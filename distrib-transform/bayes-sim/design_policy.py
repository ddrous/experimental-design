#%% 1) Imports, policy configuration, and experiment conventions
"""Sequential Bayesian experimental-design policy on top of a pretrained Bayes Transport model.

This script is intentionally notebook-like (``#%%`` cells) and has no ``main()`` function.  It is
meant to be copied into, and executed FROM, a completed Bayes-Transport run directory.  By default
it loads ``./artefacts/model_best.eqx`` and writes a separate policy run under ``./policy_runs``.

Core loop
---------
1. Draw theta* and a base-prior particle cloud.  One episode has ``experimental_budget``
   sequential policy decisions (30 by default).
2. The policy ALWAYS consumes (a) the current cloud and (b) the ENTIRE design/outcome history and
   proposes K designs jointly. Two regimes are available. The default SEPARATE policy has its own
   cloud/history Transformer. If all four Bayes-component reuse switches are enabled, the code
   automatically enters an ALINE-style JOINT regime with one shared inference/acquisition backbone.
3. Reparameterise the Gaussian observation model to simulate outcomes at those designs.
4. Update the belief with the pretrained Bayes Transport. In SEPARATE mode this is an explicit
   Bayes-map call after the policy action. In JOINT ALINE-style mode, the shared posterior Transformer
   returns both the posterior cloud and the features used to emit the NEXT design set in one call.
   Genuine multi-observation checkpoints consume the K observations jointly with observation_count=K.
5. Reward the policy by the improvement between consecutive posterior clouds.  The default
   ``reward_mode='attraction'`` uses the first term of the empirical energy score (mean distance to
   theta*).  ``reward_mode='energy_score'`` uses the full proper energy score, retaining the
   repulsion/spread term.
6. Repeat with the posterior cloud as the next prior.

Gradient estimators
-------------------
``gradient_estimator='pathwise'`` differentiates through the reparameterised policy action,
Gaussian likelihood sample, and frozen Bayes Transport.  This is the pathwise/reparameterisation
estimator of Mohamed et al. (2020), Eq. 29.

``gradient_estimator='reinforce'`` detaches sampled actions/environment transitions and uses the
score-function / REINFORCE estimator.  The default dense reward mirrors ALINE's idea of using
one-step posterior improvement (Huang et al., 2025, Eq. 10) with a policy-gradient objective
(Eq. 11), but substitutes the empirical-cloud concentration/energy-score improvement requested
for this project.  An optional EMA baseline is included as a variance-reduction control variate;
Mohamed et al. (2020), Eq. 14 shows that subtracting a constant baseline preserves unbiasedness.

Evaluation
----------
The script evaluates learned versus uniform-random design policies with common truth/prior/noise
samples.  It reports sequential Prior Contrastive Estimation (sPCE) as an EIG lower bound (the
same BED metric used in ALINE's experiments), energy score, attraction, posterior-mean RMSE,
spread, covariance log-volume, and interval coverage.  It also recreates the Bayes-Transport
``fixed_trajectory_epoch_XXXXXX.png`` style plot, now using strategic designs, and produces a
side-by-side strategic-vs-policy-specific-fixed-random trajectory comparison.

Important compatibility point
-----------------------------
The policy has its OWN configuration below, including min/max source count, min/max source
dimensionality, held-out shapes, and four explicit Bayes-component reuse switches (prior embedder,
likelihood Transformer, observation embedder, posterior Transformer). For a heterogeneous/dimension-agnostic Bayes checkpoint, downstream shapes only need to fit inside
the checkpoint's padded maxima; they may be fixed, narrower, below the original training minima, or
otherwise out-of-distribution. In JOINT mode the pretrained agnostic embedders are retained exactly.
Only a genuinely fixed-shape Bayes checkpoint requires an exact downstream shape match. Remaining legacy Bayes-Transport architecture
values are recovered automatically by safely parsing the archived training script so the Equinox
checkpoint skeleton matches the pretrained model. The old fixed-point/drifting/replay-buffer
training machinery is deliberately absent from this file.
Training is organized into explicit epochs even though every optimizer step uses newly simulated data;
per-step and per-epoch losses are both stored and plotted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import ast
import csv
import datetime
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from IPython.display import display
from tqdm.auto import tqdm
import yaml

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

Array = jax.Array


@dataclass(frozen=True)
class PolicyConfig:
    """All settings that belong to experimental-design-policy training/evaluation.

    The pretrained Bayes Transport has a separate, automatically recovered architecture config.
    In particular, there is intentionally NO policy analogue of min/max observations per step,
    no fixed-point solver, no drifting field, and no replay buffer.
    """

    # Reproducibility.  Keep the exact Bayes-Transport seed by default so the fixed diagnostic
    # uses the same theta* and prior cloud stored by the original run.
    seed: int = 2030

    # The script is executed from the Bayes-Transport run folder.
    bayes_artifact_dir: str = "./artefacts"
    bayes_checkpoint: str = "model_best.eqx"
    # None => safely parse an archived *.py file in the current run folder that contains the
    # original BayesTransportConfig.  Set an explicit path if the folder contains several copies.
    bayes_transport_source_script: str | None = None

    # Policy run bookkeeping.  Policy checkpoints/plots never overwrite Bayes-Transport artefacts.
    train_policy: bool = True
    policy_runs_base: str = "./policy_runs"
    policy_reload_dir: str | None = None

    # Sequential experimental-design problem.
    # Experimental budget = number of sequential policy decisions in ONE episode.
    # At each decision the policy proposes ``designs_per_step`` designs jointly, so the total
    # number of likelihood evaluations in an episode is experimental_budget * designs_per_step.
    # IMPORTANT: when the loaded Bayes Transport was trained with multi-observation blocks, these
    # K outcomes are passed to that Bayes map JOINTLY as one K-observation prefix.  Thus K is also
    # the observation count used by the pretrained map at each policy decision.
    # The BED benchmark default is deliberately 30 sequential decisions.
    experimental_budget: int = 30
    designs_per_step: int = 1            # K designs proposed JOINTLY at each policy decision.
    num_particles: int = 32              # empirical belief-cloud size seen by the policy.

    # Shape distribution for policy training, using the SAME min/max convention as the original
    # Bayes-Transport codebase. For a dimension-agnostic Bayes checkpoint, downstream shapes only
    # need to fit inside its padded MAX capacities; they may differ from the pretraining range.
    #
    # Embedding rule:
    #   * if Smin=Smax AND Dmin=Dmax, this is a fixed-shape policy. It does NOT use the pretrained
    #     dimensionality-agnostic/TAMO embedder. Instead, it learns fresh one-layer projections from
    #     normalized physical theta and (design,outcome) values into embedding_dim.
    #   * otherwise this is a heterogeneous/dimension-agnostic policy, for which the pretrained
    #     Bayes embedders can be reused (or fresh same-architecture copies can be trained).
    min_num_sources: int = 2
    max_num_sources: int = 2
    min_source_dim: int = 2
    max_source_dim: int = 2

    # Held-out (S,D) combinations, following the original codebase convention. The active default
    # there was empty, so the default here is also empty. Example ablations from the old script were
    # ((1,6),(6,1),(3,3),(6,6)) and ((1,4),(4,1),(2,2),(4,4)).
    heldout_shapes: tuple[tuple[int, int], ...] = ()

    # Preserve the previous safety behaviour: unless explicitly overridden, any shapes held out
    # while training the Bayes map are also excluded from policy training.
    include_bayes_heldout_shapes_in_policy_training: bool = True

    # Policy Transformer.  The cloud is first cross-attended to the processed history; learned
    # design-query tokens then attend to the fused cloud+history memory and emit a SET of designs.
    policy_hidden_dim: int = 192
    policy_heads: int = 8
    policy_mlp_ratio: int = 3
    history_depth: int = 3
    cloud_history_depth: int = 3
    decoder_depth: int = 2

    # Component-level reuse from the pretrained Bayes Transport. These are the four major learned
    # pieces of SequentialBayesModel and can now be selected independently for the policy:
    #   1) prior/theta embedder,
    #   2) causal likelihood Transformer,
    #   3) design/outcome observation embedder,
    #   4) posterior Transformer.
    #
    # If ALL FOUR are True, the script automatically enters the JOINT ALINE-style regime below:
    # the Bayes backbone is reused as one integrated inference/acquisition network and a continuous
    # set-valued acquisition head is attached to the posterior Transformer features. Otherwise the
    # existing SEPARATE policy architecture is used, optionally borrowing the selected components.
    #
    # In the SEPARATE regime, a fixed downstream policy uses fresh trainable linear projections,
    # matching the fixed-shape convention from the Bayes codebase. In JOINT mode, however, all four
    # selected Bayes components are reused literally: if the checkpoint is dimension-agnostic, its
    # pretrained agnostic embedders remain valid even when the downstream task fixes one (S,D).
    reuse_prior_embedder: bool = True
    reuse_likelihood_transformer: bool = True
    reuse_observation_embedder: bool = True
    reuse_posterior_transformer: bool = True

    # By default reused Bayes components are frozen under the POLICY objective. This mirrors the
    # separation in ALINE: the acquisition loss updates the acquisition parameters, while inference
    # components have their own objective. If Bayes fine-tuning is enabled below, those components
    # are updated by the energy-score inference loss and then synchronized back into the policy.
    train_reused_components_with_policy: bool = False

    # Fresh heterogeneous embedders can optionally be trained. Fresh fixed-shape linear projections
    # are always trainable because they are the entire fixed-shape input interface.
    train_policy_embedders: bool = False

    # ALINE-style acquisition branch used only when all four reuse flags are True. The paper uses
    # three Transformer layers, four attention heads and 128-wide feed-forward layers; those values
    # are retained as defaults. The latent width remains policy_hidden_dim so it can consume the
    # pretrained Bayes posterior features without destroying them with a bottleneck.
    joint_aline_depth: int = 3
    joint_aline_heads: int = 4
    joint_aline_ff_dim: int = 128
    joint_aline_head_width: int = 128

    # Continuous bounded stochastic policy: z ~ Normal(mu,sigma), x = box_map(tanh(z)).
    initial_policy_std: float = 0.55
    min_policy_std: float = 0.03
    max_policy_std: float = 1.50
    entropy_bonus: float = 0.0
    # Optional within-set diversity regularizer, useful only when designs_per_step > 1.
    design_set_diversity_weight: float = 0.0
    design_set_diversity_scale: float = 0.40

    # Gradient estimator and reward.
    gradient_estimator: str = "reinforce"   # {"pathwise", "reinforce"}
    reward_mode: str = "energy_score"        # {"attraction", "energy_score"}
    discount_gamma: float = 1.0
    reward_scale: float = 1.0
    reward_clip: float | None = None

    # REINFORCE-specific variance reduction / credit assignment.
    # "dense" uses gamma^t * R_t * log pi_t, matching the dense ALINE form.
    # "return_to_go" uses the conventional discounted future return from each decision.
    reinforce_credit_assignment: str = "return_to_go"  # {"dense", "return_to_go"}
    reinforce_baseline: str = "ema"                    # {"none", "ema"}
    reinforce_baseline_decay: float = 0.98
    reinforce_normalize_advantage: bool = True

    # Bayes Transport is frozen by default.  If enabled, it is fine-tuned in a SEPARATE auxiliary
    # energy-score step on data induced by the current policy; the policy objective never silently
    # changes meaning.  This mirrors ALINE's separation of policy and inference objectives.
    finetune_bayes_transport: bool = False
    bayes_finetune_learning_rate: float = 1e-6
    bayes_finetune_weight_decay: float = 1e-5
    bayes_finetune_grad_clip_norm: float = 10.0
    bayes_finetune_loss_weight: float = 1.0
    # If the policy uses frozen copies of the Bayes embedders, optionally refresh those copies after
    # every Bayes fine-tune step so policy input semantics track the evolving inference network.
    sync_reused_components_after_bayes_finetune: bool = True

    # Optimisation.  Training is explicitly organised into epochs even though EVERY optimizer
    # step draws a completely fresh simulator batch.  Thus an epoch is a bookkeeping/diagnostic
    # unit rather than a pass over a finite dataset.  The defaults retain 50,000 optimizer updates.
    epochs: int = 250*1
    train_steps_per_epoch: int = 128
    batch_size: int = 2
    learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    grad_clip_norm: float = 10.0

    # Evaluation and persistence cadence, expressed in EPOCHS.  With the default 100 fresh
    # optimizer steps per epoch these correspond to the old 1k/5k/1k step cadences.
    eval_every_epochs: int = 10
    save_every_epochs: int = 50
    plot_every_epochs: int = 10
    n_eval_trajectories: int = 128
    eval_deterministic_policy: bool = True
    # Number of independent prior contrastives used for the sPCE/EIG lower-bound diagnostic.
    # ALINE used much larger values for final paper tables; this moderate default keeps routine
    # training-time diagnostics practical.  Increase substantially for publication-quality values.
    eig_contrastive_samples: int = 1_024
    eig_eval_trajectories: int = 64
    credible_interval_mass: float = 0.90

    # Plotting / diagnostics.
    grid_size: int = 180
    confidence_z: float = 1.96
    final_plot_examples: int = 3


POLICY_CFG = PolicyConfig()


def policy_is_fixed_shape(cfg: PolicyConfig) -> bool:
    """True exactly when BOTH source count and source dimensionality are fixed."""
    return (
        cfg.min_num_sources == cfg.max_num_sources
        and cfg.min_source_dim == cfg.max_source_dim
    )


def reused_component_names(cfg: PolicyConfig) -> tuple[str, ...]:
    """Human-readable list of Bayes components requested for reuse by the policy."""
    pairs = (
        ("prior_embedder", cfg.reuse_prior_embedder),
        ("likelihood_transformer", cfg.reuse_likelihood_transformer),
        ("observation_embedder", cfg.reuse_observation_embedder),
        ("posterior_transformer", cfg.reuse_posterior_transformer),
    )
    return tuple(name for name, enabled in pairs if enabled)


def joint_aline_mode(cfg: PolicyConfig) -> bool:
    """All four reused Bayes components => integrated ALINE-style inference/acquisition mode."""
    return all((
        cfg.reuse_prior_embedder,
        cfg.reuse_likelihood_transformer,
        cfg.reuse_observation_embedder,
        cfg.reuse_posterior_transformer,
    ))


def validate_policy_config(cfg: PolicyConfig):
    if cfg.experimental_budget < 1:
        raise ValueError("experimental_budget must be >= 1.")
    if cfg.min_num_sources < 1 or cfg.min_source_dim < 1:
        raise ValueError("min_num_sources and min_source_dim must both be >= 1.")
    if cfg.max_num_sources < cfg.min_num_sources:
        raise ValueError("max_num_sources must be >= min_num_sources.")
    if cfg.max_source_dim < cfg.min_source_dim:
        raise ValueError("max_source_dim must be >= min_source_dim.")
    all_shapes = {
        (s, d)
        for s in range(cfg.min_num_sources, cfg.max_num_sources + 1)
        for d in range(cfg.min_source_dim, cfg.max_source_dim + 1)
    }
    heldout = {tuple(map(int, shape)) for shape in cfg.heldout_shapes}
    if not heldout.issubset(all_shapes):
        raise ValueError("Every heldout_shapes entry must lie inside the configured policy S,D ranges.")
    if heldout == all_shapes:
        raise ValueError("heldout_shapes cannot remove every policy-training shape.")
    if cfg.designs_per_step < 1:
        raise ValueError("designs_per_step must be >= 1.")
    if cfg.num_particles < 2:
        raise ValueError("num_particles must be >= 2 for empirical-cloud energy scores.")
    if cfg.policy_hidden_dim % cfg.policy_heads != 0:
        raise ValueError("policy_hidden_dim must be divisible by policy_heads.")
    if cfg.joint_aline_depth < 1:
        raise ValueError("joint_aline_depth must be >= 1.")
    if cfg.joint_aline_heads < 1 or cfg.policy_hidden_dim % cfg.joint_aline_heads != 0:
        raise ValueError("policy_hidden_dim must be divisible by joint_aline_heads.")
    if cfg.joint_aline_ff_dim < 1 or cfg.joint_aline_head_width < 1:
        raise ValueError("joint ALINE feed-forward/head widths must be positive.")
    if joint_aline_mode(cfg) and cfg.train_reused_components_with_policy:
        raise ValueError(
            "In joint ALINE mode reused Bayes components are optimized only by the inference "
            "fine-tuning objective. Set train_reused_components_with_policy=False; the policy "
            "gradient updates the acquisition head only, matching ALINE's objective separation."
        )
    if cfg.gradient_estimator not in {"pathwise", "reinforce"}:
        raise ValueError("gradient_estimator must be 'pathwise' or 'reinforce'.")
    if cfg.reward_mode not in {"attraction", "energy_score"}:
        raise ValueError("reward_mode must be 'attraction' or 'energy_score'.")
    if cfg.reinforce_credit_assignment not in {"dense", "return_to_go"}:
        raise ValueError("Unsupported reinforce_credit_assignment.")
    if cfg.reinforce_baseline not in {"none", "ema"}:
        raise ValueError("reinforce_baseline must be 'none' or 'ema'.")
    if not (0.0 < cfg.reinforce_baseline_decay < 1.0):
        raise ValueError("reinforce_baseline_decay must lie in (0,1).")
    if not (0.0 < cfg.discount_gamma <= 1.0):
        raise ValueError("discount_gamma must lie in (0,1].")
    if cfg.min_policy_std <= 0 or cfg.max_policy_std < cfg.min_policy_std:
        raise ValueError("Policy std bounds are invalid.")
    if cfg.epochs < 1 or cfg.train_steps_per_epoch < 1 or cfg.batch_size < 1:
        raise ValueError("epochs, train_steps_per_epoch, and batch_size must all be >= 1.")
    if cfg.eval_every_epochs < 1 or cfg.save_every_epochs < 1 or cfg.plot_every_epochs < 1:
        raise ValueError("Evaluation/save/plot epoch cadences must all be >= 1.")


validate_policy_config(POLICY_CFG)


#%% 2) Recover ONLY the pretrained Bayes-Transport architecture/simulator metadata
@dataclass(frozen=True)
class BayesTransportConfig:
    """Minimal compatibility config required to reconstruct the pretrained checkpoint.

    The observation-prefix fields are checkpoint metadata rather than policy hyperparameters.
    They are recovered from the archived training script both to rebuild the exact saved Equinox
    tree and to determine whether one policy decision should be consumed by the Bayes map as a
    genuine multi-observation block or as recurrent single-observation updates.
    """
    seed: int = 2030
    env_name: str = "location-finding"
    num_sources: int = 2
    source_dim: int = 2
    base_prior_distribution: str = "uniform"
    prior_std: float = 1.0
    synthetic_prior_match_probability: float = 1.0
    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.30
    min_num_sources: int = 2
    max_num_sources: int = 2
    min_source_dim: int = 2
    max_source_dim: int = 2
    heldout_shapes: tuple[tuple[int, int], ...] = ()
    embedding_dim: int = 192
    fixed_shape_learned_projection: bool = True
    dimension_embedder_depth: int = 4
    scalar_encoder_depth: int = 4
    embedding_heads: int = 8
    # Checkpoint metadata only; not duplicated as policy settings.  The policy-side
    # ``designs_per_step`` determines how many fresh observations are acquired at each decision.
    # For a multi-observation checkpoint those observations are consumed jointly by this map.
    min_observations_per_step: int = 1
    max_observations_per_step: int = 1
    likelihood_hidden_dim: int = 192
    likelihood_heads: int = 8
    likelihood_mlp_ratio: int = 4
    likelihood_depth: int = 4
    posterior_conditioning: str = "adaln"
    hidden_dim: int = 256
    heads: int = 8
    mlp_ratio: int = 4
    posterior_depth: int = 6
    max_embedding_displacement: float = 6.0
    canonicalize_particle_sources: bool = True
    y_center: float = 0.0
    y_scale: float = 3.0


def _safe_ast_value(node: ast.AST) -> Any:
    """Evaluate simple dataclass defaults without executing the archived training script."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_safe_ast_value(item) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_ast_value(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow)):
        left, right = _safe_ast_value(node.left), _safe_ast_value(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.FloorDiv): return left // right
        if isinstance(node.op, ast.Pow): return left ** right
    raise ValueError(f"Unsupported archived-config expression: {ast.dump(node, include_attributes=False)}")


def _extract_dataclass_defaults(script_path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            result: dict[str, Any] = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                    try:
                        result[item.target.id] = _safe_ast_value(item.value)
                    except ValueError:
                        # Architecture fields in the supplied Bayes-Transport script are simple
                        # literals.  Ignore unrelated computed runtime defaults rather than execute.
                        pass
            return result
    raise ValueError(f"Could not find class {class_name!r} in {script_path}.")


def discover_bayes_transport_source(cfg: PolicyConfig) -> Path | None:
    if cfg.bayes_transport_source_script is not None:
        path = Path(cfg.bayes_transport_source_script).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured Bayes-Transport source script does not exist: {path}")
        return path

    cwd = Path.cwd().resolve()
    this_file = Path(__file__).resolve() if "__file__" in globals() else None
    candidates: list[Path] = []
    for path in sorted(cwd.glob("*.py")):
        if this_file is not None and path.resolve() == this_file:
            continue
        try:
            prefix = path.read_text(encoding="utf-8", errors="replace")[:250_000]
        except OSError:
            continue
        if "class BayesTransportConfig" in prefix and "class SequentialBayesModel" in prefix:
            candidates.append(path)
    if not candidates:
        return None
    # Prefer filenames that look like the archived training script.
    candidates.sort(key=lambda p: ("bayes" not in p.name.lower(), "transport" not in p.name.lower(), p.name))
    return candidates[0]


def recover_bayes_transport_config(cfg: PolicyConfig) -> tuple[BayesTransportConfig, Path | None]:
    source = discover_bayes_transport_source(cfg)
    if source is None:
        print("WARNING: no archived Bayes-Transport script found; using compatibility defaults.")
        print("         Set POLICY_CFG.bayes_transport_source_script explicitly if checkpoint loading fails.")
        return BayesTransportConfig(), None

    recovered = _extract_dataclass_defaults(source, "BayesTransportConfig")
    allowed = {field.name for field in fields(BayesTransportConfig)}
    payload = {name: value for name, value in recovered.items() if name in allowed}
    bt_cfg = BayesTransportConfig(**payload)
    return bt_cfg, source


BT_CFG, BAYES_SOURCE_SCRIPT = recover_bayes_transport_config(POLICY_CFG)


def validate_bayes_compatibility_config(cfg: BayesTransportConfig):
    if cfg.min_num_sources < 1 or cfg.min_source_dim < 1:
        raise ValueError("Bayes checkpoint has invalid source-count/dimension metadata.")
    if cfg.max_num_sources < cfg.min_num_sources or cfg.max_source_dim < cfg.min_source_dim:
        raise ValueError("Bayes checkpoint shape ranges are invalid.")
    if cfg.embedding_dim < 1 or cfg.hidden_dim < 1:
        raise ValueError("Bayes checkpoint embedding dimensions must be positive.")
    fixed_shape = cfg.min_num_sources == cfg.max_num_sources and cfg.min_source_dim == cfg.max_source_dim
    if not fixed_shape and cfg.embedding_dim % cfg.embedding_heads != 0:
        raise ValueError("Recovered embedding_dim must be divisible by embedding_heads.")
    if cfg.hidden_dim % cfg.heads != 0:
        raise ValueError("Recovered hidden_dim must be divisible by heads.")
    if cfg.posterior_conditioning not in {"cross_attention", "adaln"}:
        raise ValueError("Recovered posterior_conditioning is unsupported.")


validate_bayes_compatibility_config(BT_CFG)


def policy_shape_pool_from_ranges_only(cfg: PolicyConfig) -> tuple[tuple[int, int], ...]:
    """All policy-configured shapes after policy-held-out exclusions only.

    This early helper intentionally does not apply Bayes held-out exclusions; it exists so shape
    compatibility can distinguish hard padded capacity from the Bayes model's training distribution.
    """
    excluded = {tuple(map(int, shape)) for shape in cfg.heldout_shapes}
    return tuple(
        (s, d)
        for s in range(cfg.min_num_sources, cfg.max_num_sources + 1)
        for d in range(cfg.min_source_dim, cfg.max_source_dim + 1)
        if (s, d) not in excluded
    )


def validate_policy_shape_support(cfg: PolicyConfig, bt_cfg: BayesTransportConfig):
    """Validate ARCHITECTURAL shape compatibility without confusing it with training support.

    A heterogeneous Bayes-Transport checkpoint uses the dimensionality-agnostic theta and
    observation embedders.  Those modules are explicitly designed to consume any active (S,D)
    that fits inside their padded maxima, even when the downstream policy fixes one particular
    shape or chooses shapes that were not present in the Bayes training distribution.  Such cases
    are distributional extrapolation, not an architectural error, and may be adapted by optional
    Bayes fine-tuning.

    A genuinely fixed-shape Bayes checkpoint is different: its learned linear projections have a
    fixed physical input width, so only that exact (S,D) is structurally compatible.
    """
    bayes_fixed = (
        bt_cfg.min_num_sources == bt_cfg.max_num_sources
        and bt_cfg.min_source_dim == bt_cfg.max_source_dim
    )

    if bayes_fixed:
        exact_policy_shape = (
            policy_is_fixed_shape(cfg)
            and cfg.min_num_sources == bt_cfg.min_num_sources
            and cfg.min_source_dim == bt_cfg.min_source_dim
        )
        if not exact_policy_shape:
            raise ValueError(
                "The loaded Bayes Transport is genuinely fixed-shape, so its learned physical "
                "input projections only support the exact checkpoint shape "
                f"(S={bt_cfg.min_num_sources},D={bt_cfg.min_source_dim}). The downstream policy "
                f"requests S=[{cfg.min_num_sources},{cfg.max_num_sources}], "
                f"D=[{cfg.min_source_dim},{cfg.max_source_dim}]."
            )
        return

    # Dimension-agnostic checkpoint: only padded capacity is a hard architectural limit.
    if cfg.max_num_sources > bt_cfg.max_num_sources:
        raise ValueError(
            "Policy max_num_sources exceeds the padded capacity of the dimension-agnostic Bayes "
            f"checkpoint: policy max={cfg.max_num_sources}, Bayes capacity={bt_cfg.max_num_sources}."
        )
    if cfg.max_source_dim > bt_cfg.max_source_dim:
        raise ValueError(
            "Policy max_source_dim exceeds the padded capacity of the dimension-agnostic Bayes "
            f"checkpoint: policy max={cfg.max_source_dim}, Bayes capacity={bt_cfg.max_source_dim}."
        )

    # Going below the original minima, using a Bayes-held-out shape, or fixing a downstream shape
    # is allowed.  The pretrained agnostic embedders can represent it; print a warning because the
    # inference map may be out-of-distribution until fine-tuned.
    shape_pool = policy_shape_pool_from_ranges_only(cfg)
    outside_training_range = [
        (s, d) for s, d in shape_pool
        if s < bt_cfg.min_num_sources or d < bt_cfg.min_source_dim
    ]
    bayes_heldout = {tuple(map(int, x)) for x in bt_cfg.heldout_shapes}
    reused_heldout = [x for x in shape_pool if x in bayes_heldout]
    if outside_training_range:
        print(
            "WARNING: downstream policy includes shapes below the Bayes-Transport training minima: "
            f"{outside_training_range}. The dimension-agnostic checkpoint can represent them, but "
            "this is distributional extrapolation; finetune_bayes_transport=True can adapt it."
        )
    if reused_heldout and cfg.include_bayes_heldout_shapes_in_policy_training:
        print(
            "WARNING: downstream policy explicitly includes Bayes-held-out shapes: "
            f"{reused_heldout}. This is allowed for the dimension-agnostic checkpoint and is an "
            "out-of-distribution downstream test/fine-tuning setting."
        )


validate_policy_shape_support(POLICY_CFG, BT_CFG)


def describe_bayes_observation_compatibility(policy_cfg: PolicyConfig, bt_cfg: BayesTransportConfig) -> str:
    """Human-readable description of how one policy decision enters the pretrained Bayes map."""
    K = int(policy_cfg.designs_per_step)
    omin = int(bt_cfg.min_observations_per_step)
    omax = int(bt_cfg.max_observations_per_step)
    if omin == 1 and omax == 1:
        if K == 1:
            return "strict single-observation checkpoint: one Bayes update per policy decision"
        return (
            f"strict single-observation checkpoint: {K} jointly proposed observations are consumed "
            f"as {K} recurrent one-observation Bayes updates"
        )
    range_note = f"checkpoint trained on observation prefixes {omin}..{omax}"
    if K < omin or K > omax:
        range_note += (
            f"; WARNING designs_per_step={K} is outside that trained range, so the likelihood "
            "Transformer is being extrapolated in observation-count space"
        )
    return (
        f"genuine multi-observation checkpoint: all {K} observations from each policy decision "
        f"are contextualised jointly and passed through ONE Bayes map ({range_note})"
    )


BAYES_OBSERVATION_COMPATIBILITY = describe_bayes_observation_compatibility(POLICY_CFG, BT_CFG)

if (
    joint_aline_mode(POLICY_CFG)
    and BT_CFG.min_observations_per_step == 1
    and BT_CFG.max_observations_per_step == 1
    and POLICY_CFG.designs_per_step > 1
):
    print(
        "WARNING: joint ALINE mode is reusing a checkpoint trained only with one observation per "
        "Bayes update while designs_per_step>1. The shared backbone will still run, but this is "
        "observation-count extrapolation (and the original likelihood module is the O=1 bypass). "
        "Consider finetune_bayes_transport=True for downstream adaptation."
    )


#%% 3) Persistence helpers and the physical source-localisation simulator
def make_policy_run_dir(cfg: PolicyConfig) -> Path:
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    root = Path(cfg.policy_runs_base).expanduser().resolve()
    regime = "joint_aline" if joint_aline_mode(cfg) else "separate"
    run_dir = root / f"design_policy_{regime}_{cfg.gradient_estimator}_{cfg.reward_mode}_{stamp}"
    for child in ("plots", "artefacts", "tables"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_running_script_to_run_dir(run_dir: Path) -> Path | None:
    if "__file__" not in globals():
        return None
    source = Path(__file__).expanduser().resolve()
    if not source.is_file():
        return None
    destination = run_dir / source.name
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def save_json(path: str | Path, payload: dict[str, Any]):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_model(path: str | Path, model: eqx.Module):
    eqx.tree_serialise_leaves(Path(path), model)


def sample_base_prior_np(
    rng: np.random.Generator,
    n: int,
    cfg: BayesTransportConfig,
    *,
    num_sources: int,
    source_dim: int,
) -> np.ndarray:
    """Same base-prior sampler as the Bayes-Transport script."""
    shape = (int(n), int(num_sources), int(source_dim))
    if cfg.base_prior_distribution == "uniform":
        return rng.uniform(cfg.design_low, cfg.design_high, size=shape).astype(np.float32)
    if cfg.base_prior_distribution == "gaussian":
        return rng.normal(0.0, cfg.prior_std, size=shape).astype(np.float32)
    raise ValueError("base_prior_distribution must be 'uniform' or 'gaussian'.")


def pad_theta_np(theta: np.ndarray, cfg: BayesTransportConfig) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32)
    padded = np.zeros(theta.shape[:-2] + (cfg.max_num_sources, cfg.max_source_dim), dtype=np.float32)
    padded[..., :theta.shape[-2], :theta.shape[-1]] = theta
    return padded


def source_log_mean_np(theta: np.ndarray, designs: np.ndarray, cfg: BayesTransportConfig) -> np.ndarray:
    """Copied physical likelihood mean: E[y | theta,x] on the log-intensity scale."""
    theta = np.asarray(theta, dtype=np.float64)
    designs = np.asarray(designs, dtype=np.float64)
    theta_expanded = np.expand_dims(theta, axis=-3)
    design_expanded = np.expand_dims(designs, axis=-2)
    dist_sq = np.sum((theta_expanded - design_expanded) ** 2, axis=-1)
    intensity = cfg.background + np.sum(cfg.source_strength / (cfg.softening + dist_sq), axis=-1)
    return np.log(intensity)


def source_log_mean_jax(
    theta_padded: Array, designs_padded: Array, num_sources: Array, theta_size: Array,
    cfg: BayesTransportConfig,
) -> Array:
    """Differentiable padded equivalent of source_log_mean_np for pathwise policy gradients."""
    source_dim = theta_size // num_sources
    source_valid = jnp.arange(theta_padded.shape[-2]) < num_sources
    coordinate_valid = jnp.arange(theta_padded.shape[-1]) < source_dim
    differences = theta_padded[None, :, :] - designs_padded[:, None, :]
    dist_sq = jnp.sum(differences**2 * coordinate_valid[None, None, :], axis=-1)
    contributions = cfg.source_strength / (cfg.softening + dist_sq)
    intensity = cfg.background + jnp.sum(contributions * source_valid[None, :], axis=-1)
    return jnp.log(intensity)


def log_likelihood_np(
    theta: np.ndarray, designs: np.ndarray, outcomes: np.ndarray, cfg: BayesTransportConfig
) -> np.ndarray:
    """Gaussian log p(y | x,theta); supports a leading contrastive-particle axis on theta."""
    mean = source_log_mean_np(theta, designs, cfg)
    sigma = float(cfg.observation_noise_std)
    return -0.5 * ((np.asarray(outcomes) - mean) / sigma) ** 2 - math.log(sigma * math.sqrt(2.0 * math.pi))


def _base_prior_plot_extent(cfg: BayesTransportConfig) -> float:
    if cfg.base_prior_distribution == "uniform":
        return max(abs(float(cfg.design_low)), abs(float(cfg.design_high)))
    return 3.0 * float(cfg.prior_std)


#%% 4) Exact Bayes-Transport embedding/posterior architecture copied from the training script
# The following block is copied (rather than imported) so this policy file remains standalone inside
# the run folder and can reconstruct Equinox checkpoints without executing the original training file.
#%% 4a) Source-label symmetry helpers
def canonicalize_sources_np(theta: np.ndarray) -> np.ndarray:
    """Sort ACTIVE exchangeable sources by their first coordinate."""
    theta = np.asarray(theta)
    order = np.argsort(theta[..., 0], axis=-1)
    return np.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_sources_jax(theta: Array) -> Array:
    order = jnp.argsort(theta[..., 0], axis=-1)
    return jnp.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_padded_sources_np(theta: np.ndarray, num_sources: int) -> np.ndarray:
    """Canonicalize only the active source rows, keeping padding at the end."""
    theta = np.asarray(theta)
    indices = np.arange(theta.shape[-2])
    key = np.where(indices < int(num_sources), theta[..., 0], np.inf)
    order = np.argsort(key, axis=-1)
    return np.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_padded_sources_jax(theta: Array, num_sources: Array) -> Array:
    indices = jnp.arange(theta.shape[-2])
    key = jnp.where(indices < num_sources, theta[..., 0], jnp.inf)
    order = jnp.argsort(key, axis=-1)
    return jnp.take_along_axis(theta, order[..., None], axis=-2)


#%% 4b) Token helpers shared by the dimension and posterior Transformers
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _mlp_tokens(layer: eqx.nn.MLP, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]




#%% 4c) TAMO-style dimension-agnostic scalar-to-vector embedders
class DimensionSelfAttentionBlock(eqx.Module):
    """Small Transformer block operating across scalar dimension tokens."""

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        attn_key, ff1_key, ff2_key = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=dim,
            key_size=dim,
            value_size=dim,
            output_size=dim,
            dropout_p=0.0,
            key=attn_key,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=ff2_key)

    def __call__(self, tokens: Array, valid: Array) -> Array:
        # Every query may attend only to ACTIVE dimension tokens.  Inactive query rows
        # are masked back to zero after each residual block, which avoids NaNs from an
        # all-False attention row while still preventing padding from becoming memory.
        key_mask = jnp.broadcast_to(valid[None, :], (tokens.shape[0], tokens.shape[0]))
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h, mask=key_mask)
        tokens = jnp.where(valid[:, None], tokens, 0.0)

        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        tokens = tokens + h
        return jnp.where(valid[:, None], tokens, 0.0)


def _masked_mean(tokens: Array, valid: Array) -> Array:
    weights = valid.astype(tokens.dtype)[:, None]
    return jnp.sum(tokens * weights, axis=0) / jnp.maximum(jnp.sum(weights), 1.0)


class FixedShapeObservationEmbedder(eqx.Module):
    """Fixed-shape observation normalizer with an optional one-layer learned projection."""

    projection: eqx.nn.Linear | None
    design_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)
    source_dim: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.design_scale = max(abs(cfg.design_low), abs(cfg.design_high), 1.0)
        self.y_center = cfg.y_center
        self.y_scale = max(cfg.y_scale, 1e-6)
        self.source_dim = cfg.max_source_dim
        if cfg.fixed_shape_learned_projection:
            self.projection = eqx.nn.Linear(
                self.source_dim + 1, cfg.embedding_dim, key=key
            )
            self.output_dim = cfg.embedding_dim
        else:
            self.projection = None
            self.output_dim = self.source_dim + 1

    def __call__(self, observation: Array, num_sources: Array, theta_size: Array) -> Array:
        del num_sources, theta_size
        values = jnp.concatenate(
            [
                observation[: self.source_dim] / self.design_scale,
                (observation[-1:] - self.y_center) / self.y_scale,
            ],
            axis=-1,
        )
        return values if self.projection is None else self.projection(values)


class ObservationDimensionEmbedder(eqx.Module):
    """TAMO Figure-2-style dimension aggregator for one padded (design, outcome) observation.

    The design coordinates x and outcome coordinates y use separate scalar embedders, as in
    the two bottom branches of Figure 2.  Their resulting tokens are then processed together
    by the shared dimension Transformer, producing transformed x- and y-token streams.  Each
    stream is modulated element-wise by its own learned positional vectors, averaged within
    stream, and the two stream summaries are ADDED to obtain the final E-vector.  There is
    deliberately no concatenate-and-project fusion after pooling.
    """

    design_scalar_encoder: eqx.nn.MLP
    outcome_scalar_encoder: eqx.nn.MLP
    blocks: tuple[DimensionSelfAttentionBlock, ...]
    final_norm: eqx.nn.LayerNorm
    design_position_pool: Array
    outcome_position: Array

    design_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)
    max_source_dim: int = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.dimension_embedder_depth + 4)
        E = cfg.embedding_dim
        self.design_scale = max(abs(cfg.design_low), abs(cfg.design_high), 1.0)
        self.y_center = cfg.y_center
        self.y_scale = max(cfg.y_scale, 1e-6)
        self.max_source_dim = cfg.max_source_dim
        self.design_scalar_encoder = eqx.nn.MLP(
            in_size=1,
            out_size=E,
            width_size=E,
            depth=cfg.scalar_encoder_depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=keys[0],
        )
        self.outcome_scalar_encoder = eqx.nn.MLP(
            in_size=1,
            out_size=E,
            width_size=E,
            depth=cfg.scalar_encoder_depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=keys[1],
        )
        self.blocks = tuple(
            DimensionSelfAttentionBlock(
                E, cfg.embedding_heads, cfg.mlp_ratio * E, key=keys[2 + i]
            )
            for i in range(cfg.dimension_embedder_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(E)
        # Figure 2 multiplies the transformed x and y tokens by separate learned positional
        # vectors before averaging each branch.  Starting near one preserves signal scale at
        # initialization while still making each design slot and the outcome slot distinct.
        self.design_position_pool = 1.0 + 0.02 * jax.random.normal(
            keys[-2], (cfg.max_source_dim, E)
        )
        self.outcome_position = 1.0 + 0.02 * jax.random.normal(keys[-1], (E,))

    def __call__(self, observation: Array, num_sources: Array, theta_size: Array) -> Array:
        source_dim = theta_size // num_sources
        design_values = observation[: self.max_source_dim] / self.design_scale
        outcome_value = (observation[-1:] - self.y_center) / self.y_scale

        # Figure 2: separate x/y embedders first, then one shared Transformer over both
        # token streams.  Padding is masked so inactive x coordinates never become memory.
        design_tokens = _mlp_tokens(self.design_scalar_encoder, design_values[:, None])
        outcome_token = self.outcome_scalar_encoder(outcome_value)
        tokens = jnp.concatenate([design_tokens, outcome_token[None, :]], axis=0)
        valid_design = jnp.arange(self.max_source_dim) < source_dim
        valid = jnp.concatenate([valid_design, jnp.ones((1,), dtype=bool)])
        tokens = jnp.where(valid[:, None], tokens, 0.0)
        for block in self.blocks:
            tokens = block(tokens, valid)
        tokens = _layernorm_tokens(self.final_norm, tokens)

        transformed_design = tokens[: self.max_source_dim]
        transformed_outcome = tokens[-1]
        design_embedding = _masked_mean(
            transformed_design * self.design_position_pool, valid_design
        )
        # y is scalar in this simulator, so the Figure-2 y-branch average is the single
        # transformed outcome token after its positional modulation.
        outcome_embedding = transformed_outcome * self.outcome_position
        return design_embedding + outcome_embedding


class FixedShapeThetaEmbedder(eqx.Module):
    """Fixed-shape theta normalizer with an optional one-layer learned projection."""

    projection: eqx.nn.Linear | None
    num_sources: int = eqx.field(static=True)
    source_dim: int = eqx.field(static=True)
    theta_center: float = eqx.field(static=True)
    theta_scale: float = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.num_sources = cfg.max_num_sources
        self.source_dim = cfg.max_source_dim
        if cfg.base_prior_distribution == "uniform":
            self.theta_center = 0.5 * (cfg.design_low + cfg.design_high)
            self.theta_scale = max(0.5 * (cfg.design_high - cfg.design_low), 1e-6)
        else:
            self.theta_center = 0.0
            self.theta_scale = max(cfg.prior_std, 1e-6)
        self.canonicalize = cfg.canonicalize_particle_sources
        physical_dim = self.num_sources * self.source_dim
        if cfg.fixed_shape_learned_projection:
            self.projection = eqx.nn.Linear(physical_dim, cfg.embedding_dim, key=key)
            self.output_dim = cfg.embedding_dim
        else:
            self.projection = None
            self.output_dim = physical_dim

    def __call__(self, theta: Array, num_sources: Array, theta_size: Array) -> Array:
        del theta_size
        if self.canonicalize:
            theta = canonicalize_padded_sources_jax(theta, num_sources)
        values = theta[: self.num_sources, : self.source_dim].reshape(-1)
        values = (values - self.theta_center) / self.theta_scale
        return values if self.projection is None else self.projection(values)


class ThetaDimensionEmbedder(eqx.Module):
    """TAMO-style dimension aggregator for one padded source configuration theta.

    Before flattening, exchangeable source rows are canonicalized by their first active
    coordinate.  We then compact the active [S,D] block into the FIRST S*D scalar slots
    using dynamic gather indices.  This is the important flattening detail: simply
    reshaping the padded [Smax,Dmax] array would interleave inactive padding whenever
    D < Dmax and would make theta_size metadata incorrect.
    """

    scalar_encoder: eqx.nn.MLP
    blocks: tuple[DimensionSelfAttentionBlock, ...]
    final_norm: eqx.nn.LayerNorm
    source_position_pool: Array
    coordinate_position_pool: Array

    max_num_sources: int = eqx.field(static=True)
    max_source_dim: int = eqx.field(static=True)
    max_theta_size: int = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)


    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.dimension_embedder_depth + 3)
        E = cfg.embedding_dim
        self.max_num_sources = cfg.max_num_sources
        self.max_source_dim = cfg.max_source_dim
        self.max_theta_size = cfg.max_num_sources * cfg.max_source_dim
        self.prior_std = max(cfg.prior_std, 1e-6)
        self.canonicalize = cfg.canonicalize_particle_sources
        self.scalar_encoder = eqx.nn.MLP(
            in_size=1,
            out_size=E,
            width_size=E,
            depth=cfg.scalar_encoder_depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=keys[0],
        )
        self.blocks = tuple(
            DimensionSelfAttentionBlock(
                E, cfg.embedding_heads, cfg.mlp_ratio * E, key=keys[1 + i]
            )
            for i in range(cfg.dimension_embedder_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(E)
        # Keep source and coordinate identities separate rather than assigning positional
        # meaning to the compact flat index k.  This matters when D varies: flat k=2
        # can mean (source 2, coord 1) for D=2 but (source 1, coord 3) for D=3.
        # Canonical source ordering makes the source-position pool well-defined.
        self.source_position_pool = 1.0 + 0.02 * jax.random.normal(
            keys[-2], (self.max_num_sources, E)
        )
        self.coordinate_position_pool = 1.0 + 0.02 * jax.random.normal(
            keys[-1], (self.max_source_dim, E)
        )

    def __call__(self, theta: Array, num_sources: Array, theta_size: Array) -> Array:
        source_dim = theta_size // num_sources
        if self.canonicalize:
            theta = canonicalize_padded_sources_jax(theta, num_sources)

        # Compact active theta coordinates into scalar positions k=0,...,S*D-1.
        # k maps to source=floor(k/D), coordinate=k mod D.  Gather indices are clipped
        # only for inactive k; their values are removed by `valid` immediately after.
        k = jnp.arange(self.max_theta_size)
        source_index = jnp.clip(k // source_dim, 0, self.max_num_sources - 1)
        coordinate_index = jnp.clip(k % source_dim, 0, self.max_source_dim - 1)
        values = theta[source_index, coordinate_index] / self.prior_std
        valid = k < theta_size
        values = jnp.where(valid, values, 0.0)

        tokens = _mlp_tokens(self.scalar_encoder, values[:, None])
        tokens = jnp.where(valid[:, None], tokens, 0.0)
        for block in self.blocks:
            tokens = block(tokens, valid)
        tokens = _layernorm_tokens(self.final_norm, tokens)
        positions = (
            self.source_position_pool[source_index]
            * self.coordinate_position_pool[coordinate_index]
        )
        return _masked_mean(tokens * positions, valid)



#%% 4d) Causal observation Transformer and configurable particle conditioning
class CausalObservationBlock(eqx.Module):
    """Transformer block over one Omax design-outcome sequence with a causal mask."""

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        attn_key, ff1_key, ff2_key = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=dim,
            key_size=dim,
            value_size=dim,
            output_size=dim,
            dropout_p=0.0,
            key=attn_key,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=ff2_key)

    def __call__(self, tokens: Array) -> Array:
        length = tokens.shape[0]
        index = jnp.arange(length)
        causal_mask = index[:, None] >= index[None, :]

        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h, mask=causal_mask)
        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return tokens + h


class LikelihoodSequenceEmbedder(eqx.Module):
    """Project and causally contextualise observation tokens when a sequence exists.

    Output token o can depend only on observations 0,...,o.  Training keeps every
    output from Omin-1 onward, so one causal pass supplies all conditioning signals needed by
    the vmapped direct posterior transports.  This is causal context construction, not a
    posterior recurrence.

    The exact Omin=Omax=1 case is intentionally simpler: there is no observation sequence to
    contextualise.  The module is still instantiated for code-path/API compatibility,
    but owns NO arrays and returns its single incoming observation token unchanged.  Therefore
    the Posterior Transformer itself performs the first learned conditioning projection: the
    cross-attention path learns its key/value projections from that token, while the AdaLN path
    learns the modulation projection directly from it.

    For genuine multi-observation prefixes, the dimension-aggregation module and the likelihood
    Transformer retain their previous separate responsibilities.  In heterogeneous-shape mode
    the input width is E; in fixed-shape mode it is D+1.  The active Transformer owns a learned
    input projection into `likelihood_hidden_dim`, then applies its independently configured
    causal blocks.
    """

    input_projection: eqx.nn.Linear | None
    blocks: tuple[CausalObservationBlock, ...]
    final_norm: eqx.nn.LayerNorm | None
    input_dim: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    attention_heads: int = eqx.field(static=True)
    bypass_single_observation: bool = eqx.field(static=True)

    def __init__(
        self,
        cfg: BayesTransportConfig,
        *,
        key: Array,
        input_dim: int | None = None,
    ):
        self.input_dim = cfg.embedding_dim if input_dim is None else int(input_dim)
        if self.input_dim < 1:
            raise ValueError("Likelihood input dimension must be positive.")

        self.bypass_single_observation = (
            cfg.min_observations_per_step == 1
            and cfg.max_observations_per_step == 1
        )
        if self.bypass_single_observation:
            # No sequence exists to contextualise.  Keep this module in the model tree while
            # contributing exactly zero learnable parameters and preserving the incoming width.
            self.hidden_dim = self.input_dim
            self.attention_heads = 0
            self.input_projection = None
            self.blocks = ()
            self.final_norm = None
            return

        self.hidden_dim = int(cfg.likelihood_hidden_dim)
        self.attention_heads = int(cfg.likelihood_heads)
        keys = jax.random.split(key, cfg.likelihood_depth + 1)
        self.input_projection = eqx.nn.Linear(
            self.input_dim,
            self.hidden_dim,
            key=keys[0],
        )
        self.blocks = tuple(
            CausalObservationBlock(
                self.hidden_dim,
                self.attention_heads,
                cfg.likelihood_mlp_ratio * self.hidden_dim,
                key=keys[1 + i],
            )
            for i in range(cfg.likelihood_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(self.hidden_dim)

    def __call__(self, pair_embeddings: Array) -> Array:
        if self.bypass_single_observation:
            if pair_embeddings.shape[0] != 1:
                raise ValueError(
                    "Single-observation likelihood bypass expects exactly one observation token."
                )
            return pair_embeddings

        if self.input_projection is None or self.final_norm is None:
            raise RuntimeError("Active likelihood Transformer is missing its learned layers.")
        tokens = _linear_tokens(self.input_projection, pair_embeddings)
        for block in self.blocks:
            tokens = block(tokens)
        return _layernorm_tokens(self.final_norm, tokens)


class CrossAttentionParticleBlock(eqx.Module):
    """Particle self-attention followed by cross-attention to the active observation prefix."""

    self_norm: eqx.nn.LayerNorm
    cross_query_norm: eqx.nn.LayerNorm
    memory_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    cross_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(
        self, hidden_dim: int, memory_dim: int, heads: int, mlp_dim: int, *, key: Array
    ):
        self_key, cross_key, ff1_key, ff2_key = jax.random.split(key, 4)
        self.self_norm = eqx.nn.LayerNorm(hidden_dim)
        self.cross_query_norm = eqx.nn.LayerNorm(hidden_dim)
        self.memory_norm = eqx.nn.LayerNorm(memory_dim)
        self.ff_norm = eqx.nn.LayerNorm(hidden_dim)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=self_key,
        )
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=memory_dim,
            value_size=memory_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=cross_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)

    def __call__(self, particles: Array, observation_memory: Array) -> Array:
        # Self-attention lets the cloud represent its empirical reference distribution.
        h = _layernorm_tokens(self.self_norm, particles)
        particles = particles + self.self_attention(h, h, h)

        # Only memory[:o] is passed here, so unused future observation tokens never enter.
        q = _layernorm_tokens(self.cross_query_norm, particles)
        memory = _layernorm_tokens(self.memory_norm, observation_memory)
        particles = particles + self.cross_attention(q, memory, memory)

        h = _layernorm_tokens(self.ff_norm, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + h


class AdaLNParticleBlock(eqx.Module):
    """Particle self-attention block conditioned directly through AdaLN-Zero modulation.

    The causal observation token for prefix o is a fixed-dimensional summary of observations
    1:o.  It produces shift/scale/gate vectors for both residual branches.  No observation
    token is used as a key or value: observations affect the particle residual stream only
    through adaptive LayerNorm modulation, as requested.
    """

    self_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(
        self, hidden_dim: int, conditioning_dim: int, heads: int, mlp_dim: int, *, key: Array
    ):
        self_key, ff1_key, ff2_key, modulation_key = jax.random.split(key, 4)
        self.self_norm = eqx.nn.LayerNorm(hidden_dim)
        self.ff_norm = eqx.nn.LayerNorm(hidden_dim)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=self_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)
        modulation = eqx.nn.Linear(conditioning_dim, 6 * hidden_dim, key=modulation_key)
        # AdaLN-Zero: before learning, conditioning adds no residual update.  Together with
        # the zero-initialized physical output head this preserves the identity transport.
        modulation = eqx.tree_at(
            lambda layer: layer.weight, modulation, jnp.zeros_like(modulation.weight)
        )
        modulation = eqx.tree_at(
            lambda layer: layer.bias, modulation, jnp.zeros_like(modulation.bias)
        )
        self.modulation = modulation

    def __call__(self, particles: Array, conditioning: Array) -> Array:
        modulation = self.modulation(jax.nn.silu(conditioning))
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = jnp.split(
            modulation, 6, axis=-1
        )

        h = _modulate(_layernorm_tokens(self.self_norm, particles), shift_attn, scale_attn)
        particles = particles + gate_attn[None, :] * self.self_attention(h, h, h)

        h = _modulate(_layernorm_tokens(self.ff_norm, particles), shift_ff, scale_ff)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + gate_ff[None, :] * h


#%% 4e) Physical-theta output head and direct posterior transports
def compact_theta_jax(theta: Array, num_sources: Array, theta_size: Array) -> Array:
    """Pack the active [S,D] block into the first S*D entries of one static-width vector."""
    source_dim = theta_size // num_sources
    max_theta_size = theta.shape[-2] * theta.shape[-1]
    k = jnp.arange(max_theta_size)
    source_index = jnp.clip(k // source_dim, 0, theta.shape[-2] - 1)
    coordinate_index = jnp.clip(k % source_dim, 0, theta.shape[-1] - 1)
    return jnp.where(k < theta_size, theta[source_index, coordinate_index], 0.0)


def padded_theta_jax(
    compact_theta: Array,
    num_sources: Array,
    theta_size: Array,
    max_num_sources: int,
    max_source_dim: int,
) -> Array:
    """Unpack a compact theta vector back to padded [Smax,Dmax] storage."""
    source_dim = theta_size // num_sources
    source_index = jnp.repeat(jnp.arange(max_num_sources), max_source_dim)
    coordinate_index = jnp.tile(jnp.arange(max_source_dim), max_num_sources)
    valid = (source_index < num_sources) & (coordinate_index < source_dim)
    compact_index = jnp.clip(
        source_index * source_dim + coordinate_index, 0, compact_theta.shape[0] - 1
    )
    values = jnp.where(valid, compact_theta[compact_index], 0.0)
    return values.reshape(max_num_sources, max_source_dim)


class ThetaParticleOutputHead(eqx.Module):
    """Map [N,H] posterior tokens directly to compact physical theta particles."""

    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear
    max_displacement: float = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.max_displacement = cfg.max_embedding_displacement
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)
        output = eqx.nn.Linear(
            cfg.hidden_dim,
            cfg.max_num_sources * cfg.max_source_dim,
            key=key,
        )
        # Preserve the historical identity-at-initialisation transport for every mode. In the
        # augmented DEQ, J_F is strictly lower triangular across time, so I-J_F remains invertible
        # even when the one-step transport initially equals the identity.
        output = eqx.tree_at(lambda layer: layer.weight, output, jnp.zeros_like(output.weight))
        output = eqx.tree_at(lambda layer: layer.bias, output, jnp.zeros_like(output.bias))
        self.displacement_head = output

    def __call__(
        self,
        particle_tokens: Array,
        current_theta: Array,
        theta_size: Array,
    ) -> Array:
        particle_tokens = _layernorm_tokens(self.final_norm, particle_tokens)
        displacement = self.max_displacement * jnp.tanh(
            _linear_tokens(self.displacement_head, particle_tokens)
        )
        valid = jnp.arange(current_theta.shape[-1]) < theta_size
        return jnp.where(valid[None, :], current_theta + displacement, 0.0)


class CrossAttentionPosteriorTransformer(eqx.Module):
    """Direct reference-cloud -> posterior transport using observation cross-attention."""

    particle_in: eqx.nn.Linear
    blocks: tuple[CrossAttentionParticleBlock, ...]
    output_head: ThetaParticleOutputHead

    def __init__(
        self,
        cfg: BayesTransportConfig,
        *,
        key: Array,
        particle_input_dim: int | None = None,
        conditioning_dim: int | None = None,
    ):
        particle_input_dim = (
            cfg.embedding_dim if particle_input_dim is None else int(particle_input_dim)
        )
        conditioning_dim = cfg.embedding_dim if conditioning_dim is None else int(conditioning_dim)
        keys = jax.random.split(key, cfg.posterior_depth + 2)
        self.particle_in = eqx.nn.Linear(particle_input_dim, cfg.hidden_dim, key=keys[0])
        self.blocks = tuple(
            CrossAttentionParticleBlock(
                cfg.hidden_dim,
                conditioning_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[1 + i],
            )
            for i in range(cfg.posterior_depth)
        )
        self.output_head = ThetaParticleOutputHead(cfg, key=keys[-1])

    def __call__(
        self,
        current_embeddings: Array,
        current_theta: Array,
        observation_contexts: Array,
        observation_count: Array,
        theta_size: Array,
    ) -> Array:
        count = jnp.clip(observation_count, 1, observation_contexts.shape[0]).astype(jnp.int32)

        # Static branch shapes guarantee that cross-attention sees exactly memory[:o].
        def branch_for(prefix_length: int):
            def transport(args):
                embeddings, theta_particles, full_memory = args
                memory = full_memory[:prefix_length]
                particles = _linear_tokens(self.particle_in, embeddings)
                for block in self.blocks:
                    particles = block(particles, memory)
                return self.output_head(particles, theta_particles, theta_size)
            return transport

        branches = tuple(
            branch_for(prefix_length)
            for prefix_length in range(1, observation_contexts.shape[0] + 1)
        )
        return jax.lax.switch(
            count - 1,
            branches,
            (current_embeddings, current_theta, observation_contexts),
        )


class AdaLNPosteriorTransformer(eqx.Module):
    """Direct reference-cloud -> posterior transport with AdaLN observation conditioning."""

    particle_in: eqx.nn.Linear
    blocks: tuple[AdaLNParticleBlock, ...]
    output_head: ThetaParticleOutputHead

    def __init__(
        self,
        cfg: BayesTransportConfig,
        *,
        key: Array,
        particle_input_dim: int | None = None,
        conditioning_dim: int | None = None,
    ):
        particle_input_dim = (
            cfg.embedding_dim if particle_input_dim is None else int(particle_input_dim)
        )
        conditioning_dim = cfg.embedding_dim if conditioning_dim is None else int(conditioning_dim)
        keys = jax.random.split(key, cfg.posterior_depth + 2)
        self.particle_in = eqx.nn.Linear(particle_input_dim, cfg.hidden_dim, key=keys[0])
        self.blocks = tuple(
            AdaLNParticleBlock(
                cfg.hidden_dim,
                conditioning_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[1 + i],
            )
            for i in range(cfg.posterior_depth)
        )
        self.output_head = ThetaParticleOutputHead(cfg, key=keys[-1])

    def __call__(
        self,
        current_embeddings: Array,
        current_theta: Array,
        observation_contexts: Array,
        observation_count: Array,
        theta_size: Array,
    ) -> Array:
        count = jnp.clip(observation_count, 1, observation_contexts.shape[0]).astype(jnp.int32)
        # Causal token o-1 summarizes exactly observations 1:o, so it is the direct AdaLN signal.
        conditioning = observation_contexts[count - 1]
        particles = _linear_tokens(self.particle_in, current_embeddings)
        for block in self.blocks:
            particles = block(particles, conditioning)
        return self.output_head(particles, current_theta, theta_size)


def posterior_hidden_and_output(
    posterior_transformer: CrossAttentionPosteriorTransformer | AdaLNPosteriorTransformer,
    current_embeddings: Array,
    current_theta: Array,
    observation_contexts: Array,
    observation_count: Array,
    theta_size: Array,
) -> tuple[Array, Array]:
    """Run the pretrained posterior backbone ONCE and return both hidden particles and cloud.

    This is the key hook for the joint ALINE-style regime.  The original Bayes Transport only
    exposed the physical posterior cloud, but its acquisition head needs the final shared particle
    features too.  We deliberately reproduce the exact CrossAttention/AdaLN computation and then
    call the original physical output head, so the posterior cloud and acquisition features come
    from the same posterior-Transformer pass.
    """
    count = jnp.clip(observation_count, 1, observation_contexts.shape[0]).astype(jnp.int32)

    if isinstance(posterior_transformer, CrossAttentionPosteriorTransformer):
        def branch_for(prefix_length: int):
            def transport(args):
                embeddings, theta_particles, full_memory = args
                memory = full_memory[:prefix_length]
                particles = _linear_tokens(posterior_transformer.particle_in, embeddings)
                for block in posterior_transformer.blocks:
                    particles = block(particles, memory)
                next_theta = posterior_transformer.output_head(
                    particles, theta_particles, theta_size
                )
                return particles, next_theta
            return transport

        branches = tuple(
            branch_for(prefix_length)
            for prefix_length in range(1, observation_contexts.shape[0] + 1)
        )
        return jax.lax.switch(
            count - 1,
            branches,
            (current_embeddings, current_theta, observation_contexts),
        )

    conditioning = observation_contexts[count - 1]
    particles = _linear_tokens(posterior_transformer.particle_in, current_embeddings)
    for block in posterior_transformer.blocks:
        particles = block(particles, conditioning)
    next_theta = posterior_transformer.output_head(particles, current_theta, theta_size)
    return particles, next_theta


def posterior_hidden_only(
    posterior_transformer: CrossAttentionPosteriorTransformer | AdaLNPosteriorTransformer,
    current_embeddings: Array,
    observation_contexts: Array,
    observation_count: Array,
) -> Array:
    """Return posterior-backbone particle features without applying the physical cloud head."""
    count = jnp.clip(observation_count, 1, observation_contexts.shape[0]).astype(jnp.int32)
    if isinstance(posterior_transformer, CrossAttentionPosteriorTransformer):
        def branch_for(prefix_length: int):
            def transform(args):
                embeddings, full_memory = args
                memory = full_memory[:prefix_length]
                particles = _linear_tokens(posterior_transformer.particle_in, embeddings)
                for block in posterior_transformer.blocks:
                    particles = block(particles, memory)
                return particles
            return transform
        branches = tuple(
            branch_for(prefix_length)
            for prefix_length in range(1, observation_contexts.shape[0] + 1)
        )
        return jax.lax.switch(count - 1, branches, (current_embeddings, observation_contexts))

    conditioning = observation_contexts[count - 1]
    particles = _linear_tokens(posterior_transformer.particle_in, current_embeddings)
    for block in posterior_transformer.blocks:
        particles = block(particles, conditioning)
    return particles


#%% 4f) End-to-end amortized model with reusable sequential recurrence
class SequentialBayesModel(eqx.Module):
    """Direct amortized transport plus a reusable repeated-Bayes rollout.

    `predict_prefixes` remains the historical direct one-step path. `_transport_compact_with_contexts`
    is the shared Bayes-update cell used by all modes. `__call__` performs a recurrent rollout in
    which observation block t is used exactly at update t; it is used by sequential evaluation.
    DEQ training wraps the same cell in an augmented-trajectory equilibrium. Drifting training is
    deliberately one-step and therefore uses `predict_prefixes`, not this recurrent `__call__`.
    """

    observation_embedder: ObservationDimensionEmbedder | FixedShapeObservationEmbedder
    likelihood_embedder: LikelihoodSequenceEmbedder
    theta_embedder: ThetaDimensionEmbedder | FixedShapeThetaEmbedder
    posterior_transformer: CrossAttentionPosteriorTransformer | AdaLNPosteriorTransformer

    min_observations: int = eqx.field(static=True)
    max_observations: int = eqx.field(static=True)
    conditioning_type: str = eqx.field(static=True)
    fixed_shape: bool = eqx.field(static=True)
    fixed_shape_learned_projection: bool = eqx.field(static=True)
    particle_input_dim: int = eqx.field(static=True)
    observation_input_dim: int = eqx.field(static=True)
    observation_context_dim: int = eqx.field(static=True)
    single_observation_direct: bool = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        observation_key, likelihood_key, theta_key, posterior_key = jax.random.split(key, 4)
        self.fixed_shape = (
            cfg.min_num_sources == cfg.max_num_sources
            and cfg.min_source_dim == cfg.max_source_dim
        )
        self.min_observations = int(cfg.min_observations_per_step)
        self.max_observations = int(cfg.max_observations_per_step)
        self.single_observation_direct = (
            self.min_observations == 1 and self.max_observations == 1
        )
        self.fixed_shape_learned_projection = bool(
            self.fixed_shape and cfg.fixed_shape_learned_projection
        )
        if self.fixed_shape:
            # Bypass the full dimension aggregators. Optionally use exactly one learned linear
            # projection for each fixed-shape input so both paths live in embedding_dim.
            self.observation_embedder = FixedShapeObservationEmbedder(
                cfg, key=observation_key
            )
            self.theta_embedder = FixedShapeThetaEmbedder(cfg, key=theta_key)
            self.particle_input_dim = self.theta_embedder.output_dim
            self.observation_input_dim = self.observation_embedder.output_dim
        else:
            self.particle_input_dim = cfg.embedding_dim
            self.observation_input_dim = cfg.embedding_dim
            self.observation_embedder = ObservationDimensionEmbedder(
                cfg, key=observation_key
            )
            self.theta_embedder = ThetaDimensionEmbedder(cfg, key=theta_key)
        self.likelihood_embedder = LikelihoodSequenceEmbedder(
            cfg,
            key=likelihood_key,
            input_dim=self.observation_input_dim,
        )
        # In the single-observation simplification this equals observation_input_dim; otherwise
        # it is the configured likelihood_hidden_dim produced by the causal likelihood Transformer.
        self.observation_context_dim = self.likelihood_embedder.hidden_dim
        self.conditioning_type = str(cfg.posterior_conditioning)
        if cfg.posterior_conditioning == "cross_attention":
            self.posterior_transformer = CrossAttentionPosteriorTransformer(
                cfg,
                key=posterior_key,
                particle_input_dim=self.particle_input_dim,
                conditioning_dim=self.observation_context_dim,
            )
        elif cfg.posterior_conditioning == "adaln":
            self.posterior_transformer = AdaLNPosteriorTransformer(
                cfg,
                key=posterior_key,
                particle_input_dim=self.particle_input_dim,
                conditioning_dim=self.observation_context_dim,
            )
        else:  # validate_config should already catch this.
            raise ValueError(f"Unsupported posterior_conditioning={cfg.posterior_conditioning!r}")

    def _encode_observation_block(
        self,
        observations: Array,          # [Omax,Dmax+1]
        num_sources: Array,
        theta_size: Array,
    ) -> Array:
        pair_embeddings = jax.vmap(
            lambda observation: self.observation_embedder(
                observation, num_sources, theta_size
            )
        )(observations)
        # With Omin=Omax=1 this is an identity map with zero learned parameters.  The resulting
        # token is projected only inside the selected Posterior Transformer conditioning path.
        return self.likelihood_embedder(pair_embeddings)

    def _compact_reference_cloud(
        self,
        particles: Array,             # [N,Smax,Dmax]
        num_sources: Array,
        theta_size: Array,
    ) -> Array:
        if self.theta_embedder.canonicalize:
            particles = jax.vmap(
                lambda theta: canonicalize_padded_sources_jax(theta, num_sources)
            )(particles)
        return jax.vmap(
            lambda theta: compact_theta_jax(theta, num_sources, theta_size)
        )(particles)                                                          # [N,Kmax]

    def _embed_compact_cloud(
        self,
        compact_particles: Array,
        num_sources: Array,
        theta_size: Array,
        max_num_sources: int,
        max_source_dim: int,
    ) -> Array:
        padded = jax.vmap(
            lambda theta: padded_theta_jax(
                theta,
                num_sources,
                theta_size,
                max_num_sources,
                max_source_dim,
            )
        )(compact_particles)
        return jax.vmap(
            lambda theta: self.theta_embedder(theta, num_sources, theta_size)
        )(padded)

    def _canonicalize_compact_output(
        self,
        compact_particles: Array,
        num_sources: Array,
        theta_size: Array,
        max_num_sources: int,
        max_source_dim: int,
    ) -> Array:
        if not self.theta_embedder.canonicalize:
            return compact_particles
        padded = jax.vmap(
            lambda theta: padded_theta_jax(
                theta,
                num_sources,
                theta_size,
                max_num_sources,
                max_source_dim,
            )
        )(compact_particles)
        padded = jax.vmap(
            lambda theta: canonicalize_padded_sources_jax(theta, num_sources)
        )(padded)
        return jax.vmap(
            lambda theta: compact_theta_jax(theta, num_sources, theta_size)
        )(padded)

    def _transport_compact_with_contexts(
        self,
        current_theta: Array,          # [N,Kmax]
        observation_contexts: Array,
        observation_count: Array,
        num_sources: Array,
        theta_size: Array,
        max_num_sources: int,
        max_source_dim: int,
    ) -> Array:
        """Apply one Bayes-map step to an already compact cloud using precomputed contexts."""
        current_embeddings = self._embed_compact_cloud(
            current_theta,
            num_sources,
            theta_size,
            max_num_sources,
            max_source_dim,
        )
        next_theta = self.posterior_transformer(
            current_embeddings,
            current_theta,
            observation_contexts,
            observation_count,
            theta_size,
        )
        return self._canonicalize_compact_output(
            next_theta,
            num_sources,
            theta_size,
            max_num_sources,
            max_source_dim,
        )

    def predict_prefixes(
        self,
        prior_particles: Array,       # [N,Smax,Dmax]
        observations: Array,          # [Omax,Dmax+1]
        num_sources: Array,
        theta_size: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Non-sequential training path: direct prior -> posterior for EVERY prefix.

        All prefix transports share the exact same supplied input cloud.  With Omin=Omax=1,
        the single observation bypasses likelihood self-attention and enters the Posterior
        Transformer directly; otherwise the causal observation Transformer is evaluated once.
        The Posterior Transformer is vmapped over o=Omin,...,Omax as requested.  No output prefix
        is fed into another prefix.
        """
        prior_theta = self._compact_reference_cloud(prior_particles, num_sources, theta_size)
        prior_embeddings = self._embed_compact_cloud(
            prior_theta,
            num_sources,
            theta_size,
            prior_particles.shape[-2],
            prior_particles.shape[-1],
        )
        observation_contexts = self._encode_observation_block(
            observations, num_sources, theta_size
        )
        prefix_counts = jnp.arange(
            self.min_observations, self.max_observations + 1, dtype=jnp.int32
        )

        def direct_transport(observation_count: Array) -> Array:
            transported = self.posterior_transformer(
                prior_embeddings,
                prior_theta,
                observation_contexts,
                observation_count,
                theta_size,
            )
            return self._canonicalize_compact_output(
                transported,
                num_sources,
                theta_size,
                prior_particles.shape[-2],
                prior_particles.shape[-1],
            )

        posterior_by_prefix = jax.vmap(direct_transport)(prefix_counts)   # [P,N,Kmax]
        return posterior_by_prefix, observation_contexts, prior_theta, prefix_counts

    def __call__(
        self,
        prior_particles: Array,       # [N,Smax,Dmax]
        observations: Array,          # [T,Omax,Dmax+1]
        observation_count: Array,     # scalar used for each evaluation update
        num_sources: Array,
        theta_size: Array,
    ) -> tuple[Array, Array, Array]:
        """Sequential rollout: feed each output cloud to the next DISTINCT observation block."""
        prior_theta = self._compact_reference_cloud(prior_particles, num_sources, theta_size)
        observation_contexts = jax.vmap(
            lambda block: self._encode_observation_block(block, num_sources, theta_size)
        )(observations)  # final width is observation_input_dim in bypass, likelihood_hidden_dim otherwise

        def scan_step(current_theta: Array, contexts: Array):
            current_embeddings = self._embed_compact_cloud(
                current_theta,
                num_sources,
                theta_size,
                prior_particles.shape[-2],
                prior_particles.shape[-1],
            )
            next_theta = self.posterior_transformer(
                current_embeddings,
                current_theta,
                contexts,
                observation_count,
                theta_size,
            )
            next_theta = self._canonicalize_compact_output(
                next_theta,
                num_sources,
                theta_size,
                prior_particles.shape[-2],
                prior_particles.shape[-1],
            )
            return next_theta, next_theta

        _, posterior_sequence = jax.lax.scan(scan_step, prior_theta, observation_contexts)
        return posterior_sequence, observation_contexts, prior_theta

#%% 5) Load the pretrained Bayes Transport checkpoint
def find_bayes_checkpoint(cfg: PolicyConfig) -> Path:
    artifact_dir = Path(cfg.bayes_artifact_dir).expanduser().resolve()
    requested = artifact_dir / cfg.bayes_checkpoint
    if requested.is_file():
        return requested
    fallbacks = [artifact_dir / "model_best.eqx", artifact_dir / "model_last.eqx"]
    fallbacks.extend(sorted(artifact_dir.glob("model_epoch_*.eqx"), reverse=True))
    for candidate in fallbacks:
        if candidate.is_file():
            print(f"Requested checkpoint not found; falling back to {candidate.name}")
            return candidate
    raise FileNotFoundError(
        f"No Bayes-Transport checkpoint found in {artifact_dir}. "
        "Run this script from the completed Bayes-Transport run directory."
    )


def load_bayes_model(path: str | Path, cfg: BayesTransportConfig) -> SequentialBayesModel:
    skeleton = SequentialBayesModel(cfg, key=jax.random.key(0))
    try:
        return eqx.tree_deserialise_leaves(Path(path), skeleton)
    except Exception as exc:
        raise RuntimeError(
            "Could not deserialize the Bayes-Transport checkpoint with the recovered architecture. "
            f"Checkpoint: {path}; archived source: {BAYES_SOURCE_SCRIPT}; recovered config: {asdict(cfg)}"
        ) from exc


#%% 6) Policy Transformer: cloud/history cross-attention followed by a set-valued design decoder
class MaskedSelfAttentionBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        attn_key, ff1_key, ff2_key = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=attn_key,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=ff2_key)

    def __call__(self, tokens: Array, valid: Array) -> Array:
        # Every row may attend only to valid memory keys.  Invalid query rows are zeroed after
        # each residual branch so padded history cannot become an information-carrying token.
        mask = jnp.broadcast_to(valid[None, :], (tokens.shape[0], tokens.shape[0]))
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h, mask=mask)
        tokens = jnp.where(valid[:, None], tokens, 0.0)
        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        tokens = tokens + _linear_tokens(self.ff_out, h)
        return jnp.where(valid[:, None], tokens, 0.0)


class CloudHistoryCrossBlock(eqx.Module):
    """Self-attend within the cloud, then cross-attend cloud particles to observation history."""
    cloud_norm: eqx.nn.LayerNorm
    history_norm: eqx.nn.LayerNorm
    cross_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    cloud_attention: eqx.nn.MultiheadAttention
    cross_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.cloud_norm = eqx.nn.LayerNorm(dim)
        self.history_norm = eqx.nn.LayerNorm(dim)
        self.cross_norm = eqx.nn.LayerNorm(dim)
        self.ff_norm = eqx.nn.LayerNorm(dim)
        self.cloud_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=k1,
        )
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=k2,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=k3)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=k4)

    def __call__(self, cloud: Array, history: Array, history_valid: Array) -> Array:
        h = _layernorm_tokens(self.cloud_norm, cloud)
        cloud = cloud + self.cloud_attention(h, h, h)
        q = _layernorm_tokens(self.cross_norm, cloud)
        memory = _layernorm_tokens(self.history_norm, history)
        mask = jnp.broadcast_to(history_valid[None, :], (cloud.shape[0], history.shape[0]))
        cloud = cloud + self.cross_attention(q, memory, memory, mask=mask)
        h = _layernorm_tokens(self.ff_norm, cloud)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return cloud + _linear_tokens(self.ff_out, h)


class DesignDecoderBlock(eqx.Module):
    """Set decoder: design-query self-attention + cross-attention to fused belief/history memory."""
    query_norm: eqx.nn.LayerNorm
    memory_norm: eqx.nn.LayerNorm
    cross_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    cross_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.query_norm = eqx.nn.LayerNorm(dim)
        self.memory_norm = eqx.nn.LayerNorm(dim)
        self.cross_norm = eqx.nn.LayerNorm(dim)
        self.ff_norm = eqx.nn.LayerNorm(dim)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=k1,
        )
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=k2,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=k3)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=k4)

    def __call__(self, queries: Array, memory: Array, memory_valid: Array) -> Array:
        h = _layernorm_tokens(self.query_norm, queries)
        queries = queries + self.self_attention(h, h, h)
        q = _layernorm_tokens(self.cross_norm, queries)
        mem = _layernorm_tokens(self.memory_norm, memory)
        mask = jnp.broadcast_to(memory_valid[None, :], (queries.shape[0], memory.shape[0]))
        queries = queries + self.cross_attention(q, mem, mem, mask=mask)
        h = _layernorm_tokens(self.ff_norm, queries)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        return queries + _linear_tokens(self.ff_out, h)


class ALINEAcquisitionBlock(eqx.Module):
    """ALINE-style query processing: query self-attn, query->context, query->target, FFN.

    ALINE's architecture uses a context set for acquired observations, a target set for the
    inference target, and a query set for candidate acquisitions.  Here the target set is the
    current/post-update posterior-particle representation and the query set is a learned set of K
    continuous design tokens.  The final acquisition head differs from ALINE's pool softmax because
    this project emits a continuous design set.
    """
    query_norm: eqx.nn.LayerNorm
    context_query_norm: eqx.nn.LayerNorm
    context_norm: eqx.nn.LayerNorm
    target_query_norm: eqx.nn.LayerNorm
    target_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    context_attention: eqx.nn.MultiheadAttention
    target_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, ff_dim: int, *, key: Array):
        k_self, k_ctx, k_tgt, k_ff1, k_ff2 = jax.random.split(key, 5)
        self.query_norm = eqx.nn.LayerNorm(dim)
        self.context_query_norm = eqx.nn.LayerNorm(dim)
        self.context_norm = eqx.nn.LayerNorm(dim)
        self.target_query_norm = eqx.nn.LayerNorm(dim)
        self.target_norm = eqx.nn.LayerNorm(dim)
        self.ff_norm = eqx.nn.LayerNorm(dim)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=k_self,
        )
        self.context_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=k_ctx,
        )
        self.target_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=k_tgt,
        )
        self.ff_in = eqx.nn.Linear(dim, ff_dim, key=k_ff1)
        self.ff_out = eqx.nn.Linear(ff_dim, dim, key=k_ff2)

    def __call__(
        self, queries: Array, context: Array, context_valid: Array, target: Array
    ) -> Array:
        q = _layernorm_tokens(self.query_norm, queries)
        queries = queries + self.self_attention(q, q, q)

        q = _layernorm_tokens(self.context_query_norm, queries)
        ctx = _layernorm_tokens(self.context_norm, context)
        ctx_mask = jnp.broadcast_to(context_valid[None, :], (queries.shape[0], context.shape[0]))
        queries = queries + self.context_attention(q, ctx, ctx, mask=ctx_mask)

        q = _layernorm_tokens(self.target_query_norm, queries)
        tgt = _layernorm_tokens(self.target_norm, target)
        queries = queries + self.target_attention(q, tgt, tgt)

        h = _layernorm_tokens(self.ff_norm, queries)
        h = jax.nn.relu(_linear_tokens(self.ff_in, h))
        return queries + _linear_tokens(self.ff_out, h)


class AcquisitionMLPHead(eqx.Module):
    """Two-layer ALINE-style acquisition MLP applied independently to each design query token."""
    hidden: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, in_dim: int, width: int, out_dim: int, *, key: Array, zero_output: bool = False,
                 output_bias: float = 0.0):
        k1, k2 = jax.random.split(key, 2)
        self.hidden = eqx.nn.Linear(in_dim, width, key=k1)
        output = eqx.nn.Linear(width, out_dim, key=k2)
        if zero_output:
            output = eqx.tree_at(lambda x: x.weight, output, jnp.zeros_like(output.weight))
            output = eqx.tree_at(
                lambda x: x.bias, output, jnp.full_like(output.bias, float(output_bias))
            )
        self.output = output

    def __call__(self, tokens: Array) -> Array:
        h = jax.nn.relu(_linear_tokens(self.hidden, tokens))
        return _linear_tokens(self.output, h)


def _policy_compact_cloud(
    cloud_padded: Array,
    theta_embedder: ThetaDimensionEmbedder | FixedShapeThetaEmbedder,
    num_sources: Array,
    theta_size: Array,
) -> Array:
    """Same physical compact representation used by SequentialBayesModel."""
    if theta_embedder.canonicalize:
        cloud_padded = jax.vmap(
            lambda theta: canonicalize_padded_sources_jax(theta, num_sources)
        )(cloud_padded)
    return jax.vmap(
        lambda theta: compact_theta_jax(theta, num_sources, theta_size)
    )(cloud_padded)


def _policy_pad_cloud(
    compact_cloud: Array,
    theta_embedder: ThetaDimensionEmbedder | FixedShapeThetaEmbedder,
    num_sources: Array,
    theta_size: Array,
    max_num_sources: int,
    max_source_dim: int,
) -> Array:
    """Canonicalize (if requested) and return padded [N,Smax,Dmax] cloud storage."""
    padded = jax.vmap(
        lambda theta: padded_theta_jax(
            theta, num_sources, theta_size, max_num_sources, max_source_dim
        )
    )(compact_cloud)
    if theta_embedder.canonicalize:
        padded = jax.vmap(
            lambda theta: canonicalize_padded_sources_jax(theta, num_sources)
        )(padded)
    return padded


def _embedder_output_dim(
    embedder: ThetaDimensionEmbedder | FixedShapeThetaEmbedder |
              ObservationDimensionEmbedder | FixedShapeObservationEmbedder,
    bt_cfg: BayesTransportConfig,
) -> int:
    return int(embedder.output_dim) if hasattr(embedder, "output_dim") else int(bt_cfg.embedding_dim)


def _exact_fixed_shape_match(bt_cfg: BayesTransportConfig, cfg: PolicyConfig) -> bool:
    return (
        policy_is_fixed_shape(cfg)
        and bt_cfg.min_num_sources == bt_cfg.max_num_sources == cfg.min_num_sources
        and bt_cfg.min_source_dim == bt_cfg.max_source_dim == cfg.min_source_dim
    )


def _make_policy_theta_embedder(
    bayes_model: SequentialBayesModel, bt_cfg: BayesTransportConfig, cfg: PolicyConfig, *, key: Array
) -> tuple[ThetaDimensionEmbedder | FixedShapeThetaEmbedder, bool]:
    """Return policy prior embedder plus whether it is literally reused from Bayes Transport."""
    if policy_is_fixed_shape(cfg):
        # Preserve the fixed-shape rule from the previous revision: the SEPARATE policy learns
        # its own linear projection. Reuse of the checkpoint's fixed linear map is reserved for
        # the explicit all-four JOINT setting, where component sharing is the point of the regime.
        if joint_aline_mode(cfg) and cfg.reuse_prior_embedder and _exact_fixed_shape_match(bt_cfg, cfg):
            return bayes_model.theta_embedder, True
        fixed_cfg = replace(
            bt_cfg,
            min_num_sources=cfg.min_num_sources, max_num_sources=cfg.max_num_sources,
            min_source_dim=cfg.min_source_dim, max_source_dim=cfg.max_source_dim,
            heldout_shapes=(), fixed_shape_learned_projection=True,
        )
        return FixedShapeThetaEmbedder(fixed_cfg, key=key), False
    if cfg.reuse_prior_embedder:
        return bayes_model.theta_embedder, True
    return ThetaDimensionEmbedder(bt_cfg, key=key), False


def _make_policy_observation_embedder(
    bayes_model: SequentialBayesModel, bt_cfg: BayesTransportConfig, cfg: PolicyConfig, *, key: Array
) -> tuple[ObservationDimensionEmbedder | FixedShapeObservationEmbedder, bool]:
    """Return policy observation embedder plus whether it is literally reused from Bayes Transport."""
    if policy_is_fixed_shape(cfg):
        if joint_aline_mode(cfg) and cfg.reuse_observation_embedder and _exact_fixed_shape_match(bt_cfg, cfg):
            return bayes_model.observation_embedder, True
        fixed_cfg = replace(
            bt_cfg,
            min_num_sources=cfg.min_num_sources, max_num_sources=cfg.max_num_sources,
            min_source_dim=cfg.min_source_dim, max_source_dim=cfg.max_source_dim,
            heldout_shapes=(), fixed_shape_learned_projection=True,
        )
        return FixedShapeObservationEmbedder(fixed_cfg, key=key), False
    if cfg.reuse_observation_embedder:
        return bayes_model.observation_embedder, True
    return ObservationDimensionEmbedder(bt_cfg, key=key), False


class SeparateDesignPolicy(eqx.Module):
    """Existing separate cloud/history policy, now with four independent Bayes-reuse switches."""
    theta_embedder: ThetaDimensionEmbedder | FixedShapeThetaEmbedder
    observation_embedder: ObservationDimensionEmbedder | FixedShapeObservationEmbedder
    likelihood_embedder: LikelihoodSequenceEmbedder | None
    posterior_transformer: CrossAttentionPosteriorTransformer | AdaLNPosteriorTransformer | None

    theta_projection: eqx.nn.Linear
    history_projection: eqx.nn.Linear
    posterior_feature_projection: eqx.nn.Linear | None
    history_blocks: tuple[MaskedSelfAttentionBlock, ...]
    cloud_history_blocks: tuple[CloudHistoryCrossBlock, ...]
    decoder_blocks: tuple[DesignDecoderBlock, ...]
    final_norm: eqx.nn.LayerNorm
    mean_head: eqx.nn.Linear
    log_std_head: eqx.nn.Linear

    empty_history_token: Array
    history_position: Array
    design_query_tokens: Array
    step_embedding: Array

    hidden_dim: int = eqx.field(static=True)
    max_history: int = eqx.field(static=True)
    designs_per_step: int = eqx.field(static=True)
    max_source_dim: int = eqx.field(static=True)
    fixed_shape_input: bool = eqx.field(static=True)
    joint_aline_mode: bool = eqx.field(static=True)
    uses_reused_prior: bool = eqx.field(static=True)
    uses_reused_likelihood: bool = eqx.field(static=True)
    uses_reused_observation: bool = eqx.field(static=True)
    uses_reused_posterior: bool = eqx.field(static=True)

    def __init__(self, bayes_model: SequentialBayesModel, bt_cfg: BayesTransportConfig,
                 cfg: PolicyConfig, *, key: Array):
        keys = iter(jax.random.split(
            key, 32 + cfg.history_depth + cfg.cloud_history_depth + cfg.decoder_depth
        ))
        self.fixed_shape_input = policy_is_fixed_shape(cfg)
        self.joint_aline_mode = False

        self.theta_embedder, self.uses_reused_prior = _make_policy_theta_embedder(
            bayes_model, bt_cfg, cfg, key=next(keys)
        )
        self.observation_embedder, self.uses_reused_observation = _make_policy_observation_embedder(
            bayes_model, bt_cfg, cfg, key=next(keys)
        )
        self.likelihood_embedder = bayes_model.likelihood_embedder if cfg.reuse_likelihood_transformer else None
        self.posterior_transformer = bayes_model.posterior_transformer if cfg.reuse_posterior_transformer else None
        self.uses_reused_likelihood = self.likelihood_embedder is not None
        self.uses_reused_posterior = self.posterior_transformer is not None

        self.hidden_dim = int(cfg.policy_hidden_dim)
        self.max_history = int(cfg.experimental_budget * cfg.designs_per_step)
        self.designs_per_step = int(cfg.designs_per_step)
        self.max_source_dim = int(bt_cfg.max_source_dim)

        theta_input_dim = _embedder_output_dim(self.theta_embedder, bt_cfg)
        obs_input_dim = _embedder_output_dim(self.observation_embedder, bt_cfg)
        history_input_dim = (
            int(self.likelihood_embedder.hidden_dim)
            if self.likelihood_embedder is not None and not self.likelihood_embedder.bypass_single_observation
            else obs_input_dim
        )
        self.theta_projection = eqx.nn.Linear(theta_input_dim, self.hidden_dim, key=next(keys))
        self.history_projection = eqx.nn.Linear(history_input_dim, self.hidden_dim, key=next(keys))
        self.posterior_feature_projection = (
            eqx.nn.Linear(bt_cfg.hidden_dim, self.hidden_dim, key=next(keys))
            if self.posterior_transformer is not None else None
        )

        mlp_dim = cfg.policy_mlp_ratio * self.hidden_dim
        self.history_blocks = tuple(
            MaskedSelfAttentionBlock(self.hidden_dim, cfg.policy_heads, mlp_dim, key=next(keys))
            for _ in range(cfg.history_depth)
        )
        self.cloud_history_blocks = tuple(
            CloudHistoryCrossBlock(self.hidden_dim, cfg.policy_heads, mlp_dim, key=next(keys))
            for _ in range(cfg.cloud_history_depth)
        )
        self.decoder_blocks = tuple(
            DesignDecoderBlock(self.hidden_dim, cfg.policy_heads, mlp_dim, key=next(keys))
            for _ in range(cfg.decoder_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(self.hidden_dim)
        self.mean_head = eqx.nn.Linear(self.hidden_dim, self.max_source_dim, key=next(keys))
        self.log_std_head = eqx.nn.Linear(self.hidden_dim, self.max_source_dim, key=next(keys))

        init_std_log = float(math.log(max(cfg.initial_policy_std, 1e-5)))
        self.empty_history_token = 0.02 * jax.random.normal(next(keys), (self.hidden_dim,))
        self.history_position = 0.02 * jax.random.normal(
            next(keys), (self.max_history + 1, self.hidden_dim)
        )
        self.design_query_tokens = 0.02 * jax.random.normal(
            next(keys), (self.designs_per_step, self.hidden_dim)
        )
        self.step_embedding = 0.02 * jax.random.normal(
            next(keys), (cfg.experimental_budget, self.hidden_dim)
        )
        self.log_std_head = eqx.tree_at(
            lambda layer: layer.weight, self.log_std_head, jnp.zeros_like(self.log_std_head.weight)
        )
        self.log_std_head = eqx.tree_at(
            lambda layer: layer.bias, self.log_std_head,
            jnp.full_like(self.log_std_head.bias, init_std_log),
        )

    def _history_contexts(self, history: Array, history_valid: Array,
                          num_sources: Array, theta_size: Array) -> Array:
        obs_emb = jax.vmap(
            lambda obs: self.observation_embedder(obs, num_sources, theta_size)
        )(history)
        # The pretrained likelihood Transformer is causal. Processing the static padded history is
        # therefore safe for valid prefix tokens: they never attend to future padded positions.
        if self.likelihood_embedder is not None and not self.likelihood_embedder.bypass_single_observation:
            contexts = self.likelihood_embedder(obs_emb)
        else:
            contexts = obs_emb
        return jnp.where(history_valid[:, None], contexts, 0.0)

    def __call__(self, cloud_padded: Array, history: Array, history_valid: Array,
                 decision_index: Array, num_sources: Array, theta_size: Array) -> tuple[Array, Array]:
        cloud_emb = jax.vmap(
            lambda th: self.theta_embedder(th, num_sources, theta_size)
        )(cloud_padded)
        history_contexts = self._history_contexts(history, history_valid, num_sources, theta_size)

        # If the pretrained posterior Transformer is selected, use it as the cloud/history fusion
        # backbone. Otherwise retain the original policy-specific cloud->history cross-attention.
        if self.posterior_transformer is not None:
            count = jnp.sum(history_valid.astype(jnp.int32))
            def with_history(_):
                hidden = posterior_hidden_only(
                    self.posterior_transformer, cloud_emb, history_contexts, count
                )
                return _linear_tokens(self.posterior_feature_projection, hidden)
            def without_history(_):
                return _linear_tokens(self.theta_projection, cloud_emb)
            cloud = jax.lax.cond(count > 0, with_history, without_history, operand=None)
        else:
            cloud = _linear_tokens(self.theta_projection, cloud_emb)

        hist = _linear_tokens(self.history_projection, history_contexts)
        hist = jnp.where(history_valid[:, None], hist, 0.0)
        hist = jnp.concatenate([self.empty_history_token[None, :], hist], axis=0)
        valid = jnp.concatenate([jnp.ones((1,), dtype=bool), history_valid], axis=0)
        hist = hist + self.history_position
        hist = jnp.where(valid[:, None], hist, 0.0)
        for block in self.history_blocks:
            hist = block(hist, valid)

        if self.posterior_transformer is None:
            for block in self.cloud_history_blocks:
                cloud = block(cloud, hist, valid)

        t = jnp.clip(decision_index, 0, self.step_embedding.shape[0] - 1).astype(jnp.int32)
        queries = self.design_query_tokens + self.step_embedding[t][None, :]
        memory = jnp.concatenate([cloud, hist], axis=0)
        memory_valid = jnp.concatenate([jnp.ones((cloud.shape[0],), dtype=bool), valid], axis=0)
        for block in self.decoder_blocks:
            queries = block(queries, memory, memory_valid)
        queries = _layernorm_tokens(self.final_norm, queries)
        return _linear_tokens(self.mean_head, queries), _linear_tokens(self.log_std_head, queries)


class JointALINEDesignPolicy(eqx.Module):
    """Integrated ALINE-style design policy using ALL four pretrained Bayes components.

    The Bayes observation embedder + causal likelihood Transformer form ALINE's context pathway.
    The Bayes prior embedder + posterior Transformer form the inference/target pathway. A continuous
    acquisition branch then lets learned query tokens attend to both context and target features.

    Crucially, ``update_and_propose`` performs the posterior Transformer only once: the same final
    particle features feed (a) the original physical posterior-cloud head and (b) the acquisition
    head for the next design set. This is the requested joint inference/acquisition call.
    """
    theta_embedder: ThetaDimensionEmbedder | FixedShapeThetaEmbedder
    likelihood_embedder: LikelihoodSequenceEmbedder
    observation_embedder: ObservationDimensionEmbedder | FixedShapeObservationEmbedder
    posterior_transformer: CrossAttentionPosteriorTransformer | AdaLNPosteriorTransformer

    context_projection: eqx.nn.Linear
    target_projection: eqx.nn.Linear
    acquisition_blocks: tuple[ALINEAcquisitionBlock, ...]
    final_norm: eqx.nn.LayerNorm
    mean_head: AcquisitionMLPHead
    log_std_head: AcquisitionMLPHead
    empty_context_token: Array
    design_query_tokens: Array
    step_embedding: Array

    hidden_dim: int = eqx.field(static=True)
    max_history: int = eqx.field(static=True)
    designs_per_step: int = eqx.field(static=True)
    max_source_dim: int = eqx.field(static=True)
    fixed_shape_input: bool = eqx.field(static=True)
    joint_aline_mode: bool = eqx.field(static=True)
    uses_reused_prior: bool = eqx.field(static=True)
    uses_reused_likelihood: bool = eqx.field(static=True)
    uses_reused_observation: bool = eqx.field(static=True)
    uses_reused_posterior: bool = eqx.field(static=True)

    def __init__(self, bayes_model: SequentialBayesModel, bt_cfg: BayesTransportConfig,
                 cfg: PolicyConfig, *, key: Array):
        if not joint_aline_mode(cfg):
            raise ValueError("JointALINEDesignPolicy requires all four Bayes reuse flags to be True.")
        keys = iter(jax.random.split(key, 16 + cfg.joint_aline_depth))
        self.theta_embedder = bayes_model.theta_embedder
        self.likelihood_embedder = bayes_model.likelihood_embedder
        self.observation_embedder = bayes_model.observation_embedder
        self.posterior_transformer = bayes_model.posterior_transformer
        self.uses_reused_prior = True
        self.uses_reused_likelihood = True
        self.uses_reused_observation = True
        self.uses_reused_posterior = True
        self.joint_aline_mode = True
        self.fixed_shape_input = policy_is_fixed_shape(cfg)

        self.hidden_dim = int(cfg.policy_hidden_dim)
        self.max_history = int(cfg.experimental_budget * cfg.designs_per_step)
        self.designs_per_step = int(cfg.designs_per_step)
        self.max_source_dim = int(bt_cfg.max_source_dim)

        obs_context_dim = int(self.likelihood_embedder.hidden_dim)
        self.context_projection = eqx.nn.Linear(obs_context_dim, self.hidden_dim, key=next(keys))
        self.target_projection = eqx.nn.Linear(bt_cfg.hidden_dim, self.hidden_dim, key=next(keys))
        self.acquisition_blocks = tuple(
            ALINEAcquisitionBlock(
                self.hidden_dim, cfg.joint_aline_heads, cfg.joint_aline_ff_dim, key=next(keys)
            ) for _ in range(cfg.joint_aline_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(self.hidden_dim)
        self.mean_head = AcquisitionMLPHead(
            self.hidden_dim, cfg.joint_aline_head_width, self.max_source_dim, key=next(keys)
        )
        self.log_std_head = AcquisitionMLPHead(
            self.hidden_dim, cfg.joint_aline_head_width, self.max_source_dim, key=next(keys),
            zero_output=True, output_bias=math.log(max(cfg.initial_policy_std, 1e-5)),
        )
        self.empty_context_token = 0.02 * jax.random.normal(next(keys), (self.hidden_dim,))
        self.design_query_tokens = 0.02 * jax.random.normal(
            next(keys), (self.designs_per_step, self.hidden_dim)
        )
        # +1 because update_and_propose computes the proposal for t+1 after decision t.
        self.step_embedding = 0.02 * jax.random.normal(
            next(keys), (cfg.experimental_budget + 1, self.hidden_dim)
        )

    def _encode_observation_tokens(self, observations: Array, num_sources: Array,
                                   theta_size: Array) -> Array:
        pair_embeddings = jax.vmap(
            lambda obs: self.observation_embedder(obs, num_sources, theta_size)
        )(observations)
        if self.likelihood_embedder.bypass_single_observation:
            # A strict O=1 checkpoint has no learned sequence Transformer. For a history with >1
            # entries the reusable context is simply the sequence of observation embeddings.
            return pair_embeddings
        return self.likelihood_embedder(pair_embeddings)

    def _encode_history(self, history: Array, history_valid: Array, num_sources: Array,
                        theta_size: Array) -> tuple[Array, Array]:
        contexts = self._encode_observation_tokens(history, num_sources, theta_size)
        contexts = jnp.where(history_valid[:, None], contexts, 0.0)
        projected = _linear_tokens(self.context_projection, contexts)
        projected = jnp.where(history_valid[:, None], projected, 0.0)
        projected = jnp.concatenate([self.empty_context_token[None, :], projected], axis=0)
        valid = jnp.concatenate([jnp.ones((1,), dtype=bool), history_valid], axis=0)
        return projected, valid

    def _acquisition(self, target_hidden: Array, history: Array, history_valid: Array,
                     decision_index: Array, num_sources: Array, theta_size: Array) -> tuple[Array, Array]:
        target = _linear_tokens(self.target_projection, target_hidden)
        context, context_valid = self._encode_history(
            history, history_valid, num_sources, theta_size
        )
        t = jnp.clip(decision_index, 0, self.step_embedding.shape[0] - 1).astype(jnp.int32)
        queries = self.design_query_tokens + self.step_embedding[t][None, :]
        for block in self.acquisition_blocks:
            queries = block(queries, context, context_valid, target)
        queries = _layernorm_tokens(self.final_norm, queries)
        return self.mean_head(queries), self.log_std_head(queries)

    def __call__(self, cloud_padded: Array, history: Array, history_valid: Array,
                 decision_index: Array, num_sources: Array, theta_size: Array) -> tuple[Array, Array]:
        """Proposal-only call used for the initial design and auxiliary inference fine-tuning.

        During the main joint rollout, all later proposals come from ``update_and_propose`` so the
        posterior cloud and next design set share one posterior-Transformer pass.
        """
        cloud_embeddings = jax.vmap(
            lambda theta: self.theta_embedder(theta, num_sources, theta_size)
        )(cloud_padded)
        count = jnp.sum(history_valid.astype(jnp.int32))
        raw_contexts = self._encode_observation_tokens(history, num_sources, theta_size)

        def conditioned(_):
            return posterior_hidden_only(
                self.posterior_transformer, cloud_embeddings, raw_contexts, count
            )
        def unconditioned(_):
            return _linear_tokens(self.posterior_transformer.particle_in, cloud_embeddings)
        target_hidden = jax.lax.cond(count > 0, conditioned, unconditioned, operand=None)
        return self._acquisition(
            target_hidden, history, history_valid, decision_index, num_sources, theta_size
        )

    def update_and_propose(
        self,
        cloud_padded: Array,
        new_observations: Array,
        full_history: Array,
        history_valid: Array,
        next_decision_index: Array,
        num_sources: Array,
        theta_size: Array,
    ) -> tuple[Array, Array, Array]:
        """ONE shared posterior pass -> posterior cloud + next continuous design-set parameters."""
        compact = _policy_compact_cloud(
            cloud_padded, self.theta_embedder, num_sources, theta_size
        )
        cloud_embeddings = jax.vmap(
            lambda theta: self.theta_embedder(theta, num_sources, theta_size)
        )(cloud_padded)
        block_contexts = self._encode_observation_tokens(
            full_history, num_sources, theta_size
        )
        count = jnp.sum(history_valid.astype(jnp.int32))
        target_hidden, next_compact = posterior_hidden_and_output(
            self.posterior_transformer,
            cloud_embeddings,
            compact,
            block_contexts,
            count,
            theta_size,
        )
        next_cloud = _policy_pad_cloud(
            next_compact, self.theta_embedder, num_sources, theta_size,
            cloud_padded.shape[-2], cloud_padded.shape[-1],
        )
        mean_raw, log_std_raw = self._acquisition(
            target_hidden, full_history, history_valid, next_decision_index,
            num_sources, theta_size,
        )
        return next_cloud, mean_raw, log_std_raw


PolicyModule = SeparateDesignPolicy | JointALINEDesignPolicy


def count_parameters(module: eqx.Module) -> int:
    return int(sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))
        if x is not None
    ))


def restore_frozen_policy_components(
    new_policy: PolicyModule,
    old_policy: PolicyModule,
    cfg: PolicyConfig,
) -> PolicyModule:
    """Undo optimizer/AdamW movement for policy components configured as frozen."""
    # Reused Bayes modules are frozen under the policy objective by default.
    if not cfg.train_reused_components_with_policy:
        for attr, used in (
            ("theta_embedder", new_policy.uses_reused_prior),
            ("likelihood_embedder", new_policy.uses_reused_likelihood),
            ("observation_embedder", new_policy.uses_reused_observation),
            ("posterior_transformer", new_policy.uses_reused_posterior),
        ):
            if used:
                new_policy = eqx.tree_at(
                    lambda p, attr=attr: getattr(p, attr), new_policy, getattr(old_policy, attr)
                )

    # Fresh heterogeneous embedders may also be intentionally frozen. Fresh fixed-shape linear
    # projections always train because they are the complete fixed-shape representation interface.
    if not new_policy.fixed_shape_input and not cfg.train_policy_embedders:
        if not new_policy.uses_reused_prior:
            new_policy = eqx.tree_at(
                lambda p: p.theta_embedder, new_policy, old_policy.theta_embedder
            )
        if not new_policy.uses_reused_observation:
            new_policy = eqx.tree_at(
                lambda p: p.observation_embedder, new_policy, old_policy.observation_embedder
            )
    return new_policy


def sync_reused_policy_components_from_bayes(
    policy: PolicyModule,
    bayes_model: SequentialBayesModel,
    cfg: PolicyConfig,
) -> PolicyModule:
    """Refresh reused policy-side Bayes components after inference fine-tuning/reload."""
    if not cfg.sync_reused_components_after_bayes_finetune:
        return policy
    for attr, used in (
        ("theta_embedder", policy.uses_reused_prior),
        ("likelihood_embedder", policy.uses_reused_likelihood),
        ("observation_embedder", policy.uses_reused_observation),
        ("posterior_transformer", policy.uses_reused_posterior),
    ):
        if used:
            policy = eqx.tree_at(
                lambda p, attr=attr: getattr(p, attr), policy, getattr(bayes_model, attr)
            )
    return policy


#%% 7) Empirical-cloud rewards and diagnostic metrics
def empirical_energy_score_terms_single(
    particle_theta: Array, target_theta: Array, theta_size: Array
) -> tuple[Array, Array, Array]:
    """Exact finite-cloud energy score copied from Bayes Transport."""
    valid = (jnp.arange(particle_theta.shape[-1]) < theta_size).astype(particle_theta.dtype)
    target_sq = jnp.sum((particle_theta - target_theta[None, :]) ** 2 * valid[None, :], axis=-1)
    attraction = jnp.mean(jnp.sqrt(target_sq + 1e-12))
    differences = particle_theta[:, None, :] - particle_theta[None, :, :]
    pair_sq = jnp.sum(differences**2 * valid[None, None, :], axis=-1)
    off_diagonal = 1.0 - jnp.eye(particle_theta.shape[0], dtype=particle_theta.dtype)
    repulsion = jnp.sum(jnp.sqrt(pair_sq + 1e-12) * off_diagonal) / (particle_theta.shape[0] ** 2)
    return attraction - 0.5 * repulsion, attraction, repulsion


def posterior_mean_rmse_single(particle_theta: Array, target_theta: Array, theta_size: Array) -> Array:
    valid = (jnp.arange(particle_theta.shape[-1]) < theta_size).astype(particle_theta.dtype)
    sq = (jnp.mean(particle_theta, axis=0) - target_theta) ** 2
    return jnp.sqrt(jnp.sum(sq * valid) / jnp.maximum(theta_size, 1))


def posterior_spread_single(particle_theta: Array, theta_size: Array) -> Array:
    valid = (jnp.arange(particle_theta.shape[-1]) < theta_size).astype(particle_theta.dtype)
    variance = jnp.var(particle_theta, axis=0)
    return jnp.sum(variance * valid) / jnp.maximum(theta_size, 1)


def compact_truth_jax(theta_true: Array, num_sources: Array, theta_size: Array, cfg: BayesTransportConfig) -> Array:
    if cfg.canonicalize_particle_sources:
        theta_true = canonicalize_padded_sources_jax(theta_true, num_sources)
    return compact_theta_jax(theta_true, num_sources, theta_size)


def compact_cloud_jax(cloud: Array, num_sources: Array, theta_size: Array, cfg: BayesTransportConfig) -> Array:
    if cfg.canonicalize_particle_sources:
        cloud = jax.vmap(lambda th: canonicalize_padded_sources_jax(th, num_sources))(cloud)
    return jax.vmap(lambda th: compact_theta_jax(th, num_sources, theta_size))(cloud)


def cloud_metrics_jax(
    cloud: Array, theta_true: Array, num_sources: Array, theta_size: Array, cfg: BayesTransportConfig
) -> dict[str, Array]:
    compact_cloud = compact_cloud_jax(cloud, num_sources, theta_size, cfg)
    compact_truth = compact_truth_jax(theta_true, num_sources, theta_size, cfg)
    es, attraction, repulsion = empirical_energy_score_terms_single(compact_cloud, compact_truth, theta_size)
    return {
        "energy_score": es,
        "attraction": attraction,
        "repulsion": repulsion,
        "rmse": posterior_mean_rmse_single(compact_cloud, compact_truth, theta_size),
        "spread": posterior_spread_single(compact_cloud, theta_size),
    }


def reward_from_metrics(before: dict[str, Array], after: dict[str, Array], cfg: PolicyConfig) -> Array:
    # Lower attraction / lower full energy score means greater certainty near the true theta*.
    if cfg.reward_mode == "attraction":
        reward = before["attraction"] - after["attraction"]
    else:
        reward = before["energy_score"] - after["energy_score"]
    reward = cfg.reward_scale * reward
    if cfg.reward_clip is not None:
        reward = jnp.clip(reward, -float(cfg.reward_clip), float(cfg.reward_clip))
    return reward


#%% 8) Bounded Tanh-Gaussian policy sampling and multi-observation Bayes updates
def _active_design_mask(theta_size: Array, num_sources: Array, max_source_dim: int) -> Array:
    source_dim = theta_size // num_sources
    return jnp.arange(max_source_dim) < source_dim


def sample_design_set_from_raw(
    mean_raw: Array,
    log_std_raw: Array,
    key: Array,
    num_sources: Array,
    theta_size: Array,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
    *,
    deterministic: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    """Sample the bounded continuous design set from already-computed policy parameters."""
    log_std = jnp.clip(log_std_raw, math.log(cfg.min_policy_std), math.log(cfg.max_policy_std))
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean_raw.shape)
    latent = mean_raw if deterministic else mean_raw + std * eps

    if cfg.gradient_estimator == "reinforce" and not deterministic:
        latent_for_env = jax.lax.stop_gradient(latent)
    else:
        latent_for_env = latent

    unit = jnp.tanh(latent_for_env)
    center = 0.5 * (bt_cfg.design_low + bt_cfg.design_high)
    half_range = 0.5 * (bt_cfg.design_high - bt_cfg.design_low)
    designs = center + half_range * unit

    active = _active_design_mask(theta_size, num_sources, bt_cfg.max_source_dim)
    designs = jnp.where(active[None, :], designs, 0.0)
    mean_design = center + half_range * jnp.tanh(mean_raw)
    mean_design = jnp.where(active[None, :], mean_design, 0.0)

    if deterministic:
        joint_log_prob = jnp.asarray(0.0, dtype=designs.dtype)
    else:
        latent_lp_arg = jax.lax.stop_gradient(latent) if cfg.gradient_estimator == "reinforce" else latent
        unit_lp_arg = (
            jax.lax.stop_gradient(jnp.tanh(latent))
            if cfg.gradient_estimator == "reinforce" else jnp.tanh(latent)
        )
        base_log_prob = (
            -0.5 * ((latent_lp_arg - mean_raw) / std) ** 2
            - log_std - 0.5 * math.log(2.0 * math.pi)
        )
        tanh_correction = -jnp.log(1.0 - unit_lp_arg**2 + 1e-6)
        joint_log_prob = jnp.sum((base_log_prob + tanh_correction) * active[None, :])

    latent_entropy = jnp.sum(
        (log_std + 0.5 * math.log(2.0 * math.pi * math.e)) * active[None, :]
    )
    mean_std = jnp.sum(std * active[None, :]) / jnp.maximum(
        jnp.sum(active) * cfg.designs_per_step, 1
    )
    return designs, joint_log_prob, latent_entropy, mean_design, mean_std


def sample_policy_design_set(
    policy: PolicyModule,
    cloud: Array,
    history: Array,
    history_valid: Array,
    decision_index: Array,
    num_sources: Array,
    theta_size: Array,
    key: Array,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
    *,
    deterministic: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    """Compute policy parameters from the ENTIRE history, then sample K bounded designs."""
    mean_raw, log_std_raw = policy(
        cloud, history, history_valid, decision_index, num_sources, theta_size
    )
    return sample_design_set_from_raw(
        mean_raw, log_std_raw, key, num_sources, theta_size, bt_cfg, cfg,
        deterministic=deterministic,
    )

def _bayes_update_observation_block(
    bayes_model: SequentialBayesModel,
    cloud_padded: Array,
    observations: Array,
    observation_count: int,
    num_sources: Array,
    theta_size: Array,
) -> Array:
    """Apply ONE Bayes-map update using an observation block.

    This is the policy-training counterpart of ``SequentialBayesModel.predict_prefixes``: the
    supplied observations are embedded together, the checkpoint's causal likelihood Transformer
    constructs their prefix contexts, and the Posterior Transformer consumes the context associated
    with ``observation_count``.  Consequently, for a checkpoint trained with O>1 observations per
    Bayes update, a set of K observations is *not* silently converted into K separate posterior maps.
    """
    if observation_count < 1:
        raise ValueError("observation_count must be >= 1.")
    if observations.shape[0] < observation_count:
        raise ValueError("Observation block is shorter than observation_count.")

    compact = bayes_model._compact_reference_cloud(cloud_padded, num_sources, theta_size)
    contexts = bayes_model._encode_observation_block(observations, num_sources, theta_size)
    next_compact = bayes_model._transport_compact_with_contexts(
        compact,
        contexts,
        jnp.asarray(observation_count, dtype=jnp.int32),
        num_sources,
        theta_size,
        cloud_padded.shape[-2],
        cloud_padded.shape[-1],
    )
    return jax.vmap(
        lambda th: padded_theta_jax(
            th, num_sources, theta_size, cloud_padded.shape[-2], cloud_padded.shape[-1]
        )
    )(next_compact)


def bayes_update_observation_set(
    bayes_model: SequentialBayesModel,
    cloud_padded: Array,
    observations: Array,
    num_sources: Array,
    theta_size: Array,
) -> Array:
    """Update the belief with all observations acquired at ONE policy decision.

    * Genuine multi-observation checkpoint (``min/max != 1/1``): the complete K-element set is
      consumed in ONE Bayes-map call.  This preserves the learned causal likelihood Transformer and
      exactly uses its K-observation prefix conditioning.
    * Strictly single-observation checkpoint (``min=max=1``): K=1 is a normal single update; K>1
      falls back to K recurrent one-observation updates.  This is the only case in which the set is
      intentionally split, because that checkpoint literally has no multi-observation likelihood
      Transformer.

    ``observations.shape[0]`` is static under JAX tracing, so this dispatch does not introduce a
    dynamic Python branch inside the compiled rollout.
    """
    K = int(observations.shape[0])
    if K < 1:
        raise ValueError("A policy decision must contain at least one observation.")

    if bayes_model.single_observation_direct and K > 1:
        def update_one(current_cloud: Array, observation: Array):
            next_cloud = _bayes_update_observation_block(
                bayes_model, current_cloud, observation[None, :], 1, num_sources, theta_size
            )
            return next_cloud, None

        cloud_padded, _ = jax.lax.scan(update_one, cloud_padded, observations)
        return cloud_padded

    # For a genuine multi-observation map this is the important path: all K observations are
    # contextualised jointly and only the K-prefix posterior is returned.
    return _bayes_update_observation_block(
        bayes_model, cloud_padded, observations, K, num_sources, theta_size
    )


def design_set_diversity_penalty(mean_designs: Array, theta_size: Array, num_sources: Array, cfg: PolicyConfig) -> Array:
    if cfg.designs_per_step <= 1 or cfg.design_set_diversity_weight <= 0.0:
        return jnp.asarray(0.0, dtype=mean_designs.dtype)
    active = _active_design_mask(theta_size, num_sources, mean_designs.shape[-1]).astype(mean_designs.dtype)
    delta = mean_designs[:, None, :] - mean_designs[None, :, :]
    dist_sq = jnp.sum(delta**2 * active[None, None, :], axis=-1)
    offdiag = 1.0 - jnp.eye(mean_designs.shape[0], dtype=mean_designs.dtype)
    closeness = jnp.exp(-dist_sq / max(cfg.design_set_diversity_scale**2, 1e-8)) * offdiag
    return jnp.sum(closeness) / jnp.maximum(cfg.designs_per_step * (cfg.designs_per_step - 1), 1)


#%% 9) Fresh simulator batches: same prior family, no buffers
def policy_shape_pool(bt_cfg: BayesTransportConfig, cfg: PolicyConfig) -> tuple[tuple[int, int], ...]:
    """Configured policy-training shapes after policy and optional Bayes held-out exclusions."""
    excluded = {tuple(map(int, shape)) for shape in cfg.heldout_shapes}
    if not cfg.include_bayes_heldout_shapes_in_policy_training:
        excluded |= {tuple(map(int, shape)) for shape in bt_cfg.heldout_shapes}

    pool = []
    for s in range(cfg.min_num_sources, cfg.max_num_sources + 1):
        for d in range(cfg.min_source_dim, cfg.max_source_dim + 1):
            if (s, d) not in excluded:
                pool.append((s, d))
    if not pool:
        raise ValueError(
            "No (num_sources,source_dim) shapes remain for policy training after held-out exclusions."
        )
    return tuple(pool)


def make_policy_batch_np(
    rng: np.random.Generator,
    batch_size: int,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
    *,
    balanced_shapes: bool = False,
) -> dict[str, np.ndarray]:
    """Fresh theta* + base-prior cloud batch.  Designs/outcomes are generated on-device."""
    shapes = policy_shape_pool(bt_cfg, cfg)
    if balanced_shapes:
        chosen = [shapes[i % len(shapes)] for i in range(batch_size)]
        rng.shuffle(chosen)
    else:
        chosen = [shapes[i] for i in rng.integers(0, len(shapes), size=batch_size)]

    theta_true = np.zeros((batch_size, bt_cfg.max_num_sources, bt_cfg.max_source_dim), dtype=np.float32)
    prior = np.zeros((batch_size, cfg.num_particles, bt_cfg.max_num_sources, bt_cfg.max_source_dim), dtype=np.float32)
    num_sources = np.zeros((batch_size,), dtype=np.int32)
    theta_size = np.zeros((batch_size,), dtype=np.int32)

    for b, (s, d) in enumerate(chosen):
        truth = sample_base_prior_np(rng, 1, bt_cfg, num_sources=s, source_dim=d)[0]
        cloud = sample_base_prior_np(rng, cfg.num_particles, bt_cfg, num_sources=s, source_dim=d)
        theta_true[b, :s, :d] = truth
        prior[b, :, :s, :d] = cloud
        num_sources[b] = s
        theta_size[b] = s * d

    return {
        "theta_true": theta_true,
        "prior_particles": prior,
        "num_sources": num_sources,
        "theta_size": theta_size,
    }


def batch_to_jax(batch: dict[str, np.ndarray]) -> dict[str, Array]:
    return {name: jnp.asarray(value) for name, value in batch.items()}


#%% 10) Sequential rollouts for learned and random policies

def rollout_episode_joint_aline(
    policy: JointALINEDesignPolicy,
    theta_true: Array,
    prior_particles: Array,
    num_sources: Array,
    theta_size: Array,
    key: Array,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
    *,
    deterministic: bool = False,
    fixed_observation_noise: Array | None = None,
) -> dict[str, Array]:
    """Joint ALINE-style rollout.

    Decision 0 is bootstrapped from the prior/current belief.  Thereafter each call to
    ``update_and_propose`` simultaneously produces the posterior cloud after the newly acquired
    observation block AND the distribution of the next design set, using one posterior-Transformer
    pass.  The full accumulated design/outcome history is always supplied to the acquisition head.
    """
    H = cfg.experimental_budget * cfg.designs_per_step
    history0 = jnp.zeros((H, bt_cfg.max_source_dim + 1), dtype=prior_particles.dtype)
    valid0 = jnp.zeros((H,), dtype=bool)
    mean0, log_std0 = policy(
        prior_particles, history0, valid0, jnp.asarray(0, dtype=jnp.int32),
        num_sources, theta_size,
    )

    def decision_step(carry, t):
        cloud, history, history_valid, rng_key, mean_raw, log_std_raw = carry
        rng_key, action_key, outcome_key = jax.random.split(rng_key, 3)
        before = cloud_metrics_jax(cloud, theta_true, num_sources, theta_size, bt_cfg)

        designs, log_prob, entropy, mean_designs, mean_std = sample_design_set_from_raw(
            mean_raw, log_std_raw, action_key, num_sources, theta_size, bt_cfg, cfg,
            deterministic=deterministic,
        )
        means = source_log_mean_jax(theta_true, designs, num_sources, theta_size, bt_cfg)
        if fixed_observation_noise is None:
            eps_y = jax.random.normal(outcome_key, (cfg.designs_per_step,))
        else:
            eps_y = fixed_observation_noise[t]
        outcomes = means + bt_cfg.observation_noise_std * eps_y

        observations = jnp.zeros(
            (cfg.designs_per_step, bt_cfg.max_source_dim + 1), dtype=cloud.dtype
        )
        observations = observations.at[:, :bt_cfg.max_source_dim].set(designs)
        observations = observations.at[:, -1].set(outcomes)

        start = t * cfg.designs_per_step
        history = jax.lax.dynamic_update_slice(history, observations, (start, 0))
        history_valid = jax.lax.dynamic_update_slice(
            history_valid, jnp.ones((cfg.designs_per_step,), dtype=bool), (start,)
        )

        # Shared inference/acquisition call: SAME posterior hidden particles feed the physical
        # posterior head and the ALINE-style acquisition head for decision t+1.
        cloud, next_mean_raw, next_log_std_raw = policy.update_and_propose(
            cloud, observations, history, history_valid, t + 1, num_sources, theta_size
        )
        after = cloud_metrics_jax(cloud, theta_true, num_sources, theta_size, bt_cfg)
        reward = reward_from_metrics(before, after, cfg)
        if cfg.gradient_estimator == "reinforce":
            reward = jax.lax.stop_gradient(reward)

        diversity = design_set_diversity_penalty(mean_designs, theta_size, num_sources, cfg)
        outputs = {
            "posterior_cloud": cloud,
            "designs": designs,
            "outcomes": outcomes,
            "observations": observations,
            "reward": reward,
            "log_prob": log_prob,
            "entropy": entropy,
            "mean_std": mean_std,
            "diversity_penalty": diversity,
            "energy_score": after["energy_score"],
            "attraction": after["attraction"],
            "repulsion": after["repulsion"],
            "rmse": after["rmse"],
            "spread": after["spread"],
        }
        return (
            cloud, history, history_valid, rng_key, next_mean_raw, next_log_std_raw
        ), outputs

    initial_metrics = cloud_metrics_jax(
        prior_particles, theta_true, num_sources, theta_size, bt_cfg
    )
    (_, _, _, _, _, _), trajectory = jax.lax.scan(
        decision_step,
        (prior_particles, history0, valid0, key, mean0, log_std0),
        jnp.arange(cfg.experimental_budget, dtype=jnp.int32),
    )
    trajectory["initial_energy_score"] = initial_metrics["energy_score"]
    trajectory["initial_attraction"] = initial_metrics["attraction"]
    trajectory["initial_rmse"] = initial_metrics["rmse"]
    trajectory["initial_spread"] = initial_metrics["spread"]
    return trajectory


def rollout_episode(
    policy: PolicyModule,
    bayes_model: SequentialBayesModel,
    theta_true: Array,
    prior_particles: Array,
    num_sources: Array,
    theta_size: Array,
    key: Array,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
    *,
    policy_kind: str = "learned",
    deterministic: bool = False,
    fixed_observation_noise: Array | None = None,
) -> dict[str, Array]:
    """Roll out one sequential BED episode.

    ``fixed_observation_noise`` may have shape [T,K].  It is used only for the deterministic fixed
    trajectory visualisation so learned/random policies can share the exact same Gaussian noise.
    """
    if policy_kind == "learned" and policy.joint_aline_mode:
        return rollout_episode_joint_aline(
            policy, theta_true, prior_particles, num_sources, theta_size, key, bt_cfg, cfg,
            deterministic=deterministic, fixed_observation_noise=fixed_observation_noise,
        )

    H = cfg.experimental_budget * cfg.designs_per_step
    history0 = jnp.zeros((H, bt_cfg.max_source_dim + 1), dtype=prior_particles.dtype)
    valid0 = jnp.zeros((H,), dtype=bool)

    def decision_step(carry, t):
        cloud, history, history_valid, rng_key = carry
        rng_key, action_key, outcome_key = jax.random.split(rng_key, 3)
        before = cloud_metrics_jax(cloud, theta_true, num_sources, theta_size, bt_cfg)

        if policy_kind == "learned":
            designs, log_prob, entropy, mean_designs, mean_std = sample_policy_design_set(
                policy, cloud, history, history_valid, t, num_sources, theta_size,
                action_key, bt_cfg, cfg, deterministic=deterministic,
            )
        elif policy_kind == "random":
            source_dim = theta_size // num_sources
            active = jnp.arange(bt_cfg.max_source_dim) < source_dim
            raw = jax.random.uniform(
                action_key, (cfg.designs_per_step, bt_cfg.max_source_dim),
                minval=bt_cfg.design_low, maxval=bt_cfg.design_high,
            )
            designs = jnp.where(active[None, :], raw, 0.0)
            mean_designs = designs
            log_prob = jnp.asarray(0.0, dtype=cloud.dtype)
            entropy = jnp.asarray(0.0, dtype=cloud.dtype)
            mean_std = jnp.asarray(0.0, dtype=cloud.dtype)
        else:
            raise ValueError("policy_kind must be 'learned' or 'random'.")

        means = source_log_mean_jax(theta_true, designs, num_sources, theta_size, bt_cfg)
        if fixed_observation_noise is None:
            eps_y = jax.random.normal(outcome_key, (cfg.designs_per_step,))
        else:
            eps_y = fixed_observation_noise[t]
        outcomes = means + bt_cfg.observation_noise_std * eps_y

        observations = jnp.zeros((cfg.designs_per_step, bt_cfg.max_source_dim + 1), dtype=cloud.dtype)
        observations = observations.at[:, :bt_cfg.max_source_dim].set(designs)
        observations = observations.at[:, -1].set(outcomes)

        # K designs are proposed as a set from the SAME pre-step belief/history.  A pretrained
        # multi-observation Bayes map must see that set jointly: its causal likelihood Transformer
        # builds the K-prefix context before the posterior transport is applied.  Only checkpoints
        # that were strictly single-observation fall back to recurrent K x one-observation updates.
        cloud = bayes_update_observation_set(
            bayes_model, cloud, observations, num_sources, theta_size
        )
        after = cloud_metrics_jax(cloud, theta_true, num_sources, theta_size, bt_cfg)
        reward = reward_from_metrics(before, after, cfg)
        if cfg.gradient_estimator == "reinforce":
            # Pure score-function estimator: rewards/environment are data, not pathwise gradients.
            reward = jax.lax.stop_gradient(reward)

        start = t * cfg.designs_per_step
        history = jax.lax.dynamic_update_slice(history, observations, (start, 0))
        history_valid = jax.lax.dynamic_update_slice(
            history_valid, jnp.ones((cfg.designs_per_step,), dtype=bool), (start,)
        )

        diversity = design_set_diversity_penalty(mean_designs, theta_size, num_sources, cfg)
        outputs = {
            "posterior_cloud": cloud,
            "designs": designs,
            "outcomes": outcomes,
            "observations": observations,
            "reward": reward,
            "log_prob": log_prob,
            "entropy": entropy,
            "mean_std": mean_std,
            "diversity_penalty": diversity,
            "energy_score": after["energy_score"],
            "attraction": after["attraction"],
            "repulsion": after["repulsion"],
            "rmse": after["rmse"],
            "spread": after["spread"],
        }
        return (cloud, history, history_valid, rng_key), outputs

    initial_metrics = cloud_metrics_jax(prior_particles, theta_true, num_sources, theta_size, bt_cfg)
    (_, _, _, _), trajectory = jax.lax.scan(
        decision_step,
        (prior_particles, history0, valid0, key),
        jnp.arange(cfg.experimental_budget, dtype=jnp.int32),
    )
    trajectory["initial_energy_score"] = initial_metrics["energy_score"]
    trajectory["initial_attraction"] = initial_metrics["attraction"]
    trajectory["initial_rmse"] = initial_metrics["rmse"]
    trajectory["initial_spread"] = initial_metrics["spread"]
    return trajectory


def rollout_batch(
    policy: PolicyModule,
    bayes_model: SequentialBayesModel,
    batch: dict[str, Array],
    keys: Array,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
    *,
    policy_kind: str,
    deterministic: bool,
) -> dict[str, Array]:
    return jax.vmap(
        lambda truth, prior, s, size, key: rollout_episode(
            policy, bayes_model, truth, prior, s, size, key, bt_cfg, cfg,
            policy_kind=policy_kind, deterministic=deterministic,
        )
    )(
        batch["theta_true"], batch["prior_particles"], batch["num_sources"],
        batch["theta_size"], keys,
    )


def discounted_return_to_go(rewards: Array, gamma: float) -> Array:
    """rewards [...,T] -> conventional discounted return-to-go with the same shape."""
    def reverse_step(carry, reward):
        carry = reward + gamma * carry
        return carry, carry
    _, reversed_returns = jax.lax.scan(
        reverse_step, jnp.zeros(rewards.shape[:-1], dtype=rewards.dtype),
        jnp.moveaxis(rewards, -1, 0)[::-1],
    )
    return jnp.moveaxis(reversed_returns[::-1], 0, -1)


#%% 11) Policy objective: pathwise or REINFORCE
def policy_batch_objective(
    policy: PolicyModule,
    bayes_model: SequentialBayesModel,
    batch: dict[str, Array],
    keys: Array,
    baseline_by_t: Array,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
) -> tuple[Array, dict[str, Array]]:
    trajectories = rollout_batch(
        policy, bayes_model, batch, keys, bt_cfg, cfg,
        policy_kind="learned", deterministic=False,
    )
    rewards = trajectories["reward"]                         # [B,T]
    log_prob = trajectories["log_prob"]                     # [B,T]
    entropy = trajectories["entropy"]                       # [B,T]
    diversity = trajectories["diversity_penalty"]          # [B,T]
    discounts = cfg.discount_gamma ** jnp.arange(cfg.experimental_budget, dtype=rewards.dtype)

    if cfg.gradient_estimator == "pathwise":
        # Mohamed et al. Eq. 29: all simulator randomness has been expressed through parameter-
        # independent base noise, so ordinary autodiff through the sampled trajectory estimates
        # d/dpsi E[R].  With gamma=1 and improvement rewards, the reward sum telescopes toward
        # total reduction in the selected uncertainty score.
        objective = jnp.mean(jnp.sum(discounts[None, :] * rewards, axis=-1))
        policy_loss = -objective
        signal_for_baseline = rewards
    else:
        if cfg.reinforce_credit_assignment == "return_to_go":
            signal = discounted_return_to_go(rewards, cfg.discount_gamma)
            weights = jnp.ones_like(discounts)
        else:
            # ALINE Eq. 11-style dense credit: each action is weighted by its immediate posterior
            # improvement, with gamma^t applied outside the reward.
            signal = rewards
            weights = discounts

        signal = jax.lax.stop_gradient(signal)
        if cfg.reinforce_baseline == "ema":
            advantage = signal - baseline_by_t[None, :]
        else:
            advantage = signal
        if cfg.reinforce_normalize_advantage:
            mean = jnp.mean(advantage)
            std = jnp.std(advantage) + 1e-6
            advantage = (advantage - mean) / std
        advantage = jax.lax.stop_gradient(advantage)
        policy_loss = -jnp.mean(jnp.sum(weights[None, :] * advantage * log_prob, axis=-1))
        objective = jnp.mean(jnp.sum(discounts[None, :] * rewards, axis=-1))
        signal_for_baseline = signal

    entropy_term = jnp.mean(entropy)
    diversity_term = jnp.mean(diversity)
    loss = (
        policy_loss
        - cfg.entropy_bonus * entropy_term
        + cfg.design_set_diversity_weight * diversity_term
    )

    aux = {
        "loss": loss,
        "policy_loss": policy_loss,
        "objective": objective,
        "mean_reward": jnp.mean(rewards),
        "final_reward": jnp.mean(rewards[:, -1]),
        "mean_entropy": entropy_term,
        "mean_policy_std": jnp.mean(trajectories["mean_std"]),
        "diversity_penalty": diversity_term,
        "final_energy_score": jnp.mean(trajectories["energy_score"][:, -1]),
        "final_attraction": jnp.mean(trajectories["attraction"][:, -1]),
        "final_rmse": jnp.mean(trajectories["rmse"][:, -1]),
        "baseline_signal_by_t": jnp.mean(signal_for_baseline, axis=0),
    }
    return loss, aux


def bayes_finetune_batch_objective(
    bayes_model: SequentialBayesModel,
    policy: PolicyModule,
    batch: dict[str, Array],
    keys: Array,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
) -> tuple[Array, dict[str, Array]]:
    """Optional inference-only fine-tune on the current policy-induced data distribution.

    Designs are detached from the Bayes parameters.  We optimize the proper empirical energy score
    after every decision stage, so this is an inference objective, not a hidden policy-gradient term.
    """
    def one_episode(theta_true, prior, num_sources, theta_size, key):
        H = cfg.experimental_budget * cfg.designs_per_step
        history = jnp.zeros((H, bt_cfg.max_source_dim + 1), dtype=prior.dtype)
        history_valid = jnp.zeros((H,), dtype=bool)

        def step(carry, t):
            cloud, history, history_valid, key = carry
            key, action_key, outcome_key = jax.random.split(key, 3)
            # Stop belief gradients BEFORE asking the policy for designs.  This deliberately treats
            # the acquired design as fixed data when optimizing the inference network.
            designs, _, _, _, _ = sample_policy_design_set(
                policy,
                jax.lax.stop_gradient(cloud),
                jax.lax.stop_gradient(history),
                history_valid,
                t, num_sources, theta_size, action_key, bt_cfg, cfg,
                deterministic=False,
            )
            designs = jax.lax.stop_gradient(designs)
            means = source_log_mean_jax(theta_true, designs, num_sources, theta_size, bt_cfg)
            eps = jax.random.normal(outcome_key, (cfg.designs_per_step,))
            outcomes = jax.lax.stop_gradient(means + bt_cfg.observation_noise_std * eps)
            observations = jnp.zeros((cfg.designs_per_step, bt_cfg.max_source_dim + 1), dtype=prior.dtype)
            observations = observations.at[:, :bt_cfg.max_source_dim].set(designs)
            observations = observations.at[:, -1].set(outcomes)

            cloud = bayes_update_observation_set(
                bayes_model, cloud, observations, num_sources, theta_size
            )
            metrics = cloud_metrics_jax(cloud, theta_true, num_sources, theta_size, bt_cfg)
            start = t * cfg.designs_per_step
            history = jax.lax.dynamic_update_slice(history, observations, (start, 0))
            history_valid = jax.lax.dynamic_update_slice(
                history_valid, jnp.ones((cfg.designs_per_step,), dtype=bool), (start,)
            )
            return (cloud, history, history_valid, key), metrics["energy_score"]

        (_, _, _, _), energy_by_t = jax.lax.scan(
            step, (prior, history, history_valid, key),
            jnp.arange(cfg.experimental_budget, dtype=jnp.int32),
        )
        return jnp.mean(energy_by_t), energy_by_t[-1]

    mean_es, final_es = jax.vmap(one_episode)(
        batch["theta_true"], batch["prior_particles"], batch["num_sources"],
        batch["theta_size"], keys,
    )
    loss = cfg.bayes_finetune_loss_weight * jnp.mean(mean_es)
    return loss, {"bayes_finetune_loss": loss, "bayes_finetune_final_es": jnp.mean(final_es)}


#%% 12) sPCE / EIG lower-bound and other host-side evaluation metrics
def _logmeanexp_np(values: np.ndarray, axis: int = 0) -> np.ndarray:
    vmax = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(vmax + np.log(np.mean(np.exp(values - vmax), axis=axis, keepdims=True)), axis=axis)


def estimate_spce_curve_np(
    theta_true_padded: np.ndarray,
    observations: np.ndarray,                 # [T,K,Dmax+1]
    num_sources: int,
    theta_size: int,
    rng: np.random.Generator,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
) -> np.ndarray:
    """Sequential Prior Contrastive Estimation lower bound in nats (ALINE Appendix Eq. A26)."""
    S = int(num_sources)
    D = int(theta_size // num_sources)
    flat = np.asarray(observations).reshape(-1, observations.shape[-1])
    designs = flat[:, :D]
    outcomes = flat[:, -1]
    theta_true = np.asarray(theta_true_padded)[:S, :D]

    L = int(cfg.eig_contrastive_samples)
    contrastive = sample_base_prior_np(rng, L, bt_cfg, num_sources=S, source_dim=D)
    all_theta = np.concatenate([theta_true[None, :, :], contrastive], axis=0)
    loglik = log_likelihood_np(all_theta, designs, outcomes, bt_cfg)  # [L+1, T*K]
    cumulative = np.cumsum(loglik, axis=-1)
    # log p(D_t|theta*) - log mean_{ell=0..L} p(D_t|theta_ell)
    bound_per_observation = cumulative[0] - _logmeanexp_np(cumulative, axis=0)
    decision_indices = np.arange(cfg.designs_per_step - 1, flat.shape[0], cfg.designs_per_step)
    return bound_per_observation[decision_indices].astype(np.float64)


def posterior_cloud_host_metrics(
    clouds: np.ndarray,
    theta_true: np.ndarray,
    num_sources: int,
    theta_size: int,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
) -> dict[str, np.ndarray]:
    """Covariance log-volume and marginal credible-interval coverage by decision stage."""
    S = int(num_sources); D = int(theta_size // num_sources); Kdim = int(theta_size)
    truth = np.asarray(theta_true)[:S, :D]
    if bt_cfg.canonicalize_particle_sources and S > 1:
        truth = canonicalize_sources_np(truth)
    truth_flat = truth.reshape(-1)
    alpha = 0.5 * (1.0 - cfg.credible_interval_mass)
    logdet_values, coverage_values = [], []
    for cloud in np.asarray(clouds):
        active = cloud[:, :S, :D]
        if bt_cfg.canonicalize_particle_sources and S > 1:
            active = np.stack([canonicalize_sources_np(x) for x in active], axis=0)
        x = active.reshape(active.shape[0], Kdim)
        covariance = np.cov(x, rowvar=False)
        covariance = np.atleast_2d(covariance) + 1e-6 * np.eye(Kdim)
        sign, logdet = np.linalg.slogdet(covariance)
        logdet_values.append(float(logdet) if sign > 0 else np.nan)
        lo = np.quantile(x, alpha, axis=0)
        hi = np.quantile(x, 1.0 - alpha, axis=0)
        coverage_values.append(float(np.mean((truth_flat >= lo) & (truth_flat <= hi))))
    return {
        "cov_logdet": np.asarray(logdet_values, dtype=np.float64),
        "coverage": np.asarray(coverage_values, dtype=np.float64),
    }


def mean_ci(values: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values, axis=0)
    n = np.maximum(np.sum(np.isfinite(values), axis=0), 1)
    sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(n)
    return mean, z * sem


def evaluate_policy_pair(
    policy: PolicyModule,
    bayes_model: SequentialBayesModel,
    eval_batch_np: dict[str, np.ndarray],
    key: Array,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
) -> dict[str, Any]:
    """Paired learned-vs-random evaluation using the same theta*, prior cloud and noise key."""
    batch = batch_to_jax(eval_batch_np)
    B = int(eval_batch_np["theta_true"].shape[0])
    keys = jax.random.split(key, B)
    learned = rollout_batch(
        policy, bayes_model, batch, keys, bt_cfg, cfg,
        policy_kind="learned", deterministic=cfg.eval_deterministic_policy,
    )
    # Reusing the same keys gives both policies common outcome-noise streams at each decision.
    random = rollout_batch(
        policy, bayes_model, batch, keys, bt_cfg, cfg,
        policy_kind="random", deterministic=False,
    )
    learned_np = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), learned)
    random_np = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), random)

    # sPCE can be the most expensive diagnostic.  Use a reproducible subset during training.
    n_eig = min(B, cfg.eig_eval_trajectories)
    spce_learned, spce_random = [], []
    host_learned = {"cov_logdet": [], "coverage": []}
    host_random = {"cov_logdet": [], "coverage": []}
    for b in range(B):
        S = int(eval_batch_np["num_sources"][b])
        size = int(eval_batch_np["theta_size"][b])
        for source, collector in ((learned_np, host_learned), (random_np, host_random)):
            extra = posterior_cloud_host_metrics(
                source["posterior_cloud"][b], eval_batch_np["theta_true"][b], S, size, bt_cfg, cfg
            )
            collector["cov_logdet"].append(extra["cov_logdet"])
            collector["coverage"].append(extra["coverage"])
        if b < n_eig:
            spce_rng = np.random.default_rng(cfg.seed + 90_000 + b)
            # Re-seed identically for paired contrastive theta samples.
            spce_learned.append(estimate_spce_curve_np(
                eval_batch_np["theta_true"][b], learned_np["observations"][b], S, size,
                spce_rng, bt_cfg, cfg,
            ))
            spce_rng = np.random.default_rng(cfg.seed + 90_000 + b)
            spce_random.append(estimate_spce_curve_np(
                eval_batch_np["theta_true"][b], random_np["observations"][b], S, size,
                spce_rng, bt_cfg, cfg,
            ))

    for collector in (host_learned, host_random):
        for name in collector:
            collector[name] = np.stack(collector[name], axis=0)

    result = {
        "learned": learned_np,
        "random": random_np,
        "learned_host": host_learned,
        "random_host": host_random,
        "spce_learned": np.stack(spce_learned, axis=0),
        "spce_random": np.stack(spce_random, axis=0),
    }
    return result


def evaluation_summary_rows(result: dict[str, Any], cfg: PolicyConfig) -> list[dict[str, Any]]:
    """Compact all-metric table, including ALINE-style EIG mean ± 95% CI components."""
    rows = []
    for label, traj, host, spce in (
        ("learned_policy", result["learned"], result["learned_host"], result["spce_learned"]),
        ("uniform_random", result["random"], result["random_host"], result["spce_random"]),
    ):
        final_spce = np.asarray(spce[:, -1], dtype=np.float64)
        eig_mean, eig_ci = mean_ci(final_spce[:, None], cfg.confidence_z)
        rows.append({
            "method": label,
            "sPCE_EIG_lower_bound_nats": float(eig_mean[0]),
            "sPCE_EIG_95CI_halfwidth_nats": float(eig_ci[0]),
            "energy_score": float(np.mean(traj["energy_score"][:, -1])),
            "attraction_to_theta_star": float(np.mean(traj["attraction"][:, -1])),
            "repulsion": float(np.mean(traj["repulsion"][:, -1])),
            "posterior_mean_RMSE": float(np.mean(traj["rmse"][:, -1])),
            "posterior_spread": float(np.mean(traj["spread"][:, -1])),
            "covariance_logdet": float(np.nanmean(host["cov_logdet"][:, -1])),
            f"marginal_{int(round(100*cfg.credible_interval_mass))}pct_coverage": float(np.mean(host["coverage"][:, -1])),
            "cumulative_reward": float(np.mean(np.sum(traj["reward"], axis=-1))),
        })
    return rows


def eig_final_summary_rows(result: dict[str, Any], cfg: PolicyConfig) -> list[dict[str, Any]]:
    """Paper-style final sPCE/EIG numbers: mean ± 95% CI plus paired strategic gain."""
    learned = np.asarray(result["spce_learned"][:, -1], dtype=np.float64)
    random = np.asarray(result["spce_random"][:, -1], dtype=np.float64)
    paired = learned - random

    rows: list[dict[str, Any]] = []
    for label, values in (
        ("learned_policy", learned),
        ("uniform_random", random),
        ("paired_gain_learned_minus_random", paired),
    ):
        mean, half = mean_ci(values[:, None], cfg.confidence_z)
        m = float(mean[0]); h = float(half[0])
        rows.append({
            "method": label,
            "mean_sPCE_EIG_nats": m,
            "ci95_halfwidth_nats": h,
            "ci95_lower_nats": m - h,
            "ci95_upper_nats": m + h,
            "n_paired_trajectories": int(len(values)),
            "contrastive_samples_L": int(cfg.eig_contrastive_samples),
            "experimental_budget": int(cfg.experimental_budget),
            "designs_per_step": int(cfg.designs_per_step),
        })
    return rows


def save_evaluation_table(rows: list[dict[str, Any]], table_dir: Path, stem: str):
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    md_path = table_dir / f"{stem}.md"
    headers = list(rows[0].keys())
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = [row[h] if isinstance(row[h], str) else f"{row[h]:.6g}" for h in headers]
            handle.write("| " + " | ".join(map(str, values)) + " |\n")


#%% 13) Visualisations: training curves, EIG comparison, fixed strategic trajectory
def _format_step_k(value: float, _position: int | None = None) -> str:
    if abs(value) >= 1000:
        return f"{value/1000:g}k"
    return f"{value:g}"


def plot_training_diagnostics(history: dict[str, list], destination: Path):
    """Plot both optimizer-step diagnostics and epoch-aggregated diagnostics.

    An epoch here is intentionally NOT a pass over a finite dataset: every optimizer step still
    receives a newly simulated batch.  Epochs provide a stable reporting, evaluation, checkpoint,
    and visualisation unit for this sequential policy-training problem.
    """
    steps = np.asarray(history.get("step", []), dtype=float)
    epochs = np.asarray(history.get("epoch_index", []), dtype=float)
    if len(steps) == 0 and len(epochs) == 0:
        return

    fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.8), constrained_layout=True)
    axes = axes.ravel()

    # (history key, title, x kind, log-y?)
    series = [
        ("loss", "Loss per optimizer step", "step", False),
        ("epoch_loss", "Mean loss per epoch", "epoch", False),
        ("mean_reward", "Mean step reward", "step", False),
        ("grad_norm", "Gradient norm", "step", True),
        ("mean_policy_std", "Mean policy std", "step", False),
        ("eval_learned_spce", "Eval sPCE / EIG lower bound", "eval", False),
        ("eval_learned_final_es", "Eval final energy score", "eval", False),
        ("epoch_train_final_energy_score", "Mean train final energy score / epoch", "epoch", False),
    ]

    for ax, (name, title, x_kind, logy) in zip(axes, series):
        values = np.asarray(history.get(name, []), dtype=float)

        if x_kind == "step":
            x = steps
            xlabel = "optimizer step"
        elif x_kind == "epoch":
            x = epochs
            xlabel = "epoch"
        else:
            # New runs record eval_epoch explicitly.  Fall back to global optimizer steps so old
            # policy histories can still be reloaded and visualized.
            eval_epochs = np.asarray(history.get("eval_epoch", []), dtype=float)
            eval_steps = np.asarray(history.get("eval_step", []), dtype=float)
            if len(eval_epochs) == len(values) and len(eval_epochs):
                x = eval_epochs
                xlabel = "epoch"
            else:
                x = eval_steps
                xlabel = "optimizer step"

        if len(values) == len(x) and len(values):
            ax.plot(x, values, linewidth=1.0, label="learned" if name.startswith("eval_") else None)

            if name.startswith("eval_"):
                random_name = name.replace("learned", "random")
                random_values = np.asarray(history.get(random_name, []), dtype=float)
                if len(random_values) == len(x):
                    ax.plot(x, random_values, linewidth=1.0, label="random")
                    ax.legend()

        finite_positive = values[np.isfinite(values)]
        if logy and len(finite_positive) and np.all(finite_positive > 0):
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.20)
        if x_kind in {"step", "eval"} and xlabel == "optimizer step":
            ax.xaxis.set_major_formatter(FuncFormatter(_format_step_k))

    total_updates = POLICY_CFG.epochs * POLICY_CFG.train_steps_per_epoch
    fig.suptitle(
        f"BED policy training: budget={POLICY_CFG.experimental_budget}, "
        f"epochs={POLICY_CFG.epochs}, updates={total_updates:,}, "
        f"estimator={POLICY_CFG.gradient_estimator}, reward={POLICY_CFG.reward_mode}",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


def plot_eig_final(result: dict[str, Any], destination: Path, cfg: PolicyConfig):
    """ALINE-style final EIG reporting: sequential curve plus explicit numerical estimates.

    The left panel shows how the sPCE lower bound accumulates with the experimental budget. The
    right panel reports the final-budget mean ± 95% CI numerically for learned and random policies,
    together with the paired gain.  This is intentionally closer to ALINE's BED result table than
    a curve-only diagnostic.
    """
    learned = np.asarray(result["spce_learned"], dtype=np.float64)
    random = np.asarray(result["spce_random"], dtype=np.float64)
    x = np.arange(1, cfg.experimental_budget + 1)
    lm, lci = mean_ci(learned, cfg.confidence_z)
    rm, rci = mean_ci(random, cfg.confidence_z)

    l_final = learned[:, -1]
    r_final = random[:, -1]
    gain = l_final - r_final
    lmean, lhalf = mean_ci(l_final[:, None], cfg.confidence_z)
    rmean, rhalf = mean_ci(r_final[:, None], cfg.confidence_z)
    gmean, ghalf = mean_ci(gain[:, None], cfg.confidence_z)
    lmean, lhalf = float(lmean[0]), float(lhalf[0])
    rmean, rhalf = float(rmean[0]), float(rhalf[0])
    gmean, ghalf = float(gmean[0]), float(ghalf[0])

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, lm, label="learned strategic policy")
    ax.fill_between(x, lm - lci, lm + lci, alpha=0.18)
    ax.plot(x, rm, label="uniform random policy")
    ax.fill_between(x, rm - rci, rm + rci, alpha=0.18)
    ax.set_title("Sequential sPCE / EIG lower bound", fontweight="bold")
    ax.set_xlabel("decision step")
    ax.set_ylabel("EIG lower bound (nats)")
    ax.grid(alpha=0.20)
    ax.legend()

    ax = axes[1]
    means = np.asarray([lmean, rmean], dtype=float)
    halves = np.asarray([lhalf, rhalf], dtype=float)
    ypos = np.asarray([1.0, 0.0])
    ax.errorbar(means, ypos, xerr=halves, fmt="o", capsize=5, linewidth=1.5)
    ax.set_yticks(ypos, ["Learned policy", "Uniform random"])
    ax.set_xlabel("final sPCE / EIG lower bound (nats)")
    ax.set_title("Final-budget EIG numbers (mean ± 95% CI)", fontweight="bold")
    ax.grid(axis="x", alpha=0.20)
    span = max(abs(lmean), abs(rmean), lhalf, rhalf, 1.0)
    x_text = max(lmean + lhalf, rmean + rhalf) + 0.04 * span
    ax.text(x_text, 1.0, f"{lmean:.3f} ± {lhalf:.3f}", va="center")
    ax.text(x_text, 0.0, f"{rmean:.3f} ± {rhalf:.3f}", va="center")
    ax.text(
        0.02, -0.17,
        f"Paired gain (learned − random): {gmean:+.3f} ± {ghalf:.3f} nats\n"
        f"N={len(l_final)} paired trajectories; L={cfg.eig_contrastive_samples:,} contrastives",
        transform=ax.transAxes, va="top",
    )

    fig.suptitle(
        f"Final Bayesian experimental-design EIG evaluation — budget {cfg.experimental_budget}",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(destination, dpi=190, bbox_inches="tight")
    display(fig); plt.close(fig)


def plot_policy_vs_random_metrics(result: dict[str, Any], destination: Path, cfg: PolicyConfig):
    x = np.arange(1, cfg.experimental_budget + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 9.0), constrained_layout=True)
    panels = [
        (result["spce_learned"], result["spce_random"], "sPCE / EIG lower bound", "nats", False),
        (result["learned"]["energy_score"], result["random"]["energy_score"], "Energy score", "lower is better", False),
        (result["learned"]["attraction"], result["random"]["attraction"], "Attraction to theta*", "mean distance", False),
        (result["learned"]["rmse"], result["random"]["rmse"], "Posterior mean RMSE", "lower is better", False),
        (result["learned"]["spread"], result["random"]["spread"], "Posterior spread", "mean marginal variance", False),
        (result["learned_host"]["cov_logdet"], result["random_host"]["cov_logdet"], "Covariance log-volume", "log det covariance", False),
    ]
    for panel_index, (ax, (learned, random, title, ylabel, _)) in enumerate(zip(axes.ravel(), panels)):
        lm, lci = mean_ci(learned, cfg.confidence_z)
        rm, rci = mean_ci(random, cfg.confidence_z)
        ax.plot(x, lm, label="learned strategic policy")
        ax.fill_between(x, lm - lci, lm + lci, alpha=0.18)
        ax.plot(x, rm, label="uniform random policy")
        ax.fill_between(x, rm - rci, rm + rci, alpha=0.18)
        if panel_index == 0:
            ax.text(
                0.03, 0.97,
                f"final learned: {lm[-1]:.3f} ± {lci[-1]:.3f}\n"
                f"final random:  {rm[-1]:.3f} ± {rci[-1]:.3f}",
                transform=ax.transAxes, va="top", ha="left",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82},
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("decision step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8)
    fig.suptitle("Strategic Bayesian experimental design vs random acquisition", fontsize=14, fontweight="bold")
    fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


def select_prefixes(trajectory_length: int, n_panels_after_prior: int = 5) -> list[int]:
    if trajectory_length <= n_panels_after_prior:
        return list(range(1, trajectory_length + 1))
    values = np.unique(np.linspace(1, trajectory_length, n_panels_after_prior, dtype=int))
    while len(values) > n_panels_after_prior:
        values = np.delete(values, 1)
    return values.tolist()


def _active_cloud_np(cloud_padded: np.ndarray, S: int, D: int, cfg: BayesTransportConfig) -> np.ndarray:
    active = np.asarray(cloud_padded)[:, :S, :D]
    if cfg.canonicalize_particle_sources and S > 1:
        active = np.stack([canonicalize_sources_np(x) for x in active], axis=0)
    return active


def plot_fixed_policy_evolution(
    rollout: dict[str, np.ndarray],
    theta_true_padded: np.ndarray,
    prior_particles: np.ndarray,
    num_sources: int,
    theta_size: int,
    bt_cfg: BayesTransportConfig,
    cfg: PolicyConfig,
    destination: Path,
    title: str,
):
    """Same 2x3 physical-cloud visual grammar as Bayes Transport fixed_trajectory_epoch_* plots."""
    S = int(num_sources); D = int(theta_size // num_sources)
    if D != 2:
        return
    theta_true = np.asarray(theta_true_padded)[:S, :D]
    if bt_cfg.canonicalize_particle_sources and S > 1:
        theta_true = canonicalize_sources_np(theta_true)
    prior_cloud = _active_cloud_np(prior_particles, S, D, bt_cfg)
    posterior = np.stack([_active_cloud_np(c, S, D, bt_cfg) for c in rollout["posterior_cloud"]], axis=0)
    prefixes = select_prefixes(cfg.experimental_budget, 5)
    clouds = [prior_cloud] + [posterior[t-1] for t in prefixes]
    observations_per_decision = int(np.asarray(rollout["designs"]).shape[1])
    labels = [r"base prior $p(\theta)$"] + [
        rf"$q_\phi(\theta\mid y_{{1:{t*observations_per_decision}}})$" for t in prefixes
    ]

    all_points = np.concatenate([c.reshape(-1, 2) for c in clouds] + [theta_true.reshape(-1, 2)])
    lim = max(_base_prior_plot_extent(bt_cfg), 1.12 * float(np.quantile(np.abs(all_points), 0.995)))
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.5), constrained_layout=True)
    axes = axes.ravel()
    for panel_index, (ax, cloud, label) in enumerate(zip(axes, clouds, labels)):
        ax.scatter(cloud[...,0].reshape(-1), cloud[...,1].reshape(-1), s=14, alpha=0.30,
                   color="#4C78A8", edgecolors="none", label="posterior source locations" if panel_index else "prior source locations")
        ax.scatter(theta_true[:,0], theta_true[:,1], marker="*", s=195, color="#111111",
                   edgecolors="white", linewidths=0.7, label=r"$\theta^\star$", zorder=7)
        if panel_index > 0:
            t = prefixes[panel_index-1]
            designs = rollout["designs"][:t].reshape(-1, bt_cfg.max_source_dim)[:, :D]
            ax.scatter(designs[:,0], designs[:,1], marker="x", s=38, alpha=0.78,
                       color="#8FD19E", linewidths=1.5, label="policy designs seen", zorder=5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.grid(alpha=0.18); ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
        ax.set_title(label); ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


def plot_fixed_policy_vs_random(
    learned: dict[str, np.ndarray], random: dict[str, np.ndarray], theta_true_padded: np.ndarray,
    prior_particles: np.ndarray, num_sources: int, theta_size: int,
    bt_cfg: BayesTransportConfig, cfg: PolicyConfig, destination: Path,
):
    S = int(num_sources); D = int(theta_size // num_sources)
    if D != 2:
        return
    theta_true = np.asarray(theta_true_padded)[:S, :D]
    if bt_cfg.canonicalize_particle_sources and S > 1:
        theta_true = canonicalize_sources_np(theta_true)
    selected = np.unique(np.linspace(1, cfg.experimental_budget, 4, dtype=int)).tolist()
    fig, axes = plt.subplots(2, len(selected), figsize=(4.0*len(selected), 8.0), constrained_layout=True)
    all_clouds = [learned["posterior_cloud"][t-1] for t in selected] + [random["posterior_cloud"][t-1] for t in selected]
    active_points = [_active_cloud_np(c, S, D, bt_cfg).reshape(-1,2) for c in all_clouds]
    lim = max(_base_prior_plot_extent(bt_cfg), 1.12*float(np.quantile(np.abs(np.concatenate(active_points)), 0.995)))
    for row, (name, trajectory) in enumerate((("Strategic policy", learned), ("Uniform random", random))):
        for col, t in enumerate(selected):
            ax = axes[row, col] if len(selected) > 1 else axes[row]
            cloud = _active_cloud_np(trajectory["posterior_cloud"][t-1], S, D, bt_cfg)
            designs = trajectory["designs"][:t].reshape(-1, bt_cfg.max_source_dim)[:, :D]
            ax.scatter(cloud[...,0].reshape(-1), cloud[...,1].reshape(-1), s=13, alpha=0.30)
            ax.scatter(theta_true[:,0], theta_true[:,1], marker="*", s=180, color="#111111", edgecolors="white", linewidths=0.7)
            ax.scatter(designs[:,0], designs[:,1], marker="x", s=33, alpha=0.7)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=0.18)
            ax.set_title(f"{name}\nstep {t}")
            ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    fig.suptitle("Fixed truth/prior/noise: learned strategic policy versus original random-style acquisition", fontsize=14, fontweight="bold")
    fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


def plot_design_field_comparison(
    learned: dict[str, np.ndarray], random: dict[str, np.ndarray], theta_true_padded: np.ndarray,
    num_sources: int, theta_size: int, bt_cfg: BayesTransportConfig, cfg: PolicyConfig,
    destination: Path,
):
    S = int(num_sources); D = int(theta_size // num_sources)
    if D != 2:
        return
    theta_true = np.asarray(theta_true_padded)[:S,:D]
    grid = np.linspace(bt_cfg.design_low, bt_cfg.design_high, cfg.grid_size)
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    field = source_log_mean_np(theta_true, points, bt_cfg).reshape(xx.shape)
    fig, axes = plt.subplots(1,2,figsize=(12.5,5.5), constrained_layout=True)
    for ax, (name, traj) in zip(axes, (("Strategic", learned),("Random", random))):
        contour = ax.contourf(xx, yy, field, levels=35)
        designs = traj["designs"].reshape(-1, bt_cfg.max_source_dim)[:,:2]
        steps_here = int(np.asarray(traj["designs"]).shape[0])
        designs_here = int(np.asarray(traj["designs"]).shape[1])
        step_ids = np.repeat(np.arange(1, steps_here + 1), designs_here)
        sc = ax.scatter(designs[:,0], designs[:,1], c=step_ids, s=42, marker="x")
        ax.scatter(theta_true[:,0], theta_true[:,1], marker="*", s=190, color="#111111", edgecolors="white", linewidths=0.7)
        ax.set_title(f"{name} design locations"); ax.set_aspect("equal")
        ax.set_xlabel("design x1"); ax.set_ylabel("design x2")
        fig.colorbar(sc, ax=ax, label="decision step")
    fig.suptitle("Where the policies choose to measure on the fixed source field", fontsize=14, fontweight="bold")
    fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


#%% 14) Policy-specific fixed diagnostic: fresh downstream example, independent of Bayes training
def _choose_policy_fixed_diagnostic_shape(
    bt_cfg: BayesTransportConfig, cfg: PolicyConfig
) -> tuple[int, int]:
    """Choose one deterministic downstream shape for plots that persist through policy training.

    The original Bayes-Transport ``fixed_trajectory.npz`` was only an upstream inference diagnostic.
    It is deliberately NOT a constraint on the downstream design-policy task.  For a heterogeneous
    policy we prefer a 2-D allowed shape when available so the physical source-field plots remain
    informative; otherwise we use the first allowed policy-training shape.
    """
    # shapes = policy_shape_pool(bt_cfg, cfg)
    # two_d = [shape for shape in shapes if shape[1] == 2]
    # return tuple(two_d[0] if two_d else shapes[0])

    shapes = policy_shape_pool(bt_cfg, cfg)
    preferred_shape = (2, 2)
    if preferred_shape not in shapes:
        raise ValueError(
            f"Requested visualization shape {preferred_shape} is not available "
            f"in the policy shape pool: {shapes}"
        )
    return preferred_shape

def _describe_original_bayes_fixed_trajectory(bt_cfg: BayesTransportConfig, cfg: PolicyConfig) -> None:
    """Print upstream fixed-trajectory metadata for provenance only; never reject the policy task."""
    path = Path(cfg.bayes_artifact_dir).expanduser().resolve() / "fixed_trajectory.npz"
    if not path.is_file():
        print(
            "Original Bayes fixed_trajectory.npz was not found. That is fine: the downstream "
            "policy creates and saves its own fixed visualization example."
        )
        return
    try:
        data = dict(np.load(path, allow_pickle=False))
        S = int(np.asarray(data.get("num_sources", bt_cfg.num_sources)))
        size = int(np.asarray(data.get("theta_size", S * bt_cfg.source_dim)))
        D = int(size // S)
        print(
            f"Original Bayes fixed trajectory: (S={S}, D={D}). This was only a diagnostic from "
            "Bayes-Transport training and does NOT constrain the downstream policy shapes."
        )
    except Exception as exc:
        print(
            "WARNING: could not read original Bayes fixed_trajectory.npz for provenance "
            f"({exc!r}). Downstream policy diagnostics are generated independently."
        )


def make_or_load_policy_fixed_reference(
    bt_cfg: BayesTransportConfig, cfg: PolicyConfig, policy_run_dir: Path
) -> dict[str, np.ndarray]:
    """Create one fresh, policy-shaped example and reuse it for every training visualization.

    New policy runs generate this example ONCE from deterministic policy-specific seeds and save it
    as ``artefacts/policy_fixed_reference.npz``. Reloaded runs reuse that exact saved file.  Thus all
    epoch-indexed fixed plots compare the evolving policy on the same theta*, initial cloud, random
    baseline and Gaussian observation-noise realization, without depending on the shape or horizon of
    the old Bayes-Transport diagnostic.
    """
    policy_path = policy_run_dir / "artefacts" / "policy_fixed_reference.npz"
    _describe_original_bayes_fixed_trajectory(bt_cfg, cfg)

    if policy_path.is_file():
        data = dict(np.load(policy_path, allow_pickle=False))
        print(
            "Policy fixed diagnostic: reusing saved downstream example "
            f"{policy_path} with (S={int(data['num_sources'])}, "
            f"D={int(data['theta_size']) // int(data['num_sources'])})."
        )
        return {name: np.asarray(value) for name, value in data.items()}

    S, D = _choose_policy_fixed_diagnostic_shape(bt_cfg, cfg)
    print(
        f"Policy fixed diagnostic: creating a FRESH downstream example at (S={S}, D={D}). "
        "This example is independent of the original Bayes fixed trajectory and will be reused "
        "unchanged throughout policy training."
    )

    # Distinct deterministic seeds make this genuinely policy-specific rather than silently
    # reproducing the Bayes diagnostic. The same sampled epsilon is used by strategic and random
    # policies so the fixed comparison is paired.
    truth_rng = np.random.default_rng(cfg.seed + 25_000)
    prior_rng = np.random.default_rng(cfg.seed + 26_000)
    experiment_rng = np.random.default_rng(cfg.seed + 27_000)

    theta_active = sample_base_prior_np(
        truth_rng, 1, bt_cfg, num_sources=S, source_dim=D
    )[0]
    prior_active = sample_base_prior_np(
        prior_rng, cfg.num_particles, bt_cfg, num_sources=S, source_dim=D
    )

    theta_true = np.zeros(
        (bt_cfg.max_num_sources, bt_cfg.max_source_dim), dtype=np.float32
    )
    theta_true[:S, :D] = theta_active
    prior_particles = pad_theta_np(prior_active, bt_cfg)

    T = int(cfg.experimental_budget)
    K = int(cfg.designs_per_step)
    random_designs = experiment_rng.uniform(
        bt_cfg.design_low, bt_cfg.design_high, size=(T, K, D)
    ).astype(np.float32)
    eps = experiment_rng.normal(size=(T, K)).astype(np.float32)
    random_mean = source_log_mean_np(theta_active, random_designs, bt_cfg).astype(np.float32)
    random_outcomes = random_mean + np.float32(bt_cfg.observation_noise_std) * eps

    random_observations = np.zeros(
        (T, K, bt_cfg.max_source_dim + 1), dtype=np.float32
    )
    random_observations[:, :, :D] = random_designs
    random_observations[:, :, -1] = random_outcomes

    fixed = {
        "theta_true": theta_true,
        "prior_particles": prior_particles,
        "num_sources": np.asarray(S, dtype=np.int32),
        "theta_size": np.asarray(S * D, dtype=np.int32),
        "observation_noise": eps,
        "random_reference_observations": random_observations,
    }
    np.savez_compressed(policy_path, **fixed)
    print("Saved policy-specific fixed diagnostic:", policy_path)
    return fixed


def rollout_prescribed_observations(
    bayes_model: SequentialBayesModel, theta_true: Array, prior_particles: Array,
    observations: Array, num_sources: Array, theta_size: Array,
    bt_cfg: BayesTransportConfig, cfg: PolicyConfig,
) -> dict[str, Array]:
    """Replay a fixed design/outcome sequence through Bayes Transport.

    This is used for the policy-specific fixed random baseline. The prescribed sequence is
    generated once for this downstream policy run and then held constant across all epochs.
    """
    initial_metrics = cloud_metrics_jax(prior_particles, theta_true, num_sources, theta_size, bt_cfg)

    def decision_step(cloud, obs_block):
        before = cloud_metrics_jax(cloud, theta_true, num_sources, theta_size, bt_cfg)

        cloud = bayes_update_observation_set(
            bayes_model, cloud, obs_block, num_sources, theta_size
        )
        after = cloud_metrics_jax(cloud, theta_true, num_sources, theta_size, bt_cfg)
        output = {
            "posterior_cloud": cloud,
            "designs": obs_block[:, :bt_cfg.max_source_dim],
            "outcomes": obs_block[:, -1],
            "observations": obs_block,
            "reward": reward_from_metrics(before, after, cfg),
            "log_prob": jnp.asarray(0.0, dtype=cloud.dtype),
            "entropy": jnp.asarray(0.0, dtype=cloud.dtype),
            "mean_std": jnp.asarray(0.0, dtype=cloud.dtype),
            "diversity_penalty": jnp.asarray(0.0, dtype=cloud.dtype),
            "energy_score": after["energy_score"],
            "attraction": after["attraction"],
            "repulsion": after["repulsion"],
            "rmse": after["rmse"],
            "spread": after["spread"],
        }
        return cloud, output

    _, trajectory = jax.lax.scan(decision_step, prior_particles, observations)
    trajectory["initial_energy_score"] = initial_metrics["energy_score"]
    trajectory["initial_attraction"] = initial_metrics["attraction"]
    trajectory["initial_rmse"] = initial_metrics["rmse"]
    trajectory["initial_spread"] = initial_metrics["spread"]
    return trajectory


def fixed_policy_pair_rollout(
    policy: PolicyModule, bayes_model: SequentialBayesModel, fixed: dict[str,np.ndarray],
    bt_cfg: BayesTransportConfig, cfg: PolicyConfig,
) -> tuple[dict[str,np.ndarray], dict[str,np.ndarray]]:
    truth = jnp.asarray(fixed["theta_true"]); prior = jnp.asarray(fixed["prior_particles"])
    S = jnp.asarray(fixed["num_sources"]); size = jnp.asarray(fixed["theta_size"])
    eps = jnp.asarray(fixed["observation_noise"])
    key = jax.random.key(cfg.seed + 31_000)
    learned = rollout_episode(
        policy, bayes_model, truth, prior, S, size, key, bt_cfg, cfg,
        policy_kind="learned", deterministic=True, fixed_observation_noise=eps,
    )
    random = rollout_prescribed_observations(
        bayes_model, truth, prior, jnp.asarray(fixed["random_reference_observations"]),
        S, size, bt_cfg, cfg,
    )
    to_np = lambda tree: jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)
    return to_np(learned), to_np(random)


#%% 15) Create/reload policy run and instantiate both models
np.random.seed(POLICY_CFG.seed)
print("JAX devices:", jax.devices())
print("\nPolicy configuration:\n", yaml.safe_dump(asdict(POLICY_CFG), sort_keys=False))
print(
    f"Experimental budget: {POLICY_CFG.experimental_budget} sequential decisions/episode; "
    f"{POLICY_CFG.designs_per_step} design(s)/decision -> "
    f"{POLICY_CFG.experimental_budget * POLICY_CFG.designs_per_step} likelihood evaluations/episode."
)
print(
    f"Training schedule: {POLICY_CFG.epochs} epochs x "
    f"{POLICY_CFG.train_steps_per_epoch} fresh optimizer steps/epoch = "
    f"{POLICY_CFG.epochs * POLICY_CFG.train_steps_per_epoch:,} optimizer updates."
)
print("Recovered Bayes-Transport compatibility configuration:\n", yaml.safe_dump(asdict(BT_CFG), sort_keys=False))
print("Bayes observation handling:", BAYES_OBSERVATION_COMPATIBILITY)
if joint_aline_mode(POLICY_CFG):
    max_context_observations = POLICY_CFG.experimental_budget * POLICY_CFG.designs_per_step
    print(
        "Joint ALINE context: acquisition always consumes the ENTIRE accumulated design-outcome "
        f"history, up to {max_context_observations} observations per episode. "
        f"The pretrained likelihood path saw prefixes {BT_CFG.min_observations_per_step}.."
        f"{BT_CFG.max_observations_per_step}; longer downstream histories are allowed as "
        "sequence-length extrapolation and may be adapted when finetune_bayes_transport=True."
    )
print("Archived Bayes-Transport source:", BAYES_SOURCE_SCRIPT)

BAYES_CHECKPOINT = find_bayes_checkpoint(POLICY_CFG)
bayes_model = load_bayes_model(BAYES_CHECKPOINT, BT_CFG)
print("Loaded Bayes Transport:", BAYES_CHECKPOINT)
print(f"Bayes Transport parameters: {count_parameters(bayes_model):,}")

if POLICY_CFG.train_policy:
    policy_run_dir = make_policy_run_dir(POLICY_CFG)
    archived_policy_script = copy_running_script_to_run_dir(policy_run_dir)
    save_json(policy_run_dir / "artefacts" / "policy_config.json", asdict(POLICY_CFG))
    save_json(policy_run_dir / "artefacts" / "bayes_transport_recovered_config.json", asdict(BT_CFG))
    print("Policy run directory:", policy_run_dir)
    print("Archived policy script:", archived_policy_script)
else:
    if POLICY_CFG.policy_reload_dir is None:
        raise ValueError("Set policy_reload_dir when train_policy=False.")
    policy_run_dir = Path(POLICY_CFG.policy_reload_dir).expanduser().resolve()
    for child in ("plots","artefacts","tables"):
        (policy_run_dir / child).mkdir(parents=True, exist_ok=True)
    print("Reloading policy run:", policy_run_dir)

if joint_aline_mode(POLICY_CFG):
    policy = JointALINEDesignPolicy(
        bayes_model, BT_CFG, POLICY_CFG, key=jax.random.key(POLICY_CFG.seed + 10_000)
    )
else:
    policy = SeparateDesignPolicy(
        bayes_model, BT_CFG, POLICY_CFG, key=jax.random.key(POLICY_CFG.seed + 10_000)
    )

print(f"Design policy parameters (including reused component copies): {count_parameters(policy):,}")
print("Requested Bayes components for policy reuse:", reused_component_names(POLICY_CFG) or "none")
if policy.joint_aline_mode:
    print(
        "Training regime: JOINT ALINE-style inference + acquisition. All four Bayes components "
        "are reused; one posterior-Transformer pass produces both the posterior cloud and the "
        "next design-set features. Policy gradients update the acquisition branch only."
    )
    print(
        f"Joint posterior conditioning inherited from Bayes Transport: {BT_CFG.posterior_conditioning} "
        "(AdaLN or observation cross-attention is preserved exactly)."
    )
else:
    print("Training regime: SEPARATE design policy with optional component reuse.")
    print(
        "Actually reused components:",
        tuple(name for name, used in (
            ("prior_embedder", policy.uses_reused_prior),
            ("likelihood_transformer", policy.uses_reused_likelihood),
            ("observation_embedder", policy.uses_reused_observation),
            ("posterior_transformer", policy.uses_reused_posterior),
        ) if used) or "none",
    )

if policy.fixed_shape_input:
    if policy.joint_aline_mode and not (
        BT_CFG.min_num_sources == BT_CFG.max_num_sources
        and BT_CFG.min_source_dim == BT_CFG.max_source_dim
    ):
        print(
            "Policy shape mode: fixed downstream task on a dimension-agnostic Bayes backbone. "
            "JOINT mode intentionally retains the pretrained agnostic theta/observation embedders; "
            "fixing downstream (S,D) does not require replacing them with fresh linear maps."
        )
    else:
        print(
            "Policy shape mode: fixed. In SEPARATE mode the theta/observation interfaces are fresh "
            "trainable linear projections; a genuinely fixed Bayes checkpoint can be reused only "
            "at its exact physical shape."
        )
else:
    print("Policy shape mode: heterogeneous/dimension-agnostic.")

fixed_reference = make_or_load_policy_fixed_reference(BT_CFG, POLICY_CFG, policy_run_dir)

# Fixed evaluation set is generated ONCE and kept unchanged throughout training, just like the
# original Bayes-Transport validation diagnostics.  Random/strategic evaluation shares the same
# theta*, prior clouds and JAX episode keys for paired comparisons.
eval_rng = np.random.default_rng(POLICY_CFG.seed + 20_000)
eval_batch_np = make_policy_batch_np(
    eval_rng, POLICY_CFG.n_eval_trajectories, BT_CFG, POLICY_CFG, balanced_shapes=True
)


#%% 16) Optimizers and jitted train steps
policy_optimizer = optax.chain(
    optax.clip_by_global_norm(POLICY_CFG.grad_clip_norm),
    optax.adamw(POLICY_CFG.learning_rate, weight_decay=POLICY_CFG.weight_decay),
)
policy_opt_state = policy_optimizer.init(eqx.filter(policy, eqx.is_array))

# Do not allocate Adam moments for the large Bayes Transport unless fine-tuning is requested.
# This materially reduces accelerator memory in the default frozen-inference setting.
if POLICY_CFG.finetune_bayes_transport:
    bayes_optimizer = optax.chain(
        optax.clip_by_global_norm(POLICY_CFG.bayes_finetune_grad_clip_norm),
        optax.adamw(
            POLICY_CFG.bayes_finetune_learning_rate,
            weight_decay=POLICY_CFG.bayes_finetune_weight_decay,
        ),
    )
    bayes_opt_state = bayes_optimizer.init(eqx.filter(bayes_model, eqx.is_array))
else:
    bayes_optimizer = None
    bayes_opt_state = None


@eqx.filter_jit
def policy_train_step(
    candidate_policy: PolicyModule,
    candidate_opt_state,
    frozen_bayes: SequentialBayesModel,
    batch: dict[str, Array],
    keys: Array,
    baseline_by_t: Array,
):
    (loss, aux), grads = eqx.filter_value_and_grad(policy_batch_objective, has_aux=True)(
        candidate_policy, frozen_bayes, batch, keys, baseline_by_t, BT_CFG, POLICY_CFG
    )
    params = eqx.filter(candidate_policy, eqx.is_array)
    updates, candidate_opt_state = policy_optimizer.update(grads, candidate_opt_state, params)
    updated_policy = eqx.apply_updates(candidate_policy, updates)
    # Reused Bayes components are frozen under the policy loss by default. Restore them exactly
    # after AdamW so only the acquisition/policy-specific parameters move.
    updated_policy = restore_frozen_policy_components(
        updated_policy, candidate_policy, POLICY_CFG
    )
    grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
    return updated_policy, candidate_opt_state, loss, aux, grad_norm


@eqx.filter_jit
def bayes_finetune_step(
    candidate_bayes: SequentialBayesModel,
    candidate_opt_state,
    current_policy: PolicyModule,
    batch: dict[str, Array],
    keys: Array,
):
    (loss, aux), grads = eqx.filter_value_and_grad(bayes_finetune_batch_objective, has_aux=True)(
        candidate_bayes, current_policy, batch, keys, BT_CFG, POLICY_CFG
    )
    params = eqx.filter(candidate_bayes, eqx.is_array)
    if bayes_optimizer is None:
        raise RuntimeError("bayes_finetune_step called while finetune_bayes_transport=False.")
    updates, candidate_opt_state = bayes_optimizer.update(grads, candidate_opt_state, params)
    candidate_bayes = eqx.apply_updates(candidate_bayes, updates)
    grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
    return candidate_bayes, candidate_opt_state, loss, aux, grad_norm


#%% 17) Initial learned-vs-random diagnostic before policy training
initial_eval = evaluate_policy_pair(
    policy, bayes_model, eval_batch_np, jax.random.key(POLICY_CFG.seed + 40_000), BT_CFG, POLICY_CFG
)
initial_rows = evaluation_summary_rows(initial_eval, POLICY_CFG)
save_evaluation_table(initial_rows, policy_run_dir / "tables", "comparison_epoch_000000")
plot_policy_vs_random_metrics(
    initial_eval, policy_run_dir / "plots" / "policy_vs_random_epoch_000000.png", POLICY_CFG
)
fixed_learned, fixed_random = fixed_policy_pair_rollout(
    policy, bayes_model, fixed_reference, BT_CFG, POLICY_CFG
)
plot_fixed_policy_evolution(
    fixed_learned, fixed_reference["theta_true"], fixed_reference["prior_particles"],
    int(fixed_reference["num_sources"]), int(fixed_reference["theta_size"]), BT_CFG, POLICY_CFG,
    policy_run_dir / "plots" / "fixed_trajectory_epoch_000000.png",
    "Strategic-policy posterior evolution before policy training",
)
plot_fixed_policy_vs_random(
    fixed_learned, fixed_random, fixed_reference["theta_true"], fixed_reference["prior_particles"],
    int(fixed_reference["num_sources"]), int(fixed_reference["theta_size"]), BT_CFG, POLICY_CFG,
    policy_run_dir / "plots" / "fixed_trajectory_policy_vs_random_epoch_000000.png",
)


#%% 18) Train the Bayesian experimental-design policy
# Every optimizer update below uses a newly simulated batch.  The nested epoch structure is purely
# deliberate experiment bookkeeping: it gives us meaningful epoch-level losses, evaluation cadence,
# checkpoint cadence, and epoch-indexed fixed-trajectory figures without ever recycling a dataset.
history: dict[str, list] = {
    # Per-optimizer-step traces.
    "step": [], "epoch": [], "loss": [], "policy_loss": [], "objective": [], "mean_reward": [],
    "final_reward": [], "mean_entropy": [], "mean_policy_std": [], "diversity_penalty": [],
    "grad_norm": [], "train_final_energy_score": [], "train_final_attraction": [], "train_final_rmse": [],
    "bayes_finetune_loss": [], "bayes_finetune_grad_norm": [],

    # Per-epoch summaries.  These are means over the fresh optimizer steps in that epoch.
    "epoch_index": [], "epoch_loss": [], "epoch_policy_loss": [], "epoch_mean_reward": [],
    "epoch_grad_norm": [], "epoch_train_final_energy_score": [], "epoch_train_final_attraction": [],
    "epoch_train_final_rmse": [], "epoch_bayes_finetune_loss": [],

    # Evaluation traces.
    "eval_step": [], "eval_epoch": [], "eval_learned_spce": [], "eval_random_spce": [],
    "eval_learned_final_es": [], "eval_random_final_es": [],
    "eval_learned_final_rmse": [], "eval_random_final_rmse": [],
}

baseline_by_t = np.zeros((POLICY_CFG.experimental_budget,), dtype=np.float32)
train_rng = np.random.default_rng(POLICY_CFG.seed + 50_000)
master_key = jax.random.key(POLICY_CFG.seed + 60_000)

if POLICY_CFG.train_policy:
    global_step = 0
    progress = tqdm(range(1, POLICY_CFG.epochs + 1), desc="BED policy epochs")

    for epoch in progress:
        # Remember the slice occupied by this epoch so all epoch-level summaries below are exact
        # means over the fresh data seen during this epoch.
        epoch_start = len(history["loss"])

        for _step_in_epoch in range(1, POLICY_CFG.train_steps_per_epoch + 1):
            global_step += 1

            # IMPORTANT: this is called inside the optimizer-step loop, not once per epoch.  There
            # is therefore no finite training dataset and no repeated mini-batch across epochs.
            batch_np = make_policy_batch_np(train_rng, POLICY_CFG.batch_size, BT_CFG, POLICY_CFG)
            batch = batch_to_jax(batch_np)
            master_key, step_key = jax.random.split(master_key)
            episode_keys = jax.random.split(step_key, POLICY_CFG.batch_size)

            policy, policy_opt_state, loss, aux, grad_norm = policy_train_step(
                policy, policy_opt_state, bayes_model, batch, episode_keys, jnp.asarray(baseline_by_t)
            )

            aux_np = {name: np.asarray(jax.device_get(value)) for name, value in aux.items()}
            if POLICY_CFG.gradient_estimator == "reinforce" and POLICY_CFG.reinforce_baseline == "ema":
                signal = np.asarray(aux_np["baseline_signal_by_t"], dtype=np.float32)
                beta = POLICY_CFG.reinforce_baseline_decay
                baseline_by_t = beta * baseline_by_t + (1.0 - beta) * signal

            bayes_ft_loss = np.nan
            bayes_ft_grad = np.nan
            if POLICY_CFG.finetune_bayes_transport:
                master_key, bayes_key = jax.random.split(master_key)
                bayes_keys = jax.random.split(bayes_key, POLICY_CFG.batch_size)
                bayes_model, bayes_opt_state, bt_loss, bt_aux, bt_grad_norm = bayes_finetune_step(
                    bayes_model, bayes_opt_state, policy, batch, bayes_keys
                )
                bayes_ft_loss = float(jax.device_get(bt_loss))
                bayes_ft_grad = float(jax.device_get(bt_grad_norm))
                # Joint/separate policies can both reuse Bayes components. After inference
                # fine-tuning, synchronize every selected component back into the policy copy.
                if POLICY_CFG.sync_reused_components_after_bayes_finetune:
                    policy = sync_reused_policy_components_from_bayes(
                        policy, bayes_model, POLICY_CFG
                    )

            history["step"].append(global_step)
            history["epoch"].append(epoch)
            history["loss"].append(float(jax.device_get(loss)))
            history["policy_loss"].append(float(aux_np["policy_loss"]))
            history["objective"].append(float(aux_np["objective"]))
            history["mean_reward"].append(float(aux_np["mean_reward"]))
            history["final_reward"].append(float(aux_np["final_reward"]))
            history["mean_entropy"].append(float(aux_np["mean_entropy"]))
            history["mean_policy_std"].append(float(aux_np["mean_policy_std"]))
            history["diversity_penalty"].append(float(aux_np["diversity_penalty"]))
            history["grad_norm"].append(float(jax.device_get(grad_norm)))
            history["train_final_energy_score"].append(float(aux_np["final_energy_score"]))
            history["train_final_attraction"].append(float(aux_np["final_attraction"]))
            history["train_final_rmse"].append(float(aux_np["final_rmse"]))
            history["bayes_finetune_loss"].append(bayes_ft_loss)
            history["bayes_finetune_grad_norm"].append(bayes_ft_grad)

        # ------------------------- epoch aggregation -------------------------
        epoch_slice = slice(epoch_start, len(history["loss"]))

        def _epoch_mean(name: str) -> float:
            values = np.asarray(history[name][epoch_slice], dtype=float)
            return float(np.nanmean(values)) if len(values) and np.any(np.isfinite(values)) else float("nan")

        history["epoch_index"].append(epoch)
        history["epoch_loss"].append(_epoch_mean("loss"))
        history["epoch_policy_loss"].append(_epoch_mean("policy_loss"))
        history["epoch_mean_reward"].append(_epoch_mean("mean_reward"))
        history["epoch_grad_norm"].append(_epoch_mean("grad_norm"))
        history["epoch_train_final_energy_score"].append(_epoch_mean("train_final_energy_score"))
        history["epoch_train_final_attraction"].append(_epoch_mean("train_final_attraction"))
        history["epoch_train_final_rmse"].append(_epoch_mean("train_final_rmse"))
        history["epoch_bayes_finetune_loss"].append(_epoch_mean("bayes_finetune_loss"))

        progress.set_postfix(
            epoch_loss=f"{history['epoch_loss'][-1]:.4g}",
            reward=f"{history['epoch_mean_reward'][-1]:+.3g}",
            es=f"{history['epoch_train_final_energy_score'][-1]:.3g}",
            step=global_step,
        )

        # All expensive diagnostics now have a clear epoch cadence.  The fixed-trajectory filename
        # therefore contains a real epoch number rather than an optimizer-step number.
        if epoch % POLICY_CFG.eval_every_epochs == 0 or epoch == POLICY_CFG.epochs:
            evaluation = evaluate_policy_pair(
                policy, bayes_model, eval_batch_np,
                jax.random.key(POLICY_CFG.seed + 70_000 + global_step), BT_CFG, POLICY_CFG,
            )
            rows = evaluation_summary_rows(evaluation, POLICY_CFG)
            save_evaluation_table(rows, policy_run_dir / "tables", f"comparison_epoch_{epoch:06d}")
            history["eval_step"].append(global_step)
            history["eval_epoch"].append(epoch)
            history["eval_learned_spce"].append(float(np.mean(evaluation["spce_learned"][:,-1])))
            history["eval_random_spce"].append(float(np.mean(evaluation["spce_random"][:,-1])))
            history["eval_learned_final_es"].append(float(np.mean(evaluation["learned"]["energy_score"][:,-1])))
            history["eval_random_final_es"].append(float(np.mean(evaluation["random"]["energy_score"][:,-1])))
            history["eval_learned_final_rmse"].append(float(np.mean(evaluation["learned"]["rmse"][:,-1])))
            history["eval_random_final_rmse"].append(float(np.mean(evaluation["random"]["rmse"][:,-1])))
            plot_policy_vs_random_metrics(
                evaluation, policy_run_dir / "plots" / f"policy_vs_random_epoch_{epoch:06d}.png", POLICY_CFG
            )

        if epoch % POLICY_CFG.plot_every_epochs == 0 or epoch == POLICY_CFG.epochs:
            fixed_learned, fixed_random = fixed_policy_pair_rollout(
                policy, bayes_model, fixed_reference, BT_CFG, POLICY_CFG
            )
            # This historical filename convention is now semantically exact: suffix == epoch.
            plot_fixed_policy_evolution(
                fixed_learned, fixed_reference["theta_true"], fixed_reference["prior_particles"],
                int(fixed_reference["num_sources"]), int(fixed_reference["theta_size"]),
                BT_CFG, POLICY_CFG,
                policy_run_dir / "plots" / f"fixed_trajectory_epoch_{epoch:06d}.png",
                f"Strategic-policy posterior evolution after epoch {epoch:,} "
                f"({global_step:,} optimizer steps)",
            )
            plot_fixed_policy_vs_random(
                fixed_learned, fixed_random, fixed_reference["theta_true"], fixed_reference["prior_particles"],
                int(fixed_reference["num_sources"]), int(fixed_reference["theta_size"]),
                BT_CFG, POLICY_CFG,
                policy_run_dir / "plots" / f"fixed_trajectory_policy_vs_random_epoch_{epoch:06d}.png",
            )
            plot_design_field_comparison(
                fixed_learned, fixed_random, fixed_reference["theta_true"],
                int(fixed_reference["num_sources"]), int(fixed_reference["theta_size"]),
                BT_CFG, POLICY_CFG,
                policy_run_dir / "plots" / f"fixed_design_field_epoch_{epoch:06d}.png",
            )
            plot_training_diagnostics(history, policy_run_dir / "plots" / "training_diagnostics.png")

        if epoch % POLICY_CFG.save_every_epochs == 0 or epoch == POLICY_CFG.epochs:
            save_model(policy_run_dir / "artefacts" / f"policy_epoch_{epoch:06d}.eqx", policy)
            save_model(policy_run_dir / "artefacts" / "policy_last.eqx", policy)
            if POLICY_CFG.finetune_bayes_transport:
                save_model(
                    policy_run_dir / "artefacts" / f"bayes_transport_finetuned_epoch_{epoch:06d}.eqx",
                    bayes_model,
                )
                save_model(policy_run_dir / "artefacts" / "bayes_transport_finetuned_last.eqx", bayes_model)
            np.savez_compressed(
                policy_run_dir / "artefacts" / "training_history.npz",
                **{name: np.asarray(values) for name, values in history.items()},
                reinforce_baseline_by_t=np.asarray(baseline_by_t),
            )
else:
    # Reload a policy checkpoint using a skeleton built from THIS PolicyConfig and the recovered
    # Bayes architecture.  Edit PolicyConfig to match the run you are reloading.
    artifact_dir = policy_run_dir / "artefacts"
    candidates = (
        [artifact_dir / "policy_last.eqx"]
        + sorted(artifact_dir.glob("policy_epoch_*.eqx"), reverse=True)
        + sorted(artifact_dir.glob("policy_step_*.eqx"), reverse=True)  # backward compatibility
    )
    checkpoint = next((p for p in candidates if p.is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError(f"No policy checkpoint found in {artifact_dir}.")
    policy = eqx.tree_deserialise_leaves(checkpoint, policy)
    print("Reloaded policy:", checkpoint)
    ft_path = artifact_dir / "bayes_transport_finetuned_last.eqx"
    if POLICY_CFG.finetune_bayes_transport and ft_path.is_file():
        bayes_model = eqx.tree_deserialise_leaves(ft_path, bayes_model)
        policy = sync_reused_policy_components_from_bayes(policy, bayes_model, POLICY_CFG)
        print("Reloaded fine-tuned Bayes Transport and synchronized reused policy components:", ft_path)
    history_path = artifact_dir / "training_history.npz"
    if history_path.is_file():
        loaded = dict(np.load(history_path, allow_pickle=False))
        history = {name: list(value) for name, value in loaded.items() if name != "reinforce_baseline_by_t"}


#%% 19) Final evaluation, comparison table, and paper-style metric plots
final_eval = evaluate_policy_pair(
    policy, bayes_model, eval_batch_np,
    jax.random.key(POLICY_CFG.seed + 80_000), BT_CFG, POLICY_CFG,
)
final_rows = evaluation_summary_rows(final_eval, POLICY_CFG)
save_evaluation_table(final_rows, policy_run_dir / "tables", "comparison_final")
final_eig_rows = eig_final_summary_rows(final_eval, POLICY_CFG)
save_evaluation_table(final_eig_rows, policy_run_dir / "tables", "eig_final")
plot_eig_final(
    final_eval, policy_run_dir / "plots" / "eig_final.png", POLICY_CFG
)
plot_policy_vs_random_metrics(
    final_eval, policy_run_dir / "plots" / "policy_vs_random_final.png", POLICY_CFG
)

fixed_learned, fixed_random = fixed_policy_pair_rollout(
    policy, bayes_model, fixed_reference, BT_CFG, POLICY_CFG
)
plot_fixed_policy_evolution(
    fixed_learned, fixed_reference["theta_true"], fixed_reference["prior_particles"],
    int(fixed_reference["num_sources"]), int(fixed_reference["theta_size"]), BT_CFG, POLICY_CFG,
    policy_run_dir / "plots" / "fixed_trajectory_best_policy.png",
    "Best/latest strategic-policy posterior evolution",
)
plot_fixed_policy_vs_random(
    fixed_learned, fixed_random, fixed_reference["theta_true"], fixed_reference["prior_particles"],
    int(fixed_reference["num_sources"]), int(fixed_reference["theta_size"]), BT_CFG, POLICY_CFG,
    policy_run_dir / "plots" / "fixed_trajectory_policy_vs_random_final.png",
)
plot_design_field_comparison(
    fixed_learned, fixed_random, fixed_reference["theta_true"],
    int(fixed_reference["num_sources"]), int(fixed_reference["theta_size"]), BT_CFG, POLICY_CFG,
    policy_run_dir / "plots" / "fixed_design_field_final.png",
)
if "history" in globals():
    plot_training_diagnostics(history, policy_run_dir / "plots" / "training_diagnostics.png")

# Save the final fixed trajectory arrays so every policy plot/table is reproducible without rerunning.
np.savez_compressed(
    policy_run_dir / "artefacts" / "fixed_policy_vs_random_trajectory.npz",
    theta_true=fixed_reference["theta_true"],
    prior_particles=fixed_reference["prior_particles"],
    observation_noise=fixed_reference["observation_noise"],
    learned_designs=fixed_learned["designs"],
    learned_outcomes=fixed_learned["outcomes"],
    learned_rewards=fixed_learned["reward"],
    learned_energy_score=fixed_learned["energy_score"],
    learned_attraction=fixed_learned["attraction"],
    random_designs=fixed_random["designs"],
    random_outcomes=fixed_random["outcomes"],
    random_rewards=fixed_random["reward"],
    random_energy_score=fixed_random["energy_score"],
    random_attraction=fixed_random["attraction"],
)

summary = {
    "bayes_checkpoint": str(BAYES_CHECKPOINT),
    "bayes_source_script": None if BAYES_SOURCE_SCRIPT is None else str(BAYES_SOURCE_SCRIPT),
    "policy_config": asdict(POLICY_CFG),
    "bayes_transport_recovered_config": asdict(BT_CFG),
    "final_comparison": final_rows,
    "final_eig_table": final_eig_rows,
    "notes": {
        "reward": "consecutive empirical-cloud improvement; attraction-only or full energy score",
        "pathwise": "reparameterized Tanh-Gaussian design + Gaussian likelihood + differentiable frozen Bayes map",
        "reinforce": "detached environment trajectory + score-function log probability; optional EMA baseline",
        "eig_metric": "sequential Prior Contrastive Estimation lower bound (sPCE), reported in nats",
        "experimental_budget": "number of sequential policy decisions per episode; defaults to 30",
        "epochs": "every optimizer step uses fresh simulated data; epoch metrics are means over train_steps_per_epoch fresh updates",
        "fixed_visualisation": "fresh policy-specific theta*, initial cloud, random baseline, and Gaussian-noise realization are generated once per policy run and reused unchanged throughout training; the original Bayes fixed trajectory is provenance only",
    },
}
save_json(policy_run_dir / "artefacts" / "final_summary.json", summary)

print("\nFinal strategic-vs-random comparison")
for row in final_rows:
    print(row)
print("\nFinal sPCE / EIG lower-bound numbers (mean ± 95% CI)")
for row in final_eig_rows:
    print(
        f"{row['method']}: {row['mean_sPCE_EIG_nats']:.4f} ± "
        f"{row['ci95_halfwidth_nats']:.4f} nats"
    )
print("Policy outputs saved to:", policy_run_dir)
