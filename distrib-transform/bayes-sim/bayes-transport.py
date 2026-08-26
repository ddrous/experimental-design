#%% 1) Imports, configuration, and experiment conventions
"""Single-cloud Bayes transport with energy-score, DEQ, and drifting training modes.

The simulator, posterior Transformer, observation-prefix handling, physical particle
representation, heterogeneous (S,D) support, and evaluation protocol retain the original
structure. Training is selected by ``training_mode``:

* ``"energy_score"``: one prior->posterior transport optimized by the exact empirical
  multivariate energy score.
* ``"deq_fixed_point"``: one theta* supplies a finite sequence of fresh observation blocks.
  The resulting recurrent Bayes trajectory is represented as an augmented triangular equilibrium
  and differentiated with a custom implicit VJP. The memory-safe ``triangular`` solver exploits
  this causal structure directly; Picard and Anderson remain available for forward/backward
  ablations. No forward root-finder iteration is differentiated through.
* ``"drifting"``: one ordinary transport call, followed by frozen-target regression toward
  ``x + eta V(x)``. There is NO inner time/fixed-point rollout in this mode: as in Deng et al.,
  evolution occurs across optimizer steps because the network parameters change. The default
  field combines the paper's anti-symmetric kernel attraction/repulsion field with an optional
  energy-score-gradient contribution.

The ``energy_score`` and ``drifting`` modes can additionally train on the model's own historical
predictions. A bounded host-side replay buffer stores recent final-prefix output clouds together
with the theta* and observation block that produced that output (plus minimal shape metadata needed
for heterogeneous padded batches). With configurable probability a future training row starts from
one of these detached historical clouds instead of a fresh interpolated cloud and reuses the EXACT
same observation block. Thus the model is deliberately challenged with the same evidence but a
changed prior cloud, exposing it to its own
past output distribution without creating a cross-step autodiff graph. Otherwise the original
shared-tau interpolated prior remains the default input.

Evaluation is deliberately independent of ``training_mode``. Non-sequential evaluation always
uses the original direct ``predict_prefixes`` map and exact empirical energy score; sequential
evaluation always repeatedly applies the same learned Bayes map with ``jax.lax.scan``.

Synthetic fresh training priors are intentionally simple. Draw theta* from the configured base
prior (Uniform on the physical design box or isotropic Gaussian). With probability
``synthetic_prior_match_probability`` use theta* as the interpolation anchor, otherwise draw an
independent anchor from the same base prior. Draw one shared tau ~ Uniform[0,1] for the entire
cloud and set C_tau,n=(1-tau)Z_n+tau*anchor. No particle-wise tau laws or alternative truth
sampling modes are retained.

When both source count and source dimensionality are fixed, ``fixed_shape_learned_projection``
controls the input interface. If False, the historical parameter-free normalized physical inputs
are used. If True, each fixed-shape theta particle and each fixed-shape design/outcome token gets
one learnable linear projection into ``embedding_dim``. This gives fixed-shape runs the same
embedding-space interface used by heterogeneous-shape runs without instantiating the full TAMO
aggregation Transformers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import datetime
import json
import math
import itertools
import re
import shutil
from pathlib import Path
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from IPython.display import display
from tqdm.auto import tqdm
import yaml

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

Array = jax.Array

# Execution switch.
# - train_wm=True: create a new run and train the amortized iid transport model.
# - train_wm=False: reload an existing run from the current folder.
train_wm: bool = True

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})


@dataclass(frozen=True)
class BayesTransportConfig:
    """Defaults are the experiment; edit them here rather than in an override block."""

    # Reproducibility and run bookkeeping.
    env_name: str = "location-finding"
    seed: int = 2030
    runs_base: str = "./runs"

    # Source-localisation simulator. `num_sources` and `source_dim` define only the
    # fixed 2-D diagnostic problem; heterogeneous training still uses the ranges below.
    num_sources: int = 2
    source_dim: int = 2

    # ONE base prior is used throughout the experiment; there is no distribution over distinct prior laws.
    # "uniform" samples every active source coordinate independently inside the physical
    # design box.  "gaussian" uses the original isotropic N(0, prior_std^2 I) prior.
    base_prior_distribution: str = "uniform"  # {"uniform", "gaussian"}
    prior_std: float = 1.0

    # Synthetic-prior law used for every training mode. One tau ~ Uniform[0,1] is shared by
    # the whole cloud. The interpolation anchor equals the observation-generating theta* with
    # this probability and is otherwise an independent draw from the same base prior.
    synthetic_prior_match_probability: float = 1.0

    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.10

    # Heterogeneous training-task distribution.  Arrays are padded to these maxima,
    # while masks ensure that inactive source/coordinate slots never enter an embedder.
    # The listed held-out combinations are NEVER sampled by the training stream; they are
    # reserved for the balanced dimensional-generalisation evaluation after training.
    min_num_sources: int = 2
    max_num_sources: int = 2
    min_source_dim: int = 2
    max_source_dim: int = 2
    # heldout_shapes: tuple[tuple[int, int], ...] = ((1, 6), (6, 1), (3, 3), (6, 6))
    # heldout_shapes: tuple[tuple[int, int], ...] = ((1, 4), (4, 1), (2, 2), (4, 4))
    heldout_shapes: tuple[tuple[int, int], ...] = ()

    # TAMO-style dimension aggregation is used when S and/or D varies during training. In a
    # fixed-shape run, the expensive aggregation Transformers are still bypassed, but a single
    # learnable linear projection can optionally map both theta and (design,outcome) inputs into
    # the same E-dimensional interface used by heterogeneous runs.
    embedding_dim: int = 192
    fixed_shape_learned_projection: bool = False
    dimension_embedder_depth: int = 4
    scalar_encoder_depth: int = 4
    embedding_heads: int = 8

    # Observation-prefix and particle counts.  Every iid training item contains Omax observations.
    # Every prefix o=Omin,...,Omax is optimized on every gradient step; there is no random
    # batch-level observation count.  test_observations_per_step and evaluation_trajectory_length
    # are used ONLY by evaluation-time recurrent Bayes diagnostics and physical visualisations.
    min_observations_per_step: int = 1
    max_observations_per_step: int = 16
    test_observations_per_step: int = 16
    num_particles: int = 12
    n_train_trajectories: int = 4096*2
    n_eval_trajectories: int = 256
    batch_size: int = 16*16

    # Single sequential-evaluation horizon used by evaluation diagnostics. In reload mode this
    # field is intentionally taken from the CURRENT script configuration, so a saved model can be
    # stress-tested on a longer trajectory without retraining.
    evaluation_trajectory_length: int = 8*2
    n_evaluation_trajectories_per_shape: int = 16

    # Maximum recurrent horizon used ONLY by DEQ training. One theta* is held fixed for the
    # whole row and each temporal coordinate receives a fresh observation block Y_t ~ p(Y|theta*).
    # Drifting is intentionally one-step and does not use this setting.
    training_fixed_point_max_iterations: int = 8*2

    # Continuous host-side iid simulator stream.  n_train_trajectories is retained for backward
    # configuration compatibility but now means the number of FRESH iid joint samples consumed
    # per nominal epoch.  No training trajectory is stored or revisited.
    train_dataloader_num_workers: int = 0
    train_dataloader_prefetch_factor: int = 2

    # Causal likelihood Transformer.  These hyperparameters are used when more than one
    # observation per step can occur.  Then the module owns its learned input projection and
    # causal self-attention exactly as before: fixed-shape (D+1) -> likelihood_hidden_dim,
    # heterogeneous embedding_dim -> likelihood_hidden_dim.  If Omin=Omax=1, the module is
    # retained for architectural compatibility but becomes a zero-parameter identity pass-through;
    # the Posterior Transformer receives the single normalized/aggregated observation directly.
    likelihood_hidden_dim: int = 192
    likelihood_heads: int = 8
    likelihood_mlp_ratio: int = 4
    likelihood_depth: int = 4

    # Posterior Transformer. `posterior_conditioning` selects observation cross-attention or
    # direct AdaLN conditioning. Energy-score and drifting training use one prior->posterior call
    # per prefix; DEQ training and sequential evaluation recurrently reuse the same map.
    posterior_conditioning: str = "adaln"  # {"cross_attention", "adaln"}
    hidden_dim: int = 256
    heads: int = 8
    mlp_ratio: int = 4
    posterior_depth: int = 6
    max_embedding_displacement: float = 6.0  # retained value; now caps physical theta displacement
    canonicalize_particle_sources: bool = False

    # Observation normalisation.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Training objective. Evaluation does NOT branch on this option.
    #   "energy_score"     : direct one-step empirical energy score.
    #   "deq_fixed_point" : augmented equilibrium + implicit-function custom VJP.
    #   "drifting"        : one-step frozen drifting-field target regression.
    training_mode: str = "energy_score"

    # Historical-output replay used ONLY by energy_score and drifting. The buffer contains the
    # most recent final-prefix model output clouds together with the theta* and EXACT observation
    # block used to produce each output (plus minimal S,D metadata for padded heterogeneous runs).
    # With this probability, a row replaces C_tau by a detached historical output and reuses that
    # stored theta*/observation pair unchanged. A value of 0 disables replay.
    historical_output_prior_probability: float = 0.5
    historical_output_buffer_capacity: int = 2048

    # DEQ fixed-point solver. The DEQ state is the WHOLE T-step recurrent cloud trajectory.
    # "triangular" is an exact causal solver for this particular augmented equilibrium and is
    # strongly preferred when memory matters. Picard/Anderson are retained as configurable
    # generic fixed-point solvers. The backward pass can likewise use the exact triangular
    # implicit recursion, reuse the forward solver ("same"), or choose Picard/Anderson.
    fixed_point_solver: str = "triangular"  # {"triangular", "picard", "anderson"}
    fixed_point_backward_solver: str = "triangular"  # {"same", "triangular", "picard", "anderson"}
    fixed_point_max_steps: int = 40
    fixed_point_backward_max_steps: int = 40
    fixed_point_tolerance: float = 1e-4
    fixed_point_backward_tolerance: float = 1e-5
    fixed_point_relaxation: float = 0.75
    fixed_point_anderson_history: int = 5
    fixed_point_anderson_ridge: float = 1e-4
    # Optional diagonal stabilisation of (I-J_T^T)u=g for generic backward solvers. Zero gives
    # the exact IFT equation. The triangular augmented DEQ has I-J_F invertible by construction.
    fixed_point_backward_regularization: float = 0.0

    # Drifting objective/field choice.
    #   "energy_score"          : optimize the empirical proper energy score directly. This is a
    #                             scalar objective rather than a vector field, retained as the
    #                             proper-score drifting ablation requested here.
    #   "energy_score_gradient" : frozen target x + eta*(-N grad_x ES), using the analytic gradient.
    #   "kernel"                : Deng et al. Algorithm-2 attraction/repulsion field.
    #   "kernel_energy_score_gradient": Deng field plus a weighted energy-score-gradient field.
    # Historical aliases "energy" and "kernel_energy" are accepted by the helper below.
    drifting_field: str = "energy_score_gradient"
    drifting_temperatures: tuple[float, ...] = (0.05, 0.2, 0.8)
    drifting_eta: float = 1.0
    drifting_energy_weight: float = 0.25
    drifting_distance_epsilon: float = 1e-6

    # Optimisation. The scalar differentiated loss is selected above. The direct empirical
    # energy score is still reported for every mode and remains the validation/model-selection
    # metric, preserving the original evaluation protocol.
    epochs: int = 5000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1000.0
    # Validation-driven ReduceLROnPlateau.  The fixed iid validation set is evaluated once
    # per epoch; after this many non-improving epochs the effective learning rate is halved.
    lr_plateau_patience: int = 1000
    lr_plateau_rtol: float = 1e-4

    # Persistence / visualisation cadence.
    save_every_epochs: int = 2000
    final_plot_examples: int = 3
    grid_size: int = 180

    # Reference-posterior diagnostic only; never enters the training loss.
    reference_proposals: int = 10_000
    reference_particles: int = 2_000

    # Limit / theorem diagnostics after training.
    limit_eval_trajectories: int = 192
    particle_limit_values: tuple[int, ...] = (16, 32, 64)
    trajectory_mc_values: tuple[int, ...] = (8, 16, 32, 64)
    prior_resample_repeats: int = 12


def validate_config(cfg: BayesTransportConfig):
    """Fail early on shape combinations that would invalidate the padded/JAX layout."""
    if cfg.min_num_sources < 1 or cfg.min_source_dim < 1:
        raise ValueError("min_num_sources and min_source_dim must both be >= 1.")
    if cfg.max_num_sources < cfg.min_num_sources:
        raise ValueError("max_num_sources must be >= min_num_sources.")
    if cfg.max_source_dim < cfg.min_source_dim:
        raise ValueError("max_source_dim must be >= min_source_dim.")
    max_theta_size = cfg.max_num_sources * cfg.max_source_dim
    fixed_shape = (
        cfg.min_num_sources == cfg.max_num_sources
        and cfg.min_source_dim == cfg.max_source_dim
    )
    if cfg.embedding_dim < 1:
        raise ValueError("embedding_dim must be >= 1.")
    # The full dimension-aggregation architecture is used only for heterogeneous-shape runs.
    # Fixed-shape learned projections may use E without imposing the heterogeneous aggregator's
    # max-theta-size constraint.
    if not fixed_shape and max_theta_size > cfg.embedding_dim:
        raise ValueError(
            f"max theta size S*D={max_theta_size} exceeds embedding_dim E={cfg.embedding_dim}. "
            "Increase embedding_dim or reduce the heterogeneous training range."
        )
    if not (cfg.min_num_sources <= cfg.num_sources <= cfg.max_num_sources):
        raise ValueError("fixed visualisation num_sources must lie inside the padded range.")
    if not (cfg.min_source_dim <= cfg.source_dim <= cfg.max_source_dim):
        raise ValueError("fixed visualisation source_dim must lie inside the padded range.")
    if not fixed_shape and cfg.embedding_dim % cfg.embedding_heads != 0:
        raise ValueError("embedding_dim must be divisible by embedding_heads.")
    if cfg.likelihood_hidden_dim < 1:
        raise ValueError("likelihood_hidden_dim must be >= 1.")
    if cfg.likelihood_heads < 1:
        raise ValueError("likelihood_heads must be >= 1.")
    if cfg.likelihood_hidden_dim % cfg.likelihood_heads != 0:
        raise ValueError("likelihood_hidden_dim must be divisible by likelihood_heads.")
    if cfg.likelihood_mlp_ratio < 1:
        raise ValueError("likelihood_mlp_ratio must be >= 1.")
    if cfg.likelihood_depth < 1:
        raise ValueError("likelihood_depth must be >= 1.")
    if cfg.hidden_dim % cfg.heads != 0:
        raise ValueError("hidden_dim must be divisible by heads.")
    if cfg.posterior_conditioning not in {"cross_attention", "adaln"}:
        raise ValueError("posterior_conditioning must be 'cross_attention' or 'adaln'.")
    if (
        not fixed_shape
        and cfg.posterior_conditioning == "cross_attention"
        and cfg.embedding_dim % cfg.heads != 0
    ):
        raise ValueError("embedding_dim must be divisible by heads for posterior cross-attention.")
    if cfg.min_observations_per_step < 1:
        raise ValueError("min_observations_per_step must be >= 1.")
    if cfg.max_observations_per_step < cfg.min_observations_per_step:
        raise ValueError("max_observations_per_step must be >= min_observations_per_step.")
    if not (cfg.min_observations_per_step <= cfg.test_observations_per_step <= cfg.max_observations_per_step):
        raise ValueError(
            "test_observations_per_step must lie in "
            "[min_observations_per_step, max_observations_per_step]."
        )
    if cfg.evaluation_trajectory_length < 1:
        raise ValueError("evaluation_trajectory_length must be >= 1.")
    if cfg.training_fixed_point_max_iterations < 1:
        raise ValueError("training_fixed_point_max_iterations must be >= 1.")
    if cfg.n_evaluation_trajectories_per_shape < 1:
        raise ValueError("n_evaluation_trajectories_per_shape must be >= 1.")
    all_shapes = {
        (s, d)
        for s in range(cfg.min_num_sources, cfg.max_num_sources + 1)
        for d in range(cfg.min_source_dim, cfg.max_source_dim + 1)
    }
    heldout = {tuple(map(int, shape)) for shape in cfg.heldout_shapes}
    if not heldout.issubset(all_shapes):
        raise ValueError("Every heldout_shapes entry must lie inside the configured S,D ranges.")
    if heldout == all_shapes:
        raise ValueError("heldout_shapes cannot remove every training shape.")
    if cfg.base_prior_distribution not in {"uniform", "gaussian"}:
        raise ValueError("base_prior_distribution must be 'uniform' or 'gaussian'.")
    if cfg.prior_std <= 0.0:
        raise ValueError("prior_std must be > 0 for Gaussian-base-prior support and normalization.")
    if not (0.0 <= cfg.synthetic_prior_match_probability <= 1.0):
        raise ValueError("synthetic_prior_match_probability must lie in [0, 1].")
    if cfg.training_mode not in {"energy_score", "deq_fixed_point", "drifting"}:
        raise ValueError(
            "training_mode must be 'energy_score', 'deq_fixed_point', or 'drifting'."
        )
    if not (0.0 <= cfg.historical_output_prior_probability <= 1.0):
        raise ValueError("historical_output_prior_probability must lie in [0, 1].")
    if cfg.historical_output_buffer_capacity < 1:
        raise ValueError("historical_output_buffer_capacity must be >= 1.")
    if cfg.fixed_point_solver not in {"triangular", "picard", "anderson"}:
        raise ValueError("fixed_point_solver must be 'triangular', 'picard', or 'anderson'.")
    if cfg.fixed_point_backward_solver not in {"same", "triangular", "picard", "anderson"}:
        raise ValueError(
            "fixed_point_backward_solver must be 'same', 'triangular', 'picard', or 'anderson'."
        )
    if cfg.fixed_point_max_steps < 1 or cfg.fixed_point_backward_max_steps < 1:
        raise ValueError("fixed-point max-step counts must both be >= 1.")
    if (
        cfg.training_mode == "deq_fixed_point"
        and cfg.fixed_point_solver == "picard"
        and cfg.fixed_point_max_steps < cfg.training_fixed_point_max_iterations
    ):
        raise ValueError(
            "For deq_fixed_point with the Picard solver, fixed_point_max_steps must be at "
            "least training_fixed_point_max_iterations so information can propagate through "
            "the full augmented recurrent trajectory."
        )
    resolved_backward_solver = (
        cfg.fixed_point_solver
        if cfg.fixed_point_backward_solver == "same"
        else cfg.fixed_point_backward_solver
    )
    if (
        cfg.training_mode == "deq_fixed_point"
        and resolved_backward_solver == "picard"
        and cfg.fixed_point_backward_max_steps < cfg.training_fixed_point_max_iterations
    ):
        raise ValueError(
            "For deq_fixed_point with a Picard backward solve, "
            "fixed_point_backward_max_steps must be at least "
            "training_fixed_point_max_iterations."
        )
    if cfg.fixed_point_tolerance <= 0.0 or cfg.fixed_point_backward_tolerance <= 0.0:
        raise ValueError("fixed-point tolerances must both be > 0.")
    if not (0.0 < cfg.fixed_point_relaxation <= 1.0):
        raise ValueError("fixed_point_relaxation must lie in (0, 1].")
    if cfg.fixed_point_anderson_history < 1:
        raise ValueError("fixed_point_anderson_history must be >= 1.")
    if cfg.fixed_point_anderson_ridge <= 0.0:
        raise ValueError("fixed_point_anderson_ridge must be > 0.")
    if cfg.fixed_point_backward_regularization < 0.0:
        raise ValueError("fixed_point_backward_regularization must be >= 0.")
    if cfg.drifting_field not in {
        "energy_score",
        "energy_score_gradient",
        "kernel",
        "kernel_energy_score_gradient",
        # Backward-compatible aliases from the previous revision.
        "energy",
        "kernel_energy",
    }:
        raise ValueError(
            "drifting_field must be 'energy_score', 'energy_score_gradient', 'kernel', "
            "or 'kernel_energy_score_gradient'."
        )
    if len(cfg.drifting_temperatures) < 1 or any(t <= 0.0 for t in cfg.drifting_temperatures):
        raise ValueError("drifting_temperatures must contain only positive values.")
    if cfg.drifting_eta <= 0.0:
        raise ValueError("drifting_eta must be > 0.")
    if cfg.drifting_energy_weight < 0.0:
        raise ValueError("drifting_energy_weight must be >= 0.")
    if cfg.drifting_distance_epsilon <= 0.0:
        raise ValueError("drifting_distance_epsilon must be > 0.")
    if cfg.design_high <= cfg.design_low:
        raise ValueError("design_high must be greater than design_low.")
    if cfg.train_dataloader_num_workers < 0:
        raise ValueError("train_dataloader_num_workers must be >= 0.")
    if cfg.train_dataloader_prefetch_factor < 1:
        raise ValueError("train_dataloader_prefetch_factor must be >= 1.")
    if cfg.num_particles < 2:
        raise ValueError("num_particles must be >= 2 for cloud posterior diagnostics.")
    if cfg.lr_plateau_patience < 1:
        raise ValueError("lr_plateau_patience must be >= 1.")
    if cfg.lr_plateau_rtol < 0.0:
        raise ValueError("lr_plateau_rtol must be >= 0.")


# One active configuration only.  No config file is saved or loaded.  A training run copies
# this script into its run folder; when that copied script is executed with train_wm=False,
# BayesTransportConfig below is therefore the complete source of architecture and runtime values.
def _reload_log_candidates(run_dir: Path) -> list[Path]:
    """Return a short, deterministic list of plausible nohup logs for an interrupted run."""
    directories = [run_dir, *list(run_dir.parents)[:3]]
    candidates: list[Path] = []
    for directory in directories:
        for name in ("nohup.log", "nohup.out"):
            candidate = directory / name
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _find_reload_nohup_log(run_dir: Path) -> Path | None:
    """Prefer a nearby nohup log whose recorded run directory matches this reload folder."""
    existing = [path for path in _reload_log_candidates(run_dir) if path.is_file()]
    if not existing:
        return None
    for path in existing:
        try:
            # The run-directory declaration is near the start, so avoid reading a potentially
            # very long training log merely to identify it.
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                prefix = handle.read(64_000)
            if f"Run directory: {run_dir}" in prefix or run_dir.name in prefix:
                return path
        except OSError:
            continue
    return existing[0]


_script_cfg = BayesTransportConfig()
CFG = _script_cfg
_reload_nohup_path: Path | None = None
if not train_wm:
    _reload_run_dir = Path.cwd().expanduser().resolve()
    _reload_nohup_path = _find_reload_nohup_log(_reload_run_dir)


validate_config(CFG)

ALL_SHAPES = tuple(
    (s, d)
    for s in range(CFG.min_num_sources, CFG.max_num_sources + 1)
    for d in range(CFG.min_source_dim, CFG.max_source_dim + 1)
)
HELDOUT_SHAPES = tuple(shape for shape in CFG.heldout_shapes if shape in ALL_SHAPES)
TRAIN_SHAPES = tuple(shape for shape in ALL_SHAPES if shape not in HELDOUT_SHAPES)
FIXED_SINGLE_SHAPE_SETUP = (
    CFG.min_num_sources == CFG.max_num_sources
    and CFG.min_source_dim == CFG.max_source_dim
)
RUN_DIMENSIONAL_GENERALISATION_DIAGNOSTICS = not FIXED_SINGLE_SHAPE_SETUP

#%% 2) Run directories and small persistence helpers
def make_run_dir(env_name: str, base: str | Path = "./runs") -> Path:
    """Create runs/<name>_<timestamp>/{plots,artefacts}."""
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = Path(base).expanduser().resolve() / f"{env_name}_{stamp}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=False)
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_running_script_to_run_dir(run_dir: Path) -> Path:
    """Copy the exact executable script into the run folder for future reloads.

    Reload mode intentionally reads NO saved configuration file.  Re-running this copied
    script from the run folder therefore reconstructs BayesTransportConfig directly from
    the source that created the run.
    """
    if "__file__" not in globals():
        raise RuntimeError(
            "Cannot archive the running script because __file__ is unavailable. "
            "Run this experiment from the .py script rather than an anonymous notebook cell."
        )
    source = Path(__file__).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Could not locate the running script: {source}")
    destination = run_dir / source.name
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def save_json(path: str | Path, payload: dict[str, Any]):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_model(path: str | Path, model: "SequentialBayesModel"):
    eqx.tree_serialise_leaves(Path(path), model)


def load_model(
    path: str | Path,
    cfg: BayesTransportConfig,
    *,
    key: Array | None = None,
) -> "SequentialBayesModel":
    """Rebuild the matching skeleton and load Equinox leaves."""
    if key is None:
        key = jax.random.key(0)
    skeleton = SequentialBayesModel(cfg, key=key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)

#%% 3) Base prior, synthetic interpolated training priors, and source-localisation simulator
def sample_base_prior_np(
    rng: np.random.Generator,
    n: int,
    cfg: BayesTransportConfig = CFG,
    *,
    num_sources: int | None = None,
    source_dim: int | None = None,
) -> np.ndarray:
    """Draw n iid full-theta samples from the ONE configured base prior.

    Uniform mode draws every active source coordinate independently inside the physical
    design box.  Gaussian mode preserves the historical isotropic N(0, prior_std^2 I)
    prior.  This is the t=0 distribution used for sequential evaluation and for the
    shared-tau synthetic-prior construction used only during training.
    """
    S = cfg.num_sources if num_sources is None else int(num_sources)
    D = cfg.source_dim if source_dim is None else int(source_dim)
    if S > cfg.max_num_sources or D > cfg.max_source_dim:
        raise ValueError("Requested base-prior shape exceeds configured padded limits.")
    shape = (int(n), S, D)
    if cfg.base_prior_distribution == "uniform":
        samples = rng.uniform(cfg.design_low, cfg.design_high, size=shape).astype(np.float32)
    elif cfg.base_prior_distribution == "gaussian":
        samples = rng.normal(0.0, cfg.prior_std, size=shape).astype(np.float32)
    else:
        raise ValueError("base_prior_distribution must be 'uniform' or 'gaussian'.")
    if cfg.canonicalize_particle_sources:
        samples = canonicalize_sources_np(samples)
    return samples


def sample_interpolated_training_prior_and_truth_np(
    rng: np.random.Generator,
    n_particles: int,
    cfg: BayesTransportConfig = CFG,
    *,
    num_sources: int,
    source_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Sample the only retained synthetic training law: shared-tau interpolation.

    1. theta* ~ rho_0 is the observation-generating truth.
    2. The interpolation anchor equals theta* with probability ``synthetic_prior_match_probability``;
       otherwise it is an independent draw from rho_0.
    3. Z_n iid ~ rho_0 and one tau ~ Uniform[0,1] is shared by every particle.
    4. C_tau,n = (1-tau) Z_n + tau * anchor.

    This exposes every training mode to concentrated and misspecified versions of the same base
    prior while keeping the prior-cloud design deliberately minimal.
    """
    S = int(num_sources)
    D = int(source_dim)
    n_particles = int(n_particles)
    if n_particles < 1:
        raise ValueError("n_particles must be >= 1 when constructing a synthetic prior cloud.")

    theta_true = sample_base_prior_np(
        rng, 1, cfg, num_sources=S, source_dim=D
    )[0]
    if rng.random() < cfg.synthetic_prior_match_probability:
        anchor = theta_true.copy()
    else:
        anchor = sample_base_prior_np(
            rng, 1, cfg, num_sources=S, source_dim=D
        )[0]

    base_particles = sample_base_prior_np(
        rng, n_particles, cfg, num_sources=S, source_dim=D
    )
    tau = np.float32(rng.uniform(0.0, 1.0))
    prior_particles = (
        (np.float32(1.0) - tau) * base_particles + tau * anchor[None, :, :]
    ).astype(np.float32)
    if cfg.canonicalize_particle_sources:
        prior_particles = canonicalize_sources_np(prior_particles)
    return prior_particles, np.asarray(theta_true, dtype=np.float32)


def sample_interpolated_prior_given_truth_np(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    n_particles: int,
    cfg: BayesTransportConfig = CFG,
    *,
    num_sources: int,
    source_dim: int,
) -> np.ndarray:
    """Sample C_tau conditional on an already supplied theta* under the retained shared-tau law."""
    S = int(num_sources)
    D = int(source_dim)
    theta_true = np.asarray(theta_true, dtype=np.float32)[:S, :D]
    if cfg.canonicalize_particle_sources:
        theta_true = canonicalize_sources_np(theta_true)
    if rng.random() < cfg.synthetic_prior_match_probability:
        anchor = theta_true
    else:
        anchor = sample_base_prior_np(
            rng, 1, cfg, num_sources=S, source_dim=D
        )[0]
    base_particles = sample_base_prior_np(
        rng, int(n_particles), cfg, num_sources=S, source_dim=D
    )
    tau = np.float32(rng.uniform(0.0, 1.0))
    prior_particles = (
        (np.float32(1.0) - tau) * base_particles + tau * anchor[None, :, :]
    ).astype(np.float32)
    if cfg.canonicalize_particle_sources:
        prior_particles = canonicalize_sources_np(prior_particles)
    return prior_particles

def _base_prior_plot_extent(cfg: BayesTransportConfig = CFG) -> float:
    """Natural one-coordinate plotting extent for the configured t=0 prior."""
    if cfg.base_prior_distribution == "uniform":
        return max(abs(float(cfg.design_low)), abs(float(cfg.design_high)))
    return 3.0 * float(cfg.prior_std)


def pad_theta_np(theta: np.ndarray, cfg: BayesTransportConfig = CFG) -> np.ndarray:
    """Pad [...,S,D] theta arrays to [...,Smax,Dmax] without changing active values."""
    theta = np.asarray(theta, dtype=np.float32)
    if theta.shape[-2] > cfg.max_num_sources or theta.shape[-1] > cfg.max_source_dim:
        raise ValueError("theta exceeds configured padding limits.")
    padded = np.zeros(
        theta.shape[:-2] + (cfg.max_num_sources, cfg.max_source_dim), dtype=np.float32
    )
    padded[..., : theta.shape[-2], : theta.shape[-1]] = theta
    return padded


def source_log_mean_np(
    theta: np.ndarray,
    designs: np.ndarray,
    cfg: BayesTransportConfig = CFG,
) -> np.ndarray:
    """Forward-model mean E[y | theta, x] on the log-intensity scale.

    This remains the single physical source-field function in the notebook.  It is
    already dimension-generic: the final coordinate axis can be D=1,2,3,... .
    Broadcasting supports, for example:

      theta [S,D],     designs [T,D]     -> [T]
      theta [S,D],     designs [T,O,D]   -> [T,O]
      theta [B,S,D],   designs [B,T,D]   -> [B,T]
      theta [P,S,D],   designs [T,D]     -> [P,T]
    """
    theta = np.asarray(theta, dtype=np.float64)
    designs = np.asarray(designs, dtype=np.float64)
    theta_expanded = np.expand_dims(theta, axis=-3)      # ... x 1 x S x D
    design_expanded = np.expand_dims(designs, axis=-2)   # ... x T x 1 x D
    dist_sq = np.sum((theta_expanded - design_expanded) ** 2, axis=-1)
    intensity = cfg.background + np.sum(
        cfg.source_strength / (cfg.softening + dist_sq), axis=-1
    )
    return np.log(intensity)


def _sample_problem_shapes_np(
    rng: np.random.Generator,
    n: int,
    cfg: BayesTransportConfig,
    *,
    fixed_num_sources: int | None = None,
    fixed_source_dim: int | None = None,
    shape_pool: tuple[tuple[int, int], ...] | None = None,
    balanced_shapes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample S and theta_size=S*D from an explicit set of allowed problem shapes.

    With no `shape_pool`, the training pool is the full configured S,D grid minus the
    held-out combinations.  `balanced_shapes=True` cycles through the pool before shuffling,
    which is useful for validation/test grids; the continuous training stream samples shapes
    randomly from the same pool.  Fixed S or D constraints are applied after choosing the pool.
    """
    if shape_pool is None:
        heldout = {tuple(map(int, shape)) for shape in cfg.heldout_shapes}
        # Unconstrained sampling is the training distribution and excludes held-out shapes.
        # Explicit fixed-shape diagnostics are allowed anywhere inside the configured grid.
        include_heldout = fixed_num_sources is not None or fixed_source_dim is not None
        shape_pool = tuple(
            (s, d)
            for s in range(cfg.min_num_sources, cfg.max_num_sources + 1)
            for d in range(cfg.min_source_dim, cfg.max_source_dim + 1)
            if include_heldout or (s, d) not in heldout
        )

    candidates = [
        (int(s), int(d))
        for s, d in shape_pool
        if (fixed_num_sources is None or int(s) == int(fixed_num_sources))
        and (fixed_source_dim is None or int(d) == int(fixed_source_dim))
    ]
    if not candidates:
        raise ValueError("No problem shapes remain after applying the requested constraints.")

    if balanced_shapes:
        chosen = np.asarray([candidates[i % len(candidates)] for i in range(int(n))], dtype=np.int32)
        rng.shuffle(chosen)
    else:
        chosen = np.asarray(
            [candidates[i] for i in rng.integers(0, len(candidates), size=int(n))],
            dtype=np.int32,
        )

    num_sources = chosen[:, 0]
    source_dim = chosen[:, 1]
    theta_size = (num_sources * source_dim).astype(np.int32)
    if np.any(theta_size > cfg.max_num_sources * cfg.max_source_dim):
        raise ValueError("Sampled theta_size exceeds configured padded theta width.")
    return num_sources.astype(np.int32), theta_size


def simulate_trajectories(
    rng: np.random.Generator,
    n_trajectories: int,
    trajectory_length: int,
    cfg: BayesTransportConfig = CFG,
    *,
    fixed_num_sources: int | None = None,
    fixed_source_dim: int | None = None,
    shape_pool: tuple[tuple[int, int], ...] | None = None,
    balanced_shapes: bool = False,
) -> dict[str, np.ndarray]:
    """Generate complete heterogeneous evaluation trajectories from the base prior.

    Each row draws one fresh theta* from the ONE configured base prior rho_0 and reuses that
    same theta* for all T observation blocks.  No synthetic interpolation is used here: these
    trajectories are for ordinary sequential evaluation, whose initial particle cloud is also
    sampled from rho_0 (the tau=0 case).

    Validation and diagnostics may precompute this dictionary.  Main training uses the
    ContinuousJointDataset below, which constructs its synthetic interpolated prior cloud and
    its actual theta* jointly on the CPU for every yielded row.

    Padded storage
    --------------
    theta_true[m]          [Smax,Dmax], active block [:S_m,:D_m]
    observations[m,t,o]    [Dmax+1], Omax candidate pairs per sequential step; design in
                              [:D_m], scalar y in the FINAL slot
    """
    n_trajectories = int(n_trajectories)
    trajectory_length = int(trajectory_length)
    num_sources, theta_size = _sample_problem_shapes_np(
        rng,
        n_trajectories,
        cfg,
        fixed_num_sources=fixed_num_sources,
        fixed_source_dim=fixed_source_dim,
        shape_pool=shape_pool,
        balanced_shapes=balanced_shapes,
    )

    theta_true = np.zeros(
        (n_trajectories, cfg.max_num_sources, cfg.max_source_dim), dtype=np.float32
    )
    observations = np.zeros(
        (
            n_trajectories,
            trajectory_length,
            cfg.max_observations_per_step,
            cfg.max_source_dim + 1,
        ),
        dtype=np.float32,
    )

    # Simulation remains host-side.  The neural/JAX loss never evaluates the physical likelihood.
    for m in range(n_trajectories):
        S = int(num_sources[m])
        D = int(theta_size[m] // num_sources[m])
        theta_active = sample_base_prior_np(
            rng, 1, cfg, num_sources=S, source_dim=D
        )[0]
        designs = canonicalize_designs_np(
            rng.uniform(
                cfg.design_low,
                cfg.design_high,
                size=(trajectory_length, cfg.max_observations_per_step, D),
            ).astype(np.float32)
        )
        mean = source_log_mean_np(theta_active, designs, cfg)
        readings = (
            mean + cfg.observation_noise_std * rng.normal(size=mean.shape)
        ).astype(np.float32)

        theta_true[m, :S, :D] = theta_active
        observations[m, :, :, :D] = designs
        observations[m, :, :, -1] = readings

    return {
        "theta_true": theta_true,
        "observations": observations,
        "num_sources": num_sources.astype(np.int32),
        "theta_size": theta_size.astype(np.int32),
    }


def simulate_iid_joint_samples(
    rng: np.random.Generator,
    n_samples: int,
    cfg: BayesTransportConfig = CFG,
    *,
    fixed_num_sources: int | None = None,
    fixed_source_dim: int | None = None,
    shape_pool: tuple[tuple[int, int], ...] | None = None,
    balanced_shapes: bool = False,
) -> dict[str, np.ndarray]:
    """Generate a fixed iid validation set from the SAME joint law as the training dataloader.

    Each row independently draws (S,D), C_tau, theta*, designs, and observation noise using
    `sample_interpolated_training_prior_and_truth_np`. C_tau is stored jointly with theta* so the
    validation set uses exactly the same realised cloud on every epoch.

    The Omax observations are multiple conditioning prefixes of the SAME iid joint datum,
    not recurrent posterior states.  A fixed NumPy seed makes this entire validation joint
    sample reproducible across epochs.
    """
    n_samples = int(n_samples)
    num_sources, theta_size = _sample_problem_shapes_np(
        rng,
        n_samples,
        cfg,
        fixed_num_sources=fixed_num_sources,
        fixed_source_dim=fixed_source_dim,
        shape_pool=shape_pool,
        balanced_shapes=balanced_shapes,
    )

    theta_true = np.zeros(
        (n_samples, cfg.max_num_sources, cfg.max_source_dim), dtype=np.float32
    )
    observations = np.zeros(
        (n_samples, cfg.max_observations_per_step, cfg.max_source_dim + 1),
        dtype=np.float32,
    )
    prior_particles = np.zeros(
        (
            n_samples,
            cfg.num_particles,
            cfg.max_num_sources,
            cfg.max_source_dim,
        ),
        dtype=np.float32,
    )

    for m in range(n_samples):
        S = int(num_sources[m])
        D = int(theta_size[m] // num_sources[m])
        active_prior, theta_active = sample_interpolated_training_prior_and_truth_np(
            rng,
            cfg.num_particles,
            cfg,
            num_sources=S,
            source_dim=D,
        )
        designs = canonicalize_designs_np(
            rng.uniform(
                cfg.design_low,
                cfg.design_high,
                size=(cfg.max_observations_per_step, D),
            ).astype(np.float32)
        )
        mean = source_log_mean_np(theta_active, designs, cfg)
        readings = (
            mean + cfg.observation_noise_std * rng.normal(size=mean.shape)
        ).astype(np.float32)

        theta_true[m, :S, :D] = theta_active
        observations[m, :, :D] = designs
        observations[m, :, -1] = readings
        prior_particles[m] = pad_theta_np(active_prior, cfg)

    return {
        "theta_true": theta_true,
        "observations": observations,
        "num_sources": num_sources.astype(np.int32),
        "theta_size": theta_size.astype(np.int32),
        "prior_particles": prior_particles,
    }


def make_iid_batch_np(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    rng: np.random.Generator,
    cfg: BayesTransportConfig = CFG,
    *,
    num_particles: int | None = None,
) -> dict[str, np.ndarray]:
    """Create one direct iid validation minibatch without changing its sampled joint law.

    Current validation datasets store the synthetic C_tau drawn jointly with theta* and Y,
    so this helper normally only slices those arrays. Legacy datasets without stored clouds can
    still reconstruct C_tau exactly from theta* under the retained shared-tau/match law.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n_particles = cfg.num_particles if num_particles is None else int(num_particles)
    if n_particles < 2:
        raise ValueError("Particle-cloud training requires num_particles >= 2.")

    batch_num_sources = dataset["num_sources"][indices].astype(np.int32)
    batch_theta_size = dataset["theta_size"][indices].astype(np.int32)

    if "prior_particles" in dataset:
        stored_prior = np.asarray(dataset["prior_particles"][indices], dtype=np.float32)
        if stored_prior.ndim != 4:
            raise ValueError("Stored iid prior_particles must have shape [B,N,Smax,Dmax].")
        if stored_prior.shape[1] != n_particles:
            raise ValueError(
                "This iid validation set stores the C_tau that was sampled jointly with theta*. "
                f"It contains N={stored_prior.shape[1]} particles, so it cannot be changed to "
                f"N={n_particles} without changing the validation joint law."
            )
        prior_particles = stored_prior
    else:
        # Backward compatibility for old validation dictionaries. The retained shared-tau law
        # has an exact conditional construction given theta*.
        batch_size = len(indices)
        prior_particles = np.zeros(
            (batch_size, n_particles, cfg.max_num_sources, cfg.max_source_dim),
            dtype=np.float32,
        )
        for b, (dataset_index, S_value, theta_size_value) in enumerate(
            zip(indices, batch_num_sources, batch_theta_size)
        ):
            S = int(S_value)
            theta_size_int = int(theta_size_value)
            if theta_size_int > cfg.max_num_sources * cfg.max_source_dim or theta_size_int % S != 0:
                raise ValueError("Invalid theta metadata in iid joint dataset.")
            D = theta_size_int // S
            theta_active = np.asarray(
                dataset["theta_true"][int(dataset_index)], dtype=np.float32
            )[:S, :D]
            active = sample_interpolated_prior_given_truth_np(
                rng,
                theta_active,
                n_particles,
                cfg,
                num_sources=S,
                source_dim=D,
            )
            prior_particles[b] = pad_theta_np(active, cfg)

    observations = np.asarray(dataset["observations"][indices], dtype=np.float32)
    if observations.ndim != 3:
        raise ValueError("iid training observations must have shape [B,Omax,Dmax+1].")
    return {
        "theta_true": dataset["theta_true"][indices].astype(np.float32),
        "observations": observations,
        "num_sources": batch_num_sources,
        "theta_size": batch_theta_size,
        "prior_particles": prior_particles,
    }


def make_batch_np(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    rng: np.random.Generator,
    cfg: BayesTransportConfig = CFG,
    *,
    num_particles: int | None = None,
    observations_per_step: int | None = None,
) -> dict[str, np.ndarray]:
    """Create one heterogeneous sequential-evaluation minibatch from the base prior.

    Every sequential trajectory starts from a fresh iid tau=0 cloud sampled from the ONE
    configured base prior rho_0.  This helper is evaluation-only; training uses the synthetic
    interpolated clouds produced by ContinuousJointDataset / make_iid_batch_np.

    Every stored sequential step contains Omax candidate observations.  With
    `observations_per_step=None`, this helper uses cfg.test_observations_per_step
    deterministically; passing an integer explicitly overrides it.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n_particles = cfg.num_particles if num_particles is None else int(num_particles)
    batch_size = len(indices)
    prior_particles = np.zeros(
        (
            batch_size,
            n_particles,
            cfg.max_num_sources,
            cfg.max_source_dim,
        ),
        dtype=np.float32,
    )

    batch_num_sources = dataset["num_sources"][indices].astype(np.int32)
    batch_theta_size = dataset["theta_size"][indices].astype(np.int32)
    for b, (S_value, theta_size_value) in enumerate(
        zip(batch_num_sources, batch_theta_size)
    ):
        S = int(S_value)
        theta_size_int = int(theta_size_value)
        if theta_size_int > cfg.max_num_sources * cfg.max_source_dim or theta_size_int % S != 0:
            raise ValueError("Invalid theta metadata in dataset.")
        D = theta_size_int // S
        active = sample_base_prior_np(
            rng,
            n_particles,
            cfg,
            num_sources=S,
            source_dim=D,
        )
        prior_particles[b] = pad_theta_np(active, cfg)

    # This helper is evaluation-only.  Observation count is deterministic unless the
    # caller explicitly supplies another value; training uses ALL prefixes via its dataloader.
    if observations_per_step is None:
        observation_count = np.asarray(cfg.test_observations_per_step, dtype=np.int32)
    else:
        fixed_count = int(observations_per_step)
        # print(f"Using fixed observations_per_step={fixed_count} for this batch.")
        if not (cfg.min_observations_per_step <= fixed_count <= cfg.max_observations_per_step):
            raise ValueError(
                "observations_per_step must lie in "
                "[min_observations_per_step, max_observations_per_step]."
            )
        observation_count = np.asarray(fixed_count, dtype=np.int32)

    return {
        "theta_true": dataset["theta_true"][indices].astype(np.float32),
        "observations": dataset["observations"][indices].astype(np.float32),
        "observation_count": observation_count,
        "num_sources": batch_num_sources,
        "theta_size": batch_theta_size,
        "prior_particles": prior_particles,
    }


def _trajectory_observation_count_np(
    trajectory: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
) -> np.ndarray:
    """Return the single observation count shared by all sequential steps in a direct/test call."""
    observations = np.asarray(trajectory["observations"])
    if observations.ndim == 2:
        # Backward-compatible shape handling for stored trajectories created before Omax.
        return np.asarray(1, dtype=np.int32)
    if observations.ndim != 3:
        raise ValueError("trajectory observations must have shape [T,Omax,Dmax+1].")

    if "observation_count" in trajectory:
        count = int(np.asarray(trajectory["observation_count"], dtype=np.int32).item())
    elif "observation_counts" in trajectory:
        # Compatibility with files produced by the previous implementation.  A legacy
        # schedule is accepted only when it already represents one constant count over T.
        legacy = np.asarray(trajectory["observation_counts"], dtype=np.int32)
        if legacy.ndim == 0:
            count = int(legacy.item())
        elif legacy.ndim == 1 and legacy.size > 0 and np.all(legacy == legacy[0]):
            count = int(legacy[0])
        else:
            raise ValueError(
                "Legacy observation_counts varies across sequential steps; the current "
                "model requires one observation_count shared by the full scan."
            )
    else:
        count = int(cfg.test_observations_per_step)

    if not (1 <= count <= observations.shape[1]):
        raise ValueError("observation_count is outside the available Omax range.")
    return np.asarray(count, dtype=np.int32)


def _ensure_observation_blocks_np(observations: np.ndarray) -> np.ndarray:
    """Normalise direct/test observation storage to [T,Omax,Dmax+1]."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim == 2:
        observations = observations[:, None, :]
    if observations.ndim != 3:
        raise ValueError("observations must have shape [T,Omax,Dmax+1].")
    return observations


def _flatten_used_observation_prefix_np(
    observations: np.ndarray,
    observation_count: np.ndarray,
    num_steps: int | None = None,
) -> np.ndarray:
    """Flatten the same active observation prefix from each of the first `num_steps` steps."""
    blocks = _ensure_observation_blocks_np(observations)
    count = int(np.asarray(observation_count, dtype=np.int32).item())
    steps = blocks.shape[0] if num_steps is None else int(num_steps)
    if not (0 <= steps <= blocks.shape[0]):
        raise ValueError("num_steps is outside the available sequential trajectory.")
    if not (1 <= count <= blocks.shape[1]):
        raise ValueError("observation_count is outside the available Omax range.")
    if steps == 0:
        return np.empty((0, blocks.shape[-1]), dtype=blocks.dtype)
    return blocks[:steps, :count].reshape(-1, blocks.shape[-1])


class ContinuousJointDataset(IterableDataset):
    """Infinite CPU stream of fresh simulator data plus one synthetic interpolated prior cloud.

    ``energy_score`` and ``drifting`` each receive an ordinary iid row: one theta*, one Omax
    observation block, and one C_tau cloud. Historical-output replay is applied later in the host
    training loop because it depends on model outputs from previous optimizer steps.

    ``deq_fixed_point`` instead receives one theta*, one C_tau cloud, and
    ``training_fixed_point_max_iterations`` conditionally independent observation blocks from that
    same theta*. Posterior clouds are never generated in the dataloader.
    """

    def __init__(self, cfg: BayesTransportConfig, *, seed: int):
        super().__init__()
        self.cfg = cfg
        self.seed = int(seed)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rng = np.random.default_rng(self.seed + 1_000_003 * worker_id)
        recurrent_training = self.cfg.training_mode == "deq_fixed_point"
        training_steps = (
            int(self.cfg.training_fixed_point_max_iterations)
            if recurrent_training else 1
        )

        while True:
            sampled_sources, sampled_theta_size = _sample_problem_shapes_np(
                rng, 1, self.cfg, shape_pool=TRAIN_SHAPES
            )
            S = int(sampled_sources[0])
            theta_size = int(sampled_theta_size[0])
            D = theta_size // S

            active_prior, theta_active = sample_interpolated_training_prior_and_truth_np(
                rng,
                self.cfg.num_particles,
                self.cfg,
                num_sources=S,
                source_dim=D,
            )

            designs = canonicalize_designs_np(
                rng.uniform(
                    self.cfg.design_low,
                    self.cfg.design_high,
                    size=(training_steps, self.cfg.max_observations_per_step, D),
                ).astype(np.float32)
            )
            mean = source_log_mean_np(theta_active, designs, self.cfg)
            readings = (
                mean + self.cfg.observation_noise_std * rng.normal(size=mean.shape)
            ).astype(np.float32)

            observation_trajectory = np.zeros(
                (
                    training_steps,
                    self.cfg.max_observations_per_step,
                    self.cfg.max_source_dim + 1,
                ),
                dtype=np.float32,
            )
            observation_trajectory[:, :, :D] = designs
            observation_trajectory[:, :, -1] = readings
            observations = (
                observation_trajectory
                if recurrent_training
                else observation_trajectory[0]
            )

            yield {
                "theta_true": pad_theta_np(theta_active, self.cfg),
                "observations": observations,
                "num_sources": np.asarray(S, dtype=np.int32),
                "theta_size": np.asarray(theta_size, dtype=np.int32),
                "prior_particles": pad_theta_np(active_prior, self.cfg),
            }


def _compact_cloud_to_padded_np(
    compact_cloud: np.ndarray,
    num_sources: int,
    theta_size: int,
    cfg: BayesTransportConfig,
) -> np.ndarray:
    """Inverse of compact_theta_jax for a whole N-particle cloud, on the host."""
    compact_cloud = np.asarray(compact_cloud, dtype=np.float32)
    S = int(num_sources)
    theta_size = int(theta_size)
    if compact_cloud.ndim != 2:
        raise ValueError("compact replay cloud must have shape [N,Kmax].")
    if S < 1 or theta_size < 1 or theta_size % S != 0:
        raise ValueError("Invalid compact replay cloud metadata.")
    D = theta_size // S
    active = compact_cloud[:, :theta_size].reshape(compact_cloud.shape[0], S, D)
    return pad_theta_np(active, cfg)


class HistoricalOutputPriorBuffer:
    """Bounded FIFO replay of detached model outputs for future energy/drifting rows.

    Each entry carries only the past OUTPUT cloud, its theta*, and the EXACT observation block
    used when that output was produced. Minimal S,D metadata is also retained because padded
    heterogeneous clouds cannot be interpreted unambiguously from values alone. The original input
    cloud is intentionally not stored. Replaying an entry therefore asks whether the model can take
    its own previous output as a changed prior while seeing the same evidence again.
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("Historical-output replay capacity must be >= 1.")
        self._entries: deque[dict[str, np.ndarray | int]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._entries)

    def add_batch(
        self,
        compact_clouds: np.ndarray,
        batch_np: dict[str, np.ndarray],
        cfg: BayesTransportConfig,
    ) -> None:
        """Store one final-prefix output cloud for every row in the just-trained minibatch."""
        compact_clouds = np.asarray(compact_clouds, dtype=np.float32)
        if compact_clouds.ndim != 3:
            raise ValueError("replay outputs must have shape [B,N,Kmax].")
        if compact_clouds.shape[0] != len(batch_np["theta_true"]):
            raise ValueError("replay output batch dimension does not match training metadata.")
        for b in range(compact_clouds.shape[0]):
            S = int(np.asarray(batch_np["num_sources"][b]).item())
            size = int(np.asarray(batch_np["theta_size"][b]).item())
            self._entries.append({
                # This is the model OUTPUT from the just-completed step. It is named
                # prior_particles only because that is the input field it will occupy on replay.
                "prior_particles": _compact_cloud_to_padded_np(
                    compact_clouds[b], S, size, cfg
                ).copy(),
                "theta_true": np.asarray(batch_np["theta_true"][b], dtype=np.float32).copy(),
                "observations": np.asarray(batch_np["observations"][b], dtype=np.float32).copy(),
                "num_sources": S,
                "theta_size": size,
            })

    def mix_into_batch(
        self,
        batch_np: dict[str, np.ndarray],
        rng: np.random.Generator,
        cfg: BayesTransportConfig,
    ) -> tuple[dict[str, np.ndarray], int]:
        """Replace selected C_tau rows by historical outputs and reuse their exact evidence."""
        if len(self) == 0 or cfg.historical_output_prior_probability <= 0.0:
            return batch_np, 0
        observations = np.asarray(batch_np["observations"])
        if observations.ndim != 3:
            raise ValueError(
                "Historical-output replay is only defined for one-step energy_score/drifting "
                "training batches [B,Omax,Dmax+1]."
            )

        mixed = {name: np.array(value, copy=True) for name, value in batch_np.items()}
        use_history = rng.random(len(mixed["theta_true"])) < cfg.historical_output_prior_probability
        selected = np.flatnonzero(use_history)
        for b in selected:
            entry = self._entries[int(rng.integers(0, len(self._entries)))]
            mixed["prior_particles"][b] = np.asarray(entry["prior_particles"], dtype=np.float32)
            mixed["theta_true"][b] = np.asarray(entry["theta_true"], dtype=np.float32)
            S = int(entry["num_sources"])
            size = int(entry["theta_size"])
            mixed["num_sources"][b] = np.asarray(S, dtype=np.int32)
            mixed["theta_size"][b] = np.asarray(size, dtype=np.int32)
            mixed["observations"][b] = np.asarray(
                entry["observations"], dtype=np.float32
            )
        return mixed, int(selected.size)


def _numpy_collate(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Keep DataLoader output as NumPy rather than converting through Torch tensors."""
    if not samples:
        raise ValueError("Cannot collate an empty Mode-A minibatch.")
    return {
        name: np.stack([np.asarray(sample[name]) for sample in samples], axis=0)
        for name in samples[0]
    }


def make_continuous_train_loader(
    cfg: BayesTransportConfig = CFG,
    *,
    seed: int,
) -> DataLoader:
    """Build the continuous loader; only DEQ rows contain a fresh-observation trajectory."""
    dataset = ContinuousJointDataset(cfg, seed=seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.train_dataloader_num_workers,
        "collate_fn": _numpy_collate,
        "drop_last": True,
    }
    if cfg.train_dataloader_num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.train_dataloader_prefetch_factor
    return DataLoader(**kwargs)


#%% 4) Source-label symmetry helpers
def canonicalize_sources_np(theta: np.ndarray) -> np.ndarray:
    """Sort ACTIVE exchangeable sources by increasing Euclidean norm to the origin."""
    theta = np.asarray(theta)
    key = np.linalg.norm(theta, axis=-1)
    order = np.argsort(key, axis=-1, kind="stable")
    return np.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_sources_jax(theta: Array) -> Array:
    key = jnp.linalg.norm(theta, axis=-1)
    order = jnp.argsort(key, axis=-1, stable=True)
    return jnp.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_padded_sources_np(theta: np.ndarray, num_sources: int) -> np.ndarray:
    """Canonicalize active source rows by norm, keeping inactive padding at the end."""
    theta = np.asarray(theta)
    indices = np.arange(theta.shape[-2])
    source_norm = np.linalg.norm(theta, axis=-1)
    key = np.where(indices < int(num_sources), source_norm, np.inf)
    order = np.argsort(key, axis=-1, kind="stable")
    return np.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_padded_sources_jax(theta: Array, num_sources: Array) -> Array:
    indices = jnp.arange(theta.shape[-2])
    source_norm = jnp.linalg.norm(theta, axis=-1)
    key = jnp.where(indices < num_sources, source_norm, jnp.inf)
    order = jnp.argsort(key, axis=-1, stable=True)
    return jnp.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_designs_np(designs: np.ndarray) -> np.ndarray:
    """Sort design points within each observation block by increasing norm to the origin."""
    designs = np.asarray(designs)
    key = np.linalg.norm(designs, axis=-1)
    order = np.argsort(key, axis=-1, kind="stable")
    return np.take_along_axis(designs, order[..., None], axis=-2)


def canonicalize_observation_block_np(observations: np.ndarray, source_dim: int) -> np.ndarray:
    """Sort design-outcome rows by design norm while keeping each outcome paired with its design."""
    observations = np.asarray(observations)
    key = np.linalg.norm(observations[..., : int(source_dim)], axis=-1)
    order = np.argsort(key, axis=-1, kind="stable")
    return np.take_along_axis(observations, order[..., None], axis=-2)


def canonicalize_observation_block_jax(
    observations: Array,
    num_sources: Array,
    theta_size: Array,
) -> Array:
    """JAX form of norm-ordering for one padded design-outcome observation block."""
    source_dim = theta_size // num_sources
    coordinate_index = jnp.arange(observations.shape[-1] - 1)
    valid_coordinate = coordinate_index < source_dim
    design = observations[..., :-1]
    design = jnp.where(valid_coordinate, design, 0.0)
    key = jnp.linalg.norm(design, axis=-1)
    order = jnp.argsort(key, axis=-1, stable=True)
    return jnp.take_along_axis(observations, order[..., None], axis=-2)


#%% 5) Token helpers shared by the dimension and posterior Transformers
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _mlp_tokens(layer: eqx.nn.MLP, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]




#%% 5b) TAMO-style dimension-agnostic scalar-to-vector embedders
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

    Before flattening, exchangeable source rows are canonicalized by increasing Euclidean
    norm to the origin.  We then compact the active [S,D] block into the FIRST S*D scalar slots
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



#%% 6) Causal observation Transformer and configurable particle conditioning
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


#%% 7) Physical-theta output head and direct posterior transports
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


#%% 8) End-to-end amortized model with reusable sequential recurrence
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
        observations = canonicalize_observation_block_jax(
            observations, num_sources, theta_size
        )
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

    def _canonicalize_compact_input(
        self,
        compact_particles: Array,
        num_sources: Array,
        theta_size: Array,
        max_num_sources: int,
        max_source_dim: int,
    ) -> Array:
        """Canonicalize a compact cloud immediately before it is reused as a model input."""
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

    def _canonicalize_compact_output(
        self,
        compact_particles: Array,
        num_sources: Array,
        theta_size: Array,
        max_num_sources: int,
        max_source_dim: int,
    ) -> Array:
        """Return posterior output unchanged; canonical order is expected rather than imposed."""
        del num_sources, theta_size, max_num_sources, max_source_dim
        return compact_particles

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
        current_theta = self._canonicalize_compact_input(
            current_theta,
            num_sources,
            theta_size,
            max_num_sources,
            max_source_dim,
        )
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
            current_theta = self._canonicalize_compact_input(
                current_theta,
                num_sources,
                theta_size,
                prior_particles.shape[-2],
                prior_particles.shape[-1],
            )
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

def count_parameters(module) -> int:
    return sum(
        x.size
        for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array))
    )


def print_model_parameter_count(model: SequentialBayesModel):
    observation_embedder = count_parameters(model.observation_embedder)
    likelihood_embedder = count_parameters(model.likelihood_embedder)
    theta_embedder = count_parameters(model.theta_embedder)
    posterior = count_parameters(model.posterior_transformer)

    total = observation_embedder + likelihood_embedder + theta_embedder + posterior
    if model.fixed_shape and not model.fixed_shape_learned_projection:
        if observation_embedder != 0 or theta_embedder != 0:
            raise RuntimeError(
                "Fixed-shape projection is disabled, so both input embedders must be parameter-free."
            )
    if model.fixed_shape_learned_projection:
        if observation_embedder == 0 or theta_embedder == 0:
            raise RuntimeError(
                "Fixed-shape learned projection is enabled, but an input projection has no parameters."
            )
    if model.single_observation_direct and likelihood_embedder != 0:
        raise RuntimeError(
            "Omin=Omax=1 requires the retained Likelihood Transformer to have zero parameters."
        )

    print(f"Total parameters: {total / 1e6:.3f} M")
    print(f"  Design-Outcome embedder : {observation_embedder:,}")
    print(f"  Likelihood Transformer  : {likelihood_embedder:,}")
    if model.single_observation_direct:
        print("    single-observation bypass: identity; conditioning projection lives in Posterior Transformer")
    print(f"  Theta embedder          : {theta_embedder:,}")
    print(f"  Posterior Transformer   : {posterior:,}")
    if model.fixed_shape:
        projection_label = "learned linear -> E" if model.fixed_shape_learned_projection else "parameter-free physical"
        print(
            f"  Fixed-shape input interface: {projection_label}; "
            f"theta={model.particle_input_dim}, observation={model.observation_input_dim}; "
            f"posterior conditioning context={model.observation_context_dim}"
        )

#%% 9) Empirical-cloud energy score and physical posterior diagnostics
def empirical_energy_score_terms_single(
    particle_theta: Array,
    target_theta: Array,
    theta_size: Array,
) -> tuple[Array, Array, Array]:
    """Exact multivariate energy score of one transported empirical particle measure.

    The transported cloud defines

        Q_hat = N^{-1} sum_n delta_{theta_n}.

    Its energy score at the joint-sample target theta* is exactly

        ES(Q_hat, theta*)
          = N^{-1} sum_n ||theta_n-theta*||
            - (2 N^2)^{-1} sum_{n,m} ||theta_n-theta_m||.

    Thus every particle is attracted to the SAME theta* that generated the conditioning
    observations, while the second term rewards distributional spread and prevents collapse.
    The particles do not need to be independent after transport: the score is applied to the
    whole realized empirical measure Q_hat.  This is the same finite-cloud score used in the
    original codebase.

    Only the first `theta_size` physical coordinates are active in heterogeneous padded batches.
    `repulsion` below denotes the full Q_hat x Q_hat mean distance, before the factor 1/2.
    """
    valid = (jnp.arange(particle_theta.shape[-1]) < theta_size).astype(particle_theta.dtype)
    target_sq = jnp.sum(
        (particle_theta - target_theta[None, :]) ** 2 * valid[None, :], axis=-1
    )
    attraction = jnp.mean(jnp.sqrt(target_sq + 1e-12))

    differences = particle_theta[:, None, :] - particle_theta[None, :, :]
    pair_sq = jnp.sum(differences**2 * valid[None, None, :], axis=-1)
    # The diagonal distances are exactly zero.  Masking them avoids the numerical epsilon
    # contributing N artificial self-distances while retaining the empirical N^2 denominator.
    off_diagonal = 1.0 - jnp.eye(particle_theta.shape[0], dtype=particle_theta.dtype)
    repulsion = jnp.sum(jnp.sqrt(pair_sq + 1e-12) * off_diagonal) / (
        particle_theta.shape[0] ** 2
    )
    energy_score = attraction - 0.5 * repulsion
    return energy_score, attraction, repulsion


def energy_score_single(
    particle_theta: Array,
    target_theta: Array,
    theta_size: Array,
) -> Array:
    """Return the exact empirical-cloud energy score used in training and evaluation."""
    energy_score, _, _ = empirical_energy_score_terms_single(
        particle_theta, target_theta, theta_size
    )
    return energy_score


def posterior_mean_rmse_single(
    particle_theta: Array,
    target_theta: Array,
    theta_size: Array,
) -> Array:
    """RMSE of the posterior mean over the active physical theta coordinates."""
    valid = (jnp.arange(particle_theta.shape[-1]) < theta_size).astype(particle_theta.dtype)
    squared_error = (jnp.mean(particle_theta, axis=0) - target_theta) ** 2
    return jnp.sqrt(jnp.sum(squared_error * valid) / jnp.maximum(theta_size, 1))


def posterior_spread_single(particle_theta: Array, theta_size: Array) -> Array:
    """Mean marginal posterior variance over active physical theta coordinates."""
    valid = (jnp.arange(particle_theta.shape[-1]) < theta_size).astype(particle_theta.dtype)
    variance = jnp.var(particle_theta, axis=0)
    return jnp.sum(variance * valid) / jnp.maximum(theta_size, 1)


def _compact_targets(
    theta_true: Array,
    num_sources: Array,
    theta_size: Array,
    cfg: BayesTransportConfig,
) -> Array:
    if cfg.canonicalize_particle_sources:
        theta_true = jax.vmap(canonicalize_padded_sources_jax)(theta_true, num_sources)
    return jax.vmap(compact_theta_jax)(theta_true, num_sources, theta_size)


def _prefix_metrics(
    posterior_by_prefix: Array,
    target_theta: Array,
    theta_size: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Vectorise empirical-cloud scoring and diagnostics over all observation prefixes."""
    energy, attraction, repulsion = jax.vmap(
        lambda particles: empirical_energy_score_terms_single(
            particles, target_theta, theta_size
        )
    )(posterior_by_prefix)
    rmse = jax.vmap(
        lambda particles: posterior_mean_rmse_single(particles, target_theta, theta_size)
    )(posterior_by_prefix)
    spread = jax.vmap(lambda particles: posterior_spread_single(particles, theta_size))(
        posterior_by_prefix
    )
    return energy, attraction, repulsion, rmse, spread


def _trajectory_metrics(
    posterior_sequence: Array,
    target_theta: Array,
    theta_size: Array,
) -> tuple[Array, Array, Array]:
    """Vectorise empirical-cloud diagnostics over a sequential cloud trajectory."""
    energy = jax.vmap(lambda p: energy_score_single(p, target_theta, theta_size))(
        posterior_sequence
    )
    rmse = jax.vmap(
        lambda p: posterior_mean_rmse_single(p, target_theta, theta_size)
    )(posterior_sequence)
    spread = jax.vmap(lambda p: posterior_spread_single(p, theta_size))(posterior_sequence)
    return energy, rmse, spread


def _direct_prediction_and_metrics(
    model: SequentialBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig = CFG,
) -> tuple[Array, dict[str, Array]]:
    """Original direct one-step prediction and exact empirical-energy diagnostics."""
    predicted, _, _, prefix_counts = jax.vmap(
        lambda prior, observations, sources, size: model.predict_prefixes(
            prior, observations, sources, size
        )
    )(
        batch["prior_particles"],
        batch["observations"],
        batch["num_sources"],
        batch["theta_size"],
    )
    target_theta = _compact_targets(
        batch["theta_true"], batch["num_sources"], batch["theta_size"], cfg
    )
    energy, attraction, repulsion, rmse, spread = jax.vmap(_prefix_metrics)(
        predicted, target_theta, batch["theta_size"]
    )
    metrics = {
        "loss": jnp.mean(energy),
        "energy_score": jnp.mean(energy),
        "final_energy_score": jnp.mean(energy[:, -1]),
        "posterior_mean_rmse": jnp.mean(rmse),
        "final_mean_rmse": jnp.mean(rmse[:, -1]),
        "posterior_spread": jnp.mean(spread),
        "final_spread": jnp.mean(spread[:, -1]),
        "attraction": jnp.mean(attraction),
        "repulsion": jnp.mean(repulsion),
        "energy_by_o": jnp.mean(energy, axis=0),
        "rmse_by_o": jnp.mean(rmse, axis=0),
        "spread_by_o": jnp.mean(spread, axis=0),
        "prefix_counts": prefix_counts[0],
    }
    return predicted, metrics


def _fixed_point_relative_residual(x: Array, fx: Array) -> Array:
    diff = fx - x
    numerator = jnp.sqrt(jnp.mean(jnp.square(diff)))
    denominator = jnp.maximum(
        jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(x))), jnp.sqrt(jnp.mean(jnp.square(fx)))),
        jnp.asarray(1.0, dtype=x.dtype),
    )
    return numerator / denominator


def _solve_fixed_point(
    fn,
    initial: Array,
    *,
    solver: str,
    max_steps: int,
    tolerance: float,
    relaxation: float,
    anderson_history: int,
    anderson_ridge: float,
) -> tuple[Array, Array, Array]:
    """Generic array fixed-point solver used in both DEQ forward and implicit VJP passes."""
    solver = str(solver)
    max_steps = int(max_steps)
    tol = jnp.asarray(tolerance, dtype=initial.dtype)
    relax = jnp.asarray(relaxation, dtype=initial.dtype)

    if solver == "picard":
        f0 = fn(initial)
        r0 = _fixed_point_relative_residual(initial, f0)
        init = (jnp.asarray(0, dtype=jnp.int32), initial, f0, r0)

        def cond(carry):
            step, _x, _fx, residual = carry
            return jnp.logical_and(step < max_steps, residual > tol)

        def body(carry):
            step, x, fx, _residual = carry
            x_next = x + relax * (fx - x)
            fx_next = fn(x_next)
            residual_next = _fixed_point_relative_residual(x_next, fx_next)
            return step + 1, x_next, fx_next, residual_next

        step, x, _fx, residual = eqx.internal.while_loop(
            cond, body, init, max_steps=max_steps, kind="lax"
        )
        return x, step, residual

    if solver != "anderson":
        raise ValueError(f"Unknown fixed-point solver {solver!r}.")

    history = int(anderson_history)
    flat_size = initial.size
    x_hist0 = jnp.zeros((history, flat_size), dtype=initial.dtype)
    f_hist0 = jnp.zeros((history, flat_size), dtype=initial.dtype)
    f0 = fn(initial)
    r0 = _fixed_point_relative_residual(initial, f0)
    init = (
        jnp.asarray(0, dtype=jnp.int32), initial, f0, r0, x_hist0, f_hist0
    )
    eye = jnp.eye(history, dtype=initial.dtype)
    ridge = jnp.asarray(anderson_ridge, dtype=initial.dtype)

    def cond(carry):
        step, _x, _fx, residual, _xh, _fh = carry
        return jnp.logical_and(step < max_steps, residual > tol)

    def body(carry):
        step, x, fx, _residual, x_hist, f_hist = carry
        slot = jnp.mod(step, history)
        x_hist = x_hist.at[slot].set(x.reshape(-1))
        f_hist = f_hist.at[slot].set(fx.reshape(-1))
        valid_count = jnp.minimum(step + 1, history)
        valid = (jnp.arange(history) < valid_count).astype(initial.dtype)
        g = (f_hist - x_hist) * valid[:, None]
        gram = g @ g.T + ridge * eye
        # Unused history rows are decoupled and forced to zero coefficient.
        gram = gram + (1.0 - valid)[:, None] * eye
        rhs = valid
        alpha = jnp.linalg.solve(gram, rhs)
        alpha = alpha * valid
        alpha = alpha / jnp.maximum(jnp.sum(alpha), jnp.asarray(1e-12, dtype=alpha.dtype))
        x_mix = jnp.sum(alpha[:, None] * x_hist, axis=0).reshape(initial.shape)
        f_mix = jnp.sum(alpha[:, None] * f_hist, axis=0).reshape(initial.shape)
        accelerated = (jnp.asarray(1.0, dtype=initial.dtype) - relax) * x_mix + relax * f_mix
        f_next = fn(accelerated)
        residual_next = _fixed_point_relative_residual(accelerated, f_next)
        return step + 1, accelerated, f_next, residual_next, x_hist, f_hist

    step, x, _fx, residual, _xh, _fh = eqx.internal.while_loop(
        cond, body, init, max_steps=max_steps, kind="lax"
    )
    return x, step, residual


def _deq_trajectory_operator(
    model: SequentialBayesModel,
    trajectory_state: Array,          # [T,N,Kmax]
    initial_theta: Array,             # [N,Kmax]
    observation_contexts: Array,      # [T,Omax,C]
    observation_count: Array,
    num_sources: Array,
    theta_size: Array,
    max_num_sources: int,
    max_source_dim: int,
) -> Array:
    """Augmented DEQ operator for a recurrent Bayes trajectory with fresh evidence.

    The fixed-point state is Z=(C_1,...,C_T). Its operator is triangular:

        F(Z)_1 = T_phi(C_0, Y_1)
        F(Z)_t = T_phi(C_{t-1}, Y_t),  t=2,...,T.

    Therefore the equilibrium is exactly the T-step recurrent Bayes rollout, but it can be
    differentiated with the implicit-function theorem as one augmented implicit layer. Every
    temporal coordinate calls the shared model with a DIFFERENT observation block. The outer
    root finder may reevaluate this deterministic augmented operator, so it reuses the pre-generated
    trajectory Y_1:T across root-finder evaluations; it never substitutes the same Y inside two
    different recurrent positions.
    """
    previous_clouds = jnp.concatenate(
        [initial_theta[None, ...], trajectory_state[:-1]], axis=0
    )

    def mapped_step(inputs):
        current_theta, contexts = inputs
        return model._transport_compact_with_contexts(
            current_theta,
            contexts,
            observation_count,
            num_sources,
            theta_size,
            max_num_sources,
            max_source_dim,
        )

    # lax.map intentionally serialises the temporal operator evaluations instead of creating a
    # T-way batched Transformer VJP. This is slower than vmap but dramatically lowers peak memory.
    return jax.lax.map(mapped_step, (previous_clouds, observation_contexts))


def _deq_forward_impl(
    model: SequentialBayesModel,
    prior_particles: Array,
    observations: Array,              # [T,Omax,Dmax+1]
    observation_count: Array,
    num_sources: Array,
    theta_size: Array,
    cfg: BayesTransportConfig,
) -> tuple[Array, Array, Array, Array]:
    """Solve the augmented T-observation equilibrium without constructing an autodiff tape."""
    if observations.ndim != 3:
        raise ValueError(
            "deq_fixed_point training observations must have shape [T,Omax,Dmax+1]."
        )
    if observations.shape[0] != cfg.training_fixed_point_max_iterations:
        raise ValueError(
            "DEQ training observation trajectory length does not match "
            "training_fixed_point_max_iterations."
        )

    initial_theta = model._compact_reference_cloud(prior_particles, num_sources, theta_size)
    contexts = jax.lax.map(
        lambda block: model._encode_observation_block(block, num_sources, theta_size),
        observations,
    )
    max_num_sources = prior_particles.shape[-2]
    max_source_dim = prior_particles.shape[-1]

    if cfg.fixed_point_solver == "triangular":
        # For F(Z)_t = T_phi(Z_{t-1},Y_t), with Z_{-1}=C_0, the augmented Jacobian is
        # strictly lower triangular. A single causal scan therefore solves Z=F(Z) exactly.
        # Because this function sits inside filter_custom_vjp's forward rule, JAX does NOT retain
        # a backward tape through this scan; the custom implicit rule below supplies the VJP.
        def scan_step(current_theta, current_contexts):
            next_theta = model._transport_compact_with_contexts(
                current_theta,
                current_contexts,
                observation_count,
                num_sources,
                theta_size,
                max_num_sources,
                max_source_dim,
            )
            return next_theta, next_theta

        _final, equilibrium_trajectory = jax.lax.scan(
            scan_step, initial_theta, contexts
        )
        steps = jnp.asarray(observations.shape[0], dtype=jnp.int32)
        residual = jnp.asarray(0.0, dtype=initial_theta.dtype)
        return equilibrium_trajectory, steps, residual, initial_theta

    initial_state = jnp.broadcast_to(
        initial_theta[None, ...],
        (observations.shape[0],) + initial_theta.shape,
    )

    def trajectory_operator(state):
        return _deq_trajectory_operator(
            model,
            state,
            initial_theta,
            contexts,
            observation_count,
            num_sources,
            theta_size,
            max_num_sources,
            max_source_dim,
        )

    equilibrium_trajectory, steps, residual = _solve_fixed_point(
        trajectory_operator,
        initial_state,
        solver=cfg.fixed_point_solver,
        max_steps=cfg.fixed_point_max_steps,
        tolerance=cfg.fixed_point_tolerance,
        relaxation=cfg.fixed_point_relaxation,
        anderson_history=cfg.fixed_point_anderson_history,
        anderson_ridge=cfg.fixed_point_anderson_ridge,
    )
    return equilibrium_trajectory, steps, residual, initial_theta


def _deq_triangular_implicit_model_vjp(
    model: SequentialBayesModel,
    equilibrium_trajectory: Array,
    initial_theta: Array,
    observations: Array,
    observation_count: Array,
    num_sources: Array,
    theta_size: Array,
    max_num_sources: int,
    max_source_dim: int,
    grad_final: Array,
):
    """Exact low-memory IFT VJP for the causal augmented DEQ.

    For the triangular equilibrium, (I-J_F^T)u=g is itself a reverse recurrence. We solve that
    recurrence one Bayes update at a time and accumulate parameter VJPs. The one-step function is
    rematerialized inside the reverse scan, so no T-step Transformer activation tape is retained.
    This is algebraically the implicit VJP, not differentiation through the forward scan.
    """
    previous_clouds = jnp.concatenate(
        [initial_theta[None, ...], equilibrium_trajectory[:-1]], axis=0
    )
    params, static_model = eqx.partition(model, eqx.is_inexact_array)
    zero_grad_params = jax.tree_util.tree_map(jnp.zeros_like, params)

    def reverse_step(carry, inputs):
        cotangent, grad_params_accum = carry
        previous_theta, observation_block = inputs

        def one_update(candidate_params, candidate_previous):
            candidate_model = eqx.combine(candidate_params, static_model)
            candidate_contexts = candidate_model._encode_observation_block(
                observation_block, num_sources, theta_size
            )
            return candidate_model._transport_compact_with_contexts(
                candidate_previous,
                candidate_contexts,
                observation_count,
                num_sources,
                theta_size,
                max_num_sources,
                max_source_dim,
            )

        # remat avoids saving the Transformer forward activations for the local VJP.
        rematerialized_update = jax.checkpoint(one_update)
        _output, pullback = jax.vjp(
            rematerialized_update, params, previous_theta
        )
        grad_params_step, grad_previous = pullback(cotangent)
        grad_params_accum = jax.tree_util.tree_map(
            lambda total, increment: total + increment,
            grad_params_accum,
            grad_params_step,
        )
        return (grad_previous, grad_params_accum), None

    (_grad_initial, grad_params), _ = jax.lax.scan(
        reverse_step,
        (grad_final, zero_grad_params),
        (previous_clouds[::-1], observations[::-1]),
    )
    return grad_params


@eqx.filter_custom_vjp
def deq_fixed_point_cloud(
    model: SequentialBayesModel,
    prior_particles: Array,
    observations: Array,
    observation_count: Array,
    num_sources: Array,
    theta_size: Array,
    *,
    cfg: BayesTransportConfig,
) -> tuple[Array, Array, Array]:
    trajectory, steps, residual, _initial_theta = _deq_forward_impl(
        model, prior_particles, observations, observation_count, num_sources, theta_size, cfg
    )
    # Only the last recurrent cloud enters the DEQ training objective. Keeping T clouds out of
    # the public output also prevents outer batch/prefix maps from materialising unnecessary data.
    return trajectory[-1], steps, residual


@deq_fixed_point_cloud.def_fwd
def _deq_fixed_point_cloud_fwd(
    perturbed,
    model: SequentialBayesModel,
    prior_particles: Array,
    observations: Array,
    observation_count: Array,
    num_sources: Array,
    theta_size: Array,
    *,
    cfg: BayesTransportConfig,
):
    del perturbed
    trajectory, steps, residual, initial_theta = _deq_forward_impl(
        model, prior_particles, observations, observation_count, num_sources, theta_size, cfg
    )
    # The residual contains only physical cloud states, not Transformer activations.
    return (trajectory[-1], steps, residual), (trajectory, initial_theta)


@deq_fixed_point_cloud.def_bwd
def _deq_fixed_point_cloud_bwd(
    residuals,
    grad_obj,
    perturbed,
    model: SequentialBayesModel,
    prior_particles: Array,
    observations: Array,
    observation_count: Array,
    num_sources: Array,
    theta_size: Array,
    *,
    cfg: BayesTransportConfig,
):
    equilibrium_trajectory, initial_theta = residuals
    grad_final = grad_obj[0]
    if grad_final is None:
        return jax.tree_util.tree_map(lambda _p: None, perturbed)

    max_num_sources = prior_particles.shape[-2]
    max_source_dim = prior_particles.shape[-1]
    backward_solver = (
        cfg.fixed_point_solver
        if cfg.fixed_point_backward_solver == "same"
        else cfg.fixed_point_backward_solver
    )

    if backward_solver == "triangular":
        grad_model = _deq_triangular_implicit_model_vjp(
            model,
            equilibrium_trajectory,
            initial_theta,
            observations,
            observation_count,
            num_sources,
            theta_size,
            max_num_sources,
            max_source_dim,
            grad_final,
        )
    else:
        # Generic IFT fallback retained for Picard/Anderson ablations. Unlike the memory-safe
        # triangular branch this operates on the full augmented state and can be substantially
        # more expensive for large B,T,N,H.
        contexts = jax.lax.map(
            lambda block: model._encode_observation_block(block, num_sources, theta_size),
            observations,
        )

        def state_operator(state):
            return _deq_trajectory_operator(
                model,
                state,
                initial_theta,
                contexts,
                observation_count,
                num_sources,
                theta_size,
                max_num_sources,
                max_source_dim,
            )

        _, state_pullback = eqx.filter_vjp(state_operator, equilibrium_trajectory)
        grad_trajectory = jnp.zeros_like(equilibrium_trajectory).at[-1].set(grad_final)
        regularization = jnp.asarray(
            cfg.fixed_point_backward_regularization, dtype=equilibrium_trajectory.dtype
        )

        def linear_fixed_point(u):
            jt_u = state_pullback(u)[0]
            return (grad_trajectory + jt_u) / (
                jnp.asarray(1.0, dtype=equilibrium_trajectory.dtype) + regularization
            )

        implicit_cotangent, _steps, _residual = _solve_fixed_point(
            linear_fixed_point,
            jnp.zeros_like(grad_trajectory),
            solver=backward_solver,
            max_steps=cfg.fixed_point_backward_max_steps,
            tolerance=cfg.fixed_point_backward_tolerance,
            relaxation=cfg.fixed_point_relaxation,
            anderson_history=cfg.fixed_point_anderson_history,
            anderson_ridge=cfg.fixed_point_anderson_ridge,
        )

        def model_operator(candidate_model):
            candidate_contexts = jax.lax.map(
                lambda block: candidate_model._encode_observation_block(
                    block, num_sources, theta_size
                ),
                observations,
            )
            return _deq_trajectory_operator(
                candidate_model,
                equilibrium_trajectory,
                initial_theta,
                candidate_contexts,
                observation_count,
                num_sources,
                theta_size,
                max_num_sources,
                max_source_dim,
            )

        _, model_pullback = eqx.filter_vjp(model_operator, model)
        grad_model = model_pullback(implicit_cotangent)[0]

    return jax.tree_util.tree_map(
        lambda grad, is_perturbed: grad if is_perturbed else None,
        grad_model,
        perturbed,
        is_leaf=lambda leaf: leaf is None,
    )


def _deq_batch_objective(
    model: SequentialBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig,
) -> tuple[Array, dict[str, Array], Array]:
    """Energy-score the final cloud after T distinct-observation implicit Bayes updates."""
    if batch["observations"].ndim != 4:
        raise ValueError(
            "deq_fixed_point minibatches must contain observations [B,T,Omax,Dmax+1]."
        )
    prefix_counts = jnp.arange(
        cfg.min_observations_per_step,
        cfg.max_observations_per_step + 1,
        dtype=jnp.int32,
    )

    def row_solve(inputs):
        prior, observations, sources, size = inputs

        def one_prefix(observation_count):
            return deq_fixed_point_cloud(
                model,
                prior,
                observations,
                observation_count,
                sources,
                size,
                cfg=cfg,
            )

        # Serialising prefixes and rows is intentionally conservative on memory. P is usually
        # tiny; users who prefer throughput can switch these maps back to vmap locally.
        return jax.lax.map(one_prefix, prefix_counts)

    predicted, fp_steps, fp_residuals = jax.lax.map(
        row_solve,
        (
            batch["prior_particles"],
            batch["observations"],
            batch["num_sources"],
            batch["theta_size"],
        ),
    )                                                           # [B,P,N,Kmax]
    target_theta = _compact_targets(
        batch["theta_true"], batch["num_sources"], batch["theta_size"], cfg
    )
    energy, attraction, repulsion, rmse, spread = jax.vmap(_prefix_metrics)(
        predicted, target_theta, batch["theta_size"]
    )
    loss = jnp.mean(energy)
    metrics = {
        "loss": loss,
        "energy_score": jnp.mean(energy),
        "final_energy_score": jnp.mean(energy[:, -1]),
        "posterior_mean_rmse": jnp.mean(rmse),
        "final_mean_rmse": jnp.mean(rmse[:, -1]),
        "posterior_spread": jnp.mean(spread),
        "final_spread": jnp.mean(spread[:, -1]),
        "attraction": jnp.mean(attraction),
        "repulsion": jnp.mean(repulsion),
        "energy_by_o": jnp.mean(energy, axis=0),
        "rmse_by_o": jnp.mean(rmse, axis=0),
        "spread_by_o": jnp.mean(spread, axis=0),
        "prefix_counts": prefix_counts,
        "fixed_point_steps": jnp.mean(fp_steps.astype(jnp.float32)),
        "fixed_point_residual": jnp.mean(fp_residuals),
        "training_fixed_point_iterations": jnp.asarray(
            cfg.training_fixed_point_max_iterations, dtype=jnp.float32
        ),
    }
    return loss, metrics, predicted[:, -1]


def _masked_pairwise_distance(
    x: Array,
    y: Array,
    theta_size: Array,
    eps: float,
) -> Array:
    valid = (jnp.arange(x.shape[-1]) < theta_size).astype(x.dtype)
    diff = (x[:, None, :] - y[None, :, :]) * valid[None, None, :]
    return jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1) + eps)


def _kernel_drifting_field_single(
    x: Array,
    target_theta: Array,
    theta_size: Array,
    temperature: float,
    eps: float,
) -> Array:
    """Deng et al. Algorithm-2 field with one stochastic positive posterior draw."""
    y_pos = target_theta[None, :]
    y_neg = x
    dist_pos = _masked_pairwise_distance(x, y_pos, theta_size, eps)
    dist_neg = _masked_pairwise_distance(x, y_neg, theta_size, eps)
    dist_neg = dist_neg + jnp.eye(x.shape[0], dtype=x.dtype) * jnp.asarray(1e6, x.dtype)
    logit_pos = -dist_pos / jnp.asarray(temperature, dtype=x.dtype)
    logit_neg = -dist_neg / jnp.asarray(temperature, dtype=x.dtype)
    logits = jnp.concatenate([logit_pos, logit_neg], axis=1)
    a_row = jax.nn.softmax(logits, axis=1)
    a_col = jax.nn.softmax(logits, axis=0)
    a = jnp.sqrt(jnp.maximum(a_row * a_col, jnp.asarray(0.0, dtype=x.dtype)))
    a_pos = a[:, :1]
    a_neg = a[:, 1:]
    w_pos = a_pos * jnp.sum(a_neg, axis=1, keepdims=True)
    w_neg = a_neg * jnp.sum(a_pos, axis=1, keepdims=True)
    drift = w_pos @ y_pos - w_neg @ y_neg
    valid = (jnp.arange(x.shape[-1]) < theta_size).astype(x.dtype)
    return drift * valid[None, :]


def _energy_score_drifting_field_single(
    x: Array,
    target_theta: Array,
    theta_size: Array,
    eps: float,
) -> Array:
    """Negative energy-score gradient, rescaled by N to keep an O(1) particle drift."""
    valid = (jnp.arange(x.shape[-1]) < theta_size).astype(x.dtype)
    to_target = (target_theta[None, :] - x) * valid[None, :]
    target_norm = jnp.sqrt(jnp.sum(jnp.square(to_target), axis=-1, keepdims=True) + eps)
    attraction = to_target / target_norm
    pair_diff = (x[:, None, :] - x[None, :, :]) * valid[None, None, :]
    pair_norm = jnp.sqrt(jnp.sum(jnp.square(pair_diff), axis=-1, keepdims=True) + eps)
    repulsion = jnp.mean(pair_diff / pair_norm, axis=1)
    return (attraction + repulsion) * valid[None, :]


def _canonical_drifting_field_name(name: str) -> str:
    """Map historical drifting-field names onto the explicit current names."""
    name = str(name)
    if name == "energy":
        return "energy_score_gradient"
    if name == "kernel_energy":
        return "kernel_energy_score_gradient"
    return name


def _drifting_loss_single(
    x: Array,
    target_theta: Array,
    theta_size: Array,
    cfg: BayesTransportConfig,
) -> Array:
    """Return the configured one-step drifting loss for one empirical cloud.

    ``energy_score`` is intentionally the scalar proper-score objective itself. The remaining
    choices are genuine vector fields and use the Deng-style frozen regression target
    ``stopgrad(x + eta * V)``. Keeping the scalar option explicit avoids pretending that a scalar
    energy score is itself a vector field, while still allowing the requested side-by-side ablation.
    """
    field_name = _canonical_drifting_field_name(cfg.drifting_field)
    if field_name == "energy_score":
        return energy_score_single(x, target_theta, theta_size)

    valid = (jnp.arange(x.shape[-1]) < theta_size).astype(x.dtype)
    denom = jnp.maximum(
        jnp.asarray(x.shape[0], dtype=x.dtype) * theta_size.astype(x.dtype),
        jnp.asarray(1.0, dtype=x.dtype),
    )
    energy_gradient_field = _energy_score_drifting_field_single(
        x, target_theta, theta_size, cfg.drifting_distance_epsilon
    )

    if field_name == "energy_score_gradient":
        target = jax.lax.stop_gradient(x + cfg.drifting_eta * energy_gradient_field)
        return jnp.sum(jnp.square((x - target) * valid[None, :])) / denom

    total = jnp.asarray(0.0, dtype=x.dtype)
    for temperature in cfg.drifting_temperatures:
        drift = _kernel_drifting_field_single(
            x,
            target_theta,
            theta_size,
            temperature,
            cfg.drifting_distance_epsilon,
        )
        if field_name == "kernel_energy_score_gradient":
            drift = drift + cfg.drifting_energy_weight * energy_gradient_field
        target = jax.lax.stop_gradient(x + cfg.drifting_eta * drift)
        total = total + jnp.sum(jnp.square((x - target) * valid[None, :])) / denom
    return total

def _drifting_batch_objective(
    model: SequentialBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig,
) -> tuple[Array, dict[str, Array], Array]:
    """One-step drifting-model regression; optimizer time is the only drifting iteration.

    This follows the logic of Deng et al. Algorithm 1: x is produced once by the current network,
    x itself supplies the negative cloud, x+V(x) is frozen, and the current prediction is regressed
    toward that target. There is deliberately no recurrent t-loop and no fixed-point solver here.
    Exposure to the model's own previous errors is handled across optimizer steps by the host-side
    historical-output prior buffer.
    """
    if batch["observations"].ndim != 3:
        raise ValueError(
            "drifting minibatches must contain observations [B,Omax,Dmax+1]."
        )
    predicted, direct_metrics = _direct_prediction_and_metrics(model, batch, cfg)
    target_theta = _compact_targets(
        batch["theta_true"], batch["num_sources"], batch["theta_size"], cfg
    )

    def row_loss(prefix_clouds, target, size):
        return jnp.mean(
            jax.vmap(lambda cloud: _drifting_loss_single(cloud, target, size, cfg))(
                prefix_clouds
            )
        )

    row_losses = jax.vmap(row_loss)(predicted, target_theta, batch["theta_size"])
    loss = jnp.mean(row_losses)
    metrics = dict(direct_metrics)
    metrics["loss"] = loss
    metrics["drifting_loss"] = loss
    # Final O-prefix outputs are detached on the host after train_step and may become future priors.
    return loss, metrics, predicted[:, -1]


def batch_objective(
    model: SequentialBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig = CFG,
) -> tuple[Array, tuple[dict[str, Array], Array]]:
    """Training objective dispatch plus final-prefix clouds for historical-output replay."""
    if cfg.training_mode == "energy_score":
        predicted, metrics = _direct_prediction_and_metrics(model, batch, cfg)
        return metrics["loss"], (metrics, predicted[:, -1])
    if cfg.training_mode == "deq_fixed_point":
        loss, metrics, replay_clouds = _deq_batch_objective(model, batch, cfg)
        return loss, (metrics, replay_clouds)
    if cfg.training_mode == "drifting":
        loss, metrics, replay_clouds = _drifting_batch_objective(model, batch, cfg)
        return loss, (metrics, replay_clouds)
    raise ValueError(f"Unsupported training_mode={cfg.training_mode!r}")


@eqx.filter_jit
def predict_prefix_batch(
    model: SequentialBayesModel,
    prior_particles: Array,
    observations: Array,
    num_sources: Array,
    theta_size: Array,
) -> tuple[Array, Array, Array, Array]:
    """JIT-compiled iid batch prediction for every observation prefix."""
    return jax.vmap(
        lambda prior, obs, sources, size: model.predict_prefixes(
            prior, obs, sources, size
        )
    )(prior_particles, observations, num_sources, theta_size)


@eqx.filter_jit
def predict_batch(
    model: SequentialBayesModel,
    prior_particles: Array,
    observations: Array,
    observation_count: Array,
    num_sources: Array,
    theta_size: Array,
) -> tuple[Array, Array, Array]:
    """JIT-compiled evaluation-only sequential trajectory batching."""
    return jax.vmap(model, in_axes=(0, 0, None, 0, 0))(
        prior_particles, observations, observation_count, num_sources, theta_size
    )


@eqx.filter_jit
def amortized_evaluation_batch(
    model: SequentialBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Array]:
    # Evaluation is fixed across training modes: always score the original direct one-step map.
    _, metrics = _direct_prediction_and_metrics(model, batch, cfg)
    return metrics


@eqx.filter_jit
def sequential_evaluation_batch(
    model: SequentialBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Array]:
    """Evaluate recurrence only; this objective is never differentiated during training."""
    predicted, _, _ = predict_batch(
        model,
        batch["prior_particles"],
        batch["observations"],
        batch["observation_count"],
        batch["num_sources"],
        batch["theta_size"],
    )
    target_theta = _compact_targets(
        batch["theta_true"], batch["num_sources"], batch["theta_size"], cfg
    )
    energy, rmse, spread = jax.vmap(_trajectory_metrics)(
        predicted, target_theta, batch["theta_size"]
    )
    return {
        "loss": jnp.mean(energy),
        "energy_score": jnp.mean(energy),
        "final_energy_score": jnp.mean(energy[:, -1]),
        "posterior_mean_rmse": jnp.mean(rmse),
        "final_mean_rmse": jnp.mean(rmse[:, -1]),
        "posterior_spread": jnp.mean(spread),
        "final_spread": jnp.mean(spread[:, -1]),
        "energy_by_t": jnp.mean(energy, axis=0),
        "rmse_by_t": jnp.mean(rmse, axis=0),
        "spread_by_t": jnp.mean(spread, axis=0),
    }


#%% 10) Amortized validation and sequential evaluation with reproducible prior clouds
def evaluate_amortized_model(
    model: SequentialBayesModel,
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    *,
    num_particles: int | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluate the unchanged direct one-step empirical-energy protocol used for model selection."""
    n_total = len(dataset["theta_true"])
    if max_samples is not None:
        n_total = min(n_total, int(max_samples))
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    eval_seed = cfg.seed + 89_000 if seed is None else int(seed)
    rng = np.random.default_rng(eval_seed)

    scalar_names = [
        "loss", "energy_score", "final_energy_score",
        "posterior_mean_rmse", "final_mean_rmse", "posterior_spread", "final_spread",
        "attraction", "repulsion",
    ]
    by_o_names = ["energy_by_o", "rmse_by_o", "spread_by_o"]
    scalar_values = {name: [] for name in scalar_names}
    by_o_values = {name: [] for name in by_o_names}
    weights = []
    prefix_counts = None

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        indices = np.arange(start, stop)
        batch_np = make_iid_batch_np(
            dataset, indices, rng, cfg, num_particles=num_particles
        )
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        metrics = jax.device_get(amortized_evaluation_batch(model, batch, cfg))
        weight = stop - start
        weights.append(weight)
        for name in scalar_names:
            scalar_values[name].append(float(metrics[name]))
        for name in by_o_names:
            by_o_values[name].append(np.asarray(metrics[name], dtype=np.float64))
        prefix_counts = np.asarray(metrics["prefix_counts"], dtype=np.int32)

    weight_array = np.asarray(weights, dtype=np.float64)
    result: dict[str, np.ndarray | float] = {}
    for name in scalar_names:
        result[name] = float(np.average(scalar_values[name], weights=weight_array))
    for name in by_o_names:
        result[name] = np.average(np.stack(by_o_values[name]), axis=0, weights=weight_array)
    result["prefix_counts"] = prefix_counts
    return result


def evaluate_model(
    model: SequentialBayesModel,
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    *,
    num_particles: int | None = None,
    max_trajectories: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluation-only repeated-Bayes rollout with fresh reproducible starting clouds."""
    n_total = len(dataset["theta_true"])
    if max_trajectories is not None:
        n_total = min(n_total, int(max_trajectories))
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    eval_seed = cfg.seed + 90_000 if seed is None else int(seed)
    rng = np.random.default_rng(eval_seed)

    scalar_names = [
        "loss", "energy_score", "final_energy_score", "posterior_mean_rmse",
        "final_mean_rmse", "posterior_spread", "final_spread",
    ]
    scalar_values = {name: [] for name in scalar_names}
    by_t_values = {name: [] for name in ["energy_by_t", "rmse_by_t", "spread_by_t"]}
    weights = []

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        indices = np.arange(start, stop)
        batch_np = make_batch_np(
            dataset,
            indices,
            rng,
            cfg,
            num_particles=num_particles,
            observations_per_step=cfg.test_observations_per_step,
        )
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        metrics = jax.device_get(sequential_evaluation_batch(model, batch, cfg))
        weight = stop - start
        weights.append(weight)
        for name in scalar_names:
            scalar_values[name].append(float(metrics[name]))
        for name in by_t_values:
            by_t_values[name].append(np.asarray(metrics[name], dtype=np.float64))

    weight_array = np.asarray(weights, dtype=np.float64)
    result: dict[str, np.ndarray | float] = {}
    for name in scalar_names:
        result[name] = float(np.average(scalar_values[name], weights=weight_array))
    for name in by_t_values:
        result[name] = np.average(np.stack(by_t_values[name]), axis=0, weights=weight_array)
    return result


#%% 11) Optional exact-likelihood reference posterior for plots only
def reference_posterior_particles_np(
    rng: np.random.Generator,
    observations: np.ndarray,
    prefix_length: int,
    num_sources: int,
    theta_size: int,
    cfg: BayesTransportConfig = CFG,
) -> tuple[np.ndarray, float]:
    """SNIS reference posterior used only after training for visual validation.

    Proposal is exactly the ONE configured base prior rho_0, so importance weights are
    proportional to the likelihood of the observed prefix in both uniform and Gaussian
    modes.  This function is intentionally NOT a teacher and is never called inside the
    training objective.
    """
    prefix_length = int(prefix_length)
    S = int(num_sources)
    D = int(theta_size) // S
    proposals = sample_base_prior_np(
        rng, cfg.reference_proposals, cfg, num_sources=S, source_dim=D
    )
    prefix = np.asarray(observations[:prefix_length])
    designs = prefix[:, :D]
    readings = prefix[:, -1]
    predicted_means = source_log_mean_np(proposals, designs, cfg)  # [P,t]
    residual = (readings[None, :] - predicted_means) / cfg.observation_noise_std
    log_weights = -0.5 * np.sum(residual**2, axis=1)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.maximum(weights.sum(), 1e-300)
    ess = float(1.0 / np.sum(weights**2))
    indices = rng.choice(
        len(proposals), size=cfg.reference_particles, replace=True, p=weights
    )
    posterior = proposals[indices]
    if cfg.canonicalize_particle_sources and S > 1:
        posterior = canonicalize_sources_np(posterior)
    return posterior.astype(np.float32), ess


#%% 12) Visualisation: architecture schematic
def plot_architecture_schematic(
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
):
    """Visual map separating the configurable training objective from unchanged evaluation."""
    fig, ax = plt.subplots(1, 1, figsize=(16.2, 7.0), constrained_layout=True)

    def draw_box(xy, width, height, text, title=None):
        patch = FancyBboxPatch(
            xy, width, height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.3, facecolor="white", edgecolor="black",
        )
        ax.add_patch(patch)
        label = text if title is None else f"{title}\n{text}"
        ax.text(xy[0] + width / 2, xy[1] + height / 2, label,
                ha="center", va="center", fontsize=8.7)

    def arrow(start, end, text="", connectionstyle=None):
        kwargs = {}
        if connectionstyle is not None:
            kwargs["connectionstyle"] = connectionstyle
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.2, **kwargs))
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025,
                    text, ha="center", va="bottom", fontsize=7.7)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.01, 0.96, f"TRAINING: {cfg.training_mode}",
            fontsize=12, fontweight="bold", va="top")
    ax.text(0.01, 0.43, "EVALUATION ONLY: repeatedly apply the same learned Bayes transport",
            fontsize=12, fontweight="bold", va="top")

    train_obs_text = (
        f"one theta* ~ rho_0\nT={cfg.training_fixed_point_max_iterations} fresh blocks from theta*"
        if cfg.training_mode == "deq_fixed_point"
        else "actual theta* ~ rho_0\nOmax observations from theta*"
    )
    draw_box((0.02, 0.65), 0.14, 0.20, train_obs_text, "training simulator row")
    fixed_shape = (
        cfg.min_num_sources == cfg.max_num_sources
        and cfg.min_source_dim == cfg.max_source_dim
    )
    single_observation_direct = (
        cfg.min_observations_per_step == 1
        and cfg.max_observations_per_step == 1
    )
    if fixed_shape:
        fixed_input = (
            f"linear projection -> E={cfg.embedding_dim}"
            if cfg.fixed_shape_learned_projection
            else f"parameter-free width D+1={cfg.max_source_dim + 1}"
        )
        if single_observation_direct:
            context_label = (
                f"{fixed_input}\nLikelihood Transformer bypass: 0 params\n1 token -> Posterior"
            )
        else:
            context_label = f"{fixed_input}\ncausal Transformer\nOmax tokens"
    else:
        if single_observation_direct:
            context_label = (
                f"dimension embedder\nLikelihood Transformer bypass: 0 params\n"
                f"1 x E={cfg.embedding_dim} -> Posterior"
            )
        else:
            context_label = f"dimension embedder\ncausal Transformer\nOmax x E={cfg.embedding_dim}"
    draw_box((0.20, 0.64), 0.17, 0.22, context_label, "Y -> contexts")
    draw_box(
        (0.02, 0.47), 0.14, 0.14,
        f"theta* ~ rho_0\nanchor=theta* w.p. {cfg.synthetic_prior_match_probability:.2f}\n"
        "shared tau ~ U[0,1]\nC_tau=(1-tau)z+tau anchor",
        "synthetic input cloud",
    )
    draw_box((0.42, 0.58), 0.22, 0.27,
             f"particle self-attention\n{cfg.posterior_conditioning} conditioning\nphysical displacement head",
             "Posterior Transformer")
    if cfg.training_mode == "deq_fixed_point":
        recurrent_training_text = (
            f"T={cfg.training_fixed_point_max_iterations}: C_t + fresh Y_t -> C_(t+1)\n"
        )
    elif cfg.training_mode == "drifting":
        recurrent_training_text = "one T_phi call; optimizer step supplies drifting time\n"
    else:
        recurrent_training_text = "one T_phi call per prefix\n"
    draw_box((0.71, 0.60), 0.25, 0.23,
             f"vmap over o={cfg.min_observations_per_step},...,{cfg.max_observations_per_step}\n"
             f"{recurrent_training_text}training objective: {cfg.training_mode}",
             "training clouds")

    arrow((0.16, 0.75), (0.20, 0.75))
    arrow((0.37, 0.75), (0.42, 0.73), "all causal prefix signals")
    replay_label = (
        f"C_tau or replay prior (p={cfg.historical_output_prior_probability:.2f})"
        if cfg.training_mode in {"energy_score", "drifting"}
        else "C_tau starts each DEQ rollout"
    )
    arrow((0.16, 0.54), (0.42, 0.65), replay_label)
    arrow((0.64, 0.71), (0.71, 0.71), "shared Bayes-update cell")

    draw_box((0.02, 0.11), 0.16, 0.19,
             "one theta* fixed\nnew observation block each t", "trajectory")
    draw_box((0.25, 0.10), 0.16, 0.20,
             "current particle cloud\n+ fresh observation prefix", "Bayes update input")
    draw_box((0.49, 0.10), 0.20, 0.20,
             f"same trained Posterior Transformer\n({cfg.posterior_conditioning})", "reused map")
    draw_box((0.77, 0.10), 0.19, 0.20,
             "theta_t cloud\nES / RMSE / spread\nreference comparison", "sequential diagnostics")
    arrow((0.18, 0.205), (0.25, 0.205))
    arrow((0.41, 0.205), (0.49, 0.205))
    arrow((0.69, 0.205), (0.77, 0.205))
    arrow((0.86, 0.10), (0.33, 0.10), "physical cloud reused at t+1", connectionstyle="arc3,rad=-0.27")

    fig.suptitle(
        "Interpolated-prior empirical-energy transport and repeated-Bayes evaluation",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=180)
    display(fig)
    plt.close(fig)

#%% 13) Visualisation: physical source field and one simulated trajectory
def _trajectory_shape(trajectory: dict[str, np.ndarray]) -> tuple[int, int, int]:
    S = int(np.asarray(trajectory["num_sources"]).item())
    theta_size = int(np.asarray(trajectory["theta_size"]).item())
    if theta_size % S != 0:
        raise ValueError("theta_size must be divisible by num_sources.")
    return S, theta_size // S, theta_size


def plot_source_trajectory(
    trajectory: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
    title: str = "Mode-A source-localisation trajectory",
):
    S, D, _ = _trajectory_shape(trajectory)
    if D != 2:
        raise ValueError("The physical field plot is intentionally a 2-D visual diagnostic.")
    theta_true = np.asarray(trajectory["theta_true"])[:S, :D]
    observation_blocks = _ensure_observation_blocks_np(trajectory["observations"])
    observation_count = _trajectory_observation_count_np(trajectory, cfg)
    observations = _flatten_used_observation_prefix_np(
        observation_blocks, observation_count
    )
    designs = observations[:, :D]
    readings = observations[:, -1]

    grid = np.linspace(cfg.design_low, cfg.design_high, cfg.grid_size)
    gx, gy = np.meshgrid(grid, grid)
    design_grid = np.stack([gx, gy], axis=-1)
    field = source_log_mean_np(theta_true, design_grid, cfg)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    contour = axes[0].contourf(gx, gy, field, levels=32)
    fig.colorbar(contour, ax=axes[0], label="expected log reading")
    axes[0].scatter(designs[:, 0], designs[:, 1], c=readings, s=55, marker="s",
                    edgecolors="white", linewidths=0.7, label="observed sensor pairs")
    axes[0].scatter(theta_true[:, 0], theta_true[:, 1], marker="*", s=220,
                    edgecolors="black", linewidths=0.8, label="theta*")
    axes[0].set_title("Physical sensor field and observed designs")
    axes[0].set_xlim(cfg.design_low, cfg.design_high)
    axes[0].set_ylim(cfg.design_low, cfg.design_high)
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)

    axes[1].plot(np.arange(1, len(readings) + 1), readings, marker="o", markersize=4,
                 label="observed y_t")
    expected_at_designs = source_log_mean_np(theta_true, designs, cfg)
    axes[1].plot(np.arange(1, len(readings) + 1), expected_at_designs,
                 linestyle="--", label="E[y_t | theta*, x_t]")
    axes[1].set_xlabel("observation index t")
    axes[1].set_ylabel("log reading")
    axes[1].set_title("Precomputed likelihood trajectory")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 14) Physical prior -> posterior evolution across prefixes
def select_prefixes(trajectory_length: int, n_panels_after_prior: int = 5) -> list[int]:
    values = np.unique(
        np.rint(np.geomspace(1, trajectory_length, n_panels_after_prior)).astype(int)
    )
    if values[-1] != trajectory_length:
        values = np.append(values, trajectory_length)
    while len(values) > n_panels_after_prior:
        values = np.delete(values, 1)
    return values.tolist()


def plot_posterior_evolution(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
    title: str = "Posterior evolution",
):
    """Plot physical posterior particles returned directly by the Posterior Transformer."""
    S, D, theta_size = _trajectory_shape(trajectory)
    if D != 2:
        raise ValueError("The physical posterior plot is intentionally a 2-D visual diagnostic.")
    observations = _ensure_observation_blocks_np(trajectory["observations"])
    observation_count = _trajectory_observation_count_np(trajectory, cfg)
    predicted, _, prior_theta = model(
        jnp.asarray(prior_particles),
        jnp.asarray(observations),
        jnp.asarray(observation_count),
        jnp.asarray(S),
        jnp.asarray(theta_size),
    )
    predicted = np.asarray(jax.device_get(predicted))
    prior_theta = np.asarray(jax.device_get(prior_theta))

    prior_cloud = prior_theta[:, :theta_size].reshape(len(prior_theta), S, D)
    posterior_clouds = predicted[:, :, :theta_size].reshape(
        predicted.shape[0], predicted.shape[1], S, D
    )
    theta_true = np.asarray(trajectory["theta_true"])[:S, :D]
    if cfg.canonicalize_particle_sources and S > 1:
        theta_true = canonicalize_sources_np(theta_true)

    prefixes = select_prefixes(len(observations), 5)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.5), constrained_layout=True)
    axes = axes.ravel()
    clouds = [prior_cloud] + [posterior_clouds[t - 1] for t in prefixes]
    labels = [r"base prior $p(\theta)$"] + [
        rf"$q_\phi(\theta\mid y_{{1:{t}}})$" for t in prefixes
    ]
    all_points = np.concatenate([c.reshape(-1, 2) for c in clouds] + [theta_true.reshape(-1, 2)])
    lim = max(_base_prior_plot_extent(cfg), 1.12 * float(np.quantile(np.abs(all_points), 0.995)))

    particle_color = "#4C78A8"
    design_color = "#8FD19E"  # light green: observation locations already used
    truth_color = "#111111"

    for panel_index, (ax, cloud, label) in enumerate(zip(axes, clouds, labels)):
        ax.scatter(
            cloud[..., 0].reshape(-1),
            cloud[..., 1].reshape(-1),
            s=14,
            alpha=0.30,
            color=particle_color,
            edgecolors="none",
            label="posterior source locations" if panel_index else "prior source locations",
            zorder=2,
        )
        ax.scatter(
            theta_true[:, 0],
            theta_true[:, 1],
            marker="*",
            s=195,
            color=truth_color,
            edgecolors="white",
            linewidths=0.7,
            label=r"$\theta^\star$",
            zorder=7,
        )
        if panel_index > 0:
            t = prefixes[panel_index - 1]
            designs = _flatten_used_observation_prefix_np(
                observations, observation_count, t
            )[:, :D]
            ax.scatter(
                designs[:, 0],
                designs[:, 1],
                marker="x",
                s=38,
                alpha=0.78,
                color=design_color,
                linewidths=1.5,
                label="observation locations seen",
                zorder=5,
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_title(label)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)

#%% 15) Visualisation: learned posterior versus optional likelihood-based reference
def plot_reference_comparison(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
):
    """Compare the direct physical final posterior cloud with an exact-likelihood SNIS reference."""
    S, D, theta_size = _trajectory_shape(trajectory)
    if D != 2:
        raise ValueError("Reference source-marginal plot is a 2-D visual diagnostic.")
    observations = _ensure_observation_blocks_np(trajectory["observations"])
    observation_count = _trajectory_observation_count_np(trajectory, cfg)
    used_observations = _flatten_used_observation_prefix_np(
        observations, observation_count
    )
    rng = np.random.default_rng(cfg.seed + 44_000)
    reference, ess = reference_posterior_particles_np(
        rng,
        used_observations,
        len(used_observations),
        S,
        theta_size,
        cfg,
    )
    posterior_theta, _, _ = model(
        jnp.asarray(prior_particles),
        jnp.asarray(observations),
        jnp.asarray(observation_count),
        jnp.asarray(S),
        jnp.asarray(theta_size),
    )
    learned_flat = np.asarray(jax.device_get(posterior_theta[-1]))
    learned = learned_flat[:, :theta_size].reshape(len(learned_flat), S, D)

    theta_true = np.asarray(trajectory["theta_true"])[:S, :D]
    canonical_truth = (
        canonicalize_sources_np(theta_true)
        if cfg.canonicalize_particle_sources and S > 1 else theta_true
    )
    column_names = [f"repeated Bayes\n{cfg.posterior_conditioning}", f"reference SNIS\nESS={ess:.0f}"]
    column_clouds = [learned, reference]
    lim_points = np.concatenate([cloud.reshape(-1, D) for cloud in column_clouds])
    lim = max(_base_prior_plot_extent(cfg), 1.1 * float(np.quantile(np.abs(lim_points), 0.995)))

    fig, axes = plt.subplots(
        S,
        len(column_names),
        figsize=(4.3 * len(column_names), 4.0 * S),
        squeeze=False,
        constrained_layout=True,
    )
    for source_index in range(S):
        for col, (name, cloud) in enumerate(zip(column_names, column_clouds)):
            ax = axes[source_index, col]
            ax.scatter(
                cloud[:, source_index, 0],
                cloud[:, source_index, 1],
                s=12,
                alpha=0.25,
            )
            ax.scatter(
                canonical_truth[source_index, 0],
                canonical_truth[source_index, 1],
                marker="*",
                s=190,
                edgecolors="black",
                linewidths=0.8,
            )
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            if source_index == 0:
                ax.set_title(name, fontweight="bold")
            ax.set_ylabel(f"canonical source {source_index + 1}")

    fig.suptitle(
        "Repeated-Bayes posterior source marginals versus likelihood-based reference",
        fontsize=14,
        fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)



def _smooth_hist_density_np(values: np.ndarray, grid: np.ndarray, bins: int = 90) -> np.ndarray:
    """Dependency-free smooth density used for paper-style one-dimensional marginal plots."""
    values = np.asarray(values, dtype=np.float64).ravel()
    hist, edges = np.histogram(values, bins=bins, range=(grid[0], grid[-1]), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sigma_bins = 1.5
    radius = 5
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return np.interp(grid, centers, smooth, left=0.0, right=0.0)


def plot_paper_style_marginals(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
    max_coordinates: int = 6,
):
    """Paper-style prior/reference/pushforward marginals for ONE amortized observation block.

    The paper compares rho, a reference posterior, and T_theta(.;y)#rho for a fixed observation.
    Here the first fixed-trajectory block plays that role.  This plot deliberately uses the
    NON-SEQUENTIAL training path at `test_observations_per_step`; the separate sequential
    reference study below evaluates repeated Bayes application across trajectory steps.
    """
    S, D, theta_size = _trajectory_shape(trajectory)
    observation_blocks = _ensure_observation_blocks_np(trajectory["observations"])
    observation_count = int(_trajectory_observation_count_np(trajectory, cfg).item())
    observations = observation_blocks[0]
    used_observations = np.asarray(observations[:observation_count], dtype=np.float32)
    rng = np.random.default_rng(cfg.seed + 45_000)
    reference, ess = reference_posterior_particles_np(
        rng,
        used_observations,
        len(used_observations),
        S,
        theta_size,
        cfg,
    )
    posterior_by_prefix, _, prior_theta, prefix_counts = model.predict_prefixes(
        jnp.asarray(prior_particles),
        jnp.asarray(observations),
        jnp.asarray(S),
        jnp.asarray(theta_size),
    )
    prefix_counts_np = np.asarray(jax.device_get(prefix_counts), dtype=np.int32)
    hits = np.flatnonzero(prefix_counts_np == observation_count)
    if len(hits) != 1:
        raise ValueError("test_observations_per_step must identify exactly one trained prefix.")
    learned = np.asarray(jax.device_get(posterior_by_prefix[int(hits[0])]))[:, :theta_size]
    prior_flat = np.asarray(jax.device_get(prior_theta))[:, :theta_size]
    reference_flat = reference.reshape(len(reference), -1)[:, :theta_size]
    truth = np.asarray(trajectory["theta_true"], dtype=np.float32)[:S, :D]
    if cfg.canonicalize_particle_sources and S > 1:
        truth = canonicalize_sources_np(truth)
    truth_flat = truth.reshape(-1)

    n_coordinates = min(theta_size, int(max_coordinates))
    ncols = min(3, n_coordinates)
    nrows = int(math.ceil(n_coordinates / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), squeeze=False, constrained_layout=True
    )
    axes = axes.ravel()
    for k in range(n_coordinates):
        ax = axes[k]
        values = np.concatenate([prior_flat[:, k], reference_flat[:, k], learned[:, k]])
        lo, hi = np.quantile(values, [0.002, 0.998])
        pad = max(0.15 * (hi - lo), 0.25)
        grid = np.linspace(lo - pad, hi + pad, 300)
        ax.plot(grid, _smooth_hist_density_np(prior_flat[:, k], grid), color="black", label="prior")
        ax.plot(grid, _smooth_hist_density_np(reference_flat[:, k], grid), color="tab:blue", label="SNIS reference")
        ax.plot(grid, _smooth_hist_density_np(learned[:, k], grid), color="tab:red", label="learned pushforward")
        ax.axvline(truth_flat[k], color="black", linestyle=":", linewidth=1.0, label="theta*" if k == 0 else None)
        source_index, coordinate_index = divmod(k, D)
        ax.set_title(f"source {source_index + 1}, coordinate {coordinate_index + 1}")
        ax.set_xlabel("physical theta")
        ax.set_ylabel("density")
        ax.grid(alpha=0.18)
        if k == 0:
            ax.legend(fontsize=7)
    for ax in axes[n_coordinates:]:
        ax.axis("off")
    fig.suptitle(
        f"Direct amortized posterior, o={observation_count} — learned vs reference (SNIS ESS={ess:.0f})",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=180)
    display(fig)
    plt.close(fig)


def _energy_distance_samples_np(x: np.ndarray, y: np.ndarray) -> float:
    """Finite-sample squared energy distance with off-diagonal within-sample terms."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cross = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean()
    if len(x) > 1:
        dx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        within_x = dx.sum() / (len(x) * (len(x) - 1))
    else:
        within_x = 0.0
    if len(y) > 1:
        dy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
        within_y = dy.sum() / (len(y) * (len(y) - 1))
    else:
        within_y = 0.0
    return float(max(2.0 * cross - within_x - within_y, 0.0))


def _energy_score_samples_np(samples: np.ndarray, truth: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    attraction = np.linalg.norm(samples - truth[None, :], axis=-1).mean()
    if len(samples) > 1:
        pair = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1)
        repulsion = pair.sum() / (len(samples) * (len(samples) - 1))
    else:
        repulsion = 0.0
    return float(attraction - 0.5 * repulsion)


def sequential_reference_study(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Compare repeated learned Bayes updates with a reference posterior at every step."""
    S, D, theta_size = _trajectory_shape(trajectory)
    observations = _ensure_observation_blocks_np(trajectory["observations"])
    observation_count = _trajectory_observation_count_np(trajectory, cfg)
    posterior_sequence, _, _ = model(
        jnp.asarray(prior_particles),
        jnp.asarray(observations),
        jnp.asarray(observation_count),
        jnp.asarray(S),
        jnp.asarray(theta_size),
    )
    learned_sequence = np.asarray(jax.device_get(posterior_sequence))[:, :, :theta_size]
    truth = np.asarray(trajectory["theta_true"], dtype=np.float32)[:S, :D]
    if cfg.canonicalize_particle_sources and S > 1:
        truth = canonicalize_sources_np(truth)
    truth_flat = truth.reshape(-1)
    rng = np.random.default_rng(cfg.seed + 46_000 if seed is None else int(seed))

    energy_distance = []
    learned_es = []
    reference_es = []
    learned_rmse = []
    reference_rmse = []
    ess_values = []
    for t in range(1, len(observations) + 1):
        used = _flatten_used_observation_prefix_np(observations, observation_count, t)
        reference, ess = reference_posterior_particles_np(
            rng, used, len(used), S, theta_size, cfg
        )
        reference_flat = reference.reshape(len(reference), -1)[:, :theta_size]
        # Cap the quadratic reference-reference energy calculation without changing the SNIS
        # posterior itself used for its mean/score diagnostics.
        if len(reference_flat) > 512:
            subset = rng.choice(len(reference_flat), size=512, replace=False)
            reference_for_distance = reference_flat[subset]
        else:
            reference_for_distance = reference_flat
        learned = learned_sequence[t - 1]
        energy_distance.append(_energy_distance_samples_np(learned, reference_for_distance))
        learned_es.append(_energy_score_samples_np(learned, truth_flat))
        reference_es.append(_energy_score_samples_np(reference_flat, truth_flat))
        learned_rmse.append(float(np.sqrt(np.mean((learned.mean(axis=0) - truth_flat) ** 2))))
        reference_rmse.append(float(np.sqrt(np.mean((reference_flat.mean(axis=0) - truth_flat) ** 2))))
        ess_values.append(ess)

    return {
        "energy_distance_to_reference": np.asarray(energy_distance, dtype=np.float64),
        "learned_energy_score": np.asarray(learned_es, dtype=np.float64),
        "reference_energy_score": np.asarray(reference_es, dtype=np.float64),
        "learned_mean_rmse": np.asarray(learned_rmse, dtype=np.float64),
        "reference_mean_rmse": np.asarray(reference_rmse, dtype=np.float64),
        "reference_ess": np.asarray(ess_values, dtype=np.float64),
    }


def plot_sequential_reference_study(
    study: dict[str, np.ndarray],
    destination: Path | None = None,
):
    """Strong repeated-Bayes diagnostic: learned/reference discrepancy as evidence accumulates."""
    t = np.arange(1, len(study["energy_distance_to_reference"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    axes[0, 0].plot(t, np.maximum(study["energy_distance_to_reference"], 1e-12), marker="o")
    axes[0, 0].set_title("Energy distance: learned vs reference")
    axes[0, 0].set_yscale("log")

    axes[0, 1].plot(t, np.maximum(study["learned_energy_score"], 1e-12), marker="o", label="learned")
    axes[0, 1].plot(t, np.maximum(study["reference_energy_score"], 1e-12), marker="o", label="reference")
    axes[0, 1].set_title("Energy score against fixed theta*")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(t, np.maximum(study["learned_mean_rmse"], 1e-12), marker="o", label="learned")
    axes[1, 0].plot(t, np.maximum(study["reference_mean_rmse"], 1e-12), marker="o", label="reference")
    axes[1, 0].set_title("Posterior-mean RMSE")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(t, np.maximum(study["reference_ess"], 1e-12), marker="o")
    axes[1, 1].set_title("SNIS effective sample size")
    axes[1, 1].set_yscale("log")
    for ax in axes.ravel():
        ax.set_xlabel("repeated Bayes step t")
        ax.grid(alpha=0.2)
    fig.suptitle("Sequential convergence against likelihood-based reference", fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=180)
    display(fig)
    plt.close(fig)

#%% 16) Training diagnostics visualisation
def _format_train_step_k(value: float, _position: int | None = None) -> str:
    """Compact train-step tick labels: 1000 -> 1k, 12500 -> 12.5k."""
    value = float(value)
    if abs(value) >= 1000.0:
        scaled = value / 1000.0
        return f"{scaled:g}k"
    return f"{value:g}"


def plot_training_diagnostics(
    history: dict[str, list],
    best_epoch: int,
    destination: Path | None = None,
    cfg: BayesTransportConfig = CFG,
):
    """Training-objective diagnostics plus evaluation-only sequential Bayes behaviour."""
    steps = np.arange(1, len(history["step_loss"]) + 1)
    epochs = np.arange(1, len(history["epoch_train_loss"]) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 9.0), constrained_layout=True)

    values = np.maximum(np.asarray(history["step_loss"], dtype=float), 1e-12)
    raw_loss_line = axes[0, 0].plot(
        steps, values, linewidth=0.70, alpha=0.28, label=f"training loss ({cfg.training_mode})"
    )[0]
    if len(values) >= 20:
        window = max(5, len(values) // 100)
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(
            steps[window - 1:],
            smoothed,
            linewidth=2.35,
            alpha=0.98,
            color=raw_loss_line.get_color(),
            label=f"moving average ({window})",
        )
    axes[0, 0].set_title("Training objective at every train step", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("train step")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        steps, np.maximum(np.asarray(history["step_energy_score"], dtype=float), 1e-12),
        linewidth=0.75, label="energy score"
    )
    axes[0, 1].plot(
        steps, np.maximum(np.asarray(history["step_mean_rmse"], dtype=float), 1e-12),
        linewidth=0.75, label="posterior-mean RMSE"
    )
    axes[0, 1].set_title("Physical posterior quality", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("train step")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(steps, history["step_attraction"], linewidth=0.75, label="target attraction")
    axes[0, 2].plot(steps, history["step_repulsion"], linewidth=0.75, label="cloud pairwise spread")
    axes[0, 2].set_title("Objective decomposition", loc="left", fontweight="bold")
    axes[0, 2].set_xlabel("train step")
    axes[0, 2].grid(alpha=0.2)
    axes[0, 2].legend(fontsize=8)

    axes[0, 3].plot(steps, history["step_grad_norm"], linewidth=0.75, label="gradient norm")
    axes[0, 3].set_title("Gradient / equilibrium diagnostics", loc="left", fontweight="bold")
    axes[0, 3].set_xlabel("train step")
    axes[0, 3].grid(alpha=0.2)
    fp_residual = np.asarray(
        history.get("step_fixed_point_residual", np.full(len(steps), np.nan)), dtype=float
    )
    if fp_residual.shape == steps.shape and np.any(np.isfinite(fp_residual)):
        fp_axis = axes[0, 3].twinx()
        fp_axis.plot(
            steps, np.maximum(fp_residual, 1e-12), linewidth=0.75, linestyle=":",
            label="DEQ fixed-point residual",
        )
        fp_axis.set_yscale("log")
        fp_axis.set_ylabel("fixed-point residual")
    axes[0, 3].legend(fontsize=8, loc="upper left")

    train_step_formatter = FuncFormatter(_format_train_step_k)
    for ax in axes[0, :]:
        ax.xaxis.set_major_formatter(train_step_formatter)

    axes[1, 0].plot(
        epochs, np.maximum(history["epoch_train_loss"], 1e-12), marker="o", markersize=3,
        label=f"train loss ({cfg.training_mode})"
    )
    axes[1, 0].plot(
        epochs, np.maximum(history["epoch_val_loss"], 1e-12), marker="o", markersize=3,
        label="iid validation energy score (model selection)"
    )
    axes[1, 0].axvline(best_epoch, linestyle="--", linewidth=1.0, label=f"best epoch {best_epoch}")
    if "epoch_learning_rate" in history:
        lr_axis = axes[1, 0].twinx()
        lr_axis.plot(
            epochs,
            np.asarray(history["epoch_learning_rate"], dtype=float),
            linestyle=":",
            linewidth=1.4,
            alpha=0.75,
            label="learning rate",
        )
        lr_axis.set_yscale("log")
        lr_axis.set_ylabel("learning rate")
    axes[1, 0].set_title("Model-selection objective", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    energy_by_o = np.asarray(history["epoch_val_energy_by_o"], dtype=float)
    selected_epochs = np.unique(
        np.clip(np.rint(np.linspace(0, len(energy_by_o) - 1, 5)).astype(int), 0, len(energy_by_o) - 1)
    )
    observation_axis = np.arange(cfg.min_observations_per_step, cfg.max_observations_per_step + 1)
    for epoch_index in selected_epochs:
        axes[1, 1].plot(
            observation_axis,
            np.maximum(energy_by_o[epoch_index], 1e-12),
            marker="o", markersize=3,
            label=f"epoch {epoch_index + 1}",
        )
    axes[1, 1].set_title("IID validation ES by observation prefix", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("number of observations o")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    sequential_energy = np.asarray(history["epoch_val_energy_by_t"], dtype=float)
    sequential_rmse = np.asarray(history["epoch_val_rmse_by_t"], dtype=float)
    t_axis = np.arange(1, sequential_energy.shape[1] + 1)
    for epoch_index in selected_epochs:
        axes[1, 2].plot(
            t_axis, np.maximum(sequential_energy[epoch_index], 1e-12),
            label=f"epoch {epoch_index + 1}",
        )
        axes[1, 3].plot(
            t_axis, np.maximum(sequential_rmse[epoch_index], 1e-12),
            label=f"epoch {epoch_index + 1}",
        )
    axes[1, 2].set_title("Repeated-Bayes ES during training", loc="left", fontweight="bold")
    axes[1, 3].set_title("Repeated-Bayes RMSE during training", loc="left", fontweight="bold")
    for ax in axes[1, 2:]:
        ax.set_xlabel("evaluation-only sequential step t")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Bayes transport training={cfg.training_mode} + unchanged sequential evaluation "
        f"({cfg.posterior_conditioning})",
        fontsize=14,
        fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 17) Training function
def train_model(
    train_loader: DataLoader,
    amortized_eval_data: dict[str, np.ndarray],
    sequential_eval_data: dict[str, np.ndarray],
    fixed_trajectory: dict[str, np.ndarray],
    fixed_prior_particles: np.ndarray,
    run_dir: Path,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Any]:
    """Train the configured objective while keeping the evaluation protocol unchanged.

    ``energy_score`` differentiates the original direct map. ``deq_fixed_point`` solves an
    augmented recurrent-trajectory equilibrium and uses a custom implicit VJP. ``drifting`` is a
    single network call with frozen drift-target regression; its temporal evolution is optimizer
    time, not an inner t-loop. Energy-score and drifting training can replace C_tau by detached
    historical model outputs sampled from a bounded replay buffer, with fresh observations drawn
    from the associated stored theta*. Every mode still uses all within-step observation prefixes.

    Validation/model selection always evaluates the original direct empirical energy score, and
    the separate sequential dataset always evaluates repeated application of the learned map.
    """
    model = SequentialBayesModel(cfg, key=jax.random.key(cfg.seed))
    print(
        f"\namortized Bayes transport: training={cfg.training_mode}, "
        f"conditioning={cfg.posterior_conditioning}"
    )
    print_model_parameter_count(model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    plateau = optax.contrib.reduce_on_plateau(
        factor=0.5,
        patience=cfg.lr_plateau_patience,
        rtol=cfg.lr_plateau_rtol,
        cooldown=0,
        accumulation_size=1,
        min_scale=0.01
    )
    plateau_state = plateau.init(params)

    replay_enabled = (
        cfg.training_mode in {"energy_score", "drifting"}
        and cfg.historical_output_prior_probability > 0.0
    )
    replay_buffer = HistoricalOutputPriorBuffer(cfg.historical_output_buffer_capacity)
    replay_rng = np.random.default_rng(cfg.seed + 73_001)

    @eqx.filter_jit
    def train_step(candidate_model, candidate_opt_state, learning_rate_scale, batch):
        (loss, (metrics, replay_clouds)), grads = eqx.filter_value_and_grad(
            batch_objective, has_aux=True
        )(candidate_model, batch, cfg)
        params = eqx.filter(candidate_model, eqx.is_array)
        updates, candidate_opt_state = optimizer.update(grads, candidate_opt_state, params)
        # The base AdamW update already contains cfg.learning_rate.  ReduceLROnPlateau
        # supplies a multiplicative scale that is held fixed throughout this epoch.
        updates = jax.tree_util.tree_map(lambda update: learning_rate_scale * update, updates)
        candidate_model = eqx.apply_updates(candidate_model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return candidate_model, candidate_opt_state, loss, metrics, grad_norm, replay_clouds

    history: dict[str, list] = {
        "step_loss": [],
        "step_energy_score": [],
        "step_final_energy_score": [],
        "step_mean_rmse": [],
        "step_attraction": [],
        "step_repulsion": [],
        "step_grad_norm": [],
        "step_fixed_point_steps": [],
        "step_fixed_point_residual": [],
        "step_historical_prior_fraction": [],
        "step_historical_buffer_size": [],
        "epoch_train_loss": [],
        "epoch_learning_rate": [],
        "epoch_val_loss": [],
        "epoch_val_energy_score": [],
        "epoch_val_final_energy_score": [],
        "epoch_val_mean_rmse": [],
        "epoch_val_attraction": [],
        "epoch_val_repulsion": [],
        "epoch_val_energy_by_o": [],
        "epoch_val_rmse_by_o": [],
        "epoch_val_spread_by_o": [],
        # Evaluation-only recurrence retained under the historical *_by_t names.
        "epoch_seq_energy_score": [],
        "epoch_seq_final_energy_score": [],
        "epoch_seq_mean_rmse": [],
        "epoch_val_energy_by_t": [],
        "epoch_val_rmse_by_t": [],
        "epoch_val_spread_by_t": [],
    }

    conditioning_label = "cross-attention" if cfg.posterior_conditioning == "cross_attention" else "AdaLN"
    plot_posterior_evolution(
        model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        run_dir / "plots" / "fixed_trajectory_before_training.png",
        f"evaluation-only repeated Bayes ({conditioning_label}): before training",
    )

    initial_iid = evaluate_amortized_model(
        model, amortized_eval_data, cfg, seed=cfg.seed + 91_000
    )
    initial_seq = evaluate_model(
        model, sequential_eval_data, cfg, seed=cfg.seed + 92_000
    )
    print(
        f"[amortized] initial iid ES={initial_iid['energy_score']:.6f} | "
        f"RMSE={initial_iid['posterior_mean_rmse']:.5f} || sequential final ES="
        f"{initial_seq['final_energy_score']:.6f}"
    )

    visualisation_epochs = sorted(
        set(max(1, int(math.ceil(fraction * cfg.epochs / 10.0))) for fraction in range(1, 11))
    )
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    training_started_at = time.time()
    n_steps = cfg.n_train_trajectories // cfg.batch_size
    if n_steps < 1:
        raise ValueError("n_train_trajectories must be at least one batch_size.")
    train_iterator = iter(train_loader)

    for epoch in range(1, cfg.epochs + 1):
        epoch_started_at = time.time()
        epoch_lr_scale = plateau_state.scale
        epoch_learning_rate = cfg.learning_rate * float(jax.device_get(epoch_lr_scale))
        train_losses_this_epoch: list[float] = []
        progress = tqdm(
            range(n_steps),
            desc=f"amortized epoch {epoch:03d}/{cfg.epochs:03d}",
            dynamic_ncols=True,
            leave=True,
            mininterval=5.0,
        )

        for _ in progress:
            # Fresh simulator rows always begin from C_tau. For energy_score/drifting, a configured
            # fraction can instead be replaced by detached historical model outputs. A replayed row
            # carries the theta* and EXACT observation block used when that output was first/currently
            # produced; only the prior cloud changes. DEQ never uses this host-side replay path.
            batch_np = next(train_iterator)
            replayed_rows = 0
            if replay_enabled:
                batch_np, replayed_rows = replay_buffer.mix_into_batch(
                    batch_np, replay_rng, cfg
                )
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            model, opt_state, loss, metrics, grad_norm, replay_clouds = train_step(
                model, opt_state, epoch_lr_scale, batch
            )
            if replay_enabled:
                replay_buffer.add_batch(
                    np.asarray(jax.device_get(replay_clouds)), batch_np, cfg
                )
            host = jax.device_get(metrics)
            host_loss = float(jax.device_get(loss))
            host_grad_norm = float(jax.device_get(grad_norm))
            global_step += 1

            train_losses_this_epoch.append(host_loss)
            history["step_loss"].append(host_loss)
            history["step_energy_score"].append(float(host["energy_score"]))
            history["step_final_energy_score"].append(float(host["final_energy_score"]))
            history["step_mean_rmse"].append(float(host["posterior_mean_rmse"]))
            history["step_attraction"].append(float(host["attraction"]))
            history["step_repulsion"].append(float(host["repulsion"]))
            history["step_grad_norm"].append(host_grad_norm)
            history["step_fixed_point_steps"].append(
                float(host.get("fixed_point_steps", np.nan))
            )
            history["step_fixed_point_residual"].append(
                float(host.get("fixed_point_residual", np.nan))
            )
            replay_fraction = replayed_rows / max(int(cfg.batch_size), 1)
            history["step_historical_prior_fraction"].append(float(replay_fraction))
            history["step_historical_buffer_size"].append(float(len(replay_buffer)))
            postfix = {
                "loss": f"{host_loss:.4f}",
                "RMSE": f"{float(host['posterior_mean_rmse']):.4f}",
            }
            if "fixed_point_residual" in host:
                postfix["fpR"] = f"{float(host['fixed_point_residual']):.2e}"
            if replay_enabled:
                postfix["replay"] = f"{replayed_rows}/{cfg.batch_size}"
            progress.set_postfix(**postfix, refresh=False)

        epoch_train_loss = float(np.mean(train_losses_this_epoch))
        val_metrics = evaluate_amortized_model(
            model,
            amortized_eval_data,
            cfg,
            seed=cfg.seed + 91_000,  # identical iid validation clouds every epoch
        )
        seq_metrics = evaluate_model(
            model,
            sequential_eval_data,
            cfg,
            seed=cfg.seed + 92_000,  # identical recurrent diagnostic clouds every epoch
        )

        # Update the scheduler exactly once from the SAME left-out iid validation set.
        # The new scale is used starting with the next epoch, never within the current epoch.
        _, plateau_state = plateau.update(
            updates=eqx.filter(model, eqx.is_array),
            state=plateau_state,
            value=jnp.asarray(val_metrics["loss"], dtype=jnp.float32),
        )
        next_learning_rate = cfg.learning_rate * float(jax.device_get(plateau_state.scale))

        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_learning_rate"].append(epoch_learning_rate)
        history["epoch_val_loss"].append(float(val_metrics["loss"]))
        history["epoch_val_energy_score"].append(float(val_metrics["energy_score"]))
        history["epoch_val_final_energy_score"].append(float(val_metrics["final_energy_score"]))
        history["epoch_val_mean_rmse"].append(float(val_metrics["posterior_mean_rmse"]))
        history["epoch_val_attraction"].append(float(val_metrics["attraction"]))
        history["epoch_val_repulsion"].append(float(val_metrics["repulsion"]))
        history["epoch_val_energy_by_o"].append(np.asarray(val_metrics["energy_by_o"], dtype=np.float64))
        history["epoch_val_rmse_by_o"].append(np.asarray(val_metrics["rmse_by_o"], dtype=np.float64))
        history["epoch_val_spread_by_o"].append(np.asarray(val_metrics["spread_by_o"], dtype=np.float64))

        history["epoch_seq_energy_score"].append(float(seq_metrics["energy_score"]))
        history["epoch_seq_final_energy_score"].append(float(seq_metrics["final_energy_score"]))
        history["epoch_seq_mean_rmse"].append(float(seq_metrics["posterior_mean_rmse"]))
        history["epoch_val_energy_by_t"].append(np.asarray(seq_metrics["energy_by_t"], dtype=np.float64))
        history["epoch_val_rmse_by_t"].append(np.asarray(seq_metrics["rmse_by_t"], dtype=np.float64))
        history["epoch_val_spread_by_t"].append(np.asarray(seq_metrics["spread_by_t"], dtype=np.float64))

        save_model(run_dir / "artefacts" / "model_last.eqx", model)
        if epoch % cfg.save_every_epochs == 0:
            save_model(run_dir / "artefacts" / f"model_epoch_{epoch:04d}.eqx", model)
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            save_model(run_dir / "artefacts" / "model_best.eqx", model)

        np.savez_compressed(
            run_dir / "artefacts" / "history.npz",
            **{name: np.asarray(values) for name, values in history.items()},
        )
        save_json(
            run_dir / "artefacts" / "training_state.json",
            {
                "training": (
                    "continuous simulator rows; DEQ uses one theta* with "
                    f"{cfg.training_fixed_point_max_iterations} fresh observation blocks; "
                    "energy_score/drifting are one-step and may replay historical outputs; "
                    f"training_mode={cfg.training_mode}"
                ),
                "data_assumption": (
                    "theta* ~ base prior; one shared tau ~ Uniform[0,1]; interpolation anchor "
                    "matches theta* with configured probability, otherwise it is independent"
                ),
                "conditioning": cfg.posterior_conditioning,
                "posterior_recurrence_in_training": (cfg.training_mode == "deq_fixed_point"),
                "training_fixed_point_max_iterations": cfg.training_fixed_point_max_iterations,
                "fresh_observation_block_each_training_iteration": (
                    cfg.training_mode == "deq_fixed_point"
                ),
                "historical_output_prior_probability": cfg.historical_output_prior_probability,
                "historical_output_buffer_capacity": cfg.historical_output_buffer_capacity,
                "historical_output_buffer_size": len(replay_buffer),
                "sequential_recurrence": (
                    "DEQ training and evaluation reuse the same learned Bayes-update cell; drifting "
                    "has no inner recurrence; evaluation protocol itself is unchanged"
                ),
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "learning_rate_used_this_epoch": epoch_learning_rate,
                "learning_rate_for_next_epoch": next_learning_rate,
                "lr_plateau_patience": cfg.lr_plateau_patience,
                "lr_plateau_rtol": cfg.lr_plateau_rtol,
                "elapsed_seconds": time.time() - training_started_at,
                "objective": cfg.training_mode,
                "evaluation_objective": "unchanged direct empirical multivariate energy score",
                "synthetic_input_cloud_independent_of_joint_target": False,
                "synthetic_input_clouds_per_joint_draw": 1,
                "base_prior_distribution": cfg.base_prior_distribution,
                "interpolation_tau_distribution": "one shared tau ~ Uniform[0,1]",
                "synthetic_prior_match_probability": cfg.synthetic_prior_match_probability,
                "fixed_point_solver": cfg.fixed_point_solver,
                "fixed_point_backward_solver": cfg.fixed_point_backward_solver,
                "drifting_field": cfg.drifting_field,
                "drifting_eta": cfg.drifting_eta,
                "historical_replay_observation_policy": "reuse exact stored observation block",
                "particle_pair_term": "all within-cloud pairs with empirical N^2 normalization",
                "min_observations_per_step": cfg.min_observations_per_step,
                "max_observations_per_step": cfg.max_observations_per_step,
                "test_observations_per_step": cfg.test_observations_per_step,
            },
        )

        print(
            f"[amortized] epoch {epoch:03d}: train loss={epoch_train_loss:.6f} | "
            f"val ES={float(val_metrics['energy_score']):.6f} | "
            f"val RMSE={float(val_metrics['posterior_mean_rmse']):.5f} | "
            f"lr={epoch_learning_rate:.3e} -> {next_learning_rate:.3e} || "
            f"seq final ES={float(seq_metrics['final_energy_score']):.6f} | "
            f"seq final RMSE={float(seq_metrics['final_mean_rmse']):.5f} | "
            f"{time.time() - epoch_started_at:.1f}s"
        )

        if epoch in visualisation_epochs:
            plot_posterior_evolution(
                model,
                fixed_trajectory,
                fixed_prior_particles,
                cfg,
                run_dir / "plots" / f"fixed_trajectory_epoch_{epoch:04d}.png",
                f"evaluation-only repeated Bayes after amortized epoch {epoch} ({conditioning_label})",
            )

    best_model = load_model(
        run_dir / "artefacts" / "model_best.eqx", cfg, key=jax.random.key(0)
    )
    final_amortized_metrics = evaluate_amortized_model(
        best_model, amortized_eval_data, cfg, seed=cfg.seed + 91_000
    )
    final_metrics = evaluate_model(
        best_model, sequential_eval_data, cfg, seed=cfg.seed + 92_000
    )
    plot_posterior_evolution(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        run_dir / "plots" / "fixed_trajectory_best_model.png",
        f"best model: evaluation-only repeated Bayes ({conditioning_label}, epoch {best_epoch})",
    )
    plot_training_diagnostics(
        history, best_epoch, run_dir / "plots" / "training_diagnostics.png", cfg
    )

    training_elapsed_seconds = int(time.time() - training_started_at)
    training_hours, training_remainder = divmod(training_elapsed_seconds, 3600)
    training_minutes, training_seconds = divmod(training_remainder, 60)
    print(
        "[amortized] training complete in "
        f"{training_hours:02d}:{training_minutes:02d}:{training_seconds:02d}; "
        f"best epoch={best_epoch}, iid val ES={best_val_loss:.6f}"
    )
    return {
        "model": best_model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "amortized_final_metrics": final_amortized_metrics,
        "final_metrics": final_metrics,
    }

def _recover_epoch_history_from_nohup(path: Path | None) -> dict[str, np.ndarray] | None:
    """Recover every reliable epoch-level quantity printed by an interrupted training job.

    The nohup log cannot recreate per-gradient-step arrays, objective decompositions, gradient
    norms, or the full by-prefix/by-sequential-step validation curves stored in history.npz.
    It does, however, contain one complete summary for every finished epoch.  We recover all
    quantities in those summaries so reload mode can still produce substantive training plots.
    """
    if path is None or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[reload] warning: could not read training log {path}: {exc}")
        return None

    pattern = re.compile(
        r"^\[amortized\] epoch\s+(\d+): train (?:ES|loss)=([^|\s]+) \| "
        r"val ES=([^|\s]+) \| val RMSE=([^|\s]+) \| "
        r"lr=([^|\s]+) -> ([^|\s]+) \|\| "
        r"seq final ES=([^|\s]+) \| seq final RMSE=([^|\s]+) \| "
        r"([^|\s]+)s$",
        flags=re.MULTILINE,
    )
    rows = pattern.findall(text)
    if not rows:
        return None

    values = np.asarray(rows, dtype=object)
    return {
        "epoch": values[:, 0].astype(np.int32),
        "epoch_train_loss": values[:, 1].astype(np.float64),
        "epoch_val_loss": values[:, 2].astype(np.float64),
        "epoch_val_mean_rmse": values[:, 3].astype(np.float64),
        "epoch_learning_rate": values[:, 4].astype(np.float64),
        "epoch_next_learning_rate": values[:, 5].astype(np.float64),
        "epoch_seq_final_energy_score": values[:, 6].astype(np.float64),
        "epoch_seq_final_rmse": values[:, 7].astype(np.float64),
        "epoch_seconds": values[:, 8].astype(np.float64),
    }


def _recover_initial_metrics_from_nohup(path: Path | None) -> dict[str, float] | None:
    """Recover the pre-training evaluation line when it is present in nohup output."""
    if path is None or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    match = re.search(
        r"^\[amortized\] initial iid ES=([^|\s]+) \| RMSE=([^|\s]+) "
        r"\|\| sequential final ES=([^|\s]+)$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    return {
        "initial_iid_energy_score": float(match.group(1)),
        "initial_iid_rmse": float(match.group(2)),
        "initial_seq_final_energy_score": float(match.group(3)),
    }


def _rolling_mean_np(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x-offsets and a same-information moving average for dense recovered curves."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window or window < 2:
        return np.arange(len(values)), values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.arange(window - 1, len(values)), np.convolve(values, kernel, mode="valid")


def plot_recovered_nohup_training_diagnostics(
    recovered: dict[str, np.ndarray],
    best_epoch: int | None,
    destination: Path | None = None,
    *,
    initial_metrics: dict[str, float] | None = None,
    cfg: BayesTransportConfig = CFG,
):
    """Plot all reliable training diagnostics recoverable from an interrupted nohup log.

    This is deliberately separate from plot_training_diagnostics: the original figure remains
    unchanged whenever history.npz is available.  Here we plot only quantities that were actually
    printed once per completed epoch, plus the optional pre-training evaluation line.  No missing
    per-step, attraction/repulsion, gradient-norm, or by-prefix trajectories are fabricated.
    """
    epochs = np.asarray(recovered["epoch"], dtype=np.int32)
    if not len(epochs):
        return

    train_es = np.asarray(recovered["epoch_train_loss"], dtype=np.float64)
    val_es = np.asarray(recovered["epoch_val_loss"], dtype=np.float64)
    val_rmse = np.asarray(recovered["epoch_val_mean_rmse"], dtype=np.float64)
    seq_es = np.asarray(recovered["epoch_seq_final_energy_score"], dtype=np.float64)
    seq_rmse = np.asarray(recovered["epoch_seq_final_rmse"], dtype=np.float64)
    learning_rate = np.asarray(recovered["epoch_learning_rate"], dtype=np.float64)
    next_learning_rate = np.asarray(recovered["epoch_next_learning_rate"], dtype=np.float64)
    epoch_seconds = np.asarray(recovered["epoch_seconds"], dtype=np.float64)

    smoothing_window = min(100, max(10, len(epochs) // 50)) if len(epochs) >= 20 else 1
    smooth_index, train_smooth = _rolling_mean_np(train_es, smoothing_window)
    _, val_smooth = _rolling_mean_np(val_es, smoothing_window)
    _, rmse_smooth = _rolling_mean_np(val_rmse, smoothing_window)
    _, seq_es_smooth = _rolling_mean_np(seq_es, smoothing_window)
    _, seq_rmse_smooth = _rolling_mean_np(seq_rmse, smoothing_window)
    smooth_epochs = epochs[smooth_index]

    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.2), constrained_layout=True)

    train_raw = axes[0, 0].plot(
        epochs, np.maximum(train_es, 1e-12), linewidth=0.65, alpha=0.22, label=f"train loss ({cfg.training_mode})"
    )[0]
    val_raw = axes[0, 0].plot(
        epochs, np.maximum(val_es, 1e-12), linewidth=0.65, alpha=0.22, label="iid val ES"
    )[0]
    axes[0, 0].plot(
        smooth_epochs,
        np.maximum(train_smooth, 1e-12),
        linewidth=2.15,
        alpha=0.96,
        color=train_raw.get_color(),
        label=f"train loss ({smoothing_window}-epoch mean)",
    )
    axes[0, 0].plot(
        smooth_epochs,
        np.maximum(val_smooth, 1e-12),
        linewidth=2.15,
        alpha=0.96,
        color=val_raw.get_color(),
        label=f"iid val ES ({smoothing_window}-epoch mean)",
    )
    if initial_metrics is not None and "initial_iid_energy_score" in initial_metrics:
        axes[0, 0].scatter(
            [0], [max(initial_metrics["initial_iid_energy_score"], 1e-12)],
            marker="x", s=55, label="pre-training iid ES",
        )
    axes[0, 0].set_title("Training loss and IID validation ES", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("objective / energy score")
    axes[0, 0].set_yscale("log")

    rmse_raw = axes[0, 1].plot(
        epochs, np.maximum(val_rmse, 1e-12), linewidth=0.65, alpha=0.24, label="iid val RMSE"
    )[0]
    axes[0, 1].plot(
        smooth_epochs,
        np.maximum(rmse_smooth, 1e-12),
        linewidth=2.15,
        alpha=0.96,
        color=rmse_raw.get_color(),
        label=f"{smoothing_window}-epoch mean",
    )
    if initial_metrics is not None and "initial_iid_rmse" in initial_metrics:
        axes[0, 1].scatter(
            [0], [max(initial_metrics["initial_iid_rmse"], 1e-12)],
            marker="x", s=55, label="pre-training iid RMSE",
        )
    axes[0, 1].set_title("IID validation posterior-mean RMSE", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_yscale("log")

    seq_es_raw = axes[0, 2].plot(
        epochs, np.maximum(seq_es, 1e-12), linewidth=0.65, alpha=0.24, label="sequential final ES"
    )[0]
    axes[0, 2].plot(
        smooth_epochs,
        np.maximum(seq_es_smooth, 1e-12),
        linewidth=2.15,
        alpha=0.96,
        color=seq_es_raw.get_color(),
        label=f"{smoothing_window}-epoch mean",
    )
    if initial_metrics is not None and "initial_seq_final_energy_score" in initial_metrics:
        axes[0, 2].scatter(
            [0], [max(initial_metrics["initial_seq_final_energy_score"], 1e-12)],
            marker="x", s=55, label="pre-training sequential ES",
        )
    axes[0, 2].set_title("Evaluation-only repeated-Bayes final ES", loc="left", fontweight="bold")
    axes[0, 2].set_ylabel("final energy score")
    axes[0, 2].set_yscale("log")

    seq_rmse_raw = axes[1, 0].plot(
        epochs, np.maximum(seq_rmse, 1e-12), linewidth=0.65, alpha=0.24, label="sequential final RMSE"
    )[0]
    axes[1, 0].plot(
        smooth_epochs,
        np.maximum(seq_rmse_smooth, 1e-12),
        linewidth=2.15,
        alpha=0.96,
        color=seq_rmse_raw.get_color(),
        label=f"{smoothing_window}-epoch mean",
    )
    axes[1, 0].set_title("Evaluation-only repeated-Bayes final RMSE", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("final RMSE")
    axes[1, 0].set_yscale("log")

    axes[1, 1].step(epochs, np.maximum(learning_rate, 1e-16), where="post", linewidth=1.5, label="used")
    if not np.array_equal(learning_rate, next_learning_rate):
        axes[1, 1].step(
            epochs,
            np.maximum(next_learning_rate, 1e-16),
            where="post",
            linewidth=1.1,
            linestyle="--",
            label="next epoch",
        )
    axes[1, 1].set_title("Effective learning-rate schedule", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("learning rate")
    axes[1, 1].set_yscale("log")

    finite_seconds = np.where(np.isfinite(epoch_seconds), epoch_seconds, 0.0)
    duration_raw = axes[1, 2].plot(
        epochs, epoch_seconds, linewidth=0.65, alpha=0.35, label="epoch duration"
    )[0]
    if len(epoch_seconds) >= 20:
        duration_index, duration_smooth = _rolling_mean_np(epoch_seconds, smoothing_window)
        axes[1, 2].plot(
            epochs[duration_index],
            duration_smooth,
            linewidth=2.1,
            alpha=0.96,
            color=duration_raw.get_color(),
            label=f"{smoothing_window}-epoch mean",
        )
    axes[1, 2].set_title("Epoch duration and cumulative logged time", loc="left", fontweight="bold")
    axes[1, 2].set_ylabel("seconds / epoch")
    cumulative_axis = axes[1, 2].twinx()
    cumulative_axis.plot(epochs, np.cumsum(finite_seconds) / 3600.0, linestyle=":", linewidth=1.5)
    cumulative_axis.set_ylabel("cumulative logged hours")

    for ax in axes.ravel():
        ax.set_xlabel("completed epoch")
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
        if best_epoch is not None and epochs[0] <= int(best_epoch) <= epochs[-1]:
            ax.axvline(int(best_epoch), linestyle="--", linewidth=1.0, alpha=0.75)

    finite_val = np.isfinite(val_es)
    if np.any(finite_val):
        finite_indices = np.flatnonzero(finite_val)
        best_index = int(finite_indices[np.argmin(val_es[finite_val])])
        recovered_best_epoch = int(epochs[best_index])
        recovered_best_loss = float(val_es[best_index])
        axes[0, 0].scatter(
            [recovered_best_epoch], [max(recovered_best_loss, 1e-12)], marker="*", s=90,
            label=f"log-best epoch {recovered_best_epoch}",
        )
        axes[0, 0].legend(fontsize=7)

    fig.suptitle(
        "Recovered interrupted-run training diagnostics from nohup log\n"
        f"Bayes transport training={cfg.training_mode} + unchanged sequential evaluation "
        f"({cfg.posterior_conditioning})",
        fontsize=14,
        fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


def _nohup_epoch_record(
    recovered: dict[str, np.ndarray] | None,
    epoch: int | None,
) -> dict[str, float] | None:
    if recovered is None or epoch is None:
        return None
    hits = np.flatnonzero(recovered["epoch"] == int(epoch))
    if not len(hits):
        return None
    index = int(hits[-1])
    return {
        name: float(values[index])
        for name, values in recovered.items()
        if name != "epoch"
    }


def _full_training_history_available(history: dict[str, np.ndarray] | None) -> bool:
    """Whether the exact existing training-diagnostics plot can be reproduced."""
    required = {
        "step_loss",
        "step_energy_score",
        "step_mean_rmse",
        "step_attraction",
        "step_repulsion",
        "step_grad_norm",
        "epoch_train_loss",
        "epoch_val_loss",
        "epoch_val_energy_by_o",
        "epoch_val_energy_by_t",
        "epoch_val_rmse_by_t",
    }
    return history is not None and required.issubset(history) and len(history["epoch_train_loss"]) > 0


def _periodic_checkpoint_epoch(path: Path) -> int | None:
    match = re.fullmatch(r"model_epoch_(\d+)\.eqx", path.name)
    return int(match.group(1)) if match is not None else None


def _reload_checkpoint_candidates(artefact_dir: Path) -> list[Path]:
    """Prefer the historical best model, then last model, then newest periodic checkpoint."""
    candidates = [artefact_dir / "model_best.eqx", artefact_dir / "model_last.eqx"]
    periodic = sorted(
        artefact_dir.glob("model_epoch_*.eqx"),
        key=lambda path: _periodic_checkpoint_epoch(path) or -1,
        reverse=True,
    )
    candidates.extend(periodic)
    return [path for path in candidates if path.is_file()]


#%% 18) Create a new run OR reload one existing amortized run folder
np.random.seed(CFG.seed)
print("JAX devices:", jax.devices())
print("Configuration:\n", yaml.safe_dump(asdict(CFG), sort_keys=False))

if train_wm:
    # One run directory contains the selected conditioning mechanism and both iid-training
    # and evaluation-only sequential diagnostics.  The exact script is the run configuration:
    # no separate config.yaml is written.
    run_dir = make_run_dir(CFG.env_name, CFG.runs_base)
    archived_script = copy_running_script_to_run_dir(run_dir)
    print("Run directory:", run_dir)
    print("Archived training script:", archived_script)

    # Training is an INFINITE CPU stream of iid simulator rows plus synthetic interpolated
    # input clouds. Two deterministic validation sets are kept separate: `amortized_eval_data`
    # matches the direct training distribution, while `eval_data` keeps the old fixed-theta
    # trajectory format solely for the repeated-Bayes stress test from the tau=0 base prior.
    train_loader = make_continuous_train_loader(CFG, seed=CFG.seed + 1_000)
    iid_eval_rng = np.random.default_rng(CFG.seed + 2_000)
    amortized_eval_data = simulate_iid_joint_samples(
        iid_eval_rng, CFG.n_eval_trajectories, CFG,
        shape_pool=TRAIN_SHAPES, balanced_shapes=True,
    )
    sequential_eval_rng = np.random.default_rng(CFG.seed + 2_100)
    eval_data = simulate_trajectories(
        sequential_eval_rng, CFG.n_eval_trajectories, CFG.evaluation_trajectory_length, CFG,
        shape_pool=TRAIN_SHAPES, balanced_shapes=True,
    )

    if CFG.base_prior_distribution == "uniform":
        prior_mode = f"Uniform([{CFG.design_low}, {CFG.design_high}] per active coordinate)"
    else:
        prior_mode = f"Gaussian N(0, {CFG.prior_std}^2 I)"
    print("Continuous iid training stream:")
    print(f"  fresh joint samples per nominal epoch: {CFG.n_train_trajectories}")
    print(f"  batch size (unchanged): {CFG.batch_size}")
    print(f"  S,D grid: [{CFG.min_num_sources},{CFG.max_num_sources}] x [{CFG.min_source_dim},{CFG.max_source_dim}]")
    print(f"  held-out training shapes: {HELDOUT_SHAPES}")
    print(f"  training shapes: {len(TRAIN_SHAPES)} / {len(ALL_SHAPES)} combinations")
    print(f"  base prior rho_0: {prior_mode}")
    print("  synthetic prior interpolation: one shared tau ~ Uniform[0,1]")
    print(f"  P(interpolation anchor = theta*): {CFG.synthetic_prior_match_probability:.3f}")
    print(f"  training mode: {CFG.training_mode}")
    if CFG.training_mode == "deq_fixed_point":
        print(
            f"  DEQ solver: forward={CFG.fixed_point_solver}, "
            f"backward={CFG.fixed_point_backward_solver}"
        )
    if CFG.training_mode == "drifting":
        print(
            f"  drifting field: {CFG.drifting_field}; eta={CFG.drifting_eta:g}; "
            f"temperatures={CFG.drifting_temperatures}"
        )
    if CFG.training_mode in {"energy_score", "drifting"}:
        print(
            "  historical-output prior replay: "
            f"p={CFG.historical_output_prior_probability:.3f}, "
            f"capacity={CFG.historical_output_buffer_capacity}"
        )
    print(f"  training prefixes: {list(range(CFG.min_observations_per_step, CFG.max_observations_per_step + 1))}")
    print(f"  posterior conditioning: {CFG.posterior_conditioning}")
    if CFG.min_observations_per_step == 1 and CFG.max_observations_per_step == 1:
        print("  likelihood Transformer: zero-parameter single-observation bypass")
    else:
        print(
            "  likelihood Transformer: "
            f"hidden={CFG.likelihood_hidden_dim}, heads={CFG.likelihood_heads}, "
            f"mlp_ratio={CFG.likelihood_mlp_ratio}, depth={CFG.likelihood_depth}"
        )

    # Keep one fixed 2-D problem for physical posterior plots. It is generated separately
    # so heterogeneous eval_data is free to begin with any shape.  Sequential evaluation
    # always starts from a fresh tau=0 cloud sampled from the same base prior rho_0.
    fixed_rng = np.random.default_rng(CFG.seed + 2_500)
    fixed_data = simulate_trajectories(
        fixed_rng,
        1,
        CFG.evaluation_trajectory_length,
        CFG,
        fixed_num_sources=CFG.num_sources,
        fixed_source_dim=CFG.source_dim,
    )
    fixed_trajectory = {
        "theta_true": fixed_data["theta_true"][0],
        "observations": fixed_data["observations"][0],
        "observation_count": np.asarray(CFG.test_observations_per_step, dtype=np.int32),
        "num_sources": fixed_data["num_sources"][0],
        "theta_size": fixed_data["theta_size"][0],
    }
    fixed_prior_active = sample_base_prior_np(
        np.random.default_rng(CFG.seed + 3_000),
        CFG.num_particles,
        CFG,
        num_sources=CFG.num_sources,
        source_dim=CFG.source_dim,
    )
    fixed_prior_particles = pad_theta_np(fixed_prior_active, CFG)
    np.savez_compressed(
        run_dir / "artefacts" / "fixed_trajectory.npz",
        **{key: np.asarray(value) for key, value in fixed_trajectory.items()},
        prior_particles=fixed_prior_particles,
    )
else:
    # Reload path.  Run the archived script from the existing amortized run folder itself.
    # BayesTransportConfig in THIS script is used verbatim; no config file is loaded.  All
    # refreshed plots and diagnostics remain in that same run; no new run is made.
    run_dir = Path.cwd().expanduser().resolve()
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
    print("Existing amortized run directory:", run_dir)
    print("Main-model training is disabled; using BayesTransportConfig from this script.")
    print("Reloading saved model and regenerating available diagnostics.")

    # Regenerate both deterministic validation sets from the CURRENT script configuration.
    iid_eval_rng = np.random.default_rng(CFG.seed + 2_000)
    amortized_eval_data = simulate_iid_joint_samples(
        iid_eval_rng, CFG.n_eval_trajectories, CFG,
        shape_pool=TRAIN_SHAPES, balanced_shapes=True,
    )
    sequential_eval_rng = np.random.default_rng(CFG.seed + 2_100)
    eval_data = simulate_trajectories(
        sequential_eval_rng, CFG.n_eval_trajectories, CFG.evaluation_trajectory_length, CFG,
        shape_pool=TRAIN_SHAPES, balanced_shapes=True,
    )

    # Rebuild the fixed physical evaluation problem at the CURRENT requested horizon rather
    # than truncating to a horizon used during training.  The deterministic simulator seed and
    # the base-prior sampler are the same as in train mode.
    fixed_rng = np.random.default_rng(CFG.seed + 2_500)
    fixed_data = simulate_trajectories(
        fixed_rng,
        1,
        CFG.evaluation_trajectory_length,
        CFG,
        fixed_num_sources=CFG.num_sources,
        fixed_source_dim=CFG.source_dim,
    )
    fixed_trajectory = {
        "theta_true": fixed_data["theta_true"][0],
        "observations": fixed_data["observations"][0],
        "observation_count": np.asarray(CFG.test_observations_per_step, dtype=np.int32),
        "num_sources": fixed_data["num_sources"][0],
        "theta_size": fixed_data["theta_size"][0],
    }
    fixed_prior_active = sample_base_prior_np(
        np.random.default_rng(CFG.seed + 3_000),
        CFG.num_particles,
        CFG,
        num_sources=CFG.num_sources,
        source_dim=CFG.source_dim,
    )
    fixed_prior_particles = pad_theta_np(fixed_prior_active, CFG)

# These descriptive plots are cheap and are regenerated in both modes.
plot_architecture_schematic(CFG, run_dir / "plots" / "architecture_schematic.png")
plot_source_trajectory(
    fixed_trajectory, CFG, run_dir / "plots" / "fixed_trajectory_sensor_field.png"
)

#%% 19) Train the single-cloud amortized model, or reload the best available local checkpoint
# Observation embedder + theta embedder + Posterior Transformer are optimized jointly from
# the SAME physical-theta empirical energy-score objective.  The training graph contains no
# posterior recurrence.  The historical lax.scan path is called only by sequential evaluation.
if train_wm:
    result = train_model(
        train_loader,
        amortized_eval_data,
        eval_data,
        fixed_trajectory,
        fixed_prior_particles,
        run_dir,
        CFG,
    )
else:
    artefact_dir = run_dir / "artefacts"
    history_path = artefact_dir / "history.npz"
    state_path = artefact_dir / "training_state.json"

    # Interrupted jobs can leave a perfectly usable model checkpoint without the auxiliary
    # history/state files.  Treat the model as essential and everything else as recoverable or
    # optional.  If model_best is absent/corrupt, fall back to model_last and then the newest
    # periodic checkpoint produced by save_every_epochs.
    checkpoint_candidates = _reload_checkpoint_candidates(artefact_dir)
    if not checkpoint_candidates:
        raise FileNotFoundError(
            "Reload mode needs at least one model checkpoint in artefacts/: expected one of "
            "model_best.eqx, model_last.eqx, or model_epoch_*.eqx."
        )

    best_model = None
    model_path = None
    checkpoint_errors: list[str] = []
    for candidate in checkpoint_candidates:
        try:
            print("[reload] trying model checkpoint:", candidate)
            best_model = load_model(candidate, CFG, key=jax.random.key(0))
            model_path = candidate
            break
        except Exception as exc:
            checkpoint_errors.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
            print(f"[reload] warning: could not load {candidate.name}; trying the next checkpoint.")
    if best_model is None or model_path is None:
        joined_errors = "\n  ".join(checkpoint_errors)
        raise RuntimeError(f"No compatible saved model checkpoint could be loaded.\n  {joined_errors}")
    print("[reload] loaded amortized model from:", model_path)

    # Full history is optional.  A stopped process may have a checkpoint but no usable NPZ.
    history: dict[str, np.ndarray] | None = None
    if history_path.is_file():
        try:
            with np.load(history_path, allow_pickle=False) as saved_history:
                history = {key: np.asarray(saved_history[key]) for key in saved_history.files}
            print(f"[reload] loaded full training history from {history_path}")
        except Exception as exc:
            print(f"[reload] warning: could not read {history_path}: {exc}")
    else:
        print(f"[reload] history file not found; continuing without {history_path.name}.")

    # training_state.json is also optional.  Recover its important scalar information from
    # nohup.log when possible; this is enough to identify the historical best checkpoint.
    training_state: dict[str, Any] = {}
    if state_path.is_file():
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                loaded_state = json.load(handle)
            if isinstance(loaded_state, dict):
                training_state = loaded_state
                print(f"[reload] loaded training state from {state_path}")
        except Exception as exc:
            print(f"[reload] warning: could not read {state_path}: {exc}")
    else:
        print(f"[reload] training-state file not found; continuing without {state_path.name}.")

    recovered_epoch_history = _recover_epoch_history_from_nohup(_reload_nohup_path)
    recovered_initial_metrics = _recover_initial_metrics_from_nohup(_reload_nohup_path)
    if recovered_epoch_history is not None:
        print(
            f"[reload] recovered {len(recovered_epoch_history['epoch'])} completed epoch summaries "
            f"from {_reload_nohup_path}."
        )
        # Preserve the recovered compact information separately; it is intentionally not named
        # history.npz because the nohup log cannot reconstruct per-step/by-prefix training arrays.
        try:
            np.savez_compressed(
                artefact_dir / "reload_epoch_history_from_nohup.npz",
                **recovered_epoch_history,
            )
            csv_names = (
                "epoch", "epoch_train_loss", "epoch_val_loss", "epoch_val_mean_rmse",
                "epoch_learning_rate", "epoch_next_learning_rate",
                "epoch_seq_final_energy_score", "epoch_seq_final_rmse", "epoch_seconds",
            )
            csv_values = np.column_stack([recovered_epoch_history[name] for name in csv_names])
            np.savetxt(
                artefact_dir / "reload_epoch_history_from_nohup.csv",
                csv_values,
                delimiter=",",
                header=",".join(csv_names),
                comments="",
            )
            if recovered_initial_metrics is not None:
                save_json(
                    artefact_dir / "reload_initial_metrics_from_nohup.json",
                    recovered_initial_metrics,
                )
        except OSError as exc:
            print(f"[reload] warning: could not save recovered nohup history: {exc}")

    history_best_epoch: int | None = None
    history_best_val_loss: float | None = None
    history_last_epoch: int | None = None
    if history is not None and "epoch_val_loss" in history:
        history_val = np.asarray(history["epoch_val_loss"], dtype=np.float64)
        finite = np.isfinite(history_val)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            best_index = int(finite_indices[np.argmin(history_val[finite])])
            history_best_epoch = best_index + 1
            history_best_val_loss = float(history_val[best_index])
        if len(history_val):
            history_last_epoch = len(history_val)

    nohup_best_epoch: int | None = None
    nohup_best_val_loss: float | None = None
    nohup_last_epoch: int | None = None
    if recovered_epoch_history is not None:
        recovered_epochs = recovered_epoch_history["epoch"]
        recovered_val = recovered_epoch_history["epoch_val_loss"]
        finite = np.isfinite(recovered_val)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            best_index = int(finite_indices[np.argmin(recovered_val[finite])])
            nohup_best_epoch = int(recovered_epochs[best_index])
            nohup_best_val_loss = float(recovered_val[best_index])
        nohup_last_epoch = int(recovered_epochs[-1])

    checkpoint_kind = (
        "best" if model_path.name == "model_best.eqx"
        else "last" if model_path.name == "model_last.eqx"
        else "periodic"
    )
    periodic_epoch = _periodic_checkpoint_epoch(model_path)
    if checkpoint_kind == "best":
        if "best_epoch" in training_state and "best_val_loss" in training_state:
            loaded_epoch = int(training_state["best_epoch"])
            best_val_loss = float(training_state["best_val_loss"])
        else:
            best_candidates = [
                (epoch, loss)
                for epoch, loss in (
                    (history_best_epoch, history_best_val_loss),
                    (nohup_best_epoch, nohup_best_val_loss),
                )
                if epoch is not None and loss is not None and np.isfinite(loss)
            ]
            if best_candidates:
                loaded_epoch, best_val_loss = min(best_candidates, key=lambda item: item[1])
            else:
                loaded_epoch, best_val_loss = None, None
    elif checkpoint_kind == "last":
        if "epoch" in training_state:
            loaded_epoch = int(training_state["epoch"])
        else:
            last_candidates = [
                epoch for epoch in (history_last_epoch, nohup_last_epoch) if epoch is not None
            ]
            loaded_epoch = max(last_candidates) if last_candidates else None
        record = _nohup_epoch_record(recovered_epoch_history, loaded_epoch)
        best_val_loss = None if record is None else float(record["epoch_val_loss"])
    else:
        loaded_epoch = periodic_epoch
        record = _nohup_epoch_record(recovered_epoch_history, loaded_epoch)
        best_val_loss = None if record is None else float(record["epoch_val_loss"])

    # Keep the historical result key for downstream compatibility.  When we had to fall back
    # from model_best to another checkpoint, this is the epoch of the model actually loaded.
    best_epoch = loaded_epoch

    # These deterministic validation quantities are cheap enough to regenerate and remove any
    # dependence on training-time NPZ files.  They also provide a sensible scalar fallback when
    # the historical validation loss could not be recovered from state/log metadata.
    final_amortized_metrics = evaluate_amortized_model(
        best_model, amortized_eval_data, CFG, seed=CFG.seed + 91_000
    )
    final_metrics = evaluate_model(best_model, eval_data, CFG, seed=CFG.seed + 92_000)
    if best_val_loss is None or not np.isfinite(best_val_loss):
        best_val_loss = float(final_amortized_metrics["loss"])
        print(
            "[reload] historical validation loss unavailable; using the regenerated iid "
            f"validation loss {best_val_loss:.6f} for summary metadata."
        )

    has_full_training_history = _full_training_history_available(history)
    if has_full_training_history:
        plot_training_diagnostics(
            history,
            0 if best_epoch is None else int(best_epoch),
            run_dir / "plots" / "training_diagnostics.png",
            CFG,
        )

    if recovered_epoch_history is not None:
        # The nohup figure uses every reliable epoch-level quantity available in the interrupted
        # log.  If the exact history is absent it becomes the main training_diagnostics.png;
        # otherwise it is retained as a complementary recovered-log diagnostic.
        recovered_destination = (
            run_dir / "plots" / "training_diagnostics_from_nohup.png"
            if has_full_training_history
            else run_dir / "plots" / "training_diagnostics.png"
        )
        plot_recovered_nohup_training_diagnostics(
            recovered_epoch_history,
            best_epoch,
            recovered_destination,
            initial_metrics=recovered_initial_metrics,
            cfg=CFG,
        )
        print(f"[reload] wrote recovered training diagnostics to {recovered_destination}")
    elif not has_full_training_history:
        print(
            "[reload] no full history.npz and no recoverable epoch summaries in nohup output; "
            "training-history plots are unavailable, but all model-based diagnostics continue."
        )

    epoch_label = "epoch unknown" if best_epoch is None else f"epoch {best_epoch}"
    plot_posterior_evolution(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        CFG,
        run_dir / "plots" / "fixed_trajectory_best_model.png",
        f"evaluation-only repeated Bayes ({CFG.posterior_conditioning}, {epoch_label})",
    )
    result = {
        "model": best_model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "amortized_final_metrics": final_amortized_metrics,
        "final_metrics": final_metrics,
        "reload_checkpoint": str(model_path),
        "reload_checkpoint_kind": checkpoint_kind,
        "reload_nohup_log": None if _reload_nohup_path is None else str(_reload_nohup_path),
    }

model = result["model"]
if not train_wm:
    print(
        f"\namortized Bayes transport: training={CFG.training_mode}, "
        f"conditioning={CFG.posterior_conditioning}"
    )
    print_model_parameter_count(model)


#%% 19b) Direct physical best-model posterior visualisation
plot_posterior_evolution(
    model,
    fixed_trajectory,
    fixed_prior_particles,
    CFG,
    run_dir / "plots" / "fixed_trajectory_best_model.png",
    f"evaluation-only repeated Bayes ({CFG.posterior_conditioning}): direct physical posterior evolution",
)

#%% 19c) Paper-style marginals and sequential likelihood-reference convergence
plot_paper_style_marginals(
    model,
    fixed_trajectory,
    fixed_prior_particles,
    CFG,
    run_dir / "plots" / "paper_style_posterior_marginals.png",
)

sequential_reference = sequential_reference_study(
    model,
    fixed_trajectory,
    fixed_prior_particles,
    CFG,
    seed=CFG.seed + 46_000,
)
np.savez_compressed(
    run_dir / "artefacts" / "sequential_reference_study.npz",
    **{name: np.asarray(value) for name, value in sequential_reference.items()},
)
plot_sequential_reference_study(
    sequential_reference,
    run_dir / "plots" / "sequential_reference_convergence.png",
)

#%% 19d) Balanced dimensional-generalisation evaluation, including held-out S,D shapes
def evaluate_dimension_generalisation(
    model: SequentialBayesModel,
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Evaluate every trajectory individually so errors can be grouped by (S,D).

    This uses the same physical energy score and RMSE as training, but stores the final
    posterior particles and per-prefix metrics so dimensional generalisation can be inspected
    rather than reduced to one scalar.  The input dataset is balanced across ALL_SHAPES.
    """
    rng = np.random.default_rng(seed)
    final_particles, final_means, compact_targets = [], [], []
    energy_rows, rmse_rows, spread_rows = [], [], []

    for start in range(0, len(dataset["theta_true"]), cfg.batch_size):
        stop = min(start + cfg.batch_size, len(dataset["theta_true"]))
        indices = np.arange(start, stop)
        batch_np = make_batch_np(
            dataset,
            indices,
            rng,
            cfg,
            observations_per_step=cfg.test_observations_per_step,
        )
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        predicted, _, _ = predict_batch(
            model,
            batch["prior_particles"],
            batch["observations"],
            batch["observation_count"],
            batch["num_sources"],
            batch["theta_size"],
        )
        targets = batch["theta_true"]
        if cfg.canonicalize_particle_sources:
            targets = jax.vmap(canonicalize_padded_sources_jax)(targets, batch["num_sources"])
        targets = jax.vmap(compact_theta_jax)(targets, batch["num_sources"], batch["theta_size"])
        energy, rmse, spread = jax.vmap(_trajectory_metrics)(predicted, targets, batch["theta_size"])

        predicted_np = np.asarray(jax.device_get(predicted), dtype=np.float32)
        final = predicted_np[:, -1]
        final_particles.append(final)
        final_means.append(final.mean(axis=1))
        compact_targets.append(np.asarray(jax.device_get(targets), dtype=np.float32))
        energy_rows.append(np.asarray(jax.device_get(energy), dtype=np.float64))
        rmse_rows.append(np.asarray(jax.device_get(rmse), dtype=np.float64))
        spread_rows.append(np.asarray(jax.device_get(spread), dtype=np.float64))

    final_particles = np.concatenate(final_particles, axis=0)
    final_means = np.concatenate(final_means, axis=0)
    compact_targets = np.concatenate(compact_targets, axis=0)
    energy_by_t = np.concatenate(energy_rows, axis=0)
    rmse_by_t = np.concatenate(rmse_rows, axis=0)
    spread_by_t = np.concatenate(spread_rows, axis=0)
    num_sources = np.asarray(dataset["num_sources"], dtype=np.int32)
    source_dim = np.asarray(dataset["theta_size"] // dataset["num_sources"], dtype=np.int32)
    heldout = np.asarray([(int(s), int(d)) in HELDOUT_SHAPES for s, d in zip(num_sources, source_dim)])

    rmse_grid = np.full((cfg.max_num_sources, cfg.max_source_dim), np.nan)
    energy_grid = np.full_like(rmse_grid, np.nan)
    spread_grid = np.full_like(rmse_grid, np.nan)
    shape_rmse_by_t = np.full((cfg.max_num_sources, cfg.max_source_dim, rmse_by_t.shape[1]), np.nan)
    shape_energy_by_t = np.full_like(shape_rmse_by_t, np.nan)
    rows = []
    for s, d in ALL_SHAPES:
        select = (num_sources == s) & (source_dim == d)
        rmse_grid[s - 1, d - 1] = np.mean(rmse_by_t[select, -1])
        energy_grid[s - 1, d - 1] = np.mean(energy_by_t[select, -1])
        spread_grid[s - 1, d - 1] = np.mean(spread_by_t[select, -1])
        shape_rmse_by_t[s - 1, d - 1] = np.mean(rmse_by_t[select], axis=0)
        shape_energy_by_t[s - 1, d - 1] = np.mean(energy_by_t[select], axis=0)
        rows.append((s, d, int((s, d) in HELDOUT_SHAPES), int(np.sum(select)),
                     rmse_grid[s - 1, d - 1], energy_grid[s - 1, d - 1], spread_grid[s - 1, d - 1]))

    np.savetxt(
        run_dir / "artefacts" / "dimensional_generalisation_by_shape.csv",
        np.asarray(rows, dtype=float),
        delimiter=",",
        header="S,D,heldout,n,final_rmse,final_energy,final_spread",
        comments="",
    )
    return {
        "num_sources": num_sources,
        "source_dim": source_dim,
        "heldout": heldout,
        "final_particles": final_particles,
        "final_means": final_means,
        "targets": compact_targets,
        "energy_by_t": energy_by_t,
        "rmse_by_t": rmse_by_t,
        "spread_by_t": spread_by_t,
        "rmse_grid": rmse_grid,
        "energy_grid": energy_grid,
        "spread_grid": spread_grid,
        "shape_rmse_by_t": shape_rmse_by_t,
        "shape_energy_by_t": shape_energy_by_t,
    }


def plot_dimension_generalisation(
    study: dict[str, np.ndarray],
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
):
    """Benchmark-style shape diagnostics plus source-identification views."""
    if not RUN_DIMENSIONAL_GENERALISATION_DIAGNOSTICS:
        print(
            "[dimensional generalisation] skipped plotting: fixed single-shape setup "
            "(min_num_sources=max_num_sources and min_source_dim=max_source_dim)."
        )
        return

    rmse_grid = study["rmse_grid"]
    energy_grid = study["energy_grid"]

    # Benchmark-style summary: shape heatmaps, true-vs-posterior-mean scatter, and difficulty.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)
    for ax, grid, title, label in (
        (axes[0, 0], rmse_grid, "Final posterior-mean RMSE by shape", "RMSE"),
        (axes[0, 1], energy_grid, "Final energy score by shape", "energy score"),
    ):
        positive = grid[np.isfinite(grid) & (grid > 0)]
        if positive.size == 0:
            image = np.ones_like(grid, dtype=np.float64)
            im = ax.imshow(image, origin="lower", aspect="auto")
        else:
            norm = LogNorm(
                vmin=max(float(np.min(positive)), 1e-8),
                vmax=max(float(np.max(positive)), float(np.min(positive)) * 1.0001),
            )
            im = ax.imshow(grid, origin="lower", aspect="auto", norm=norm)
        ax.set_xticks(np.arange(cfg.max_source_dim), np.arange(1, cfg.max_source_dim + 1))
        ax.set_yticks(np.arange(cfg.max_num_sources), np.arange(1, cfg.max_num_sources + 1))
        ax.set_xlabel("source dimension D")
        ax.set_ylabel("number of sources S")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, label=label)
        for s, d in HELDOUT_SHAPES:
            ax.text(
                d - 1,
                s - 1,
                "H",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=11,
                bbox=dict(boxstyle="circle,pad=0.18", facecolor="white", alpha=0.75, edgecolor="none"),
            )

    active_true, active_pred = [], []
    for i, theta_size in enumerate(dataset["theta_size"]):
        k = int(theta_size)
        active_true.append(study["targets"][i, :k])
        active_pred.append(study["final_means"][i, :k])
    active_true = np.concatenate(active_true)
    active_pred = np.concatenate(active_pred)
    axes[1, 0].scatter(active_true, active_pred, s=7, alpha=0.14)
    lo = float(min(active_true.min(), active_pred.min()))
    hi = float(max(active_true.max(), active_pred.max()))
    axes[1, 0].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    axes[1, 0].set_xlabel("true active theta coordinate")
    axes[1, 0].set_ylabel("final posterior-mean coordinate")
    axes[1, 0].set_title("All active coordinates", fontweight="bold")
    axes[1, 0].grid(alpha=0.2)

    shape_size, shape_rmse, is_heldout = [], [], []
    for s, d in ALL_SHAPES:
        shape_size.append(s * d)
        shape_rmse.append(rmse_grid[s - 1, d - 1])
        is_heldout.append((s, d) in HELDOUT_SHAPES)
    shape_size = np.asarray(shape_size)
    shape_rmse = np.asarray(shape_rmse)
    is_heldout = np.asarray(is_heldout)
    if np.any(~is_heldout):
        axes[1, 1].scatter(shape_size[~is_heldout], shape_rmse[~is_heldout], alpha=0.75, label="seen shape")
    if np.any(is_heldout):
        axes[1, 1].scatter(shape_size[is_heldout], shape_rmse[is_heldout], marker="X", s=90, label="held-out shape")
    for (s, d), x, y, h in zip(ALL_SHAPES, shape_size, shape_rmse, is_heldout):
        if h:
            axes[1, 1].annotate(f"{s}x{d}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("theta size S x D")
    axes[1, 1].set_ylabel("final RMSE")
    axes[1, 1].set_title("Difficulty versus active theta size", fontweight="bold")
    axes[1, 1].grid(alpha=0.2)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend(fontsize=8)
    fig.suptitle("Dimensional generalisation: seen versus deliberately held-out shapes", fontsize=15, fontweight="bold")
    fig.savefig(run_dir / "plots" / "dimensional_generalisation_summary.png", dpi=180)
    display(fig); plt.close(fig)

    # Benchmark-style fixed examples, now with posterior-particle uncertainty bands.
    preferred = ((1, 1), (2, 2), (3, 2), (2, 4), (4, 1), (4, 4), (1, 6), (6, 1), (3, 3), (6, 6))
    selected = []
    for shape in preferred:
        hits = np.flatnonzero((study["num_sources"] == shape[0]) & (study["source_dim"] == shape[1]))
        if len(hits):
            selected.append(int(hits[0]))
    if not selected:
        selected = list(range(min(len(study["num_sources"]), 6)))
    fig, axes = plt.subplots(len(selected), 1, figsize=(12.5, 2.45 * len(selected)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, selected):
        s = int(study["num_sources"][idx]); d = int(study["source_dim"][idx]); k = s * d
        truth = study["targets"][idx, :k]
        cloud = study["final_particles"][idx, :, :k]
        mean = cloud.mean(axis=0)
        q10, q90 = np.quantile(cloud, [0.10, 0.90], axis=0)
        x = np.arange(k)
        ax.fill_between(x, q10, q90, alpha=0.16, label="10-90% posterior particles")
        ax.plot(x, truth, marker="o", linewidth=1.3, label="true theta")
        ax.plot(x, mean, marker="x", linewidth=1.3, label="posterior mean")
        ax.set_title(
            f"S={s}, D={d}" + ("  HELD OUT" if (s, d) in HELDOUT_SHAPES else ""),
            loc="left",
            fontweight="bold",
        )
        ax.set_ylabel("theta")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("compact active theta coordinate")
    fig.suptitle(
        "Fixed dimensional-generalisation examples at the final evaluation step",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(run_dir / "plots" / "dimensional_generalisation_fixed_examples.png", dpi=180)
    display(fig); plt.close(fig)

    # How quickly evidence is used: every shape as a faint line, with seen/held-out group means.
    t = np.arange(1, study["rmse_by_t"].shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), constrained_layout=True)
    seen_curves_rmse, held_curves_rmse, seen_curves_es, held_curves_es = [], [], [], []
    for s, d in ALL_SHAPES:
        r = study["shape_rmse_by_t"][s - 1, d - 1]
        e = study["shape_energy_by_t"][s - 1, d - 1]
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(e))):
            continue
        held = (s, d) in HELDOUT_SHAPES
        axes[0].plot(t, np.maximum(r, 1e-12), alpha=0.16, linewidth=0.9, linestyle="--" if held else "-")
        axes[1].plot(t, np.maximum(e, 1e-12), alpha=0.16, linewidth=0.9, linestyle="--" if held else "-")
        (held_curves_rmse if held else seen_curves_rmse).append(r)
        (held_curves_es if held else seen_curves_es).append(e)
    if seen_curves_rmse:
        axes[0].plot(
            t,
            np.maximum(np.mean(np.asarray(seen_curves_rmse), axis=0), 1e-12),
            linewidth=2.5,
            label="seen-shape mean",
        )
    if held_curves_rmse:
        axes[0].plot(
            t,
            np.maximum(np.mean(np.asarray(held_curves_rmse), axis=0), 1e-12),
            linewidth=2.5,
            linestyle="--",
            label="held-out-shape mean",
        )
    if seen_curves_es:
        axes[1].plot(
            t,
            np.maximum(np.mean(np.asarray(seen_curves_es), axis=0), 1e-12),
            linewidth=2.5,
            label="seen-shape mean",
        )
    if held_curves_es:
        axes[1].plot(
            t,
            np.maximum(np.mean(np.asarray(held_curves_es), axis=0), 1e-12),
            linewidth=2.5,
            linestyle="--",
            label="held-out-shape mean",
        )
    for ax, title, ylabel in zip(
        axes,
        ("Posterior-mean RMSE versus evidence", "Energy score versus evidence"),
        ("RMSE", "energy score"),
    ):
        ax.set_yscale("log")
        ax.set_xlabel("sequential step t")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle("Generalisation through a longer evaluation sequence", fontsize=14, fontweight="bold")
    fig.savefig(run_dir / "plots" / "dimensional_generalisation_by_prefix.png", dpi=180)
    display(fig); plt.close(fig)

    # Source-identification gallery for the directly visualisable D=2 cases.  A best permutation
    # is used ONLY for display because physical sources are exchangeable; it never changes a score.
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.2), constrained_layout=True)
    for ax, s in zip(axes.ravel(), range(1, min(cfg.max_num_sources, 6) + 1)):
        hits = np.flatnonzero((study["num_sources"] == s) & (study["source_dim"] == 2))
        if not len(hits):
            ax.axis("off")
            continue
        idx = int(hits[0]); k = 2 * s
        truth = study["targets"][idx, :k].reshape(s, 2)
        cloud = study["final_particles"][idx, :, :k].reshape(-1, s, 2)
        mean = cloud.mean(axis=0)
        best_perm = min(itertools.permutations(range(s)), key=lambda p: float(np.sum((mean[list(p)] - truth) ** 2)))
        cloud = cloud[:, best_perm, :]
        mean = cloud.mean(axis=0)
        matched_rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
        for source_index in range(s):
            points = cloud[:, source_index]
            ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.08, color=f"C{source_index % 10}")
            covariance = np.cov(points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]; eigenvectors = eigenvectors[:, order]
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            ellipse = Ellipse(mean[source_index], 2 * np.sqrt(4.605 * eigenvalues[0]),
                              2 * np.sqrt(4.605 * eigenvalues[1]), angle=angle,
                              fill=False, linewidth=1.4, alpha=0.8, color=f"C{source_index % 10}")
            ax.add_patch(ellipse)
            ax.scatter(mean[source_index, 0], mean[source_index, 1], marker="x", s=65, color=f"C{source_index % 10}")
            ax.scatter(truth[source_index, 0], truth[source_index, 1], marker="*", s=150,
                       color=f"C{source_index % 10}", edgecolors="black", linewidths=0.6)
            ax.plot([mean[source_index, 0], truth[source_index, 0]],
                    [mean[source_index, 1], truth[source_index, 1]], linewidth=0.8,
                    alpha=0.55, color=f"C{source_index % 10}")
        ax.set_xlim(cfg.design_low, cfg.design_high); ax.set_ylim(cfg.design_low, cfg.design_high)
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_title(f"S={s}, D=2 | matched RMSE={matched_rmse:.3f}", fontweight="bold")
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
    for ax in axes.ravel()[min(cfg.max_num_sources, 6):]:
        ax.axis("off")
    fig.suptitle(
        "Source identification: truth stars, posterior means, and 90% marginal ellipses\n"
        "(best source permutation used for display only)",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(run_dir / "plots" / "source_identification_gallery.png", dpi=180)
    display(fig); plt.close(fig)


dimensional_generalisation = None
post_eval_data = None
if RUN_DIMENSIONAL_GENERALISATION_DIAGNOSTICS:
    post_eval_rng = np.random.default_rng(CFG.seed + 120_000)
    post_eval_data = simulate_trajectories(
        post_eval_rng,
        CFG.n_evaluation_trajectories_per_shape * len(ALL_SHAPES),
        CFG.evaluation_trajectory_length,
        CFG,
        shape_pool=ALL_SHAPES,
        balanced_shapes=True,
    )
    dimensional_generalisation = evaluate_dimension_generalisation(
        model, post_eval_data, CFG, seed=CFG.seed + 121_000
    )
    np.savez_compressed(
        run_dir / "artefacts" / "dimensional_generalisation.npz",
        **{name: np.asarray(value) for name, value in dimensional_generalisation.items()},
    )
    plot_dimension_generalisation(dimensional_generalisation, post_eval_data, CFG)

    seen = ~dimensional_generalisation["heldout"]
    held = dimensional_generalisation["heldout"]
    summary_parts = [
        "[dimensional generalisation]",
        f"T_eval={CFG.evaluation_trajectory_length}",
    ]
    if np.any(seen):
        summary_parts.append(
            f"seen final RMSE={np.mean(dimensional_generalisation['rmse_by_t'][seen, -1]):.5f}"
        )
        summary_parts.append(
            f"seen final ES={np.mean(dimensional_generalisation['energy_by_t'][seen, -1]):.5f}"
        )
    if np.any(held):
        summary_parts.append(
            f"held-out final RMSE={np.mean(dimensional_generalisation['rmse_by_t'][held, -1]):.5f}"
        )
        summary_parts.append(
            f"held-out final ES={np.mean(dimensional_generalisation['energy_by_t'][held, -1]):.5f}"
        )
    else:
        summary_parts.append("held-out shapes: none")
    print(" | ".join(summary_parts))
else:
    print(
        "[dimensional generalisation] skipped: fixed single-shape setup "
        "(min_num_sources=max_num_sources and min_source_dim=max_source_dim)."
    )

#%% 20) Final repeated-Bayes source-space comparison with a likelihood reference
plot_reference_comparison(
    model,
    fixed_trajectory,
    fixed_prior_particles,
    CFG,
    run_dir / "plots" / "reference_posterior_comparison.png",
)

#%% 21) Numerical architecture checks: causality and particle equivariance
def structural_checks(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, float]:
    """Numerically test identities used by both training and sequential evaluation.

    1. Direct-prefix causality: a training prediction using o observations ignores suffix o+1:O.
    2. Sequential causality: perturb future observation blocks and preserve outputs through t.
    3. Memory causality: contextual observation memories through t ignore future blocks.
    4. Particle equivariance: permute the particle axis, undo it on outputs, verify equality.

    Observation order is intentionally meaningful: designs inside each block use canonical
    norm order, while the evaluation rollout still applies distinct Bayes updates in arrival order.
    """
    obs = _ensure_observation_blocks_np(trajectory["observations"])
    observation_count = _trajectory_observation_count_np(trajectory, cfg)
    prior = np.asarray(prior_particles, dtype=np.float32)
    S, D, theta_size = _trajectory_shape(trajectory)
    rng = np.random.default_rng(cfg.seed + 500_000)
    t = max(2, len(obs) // 2)

    baseline, baseline_conditions, _ = model(
        jnp.asarray(prior), jnp.asarray(obs), jnp.asarray(observation_count),
        jnp.asarray(S), jnp.asarray(theta_size)
    )
    baseline = np.asarray(jax.device_get(baseline))
    baseline_conditions = np.asarray(jax.device_get(baseline_conditions))

    # Training-path causality: direct posterior prefixes may not depend on observations after o.
    direct_baseline, _, _, direct_counts = model.predict_prefixes(
        jnp.asarray(prior), jnp.asarray(obs[0]), jnp.asarray(S), jnp.asarray(theta_size)
    )
    direct_baseline = np.asarray(jax.device_get(direct_baseline))
    direct_counts = np.asarray(jax.device_get(direct_counts), dtype=np.int32)
    cutoff = int((cfg.min_observations_per_step + cfg.max_observations_per_step) // 2)
    direct_perturbed_obs = obs[0].copy()
    if cutoff < direct_perturbed_obs.shape[0]:
        # Preserve each suffix design norm so canonical sorting cannot move a perturbed row
        # into the protected prefix; random sign flips still change the physical design.
        direct_perturbed_obs[cutoff:, :D] *= rng.choice(
            np.asarray([-1.0, 1.0], dtype=np.float32),
            size=direct_perturbed_obs[cutoff:, :D].shape,
        )
        direct_perturbed_obs[cutoff:, -1] += rng.normal(
            0.0, 5.0, size=direct_perturbed_obs[cutoff:, -1].shape
        )
    direct_perturbed, _, _, _ = model.predict_prefixes(
        jnp.asarray(prior), jnp.asarray(direct_perturbed_obs),
        jnp.asarray(S), jnp.asarray(theta_size)
    )
    direct_perturbed = np.asarray(jax.device_get(direct_perturbed))
    protected = direct_counts <= cutoff
    direct_prefix_causality_error = (
        float(np.max(np.abs(direct_perturbed[protected] - direct_baseline[protected])))
        if np.any(protected) else 0.0
    )

    future_perturbed = obs.copy()
    if t < len(obs):
        future_perturbed[t:, :, :D] = canonicalize_designs_np(
            rng.uniform(
                cfg.design_low, cfg.design_high, size=future_perturbed[t:, :, :D].shape
            )
        )
        future_perturbed[t:, :, -1] += rng.normal(
            0.0, 5.0, size=future_perturbed[t:, :, -1].shape
        )
        future_perturbed[t:] = canonicalize_observation_block_np(future_perturbed[t:], D)
    causal_output, causal_conditions, _ = model(
        jnp.asarray(prior), jnp.asarray(future_perturbed), jnp.asarray(observation_count),
        jnp.asarray(S), jnp.asarray(theta_size)
    )
    causal_output = np.asarray(jax.device_get(causal_output))
    causal_conditions = np.asarray(jax.device_get(causal_conditions))
    causal_error = float(np.max(np.abs(causal_output[:t] - baseline[:t])))
    causal_condition_error = float(np.max(np.abs(causal_conditions[:t] - baseline_conditions[:t])))

    particle_perm = rng.permutation(len(prior)); inverse_perm = np.argsort(particle_perm)
    permuted_output, _, _ = model(
        jnp.asarray(prior[particle_perm]), jnp.asarray(obs), jnp.asarray(observation_count),
        jnp.asarray(S), jnp.asarray(theta_size)
    )
    permuted_output = np.asarray(jax.device_get(permuted_output))[:, inverse_perm]
    particle_equivariance_error = float(np.max(np.abs(permuted_output - baseline)))

    return {
        "direct_prefix_causality_max_abs_error": direct_prefix_causality_error,
        "causal_output_max_abs_error": causal_error,
        "causal_condition_max_abs_error": causal_condition_error,
        "particle_equivariance_max_abs_error": particle_equivariance_error,
    }


structure_results = structural_checks(model, fixed_trajectory, fixed_prior_particles, CFG)
print("Structural architecture checks:", structure_results)
save_json(run_dir / "artefacts" / "structural_checks.json", structure_results)


def plot_structural_checks(
    structure_results: dict[str, float],
    destination: Path | None = None,
):
    metric_names = list(structure_results.keys())
    x = np.arange(len(metric_names))
    heights = [max(structure_results[m], 1e-16) for m in metric_names]
    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)
    ax.bar(x, heights)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([
        "direct-prefix\ncausality", "sequential\nposterior causality",
        "sequential\nmemory causality", "particle\nequivariance"
    ])
    ax.set_ylabel("max absolute discrepancy")
    ax.set_title("Direct-training and sequential-evaluation identities should be near precision",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


plot_structural_checks(structure_results, run_dir / "plots" / "structural_theorem_checks.png")


#%% 22) Numerical theorem check: single-global-truth proper-score collapse
def energy_score_np(
    particles: np.ndarray,
    target: np.ndarray,
    theta_size: int,
) -> float:
    return float(
        jax.device_get(
            energy_score_single(
                jnp.asarray(particles),
                jnp.asarray(target),
                jnp.asarray(theta_size),
            )
        )
    )


def mode_b_collapse_curve(
    theta_star_padded: np.ndarray,
    num_sources: int,
    theta_size: int,
    cfg: BayesTransportConfig = CFG,
    n_particles: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Show the fixed-target proper-score collapse theorem directly in physical theta space."""
    S = int(num_sources)
    D = int(theta_size) // S
    theta_active = np.asarray(theta_star_padded)[:S, :D]
    if cfg.canonicalize_particle_sources and S > 1:
        theta_active = canonicalize_sources_np(theta_active)

    target = np.zeros((cfg.max_num_sources * cfg.max_source_dim,), dtype=np.float32)
    target[:theta_size] = theta_active.reshape(-1)

    rng = np.random.default_rng(cfg.seed + 600_000)
    base_noise = rng.normal(size=(n_particles, S, D)).astype(np.float32)
    scales = np.concatenate([[0.0], np.geomspace(1e-3, 2.0, 34)])
    scores = []
    for scale in scales:
        cloud = theta_active[None, :, :] + float(scale) * base_noise
        if cfg.canonicalize_particle_sources and S > 1:
            cloud = canonicalize_sources_np(cloud)
        compact = np.zeros(
            (n_particles, cfg.max_num_sources * cfg.max_source_dim),
            dtype=np.float32,
        )
        compact[:, :theta_size] = cloud.reshape(n_particles, theta_size)
        scores.append(energy_score_np(compact, target, theta_size))
    return scales, np.asarray(scores)


fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
S_fixed, D_fixed, theta_size_fixed = _trajectory_shape(fixed_trajectory)
collapse_scales, collapse_scores = mode_b_collapse_curve(
    fixed_trajectory["theta_true"], S_fixed, theta_size_fixed, CFG
)
ax.plot(collapse_scales, collapse_scores, marker="o", markersize=3)
ax.set_xscale("symlog", linthresh=1e-3)
ax.set_yscale("symlog", linthresh=1e-6)
ax.set_xlabel("physical cloud scale around one fixed theta*")
ax.set_ylabel("physical-theta energy score against theta*")
ax.set_title("Mode B diagnostic: a fixed physical target favors a point mass", fontweight="bold")
ax.grid(alpha=0.25)
fig.savefig(run_dir / "plots" / "mode_b_collapse_theorem.png", dpi=170)
display(fig)
plt.close(fig)

#%% 23) Limit study N -> large: particle count, energy score, and runtime
def particle_limit_study(
    model: SequentialBayesModel,
    eval_data: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, np.ndarray]:
    """Evaluate the same trained set-valued model at several particle counts.

    The architecture has no particle positional encoding and all weights are shared,
    so changing N does not change parameter shapes.  JAX will compile once per new N.
    This is an empirical finite-particle study, not a proof of a mean-field limit.
    """
    final_energy = []
    mean_energy = []
    final_rmse = []
    seconds = []
    for n_particles in cfg.particle_limit_values:
        # Warm up the shape first so the runtime curve is not dominated by the one-off
        # JAX compilation that occurs whenever N changes.
        _ = evaluate_model(
            model,
            eval_data,
            cfg,
            num_particles=n_particles,
            max_trajectories=min(cfg.batch_size, cfg.limit_eval_trajectories),
            seed=cfg.seed + 699_000,
        )
        started = time.perf_counter()
        metrics = evaluate_model(
            model,
            eval_data,
            cfg,
            num_particles=n_particles,
            max_trajectories=cfg.limit_eval_trajectories,
            seed=cfg.seed + 700_000,
        )
        # evaluate_model uses jax.device_get internally, so device work is complete here.
        seconds.append(time.perf_counter() - started)
        final_energy.append(float(metrics["final_energy_score"]))
        mean_energy.append(float(metrics["energy_score"]))
        final_rmse.append(float(metrics["final_mean_rmse"]))
    return {
        "num_particles": np.asarray(cfg.particle_limit_values, dtype=int),
        "final_energy": np.asarray(final_energy),
        "mean_energy": np.asarray(mean_energy),
        "final_rmse": np.asarray(final_rmse),
        "seconds": np.asarray(seconds),
    }


particle_study = particle_limit_study(model, eval_data, CFG)
fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), constrained_layout=True)
n = particle_study["num_particles"]
axes[0].plot(n, particle_study["final_energy"], marker="o")
axes[1].plot(n, particle_study["final_rmse"], marker="o")
axes[2].plot(n, particle_study["seconds"], marker="o")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)
axes[0].set_yscale("log")
axes[1].set_yscale("log")
axes[0].set_title("Final-step energy score")
axes[1].set_title("Final physical posterior-mean RMSE")
axes[2].set_title("Evaluation wall time")
axes[0].set_xlabel("particles N")
axes[1].set_xlabel("particles N")
axes[2].set_xlabel("particles N")
axes[2].set_ylabel("seconds")
fig.suptitle("Finite-particle limit study: physical accuracy and the O(N^2 theta_size) score cost",
             fontsize=14, fontweight="bold")
fig.savefig(run_dir / "plots" / "particle_limit_study.png", dpi=170)
display(fig)
plt.close(fig)


#%% 24) Sequential behaviour across the requested evaluation horizon
long_eval_rng = np.random.default_rng(CFG.seed + 800_000)
long_eval_data = simulate_trajectories(
    long_eval_rng,
    CFG.limit_eval_trajectories,
    CFG.evaluation_trajectory_length,
    CFG,
)

long_trajectory_study = evaluate_model(
    model,
    long_eval_data,
    CFG,
    max_trajectories=CFG.limit_eval_trajectories,
    seed=CFG.seed + 801_000,
)

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
t = np.arange(1, len(long_trajectory_study["energy_by_t"]) + 1)
axes[0].plot(t, long_trajectory_study["energy_by_t"])
axes[1].plot(t, long_trajectory_study["rmse_by_t"])
axes[2].plot(t, long_trajectory_study["spread_by_t"])
for ax in axes:
    ax.set_xlabel("sequential step t")
    ax.grid(alpha=0.25)
for ax in axes:
    ax.set_yscale("log")
axes[0].set_title("Energy score")
axes[1].set_title("Physical posterior-mean RMSE")
axes[2].set_title("Physical posterior spread")
fig.suptitle(
    f"Repeated-Bayes behaviour through requested evaluation horizon T={CFG.evaluation_trajectory_length}",
    fontsize=14,
    fontweight="bold",
)
fig.savefig(run_dir / "plots" / "trajectory_length_limit_study.png", dpi=170)
display(fig)
plt.close(fig)


#%% 25) Limit study M -> large: empirical trajectory-average convergence
def per_trajectory_final_energy(
    model: SequentialBayesModel,
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    *,
    seed: int,
) -> np.ndarray:
    """Return one final-prefix physical theta energy score per independent trajectory."""
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, len(dataset["theta_true"]), cfg.batch_size):
        stop = min(start + cfg.batch_size, len(dataset["theta_true"]))
        indices = np.arange(start, stop)
        batch_np = make_batch_np(
            dataset, indices, rng, cfg,
            observations_per_step=cfg.test_observations_per_step,
        )
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        predicted, _, _ = predict_batch(
            model,
            batch["prior_particles"],
            batch["observations"],
            batch["observation_count"],
            batch["num_sources"],
            batch["theta_size"],
        )
        targets = batch["theta_true"]
        if cfg.canonicalize_particle_sources:
            targets = jax.vmap(canonicalize_padded_sources_jax)(
                targets, batch["num_sources"]
            )
        targets = jax.vmap(compact_theta_jax)(
            targets, batch["num_sources"], batch["theta_size"]
        )
        final_posteriors = predicted[:, -1]
        batch_scores = jax.vmap(energy_score_single)(
            final_posteriors, targets, batch["theta_size"]
        )
        values.append(np.asarray(jax.device_get(batch_scores), dtype=np.float64))
    return np.concatenate(values)


mc_pool_rng = np.random.default_rng(CFG.seed + 900_000)
mc_pool_size = max(CFG.trajectory_mc_values)
mc_pool = simulate_trajectories(
    mc_pool_rng, mc_pool_size, CFG.evaluation_trajectory_length, CFG
)
scores = per_trajectory_final_energy(model, mc_pool, CFG, seed=CFG.seed + 901_000)
rng = np.random.default_rng(CFG.seed + 902_000)
scores = scores[rng.permutation(len(scores))]
means, lower, upper = [], [], []
for m in CFG.trajectory_mc_values:
    sample = scores[:m]
    mean = float(np.mean(sample))
    se = float(np.std(sample, ddof=1) / math.sqrt(m)) if m > 1 else 0.0
    means.append(mean); lower.append(mean - 1.96 * se); upper.append(mean + 1.96 * se)
trajectory_mc_study = {
    "M": np.asarray(CFG.trajectory_mc_values, dtype=int),
    "mean": np.asarray(means), "lower": np.asarray(lower), "upper": np.asarray(upper),
}

fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
ax.plot(trajectory_mc_study["M"], trajectory_mc_study["mean"], marker="o")
ax.fill_between(trajectory_mc_study["M"], trajectory_mc_study["lower"],
                trajectory_mc_study["upper"], alpha=0.16)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("independent evaluation trajectories M")
ax.set_ylabel("empirical mean final-step physical-theta energy score")
ax.set_title("M -> large: Monte Carlo estimate of population risk stabilises", fontweight="bold")
ax.grid(alpha=0.25)
fig.savefig(run_dir / "plots" / "trajectory_count_limit_study.png", dpi=170)
display(fig); plt.close(fig)

#%% 26) Finite prior-cloud stability: repeated prior draws for the SAME observations
def prior_cloud_stability_study(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, np.ndarray]:
    """How much does the FINAL PHYSICAL posterior mean move when the prior cloud is re-drawn?"""
    observations = _ensure_observation_blocks_np(trajectory["observations"])
    observation_count = _trajectory_observation_count_np(trajectory, cfg)
    S, D, theta_size = _trajectory_shape(trajectory)
    stds = []
    for n_particles in cfg.particle_limit_values:
        means = []
        for repeat in range(cfg.prior_resample_repeats):
            rng = np.random.default_rng(cfg.seed + 1_000_000 + 1000 * n_particles + repeat)
            active_prior = sample_base_prior_np(
                rng, n_particles, cfg, num_sources=S, source_dim=D
            )
            prior = pad_theta_np(active_prior, cfg)
            posterior, _, _ = model(
                jnp.asarray(prior), jnp.asarray(observations), jnp.asarray(observation_count),
                jnp.asarray(S), jnp.asarray(theta_size),
            )
            final = np.asarray(jax.device_get(posterior[-1]))
            means.append(final[:, :theta_size].mean(axis=0))
        means = np.stack(means)
        stds.append(float(np.sqrt(np.mean(np.var(means, axis=0, ddof=1)))))
    return {
        "num_particles": np.asarray(cfg.particle_limit_values, dtype=int),
        "posterior_mean_sd_across_prior_clouds": np.asarray(stds),
    }


prior_cloud_study = prior_cloud_stability_study(model, fixed_trajectory, CFG)
fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
ax.plot(prior_cloud_study["num_particles"],
        prior_cloud_study["posterior_mean_sd_across_prior_clouds"], marker="o")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("prior particles N")
ax.set_ylabel("RMS SD of physical posterior mean across fresh prior clouds")
ax.set_title("Finite-prior representation stability for fixed observed data", fontweight="bold")
ax.grid(alpha=0.25)
fig.savefig(run_dir / "plots" / "prior_cloud_stability.png", dpi=170)
display(fig); plt.close(fig)

#%% 27) Causal truncation consistency: full T versus running only the first t observation blocks
def truncation_consistency_study(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct check that future observation blocks are not needed to compute physical step-t posterior particles."""
    observations = _ensure_observation_blocks_np(trajectory["observations"])
    observation_count = _trajectory_observation_count_np(trajectory, CFG)
    S, D, theta_size = _trajectory_shape(trajectory)
    full, _, _ = model(
        jnp.asarray(prior_particles), jnp.asarray(observations), jnp.asarray(observation_count),
        jnp.asarray(S), jnp.asarray(theta_size),
    )
    full = np.asarray(jax.device_get(full))
    prefix_values = select_prefixes(len(observations), 6)
    errors = []
    for t in prefix_values:
        truncated, _, _ = model(
            jnp.asarray(prior_particles), jnp.asarray(observations[:t]),
            jnp.asarray(observation_count), jnp.asarray(S), jnp.asarray(theta_size),
        )
        truncated = np.asarray(jax.device_get(truncated))
        errors.append(float(np.max(np.abs(full[t - 1] - truncated[-1]))))
    return np.asarray(prefix_values), np.asarray(errors)


fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
t_values, errors = truncation_consistency_study(model, fixed_trajectory, fixed_prior_particles)
ax.plot(t_values, np.maximum(errors, 1e-16), marker="o")
ax.set_yscale("log"); ax.set_xlabel("sequential step t")
ax.set_ylabel("max |full-run theta_q,t - truncated-run theta_q,t|")
ax.set_title("Full lax.scan agrees with separately truncated sequential inference", fontweight="bold")
ax.grid(alpha=0.25)
fig.savefig(run_dir / "plots" / "causal_truncation_consistency.png", dpi=170)
display(fig); plt.close(fig)

#%% 28) Save limit-study arrays and final summary
for study_name, study in {
    "particle_limit": particle_study,
    "trajectory_mc": trajectory_mc_study,
    "prior_cloud_stability": prior_cloud_study,
}.items():
    np.savez_compressed(
        run_dir / "artefacts" / f"{study_name}.npz",
        **{metric_name: np.asarray(value) for metric_name, value in study.items()},
    )

if dimensional_generalisation is None:
    dimensional_generalisation_summary = {
        "executed": False,
        "reason": (
            "fixed single-shape setup "
            "(min_num_sources=max_num_sources and min_source_dim=max_source_dim)"
        ),
    }
else:
    _dim_seen = ~dimensional_generalisation["heldout"]
    _dim_held = dimensional_generalisation["heldout"]
    dimensional_generalisation_summary = {
        "executed": True,
        "seen_final_rmse": (
            float(np.mean(dimensional_generalisation["rmse_by_t"][_dim_seen, -1]))
            if np.any(_dim_seen) else None
        ),
        "heldout_final_rmse": (
            float(np.mean(dimensional_generalisation["rmse_by_t"][_dim_held, -1]))
            if np.any(_dim_held) else None
        ),
        "seen_final_energy": (
            float(np.mean(dimensional_generalisation["energy_by_t"][_dim_seen, -1]))
            if np.any(_dim_seen) else None
        ),
        "heldout_final_energy": (
            float(np.mean(dimensional_generalisation["energy_by_t"][_dim_held, -1]))
            if np.any(_dim_held) else None
        ),
    }

summary = {

    "objective": CFG.training_mode,
    "evaluation_objective": "unchanged direct empirical multivariate energy score in physical theta space",
    "training_mode": CFG.training_mode,
    "data_assumption": (
        "energy_score/drifting use one-step (prior,theta*,Y) rows, where prior is C_tau or a "
        "historical model output; DEQ uses one C_tau and theta* with a T-step sequence of fresh "
        "conditionally independent observation blocks from theta*"
    ),
    "training_data": "infinite PyTorch IterableDataset/DataLoader with NumPy-only simulator randomness",
    "fresh_joint_samples_per_nominal_epoch": CFG.n_train_trajectories,
    "training_prefixes": list(range(CFG.min_observations_per_step, CFG.max_observations_per_step + 1)),
    "all_prefixes_used_each_training_batch": True,
    "posterior_recurrence_in_training": (CFG.training_mode == "deq_fixed_point"),
    "training_fixed_point_max_iterations": CFG.training_fixed_point_max_iterations,
    "fresh_observation_block_each_training_iteration": (CFG.training_mode == "deq_fixed_point"),
    "historical_output_prior_probability": CFG.historical_output_prior_probability,
    "historical_output_buffer_capacity": CFG.historical_output_buffer_capacity,
    "sequential_evaluation": True,
    "sequential_evaluation_recurrence": "jax.lax.scan repeatedly applies the same learned transport",
    "posterior_conditioning": CFG.posterior_conditioning,
    "learning_rate_schedule": {
        "type": "validation reduce_on_plateau",
        "base_learning_rate": CFG.learning_rate,
        "factor": 0.5,
        "patience_epochs": CFG.lr_plateau_patience,
        "rtol": CFG.lr_plateau_rtol,
    },
    "synthetic_input_clouds_per_training_joint_draw": 1,
    "synthetic_input_cloud_independent_of_joint_target": False,
    "base_prior_distribution": CFG.base_prior_distribution,
    "base_prior_uniform_bounds": [CFG.design_low, CFG.design_high],
    "base_prior_gaussian_std": CFG.prior_std,
    "synthetic_prior_interpolation": "C_n=(1-tau)Z_n+tau*anchor with one shared tau~Uniform[0,1]",
    "synthetic_prior_match_probability": CFG.synthetic_prior_match_probability,
    "fixed_point_solver": CFG.fixed_point_solver,
    "fixed_point_backward_solver": CFG.fixed_point_backward_solver,
    "fixed_point_max_steps": CFG.fixed_point_max_steps,
    "fixed_point_backward_max_steps": CFG.fixed_point_backward_max_steps,
    "drifting_field": CFG.drifting_field,
    "drifting_eta": CFG.drifting_eta,
    "drifting_temperatures": list(CFG.drifting_temperatures),
    "historical_replay_observation_policy": "reuse exact stored observation block",
    "sequential_initial_prior": "fresh iid base-prior cloud (tau=0)",
    "particle_cloud_transport": "permutation-equivariant cloud-valued map with particle self-attention",
    "spread_term": "all within-cloud empirical pairs with N^2 normalization",
    "dimension_agnostic": not FIXED_SINGLE_SHAPE_SETUP,
    "fixed_shape_bypass_dimension_aggregators": FIXED_SINGLE_SHAPE_SETUP,
    "fixed_shape_learned_projection": (CFG.fixed_shape_learned_projection if FIXED_SINGLE_SHAPE_SETUP else None),
    "train_num_sources_range": [CFG.min_num_sources, CFG.max_num_sources],
    "train_source_dim_range": [CFG.min_source_dim, CFG.max_source_dim],
    "embedding_dim": CFG.embedding_dim,
    "fixed_shape_particle_input_dim": (model.particle_input_dim if FIXED_SINGLE_SHAPE_SETUP else None),
    "fixed_shape_observation_input_dim": (
        model.observation_input_dim if FIXED_SINGLE_SHAPE_SETUP else None
    ),
    "single_observation_direct_likelihood_bypass": model.single_observation_direct,
    "likelihood_transformer_active": not model.single_observation_direct,
    "observation_context_dim": model.observation_context_dim,
    "likelihood_hidden_dim": CFG.likelihood_hidden_dim,
    "likelihood_heads": CFG.likelihood_heads,
    "likelihood_mlp_ratio": CFG.likelihood_mlp_ratio,
    "likelihood_depth": CFG.likelihood_depth,
    "max_theta_size": CFG.max_num_sources * CFG.max_source_dim,
    "posterior_output": "compact physical theta; first S*D entries active",
    "theta_true_embedded_for_loss": False,
    "test_observations_per_step": CFG.test_observations_per_step,
    "evaluation_trajectory_length": CFG.evaluation_trajectory_length,
    "heldout_shapes": [list(shape) for shape in HELDOUT_SHAPES],
    "num_particles": CFG.num_particles,
    "best_epoch": (None if result["best_epoch"] is None else int(result["best_epoch"])),
    "best_val_energy_score": float(result["best_val_loss"]),
    "reload_checkpoint": result.get("reload_checkpoint"),
    "reload_checkpoint_kind": result.get("reload_checkpoint_kind"),
    "reload_nohup_log": result.get("reload_nohup_log"),
    "final_amortized_metrics": {
        key: float(value)
        for key, value in result["amortized_final_metrics"].items()
        if np.ndim(value) == 0
    },
    "final_sequential_metrics": {
        key: float(value)
        for key, value in result["final_metrics"].items()
        if np.ndim(value) == 0
    },
    "fixed_trajectory_sequential_reference": {
        "final_energy_distance_to_reference": float(sequential_reference["energy_distance_to_reference"][-1]),
        "final_learned_energy_score": float(sequential_reference["learned_energy_score"][-1]),
        "final_reference_energy_score": float(sequential_reference["reference_energy_score"][-1]),
        "final_learned_mean_rmse": float(sequential_reference["learned_mean_rmse"][-1]),
        "final_reference_mean_rmse": float(sequential_reference["reference_mean_rmse"][-1]),
        "final_reference_ess": float(sequential_reference["reference_ess"][-1]),
    },
    "dimensional_generalisation": dimensional_generalisation_summary,
}

save_json(run_dir / "artefacts" / "final_summary.json", summary)

print("\nFinal single-cloud energy-score + sequential-evaluation summary")
print(json.dumps(summary, indent=2))
print("All artefacts saved under:", run_dir)
