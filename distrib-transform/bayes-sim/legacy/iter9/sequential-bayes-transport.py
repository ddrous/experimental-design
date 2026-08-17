#%% 1) Imports, configuration, and experiment conventions
"""Dimension-agnostic Mode-A Bayesian source localisation with sequential posterior training.

This notebook-style file preserves the original Mode-A simulator, causal observation model,
and dimension-agnostic theta encoder while moving the proper-score objective back into
physical theta space.

For every simulated trajectory m we draw a NEW problem shape and a trajectory-specific
prior law pi_m.  By default pi_m is still the original Gaussian N(0, prior_std^2 I); an
optional meta-prior can instead draw a fresh Gaussian-mixture prior for each trajectory.
The simulator truth and the input prior point cloud are always independent draws from the
SAME pi_m:

    S_m ~ p(S),    D_m ~ p(D),    pi_m ~ H,    theta_m^* ~ pi_m,

hold that theta_m^* fixed while simulating the whole trajectory

    {(x_{m,t,o}, y_{m,t,o}) : o=1,...,Omax}_{t=1,...,T} ~ p(x,y | theta_m^*),

and train on every sequential posterior state.  The same theta_m^* is therefore the
proper-score target for all t inside one trajectory, but theta_m^*, S_m, and D_m are
re-drawn between trajectories.  This is the Bayes-consistent "fixed within a trajectory"
case, not the single-global-truth collapse case.

The dimensionality-agnostic input interface follows the dimension-aggregating embedder in
TAMO Figure 2.  For each design-outcome pair, x and y are first mapped by separate scalar
embedders into x- and y-tokens.  Those tokens then interact through the same dimension
Transformer, after which the transformed x and y streams receive their own learned
positional modulation, are averaged within stream, and the two stream summaries are ADDED
to form the common E-dimensional observation representation.  ThetaDimensionEmbedder
independently maps padded source coordinates theta -> R^E.

At every sequential step the simulator provides Omax candidate design-outcome pairs.  A
causal LikelihoodSequenceEmbedder contextualises the full Omax pair sequence once, with
each output token attending only to itself and earlier pairs.  Training samples ONE active
prefix length o in [Omin,Omax] when each minibatch is formed and uses that same o at EVERY
sequential step in that minibatch; validation/test-time can fix o through
test_observations_per_step.  The Posterior Transformer first self-attends across particle
tokens, then cross-attends only to the first o causal observation-memory tokens.

The posterior state is represented physically.  The current theta particles are embedded
by the shared ThetaDimensionEmbedder before each posterior update; the Posterior Transformer
then predicts a compact physical theta vector for every particle.  That physical output is
the scan carry for the next step and is re-embedded before the following update.  The output
array has static width Smax*Dmax for JAX compilation, but only the first S*D entries are active
and all remaining entries are zero.

The energy score is computed directly between those physical posterior particles and the
true theta.  theta_true is therefore never embedded for the loss.  Gradients from the same
physical-space proper score train the theta encoder, observation encoder, likelihood
Transformer, Posterior Transformer, and physical output head jointly.  No SIGReg, JEPA,
stop-gradient, EMA target, or separately trained visualisation decoder is used.

Because posterior particles are already returned in physical theta coordinates, the same
source-location plots can be generated during the main world-model training itself.

Notation used in arrays
-----------------------
B : number of trajectories in a minibatch
T : trajectory length / number of sequential posterior updates
N : number of prior/output particles
S : active number of exchangeable physical sources for one trajectory
D : active coordinate dimension of each source for one trajectory
E : fixed theta/observation embedding dimension
H : hidden dimension of the Posterior Transformer
Kmax = Smax*Dmax : static compact-theta output width used for heterogeneous JAX batches
Smax, Dmax : padding limits used to mix heterogeneous problems in one JAX minibatch
Omax : maximum number of fresh design-outcome observations available at one sequential step

theta_true             [B, Smax, Dmax]        padded; active block is [:S,:D]
theta_size             [B]                    equals S*D
num_sources            [B]                    sampled afresh by the training stream
observations           [B, T, Omax, Dmax+1]   padded design + outcome in final slot
observation_count      scalar                 active prefix length o shared by all T steps in the minibatch
prior_particles        [B, N, Smax, Dmax]     iid from the SAME pi_m as theta_true, padded
observation_conditions [B, T, Omax, E]        causal likelihood/context memory
posterior_theta         [B, T, N, Kmax]        first theta_size entries active
energy_by_t             [B, T]

Training trajectories are refreshed continuously by an infinite PyTorch IterableDataset
and DataLoader.  Every gradient step therefore receives fresh theta_true, designs, noisy
observations, and a fresh prior point cloud from the same trajectory-specific prior law.
The neural model itself still never calls the likelihood; simulation remains host-side.
Validation trajectories stay fixed on the seen (S,D) combinations for comparable learning
curves.  After training, a separate balanced evaluation covers every configured (S,D),
including combinations deliberately omitted from training, and may use a longer sequence.
The known likelihood is used again only in an OPTIONAL reference-posterior diagnostic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
import json
import math
import itertools
from pathlib import Path
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.colors import LogNorm
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
# - train_wm=True: create a new run and train the sequential world model.
# - train_wm=False: reload an existing AE-style sequential run from the current folder.
train_wm: bool = True

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})


@dataclass(frozen=True)
class BayesTransportConfig:
    """Defaults are the experiment; edit them here rather than in an override block."""

    # Reproducibility and run bookkeeping.
    env_name: str = "sequential"
    seed: int = 2030
    runs_base: str = "./runs"

    # Source-localisation simulator. `num_sources` and `source_dim` define only the
    # fixed 2-D diagnostic problem; heterogeneous training still uses the ranges below.
    num_sources: int = 2
    source_dim: int = 2
    prior_std: float = 1.0

    # Optional trajectory-level meta-prior.  OFF by default, so the default experiment is
    # exactly the original N(0, prior_std^2 I) initialization.  When enabled, every fresh
    # trajectory draws a NEW exchangeable diagonal Gaussian-mixture prior.  The number of
    # mixture components, component means/scales, and weights are all random.  Each Gaussian
    # component has a small configurable chance of becoming (near-)degenerate.
    use_meta_prior: bool = False
    meta_prior_min_components: int = 1
    meta_prior_max_components: int = 8
    meta_prior_component_mean_std: float = 1.5
    meta_prior_component_std_min: float = 0.20
    meta_prior_component_std_max: float = 2.00
    meta_prior_dirichlet_concentration: float = 1.0
    meta_prior_degenerate_probability: float = 0.01
    meta_prior_degenerate_std: float = 0.0

    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.30

    # Heterogeneous training-task distribution.  Arrays are padded to these maxima,
    # while masks ensure that inactive source/coordinate slots never enter an embedder.
    # The listed held-out combinations are NEVER sampled by the training stream; they are
    # reserved for the balanced dimensional-generalisation evaluation after training.
    min_num_sources: int = 1
    max_num_sources: int = 6
    min_source_dim: int = 1
    max_source_dim: int = 6
    heldout_shapes: tuple[tuple[int, int], ...] = ((1, 6), (6, 1), (3, 3), (6, 6))

    # TAMO-style dimension aggregation.  Every observation pair and every theta particle
    # is mapped to one fixed E-vector before posterior conditioning/transport.  The hard
    # check below guarantees max(S*D) <= E, as requested.
    embedding_dim: int = 192
    dimension_embedder_depth: int = 4
    scalar_encoder_depth: int = 4
    embedding_heads: int = 8

    # Mode-A trajectory and particle counts.  Each sequential posterior update may consume
    # between min_observations_per_step and max_observations_per_step fresh design-outcome
    # pairs.  Training samples ONE count in [Omin,Omax] when a minibatch is formed and uses that same count
    # at every sequential step; test_observations_per_step fixes it for validation, physical
    # visualisation, and deterministic diagnostics.
    trajectory_length: int = 8
    min_observations_per_step: int = 4
    max_observations_per_step: int = 8
    test_observations_per_step: int = 6
    num_particles: int = 32
    n_train_trajectories: int = 4096
    n_eval_trajectories: int = 256
    batch_size: int = 8

    # Balanced post-training dimensional-generalisation evaluation.  This is deliberately
    # separate from the fixed validation set used for model selection, so evaluation can
    # run for longer than the training horizon without changing training behaviour.
    evaluation_trajectory_length: int = 64
    n_evaluation_trajectories_per_shape: int = 16

    # Continuous host-side simulator stream.  n_train_trajectories now means the number of
    # FRESH trajectories consumed per nominal epoch; batch_size and all model hyperparameters
    # are unchanged.  num_workers=0 is notebook-safe; increase only if host simulation is the
    # bottleneck and your execution environment supports multiprocessing cleanly.
    train_dataloader_num_workers: int = 0
    train_dataloader_prefetch_factor: int = 2

    # Sequential Posterior Transformer. At each scan step particle embeddings first interact
    # through self-attention, then cross-attend to the active causal observation prefix,
    # and finally predict one compact physical theta vector per particle.
    hidden_dim: int = 256
    heads: int = 8
    mlp_ratio: int = 4
    posterior_depth: int = 6
    max_embedding_displacement: float = 6.0  # retained value; now caps physical theta displacement
    canonicalize_particle_sources: bool = False

    # Observation normalisation.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Optimisation. The proper-score term is the mean PHYSICAL-theta energy score
    # over B x T. The theta encoder is trained end-to-end through the same objective.
    epochs: int = 50
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1000.0

    # Persistence / visualisation cadence.
    save_every_epochs: int = 10
    final_plot_examples: int = 3
    grid_size: int = 180

    # Reference-posterior diagnostic only; never enters the training loss.
    reference_proposals: int = 10_000
    reference_particles: int = 2_000

    # Limit / theorem diagnostics after training.
    limit_eval_trajectories: int = 192
    particle_limit_values: tuple[int, ...] = (16, 32, 64)
    long_trajectory_length: int = 192
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
    if max_theta_size > cfg.embedding_dim:
        raise ValueError(
            f"max theta size S*D={max_theta_size} exceeds embedding_dim E={cfg.embedding_dim}. "
            "Increase embedding_dim or reduce the heterogeneous training range."
        )
    if not (cfg.min_num_sources <= cfg.num_sources <= cfg.max_num_sources):
        raise ValueError("fixed visualisation num_sources must lie inside the padded range.")
    if not (cfg.min_source_dim <= cfg.source_dim <= cfg.max_source_dim):
        raise ValueError("fixed visualisation source_dim must lie inside the padded range.")
    if cfg.embedding_dim % cfg.embedding_heads != 0:
        raise ValueError("embedding_dim must be divisible by embedding_heads.")
    if cfg.hidden_dim % cfg.heads != 0:
        raise ValueError("hidden_dim must be divisible by heads.")
    if cfg.embedding_dim % cfg.heads != 0:
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
    if cfg.meta_prior_min_components < 1:
        raise ValueError("meta_prior_min_components must be >= 1.")
    if cfg.meta_prior_max_components < cfg.meta_prior_min_components:
        raise ValueError("meta_prior_max_components must be >= meta_prior_min_components.")
    if cfg.meta_prior_component_mean_std < 0.0:
        raise ValueError("meta_prior_component_mean_std must be >= 0.")
    if cfg.meta_prior_component_std_min <= 0.0:
        raise ValueError("meta_prior_component_std_min must be > 0.")
    if cfg.meta_prior_component_std_max < cfg.meta_prior_component_std_min:
        raise ValueError("meta_prior_component_std_max must be >= meta_prior_component_std_min.")
    if cfg.meta_prior_dirichlet_concentration <= 0.0:
        raise ValueError("meta_prior_dirichlet_concentration must be > 0.")
    if not (0.0 <= cfg.meta_prior_degenerate_probability <= 1.0):
        raise ValueError("meta_prior_degenerate_probability must lie in [0, 1].")
    if cfg.meta_prior_degenerate_std < 0.0:
        raise ValueError("meta_prior_degenerate_std must be >= 0.")
    if cfg.train_dataloader_num_workers < 0:
        raise ValueError("train_dataloader_num_workers must be >= 0.")
    if cfg.train_dataloader_prefetch_factor < 1:
        raise ValueError("train_dataloader_prefetch_factor must be >= 1.")


# One active configuration only. In reload mode the architecture is reconstructed from the
# saved AE-style run configuration. Obsolete keys from older decoder/SIGReg runs are ignored.
_script_cfg = BayesTransportConfig()
if train_wm:
    CFG = _script_cfg
else:
    _reload_run_dir = Path.cwd().expanduser().resolve()
    _config_path = _reload_run_dir / "config.yaml"
    if not _config_path.is_file():
        raise FileNotFoundError(
            "With train_wm=False, run the notebook from the existing sequential run "
            f"folder itself. Could not find: {_config_path}"
        )
    with _config_path.open("r", encoding="utf-8") as handle:
        _saved_cfg_dict = yaml.safe_load(handle)
    _valid_cfg_fields = set(BayesTransportConfig.__dataclass_fields__)
    _saved_cfg_dict = {k: v for k, v in _saved_cfg_dict.items() if k in _valid_cfg_fields}
    for _tuple_field in ("particle_limit_values", "trajectory_mc_values"):
        if _tuple_field in _saved_cfg_dict:
            _saved_cfg_dict[_tuple_field] = tuple(_saved_cfg_dict[_tuple_field])
    if "heldout_shapes" in _saved_cfg_dict:
        _saved_cfg_dict["heldout_shapes"] = tuple(
            tuple(map(int, shape)) for shape in _saved_cfg_dict["heldout_shapes"]
        )
    CFG = BayesTransportConfig(**_saved_cfg_dict)


validate_config(CFG)

ALL_SHAPES = tuple(
    (s, d)
    for s in range(CFG.min_num_sources, CFG.max_num_sources + 1)
    for d in range(CFG.min_source_dim, CFG.max_source_dim + 1)
)
HELDOUT_SHAPES = tuple(shape for shape in CFG.heldout_shapes if shape in ALL_SHAPES)
TRAIN_SHAPES = tuple(shape for shape in ALL_SHAPES if shape not in HELDOUT_SHAPES)

#%% 2) Run directories and small persistence helpers
def make_run_dir(env_name: str, base: str | Path = "./runs") -> Path:
    """Create runs/<name>_<timestamp>/{plots,artefacts}."""
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = Path(base).expanduser().resolve() / f"{env_name}_{stamp}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=False)
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
    return run_dir


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

#%% 3) Prior and source-localisation simulator
PRIOR_SPEC_KEYS = (
    "prior_num_components",
    "prior_weights",
    "prior_component_means",
    "prior_component_stds",
)


def _default_prior_spec_np(
    cfg: BayesTransportConfig,
    *,
    source_dim: int,
) -> dict[str, np.ndarray]:
    """Return the original single-Gaussian prior in the padded prior-spec representation."""
    D = int(source_dim)
    Kmax = int(cfg.meta_prior_max_components)
    weights = np.zeros((Kmax,), dtype=np.float32)
    means = np.zeros((Kmax, cfg.max_source_dim), dtype=np.float32)
    stds = np.zeros((Kmax, cfg.max_source_dim), dtype=np.float32)
    weights[0] = 1.0
    stds[0, :D] = float(cfg.prior_std)
    return {
        "prior_num_components": np.asarray(1, dtype=np.int32),
        "prior_weights": weights,
        "prior_component_means": means,
        "prior_component_stds": stds,
    }


def sample_prior_spec_np(
    rng: np.random.Generator,
    cfg: BayesTransportConfig = CFG,
    *,
    source_dim: int,
    use_meta_prior: bool | None = None,
) -> dict[str, np.ndarray]:
    """Draw one trajectory-specific prior law pi_m.

    Default mode (use_meta_prior=False) is exactly the original isotropic Gaussian

        pi_m = N(0, prior_std^2 I).

    Optional meta-prior mode draws K at random and returns an exchangeable diagonal
    Gaussian mixture.  Component means/stds are D-vectors shared across exchangeable
    source rows, so each full theta component remains source-label symmetric.  A single
    mixture component is chosen per full theta draw; conditional on that component, the
    S source rows are iid Gaussian with the same D-dimensional parameters.

    `meta_prior_degenerate_probability` is applied independently to mixture components.
    For a selected component all active standard deviations are set to
    `meta_prior_degenerate_std`; setting that value to 0.0 gives an exactly degenerate
    Gaussian component (a point mass in each source coordinate).
    """
    D = int(source_dim)
    if D < 1 or D > cfg.max_source_dim:
        raise ValueError("source_dim is outside configured padding limits.")
    use_meta = cfg.use_meta_prior if use_meta_prior is None else bool(use_meta_prior)
    if not use_meta:
        return _default_prior_spec_np(cfg, source_dim=D)

    K = int(rng.integers(cfg.meta_prior_min_components, cfg.meta_prior_max_components + 1))
    Kmax = int(cfg.meta_prior_max_components)
    weights = np.zeros((Kmax,), dtype=np.float32)
    means = np.zeros((Kmax, cfg.max_source_dim), dtype=np.float32)
    stds = np.zeros((Kmax, cfg.max_source_dim), dtype=np.float32)

    alpha = np.full((K,), cfg.meta_prior_dirichlet_concentration, dtype=np.float64)
    weights[:K] = rng.dirichlet(alpha).astype(np.float32)
    means[:K, :D] = rng.normal(
        0.0,
        cfg.meta_prior_component_mean_std,
        size=(K, D),
    ).astype(np.float32)
    log_std_min = math.log(cfg.meta_prior_component_std_min)
    log_std_max = math.log(cfg.meta_prior_component_std_max)
    stds[:K, :D] = np.exp(
        rng.uniform(log_std_min, log_std_max, size=(K, D))
    ).astype(np.float32)

    degenerate = rng.random(K) < cfg.meta_prior_degenerate_probability
    if np.any(degenerate):
        stds[np.flatnonzero(degenerate), :D] = float(cfg.meta_prior_degenerate_std)

    return {
        "prior_num_components": np.asarray(K, dtype=np.int32),
        "prior_weights": weights,
        "prior_component_means": means,
        "prior_component_stds": stds,
    }


def sample_prior_np(
    rng: np.random.Generator,
    n: int,
    cfg: BayesTransportConfig = CFG,
    *,
    num_sources: int | None = None,
    source_dim: int | None = None,
    prior_spec: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Draw n independent theta samples in ACTIVE, unpadded shape.

    With `prior_spec=None` this preserves the original helper semantics and draws from the
    fixed Gaussian N(0, prior_std^2 I).  Training/evaluation code that needs a trajectory-
    specific meta-prior passes the explicit stored `prior_spec`, ensuring theta_true and the
    input point cloud are independent draws from the SAME prior law.
    """
    S = cfg.num_sources if num_sources is None else int(num_sources)
    D = cfg.source_dim if source_dim is None else int(source_dim)
    if S * D > cfg.embedding_dim:
        raise ValueError(f"theta size {S*D} exceeds embedding_dim={cfg.embedding_dim}.")
    if prior_spec is None:
        return rng.normal(0.0, cfg.prior_std, size=(int(n), S, D)).astype(np.float32)

    K = int(np.asarray(prior_spec["prior_num_components"]).item())
    if K < 1 or K > cfg.meta_prior_max_components:
        raise ValueError("Invalid number of components in prior_spec.")
    weights = np.asarray(prior_spec["prior_weights"], dtype=np.float64)[:K]
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Prior mixture weights must have positive finite sum.")
    weights = weights / weight_sum
    means = np.asarray(prior_spec["prior_component_means"], dtype=np.float32)[:K, :D]
    stds = np.asarray(prior_spec["prior_component_stds"], dtype=np.float32)[:K, :D]
    if np.any(stds < 0.0):
        raise ValueError("Prior component standard deviations must be non-negative.")

    # One mixture component per full theta draw.  Source rows are iid conditional on that
    # component, preserving exchangeability under source-label permutations.
    component = rng.choice(K, size=int(n), replace=True, p=weights)
    selected_means = means[component][:, None, :]                         # [n,1,D]
    selected_stds = stds[component][:, None, :]                           # [n,1,D]
    noise = rng.normal(size=(int(n), S, D)).astype(np.float32)
    return (selected_means + selected_stds * noise).astype(np.float32)


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
    if np.any(theta_size > cfg.embedding_dim):
        raise ValueError("Sampled theta_size exceeds embedding_dim; validate CFG ranges.")
    return num_sources.astype(np.int32), theta_size


def _prior_spec_from_dataset_row(
    dataset: dict[str, np.ndarray],
    index: int,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, np.ndarray]:
    """Recover the stored prior law for one trajectory; old files fall back to Gaussian."""
    S = int(dataset["num_sources"][index])
    D = int(dataset["theta_size"][index] // S)
    if not all(key in dataset for key in PRIOR_SPEC_KEYS):
        return _default_prior_spec_np(cfg, source_dim=D)
    return {
        "prior_num_components": np.asarray(dataset["prior_num_components"][index], dtype=np.int32),
        "prior_weights": np.asarray(dataset["prior_weights"][index], dtype=np.float32),
        "prior_component_means": np.asarray(dataset["prior_component_means"][index], dtype=np.float32),
        "prior_component_stds": np.asarray(dataset["prior_component_stds"][index], dtype=np.float32),
    }


def _prior_spec_from_trajectory(
    trajectory: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, np.ndarray]:
    """Recover a fixed trajectory's prior law for matched diagnostics and resampling."""
    S = int(np.asarray(trajectory["num_sources"]).item())
    D = int(np.asarray(trajectory["theta_size"]).item()) // S
    if not all(key in trajectory for key in PRIOR_SPEC_KEYS):
        return _default_prior_spec_np(cfg, source_dim=D)
    return {
        "prior_num_components": np.asarray(trajectory["prior_num_components"], dtype=np.int32),
        "prior_weights": np.asarray(trajectory["prior_weights"], dtype=np.float32),
        "prior_component_means": np.asarray(trajectory["prior_component_means"], dtype=np.float32),
        "prior_component_stds": np.asarray(trajectory["prior_component_stds"], dtype=np.float32),
    }


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
    """Generate complete heterogeneous Mode-A trajectories.

    Critical sampling provenance
    ----------------------------
    1. Each row m receives its own S_m and D_m and its own prior law pi_m.  With the
       default configuration pi_m is always the original N(0, prior_std^2 I).  With
       `use_meta_prior=True`, pi_m is a fresh random Gaussian mixture.
    2. theta_true[m] is drawn ONCE from pi_m and all T sensor readings in row m are
       simulated conditional on that SAME theta.
    3. The next row draws a fresh problem shape, prior law, and theta_true.
    4. The prior-law parameters are stored with the trajectory.  Whenever an input prior
       cloud is needed, it is drawn independently from that SAME pi_m.

    Validation and diagnostics may precompute this dictionary.  Main training instead uses
    the infinite PyTorch stream below, so these simulator draws are refreshed every step.

    Padded storage
    --------------
    theta_true[m]          [Smax,Dmax], active block [:S_m,:D_m]
    observations[m,t,o]    [Dmax+1], Omax candidate pairs per sequential step; design in
                              [:D_m], scalar y in the FINAL slot
    prior_weights[m]       [Kmax]
    prior_component_means  [M,Kmax,Dmax]
    prior_component_stds   [M,Kmax,Dmax]
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
    prior_num_components = np.zeros((n_trajectories,), dtype=np.int32)
    prior_weights = np.zeros(
        (n_trajectories, cfg.meta_prior_max_components), dtype=np.float32
    )
    prior_component_means = np.zeros(
        (n_trajectories, cfg.meta_prior_max_components, cfg.max_source_dim), dtype=np.float32
    )
    prior_component_stds = np.zeros_like(prior_component_means)

    # Simulation remains a host-side step.  In training this function is called continuously
    # by the IterableDataset; the neural/JAX loss itself never evaluates the physical likelihood.
    for m in range(n_trajectories):
        S = int(num_sources[m])
        D = int(theta_size[m] // num_sources[m])
        prior_spec = sample_prior_spec_np(rng, cfg, source_dim=D)
        theta_active = sample_prior_np(
            rng, 1, cfg, num_sources=S, source_dim=D, prior_spec=prior_spec
        )[0]
        designs = rng.uniform(
            cfg.design_low,
            cfg.design_high,
            size=(trajectory_length, cfg.max_observations_per_step, D),
        ).astype(np.float32)
        mean = source_log_mean_np(theta_active, designs, cfg)
        readings = (
            mean + cfg.observation_noise_std * rng.normal(size=mean.shape)
        ).astype(np.float32)

        theta_true[m, :S, :D] = theta_active
        observations[m, :, :, :D] = designs
        observations[m, :, :, -1] = readings
        prior_num_components[m] = prior_spec["prior_num_components"]
        prior_weights[m] = prior_spec["prior_weights"]
        prior_component_means[m] = prior_spec["prior_component_means"]
        prior_component_stds[m] = prior_spec["prior_component_stds"]

    return {
        "theta_true": theta_true,
        "observations": observations,
        "num_sources": num_sources.astype(np.int32),
        "theta_size": theta_size.astype(np.int32),
        "prior_num_components": prior_num_components,
        "prior_weights": prior_weights,
        "prior_component_means": prior_component_means,
        "prior_component_stds": prior_component_stds,
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
    """Create one heterogeneous minibatch with matched independent prior clouds.

    For each trajectory b, theta_true[b] and prior_particles[b] are independent samples
    from the SAME stored trajectory-level prior pi_b.  This remains true in both the default
    single-Gaussian experiment and the optional Gaussian-mixture meta-prior experiment.

    Every stored sequential step contains Omax candidate observations.  With
    `observations_per_step=None`, this minibatch samples ONE integer o in [Omin,Omax] and uses
    that same prefix length for every sequential step and every trajectory in the batch.
    Passing an integer fixes o, which is used for reproducible validation/test-time and
    physical posterior visualisations.
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
    for b, (dataset_index, S_value, theta_size_value) in enumerate(
        zip(indices, batch_num_sources, batch_theta_size)
    ):
        S = int(S_value)
        theta_size_int = int(theta_size_value)
        if theta_size_int > cfg.embedding_dim or theta_size_int % S != 0:
            raise ValueError("Invalid theta metadata in dataset.")
        D = theta_size_int // S
        prior_spec = _prior_spec_from_dataset_row(dataset, int(dataset_index), cfg)
        active = sample_prior_np(
            rng,
            n_particles,
            cfg,
            num_sources=S,
            source_dim=D,
            prior_spec=prior_spec,
        )
        prior_particles[b] = pad_theta_np(active, cfg)

    # One observation count is chosen for the WHOLE minibatch.  Every sequential step in
    # this batch consumes that same number of fresh observations; only the actual pairs
    # differ with t.  This trains the same model across different downstream evidence sizes
    # while keeping one forward scan internally homogeneous.
    if observations_per_step is None:
        observation_count = np.asarray(
            rng.integers(cfg.min_observations_per_step, cfg.max_observations_per_step + 1), dtype=np.int32
        )
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


class ContinuousTrajectoryDataset(IterableDataset):
    """Infinite stream of fresh Mode-A trajectories and their matched prior point clouds.

    No training trajectory is stored or revisited by design.  Each yielded item contains a
    freshly sampled problem shape, prior law, theta_true, sensor design/noise trajectory, and
    an independent N-particle cloud from that exact same prior law.  PyTorch only orchestrates
    CPU-side data generation/batching; the returned arrays are NumPy and are converted directly
    to JAX arrays by the existing training loop.
    """

    def __init__(self, cfg: BayesTransportConfig, *, seed: int):
        super().__init__()
        self.cfg = cfg
        self.seed = int(seed)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rng = np.random.default_rng(self.seed + 1_000_003 * worker_id)
        while True:
            trajectory = simulate_trajectories(
                rng,
                1,
                self.cfg.trajectory_length,
                self.cfg,
            )
            item = make_batch_np(
                trajectory,
                np.asarray([0], dtype=np.int64),
                rng,
                self.cfg,
                observations_per_step=self.cfg.min_observations_per_step,  # training chooses the real batch-level count below
            )
            yield {
                name: np.asarray(value[0])
                for name, value in item.items()
                if name != "observation_count"
            }


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
    """Build the infinite PyTorch DataLoader used by the JAX training loop."""
    dataset = ContinuousTrajectoryDataset(cfg, seed=seed)
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



#%% 6) Causal likelihood/context Transformer and cross-attention particle block
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
    """Causally contextualise all Omax pair embeddings once before posterior cross-attention.

    Output token o can depend only on pair embeddings 0,...,o.  Consequently a later
    posterior update can use the first `observation_count` outputs without any information
    leaking from the precomputed but unused suffix.
    """

    blocks: tuple[CausalObservationBlock, ...]
    final_norm: eqx.nn.LayerNorm

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.dimension_embedder_depth + 1)
        self.blocks = tuple(
            CausalObservationBlock(
                cfg.embedding_dim,
                cfg.embedding_heads,
                cfg.mlp_ratio * cfg.embedding_dim,
                key=keys[i],
            )
            for i in range(cfg.dimension_embedder_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.embedding_dim)

    def __call__(self, pair_embeddings: Array) -> Array:
        tokens = pair_embeddings
        for block in self.blocks:
            tokens = block(tokens)
        return _layernorm_tokens(self.final_norm, tokens)


class CrossAttentionParticleBlock(eqx.Module):
    """Particle self-attention followed by cross-attention to observation-memory tokens."""

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
        # Self-attention lets the particle cloud internally represent its empirical
        # distribution before any likelihood/context information is injected.
        h = _layernorm_tokens(self.self_norm, particles)
        particles = particles + self.self_attention(h, h, h)

        # Cross-attention is the only observation-to-particle interaction.  The caller
        # passes the exact active memory slice memory[:o], so no unused suffix participates.
        q = _layernorm_tokens(self.cross_query_norm, particles)
        memory = _layernorm_tokens(self.memory_norm, observation_memory)
        particles = particles + self.cross_attention(q, memory, memory)

        h = _layernorm_tokens(self.ff_norm, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + h


#%% 7) One-step physical-theta output head and sequential cross-attention posterior update
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
    compact_index = jnp.clip(source_index * source_dim + coordinate_index, 0, compact_theta.shape[0] - 1)
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

        # Identity transport remains the natural initialization: before learning, every
        # posterior update returns the current physical theta particles unchanged.
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
    """One posterior update: embedded current particles + observations -> physical theta."""

    particle_in: eqx.nn.Linear
    blocks: tuple[CrossAttentionParticleBlock, ...]
    output_head: ThetaParticleOutputHead

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 2)
        self.particle_in = eqx.nn.Linear(cfg.embedding_dim, cfg.hidden_dim, key=keys[0])
        self.blocks = tuple(
            CrossAttentionParticleBlock(
                cfg.hidden_dim,
                cfg.embedding_dim,
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
        observation_memory: Array,
        observation_count: Array,
        theta_size: Array,
    ) -> Array:
        # The count is shared across the minibatch and every scan step. lax.switch gives each
        # branch a static memory-prefix shape, so unused Omax-o tokens never enter cross-attention.
        count = jnp.clip(observation_count, 1, observation_memory.shape[0]).astype(jnp.int32)

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
            for prefix_length in range(1, observation_memory.shape[0] + 1)
        )
        return jax.lax.switch(
            count - 1,
            branches,
            (current_embeddings, current_theta, observation_memory),
        )

#%% 8) End-to-end dimension-agnostic sequential model
class SequentialBayesModel(eqx.Module):
    """Observation embedders + theta embedder + physical-output posterior Transformer."""

    observation_embedder: ObservationDimensionEmbedder
    likelihood_embedder: LikelihoodSequenceEmbedder
    theta_embedder: ThetaDimensionEmbedder
    posterior_transformer: CrossAttentionPosteriorTransformer

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        observation_key, likelihood_key, theta_key, posterior_key = jax.random.split(key, 4)
        self.observation_embedder = ObservationDimensionEmbedder(cfg, key=observation_key)
        self.likelihood_embedder = LikelihoodSequenceEmbedder(cfg, key=likelihood_key)
        self.theta_embedder = ThetaDimensionEmbedder(cfg, key=theta_key)
        self.posterior_transformer = CrossAttentionPosteriorTransformer(cfg, key=posterior_key)

    def __call__(
        self,
        prior_particles: Array,      # [N,Smax,Dmax]
        observations: Array,         # [T,Omax,Dmax+1]
        observation_count: Array,    # scalar, same active prefix length at every scan step
        num_sources: Array,          # scalar S
        theta_size: Array,           # scalar S*D
    ) -> tuple[Array, Array, Array]:
        # The theta embedder canonicalizes exchangeable source rows. The physical scan carry
        # must use the same ordering, otherwise an embedding would refer to a different source
        # labelling than the residual physical output head.
        if self.theta_embedder.canonicalize:
            prior_particles = jax.vmap(
                lambda theta: canonicalize_padded_sources_jax(theta, num_sources)
            )(prior_particles)

        prior_theta = jax.vmap(
            lambda theta: compact_theta_jax(theta, num_sources, theta_size)
        )(prior_particles)                                               # [N,Kmax]

        # Embed every candidate observation pair independently, then causally contextualise
        # each Omax block once. The first o outputs cannot depend on the unused suffix.
        pair_embeddings = jax.vmap(
            lambda observation_block: jax.vmap(
                lambda observation: self.observation_embedder(
                    observation, num_sources, theta_size
                )
            )(observation_block)
        )(observations)                                                   # [T,Omax,E]
        observation_contexts = jax.vmap(self.likelihood_embedder)(
            pair_embeddings
        )                                                                 # [T,Omax,E]

        # The scan carry is now the physical compact theta cloud. Before each update it is
        # converted back to padded [Smax,Dmax] form and passed through the SAME trainable
        # dimension-agnostic theta encoder. The Posterior Transformer directly returns the
        # next physical theta cloud, which is used both by the loss and by the next scan step.
        def scan_step(current_theta: Array, observation_memory: Array):
            current_padded = jax.vmap(
                lambda theta: padded_theta_jax(
                    theta,
                    num_sources,
                    theta_size,
                    prior_particles.shape[-2],
                    prior_particles.shape[-1],
                )
            )(current_theta)
            current_embeddings = jax.vmap(
                lambda theta: self.theta_embedder(theta, num_sources, theta_size)
            )(current_padded)
            next_theta = self.posterior_transformer(
                current_embeddings,
                current_theta,
                observation_memory,
                observation_count,
                theta_size,
            )                                                             # [N,Kmax]

            # Keep the physical carry in canonical source order as well. Without this,
            # the encoder could re-sort a predicted cloud while the residual output head
            # still receives the old source ordering on its physical skip connection.
            if self.theta_embedder.canonicalize:
                next_padded = jax.vmap(
                    lambda theta: padded_theta_jax(
                        theta,
                        num_sources,
                        theta_size,
                        prior_particles.shape[-2],
                        prior_particles.shape[-1],
                    )
                )(next_theta)
                next_padded = jax.vmap(
                    lambda theta: canonicalize_padded_sources_jax(theta, num_sources)
                )(next_padded)
                next_theta = jax.vmap(
                    lambda theta: compact_theta_jax(theta, num_sources, theta_size)
                )(next_padded)

            return next_theta, next_theta

        _, posterior_sequence = jax.lax.scan(
            scan_step, prior_theta, observation_contexts
        )
        return posterior_sequence, observation_contexts, prior_theta      # [T,N,Kmax], [T,Omax,E], [N,Kmax]


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

    print(f"Total parameters: {total / 1e6:.3f} M")
    print(f"  Design-Outcome embedder : {observation_embedder:,}")
    print(f"  Likelihood Transformer  : {likelihood_embedder:,}")
    print(f"  Theta embedder          : {theta_embedder:,}")
    print(f"  Posterior Transformer   : {posterior:,}")

#%% 9) Physical-theta energy score and simple posterior diagnostics
def energy_score_single(
    particle_theta: Array,
    target_theta: Array,
    theta_size: Array,
) -> Array:
    """Exact empirical multivariate energy score on the active physical theta coordinates.

    For q^N = N^{-1} sum_n delta_{theta_n},

        ES(q^N, theta*)
          = N^{-1} sum_n ||theta_n-theta*||
            - (2 N^2)^{-1} sum_{n,m} ||theta_n-theta_m||.

    Arrays have static width Kmax for heterogeneous JAX batches, but only the first S*D
    coordinates enter either distance.
    """
    valid = (jnp.arange(particle_theta.shape[-1]) < theta_size).astype(particle_theta.dtype)
    truth_distance = jnp.mean(
        jnp.sqrt(
            jnp.sum((particle_theta - target_theta[None, :]) ** 2 * valid[None, :], axis=-1)
            + 1e-12
        )
    )
    differences = particle_theta[:, None, :] - particle_theta[None, :, :]
    squared_distance = jnp.sum(differences**2 * valid[None, None, :], axis=-1)
    off_diagonal = 1.0 - jnp.eye(particle_theta.shape[0], dtype=particle_theta.dtype)
    pairwise_distance = jnp.sum(
        jnp.sqrt(squared_distance + 1e-12) * off_diagonal
    ) / (particle_theta.shape[0] ** 2)
    return truth_distance - 0.5 * pairwise_distance


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


def _trajectory_metrics(
    posterior_sequence: Array,
    target_theta: Array,
    theta_size: Array,
) -> tuple[Array, Array, Array]:
    """Vectorise all per-prefix physical posterior metrics over T."""
    energy = jax.vmap(lambda p: energy_score_single(p, target_theta, theta_size))(posterior_sequence)
    rmse = jax.vmap(lambda p: posterior_mean_rmse_single(p, target_theta, theta_size))(posterior_sequence)
    spread = jax.vmap(lambda p: posterior_spread_single(p, theta_size))(posterior_sequence)
    return energy, rmse, spread


def batch_objective(
    model: SequentialBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig = CFG,
) -> tuple[Array, dict[str, Array]]:
    """Mean Mode-A physical-theta energy score over B x T.

    `predicted` has shape [B,T,N,Kmax]. All sequential losses come from one lax.scan forward
    pass and one gradient call. The true theta is used directly in physical space; it is not
    embedded. Observation embedding, theta embedding, and posterior prediction are all
    differentiated jointly through this same proper-score objective.
    """
    predicted, _, _ = jax.vmap(
        model, in_axes=(0, 0, None, 0, 0)
    )(
        batch["prior_particles"],
        batch["observations"],
        batch["observation_count"],
        batch["num_sources"],
        batch["theta_size"],
    )

    target_theta = batch["theta_true"]
    if cfg.canonicalize_particle_sources:
        target_theta = jax.vmap(canonicalize_padded_sources_jax)(
            target_theta, batch["num_sources"]
        )
    target_theta = jax.vmap(compact_theta_jax)(
        target_theta, batch["num_sources"], batch["theta_size"]
    )                                                                  # [B,Kmax]

    energy, rmse, spread = jax.vmap(_trajectory_metrics)(
        predicted, target_theta, batch["theta_size"]
    )                                                                  # each [B,T]

    loss = jnp.mean(energy)
    metrics = {
        "loss": loss,
        "energy_score": loss,
        "final_energy_score": jnp.mean(energy[:, -1]),
        "posterior_mean_rmse": jnp.mean(rmse),
        "final_mean_rmse": jnp.mean(rmse[:, -1]),
        "posterior_spread": jnp.mean(spread),
        "final_spread": jnp.mean(spread[:, -1]),
        "energy_by_t": jnp.mean(energy, axis=0),
        "rmse_by_t": jnp.mean(rmse, axis=0),
        "spread_by_t": jnp.mean(spread, axis=0),
    }
    return loss, metrics


@eqx.filter_jit
def predict_batch(
    model: SequentialBayesModel,
    prior_particles: Array,
    observations: Array,
    observation_count: Array,
    num_sources: Array,
    theta_size: Array,
) -> tuple[Array, Array, Array]:
    """JIT-compiled trajectory batching; sequential time remains inside lax.scan."""
    return jax.vmap(model, in_axes=(0, 0, None, 0, 0))(
        prior_particles, observations, observation_count, num_sources, theta_size
    )


@eqx.filter_jit
def evaluation_batch(
    model: SequentialBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Array]:
    _, metrics = batch_objective(model, batch, cfg)
    return metrics

#%% 10) Evaluation helper with reproducible fresh prior clouds
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
    """Evaluate with fresh reproducible prior clouds."""
    n_total = len(dataset["theta_true"])
    if max_trajectories is not None:
        n_total = min(n_total, int(max_trajectories))
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    eval_seed = cfg.seed + 90_000 if seed is None else int(seed)
    rng = np.random.default_rng(eval_seed)

    scalar_names = [
        "loss",
        "energy_score",
        "final_energy_score",
        "posterior_mean_rmse",
        "final_mean_rmse",
        "posterior_spread",
        "final_spread",
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
        host = jax.device_get(evaluation_batch(model, batch, cfg))
        weight = stop - start
        weights.append(weight)
        for name in scalar_names:
            scalar_values[name].append(float(host[name]))
        for name in by_t_values:
            by_t_values[name].append(np.asarray(host[name], dtype=np.float64))

    weights = np.asarray(weights, dtype=np.float64)
    result: dict[str, np.ndarray | float] = {}
    for name, values in scalar_values.items():
        result[name] = float(np.average(np.asarray(values), weights=weights))
    for name, values in by_t_values.items():
        stacked = np.stack(values, axis=0)
        result[name] = np.average(stacked, axis=0, weights=weights)
    return result

#%% 11) Optional exact-likelihood reference posterior for plots only
def reference_posterior_particles_np(
    rng: np.random.Generator,
    observations: np.ndarray,
    prefix_length: int,
    num_sources: int,
    theta_size: int,
    cfg: BayesTransportConfig = CFG,
    *,
    prior_spec: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, float]:
    """SNIS reference posterior used only after training for visual validation.

    Proposal is exactly the trajectory's prior pi_m when `prior_spec` is supplied (or the
    original Gaussian prior when it is omitted), so importance weights are proportional to
    the likelihood of the observed prefix.  This function is intentionally NOT a teacher
    and is never called inside the training objective.
    """
    prefix_length = int(prefix_length)
    S = int(num_sources)
    D = int(theta_size) // S
    proposals = sample_prior_np(
        rng, cfg.reference_proposals, cfg, num_sources=S, source_dim=D, prior_spec=prior_spec
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
    """Visual map of the sequential cross-attention architecture with physical theta outputs."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 5.2), constrained_layout=True)

    def draw_box(xy, width, height, text, title=None):
        patch = FancyBboxPatch(
            xy, width, height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.3, facecolor="white", edgecolor="black",
        )
        ax.add_patch(patch)
        label = text if title is None else f"{title}\n{text}"
        ax.text(xy[0] + width / 2, xy[1] + height / 2, label,
                ha="center", va="center", fontsize=9)

    def arrow(start, end, text="", connectionstyle=None):
        kwargs = {}
        if connectionstyle is not None:
            kwargs["connectionstyle"] = connectionstyle
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.25, **kwargs))
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035,
                    text, ha="center", va="bottom", fontsize=8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(
        "Sequential cross-attention posterior conditioning — recurrence compiled with jax.lax.scan",
        loc="left", fontweight="bold",
    )

    draw_box(
        (0.01, 0.60), 0.14, 0.24,
        f"Omax={cfg.max_observations_per_step} candidate pairs\none batch-level prefix o used at every step",
        "scan input",
    )
    draw_box(
        (0.19, 0.58), 0.17, 0.28,
        f"separate x/y scalar embedders\nshared dimension Transformer; avg + add\ncausal likelihood Transformer; Omax x E={cfg.embedding_dim}",
        "observation memory",
    )
    draw_box((0.01, 0.12), 0.14, 0.22, "N iid padded draws\nvariable S,D", "prior cloud")
    draw_box((0.19, 0.09), 0.17, 0.28,
             f"canonicalize physical theta\ncompact S*D scalars\ndimension Transformer\n-> E={cfg.embedding_dim}",
             "theta embedder")
    draw_box((0.48, 0.22), 0.22, 0.42,
             "particle self-attention\ncross-attention to memory[:o]\nphysical theta output head",
             "Posterior Transformer")
    draw_box((0.82, 0.30), 0.16, 0.26,
             f"[N,Kmax] at step t\nKmax={cfg.max_num_sources * cfg.max_source_dim}; first S*D active", "posterior theta")

    arrow((0.15, 0.72), (0.19, 0.72))
    arrow((0.36, 0.72), (0.48, 0.55), "cross-attention memory")
    arrow((0.15, 0.23), (0.19, 0.23))
    arrow((0.36, 0.23), (0.48, 0.35), "initial scan carry")
    arrow((0.70, 0.43), (0.82, 0.43), "theta_t")
    arrow((0.90, 0.30), (0.28, 0.09), "physical carry re-embedded at t+1", connectionstyle="arc3,rad=0.30")

    fig.suptitle(
        "Dimension-agnostic Mode A: E-space conditioning with direct physical-theta posterior output",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
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
    labels = ["physical prior"] + [f"q_phi(theta | steps 1:{t})" for t in prefixes]
    all_points = np.concatenate([c.reshape(-1, 2) for c in clouds] + [theta_true.reshape(-1, 2)])
    lim = max(3.0 * cfg.prior_std, 1.12 * float(np.quantile(np.abs(all_points), 0.995)))

    for panel_index, (ax, cloud, label) in enumerate(zip(axes, clouds, labels)):
        ax.scatter(
            cloud[..., 0].reshape(-1),
            cloud[..., 1].reshape(-1),
            s=13,
            alpha=0.30,
            label="posterior source locations" if panel_index else "prior source locations",
        )
        ax.scatter(
            theta_true[:, 0],
            theta_true[:, 1],
            marker="*",
            s=190,
            edgecolors="black",
            linewidths=0.8,
            label="theta*",
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
                s=33,
                alpha=0.65,
                label="designs seen",
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
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
    prior_spec = _prior_spec_from_trajectory(trajectory, cfg)
    reference, ess = reference_posterior_particles_np(
        rng,
        used_observations,
        len(used_observations),
        S,
        theta_size,
        cfg,
        prior_spec=prior_spec,
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
    column_names = ["sequential cross-attention", f"reference SNIS\nESS={ess:.0f}"]
    column_clouds = [learned, reference]
    lim_points = np.concatenate([cloud.reshape(-1, D) for cloud in column_clouds])
    lim = max(3.0 * cfg.prior_std, 1.1 * float(np.quantile(np.abs(lim_points), 0.995)))

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
        "Final sequential posterior source marginals versus likelihood-based reference",
        fontsize=14,
        fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)

#%% 16) Training diagnostics visualisation
def plot_training_diagnostics(
    history: dict[str, list],
    best_epoch: int,
    destination: Path | None = None,
):
    steps = np.arange(1, len(history["step_loss"]) + 1)
    epochs = np.arange(1, len(history["epoch_train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0), constrained_layout=True)

    values = np.maximum(np.asarray(history["step_loss"], dtype=float), 1e-12)
    axes[0, 0].plot(steps, values, linewidth=0.70, alpha=0.60, label="physical energy score")
    if len(values) >= 20:
        window = max(5, len(values) // 100)
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(steps[window - 1:], smoothed, linewidth=1.8, label=f"moving average ({window})")
    axes[0, 0].set_title("Physical-theta loss at every gradient step", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("gradient step")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)

    rmse = np.maximum(np.asarray(history["step_mean_rmse"], dtype=float), 1e-12)
    axes[0, 1].plot(steps, rmse, linewidth=0.75)
    axes[0, 1].set_title("Posterior-mean RMSE at every gradient step", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("gradient step")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.2)

    axes[0, 2].plot(steps, history["step_grad_norm"], linewidth=0.75)
    axes[0, 2].set_title("Gradient norm at every step", loc="left", fontweight="bold")
    axes[0, 2].set_xlabel("gradient step")
    axes[0, 2].grid(alpha=0.2)

    axes[1, 0].plot(epochs, np.maximum(history["epoch_train_loss"], 1e-12), marker="o", markersize=3, label="train")
    axes[1, 0].plot(epochs, np.maximum(history["epoch_val_loss"], 1e-12), marker="o", markersize=3, label="validation")
    axes[1, 0].axvline(best_epoch, linestyle="--", linewidth=1.0, label=f"best epoch {best_epoch}")
    axes[1, 0].set_title("Per-epoch physical energy score", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    energy_by_t = np.asarray(history["epoch_val_energy_by_t"], dtype=float)
    rmse_by_t = np.asarray(history["epoch_val_rmse_by_t"], dtype=float)
    selected_epochs = np.unique(np.clip(np.rint(np.linspace(0, len(energy_by_t) - 1, 5)).astype(int), 0, len(energy_by_t) - 1))
    prefix_axis = np.arange(1, energy_by_t.shape[1] + 1)
    for epoch_index in selected_epochs:
        axes[1, 1].plot(prefix_axis, np.maximum(energy_by_t[epoch_index], 1e-12), label=f"epoch {epoch_index + 1}")
        axes[1, 2].plot(prefix_axis, np.maximum(rmse_by_t[epoch_index], 1e-12), label=f"epoch {epoch_index + 1}")
    axes[1, 1].set_title("Validation energy score by prefix", loc="left", fontweight="bold")
    axes[1, 2].set_title("Validation RMSE by prefix", loc="left", fontweight="bold")
    for ax in axes[1, 1:]:
        ax.set_xlabel("sequential step t")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle("Mode-A dimension-agnostic sequential training diagnostics", fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)

#%% 17) Training function
def train_model(
    train_loader: DataLoader,
    eval_data: dict[str, np.ndarray],
    fixed_trajectory: dict[str, np.ndarray],
    fixed_prior_particles: np.ndarray,
    run_dir: Path,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Any]:
    """Train one end-to-end dimension-agnostic sequential posterior model.

    Every trajectory is processed in observation order by one jax.lax.scan. The scan carry
    is the current physical posterior cloud. Before every update those particles pass through
    the shared theta encoder; all T physical energy-score terms are returned by the same
    compiled forward pass and differentiated jointly with the full world model.

    Training data come from an infinite PyTorch DataLoader. n_train_trajectories therefore
    retains its old role as the nominal amount of work per epoch, but every one of those
    trajectories is newly simulated rather than revisited from a finite stored dataset.
    One observation_count is sampled when each minibatch is formed and is held fixed through
    all T scan steps for that minibatch; a new count is sampled for the next minibatch.
    """
    model = SequentialBayesModel(cfg, key=jax.random.key(cfg.seed))
    print("\nsequential cross-attention")
    print_model_parameter_count(model)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(candidate_model, candidate_opt_state, batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            batch_objective, has_aux=True
        )(candidate_model, batch, cfg)
        params = eqx.filter(candidate_model, eqx.is_array)
        updates, candidate_opt_state = optimizer.update(grads, candidate_opt_state, params)
        candidate_model = eqx.apply_updates(candidate_model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return candidate_model, candidate_opt_state, loss, metrics, grad_norm

    history: dict[str, list] = {
        "step_loss": [],
        "step_energy_score": [],
        "step_final_energy_score": [],
        "step_mean_rmse": [],
        "step_grad_norm": [],
        "epoch_train_loss": [],
        "epoch_val_loss": [],
        "epoch_val_energy_score": [],
        "epoch_val_final_energy_score": [],
        "epoch_val_mean_rmse": [],
        "epoch_val_energy_by_t": [],
        "epoch_val_rmse_by_t": [],
        "epoch_val_spread_by_t": [],
    }

    # Physical posterior particles are available immediately; no post-hoc decoder is needed.
    plot_posterior_evolution(
        model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        run_dir / "plots" / "fixed_trajectory_before_training.png",
        "sequential cross-attention: before training (identity physical transport)",
    )

    initial_metrics = evaluate_model(model, eval_data, cfg, seed=cfg.seed + 91_000)
    print(
        f"[sequential] initial validation ES={initial_metrics['energy_score']:.6f} | "
        f"theta RMSE={initial_metrics['posterior_mean_rmse']:.5f}"
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
    observation_count_rng = np.random.default_rng(cfg.seed + 654_321)

    for epoch in range(1, cfg.epochs + 1):
        epoch_started_at = time.time()
        train_losses_this_epoch: list[float] = []
        progress = tqdm(
            range(n_steps),
            desc=f"sequential epoch {epoch:03d}/{cfg.epochs:03d}",
            dynamic_ncols=True,
            leave=True,
            mininterval=5.0,
        )

        for batch_index in progress:
            # Infinite IterableDataset: every next() call creates a fresh theta*, fresh sensor
            # trajectory, and a fresh prior cloud from the same newly drawn prior pi_m.
            batch_np = next(train_iterator)
            # One evidence size per minibatch, held fixed for the ENTIRE sequential scan.
            # The next minibatch samples a fresh value, so training covers downstream calls
            # with different numbers of observations without varying o inside one scan.
            batch_np["observation_count"] = np.asarray(
                observation_count_rng.integers(
                    cfg.min_observations_per_step, cfg.max_observations_per_step + 1
                ),
                dtype=np.int32,
            )
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            model, opt_state, loss, metrics, grad_norm = train_step(
                model, opt_state, batch
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
            history["step_grad_norm"].append(host_grad_norm)
            progress.set_postfix(
                ES=f"{float(host['energy_score']):.4f}",
                RMSE=f"{float(host['posterior_mean_rmse']):.4f}",
                grad=f"{host_grad_norm:.3f}",
                refresh=False,
            )

        epoch_train_loss = float(np.mean(train_losses_this_epoch))
        val_metrics = evaluate_model(
            model,
            eval_data,
            cfg,
            seed=cfg.seed + 91_000,  # identical validation draws every epoch
        )
        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_val_loss"].append(float(val_metrics["loss"]))
        history["epoch_val_energy_score"].append(float(val_metrics["energy_score"]))
        history["epoch_val_final_energy_score"].append(float(val_metrics["final_energy_score"]))
        history["epoch_val_mean_rmse"].append(float(val_metrics["posterior_mean_rmse"]))
        history["epoch_val_energy_by_t"].append(np.asarray(val_metrics["energy_by_t"], dtype=np.float64))
        history["epoch_val_rmse_by_t"].append(np.asarray(val_metrics["rmse_by_t"], dtype=np.float64))
        history["epoch_val_spread_by_t"].append(np.asarray(val_metrics["spread_by_t"], dtype=np.float64))

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
                "training": "continuous fresh-simulator stream + physical posterior recurrence with jax.lax.scan",
                "conditioning": "particle self-attention + cross-attention to one batch-level causal observation-memory prefix used at every step",
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "elapsed_seconds": time.time() - training_started_at,
                "objective": "mean physical-theta energy score over B x T",
                "min_observations_per_step": cfg.min_observations_per_step,
                "max_observations_per_step": cfg.max_observations_per_step,
                "test_observations_per_step": cfg.test_observations_per_step,
            },
        )

        print(
            f"[sequential] epoch {epoch:03d}: "
            f"train ES={epoch_train_loss:.6f} | val ES={float(val_metrics['energy_score']):.6f} | "
            f"final ES={float(val_metrics['final_energy_score']):.6f} | "
            f"theta RMSE={float(val_metrics['posterior_mean_rmse']):.5f} | "
            f"{time.time() - epoch_started_at:.1f}s"
        )

        if epoch in visualisation_epochs:
            plot_posterior_evolution(
                model,
                fixed_trajectory,
                fixed_prior_particles,
                cfg,
                run_dir / "plots" / f"fixed_trajectory_epoch_{epoch:04d}.png",
                f"sequential cross-attention: physical posterior evolution after epoch {epoch}",
            )

    best_model = load_model(
        run_dir / "artefacts" / "model_best.eqx", cfg, key=jax.random.key(0)
    )
    final_metrics = evaluate_model(best_model, eval_data, cfg, seed=cfg.seed + 91_000)
    plot_posterior_evolution(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        run_dir / "plots" / "fixed_trajectory_best_model.png",
        f"sequential cross-attention: best physical posterior (epoch {best_epoch})",
    )
    plot_training_diagnostics(
        history, best_epoch, run_dir / "plots" / "training_diagnostics.png"
    )

    training_elapsed_seconds = int(time.time() - training_started_at)
    training_hours, training_remainder = divmod(training_elapsed_seconds, 3600)
    training_minutes, training_seconds = divmod(training_remainder, 60)
    print(
        "[sequential] training complete in "
        f"{training_hours:02d}:{training_minutes:02d}:{training_seconds:02d}; "
        f"best epoch={best_epoch}, val ES={best_val_loss:.6f}"
    )
    return {
        "model": best_model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_metrics": final_metrics,
    }

#%% 18) Create a new run OR reload one existing sequential run folder
np.random.seed(CFG.seed)
print("JAX devices:", jax.devices())
print("Configuration:\n", yaml.safe_dump(asdict(CFG), sort_keys=False))

if train_wm:
    # Original full-training path.  Sequential is intentionally lean: there is no
    # architecture-specific subfolder because cross-attention is the only conditioning mechanism.
    run_dir = make_run_dir(CFG.env_name, CFG.runs_base)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(CFG), handle, sort_keys=False)
    print("Run directory:", run_dir)

    # Main training is now an INFINITE fresh simulator stream.  The nominal epoch length is
    # still CFG.n_train_trajectories and batch_size is unchanged, but no finite train_data is
    # cached or replayed across epochs.  Validation remains fixed for comparable curves.
    train_loader = make_continuous_train_loader(CFG, seed=CFG.seed + 1_000)
    eval_rng = np.random.default_rng(CFG.seed + 2_000)
    eval_data = simulate_trajectories(
        eval_rng, CFG.n_eval_trajectories, CFG.trajectory_length, CFG,
        shape_pool=TRAIN_SHAPES, balanced_shapes=True,
    )

    prior_mode = (
        f"random Gaussian-mixture meta-prior, K in "
        f"[{CFG.meta_prior_min_components}, {CFG.meta_prior_max_components}]"
        if CFG.use_meta_prior else f"fixed Gaussian N(0, {CFG.prior_std}^2 I)"
    )
    print("Continuous training stream:")
    print(f"  fresh trajectories per nominal epoch: {CFG.n_train_trajectories}")
    print(f"  batch size (unchanged): {CFG.batch_size}")
    print(f"  S,D grid: [{CFG.min_num_sources},{CFG.max_num_sources}] x [{CFG.min_source_dim},{CFG.max_source_dim}]")
    print(f"  held-out training shapes: {HELDOUT_SHAPES}")
    print(f"  training shapes: {len(TRAIN_SHAPES)} / {len(ALL_SHAPES)} combinations")
    print(f"  t=0 prior mode: {prior_mode}")

    # Keep one fixed 2-D problem for physical posterior plots. It is generated separately
    # so heterogeneous eval_data is free to begin with any shape.
    fixed_rng = np.random.default_rng(CFG.seed + 2_500)
    fixed_data = simulate_trajectories(
        fixed_rng,
        1,
        CFG.trajectory_length,
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
        **{key: fixed_data[key][0] for key in PRIOR_SPEC_KEYS},
    }
    fixed_prior_spec = _prior_spec_from_trajectory(fixed_trajectory, CFG)
    fixed_prior_active = sample_prior_np(
        np.random.default_rng(CFG.seed + 3_000),
        CFG.num_particles,
        CFG,
        num_sources=CFG.num_sources,
        source_dim=CFG.source_dim,
        prior_spec=fixed_prior_spec,
    )
    fixed_prior_particles = pad_theta_np(fixed_prior_active, CFG)
    np.savez_compressed(
        run_dir / "artefacts" / "fixed_trajectory.npz",
        **{key: np.asarray(value) for key, value in fixed_trajectory.items()},
        prior_particles=fixed_prior_particles,
    )
else:
    # Reload path.  Run the notebook from the existing sequential run folder itself.
    # All refreshed plots and diagnostics remain in that same run; no new run is made.
    run_dir = Path.cwd().expanduser().resolve()
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
    print("Existing sequential run directory:", run_dir)
    print("Main-model training is disabled; reloading saved model and diagnostics.")

    # The evaluation trajectories are deterministic preprocessing, so regenerate them from
    # the original saved configuration for fresh diagnostics without needing the train stream.
    eval_rng = np.random.default_rng(CFG.seed + 2_000)
    eval_data = simulate_trajectories(
        eval_rng, CFG.n_eval_trajectories, CFG.trajectory_length, CFG,
        shape_pool=TRAIN_SHAPES, balanced_shapes=True,
    )

    fixed_path = run_dir / "artefacts" / "fixed_trajectory.npz"
    if not fixed_path.is_file():
        raise FileNotFoundError(f"Missing saved fixed trajectory: {fixed_path}")
    with np.load(fixed_path, allow_pickle=False) as fixed_npz:
        fixed_trajectory = {
            key: np.asarray(fixed_npz[key])
            for key in ("theta_true", "observations", "num_sources", "theta_size")
        }
        fixed_trajectory["observations"] = _ensure_observation_blocks_np(
            fixed_trajectory["observations"]
        )
        fixed_trajectory["observation_count"] = np.asarray(
            CFG.test_observations_per_step, dtype=np.int32
        )
        for key in PRIOR_SPEC_KEYS:
            if key in fixed_npz.files:
                fixed_trajectory[key] = np.asarray(fixed_npz[key])
        fixed_prior_particles = np.asarray(fixed_npz["prior_particles"], dtype=np.float32)

# These descriptive plots are cheap and are regenerated in both modes.
plot_architecture_schematic(CFG, run_dir / "plots" / "architecture_schematic.png")
plot_source_trajectory(
    fixed_trajectory, CFG, run_dir / "plots" / "fixed_trajectory_sensor_field.png"
)

#%% 19) Train the sequential cross-attention model, or reload the local best model
# Observation embedder + theta embedder + Posterior Transformer are optimized jointly from
# the SAME physical-theta energy score; there are no stop-gradients or auxiliary latent
# regularizers. Observation time is handled only by the posterior recurrence inside lax.scan.
if train_wm:
    result = train_model(
        train_loader,
        eval_data,
        fixed_trajectory,
        fixed_prior_particles,
        run_dir,
        CFG,
    )
else:
    artefact_dir = run_dir / "artefacts"
    model_path = artefact_dir / "model_best.eqx"
    history_path = artefact_dir / "history.npz"
    state_path = artefact_dir / "training_state.json"
    for required_path in (model_path, history_path, state_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing saved training artefact: {required_path}")

    print("Reloading best sequential model from:", model_path)
    best_model = load_model(model_path, CFG, key=jax.random.key(0))
    with np.load(history_path, allow_pickle=False) as saved_history:
        history = {key: np.asarray(saved_history[key]) for key in saved_history.files}
    with state_path.open("r", encoding="utf-8") as handle:
        training_state = json.load(handle)
    best_epoch = int(training_state["best_epoch"])
    best_val_loss = float(training_state["best_val_loss"])
    final_metrics = evaluate_model(best_model, eval_data, CFG, seed=CFG.seed + 91_000)

    plot_training_diagnostics(
        history, best_epoch, run_dir / "plots" / "training_diagnostics.png"
    )
    plot_posterior_evolution(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        CFG,
        run_dir / "plots" / "fixed_trajectory_best_model.png",
        f"sequential cross-attention: best physical posterior (epoch {best_epoch})",
    )
    result = {
        "model": best_model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_metrics": final_metrics,
    }

model = result["model"]
if not train_wm:
    print("\nsequential cross-attention")
    print_model_parameter_count(model)


#%% 19b) Direct physical best-model posterior visualisation
plot_posterior_evolution(
    model,
    fixed_trajectory,
    fixed_prior_particles,
    CFG,
    run_dir / "plots" / "fixed_trajectory_best_model.png",
    "sequential cross-attention: direct physical posterior evolution",
)

#%% 19c) Balanced dimensional-generalisation evaluation, including held-out S,D shapes
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
    rmse_grid = study["rmse_grid"]
    energy_grid = study["energy_grid"]

    # Benchmark-style summary: shape heatmaps, true-vs-posterior-mean scatter, and difficulty.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)
    for ax, grid, title, label in (
        (axes[0, 0], rmse_grid, "Final posterior-mean RMSE by shape", "RMSE"),
        (axes[0, 1], energy_grid, "Final energy score by shape", "energy score"),
    ):
        positive = grid[np.isfinite(grid) & (grid > 0)]
        norm = LogNorm(vmin=max(float(np.min(positive)), 1e-8), vmax=max(float(np.max(positive)), float(np.min(positive)) * 1.0001))
        im = ax.imshow(grid, origin="lower", aspect="auto", norm=norm)
        ax.set_xticks(np.arange(cfg.max_source_dim), np.arange(1, cfg.max_source_dim + 1))
        ax.set_yticks(np.arange(cfg.max_num_sources), np.arange(1, cfg.max_num_sources + 1))
        ax.set_xlabel("source dimension D")
        ax.set_ylabel("number of sources S")
        ax.set_title(title, fontweight="bold")
        fig.colorbar(im, ax=ax, label=label)
        for s, d in HELDOUT_SHAPES:
            ax.text(d - 1, s - 1, "H", ha="center", va="center", fontweight="bold", fontsize=11,
                    bbox=dict(boxstyle="circle,pad=0.18", facecolor="white", alpha=0.75, edgecolor="none"))

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
    axes[1, 1].scatter(shape_size[~is_heldout], shape_rmse[~is_heldout], alpha=0.75, label="seen shape")
    axes[1, 1].scatter(shape_size[is_heldout], shape_rmse[is_heldout], marker="X", s=90, label="held-out shape")
    for (s, d), x, y, h in zip(ALL_SHAPES, shape_size, shape_rmse, is_heldout):
        if h:
            axes[1, 1].annotate(f"{s}x{d}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("theta size S x D")
    axes[1, 1].set_ylabel("final RMSE")
    axes[1, 1].set_title("Difficulty versus active theta size", fontweight="bold")
    axes[1, 1].grid(alpha=0.2)
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
        ax.set_title(f"S={s}, D={d}" + ("  HELD OUT" if (s, d) in HELDOUT_SHAPES else ""), loc="left", fontweight="bold")
        ax.set_ylabel("theta")
        ax.grid(alpha=0.2)
        ax.legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("compact active theta coordinate")
    fig.suptitle("Fixed dimensional-generalisation examples at the final evaluation step", fontsize=14, fontweight="bold")
    fig.savefig(run_dir / "plots" / "dimensional_generalisation_fixed_examples.png", dpi=180)
    display(fig); plt.close(fig)

    # How quickly evidence is used: every shape as a faint line, with seen/held-out group means.
    t = np.arange(1, study["rmse_by_t"].shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), constrained_layout=True)
    seen_curves_rmse, held_curves_rmse, seen_curves_es, held_curves_es = [], [], [], []
    for s, d in ALL_SHAPES:
        r = study["shape_rmse_by_t"][s - 1, d - 1]
        e = study["shape_energy_by_t"][s - 1, d - 1]
        held = (s, d) in HELDOUT_SHAPES
        axes[0].plot(t, np.maximum(r, 1e-12), alpha=0.16, linewidth=0.9, linestyle="--" if held else "-")
        axes[1].plot(t, np.maximum(e, 1e-12), alpha=0.16, linewidth=0.9, linestyle="--" if held else "-")
        (held_curves_rmse if held else seen_curves_rmse).append(r)
        (held_curves_es if held else seen_curves_es).append(e)
    axes[0].plot(t, np.maximum(np.mean(seen_curves_rmse, axis=0), 1e-12), linewidth=2.5, label="seen-shape mean")
    axes[0].plot(t, np.maximum(np.mean(held_curves_rmse, axis=0), 1e-12), linewidth=2.5, linestyle="--", label="held-out-shape mean")
    axes[1].plot(t, np.maximum(np.mean(seen_curves_es, axis=0), 1e-12), linewidth=2.5, label="seen-shape mean")
    axes[1].plot(t, np.maximum(np.mean(held_curves_es, axis=0), 1e-12), linewidth=2.5, linestyle="--", label="held-out-shape mean")
    for ax, title, ylabel in zip(axes, ("Posterior-mean RMSE versus evidence", "Energy score versus evidence"), ("RMSE", "energy score")):
        ax.axvline(cfg.trajectory_length, linestyle=":", linewidth=1.3, label="training horizon")
        ax.set_yscale("log")
        ax.set_xlabel("sequential step t")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.2)
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
            ax.axis("off"); continue
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
    fig.suptitle("Source identification: truth stars, posterior means, and 90% marginal ellipses\n(best source permutation used for display only)", fontsize=14, fontweight="bold")
    fig.savefig(run_dir / "plots" / "source_identification_gallery.png", dpi=180)
    display(fig); plt.close(fig)


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
print(
    "[dimensional generalisation] "
    f"T_eval={CFG.evaluation_trajectory_length} | "
    f"seen final RMSE={np.mean(dimensional_generalisation['rmse_by_t'][seen, -1]):.5f} | "
    f"held-out final RMSE={np.mean(dimensional_generalisation['rmse_by_t'][held, -1]):.5f} | "
    f"seen final ES={np.mean(dimensional_generalisation['energy_by_t'][seen, -1]):.5f} | "
    f"held-out final ES={np.mean(dimensional_generalisation['energy_by_t'][held, -1]):.5f}"
)

#%% 20) Direct visual comparison with a likelihood-based posterior reference
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
    """Numerically test the exact identities built into the sequential architecture.

    1. Causality: perturb future observations and verify physical posterior outputs through step t do not change.
    2. Memory causality: the contextual observation memories through t ignore future steps.
    3. Particle equivariance: permute prior-particle axis, undo it on outputs, verify equality.

    Prefix-set permutation invariance is deliberately NOT a property anymore: the posterior
    consumes observations sequentially, so changing their order changes the recurrence.
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

    future_perturbed = obs.copy()
    if t < len(obs):
        future_perturbed[t:, :, :D] = rng.uniform(
            cfg.design_low, cfg.design_high, size=future_perturbed[t:, :, :D].shape
        )
        future_perturbed[t:, :, -1] += rng.normal(
            0.0, 5.0, size=future_perturbed[t:, :, -1].shape
        )
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
    ax.set_xticklabels(["causal\nposterior", "causal\nmemory", "particle\nequivariance"])
    ax.set_ylabel("max absolute discrepancy")
    ax.set_title("Sequential architectural identities should be near floating-point precision",
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


#%% 24) Limit study T -> larger: within-horizon and out-of-horizon prefix behaviour
long_eval_rng = np.random.default_rng(CFG.seed + 800_000)
long_eval_data = simulate_trajectories(
    long_eval_rng,
    CFG.limit_eval_trajectories,
    CFG.long_trajectory_length,
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
    ax.axvline(CFG.trajectory_length, linestyle="--", linewidth=1.0,
               label="training horizon")
    ax.set_xlabel("sequential step t")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
for ax in axes:
    ax.set_yscale("log")
axes[0].set_title("Energy score")
axes[1].set_title("Physical posterior-mean RMSE")
axes[2].set_title("Physical posterior spread")
fig.suptitle(
    "Trajectory-length study: solid region is trained horizon; right side is extrapolation",
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
    mc_pool_rng, mc_pool_size, CFG.trajectory_length, CFG
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
    prior_spec = _prior_spec_from_trajectory(trajectory, cfg)
    stds = []
    for n_particles in cfg.particle_limit_values:
        means = []
        for repeat in range(cfg.prior_resample_repeats):
            rng = np.random.default_rng(cfg.seed + 1_000_000 + 1000 * n_particles + repeat)
            active_prior = sample_prior_np(
                rng, n_particles, cfg, num_sources=S, source_dim=D, prior_spec=prior_spec
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

summary = {
    "objective": "physical-theta energy score",
    "mode": "Mode A: theta* fixed within trajectory, re-drawn across continuously refreshed trajectories",
    "training_data": "infinite PyTorch IterableDataset/DataLoader simulator stream",
    "fresh_train_trajectories_per_nominal_epoch": CFG.n_train_trajectories,
    "t0_meta_prior_enabled": CFG.use_meta_prior,
    "t0_prior_mode": (
        "random Gaussian-mixture meta-prior" if CFG.use_meta_prior
        else "fixed Gaussian N(0, prior_std^2 I)"
    ),
    "meta_prior_component_range": [CFG.meta_prior_min_components, CFG.meta_prior_max_components],
    "meta_prior_degenerate_probability": CFG.meta_prior_degenerate_probability,
    "dimension_agnostic": True,
    "train_num_sources_range": [CFG.min_num_sources, CFG.max_num_sources],
    "train_source_dim_range": [CFG.min_source_dim, CFG.max_source_dim],
    "embedding_dim": CFG.embedding_dim,
    "max_theta_size": CFG.max_num_sources * CFG.max_source_dim,
    "posterior_output": "compact physical theta; first S*D entries active",
    "theta_true_embedded_for_loss": False,
    "sequential_posterior_training": True,
    "time_recurrence": "jax.lax.scan",
    "conditioning": "particle self-attention + cross-attention to one batch-level causal observation-memory prefix used at every step",
    "min_observations_per_step": CFG.min_observations_per_step,
    "max_observations_per_step": CFG.max_observations_per_step,
    "test_observations_per_step": CFG.test_observations_per_step,
    "trajectory_length": CFG.trajectory_length,
    "evaluation_trajectory_length": CFG.evaluation_trajectory_length,
    "heldout_shapes": [list(shape) for shape in HELDOUT_SHAPES],
    "num_particles": CFG.num_particles,
    "best_epoch": int(result["best_epoch"]),
    "best_val_loss": float(result["best_val_loss"]),
    "final_metrics": {
        key: float(value)
        for key, value in result["final_metrics"].items()
        if np.ndim(value) == 0
    },
    "dimensional_generalisation": {
        "seen_final_rmse": float(np.mean(dimensional_generalisation["rmse_by_t"][~dimensional_generalisation["heldout"], -1])),
        "heldout_final_rmse": float(np.mean(dimensional_generalisation["rmse_by_t"][dimensional_generalisation["heldout"], -1])),
        "seen_final_energy": float(np.mean(dimensional_generalisation["energy_by_t"][~dimensional_generalisation["heldout"], -1])),
        "heldout_final_energy": float(np.mean(dimensional_generalisation["energy_by_t"][dimensional_generalisation["heldout"], -1])),
    },
}
save_json(run_dir / "artefacts" / "final_summary.json", summary)

print("\nFinal dimension-agnostic sequential Mode-A summary")
print(json.dumps(summary, indent=2))
print("All artefacts saved under:", run_dir)
