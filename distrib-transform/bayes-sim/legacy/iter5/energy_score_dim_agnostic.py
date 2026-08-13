#%% 1) Imports, configuration, and experiment conventions
"""Dimension-agnostic Mode-A Bayesian source localisation with parallel prefix training.

This notebook-style file preserves the original Mode-A construction while making the
learned inference system agnostic to both the number of physical sources S and the
coordinate dimension D of each source.

For every simulated trajectory m we draw a NEW problem shape and generating parameter

    S_m ~ p(S),    D_m ~ p(D),    theta_m^* ~ p(theta | S_m, D_m),

hold that theta_m^* fixed while simulating the whole trajectory

    (x_{m,1}, y_{m,1}), ..., (x_{m,T}, y_{m,T}) ~ p(x_{1:T}, y_{1:T} | theta_m^*),

and train on every prefix x_{1:t}, y_{1:t}.  The same theta_m^* is therefore the
proper-score target for all t inside one trajectory, but theta_m^*, S_m, and D_m are
re-drawn between trajectories.  This is the Bayes-consistent "fixed within a trajectory"
case, not the single-global-truth collapse case.

The key dimensionality-agnostic change follows the dimension-aggregating embedder in
TAMO Figure 2.  Scalar coordinates are first mapped to vector tokens, those tokens
interact through a small Transformer across the active dimensions, learned positional
vectors modulate the resulting dimension tokens element-wise, and a masked mean pool
produces one fixed-dimensional representation E.  There are TWO such embedders here:

1. ObservationDimensionEmbedder: padded [design coordinates, outcome] -> R^E.
2. ThetaDimensionEmbedder: padded source coordinates theta -> R^E.

The Likelihood Transformer therefore never sees a raw dimension-dependent observation,
and the Posterior Transformer never sees a raw dimension-dependent theta particle.  The
Posterior Transformer transports particles entirely in the E-dimensional embedding space.
The energy score is also computed in that embedding space: theta_true is embedded by the
same end-to-end theta embedder before being compared with posterior particle embeddings.

A configurable SIGReg term acts on fresh embedded PRIOR clouds to discourage the shared
theta embedder from collapsing to a constant representation.  Setting sigreg_weight=0.0
disables it.  No stop-gradient is used anywhere in the end-to-end inference model.

After the main model has finished training, a separate lightweight Transformer decoder is
trained for ONE fixed visualisation problem dimensionality.  It learns only to invert the
already-trained theta embedding on prior draws; it is not part of the main objective and
is used only to map latent prior/posterior particles back to physical source coordinates
for the same source/design/outcome plots as before.

Two end-to-end posterior-conditioning architectures are still trained and compared:

1. AdaLN conditioning
   padded observations -> dimension embedder -> causal prefix-set Likelihood Transformer
   embedded prior particles + summary_t -> AdaLN Posterior Transformer -> q_phi(z_theta|x_1:t)

2. Cross-attention conditioning
   padded observations -> dimension embedder -> causal prefix-set Likelihood Transformer
   embedded prior particles cross-attend to the ONE prefix summary vector for prefix t.

The important parallelism is still across prefixes.  The Likelihood Transformer computes
all t=1,...,T prefix representations in one JAX program, and the Posterior Transformer
maps the same embedded prior cloud to all T posterior clouds in one JAX program.  There
is no posterior recurrence z_t -> z_{t+1} and no Python loop over t in the model or loss.
Python loops remain only over Transformer depth, minibatches, epochs, simulation, and
high-level diagnostic sweeps.

Notation used in arrays
-----------------------
B : number of trajectories in a minibatch
T : trajectory length / number of scored prefixes
N : number of prior/output particles
S : active number of exchangeable physical sources for one trajectory
D : active coordinate dimension of each source for one trajectory
E : fixed theta/observation embedding dimension
H : hidden dimension of the main Transformers
Smax, Dmax : padding limits used to mix heterogeneous problems in one JAX minibatch

theta_true           [B, Smax, Dmax]      padded; active block is [:S,:D]
theta_size           [B]                  equals S*D
num_sources          [B]                  stored explicitly in train_data
observations         [B, T, Dmax+1]       padded design + outcome in final slot
prior_particles      [B, N, Smax, Dmax]   iid from p(theta|S,D), padded
likelihood_summaries [B, T, H]             one vector per causal prefix
posterior_embeddings [B, T, N, E]
theta_true_embedding [B, E]
energy_by_t           [B, T]

The observation pairs are generated in advance.  The neural model never calls the
likelihood during training; it sees only the padded observations, shape metadata, fresh
prior particles, and the proper-score target theta_true.  The known likelihood is used
again only in an OPTIONAL reference-posterior diagnostic after training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from IPython.display import display
from tqdm.auto import tqdm
import yaml

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

Array = jax.Array
ConditioningMode = Literal["adaln", "cross_attention"]

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})


@dataclass(frozen=True)
class BayesTransportConfig:
    """Defaults are the experiment; edit them here rather than in an override block."""

    # Reproducibility and run bookkeeping.
    env_name: str = "energy_score_dimensionality_agnostic"
    seed: int = 2030
    runs_base: str = "./runs"

    # Source-localisation simulator.  `num_sources` and `source_dim` are now ONLY the
    # fixed problem used by the post-hoc visualisation decoder and 2-D diagnostic plots.
    num_sources: int = 2
    source_dim: int = 2
    prior_std: float = 1.0
    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.30

    # Heterogeneous training-task distribution.  Arrays are padded to these maxima,
    # while masks ensure that inactive source/coordinate slots never enter an embedder.
    min_num_sources: int = 1
    max_num_sources: int = 8
    min_source_dim: int = 1
    max_source_dim: int = 8

    # TAMO-style dimension aggregation.  Every observation and every theta particle is
    # mapped to one fixed E-vector before it reaches either main Transformer.  The hard
    # check below guarantees max(S*D) <= E, as requested.
    embedding_dim: int = 64
    dimension_embedder_depth: int = 2
    scalar_encoder_depth: int = 2
    embedding_heads: int = 4

    # Mode-A trajectory and particle counts.
    trajectory_length: int = 128
    num_particles: int = 64
    n_train_trajectories: int = 4096
    n_eval_trajectories: int = 256
    batch_size: int = 16

    # Likelihood Transformer.  There is deliberately NO likelihood_summary_tokens:
    # one learned prefix summary vector is sufficient and avoids a fake R dimension.
    hidden_dim: int = 96
    heads: int = 4
    mlp_ratio: int = 4
    likelihood_depth: int = 2

    # Posterior particle Transformer.  The residual transport is now in R^E.
    posterior_depth: int = 3
    max_embedding_displacement: float = 6.0
    canonicalize_particle_sources: bool = True
    # architectures_to_train: tuple[str, ...] = ("adaln", "cross_attention")
    architectures_to_train: tuple[str, ...] = ("cross_attention", )

    # Observation normalisation.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Optimisation.  The proper-score term is the mean EMBEDDING-space energy score
    # over B x T.  SIGReg is optional anti-collapse regularisation for theta embeddings.
    epochs: int = 1500
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0
    sigreg_weight: float = 0.1              # set to 0.0 to disable
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sigreg_t_max: float = 3.0

    # Lightweight post-hoc visualisation decoder.  This is intentionally trained only
    # AFTER the main end-to-end model and is fixed to (num_sources, source_dim).
    decoder_hidden_dim: int = 64
    decoder_heads: int = 4
    decoder_depth: int = 2
    decoder_epochs: int = 5000
    decoder_learning_rate: float = 3e-4
    decoder_batch_size: int = 128
    decoder_train_samples: int = 8192

    # Persistence / visualisation cadence.
    save_every_epochs: int = 10
    final_plot_examples: int = 3
    grid_size: int = 180

    # Reference-posterior diagnostic only; never enters the training loss.
    reference_proposals: int = 50_000
    reference_particles: int = 2_000

    # Limit / theorem diagnostics after training.
    limit_eval_trajectories: int = 192
    particle_limit_values: tuple[int, ...] = (16, 32, 64, 128, 256)
    long_trajectory_length: int = 192
    trajectory_mc_values: tuple[int, ...] = (8, 16, 32, 64, 128, 192)
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
    if cfg.decoder_hidden_dim % cfg.decoder_heads != 0:
        raise ValueError("decoder_hidden_dim must be divisible by decoder_heads.")


# One default instantiation only: no second CFG = BayesTransportConfig(...) override block.
CFG = BayesTransportConfig()
validate_config(CFG)


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


def save_model(path: str | Path, model: "ModeAParallelBayesModel"):
    eqx.tree_serialise_leaves(Path(path), model)


def load_model(
    path: str | Path,
    cfg: BayesTransportConfig,
    conditioning: ConditioningMode,
    *,
    key: Array | None = None,
) -> "ModeAParallelBayesModel":
    """Rebuild the matching skeleton and load Equinox leaves."""
    if key is None:
        key = jax.random.key(0)
    skeleton = ModeAParallelBayesModel(cfg, conditioning=conditioning, key=key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)



def save_visualization_decoder(path: str | Path, decoder: "ThetaVisualizationDecoder"):
    eqx.tree_serialise_leaves(Path(path), decoder)


def load_visualization_decoder(
    path: str | Path,
    cfg: BayesTransportConfig,
    *,
    key: Array | None = None,
) -> "ThetaVisualizationDecoder":
    """Rebuild the fixed-problem post-hoc decoder skeleton and load its leaves."""
    if key is None:
        key = jax.random.key(0)
    skeleton = ThetaVisualizationDecoder(cfg, key=key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)




#%% 3) Prior and source-localisation simulator
def sample_prior_np(
    rng: np.random.Generator,
    n: int,
    cfg: BayesTransportConfig = CFG,
    *,
    num_sources: int | None = None,
    source_dim: int | None = None,
) -> np.ndarray:
    """Draw n independent theta ~ p(theta | S,D) in its ACTIVE, unpadded shape.

    `num_sources` and `source_dim` default to the fixed visualisation problem only.
    Heterogeneous training batches call this helper with the per-trajectory metadata.
    """
    S = cfg.num_sources if num_sources is None else int(num_sources)
    D = cfg.source_dim if source_dim is None else int(source_dim)
    if S * D > cfg.embedding_dim:
        raise ValueError(f"theta size {S*D} exceeds embedding_dim={cfg.embedding_dim}.")
    return rng.normal(0.0, cfg.prior_std, size=(int(n), S, D)).astype(np.float32)


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

      theta [S,D],     designs [T,D]   -> [T]
      theta [B,S,D],   designs [B,T,D] -> [B,T]
      theta [P,S,D],   designs [T,D]   -> [P,T]
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
) -> tuple[np.ndarray, np.ndarray]:
    """Sample S and theta_size=S*D; D is intentionally derivable from those two."""
    if fixed_num_sources is None:
        num_sources = rng.integers(
            cfg.min_num_sources, cfg.max_num_sources + 1, size=int(n), dtype=np.int32
        )
    else:
        num_sources = np.full(int(n), int(fixed_num_sources), dtype=np.int32)

    if fixed_source_dim is None:
        source_dim = rng.integers(
            cfg.min_source_dim, cfg.max_source_dim + 1, size=int(n), dtype=np.int32
        )
    else:
        source_dim = np.full(int(n), int(fixed_source_dim), dtype=np.int32)

    theta_size = (num_sources * source_dim).astype(np.int32)
    if np.any(theta_size > cfg.embedding_dim):
        raise ValueError("Sampled theta_size exceeds embedding_dim; validate CFG ranges.")
    return num_sources, theta_size


def simulate_mode_a_trajectories(
    rng: np.random.Generator,
    n_trajectories: int,
    trajectory_length: int,
    cfg: BayesTransportConfig = CFG,
    *,
    fixed_num_sources: int | None = None,
    fixed_source_dim: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate complete heterogeneous Mode-A trajectories before neural training.

    Critical sampling provenance
    ----------------------------
    1. Each row m receives its own S_m and D_m, then theta_true[m] is drawn once from
       p(theta | S_m,D_m).
    2. All T sensor readings in row m are simulated conditional on that SAME theta.
    3. The next row draws a fresh problem shape and theta_true.
    4. Prior particles are deliberately NOT stored here.  They are drawn independently
       from p(theta | S_m,D_m) when a minibatch is formed.

    `train_data` explicitly stores `num_sources`.  It also stores `theta_size=S*D`, so
    source_dim is always recovered as theta_size // num_sources rather than maintained
    as a second independent source of truth.

    Padded storage
    --------------
    theta_true[m]          [Smax,Dmax], active block [:S_m,:D_m]
    observations[m,t]      [Dmax+1], design in [:D_m], scalar y in the FINAL slot
    """
    n_trajectories = int(n_trajectories)
    trajectory_length = int(trajectory_length)
    num_sources, theta_size = _sample_problem_shapes_np(
        rng,
        n_trajectories,
        cfg,
        fixed_num_sources=fixed_num_sources,
        fixed_source_dim=fixed_source_dim,
    )

    theta_true = np.zeros(
        (n_trajectories, cfg.max_num_sources, cfg.max_source_dim), dtype=np.float32
    )
    observations = np.zeros(
        (n_trajectories, trajectory_length, cfg.max_source_dim + 1), dtype=np.float32
    )

    # Simulation is intentionally a host-side preprocessing step.  The neural training
    # loop still never evaluates the physical likelihood.
    for m in range(n_trajectories):
        S = int(num_sources[m])
        D = int(theta_size[m] // num_sources[m])
        theta_active = sample_prior_np(
            rng, 1, cfg, num_sources=S, source_dim=D
        )[0]
        designs = rng.uniform(
            cfg.design_low,
            cfg.design_high,
            size=(trajectory_length, D),
        ).astype(np.float32)
        mean = source_log_mean_np(theta_active, designs, cfg)
        readings = (
            mean + cfg.observation_noise_std * rng.normal(size=mean.shape)
        ).astype(np.float32)

        theta_true[m, :S, :D] = theta_active
        observations[m, :, :D] = designs
        observations[m, :, -1] = readings

    return {
        "theta_true": theta_true,
        "observations": observations,
        "num_sources": num_sources.astype(np.int32),
        "theta_size": theta_size.astype(np.int32),
    }


def make_batch_np(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    rng: np.random.Generator,
    cfg: BayesTransportConfig = CFG,
    *,
    num_particles: int | None = None,
) -> dict[str, np.ndarray]:
    """Create one heterogeneous minibatch and draw its independent padded prior clouds.

    The prior cloud and theta_true are both sampled from the matching p(theta | S,D),
    but independently.  The prior cloud is an INPUT numerical measure; theta_true is the
    simulator truth and proper-score TARGET.  Keeping those roles separate is central
    to Mode A.
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
        if theta_size_int > cfg.embedding_dim or theta_size_int % S != 0:
            raise ValueError("Invalid theta metadata in dataset.")
        D = theta_size_int // S
        active = sample_prior_np(
            rng, n_particles, cfg, num_sources=S, source_dim=D
        )
        prior_particles[b] = pad_theta_np(active, cfg)

    return {
        "theta_true": dataset["theta_true"][indices].astype(np.float32),
        "observations": dataset["observations"][indices].astype(np.float32),
        "num_sources": batch_num_sources,
        "theta_size": batch_theta_size,
        "prior_particles": prior_particles,
    }


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


#%% 5) Token helpers shared by both Transformers
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _mlp_tokens(layer: eqx.nn.MLP, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _time_layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    """LayerNorm [T,N,H] or [T,R,H] without writing a time-step loop."""
    return jax.vmap(lambda tokens: _layernorm_tokens(layer, tokens))(x)


def _time_linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    """Linear map over the last axis of [T,N,H] without a time-step loop."""
    return jax.vmap(lambda tokens: _linear_tokens(layer, tokens))(x)


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
    """TAMO-style dimension aggregator for one padded (design, outcome) observation.

    TAMO Figure 2 applies scalar-to-vector MLPs dimension-wise, runs a Transformer across
    dimension tokens, then modulates each token element-wise by a learned positional
    vector before mean pooling.  We keep that mechanism.  Unlike TAMO's random pool
    assignment, coordinate slots are stable here because x_1,x_2,... are shared physical
    axes across source-localisation tasks; the learned pool still removes dimensionality
    from the downstream Transformer's interface.
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
        # Figure 2 uses an element-wise product with learned positional tokens.  Starting
        # near one preserves signal scale at initialization while still making each slot
        # learnably distinct.
        self.design_position_pool = 1.0 + 0.02 * jax.random.normal(
            keys[-2], (cfg.max_source_dim, E)
        )
        self.outcome_position = 1.0 + 0.02 * jax.random.normal(keys[-1], (E,))

    def __call__(self, observation: Array, num_sources: Array, theta_size: Array) -> Array:
        source_dim = theta_size // num_sources
        design_values = observation[: self.max_source_dim] / self.design_scale
        outcome_value = (observation[-1:] - self.y_center) / self.y_scale

        design_tokens = _mlp_tokens(self.design_scalar_encoder, design_values[:, None])
        outcome_token = self.outcome_scalar_encoder(outcome_value)
        tokens = jnp.concatenate([design_tokens, outcome_token[None, :]], axis=0)
        valid_design = jnp.arange(self.max_source_dim) < source_dim
        valid = jnp.concatenate([valid_design, jnp.ones((1,), dtype=bool)])

        for block in self.blocks:
            tokens = block(tokens, valid)
        tokens = _layernorm_tokens(self.final_norm, tokens)

        positions = jnp.concatenate(
            [self.design_position_pool, self.outcome_position[None, :]], axis=0
        )
        return _masked_mean(tokens * positions, valid)


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


#%% 6) Causal, prefix-permutation-invariant Likelihood Transformer
class PrefixSummaryBlock(eqx.Module):
    """One learned prefix vector cross-attending to the active observation set.

    The original `likelihood_summary_tokens` dimension has been removed.  With R=1 it
    carried no set-valued information, so the summary is now directly [H].
    """

    query_norm: eqx.nn.LayerNorm
    memory_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    cross_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, *, key: Array):
        cross_key, ff1_key, ff2_key = jax.random.split(key, 3)
        self.query_norm = eqx.nn.LayerNorm(hidden_dim)
        self.memory_norm = eqx.nn.LayerNorm(hidden_dim)
        self.ff_norm = eqx.nn.LayerNorm(hidden_dim)
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=cross_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)

    def __call__(
        self,
        summary: Array,              # [H]
        observation_tokens: Array,   # [T,H]
        prefix_mask: Array,          # [T]
    ) -> Array:
        query = self.query_norm(summary)[None, :]
        memory = _layernorm_tokens(self.memory_norm, observation_tokens)
        cross_mask = prefix_mask[None, :] > 0.5
        summary = summary + self.cross_attention(
            query, memory, memory, mask=cross_mask
        )[0]

        h = self.ff_norm(summary)
        h = jax.nn.gelu(self.ff_in(h))
        h = self.ff_out(h)
        return summary + h


class CausalPrefixLikelihoodTransformer(eqx.Module):
    """Map all padded (x_t,y_t) pairs to all one-vector prefix summaries in one pass.

    Input
    -----
    observations : [T,Dmax+1]
        Padded design coordinates followed by observed_y in the FINAL slot.
    num_sources, theta_size : scalars
        source_dim is reconstructed as theta_size // num_sources.

    Output
    ------
    summaries : [T,H]
        summaries[t] depends only on observations[:t+1] and is permutation invariant
        with respect to reordering those active observations.

    There are still no positional encodings on observation-time tokens.  Dimensional
    identity is handled INSIDE ObservationDimensionEmbedder before each observation is
    compressed to one E-vector.  Prefix cardinality remains explicitly embedded.
    """

    observation_embedder: ObservationDimensionEmbedder
    observation_projection: eqx.nn.Linear
    count_projection: eqx.nn.Linear
    summary_query: Array
    blocks: tuple[PrefixSummaryBlock, ...]
    final_norm: eqx.nn.LayerNorm

    count_scale: float = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.likelihood_depth + 5)
        self.count_scale = float(max(cfg.trajectory_length, 1))
        self.observation_embedder = ObservationDimensionEmbedder(cfg, key=keys[0])
        self.observation_projection = eqx.nn.Linear(
            cfg.embedding_dim, cfg.hidden_dim, key=keys[1]
        )
        self.count_projection = eqx.nn.Linear(1, cfg.hidden_dim, key=keys[2])
        self.summary_query = 0.02 * jax.random.normal(keys[3], (cfg.hidden_dim,))
        self.blocks = tuple(
            PrefixSummaryBlock(
                cfg.hidden_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[4 + i],
            )
            for i in range(cfg.likelihood_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)

    def __call__(self, observations: Array, num_sources: Array, theta_size: Array) -> Array:
        trajectory_length = observations.shape[0]
        embedded_observations = jax.vmap(
            lambda obs: self.observation_embedder(obs, num_sources, theta_size)
        )(observations)                                              # [T,E]
        observation_tokens = _linear_tokens(
            self.observation_projection, embedded_observations
        )                                                            # [T,H]

        # Row t is the membership mask of observations 1,...,t+1.
        prefix_masks = jnp.tril(
            jnp.ones((trajectory_length, trajectory_length), dtype=bool)
        )
        normalized_counts = (
            jnp.arange(1, trajectory_length + 1, dtype=observations.dtype)
            / self.count_scale
        )[:, None]
        count_tokens = jax.vmap(self.count_projection)(normalized_counts)  # [T,H]

        summaries = jnp.broadcast_to(
            self.summary_query[None, :], (trajectory_length, self.summary_query.shape[0])
        ) + count_tokens

        # Each vmap batch element is a different prefix mask.  All T prefixes are
        # compiled/evaluated together; this is not a recurrent scan over time.
        for block in self.blocks:
            summaries = jax.vmap(
                lambda summary_t, mask_t: block(summary_t, observation_tokens, mask_t)
            )(summaries, prefix_masks)

        return jax.vmap(self.final_norm)(summaries)


#%% 7) Posterior Transformer: AdaLN path
class AdaLNParticleBlock(eqx.Module):
    """Permutation-equivariant particle self-attention conditioned by one prefix."""

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, *, key: Array):
        attn_key, ff1_key, ff2_key, mod_key = jax.random.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm(
            hidden_dim, eps=1e-6, use_weight=False, use_bias=False
        )
        self.norm2 = eqx.nn.LayerNorm(
            hidden_dim, eps=1e-6, use_weight=False, use_bias=False
        )
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            use_query_bias=False,
            use_key_bias=False,
            use_value_bias=False,
            use_output_bias=True,
            dropout_p=0.0,
            key=attn_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)

        modulation = eqx.nn.Linear(hidden_dim, 6 * hidden_dim, key=mod_key)
        # AdaLN-zero: identity residual blocks at initialization.
        modulation = eqx.tree_at(
            lambda layer: layer.weight,
            modulation,
            jnp.zeros_like(modulation.weight),
        )
        modulation = eqx.tree_at(
            lambda layer: layer.bias,
            modulation,
            jnp.zeros_like(modulation.bias),
        )
        self.modulation = modulation

    def __call__(self, particles: Array, condition: Array) -> Array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(
            self.modulation(jax.nn.silu(condition)), 6, axis=-1
        )

        h = _modulate(_layernorm_tokens(self.norm1, particles), shift_a, scale_a)
        h = self.attention(h, h, h)
        particles = particles + gate_a[None, :] * h

        h = _modulate(_layernorm_tokens(self.norm2, particles), shift_f, scale_f)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + gate_f[None, :] * h




#%% 8) Posterior Transformer: cross-attention path
class ParticleLikelihoodCrossBlock(eqx.Module):
    """Particle self-attention followed by particle-to-likelihood cross-attention."""

    particle_norm: eqx.nn.LayerNorm
    cross_query_norm: eqx.nn.LayerNorm
    likelihood_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    particle_attention: eqx.nn.MultiheadAttention
    likelihood_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, *, key: Array):
        self_key, cross_key, ff1_key, ff2_key = jax.random.split(key, 4)
        self.particle_norm = eqx.nn.LayerNorm(hidden_dim)
        self.cross_query_norm = eqx.nn.LayerNorm(hidden_dim)
        self.likelihood_norm = eqx.nn.LayerNorm(hidden_dim)
        self.ff_norm = eqx.nn.LayerNorm(hidden_dim)
        self.particle_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=self_key,
        )
        self.likelihood_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=cross_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)

    def __call__(self, particles: Array, likelihood_tokens: Array) -> Array:
        h = _layernorm_tokens(self.particle_norm, particles)
        particles = particles + self.particle_attention(h, h, h)

        query = _layernorm_tokens(self.cross_query_norm, particles)
        memory = _layernorm_tokens(self.likelihood_norm, likelihood_tokens)
        particles = particles + self.likelihood_attention(
            query, memory, memory
        )

        h = _layernorm_tokens(self.ff_norm, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + h




#%% 9) Shared embedding-space particle decoder and the two posterior architectures
class EmbeddingParticleDecoder(eqx.Module):
    """Decode [T,N,H] tokens as residual transports of the same embedded prior cloud."""

    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear
    max_displacement: float = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.max_displacement = cfg.max_embedding_displacement
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)
        output = eqx.nn.Linear(cfg.hidden_dim, cfg.embedding_dim, key=key)

        # Identity transport at initialization is still the natural prior-to-posterior
        # starting point, but identity now means z_post = z_prior in the common E-space.
        output = eqx.tree_at(
            lambda layer: layer.weight, output, jnp.zeros_like(output.weight)
        )
        output = eqx.tree_at(
            lambda layer: layer.bias, output, jnp.zeros_like(output.bias)
        )
        self.displacement_head = output

    def __call__(self, particle_tokens: Array, prior_embeddings: Array) -> Array:
        # particle_tokens [T,N,H], prior_embeddings [N,E]
        particle_tokens = _time_layernorm_tokens(self.final_norm, particle_tokens)
        displacement = self.max_displacement * jnp.tanh(
            _time_linear_tokens(self.displacement_head, particle_tokens)
        )
        return prior_embeddings[None, :, :] + displacement              # [T,N,E]


class AdaLNPosteriorTransformer(eqx.Module):
    """All-prefix posterior transport using AdaLN conditions from one summary per prefix."""

    particle_in: eqx.nn.Linear
    condition_encoder: eqx.nn.MLP
    blocks: tuple[AdaLNParticleBlock, ...]
    decoder: EmbeddingParticleDecoder

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 3)
        self.particle_in = eqx.nn.Linear(cfg.embedding_dim, cfg.hidden_dim, key=keys[0])
        self.condition_encoder = eqx.nn.MLP(
            in_size=cfg.hidden_dim,
            out_size=cfg.hidden_dim,
            width_size=cfg.hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=keys[1],
        )
        self.blocks = tuple(
            AdaLNParticleBlock(
                cfg.hidden_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[2 + i],
            )
            for i in range(cfg.posterior_depth)
        )
        self.decoder = EmbeddingParticleDecoder(cfg, key=keys[-1])

    def __call__(self, prior_embeddings: Array, likelihood_summaries: Array) -> Array:
        base_particles = _linear_tokens(self.particle_in, prior_embeddings)  # [N,H]
        conditions = jax.vmap(self.condition_encoder)(likelihood_summaries)  # [T,H]
        particles = jnp.broadcast_to(
            base_particles[None, :, :],
            (likelihood_summaries.shape[0],) + base_particles.shape,
        )                                                                  # [T,N,H]

        for block in self.blocks:
            particles = jax.vmap(block)(particles, conditions)

        return self.decoder(particles, prior_embeddings)                   # [T,N,E]


class CrossAttentionPosteriorTransformer(eqx.Module):
    """All-prefix posterior transport with direct particle-to-summary cross-attention."""

    particle_in: eqx.nn.Linear
    blocks: tuple[ParticleLikelihoodCrossBlock, ...]
    decoder: EmbeddingParticleDecoder

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 2)
        self.particle_in = eqx.nn.Linear(cfg.embedding_dim, cfg.hidden_dim, key=keys[0])
        self.blocks = tuple(
            ParticleLikelihoodCrossBlock(
                cfg.hidden_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[1 + i],
            )
            for i in range(cfg.posterior_depth)
        )
        self.decoder = EmbeddingParticleDecoder(cfg, key=keys[-1])

    def __call__(self, prior_embeddings: Array, likelihood_summaries: Array) -> Array:
        base_particles = _linear_tokens(self.particle_in, prior_embeddings)  # [N,H]
        particles = jnp.broadcast_to(
            base_particles[None, :, :],
            (likelihood_summaries.shape[0],) + base_particles.shape,
        )                                                                  # [T,N,H]

        # At prefix t, N particle tokens cross-attend to the ONE [H] likelihood vector
        # summarising observations 1:t.  The singleton memory axis is created locally;
        # there is no configurable likelihood-summary-token dimension anymore.
        for block in self.blocks:
            particles = jax.vmap(
                lambda particle_t, summary_t: block(particle_t, summary_t[None, :])
            )(particles, likelihood_summaries)

        return self.decoder(particles, prior_embeddings)                   # [T,N,E]


#%% 10) End-to-end dimension-agnostic two-Transformer model
class ModeAParallelBayesModel(eqx.Module):
    """Observation embedder + Likelihood Transformer + theta embedder + Posterior Transformer."""

    likelihood_transformer: CausalPrefixLikelihoodTransformer
    theta_embedder: ThetaDimensionEmbedder
    posterior_transformer: AdaLNPosteriorTransformer | CrossAttentionPosteriorTransformer
    sigreg: "SIGReg"
    conditioning: str = eqx.field(static=True)

    def __init__(
        self,
        cfg: BayesTransportConfig,
        *,
        conditioning: ConditioningMode,
        key: Array,
    ):
        if conditioning not in {"adaln", "cross_attention"}:
            raise ValueError("conditioning must be 'adaln' or 'cross_attention'.")
        likelihood_key, theta_key, posterior_key = jax.random.split(key, 3)
        self.conditioning = conditioning
        self.likelihood_transformer = CausalPrefixLikelihoodTransformer(
            cfg, key=likelihood_key
        )
        self.theta_embedder = ThetaDimensionEmbedder(cfg, key=theta_key)
        if conditioning == "adaln":
            self.posterior_transformer = AdaLNPosteriorTransformer(
                cfg, key=posterior_key
            )
        else:
            self.posterior_transformer = CrossAttentionPosteriorTransformer(
                cfg, key=posterior_key
            )
        self.sigreg = SIGReg(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            t_max=cfg.sigreg_t_max,
        )

    def encode_theta(self, theta: Array, num_sources: Array, theta_size: Array) -> Array:
        return self.theta_embedder(theta, num_sources, theta_size)

    def __call__(
        self,
        prior_particles: Array,  # [N,Smax,Dmax]
        observations: Array,     # [T,Dmax+1]
        num_sources: Array,      # scalar S
        theta_size: Array,       # scalar S*D
    ) -> tuple[Array, Array, Array]:
        prior_embeddings = jax.vmap(
            lambda theta: self.theta_embedder(theta, num_sources, theta_size)
        )(prior_particles)                                              # [N,E]
        summaries = self.likelihood_transformer(
            observations, num_sources, theta_size
        )                                                               # [T,H]
        posterior = self.posterior_transformer(prior_embeddings, summaries)
        return posterior, summaries, prior_embeddings                    # [T,N,E], [T,H], [N,E]


#%% 11) Embedding-space energy score, SIGReg, and simple posterior diagnostics
class SIGReg(eqx.Module):
    """Epps-Pulley normality regularizer adapted to theta embeddings.

    Input z has shape (T,B,D).  In this notebook we pass the independently re-sampled
    embedded prior clouds as z=[trajectory_in_minibatch, prior_particle, E].  Therefore
    each trajectory is one independent normality-test slice and the sample axis is the
    N prior particles for that trajectory.  This is the appropriate unconditioned latent
    distribution to regularize: posterior clouds are NOT expected to remain N(0,I).

    As in the supplied implementation, the integrated ECF error is multiplied by the
    number of samples B used to estimate the empirical characteristic function.
    """

    knots: int = eqx.field(static=True)
    num_proj: int = eqx.field(static=True)
    t_max: float = eqx.field(static=True)

    def __init__(self, knots: int = 17, num_proj: int = 1024, t_max: float = 3.0):
        self.knots = knots
        self.num_proj = num_proj
        self.t_max = t_max

    def __call__(self, z: Array, key: Array) -> Array:
        """z: (T,B,D) latent embeddings."""
        T, B, D = z.shape

        # Random unit-norm projection directions, re-sampled every call.
        A = jax.random.normal(key, (D, self.num_proj))
        A = A / (jnp.linalg.norm(A, axis=0, keepdims=True) + 1e-12)

        t = jnp.linspace(0.0, self.t_max, self.knots)
        dt = self.t_max / (self.knots - 1)
        weights = jnp.full((self.knots,), 2.0 * dt).at[0].set(dt).at[-1].set(dt)
        window = jnp.exp(-0.5 * t ** 2)
        weights = weights * window
        phi = window  # target real characteristic function of N(0,1)

        h = z @ A                                           # (T,B,num_proj)
        x_t = h[..., None] * t                              # (T,B,num_proj,knots)
        ecf_real = jnp.mean(jnp.cos(x_t), axis=1)           # (T,num_proj,knots)
        ecf_imag = jnp.mean(jnp.sin(x_t), axis=1)
        err = (ecf_real - phi) ** 2 + ecf_imag ** 2
        statistic = jnp.einsum("tpk,k->tp", err, weights) * B
        return statistic.mean()


def energy_score_single(particle_embeddings: Array, target_embedding: Array) -> Array:
    """Exact empirical multivariate energy score directly in R^E.

    For q^N = N^{-1} sum_n delta_{z_n},

        ES(q^N, z*)
          = N^{-1} sum_n ||z_n-z*||
            - (2 N^2)^{-1} sum_{n,m} ||z_n-z_m||.

    The pair term remains O(N^2 E), but physical theta dimensionality no longer changes
    the scorer shape or the Posterior Transformer output head.
    """
    truth_distance = jnp.mean(
        jnp.sqrt(jnp.sum((particle_embeddings - target_embedding[None, :]) ** 2, axis=-1) + 1e-12)
    )
    differences = particle_embeddings[:, None, :] - particle_embeddings[None, :, :]
    squared_distance = jnp.sum(differences**2, axis=-1)
    off_diagonal = 1.0 - jnp.eye(particle_embeddings.shape[0], dtype=particle_embeddings.dtype)
    pairwise_distance = jnp.sum(
        jnp.sqrt(squared_distance + 1e-12) * off_diagonal
    ) / (particle_embeddings.shape[0] ** 2)
    return truth_distance - 0.5 * pairwise_distance


def posterior_mean_rmse_single(particle_embeddings: Array, target_embedding: Array) -> Array:
    """RMSE of the posterior mean in embedding space (physical RMSE comes post-hoc)."""
    return jnp.sqrt(jnp.mean((jnp.mean(particle_embeddings, axis=0) - target_embedding) ** 2))


def posterior_spread_single(particle_embeddings: Array) -> Array:
    """Mean marginal variance in embedding space."""
    return jnp.mean(jnp.var(particle_embeddings, axis=0))


def _trajectory_metrics(
    posterior_sequence: Array,
    target_embedding: Array,
) -> tuple[Array, Array, Array]:
    """Vectorise all per-prefix embedding metrics over T without a Python loop."""
    energy = jax.vmap(lambda p: energy_score_single(p, target_embedding))(posterior_sequence)
    rmse = jax.vmap(lambda p: posterior_mean_rmse_single(p, target_embedding))(posterior_sequence)
    spread = jax.vmap(posterior_spread_single)(posterior_sequence)
    return energy, rmse, spread


def batch_objective(
    model: ModeAParallelBayesModel,
    batch: dict[str, Array],
    sigreg_key: Array,
    cfg: BayesTransportConfig = CFG,
) -> tuple[Array, dict[str, Array]]:
    """Mean Mode-A embedding energy score over B x T, optionally plus SIGReg.

    `predicted` has shape [B,T,N,E].  All prefix losses are therefore available in one
    forward pass and one gradient call.  Observation embedding, theta embedding,
    likelihood summarisation, and posterior transport are all differentiated jointly;
    there is no stop-gradient in this end-to-end path.
    """
    predicted, _, prior_embeddings = jax.vmap(model)(
        batch["prior_particles"],
        batch["observations"],
        batch["num_sources"],
        batch["theta_size"],
    )
    target_embeddings = jax.vmap(model.encode_theta)(
        batch["theta_true"], batch["num_sources"], batch["theta_size"]
    )                                                                  # [B,E]
    energy, rmse, spread = jax.vmap(_trajectory_metrics)(
        predicted, target_embeddings
    )                                                                  # each [B,T]

    energy_loss = jnp.mean(energy)
    if cfg.sigreg_weight > 0.0:
        # prior_embeddings is [B,N,E], exactly matching SIGReg's (T,B,D) convention
        # with heterogeneous trajectories as the outer independent slice and N fresh
        # prior draws as the empirical sample axis.
        sigreg_loss = model.sigreg(prior_embeddings, sigreg_key)
    else:
        sigreg_loss = jnp.asarray(0.0, dtype=energy_loss.dtype)
    loss = energy_loss + cfg.sigreg_weight * sigreg_loss

    metrics = {
        "loss": loss,
        "energy_score": energy_loss,
        "sigreg_loss": sigreg_loss,
        "weighted_sigreg_loss": cfg.sigreg_weight * sigreg_loss,
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
    model: ModeAParallelBayesModel,
    prior_particles: Array,
    observations: Array,
    num_sources: Array,
    theta_size: Array,
) -> tuple[Array, Array, Array]:
    """JIT-compiled trajectory batching; time prefixes remain inside each model call."""
    return jax.vmap(model)(prior_particles, observations, num_sources, theta_size)


@eqx.filter_jit
def evaluation_batch(
    model: ModeAParallelBayesModel,
    batch: dict[str, Array],
    sigreg_key: Array,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Array]:
    _, metrics = batch_objective(model, batch, sigreg_key, cfg)
    return metrics


#%% 12) Evaluation helper with reproducible fresh prior clouds
def evaluate_model(
    model: ModeAParallelBayesModel,
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    *,
    num_particles: int | None = None,
    max_trajectories: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray | float]:
    """Evaluate with fresh reproducible prior clouds and reproducible SIGReg projections."""
    n_total = len(dataset["theta_true"])
    if max_trajectories is not None:
        n_total = min(n_total, int(max_trajectories))
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    eval_seed = cfg.seed + 90_000 if seed is None else int(seed)
    rng = np.random.default_rng(eval_seed)
    base_sigreg_key = jax.random.key(eval_seed + 17)

    scalar_names = [
        "loss",
        "energy_score",
        "sigreg_loss",
        "weighted_sigreg_loss",
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
        batch_np = make_batch_np(dataset, indices, rng, cfg, num_particles=num_particles)
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        sigreg_key = jax.random.fold_in(base_sigreg_key, start)
        host = jax.device_get(evaluation_batch(model, batch, sigreg_key, cfg))
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


#%% 13) Optional exact-likelihood reference posterior for plots only
def reference_posterior_particles_np(
    rng: np.random.Generator,
    observations: np.ndarray,
    prefix_length: int,
    num_sources: int,
    theta_size: int,
    cfg: BayesTransportConfig = CFG,
) -> tuple[np.ndarray, float]:
    """SNIS reference posterior used only after training for visual validation.

    Proposal is exactly p(theta|S,D), so importance weights are proportional to the
    likelihood of the observed prefix.  This function is intentionally NOT a teacher
    and is never called inside the training objective or embedding-decoder objective.
    """
    prefix_length = int(prefix_length)
    S = int(num_sources)
    D = int(theta_size) // S
    proposals = sample_prior_np(
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


#%% 14) Visualisation: architecture schematic
def plot_architecture_schematic(
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
):
    """Visual map of the two architectures, fixed E interface, and parallel T dimension."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.4), constrained_layout=True)

    def draw_box(ax, xy, width, height, text, title=None):
        patch = FancyBboxPatch(
            xy, width, height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.3, facecolor="white", edgecolor="black",
        )
        ax.add_patch(patch)
        label = text if title is None else f"{title}\n{text}"
        ax.text(xy[0] + width / 2, xy[1] + height / 2, label,
                ha="center", va="center", fontsize=9)

    def arrow(ax, start, end, text=""):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.25))
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035,
                    text, ha="center", va="bottom", fontsize=8)

    for ax, mode in zip(axes, ["AdaLN", "Cross-attention"]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"{mode} posterior conditioning — all T prefixes in parallel",
                     loc="left", fontweight="bold")

        draw_box(ax, (0.01, 0.60), 0.15, 0.24,
                 "[x_t,y_t] padded\nvariable D", "observations")
        draw_box(ax, (0.19, 0.58), 0.16, 0.28,
                 f"scalar -> vector\ndimension Transformer\nmasked mean -> E={cfg.embedding_dim}",
                 "observation embedder")
        draw_box(ax, (0.39, 0.56), 0.19, 0.32,
                 "triangular prefix mask\nONE summary vector\nno time positions",
                 "causal prefix-set\nLikelihood Transformer")
        draw_box(ax, (0.62, 0.61), 0.12, 0.22,
                 "shape [T,H]", "prefix summaries")

        draw_box(ax, (0.01, 0.12), 0.15, 0.22,
                 "N iid padded draws\nvariable S,D", "prior cloud")
        draw_box(ax, (0.19, 0.09), 0.16, 0.28,
                 f"canonicalize\ncompact S*D scalars\ndimension Transformer\n-> E={cfg.embedding_dim}",
                 "theta embedder")
        posterior_text = (
            "particle self-attention\nAdaLN(summary_t)\nresidual transport in E"
            if mode == "AdaLN"
            else "particle self-attention\nparticle -> 1 summary\ncross-attention\ntransport in E"
        )
        draw_box(ax, (0.62, 0.08), 0.18, 0.30, posterior_text, "Posterior Transformer")
        draw_box(ax, (0.84, 0.13), 0.15, 0.20,
                 "[T,N,E]\nq_phi(z_theta|x_1:t)", "posterior embeddings")

        arrow(ax, (0.16, 0.72), (0.19, 0.72))
        arrow(ax, (0.35, 0.72), (0.39, 0.72))
        arrow(ax, (0.58, 0.72), (0.62, 0.72), "all t")
        arrow(ax, (0.16, 0.23), (0.19, 0.23))
        arrow(ax, (0.35, 0.23), (0.62, 0.23), "broadcast embedded prior over T")
        arrow(ax, (0.68, 0.61), (0.70, 0.38), "condition per t")
        arrow(ax, (0.80, 0.23), (0.84, 0.23))

    fig.suptitle(
        "Dimension-agnostic Mode A: energy score and posterior transport both live in E-space",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 15) Visualisation: physical source field and one simulated trajectory
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
    observations = np.asarray(trajectory["observations"])
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


#%% 16) Visualisation decoder and prior -> posterior evolution across prefixes
def select_prefixes(trajectory_length: int, n_panels_after_prior: int = 5) -> list[int]:
    values = np.unique(
        np.rint(np.geomspace(1, trajectory_length, n_panels_after_prior)).astype(int)
    )
    if values[-1] != trajectory_length:
        values = np.append(values, trajectory_length)
    while len(values) > n_panels_after_prior:
        values = np.delete(values, 1)
    return values.tolist()


class VisualizationDecoderBlock(eqx.Module):
    """Light Transformer-decoder block: coordinate queries attend to one latent token."""
    self_norm: eqx.nn.LayerNorm
    cross_query_norm: eqx.nn.LayerNorm
    memory_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    cross_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        self_key, cross_key, ff1_key, ff2_key = jax.random.split(key, 4)
        self.self_norm = eqx.nn.LayerNorm(dim)
        self.cross_query_norm = eqx.nn.LayerNorm(dim)
        self.memory_norm = eqx.nn.LayerNorm(dim)
        self.ff_norm = eqx.nn.LayerNorm(dim)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=self_key,
        )
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=cross_key,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=ff2_key)

    def __call__(self, queries: Array, memory: Array) -> Array:
        h = _layernorm_tokens(self.self_norm, queries)
        queries = queries + self.self_attention(h, h, h)
        q = _layernorm_tokens(self.cross_query_norm, queries)
        m = _layernorm_tokens(self.memory_norm, memory)
        queries = queries + self.cross_attention(q, m, m)
        h = _layernorm_tokens(self.ff_norm, queries)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return queries + h


class ThetaVisualizationDecoder(eqx.Module):
    """Post-hoc E -> fixed [S,D] decoder used only for physical visualisation."""
    latent_in: eqx.nn.Linear
    coordinate_queries: Array
    blocks: tuple[VisualizationDecoderBlock, ...]
    final_norm: eqx.nn.LayerNorm
    scalar_head: eqx.nn.Linear
    num_sources: int = eqx.field(static=True)
    source_dim: int = eqx.field(static=True)
    theta_size: int = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.decoder_depth + 4)
        self.num_sources = cfg.num_sources
        self.source_dim = cfg.source_dim
        self.theta_size = cfg.num_sources * cfg.source_dim
        self.latent_in = eqx.nn.Linear(
            cfg.embedding_dim, cfg.decoder_hidden_dim, key=keys[0]
        )
        self.coordinate_queries = 0.02 * jax.random.normal(
            keys[1], (self.theta_size, cfg.decoder_hidden_dim)
        )
        self.blocks = tuple(
            VisualizationDecoderBlock(
                cfg.decoder_hidden_dim,
                cfg.decoder_heads,
                cfg.mlp_ratio * cfg.decoder_hidden_dim,
                key=keys[2 + i],
            )
            for i in range(cfg.decoder_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.decoder_hidden_dim)
        self.scalar_head = eqx.nn.Linear(cfg.decoder_hidden_dim, 1, key=keys[-1])

    def __call__(self, embedding: Array) -> Array:
        memory = self.latent_in(embedding)[None, :]
        queries = self.coordinate_queries
        for block in self.blocks:
            queries = block(queries, memory)
        queries = _layernorm_tokens(self.final_norm, queries)
        values = jax.vmap(self.scalar_head)(queries)[:, 0]
        return values.reshape(self.num_sources, self.source_dim)


def plot_latent_posterior_evolution(
    model: ModeAParallelBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
    title: str = "Latent posterior evolution",
):
    """Training-time snapshot in the first two E coordinates; no decoder is needed."""
    S, D, theta_size = _trajectory_shape(trajectory)
    observations = np.asarray(trajectory["observations"])
    predicted, _, prior_embeddings = model(
        jnp.asarray(prior_particles), jnp.asarray(observations),
        jnp.asarray(S), jnp.asarray(theta_size),
    )
    target_embedding = model.encode_theta(
        jnp.asarray(trajectory["theta_true"]), jnp.asarray(S), jnp.asarray(theta_size)
    )
    predicted = np.asarray(jax.device_get(predicted))
    prior_embeddings = np.asarray(jax.device_get(prior_embeddings))
    target_embedding = np.asarray(jax.device_get(target_embedding))

    prefixes = select_prefixes(len(observations), 5)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.5), constrained_layout=True)
    axes = axes.ravel()
    clouds = [prior_embeddings] + [predicted[t - 1] for t in prefixes]
    labels = ["embedded prior"] + [f"q_phi(z_theta | x_1:{t})" for t in prefixes]
    all_points = np.concatenate([c[:, :2] for c in clouds] + [target_embedding[None, :2]])
    lim = max(2.0, 1.12 * float(np.quantile(np.abs(all_points), 0.995)))

    for ax, cloud, label in zip(axes, clouds, labels):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=13, alpha=0.30, label="latent particles")
        ax.scatter(target_embedding[0], target_embedding[1], marker="*", s=190,
                   edgecolors="black", linewidths=0.8, label="embedded theta*")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.grid(alpha=0.2); ax.set_title(label); ax.legend(fontsize=7)
    fig.suptitle(title + " (first two embedding coordinates)", fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


def plot_posterior_evolution(
    model: ModeAParallelBayesModel,
    decoder: ThetaVisualizationDecoder,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
    title: str = "Posterior evolution",
):
    """Decode latent posterior particles back to the fixed 2-D visualisation problem."""
    S, D, theta_size = _trajectory_shape(trajectory)
    if (S, D) != (cfg.num_sources, cfg.source_dim) or D != 2:
        raise ValueError("Physical posterior plot requires the decoder's fixed 2-D problem.")
    observations = np.asarray(trajectory["observations"])
    predicted, _, prior_embeddings = model(
        jnp.asarray(prior_particles), jnp.asarray(observations),
        jnp.asarray(S), jnp.asarray(theta_size),
    )
    decoded_prior = jax.vmap(decoder)(prior_embeddings)
    decoded_post = jax.vmap(lambda z_t: jax.vmap(decoder)(z_t))(predicted)
    decoded_prior = np.asarray(jax.device_get(decoded_prior))
    decoded_post = np.asarray(jax.device_get(decoded_post))
    theta_true = np.asarray(trajectory["theta_true"])[:S, :D]
    if cfg.canonicalize_particle_sources and S > 1:
        theta_true = canonicalize_sources_np(theta_true)

    prefixes = select_prefixes(len(observations), 5)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.5), constrained_layout=True)
    axes = axes.ravel()
    clouds = [decoded_prior] + [decoded_post[t - 1] for t in prefixes]
    labels = ["decoded embedded prior"] + [f"decoded q_phi(theta | x_1:{t})" for t in prefixes]
    all_points = np.concatenate([c.reshape(-1, 2) for c in clouds] + [theta_true.reshape(-1, 2)])
    lim = max(3.0 * cfg.prior_std, 1.12 * float(np.quantile(np.abs(all_points), 0.995)))

    for panel_index, (ax, cloud, label) in enumerate(zip(axes, clouds, labels)):
        ax.scatter(cloud[..., 0].reshape(-1), cloud[..., 1].reshape(-1),
                   s=13, alpha=0.30, label="decoded source locations")
        ax.scatter(theta_true[:, 0], theta_true[:, 1], marker="*", s=190,
                   edgecolors="black", linewidths=0.8, label="theta*")
        if panel_index > 0:
            t = prefixes[panel_index - 1]
            designs = observations[:t, :D]
            ax.scatter(designs[:, 0], designs[:, 1], marker="x", s=33,
                       alpha=0.65, label="designs seen")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.grid(alpha=0.2); ax.set_title(label); ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 17) Visualisation: learned posterior versus optional likelihood-based reference
def plot_reference_comparison(
    models: dict[str, ModeAParallelBayesModel],
    decoders: dict[str, ThetaVisualizationDecoder],
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
):
    """Compare decoded final latent clouds with an exact-likelihood SNIS reference."""
    S, D, theta_size = _trajectory_shape(trajectory)
    if D != 2:
        raise ValueError("Reference source-marginal plot is a 2-D visual diagnostic.")
    observations = np.asarray(trajectory["observations"])
    rng = np.random.default_rng(cfg.seed + 44_000)
    reference, ess = reference_posterior_particles_np(
        rng, observations, len(observations), S, theta_size, cfg
    )
    learned = {}
    for name, model in models.items():
        posterior_z, _, _ = model(
            jnp.asarray(prior_particles), jnp.asarray(observations),
            jnp.asarray(S), jnp.asarray(theta_size),
        )
        decoded = jax.vmap(decoders[name])(posterior_z[-1])
        learned[name] = np.asarray(jax.device_get(decoded))

    theta_true = np.asarray(trajectory["theta_true"])[:S, :D]
    canonical_truth = (
        canonicalize_sources_np(theta_true)
        if cfg.canonicalize_particle_sources and S > 1 else theta_true
    )
    column_names = list(learned.keys()) + [f"reference SNIS\nESS={ess:.0f}"]
    column_clouds = list(learned.values()) + [reference]
    lim_points = np.concatenate([cloud.reshape(-1, D) for cloud in column_clouds])
    lim = max(3.0 * cfg.prior_std, 1.1 * float(np.quantile(np.abs(lim_points), 0.995)))

    fig, axes = plt.subplots(
        S, len(column_names), figsize=(4.3 * len(column_names), 4.0 * S),
        squeeze=False, constrained_layout=True,
    )
    for source_index in range(S):
        for col, (name, cloud) in enumerate(zip(column_names, column_clouds)):
            ax = axes[source_index, col]
            ax.scatter(cloud[:, source_index, 0], cloud[:, source_index, 1], s=12, alpha=0.25)
            ax.scatter(canonical_truth[source_index, 0], canonical_truth[source_index, 1],
                       marker="*", s=190, edgecolors="black", linewidths=0.8)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            if source_index == 0:
                ax.set_title(name, fontweight="bold")
            ax.set_ylabel(f"canonical source {source_index + 1}")

    fig.suptitle(
        "Final-prefix decoded posterior source marginals versus likelihood-based reference",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 18) Training diagnostics visualisation
def plot_training_diagnostics(
    history: dict[str, list],
    best_epoch: int,
    conditioning: str,
    destination: Path | None = None,
):
    steps = np.arange(1, len(history["step_loss"]) + 1)
    epochs = np.arange(1, len(history["epoch_train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)

    values = np.asarray(history["step_loss"])
    axes[0, 0].plot(steps, values, linewidth=0.70, alpha=0.65, label="total loss")
    axes[0, 0].plot(steps, history["step_energy_score"], linewidth=0.70, alpha=0.65,
                    label="embedding energy score")
    if np.any(np.asarray(history["step_sigreg_loss"]) != 0.0):
        axes[0, 0].plot(steps, history["step_weighted_sigreg_loss"], linewidth=0.65,
                        alpha=0.60, label="weighted SIGReg")
    if len(values) >= 20:
        window = max(5, len(values) // 100)
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(steps[window - 1:], smoothed, linewidth=1.8,
                        label=f"total moving average ({window})")
    axes[0, 0].set_title("Loss terms at every gradient step", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("gradient step")
    axes[0, 0].set_yscale("symlog", linthresh=1e-5)
    axes[0, 0].grid(alpha=0.2); axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(steps, history["step_grad_norm"], linewidth=0.75)
    axes[0, 1].set_title("Gradient norm at every step", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("gradient step"); axes[0, 1].set_yscale("log"); axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(epochs, history["epoch_train_loss"], marker="o", markersize=3, label="train total")
    axes[1, 0].plot(epochs, history["epoch_val_loss"], marker="o", markersize=3, label="validation total")
    axes[1, 0].plot(epochs, history["epoch_val_energy_score"], marker="o", markersize=2,
                    label="validation energy")
    axes[1, 0].axvline(best_epoch, linestyle="--", linewidth=1.0, label=f"best epoch {best_epoch}")
    axes[1, 0].set_title("Per-epoch objective", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].grid(alpha=0.2); axes[1, 0].legend(fontsize=8)

    energy_by_t = np.asarray(history["epoch_val_energy_by_t"])
    selected_epochs = np.unique(
        np.clip(np.rint(np.linspace(0, len(energy_by_t) - 1, 5)).astype(int), 0, len(energy_by_t) - 1)
    )
    prefix_axis = np.arange(1, energy_by_t.shape[1] + 1)
    for epoch_index in selected_epochs:
        axes[1, 1].plot(prefix_axis, energy_by_t[epoch_index], label=f"epoch {epoch_index + 1}")
    axes[1, 1].set_title("Validation embedding energy score by prefix", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("prefix length t"); axes[1, 1].grid(alpha=0.2); axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"Mode-A dimension-agnostic training diagnostics — {conditioning}",
                 fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 19) Training function shared by both architectural variants
def train_variant(
    conditioning: ConditioningMode,
    train_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    fixed_trajectory: dict[str, np.ndarray],
    fixed_prior_particles: np.ndarray,
    run_dir: Path,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Any]:
    """Train one end-to-end dimension-agnostic model; all T prefix losses are parallel.

    The AdaLN and cross-attention variants use this same training function, same mixed-
    dimensional data, same minibatch order, same fresh-prior RNG seed, same embedding
    energy-score + optional SIGReg objective, and same evaluation protocol.  Only the
    posterior-conditioning mechanism changes.
    """
    variant_dir = run_dir / conditioning
    (variant_dir / "plots").mkdir(parents=True, exist_ok=True)
    (variant_dir / "artefacts").mkdir(parents=True, exist_ok=True)

    model_seed_offset = 0 if conditioning == "adaln" else 10_000
    model = ModeAParallelBayesModel(
        cfg, conditioning=conditioning, key=jax.random.key(cfg.seed + model_seed_offset)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(candidate_model, candidate_opt_state, batch, sigreg_key):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            batch_objective, has_aux=True
        )(candidate_model, batch, sigreg_key, cfg)
        params = eqx.filter(candidate_model, eqx.is_array)
        updates, candidate_opt_state = optimizer.update(grads, candidate_opt_state, params)
        candidate_model = eqx.apply_updates(candidate_model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return candidate_model, candidate_opt_state, loss, metrics, grad_norm

    # Keep the same detailed per-gradient-step and per-epoch collection pattern from the
    # original notebook; we simply add the SIGReg terms rather than replacing anything.
    history: dict[str, list] = {
        "step_loss": [],
        "step_energy_score": [],
        "step_sigreg_loss": [],
        "step_weighted_sigreg_loss": [],
        "step_final_energy_score": [],
        "step_mean_rmse": [],
        "step_grad_norm": [],
        "epoch_train_loss": [],
        "epoch_val_loss": [],
        "epoch_val_energy_score": [],
        "epoch_val_sigreg_loss": [],
        "epoch_val_final_energy_score": [],
        "epoch_val_mean_rmse": [],
        "epoch_val_energy_by_t": [],
        "epoch_val_rmse_by_t": [],
        "epoch_val_spread_by_t": [],
    }

    # Snapshot the initial identity transport in latent space.  Physical source plots are
    # intentionally postponed until AFTER the separate visualisation decoder is trained.
    plot_latent_posterior_evolution(
        model, fixed_trajectory, fixed_prior_particles, cfg,
        variant_dir / "plots" / "fixed_trajectory_before_training_latent.png",
        f"{conditioning}: before training (identity transport in E-space)",
    )

    initial_metrics = evaluate_model(model, eval_data, cfg, seed=cfg.seed + 91_000)
    print(
        f"[{conditioning}] initial validation total={initial_metrics['loss']:.6f} | "
        f"ES={initial_metrics['energy_score']:.6f} | SIGReg={initial_metrics['sigreg_loss']:.4f}"
    )

    visualisation_epochs = sorted(
        set(max(1, int(math.ceil(fraction * cfg.epochs / 10.0))) for fraction in range(1, 11))
    )
    rng = np.random.default_rng(cfg.seed + 30_000)  # identical data order for both variants
    sigreg_base_key = jax.random.key(cfg.seed + model_seed_offset + 123_456)
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    training_started_at = time.time()
    n_train = len(train_data["theta_true"])

    for epoch in range(1, cfg.epochs + 1):
        epoch_started_at = time.time()
        order = rng.permutation(n_train)
        train_losses_this_epoch: list[float] = []
        n_steps = n_train // cfg.batch_size
        progress = tqdm(
            range(n_steps),
            desc=f"{conditioning:>15s} epoch {epoch:03d}/{cfg.epochs:03d}",
            dynamic_ncols=True,
            leave=True,
        )

        for batch_index in progress:
            start = batch_index * cfg.batch_size
            stop = start + cfg.batch_size
            indices = order[start:stop]
            batch_np = make_batch_np(train_data, indices, rng, cfg)
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            sigreg_key = jax.random.fold_in(sigreg_base_key, global_step)
            model, opt_state, loss, metrics, grad_norm = train_step(
                model, opt_state, batch, sigreg_key
            )
            host = jax.device_get(metrics)
            host_loss = float(jax.device_get(loss))
            host_grad_norm = float(jax.device_get(grad_norm))
            global_step += 1

            train_losses_this_epoch.append(host_loss)
            history["step_loss"].append(host_loss)
            history["step_energy_score"].append(float(host["energy_score"]))
            history["step_sigreg_loss"].append(float(host["sigreg_loss"]))
            history["step_weighted_sigreg_loss"].append(float(host["weighted_sigreg_loss"]))
            history["step_final_energy_score"].append(float(host["final_energy_score"]))
            history["step_mean_rmse"].append(float(host["posterior_mean_rmse"]))
            history["step_grad_norm"].append(host_grad_norm)
            progress.set_postfix(
                L=f"{host_loss:.4f}", ES=f"{float(host['energy_score']):.4f}",
                SIG=f"{float(host['sigreg_loss']):.2f}", grad=f"{host_grad_norm:.3f}",
            )

        epoch_train_loss = float(np.mean(train_losses_this_epoch))
        val_metrics = evaluate_model(
            model, eval_data, cfg, seed=cfg.seed + 91_000  # identical validation draws every epoch
        )
        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_val_loss"].append(float(val_metrics["loss"]))
        history["epoch_val_energy_score"].append(float(val_metrics["energy_score"]))
        history["epoch_val_sigreg_loss"].append(float(val_metrics["sigreg_loss"]))
        history["epoch_val_final_energy_score"].append(float(val_metrics["final_energy_score"]))
        history["epoch_val_mean_rmse"].append(float(val_metrics["posterior_mean_rmse"]))
        history["epoch_val_energy_by_t"].append(np.asarray(val_metrics["energy_by_t"], dtype=np.float64))
        history["epoch_val_rmse_by_t"].append(np.asarray(val_metrics["rmse_by_t"], dtype=np.float64))
        history["epoch_val_spread_by_t"].append(np.asarray(val_metrics["spread_by_t"], dtype=np.float64))

        save_model(variant_dir / "artefacts" / "model_last.eqx", model)
        if epoch % cfg.save_every_epochs == 0:
            save_model(variant_dir / "artefacts" / f"model_epoch_{epoch:04d}.eqx", model)
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            save_model(variant_dir / "artefacts" / "model_best.eqx", model)

        np.savez_compressed(
            variant_dir / "artefacts" / "history.npz",
            **{name: np.asarray(values) for name, values in history.items()},
        )
        save_json(
            variant_dir / "artefacts" / "training_state.json",
            {
                "conditioning": conditioning,
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "elapsed_seconds": time.time() - training_started_at,
                "objective": "mean embedding-space energy score over B x T + sigreg_weight * SIGReg",
                "sigreg_weight": cfg.sigreg_weight,
            },
        )

        print(
            f"[{conditioning}] epoch {epoch:03d}: "
            f"train total={epoch_train_loss:.6f} | val total={float(val_metrics['loss']):.6f} | "
            f"val ES={float(val_metrics['energy_score']):.6f} | "
            f"SIGReg={float(val_metrics['sigreg_loss']):.3f} | "
            f"final ES={float(val_metrics['final_energy_score']):.6f} | "
            f"embedding RMSE={float(val_metrics['posterior_mean_rmse']):.5f} | "
            f"{time.time() - epoch_started_at:.1f}s"
        )

        if epoch in visualisation_epochs:
            plot_latent_posterior_evolution(
                model, fixed_trajectory, fixed_prior_particles, cfg,
                variant_dir / "plots" / f"fixed_trajectory_epoch_{epoch:04d}_latent.png",
                f"{conditioning}: latent posterior evolution after epoch {epoch}",
            )

    best_model = load_model(
        variant_dir / "artefacts" / "model_best.eqx", cfg, conditioning, key=jax.random.key(0)
    )
    final_metrics = evaluate_model(best_model, eval_data, cfg, seed=cfg.seed + 91_000)
    plot_latent_posterior_evolution(
        best_model, fixed_trajectory, fixed_prior_particles, cfg,
        variant_dir / "plots" / "fixed_trajectory_best_model_latent.png",
        f"{conditioning}: best model (epoch {best_epoch}) in E-space",
    )
    plot_training_diagnostics(
        history, best_epoch, conditioning, variant_dir / "plots" / "training_diagnostics.png"
    )

    print(
        f"[{conditioning}] training complete in "
        f"{datetime.timedelta(seconds=int(time.time() - training_started_at))}; "
        f"best epoch={best_epoch}, val total={best_val_loss:.6f}"
    )
    return {
        "model": best_model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_metrics": final_metrics,
        "variant_dir": variant_dir,
    }


#%% 20) Create the run, heterogeneous datasets, and fixed visualisation trajectory
np.random.seed(CFG.seed)
print("JAX devices:", jax.devices())
print("Configuration:\n", yaml.safe_dump(asdict(CFG), sort_keys=False))

run_dir = make_run_dir(CFG.env_name, CFG.runs_base)
with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
    yaml.safe_dump(asdict(CFG), handle, sort_keys=False)
print("Run directory:", run_dir)

# Precompute complete heterogeneous likelihood trajectories.  The neural training loop
# does not call the simulator again; each batch only adds fresh independent prior particles
# that match each row's stored (num_sources, theta_size) metadata.
train_rng = np.random.default_rng(CFG.seed + 1_000)
eval_rng = np.random.default_rng(CFG.seed + 2_000)
train_data = simulate_mode_a_trajectories(
    train_rng, CFG.n_train_trajectories, CFG.trajectory_length, CFG
)
eval_data = simulate_mode_a_trajectories(
    eval_rng, CFG.n_eval_trajectories, CFG.trajectory_length, CFG
)

print("Training problem-shape counts (S, D):")
train_shapes = np.stack(
    [train_data["num_sources"], train_data["theta_size"] // train_data["num_sources"]], axis=1
)
for shape, count in zip(*np.unique(train_shapes, axis=0, return_counts=True)):
    print(f"  S={int(shape[0])}, D={int(shape[1])}: {int(count)} trajectories")

# Keep one fixed 2-D problem for physical plots and for the post-hoc decoder.  It is
# generated separately so heterogeneous eval_data is free to begin with any shape.
fixed_rng = np.random.default_rng(CFG.seed + 2_500)
fixed_data = simulate_mode_a_trajectories(
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
    "num_sources": fixed_data["num_sources"][0],
    "theta_size": fixed_data["theta_size"][0],
}
fixed_prior_active = sample_prior_np(
    np.random.default_rng(CFG.seed + 3_000),
    CFG.num_particles,
    CFG,
    num_sources=CFG.num_sources,
    source_dim=CFG.source_dim,
)
fixed_prior_particles = pad_theta_np(fixed_prior_active, CFG)
np.savez_compressed(
    run_dir / "artefacts" / "fixed_trajectory.npz",
    theta_true=fixed_trajectory["theta_true"],
    observations=fixed_trajectory["observations"],
    num_sources=fixed_trajectory["num_sources"],
    theta_size=fixed_trajectory["theta_size"],
    prior_particles=fixed_prior_particles,
)

plot_architecture_schematic(CFG, run_dir / "plots" / "architecture_schematic.png")
plot_source_trajectory(
    fixed_trajectory, CFG, run_dir / "plots" / "fixed_trajectory_sensor_field.png"
)


#%% 21) Train BOTH conditioning architectures on the same heterogeneous Mode-A problem
# The variants are separate end-to-end models so their dimension embedders and likelihood
# representations can specialize to their posterior-conditioning mechanism.  Within each
# model, observation embedder + likelihood Transformer + theta embedder + posterior
# Transformer are optimized jointly from the same objective; there are no stop-gradients.
results: dict[str, dict[str, Any]] = {}
for conditioning_name in CFG.architectures_to_train:
    if conditioning_name not in {"adaln", "cross_attention"}:
        raise ValueError(f"Unknown architecture {conditioning_name!r}.")
    results[conditioning_name] = train_variant(
        conditioning_name,  # type: ignore[arg-type]
        train_data,
        eval_data,
        fixed_trajectory,
        fixed_prior_particles,
        run_dir,
        CFG,
    )

models = {name: result["model"] for name, result in results.items()}


#%% 21b) Train lightweight fixed-dimensional visualisation decoders AFTER main training
def _make_fixed_decoder_training_set(
    model: ModeAParallelBayesModel,
    cfg: BayesTransportConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample prior theta, embed with the finished model, and return (z, canonical theta)."""
    rng = np.random.default_rng(seed)
    active = sample_prior_np(
        rng, cfg.decoder_train_samples, cfg,
        num_sources=cfg.num_sources, source_dim=cfg.source_dim,
    )
    targets = (
        canonicalize_sources_np(active)
        if cfg.canonicalize_particle_sources and cfg.num_sources > 1 else active
    ).astype(np.float32)
    padded = pad_theta_np(active, cfg)
    embeddings = []
    batch = max(cfg.decoder_batch_size, 1)

    @eqx.filter_jit
    def encode_batch(theta_batch):
        return jax.vmap(
            lambda theta: model.encode_theta(
                theta,
                jnp.asarray(cfg.num_sources),
                jnp.asarray(cfg.num_sources * cfg.source_dim),
            )
        )(theta_batch)

    for start in range(0, len(padded), batch):
        encoded = encode_batch(jnp.asarray(padded[start:start + batch]))
        embeddings.append(np.asarray(jax.device_get(encoded), dtype=np.float32))
    return np.concatenate(embeddings, axis=0), targets


def train_visualization_decoder(
    model: ModeAParallelBayesModel,
    conditioning: str,
    variant_dir: Path,
    cfg: BayesTransportConfig = CFG,
) -> tuple[ThetaVisualizationDecoder, dict[str, list[float]]]:
    """Train ONLY the small fixed-problem inverse map after end-to-end main training."""
    z_train, theta_train = _make_fixed_decoder_training_set(
        model, cfg, seed=cfg.seed + 1_200_000 + (0 if conditioning == "adaln" else 10_000)
    )
    decoder = ThetaVisualizationDecoder(
        cfg,
        key=jax.random.key(cfg.seed + 1_210_000 + (0 if conditioning == "adaln" else 10_000)),
    )
    optimizer = optax.adamw(learning_rate=cfg.decoder_learning_rate, weight_decay=1e-5)
    opt_state = optimizer.init(eqx.filter(decoder, eqx.is_array))

    @eqx.filter_jit
    def decoder_step(candidate_decoder, candidate_state, z_batch, target_batch):
        def loss_fn(dec):
            predicted = jax.vmap(dec)(z_batch)
            return jnp.mean((predicted - target_batch) ** 2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(candidate_decoder)
        updates, candidate_state = optimizer.update(
            grads, candidate_state, eqx.filter(candidate_decoder, eqx.is_array)
        )
        candidate_decoder = eqx.apply_updates(candidate_decoder, updates)
        return candidate_decoder, candidate_state, loss

    rng = np.random.default_rng(cfg.seed + 1_220_000)
    history = {"epoch_mse": []}
    for epoch in range(1, cfg.decoder_epochs + 1):
        order = rng.permutation(len(z_train))
        losses = []
        for start in range(0, len(order), cfg.decoder_batch_size):
            idx = order[start:start + cfg.decoder_batch_size]
            if len(idx) == 0:
                continue
            decoder, opt_state, loss = decoder_step(
                decoder, opt_state, jnp.asarray(z_train[idx]), jnp.asarray(theta_train[idx])
            )
            losses.append(float(jax.device_get(loss)))
        epoch_mse = float(np.mean(losses))
        history["epoch_mse"].append(epoch_mse)
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.decoder_epochs:
            print(f"[{conditioning}] visual decoder epoch {epoch:03d}: MSE={epoch_mse:.6f}")

    save_visualization_decoder(variant_dir / "artefacts" / "visualization_decoder.eqx", decoder)
    np.savez_compressed(
        variant_dir / "artefacts" / "visualization_decoder_history.npz",
        epoch_mse=np.asarray(history["epoch_mse"]),
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(np.arange(1, len(history["epoch_mse"]) + 1), history["epoch_mse"])
    ax.set_xlabel("decoder epoch"); ax.set_ylabel("physical theta reconstruction MSE")
    ax.set_title(f"Post-hoc visualisation decoder — {conditioning}", fontweight="bold")
    ax.grid(alpha=0.25)
    fig.savefig(variant_dir / "plots" / "visualization_decoder_training.png", dpi=170)
    display(fig); plt.close(fig)
    return decoder, history


visualization_decoders: dict[str, ThetaVisualizationDecoder] = {}
for name, result in results.items():
    decoder, decoder_history = train_visualization_decoder(
        result["model"], name, result["variant_dir"], CFG
    )
    visualization_decoders[name] = decoder
    result["visualization_decoder"] = decoder
    result["visualization_decoder_history"] = decoder_history
    plot_posterior_evolution(
        result["model"], decoder, fixed_trajectory, fixed_prior_particles, CFG,
        result["variant_dir"] / "plots" / "fixed_trajectory_best_model_decoded.png",
        f"{name}: decoded posterior evolution after post-hoc decoder training",
    )


#%% 22) Direct visual comparison with a likelihood-based posterior reference
plot_reference_comparison(
    models,
    visualization_decoders,
    fixed_trajectory,
    fixed_prior_particles,
    CFG,
    run_dir / "plots" / "reference_posterior_comparison.png",
)


#%% 23) Architecture comparison across trajectory length
def plot_architecture_comparison(
    results: dict[str, dict[str, Any]],
    destination: Path | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    for name, result in results.items():
        metrics = result["final_metrics"]
        t = np.arange(1, len(metrics["energy_by_t"]) + 1)
        axes[0].plot(t, metrics["energy_by_t"], marker="o", markersize=3, label=name)
        axes[1].plot(t, metrics["rmse_by_t"], marker="o", markersize=3, label=name)
        axes[2].plot(t, metrics["spread_by_t"], marker="o", markersize=3, label=name)

    axes[0].set_title("Energy score by prefix", fontweight="bold")
    axes[1].set_title("Embedding posterior-mean RMSE by prefix", fontweight="bold")
    axes[2].set_title("Embedding posterior spread by prefix", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("prefix length t")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("energy score")
    axes[1].set_ylabel("embedding RMSE")
    axes[2].set_ylabel("embedding marginal variance")
    fig.suptitle("AdaLN versus cross-attention on the same heterogeneous Mode-A trajectories",
                 fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


plot_architecture_comparison(
    results,
    run_dir / "plots" / "architecture_comparison_by_prefix.png",
)




#%% 24) Numerical theorem check: causality, prefix permutation invariance, particle equivariance
def structural_checks(
    model: ModeAParallelBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, float]:
    """Numerically test the exact symmetries built into the dimension-agnostic architecture.

    1. Causality: perturb future observations and verify outputs through prefix t do not change.
    2. Prefix-set invariance: permute the first t observations and verify final prefix output.
    3. Particle equivariance: permute prior-particle axis, undo it on outputs, verify equality.
    """
    obs = np.asarray(trajectory["observations"], dtype=np.float32)
    prior = np.asarray(prior_particles, dtype=np.float32)
    S, D, theta_size = _trajectory_shape(trajectory)
    rng = np.random.default_rng(cfg.seed + 500_000)
    t = max(2, len(obs) // 2)

    baseline, baseline_summary, _ = model(
        jnp.asarray(prior), jnp.asarray(obs), jnp.asarray(S), jnp.asarray(theta_size)
    )
    baseline = np.asarray(jax.device_get(baseline))
    baseline_summary = np.asarray(jax.device_get(baseline_summary))

    future_perturbed = obs.copy()
    if t < len(obs):
        future_perturbed[t:, :D] = rng.uniform(
            cfg.design_low, cfg.design_high, size=future_perturbed[t:, :D].shape
        )
        future_perturbed[t:, -1] += rng.normal(0.0, 5.0, size=len(obs) - t)
    causal_output, causal_summary, _ = model(
        jnp.asarray(prior), jnp.asarray(future_perturbed), jnp.asarray(S), jnp.asarray(theta_size)
    )
    causal_output = np.asarray(jax.device_get(causal_output))
    causal_summary = np.asarray(jax.device_get(causal_summary))
    causal_error = float(np.max(np.abs(causal_output[:t] - baseline[:t])))
    causal_summary_error = float(np.max(np.abs(causal_summary[:t] - baseline_summary[:t])))

    truncated = obs[:t].copy()
    permutation = rng.permutation(t)
    permuted = truncated[permutation]
    output_a, summary_a, _ = model(
        jnp.asarray(prior), jnp.asarray(truncated), jnp.asarray(S), jnp.asarray(theta_size)
    )
    output_b, summary_b, _ = model(
        jnp.asarray(prior), jnp.asarray(permuted), jnp.asarray(S), jnp.asarray(theta_size)
    )
    output_a = np.asarray(jax.device_get(output_a)); output_b = np.asarray(jax.device_get(output_b))
    summary_a = np.asarray(jax.device_get(summary_a)); summary_b = np.asarray(jax.device_get(summary_b))
    prefix_invariance_error = float(np.max(np.abs(output_a[-1] - output_b[-1])))
    prefix_summary_invariance_error = float(np.max(np.abs(summary_a[-1] - summary_b[-1])))

    particle_perm = rng.permutation(len(prior)); inverse_perm = np.argsort(particle_perm)
    permuted_output, _, _ = model(
        jnp.asarray(prior[particle_perm]), jnp.asarray(obs), jnp.asarray(S), jnp.asarray(theta_size)
    )
    permuted_output = np.asarray(jax.device_get(permuted_output))[:, inverse_perm]
    particle_equivariance_error = float(np.max(np.abs(permuted_output - baseline)))

    return {
        "causal_output_max_abs_error": causal_error,
        "causal_summary_max_abs_error": causal_summary_error,
        "prefix_permutation_output_max_abs_error": prefix_invariance_error,
        "prefix_permutation_summary_max_abs_error": prefix_summary_invariance_error,
        "particle_equivariance_max_abs_error": particle_equivariance_error,
    }


structure_results = {
    name: structural_checks(model, fixed_trajectory, fixed_prior_particles, CFG)
    for name, model in models.items()
}
print("Structural theorem checks:")
for name, checks in structure_results.items():
    print(name, checks)
save_json(run_dir / "artefacts" / "structural_checks.json", structure_results)


def plot_structural_checks(
    structure_results: dict[str, dict[str, float]],
    destination: Path | None = None,
):
    metric_names = list(next(iter(structure_results.values())).keys())
    x = np.arange(len(metric_names)); width = 0.8 / len(structure_results)
    fig, ax = plt.subplots(figsize=(13.5, 5.2), constrained_layout=True)
    for i, (name, values) in enumerate(structure_results.items()):
        heights = [max(values[m], 1e-16) for m in metric_names]
        ax.bar(x + (i - (len(structure_results) - 1) / 2) * width,
               heights, width=width, label=name)
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([
        "causal\nposterior", "causal\nlikelihood", "prefix-set\nposterior",
        "prefix-set\nlikelihood", "particle\nequivariance",
    ])
    ax.set_ylabel("max absolute discrepancy")
    ax.set_title("Architectural identities should be near floating-point precision", fontweight="bold")
    ax.grid(axis="y", alpha=0.25); ax.legend()
    if destination is not None: fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


plot_structural_checks(structure_results, run_dir / "plots" / "structural_theorem_checks.png")


#%% 25) Numerical theorem check: single-global-truth proper-score collapse
def energy_score_np(embeddings: np.ndarray, target_embedding: np.ndarray) -> float:
    return float(jax.device_get(energy_score_single(jnp.asarray(embeddings), jnp.asarray(target_embedding))))


def mode_b_collapse_curve(
    model: ModeAParallelBayesModel,
    theta_star_padded: np.ndarray,
    num_sources: int,
    theta_size: int,
    cfg: BayesTransportConfig = CFG,
    n_particles: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Show the fixed-target energy-score collapse theorem in the LEARNED E-space."""
    S = int(num_sources); D = int(theta_size) // S
    theta_active = np.asarray(theta_star_padded)[:S, :D]
    if cfg.canonicalize_particle_sources and S > 1:
        theta_active = canonicalize_sources_np(theta_active)
    rng = np.random.default_rng(cfg.seed + 600_000)
    base_noise = rng.normal(size=(n_particles, S, D)).astype(np.float32)
    scales = np.concatenate([[0.0], np.geomspace(1e-3, 2.0, 34)])
    target_z = np.asarray(jax.device_get(model.encode_theta(
        jnp.asarray(pad_theta_np(theta_active, cfg)), jnp.asarray(S), jnp.asarray(theta_size)
    )))
    scores = []
    for scale in scales:
        cloud = theta_active[None, :, :] + float(scale) * base_noise
        padded = pad_theta_np(cloud.astype(np.float32), cfg)
        z = jax.vmap(lambda th: model.encode_theta(th, jnp.asarray(S), jnp.asarray(theta_size)))(
            jnp.asarray(padded)
        )
        scores.append(energy_score_np(np.asarray(jax.device_get(z)), target_z))
    return scales, np.asarray(scores)


fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
S_fixed, D_fixed, theta_size_fixed = _trajectory_shape(fixed_trajectory)
for name, model in models.items():
    collapse_scales, collapse_scores = mode_b_collapse_curve(
        model, fixed_trajectory["theta_true"], S_fixed, theta_size_fixed, CFG
    )
    ax.plot(collapse_scales, collapse_scores, marker="o", markersize=3, label=name)
ax.set_xscale("symlog", linthresh=1e-3); ax.set_yscale("symlog", linthresh=1e-6)
ax.set_xlabel("physical cloud scale around one fixed theta*")
ax.set_ylabel("embedding-space energy score against embedded theta*")
ax.set_title("Mode B diagnostic: a fixed embedded target still favors a point mass", fontweight="bold")
ax.grid(alpha=0.25); ax.legend()
fig.savefig(run_dir / "plots" / "mode_b_collapse_theorem.png", dpi=170)
display(fig); plt.close(fig)


#%% 26) Limit study N -> large: particle count, energy score, and runtime
def particle_limit_study(
    models: dict[str, ModeAParallelBayesModel],
    eval_data: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate the same trained set-valued model at several particle counts.

    The architecture has no particle positional encoding and all weights are shared,
    so changing N does not change parameter shapes.  JAX will compile once per new N.
    This is an empirical finite-particle study, not a proof of a mean-field limit.
    """
    study = {}
    for name, model in models.items():
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
        study[name] = {
            "num_particles": np.asarray(cfg.particle_limit_values, dtype=int),
            "final_energy": np.asarray(final_energy),
            "mean_energy": np.asarray(mean_energy),
            "final_rmse": np.asarray(final_rmse),
            "seconds": np.asarray(seconds),
        }
    return study


particle_study = particle_limit_study(models, eval_data, CFG)
fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.5), constrained_layout=True)
for name, values in particle_study.items():
    n = values["num_particles"]
    axes[0].plot(n, values["final_energy"], marker="o", label=name)
    axes[1].plot(n, values["final_rmse"], marker="o", label=name)
    axes[2].plot(n, values["seconds"], marker="o", label=name)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25)
    ax.legend()
axes[0].set_title("Final-prefix energy score")
axes[1].set_title("Final embedding posterior-mean RMSE")
axes[2].set_title("Evaluation wall time")
axes[0].set_xlabel("particles N")
axes[1].set_xlabel("particles N")
axes[2].set_xlabel("particles N")
axes[2].set_ylabel("seconds")
fig.suptitle("Finite-particle limit study: embedding accuracy and the O(N^2 E) cost pressure",
             fontsize=14, fontweight="bold")
fig.savefig(run_dir / "plots" / "particle_limit_study.png", dpi=170)
display(fig)
plt.close(fig)




#%% 27) Limit study T -> larger: within-horizon and out-of-horizon prefix behaviour
long_eval_rng = np.random.default_rng(CFG.seed + 800_000)
long_eval_data = simulate_mode_a_trajectories(
    long_eval_rng,
    CFG.limit_eval_trajectories,
    CFG.long_trajectory_length,
    CFG,
)

long_trajectory_study: dict[str, dict[str, np.ndarray | float]] = {}
for name, model in models.items():
    long_trajectory_study[name] = evaluate_model(
        model,
        long_eval_data,
        CFG,
        max_trajectories=CFG.limit_eval_trajectories,
        seed=CFG.seed + 801_000,
    )

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
for name, metrics in long_trajectory_study.items():
    t = np.arange(1, len(metrics["energy_by_t"]) + 1)
    axes[0].plot(t, metrics["energy_by_t"], label=name)
    axes[1].plot(t, metrics["rmse_by_t"], label=name)
    axes[2].plot(t, metrics["spread_by_t"], label=name)
for ax in axes:
    ax.axvline(CFG.trajectory_length, linestyle="--", linewidth=1.0,
               label="training horizon")
    ax.set_xlabel("prefix length t")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
axes[0].set_title("Energy score")
axes[1].set_title("Embedding posterior-mean RMSE")
axes[2].set_title("Embedding posterior spread")
fig.suptitle(
    "Trajectory-length study: solid region is trained horizon; right side is extrapolation",
    fontsize=14,
    fontweight="bold",
)
fig.savefig(run_dir / "plots" / "trajectory_length_limit_study.png", dpi=170)
display(fig)
plt.close(fig)




#%% 28) Limit study M -> large: empirical trajectory-average convergence
def per_trajectory_final_energy(
    model: ModeAParallelBayesModel,
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    *,
    seed: int,
) -> np.ndarray:
    """Return one final-prefix embedding ES per independent Mode-A trajectory."""
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, len(dataset["theta_true"]), cfg.batch_size):
        stop = min(start + cfg.batch_size, len(dataset["theta_true"]))
        indices = np.arange(start, stop)
        batch_np = make_batch_np(dataset, indices, rng, cfg)
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        predicted, _, _ = predict_batch(
            model,
            batch["prior_particles"], batch["observations"],
            batch["num_sources"], batch["theta_size"],
        )
        targets = jax.vmap(model.encode_theta)(
            batch["theta_true"], batch["num_sources"], batch["theta_size"]
        )
        final_posteriors = predicted[:, -1]
        batch_scores = jax.vmap(energy_score_single)(final_posteriors, targets)
        values.append(np.asarray(jax.device_get(batch_scores), dtype=np.float64))
    return np.concatenate(values)


mc_pool_rng = np.random.default_rng(CFG.seed + 900_000)
mc_pool_size = max(CFG.trajectory_mc_values)
mc_pool = simulate_mode_a_trajectories(
    mc_pool_rng, mc_pool_size, CFG.trajectory_length, CFG
)
trajectory_mc_study: dict[str, dict[str, np.ndarray]] = {}
for name, model in models.items():
    scores = per_trajectory_final_energy(model, mc_pool, CFG, seed=CFG.seed + 901_000)
    rng = np.random.default_rng(CFG.seed + 902_000)
    scores = scores[rng.permutation(len(scores))]
    means, lower, upper = [], [], []
    for m in CFG.trajectory_mc_values:
        sample = scores[:m]
        mean = float(np.mean(sample))
        se = float(np.std(sample, ddof=1) / math.sqrt(m)) if m > 1 else 0.0
        means.append(mean); lower.append(mean - 1.96 * se); upper.append(mean + 1.96 * se)
    trajectory_mc_study[name] = {
        "M": np.asarray(CFG.trajectory_mc_values, dtype=int),
        "mean": np.asarray(means), "lower": np.asarray(lower), "upper": np.asarray(upper),
    }

fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
for name, values in trajectory_mc_study.items():
    ax.plot(values["M"], values["mean"], marker="o", label=name)
    ax.fill_between(values["M"], values["lower"], values["upper"], alpha=0.16)
ax.set_xscale("log", base=2)
ax.set_xlabel("independent evaluation trajectories M")
ax.set_ylabel("empirical mean final-prefix embedding energy score")
ax.set_title("M -> large: Monte Carlo estimate of population risk stabilises", fontweight="bold")
ax.grid(alpha=0.25); ax.legend()
fig.savefig(run_dir / "plots" / "trajectory_count_limit_study.png", dpi=170)
display(fig); plt.close(fig)


#%% 29) Finite prior-cloud stability: repeated prior draws for the SAME observations
def prior_cloud_stability_study(
    models: dict[str, ModeAParallelBayesModel],
    trajectory: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, dict[str, np.ndarray]]:
    """How much does the FINAL EMBEDDING posterior mean move when prior cloud is re-drawn?"""
    observations = np.asarray(trajectory["observations"])
    S, D, theta_size = _trajectory_shape(trajectory)
    study = {}
    for name, model in models.items():
        stds = []
        for n_particles in cfg.particle_limit_values:
            means = []
            for repeat in range(cfg.prior_resample_repeats):
                rng = np.random.default_rng(cfg.seed + 1_000_000 + 1000 * n_particles + repeat)
                active_prior = sample_prior_np(
                    rng, n_particles, cfg, num_sources=S, source_dim=D
                )
                prior = pad_theta_np(active_prior, cfg)
                posterior, _, _ = model(
                    jnp.asarray(prior), jnp.asarray(observations),
                    jnp.asarray(S), jnp.asarray(theta_size),
                )
                final = np.asarray(jax.device_get(posterior[-1]))
                means.append(final.mean(axis=0))
            means = np.stack(means)
            stds.append(float(np.sqrt(np.mean(np.var(means, axis=0, ddof=1)))))
        study[name] = {
            "num_particles": np.asarray(cfg.particle_limit_values, dtype=int),
            "posterior_mean_sd_across_prior_clouds": np.asarray(stds),
        }
    return study


prior_cloud_study = prior_cloud_stability_study(models, fixed_trajectory, CFG)
fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
for name, values in prior_cloud_study.items():
    ax.plot(values["num_particles"], values["posterior_mean_sd_across_prior_clouds"],
            marker="o", label=name)
ax.set_xscale("log", base=2)
ax.set_xlabel("prior particles N")
ax.set_ylabel("RMS SD of embedding posterior mean across fresh prior clouds")
ax.set_title("Finite-prior representation stability for fixed observed data", fontweight="bold")
ax.grid(alpha=0.25); ax.legend()
fig.savefig(run_dir / "plots" / "prior_cloud_stability.png", dpi=170)
display(fig); plt.close(fig)


#%% 30) Causal truncation consistency: full T versus running only the first t observations
def truncation_consistency_study(
    model: ModeAParallelBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct check that future points are not needed to compute prefix t in E-space."""
    observations = np.asarray(trajectory["observations"])
    S, D, theta_size = _trajectory_shape(trajectory)
    full, _, _ = model(
        jnp.asarray(prior_particles), jnp.asarray(observations),
        jnp.asarray(S), jnp.asarray(theta_size),
    )
    full = np.asarray(jax.device_get(full))
    prefix_values = select_prefixes(len(observations), 6)
    errors = []
    for t in prefix_values:
        truncated, _, _ = model(
            jnp.asarray(prior_particles), jnp.asarray(observations[:t]),
            jnp.asarray(S), jnp.asarray(theta_size),
        )
        truncated = np.asarray(jax.device_get(truncated))
        errors.append(float(np.max(np.abs(full[t - 1] - truncated[-1]))))
    return np.asarray(prefix_values), np.asarray(errors)


fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
for name, model in models.items():
    t_values, errors = truncation_consistency_study(model, fixed_trajectory, fixed_prior_particles)
    ax.plot(t_values, np.maximum(errors, 1e-16), marker="o", label=name)
ax.set_yscale("log"); ax.set_xlabel("prefix length t")
ax.set_ylabel("max |full-run z_q,t - truncated-run z_q,t|")
ax.set_title("Parallel causal computation agrees with separately truncated inference", fontweight="bold")
ax.grid(alpha=0.25); ax.legend()
fig.savefig(run_dir / "plots" / "causal_truncation_consistency.png", dpi=170)
display(fig); plt.close(fig)


#%% 31) Save limit-study arrays and final summary
for study_name, study in {
    "particle_limit": particle_study,
    "trajectory_mc": trajectory_mc_study,
    "prior_cloud_stability": prior_cloud_study,
}.items():
    flat_payload = {}
    for architecture, values in study.items():
        for metric_name, value in values.items():
            flat_payload[f"{architecture}__{metric_name}"] = np.asarray(value)
    np.savez_compressed(run_dir / "artefacts" / f"{study_name}.npz", **flat_payload)

summary = {
    "objective": "embedding-space energy score + optional SIGReg",
    "sigreg_weight": CFG.sigreg_weight,
    "mode": "Mode A: theta* fixed within trajectory, re-drawn across trajectories",
    "dimension_agnostic": True,
    "train_num_sources_range": [CFG.min_num_sources, CFG.max_num_sources],
    "train_source_dim_range": [CFG.min_source_dim, CFG.max_source_dim],
    "embedding_dim": CFG.embedding_dim,
    "max_theta_size": CFG.max_num_sources * CFG.max_source_dim,
    "parallel_prefix_training": True,
    "trajectory_length": CFG.trajectory_length,
    "num_particles": CFG.num_particles,
    "posthoc_visualization_decoder_problem": [CFG.num_sources, CFG.source_dim],
    "architectures": {},
}
for name, result in results.items():
    decoder_history = result.get("visualization_decoder_history", {"epoch_mse": [np.nan]})
    summary["architectures"][name] = {
        "best_epoch": int(result["best_epoch"]),
        "best_val_loss": float(result["best_val_loss"]),
        "decoder_final_mse": float(decoder_history["epoch_mse"][-1]),
        "final_metrics": {
            key: float(value)
            for key, value in result["final_metrics"].items()
            if np.ndim(value) == 0
        },
    }
save_json(run_dir / "artefacts" / "final_summary.json", summary)

print("\nFinal dimension-agnostic Mode-A summary")
print(json.dumps(summary, indent=2))
print("All artefacts saved under:", run_dir)

