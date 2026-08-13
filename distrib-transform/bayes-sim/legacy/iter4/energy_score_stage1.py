#%% 1) Imports, configuration, and experiment conventions
"""Mode-A Bayesian source localisation with parallel prefix training.

This notebook-style file implements the Mode A construction from the accompanying
technical note for a source-location inverse problem.

For every simulated trajectory m we draw a NEW generating parameter

    theta_m^* ~ p(theta),

hold that theta_m^* fixed while simulating the whole trajectory

    (x_{m,1}, y_{m,1}), ..., (x_{m,T}, y_{m,T}) ~ p(x_{1:T}, y_{1:T} | theta_m^*),

and train on every prefix x_{1:t}, y_{1:t}.  The same theta_m^* is therefore the
proper-score target for all t inside one trajectory, but theta_m^* is re-drawn
between trajectories.  This is the Bayes-consistent "fixed within a trajectory"
case, not the single-global-truth collapse case.

The code is deliberately focused on exactly one training objective: the multivariate
energy score.  There is no ELBO, KDE objective, sliced-Wasserstein objective,
approximate-posterior teacher, or kinetic penalty in the training loss.

Two end-to-end architectures are trained and compared:

1. AdaLN conditioning
   observations -> causal prefix-set Likelihood Transformer -> prefix summaries
   prior particles + pooled summary_t -> AdaLN Posterior Transformer -> q_phi(theta|x_1:t)

2. Cross-attention conditioning
   observations -> causal prefix-set Likelihood Transformer -> prefix summary tokens
   prior particles cross-attend directly to the R summary tokens for prefix t.

The important parallelism is across prefixes.  The Likelihood Transformer computes
all t=1,...,T prefix representations in one JAX program, and the Posterior Transformer
maps the same prior cloud to all T posterior clouds in one JAX program.  There is no
posterior recurrence theta_t -> theta_{t+1} and no Python loop over t in the model or
loss.  Python loops remain only over Transformer depth, minibatches, epochs, and
high-level diagnostic sweeps.

Notation used in arrays
-----------------------
B : number of trajectories in a minibatch
T : trajectory length / number of scored prefixes
N : number of prior/output particles
S : number of exchangeable physical sources
R : number of likelihood-summary tokens per prefix
H : hidden dimension

theta_true          [B, S, 2]
observations         [B, T, 3]      = concat(sensor_xy, observed_y)
prior_particles      [B, N, S, 2]   iid from p(theta), independent of theta_true
likelihood_summaries [B, T, R, H]   each t uses only observations 1:t
posterior_particles  [B, T, N, S, 2]
energy_by_t          [B, T]

The observation pairs are generated in advance.  The neural model never calls the
likelihood during training; it sees only the concatenated (x, y) observations and the
proper-score target theta_true.  The known likelihood is used again only in an
OPTIONAL reference-posterior diagnostic after training.
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
    env_name: str = "energy_score"
    seed: int = 2030
    runs_base: str = "./runs"

    # Source-localisation simulator.
    num_sources: int = 2
    prior_std: float = 1.0
    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.30

    # Mode-A trajectory and particle counts.
    trajectory_length: int = 128
    num_particles: int = 64
    n_train_trajectories: int = 4096
    n_eval_trajectories: int = 256
    batch_size: int = 16

    # Likelihood Transformer.  R > 1 makes the non-AdaLN cross-attention path
    # genuinely set-valued rather than cross-attention to a single pooled vector.
    hidden_dim: int = 96
    heads: int = 4
    mlp_ratio: int = 4
    likelihood_depth: int = 2
    likelihood_summary_tokens: int = 1
    pair_encoder_depth: int = 2

    # Posterior particle Transformer.
    posterior_depth: int = 3
    max_particle_displacement: float = 6.0
    canonicalize_particle_sources: bool = True
    architectures_to_train: tuple[str, ...] = ("adaln", "cross_attention")

    # Observation normalisation.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Optimisation.  The loss is exactly the mean energy score over B x T.
    epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0

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
    long_trajectory_length: int = 48
    trajectory_mc_values: tuple[int, ...] = (8, 16, 32, 64, 128, 192)
    prior_resample_repeats: int = 12


# One default instantiation only: no second CFG = BayesTransportConfig(...) override block.
CFG = BayesTransportConfig()


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


#%% 3) Prior and source-localisation simulator
def sample_prior_np(
    rng: np.random.Generator,
    n: int,
    cfg: BayesTransportConfig = CFG,
) -> np.ndarray:
    """Draw n independent theta ~ p(theta).

    Each theta contains S exchangeable two-dimensional source positions, so the
    returned shape is [n, S, 2].  This one sampler is used both for trajectory truths
    and for independent input prior particles; their roles are kept distinct by where
    the samples are drawn in the data pipeline.
    """
    return rng.normal(
        0.0,
        cfg.prior_std,
        size=(int(n), cfg.num_sources, 2),
    ).astype(np.float32)


def source_log_mean_np(
    theta: np.ndarray,
    designs: np.ndarray,
    cfg: BayesTransportConfig = CFG,
) -> np.ndarray:
    """Forward-model mean E[y | theta, x] on the log-intensity scale.

    This is intentionally the single source-field function in the notebook.  We do
    not maintain separate NumPy/JAX `source_log_likelihood` variants because training
    does not evaluate a likelihood.  Broadcasting supports, for example:

      theta   [S,2],     designs [T,2]   -> [T]
      theta   [B,S,2],   designs [B,T,2] -> [B,T]
      theta   [P,S,2],   designs [T,2]   -> [P,T]
      theta   [S,2],     designs [G,G,2] -> [G,G]
    """
    theta = np.asarray(theta, dtype=np.float64)
    designs = np.asarray(designs, dtype=np.float64)
    theta_expanded = np.expand_dims(theta, axis=-3)      # ... x 1 x S x 2
    design_expanded = np.expand_dims(designs, axis=-2)   # ... x T x 1 x 2
    dist_sq = np.sum((theta_expanded - design_expanded) ** 2, axis=-1)
    intensity = cfg.background + np.sum(
        cfg.source_strength / (cfg.softening + dist_sq), axis=-1
    )
    return np.log(intensity)


def simulate_mode_a_trajectories(
    rng: np.random.Generator,
    n_trajectories: int,
    trajectory_length: int,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, np.ndarray]:
    """Generate complete Mode-A trajectories before neural-network training.

    Critical sampling provenance
    ----------------------------
    1. theta_true[m] is drawn once from p(theta).
    2. All T sensor readings in row m are simulated conditional on that SAME theta.
    3. The next row draws a fresh theta_true[m+1] from p(theta).
    4. Prior particles are deliberately NOT stored here.  They are drawn independently
       from p(theta) when a minibatch is formed, so the network sees fresh numerical
       representations of the same prior over training epochs.

    observations[..., :2] are design/sensor locations x_t.
    observations[..., 2:] are scalar outcomes y_t.
    """
    n_trajectories = int(n_trajectories)
    trajectory_length = int(trajectory_length)
    theta_true = sample_prior_np(rng, n_trajectories, cfg)
    designs = rng.uniform(
        cfg.design_low,
        cfg.design_high,
        size=(n_trajectories, trajectory_length, 2),
    ).astype(np.float32)
    mean = source_log_mean_np(theta_true, designs, cfg)
    readings = (
        mean
        + cfg.observation_noise_std
        * rng.normal(size=mean.shape)
    ).astype(np.float32)
    observations = np.concatenate([designs, readings[..., None]], axis=-1).astype(np.float32)
    return {
        "theta_true": theta_true.astype(np.float32),
        "observations": observations,
    }


def make_batch_np(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    rng: np.random.Generator,
    cfg: BayesTransportConfig = CFG,
    *,
    num_particles: int | None = None,
) -> dict[str, np.ndarray]:
    """Create one minibatch and draw its independent prior clouds.

    The prior cloud and theta_true are both sampled from p(theta), but independently.
    The prior cloud is an INPUT numerical measure; theta_true is the simulator truth
    and proper-score TARGET.  Keeping those roles separate is central to Mode A.
    """
    indices = np.asarray(indices, dtype=np.int64)
    n_particles = cfg.num_particles if num_particles is None else int(num_particles)
    batch_size = len(indices)
    prior_particles = sample_prior_np(rng, batch_size * n_particles, cfg).reshape(
        batch_size, n_particles, cfg.num_sources, 2
    )
    return {
        "theta_true": dataset["theta_true"][indices].astype(np.float32),
        "observations": dataset["observations"][indices].astype(np.float32),
        "prior_particles": prior_particles.astype(np.float32),
    }


#%% 4) Source-label symmetry helpers
def canonicalize_sources_np(theta: np.ndarray) -> np.ndarray:
    """Sort exchangeable sources by x-coordinate inside each theta sample."""
    theta = np.asarray(theta)
    order = np.argsort(theta[..., 0], axis=-1)
    return np.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_sources_jax(theta: Array) -> Array:
    order = jnp.argsort(theta[..., 0], axis=-1)
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


#%% 6) Causal, prefix-permutation-invariant Likelihood Transformer
class PrefixSummaryBlock(eqx.Module):
    """Transformer block for R learned summary queries attending to one prefix set.

    There are no positional encodings on observation tokens.  For a fixed prefix mask,
    the cross-attention output is therefore invariant to a joint permutation of the
    allowed observation keys/values.  The mask controls membership of x_{1:t}; it does
    not assign semantic positions within that prefix.
    """

    query_norm: eqx.nn.LayerNorm
    memory_norm: eqx.nn.LayerNorm
    self_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    cross_attention: eqx.nn.MultiheadAttention
    self_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, *, key: Array):
        cross_key, self_key, ff1_key, ff2_key = jax.random.split(key, 4)
        self.query_norm = eqx.nn.LayerNorm(hidden_dim)
        self.memory_norm = eqx.nn.LayerNorm(hidden_dim)
        self.self_norm = eqx.nn.LayerNorm(hidden_dim)
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

    def __call__(
        self,
        summaries: Array,       # [R,H]
        observation_tokens: Array,  # [T,H]
        prefix_mask: Array,     # [T]
    ) -> Array:
        query = _layernorm_tokens(self.query_norm, summaries)
        memory = _layernorm_tokens(self.memory_norm, observation_tokens)
        cross_mask = jnp.broadcast_to(
            prefix_mask[None, :] > 0.5,
            (summaries.shape[0], observation_tokens.shape[0]),
        )
        summaries = summaries + self.cross_attention(
            query, memory, memory, mask=cross_mask
        )

        h = _layernorm_tokens(self.self_norm, summaries)
        summaries = summaries + self.self_attention(h, h, h)

        h = _layernorm_tokens(self.ff_norm, summaries)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return summaries + h


class CausalPrefixLikelihoodTransformer(eqx.Module):
    """Map all precomputed (x_t,y_t) pairs to all prefix summaries in one pass.

    Input
    -----
    observations : [T,3]
        [sensor_x, sensor_y, observed_y] for the complete simulated trajectory.

    Output
    ------
    summaries : [T,R,H]
        summaries[t] depends only on observations[:t+1].  Each R-token set is
        permutation invariant with respect to reordering those active observations.

    Why not a standard causal self-attention encoder?
    -------------------------------------------------
    In a multilayer causal sequence Transformer, early token states depend on smaller
    prefixes, so the final token can indirectly encode the order of observations.
    That would contradict the desired set semantics.  Here learned summary queries
    cross-attend directly to independently encoded observation tokens under a
    triangular membership mask.  This retains causality without smuggling in order.

    Prefix cardinality is explicitly embedded.  Attention is a normalized average,
    so without a count signal it could not distinguish one observation from repeated
    identical observations, even though Bayes' rule certainly can.
    """

    pair_encoder: eqx.nn.MLP
    count_projection: eqx.nn.Linear
    summary_queries: Array
    blocks: tuple[PrefixSummaryBlock, ...]
    final_norm: eqx.nn.LayerNorm

    design_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)
    count_scale: float = eqx.field(static=True)
    num_summary_tokens: int = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.likelihood_depth + 4)
        self.design_scale = max(abs(cfg.design_low), abs(cfg.design_high), 1.0)
        self.y_center = cfg.y_center
        self.y_scale = max(cfg.y_scale, 1e-6)
        self.count_scale = float(max(cfg.trajectory_length, 1))
        self.num_summary_tokens = cfg.likelihood_summary_tokens

        self.pair_encoder = eqx.nn.MLP(
            in_size=3,
            out_size=cfg.hidden_dim,
            width_size=cfg.hidden_dim,
            depth=cfg.pair_encoder_depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=keys[0],
        )
        self.count_projection = eqx.nn.Linear(1, cfg.hidden_dim, key=keys[1])
        self.summary_queries = 0.02 * jax.random.normal(
            keys[2], (cfg.likelihood_summary_tokens, cfg.hidden_dim)
        )
        self.blocks = tuple(
            PrefixSummaryBlock(
                cfg.hidden_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[3 + i],
            )
            for i in range(cfg.likelihood_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)

    def __call__(self, observations: Array) -> Array:
        trajectory_length = observations.shape[0]
        normalized = jnp.concatenate(
            [
                observations[:, :2] / self.design_scale,
                (observations[:, 2:3] - self.y_center) / self.y_scale,
            ],
            axis=-1,
        )
        observation_tokens = _mlp_tokens(self.pair_encoder, normalized)  # [T,H]

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
            self.summary_queries[None, :, :],
            (trajectory_length, self.num_summary_tokens, self.summary_queries.shape[-1]),
        )
        summaries = summaries + count_tokens[:, None, :]

        # Each vmap batch element is a different prefix mask.  All T prefixes are
        # compiled/evaluated together; this is not a recurrent scan over time.
        for block in self.blocks:
            summaries = jax.vmap(
                lambda summary_t, mask_t: block(
                    summary_t, observation_tokens, mask_t
                )
            )(summaries, prefix_masks)

        return _time_layernorm_tokens(self.final_norm, summaries)


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


#%% 9) Shared particle decoder and the two posterior architectures
class ParticleDecoder(eqx.Module):
    """Decode [T,N,H] tokens as residual transports of the same prior cloud."""

    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear

    num_sources: int = eqx.field(static=True)
    theta_dim: int = eqx.field(static=True)
    max_displacement: float = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.num_sources = cfg.num_sources
        self.theta_dim = 2 * cfg.num_sources
        self.max_displacement = cfg.max_particle_displacement
        self.canonicalize = cfg.canonicalize_particle_sources
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)
        output = eqx.nn.Linear(cfg.hidden_dim, self.theta_dim, key=key)

        # Identity transport at initialization is a natural prior-to-posterior starting
        # point.  The learned likelihood-conditioning path must earn every displacement.
        output = eqx.tree_at(
            lambda layer: layer.weight,
            output,
            jnp.zeros_like(output.weight),
        )
        output = eqx.tree_at(
            lambda layer: layer.bias,
            output,
            jnp.zeros_like(output.bias),
        )
        self.displacement_head = output

    def __call__(self, particle_tokens: Array, flat_prior: Array) -> Array:
        # particle_tokens [T,N,H], flat_prior [N,d_theta]
        particle_tokens = _time_layernorm_tokens(self.final_norm, particle_tokens)
        displacement = self.max_displacement * jnp.tanh(
            _time_linear_tokens(self.displacement_head, particle_tokens)
        )
        transported = (flat_prior[None, :, :] + displacement).reshape(
            particle_tokens.shape[0],
            flat_prior.shape[0],
            self.num_sources,
            2,
        )
        if self.canonicalize and self.num_sources > 1:
            transported = canonicalize_sources_jax(transported)
        return transported


def _prepare_prior_jax(
    prior_particles: Array,
    num_sources: int,
    canonicalize: bool,
) -> tuple[Array, Array]:
    if canonicalize and num_sources > 1:
        prior_particles = canonicalize_sources_jax(prior_particles)
    flat_prior = prior_particles.reshape(prior_particles.shape[0], 2 * num_sources)
    return prior_particles, flat_prior


class AdaLNPosteriorTransformer(eqx.Module):
    """All-prefix posterior transport using AdaLN conditions from likelihood summaries."""

    particle_in: eqx.nn.Linear
    condition_encoder: eqx.nn.MLP
    blocks: tuple[AdaLNParticleBlock, ...]
    decoder: ParticleDecoder

    num_sources: int = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 3)
        self.num_sources = cfg.num_sources
        self.canonicalize = cfg.canonicalize_particle_sources
        self.particle_in = eqx.nn.Linear(2 * cfg.num_sources, cfg.hidden_dim, key=keys[0])
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
        self.decoder = ParticleDecoder(cfg, key=keys[-1])

    def __call__(self, prior_particles: Array, likelihood_summaries: Array) -> Array:
        _, flat_prior = _prepare_prior_jax(
            prior_particles, self.num_sources, self.canonicalize
        )
        base_particles = _linear_tokens(self.particle_in, flat_prior)  # [N,H]

        # Pool only across the R summary tokens of the SAME prefix.  No information from
        # another t is needed because summaries[t] already represents observations 1:t.
        pooled = jnp.mean(likelihood_summaries, axis=1)                # [T,H]
        conditions = jax.vmap(self.condition_encoder)(pooled)         # [T,H]
        particles = jnp.broadcast_to(
            base_particles[None, :, :],
            (likelihood_summaries.shape[0],) + base_particles.shape,
        )                                                             # [T,N,H]

        for block in self.blocks:
            particles = jax.vmap(block)(particles, conditions)

        return self.decoder(particles, flat_prior)                     # [T,N,S,2]


class CrossAttentionPosteriorTransformer(eqx.Module):
    """All-prefix posterior transport with direct particle-to-likelihood cross-attention."""

    particle_in: eqx.nn.Linear
    blocks: tuple[ParticleLikelihoodCrossBlock, ...]
    decoder: ParticleDecoder

    num_sources: int = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 2)
        self.num_sources = cfg.num_sources
        self.canonicalize = cfg.canonicalize_particle_sources
        self.particle_in = eqx.nn.Linear(2 * cfg.num_sources, cfg.hidden_dim, key=keys[0])
        self.blocks = tuple(
            ParticleLikelihoodCrossBlock(
                cfg.hidden_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[1 + i],
            )
            for i in range(cfg.posterior_depth)
        )
        self.decoder = ParticleDecoder(cfg, key=keys[-1])

    def __call__(self, prior_particles: Array, likelihood_summaries: Array) -> Array:
        _, flat_prior = _prepare_prior_jax(
            prior_particles, self.num_sources, self.canonicalize
        )
        base_particles = _linear_tokens(self.particle_in, flat_prior)  # [N,H]
        particles = jnp.broadcast_to(
            base_particles[None, :, :],
            (likelihood_summaries.shape[0],) + base_particles.shape,
        )                                                             # [T,N,H]

        # At prefix t, the N particle tokens cross-attend to the R likelihood-summary
        # tokens that summarize exactly observations 1:t.  Because those R tokens are
        # themselves prefix-permutation-invariant, the final posterior remains invariant
        # to reordering observations within a fixed prefix.
        for block in self.blocks:
            particles = jax.vmap(block)(particles, likelihood_summaries)

        return self.decoder(particles, flat_prior)


#%% 10) End-to-end two-Transformer model
class ModeAParallelBayesModel(eqx.Module):
    """Likelihood Transformer + one of the two Posterior Transformer variants."""

    likelihood_transformer: CausalPrefixLikelihoodTransformer
    posterior_transformer: AdaLNPosteriorTransformer | CrossAttentionPosteriorTransformer
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
        likelihood_key, posterior_key = jax.random.split(key)
        self.conditioning = conditioning
        self.likelihood_transformer = CausalPrefixLikelihoodTransformer(
            cfg, key=likelihood_key
        )
        if conditioning == "adaln":
            self.posterior_transformer = AdaLNPosteriorTransformer(
                cfg, key=posterior_key
            )
        else:
            self.posterior_transformer = CrossAttentionPosteriorTransformer(
                cfg, key=posterior_key
            )

    def __call__(
        self,
        prior_particles: Array,  # [N,S,2]
        observations: Array,     # [T,3]
    ) -> tuple[Array, Array]:
        summaries = self.likelihood_transformer(observations)          # [T,R,H]
        posterior = self.posterior_transformer(prior_particles, summaries)
        return posterior, summaries                                    # [T,N,S,2], [T,R,H]


#%% 11) Exact empirical energy score and simple posterior diagnostics
def flatten_particle_measure(
    particles: Array,
    cfg: BayesTransportConfig = CFG,
) -> Array:
    """Represent each S-source theta sample as one vector in R^(2S)."""
    if cfg.canonicalize_particle_sources and cfg.num_sources > 1:
        particles = canonicalize_sources_jax(particles)
    return particles.reshape(particles.shape[0], 2 * cfg.num_sources)


def energy_score_single(
    particles: Array,
    theta_true: Array,
    cfg: BayesTransportConfig = CFG,
) -> Array:
    """Exact energy score of an equally weighted empirical posterior cloud.

    For q^N = N^{-1} sum_n delta_{theta_n},

        ES(q^N, theta*)
          = N^{-1} sum_n ||theta_n-theta*||
            - (2 N^2)^{-1} sum_{n,m} ||theta_n-theta_m||.

    The N truth distances are O(N d_theta).  The pair term contains N(N-1)/2
    unique non-zero particle interactions and dominates at O(N^2 d_theta).
    """
    samples = flatten_particle_measure(particles, cfg)
    target = theta_true
    if cfg.canonicalize_particle_sources and cfg.num_sources > 1:
        target = canonicalize_sources_jax(target)
    target = target.reshape(-1)

    truth_distance = jnp.mean(
        jnp.sqrt(jnp.sum((samples - target[None, :]) ** 2, axis=-1) + 1e-12)
    )
    differences = samples[:, None, :] - samples[None, :, :]
    squared_distance = jnp.sum(differences**2, axis=-1)
    off_diagonal = 1.0 - jnp.eye(samples.shape[0], dtype=samples.dtype)
    pairwise_distance = jnp.sum(
        jnp.sqrt(squared_distance + 1e-12) * off_diagonal
    ) / (samples.shape[0] ** 2)
    return truth_distance - 0.5 * pairwise_distance


def posterior_mean_rmse_single(
    particles: Array,
    theta_true: Array,
    cfg: BayesTransportConfig = CFG,
) -> Array:
    samples = flatten_particle_measure(particles, cfg)
    target = theta_true
    if cfg.canonicalize_particle_sources and cfg.num_sources > 1:
        target = canonicalize_sources_jax(target)
    target = target.reshape(-1)
    return jnp.sqrt(jnp.mean((jnp.mean(samples, axis=0) - target) ** 2))


def posterior_spread_single(
    particles: Array,
    cfg: BayesTransportConfig = CFG,
) -> Array:
    samples = flatten_particle_measure(particles, cfg)
    return jnp.mean(jnp.var(samples, axis=0))


def _trajectory_metrics(
    posterior_sequence: Array,
    theta_true: Array,
    cfg: BayesTransportConfig,
) -> tuple[Array, Array, Array]:
    """Vectorise all per-prefix metrics over T without a Python loop."""
    energy = jax.vmap(lambda p: energy_score_single(p, theta_true, cfg))(
        posterior_sequence
    )
    rmse = jax.vmap(lambda p: posterior_mean_rmse_single(p, theta_true, cfg))(
        posterior_sequence
    )
    spread = jax.vmap(lambda p: posterior_spread_single(p, cfg))(
        posterior_sequence
    )
    return energy, rmse, spread


def batch_objective(
    model: ModeAParallelBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig = CFG,
) -> tuple[Array, dict[str, Array]]:
    """Mean Mode-A energy score over every trajectory and every prefix.

    `predicted` has shape [B,T,N,S,2].  Thus all prefix losses are available in one
    forward pass and one gradient call.  This is the key change from naively stepping
    a posterior state through t=1,...,T.
    """
    predicted, _ = jax.vmap(model)(
        batch["prior_particles"],
        batch["observations"],
    )
    energy, rmse, spread = jax.vmap(
        lambda posterior_sequence, theta: _trajectory_metrics(
            posterior_sequence, theta, cfg
        )
    )(predicted, batch["theta_true"])

    loss = jnp.mean(energy)
    metrics = {
        "loss": loss,
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
    return loss, metrics


@eqx.filter_jit
def predict_batch(
    model: ModeAParallelBayesModel,
    prior_particles: Array,
    observations: Array,
) -> tuple[Array, Array]:
    """JIT-compiled trajectory batching; time prefixes remain inside each model call."""
    return jax.vmap(model)(prior_particles, observations)


@eqx.filter_jit
def evaluation_batch(
    model: ModeAParallelBayesModel,
    batch: dict[str, Array],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Array]:
    _, metrics = batch_objective(model, batch, cfg)
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
    """Evaluate a model with fresh, reproducible prior clouds.

    Returning by-prefix arrays is useful for the T -> large diagnostics; the scalar
    metrics are averages over the same underlying arrays.
    """
    n_total = len(dataset["theta_true"])
    if max_trajectories is not None:
        n_total = min(n_total, int(max_trajectories))
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    rng = np.random.default_rng(cfg.seed + 90_000 if seed is None else seed)

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


#%% 13) Optional exact-likelihood reference posterior for plots only
def reference_posterior_particles_np(
    rng: np.random.Generator,
    observations: np.ndarray,
    prefix_length: int,
    cfg: BayesTransportConfig = CFG,
) -> tuple[np.ndarray, float]:
    """SNIS reference posterior used only after training for visual validation.

    Proposal is exactly p(theta), so the importance weights are proportional to the
    likelihood of the observed prefix.  This function is intentionally NOT a teacher
    and is never called inside the training objective.

    No separate `source_log_likelihood` function is introduced: the Gaussian residual
    is written once here from the single shared forward-model function.
    """
    prefix_length = int(prefix_length)
    proposals = sample_prior_np(rng, cfg.reference_proposals, cfg)
    prefix = np.asarray(observations[:prefix_length])
    designs = prefix[:, :2]
    readings = prefix[:, 2]
    predicted_means = source_log_mean_np(proposals, designs, cfg)  # [P,t]
    residual = (readings[None, :] - predicted_means) / cfg.observation_noise_std
    log_weights = -0.5 * np.sum(residual**2, axis=1)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.maximum(weights.sum(), 1e-300)
    ess = float(1.0 / np.sum(weights**2))
    indices = rng.choice(
        len(proposals),
        size=cfg.reference_particles,
        replace=True,
        p=weights,
    )
    posterior = proposals[indices]
    if cfg.canonicalize_particle_sources and cfg.num_sources > 1:
        posterior = canonicalize_sources_np(posterior)
    return posterior.astype(np.float32), ess


#%% 14) Visualisation: architecture schematic
def plot_architecture_schematic(
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
):
    """Visual map of the two architectures and the parallel T dimension."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.8), constrained_layout=True)

    def draw_box(ax, xy, width, height, text, title=None):
        patch = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.3,
            facecolor="white",
            edgecolor="black",
        )
        ax.add_patch(patch)
        label = text if title is None else f"{title}\n{text}"
        ax.text(xy[0] + width / 2, xy[1] + height / 2, label,
                ha="center", va="center", fontsize=10)

    def arrow(ax, start, end, text=""):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                     linewidth=1.25))
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.045,
                    text, ha="center", va="bottom", fontsize=8)

    for ax, mode in zip(axes, ["AdaLN", "Cross-attention"]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(f"{mode} posterior conditioning — all T prefixes in parallel",
                     loc="left", fontweight="bold")

        draw_box(ax, (0.02, 0.58), 0.18, 0.25,
                 "[x_t, y_t]_{t=1:T}\nprecomputed trajectory",
                 "observations")
        draw_box(ax, (0.25, 0.54), 0.24, 0.33,
                 f"triangular prefix mask\nR={cfg.likelihood_summary_tokens} summary queries\nno observation positions",
                 "causal prefix-set\nLikelihood Transformer")
        draw_box(ax, (0.54, 0.58), 0.18, 0.25,
                 r"$\tilde{x}_{1:T}$" + f"\nshape [T,R,H]",
                 "likelihood summaries")
        draw_box(ax, (0.02, 0.10), 0.18, 0.25,
                 f"N={cfg.num_particles} iid draws\nfrom p(theta)",
                 "prior cloud")
        posterior_text = (
            "particle self-attention\nAdaLN(summary_t)\nshared weights for every t"
            if mode == "AdaLN"
            else "particle self-attention\nparticle -> R summary tokens\ncross-attention"
        )
        draw_box(ax, (0.54, 0.08), 0.22, 0.30, posterior_text, "Posterior Transformer")
        draw_box(ax, (0.81, 0.14), 0.17, 0.20,
                 "[T,N,S,2]\nq_phi(theta | x_1:t)",
                 "posterior clouds")

        arrow(ax, (0.20, 0.705), (0.25, 0.705))
        arrow(ax, (0.49, 0.705), (0.54, 0.705), "all t")
        arrow(ax, (0.20, 0.225), (0.54, 0.225), "broadcast over T")
        arrow(ax, (0.63, 0.58), (0.65, 0.38),
              "pool per t" if mode == "AdaLN" else "R tokens per t")
        arrow(ax, (0.76, 0.23), (0.81, 0.23))

    fig.suptitle(
        "Mode A: theta* fixed within each simulated trajectory, re-drawn across trajectories",
        fontsize=14,
        fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 15) Visualisation: physical source field and one simulated trajectory
def plot_source_trajectory(
    trajectory: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
    title: str = "Mode-A source-localisation trajectory",
):
    theta_true = np.asarray(trajectory["theta_true"])
    observations = np.asarray(trajectory["observations"])
    designs = observations[:, :2]
    readings = observations[:, 2]

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


#%% 16) Visualisation: prior -> posterior evolution across prefixes
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
    model: ModeAParallelBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
    title: str = "Posterior evolution",
):
    theta_true = np.asarray(trajectory["theta_true"])
    observations = np.asarray(trajectory["observations"])
    predicted, _ = model(
        jnp.asarray(prior_particles),
        jnp.asarray(observations),
    )
    predicted = np.asarray(jax.device_get(predicted))

    prefixes = select_prefixes(len(observations), 5)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.5), constrained_layout=True)
    axes = axes.ravel()

    clouds = [prior_particles] + [predicted[t - 1] for t in prefixes]
    labels = ["prior p(theta)"] + [f"q_phi(theta | x_1:{t})" for t in prefixes]
    all_points = np.concatenate([c.reshape(-1, 2) for c in clouds] + [theta_true.reshape(-1, 2)])
    lim = max(3.0 * cfg.prior_std, 1.12 * float(np.quantile(np.abs(all_points), 0.995)))

    for panel_index, (ax, cloud, label) in enumerate(zip(axes, clouds, labels)):
        ax.scatter(cloud[..., 0].reshape(-1), cloud[..., 1].reshape(-1),
                   s=13, alpha=0.30, label="particle source locations")
        ax.scatter(theta_true[:, 0], theta_true[:, 1], marker="*", s=190,
                   edgecolors="black", linewidths=0.8, label="theta*")
        if panel_index > 0:
            t = prefixes[panel_index - 1]
            designs = observations[:t, :2]
            ax.scatter(designs[:, 0], designs[:, 1], marker="x", s=33,
                       alpha=0.65, label="designs seen")
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


#%% 17) Visualisation: learned posterior versus optional likelihood-based reference
def plot_reference_comparison(
    models: dict[str, ModeAParallelBayesModel],
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
):
    """Compare source marginals at the final prefix; reference is diagnostic only."""
    observations = np.asarray(trajectory["observations"])
    theta_true = np.asarray(trajectory["theta_true"])
    rng = np.random.default_rng(cfg.seed + 444_000)
    reference, ess = reference_posterior_particles_np(
        rng, observations, len(observations), cfg
    )

    learned = {}
    for name, model in models.items():
        posterior, _ = model(jnp.asarray(prior_particles), jnp.asarray(observations))
        learned[name] = np.asarray(jax.device_get(posterior[-1]))

    column_names = ["Prior", f"Reference posterior\nSNIS ESS={ess:.0f}"] + list(learned.keys())
    column_clouds = [prior_particles, reference] + list(learned.values())
    n_cols = len(column_clouds)
    fig, axes = plt.subplots(
        cfg.num_sources,
        n_cols,
        figsize=(4.0 * n_cols, 3.8 * cfg.num_sources),
        squeeze=False,
        constrained_layout=True,
    )

    canonical_prior = (
        canonicalize_sources_np(prior_particles)
        if cfg.canonicalize_particle_sources and cfg.num_sources > 1
        else prior_particles
    )
    column_clouds[0] = canonical_prior
    canonical_truth = (
        canonicalize_sources_np(theta_true)
        if cfg.canonicalize_particle_sources and cfg.num_sources > 1
        else theta_true
    )

    all_points = np.concatenate([c.reshape(-1, 2) for c in column_clouds] + [canonical_truth])
    lim = max(3.0 * cfg.prior_std, 1.10 * float(np.quantile(np.abs(all_points), 0.995)))

    for source_index in range(cfg.num_sources):
        for col, (name, cloud) in enumerate(zip(column_names, column_clouds)):
            ax = axes[source_index, col]
            ax.scatter(cloud[:, source_index, 0], cloud[:, source_index, 1],
                       s=12, alpha=0.25)
            ax.scatter(canonical_truth[source_index, 0], canonical_truth[source_index, 1],
                       marker="*", s=190, edgecolors="black", linewidths=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            if source_index == 0:
                ax.set_title(name, fontweight="bold")
            ax.set_ylabel(f"canonical source {source_index + 1}")

    fig.suptitle(
        "Final-prefix posterior source marginals: learned clouds versus a likelihood-based reference",
        fontsize=14,
        fontweight="bold",
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

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8), constrained_layout=True)

    values = np.asarray(history["step_loss"])
    axes[0, 0].plot(steps, values, linewidth=0.75, alpha=0.75, label="step energy loss")
    if len(values) >= 20:
        window = max(5, len(values) // 100)
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(steps[window - 1:], smoothed, linewidth=1.8,
                        label=f"moving average ({window})")
    axes[0, 0].set_title("Energy score at every gradient step", loc="left", fontweight="bold")
    axes[0, 0].set_xlabel("gradient step")
    axes[0, 0].set_yscale("symlog", linthresh=1e-5)
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(steps, history["step_grad_norm"], linewidth=0.75)
    axes[0, 1].set_title("Gradient norm at every step", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("gradient step")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(epochs, history["epoch_train_loss"], marker="o", markersize=3,
                    label="train")
    axes[1, 0].plot(epochs, history["epoch_val_loss"], marker="o", markersize=3,
                    label="validation")
    axes[1, 0].axvline(best_epoch, linestyle="--", linewidth=1.0,
                       label=f"best epoch {best_epoch}")
    axes[1, 0].set_title("Per-epoch mean energy score", loc="left", fontweight="bold")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    energy_by_t = np.asarray(history["epoch_val_energy_by_t"])
    selected_epochs = np.unique(
        np.clip(np.rint(np.linspace(0, len(energy_by_t) - 1, 5)).astype(int),
                0, len(energy_by_t) - 1)
    )
    prefix_axis = np.arange(1, energy_by_t.shape[1] + 1)
    for epoch_index in selected_epochs:
        axes[1, 1].plot(prefix_axis, energy_by_t[epoch_index],
                        label=f"epoch {epoch_index + 1}")
    axes[1, 1].set_title("Validation energy score by prefix", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("prefix length t")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        f"Mode-A parallel training diagnostics — {conditioning}",
        fontsize=14,
        fontweight="bold",
    )
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
    """Train one end-to-end model pair; all T prefix losses are parallel.

    The AdaLN and cross-attention variants use this same training function, same data,
    same minibatch order, same fresh-prior RNG seed, same energy-score objective, and
    same evaluation protocol.  Only the posterior-conditioning mechanism changes.
    """
    variant_dir = run_dir / conditioning
    (variant_dir / "plots").mkdir(parents=True, exist_ok=True)
    (variant_dir / "artefacts").mkdir(parents=True, exist_ok=True)

    model_seed_offset = 0 if conditioning == "adaln" else 10_000
    model = ModeAParallelBayesModel(
        cfg,
        conditioning=conditioning,
        key=jax.random.key(cfg.seed + model_seed_offset),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        ),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(candidate_model, candidate_opt_state, batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            batch_objective, has_aux=True
        )(candidate_model, batch, cfg)
        params = eqx.filter(candidate_model, eqx.is_array)
        updates, candidate_opt_state = optimizer.update(
            grads, candidate_opt_state, params
        )
        candidate_model = eqx.apply_updates(candidate_model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return candidate_model, candidate_opt_state, loss, metrics, grad_norm

    history: dict[str, list] = {
        "step_loss": [],
        "step_final_energy_score": [],
        "step_mean_rmse": [],
        "step_grad_norm": [],
        "epoch_train_loss": [],
        "epoch_val_loss": [],
        "epoch_val_final_energy_score": [],
        "epoch_val_mean_rmse": [],
        "epoch_val_energy_by_t": [],
        "epoch_val_rmse_by_t": [],
        "epoch_val_spread_by_t": [],
    }

    # Snapshot the initial identity transport, as in the supplied implementation.
    plot_posterior_evolution(
        model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        variant_dir / "plots" / "fixed_trajectory_before_training.png",
        f"{conditioning}: before training (identity transport)",
    )

    initial_metrics = evaluate_model(
        model,
        eval_data,
        cfg,
        seed=cfg.seed + 91_000,
    )
    print(f"[{conditioning}] initial validation loss = {initial_metrics['loss']:.6f}")

    visualisation_epochs = sorted(
        set(
            max(1, int(math.ceil(fraction * cfg.epochs / 10.0)))
            for fraction in range(1, 11)
        )
    )
    rng = np.random.default_rng(cfg.seed + 30_000)  # identical data order for both variants
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
            model, opt_state, loss, metrics, grad_norm = train_step(
                model, opt_state, batch
            )
            host = jax.device_get(metrics)
            host_loss = float(jax.device_get(loss))
            host_grad_norm = float(jax.device_get(grad_norm))
            global_step += 1

            train_losses_this_epoch.append(host_loss)
            history["step_loss"].append(host_loss)
            history["step_final_energy_score"].append(float(host["final_energy_score"]))
            history["step_mean_rmse"].append(float(host["posterior_mean_rmse"]))
            history["step_grad_norm"].append(host_grad_norm)
            progress.set_postfix(
                ES=f"{host_loss:.4f}",
                final=f"{float(host['final_energy_score']):.4f}",
                grad=f"{host_grad_norm:.3f}",
            )

        epoch_train_loss = float(np.mean(train_losses_this_epoch))
        val_metrics = evaluate_model(
            model,
            eval_data,
            cfg,
            seed=cfg.seed + 91_000,  # same validation prior clouds every epoch
        )
        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_val_loss"].append(float(val_metrics["loss"]))
        history["epoch_val_final_energy_score"].append(
            float(val_metrics["final_energy_score"])
        )
        history["epoch_val_mean_rmse"].append(
            float(val_metrics["posterior_mean_rmse"])
        )
        history["epoch_val_energy_by_t"].append(
            np.asarray(val_metrics["energy_by_t"], dtype=np.float64)
        )
        history["epoch_val_rmse_by_t"].append(
            np.asarray(val_metrics["rmse_by_t"], dtype=np.float64)
        )
        history["epoch_val_spread_by_t"].append(
            np.asarray(val_metrics["spread_by_t"], dtype=np.float64)
        )

        save_model(variant_dir / "artefacts" / "model_last.eqx", model)
        if epoch % cfg.save_every_epochs == 0:
            save_model(
                variant_dir / "artefacts" / f"model_epoch_{epoch:04d}.eqx",
                model,
            )
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
                "objective": "mean empirical energy score over all B x T prefixes",
            },
        )

        print(
            f"[{conditioning}] epoch {epoch:03d}: "
            f"train ES={epoch_train_loss:.6f} | "
            f"val ES={float(val_metrics['loss']):.6f} | "
            f"final-prefix ES={float(val_metrics['final_energy_score']):.6f} | "
            f"mean RMSE={float(val_metrics['posterior_mean_rmse']):.5f} | "
            f"{time.time() - epoch_started_at:.1f}s"
        )

        if epoch in visualisation_epochs:
            plot_posterior_evolution(
                model,
                fixed_trajectory,
                fixed_prior_particles,
                cfg,
                variant_dir / "plots" / f"fixed_trajectory_epoch_{epoch:04d}.png",
                f"{conditioning}: posterior evolution after epoch {epoch}",
            )

    best_model = load_model(
        variant_dir / "artefacts" / "model_best.eqx",
        cfg,
        conditioning,
        key=jax.random.key(0),
    )
    final_metrics = evaluate_model(
        best_model,
        eval_data,
        cfg,
        seed=cfg.seed + 91_000,
    )
    plot_posterior_evolution(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        variant_dir / "plots" / "fixed_trajectory_best_model.png",
        f"{conditioning}: best model (epoch {best_epoch})",
    )
    plot_training_diagnostics(
        history,
        best_epoch,
        conditioning,
        variant_dir / "plots" / "training_diagnostics.png",
    )

    print(
        f"[{conditioning}] training complete in "
        f"{datetime.timedelta(seconds=int(time.time() - training_started_at))}; "
        f"best epoch={best_epoch}, val ES={best_val_loss:.6f}"
    )
    return {
        "model": best_model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_metrics": final_metrics,
        "variant_dir": variant_dir,
    }


#%% 20) Create the run, datasets, and fixed visualisation trajectory
np.random.seed(CFG.seed)
print("JAX devices:", jax.devices())
print("Configuration:\n", yaml.safe_dump(asdict(CFG), sort_keys=False))

run_dir = make_run_dir(CFG.env_name, CFG.runs_base)
with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
    yaml.safe_dump(asdict(CFG), handle, sort_keys=False)
print("Run directory:", run_dir)

# Precompute complete likelihood trajectories.  The neural training loop does not call
# the simulator again; each batch only adds fresh independent prior particles.
train_rng = np.random.default_rng(CFG.seed + 1_000)
eval_rng = np.random.default_rng(CFG.seed + 2_000)
train_data = simulate_mode_a_trajectories(
    train_rng,
    CFG.n_train_trajectories,
    CFG.trajectory_length,
    CFG,
)
eval_data = simulate_mode_a_trajectories(
    eval_rng,
    CFG.n_eval_trajectories,
    CFG.trajectory_length,
    CFG,
)

fixed_trajectory = {
    "theta_true": eval_data["theta_true"][0],
    "observations": eval_data["observations"][0],
}
fixed_prior_particles = sample_prior_np(
    np.random.default_rng(CFG.seed + 3_000),
    CFG.num_particles,
    CFG,
)
np.savez_compressed(
    run_dir / "artefacts" / "fixed_trajectory.npz",
    theta_true=fixed_trajectory["theta_true"],
    observations=fixed_trajectory["observations"],
    prior_particles=fixed_prior_particles,
)

plot_architecture_schematic(CFG, run_dir / "plots" / "architecture_schematic.png")
plot_source_trajectory(
    fixed_trajectory,
    CFG,
    run_dir / "plots" / "fixed_trajectory_sensor_field.png",
)


#%% 21) Train BOTH conditioning architectures on the same Mode-A problem
# The variants are separate end-to-end models so their likelihood representations can
# specialize to their posterior-conditioning mechanism.  Within each model, however,
# all T prefixes are trained in parallel as [B,T,N,S,2]; nothing is stepped recurrently.
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


#%% 22) Direct visual comparison with a likelihood-based posterior reference
plot_reference_comparison(
    models,
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
    axes[1].set_title("Posterior-mean RMSE by prefix", fontweight="bold")
    axes[2].set_title("Posterior spread by prefix", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("prefix length t")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("energy score")
    axes[1].set_ylabel("RMSE")
    axes[2].set_ylabel("mean marginal variance")
    fig.suptitle("AdaLN versus cross-attention on the same Mode-A trajectories",
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
    """Numerically test the exact symmetries built into the architecture.

    1. Causality: perturb future observations and verify outputs through prefix t do
       not change.
    2. Prefix-set invariance: truncate at t, randomly permute those t observations,
       and verify the FINAL prefix posterior is unchanged.
    3. Particle equivariance: permute the prior-particle axis, run the model, undo that
       permutation on outputs, and verify equality.

    These are architectural identities up to floating-point error, not statistical
    claims about whether the learned distribution is the exact Bayesian posterior.
    """
    obs = np.asarray(trajectory["observations"], dtype=np.float32)
    prior = np.asarray(prior_particles, dtype=np.float32)
    rng = np.random.default_rng(cfg.seed + 500_000)
    t = max(2, len(obs) // 2)

    baseline, baseline_summary = model(jnp.asarray(prior), jnp.asarray(obs))
    baseline = np.asarray(jax.device_get(baseline))
    baseline_summary = np.asarray(jax.device_get(baseline_summary))

    future_perturbed = obs.copy()
    if t < len(obs):
        future_perturbed[t:, :2] = rng.uniform(
            cfg.design_low, cfg.design_high, size=future_perturbed[t:, :2].shape
        )
        future_perturbed[t:, 2] += rng.normal(0.0, 5.0, size=len(obs) - t)
    causal_output, causal_summary = model(
        jnp.asarray(prior), jnp.asarray(future_perturbed)
    )
    causal_output = np.asarray(jax.device_get(causal_output))
    causal_summary = np.asarray(jax.device_get(causal_summary))
    causal_error = float(np.max(np.abs(causal_output[:t] - baseline[:t])))
    causal_summary_error = float(np.max(np.abs(causal_summary[:t] - baseline_summary[:t])))

    truncated = obs[:t].copy()
    permutation = rng.permutation(t)
    permuted = truncated[permutation]
    output_a, summary_a = model(jnp.asarray(prior), jnp.asarray(truncated))
    output_b, summary_b = model(jnp.asarray(prior), jnp.asarray(permuted))
    output_a = np.asarray(jax.device_get(output_a))
    output_b = np.asarray(jax.device_get(output_b))
    summary_a = np.asarray(jax.device_get(summary_a))
    summary_b = np.asarray(jax.device_get(summary_b))
    prefix_invariance_error = float(np.max(np.abs(output_a[-1] - output_b[-1])))
    prefix_summary_invariance_error = float(np.max(np.abs(summary_a[-1] - summary_b[-1])))

    particle_perm = rng.permutation(len(prior))
    inverse_perm = np.argsort(particle_perm)
    permuted_output, _ = model(jnp.asarray(prior[particle_perm]), jnp.asarray(obs))
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
    x = np.arange(len(metric_names))
    width = 0.8 / len(structure_results)
    fig, ax = plt.subplots(figsize=(13.5, 5.2), constrained_layout=True)
    for i, (name, values) in enumerate(structure_results.items()):
        heights = [max(values[m], 1e-16) for m in metric_names]
        ax.bar(x + (i - (len(structure_results) - 1) / 2) * width,
               heights, width=width, label=name)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([
        "causal\nposterior",
        "causal\nlikelihood",
        "prefix-set\nposterior",
        "prefix-set\nlikelihood",
        "particle\nequivariance",
    ])
    ax.set_ylabel("max absolute discrepancy")
    ax.set_title("Architectural identities should be near floating-point precision",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


plot_structural_checks(
    structure_results,
    run_dir / "plots" / "structural_theorem_checks.png",
)


#%% 25) Numerical theorem check: single-global-truth proper-score collapse
def energy_score_np(
    particles: np.ndarray,
    theta_true: np.ndarray,
    cfg: BayesTransportConfig = CFG,
) -> float:
    return float(
        jax.device_get(
            energy_score_single(
                jnp.asarray(particles),
                jnp.asarray(theta_true),
                cfg,
            )
        )
    )


def mode_b_collapse_curve(
    theta_star: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    n_particles: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirically show that ES(q,theta*) is minimized by delta_{theta*}.

    We are NOT training Mode B.  We simply construct increasingly diffuse clouds
    centered on one fixed truth and evaluate the exact empirical energy score.  The
    theorem says the point mass is the unique distributional optimum.
    """
    rng = np.random.default_rng(cfg.seed + 600_000)
    canonical_truth = (
        canonicalize_sources_np(theta_star)
        if cfg.canonicalize_particle_sources and cfg.num_sources > 1
        else theta_star
    )
    base_noise = rng.normal(
        size=(n_particles, cfg.num_sources, 2)
    ).astype(np.float32)
    scales = np.concatenate([[0.0], np.geomspace(1e-3, 2.0, 34)])
    scores = []
    for scale in scales:
        cloud = canonical_truth[None, :, :] + float(scale) * base_noise
        scores.append(energy_score_np(cloud.astype(np.float32), canonical_truth, cfg))
    return scales, np.asarray(scores)


collapse_scales, collapse_scores = mode_b_collapse_curve(
    fixed_trajectory["theta_true"], CFG
)
fig, ax = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
ax.plot(collapse_scales, collapse_scores, marker="o", markersize=3)
ax.set_xscale("symlog", linthresh=1e-3)
ax.set_yscale("symlog", linthresh=1e-6)
ax.set_xlabel("cloud scale around one fixed theta*")
ax.set_ylabel("energy score against the same fixed theta*")
ax.set_title("Mode B diagnostic: the proper-score optimum collapses to delta(theta*)",
             fontweight="bold")
ax.grid(alpha=0.25)
fig.savefig(run_dir / "plots" / "mode_b_collapse_theorem.png", dpi=170)
display(fig)
plt.close(fig)


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
axes[1].set_title("Final posterior-mean RMSE")
axes[2].set_title("Evaluation wall time")
axes[0].set_xlabel("particles N")
axes[1].set_xlabel("particles N")
axes[2].set_xlabel("particles N")
axes[2].set_ylabel("seconds")
fig.suptitle("Finite-particle limit study: accuracy and the O(N^2) cost pressure",
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
axes[1].set_title("Posterior-mean RMSE")
axes[2].set_title("Posterior spread")
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
    """Return one final-prefix ES per independent Mode-A trajectory."""
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, len(dataset["theta_true"]), cfg.batch_size):
        stop = min(start + cfg.batch_size, len(dataset["theta_true"]))
        indices = np.arange(start, stop)
        batch_np = make_batch_np(dataset, indices, rng, cfg)
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        predicted, _ = predict_batch(
            model, batch["prior_particles"], batch["observations"]
        )
        final_posteriors = predicted[:, -1]
        batch_scores = jax.vmap(
            lambda particles, theta: energy_score_single(particles, theta, cfg)
        )(final_posteriors, batch["theta_true"])
        values.append(np.asarray(jax.device_get(batch_scores), dtype=np.float64))
    return np.concatenate(values)


mc_pool_rng = np.random.default_rng(CFG.seed + 900_000)
mc_pool_size = max(CFG.trajectory_mc_values)
mc_pool = simulate_mode_a_trajectories(
    mc_pool_rng,
    mc_pool_size,
    CFG.trajectory_length,
    CFG,
)
trajectory_mc_study: dict[str, dict[str, np.ndarray]] = {}
for name, model in models.items():
    scores = per_trajectory_final_energy(
        model,
        mc_pool,
        CFG,
        seed=CFG.seed + 901_000,
    )
    rng = np.random.default_rng(CFG.seed + 902_000)
    scores = scores[rng.permutation(len(scores))]
    means, lower, upper = [], [], []
    for m in CFG.trajectory_mc_values:
        sample = scores[:m]
        mean = float(np.mean(sample))
        se = float(np.std(sample, ddof=1) / math.sqrt(m)) if m > 1 else 0.0
        means.append(mean)
        lower.append(mean - 1.96 * se)
        upper.append(mean + 1.96 * se)
    trajectory_mc_study[name] = {
        "M": np.asarray(CFG.trajectory_mc_values, dtype=int),
        "mean": np.asarray(means),
        "lower": np.asarray(lower),
        "upper": np.asarray(upper),
    }

fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
for name, values in trajectory_mc_study.items():
    ax.plot(values["M"], values["mean"], marker="o", label=name)
    ax.fill_between(values["M"], values["lower"], values["upper"], alpha=0.16)
ax.set_xscale("log", base=2)
ax.set_xlabel("independent evaluation trajectories M")
ax.set_ylabel("empirical mean final-prefix energy score")
ax.set_title("M -> large: Monte Carlo estimate of population risk stabilises",
             fontweight="bold")
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(run_dir / "plots" / "trajectory_count_limit_study.png", dpi=170)
display(fig)
plt.close(fig)


#%% 29) Finite prior-cloud stability: repeated prior draws for the SAME observations
def prior_cloud_stability_study(
    models: dict[str, ModeAParallelBayesModel],
    trajectory: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
) -> dict[str, dict[str, np.ndarray]]:
    """How much does posterior mean move when the numerical prior cloud is re-drawn?"""
    observations = np.asarray(trajectory["observations"])
    study = {}
    for name, model in models.items():
        stds = []
        for n_particles in cfg.particle_limit_values:
            means = []
            for repeat in range(cfg.prior_resample_repeats):
                rng = np.random.default_rng(
                    cfg.seed + 1_000_000 + 1000 * n_particles + repeat
                )
                prior = sample_prior_np(rng, n_particles, cfg)
                posterior, _ = model(jnp.asarray(prior), jnp.asarray(observations))
                final = np.asarray(jax.device_get(posterior[-1]))
                if cfg.canonicalize_particle_sources and cfg.num_sources > 1:
                    final = canonicalize_sources_np(final)
                means.append(final.reshape(len(final), -1).mean(axis=0))
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
ax.set_ylabel("RMS SD of posterior mean across fresh prior clouds")
ax.set_title("Finite-prior representation stability for fixed observed data",
             fontweight="bold")
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(run_dir / "plots" / "prior_cloud_stability.png", dpi=170)
display(fig)
plt.close(fig)


#%% 30) Causal truncation consistency: full T versus running only the first t observations
def truncation_consistency_study(
    model: ModeAParallelBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """A direct numerical check that future points are not needed to compute prefix t."""
    observations = np.asarray(trajectory["observations"])
    full, _ = model(jnp.asarray(prior_particles), jnp.asarray(observations))
    full = np.asarray(jax.device_get(full))
    prefix_values = select_prefixes(len(observations), 6)
    errors = []
    for t in prefix_values:
        truncated, _ = model(
            jnp.asarray(prior_particles),
            jnp.asarray(observations[:t]),
        )
        truncated = np.asarray(jax.device_get(truncated))
        errors.append(float(np.max(np.abs(full[t - 1] - truncated[-1]))))
    return np.asarray(prefix_values), np.asarray(errors)


fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
for name, model in models.items():
    t_values, errors = truncation_consistency_study(
        model, fixed_trajectory, fixed_prior_particles
    )
    ax.plot(t_values, np.maximum(errors, 1e-16), marker="o", label=name)
ax.set_yscale("log")
ax.set_xlabel("prefix length t")
ax.set_ylabel("max |full-run q_t - truncated-run q_t|")
ax.set_title("Parallel causal computation agrees with separately truncated inference",
             fontweight="bold")
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(run_dir / "plots" / "causal_truncation_consistency.png", dpi=170)
display(fig)
plt.close(fig)


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
    np.savez_compressed(
        run_dir / "artefacts" / f"{study_name}.npz",
        **flat_payload,
    )

summary = {
    "objective": "energy score only",
    "mode": "Mode A: theta* fixed within trajectory, re-drawn across trajectories",
    "parallel_prefix_training": True,
    "trajectory_length": CFG.trajectory_length,
    "num_particles": CFG.num_particles,
    "architectures": {},
}
for name, result in results.items():
    summary["architectures"][name] = {
        "best_epoch": int(result["best_epoch"]),
        "best_val_loss": float(result["best_val_loss"]),
        "final_metrics": {
            key: float(value)
            for key, value in result["final_metrics"].items()
            if np.ndim(value) == 0
        },
    }
save_json(run_dir / "artefacts" / "final_summary.json", summary)

print("\nFinal Mode-A summary")
print(json.dumps(summary, indent=2))
print("All artefacts saved under:", run_dir)
