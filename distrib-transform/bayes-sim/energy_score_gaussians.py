#%% 1) Imports, configuration, and experiment conventions
"""Mode-A Bayesian transport on a conjugate linear-Gaussian benchmark.

This notebook-style script is deliberately parallel to the source-localisation
experiment, but replaces the nonlinear source simulator with a model whose posterior
is known exactly at every prefix.

Generative model
----------------
For every simulated trajectory m we draw a NEW parameter

    theta_m^* ~ N(mu_0, Sigma_0),

hold it fixed for the entire trajectory, draw design vectors x_{m,t}, and simulate

    y_{m,t} | x_{m,t}, theta_m^* ~ N(x_{m,t}^T theta_m^*, sigma_y^2).

The complete observation token is z_t = concat(x_t, y_t).  With theta_dim=2 this has
three entries, exactly like the [x-coordinate, y-coordinate, scalar-reading] token in
the source-localisation notebook.

Mode A
------
The same theta_m^* is the energy-score target for every prefix inside trajectory m,
but theta_m^* is re-drawn independently between trajectories.  Therefore the
population proper-score target at prefix t is the Bayesian posterior

    p(theta | D_t),   D_t = {(x_i, y_i)}_{i=1}^t.

Closed-form posterior
---------------------
For the spherical Gaussian prior Sigma_0 = prior_std^2 I,

    Sigma_t^{-1} = Sigma_0^{-1} + sigma_y^{-2} X_t^T X_t,
    mu_t = Sigma_t [Sigma_0^{-1} mu_0 + sigma_y^{-2} X_t^T y_t].

We compute (mu_t, Sigma_t) for EVERY prefix exactly.  These analytic quantities never
enter the training loss; they are diagnostics and an oracle reference only.  Training
still uses exactly one objective: the empirical multivariate energy score against the
generating theta^*.

Two end-to-end architectures
----------------------------
1. AdaLN conditioning
   observations -> causal prefix-set Likelihood Transformer -> prefix summaries
   prior particles + pooled summary_t -> AdaLN Posterior Transformer

2. Cross-attention conditioning
   observations -> causal prefix-set Likelihood Transformer -> prefix summary tokens
   prior particles cross-attend directly to the R summary tokens for prefix t.

All t=1,...,T prefixes are evaluated in one JAX program.  There is no recurrent
posterior update and no Python loop over t in the model or loss.

Array notation
--------------
B : minibatch trajectories
T : trajectory length / number of prefixes
N : prior/output particles
D : parameter dimension (2 by default)
R : learned summary tokens per prefix; R=1 is perfectly valid
H : Transformer hidden dimension

theta_true          [B,D]
observations         [B,T,D+1]       = concat(x_t, y_t)
prior_particles      [B,N,D]         iid p(theta), independent of theta_true
exact_mean           [B,T,D]
exact_cov            [B,T,D,D]
likelihood_summaries [B,T,R,H]
posterior_particles  [B,T,N,D]
energy_by_t          [B,T]

Notebook conventions
--------------------
* no main() wrapper: cells execute sequentially;
* every gradient-step loss is retained;
* both conditioning architectures use the same data and training recipe;
* figures and checkpoints are saved throughout training;
* closed-form posterior diagnostics are richer than in the nonlinear experiment.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse
from IPython.display import display
from tqdm.auto import tqdm
import yaml

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})


import jax
import jax.numpy as jnp
import equinox as eqx
import optax

Array = jax.Array
ConditioningMode = Literal["adaln", "cross_attention"]


@dataclass(frozen=True)
class GaussianBayesConfig:
    """Single source of experiment defaults; edit values here for notebook runs."""

    # Reproducibility and run bookkeeping.
    env_name: str = "mode_a_parallel_gaussian_conjugate"
    seed: int = 2030
    runs_base: str = "./runs"

    # Conjugate Bayesian model.
    theta_dim: int = 2
    prior_mean: tuple[float, ...] = (0.0, 0.0)
    prior_std: float = 1.0
    design_low: float = -3.0
    design_high: float = 3.0
    observation_noise_std: float = 0.30

    # Mode-A trajectory and particle counts: inherited from the source benchmark.
    trajectory_length: int = 24
    num_particles: int = 64
    n_train_trajectories: int = 4096
    n_eval_trajectories: int = 256
    batch_size: int = 16

    # Likelihood/context Transformer: same capacity as the source benchmark.
    hidden_dim: int = 96
    heads: int = 4
    mlp_ratio: int = 4
    likelihood_depth: int = 2
    likelihood_summary_tokens: int = 1
    pair_encoder_depth: int = 2

    # Posterior particle Transformer.
    posterior_depth: int = 3
    max_particle_displacement: float = 6.0
    architectures_to_train: tuple[str, ...] = ("adaln", "cross_attention")

    # Observation normalisation.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Optimisation: identical recipe and energy-score-only objective.
    epochs: int = 40
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0

    # Persistence / visualisation cadence.
    save_every_epochs: int = 5
    exact_reference_particles: int = 2_000

    # Limit / theorem diagnostics after training.
    limit_eval_trajectories: int = 192
    particle_limit_values: tuple[int, ...] = (16, 32, 64, 128, 256)
    long_trajectory_length: int = 48
    trajectory_mc_values: tuple[int, ...] = (8, 16, 32, 64, 128, 192)
    prior_resample_repeats: int = 12


CFG = GaussianBayesConfig()
if len(CFG.prior_mean) != CFG.theta_dim:
    raise ValueError("prior_mean must contain exactly theta_dim entries.")


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


def save_model(path: str | Path, model: "ModeAParallelGaussianModel"):
    eqx.tree_serialise_leaves(Path(path), model)


def load_model(
    path: str | Path,
    cfg: GaussianBayesConfig,
    conditioning: ConditioningMode,
    *,
    key: Array | None = None,
) -> "ModeAParallelGaussianModel":
    if key is None:
        key = jax.random.key(0)
    skeleton = ModeAParallelGaussianModel(cfg, conditioning=conditioning, key=key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)


#%% 3) Gaussian prior, Gaussian likelihood, and exact conjugate posterior
def prior_mean_np(cfg: GaussianBayesConfig = CFG) -> np.ndarray:
    return np.asarray(cfg.prior_mean, dtype=np.float64)


def prior_cov_np(cfg: GaussianBayesConfig = CFG) -> np.ndarray:
    return (cfg.prior_std**2) * np.eye(cfg.theta_dim, dtype=np.float64)


def sample_prior_np(
    rng: np.random.Generator,
    n: int,
    cfg: GaussianBayesConfig = CFG,
) -> np.ndarray:
    """Draw n iid theta ~ N(mu_0, prior_std^2 I), shape [n,D]."""
    return rng.normal(
        loc=prior_mean_np(cfg),
        scale=cfg.prior_std,
        size=(int(n), cfg.theta_dim),
    ).astype(np.float32)


def gaussian_likelihood_mean_np(theta: np.ndarray, designs: np.ndarray) -> np.ndarray:
    """Return x^T theta, broadcasting over leading theta/design dimensions.

    Examples
    --------
    theta [D],     designs [T,D]   -> [T]
    theta [B,D],   designs [B,T,D] -> [B,T]
    theta [P,D],   designs [T,D]   -> [P,T]
    """
    theta = np.asarray(theta, dtype=np.float64)
    designs = np.asarray(designs, dtype=np.float64)
    if theta.ndim == 1:
        return np.einsum("...d,d->...", designs, theta)
    if designs.ndim == 2:
        return np.einsum("pd,td->pt", theta, designs)
    return np.einsum("...d,...td->...t", theta, designs)


def exact_posterior_sequence_np(
    observations: np.ndarray,
    cfg: GaussianBayesConfig = CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form p(theta | D_t) for every t=1,...,T.

    observations[t] = concat(x_t, y_t), with x_t in R^D.

    Sigma_t^{-1} = Sigma_0^{-1} + sigma_y^{-2} sum_{i<=t} x_i x_i^T
    mu_t          = Sigma_t [Sigma_0^{-1} mu_0
                              + sigma_y^{-2} sum_{i<=t} x_i y_i].

    Returns
    -------
    means : [T,D]
    covs  : [T,D,D]
    """
    observations = np.asarray(observations, dtype=np.float64)
    x = observations[:, : cfg.theta_dim]
    y = observations[:, cfg.theta_dim]

    prior_precision = np.eye(cfg.theta_dim, dtype=np.float64) / (cfg.prior_std**2)
    prior_information = prior_precision @ prior_mean_np(cfg)
    noise_precision = 1.0 / (cfg.observation_noise_std**2)

    cumulative_xx = np.cumsum(
        np.einsum("ti,tj->tij", x, x), axis=0
    )
    cumulative_xy = np.cumsum(x * y[:, None], axis=0)

    precisions = prior_precision[None, :, :] + noise_precision * cumulative_xx
    covs = np.linalg.inv(precisions)
    information = prior_information[None, :] + noise_precision * cumulative_xy
    means = np.einsum("tij,tj->ti", covs, information)
    return means.astype(np.float32), covs.astype(np.float32)


def exact_posterior_batch_np(
    observations: np.ndarray,
    cfg: GaussianBayesConfig = CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised conjugate posterior for observations [B,T,D+1]."""
    observations = np.asarray(observations, dtype=np.float64)
    x = observations[..., : cfg.theta_dim]
    y = observations[..., cfg.theta_dim]

    prior_precision = np.eye(cfg.theta_dim, dtype=np.float64) / (cfg.prior_std**2)
    prior_information = prior_precision @ prior_mean_np(cfg)
    noise_precision = 1.0 / (cfg.observation_noise_std**2)

    cumulative_xx = np.cumsum(
        np.einsum("bti,btj->btij", x, x), axis=1
    )
    cumulative_xy = np.cumsum(x * y[..., None], axis=1)
    precisions = prior_precision[None, None, :, :] + noise_precision * cumulative_xx
    covs = np.linalg.inv(precisions)
    information = prior_information[None, None, :] + noise_precision * cumulative_xy
    means = np.einsum("btij,btj->bti", covs, information)
    return means.astype(np.float32), covs.astype(np.float32)


def sample_exact_gaussian_np(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray,
    n: int,
) -> np.ndarray:
    """Draw diagnostic samples from an analytically known Gaussian posterior."""
    return rng.multivariate_normal(
        np.asarray(mean, dtype=np.float64),
        np.asarray(cov, dtype=np.float64),
        size=int(n),
    ).astype(np.float32)


def simulate_mode_a_trajectories(
    rng: np.random.Generator,
    n_trajectories: int,
    trajectory_length: int,
    cfg: GaussianBayesConfig = CFG,
) -> dict[str, np.ndarray]:
    """Precompute complete Mode-A trajectories and exact posterior sequences.

    Sampling provenance is deliberately explicit:
      1. theta_true[m] ~ p(theta) once per trajectory;
      2. all T outcomes in that trajectory use the same theta_true[m];
      3. theta_true is re-drawn for the next trajectory;
      4. input prior particles are NOT stored here and are drawn independently later.
    """
    n_trajectories = int(n_trajectories)
    trajectory_length = int(trajectory_length)
    theta_true = sample_prior_np(rng, n_trajectories, cfg)
    designs = rng.uniform(
        cfg.design_low,
        cfg.design_high,
        size=(n_trajectories, trajectory_length, cfg.theta_dim),
    ).astype(np.float32)
    means = gaussian_likelihood_mean_np(theta_true, designs)
    outcomes = (
        means
        + cfg.observation_noise_std * rng.normal(size=means.shape)
    ).astype(np.float32)
    observations = np.concatenate([designs, outcomes[..., None]], axis=-1).astype(np.float32)
    exact_mean, exact_cov = exact_posterior_batch_np(observations, cfg)
    return {
        "theta_true": theta_true.astype(np.float32),
        "observations": observations,
        "exact_mean": exact_mean,
        "exact_cov": exact_cov,
    }


#%% 4) Minibatches with fresh independent prior clouds
def make_batch_np(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    rng: np.random.Generator,
    cfg: GaussianBayesConfig = CFG,
    *,
    num_particles: int | None = None,
) -> dict[str, np.ndarray]:
    """Create a minibatch and re-draw the finite numerical prior representation."""
    indices = np.asarray(indices, dtype=np.int64)
    n_particles = cfg.num_particles if num_particles is None else int(num_particles)
    b = len(indices)
    prior_particles = sample_prior_np(rng, b * n_particles, cfg).reshape(
        b, n_particles, cfg.theta_dim
    )
    return {
        "theta_true": dataset["theta_true"][indices].astype(np.float32),
        "observations": dataset["observations"][indices].astype(np.float32),
        "exact_mean": dataset["exact_mean"][indices].astype(np.float32),
        "exact_cov": dataset["exact_cov"][indices].astype(np.float32),
        "prior_particles": prior_particles.astype(np.float32),
    }


#%% 5) Token helpers shared by both Transformers
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _mlp_tokens(layer: eqx.nn.MLP, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _time_layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(lambda tokens: _layernorm_tokens(layer, tokens))(x)


def _time_linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(lambda tokens: _linear_tokens(layer, tokens))(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


#%% 6) Causal, prefix-permutation-invariant Likelihood Transformer
class PrefixSummaryBlock(eqx.Module):
    """R learned queries summarize one observation prefix without using token positions."""

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
        summaries: Array,          # [R,H]
        observation_tokens: Array, # [T,H]
        prefix_mask: Array,        # [T]
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
    """Encode all D_t in parallel while remaining invariant within each prefix set.

    Important: this network does NOT evaluate the known Gaussian likelihood formula.
    It learns a representation of observation tokens z_t=(x_t,y_t).  The exact
    conjugate posterior is kept outside the network and is used only for diagnostics.

    R is an architectural capacity parameter.  R=1 is valid.  For R>1, the
    cross-attention posterior path can select among multiple learned prefix summaries.
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
    theta_dim: int = eqx.field(static=True)

    def __init__(self, cfg: GaussianBayesConfig, *, key: Array):
        keys = jax.random.split(key, cfg.likelihood_depth + 4)
        self.design_scale = max(abs(cfg.design_low), abs(cfg.design_high), 1.0)
        self.y_center = cfg.y_center
        self.y_scale = max(cfg.y_scale, 1e-6)
        self.count_scale = float(max(cfg.trajectory_length, 1))
        self.num_summary_tokens = cfg.likelihood_summary_tokens
        self.theta_dim = cfg.theta_dim

        self.pair_encoder = eqx.nn.MLP(
            in_size=cfg.theta_dim + 1,
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
        t_length = observations.shape[0]
        normalized = jnp.concatenate(
            [
                observations[:, : self.theta_dim] / self.design_scale,
                (observations[:, self.theta_dim : self.theta_dim + 1] - self.y_center)
                / self.y_scale,
            ],
            axis=-1,
        )
        observation_tokens = _mlp_tokens(self.pair_encoder, normalized)  # [T,H]

        # Row t contains exactly observations 1,...,t+1.  vmap evaluates all rows
        # together; this is a causal information restriction, not sequential execution.
        prefix_masks = jnp.tril(jnp.ones((t_length, t_length), dtype=bool))
        normalized_counts = (
            jnp.arange(1, t_length + 1, dtype=observations.dtype) / self.count_scale
        )[:, None]
        count_tokens = jax.vmap(self.count_projection)(normalized_counts)

        summaries = jnp.broadcast_to(
            self.summary_queries[None, :, :],
            (t_length, self.num_summary_tokens, self.summary_queries.shape[-1]),
        )
        summaries = summaries + count_tokens[:, None, :]

        for block in self.blocks:
            summaries = jax.vmap(
                lambda summary_t, mask_t: block(summary_t, observation_tokens, mask_t)
            )(summaries, prefix_masks)

        return _time_layernorm_tokens(self.final_norm, summaries)  # [T,R,H]


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
        modulation = eqx.tree_at(
            lambda layer: layer.weight, modulation, jnp.zeros_like(modulation.weight)
        )
        modulation = eqx.tree_at(
            lambda layer: layer.bias, modulation, jnp.zeros_like(modulation.bias)
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
    """Particle self-attention followed by particle-to-prefix-summary attention."""

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
        particles = particles + self.likelihood_attention(query, memory, memory)

        h = _layernorm_tokens(self.ff_norm, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + h


#%% 9) Shared particle decoder and both posterior architectures
class ParticleDecoder(eqx.Module):
    """Decode [T,N,H] tokens as residual transports of one [N,D] prior cloud."""

    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear
    theta_dim: int = eqx.field(static=True)
    max_displacement: float = eqx.field(static=True)

    def __init__(self, cfg: GaussianBayesConfig, *, key: Array):
        self.theta_dim = cfg.theta_dim
        self.max_displacement = cfg.max_particle_displacement
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)
        output = eqx.nn.Linear(cfg.hidden_dim, cfg.theta_dim, key=key)

        # Identity pushforward at initialization: q starts as the prior cloud.
        output = eqx.tree_at(
            lambda layer: layer.weight, output, jnp.zeros_like(output.weight)
        )
        output = eqx.tree_at(
            lambda layer: layer.bias, output, jnp.zeros_like(output.bias)
        )
        self.displacement_head = output

    def __call__(self, particle_tokens: Array, prior_particles: Array) -> Array:
        particle_tokens = _time_layernorm_tokens(self.final_norm, particle_tokens)
        displacement = self.max_displacement * jnp.tanh(
            _time_linear_tokens(self.displacement_head, particle_tokens)
        )
        return prior_particles[None, :, :] + displacement  # [T,N,D]


class AdaLNPosteriorTransformer(eqx.Module):
    particle_in: eqx.nn.Linear
    condition_encoder: eqx.nn.MLP
    blocks: tuple[AdaLNParticleBlock, ...]
    decoder: ParticleDecoder

    def __init__(self, cfg: GaussianBayesConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 3)
        self.particle_in = eqx.nn.Linear(cfg.theta_dim, cfg.hidden_dim, key=keys[0])
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

    def __call__(self, prior_particles: Array, summaries: Array) -> Array:
        base = _linear_tokens(self.particle_in, prior_particles)  # [N,H]
        pooled = jnp.mean(summaries, axis=1)                     # [T,H]
        conditions = jax.vmap(self.condition_encoder)(pooled)    # [T,H]
        particles = jnp.broadcast_to(
            base[None, :, :], (summaries.shape[0],) + base.shape
        )                                                        # [T,N,H]
        for block in self.blocks:
            particles = jax.vmap(block)(particles, conditions)
        return self.decoder(particles, prior_particles)


class CrossAttentionPosteriorTransformer(eqx.Module):
    particle_in: eqx.nn.Linear
    blocks: tuple[ParticleLikelihoodCrossBlock, ...]
    decoder: ParticleDecoder

    def __init__(self, cfg: GaussianBayesConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 2)
        self.particle_in = eqx.nn.Linear(cfg.theta_dim, cfg.hidden_dim, key=keys[0])
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

    def __call__(self, prior_particles: Array, summaries: Array) -> Array:
        base = _linear_tokens(self.particle_in, prior_particles)  # [N,H]
        particles = jnp.broadcast_to(
            base[None, :, :], (summaries.shape[0],) + base.shape
        )                                                        # [T,N,H]
        for block in self.blocks:
            particles = jax.vmap(block)(particles, summaries)
        return self.decoder(particles, prior_particles)


#%% 10) End-to-end two-Transformer model
class ModeAParallelGaussianModel(eqx.Module):
    """Prefix Likelihood Transformer + one Posterior Transformer variant."""

    likelihood_transformer: CausalPrefixLikelihoodTransformer
    posterior_transformer: AdaLNPosteriorTransformer | CrossAttentionPosteriorTransformer
    conditioning: str = eqx.field(static=True)

    def __init__(
        self,
        cfg: GaussianBayesConfig,
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
        prior_particles: Array,  # [N,D]
        observations: Array,     # [T,D+1]
    ) -> tuple[Array, Array]:
        summaries = self.likelihood_transformer(observations)  # [T,R,H]
        posterior = self.posterior_transformer(prior_particles, summaries)
        return posterior, summaries                            # [T,N,D], [T,R,H]


#%% 11) Energy score and diagnostics against the exact Gaussian posterior
def energy_score_single(particles: Array, theta_true: Array) -> Array:
    """Exact energy score of q^N = N^{-1} sum_n delta_{theta_n}."""
    truth_distance = jnp.mean(
        jnp.sqrt(jnp.sum((particles - theta_true[None, :]) ** 2, axis=-1) + 1e-12)
    )
    differences = particles[:, None, :] - particles[None, :, :]
    squared_distance = jnp.sum(differences**2, axis=-1)
    off_diagonal = 1.0 - jnp.eye(particles.shape[0], dtype=particles.dtype)
    pairwise_distance = jnp.sum(
        jnp.sqrt(squared_distance + 1e-12) * off_diagonal
    ) / (particles.shape[0] ** 2)
    return truth_distance - 0.5 * pairwise_distance


def empirical_mean_cov_jax(particles: Array) -> tuple[Array, Array]:
    """Population-moment convention for the empirical particle measure."""
    mean = jnp.mean(particles, axis=0)
    centered = particles - mean[None, :]
    cov = (centered.T @ centered) / particles.shape[0]
    return mean, cov


def _prefix_metrics(
    particles: Array,
    theta_true: Array,
    exact_mean: Array,
    exact_cov: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    learned_mean, learned_cov = empirical_mean_cov_jax(particles)
    energy = energy_score_single(particles, theta_true)
    truth_rmse = jnp.sqrt(jnp.mean((learned_mean - theta_true) ** 2))
    exact_mean_rmse = jnp.sqrt(jnp.mean((learned_mean - exact_mean) ** 2))
    cov_fro_error = jnp.sqrt(jnp.sum((learned_cov - exact_cov) ** 2))
    learned_spread = jnp.trace(learned_cov) / learned_cov.shape[0]
    exact_spread = jnp.trace(exact_cov) / exact_cov.shape[0]
    return energy, truth_rmse, exact_mean_rmse, cov_fro_error, learned_spread, exact_spread


def _trajectory_metrics(
    posterior_sequence: Array,
    theta_true: Array,
    exact_mean_sequence: Array,
    exact_cov_sequence: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    return jax.vmap(_prefix_metrics)(
        posterior_sequence,
        jnp.broadcast_to(theta_true[None, :], exact_mean_sequence.shape),
        exact_mean_sequence,
        exact_cov_sequence,
    )


def batch_objective(
    model: ModeAParallelGaussianModel,
    batch: dict[str, Array],
    cfg: GaussianBayesConfig = CFG,
) -> tuple[Array, dict[str, Array]]:
    """Mean energy score over B x T; analytic posterior appears only in metrics."""
    predicted, _ = jax.vmap(model)(batch["prior_particles"], batch["observations"])
    energy, truth_rmse, exact_mean_rmse, cov_error, spread, exact_spread = jax.vmap(
        _trajectory_metrics
    )(
        predicted,
        batch["theta_true"],
        batch["exact_mean"],
        batch["exact_cov"],
    )
    loss = jnp.mean(energy)
    return loss, {
        "loss": loss,
        "energy_score": jnp.mean(energy),
        "final_energy_score": jnp.mean(energy[:, -1]),
        "truth_mean_rmse": jnp.mean(truth_rmse),
        "final_truth_mean_rmse": jnp.mean(truth_rmse[:, -1]),
        "exact_mean_rmse": jnp.mean(exact_mean_rmse),
        "final_exact_mean_rmse": jnp.mean(exact_mean_rmse[:, -1]),
        "exact_cov_fro_error": jnp.mean(cov_error),
        "final_exact_cov_fro_error": jnp.mean(cov_error[:, -1]),
        "posterior_spread": jnp.mean(spread),
        "final_spread": jnp.mean(spread[:, -1]),
        "exact_spread": jnp.mean(exact_spread),
        "final_exact_spread": jnp.mean(exact_spread[:, -1]),
        "energy_by_t": jnp.mean(energy, axis=0),
        "truth_rmse_by_t": jnp.mean(truth_rmse, axis=0),
        "exact_mean_rmse_by_t": jnp.mean(exact_mean_rmse, axis=0),
        "cov_error_by_t": jnp.mean(cov_error, axis=0),
        "spread_by_t": jnp.mean(spread, axis=0),
        "exact_spread_by_t": jnp.mean(exact_spread, axis=0),
    }


@eqx.filter_jit
def predict_batch(
    model: ModeAParallelGaussianModel,
    prior_particles: Array,
    observations: Array,
) -> tuple[Array, Array]:
    return jax.vmap(model)(prior_particles, observations)


@eqx.filter_jit
def evaluation_batch(
    model: ModeAParallelGaussianModel,
    batch: dict[str, Array],
    cfg: GaussianBayesConfig = CFG,
) -> dict[str, Array]:
    _, metrics = batch_objective(model, batch, cfg)
    return metrics


#%% 12) Evaluation helper with reproducible fresh prior clouds
def evaluate_model(
    model: ModeAParallelGaussianModel,
    dataset: dict[str, np.ndarray],
    cfg: GaussianBayesConfig = CFG,
    *,
    num_particles: int | None = None,
    max_trajectories: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> dict[str, np.ndarray | float]:
    n_total = len(dataset["theta_true"])
    if max_trajectories is not None:
        n_total = min(n_total, int(max_trajectories))
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    rng = np.random.default_rng(cfg.seed + 90_000 if seed is None else seed)

    scalar_names = [
        "loss", "energy_score", "final_energy_score",
        "truth_mean_rmse", "final_truth_mean_rmse",
        "exact_mean_rmse", "final_exact_mean_rmse",
        "exact_cov_fro_error", "final_exact_cov_fro_error",
        "posterior_spread", "final_spread", "exact_spread", "final_exact_spread",
    ]
    by_t_names = [
        "energy_by_t", "truth_rmse_by_t", "exact_mean_rmse_by_t",
        "cov_error_by_t", "spread_by_t", "exact_spread_by_t",
    ]
    scalar_values = {name: [] for name in scalar_names}
    by_t_values = {name: [] for name in by_t_names}
    weights = []

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch_np = make_batch_np(
            dataset, np.arange(start, stop), rng, cfg, num_particles=num_particles
        )
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        host = jax.device_get(evaluation_batch(model, batch, cfg))
        weights.append(stop - start)
        for name in scalar_names:
            scalar_values[name].append(float(host[name]))
        for name in by_t_names:
            by_t_values[name].append(np.asarray(host[name], dtype=np.float64))

    weights = np.asarray(weights, dtype=np.float64)
    result: dict[str, np.ndarray | float] = {}
    for name, values in scalar_values.items():
        result[name] = float(np.average(np.asarray(values), weights=weights))
    for name, values in by_t_values.items():
        result[name] = np.average(np.stack(values), axis=0, weights=weights)
    return result


#%% 13) Exact Gaussian posterior utilities for figures and oracle diagnostics
def exact_reference_for_prefix(
    rng: np.random.Generator,
    trajectory: dict[str, np.ndarray],
    prefix_length: int,
    cfg: GaussianBayesConfig = CFG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact mean, covariance, and independent exact posterior samples."""
    t = int(prefix_length) - 1
    mean = np.asarray(trajectory["exact_mean"][t])
    cov = np.asarray(trajectory["exact_cov"][t])
    particles = sample_exact_gaussian_np(
        rng, mean, cov, cfg.exact_reference_particles
    )
    return mean, cov, particles


def gaussian_ellipse_parameters(
    mean: np.ndarray,
    cov: np.ndarray,
    probability: float = 0.95,
) -> tuple[float, float, float]:
    """Width, height, angle for a 2D Gaussian credible ellipse.

    The script uses theta_dim=2 for all geometric figures.  For 95% mass,
    chi2_2(0.95)=5.991464547.  Other probability values use the familiar 1-sigma
    approximation only for visual convenience; the default is the one used below.
    """
    if len(mean) != 2:
        raise ValueError("Ellipse visualisations require theta_dim=2.")
    if abs(probability - 0.95) < 1e-12:
        radius_sq = 5.991464547107979
    else:
        radius_sq = -2.0 * math.log(max(1.0 - probability, 1e-12))
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    angle = math.degrees(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.0 * np.sqrt(radius_sq * eigenvalues)
    return float(width), float(height), float(angle)


def add_gaussian_ellipse(
    ax,
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    label: str = "exact 95% ellipse",
    linewidth: float = 2.0,
    linestyle: str = "--",
):
    width, height, angle = gaussian_ellipse_parameters(mean, cov, 0.95)
    ellipse = Ellipse(
        xy=np.asarray(mean), width=width, height=height, angle=angle,
        fill=False, linewidth=linewidth, linestyle=linestyle, label=label,
    )
    ax.add_patch(ellipse)
    return ellipse


#%% 14) Visualisation: architecture schematic
def plot_architecture_schematic(
    cfg: GaussianBayesConfig = CFG,
    destination: Path | None = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.8), constrained_layout=True)

    def draw_box(ax, xy, width, height, text, title=None):
        patch = FancyBboxPatch(
            xy, width, height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.3, facecolor="white", edgecolor="black",
        )
        ax.add_patch(patch)
        label = text if title is None else f"{title}\n{text}"
        ax.text(xy[0] + width/2, xy[1] + height/2, label,
                ha="center", va="center", fontsize=10)

    def arrow(ax, start, end, text=""):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=15, linewidth=1.25))
        if text:
            ax.text((start[0]+end[0])/2, (start[1]+end[1])/2 + 0.045,
                    text, ha="center", va="bottom", fontsize=8)

    for ax, mode in zip(axes, ["AdaLN", "Cross-attention"]):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(f"{mode} conditioning — all T Gaussian posteriors in parallel",
                     loc="left", fontweight="bold")
        draw_box(ax, (0.02, 0.58), 0.18, 0.25,
                 f"z_t=[x_t,y_t]\nshape [T,{cfg.theta_dim+1}]", "observations")
        draw_box(ax, (0.25, 0.54), 0.24, 0.33,
                 f"triangular prefix mask\nR={cfg.likelihood_summary_tokens} summary queries\nno observation positions",
                 "causal prefix-set\nLikelihood Transformer")
        draw_box(ax, (0.54, 0.58), 0.18, 0.25,
                 r"$\tilde{x}_{1:T}$" + "\nshape [T,R,H]", "prefix summaries")
        draw_box(ax, (0.02, 0.10), 0.18, 0.25,
                 f"N={cfg.num_particles} iid draws\nfrom N(mu_0,Sigma_0)", "prior cloud")
        posterior_text = (
            "particle self-attention\nAdaLN(summary_t)\nshared weights for every t"
            if mode == "AdaLN" else
            "particle self-attention\nparticle -> R summaries\ncross-attention"
        )
        draw_box(ax, (0.54, 0.08), 0.22, 0.30, posterior_text, "Posterior Transformer")
        draw_box(ax, (0.81, 0.14), 0.17, 0.20,
                 f"[T,N,{cfg.theta_dim}]\nq_phi(theta | D_t)", "posterior clouds")
        arrow(ax, (0.20, 0.705), (0.25, 0.705))
        arrow(ax, (0.49, 0.705), (0.54, 0.705), "all prefixes")
        arrow(ax, (0.20, 0.225), (0.54, 0.225), "broadcast over T")
        arrow(ax, (0.63, 0.58), (0.65, 0.38),
              "pool per t" if mode == "AdaLN" else "R tokens per t")
        arrow(ax, (0.76, 0.23), (0.81, 0.23))

    fig.suptitle(
        "Mode A conjugate benchmark: exact p(theta | D_t) is available for every t",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


#%% 15) Visualisation: one Gaussian trajectory and its exact posterior contraction
def plot_gaussian_trajectory(
    trajectory: dict[str, np.ndarray],
    cfg: GaussianBayesConfig = CFG,
    destination: Path | None = None,
    title: str = "Mode-A linear-Gaussian trajectory",
):
    if cfg.theta_dim != 2:
        raise ValueError("This visualisation is written for theta_dim=2.")
    theta_true = np.asarray(trajectory["theta_true"])
    observations = np.asarray(trajectory["observations"])
    x = observations[:, :2]
    y = observations[:, 2]
    expected = x @ theta_true
    exact_mean = np.asarray(trajectory["exact_mean"])
    exact_cov = np.asarray(trajectory["exact_cov"])

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0), constrained_layout=True)

    scatter = axes[0].scatter(x[:, 0], x[:, 1], c=y, s=70, marker="s")
    fig.colorbar(scatter, ax=axes[0], label="observed y_t")
    axes[0].arrow(0, 0, theta_true[0], theta_true[1], width=0.025,
                  length_includes_head=True, label="theta*")
    axes[0].scatter([theta_true[0]], [theta_true[1]], marker="*", s=190,
                    edgecolors="black", linewidths=0.8, label="theta*")
    axes[0].set_xlim(cfg.design_low, cfg.design_high)
    axes[0].set_ylim(cfg.design_low, cfg.design_high)
    axes[0].set_aspect("equal")
    axes[0].set_title("Design vectors x_t, coloured by y_t")
    axes[0].set_xlabel("x_{t,1}"); axes[0].set_ylabel("x_{t,2}")
    axes[0].legend(fontsize=8)

    t = np.arange(1, len(y)+1)
    axes[1].plot(t, y, marker="o", markersize=4, label="observed y_t")
    axes[1].plot(t, expected, linestyle="--", label="x_t^T theta*")
    axes[1].set_xlabel("observation index t")
    axes[1].set_ylabel("scalar outcome")
    axes[1].set_title("Gaussian likelihood trajectory")
    axes[1].grid(alpha=0.25); axes[1].legend()

    axes[2].plot(exact_mean[:, 0], exact_mean[:, 1], marker="o", markersize=3,
                 label="exact posterior mean")
    axes[2].scatter(theta_true[0], theta_true[1], marker="*", s=190,
                    edgecolors="black", linewidths=0.8, label="theta*")
    for idx in np.unique(np.rint(np.geomspace(1, len(y), 5)).astype(int)):
        add_gaussian_ellipse(
            axes[2], exact_mean[idx-1], exact_cov[idx-1],
            label="exact 95% ellipse" if idx == 1 else "", linewidth=1.2,
        )
    axes[2].set_aspect("equal")
    axes[2].set_title("Closed-form posterior contraction")
    axes[2].set_xlabel("theta_1"); axes[2].set_ylabel("theta_2")
    axes[2].grid(alpha=0.25); axes[2].legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


#%% 16) Visualisation: prior -> learned posterior evolution with exact Gaussian ellipses
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
    model: ModeAParallelGaussianModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: GaussianBayesConfig = CFG,
    destination: Path | None = None,
    title: str = "Posterior evolution",
):
    if cfg.theta_dim != 2:
        raise ValueError("Posterior-evolution figures require theta_dim=2.")
    theta_true = np.asarray(trajectory["theta_true"])
    observations = np.asarray(trajectory["observations"])
    exact_mean = np.asarray(trajectory["exact_mean"])
    exact_cov = np.asarray(trajectory["exact_cov"])
    predicted, _ = model(jnp.asarray(prior_particles), jnp.asarray(observations))
    predicted = np.asarray(jax.device_get(predicted))

    prefixes = select_prefixes(len(observations), 5)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.5), constrained_layout=True)
    axes = axes.ravel()
    clouds = [prior_particles] + [predicted[t-1] for t in prefixes]
    labels = ["prior p(theta)"] + [f"q_phi(theta | D_{t})" for t in prefixes]

    exact_reference_points = np.concatenate([
        exact_mean, theta_true[None, :], np.asarray(cfg.prior_mean)[None, :]
    ], axis=0)
    all_points = np.concatenate(clouds + [exact_reference_points], axis=0)
    lim = max(3.0 * cfg.prior_std, 1.12 * float(np.quantile(np.abs(all_points), 0.995)))

    for panel_index, (ax, cloud, label) in enumerate(zip(axes, clouds, labels)):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=16, alpha=0.32,
                   label="learned particles" if panel_index else "prior particles")
        ax.scatter(theta_true[0], theta_true[1], marker="*", s=190,
                   edgecolors="black", linewidths=0.8, label="theta*")
        if panel_index == 0:
            mean = prior_mean_np(cfg)
            cov = prior_cov_np(cfg)
            exact_label = "prior 95% ellipse"
        else:
            t = prefixes[panel_index-1]
            mean = exact_mean[t-1]
            cov = exact_cov[t-1]
            exact_label = "exact posterior 95% ellipse"
            ax.scatter(mean[0], mean[1], marker="x", s=70, linewidths=2,
                       label="exact posterior mean")
        add_gaussian_ellipse(ax, mean, cov, label=exact_label)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.grid(alpha=0.2); ax.set_title(label); ax.legend(fontsize=7)
        ax.set_xlabel("theta_1"); ax.set_ylabel("theta_2")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


#%% 17) Visualisation: exact final posterior versus both learned clouds
def plot_exact_reference_comparison(
    models: dict[str, ModeAParallelGaussianModel],
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: GaussianBayesConfig = CFG,
    destination: Path | None = None,
):
    """The reference column is now genuinely exact, not SNIS or another teacher."""
    if cfg.theta_dim != 2:
        raise ValueError("Reference comparison requires theta_dim=2.")
    observations = np.asarray(trajectory["observations"])
    theta_true = np.asarray(trajectory["theta_true"])
    mean = np.asarray(trajectory["exact_mean"][-1])
    cov = np.asarray(trajectory["exact_cov"][-1])
    rng = np.random.default_rng(cfg.seed + 123_000)
    exact_particles = sample_exact_gaussian_np(
        rng, mean, cov, cfg.exact_reference_particles
    )

    learned = {}
    for name, model in models.items():
        posterior, _ = model(jnp.asarray(prior_particles), jnp.asarray(observations))
        learned[name] = np.asarray(jax.device_get(posterior[-1]))

    columns = [("Prior", prior_particles), ("Exact posterior", exact_particles)]
    columns.extend((name, learned[name]) for name in models)
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2*len(columns), 4.6),
                             constrained_layout=True)
    all_points = np.concatenate([c for _, c in columns] + [theta_true[None, :]])
    lim = max(3.0*cfg.prior_std, 1.12*float(np.quantile(np.abs(all_points), 0.997)))

    for ax, (name, cloud) in zip(axes, columns):
        ax.scatter(cloud[:, 0], cloud[:, 1], s=13, alpha=0.28)
        ax.scatter(theta_true[0], theta_true[1], marker="*", s=190,
                   edgecolors="black", linewidths=0.8, label="theta*")
        if name == "Prior":
            add_gaussian_ellipse(ax, prior_mean_np(cfg), prior_cov_np(cfg),
                                 label="prior 95%")
        else:
            add_gaussian_ellipse(ax, mean, cov, label="exact 95%")
            ax.scatter(mean[0], mean[1], marker="x", s=70, linewidths=2,
                       label="exact mean")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.grid(alpha=0.2); ax.set_title(name, fontweight="bold")
        ax.set_xlabel("theta_1"); ax.set_ylabel("theta_2")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Final-prefix posterior: learned particle transports versus the closed-form Gaussian",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


#%% 18) Unified training-diagnostics visualisation
def plot_training_diagnostics(
    history: dict[str, list],
    best_epoch: int,
    conditioning: str,
    destination: Path | None = None,
):
    steps = np.arange(1, len(history["step_loss"])+1)
    epochs = np.arange(1, len(history["epoch_train_loss"])+1)
    fig = plt.figure(figsize=(12.0, 10.2), constrained_layout=True)
    fig.suptitle(f"{conditioning}: energy-score training diagnostics",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.4, 1.8])

    step_panels = [
        ("step_loss", "energy-score loss"),
        ("step_exact_mean_rmse", "learned mean vs exact mean RMSE"),
        ("step_exact_cov_error", "learned covariance vs exact Frobenius error"),
        ("step_grad_norm", "gradient norm"),
    ]
    for ax, (key, title) in zip([fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                                 fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])],
                                step_panels):
        values = np.asarray(history[key], dtype=float)
        ax.plot(steps, values, linewidth=0.8, alpha=0.8)
        if len(values) >= 20:
            window = max(5, len(values)//100)
            smoothed = np.convolve(values, np.ones(window)/window, mode="valid")
            ax.plot(steps[window-1:], smoothed, linewidth=1.8,
                    label=f"moving avg ({window})")
            ax.legend(fontsize=7, frameon=False)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("gradient step"); ax.grid(alpha=0.2)
        ax.set_yscale("symlog", linthresh=1e-8)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(epochs, history["epoch_train_loss"], marker="o", markersize=3,
            label="train energy score")
    ax.plot(epochs, history["epoch_val_loss"], marker="o", markersize=3,
            label="validation energy score")
    ax.plot(epochs, history["epoch_val_exact_mean_rmse"], linestyle="--",
            label="validation exact-mean RMSE")
    ax.plot(epochs, history["epoch_val_exact_cov_error"], linestyle=":",
            label="validation exact-cov error")
    ax.axvline(best_epoch, linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"best epoch {best_epoch}")
    ax.set_xlabel("epoch"); ax.set_title("Epoch-level training and exact-posterior diagnostics",
                                         loc="left", fontweight="bold")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


#%% 19) Training function shared by both architectural variants
def train_variant(
    conditioning: ConditioningMode,
    train_data: dict[str, np.ndarray],
    eval_data: dict[str, np.ndarray],
    fixed_trajectory: dict[str, np.ndarray],
    fixed_prior_particles: np.ndarray,
    run_dir: Path,
    cfg: GaussianBayesConfig = CFG,
) -> dict[str, Any]:
    """Train one model pair; all T prefix losses are parallel and ES-only."""
    variant_dir = run_dir / conditioning
    (variant_dir / "plots").mkdir(parents=True, exist_ok=True)
    (variant_dir / "artefacts").mkdir(parents=True, exist_ok=True)

    model_seed_offset = 0 if conditioning == "adaln" else 10_000
    model = ModeAParallelGaussianModel(
        cfg, conditioning=conditioning,
        key=jax.random.key(cfg.seed + model_seed_offset),
    )
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
        "step_final_energy_score": [],
        "step_truth_mean_rmse": [],
        "step_exact_mean_rmse": [],
        "step_exact_cov_error": [],
        "step_grad_norm": [],
        "epoch_train_loss": [],
        "epoch_val_loss": [],
        "epoch_val_final_energy_score": [],
        "epoch_val_truth_mean_rmse": [],
        "epoch_val_exact_mean_rmse": [],
        "epoch_val_exact_cov_error": [],
        "epoch_val_energy_by_t": [],
        "epoch_val_exact_mean_rmse_by_t": [],
        "epoch_val_cov_error_by_t": [],
        "epoch_val_spread_by_t": [],
        "epoch_val_exact_spread_by_t": [],
    }

    plot_posterior_evolution(
        model, fixed_trajectory, fixed_prior_particles, cfg,
        variant_dir / "plots" / "fixed_trajectory_before_training.png",
        f"{conditioning}: before training (identity transport)",
    )
    initial_metrics = evaluate_model(model, eval_data, cfg, seed=cfg.seed + 91_000)
    print(f"[{conditioning}] initial validation ES = {initial_metrics['loss']:.6f}")

    visualisation_epochs = sorted(set(
        max(1, int(math.ceil(fraction * cfg.epochs / 10.0)))
        for fraction in range(1, 11)
    ))
    rng = np.random.default_rng(cfg.seed + 30_000)  # identical data order across variants
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    training_started_at = time.time()
    n_train = len(train_data["theta_true"])

    for epoch in range(1, cfg.epochs + 1):
        epoch_started_at = time.time()
        order = rng.permutation(n_train)
        train_losses = []
        n_steps = n_train // cfg.batch_size
        progress = tqdm(
            range(n_steps),
            desc=f"{conditioning:>15s} epoch {epoch:03d}/{cfg.epochs:03d}",
            dynamic_ncols=True, leave=True,
        )

        for batch_index in progress:
            start = batch_index * cfg.batch_size
            indices = order[start:start + cfg.batch_size]
            batch_np = make_batch_np(train_data, indices, rng, cfg)
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            model, opt_state, loss, metrics, grad_norm = train_step(model, opt_state, batch)
            host = jax.device_get(metrics)
            host_loss = float(jax.device_get(loss))
            host_grad = float(jax.device_get(grad_norm))
            global_step += 1
            train_losses.append(host_loss)
            history["step_loss"].append(host_loss)
            history["step_final_energy_score"].append(float(host["final_energy_score"]))
            history["step_truth_mean_rmse"].append(float(host["truth_mean_rmse"]))
            history["step_exact_mean_rmse"].append(float(host["exact_mean_rmse"]))
            history["step_exact_cov_error"].append(float(host["exact_cov_fro_error"]))
            history["step_grad_norm"].append(host_grad)
            progress.set_postfix(
                ES=f"{host_loss:.4f}",
                exact_mu=f"{float(host['exact_mean_rmse']):.4f}",
                exact_cov=f"{float(host['exact_cov_fro_error']):.4f}",
            )

        epoch_train_loss = float(np.mean(train_losses))
        val = evaluate_model(model, eval_data, cfg, seed=cfg.seed + 91_000)
        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_val_loss"].append(float(val["loss"]))
        history["epoch_val_final_energy_score"].append(float(val["final_energy_score"]))
        history["epoch_val_truth_mean_rmse"].append(float(val["truth_mean_rmse"]))
        history["epoch_val_exact_mean_rmse"].append(float(val["exact_mean_rmse"]))
        history["epoch_val_exact_cov_error"].append(float(val["exact_cov_fro_error"]))
        history["epoch_val_energy_by_t"].append(np.asarray(val["energy_by_t"]))
        history["epoch_val_exact_mean_rmse_by_t"].append(np.asarray(val["exact_mean_rmse_by_t"]))
        history["epoch_val_cov_error_by_t"].append(np.asarray(val["cov_error_by_t"]))
        history["epoch_val_spread_by_t"].append(np.asarray(val["spread_by_t"]))
        history["epoch_val_exact_spread_by_t"].append(np.asarray(val["exact_spread_by_t"]))

        save_model(variant_dir / "artefacts" / "model_last.eqx", model)
        if epoch % cfg.save_every_epochs == 0:
            save_model(variant_dir / "artefacts" / f"model_epoch_{epoch:04d}.eqx", model)
        if float(val["loss"]) < best_val_loss:
            best_val_loss = float(val["loss"])
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
                "exact_posterior_used_in_loss": False,
            },
        )

        print(
            f"[{conditioning}] epoch {epoch:03d}: "
            f"train ES={epoch_train_loss:.6f} | val ES={float(val['loss']):.6f} | "
            f"exact-mean RMSE={float(val['exact_mean_rmse']):.5f} | "
            f"exact-cov error={float(val['exact_cov_fro_error']):.5f} | "
            f"{time.time()-epoch_started_at:.1f}s"
        )
        if epoch in visualisation_epochs:
            plot_posterior_evolution(
                model, fixed_trajectory, fixed_prior_particles, cfg,
                variant_dir / "plots" / f"fixed_trajectory_epoch_{epoch:04d}.png",
                f"{conditioning}: posterior evolution after epoch {epoch}",
            )

    best_model = load_model(
        variant_dir / "artefacts" / "model_best.eqx", cfg, conditioning,
        key=jax.random.key(0),
    )
    final_metrics = evaluate_model(best_model, eval_data, cfg, seed=cfg.seed + 91_000)
    plot_posterior_evolution(
        best_model, fixed_trajectory, fixed_prior_particles, cfg,
        variant_dir / "plots" / "fixed_trajectory_best_model.png",
        f"{conditioning}: best model (epoch {best_epoch})",
    )
    plot_training_diagnostics(
        history, best_epoch, conditioning,
        variant_dir / "plots" / "training_diagnostics.png",
    )
    print(
        f"[{conditioning}] complete in "
        f"{datetime.timedelta(seconds=int(time.time()-training_started_at))}; "
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

train_rng = np.random.default_rng(CFG.seed + 1_000)
eval_rng = np.random.default_rng(CFG.seed + 2_000)
train_data = simulate_mode_a_trajectories(
    train_rng, CFG.n_train_trajectories, CFG.trajectory_length, CFG
)
eval_data = simulate_mode_a_trajectories(
    eval_rng, CFG.n_eval_trajectories, CFG.trajectory_length, CFG
)

fixed_trajectory = {
    "theta_true": eval_data["theta_true"][0],
    "observations": eval_data["observations"][0],
    "exact_mean": eval_data["exact_mean"][0],
    "exact_cov": eval_data["exact_cov"][0],
}
fixed_prior_particles = sample_prior_np(
    np.random.default_rng(CFG.seed + 3_000), CFG.num_particles, CFG
)
np.savez_compressed(
    run_dir / "artefacts" / "fixed_trajectory.npz",
    theta_true=fixed_trajectory["theta_true"],
    observations=fixed_trajectory["observations"],
    exact_mean=fixed_trajectory["exact_mean"],
    exact_cov=fixed_trajectory["exact_cov"],
    prior_particles=fixed_prior_particles,
)

plot_architecture_schematic(CFG, run_dir / "plots" / "architecture_schematic.png")
plot_gaussian_trajectory(
    fixed_trajectory, CFG, run_dir / "plots" / "fixed_gaussian_trajectory.png"
)


#%% 21) Train BOTH conditioning architectures on the same Mode-A trajectories
results: dict[str, dict[str, Any]] = {}
for conditioning_name in CFG.architectures_to_train:
    if conditioning_name not in {"adaln", "cross_attention"}:
        raise ValueError(f"Unknown architecture {conditioning_name!r}.")
    results[conditioning_name] = train_variant(
        conditioning_name,  # type: ignore[arg-type]
        train_data, eval_data, fixed_trajectory, fixed_prior_particles, run_dir, CFG
    )
models = {name: result["model"] for name, result in results.items()}


#%% 22) Direct final-prefix comparison with the CLOSED-FORM posterior
plot_exact_reference_comparison(
    models, fixed_trajectory, fixed_prior_particles, CFG,
    run_dir / "plots" / "exact_posterior_comparison.png",
)


#%% 23) Architecture comparison across prefix length, now with oracle errors
def plot_architecture_comparison(
    results: dict[str, dict[str, Any]],
    destination: Path | None = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    axes = axes.ravel()
    for name, result in results.items():
        m = result["final_metrics"]
        t = np.arange(1, len(m["energy_by_t"])+1)
        axes[0].plot(t, m["energy_by_t"], label=name)
        axes[1].plot(t, m["exact_mean_rmse_by_t"], label=name)
        axes[2].plot(t, m["cov_error_by_t"], label=name)
        axes[3].plot(t, m["spread_by_t"], label=f"{name} learned")
    exact_t = np.arange(1, len(next(iter(results.values()))["final_metrics"]["exact_spread_by_t"])+1)
    exact_spread = next(iter(results.values()))["final_metrics"]["exact_spread_by_t"]
    axes[3].plot(exact_t, exact_spread, linestyle="--", linewidth=2,
                 label="exact Gaussian")
    titles = ["Energy score", "Mean error to exact posterior",
              "Covariance error to exact posterior", "Posterior spread"]
    ylabels = ["energy score", "RMSE", "Frobenius error", "mean variance"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("prefix length t")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("AdaLN versus cross-attention against the analytic Gaussian posterior",
                 fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig); plt.close(fig)


plot_architecture_comparison(
    results, run_dir / "plots" / "architecture_comparison_by_prefix.png"
)


#%% 24) Numerical architectural checks: causality, prefix-set invariance, particle equivariance
def structural_checks(
    model: ModeAParallelGaussianModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: GaussianBayesConfig = CFG,
) -> dict[str, float]:
    obs = np.asarray(trajectory["observations"], dtype=np.float32)
    prior = np.asarray(prior_particles, dtype=np.float32)
    rng = np.random.default_rng(cfg.seed + 500_000)
    t = max(2, len(obs)//2)

    baseline, baseline_summary = model(jnp.asarray(prior), jnp.asarray(obs))
    baseline = np.asarray(jax.device_get(baseline))
    baseline_summary = np.asarray(jax.device_get(baseline_summary))

    future = obs.copy()
    if t < len(obs):
        future[t:, :cfg.theta_dim] = rng.uniform(
            cfg.design_low, cfg.design_high, size=future[t:, :cfg.theta_dim].shape
        )
        future[t:, cfg.theta_dim] += rng.normal(0.0, 5.0, size=len(obs)-t)
    causal_output, causal_summary = model(jnp.asarray(prior), jnp.asarray(future))
    causal_output = np.asarray(jax.device_get(causal_output))
    causal_summary = np.asarray(jax.device_get(causal_summary))

    truncated = obs[:t].copy()
    permutation = rng.permutation(t)
    output_a, summary_a = model(jnp.asarray(prior), jnp.asarray(truncated))
    output_b, summary_b = model(jnp.asarray(prior), jnp.asarray(truncated[permutation]))
    output_a = np.asarray(jax.device_get(output_a)); output_b = np.asarray(jax.device_get(output_b))
    summary_a = np.asarray(jax.device_get(summary_a)); summary_b = np.asarray(jax.device_get(summary_b))

    particle_perm = rng.permutation(len(prior)); inverse = np.argsort(particle_perm)
    permuted_output, _ = model(jnp.asarray(prior[particle_perm]), jnp.asarray(obs))
    permuted_output = np.asarray(jax.device_get(permuted_output))[:, inverse]

    return {
        "causal_output_max_abs_error": float(np.max(np.abs(causal_output[:t]-baseline[:t]))),
        "causal_summary_max_abs_error": float(np.max(np.abs(causal_summary[:t]-baseline_summary[:t]))),
        "prefix_permutation_output_max_abs_error": float(np.max(np.abs(output_a[-1]-output_b[-1]))),
        "prefix_permutation_summary_max_abs_error": float(np.max(np.abs(summary_a[-1]-summary_b[-1]))),
        "particle_equivariance_max_abs_error": float(np.max(np.abs(permuted_output-baseline))),
    }


structure_results = {
    name: structural_checks(model, fixed_trajectory, fixed_prior_particles, CFG)
    for name, model in models.items()
}
print("Structural checks:")
for name, checks in structure_results.items():
    print(name, checks)
save_json(run_dir / "artefacts" / "structural_checks.json", structure_results)

fig, ax = plt.subplots(figsize=(13.0, 5.0), constrained_layout=True)
metric_names = list(next(iter(structure_results.values())).keys())
x = np.arange(len(metric_names)); width = 0.8/len(structure_results)
for i, (name, values) in enumerate(structure_results.items()):
    heights = [max(values[m], 1e-16) for m in metric_names]
    ax.bar(x + (i-(len(structure_results)-1)/2)*width, heights, width=width, label=name)
ax.set_yscale("log"); ax.set_xticks(x)
ax.set_xticklabels(["causal\nposterior", "causal\nsummary", "prefix-set\nposterior",
                    "prefix-set\nsummary", "particle\nequivariance"])
ax.set_ylabel("max absolute discrepancy")
ax.set_title("Architectural identities: expected near floating-point precision", fontweight="bold")
ax.grid(axis="y", alpha=0.25); ax.legend()
fig.savefig(run_dir / "plots" / "structural_checks.png", dpi=170)
display(fig); plt.close(fig)


#%% 25) Numerical theorem sanity check: single-global-truth energy-score collapse
def mode_b_collapse_curve(
    theta_fixed: np.ndarray,
    cfg: GaussianBayesConfig = CFG,
) -> tuple[np.ndarray, np.ndarray]:
    """Energy score of increasingly concentrated Gaussian clouds around one fixed truth.

    This does not train the Transformer.  It simply illustrates why reusing one global
    theta* as every training target drives a proper score toward delta_{theta*}.
    """
    rng = np.random.default_rng(cfg.seed + 600_000)
    scales = np.geomspace(1e-3, 2.0, 50)
    scores = []
    for scale in scales:
        cloud = theta_fixed[None, :] + scale * rng.normal(
            size=(max(cfg.num_particles, 256), cfg.theta_dim)
        )
        diff_truth = np.linalg.norm(cloud-theta_fixed[None, :], axis=1).mean()
        diff_pairs = np.linalg.norm(cloud[:,None,:]-cloud[None,:,:], axis=-1).mean()
        scores.append(diff_truth - 0.5*diff_pairs)
    return scales, np.asarray(scores)


collapse_scales, collapse_scores = mode_b_collapse_curve(fixed_trajectory["theta_true"], CFG)
fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
ax.plot(collapse_scales, collapse_scores, marker="o", markersize=3)
ax.set_xscale("log"); ax.set_xlabel("cloud standard deviation around one global theta*")
ax.set_ylabel("energy score against that same theta*")
ax.set_title("Mode-B sanity check: one global target rewards point-mass collapse", fontweight="bold")
ax.grid(alpha=0.25)
fig.savefig(run_dir / "plots" / "mode_b_collapse_sanity.png", dpi=170)
display(fig); plt.close(fig)


#%% 26) Limit study N -> large: particle accuracy, exact moments, and runtime
particle_study: dict[str, dict[str, list[float]]] = {
    name: {"N": [], "energy": [], "mean_error": [], "cov_error": [], "seconds": []}
    for name in models
}
for name, model in models.items():
    for n_particles in CFG.particle_limit_values:
        started = time.perf_counter()
        metrics = evaluate_model(
            model, eval_data, CFG,
            num_particles=n_particles,
            max_trajectories=CFG.limit_eval_trajectories,
            seed=CFG.seed + 700_000,
        )
        elapsed = time.perf_counter()-started
        particle_study[name]["N"].append(n_particles)
        particle_study[name]["energy"].append(float(metrics["final_energy_score"]))
        particle_study[name]["mean_error"].append(float(metrics["final_exact_mean_rmse"]))
        particle_study[name]["cov_error"].append(float(metrics["final_exact_cov_fro_error"]))
        particle_study[name]["seconds"].append(elapsed)

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.7), constrained_layout=True)
axes = axes.ravel()
for name, values in particle_study.items():
    axes[0].plot(values["N"], values["energy"], marker="o", label=name)
    axes[1].plot(values["N"], values["mean_error"], marker="o", label=name)
    axes[2].plot(values["N"], values["cov_error"], marker="o", label=name)
    axes[3].plot(values["N"], values["seconds"], marker="o", label=name)
for ax in axes:
    ax.set_xscale("log", base=2); ax.set_xlabel("particles N"); ax.grid(alpha=0.25); ax.legend()
axes[0].set_title("Final energy score"); axes[1].set_title("Exact-mean RMSE")
axes[2].set_title("Exact-covariance error"); axes[3].set_title("Evaluation wall time")
fig.suptitle("Finite-particle study: oracle accuracy and O(N^2) energy/self-attention pressure",
             fontsize=14, fontweight="bold")
fig.savefig(run_dir / "plots" / "particle_limit_study.png", dpi=170)
display(fig); plt.close(fig)


#%% 27) Limit study T -> larger: trained horizon and extrapolated Gaussian prefixes
long_eval_rng = np.random.default_rng(CFG.seed + 800_000)
long_eval_data = simulate_mode_a_trajectories(
    long_eval_rng, CFG.limit_eval_trajectories, CFG.long_trajectory_length, CFG
)
long_trajectory_study = {
    name: evaluate_model(
        model, long_eval_data, CFG,
        max_trajectories=CFG.limit_eval_trajectories,
        seed=CFG.seed + 801_000,
    )
    for name, model in models.items()
}
fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), constrained_layout=True)
axes = axes.ravel()
for name, m in long_trajectory_study.items():
    t = np.arange(1, len(m["energy_by_t"])+1)
    axes[0].plot(t, m["energy_by_t"], label=name)
    axes[1].plot(t, m["exact_mean_rmse_by_t"], label=name)
    axes[2].plot(t, m["cov_error_by_t"], label=name)
    axes[3].plot(t, m["spread_by_t"], label=f"{name} learned")
exact_spread = next(iter(long_trajectory_study.values()))["exact_spread_by_t"]
axes[3].plot(np.arange(1, len(exact_spread)+1), exact_spread, linestyle="--",
             linewidth=2, label="exact Gaussian")
for ax in axes:
    ax.axvline(CFG.trajectory_length, linestyle="--", linewidth=1.0, label="training horizon")
    ax.set_xlabel("prefix length t"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
axes[0].set_title("Energy score"); axes[1].set_title("Exact-mean RMSE")
axes[2].set_title("Exact-covariance error"); axes[3].set_title("Posterior spread")
fig.suptitle("Trajectory-length study: exact conjugate posterior remains available beyond training T",
             fontsize=14, fontweight="bold")
fig.savefig(run_dir / "plots" / "trajectory_length_limit_study.png", dpi=170)
display(fig); plt.close(fig)


#%% 28) Limit study M -> large: empirical trajectory-average convergence
def per_trajectory_final_energy(
    model: ModeAParallelGaussianModel,
    dataset: dict[str, np.ndarray],
    cfg: GaussianBayesConfig = CFG,
    *, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, len(dataset["theta_true"]), cfg.batch_size):
        stop = min(start+cfg.batch_size, len(dataset["theta_true"]))
        batch_np = make_batch_np(dataset, np.arange(start, stop), rng, cfg)
        batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
        predicted, _ = predict_batch(model, batch["prior_particles"], batch["observations"])
        scores = jax.vmap(energy_score_single)(predicted[:, -1], batch["theta_true"])
        values.append(np.asarray(jax.device_get(scores), dtype=np.float64))
    return np.concatenate(values)


mc_pool_rng = np.random.default_rng(CFG.seed + 900_000)
mc_pool = simulate_mode_a_trajectories(
    mc_pool_rng, max(CFG.trajectory_mc_values), CFG.trajectory_length, CFG
)
trajectory_mc_study = {}
for name, model in models.items():
    scores = per_trajectory_final_energy(model, mc_pool, CFG, seed=CFG.seed + 901_000)
    rng = np.random.default_rng(CFG.seed + 902_000)
    scores = scores[rng.permutation(len(scores))]
    means, lower, upper = [], [], []
    for m in CFG.trajectory_mc_values:
        sample = scores[:m]
        mean = float(np.mean(sample))
        se = float(np.std(sample, ddof=1)/math.sqrt(m)) if m > 1 else 0.0
        means.append(mean); lower.append(mean-1.96*se); upper.append(mean+1.96*se)
    trajectory_mc_study[name] = {
        "M": np.asarray(CFG.trajectory_mc_values),
        "mean": np.asarray(means), "lower": np.asarray(lower), "upper": np.asarray(upper),
    }

fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
for name, v in trajectory_mc_study.items():
    ax.plot(v["M"], v["mean"], marker="o", label=name)
    ax.fill_between(v["M"], v["lower"], v["upper"], alpha=0.16)
ax.set_xscale("log", base=2); ax.set_xlabel("independent evaluation trajectories M")
ax.set_ylabel("empirical mean final-prefix energy score")
ax.set_title("M -> large: Monte Carlo estimate of population risk stabilises", fontweight="bold")
ax.grid(alpha=0.25); ax.legend()
fig.savefig(run_dir / "plots" / "trajectory_count_limit_study.png", dpi=170)
display(fig); plt.close(fig)


#%% 29) Finite prior-cloud stability for the SAME observations
def prior_cloud_stability_study(
    models: dict[str, ModeAParallelGaussianModel],
    trajectory: dict[str, np.ndarray],
    cfg: GaussianBayesConfig = CFG,
) -> dict[str, dict[str, np.ndarray]]:
    observations = np.asarray(trajectory["observations"])
    exact_mean = np.asarray(trajectory["exact_mean"][-1])
    study = {}
    for name, model in models.items():
        stds, biases = [], []
        for n_particles in cfg.particle_limit_values:
            means = []
            for repeat in range(cfg.prior_resample_repeats):
                rng = np.random.default_rng(cfg.seed + 1_000_000 + 1000*n_particles + repeat)
                prior = sample_prior_np(rng, n_particles, cfg)
                posterior, _ = model(jnp.asarray(prior), jnp.asarray(observations))
                final = np.asarray(jax.device_get(posterior[-1]))
                means.append(final.mean(axis=0))
            means = np.stack(means)
            stds.append(float(np.sqrt(np.mean(np.var(means, axis=0, ddof=1)))))
            biases.append(float(np.sqrt(np.mean((means.mean(axis=0)-exact_mean)**2))))
        study[name] = {
            "num_particles": np.asarray(cfg.particle_limit_values),
            "posterior_mean_sd_across_prior_clouds": np.asarray(stds),
            "posterior_mean_bias_to_exact": np.asarray(biases),
        }
    return study


prior_cloud_study = prior_cloud_stability_study(models, fixed_trajectory, CFG)
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
for name, v in prior_cloud_study.items():
    axes[0].plot(v["num_particles"], v["posterior_mean_sd_across_prior_clouds"], marker="o", label=name)
    axes[1].plot(v["num_particles"], v["posterior_mean_bias_to_exact"], marker="o", label=name)
for ax in axes:
    ax.set_xscale("log", base=2); ax.set_xlabel("prior particles N"); ax.grid(alpha=0.25); ax.legend()
axes[0].set_title("Variation across fresh prior clouds")
axes[0].set_ylabel("RMS SD of learned posterior mean")
axes[1].set_title("Bias of average learned mean to exact mean")
axes[1].set_ylabel("RMSE")
fig.suptitle("Finite numerical prior representation for fixed observed Gaussian data",
             fontsize=14, fontweight="bold")
fig.savefig(run_dir / "plots" / "prior_cloud_stability.png", dpi=170)
display(fig); plt.close(fig)


#%% 30) Causal truncation consistency: full T versus separately truncated inference
def truncation_consistency_study(
    model: ModeAParallelGaussianModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(trajectory["observations"])
    full, _ = model(jnp.asarray(prior_particles), jnp.asarray(observations))
    full = np.asarray(jax.device_get(full))
    prefix_values = select_prefixes(len(observations), 6)
    errors = []
    for t in prefix_values:
        truncated, _ = model(jnp.asarray(prior_particles), jnp.asarray(observations[:t]))
        truncated = np.asarray(jax.device_get(truncated))
        errors.append(float(np.max(np.abs(full[t-1]-truncated[-1]))))
    return np.asarray(prefix_values), np.asarray(errors)


fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
for name, model in models.items():
    t_values, errors = truncation_consistency_study(model, fixed_trajectory, fixed_prior_particles)
    ax.plot(t_values, np.maximum(errors, 1e-16), marker="o", label=name)
ax.set_yscale("log"); ax.set_xlabel("prefix length t")
ax.set_ylabel("max |full-run q_t - truncated-run q_t|")
ax.set_title("Parallel causal computation equals separately truncated inference", fontweight="bold")
ax.grid(alpha=0.25); ax.legend()
fig.savefig(run_dir / "plots" / "causal_truncation_consistency.png", dpi=170)
display(fig); plt.close(fig)


#%% 31) Exact-posterior calibration diagnostic: 95% ellipse coverage
def mahalanobis_sq_np(value: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    delta = np.asarray(value, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return float(delta @ np.linalg.solve(np.asarray(cov, dtype=np.float64), delta))


def learned_gaussian_coverage(
    model: ModeAParallelGaussianModel,
    dataset: dict[str, np.ndarray],
    cfg: GaussianBayesConfig = CFG,
    *, seed: int,
) -> tuple[float, float]:
    """Compare exact and moment-matched learned 95% final-prefix ellipse coverage."""
    if cfg.theta_dim != 2:
        raise ValueError("Coverage helper currently uses chi2_2(0.95).")
    threshold = 5.991464547107979
    rng = np.random.default_rng(seed)
    exact_hits, learned_hits = [], []
    for start in range(0, len(dataset["theta_true"]), cfg.batch_size):
        stop = min(start+cfg.batch_size, len(dataset["theta_true"]))
        batch_np = make_batch_np(dataset, np.arange(start, stop), rng, cfg)
        posterior, _ = predict_batch(
            model,
            jnp.asarray(batch_np["prior_particles"]),
            jnp.asarray(batch_np["observations"]),
        )
        posterior = np.asarray(jax.device_get(posterior))[:, -1]
        for local_i, global_i in enumerate(range(start, stop)):
            theta = dataset["theta_true"][global_i]
            exact_mean = dataset["exact_mean"][global_i, -1]
            exact_cov = dataset["exact_cov"][global_i, -1]
            exact_hits.append(mahalanobis_sq_np(theta, exact_mean, exact_cov) <= threshold)
            cloud = posterior[local_i]
            learned_mean = cloud.mean(axis=0)
            centered = cloud - learned_mean
            learned_cov = centered.T @ centered / len(cloud) + 1e-6*np.eye(cfg.theta_dim)
            learned_hits.append(mahalanobis_sq_np(theta, learned_mean, learned_cov) <= threshold)
    return float(np.mean(exact_hits)), float(np.mean(learned_hits))


coverage_results = {}
for name, model in models.items():
    exact_cov_rate, learned_cov_rate = learned_gaussian_coverage(
        model, eval_data, CFG, seed=CFG.seed + 1_100_000
    )
    coverage_results[name] = {
        "exact_95pct_coverage": exact_cov_rate,
        "learned_moment_matched_95pct_coverage": learned_cov_rate,
    }
print("95% ellipse coverage:", coverage_results)
save_json(run_dir / "artefacts" / "coverage_results.json", coverage_results)

fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
names = list(coverage_results)
x = np.arange(len(names)); width = 0.34
ax.bar(x-width/2, [coverage_results[n]["exact_95pct_coverage"] for n in names],
       width=width, label="exact posterior")
ax.bar(x+width/2, [coverage_results[n]["learned_moment_matched_95pct_coverage"] for n in names],
       width=width, label="learned particles (moment-matched)")
ax.axhline(0.95, linestyle="--", linewidth=1.2, label="nominal 95%")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.02)
ax.set_ylabel("empirical coverage")
ax.set_title("Closed-form benchmark makes posterior calibration directly testable", fontweight="bold")
ax.grid(axis="y", alpha=0.25); ax.legend()
fig.savefig(run_dir / "plots" / "posterior_coverage.png", dpi=170)
display(fig); plt.close(fig)


#%% 32) Save limit-study arrays and final summary
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
    "model": "Bayesian linear regression with Gaussian prior and Gaussian likelihood",
    "posterior": "closed-form Gaussian at every prefix",
    "objective": "energy score only",
    "exact_posterior_used_in_training_loss": False,
    "mode": "Mode A: theta* fixed within trajectory, re-drawn across trajectories",
    "parallel_prefix_training": True,
    "theta_dim": CFG.theta_dim,
    "trajectory_length": CFG.trajectory_length,
    "num_particles": CFG.num_particles,
    "architectures": {},
    "coverage": coverage_results,
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

print("\nFinal conjugate-Gaussian Mode-A summary")
print(json.dumps(summary, indent=2))
print("All artefacts saved under:", run_dir)
