#%% 1) Imports and configuration
"""
AdaLN posterior-particle transformer for a source-localisation simulator.

This file is deliberately written as a sequence of notebook-friendly cells. Open it in
VS Code, Spyder, PyCharm, or JupyterLab with a percent-script extension and run each cell in order.

The prototype learns a conditional *set transport*:

    prior particles {theta_i} + condition c=(x, y)
        -> approximate posterior particles {theta'_i}

Training targets are generated with self-normalised importance sampling (SNIS) using
an explicitly coded source-localisation likelihood. The transport is deterministic;
the random prior particles provide its source of randomness.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import partial
from itertools import islice
from typing import Any, Iterator
import math
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm.auto import tqdm
from IPython.display import display
import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

print("JAX devices:", jax.devices())


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 2030

    # Source-localisation simulator
    K: int = 1                         # number of sources; visualisations assume K=1
    prior_std: float = 1.0
    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10             # keeps inverse-square signal finite
    observation_noise_std: float = 0.30 # noise on log intensity

    # Particle approximation
    num_particles: int = 64
    posterior_proposals: int = 2048     # SNIS proposal count; increase if ESS is poor

    # Data loading
    data_mode: str = "finite"           # "finite" or "infinite"
    n_train_episodes: int = 20_000
    n_eval_episodes: int = 512
    batch_size: int = 64
    num_workers: int = 0                # 0 is safest in notebooks
    steps_per_epoch: int = 400          # only used in infinite mode

    # AdaLN Transformer
    hidden_dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4

    # Optimisation
    epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Distribution-matching objective
    swd_projections: int = 48
    swd_weight: float = 1.0
    moment_weight: float = 0.10
    displacement_weight: float = 1e-5

    # Conditioning normalisation. The AdaLN signal is concat(x, y) after these
    # deterministic scale changes.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Notebook visualisation
    live_plots_per_epoch: int = 1
    final_plot_examples: int = 6
    grid_size: int = 180


CFG = Config()
assert CFG.hidden_dim % CFG.heads == 0
assert CFG.posterior_proposals >= CFG.num_particles
assert CFG.y_scale > 0
print(asdict(CFG))


#%% 2) Data loader, likelihood, and one fixed example
class SourceLocPrior:
    """theta_k ~ N(0, prior_std^2 I_2), independently for k=1,...,K."""

    def __init__(self, K: int = 1, prior_std: float = 1.0):
        self.K = int(K)
        self.prior_std = float(prior_std)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.prior_std, size=(self.K, 2)).astype(np.float32)

    def sample_n(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, self.prior_std, size=(n, self.K, 2)).astype(np.float32)


def source_log_signal_np(theta: np.ndarray, x: np.ndarray, cfg: Config = CFG) -> np.ndarray:
    """Mean log intensity for the location-finding likelihood.

    theta: (..., K, 2)
    x:     (..., 2), or simply (2,)
    returns: broadcast batch shape (...,)
    """
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dist_sq = np.sum((theta - np.expand_dims(x, axis=-2)) ** 2, axis=-1)
    intensity = cfg.background + np.sum(
        cfg.source_strength / (cfg.softening + dist_sq), axis=-1
    )
    return np.log(intensity)


def source_log_likelihood_np(
    y: np.ndarray | float,
    theta: np.ndarray,
    x: np.ndarray,
    cfg: Config = CFG,
) -> np.ndarray:
    """log p(y | theta, x), with y modelled as noisy log intensity."""
    mean = source_log_signal_np(theta, x, cfg)
    z = (np.asarray(y, dtype=np.float64) - mean) / cfg.observation_noise_std
    return -0.5 * z**2 - math.log(cfg.observation_noise_std * math.sqrt(2.0 * math.pi))


def make_condition_np(x: np.ndarray, y: np.ndarray | float, cfg: Config = CFG) -> np.ndarray:
    """Normalised concat(x, y) used by the AdaLN condition encoder."""
    x_scale = max(abs(cfg.design_low), abs(cfg.design_high), 1.0)
    x_norm = np.asarray(x, dtype=np.float32) / np.float32(x_scale)
    y_norm = (np.asarray(y, dtype=np.float32).reshape(1) - cfg.y_center) / cfg.y_scale
    return np.concatenate([x_norm.reshape(-1), y_norm], axis=0).astype(np.float32)


def systematic_resample_np(
    rng: np.random.Generator,
    weights: np.ndarray,
    n: int,
) -> np.ndarray:
    """Low-variance resampling from a one-dimensional categorical distribution."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="right")


class SourceLocationEpisodeGenerator:
    """Draw one complete conditional posterior-learning episode from a seed.

    The likelihood is intentionally embedded here. Each episode contains:
      - one true theta,
      - one random design x,
      - one simulated observation y,
      - a sequence of input prior particles,
      - an SNIS-resampled reference posterior sequence,
      - the SNIS effective sample size (ESS).
    """

    def __init__(self, prior: SourceLocPrior, cfg: Config = CFG):
        self.prior = prior
        self.cfg = cfg

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        cfg = self.cfg
        theta_true = self.prior.sample(rng)
        x = rng.uniform(cfg.design_low, cfg.design_high, size=(2,)).astype(np.float32)

        mean_y = float(source_log_signal_np(theta_true, x, cfg))
        y = np.float32(mean_y + cfg.observation_noise_std * rng.normal())

        # The model's source randomness.
        prior_particles = self.prior.sample_n(rng, cfg.num_particles)

        # Include the input particles in the larger importance-sampling pool, then add
        # independent prior proposals. This makes the target support faithful to the
        # prior without requiring any posterior MCMC inside the training loop.
        n_extra = cfg.posterior_proposals - cfg.num_particles
        if n_extra > 0:
            extra = self.prior.sample_n(rng, n_extra)
            proposals = np.concatenate([prior_particles, extra], axis=0)
        else:
            proposals = prior_particles.copy()

        log_w = source_log_likelihood_np(y, proposals, x, cfg)
        log_w = log_w - np.max(log_w)
        weights = np.exp(log_w)
        weights = weights / np.sum(weights)
        ess = np.float32(1.0 / np.sum(weights**2))

        indices = systematic_resample_np(rng, weights, cfg.num_particles)
        posterior_particles = proposals[indices].astype(np.float32)

        return {
            "theta_true": theta_true.astype(np.float32),
            "x": x.astype(np.float32),
            "y": np.asarray([y], dtype=np.float32),
            "condition": make_condition_np(x, y, cfg),
            "prior_particles": prior_particles.astype(np.float32),
            "posterior_particles": posterior_particles,
            "ess": np.asarray([ess], dtype=np.float32),
        }


class FiniteEpisodes(Dataset):
    """Reproducible map-style episode set, following the supplied loader design."""

    def __init__(
        self,
        generator: SourceLocationEpisodeGenerator,
        n_episodes: int,
        base_seed: int = 0,
    ):
        self.generator = generator
        self.n_episodes = int(n_episodes)
        self.seeds = (np.arange(self.n_episodes, dtype=np.int64) + base_seed).tolist()

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seeds[idx])
        return self.generator.sample(rng)


class TrainEpisodes(IterableDataset):
    """Infinite stream of fresh, worker-separated simulated episodes."""

    def __init__(self, generator: SourceLocationEpisodeGenerator, base_seed: int = 0):
        self.generator = generator
        self.base_seed = int(base_seed)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.base_seed + 1_000_003 * worker_id)
        while True:
            yield self.generator.sample(rng)


class EvalEpisodes(Dataset):
    """Materialised, seeded evaluation episodes; identical across evaluations."""

    def __init__(
        self,
        generator: SourceLocationEpisodeGenerator,
        n_episodes: int = 512,
        seed: int = 12345,
    ):
        seed_rng = np.random.default_rng(seed)
        episode_seeds = seed_rng.integers(0, np.iinfo(np.int64).max, size=n_episodes, dtype=np.int64)
        self.episodes = [
            generator.sample(np.random.default_rng(int(s))) for s in episode_seeds
        ]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.episodes[idx]


def collate_episodes(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([item[key] for item in batch], axis=0) for key in batch[0]}


def make_train_loader(
    generator: SourceLocationEpisodeGenerator,
    cfg: Config = CFG,
) -> DataLoader:
    if cfg.data_mode == "finite":
        dataset = FiniteEpisodes(
            generator,
            n_episodes=cfg.n_train_episodes,
            base_seed=cfg.seed,
        )
        torch_generator = torch.Generator()
        torch_generator.manual_seed(cfg.seed)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_episodes,
            num_workers=cfg.num_workers,
            generator=torch_generator,
            drop_last=True,
        )
    if cfg.data_mode == "infinite":
        dataset = TrainEpisodes(generator, base_seed=cfg.seed)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_episodes,
            num_workers=cfg.num_workers,
        )
    raise ValueError("data_mode must be 'finite' or 'infinite'.")


def make_eval_loader(
    generator: SourceLocationEpisodeGenerator,
    cfg: Config = CFG,
) -> DataLoader:
    dataset = EvalEpisodes(
        generator,
        n_episodes=cfg.n_eval_episodes,
        seed=cfg.seed + 20_000,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
        num_workers=0,
        drop_last=False,
    )


prior = SourceLocPrior(K=CFG.K, prior_std=CFG.prior_std)
episode_generator = SourceLocationEpisodeGenerator(prior, CFG)
train_loader = make_train_loader(episode_generator, CFG)
eval_loader = make_eval_loader(episode_generator, CFG)
fixed_example = eval_loader.dataset[0]

print("Train batches per finite epoch:" if CFG.data_mode == "finite" else "Configured steps per epoch:",
      len(train_loader) if CFG.data_mode == "finite" else CFG.steps_per_epoch)
print("Fixed-example SNIS ESS:", float(fixed_example["ess"][0]), "/", CFG.posterior_proposals)


def posterior_density_grid_np(
    episode: dict[str, np.ndarray],
    cfg: Config = CFG,
    grid_size: int | None = None,
    lim: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically normalised posterior grid for K=1 visual diagnostics only."""
    if cfg.K != 1:
        raise ValueError("The 2-D posterior grid visualisation is defined only for K=1.")
    grid_size = cfg.grid_size if grid_size is None else grid_size
    if lim is None:
        points = np.concatenate(
            [
                episode["prior_particles"].reshape(-1, 2),
                episode["posterior_particles"].reshape(-1, 2),
                episode["theta_true"].reshape(-1, 2),
                episode["x"].reshape(-1, 2),
            ],
            axis=0,
        )
        lim = max(3.0 * cfg.prior_std, float(np.quantile(np.abs(points), 0.995)) * 1.15)
    gx = np.linspace(-lim, lim, grid_size)
    gy = np.linspace(-lim, lim, grid_size)
    xx, yy = np.meshgrid(gx, gy)
    theta_grid = np.stack([xx, yy], axis=-1)[:, :, None, :]

    log_prior = -0.5 * np.sum((theta_grid[:, :, 0, :] / cfg.prior_std) ** 2, axis=-1)
    log_like = source_log_likelihood_np(
        float(episode["y"][0]), theta_grid, episode["x"], cfg
    )
    log_post = log_prior + log_like
    density = np.exp(log_post - np.max(log_post))
    density = density / np.maximum(np.sum(density), 1e-12)
    return xx, yy, density


def common_episode_limit(
    episode: dict[str, np.ndarray],
    predicted: np.ndarray | None = None,
    cfg: Config = CFG,
) -> float:
    arrays = [
        episode["prior_particles"].reshape(-1, 2),
        episode["posterior_particles"].reshape(-1, 2),
        episode["theta_true"].reshape(-1, 2),
        episode["x"].reshape(-1, 2),
    ]
    if predicted is not None:
        arrays.append(np.asarray(predicted).reshape(-1, 2))
    points = np.concatenate(arrays, axis=0)
    q = float(np.quantile(np.abs(points), 0.995))
    return max(3.0 * cfg.prior_std, 1.15 * q)


def plot_raw_episode(
    episode: dict[str, np.ndarray],
    cfg: Config = CFG,
) -> Figure:
    if cfg.K != 1:
        raise ValueError("This diagnostic plot currently assumes K=1.")
    lim = common_episode_limit(episode, cfg=cfg)
    xx, yy, density = posterior_density_grid_np(episode, cfg, lim=lim)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    true_theta = episode["theta_true"][0]
    x = episode["x"]

    axes[0].contourf(xx, yy, density / density.max(), levels=24)
    axes[0].scatter(*true_theta, marker="*", s=150, label="true source")
    axes[0].scatter(*x, marker="X", s=90, label="design x")
    axes[0].set_title(f"Reference posterior density\ny={episode['y'][0]:.3f}")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].scatter(
        episode["prior_particles"][:, 0, 0],
        episode["prior_particles"][:, 0, 1],
        s=14,
        alpha=0.65,
    )
    axes[1].scatter(*true_theta, marker="*", s=140)
    axes[1].set_title("Input prior-particle sequence")

    axes[2].scatter(
        episode["posterior_particles"][:, 0, 0],
        episode["posterior_particles"][:, 0, 1],
        s=14,
        alpha=0.65,
    )
    axes[2].scatter(*true_theta, marker="*", s=140)
    axes[2].set_title("SNIS posterior target sequence")

    log_like = source_log_likelihood_np(
        float(episode["y"][0]),
        np.stack([xx, yy], axis=-1)[:, :, None, :],
        episode["x"],
        cfg,
    )
    likelihood = np.exp(log_like - np.max(log_like))
    axes[3].contourf(xx, yy, likelihood, levels=24)
    axes[3].scatter(*true_theta, marker="*", s=150)
    axes[3].scatter(*x, marker="X", s=90)
    axes[3].set_title(f"Likelihood over source location\nESS={episode['ess'][0]:.1f}")

    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.grid(alpha=0.15)
    return fig


fig = plot_raw_episode(fixed_example, CFG)
display(fig)
plt.close(fig)


#%% 3) AdaLN-zero posterior-particle Transformer
Array = jax.Array


def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    """Apply an Equinox Linear independently to every token."""
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class AdaLNZeroBlock(eqx.Module):
    """Permutation-equivariant full-self-attention block with AdaLN-zero."""

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        mlp_dim: int,
        *,
        key: Array,
    ):
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
            lambda m: m.weight, modulation, jnp.zeros_like(modulation.weight)
        )
        modulation = eqx.tree_at(
            lambda m: m.bias, modulation, jnp.zeros_like(modulation.bias)
        )
        self.modulation = modulation

    def __call__(self, x: Array, condition: Array) -> Array:
        modulation = self.modulation(jax.nn.silu(condition))
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = jnp.split(
            modulation, 6, axis=-1
        )

        h = _layernorm_tokens(self.norm1, x)
        h = modulate(h, shift_attn, scale_attn)
        # No positional embedding and no causal mask: the particles are an exchangeable set.
        h = self.attention(h, h, h)
        x = x + gate_attn[None, :] * h

        h = _layernorm_tokens(self.norm2, x)
        h = modulate(h, shift_ff, scale_ff)
        h = _linear_tokens(self.ff_in, h)
        h = jax.nn.gelu(h)
        h = _linear_tokens(self.ff_out, h)
        x = x + gate_ff[None, :] * h
        return x


class PosteriorParticleTransformer(eqx.Module):
    particle_in: eqx.nn.Linear
    condition_encoder: eqx.nn.MLP
    blocks: tuple[AdaLNZeroBlock, ...]
    final_norm: eqx.nn.LayerNorm
    output_delta: eqx.nn.Linear

    theta_dim: int
    K: int
    prior_std: float

    def __init__(self, cfg: Config = CFG, *, key: Array):
        self.theta_dim = 2 * cfg.K
        self.K = cfg.K
        self.prior_std = cfg.prior_std

        keys = jax.random.split(key, cfg.depth + 3)
        self.particle_in = eqx.nn.Linear(self.theta_dim, cfg.hidden_dim, key=keys[0])
        self.condition_encoder = eqx.nn.MLP(
            in_size=3,
            out_size=cfg.hidden_dim,
            width_size=cfg.hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            final_activation=lambda z: z,
            key=keys[1],
        )
        self.blocks = tuple(
            AdaLNZeroBlock(
                cfg.hidden_dim,
                cfg.heads,
                cfg.mlp_ratio * cfg.hidden_dim,
                key=keys[2 + i],
            )
            for i in range(cfg.depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)

        output_delta = eqx.nn.Linear(
            cfg.hidden_dim, self.theta_dim, key=keys[-1]
        )
        # Near-identity initial transport while retaining a gradient path.
        output_delta = eqx.tree_at(
            lambda m: m.weight, output_delta, 1e-3 * output_delta.weight
        )
        output_delta = eqx.tree_at(
            lambda m: m.bias, output_delta, jnp.zeros_like(output_delta.bias)
        )
        self.output_delta = output_delta

    def __call__(self, prior_particles: Array, condition: Array) -> Array:
        """prior_particles: (P,K,2), condition: normalised concat(x,y), shape (3,)."""
        flat = prior_particles.reshape(prior_particles.shape[0], self.theta_dim)
        h = _linear_tokens(self.particle_in, flat / self.prior_std)
        c = self.condition_encoder(condition)
        for block in self.blocks:
            h = block(h, c)
        h = _layernorm_tokens(self.final_norm, h)
        delta = _linear_tokens(self.output_delta, h)
        transported = flat + self.prior_std * delta
        return transported.reshape(prior_particles.shape[0], self.K, 2)


model_key = jax.random.PRNGKey(CFG.seed)
model = PosteriorParticleTransformer(CFG, key=model_key)
# print(model)

# A shape smoke test before defining optimisation.
smoke_prior = jnp.asarray(fixed_example["prior_particles"])
smoke_condition = jnp.asarray(fixed_example["condition"])
smoke_output = model(smoke_prior, smoke_condition)
print("Smoke-test output shape:", smoke_output.shape)


#%% 4) Train/evaluation utilities
Batch = dict[str, Array]


def to_jax_batch(batch_np: dict[str, np.ndarray]) -> Batch:
    """One host-to-device conversion per NumPy field."""
    return {name: jnp.asarray(value) for name, value in batch_np.items()}


def predict_batch(model: PosteriorParticleTransformer, batch: Batch) -> Array:
    return jax.vmap(lambda p, c: model(p, c))(
        batch["prior_particles"], batch["condition"]
    )


def random_unit_projections(key: Array, n_proj: int, dim: int) -> Array:
    proj = jax.random.normal(key, shape=(n_proj, dim))
    return proj / jnp.maximum(jnp.linalg.norm(proj, axis=-1, keepdims=True), 1e-8)


def sliced_wasserstein_single(pred: Array, target: Array, projections: Array) -> Array:
    """Squared sliced-Wasserstein estimate for equal-size empirical particle sets."""
    pred_projected = pred @ projections.T
    target_projected = target @ projections.T
    pred_sorted = jnp.sort(pred_projected, axis=0)
    target_sorted = jnp.sort(target_projected, axis=0)
    return jnp.mean((pred_sorted - target_sorted) ** 2)


def moments_single(samples: Array) -> tuple[Array, Array]:
    mean = jnp.mean(samples, axis=0)
    centered = samples - mean
    denom = jnp.maximum(samples.shape[0] - 1, 1)
    cov = (centered.T @ centered) / denom
    return mean, cov


def loss_fn(
    model: PosteriorParticleTransformer,
    batch: Batch,
    key: Array,
) -> tuple[Array, dict[str, Array]]:
    pred = predict_batch(model, batch)
    batch_size = pred.shape[0]
    flat_dim = 2 * CFG.K
    pred_flat = pred.reshape(batch_size, CFG.num_particles, flat_dim)
    target_flat = batch["posterior_particles"].reshape(
        batch_size, CFG.num_particles, flat_dim
    )
    prior_flat = batch["prior_particles"].reshape(
        batch_size, CFG.num_particles, flat_dim
    )

    projections = random_unit_projections(key, CFG.swd_projections, flat_dim)
    swd_per_example = jax.vmap(
        lambda p, t: sliced_wasserstein_single(p, t, projections)
    )(pred_flat, target_flat)
    swd = jnp.mean(swd_per_example)

    pred_mean, pred_cov = jax.vmap(moments_single)(pred_flat)
    target_mean, target_cov = jax.vmap(moments_single)(target_flat)
    mean_loss = jnp.mean((pred_mean - target_mean) ** 2)
    cov_loss = jnp.mean((pred_cov - target_cov) ** 2)
    moment_loss = mean_loss + cov_loss

    displacement = jnp.mean((pred_flat - prior_flat) ** 2)
    total = (
        CFG.swd_weight * swd
        + CFG.moment_weight * moment_loss
        + CFG.displacement_weight * displacement
    )

    mean_rmse = jnp.sqrt(jnp.mean((pred_mean - target_mean) ** 2))
    metrics = {
        "loss": total,
        "swd": swd,
        "moment": moment_loss,
        "mean_rmse": mean_rmse,
        "displacement": displacement,
        "ess": jnp.mean(batch["ess"]),
    }
    return total, metrics


optimizer = optax.chain(
    optax.clip_by_global_norm(CFG.grad_clip_norm),
    optax.adamw(
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    ),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(
    model: PosteriorParticleTransformer,
    opt_state: optax.OptState,
    batch: Batch,
    key: Array,
) -> tuple[PosteriorParticleTransformer, optax.OptState, dict[str, Array], Array]:
    (_, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, batch, key
    )
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
    return model, opt_state, metrics, grad_norm


@eqx.filter_jit
def eval_step(
    model: PosteriorParticleTransformer,
    batch: Batch,
    key: Array,
) -> dict[str, Array]:
    _, metrics = loss_fn(model, batch, key)
    return metrics


@eqx.filter_jit
def predict_single_jit(
    model: PosteriorParticleTransformer,
    prior_particles: Array,
    condition: Array,
) -> Array:
    return model(prior_particles, condition)


def evaluate_model(
    model: PosteriorParticleTransformer,
    loader: DataLoader,
    base_key: Array,
) -> dict[str, float]:
    collected: dict[str, list[float]] = {}
    for batch_index, batch_np in enumerate(loader):
        batch = to_jax_batch(batch_np)
        batch_key = jax.random.fold_in(base_key, batch_index)
        metrics = eval_step(model, batch, batch_key)
        host_metrics = jax.device_get(metrics)
        for name, value in host_metrics.items():
            collected.setdefault(name, []).append(float(value))
    return {name: float(np.mean(values)) for name, values in collected.items()}


def predict_episode_np(
    model: PosteriorParticleTransformer,
    episode: dict[str, np.ndarray],
) -> np.ndarray:
    pred = predict_single_jit(
        model,
        jnp.asarray(episode["prior_particles"]),
        jnp.asarray(episode["condition"]),
    )
    return np.asarray(jax.device_get(pred))


def plot_model_on_example(
    model: PosteriorParticleTransformer,
    episode: dict[str, np.ndarray],
    *,
    title: str,
    cfg: Config = CFG,
) -> Figure:
    if cfg.K != 1:
        raise ValueError("This live diagnostic currently assumes K=1.")
    pred = predict_episode_np(model, episode)
    lim = common_episode_limit(episode, predicted=pred, cfg=cfg)
    xx, yy, density = posterior_density_grid_np(episode, cfg, lim=lim)
    true_theta = episode["theta_true"][0]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    axes = axes.ravel()

    axes[0].contourf(xx, yy, density / density.max(), levels=24)
    axes[0].scatter(*true_theta, marker="*", s=150)
    axes[0].scatter(*episode["x"], marker="X", s=90)
    axes[0].set_title("Reference density")

    axes[1].scatter(
        episode["prior_particles"][:, 0, 0],
        episode["prior_particles"][:, 0, 1],
        s=15,
        alpha=0.65,
    )
    axes[1].scatter(*true_theta, marker="*", s=140)
    axes[1].set_title("Prior particles")

    axes[2].scatter(
        episode["posterior_particles"][:, 0, 0],
        episode["posterior_particles"][:, 0, 1],
        s=15,
        alpha=0.65,
    )
    axes[2].scatter(*true_theta, marker="*", s=140)
    axes[2].set_title(f"SNIS target (ESS={episode['ess'][0]:.1f})")

    axes[3].scatter(pred[:, 0, 0], pred[:, 0, 1], s=15, alpha=0.65)
    axes[3].scatter(*true_theta, marker="*", s=140)
    axes[3].set_title("Transformer posterior particles")

    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
    fig.suptitle(title)
    return fig


# Evaluate the near-identity initial model.
initial_eval_key = jax.random.PRNGKey(CFG.seed + 100_000)
initial_metrics = evaluate_model(model, eval_loader, initial_eval_key)
print("Initial validation metrics:", initial_metrics)


#%% 5) Training loop: every-step losses, epoch evaluation, and 10 live grids per epoch
history: dict[str, list[float]] = {
    "step_loss": [],
    "step_swd": [],
    "step_moment": [],
    "step_mean_rmse": [],
    "step_grad_norm": [],
    "step_ess": [],
    "epoch_train_loss": [],
    "epoch_val_loss": [],
    "epoch_val_swd": [],
    "epoch_val_mean_rmse": [],
}

train_rng = jax.random.PRNGKey(CFG.seed + 1)
global_step = 0
steps_per_epoch = len(train_loader) if CFG.data_mode == "finite" else CFG.steps_per_epoch
live_every = max(1, steps_per_epoch // max(CFG.live_plots_per_epoch, 1))

train_start_time = time.time()
for epoch in range(1, CFG.epochs + 1):
    epoch_start = time.time()
    epoch_losses: list[float] = []

    if CFG.data_mode == "finite":
        epoch_iterator = iter(train_loader)
    else:
        epoch_iterator = islice(iter(train_loader), CFG.steps_per_epoch)

    progress = tqdm(
        enumerate(epoch_iterator, start=1),
        total=steps_per_epoch,
        desc=f"Epoch {epoch:02d}/{CFG.epochs:02d}",
        leave=True,
    )

    for step_in_epoch, batch_np in progress:
        batch = to_jax_batch(batch_np)
        train_rng, step_key = jax.random.split(train_rng)
        model, opt_state, metrics, grad_norm = train_step(
            model, opt_state, batch, step_key
        )

        host_metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        host_grad_norm = float(jax.device_get(grad_norm))
        epoch_losses.append(host_metrics["loss"])

        history["step_loss"].append(host_metrics["loss"])
        history["step_swd"].append(host_metrics["swd"])
        history["step_moment"].append(host_metrics["moment"])
        history["step_mean_rmse"].append(host_metrics["mean_rmse"])
        history["step_grad_norm"].append(host_grad_norm)
        history["step_ess"].append(host_metrics["ess"])
        global_step += 1

        progress.set_postfix(
            loss=f"{host_metrics['loss']:.4f}",
            swd=f"{host_metrics['swd']:.4f}",
            ess=f"{host_metrics['ess']:.0f}",
        )

        # Exactly the requested repeated view of the same fixed evaluation episode.
        if (
            CFG.live_plots_per_epoch > 0
            and (step_in_epoch % live_every == 0 or step_in_epoch == steps_per_epoch)
        ):
            fraction = step_in_epoch / steps_per_epoch
            fig = plot_model_on_example(
                model,
                fixed_example,
                title=(
                    f"Fixed episode — epoch {epoch}/{CFG.epochs}, "
                    f"{fraction:.0%} through epoch, global step {global_step}"
                ),
                cfg=CFG,
            )
            display(fig)
            plt.close(fig)

    train_epoch_loss = float(np.mean(epoch_losses))
    eval_key = jax.random.PRNGKey(CFG.seed + 100_000)  # fixed projections across epochs
    val_metrics = evaluate_model(model, eval_loader, eval_key)

    history["epoch_train_loss"].append(train_epoch_loss)
    history["epoch_val_loss"].append(val_metrics["loss"])
    history["epoch_val_swd"].append(val_metrics["swd"])
    history["epoch_val_mean_rmse"].append(val_metrics["mean_rmse"])

    elapsed = time.time() - epoch_start
    print(
        f"Epoch {epoch:02d}: train={train_epoch_loss:.6f} | "
        f"val={val_metrics['loss']:.6f} | val SWD={val_metrics['swd']:.6f} | "
        f"val mean-RMSE={val_metrics['mean_rmse']:.6f} | {elapsed:.1f}s"
    )

print(f"Total training time in hh:mm:ss: {str(datetime.timedelta(seconds=int(time.time() - train_start_time)))}")


#%% 6) Plot step losses and epoch train/validation losses
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

axes[0, 0].plot(history["step_loss"], alpha=0.65)
axes[0, 0].set_yscale("log")
axes[0, 0].set_title("Total training loss per optimiser step")
axes[0, 0].set_xlabel("Global train step")
axes[0, 0].grid(alpha=0.25)

axes[0, 1].plot(history["step_swd"], label="sliced Wasserstein", alpha=0.7)
axes[0, 1].plot(history["step_moment"], label="moment loss", alpha=0.7)
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Distribution-loss components per step")
axes[0, 1].set_xlabel("Global train step")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.25)

epoch_axis = np.arange(1, len(history["epoch_train_loss"]) + 1)
axes[1, 0].plot(epoch_axis, history["epoch_train_loss"], marker="o", label="train")
axes[1, 0].plot(epoch_axis, history["epoch_val_loss"], marker="s", label="validation")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("Epoch losses")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.25)

axes[1, 1].plot(history["step_grad_norm"], alpha=0.65, label="gradient norm")
axes[1, 1].plot(history["step_mean_rmse"], alpha=0.65, label="posterior-mean RMSE")
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("Optimisation and posterior-mean diagnostics")
axes[1, 1].set_xlabel("Global train step")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.25)

display(fig)
plt.close(fig)


#%% 7) Giant multi-axis comparison over multiple location-finding episodes
def plot_many_location_finding_episodes(
    model: PosteriorParticleTransformer,
    dataset: Dataset,
    n_examples: int = 6,
    cfg: Config = CFG,
) -> Figure:
    if cfg.K != 1:
        raise ValueError("The final 2-D location-finding plot assumes K=1.")

    episodes = [dataset[i] for i in range(min(n_examples, len(dataset)))]
    predictions = [predict_episode_np(model, episode) for episode in episodes]

    # One shared normalisation/axis range makes rows visually comparable. Robust
    # quantiles prevent a single runaway particle from making every panel unreadable.
    all_points = []
    for episode, pred in zip(episodes, predictions):
        all_points.extend(
            [
                episode["prior_particles"].reshape(-1, 2),
                episode["posterior_particles"].reshape(-1, 2),
                pred.reshape(-1, 2),
                episode["theta_true"].reshape(-1, 2),
                episode["x"].reshape(-1, 2),
            ]
        )
    all_points_np = np.concatenate(all_points, axis=0)
    shared_lim = max(
        3.0 * cfg.prior_std,
        1.15 * float(np.quantile(np.abs(all_points_np), 0.995)),
    )

    fig, axes = plt.subplots(
        len(episodes),
        4,
        figsize=(17, 4.0 * len(episodes)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (episode, pred) in enumerate(zip(episodes, predictions)):
        xx, yy, density = posterior_density_grid_np(
            episode, cfg, grid_size=cfg.grid_size, lim=shared_lim
        )
        true_theta = episode["theta_true"][0]
        design = episode["x"]

        axes[row, 0].contourf(xx, yy, density / density.max(), levels=24)
        axes[row, 0].scatter(*true_theta, marker="*", s=135)
        axes[row, 0].scatter(*design, marker="X", s=80)
        axes[row, 0].set_title(
            f"Episode {row + 1}: posterior density\n"
            f"y={episode['y'][0]:.2f}, ESS={episode['ess'][0]:.0f}"
        )

        axes[row, 1].scatter(
            episode["prior_particles"][:, 0, 0],
            episode["prior_particles"][:, 0, 1],
            s=13,
            alpha=0.62,
        )
        axes[row, 1].scatter(*true_theta, marker="*", s=130)
        axes[row, 1].set_title("Prior sequence")

        axes[row, 2].scatter(
            episode["posterior_particles"][:, 0, 0],
            episode["posterior_particles"][:, 0, 1],
            s=13,
            alpha=0.62,
        )
        axes[row, 2].scatter(*true_theta, marker="*", s=130)
        axes[row, 2].set_title("SNIS posterior target")

        axes[row, 3].scatter(pred[:, 0, 0], pred[:, 0, 1], s=13, alpha=0.62)
        axes[row, 3].scatter(*true_theta, marker="*", s=130)
        axes[row, 3].set_title("AdaLN transport output")

        for col in range(4):
            ax = axes[row, col]
            ax.set_xlim(-shared_lim, shared_lim)
            ax.set_ylim(-shared_lim, shared_lim)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$\theta_1$")
            ax.set_ylabel(r"$\theta_2$")
            ax.grid(alpha=0.14)

    fig.suptitle(
        "Location-finding posterior transport across fixed evaluation episodes",
        fontsize=16,
    )
    return fig


final_fig = plot_many_location_finding_episodes(
    model,
    eval_loader.dataset,
    n_examples=CFG.final_plot_examples,
    cfg=CFG,
)
display(final_fig)
plt.close(final_fig)
