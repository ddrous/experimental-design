"""Shared JAX/Equinox components for the posterior and design-policy experiments.

This module intentionally contains *mechanics* rather than notebook plotting code.
The two notebook-style scripts keep their figures directly inside ``#%%`` cells so
that a researcher can edit the plots without jumping through helper functions.

Mathematical objects
--------------------
For K unknown two-dimensional source locations,

    theta = (theta_1, ..., theta_K),  theta_k in R^2,

we use the location-finding likelihood

    s(theta, x) = b + sum_k alpha / (m + ||theta_k - x||^2),
    y | theta, x ~ Normal(log s(theta, x), sigma_y^2).

The posterior network is a simulation-based conditional density estimator

    q_phi(theta | B, x, y),

where B = {theta^(n)}_{n=1}^N is an unweighted particle representation of the
*current* belief. Its Transformer backbone processes B as an exchangeable set;
AdaLN-zero injects the new observation condition c = concat(x, y). The network
outputs a diagonal Gaussian mixture, not transformed particles.

For indistinguishable sources, source labels are arbitrary. The training objective
therefore marginalises over source permutations:

    q_sym(theta | ...) = (1 / |P|) sum_{pi in P} q_phi(pi theta | ...),
    L_NLL = -log q_sym(theta_true | ...).

When all K! permutations are enumerated this is exact. For large K, the scripts use
a fixed Monte-Carlo subset of permutations and clearly report that approximation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Sequence
import json
import math
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import yaml

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import equinox as eqx

Array = jax.Array


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PosteriorConfig:
    """Configuration for the amortised Bayesian update model."""

    env_name: str = "posterior_adaln"
    seed: int = 7

    # Source-localisation simulator.
    K: int = 1
    prior_std: float = 1.0
    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.30

    # An input belief is represented by N unweighted particles. During data
    # generation, SNIS constructs beliefs after a random synthetic history.
    num_particles: int = 64
    belief_proposals: int = 4096
    max_history_steps: int = 7

    # PyTorch is used only as the requested batching/shuffling front end. Every
    # numerical model operation remains JAX + Equinox + Optax.
    data_mode: str = "finite"
    n_train_episodes: int = 20_000
    n_eval_episodes: int = 512
    batch_size: int = 64
    num_workers: int = 0
    steps_per_epoch: int = 400

    # AdaLN set Transformer and diagonal Gaussian-mixture output.
    hidden_dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    num_mixture_components: int = 16
    min_scale: float = 1e-3
    max_scale: float = 8.0
    canonicalize_particle_sources: bool = True

    # Source-label symmetry. Enumerating K! is practical only for small K.
    exact_permutation_max_k: int = 6
    sampled_permutations: int = 256

    # Optimisation.
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Conditioning normalisation.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Diagnostics and persistence.
    final_plot_examples: int = 6
    grid_size: int = 180
    runs_base: str = "./runs"
    save_every_epochs: int = 1


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for the downstream differentiable design policy."""

    env_name: str = "design_policy_adaln"
    seed: int = 17

    # Set this explicitly to a posterior run directory, or leave None to select
    # the newest matching run under runs_base.
    inference_run_dir: str | None = None
    inference_env_name: str = "posterior_adaln"
    runs_base: str = "./runs"
    inference_checkpoint_name: str = "model_best.eqx"

    # Sequential experiment budget.
    horizon: int = 6
    num_belief_particles: int = 64
    relaxed_mixture_temperature: float = 0.35
    design_exploration_std: float = 0.08

    # Policy architecture.
    hidden_dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4

    # Training data and optimisation.
    data_mode: str = "finite"
    n_train_episodes: int = 20_000
    n_eval_episodes: int = 256
    batch_size: int = 64
    num_workers: int = 0
    steps_per_epoch: int = 400
    epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # Objective. NLL is the principled term. The optional oracle coverage term
    # is simulation-only shaping: it encourages the designs to visit the true
    # sources during training but is never needed at deployment.
    intermediate_nll_weight: float = 0.25
    terminal_nll_weight: float = 1.0
    oracle_coverage_weight: float = 0.05
    design_smoothness_weight: float = 1e-3

    # Diagnostics/persistence.
    final_plot_examples: int = 6
    runs_base_policy: str = "./runs"
    save_every_epochs: int = 1


def dataclass_from_dict(cls, values: dict[str, Any]):
    """Construct a dataclass while ignoring unrelated YAML metadata fields."""
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in allowed})


# -----------------------------------------------------------------------------
# Run directories, snapshots, and checkpoint metadata
# -----------------------------------------------------------------------------
def make_run_dir(env_name: str, base: str | Path = "./runs") -> Path:
    """Create ``runs/<name>_<timestamp>/{plots,artefacts}``.

    This follows the experiment layout supplied by the user. The timestamp keeps
    runs immutable and makes selecting the newest trained model straightforward.
    """
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = Path(base).expanduser().resolve() / f"{env_name}_{stamp}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=False)
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_yaml(config: Any, path: str | Path, extra: dict[str, Any] | None = None):
    payload = asdict(config)
    if extra:
        payload.update(extra)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def snapshot_files(run_dir: str | Path, paths: Sequence[str | Path]):
    """Copy available source files into a run-level ``src`` directory."""
    destination = Path(run_dir) / "src"
    destination.mkdir(parents=True, exist_ok=True)
    for source_like in paths:
        source = Path(source_like).expanduser()
        if source.is_file():
            shutil.copy2(source, destination / source.name)


def save_json(path: str | Path, payload: dict[str, Any]):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def find_latest_run(runs_base: str | Path, env_name: str) -> Path:
    candidates = sorted(
        (p for p in Path(runs_base).expanduser().resolve().glob(f"{env_name}_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run matching {env_name!r} found under {Path(runs_base).resolve()}."
        )
    return candidates[-1]


# -----------------------------------------------------------------------------
# Source prior and likelihood: NumPy for data-loader workers, JAX for training
# -----------------------------------------------------------------------------
class SourceLocPrior:
    """theta_k ~ Normal(0, prior_std^2 I_2), independently for k=1,...,K."""

    def __init__(self, K: int = 1, prior_std: float = 1.0):
        self.K = int(K)
        self.prior_std = float(prior_std)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.prior_std, size=(self.K, 2)).astype(np.float32)

    def sample_n(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, self.prior_std, size=(n, self.K, 2)).astype(np.float32)


def source_log_signal_np(theta: np.ndarray, x: np.ndarray, cfg: PosteriorConfig) -> np.ndarray:
    """Return log mean intensity, broadcasting over leading theta dimensions."""
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
    cfg: PosteriorConfig,
) -> np.ndarray:
    mean = source_log_signal_np(theta, x, cfg)
    z = (np.asarray(y, dtype=np.float64) - mean) / cfg.observation_noise_std
    normalizer = math.log(cfg.observation_noise_std * math.sqrt(2.0 * math.pi))
    return -0.5 * z**2 - normalizer


def source_log_signal_jax(theta: Array, x: Array, cfg: PosteriorConfig) -> Array:
    """JAX likelihood mean for theta (..., K, 2) and x (..., 2).

    The policy rollout uses theta with shape (B,K,2) and x with shape (B,2), so
    inserting the source axis at -2 gives exactly the required broadcasting.
    """
    dist_sq = jnp.sum((theta - jnp.expand_dims(x, axis=-2)) ** 2, axis=-1)
    intensity = cfg.background + jnp.sum(
        cfg.source_strength / (cfg.softening + dist_sq), axis=-1
    )
    return jnp.log(intensity)


def make_condition_np(x: np.ndarray, y: np.ndarray | float, cfg: PosteriorConfig) -> np.ndarray:
    x_scale = max(abs(cfg.design_low), abs(cfg.design_high), 1.0)
    x_norm = np.asarray(x, dtype=np.float32).reshape(2) / np.float32(x_scale)
    y_norm = (np.asarray(y, dtype=np.float32).reshape(1) - cfg.y_center) / cfg.y_scale
    return np.concatenate([x_norm, y_norm], axis=0).astype(np.float32)


def make_condition_jax(x: Array, y: Array, cfg: PosteriorConfig) -> Array:
    """Batch-safe c = concat(x / x_scale, (y-y_center)/y_scale)."""
    x_scale = max(abs(cfg.design_low), abs(cfg.design_high), 1.0)
    y = jnp.asarray(y)
    if y.ndim == x.ndim - 1:
        y = y[..., None]
    return jnp.concatenate(
        [x / x_scale, (y - cfg.y_center) / cfg.y_scale], axis=-1
    )


def systematic_resample_np(rng: np.random.Generator, weights: np.ndarray, n: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-300)
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="right")


def canonicalize_sources_np(theta: np.ndarray) -> np.ndarray:
    """Sort sources by x coordinate inside every sample; ties are measure-zero here."""
    theta = np.asarray(theta)
    order = np.argsort(theta[..., 0], axis=-1)
    return np.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_sources_jax(theta: Array) -> Array:
    order = jnp.argsort(theta[..., 0], axis=-1)
    return jnp.take_along_axis(theta, order[..., None], axis=-2)


# -----------------------------------------------------------------------------
# Likelihood-aware one-step update episodes
# -----------------------------------------------------------------------------
class SourceLocationUpdateGenerator:
    """Generate examples for a reusable one-step Bayesian update operator.

    Instead of always feeding the original Gaussian prior, each episode first draws
    a random history length H. SNIS constructs an approximate current belief

        B_H ~= p(theta | x_1:H, y_1:H).

    A fresh design/observation pair (x_{H+1}, y_{H+1}) is then generated. The model
    receives (B_H, x_{H+1}, y_{H+1}) and is trained by conditional NLL on the same
    theta_true used to simulate every observation. This is essential for recursively
    reusing the updater inside the downstream policy rollout.
    """

    def __init__(self, prior: SourceLocPrior, cfg: PosteriorConfig):
        self.prior = prior
        self.cfg = cfg

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        cfg = self.cfg
        theta_true = self.prior.sample(rng)
        history_length = int(rng.integers(0, cfg.max_history_steps + 1))

        history_x = np.zeros((cfg.max_history_steps, 2), dtype=np.float32)
        history_y = np.zeros((cfg.max_history_steps, 1), dtype=np.float32)

        if history_length > 0:
            x_hist = rng.uniform(
                cfg.design_low, cfg.design_high, size=(history_length, 2)
            ).astype(np.float32)
            y_hist_mean = np.asarray(
                [source_log_signal_np(theta_true, x_t, cfg) for x_t in x_hist],
                dtype=np.float64,
            )
            y_hist = (
                y_hist_mean + cfg.observation_noise_std * rng.normal(size=history_length)
            ).astype(np.float32)
            history_x[:history_length] = x_hist
            history_y[:history_length, 0] = y_hist
        else:
            x_hist = np.empty((0, 2), dtype=np.float32)
            y_hist = np.empty((0,), dtype=np.float32)

        # Build the current belief from prior proposals and the complete synthetic
        # history. The model never sees the importance weights; it sees only the
        # resulting unweighted particles, matching its intended deployment API.
        proposals = self.prior.sample_n(rng, cfg.belief_proposals)
        log_weights = np.zeros((cfg.belief_proposals,), dtype=np.float64)
        for x_t, y_t in zip(x_hist, y_hist):
            log_weights += source_log_likelihood_np(y_t, proposals, x_t, cfg)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.maximum(np.sum(weights), 1e-300)
        belief_ess = np.float32(1.0 / np.sum(weights**2))
        indices = systematic_resample_np(rng, weights, cfg.num_particles)
        belief_particles = proposals[indices].astype(np.float32)
        if cfg.canonicalize_particle_sources and cfg.K > 1:
            belief_particles = canonicalize_sources_np(belief_particles)

        # The new likelihood factor to be amortised by q_phi.
        x = rng.uniform(cfg.design_low, cfg.design_high, size=(2,)).astype(np.float32)
        y_mean = float(source_log_signal_np(theta_true, x, cfg))
        y = np.float32(y_mean + cfg.observation_noise_std * rng.normal())

        # Requested label permutation. The loss additionally marginalises over all
        # enumerated permutations, which avoids forcing an arbitrary source ordering.
        if cfg.K > 1:
            theta_target = theta_true[rng.permutation(cfg.K)]
        else:
            theta_target = theta_true.copy()

        return {
            "theta_true": theta_true.astype(np.float32),
            "theta_target": theta_target.astype(np.float32),
            "belief_particles": belief_particles,
            "history_x": history_x,
            "history_y": history_y,
            "history_length": np.asarray([history_length], dtype=np.int32),
            "x": x,
            "y": np.asarray([y], dtype=np.float32),
            "condition": make_condition_np(x, y, cfg),
            "belief_ess": np.asarray([belief_ess], dtype=np.float32),
        }


class FiniteUpdateEpisodes(Dataset):
    def __init__(self, generator: SourceLocationUpdateGenerator, n_episodes: int, base_seed: int):
        self.generator = generator
        self.n_episodes = int(n_episodes)
        self.seeds = (np.arange(self.n_episodes, dtype=np.int64) + int(base_seed)).tolist()

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.generator.sample(np.random.default_rng(self.seeds[idx]))


class InfiniteUpdateEpisodes(IterableDataset):
    def __init__(self, generator: SourceLocationUpdateGenerator, base_seed: int):
        self.generator = generator
        self.base_seed = int(base_seed)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.base_seed + 1_000_003 * worker_id)
        while True:
            yield self.generator.sample(rng)


class EvalUpdateEpisodes(Dataset):
    def __init__(self, generator: SourceLocationUpdateGenerator, n_episodes: int, seed: int):
        seed_rng = np.random.default_rng(seed)
        episode_seeds = seed_rng.integers(
            0, np.iinfo(np.int64).max, size=n_episodes, dtype=np.int64
        )
        self.episodes = [
            generator.sample(np.random.default_rng(int(s))) for s in episode_seeds
        ]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.episodes[idx]


def collate_dicts(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack([item[name] for item in batch], axis=0) for name in batch[0]}


def make_update_train_loader(generator: SourceLocationUpdateGenerator, cfg: PosteriorConfig):
    if cfg.data_mode == "finite":
        dataset = FiniteUpdateEpisodes(generator, cfg.n_train_episodes, cfg.seed)
        torch_generator = torch.Generator()
        torch_generator.manual_seed(cfg.seed)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_dicts,
            num_workers=cfg.num_workers,
            generator=torch_generator,
            drop_last=True,
        )
    if cfg.data_mode == "infinite":
        dataset = InfiniteUpdateEpisodes(generator, cfg.seed)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_dicts,
            num_workers=cfg.num_workers,
        )
    raise ValueError("data_mode must be 'finite' or 'infinite'.")


def make_update_eval_loader(generator: SourceLocationUpdateGenerator, cfg: PosteriorConfig):
    dataset = EvalUpdateEpisodes(generator, cfg.n_eval_episodes, cfg.seed + 20_000)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_dicts,
        num_workers=0,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Theta-only loaders for policy rollouts
# -----------------------------------------------------------------------------
class FiniteThetaEpisodes(Dataset):
    def __init__(self, prior: SourceLocPrior, n_episodes: int, base_seed: int):
        self.prior = prior
        self.n_episodes = int(n_episodes)
        self.seeds = (np.arange(self.n_episodes, dtype=np.int64) + int(base_seed)).tolist()

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        return self.prior.sample(np.random.default_rng(self.seeds[idx]))


class InfiniteThetaEpisodes(IterableDataset):
    def __init__(self, prior: SourceLocPrior, base_seed: int):
        self.prior = prior
        self.base_seed = int(base_seed)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.base_seed + 1_000_003 * worker_id)
        while True:
            yield self.prior.sample(rng)


class EvalThetaEpisodes(Dataset):
    def __init__(self, prior: SourceLocPrior, n_episodes: int, seed: int):
        rng = np.random.default_rng(seed)
        self.thetas = np.stack([prior.sample(rng) for _ in range(n_episodes)], axis=0)

    def __len__(self):
        return self.thetas.shape[0]

    def __getitem__(self, idx):
        return self.thetas[idx]


def collate_thetas(batch):
    return np.stack(batch, axis=0)


def make_theta_train_loader(prior: SourceLocPrior, cfg: PolicyConfig):
    if cfg.data_mode == "finite":
        dataset = FiniteThetaEpisodes(prior, cfg.n_train_episodes, cfg.seed)
        torch_generator = torch.Generator()
        torch_generator.manual_seed(cfg.seed)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_thetas,
            num_workers=cfg.num_workers,
            generator=torch_generator,
            drop_last=True,
        )
    if cfg.data_mode == "infinite":
        dataset = InfiniteThetaEpisodes(prior, cfg.seed)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=collate_thetas,
            num_workers=cfg.num_workers,
        )
    raise ValueError("data_mode must be 'finite' or 'infinite'.")


def make_theta_eval_loader(prior: SourceLocPrior, cfg: PolicyConfig):
    dataset = EvalThetaEpisodes(prior, cfg.n_eval_episodes, cfg.seed + 20_000)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_thetas,
        num_workers=0,
        drop_last=False,
    )


# -----------------------------------------------------------------------------
# Source permutations and permutation-symmetrised conditional likelihood
# -----------------------------------------------------------------------------
def build_source_permutations(
    K: int,
    exact_max_k: int,
    sampled_permutations: int,
    seed: int,
) -> tuple[np.ndarray, bool]:
    """Return permutation indices and whether the set is the exact K! set."""
    if K <= exact_max_k:
        return np.asarray(list(permutations(range(K))), dtype=np.int32), True

    rng = np.random.default_rng(seed)
    unique: set[tuple[int, ...]] = {tuple(range(K))}
    while len(unique) < sampled_permutations:
        unique.add(tuple(int(i) for i in rng.permutation(K)))
    return np.asarray(sorted(unique), dtype=np.int32), False


class GMMParams(NamedTuple):
    logits: Array       # (..., M)
    means: Array        # (..., M, D)
    log_scales: Array   # (..., M, D)


def diagonal_gmm_log_prob_single(params: GMMParams, theta_flat: Array) -> Array:
    """log sum_m w_m Normal(theta; mu_m, diag(sigma_m^2))."""
    scales = jnp.exp(params.log_scales)
    z = (theta_flat[None, :] - params.means) / scales
    component_log_prob = -0.5 * jnp.sum(z**2, axis=-1)
    component_log_prob -= jnp.sum(params.log_scales, axis=-1)
    component_log_prob -= 0.5 * theta_flat.shape[-1] * jnp.log(2.0 * jnp.pi)
    return jsp.special.logsumexp(jax.nn.log_softmax(params.logits) + component_log_prob)


def permutation_marginal_log_prob_single(
    params: GMMParams,
    theta: Array,
    source_permutations: Array,
) -> Array:
    """Compute log average_pi q_phi(pi theta | ...)."""
    permuted = theta[source_permutations].reshape(source_permutations.shape[0], -1)
    log_probs = jax.vmap(lambda value: diagonal_gmm_log_prob_single(params, value))(permuted)
    return jsp.special.logsumexp(log_probs) - jnp.log(source_permutations.shape[0])


def permutation_marginal_nll(
    params: GMMParams,
    theta: Array,
    source_permutations: Array,
) -> Array:
    log_prob = jax.vmap(
        lambda p, t: permutation_marginal_log_prob_single(p, t, source_permutations)
    )(params, theta)
    return -log_prob


def best_component_assignment_rmse_single(
    params: GMMParams,
    theta: Array,
    source_permutations: Array,
) -> Array:
    means = params.means.reshape(params.means.shape[0], theta.shape[0], 2)
    permuted = theta[source_permutations]
    squared = jnp.mean(
        (means[:, None, :, :] - permuted[None, :, :, :]) ** 2,
        axis=(-1, -2),
    )
    return jnp.sqrt(jnp.min(squared))


# -----------------------------------------------------------------------------
# AdaLN-zero Transformer blocks and posterior density model
# -----------------------------------------------------------------------------
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class AdaLNZeroBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, *, key: Array):
        attn_key, ff1_key, ff2_key, mod_key = jax.random.split(key, 4)
        self.norm1 = eqx.nn.LayerNorm(hidden_dim, eps=1e-6, use_weight=False, use_bias=False)
        self.norm2 = eqx.nn.LayerNorm(hidden_dim, eps=1e-6, use_weight=False, use_bias=False)
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
        modulation = eqx.tree_at(lambda m: m.weight, modulation, jnp.zeros_like(modulation.weight))
        modulation = eqx.tree_at(lambda m: m.bias, modulation, jnp.zeros_like(modulation.bias))
        self.modulation = modulation

    def __call__(self, x: Array, condition: Array) -> Array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(
            self.modulation(jax.nn.silu(condition)), 6, axis=-1
        )

        h = _modulate(_layernorm_tokens(self.norm1, x), shift_a, scale_a)
        h = self.attention(h, h, h)
        x = x + gate_a[None, :] * h

        h = _modulate(_layernorm_tokens(self.norm2, x), shift_f, scale_f)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return x + gate_f[None, :] * h


class PosteriorGMMTransformer(eqx.Module):
    """Set Transformer parameterising q_phi(theta | belief particles, x, y)."""

    particle_in: eqx.nn.Linear
    condition_encoder: eqx.nn.MLP
    blocks: tuple[AdaLNZeroBlock, ...]
    final_norm: eqx.nn.LayerNorm
    pool_projection: eqx.nn.Linear
    output_head: eqx.nn.Linear

    K: int = eqx.field(static=True)
    theta_dim: int = eqx.field(static=True)
    num_components: int = eqx.field(static=True)
    min_scale: float = eqx.field(static=True)
    max_scale: float = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)

    def __init__(self, cfg: PosteriorConfig, *, key: Array):
        self.K = cfg.K
        self.theta_dim = 2 * cfg.K
        self.num_components = cfg.num_mixture_components
        self.min_scale = cfg.min_scale
        self.max_scale = cfg.max_scale
        self.canonicalize = cfg.canonicalize_particle_sources

        keys = jax.random.split(key, cfg.depth + 4)
        self.particle_in = eqx.nn.Linear(self.theta_dim, cfg.hidden_dim, key=keys[0])
        self.condition_encoder = eqx.nn.MLP(
            in_size=3,
            out_size=cfg.hidden_dim,
            width_size=cfg.hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
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
        self.pool_projection = eqx.nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim, key=keys[-2])

        output_size = cfg.num_mixture_components * (1 + 2 * self.theta_dim)
        output_head = eqx.nn.Linear(cfg.hidden_dim, output_size, key=keys[-1])

        # Initial q is a uniform mixture of the original prior. This gives finite,
        # meaningful NLL before training instead of an arbitrary random density.
        output_head = eqx.tree_at(
            lambda layer: layer.weight,
            output_head,
            jnp.zeros_like(output_head.weight),
        )
        initial_bias = np.zeros((output_size,), dtype=np.float32)
        stride = 1 + 2 * self.theta_dim
        # The scale head uses a bounded sigmoid map. Initialise its unconstrained
        # value so that the resulting standard deviation is exactly prior_std.
        initial_scale_fraction = (
            np.clip(cfg.prior_std, cfg.min_scale + 1e-6, cfg.max_scale - 1e-6)
            - cfg.min_scale
        ) / (cfg.max_scale - cfg.min_scale)
        initial_raw_scale = math.log(initial_scale_fraction / (1.0 - initial_scale_fraction))
        for component in range(cfg.num_mixture_components):
            offset = component * stride
            initial_bias[offset + 1 + self.theta_dim : offset + 1 + 2 * self.theta_dim] = initial_raw_scale
        output_head = eqx.tree_at(
            lambda layer: layer.bias,
            output_head,
            jnp.asarray(initial_bias),
        )
        self.output_head = output_head

    def __call__(self, belief_particles: Array, condition: Array) -> GMMParams:
        # belief_particles: (N,K,2). Source sorting makes the flattened token less
        # sensitive to arbitrary labels; the output NLL is separately symmetrised.
        if self.canonicalize and self.K > 1:
            belief_particles = canonicalize_sources_jax(belief_particles)
        tokens = belief_particles.reshape(belief_particles.shape[0], self.theta_dim)
        tokens = _linear_tokens(self.particle_in, tokens)
        condition_embedding = self.condition_encoder(condition)
        for block in self.blocks:
            tokens = block(tokens, condition_embedding)
        tokens = _layernorm_tokens(self.final_norm, tokens)

        # Mean and standard deviation are permutation-invariant summaries over the
        # particle/token axis. Combining both preserves location and spread information.
        pooled_mean = jnp.mean(tokens, axis=0)
        pooled_std = jnp.sqrt(jnp.var(tokens, axis=0) + 1e-6)
        pooled = jax.nn.silu(self.pool_projection(jnp.concatenate([pooled_mean, pooled_std])))
        raw = self.output_head(pooled).reshape(
            self.num_components, 1 + 2 * self.theta_dim
        )
        logits = raw[:, 0]
        means = raw[:, 1 : 1 + self.theta_dim]
        raw_scales = raw[:, 1 + self.theta_dim :]
        # Bounded positive scale avoids both exact zero variance and numerical overflow:
        # sigma = sigma_min + (sigma_max-sigma_min) sigmoid(raw).
        scales = self.min_scale + (self.max_scale - self.min_scale) * jax.nn.sigmoid(raw_scales)
        return GMMParams(logits=logits, means=means, log_scales=jnp.log(scales))


def sample_gmm_single(params: GMMParams, key: Array, n_samples: int, K: int) -> Array:
    """Exact categorical sampling for evaluation/deployment."""
    component_key, noise_key = jax.random.split(key)
    components = jax.random.categorical(
        component_key, params.logits, shape=(n_samples,)
    )
    means = params.means[components]
    scales = jnp.exp(params.log_scales[components])
    noise = jax.random.normal(noise_key, shape=means.shape)
    return (means + scales * noise).reshape(n_samples, K, 2)


def relaxed_sample_gmm_single(
    params: GMMParams,
    key: Array,
    n_samples: int,
    K: int,
    temperature: float,
) -> Array:
    """Differentiable Gumbel-softmax mixture sampling for policy training.

    For each particle n and component m,

        a_nm = softmax((log w_m + g_nm) / tau),
        theta_n = sum_m a_nm (mu_m + sigma_m epsilon_nm).

    As tau -> 0 this approaches categorical mixture sampling, but gradients become
    less smooth. The policy script uses a small positive temperature.
    """
    gumbel_key, noise_key = jax.random.split(key)
    gumbel = jax.random.gumbel(
        gumbel_key, shape=(n_samples, params.logits.shape[-1])
    )
    assignment = jax.nn.softmax((params.logits[None, :] + gumbel) / temperature, axis=-1)
    noise = jax.random.normal(
        noise_key,
        shape=(n_samples, params.means.shape[0], params.means.shape[1]),
    )
    candidates = params.means[None, :, :] + jnp.exp(params.log_scales)[None, :, :] * noise
    samples = jnp.einsum("nm,nmd->nd", assignment, candidates)
    return samples.reshape(n_samples, K, 2)


def save_posterior_model(path: str | Path, model: PosteriorGMMTransformer):
    eqx.tree_serialise_leaves(Path(path), model)


def load_posterior_model(
    path: str | Path,
    cfg: PosteriorConfig,
    *,
    key: Array | None = None,
) -> PosteriorGMMTransformer:
    if key is None:
        key = jax.random.key(0)
    skeleton = PosteriorGMMTransformer(cfg, key=key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)


# -----------------------------------------------------------------------------
# Set Transformer policy
# -----------------------------------------------------------------------------
class SetTransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, *, key: Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(hidden_dim)
        self.norm2 = eqx.nn.LayerNorm(hidden_dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=k1,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=k2)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=k3)

    def __call__(self, x: Array) -> Array:
        h = _layernorm_tokens(self.norm1, x)
        x = x + self.attention(h, h, h)
        h = _layernorm_tokens(self.norm2, x)
        h = _linear_tokens(self.ff_out, jax.nn.gelu(_linear_tokens(self.ff_in, h)))
        return x + h


class DesignPolicyTransformer(eqx.Module):
    """Map an unweighted current belief to the next bounded design x_t in R^2."""

    particle_in: eqx.nn.Linear
    step_encoder: eqx.nn.MLP
    blocks: tuple[SetTransformerBlock, ...]
    final_norm: eqx.nn.LayerNorm
    pool_projection: eqx.nn.Linear
    output: eqx.nn.Linear

    K: int = eqx.field(static=True)
    theta_dim: int = eqx.field(static=True)
    design_low: float = eqx.field(static=True)
    design_high: float = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)

    def __init__(self, posterior_cfg: PosteriorConfig, policy_cfg: PolicyConfig, *, key: Array):
        self.K = posterior_cfg.K
        self.theta_dim = 2 * posterior_cfg.K
        self.design_low = posterior_cfg.design_low
        self.design_high = posterior_cfg.design_high
        self.canonicalize = posterior_cfg.canonicalize_particle_sources

        keys = jax.random.split(key, policy_cfg.depth + 4)
        self.particle_in = eqx.nn.Linear(self.theta_dim, policy_cfg.hidden_dim, key=keys[0])
        self.step_encoder = eqx.nn.MLP(
            in_size=1,
            out_size=policy_cfg.hidden_dim,
            width_size=policy_cfg.hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=keys[1],
        )
        self.blocks = tuple(
            SetTransformerBlock(
                policy_cfg.hidden_dim,
                policy_cfg.heads,
                policy_cfg.mlp_ratio * policy_cfg.hidden_dim,
                key=keys[2 + i],
            )
            for i in range(policy_cfg.depth)
        )
        self.final_norm = eqx.nn.LayerNorm(policy_cfg.hidden_dim)
        self.pool_projection = eqx.nn.Linear(
            2 * policy_cfg.hidden_dim,
            policy_cfg.hidden_dim,
            key=keys[-2],
        )
        self.output = eqx.nn.Linear(policy_cfg.hidden_dim, 2, key=keys[-1])

    def __call__(self, belief_particles: Array, step_fraction: Array) -> Array:
        if self.canonicalize and self.K > 1:
            belief_particles = canonicalize_sources_jax(belief_particles)
        tokens = belief_particles.reshape(belief_particles.shape[0], self.theta_dim)
        tokens = _linear_tokens(self.particle_in, tokens)
        tokens = tokens + self.step_encoder(jnp.asarray([step_fraction]))[None, :]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = _layernorm_tokens(self.final_norm, tokens)
        pooled = jnp.concatenate(
            [jnp.mean(tokens, axis=0), jnp.sqrt(jnp.var(tokens, axis=0) + 1e-6)]
        )
        hidden = jax.nn.silu(self.pool_projection(pooled))
        unit = jnp.tanh(self.output(hidden))
        midpoint = 0.5 * (self.design_low + self.design_high)
        half_range = 0.5 * (self.design_high - self.design_low)
        return midpoint + half_range * unit


def save_policy_model(path: str | Path, model: DesignPolicyTransformer):
    eqx.tree_serialise_leaves(Path(path), model)


def load_policy_model(
    path: str | Path,
    posterior_cfg: PosteriorConfig,
    policy_cfg: PolicyConfig,
    *,
    key: Array | None = None,
) -> DesignPolicyTransformer:
    if key is None:
        key = jax.random.key(0)
    skeleton = DesignPolicyTransformer(posterior_cfg, policy_cfg, key=key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)
