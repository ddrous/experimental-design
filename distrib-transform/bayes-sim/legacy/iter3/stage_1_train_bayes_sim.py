#%% 1) Imports, configuration, and experiment conventions
"""Train an in-context particle pushforward that approximates Bayes' rule.

The model in this file is deliberately a *sample transport*, not a conditional
mixture-density estimator.  Given a fresh unweighted sample from the prior and an
unordered, padded context

    D = {(x_i, y_i)}_{i=1}^m,

it implements

    Z = {z_n}_{n=1}^N,  z_n ~ p(theta)
        -> T_phi(Z, D) = {theta_hat_n}_{n=1}^N
        ~= samples from p(theta | D).

This is an in-context approximation of the Bayesian pushforward.  The particle axis
is exchangeable, the context-pair axis is exchangeable, and active context pairs either
condition every Transformer block through AdaLN-zero or are attended directly through
an order-invariant observation encoder and particle-to-observation cross-attention.

Approximate Bayes posterior samples are retained only as diagnostics.  Two teachers are
implemented:

1. ``snis``: self-normalised importance sampling, used only when the likelihood
   p(y | theta, x) can be evaluated.
2. ``abc``: a simulator-only ABC kernel teacher, used when likelihood evaluation is
   unavailable but forward simulation y ~ p(y | theta, x) remains possible.

The primary objective is a particle-KDE negative ELBO.  The posterior normalising
constant is never evaluated.  Sliced 2-Wasserstein sample matching is retained only as
a diagnostic, and an optional transport-kinetic penalty is disabled by default.
No sequential ordering is imposed in Stage I: context sizes, slot locations, pair
order, and source labels are randomised independently in every episode.

The source-location simulator is retained from the supplied code.  Its likelihood is

    p(y | theta, x) = Normal(y; log s(theta, x), sigma_y^2),

whereas ``posterior`` always refers to p(theta | D), never to p(y | theta, x).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Sequence
import json
import math
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import matplotlib.pyplot as plt
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


@dataclass(frozen=True)
class BayesTransportConfig:
    """Configuration for the amortised Bayesian particle pushforward."""

    env_name: str = "bayes_pushforward_adaln"
    seed: int = 2030

    # Source-localisation simulator.
    K: int = 2
    prior_std: float = 1.0
    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.30

    # ``likelihood_available=False`` is the important implicit-simulator setting.
    # The code can still *simulate* y ~ p(y | theta, x), but does not evaluate a
    # numerical likelihood in the ABC teacher or likelihood diagnostic panels.
    likelihood_available: bool = True
    teacher_method: str = "auto"  # one of: auto, snis, abc
    teacher_proposals: int = 4096
    teacher_tempering: float = 1.0
    abc_bandwidth: float = 0.60
    abc_replicates: int = 2

    # Prior and approximate-posterior particle sets.
    num_particles: int = 128
    min_context_pairs: int = 0
    max_context_pairs: int = 8
    randomise_context_slots: bool = True
    randomise_context_order: bool = True
    randomise_source_labels: bool = True
    canonicalize_particle_sources: bool = True

    # PyTorch remains only the batching/shuffling front end.  Numerical model
    # operations are JAX + Equinox + Optax, as in the supplied implementation.
    data_mode: str = "finite"
    n_train_episodes: int = 20_000
    n_eval_episodes: int = 512
    batch_size: int = 64
    num_workers: int = 0
    steps_per_epoch: int = 400

    # AdaLN-zero particle Transformer and context-set encoder.
    use_adaln: bool = True
    hidden_dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    context_encoder_depth: int = 2
    max_particle_displacement: float = 6.0

    # Sample-matching objective.  The default is exactly one objective term.
    num_swd_projections: int = 64
    kinetic_weight: float = 0.0
    use_proper_scoring_rule: bool = True
    elbo_kde_bandwidth: float = 0.35

    # Optimisation.
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.0

    # Conditioning normalisation.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Diagnostics and persistence.
    final_plot_examples: int = 6
    grid_size: int = 180
    runs_base: str = "./runs"
    save_every_epochs: int = 1


CFG = BayesTransportConfig(
    # Set likelihood_available=False to exercise the simulator-only ABC path.
    likelihood_available=True,
    teacher_method="auto",
    use_adaln=True,
    max_context_pairs=20,
    num_particles=256,
    epochs=100,
    n_train_episodes=20_000,
    n_eval_episodes=512,
    batch_size=128,
    kinetic_weight=0.0,
    use_proper_scoring_rule=True,
)


#%% 2) Run directories, configuration IO, and source snapshots
def dataclass_from_dict(cls, values: dict[str, Any]):
    """Construct a dataclass while ignoring unrelated YAML metadata fields."""
    allowed = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in allowed})


def make_run_dir(env_name: str, base: str | Path = "./runs") -> Path:
    """Create ``runs/<name>_<timestamp>/{plots,artefacts}``."""
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
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


#%% 3) Prior, simulator, likelihood, and source-label conventions
class SourceLocPrior:
    """theta_k ~ Normal(0, prior_std^2 I_2), independently for k=1,...,K."""

    def __init__(self, K: int = 1, prior_std: float = 1.0):
        self.K = int(K)
        self.prior_std = float(prior_std)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.prior_std, size=(self.K, 2)).astype(np.float32)

    def sample_n(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(0.0, self.prior_std, size=(n, self.K, 2)).astype(np.float32)


def source_log_signal_np(
    theta: np.ndarray,
    x: np.ndarray,
    CFG: BayesTransportConfig,
) -> np.ndarray:
    """Return log mean intensity, broadcasting over leading theta dimensions."""
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    dist_sq = np.sum((theta - np.expand_dims(x, axis=-2)) ** 2, axis=-1)
    intensity = CFG.background + np.sum(
        CFG.source_strength / (CFG.softening + dist_sq), axis=-1
    )
    return np.log(intensity)


def source_log_signal_jax(
    theta: Array,
    x: Array,
    CFG: BayesTransportConfig,
) -> Array:
    """JAX mean of p(y | theta, x), broadcasting over leading dimensions."""
    dist_sq = jnp.sum((theta - jnp.expand_dims(x, axis=-2)) ** 2, axis=-1)
    intensity = CFG.background + jnp.sum(
        CFG.source_strength / (CFG.softening + dist_sq), axis=-1
    )
    return jnp.log(intensity)


def source_log_likelihood_np(
    y: np.ndarray | float,
    theta: np.ndarray,
    x: np.ndarray,
    CFG: BayesTransportConfig,
) -> np.ndarray:
    """Evaluate log p(y | theta, x); never call this in simulator-only mode."""
    if not CFG.likelihood_available:
        raise RuntimeError(
            "The configuration declares p(y | theta, x) unavailable. "
            "Use the ABC teacher, which needs only forward simulation."
        )
    mean = source_log_signal_np(theta, x, CFG)
    z = (np.asarray(y, dtype=np.float64) - mean) / CFG.observation_noise_std
    normalizer = math.log(CFG.observation_noise_std * math.sqrt(2.0 * math.pi))
    return -0.5 * z**2 - normalizer


def simulate_observation_np(
    theta: np.ndarray,
    x: np.ndarray,
    CFG: BayesTransportConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw y ~ p(y | theta, x) without requiring likelihood evaluation."""
    mean = source_log_signal_np(theta, x, CFG)
    return np.asarray(
        mean + CFG.observation_noise_std * rng.normal(size=np.shape(mean)),
        dtype=np.float32,
    )


def simulate_observation_jax(
    theta: Array,
    x: Array,
    CFG: BayesTransportConfig,
    key: Array,
) -> Array:
    """Differentiable reparameterised simulation from p(y | theta, x)."""
    mean = source_log_signal_jax(theta, x, CFG)
    return mean + CFG.observation_noise_std * jax.random.normal(key, mean.shape)


def canonicalize_sources_np(theta: np.ndarray) -> np.ndarray:
    """Sort sources by x coordinate inside each sample; ties are measure-zero here."""
    theta = np.asarray(theta)
    order = np.argsort(theta[..., 0], axis=-1)
    return np.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_sources_jax(theta: Array) -> Array:
    order = jnp.argsort(theta[..., 0], axis=-1)
    return jnp.take_along_axis(theta, order[..., None], axis=-2)


def randomise_source_labels_np(
    theta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independently relabel the K exchangeable sources in every particle/sample."""
    theta = np.asarray(theta)
    if theta.shape[-2] <= 1:
        return theta.copy()
    flat = theta.reshape((-1,) + theta.shape[-2:]).copy()
    for row in range(flat.shape[0]):
        flat[row] = flat[row, rng.permutation(flat.shape[1])]
    return flat.reshape(theta.shape)


#%% 4) Approximate Bayes teachers: SNIS or simulator-only ABC
def systematic_resample_np(
    rng: np.random.Generator,
    weights: np.ndarray,
    n: int,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.maximum(weights.sum(), 1e-300)
    positions = (rng.random() + np.arange(n, dtype=np.float64)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, positions, side="right")


def resolve_teacher_method(CFG: BayesTransportConfig) -> str:
    method = CFG.teacher_method.lower()
    if method == "auto":
        return "snis" if CFG.likelihood_available else "abc"
    if method not in {"snis", "abc"}:
        raise ValueError("teacher_method must be one of: auto, snis, abc.")
    if method == "snis" and not CFG.likelihood_available:
        raise ValueError(
            "teacher_method='snis' requires likelihood_available=True. "
            "Choose 'abc' or 'auto' for an implicit simulator."
        )
    return method


class TeacherPosterior(NamedTuple):
    particles: np.ndarray
    ess: np.float32
    method_code: np.int32  # 0 = SNIS, 1 = ABC


def approximate_posterior_particles_np(
    prior: SourceLocPrior,
    CFG: BayesTransportConfig,
    rng: np.random.Generator,
    context_x: np.ndarray,
    context_y: np.ndarray,
    context_mask: np.ndarray,
) -> TeacherPosterior:
    """Approximate p(theta | D) and return an unweighted posterior sample.

    SNIS uses the evaluable product likelihood

        p(D | theta) = product_i p(y_i | theta, x_i).

    ABC never evaluates p(y | theta, x).  It repeatedly simulates synthetic outcomes
    at the observed designs, compares them with the observed outcomes, and forms a
    smooth kernel weight.  This is intentionally a simple baseline teacher; replacing
    it with SMC-ABC, neural ratio estimation, or another SBI method does not change the
    transport model's API.
    """
    method = resolve_teacher_method(CFG)
    proposals = prior.sample_n(rng, CFG.teacher_proposals)
    active = np.flatnonzero(np.asarray(context_mask) > 0.5)

    if active.size == 0:
        indices = rng.integers(0, CFG.teacher_proposals, size=CFG.num_particles)
        posterior = proposals[indices]
        return TeacherPosterior(
            particles=posterior.astype(np.float32),
            ess=np.float32(CFG.teacher_proposals),
            method_code=np.int32(0 if method == "snis" else 1),
        )

    log_weights = np.zeros((CFG.teacher_proposals,), dtype=np.float64)
    if method == "snis":
        for slot in active:
            log_weights += source_log_likelihood_np(
                float(context_y[slot, 0]), proposals, context_x[slot], CFG
            )
    else:
        # Simulator-only ABC kernel.  Each proposal generates R pseudo-observations
        # from p(y | theta, x); no likelihood value is queried or approximated by name.
        normalising_scale = max(CFG.observation_noise_std, 1e-6)
        for slot in active:
            mean = source_log_signal_np(proposals, context_x[slot], CFG)
            pseudo_y = mean[:, None] + CFG.observation_noise_std * rng.normal(
                size=(CFG.teacher_proposals, CFG.abc_replicates)
            )
            discrepancy_sq = np.mean(
                ((pseudo_y - float(context_y[slot, 0])) / normalising_scale) ** 2,
                axis=1,
            )
            log_weights += -0.5 * discrepancy_sq / max(CFG.abc_bandwidth**2, 1e-8)

    log_weights /= max(CFG.teacher_tempering, 1e-6)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.maximum(weights.sum(), 1e-300)
    ess = np.float32(1.0 / np.sum(weights**2))
    indices = systematic_resample_np(rng, weights, CFG.num_particles)
    posterior = proposals[indices].astype(np.float32)
    return TeacherPosterior(
        particles=posterior,
        ess=ess,
        method_code=np.int32(0 if method == "snis" else 1),
    )


#%% 5) Random, non-sequential Stage-I episodes and PyTorch loaders
class BayesTransportEpisodeGenerator:
    """Generate independent set-conditioned Bayes-transport training episodes.

    The context is a random *set*, not a trajectory.  Active pairs are sampled jointly,
    optionally shuffled, and optionally written into random padded slots.  Consequently,
    the model cannot rely on temporal position or insertion order.  The context mask is
    the only indicator of which slots are active.
    """

    def __init__(self, prior: SourceLocPrior, CFG: BayesTransportConfig):
        self.prior = prior
        self.CFG = CFG

    def sample(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        CFG = self.CFG
        theta_true = self.prior.sample(rng)
        context_size = int(
            rng.integers(CFG.min_context_pairs, CFG.max_context_pairs + 1)
        )

        active_x = rng.uniform(
            CFG.design_low, CFG.design_high, size=(context_size, 2)
        ).astype(np.float32)
        active_y = np.asarray(
            [simulate_observation_np(theta_true, x_i, CFG, rng) for x_i in active_x],
            dtype=np.float32,
        ).reshape(context_size, 1)

        if CFG.randomise_context_order and context_size > 1:
            order = rng.permutation(context_size)
            active_x = active_x[order]
            active_y = active_y[order]

        context_x = np.zeros((CFG.max_context_pairs, 2), dtype=np.float32)
        context_y = np.zeros((CFG.max_context_pairs, 1), dtype=np.float32)
        context_mask = np.zeros((CFG.max_context_pairs,), dtype=np.float32)
        if context_size > 0:
            if CFG.randomise_context_slots:
                slots = rng.choice(
                    CFG.max_context_pairs, size=context_size, replace=False
                )
            else:
                slots = np.arange(context_size)
            context_x[slots] = active_x
            context_y[slots] = active_y
            context_mask[slots] = 1.0

        prior_particles = self.prior.sample_n(rng, CFG.num_particles)
        teacher = approximate_posterior_particles_np(
            self.prior, CFG, rng, context_x, context_y, context_mask
        )
        target_particles = teacher.particles

        # Random relabelling is applied independently.  The model/loss may then
        # canonicalise each K-source sample, so no semantically meaningful source ID is
        # introduced by data generation.
        if CFG.randomise_source_labels and CFG.K > 1:
            theta_true = randomise_source_labels_np(theta_true, rng)
            prior_particles = randomise_source_labels_np(prior_particles, rng)
            target_particles = randomise_source_labels_np(target_particles, rng)

        return {
            "theta_true": theta_true.astype(np.float32),
            "prior_particles": prior_particles.astype(np.float32),
            "target_particles": target_particles.astype(np.float32),
            "context_x": context_x,
            "context_y": context_y,
            "context_mask": context_mask,
            "context_size": np.asarray([context_size], dtype=np.int32),
            "teacher_ess": np.asarray([teacher.ess], dtype=np.float32),
            "teacher_method_code": np.asarray([teacher.method_code], dtype=np.int32),
        }


class FiniteTransportEpisodes(Dataset):
    def __init__(
        self,
        generator: BayesTransportEpisodeGenerator,
        n_episodes: int,
        base_seed: int,
    ):
        self.generator = generator
        self.n_episodes = int(n_episodes)
        self.seeds = (
            np.arange(self.n_episodes, dtype=np.int64) + int(base_seed)
        ).tolist()

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.generator.sample(np.random.default_rng(self.seeds[idx]))


class InfiniteTransportEpisodes(IterableDataset):
    def __init__(self, generator: BayesTransportEpisodeGenerator, base_seed: int):
        self.generator = generator
        self.base_seed = int(base_seed)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.base_seed + 1_000_003 * worker_id)
        while True:
            yield self.generator.sample(rng)


class EvalTransportEpisodes(Dataset):
    def __init__(
        self,
        generator: BayesTransportEpisodeGenerator,
        n_episodes: int,
        seed: int,
    ):
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


def make_transport_train_loader(
    generator: BayesTransportEpisodeGenerator,
    CFG: BayesTransportConfig,
):
    if CFG.data_mode == "finite":
        dataset = FiniteTransportEpisodes(generator, CFG.n_train_episodes, CFG.seed)
        torch_generator = torch.Generator()
        torch_generator.manual_seed(CFG.seed)
        return DataLoader(
            dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            collate_fn=collate_dicts,
            num_workers=CFG.num_workers,
            generator=torch_generator,
            drop_last=True,
        )
    if CFG.data_mode == "infinite":
        dataset = InfiniteTransportEpisodes(generator, CFG.seed)
        return DataLoader(
            dataset,
            batch_size=CFG.batch_size,
            collate_fn=collate_dicts,
            num_workers=CFG.num_workers,
        )
    raise ValueError("data_mode must be 'finite' or 'infinite'.")


def make_transport_eval_loader(
    generator: BayesTransportEpisodeGenerator,
    CFG: BayesTransportConfig,
):
    dataset = EvalTransportEpisodes(generator, CFG.n_eval_episodes, CFG.seed + 20_000)
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_dicts,
        num_workers=0,
        drop_last=False,
    )


#%% 6) AdaLN-zero blocks, context-set conditioning, and particle transport
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

    def __call__(self, tokens: Array, condition: Array) -> Array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(
            self.modulation(jax.nn.silu(condition)), 6, axis=-1
        )

        h = _modulate(_layernorm_tokens(self.norm1, tokens), shift_a, scale_a)
        h = self.attention(h, h, h)
        tokens = tokens + gate_a[None, :] * h

        h = _modulate(_layernorm_tokens(self.norm2, tokens), shift_f, scale_f)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return tokens + gate_f[None, :] * h


class ContextSetEncoder(eqx.Module):
    """Permutation-invariant encoder for many design--outcome pairs."""

    pair_encoder: eqx.nn.MLP
    summary_encoder: eqx.nn.MLP
    empty_context: Array

    design_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)

    def __init__(self, CFG: BayesTransportConfig, *, key: Array):
        pair_key, summary_key, empty_key = jax.random.split(key, 3)
        self.design_scale = max(abs(CFG.design_low), abs(CFG.design_high), 1.0)
        self.y_center = CFG.y_center
        self.y_scale = CFG.y_scale
        self.pair_encoder = eqx.nn.MLP(
            in_size=3,
            out_size=CFG.hidden_dim,
            width_size=CFG.hidden_dim,
            depth=CFG.context_encoder_depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=pair_key,
        )
        self.summary_encoder = eqx.nn.MLP(
            in_size=2 * CFG.hidden_dim + 1,
            out_size=CFG.hidden_dim,
            width_size=CFG.hidden_dim,
            depth=2,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=summary_key,
        )
        self.empty_context = 0.02 * jax.random.normal(empty_key, (CFG.hidden_dim,))

    def __call__(
        self,
        context_x: Array,
        context_y: Array,
        context_mask: Array,
    ) -> Array:
        features = jnp.concatenate(
            [
                context_x / self.design_scale,
                (context_y - self.y_center) / self.y_scale,
            ],
            axis=-1,
        )
        encoded = jax.vmap(self.pair_encoder)(features)
        mask = context_mask[:, None]
        count = jnp.sum(context_mask)
        safe_count = jnp.maximum(count, 1.0)
        masked_mean = jnp.sum(encoded * mask, axis=0) / safe_count
        masked_max = jnp.max(
            jnp.where(mask > 0.5, encoded, -jnp.inf), axis=0
        )
        masked_max = jnp.where(count > 0.5, masked_max, jnp.zeros_like(masked_max))
        count_fraction = count / jnp.maximum(context_mask.shape[0], 1)
        summary = self.summary_encoder(
            jnp.concatenate([masked_mean, masked_max, jnp.asarray([count_fraction])])
        )
        return jnp.where(count > 0.5, summary, self.empty_context)


class MaskedSetAttentionBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, *, key: Array):
        attn_key, ff1_key, ff2_key = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(hidden_dim)
        self.norm2 = eqx.nn.LayerNorm(hidden_dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=attn_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)

    def __call__(self, tokens: Array, token_mask: Array) -> Array:
        attention_mask = jnp.broadcast_to(
            token_mask[None, :] > 0.5,
            (tokens.shape[0], tokens.shape[0]),
        )
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h, mask=attention_mask)
        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        tokens = tokens + h
        return jnp.where(token_mask[:, None] > 0.5, tokens, jnp.zeros_like(tokens))


class ObservationSetEncoder(eqx.Module):
    """Permutation-equivariant tokens for an unordered design--outcome set."""

    pair_encoder: eqx.nn.MLP
    blocks: tuple[MaskedSetAttentionBlock, ...]
    empty_context: Array

    design_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)

    def __init__(self, CFG: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, CFG.context_encoder_depth + 2)
        self.design_scale = max(abs(CFG.design_low), abs(CFG.design_high), 1.0)
        self.y_center = CFG.y_center
        self.y_scale = CFG.y_scale
        self.pair_encoder = eqx.nn.MLP(
            in_size=3,
            out_size=CFG.hidden_dim,
            width_size=CFG.hidden_dim,
            depth=CFG.context_encoder_depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=keys[0],
        )
        self.blocks = tuple(
            MaskedSetAttentionBlock(
                CFG.hidden_dim,
                CFG.heads,
                CFG.mlp_ratio * CFG.hidden_dim,
                key=keys[1 + i],
            )
            for i in range(CFG.context_encoder_depth)
        )
        self.empty_context = 0.02 * jax.random.normal(keys[-1], (CFG.hidden_dim,))

    def __call__(
        self,
        context_x: Array,
        context_y: Array,
        context_mask: Array,
    ) -> tuple[Array, Array]:
        features = jnp.concatenate(
            [
                context_x / self.design_scale,
                (context_y - self.y_center) / self.y_scale,
            ],
            axis=-1,
        )
        encoded = jax.vmap(self.pair_encoder)(features)
        encoded = jnp.where(
            context_mask[:, None] > 0.5, encoded, jnp.zeros_like(encoded)
        )
        tokens = jnp.concatenate([self.empty_context[None, :], encoded], axis=0)
        token_mask = jnp.concatenate(
            [jnp.ones((1,), dtype=context_mask.dtype), context_mask], axis=0
        )
        for block in self.blocks:
            tokens = block(tokens, token_mask)
        return tokens, token_mask


class ParticleObservationBlock(eqx.Module):
    particle_norm: eqx.nn.LayerNorm
    cross_norm: eqx.nn.LayerNorm
    observation_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    particle_attention: eqx.nn.MultiheadAttention
    observation_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, hidden_dim: int, heads: int, mlp_dim: int, *, key: Array):
        particle_key, observation_key, ff1_key, ff2_key = jax.random.split(key, 4)
        self.particle_norm = eqx.nn.LayerNorm(hidden_dim)
        self.cross_norm = eqx.nn.LayerNorm(hidden_dim)
        self.observation_norm = eqx.nn.LayerNorm(hidden_dim)
        self.ff_norm = eqx.nn.LayerNorm(hidden_dim)
        self.particle_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=particle_key,
        )
        self.observation_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=observation_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)

    def __call__(
        self,
        particles: Array,
        observations: Array,
        observation_mask: Array,
    ) -> Array:
        h = _layernorm_tokens(self.particle_norm, particles)
        particles = particles + self.particle_attention(h, h, h)

        query = _layernorm_tokens(self.cross_norm, particles)
        key_value = _layernorm_tokens(self.observation_norm, observations)
        cross_mask = jnp.broadcast_to(
            observation_mask[None, :] > 0.5,
            (particles.shape[0], observations.shape[0]),
        )
        particles = particles + self.observation_attention(
            query, key_value, key_value, mask=cross_mask
        )

        h = _layernorm_tokens(self.ff_norm, particles)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + h


class BayesPushforwardTransformer(eqx.Module):
    """Exchangeable transport T_phi(Z, D) from prior to posterior particles."""

    particle_in: eqx.nn.Linear | eqx.nn.MLP
    context_encoder: ContextSetEncoder | ObservationSetEncoder
    blocks: tuple[AdaLNZeroBlock, ...] | tuple[ParticleObservationBlock, ...]
    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear

    K: int = eqx.field(static=True)
    theta_dim: int = eqx.field(static=True)
    max_displacement: float = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)
    use_adaln: bool = eqx.field(static=True)

    def __init__(self, CFG: BayesTransportConfig, *, key: Array):
        self.K = CFG.K
        self.theta_dim = 2 * CFG.K
        self.max_displacement = CFG.max_particle_displacement
        self.canonicalize = CFG.canonicalize_particle_sources
        self.use_adaln = CFG.use_adaln

        keys = jax.random.split(key, CFG.depth + 4)
        if self.use_adaln:
            self.particle_in = eqx.nn.Linear(
                self.theta_dim, CFG.hidden_dim, key=keys[0]
            )
            self.context_encoder = ContextSetEncoder(CFG, key=keys[1])
            self.blocks = tuple(
                AdaLNZeroBlock(
                    CFG.hidden_dim,
                    CFG.heads,
                    CFG.mlp_ratio * CFG.hidden_dim,
                    key=keys[2 + i],
                )
                for i in range(CFG.depth)
            )
        else:
            self.particle_in = eqx.nn.MLP(
                in_size=self.theta_dim,
                out_size=CFG.hidden_dim,
                width_size=CFG.hidden_dim,
                depth=CFG.context_encoder_depth,
                activation=jax.nn.silu,
                final_activation=jax.nn.silu,
                key=keys[0],
            )
            self.context_encoder = ObservationSetEncoder(CFG, key=keys[1])
            self.blocks = tuple(
                ParticleObservationBlock(
                    CFG.hidden_dim,
                    CFG.heads,
                    CFG.mlp_ratio * CFG.hidden_dim,
                    key=keys[2 + i],
                )
                for i in range(CFG.depth)
            )
        self.final_norm = eqx.nn.LayerNorm(CFG.hidden_dim)
        output = eqx.nn.Linear(CFG.hidden_dim, self.theta_dim, key=keys[-1])

        # Identity initialization is natural for a pushforward: before training,
        # empty-context inputs remain prior samples and non-empty contexts initially
        # induce no arbitrary displacement.
        output = eqx.tree_at(
            lambda layer: layer.weight, output, jnp.zeros_like(output.weight)
        )
        output = eqx.tree_at(
            lambda layer: layer.bias, output, jnp.zeros_like(output.bias)
        )
        self.displacement_head = output

    def __call__(
        self,
        prior_particles: Array,
        context_x: Array,
        context_y: Array,
        context_mask: Array,
    ) -> Array:
        if self.canonicalize and self.K > 1:
            prior_particles = canonicalize_sources_jax(prior_particles)
        flat_prior = prior_particles.reshape(prior_particles.shape[0], self.theta_dim)
        if self.use_adaln:
            tokens = _linear_tokens(self.particle_in, flat_prior)
            condition = self.context_encoder(context_x, context_y, context_mask)
            for block in self.blocks:
                tokens = block(tokens, condition)
        else:
            tokens = jax.vmap(self.particle_in)(flat_prior)
            observations, observation_mask = self.context_encoder(
                context_x, context_y, context_mask
            )
            for block in self.blocks:
                tokens = block(tokens, observations, observation_mask)
        tokens = _layernorm_tokens(self.final_norm, tokens)
        displacement = self.max_displacement * jnp.tanh(
            _linear_tokens(self.displacement_head, tokens)
        )
        transported = (flat_prior + displacement).reshape(
            prior_particles.shape[0], self.K, 2
        )
        if self.canonicalize and self.K > 1:
            transported = canonicalize_sources_jax(transported)
        return transported


def save_transport_model(path: str | Path, model: BayesPushforwardTransformer):
    eqx.tree_serialise_leaves(Path(path), model)


def load_transport_model(
    path: str | Path,
    CFG: BayesTransportConfig,
    *,
    key: Array | None = None,
) -> BayesPushforwardTransformer:
    if key is None:
        key = jax.random.key(0)
    skeleton = BayesPushforwardTransformer(CFG, key=key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)


#%% 7) Sliced-Wasserstein objective and sample diagnostics
def flatten_particle_measure(particles: Array, CFG: BayesTransportConfig) -> Array:
    if CFG.canonicalize_particle_sources and CFG.K > 1:
        particles = canonicalize_sources_jax(particles)
    return particles.reshape(particles.shape[0], 2 * CFG.K)


def random_unit_directions(key: Array, n_directions: int, dimension: int) -> Array:
    directions = jax.random.normal(key, (n_directions, dimension))
    return directions / jnp.maximum(
        jnp.linalg.norm(directions, axis=-1, keepdims=True), 1e-8
    )


def sliced_wasserstein_squared(
    predicted_particles: Array,
    target_particles: Array,
    directions: Array,
    CFG: BayesTransportConfig,
) -> Array:
    """Monte-Carlo sliced W_2^2 between two equally weighted particle measures."""
    predicted = flatten_particle_measure(predicted_particles, CFG)
    target = flatten_particle_measure(target_particles, CFG)
    predicted_projection = predicted @ directions.T
    target_projection = target @ directions.T
    predicted_sorted = jnp.sort(predicted_projection, axis=0)
    target_sorted = jnp.sort(target_projection, axis=0)
    return jnp.mean((predicted_sorted - target_sorted) ** 2)


def transport_kinetic_energy(
    prior_particles: Array,
    transported_particles: Array,
) -> Array:
    """Optional OT-style displacement cost; zero-weighted in the default objective."""
    return jnp.mean((transported_particles - prior_particles) ** 2)


def sample_mean_rmse(
    particles: Array,
    theta_true: Array,
    CFG: BayesTransportConfig,
) -> Array:
    particles_flat = flatten_particle_measure(particles, CFG)
    theta = theta_true
    if CFG.canonicalize_particle_sources and CFG.K > 1:
        theta = canonicalize_sources_jax(theta)
    return jnp.sqrt(jnp.mean((jnp.mean(particles_flat, axis=0) - theta.reshape(-1)) ** 2))


def sample_spread(particles: Array, CFG: BayesTransportConfig) -> Array:
    flat = flatten_particle_measure(particles, CFG)
    return jnp.mean(jnp.var(flat, axis=0))


def energy_score_single(
    particles: Array,
    theta_true: Array,
    CFG: BayesTransportConfig,
) -> Array:
    """Multivariate energy score for a posterior particle ensemble."""
    samples = flatten_particle_measure(particles, CFG)
    target = theta_true
    if CFG.canonicalize_particle_sources and CFG.K > 1:
        target = canonicalize_sources_jax(target)
    target = target.reshape(-1)
    truth_distance = jnp.mean(
        jnp.sqrt(jnp.sum((samples - target[None, :]) ** 2, axis=-1) + 1e-12)
    )
    differences = samples[:, None, :] - samples[None, :, :]
    squared_distance = jnp.sum(differences**2, axis=-1)
    off_diagonal = 1.0 - jnp.eye(samples.shape[0])
    pairwise_distance = jnp.sum(
        jnp.sqrt(squared_distance + 1e-12) * off_diagonal
    ) / (samples.shape[0] ** 2)
    return truth_distance - 0.5 * pairwise_distance


def leave_one_out_kde_log_density(
    particles: Array,
    CFG: BayesTransportConfig,
) -> Array:
    """Evaluate a differentiable leave-one-out KDE at every transported particle."""
    flat = flatten_particle_measure(particles, CFG)
    if flat.shape[0] < 2:
        raise ValueError("The ELBO KDE requires at least two posterior particles.")
    bandwidth = max(CFG.elbo_kde_bandwidth, 1e-6)
    dimension = flat.shape[-1]
    differences = flat[:, None, :] - flat[None, :, :]
    squared_distance = jnp.sum(differences**2, axis=-1)
    log_kernel = (
        -0.5 * squared_distance / bandwidth**2
        - dimension * math.log(bandwidth)
        - 0.5 * dimension * math.log(2.0 * math.pi)
    )
    log_kernel = jnp.where(
        jnp.eye(flat.shape[0], dtype=bool), -jnp.inf, log_kernel
    )
    return jax.nn.logsumexp(log_kernel, axis=1) - math.log(flat.shape[0] - 1)


def unnormalised_log_joint_particles(
    particles: Array,
    context_x: Array,
    context_y: Array,
    context_mask: Array,
    CFG: BayesTransportConfig,
) -> Array:
    """Return log p(theta, D) up to constants independent of transported particles."""
    prior_scale = max(CFG.prior_std, 1e-6)
    noise_scale = max(CFG.observation_noise_std, 1e-6)

    def one_particle(theta):
        log_prior = -0.5 * jnp.sum((theta / prior_scale) ** 2)
        mean = source_log_signal_jax(theta, context_x, CFG)
        residual = (context_y[:, 0] - mean) / noise_scale
        log_likelihood = -0.5 * jnp.sum(context_mask * residual**2)
        return log_prior + log_likelihood

    return jax.vmap(one_particle)(particles)


def negative_elbo_single(
    particles: Array,
    context_x: Array,
    context_y: Array,
    context_mask: Array,
    CFG: BayesTransportConfig,
) -> tuple[Array, Array, Array]:
    """Empirical negative ELBO using a particle KDE for q_phi(theta | D)."""
    log_q = leave_one_out_kde_log_density(particles, CFG)
    log_joint = unnormalised_log_joint_particles(
        particles, context_x, context_y, context_mask, CFG
    )
    negative_elbo = jnp.mean(log_q - log_joint)
    return negative_elbo, jnp.mean(log_q), jnp.mean(log_joint)


#%% 8) Plot helpers with explicit likelihood/posterior terminology
def _active_context(episode: dict[str, np.ndarray]):
    mask = np.asarray(episode["context_mask"]) > 0.5
    return np.asarray(episode["context_x"])[mask], np.asarray(episode["context_y"])[mask, 0]


def plot_transport_episode(
    model: BayesPushforwardTransformer,
    episode: dict[str, np.ndarray],
    CFG: BayesTransportConfig,
    destination: Path,
    title_prefix: str,
):
    prior_particles = np.asarray(episode["prior_particles"])
    target_particles = np.asarray(episode["target_particles"])
    predicted_particles = np.asarray(
        jax.device_get(
            model(
                jnp.asarray(prior_particles),
                jnp.asarray(episode["context_x"]),
                jnp.asarray(episode["context_y"]),
                jnp.asarray(episode["context_mask"]),
            )
        )
    )
    theta_true = np.asarray(episode["theta_true"])
    context_x, context_y = _active_context(episode)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5), constrained_layout=True)
    point_sets = [
        prior_particles.reshape(-1, 2),
        target_particles.reshape(-1, 2),
        predicted_particles.reshape(-1, 2),
        theta_true.reshape(-1, 2),
    ]
    if context_x.size:
        point_sets.append(context_x.reshape(-1, 2))
    all_points = np.concatenate(point_sets, axis=0)
    lim = max(
        3.0 * CFG.prior_std,
        1.15 * float(np.quantile(np.abs(all_points), 0.995)),
    )

    panels = [
        (prior_particles, "Prior samples  z_n ~ p(theta)"),
        (target_particles, "Teacher samples  theta~ ~ p_A(theta | D)"),
        (predicted_particles, "Transported samples  T_phi#p(theta | D)"),
    ]
    for ax, (particles, panel_title) in zip(axes[0], panels):
        ax.scatter(
            particles[..., 0].reshape(-1),
            particles[..., 1].reshape(-1),
            s=13,
            alpha=0.32,
            label=panel_title.split("  ")[0],
        )
        ax.scatter(
            theta_true[:, 0], theta_true[:, 1], marker="*", s=190,
            label="simulator parameter theta",
        )
        if context_x.size:
            ax.scatter(
                context_x[:, 0], context_x[:, 1], marker="x", s=55,
                label="designs in D",
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(panel_title)
        ax.legend(fontsize=8)

    overlay = axes[1, 0]
    overlay.scatter(
        target_particles[..., 0].reshape(-1),
        target_particles[..., 1].reshape(-1),
        s=16,
        alpha=0.25,
        label="teacher approximate posterior",
    )
    overlay.scatter(
        predicted_particles[..., 0].reshape(-1),
        predicted_particles[..., 1].reshape(-1),
        s=14,
        alpha=0.28,
        label="learned pushforward",
    )
    overlay.scatter(theta_true[:, 0], theta_true[:, 1], marker="*", s=190, label="theta")
    overlay.set_xlim(-lim, lim)
    overlay.set_ylim(-lim, lim)
    overlay.set_aspect("equal")
    overlay.grid(alpha=0.2)
    overlay.set_title("Approximate posterior p(theta | D): teacher versus transport")
    overlay.legend(fontsize=8)

    pair_ax = axes[1, 1]
    if context_x.size:
        scatter = pair_ax.scatter(
            context_x[:, 0], context_x[:, 1], c=context_y, s=90, marker="s"
        )
        fig.colorbar(scatter, ax=pair_ax, label="observed outcome y")
    pair_ax.scatter(theta_true[:, 0], theta_true[:, 1], marker="*", s=190, label="theta")
    pair_ax.set_xlim(CFG.design_low, CFG.design_high)
    pair_ax.set_ylim(CFG.design_low, CFG.design_high)
    pair_ax.set_aspect("equal")
    pair_ax.grid(alpha=0.2)
    pair_ax.set_title("Unordered conditioning set D = {(x_i, y_i)}")
    pair_ax.legend(fontsize=8)

    likelihood_ax = axes[1, 2]
    # if CFG.K >= 1:
    # Physically intuitive panel: expected sensor reading E[y | theta, x] as a
    # function of design location x, given the true source theta. This is the
    # forward model itself (not a likelihood-over-theta slice), so it directly
    # shows the "closer to theta -> higher reading" intuition and lets us check
    # observed (x_i, y_i) context points against the field they were drawn from.
    grid = np.linspace(CFG.design_low, CFG.design_high, CFG.grid_size)
    gx, gy = np.meshgrid(grid, grid)
    x_grid = np.stack([gx, gy], axis=-1)
    field = source_log_signal_np(theta_true, x_grid, CFG)  # log E[y | theta, x]

    vmin = min(field.min(), context_y.min()) if context_x.size else field.min()
    vmax = max(field.max(), context_y.max()) if context_x.size else field.max()

    contour = likelihood_ax.contourf(
        gx, gy, field, levels=30, cmap="magma", vmin=vmin, vmax=vmax
    )
    fig.colorbar(contour, ax=likelihood_ax, label="expected log outcome  log E[y | theta, x]")

    if context_x.size:
        likelihood_ax.scatter(
            context_x[:, 0], context_x[:, 1],
            c=context_y, cmap="magma", vmin=vmin, vmax=vmax,
            s=110, marker="s", edgecolors="white", linewidths=1.2,
            label="observed (x_i, y_i) in D",
        )
    likelihood_ax.scatter(
        theta_true[:, 0], theta_true[:, 1], marker="*", s=220,
        color="white", edgecolors="black", linewidths=0.8, label="theta",
    )
    likelihood_ax.set_xlim(CFG.design_low, CFG.design_high)
    likelihood_ax.set_ylim(CFG.design_low, CFG.design_high)
    likelihood_ax.set_aspect("equal")
    likelihood_ax.set_title("Sensor field around the true source")
    likelihood_ax.legend(fontsize=8, loc="upper right")
    # else:
    #     likelihood_ax.axis("off")
    #     likelihood_ax.text(
    #         0.5, 0.5,
    #         "K > 1: the sensor field is a superposition of K sources and\n"
    #         "is no longer a simple function of a single theta.",
    #         ha="center", va="center", wrap=True,
    #         transform=likelihood_ax.transAxes,
    #     )

    teacher_name = "SNIS" if int(episode["teacher_method_code"][0]) == 0 else "ABC"
    fig.suptitle(
        f"{title_prefix} | |D|={int(episode['context_size'][0])} | "
        f"teacher={teacher_name} | ESS={float(episode['teacher_ess'][0]):.1f}",
        fontsize=14,
    )
    fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 9) Training entry point


if __name__ == "__main__":

    np.random.seed(CFG.seed)
    print("JAX devices:", jax.devices())
    print("Resolved approximate-Bayes teacher:", resolve_teacher_method(CFG))
    if not CFG.use_proper_scoring_rule and not CFG.likelihood_available:
        raise ValueError(
            "The Stage-I ELBO requires an evaluable likelihood. "
            "Set likelihood_available=True."
        )

    run_dir = make_run_dir(CFG.env_name, CFG.runs_base)
    script_path = Path(globals().get("__file__", "stage_1_train_bayes_pushforward.py")).resolve()
    snapshot_files(run_dir, [script_path])
    save_config_yaml(
        CFG,
        run_dir / "config.yaml",
        extra={
            "training_complete": False,
            "resolved_teacher_method": resolve_teacher_method(CFG),
            "stage": 1,
        },
    )
    print("Stage-I run directory:", run_dir)
    print("Configuration:\n", yaml.safe_dump(asdict(CFG), sort_keys=False))

    prior = SourceLocPrior(K=CFG.K, prior_std=CFG.prior_std)
    generator = BayesTransportEpisodeGenerator(prior, CFG)
    train_loader = make_transport_train_loader(generator, CFG)
    eval_loader = make_transport_eval_loader(generator, CFG)
    steps_per_epoch = (
        len(train_loader) if CFG.data_mode == "finite" else CFG.steps_per_epoch
    )
    fixed_episode = eval_loader.dataset[0]
    np.savez_compressed(run_dir / "artefacts" / "fixed_episode.npz", **fixed_episode)

    model_key = jax.random.key(CFG.seed)
    model = BayesPushforwardTransformer(CFG, key=model_key)
    # eqx.tree_pprint(model)

    optimizer = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip_norm),
        optax.adamw(
            learning_rate=CFG.learning_rate,
            weight_decay=CFG.weight_decay,
        ),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def batch_objective(
        candidate_model: BayesPushforwardTransformer,
        batch: dict[str, Array],
        key: Array,
    ):
        direction_key = jax.random.fold_in(key, 17)
        directions = random_unit_directions(
            direction_key, CFG.num_swd_projections, 2 * CFG.K
        )
        predicted = jax.vmap(candidate_model)(
            batch["prior_particles"],
            batch["context_x"],
            batch["context_y"],
            batch["context_mask"],
        )
        energy_score = jax.vmap(
            lambda prediction, theta: energy_score_single(prediction, theta, CFG)
        )(predicted, batch["theta_true"])
        if CFG.use_proper_scoring_rule:
            negative_elbo = jnp.zeros_like(energy_score)
            mean_log_q = jnp.zeros_like(energy_score)
            mean_log_joint = jnp.zeros_like(energy_score)
        else:
            negative_elbo, mean_log_q, mean_log_joint = jax.vmap(
                lambda prediction, x, y, mask: negative_elbo_single(
                    prediction, x, y, mask, CFG
                )
            )(
                predicted,
                batch["context_x"],
                batch["context_y"],
                batch["context_mask"],
            )
        swd = jax.vmap(
            lambda prediction, target: sliced_wasserstein_squared(
                prediction, target, directions, CFG
            )
        )(predicted, batch["target_particles"])
        kinetic = jax.vmap(transport_kinetic_energy)(
            batch["prior_particles"], predicted
        )
        mean_rmse = jax.vmap(
            lambda particles, theta: sample_mean_rmse(particles, theta, CFG)
        )(predicted, batch["theta_true"])
        target_rmse = jax.vmap(
            lambda particles, theta: sample_mean_rmse(particles, theta, CFG)
        )(batch["target_particles"], batch["theta_true"])
        predicted_spread = jax.vmap(lambda p: sample_spread(p, CFG))(predicted)
        target_spread = jax.vmap(lambda p: sample_spread(p, CFG))(
            batch["target_particles"]
        )

        # At most two terms.  With the default kinetic_weight=0 this is exactly the
        # single negative-ELBO objective requested by the user.
        primary_objective = energy_score if CFG.use_proper_scoring_rule else negative_elbo
        loss = jnp.mean(primary_objective) + CFG.kinetic_weight * jnp.mean(kinetic)
        metrics = {
            "loss": loss,
            "energy_score": jnp.mean(energy_score),
            "negative_elbo": jnp.mean(negative_elbo),
            "mean_log_q": jnp.mean(mean_log_q),
            "mean_log_joint": jnp.mean(mean_log_joint),
            "sliced_wasserstein_sq": jnp.mean(swd),
            "kinetic_energy": jnp.mean(kinetic),
            "transport_mean_rmse": jnp.mean(mean_rmse),
            "teacher_mean_rmse": jnp.mean(target_rmse),
            "transport_spread": jnp.mean(predicted_spread),
            "teacher_spread": jnp.mean(target_spread),
            "teacher_ess": jnp.mean(batch["teacher_ess"]),
        }
        return loss, metrics

    @eqx.filter_jit
    def train_step(candidate_model, candidate_opt_state, batch, key):
        (_, metrics), grads = eqx.filter_value_and_grad(
            batch_objective, has_aux=True
        )(candidate_model, batch, key)
        params = eqx.filter(candidate_model, eqx.is_array)
        updates, candidate_opt_state = optimizer.update(
            grads, candidate_opt_state, params
        )
        candidate_model = eqx.apply_updates(candidate_model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return candidate_model, candidate_opt_state, metrics, grad_norm

    @eqx.filter_jit
    def eval_step(candidate_model, batch, key):
        _, metrics = batch_objective(candidate_model, batch, key)
        return metrics

    def evaluate(candidate_model, loader, evaluation_seed):
        collected: dict[str, list[float]] = {}
        root_key = jax.random.key(evaluation_seed)
        for batch_index, batch_np in enumerate(loader):
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            batch_key = jax.random.fold_in(root_key, batch_index)
            metrics = jax.device_get(eval_step(candidate_model, batch, batch_key))
            for name, value in metrics.items():
                collected.setdefault(name, []).append(float(value))
        return {name: float(np.mean(values)) for name, values in collected.items()}



    plot_transport_episode(
        model,
        fixed_episode,
        CFG,
        run_dir / "plots" / "fixed_episode_before_training.png",
        "Stage I before training",
    )
    initial_metrics = evaluate(model, eval_loader, CFG.seed + 100_000)
    print("Initial validation metrics:", initial_metrics)

    history: dict[str, list[float]] = {
        "step_loss": [],
        "step_energy_score": [],
        "step_negative_elbo": [],
        "step_mean_log_q": [],
        "step_mean_log_joint": [],
        "step_swd": [],
        "step_kinetic": [],
        "step_grad_norm": [],
        "epoch_train_loss": [],
        "epoch_val_loss": [],
        "epoch_val_energy_score": [],
        "epoch_val_negative_elbo": [],
        "epoch_val_mean_log_q": [],
        "epoch_val_mean_log_joint": [],
        "epoch_val_swd": [],
        "epoch_val_transport_mean_rmse": [],
        "epoch_val_teacher_mean_rmse": [],
        "epoch_val_teacher_ess": [],
    }
    visualisation_epochs = sorted(
        set(
            int(math.ceil(fraction * CFG.epochs / 10.0))
            for fraction in range(1, 11)
        )
    )
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    training_started_at = time.time()
    train_key = jax.random.key(CFG.seed + 1)
    train_start_time = time.time()

    for epoch in range(1, CFG.epochs + 1):
        epoch_started_at = time.time()
        train_losses_this_epoch: list[float] = []
        epoch_iterator = (
            iter(train_loader)
            if CFG.data_mode == "finite"
            else islice(iter(train_loader), CFG.steps_per_epoch)
        )
        progress = tqdm(
            enumerate(epoch_iterator, start=1),
            total=steps_per_epoch,
            desc=f"Bayes transport epoch {epoch:03d}/{CFG.epochs:03d}",
            dynamic_ncols=True,
            leave=True,
        )

        for _, batch_np in progress:
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            train_key, step_key = jax.random.split(train_key)
            model, opt_state, metrics, grad_norm = train_step(
                model, opt_state, batch, step_key
            )
            host = {name: float(value) for name, value in jax.device_get(metrics).items()}
            host_grad_norm = float(jax.device_get(grad_norm))
            global_step += 1
            train_losses_this_epoch.append(host["loss"])
            history["step_loss"].append(host["loss"])
            history["step_energy_score"].append(host["energy_score"])
            history["step_negative_elbo"].append(host["negative_elbo"])
            history["step_mean_log_q"].append(host["mean_log_q"])
            history["step_mean_log_joint"].append(host["mean_log_joint"])
            history["step_swd"].append(host["sliced_wasserstein_sq"])
            history["step_kinetic"].append(host["kinetic_energy"])
            history["step_grad_norm"].append(host_grad_norm)
            progress.set_postfix(
                loss=f"{host['loss']:.4f}",
                swd=f"{host['sliced_wasserstein_sq']:.4f}",
                ess=f"{host['teacher_ess']:.0f}",
            )

        epoch_train_loss = float(np.mean(train_losses_this_epoch))
        val_metrics = evaluate(model, eval_loader, CFG.seed + 100_000)
        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_val_loss"].append(val_metrics["loss"])
        history["epoch_val_energy_score"].append(val_metrics["energy_score"])
        history["epoch_val_negative_elbo"].append(val_metrics["negative_elbo"])
        history["epoch_val_mean_log_q"].append(val_metrics["mean_log_q"])
        history["epoch_val_mean_log_joint"].append(val_metrics["mean_log_joint"])
        history["epoch_val_swd"].append(val_metrics["sliced_wasserstein_sq"])
        history["epoch_val_transport_mean_rmse"].append(
            val_metrics["transport_mean_rmse"]
        )
        history["epoch_val_teacher_mean_rmse"].append(
            val_metrics["teacher_mean_rmse"]
        )
        history["epoch_val_teacher_ess"].append(val_metrics["teacher_ess"])

        save_transport_model(run_dir / "artefacts" / "model_last.eqx", model)
        eqx.tree_serialise_leaves(
            run_dir / "artefacts" / "training_state_last.eqx",
            (model, opt_state, jax.random.key_data(train_key)),
        )
        if epoch % CFG.save_every_epochs == 0:
            save_transport_model(
                run_dir / "artefacts" / f"model_epoch_{epoch:04d}.eqx", model
            )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_transport_model(run_dir / "artefacts" / "model_best.eqx", model)

        np.savez_compressed(
            run_dir / "artefacts" / "history.npz",
            **{name: np.asarray(values, dtype=np.float64) for name, values in history.items()},
        )
        save_json(
            run_dir / "artefacts" / "training_state.json",
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "elapsed_seconds": time.time() - training_started_at,
                "resolved_teacher_method": resolve_teacher_method(CFG),
            },
        )

        print(
            f"Epoch {epoch:03d}: train={epoch_train_loss:.6f} | "
            f"val={val_metrics['loss']:.6f} | "
            f"val SWD^2={val_metrics['sliced_wasserstein_sq']:.6f} | "
            f"transport RMSE={val_metrics['transport_mean_rmse']:.5f} | "
            f"teacher RMSE={val_metrics['teacher_mean_rmse']:.5f} | "
            f"ESS={val_metrics['teacher_ess']:.1f} | "
            f"{time.time() - epoch_started_at:.1f}s"
        )

        if epoch in visualisation_epochs:
            plot_transport_episode(
                model,
                fixed_episode,
                CFG,
                run_dir / "plots" / f"fixed_episode_epoch_{epoch:04d}.png",
                f"Stage I after epoch {epoch}",
            )
    print(f"Training complete in {datetime.timedelta(seconds=int(time.time() - training_started_at))} (hh:mm:ss)")

    best_model = load_transport_model(
        run_dir / "artefacts" / "model_best.eqx", CFG, key=jax.random.key(0)
    )
    final_metrics = evaluate(best_model, eval_loader, CFG.seed + 100_000)
    plot_transport_episode(
        best_model,
        fixed_episode,
        CFG,
        run_dir / "plots" / "fixed_episode_best_model.png",
        f"Stage I best model (epoch {best_epoch})",
    )

    #%%
    #%% Unified training diagnostics: per-step metrics (top) + per-epoch curves (bottom)

    steps = np.arange(1, len(history["step_loss"]) + 1)
    epochs = np.arange(1, len(history["epoch_train_loss"]) + 1)
    objective_label = (
        "energy score + penalty"
        if CFG.use_proper_scoring_rule
        else "negative ELBO + penalty"
    )

    plt.style.use("default")
    fig = plt.figure(figsize=(11.0, 10.0), constrained_layout=True)
    fig.suptitle("Stage-I Bayesian pushforward — training diagnostics", fontsize=14, fontweight="bold")

    # Top block: 2x2 grid of per-step metrics
    top_gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.0, 2.6])
    step_gs = top_gs[0:2].subgridspec(2, 2, hspace=0.35, wspace=0.28)

    step_panels = [
        ("step_loss", objective_label, "#1f77b4"),
        ("step_swd", "sliced $W_2^2$", "#d62728"),
        ("step_kinetic", "kinetic energy", "#2ca02c"),
        ("step_grad_norm", "gradient norm", "#9467bd"),
    ]

    for gs_cell, (key, ylabel, color) in zip(step_gs, step_panels):
        ax = fig.add_subplot(gs_cell)
        values = np.asarray(history[key])
        ax.plot(steps, values, color=color, linewidth=0.8, alpha=0.85)
        if len(values) >= 20:
            window = max(5, len(values) // 100)
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(
                steps[window - 1:], smoothed,
                color=color, linewidth=1.8, alpha=1.0,
                label=f"moving avg ({window})",
            )
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        ax.set_title(ylabel, fontsize=10, fontweight="bold", loc="left")
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

        ## Set symetric log scale for SWD and kinetic energy, if applicable
        # if key in ["step_swd", "step_kinetic"]:
        ax.set_yscale("symlog", linthresh=1e-6)

    # Bottom block: per-epoch train/val loss + SWD
    ax_bottom = fig.add_subplot(top_gs[2])
    ax_bottom.plot(epochs, history["epoch_train_loss"], marker="o", markersize=3,
                color="#1f77b4", label="train loss")
    ax_bottom.plot(epochs, history["epoch_val_loss"], marker="o", markersize=3,
                color="#ff7f0e", label="val loss")
    ax_bottom.axvline(best_epoch, color="grey", linestyle="--", linewidth=1.0, alpha=0.7,
                    label=f"best epoch ({best_epoch})")
    ax_bottom.set_xlabel("epoch", fontsize=10)
    ax_bottom.set_ylabel(objective_label, fontsize=10)
    ax_bottom.set_title("Per-epoch train / validation objective", fontsize=10, fontweight="bold", loc="left")
    ax_bottom.grid(alpha=0.25)
    ax_bottom.spines[["top", "right"]].set_visible(False)
    ax_bottom.tick_params(labelsize=9)

    ax_bottom2 = ax_bottom.twinx()
    ax_bottom2.plot(epochs, history["epoch_val_swd"], marker="s", markersize=3,
                    color="#2ca02c", linestyle=":", label="val sliced $W_2^2$")
    ax_bottom2.set_ylabel("validation sliced $W_2^2$", fontsize=10, color="#2ca02c")
    ax_bottom2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=9)
    ax_bottom2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax_bottom.get_legend_handles_labels()
    lines2, labels2 = ax_bottom2.get_legend_handles_labels()
    ax_bottom.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right", frameon=False)

    plt.tight_layout()

    fig.savefig(run_dir / "plots" / "training_diagnostics.png", dpi=170)
    display(fig)
    plt.close(fig)

    print("Best epoch:", best_epoch)
    print("Final validation metrics:", final_metrics)
    print("Saved Stage-I artefacts under:", run_dir)


