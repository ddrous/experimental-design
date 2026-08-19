#%% 1) Imports, configuration, and experiment description
"""Single-cloud particle transport trained with an empirical energy score.

This is a stripped-down rewrite of the original heterogeneous-shape / multi-observation
notebook.  It keeps ONLY the code path selected by the fixed hyperparameters actually used:

  * ONE fixed problem shape:            num_sources S=2, source_dim D=3 (no padding, no
                                         dimension-agnostic TAMO embedders, no held-out shapes).
  * ONE observation per evaluation step: Omin=Omax=1, so there is no causal likelihood
                                         Transformer and no vmap over observation prefixes.
  * ONE base prior:                     Uniform on the physical design box [low, high]^D.
  * ONE synthetic-truth rule:           the default "closed_form" interpolated-prior law.
  * ONE posterior conditioning:         AdaLN-Zero modulation (no cross-attention branch).

Training algorithm
-------------------
For each fresh iid training row:
  1. Draw N iid particles z_1..z_N ~ rho_0 (uniform base prior), one synthetic "potential
     truth" theta_tilde ~ rho_0, and an interpolation time tau ~ Uniform[0,1].
  2. Form the synthetic input cloud C_tau = (1-tau) z_n + tau theta_tilde.
  3. Draw the actual truth theta* from the exact conditional law of C_tau given
     (theta_tilde, tau) -- i.e. theta* and the cloud are iid draws from the SAME
     interpolated prior.  tau=0 recovers an ordinary base-prior cloud; tau near 1
     concentrates the cloud near theta_tilde.
  4. Draw one design/outcome observation from the physical likelihood conditional on theta*.
  5. Pass the single normalized observation and the (canonicalized, flattened) input cloud
     directly to the Posterior Transformer -- there is no learned input-embedding step and no
     posterior recurrence in the training graph.
  6. Score the transported cloud against theta* with the exact empirical multivariate energy
     score, average over the batch, and take one AdamW step.

For one output cloud Q_hat = N^-1 sum_n delta_{theta_n}, the optimized score is

    ES(Q_hat, theta*) = N^-1 sum_n ||theta_n - theta*|| - (2N^2)^-1 sum_{n,m} ||theta_n - theta_m||.

Sequential evaluation
----------------------
Repeated-Bayes behaviour is retained as an EVALUATION-ONLY stress test: starting from a fresh
tau=0 (ordinary base-prior) cloud, the SAME learned transport is applied repeatedly, once per
fresh single-observation step, via jax.lax.scan.  This never contributes a training gradient.
It is compared against an SNIS reference posterior built from the true (known) likelihood, as
a soundness check that repeated application of the learned map tracks real Bayesian updating.

Array shapes
------------
B : batch size            N : particles per cloud        S=2, D=3 : fixed problem shape
K = S*D = 6 : flattened physical theta width              obs width = D+1 = 4

theta_true        [B,S,D]
observation       [B,D+1]           (single design + outcome)
prior_particles   [B,N,S,D]
posterior_theta   [B,N,S,D]         (training output, one prefix only)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
import json
import math
import shutil
from pathlib import Path
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm
import yaml

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

Array = jax.Array

# Execution switch: True trains a fresh model into a new run folder; False reloads the best
# checkpoint from the current run folder (cwd) and only regenerates the core diagnostic plots.
train_wm: bool = True

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})


@dataclass(frozen=True)
class BayesTransportConfig:
    """Every field here is on the active code path; nothing is a dead switch."""

    env_name: str = "parallel"
    seed: int = 2030
    runs_base: str = "./runs"

    # Fixed source-localisation problem shape.
    num_sources: int = 2
    source_dim: int = 3

    # ONE base prior: uniform on the physical design box.
    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.30

    # Particle-cloud / training-stream sizes.
    num_particles: int = 64
    n_train_trajectories: int = 4096
    n_eval_trajectories: int = 256
    batch_size: int = 16 * 16
    train_dataloader_num_workers: int = 0
    train_dataloader_prefetch_factor: int = 2

    # Evaluation-only sequential rollout horizon.
    evaluation_trajectory_length: int = 16

    # Posterior Transformer (AdaLN-conditioned particle self-attention).
    hidden_dim: int = 256
    heads: int = 8
    mlp_ratio: int = 4
    posterior_depth: int = 6
    max_theta_displacement: float = 6.0
    canonicalize_particle_sources: bool = True

    # Observation normalisation.
    y_center: float = 0.0
    y_scale: float = 3.0

    # Optimisation.
    epochs: int = 2000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1000.0
    lr_plateau_patience: int = 50
    lr_plateau_rtol: float = 1e-4

    # Persistence / visualisation cadence.
    save_every_epochs: int = 500

    # SNIS reference posterior, used only for the post-hoc soundness check plot.
    reference_proposals: int = 10_000
    reference_particles: int = 2_000


CFG = BayesTransportConfig()
S, D, K = CFG.num_sources, CFG.source_dim, CFG.num_sources * CFG.source_dim
OBS_DIM = D + 1

#%% 2) Run directories and small persistence helpers
def make_run_dir(env_name: str, base: str | Path = "./runs") -> Path:
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = Path(base).expanduser().resolve() / f"{env_name}_{stamp}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=False)
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_running_script_to_run_dir(run_dir: Path) -> Path | None:
    """Archive the exact script into the run folder, when running as a real .py file."""
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


def save_model(path: str | Path, model: "SequentialBayesModel"):
    eqx.tree_serialise_leaves(Path(path), model)


def load_model(path: str | Path, cfg: BayesTransportConfig, *, key: Array | None = None) -> "SequentialBayesModel":
    skeleton = SequentialBayesModel(cfg, key=jax.random.key(0) if key is None else key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)


#%% 3) Base prior, synthetic interpolated training prior, and source-localisation simulator
def sample_base_prior_np(rng: np.random.Generator, n: int, cfg: BayesTransportConfig = CFG) -> np.ndarray:
    """Draw n iid full-theta [S,D] samples from the uniform base prior rho_0."""
    return rng.uniform(cfg.design_low, cfg.design_high, size=(int(n), S, D)).astype(np.float32)


def sample_interpolated_prior_and_truth_np(
    rng: np.random.Generator, n_particles: int, cfg: BayesTransportConfig = CFG
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one synthetic training-prior cloud C_tau and the closed-form truth theta*.

    z_n ~ rho_0, theta_tilde ~ rho_0, tau ~ Uniform[0,1];  C_tau = (1-tau) z_n + tau theta_tilde.
    theta* is then an independent draw from the EXACT population law realised by C_tau:
    coordinatewise Uniform on [(1-tau)*low + tau*theta_tilde, (1-tau)*high + tau*theta_tilde].
    """
    base_particles = sample_base_prior_np(rng, n_particles, cfg)
    potential_theta = sample_base_prior_np(rng, 1, cfg)[0]
    tau = np.float32(rng.uniform(0.0, 1.0))
    one_minus_tau = np.float32(1.0) - tau

    prior_particles = (one_minus_tau * base_particles + tau * potential_theta[None]).astype(np.float32)

    low = one_minus_tau * np.float32(cfg.design_low) + tau * potential_theta
    high = one_minus_tau * np.float32(cfg.design_high) + tau * potential_theta
    if np.all(high == low):
        theta_true = np.asarray(low, dtype=np.float32).copy()
    else:
        theta_true = rng.uniform(
            low=np.asarray(low, dtype=np.float64), high=np.asarray(high, dtype=np.float64), size=(S, D)
        ).astype(np.float32)
    return prior_particles, theta_true


def source_log_mean_np(theta: np.ndarray, designs: np.ndarray, cfg: BayesTransportConfig = CFG) -> np.ndarray:
    """Forward-model mean E[y | theta, x] on the log-intensity scale.

    theta [...,S,D], designs [...,T,D] -> [...,T]  (dimension-generic broadcasting).
    """
    theta = np.asarray(theta, dtype=np.float64)
    designs = np.asarray(designs, dtype=np.float64)
    dist_sq = np.sum((np.expand_dims(theta, -3) - np.expand_dims(designs, -2)) ** 2, axis=-1)
    intensity = cfg.background + np.sum(cfg.source_strength / (cfg.softening + dist_sq), axis=-1)
    return np.log(intensity)


def simulate_observation_np(
    rng: np.random.Generator, theta_true: np.ndarray, cfg: BayesTransportConfig = CFG
) -> np.ndarray:
    """Draw one [D+1] (design, outcome) pair from the physical likelihood given theta_true."""
    design = rng.uniform(cfg.design_low, cfg.design_high, size=(D,)).astype(np.float32)
    mean = source_log_mean_np(theta_true, design[None, :], cfg)[0]
    reading = np.float32(mean + cfg.observation_noise_std * rng.normal())
    return np.concatenate([design, [reading]]).astype(np.float32)


def simulate_trajectories(
    rng: np.random.Generator, n_trajectories: int, trajectory_length: int, cfg: BayesTransportConfig = CFG
) -> dict[str, np.ndarray]:
    """Evaluation-only trajectories: one fresh theta* ~ rho_0, reused for T iid observations."""
    n_trajectories, trajectory_length = int(n_trajectories), int(trajectory_length)
    theta_true = sample_base_prior_np(rng, n_trajectories, cfg)
    observations = np.zeros((n_trajectories, trajectory_length, OBS_DIM), dtype=np.float32)
    for m in range(n_trajectories):
        for t in range(trajectory_length):
            observations[m, t] = simulate_observation_np(rng, theta_true[m], cfg)
    return {"theta_true": theta_true, "observations": observations}


def simulate_iid_joint_samples(
    rng: np.random.Generator, n_samples: int, cfg: BayesTransportConfig = CFG
) -> dict[str, np.ndarray]:
    """Fixed, reproducible iid (C_tau, theta*, y) triples used for amortized validation."""
    n_samples = int(n_samples)
    theta_true = np.zeros((n_samples, S, D), dtype=np.float32)
    observation = np.zeros((n_samples, OBS_DIM), dtype=np.float32)
    prior_particles = np.zeros((n_samples, cfg.num_particles, S, D), dtype=np.float32)
    for m in range(n_samples):
        active_prior, theta_active = sample_interpolated_prior_and_truth_np(rng, cfg.num_particles, cfg)
        theta_true[m] = theta_active
        observation[m] = simulate_observation_np(rng, theta_active, cfg)
        prior_particles[m] = active_prior
    return {"theta_true": theta_true, "observation": observation, "prior_particles": prior_particles}


class ContinuousJointDataset(IterableDataset):
    """Infinite CPU stream of fresh iid (C_tau, theta*, y) training rows."""

    def __init__(self, cfg: BayesTransportConfig, *, seed: int):
        super().__init__()
        self.cfg = cfg
        self.seed = int(seed)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rng = np.random.default_rng(self.seed + 1_000_003 * worker_id)
        while True:
            prior_particles, theta_true = sample_interpolated_prior_and_truth_np(
                rng, self.cfg.num_particles, self.cfg
            )
            observation = simulate_observation_np(rng, theta_true, self.cfg)
            yield {
                "theta_true": theta_true,
                "observation": observation,
                "prior_particles": prior_particles,
            }


def _numpy_collate(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack([np.asarray(s[name]) for s in samples], axis=0) for name in samples[0]}


def make_continuous_train_loader(cfg: BayesTransportConfig = CFG, *, seed: int) -> DataLoader:
    dataset = ContinuousJointDataset(cfg, seed=seed)
    kwargs: dict[str, Any] = dict(
        dataset=dataset, batch_size=cfg.batch_size, num_workers=cfg.train_dataloader_num_workers,
        collate_fn=_numpy_collate, drop_last=True,
    )
    if cfg.train_dataloader_num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.train_dataloader_prefetch_factor
    return DataLoader(**kwargs)


#%% 4) Source-label symmetry and small array/token helpers
def canonicalize_sources_np(theta: np.ndarray) -> np.ndarray:
    """Sort exchangeable sources by their first coordinate (removes label symmetry)."""
    order = np.argsort(theta[..., 0], axis=-1)
    return np.take_along_axis(theta, order[..., None], axis=-2)


def canonicalize_sources_jax(theta: Array) -> Array:
    order = jnp.argsort(theta[..., 0], axis=-1)
    return jnp.take_along_axis(theta, order[..., None], axis=-2)


def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


#%% 5) AdaLN particle-conditioning block and the Posterior Transformer
class AdaLNParticleBlock(eqx.Module):
    """Particle self-attention conditioned on the single observation via AdaLN-Zero.

    The observation never appears as an attention key/value; it only produces shift/scale/gate
    vectors that modulate the particle residual stream. Zero-initialized modulation means the
    block starts as an (untrained) identity-preserving transport.
    """

    self_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(self, hidden_dim: int, conditioning_dim: int, heads: int, mlp_dim: int, *, key: Array):
        self_key, ff1_key, ff2_key, mod_key = jax.random.split(key, 4)
        self.self_norm = eqx.nn.LayerNorm(hidden_dim)
        self.ff_norm = eqx.nn.LayerNorm(hidden_dim)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=hidden_dim, key_size=hidden_dim, value_size=hidden_dim,
            output_size=hidden_dim, dropout_p=0.0, key=self_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)
        modulation = eqx.nn.Linear(conditioning_dim, 6 * hidden_dim, key=mod_key)
        modulation = eqx.tree_at(lambda l: l.weight, modulation, jnp.zeros_like(modulation.weight))
        modulation = eqx.tree_at(lambda l: l.bias, modulation, jnp.zeros_like(modulation.bias))
        self.modulation = modulation

    def __call__(self, particles: Array, conditioning: Array) -> Array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(
            self.modulation(jax.nn.silu(conditioning)), 6, axis=-1
        )
        h = _modulate(_layernorm_tokens(self.self_norm, particles), shift_a, scale_a)
        particles = particles + gate_a[None, :] * self.self_attention(h, h, h)
        h = _modulate(_layernorm_tokens(self.ff_norm, particles), shift_f, scale_f)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + gate_f[None, :] * h


class ThetaParticleOutputHead(eqx.Module):
    """Map [N,hidden] posterior tokens to a bounded displacement of the physical theta cloud."""

    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear
    max_displacement: float = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.max_displacement = cfg.max_theta_displacement
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)
        head = eqx.nn.Linear(cfg.hidden_dim, K, key=key)
        head = eqx.tree_at(lambda l: l.weight, head, jnp.zeros_like(head.weight))
        head = eqx.tree_at(lambda l: l.bias, head, jnp.zeros_like(head.bias))
        self.displacement_head = head

    def __call__(self, particle_tokens: Array, current_theta_flat: Array) -> Array:
        particle_tokens = _layernorm_tokens(self.final_norm, particle_tokens)
        displacement = self.max_displacement * jnp.tanh(_linear_tokens(self.displacement_head, particle_tokens))
        return current_theta_flat + displacement


class AdaLNPosteriorTransformer(eqx.Module):
    """Direct reference-cloud -> posterior transport, AdaLN-conditioned on one observation."""

    particle_in: eqx.nn.Linear
    blocks: tuple[AdaLNParticleBlock, ...]
    output_head: ThetaParticleOutputHead

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 2)
        self.particle_in = eqx.nn.Linear(K, cfg.hidden_dim, key=keys[0])
        self.blocks = tuple(
            AdaLNParticleBlock(cfg.hidden_dim, OBS_DIM, cfg.heads, cfg.mlp_ratio * cfg.hidden_dim, key=keys[1 + i])
            for i in range(cfg.posterior_depth)
        )
        self.output_head = ThetaParticleOutputHead(cfg, key=keys[-1])

    def __call__(self, particle_embeddings: Array, current_theta_flat: Array, observation: Array) -> Array:
        particles = _linear_tokens(self.particle_in, particle_embeddings)
        for block in self.blocks:
            particles = block(particles, observation)
        return self.output_head(particles, current_theta_flat)


#%% 6) End-to-end amortized model with an evaluation-only sequential rollout
class SequentialBayesModel(eqx.Module):
    """`predict` is the ONLY path used by the training loss (one prior -> one posterior cloud
    given one observation). `__call__` retains the evaluation-only sequential interface: it
    repeatedly applies the same learned transport to a running cloud, once per fresh
    observation, via jax.lax.scan. No gradient step ever trains through that recurrence.
    """

    posterior_transformer: AdaLNPosteriorTransformer
    theta_center: float = eqx.field(static=True)
    theta_scale: float = eqx.field(static=True)
    design_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.posterior_transformer = AdaLNPosteriorTransformer(cfg, key=key)
        self.theta_center = 0.5 * (cfg.design_low + cfg.design_high)
        self.theta_scale = max(0.5 * (cfg.design_high - cfg.design_low), 1e-6)
        self.design_scale = max(abs(cfg.design_low), abs(cfg.design_high), 1.0)
        self.y_center = cfg.y_center
        self.y_scale = max(cfg.y_scale, 1e-6)
        self.canonicalize = cfg.canonicalize_particle_sources

    def _flatten(self, theta: Array) -> Array:
        """[...,S,D] physical theta -> [...,K], canonicalizing source labels first."""
        if self.canonicalize:
            theta = canonicalize_sources_jax(theta)
        return theta.reshape(theta.shape[:-2] + (K,))

    def _normalize_particles(self, theta_flat: Array) -> Array:
        return (theta_flat - self.theta_center) / self.theta_scale

    def _normalize_observation(self, observation: Array) -> Array:
        return jnp.concatenate(
            [observation[:D] / self.design_scale, (observation[-1:] - self.y_center) / self.y_scale]
        )

    def predict(self, prior_particles: Array, observation: Array) -> Array:
        """Training/amortized path: one synthetic cloud + one observation -> one posterior cloud.

        prior_particles [N,S,D], observation [D+1] -> posterior_theta [N,S,D].
        """
        prior_flat = self._flatten(prior_particles)                     # [N,K]
        embeddings = self._normalize_particles(prior_flat)               # [N,K]
        obs = self._normalize_observation(observation)                   # [D+1]
        posterior_flat = self.posterior_transformer(embeddings, prior_flat, obs)
        posterior_flat = self._flatten(posterior_flat.reshape(-1, S, D))  # re-canonicalize output
        return posterior_flat.reshape(-1, S, D)

    def __call__(self, prior_particles: Array, observations: Array) -> Array:
        """Evaluation-only rollout: repeatedly apply the SAME learned map, one fresh
        observation per step. observations [T,D+1] -> posterior_sequence [T,N,S,D].
        """
        prior_flat = self._flatten(prior_particles)

        def scan_step(current_flat: Array, observation: Array):
            embeddings = self._normalize_particles(current_flat)
            obs = self._normalize_observation(observation)
            next_flat = self.posterior_transformer(embeddings, current_flat, obs)
            next_flat = self._flatten(next_flat.reshape(-1, S, D))
            return next_flat, next_flat

        _, sequence_flat = jax.lax.scan(scan_step, prior_flat, observations)
        return sequence_flat.reshape(observations.shape[0], -1, S, D)


def count_parameters(module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array)))


def print_model_parameter_count(model: SequentialBayesModel):
    total = count_parameters(model.posterior_transformer)
    print(f"Total parameters: {total / 1e6:.3f} M  (Posterior Transformer, AdaLN-conditioned)")


#%% 7) Empirical-cloud energy score and physical posterior diagnostics
def empirical_energy_score_terms_single(particle_theta: Array, target_theta: Array) -> tuple[Array, Array, Array]:
    """Exact multivariate energy score of one transported empirical particle measure.

        ES(Q_hat, theta*) = N^-1 sum_n ||theta_n-theta*|| - (2N^2)^-1 sum_{n,m} ||theta_n-theta_m||.

    particle_theta [N,K], target_theta [K].
    """
    target_sq = jnp.sum((particle_theta - target_theta[None, :]) ** 2, axis=-1)
    attraction = jnp.mean(jnp.sqrt(target_sq + 1e-12))

    differences = particle_theta[:, None, :] - particle_theta[None, :, :]
    pair_sq = jnp.sum(differences**2, axis=-1)
    off_diagonal = 1.0 - jnp.eye(particle_theta.shape[0], dtype=particle_theta.dtype)
    repulsion = jnp.sum(jnp.sqrt(pair_sq + 1e-12) * off_diagonal) / (particle_theta.shape[0] ** 2)
    energy_score = attraction - 0.5 * repulsion
    return energy_score, attraction, repulsion


def energy_score_single(particle_theta: Array, target_theta: Array) -> Array:
    energy_score, _, _ = empirical_energy_score_terms_single(particle_theta, target_theta)
    return energy_score


def posterior_mean_rmse_single(particle_theta: Array, target_theta: Array) -> Array:
    squared_error = (jnp.mean(particle_theta, axis=0) - target_theta) ** 2
    return jnp.sqrt(jnp.mean(squared_error))


def posterior_spread_single(particle_theta: Array) -> Array:
    return jnp.mean(jnp.var(particle_theta, axis=0))


def _flatten_theta(theta: Array) -> Array:
    """[...,S,D] -> [...,K], canonicalized (matches the model's own convention)."""
    theta = canonicalize_sources_jax(theta) if CFG.canonicalize_particle_sources else theta
    return theta.reshape(theta.shape[:-2] + (K,))


def batch_objective(model: SequentialBayesModel, batch: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
    """Non-sequential iid empirical-energy-score training objective (one prefix, one prior cloud)."""
    predicted = jax.vmap(model.predict)(batch["prior_particles"], batch["observation"])   # [B,N,S,D]
    predicted_flat = jax.vmap(_flatten_theta)(predicted)                                   # [B,N,K]
    target_flat = jax.vmap(_flatten_theta)(batch["theta_true"])                            # [B,K]

    energy, attraction, repulsion = jax.vmap(empirical_energy_score_terms_single)(predicted_flat, target_flat)
    rmse = jax.vmap(posterior_mean_rmse_single)(predicted_flat, target_flat)
    spread = jax.vmap(posterior_spread_single)(predicted_flat)

    loss = jnp.mean(energy)
    metrics = {
        "loss": loss,
        "energy_score": loss,
        "posterior_mean_rmse": jnp.mean(rmse),
        "posterior_spread": jnp.mean(spread),
        "attraction": jnp.mean(attraction),
        "repulsion": jnp.mean(repulsion),
    }
    return loss, metrics


@eqx.filter_jit
def predict_batch(model: SequentialBayesModel, prior_particles: Array, observation: Array) -> Array:
    return jax.vmap(model.predict)(prior_particles, observation)


@eqx.filter_jit
def sequential_predict_batch(model: SequentialBayesModel, prior_particles: Array, observations: Array) -> Array:
    return jax.vmap(model)(prior_particles, observations)


@eqx.filter_jit
def amortized_evaluation_batch(model: SequentialBayesModel, batch: dict[str, Array]) -> dict[str, Array]:
    _, metrics = batch_objective(model, batch)
    return metrics


@eqx.filter_jit
def sequential_evaluation_batch(model: SequentialBayesModel, batch: dict[str, Array]) -> dict[str, Array]:
    """Evaluate the repeated-Bayes recurrence only; never differentiated during training."""
    predicted = sequential_predict_batch(model, batch["prior_particles"], batch["observations"])  # [B,T,N,S,D]
    predicted_flat = jax.vmap(jax.vmap(_flatten_theta))(predicted)                                 # [B,T,N,K]
    target_flat = jax.vmap(_flatten_theta)(batch["theta_true"])                                     # [B,K]

    energy = jax.vmap(lambda seq, target: jax.vmap(lambda p: energy_score_single(p, target))(seq))(
        predicted_flat, target_flat
    )
    rmse = jax.vmap(lambda seq, target: jax.vmap(lambda p: posterior_mean_rmse_single(p, target))(seq))(
        predicted_flat, target_flat
    )
    return {
        "energy_score": jnp.mean(energy),
        "final_energy_score": jnp.mean(energy[:, -1]),
        "posterior_mean_rmse": jnp.mean(rmse),
        "final_mean_rmse": jnp.mean(rmse[:, -1]),
        "energy_by_t": jnp.mean(energy, axis=0),
        "rmse_by_t": jnp.mean(rmse, axis=0),
    }


#%% 8) Amortized validation and sequential evaluation with reproducible data
def evaluate_amortized_model(
    model: SequentialBayesModel, dataset: dict[str, np.ndarray], cfg: BayesTransportConfig = CFG,
    *, batch_size: int | None = None,
) -> dict[str, float]:
    n_total = len(dataset["theta_true"])
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    names = ["loss", "energy_score", "posterior_mean_rmse", "posterior_spread", "attraction", "repulsion"]
    values = {name: [] for name in names}
    weights = []
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = {name: jnp.asarray(dataset[name][start:stop]) for name in ("theta_true", "observation", "prior_particles")}
        metrics = jax.device_get(amortized_evaluation_batch(model, batch))
        weights.append(stop - start)
        for name in names:
            values[name].append(float(metrics[name]))
    weight_array = np.asarray(weights, dtype=np.float64)
    return {name: float(np.average(values[name], weights=weight_array)) for name in names}


def evaluate_sequential_model(
    model: SequentialBayesModel, dataset: dict[str, np.ndarray], cfg: BayesTransportConfig = CFG,
    *, batch_size: int | None = None,
) -> dict[str, np.ndarray | float]:
    n_total = len(dataset["theta_true"])
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    scalar_names = ["energy_score", "final_energy_score", "posterior_mean_rmse", "final_mean_rmse"]
    scalar_values = {name: [] for name in scalar_names}
    by_t_values = {name: [] for name in ["energy_by_t", "rmse_by_t"]}
    weights = []
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = {
            "theta_true": jnp.asarray(dataset["theta_true"][start:stop]),
            "observations": jnp.asarray(dataset["observations"][start:stop]),
            "prior_particles": jnp.asarray(dataset["prior_particles"][start:stop]),
        }
        metrics = jax.device_get(sequential_evaluation_batch(model, batch))
        weights.append(stop - start)
        for name in scalar_names:
            scalar_values[name].append(float(metrics[name]))
        for name in by_t_values:
            by_t_values[name].append(np.asarray(metrics[name], dtype=np.float64))
    weight_array = np.asarray(weights, dtype=np.float64)
    result: dict[str, np.ndarray | float] = {
        name: float(np.average(scalar_values[name], weights=weight_array)) for name in scalar_names
    }
    for name in by_t_values:
        result[name] = np.average(np.stack(by_t_values[name]), axis=0, weights=weight_array)
    return result


#%% 9) Optional exact-likelihood SNIS reference posterior (soundness check only)
def reference_posterior_particles_np(
    rng: np.random.Generator, observations: np.ndarray, cfg: BayesTransportConfig = CFG
) -> tuple[np.ndarray, float]:
    """SNIS reference posterior from the TRUE likelihood; used only for the post-hoc plot.

    Proposal is the base prior rho_0, so importance weights are exactly the likelihood of the
    observed prefix. This is never called inside the training objective.
    """
    proposals = sample_base_prior_np(rng, cfg.reference_proposals, cfg)             # [P,S,D]
    designs = observations[:, :D]
    readings = observations[:, -1]
    predicted_means = source_log_mean_np(proposals, designs, cfg)                    # [P,t]
    residual = (readings[None, :] - predicted_means) / cfg.observation_noise_std
    log_weights = -0.5 * np.sum(residual**2, axis=1)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.maximum(weights.sum(), 1e-300)
    ess = float(1.0 / np.sum(weights**2))
    indices = rng.choice(len(proposals), size=cfg.reference_particles, replace=True, p=weights)
    posterior = proposals[indices]
    if cfg.canonicalize_particle_sources:
        posterior = canonicalize_sources_np(posterior)
    return posterior.astype(np.float32), ess


#%% 10) Visualisation: sequential posterior evolution during training
def _project2d(theta: np.ndarray, dims: tuple[int, int] = (0, 1)) -> np.ndarray:
    """Project [...,S,D] physical theta onto two coordinate axes for 2-D plotting."""
    return theta[..., list(dims)]


def select_steps(trajectory_length: int, n_panels_after_prior: int = 5) -> list[int]:
    values = np.unique(np.rint(np.geomspace(1, trajectory_length, n_panels_after_prior)).astype(int))
    if values[-1] != trajectory_length:
        values = np.append(values, trajectory_length)
    while len(values) > n_panels_after_prior:
        values = np.delete(values, 1)
    return values.tolist()


def plot_posterior_evolution(
    model: SequentialBayesModel, trajectory: dict[str, np.ndarray], prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG, destination: Path | None = None, title: str = "Posterior evolution",
    plot_dims: tuple[int, int] = (0, 1),
):
    """Plot the evaluation-only repeated-Bayes rollout on a fixed problem, projected to 2-D.

    D=3 here, so this shows the (dims[0], dims[1]) coordinate plane; the third coordinate is
    marginalised out for plotting only, exactly as the third panel of a corner plot would.
    """
    observations = np.asarray(trajectory["observations"])
    posterior_sequence = np.asarray(jax.device_get(
        sequential_predict_batch(model, jnp.asarray(prior_particles)[None], jnp.asarray(observations)[None])[0]
    ))  # [T,N,S,D]
    theta_true = np.asarray(trajectory["theta_true"])
    if cfg.canonicalize_particle_sources:
        theta_true = canonicalize_sources_np(theta_true)

    steps = select_steps(len(observations), 5)
    clouds = [prior_particles] + [posterior_sequence[t - 1] for t in steps]
    labels = ["physical prior"] + [f"q_phi(theta | steps 1:{t})" for t in steps]

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.5), constrained_layout=True)
    axes = axes.ravel()
    all_points = np.concatenate(
        [_project2d(c, plot_dims).reshape(-1, 2) for c in clouds] + [_project2d(theta_true, plot_dims).reshape(-1, 2)]
    )
    lim = max(abs(cfg.design_low), abs(cfg.design_high), 1.12 * float(np.quantile(np.abs(all_points), 0.995)))

    for panel_index, (ax, cloud, label) in enumerate(zip(axes, clouds, labels)):
        p = _project2d(cloud, plot_dims)
        ax.scatter(p[..., 0].reshape(-1), p[..., 1].reshape(-1), s=13, alpha=0.30,
                   label="posterior source locations" if panel_index else "prior source locations")
        truth_p = _project2d(theta_true, plot_dims)
        ax.scatter(truth_p[:, 0], truth_p[:, 1], marker="*", s=190, edgecolors="black", linewidths=0.8, label="theta*")
        if panel_index > 0:
            t = steps[panel_index - 1]
            designs = observations[:t, :D]
            d = designs[:, list(plot_dims)]
            ax.scatter(d[:, 0], d[:, 1], marker="x", s=33, alpha=0.65, label="designs seen")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_title(label); ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"{title}  (projected onto coords {plot_dims})", fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


def plot_sequential_reference_check(
    model: SequentialBayesModel, trajectory: dict[str, np.ndarray], prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG, destination: Path | None = None, seed: int | None = None,
):
    """Soundness check: does the repeated learned Bayes update track a true-likelihood SNIS
    reference posterior as evidence accumulates?"""
    observations = np.asarray(trajectory["observations"])
    theta_true = np.asarray(trajectory["theta_true"])
    posterior_sequence = np.asarray(jax.device_get(
        sequential_predict_batch(model, jnp.asarray(prior_particles)[None], jnp.asarray(observations)[None])[0]
    )).reshape(len(observations), -1, K)

    rng = np.random.default_rng(cfg.seed + 44_000 if seed is None else seed)
    truth_flat = _flatten_theta(jnp.asarray(theta_true))
    truth_flat = np.asarray(jax.device_get(truth_flat))

    learned_rmse, reference_rmse, ess_values = [], [], []
    for t in range(1, len(observations) + 1):
        reference, ess = reference_posterior_particles_np(rng, observations[:t], cfg)
        reference_flat = reference.reshape(len(reference), -1)
        learned = posterior_sequence[t - 1]
        learned_rmse.append(float(np.sqrt(np.mean((learned.mean(axis=0) - truth_flat) ** 2))))
        reference_rmse.append(float(np.sqrt(np.mean((reference_flat.mean(axis=0) - truth_flat) ** 2))))
        ess_values.append(ess)

    t_axis = np.arange(1, len(observations) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axes[0].plot(t_axis, np.maximum(learned_rmse, 1e-12), marker="o", label="learned (repeated Bayes)")
    axes[0].plot(t_axis, np.maximum(reference_rmse, 1e-12), marker="o", label="SNIS reference (true likelihood)")
    axes[0].set_title("Posterior-mean RMSE vs theta*"); axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
    axes[1].plot(t_axis, np.maximum(ess_values, 1e-12), marker="o")
    axes[1].set_title("SNIS effective sample size"); axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("repeated Bayes step t"); ax.grid(alpha=0.2)
    fig.suptitle("Sequential soundness check: learned map vs true-likelihood reference", fontsize=13, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 11) Visualisation: training diagnostics
def plot_training_diagnostics(
    history: dict[str, list], best_epoch: int, destination: Path | None = None, cfg: BayesTransportConfig = CFG,
):
    steps = np.arange(1, len(history["step_loss"]) + 1)
    epochs = np.arange(1, len(history["epoch_train_loss"]) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0), constrained_layout=True)

    values = np.maximum(np.asarray(history["step_loss"], dtype=float), 1e-12)
    axes[0, 0].plot(steps, values, linewidth=0.70, alpha=0.60, label="empirical cloud energy score")
    if len(values) >= 20:
        window = max(5, len(values) // 100)
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(steps[window - 1:], smoothed, linewidth=1.8, label=f"moving average ({window})")
    axes[0, 0].set_title("Training objective at every gradient step", loc="left", fontweight="bold")
    axes[0, 0].set_yscale("log"); axes[0, 0].set_xlabel("gradient step"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(steps, history["step_attraction"], linewidth=0.75, label="target attraction")
    axes[0, 1].plot(steps, history["step_repulsion"], linewidth=0.75, label="cloud pairwise spread")
    axes[0, 1].set_title("Objective decomposition", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("gradient step"); axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot(steps, history["step_grad_norm"], linewidth=0.75)
    axes[0, 2].set_title("Gradient norm", loc="left", fontweight="bold"); axes[0, 2].set_xlabel("gradient step")

    axes[1, 0].plot(epochs, np.maximum(history["epoch_train_loss"], 1e-12), marker="o", markersize=3, label="train ES")
    axes[1, 0].plot(epochs, np.maximum(history["epoch_val_loss"], 1e-12), marker="o", markersize=3, label="iid val ES")
    axes[1, 0].axvline(best_epoch, linestyle="--", linewidth=1.0, label=f"best epoch {best_epoch}")
    lr_axis = axes[1, 0].twinx()
    lr_axis.plot(epochs, np.asarray(history["epoch_learning_rate"], dtype=float), linestyle=":", linewidth=1.4, alpha=0.75)
    lr_axis.set_yscale("log"); lr_axis.set_ylabel("learning rate")
    axes[1, 0].set_title("Model-selection objective", loc="left", fontweight="bold")
    axes[1, 0].set_yscale("log"); axes[1, 0].set_xlabel("epoch"); axes[1, 0].legend(fontsize=8)

    seq_energy = np.asarray(history["epoch_val_energy_by_t"], dtype=float)
    seq_rmse = np.asarray(history["epoch_val_rmse_by_t"], dtype=float)
    t_axis = np.arange(1, seq_energy.shape[1] + 1)
    selected_epochs = np.unique(np.clip(np.rint(np.linspace(0, len(seq_energy) - 1, 5)).astype(int), 0, len(seq_energy) - 1))
    for epoch_index in selected_epochs:
        axes[1, 1].plot(t_axis, np.maximum(seq_energy[epoch_index], 1e-12), label=f"epoch {epoch_index + 1}")
        axes[1, 2].plot(t_axis, np.maximum(seq_rmse[epoch_index], 1e-12), label=f"epoch {epoch_index + 1}")
    axes[1, 1].set_title("Repeated-Bayes ES during training", loc="left", fontweight="bold")
    axes[1, 2].set_title("Repeated-Bayes RMSE during training", loc="left", fontweight="bold")
    for ax in axes[1, 1:]:
        ax.set_xlabel("evaluation-only sequential step t"); ax.set_yscale("log"); ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)

    fig.suptitle("Single-cloud energy-score training + sequential evaluation (AdaLN)", fontsize=14, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)


#%% 12) Training function
def train_model(
    train_loader: DataLoader, amortized_eval_data: dict[str, np.ndarray], sequential_eval_data: dict[str, np.ndarray],
    fixed_trajectory: dict[str, np.ndarray], fixed_prior_particles: np.ndarray, run_dir: Path,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Any]:
    model = SequentialBayesModel(cfg, key=jax.random.key(cfg.seed))
    print("\namortized single-cloud energy transport (AdaLN)")
    print_model_parameter_count(model)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    plateau = optax.contrib.reduce_on_plateau(
        factor=0.5, patience=cfg.lr_plateau_patience, rtol=cfg.lr_plateau_rtol, cooldown=0,
        accumulation_size=1, min_scale=0.01,
    )
    plateau_state = plateau.init(params)

    @eqx.filter_jit
    def train_step(candidate_model, candidate_opt_state, learning_rate_scale, batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(batch_objective, has_aux=True)(candidate_model, batch)
        params = eqx.filter(candidate_model, eqx.is_array)
        updates, candidate_opt_state = optimizer.update(grads, candidate_opt_state, params)
        updates = jax.tree_util.tree_map(lambda u: learning_rate_scale * u, updates)
        candidate_model = eqx.apply_updates(candidate_model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return candidate_model, candidate_opt_state, loss, metrics, grad_norm

    history: dict[str, list] = {
        "step_loss": [], "step_attraction": [], "step_repulsion": [], "step_grad_norm": [],
        "epoch_train_loss": [], "epoch_learning_rate": [], "epoch_val_loss": [],
        "epoch_val_mean_rmse": [], "epoch_seq_final_energy_score": [], "epoch_seq_final_mean_rmse": [],
        "epoch_val_energy_by_t": [], "epoch_val_rmse_by_t": [],
    }

    plot_posterior_evolution(
        model, fixed_trajectory, fixed_prior_particles, cfg,
        run_dir / "plots" / "fixed_trajectory_before_training.png", "evaluation-only repeated Bayes: before training",
    )

    visualisation_epochs = sorted(set(max(1, int(math.ceil(f * cfg.epochs / 10.0))) for f in range(1, 11)))
    best_val_loss = float("inf")
    best_epoch = 0
    n_steps = cfg.n_train_trajectories // cfg.batch_size
    if n_steps < 1:
        raise ValueError("n_train_trajectories must be at least one batch_size.")
    train_iterator = iter(train_loader)
    training_started_at = time.time()

    # progress = tqdm(range(n_steps), desc=f"epoch 001/{cfg.epochs:03d}",
    #                     dynamic_ncols=True, leave=True, mininterval=5.0)

    for epoch in range(1, cfg.epochs + 1):
        epoch_started_at = time.time()
        epoch_lr_scale = plateau_state.scale
        epoch_learning_rate = cfg.learning_rate * float(jax.device_get(epoch_lr_scale))
        train_losses_this_epoch: list[float] = []
        progress = tqdm(range(n_steps), desc=f"epoch {epoch:03d}/{cfg.epochs:03d}",
                         dynamic_ncols=True, leave=True, mininterval=5.0)

        # if epoch > 1:
        #     progress.reset(total=n_steps)
        # progress.set_description(f"epoch {epoch:03d}/{cfg.epochs:03d}", refresh=False)

        for _ in progress:
            batch_np = next(train_iterator)
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            model, opt_state, loss, metrics, grad_norm = train_step(model, opt_state, epoch_lr_scale, batch)
            host = jax.device_get(metrics)
            host_loss = float(jax.device_get(loss))
            train_losses_this_epoch.append(host_loss)
            history["step_loss"].append(host_loss)
            history["step_attraction"].append(float(host["attraction"]))
            history["step_repulsion"].append(float(host["repulsion"]))
            history["step_grad_norm"].append(float(jax.device_get(grad_norm)))
            progress.set_postfix(ES=f"{host_loss:.4f}", RMSE=f"{float(host['posterior_mean_rmse']):.4f}", refresh=False)
            # progress.update(1)

        epoch_train_loss = float(np.mean(train_losses_this_epoch))
        val_metrics = evaluate_amortized_model(model, amortized_eval_data, cfg)
        seq_metrics = evaluate_sequential_model(model, sequential_eval_data, cfg)

        _, plateau_state = plateau.update(
            updates=eqx.filter(model, eqx.is_array), state=plateau_state,
            value=jnp.asarray(val_metrics["loss"], dtype=jnp.float32),
        )
        next_learning_rate = cfg.learning_rate * float(jax.device_get(plateau_state.scale))

        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_learning_rate"].append(epoch_learning_rate)
        history["epoch_val_loss"].append(float(val_metrics["loss"]))
        history["epoch_val_mean_rmse"].append(float(val_metrics["posterior_mean_rmse"]))
        history["epoch_seq_final_energy_score"].append(float(seq_metrics["final_energy_score"]))
        history["epoch_seq_final_mean_rmse"].append(float(seq_metrics["final_mean_rmse"]))
        history["epoch_val_energy_by_t"].append(np.asarray(seq_metrics["energy_by_t"], dtype=np.float64))
        history["epoch_val_rmse_by_t"].append(np.asarray(seq_metrics["rmse_by_t"], dtype=np.float64))

        save_model(run_dir / "artefacts" / "model_last.eqx", model)
        if epoch % cfg.save_every_epochs == 0:
            save_model(run_dir / "artefacts" / f"model_epoch_{epoch:04d}.eqx", model)
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss, best_epoch = float(val_metrics["loss"]), epoch
            save_model(run_dir / "artefacts" / "model_best.eqx", model)

        np.savez_compressed(run_dir / "artefacts" / "history.npz", **{n: np.asarray(v) for n, v in history.items()})
        print(
            f"epoch {epoch:03d}: train ES={epoch_train_loss:.6f} | val ES={float(val_metrics['loss']):.6f} | "
            f"val RMSE={float(val_metrics['posterior_mean_rmse']):.5f} | lr={epoch_learning_rate:.3e} -> {next_learning_rate:.3e} || "
            f"seq final ES={float(seq_metrics['final_energy_score']):.6f} | seq final RMSE={float(seq_metrics['final_mean_rmse']):.5f} | "
            f"{time.time() - epoch_started_at:.1f}s"
        )

        if epoch in visualisation_epochs:
            plot_posterior_evolution(
                model, fixed_trajectory, fixed_prior_particles, cfg,
                run_dir / "plots" / f"fixed_trajectory_epoch_{epoch:04d}.png",
                f"evaluation-only repeated Bayes after amortized epoch {epoch}",
            )

    best_model = load_model(run_dir / "artefacts" / "model_best.eqx", cfg)
    final_amortized_metrics = evaluate_amortized_model(best_model, amortized_eval_data, cfg)
    final_metrics = evaluate_sequential_model(best_model, sequential_eval_data, cfg)
    plot_posterior_evolution(
        best_model, fixed_trajectory, fixed_prior_particles, cfg,
        run_dir / "plots" / "fixed_trajectory_best_model.png", f"best model (epoch {best_epoch}): evaluation-only repeated Bayes",
    )
    plot_sequential_reference_check(
        best_model, fixed_trajectory, fixed_prior_particles, cfg,
        run_dir / "plots" / "sequential_reference_check.png",
    )
    plot_training_diagnostics(history, best_epoch, run_dir / "plots" / "training_diagnostics.png", cfg)

    elapsed = int(time.time() - training_started_at)
    print(f"training complete in {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}; "
          f"best epoch={best_epoch}, iid val ES={best_val_loss:.6f}")
    return {
        "model": best_model, "history": history, "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "amortized_final_metrics": final_amortized_metrics, "final_metrics": final_metrics,
    }


#%% 13) Build run, data, and either train a fresh model or reload the best checkpoint
if __name__ == "__main__":
    np.random.seed(CFG.seed)
    print("JAX devices:", jax.devices())
    print("Configuration:\n", yaml.safe_dump(asdict(CFG), sort_keys=False))

    if train_wm:
        run_dir = make_run_dir(CFG.env_name, CFG.runs_base)
        archived_script = copy_running_script_to_run_dir(run_dir)
        print("Run directory:", run_dir)
        if archived_script is not None:
            print("Archived training script:", archived_script)

        train_loader = make_continuous_train_loader(CFG, seed=CFG.seed + 1_000)
        amortized_eval_data = simulate_iid_joint_samples(np.random.default_rng(CFG.seed + 2_000), CFG.n_eval_trajectories, CFG)
        sequential_eval_data = simulate_trajectories(
            np.random.default_rng(CFG.seed + 2_100), CFG.n_eval_trajectories, CFG.evaluation_trajectory_length, CFG
        )
        # Sequential eval also needs a starting prior cloud per trajectory (fresh tau=0 clouds).
        prior_rng = np.random.default_rng(CFG.seed + 2_200)
        sequential_eval_data["prior_particles"] = np.stack(
            [sample_base_prior_np(prior_rng, CFG.num_particles, CFG) for _ in range(len(sequential_eval_data["theta_true"]))]
        )

        # One fixed problem kept for the periodic-during-training and final diagnostic plots.
        fixed_data = simulate_trajectories(np.random.default_rng(CFG.seed + 2_500), 1, CFG.evaluation_trajectory_length, CFG)
        fixed_trajectory = {"theta_true": fixed_data["theta_true"][0], "observations": fixed_data["observations"][0]}
        fixed_prior_particles = sample_base_prior_np(np.random.default_rng(CFG.seed + 3_000), CFG.num_particles, CFG)
        np.savez_compressed(
            run_dir / "artefacts" / "fixed_trajectory.npz",
            theta_true=fixed_trajectory["theta_true"], observations=fixed_trajectory["observations"],
            prior_particles=fixed_prior_particles,
        )

        result = train_model(
            train_loader, amortized_eval_data, sequential_eval_data, fixed_trajectory, fixed_prior_particles, run_dir, CFG,
        )
    else:
        run_dir = Path.cwd().expanduser().resolve()
        (run_dir / "plots").mkdir(parents=True, exist_ok=True)
        (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
        print("Existing run directory:", run_dir)

        amortized_eval_data = simulate_iid_joint_samples(np.random.default_rng(CFG.seed + 2_000), CFG.n_eval_trajectories, CFG)
        sequential_eval_data = simulate_trajectories(
            np.random.default_rng(CFG.seed + 2_100), CFG.n_eval_trajectories, CFG.evaluation_trajectory_length, CFG
        )
        prior_rng = np.random.default_rng(CFG.seed + 2_200)
        sequential_eval_data["prior_particles"] = np.stack(
            [sample_base_prior_np(prior_rng, CFG.num_particles, CFG) for _ in range(len(sequential_eval_data["theta_true"]))]
        )
        fixed = np.load(run_dir / "artefacts" / "fixed_trajectory.npz")
        fixed_trajectory = {"theta_true": fixed["theta_true"], "observations": fixed["observations"]}
        fixed_prior_particles = fixed["prior_particles"]

        best_model = load_model(run_dir / "artefacts" / "model_best.eqx", CFG)
        history_path = run_dir / "artefacts" / "history.npz"
        history = dict(np.load(history_path)) if history_path.is_file() else None

        final_amortized_metrics = evaluate_amortized_model(best_model, amortized_eval_data, CFG)
        final_metrics = evaluate_sequential_model(best_model, sequential_eval_data, CFG)
        print_model_parameter_count(best_model)
        if history is not None and len(history["epoch_val_loss"]):
            best_epoch = int(np.argmin(history["epoch_val_loss"])) + 1
            plot_training_diagnostics(history, best_epoch, run_dir / "plots" / "training_diagnostics.png", CFG)
        plot_posterior_evolution(
            best_model, fixed_trajectory, fixed_prior_particles, CFG,
            run_dir / "plots" / "fixed_trajectory_reloaded.png", "reloaded best model: evaluation-only repeated Bayes",
        )
        plot_sequential_reference_check(
            best_model, fixed_trajectory, fixed_prior_particles, CFG, run_dir / "plots" / "sequential_reference_check.png",
        )
        result = {"model": best_model, "history": history, "amortized_final_metrics": final_amortized_metrics, "final_metrics": final_metrics}

    model = result["model"]

    summary = {
        "objective": "exact empirical multivariate energy score in physical theta space",
        "training_mode": "non-sequential amortized transport from synthetic interpolated input clouds",
        "problem_shape": {"num_sources": S, "source_dim": D},
        "observations_per_training_step": 1,
        "posterior_conditioning": "adaln",
        "base_prior_distribution": "uniform",
        "synthetic_truth_sampling_mode": "closed_form",
        "sequential_evaluation": True,
        "final_amortized_metrics": result["amortized_final_metrics"],
        "final_sequential_metrics": {
            k: float(v) for k, v in result["final_metrics"].items() if np.ndim(v) == 0
        },
    }
    save_json(run_dir / "artefacts" / "final_summary.json", summary)
    print("\nFinal summary:")
    print(json.dumps(summary, indent=2))
    print("All artefacts saved under:", run_dir)
