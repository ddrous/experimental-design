#%% 1) Imports, policy configuration, and frozen Bayes-pushforward loading
"""Train sequential design policies through the Stage-I approximate Bayes operator.

The frozen Stage-I model implements an in-context sample transport

    B_t = T_phi(Z, D_t) ~= p(theta | D_t),

where Z is a fixed prior sample and D_t = {(x_i, y_i)}_{i=1}^t.  Every posterior
refresh starts from Z and the complete context D_t; the code does *not* recursively
transport B_{t-1}.  This avoids compounding a one-step approximation and matches the
non-sequential Stage-I training distribution.

Two policy-training variants are provided in this single file.

``simple_energy_score``
    Roll out the policy through the differentiable simulator and frozen transport,
    then minimise one terminal energy-score term for the sample approximation to
    p(theta | D_T).  The energy score is proper for sample distributions and does not
    require evaluating q_phi(theta | D_T) or p(y | theta, x).

``self_distilled_eig``
    At a random posterior state, generate a candidate design set.  The frozen
    approximate Bayes operator evaluates each candidate by simulator-only posterior
    contraction under theta~ ~ T_phi(Z,D_t) and y~p(y | theta~,x).  A soft best-design
    target is formed from the estimated information gains and the policy is distilled
    toward it with one mean-squared-error objective.  No ground-truth theta is used in
    the teacher target once the current context has been generated.

The default ``objective_mode='both'`` trains and saves both variants in separate run
directories.  At deployment, either policy sees only approximate posterior particles
and the current step fraction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Iterator
import math
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm
import yaml

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import stage_1_train_bayes_pushforward as s1
from stage_1_train_bayes_pushforward import (
    Array,
    BayesPushforwardTransformer,
    BayesTransportConfig,
    SourceLocPrior,
    canonicalize_sources_jax,
    dataclass_from_dict,
    find_latest_run,
    load_transport_model,
    make_run_dir,
    save_config_yaml,
    save_json,
    simulate_observation_jax,
    snapshot_files,
    source_log_likelihood_np,
    source_log_signal_jax,
)


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for the downstream amortised design policies."""

    env_name: str = "design_policy_pushforward"
    seed: int = 17

    # Set this explicitly to a Stage-I run directory, or leave None to select the
    # newest matching run under runs_base.
    inference_run_dir: str | None = None
    inference_env_name: str = "bayes_pushforward_adaln"
    inference_checkpoint_name: str = "model_best.eqx"
    runs_base: str = "./runs"
    runs_base_policy: str = "./runs"

    # ``both`` trains the two requested implementations in separate run directories.
    objective_mode: str = "both"  # both, simple_energy_score, self_distilled_eig

    # Sequential experiment budget.  horizon must not exceed the Stage-I maximum
    # context size because D_t is passed to T_phi as one padded unordered set.
    horizon: int = 6
    num_prior_particles: int = 64
    design_exploration_std: float = 0.05

    # Policy architecture.
    hidden_dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: int = 4

    # Self-distilled candidate teacher.  Candidate utilities use only posterior
    # particles and forward simulation, not an evaluable likelihood.
    teacher_candidates: int = 16
    teacher_outcome_samples: int = 2
    teacher_softmax_temperature: float = 0.15
    posterior_logdet_jitter: float = 1e-4

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

    # Diagnostics/persistence.
    final_plot_examples: int = 6
    save_every_epochs: int = 1


POLICY_CFG = PolicyConfig(
    inference_run_dir=None,
    objective_mode="both",
    horizon=6,
    num_prior_particles=64,
    epochs=30,
    n_train_episodes=20_000,
    n_eval_episodes=256,
    batch_size=64,
)


#%% 2) Theta-only datasets for simulator rollouts
class FiniteThetaEpisodes(Dataset):
    def __init__(self, prior: SourceLocPrior, n_episodes: int, base_seed: int):
        self.prior = prior
        self.n_episodes = int(n_episodes)
        self.seeds = (
            np.arange(self.n_episodes, dtype=np.int64) + int(base_seed)
        ).tolist()

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        return self.prior.sample(np.random.default_rng(self.seeds[idx]))


class InfiniteThetaEpisodes(IterableDataset):
    def __init__(self, prior: SourceLocPrior, base_seed: int):
        self.prior = prior
        self.base_seed = int(base_seed)

    def __iter__(self) -> Iterator[np.ndarray]:
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


#%% 3) Set Transformer design policy
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


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

    def __call__(self, tokens: Array) -> Array:
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h)
        h = _layernorm_tokens(self.norm2, tokens)
        h = _linear_tokens(self.ff_out, jax.nn.gelu(_linear_tokens(self.ff_in, h)))
        return tokens + h


class DesignPolicyTransformer(eqx.Module):
    """Map samples from p(theta | D_t) to the next bounded design x_{t+1}."""

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

    def __init__(
        self,
        transport_cfg: BayesTransportConfig,
        policy_cfg: PolicyConfig,
        *,
        key: Array,
    ):
        self.K = transport_cfg.K
        self.theta_dim = 2 * transport_cfg.K
        self.design_low = transport_cfg.design_low
        self.design_high = transport_cfg.design_high
        self.canonicalize = transport_cfg.canonicalize_particle_sources

        keys = jax.random.split(key, policy_cfg.depth + 4)
        self.particle_in = eqx.nn.Linear(
            self.theta_dim, policy_cfg.hidden_dim, key=keys[0]
        )
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

    def __call__(self, posterior_particles: Array, step_fraction: Array) -> Array:
        if self.canonicalize and self.K > 1:
            posterior_particles = canonicalize_sources_jax(posterior_particles)
        tokens = posterior_particles.reshape(
            posterior_particles.shape[0], self.theta_dim
        )
        tokens = _linear_tokens(self.particle_in, tokens)
        tokens = tokens + self.step_encoder(jnp.asarray([step_fraction]))[None, :]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = _layernorm_tokens(self.final_norm, tokens)
        pooled = jnp.concatenate(
            [
                jnp.mean(tokens, axis=0),
                jnp.sqrt(jnp.var(tokens, axis=0) + 1e-6),
            ]
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
    transport_cfg: BayesTransportConfig,
    policy_cfg: PolicyConfig,
    *,
    key: Array | None = None,
) -> DesignPolicyTransformer:
    if key is None:
        key = jax.random.key(0)
    skeleton = DesignPolicyTransformer(transport_cfg, policy_cfg, key=key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)


#%% 4) Sample-only posterior scores and uncertainty summaries
def flatten_measure(particles: Array, transport_cfg: BayesTransportConfig) -> Array:
    if transport_cfg.canonicalize_particle_sources and transport_cfg.K > 1:
        particles = canonicalize_sources_jax(particles)
    return particles.reshape(particles.shape[0], 2 * transport_cfg.K)


def flatten_theta(theta: Array, transport_cfg: BayesTransportConfig) -> Array:
    if transport_cfg.canonicalize_particle_sources and transport_cfg.K > 1:
        theta = canonicalize_sources_jax(theta)
    return theta.reshape(2 * transport_cfg.K)


def energy_score_single(
    posterior_particles: Array,
    theta_true: Array,
    transport_cfg: BayesTransportConfig,
) -> Array:
    """Multivariate energy score for a sample approximation to p(theta | D).

    ES(B, theta) = mean_n ||theta_n-theta||
                   - 0.5 mean_{n,m} ||theta_n-theta_m||.

    It is a proper sample-based scoring rule: the second term prevents the degenerate
    policy objective that would reward collapsing every posterior sample to one point.
    """
    samples = flatten_measure(posterior_particles, transport_cfg)
    target = flatten_theta(theta_true, transport_cfg)
    truth_distance = jnp.mean(jnp.linalg.norm(samples - target[None, :], axis=-1))
    pairwise_distance = jnp.mean(
        jnp.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1)
    )
    return truth_distance - 0.5 * pairwise_distance


def posterior_logdet_single(
    posterior_particles: Array,
    transport_cfg: BayesTransportConfig,
    jitter: float,
) -> Array:
    """Gaussian log-volume proxy computed only from posterior samples."""
    samples = flatten_measure(posterior_particles, transport_cfg)
    centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    denominator = jnp.maximum(samples.shape[0] - 1, 1)
    covariance = centered.T @ centered / denominator
    covariance = covariance + jitter * jnp.eye(covariance.shape[0])
    sign, logdet = jnp.linalg.slogdet(covariance)
    return jnp.where(sign > 0, logdet, jnp.asarray(50.0))


def posterior_mean_rmse_single(
    posterior_particles: Array,
    theta_true: Array,
    transport_cfg: BayesTransportConfig,
) -> Array:
    samples = flatten_measure(posterior_particles, transport_cfg)
    target = flatten_theta(theta_true, transport_cfg)
    return jnp.sqrt(jnp.mean((jnp.mean(samples, axis=0) - target) ** 2))


#%% 5) Differentiable simple policy objective
def simple_energy_score_objective(
    candidate_policy: DesignPolicyTransformer,
    theta_true: Array,
    key: Array,
    frozen_transport: BayesPushforwardTransformer,
    transport_cfg: BayesTransportConfig,
    policy_cfg: PolicyConfig,
):
    """End-to-end terminal energy-score objective for a batch of true parameters."""
    batch_size = theta_true.shape[0]
    key, prior_key = jax.random.split(key)
    base_particles = transport_cfg.prior_std * jax.random.normal(
        prior_key,
        shape=(
            batch_size,
            policy_cfg.num_prior_particles,
            transport_cfg.K,
            2,
        ),
    )
    context_x = jnp.zeros(
        (batch_size, transport_cfg.max_context_pairs, 2), dtype=jnp.float32
    )
    context_y = jnp.zeros(
        (batch_size, transport_cfg.max_context_pairs, 1), dtype=jnp.float32
    )
    context_mask = jnp.zeros(
        (batch_size, transport_cfg.max_context_pairs), dtype=jnp.float32
    )
    posterior = jax.vmap(frozen_transport)(
        base_particles, context_x, context_y, context_mask
    )
    designs = []

    for step in range(policy_cfg.horizon):
        step_fraction = jnp.asarray(
            step / max(policy_cfg.horizon - 1, 1), dtype=jnp.float32
        )
        proposed_design = jax.vmap(
            lambda particles: candidate_policy(particles, step_fraction)
        )(posterior)
        key, design_noise_key, observation_key = jax.random.split(key, 3)
        exploration = jax.random.normal(design_noise_key, proposed_design.shape)
        design = proposed_design + (
            policy_cfg.design_exploration_std
            * (transport_cfg.design_high - transport_cfg.design_low)
            * exploration
        )
        design = jnp.clip(
            design, transport_cfg.design_low, transport_cfg.design_high
        )
        observation = simulate_observation_jax(
            theta_true, design, transport_cfg, observation_key
        )
        context_x = context_x.at[:, step, :].set(design)
        context_y = context_y.at[:, step, 0].set(observation)
        context_mask = context_mask.at[:, step].set(1.0)

        # Every posterior p(theta | D_t) is recomputed from the same prior sample Z
        # and the complete context D_t.  Stage I was not trained as a one-step chain.
        posterior = jax.vmap(frozen_transport)(
            base_particles, context_x, context_y, context_mask
        )
        designs.append(design)

    energy = jax.vmap(
        lambda particles, theta: energy_score_single(
            particles, theta, transport_cfg
        )
    )(posterior, theta_true)
    logdet = jax.vmap(
        lambda particles: posterior_logdet_single(
            particles, transport_cfg, policy_cfg.posterior_logdet_jitter
        )
    )(posterior)
    rmse = jax.vmap(
        lambda particles, theta: posterior_mean_rmse_single(
            particles, theta, transport_cfg
        )
    )(posterior, theta_true)

    # Exactly one policy objective term.
    loss = jnp.mean(energy)
    metrics = {
        "loss": loss,
        "terminal_energy_score": jnp.mean(energy),
        "terminal_logdet": jnp.mean(logdet),
        "terminal_mean_rmse": jnp.mean(rmse),
        "distillation_mse": jnp.asarray(0.0),
        "teacher_expected_gain": jnp.asarray(0.0),
        "teacher_target_distance": jnp.asarray(0.0),
    }
    return loss, metrics


#%% 6) Self-distilled approximate-information-gain objective
def self_distilled_single_objective(
    candidate_policy: DesignPolicyTransformer,
    theta_true: Array,
    key: Array,
    frozen_transport: BayesPushforwardTransformer,
    transport_cfg: BayesTransportConfig,
    policy_cfg: PolicyConfig,
):
    """Distil a sample-based one-step information-gain teacher at one random state.

    A random context D_t is first generated from the training simulator.  The teacher
    then stops using theta_true: it samples theta~ from the current approximate
    posterior, simulates y~p(y | theta~, x), refreshes p(theta | D_t union {(x,y)})
    through T_phi, and scores the reduction in posterior log-volume.
    """
    (
        history_key,
        prior_key,
        policy_key,
        candidate_key,
        teacher_key,
    ) = jax.random.split(key, 5)
    history_length = jax.random.randint(
        history_key, shape=(), minval=0, maxval=policy_cfg.horizon
    )

    base_particles = transport_cfg.prior_std * jax.random.normal(
        prior_key,
        shape=(
            policy_cfg.num_prior_particles,
            transport_cfg.K,
            2,
        ),
    )
    context_x = jax.random.uniform(
        candidate_key,
        shape=(transport_cfg.max_context_pairs, 2),
        minval=transport_cfg.design_low,
        maxval=transport_cfg.design_high,
    )
    teacher_key, history_noise_key, random_candidate_key = jax.random.split(
        teacher_key, 3
    )
    history_mean = source_log_signal_jax(theta_true, context_x, transport_cfg)
    history_y = history_mean + transport_cfg.observation_noise_std * jax.random.normal(
        history_noise_key, history_mean.shape
    )
    context_mask = (
        jnp.arange(transport_cfg.max_context_pairs) < history_length
    ).astype(jnp.float32)
    context_y = history_y[:, None] * context_mask[:, None]
    context_x = context_x * context_mask[:, None]

    posterior = frozen_transport(
        base_particles, context_x, context_y, context_mask
    )
    step_fraction = history_length.astype(jnp.float32) / max(
        policy_cfg.horizon - 1, 1
    )
    policy_design = candidate_policy(posterior, step_fraction)

    if policy_cfg.teacher_candidates < 2:
        raise ValueError("teacher_candidates must be at least 2.")
    random_candidates = jax.random.uniform(
        random_candidate_key,
        shape=(policy_cfg.teacher_candidates - 1, 2),
        minval=transport_cfg.design_low,
        maxval=transport_cfg.design_high,
    )
    # Include the policy's own proposal in the candidate pool.  The distilled target
    # is stop-gradient, so the teacher cannot be gamed through this inclusion.
    candidates = jnp.concatenate(
        [jax.lax.stop_gradient(policy_design[None, :]), random_candidates], axis=0
    )
    current_logdet = posterior_logdet_single(
        posterior, transport_cfg, policy_cfg.posterior_logdet_jitter
    )
    candidate_keys = jax.random.split(teacher_key, policy_cfg.teacher_candidates)

    def candidate_expected_next_logdet(candidate: Array, candidate_root_key: Array):
        outcome_keys = jax.random.split(
            candidate_root_key, policy_cfg.teacher_outcome_samples
        )

        def one_hypothetical_outcome(outcome_key: Array):
            index_key, observation_key = jax.random.split(outcome_key)
            particle_index = jax.random.randint(
                index_key,
                shape=(),
                minval=0,
                maxval=policy_cfg.num_prior_particles,
            )
            theta_hypothetical = posterior[particle_index]
            observation = simulate_observation_jax(
                theta_hypothetical, candidate, transport_cfg, observation_key
            )
            next_x = context_x.at[history_length].set(candidate)
            next_y = context_y.at[history_length, 0].set(observation)
            next_mask = context_mask.at[history_length].set(1.0)
            next_posterior = frozen_transport(
                base_particles, next_x, next_y, next_mask
            )
            return posterior_logdet_single(
                next_posterior,
                transport_cfg,
                policy_cfg.posterior_logdet_jitter,
            )

        return jnp.mean(jax.vmap(one_hypothetical_outcome)(outcome_keys))

    expected_next_logdet = jax.vmap(candidate_expected_next_logdet)(
        candidates, candidate_keys
    )
    estimated_gain = current_logdet - expected_next_logdet
    target_weights = jax.nn.softmax(
        estimated_gain / max(policy_cfg.teacher_softmax_temperature, 1e-6)
    )
    target_design = jax.lax.stop_gradient(
        jnp.sum(target_weights[:, None] * candidates, axis=0)
    )

    # Exactly one self-distillation objective term.
    loss = jnp.mean((policy_design - target_design) ** 2)
    best_candidate = candidates[jnp.argmax(estimated_gain)]
    target_distance = jnp.sqrt(jnp.mean((policy_design - best_candidate) ** 2))
    metrics = {
        "loss": loss,
        "terminal_energy_score": jnp.asarray(0.0),
        "terminal_logdet": current_logdet,
        "terminal_mean_rmse": posterior_mean_rmse_single(
            posterior, theta_true, transport_cfg
        ),
        "distillation_mse": loss,
        "teacher_expected_gain": jnp.sum(target_weights * estimated_gain),
        "teacher_target_distance": target_distance,
    }
    return loss, metrics


def self_distilled_eig_objective(
    candidate_policy: DesignPolicyTransformer,
    theta_true: Array,
    key: Array,
    frozen_transport: BayesPushforwardTransformer,
    transport_cfg: BayesTransportConfig,
    policy_cfg: PolicyConfig,
):
    batch_keys = jax.random.split(key, theta_true.shape[0])
    losses, metrics = jax.vmap(
        lambda theta, sample_key: self_distilled_single_objective(
            candidate_policy,
            theta,
            sample_key,
            frozen_transport,
            transport_cfg,
            policy_cfg,
        )
    )(theta_true, batch_keys)
    mean_metrics = jax.tree.map(lambda value: jnp.mean(value), metrics)
    return jnp.mean(losses), mean_metrics


#%% 7) Deployment-faithful rollout and adapted diagnostic plots
def rollout_single(
    candidate_policy: DesignPolicyTransformer,
    theta_true_np: np.ndarray,
    seed: int,
    frozen_transport: BayesPushforwardTransformer,
    transport_cfg: BayesTransportConfig,
    policy_cfg: PolicyConfig,
):
    """Run one sequential experiment and return arrays for direct notebook plots."""
    theta_true = jnp.asarray(theta_true_np)
    key = jax.random.key(seed)
    key, prior_key = jax.random.split(key)
    base_particles = transport_cfg.prior_std * jax.random.normal(
        prior_key,
        shape=(
            policy_cfg.num_prior_particles,
            transport_cfg.K,
            2,
        ),
    )
    context_x = jnp.zeros((transport_cfg.max_context_pairs, 2), dtype=jnp.float32)
    context_y = jnp.zeros((transport_cfg.max_context_pairs, 1), dtype=jnp.float32)
    context_mask = jnp.zeros((transport_cfg.max_context_pairs,), dtype=jnp.float32)
    posterior = frozen_transport(base_particles, context_x, context_y, context_mask)

    posteriors = [np.asarray(jax.device_get(posterior))]
    designs = []
    observations = []
    energy_scores = [
        float(jax.device_get(energy_score_single(posterior, theta_true, transport_cfg)))
    ]
    logdets = [
        float(
            jax.device_get(
                posterior_logdet_single(
                    posterior,
                    transport_cfg,
                    policy_cfg.posterior_logdet_jitter,
                )
            )
        )
    ]

    for step in range(policy_cfg.horizon):
        step_fraction = jnp.asarray(
            step / max(policy_cfg.horizon - 1, 1), dtype=jnp.float32
        )
        design = candidate_policy(posterior, step_fraction)
        key, observation_key = jax.random.split(key)
        observation = simulate_observation_jax(
            theta_true, design, transport_cfg, observation_key
        )
        context_x = context_x.at[step].set(design)
        context_y = context_y.at[step, 0].set(observation)
        context_mask = context_mask.at[step].set(1.0)
        posterior = frozen_transport(
            base_particles, context_x, context_y, context_mask
        )

        designs.append(np.asarray(jax.device_get(design)))
        observations.append(float(jax.device_get(observation)))
        posteriors.append(np.asarray(jax.device_get(posterior)))
        energy_scores.append(
            float(
                jax.device_get(
                    energy_score_single(posterior, theta_true, transport_cfg)
                )
            )
        )
        logdets.append(
            float(
                jax.device_get(
                    posterior_logdet_single(
                        posterior,
                        transport_cfg,
                        policy_cfg.posterior_logdet_jitter,
                    )
                )
            )
        )

    return {
        "theta_true": np.asarray(theta_true_np),
        "base_particles": np.asarray(jax.device_get(base_particles)),
        "posteriors": posteriors,
        "designs": np.stack(designs, axis=0),
        "observations": np.asarray(observations),
        "energy_scores": np.asarray(energy_scores),
        "logdets": np.asarray(logdets),
    }


def plot_policy_rollout(
    trajectory: dict[str, np.ndarray],
    transport_cfg: BayesTransportConfig,
    policy_cfg: PolicyConfig,
    destination: Path,
    title_prefix: str,
):
    theta_true = trajectory["theta_true"]
    all_points = [theta_true.reshape(-1, 2), trajectory["designs"].reshape(-1, 2)]
    all_points.extend(p.reshape(-1, 2) for p in trajectory["posteriors"])
    shared = np.concatenate(all_points, axis=0)
    lim = max(
        3.0 * transport_cfg.prior_std,
        1.15 * float(np.quantile(np.abs(shared), 0.995)),
    )

    fig, axes = plt.subplots(
        2,
        policy_cfg.horizon + 1,
        figsize=(3.25 * (policy_cfg.horizon + 1), 6.8),
        squeeze=False,
        constrained_layout=True,
    )
    for stage in range(policy_cfg.horizon + 1):
        posterior_ax = axes[0, stage]
        posterior = trajectory["posteriors"][stage]
        posterior_ax.scatter(
            posterior[..., 0].reshape(-1),
            posterior[..., 1].reshape(-1),
            s=11,
            alpha=0.30,
            label=f"samples for p(theta | D_{stage})",
        )
        posterior_ax.scatter(
            theta_true[:, 0], theta_true[:, 1], marker="*", s=165,
            label="simulator parameter theta",
        )
        if stage > 0:
            posterior_ax.plot(
                trajectory["designs"][:stage, 0],
                trajectory["designs"][:stage, 1],
                marker="x",
                linewidth=1.2,
                label="design history",
            )
        posterior_ax.set_xlim(-lim, lim)
        posterior_ax.set_ylim(-lim, lim)
        posterior_ax.set_aspect("equal")
        posterior_ax.grid(alpha=0.2)
        posterior_ax.set_title(f"Approximate posterior p(theta | D_{stage})")
        if stage in {0, policy_cfg.horizon}:
            posterior_ax.legend(fontsize=7)

        likelihood_ax = axes[1, stage]
        if stage == 0:
            likelihood_ax.axis("off")
            likelihood_ax.text(
                0.5,
                0.5,
                "No observation yet\nD_0 = empty set",
                ha="center",
                va="center",
                transform=likelihood_ax.transAxes,
            )
        elif transport_cfg.K == 1 and transport_cfg.likelihood_available:
            design = trajectory["designs"][stage - 1]
            observation = trajectory["observations"][stage - 1]
            grid = np.linspace(-lim, lim, transport_cfg.grid_size)
            gx, gy = np.meshgrid(grid, grid)
            theta_grid = np.stack([gx, gy], axis=-1)[:, :, None, :]
            log_likelihood = source_log_likelihood_np(
                observation, theta_grid, design, transport_cfg
            )
            relative_likelihood = np.exp(log_likelihood - np.max(log_likelihood))
            contour = likelihood_ax.contourf(
                gx, gy, relative_likelihood, levels=28
            )
            likelihood_ax.scatter(
                design[0], design[1], marker="x", s=80, label=f"design x_{stage}"
            )
            likelihood_ax.scatter(
                theta_true[0, 0], theta_true[0, 1], marker="*", s=165,
                label="theta",
            )
            likelihood_ax.set_xlim(-lim, lim)
            likelihood_ax.set_ylim(-lim, lim)
            likelihood_ax.set_aspect("equal")
            likelihood_ax.set_title(
                f"Likelihood p(y_{stage} | theta, x_{stage})"
            )
            if stage == policy_cfg.horizon:
                fig.colorbar(contour, ax=likelihood_ax, shrink=0.75)
                likelihood_ax.legend(fontsize=7)
        else:
            likelihood_ax.axis("off")
            likelihood_ax.text(
                0.5,
                0.5,
                "p(y | theta, x) not plotted\n(simulator-only mode or K > 1)",
                ha="center",
                va="center",
                transform=likelihood_ax.transAxes,
            )

    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(destination, dpi=165)
    display(fig)
    plt.close(fig)

    steps = np.arange(policy_cfg.horizon + 1)
    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    ax.plot(steps, trajectory["energy_scores"], marker="o", label="posterior energy score")
    ax.plot(steps, trajectory["logdets"], marker="s", label="posterior log-volume proxy")
    ax.set_xlabel("number of acquired design--outcome pairs |D_t|")
    ax.set_title("Sample-based quality of p(theta | D_t) along the policy rollout")
    ax.grid(alpha=0.25)
    ax.legend()
    metrics_path = destination.with_name(destination.stem + "_metrics.png")
    fig.savefig(metrics_path, dpi=165)
    display(fig)
    plt.close(fig)


#%% 8) One complete training run for one policy variant
def train_variant(
    cfg: PolicyConfig,
    transport_cfg: BayesTransportConfig,
    frozen_transport: BayesPushforwardTransformer,
    inference_run_dir: Path,
    inference_checkpoint: Path,
):
    if cfg.objective_mode not in {"simple_energy_score", "self_distilled_eig"}:
        raise ValueError("train_variant expects one concrete objective mode.")

    np.random.seed(cfg.seed)
    prior = SourceLocPrior(K=transport_cfg.K, prior_std=transport_cfg.prior_std)
    train_loader = make_theta_train_loader(prior, cfg)
    eval_loader = make_theta_eval_loader(prior, cfg)
    fixed_theta = eval_loader.dataset[0]
    steps_per_epoch = (
        len(train_loader) if cfg.data_mode == "finite" else cfg.steps_per_epoch
    )

    run_name = f"{cfg.env_name}_{cfg.objective_mode}"
    run_dir = make_run_dir(run_name, cfg.runs_base_policy)
    script_path = Path(globals().get("__file__", "stage_2_train_design_policy.py")).resolve()
    stage1_path = Path(s1.__file__).resolve()
    snapshot_files(run_dir, [script_path, stage1_path])
    shutil.copy2(
        inference_run_dir / "config.yaml",
        run_dir / "artefacts" / "inference_config.yaml",
    )
    shutil.copy2(
        inference_checkpoint,
        run_dir / "artefacts" / "frozen_bayes_transport.eqx",
    )
    np.save(run_dir / "artefacts" / "fixed_theta.npy", fixed_theta)
    save_config_yaml(
        cfg,
        run_dir / "config.yaml",
        extra={
            "training_complete": False,
            "stage": 2,
            "objective_mode_resolved": cfg.objective_mode,
            "inference_run_dir_resolved": str(inference_run_dir),
            "inference_checkpoint_resolved": str(inference_checkpoint),
        },
    )

    print("\n" + "=" * 88)
    print("Training Stage-II variant:", cfg.objective_mode)
    print("Run directory:", run_dir)
    print("Frozen Stage-I run:", inference_run_dir)
    print("Policy configuration:\n", yaml.safe_dump(asdict(cfg), sort_keys=False))

    policy_key = jax.random.key(cfg.seed)
    policy = DesignPolicyTransformer(transport_cfg, cfg, key=policy_key)
    eqx.tree_pprint(policy)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        ),
    )
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    if cfg.objective_mode == "simple_energy_score":
        objective_fn = simple_energy_score_objective
    else:
        objective_fn = self_distilled_eig_objective

    @eqx.filter_jit
    def train_step(candidate_policy, candidate_opt_state, theta_true, key):
        (_, metrics), grads = eqx.filter_value_and_grad(
            objective_fn, has_aux=True
        )(
            candidate_policy,
            theta_true,
            key,
            frozen_transport,
            transport_cfg,
            cfg,
        )
        params = eqx.filter(candidate_policy, eqx.is_array)
        updates, candidate_opt_state = optimizer.update(
            grads, candidate_opt_state, params
        )
        candidate_policy = eqx.apply_updates(candidate_policy, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return candidate_policy, candidate_opt_state, metrics, grad_norm

    @eqx.filter_jit
    def eval_objective_step(candidate_policy, theta_true, key):
        _, metrics = objective_fn(
            candidate_policy,
            theta_true,
            key,
            frozen_transport,
            transport_cfg,
            cfg,
        )
        return metrics

    @eqx.filter_jit
    def eval_rollout_step(candidate_policy, theta_true, key):
        _, metrics = simple_energy_score_objective(
            candidate_policy,
            theta_true,
            key,
            frozen_transport,
            transport_cfg,
            replace(cfg, design_exploration_std=0.0),
        )
        return metrics

    def evaluate(candidate_policy, loader, evaluation_seed):
        objective_collected: dict[str, list[float]] = {}
        rollout_collected: dict[str, list[float]] = {}
        root_key = jax.random.key(evaluation_seed)
        for batch_index, theta_np in enumerate(loader):
            objective_key = jax.random.fold_in(root_key, 2 * batch_index)
            rollout_key = jax.random.fold_in(root_key, 2 * batch_index + 1)
            theta = jnp.asarray(theta_np)
            objective_metrics = jax.device_get(
                eval_objective_step(candidate_policy, theta, objective_key)
            )
            rollout_metrics = jax.device_get(
                eval_rollout_step(candidate_policy, theta, rollout_key)
            )
            for name, value in objective_metrics.items():
                objective_collected.setdefault(name, []).append(float(value))
            for name, value in rollout_metrics.items():
                rollout_collected.setdefault(name, []).append(float(value))
        result = {
            f"objective_{name}": float(np.mean(values))
            for name, values in objective_collected.items()
        }
        result.update(
            {
                f"rollout_{name}": float(np.mean(values))
                for name, values in rollout_collected.items()
            }
        )
        return result

    initial_trajectory = rollout_single(
        policy,
        fixed_theta,
        cfg.seed + 700_000,
        frozen_transport,
        transport_cfg,
        replace(cfg, design_exploration_std=0.0),
    )
    plot_policy_rollout(
        initial_trajectory,
        transport_cfg,
        cfg,
        run_dir / "plots" / "fixed_rollout_before_training.png",
        f"{cfg.objective_mode}: fixed rollout before training",
    )
    initial_metrics = evaluate(policy, eval_loader, cfg.seed + 100_000)
    print("Initial validation metrics:", initial_metrics)

    history: dict[str, list[float]] = {
        "step_loss": [],
        "step_energy": [],
        "step_distillation": [],
        "step_teacher_gain": [],
        "step_grad_norm": [],
        "epoch_train_loss": [],
        "epoch_val_objective": [],
        "epoch_val_rollout_energy": [],
        "epoch_val_rollout_logdet": [],
        "epoch_val_rollout_rmse": [],
    }
    visualisation_epochs = sorted(
        set(
            int(math.ceil(fraction * cfg.epochs / 10.0))
            for fraction in range(1, 11)
        )
    )
    best_val_loss = float("inf")
    best_epoch = 0
    global_step = 0
    training_started_at = time.time()
    train_key = jax.random.key(cfg.seed + 1)

    for epoch in range(1, cfg.epochs + 1):
        epoch_started_at = time.time()
        train_losses_this_epoch: list[float] = []
        epoch_iterator = (
            iter(train_loader)
            if cfg.data_mode == "finite"
            else islice(iter(train_loader), cfg.steps_per_epoch)
        )
        progress = tqdm(
            enumerate(epoch_iterator, start=1),
            total=steps_per_epoch,
            desc=f"{cfg.objective_mode} epoch {epoch:03d}/{cfg.epochs:03d}",
            dynamic_ncols=True,
            leave=True,
        )
        for _, theta_np in progress:
            train_key, step_key = jax.random.split(train_key)
            policy, opt_state, metrics, grad_norm = train_step(
                policy, opt_state, jnp.asarray(theta_np), step_key
            )
            host = {name: float(value) for name, value in jax.device_get(metrics).items()}
            host_grad_norm = float(jax.device_get(grad_norm))
            global_step += 1
            train_losses_this_epoch.append(host["loss"])
            history["step_loss"].append(host["loss"])
            history["step_energy"].append(host["terminal_energy_score"])
            history["step_distillation"].append(host["distillation_mse"])
            history["step_teacher_gain"].append(host["teacher_expected_gain"])
            history["step_grad_norm"].append(host_grad_norm)
            if cfg.objective_mode == "simple_energy_score":
                progress.set_postfix(
                    loss=f"{host['loss']:.4f}",
                    energy=f"{host['terminal_energy_score']:.4f}",
                    rmse=f"{host['terminal_mean_rmse']:.3f}",
                )
            else:
                progress.set_postfix(
                    loss=f"{host['loss']:.4f}",
                    gain=f"{host['teacher_expected_gain']:.3f}",
                    target=f"{host['teacher_target_distance']:.3f}",
                )

        epoch_train_loss = float(np.mean(train_losses_this_epoch))
        val_metrics = evaluate(policy, eval_loader, cfg.seed + 100_000)
        val_objective = val_metrics["objective_loss"]
        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_val_objective"].append(val_objective)
        history["epoch_val_rollout_energy"].append(
            val_metrics["rollout_terminal_energy_score"]
        )
        history["epoch_val_rollout_logdet"].append(
            val_metrics["rollout_terminal_logdet"]
        )
        history["epoch_val_rollout_rmse"].append(
            val_metrics["rollout_terminal_mean_rmse"]
        )

        save_policy_model(run_dir / "artefacts" / "model_last.eqx", policy)
        eqx.tree_serialise_leaves(
            run_dir / "artefacts" / "training_state_last.eqx",
            (policy, opt_state, train_key),
        )
        if epoch % cfg.save_every_epochs == 0:
            save_policy_model(
                run_dir / "artefacts" / f"model_epoch_{epoch:04d}.eqx", policy
            )
        if val_objective < best_val_loss:
            best_val_loss = val_objective
            best_epoch = epoch
            save_policy_model(run_dir / "artefacts" / "model_best.eqx", policy)

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
                "best_val_objective": best_val_loss,
                "elapsed_seconds": time.time() - training_started_at,
                "objective_mode": cfg.objective_mode,
                "frozen_inference_run": str(inference_run_dir),
            },
        )

        print(
            f"Epoch {epoch:03d}: train={epoch_train_loss:.6f} | "
            f"val objective={val_objective:.6f} | "
            f"rollout energy={val_metrics['rollout_terminal_energy_score']:.6f} | "
            f"rollout logdet={val_metrics['rollout_terminal_logdet']:.5f} | "
            f"rollout RMSE={val_metrics['rollout_terminal_mean_rmse']:.5f} | "
            f"{time.time() - epoch_started_at:.1f}s"
        )

        if epoch in visualisation_epochs:
            trajectory = rollout_single(
                policy,
                fixed_theta,
                cfg.seed + 700_000,
                frozen_transport,
                transport_cfg,
                replace(cfg, design_exploration_std=0.0),
            )
            plot_policy_rollout(
                trajectory,
                transport_cfg,
                cfg,
                run_dir / "plots" / f"fixed_rollout_epoch_{epoch:04d}.png",
                f"{cfg.objective_mode}: fixed rollout after epoch {epoch}",
            )

    best_policy = load_policy_model(
        run_dir / "artefacts" / "model_best.eqx",
        transport_cfg,
        cfg,
        key=jax.random.key(0),
    )
    final_metrics = evaluate(best_policy, eval_loader, cfg.seed + 100_000)
    final_trajectory = rollout_single(
        best_policy,
        fixed_theta,
        cfg.seed + 700_000,
        frozen_transport,
        transport_cfg,
        replace(cfg, design_exploration_std=0.0),
    )
    plot_policy_rollout(
        final_trajectory,
        transport_cfg,
        cfg,
        run_dir / "plots" / "fixed_rollout_best_model.png",
        f"{cfg.objective_mode}: best policy (epoch {best_epoch})",
    )

    epochs = np.arange(1, len(history["epoch_train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    ax.plot(epochs, history["epoch_train_loss"], label="training objective")
    ax.plot(epochs, history["epoch_val_objective"], label="validation objective")
    ax.plot(
        epochs,
        history["epoch_val_rollout_energy"],
        label="validation terminal energy score",
    )
    ax.set_xlabel("epoch")
    ax.set_title(f"Stage-II training: {cfg.objective_mode}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(run_dir / "plots" / "training_curves.png", dpi=170)
    display(fig)
    plt.close(fig)

    save_config_yaml(
        cfg,
        run_dir / "config.yaml",
        extra={
            "training_complete": True,
            "stage": 2,
            "objective_mode_resolved": cfg.objective_mode,
            "best_epoch": best_epoch,
            "best_val_objective": best_val_loss,
            "final_validation_metrics": final_metrics,
            "inference_run_dir_resolved": str(inference_run_dir),
            "inference_checkpoint_resolved": str(inference_checkpoint),
        },
    )
    print("Best epoch:", best_epoch)
    print("Final validation metrics:", final_metrics)
    print("Saved policy artefacts under:", run_dir)
    return run_dir


#%% 9) Top-level driver: resolve Stage I and train one or both variants
def main(policy_cfg: PolicyConfig = POLICY_CFG):
    print("JAX devices:", jax.devices())
    if policy_cfg.inference_run_dir is None:
        inference_run_dir = find_latest_run(
            policy_cfg.runs_base, policy_cfg.inference_env_name
        )
    else:
        inference_run_dir = Path(policy_cfg.inference_run_dir).expanduser().resolve()

    with (inference_run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        inference_config_payload = yaml.safe_load(handle)
    transport_cfg = dataclass_from_dict(
        BayesTransportConfig, inference_config_payload
    )
    inference_checkpoint = (
        inference_run_dir
        / "artefacts"
        / policy_cfg.inference_checkpoint_name
    )
    if not inference_checkpoint.is_file():
        raise FileNotFoundError(f"Missing Stage-I checkpoint: {inference_checkpoint}")
    if policy_cfg.horizon > transport_cfg.max_context_pairs:
        raise ValueError(
            "Policy horizon exceeds the Stage-I context capacity: "
            f"horizon={policy_cfg.horizon} > "
            f"max_context_pairs={transport_cfg.max_context_pairs}."
        )
    if policy_cfg.num_prior_particles != transport_cfg.num_particles:
        print(
            "WARNING: the particle-set architecture accepts variable N, but Stage I "
            "was trained with",
            transport_cfg.num_particles,
            "particles and Stage II requests",
            policy_cfg.num_prior_particles,
            ". Matching them is recommended.",
        )

    frozen_transport = load_transport_model(
        inference_checkpoint, transport_cfg, key=jax.random.key(0)
    )
    print("Frozen Stage-I run:", inference_run_dir)
    print("Transport configuration:\n", yaml.safe_dump(asdict(transport_cfg), sort_keys=False))

    mode = policy_cfg.objective_mode.lower()
    if mode == "both":
        variants = ["simple_energy_score", "self_distilled_eig"]
    elif mode in {"simple_energy_score", "self_distilled_eig"}:
        variants = [mode]
    else:
        raise ValueError(
            "objective_mode must be one of: both, simple_energy_score, "
            "self_distilled_eig."
        )

    run_dirs = []
    for variant_index, variant in enumerate(variants):
        variant_cfg = replace(
            policy_cfg,
            objective_mode=variant,
            seed=policy_cfg.seed + 10_000 * variant_index,
        )
        run_dirs.append(
            train_variant(
                variant_cfg,
                transport_cfg,
                frozen_transport,
                inference_run_dir,
                inference_checkpoint,
            )
        )
    print("Completed Stage-II runs:")
    for run_dir in run_dirs:
        print(" -", run_dir)
    return run_dirs


if __name__ == "__main__":
    main()
