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
    then minimise an energy-score objective for the sample approximation to
    p(theta | D_t). The energy score is proper for sample distributions and does not
    require evaluating q_phi(theta | D_T) or p(y | theta, x). By default this loss is
    summed over every step's posterior along the rollout, not just the terminal one
    (see CHANGE markers below).

``self_distilled_eig``
    At a random posterior state, generate a candidate design set.  The frozen
    approximate Bayes operator evaluates each candidate by simulator-only posterior
    contraction under theta~ ~ T_phi(Z,D_t) and y~p(y | theta~,x).  A soft best-design
    target is formed from the estimated information gains and the policy is distilled
    toward it with one mean-squared-error objective.  No ground-truth theta is used in
    the teacher target once the current context has been generated.

``objective_mode`` selects which variant(s) to train. The default is now
``simple_energy_score`` (see CHANGE marker in POLICY_CFG below); set it to
``self_distilled_eig`` or ``both`` to restore the other behaviours. At deployment,
either policy sees only approximate posterior particles and (optionally) the current
step fraction.
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

## JAX stop at NaN
# jax.config.update("jax_debug_nans", True)


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
    source_log_signal_np,  # CHANGE: needed for the new sensor-field rollout panel
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

    # CHANGE (reversible): step-fraction conditioning is now config-controlled.
    # True reproduces the original behaviour (the policy always sees t). Set to
    # False to train a policy that ignores the step index entirely.
    condition_on_step: bool = True

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
    # CHANGE (reversible): default objective is now the (step-summed) energy-score
    # policy instead of "both". Set back to "both" or "self_distilled_eig" to
    # restore the earlier default.
    objective_mode="simple_energy_score",  
    horizon=16,
    num_prior_particles=64,
    epochs=100,
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
    # CHANGE (reversible): step conditioning is now a static config flag rather than
    # unconditional. The step_encoder weights still exist (so checkpoints stay
    # shape-compatible whichever way the flag is set) but are simply unused when
    # condition_on_step=False.
    condition_on_step: bool = eqx.field(static=True)

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
        self.condition_on_step = policy_cfg.condition_on_step  # CHANGE

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
        # CHANGE (reversible): step-fraction conditioning is now gated by
        # self.condition_on_step (set from policy_cfg.condition_on_step). The
        # original code unconditionally added this term:
        #     tokens = tokens + self.step_encoder(jnp.asarray([step_fraction]))[None, :]
        if self.condition_on_step:
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
    # pairwise_distance = jnp.mean(
    #     jnp.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1)
    # )

    diff = samples[:, None, :] - samples[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    n = samples.shape[0]
    mask = 1.0 - jnp.eye(n)
    pairwise_distance = jnp.sum(jnp.sqrt(sq_dist + 1e-12) * mask) / jnp.sum(mask)

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
    """Energy-score objective for a batch of true parameters, summed over steps.

    CHANGE (reversible): the loss is now the mean-over-batch of the *sum over every
    rollout step* of the per-step energy score, instead of only the terminal step's
    energy score. Set SUM_LOSS_OVER_STEPS=False below to restore the original
    terminal-only objective (`loss = jnp.mean(terminal_energy)`).
    """
    SUM_LOSS_OVER_STEPS = True  # CHANGE (reversible): toggle to restore old behaviour

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
    # CHANGE: collect the per-step energy score instead of only keeping it for the
    # final step.
    step_energy_scores = []

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

        # CHANGE: score this step's posterior and stash it for the (optional) sum.
        step_energy = jax.vmap(
            lambda particles, theta: energy_score_single(
                particles, theta, transport_cfg
            )
        )(posterior, theta_true)
        step_energy_scores.append(step_energy)

    stacked_step_energy = jnp.stack(step_energy_scores, axis=0)  # [horizon, batch]
    terminal_energy = stacked_step_energy[-1]

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

    if SUM_LOSS_OVER_STEPS:
        # CHANGE: sum the energy score across all horizon steps (per sample), then
        # average over the batch.
        summed_energy_per_sample = jnp.sum(stacked_step_energy, axis=0)
        loss = jnp.mean(summed_energy_per_sample)
    else:
        # Original terminal-only objective, kept for easy reversion.
        loss = jnp.mean(terminal_energy)

    metrics = {
        "loss": loss,
        "terminal_energy_score": jnp.mean(terminal_energy),
        "mean_step_energy_score": jnp.mean(stacked_step_energy),  # CHANGE: new diagnostic
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
    max_displayed_stages: int = 8,  # CHANGE: cap panel columns instead of scaling fonts/sizes down
):
    theta_true = trajectory["theta_true"]
    all_points = [theta_true.reshape(-1, 2), trajectory["designs"].reshape(-1, 2)]
    all_points.extend(p.reshape(-1, 2) for p in trajectory["posteriors"])
    shared = np.concatenate(all_points, axis=0)
    lim = max(
        3.0 * transport_cfg.prior_std,
        1.15 * float(np.quantile(np.abs(shared), 0.995)),
    )

    # CHANGE: pick at most `max_displayed_stages` stages out of [0, horizon], always
    # including stage 0 and the final stage, evenly spaced in between. This replaces
    # the previous approach of squeezing every single stage into the figure.
    n_total_stages = policy_cfg.horizon + 1
    n_display = min(max_displayed_stages, n_total_stages)
    displayed_stages = sorted(
        set(int(round(i)) for i in np.linspace(0, n_total_stages - 1, n_display))
    )
    n_cols = len(displayed_stages)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(3.1 * n_cols, 7.0),
        squeeze=False,
        constrained_layout=True,
    )

    for col, stage in enumerate(displayed_stages):
        posterior_ax = axes[0, col]
        posterior = trajectory["posteriors"][stage]
        posterior_ax.scatter(
            posterior[..., 0].reshape(-1),
            posterior[..., 1].reshape(-1),
            s=13,
            alpha=0.30,
            label=f"samples for p(theta | D_{stage})",
        )
        posterior_ax.scatter(
            theta_true[:, 0], theta_true[:, 1], marker="*", s=170,
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
        posterior_ax.set_title(f"$p(\\theta \\mid D_{{{stage}}})$", fontsize=11)
        if col in {0, n_cols - 1}:
            posterior_ax.legend(fontsize=7, loc="upper right")

        field_ax = axes[1, col]
        if stage == 0:
            field_ax.axis("off")
            field_ax.text(
                0.5, 0.5, "No observations yet\n$D_0$ = empty set",
                ha="center", va="center", fontsize=9,
                transform=field_ax.transAxes,
            )
        else:
            grid = np.linspace(-lim, lim, transport_cfg.grid_size)
            gx, gy = np.meshgrid(grid, grid)
            x_grid = np.stack([gx, gy], axis=-1)
            field = source_log_signal_np(theta_true, x_grid, transport_cfg)

            history_designs = trajectory["designs"][:stage]
            history_obs = trajectory["observations"][:stage]
            vmin = min(field.min(), history_obs.min())
            vmax = max(field.max(), history_obs.max())

            contour = field_ax.contourf(
                gx, gy, field, levels=26, cmap="magma", vmin=vmin, vmax=vmax
            )
            field_ax.scatter(
                history_designs[:, 0], history_designs[:, 1],
                c=history_obs, cmap="magma", vmin=vmin, vmax=vmax,
                s=85, marker="s", edgecolors="white", linewidths=1.0,
                label="observed (x_i, y_i)",
            )
            field_ax.scatter(
                theta_true[:, 0], theta_true[:, 1], marker="*", s=170,
                color="white", edgecolors="black", linewidths=0.7, label="theta",
            )
            field_ax.set_xlim(-lim, lim)
            field_ax.set_ylim(-lim, lim)
            field_ax.set_aspect("equal")
            field_ax.set_title(f"Sensor field, $t={stage}$", fontsize=11)
            if col == n_cols - 1:
                fig.colorbar(
                    contour, ax=field_ax, shrink=0.75,
                    label="log E[y | theta, x]",
                )
                field_ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(destination, dpi=170)
    display(fig)
    plt.close(fig)

    # Full-resolution metrics line plot below is unaffected by the subsampling above.
    steps = np.arange(policy_cfg.horizon + 1)
    energy = trajectory["energy_scores"]
    logdet = trajectory["logdets"]

    fig, ax_energy = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax_logdet = ax_energy.twinx()

    color_energy = "#1f77b4"
    color_logdet = "#d62728"

    ax_energy.plot(
        steps, energy, marker="o", markersize=6, linewidth=2.2,
        color=color_energy, markerfacecolor="white", markeredgewidth=1.8,
        label="posterior energy score",
    )
    ax_logdet.plot(
        steps, logdet, marker="s", markersize=6, linewidth=2.2,
        color=color_logdet, markerfacecolor="white", markeredgewidth=1.8,
        linestyle="--",
        label="posterior log-volume proxy",
    )

    for ax, values, color in ((ax_energy, energy, color_energy), (ax_logdet, logdet, color_logdet)):
        ax.scatter([steps[0], steps[-1]], [values[0], values[-1]],
                   s=70, color=color, zorder=5, edgecolor="white", linewidth=1.2)
    ax_energy.annotate(
        f"{energy[0]:.2f} → {energy[-1]:.2f}",
        xy=(steps[-1], energy[-1]), xytext=(6, 8), textcoords="offset points",
        fontsize=9, color=color_energy, fontweight="bold",
    )
    ax_logdet.annotate(
        f"{logdet[0]:.2f} → {logdet[-1]:.2f}",
        xy=(steps[-1], logdet[-1]), xytext=(6, -14), textcoords="offset points",
        fontsize=9, color=color_logdet, fontweight="bold",
    )

    ax_energy.set_xlabel("number of acquired design–outcome pairs  $|D_t|$", fontsize=10.5)
    ax_energy.set_ylabel("posterior energy score", fontsize=10.5, color=color_energy)
    ax_logdet.set_ylabel("posterior log-volume proxy", fontsize=10.5, color=color_logdet)
    ax_energy.tick_params(axis="y", labelcolor=color_energy)
    ax_logdet.tick_params(axis="y", labelcolor=color_logdet)
    if policy_cfg.horizon <= 20:
        ax_energy.set_xticks(steps)

    ax_energy.set_title(
        "Posterior contraction along the policy rollout",
        fontsize=13, fontweight="bold", loc="left", pad=12,
    )
    ax_energy.text(
        0.0, 1.02,
        r"lower is tighter/closer to $\theta$  ·  $p(\theta \mid D_t)$",
        transform=ax_energy.transAxes, fontsize=9, color="#666666", style="italic",
    )

    ax_energy.grid(alpha=0.2)
    ax_energy.spines[["top"]].set_visible(False)
    ax_logdet.spines[["top"]].set_visible(False)

    lines1, labels1 = ax_energy.get_legend_handles_labels()
    lines2, labels2 = ax_logdet.get_legend_handles_labels()
    ax_energy.legend(
        lines1 + lines2, labels1 + labels2,
        loc="upper right", fontsize=9, frameon=True, framealpha=0.9,
    )

    metrics_path = destination.with_name(destination.stem + "_metrics.png")
    fig.savefig(metrics_path, dpi=170)
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
    # eqx.tree_pprint(policy)

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
            (policy, opt_state, jax.random.key_data(train_key))
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

    # model_to_load = "model_best.eqx" if best_epoch > 0 else "model_last.eqx"

    best_policy = load_policy_model(
        run_dir / "artefacts" / "model_last.eqx",
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
    # CHANGE: replaced the single flat line plot with a unified diagnostics figure
    # matching the Stage-I style — per-step metrics on top, per-epoch curves below.
    steps = np.arange(1, len(history["step_loss"]) + 1)
    epochs = np.arange(1, len(history["epoch_train_loss"]) + 1)

    fig = plt.figure(figsize=(11.0, 10.0), constrained_layout=True)
    fig.suptitle(
        f"Stage-II policy training — {cfg.objective_mode}",
        fontsize=14, fontweight="bold",
    )

    top_gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.0, 2.6])
    step_gs = top_gs[0:2].subgridspec(2, 2, hspace=0.35, wspace=0.28)

    # For self_distilled_eig, step_energy/step_teacher_gain carry the meaningful
    # signal; for simple_energy_score, step_energy/step_grad_norm do. Both are always
    # populated (zeros for the inactive objective's fields), so the same four panels
    # work for either variant.
    step_panels = [
        ("step_loss", "training loss", "#1f77b4"),
        ("step_energy", "terminal energy score", "#d62728"),
        ("step_distillation", "distillation MSE", "#2ca02c"),
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

    # Bottom block: per-epoch train/val objective + validation rollout energy score
    ax_bottom = fig.add_subplot(top_gs[2])
    # ax_bottom.set_y_scale("log")
    ax_bottom.plot(epochs, history["epoch_train_loss"], marker="o", markersize=3,
                   color="#1f77b4", label="train objective")
    ax_bottom.plot(epochs, history["epoch_val_objective"], marker="o", markersize=3,
                   color="#ff7f0e", label="val objective")
    ax_bottom.set_xlabel("epoch", fontsize=10)
    ax_bottom.set_ylabel("objective", fontsize=10)
    ax_bottom.set_title(
        "Per-epoch train / validation objective", fontsize=10, fontweight="bold", loc="left"
    )
    ax_bottom.grid(alpha=0.25)
    ax_bottom.spines[["top", "right"]].set_visible(False)
    ax_bottom.tick_params(labelsize=9)

    ax_bottom2 = ax_bottom.twinx()
    ax_bottom2.plot(
        epochs, history["epoch_val_rollout_energy"], marker="s", markersize=3,
        color="#2ca02c", linestyle=":", label="val rollout terminal energy score",
    )
    ax_bottom2.set_ylabel("validation rollout terminal energy score", fontsize=10, color="#2ca02c")
    ax_bottom2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=9)
    ax_bottom2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax_bottom.get_legend_handles_labels()
    lines2, labels2 = ax_bottom2.get_legend_handles_labels()
    ax_bottom.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right", frameon=False)

    plt.tight_layout()

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
print("JAX devices:", jax.devices())
if POLICY_CFG.inference_run_dir is None:
    inference_run_dir = find_latest_run(
        POLICY_CFG.runs_base, POLICY_CFG.inference_env_name
    )
else:
    inference_run_dir = Path(POLICY_CFG.inference_run_dir).expanduser().resolve()

with (inference_run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
    inference_config_payload = yaml.safe_load(handle)
transport_cfg = dataclass_from_dict(
    BayesTransportConfig, inference_config_payload
)
inference_checkpoint = (
    inference_run_dir
    / "artefacts"
    / POLICY_CFG.inference_checkpoint_name
)
if not inference_checkpoint.is_file():
    raise FileNotFoundError(f"Missing Stage-I checkpoint: {inference_checkpoint}")
if POLICY_CFG.horizon > transport_cfg.max_context_pairs:
    raise ValueError(
        "Policy horizon exceeds the Stage-I context capacity: "
        f"horizon={POLICY_CFG.horizon} > "
        f"max_context_pairs={transport_cfg.max_context_pairs}."
    )
if POLICY_CFG.num_prior_particles != transport_cfg.num_particles:
    print(
        "WARNING: the particle-set architecture accepts variable N, but Stage I "
        "was trained with",
        transport_cfg.num_particles,
        "particles and Stage II requests",
        POLICY_CFG.num_prior_particles,
        ". Matching them is recommended.",
    )

frozen_transport = load_transport_model(
    inference_checkpoint, transport_cfg, key=jax.random.key(0)
)
print("Frozen Stage-I run:", inference_run_dir)
print("Transport configuration:\n", yaml.safe_dump(asdict(transport_cfg), sort_keys=False))

mode = POLICY_CFG.objective_mode.lower()
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
        POLICY_CFG,
        objective_mode=variant,
        seed=POLICY_CFG.seed + 10_000 * variant_index,
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




#%% 10) Concentration animation: watch posterior samples collapse onto theta
# CHANGE: new cell. Renders how the approximate posterior p(theta | D_t) particles
# gradually concentrate around the true theta as the policy acquires more
# design-outcome pairs, and saves the result as a GIF via matplotlib's PillowWriter.
import matplotlib.animation as mpl_animation
from IPython.display import Image


def make_concentration_gif(
    trajectory: dict[str, np.ndarray],
    transport_cfg: BayesTransportConfig,
    destination: Path,
    title_prefix: str,
    fps: int = 2,
):
    """Animate p(theta | D_t) particles concentrating on theta as t grows."""
    theta_true = trajectory["theta_true"]
    posteriors = trajectory["posteriors"]
    all_points = np.concatenate(
        [p.reshape(-1, 2) for p in posteriors] + [theta_true.reshape(-1, 2)], axis=0
    )
    lim = max(
        3.0 * transport_cfg.prior_std,
        1.15 * float(np.quantile(np.abs(all_points), 0.995)),
    )

    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)
    scatter = ax.scatter([], [], s=20, alpha=0.4, color="#1f77b4", label="posterior samples")
    ax.scatter(
        theta_true[:, 0], theta_true[:, 1], marker="*", s=220,
        color="#d62728", label="theta (truth)", zorder=5,
    )
    if trajectory["designs"].size:
        ax.plot(
            trajectory["designs"][:, 0], trajectory["designs"][:, 1],
            color="grey", marker="x", linewidth=1.0, alpha=0.5,
            label="design history", zorder=2,
        )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")
    title = ax.set_title(f"{title_prefix} | t=0  (|D_t|=0)")

    def update(frame_index: int):
        particles = posteriors[frame_index].reshape(-1, 2)
        scatter.set_offsets(particles)
        title.set_text(f"{title_prefix} | t={frame_index}  (|D_t|={frame_index})")
        return scatter, title

    anim = mpl_animation.FuncAnimation(
        fig, update, frames=len(posteriors), interval=600, blit=False,
    )
    anim.save(destination, writer=mpl_animation.PillowWriter(fps=fps))
    plt.close(fig)
    print("Saved concentration animation to:", destination)


# Reload the best checkpoint from the first Stage-II run just trained and animate its
# fixed-theta rollout. Assumes `run_dirs` exists from the `main()` cell above.
if "run_dirs" in globals() and run_dirs:
    animation_run_dir = run_dirs[0]

    with (animation_run_dir / "config.yaml").open("r", encoding="utf-8") as handle:
        animation_cfg_payload = yaml.safe_load(handle)
    animation_policy_cfg = dataclass_from_dict(PolicyConfig, animation_cfg_payload)

    inference_run_dir_for_anim = Path(animation_cfg_payload["inference_run_dir_resolved"])
    with (inference_run_dir_for_anim / "config.yaml").open("r", encoding="utf-8") as handle:
        transport_cfg_payload = yaml.safe_load(handle)
    animation_transport_cfg = dataclass_from_dict(BayesTransportConfig, transport_cfg_payload)

    animation_frozen_transport = load_transport_model(
        animation_run_dir / "artefacts" / "frozen_bayes_transport.eqx",
        animation_transport_cfg,
        key=jax.random.key(0),
    )
    animation_policy = load_policy_model(
        animation_run_dir / "artefacts" / "model_best.eqx",
        animation_transport_cfg,
        animation_policy_cfg,
        key=jax.random.key(0),
    )
    fixed_theta_for_anim = np.load(animation_run_dir / "artefacts" / "fixed_theta.npy")

    animation_trajectory = rollout_single(
        animation_policy,
        fixed_theta_for_anim,
        animation_policy_cfg.seed + 700_000,
        animation_frozen_transport,
        animation_transport_cfg,
        replace(animation_policy_cfg, design_exploration_std=0.0),
    )
    make_concentration_gif(
        animation_trajectory,
        animation_transport_cfg,
        animation_run_dir / "plots" / "posterior_concentration.gif",
        f"{animation_policy_cfg.objective_mode}: posterior concentration",
    )

    ## Display the gif here
    gif_path = animation_run_dir / "plots" / "posterior_concentration.gif"
    if gif_path.is_file():
        display(Image(filename=str(gif_path))) 
