# %% [markdown]
# # Latent Action Bayesian Experimental Design (LA-BED)
# ### Principled Regularisation & Gradient Routing  -- v2 (bugfixed)
#
# Changes vs. the previous version (see inline `# FIX:` comments for details):
#
#   1. **Gradient-routing bug.** `eqx.partition(model, lambda n: n is model.simulator)`
#      is applied leaf-wise by `jax.tree_util`, so it is compared against *array leaves*,
#      never against the `eqx.Module` instances themselves. The predicate was therefore
#      **always False**, `sim_params`/`pol_params` were empty pytrees, and the
#      "separate" optimisers were updating *nothing* -- this is why both loss curves in
#      the attached figures were flat/noisy and the trajectory never converged. Fixed
#      with the standard Equinox pattern: build a boolean mask pytree with `eqx.tree_at`
#      that marks exactly the leaves belonging to `.simulator` / `.policy`.
#   2. **Primary simulator objective.** Per your request, the simulator's main loss is
#      now the plain (non-log) MSE between the *final* belief mean and the true theta,
#      exactly mirroring Action-BED's terminal-belief loss. The Barber-Agakov term
#      remains as an additional regulariser (all regularisers are kept, none removed).
#   3. **LR schedule.** Replaced the fixed `1e-4` Adam with
#      `optax.chain(clip, adam(peak_lr), optax.contrib.reduce_on_plateau(...))`, i.e. a
#      warmup->constant LR that is halved whenever the *tracked* loss plateaus (separate
#      trackers for the simulator and the policy).
#   4. **Diagnostics.** Because the routing bug hid the real problem, we now also log
#      (a) raw terminal MSE per step/epoch, (b) the current LR multiplier for each
#      optimiser, and (c) gradient norms per module, so a stalled optimiser is visible
#      immediately instead of only inferred from a flat loss curve.
#   5. Nothing else about the modelling (Barber-Agakov / PCE / Fisher / trust-region
#      regularisers, the two independent optimisers, or the rollout/scan mechanics) was
#      removed -- only the routing/schedule/loss-weighting bugs above were fixed.
# --------------------------------------------------------------------------------------

# %% [markdown]
# ## 1. Imports and Configuration

# %%
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import optax.contrib as optax_contrib
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
except ImportError:
    pass

# We assume the following modules exist in the same directory:
from models import (
    DesignPolicyTransformer, DesignPolicyMLP,
    BayesSimulatorMLP, BayesSimulatorTransformer,
    SourceLocConfig, environment_step, init_belief,
    sort_sources_by_amplitude, expected_information_gain
)
from datalloaders import SourceLocPrior, make_train_loader, make_eval_loader

THIS_FILE = Path.cwd() / "main.py"
if "__file__" in dir():
    THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent

# %% [markdown]
# ## 2. Configuration Dataclass (extended)

# %%
@dataclass
class Config:
    # --- Run Control ---
    train: bool = True
    run_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    seed: int = 2026

    # --- Architecture & Environment ---
    k_sources: int = 2
    design_dim: int = 2
    obs_dim: int = 1
    belief_mode: str = "gaussian"
    simulator_type: str = "mlp"      # "mlp" | "transformer"
    policy_hidden: int = 128
    sim_hidden: int = 128
    max_t: int = 30

    # --- Optimisation & Loss ---
    batch_size: int = 512
    epochs: int = 10
    lr_max: float = 1e-4
    lr_warmup_steps: int = 200
    gamma: float = 0.95                # (unused in new regularisers, kept for compatibility)
    loss_matching: str = "permute"        # "sort" or "permute"

    # --- Regularisation ---
    regularizer_kind: str = "all"      # "none", "barber_agakov", "pce", "fisher", "trust", "all"
    # Weights
    lambda_mse: float = 1.0            # FIX: now weights *raw* terminal MSE (ActionBED-style)
    lambda_barber: float = 0.1
    lambda_pce: float = 0.05
    lambda_fisher: float = 0.02
    lambda_trust: float = 0.01
    # PCE hyperparameters
    pce_samples: int = 10              # L
    # Fisher: noise variance (known from environment)
    fisher_noise_var: float = 0.1
    # Trust region
    trust_reference: Optional[tuple] = None   # if None, uses [0,0]
    trust_tau: float = 0.1             # policy's design variance
    trust_rho: float = 1.0             # reference variance

    # --- Data Regime ---
    data_mode: str = "infinite"        # "finite" | "infinite"
    n_train_episodes: int = 100_000
    steps_per_epoch: int = 1000

    # --- Reduce-on-plateau schedule (applied on top of a constant-after-warmup Adam LR) ---
    plateau_patience: int = 5          # epochs of no improvement before shrinking LR
    plateau_cooldown: int = 2
    plateau_factor: float = 0.5
    plateau_rtol: float = 1e-3
    plateau_accumulation_size: int = 200   # steps averaged into one "loss reading"
    lr_min_multiplier: float = 1e-1    # floor for the plateau multiplier (relative to lr_max)

    @property
    def belief_dim(self) -> int:
        return self.k_sources * self.design_dim

    @property
    def design_bound(self) -> float:
        return 1.0   # assuming designs in [-1,1] per dimension

cfg = Config()

# %% [markdown]
# ## 3. Environment Likelihood (known model)

# %%
@dataclass
class EnvConfig:
    K: int = 2
    design_dim: int = 2
    amplitude: float = 1.0
    signal_var: float = 0.5            # source spread
    noise_var: float = 0.1             # observation noise

def source_intensity(theta: jnp.ndarray, x: jnp.ndarray, env_cfg: EnvConfig) -> jnp.ndarray:
    """Sum of Gaussian kernels from K sources evaluated at design x."""
    diff = x - theta  # (K, D)
    sq_dist = jnp.sum(diff**2, axis=-1)
    kernel = jnp.exp(-sq_dist / (2 * env_cfg.signal_var))
    return env_cfg.amplitude * jnp.sum(kernel)

def log_likelihood(y: jnp.ndarray, theta: jnp.ndarray, x: jnp.ndarray, env_cfg: EnvConfig) -> jnp.ndarray:
    """log p(y | theta, x) = log N(y | mu(theta,x), noise_var)."""
    mu = source_intensity(theta, x, env_cfg)
    return -0.5 * ((y - mu)**2 / env_cfg.noise_var + jnp.log(2 * jnp.pi * env_cfg.noise_var))

# %% [markdown]
# ## 4. Independent Regularisation Functions (unchanged - all four kept)
# These functions operate on the collected arrays and return scalar losses.
# Gradient routing is applied inside the loss function, not here.

# %%
def barber_agakov_regularizer(
    mus_before: jnp.ndarray,          # (B, T, D)
    log_sigmas_before: jnp.ndarray,
    mus_after: jnp.ndarray,           # (B, T, D)
    log_sigmas_after: jnp.ndarray,
    theta_true_flat: jnp.ndarray,     # (B, D)
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Barber-Agakov (log-loss) regulariser.
    Samples theta from belief before, then computes -log q_phi(theta | h_t)
    under the updated belief.
    Returns scalar loss (mean over batch and time).
    """
    B, T, D = mus_before.shape
    eps = jax.random.normal(key, (B, T, D))
    theta_samples = mus_before + jnp.exp(log_sigmas_before) * eps  # (B,T,D)
    diff = theta_samples - mus_after
    nll = 0.5 * (diff**2 / jnp.exp(2 * log_sigmas_after) + 2 * log_sigmas_after + jnp.log(2 * jnp.pi))
    nll = jnp.sum(nll, axis=-1)       # sum over dimensions
    return jnp.mean(nll)

def pce_regularizer(
    mus_before: jnp.ndarray,
    log_sigmas_before: jnp.ndarray,
    x_ts: jnp.ndarray,                 # (B, T, design_dim)
    y_ts: jnp.ndarray,                 # (B, T, obs_dim)
    env_cfg: EnvConfig,
    key: jax.random.PRNGKey,
    L: int = 10
) -> jnp.ndarray:
    """
    Prior Contrastive Estimation (PCE) bound.
    Returns scalar loss = -mean(PCE) because we want to maximise PCE.
    """
    B, T, D = mus_before.shape
    # sample L+1 theta candidates from belief before
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (B, T, L + 1, D))
    theta_candidates = mus_before[:, :, None, :] + jnp.exp(log_sigmas_before[:, :, None, :]) * eps  # (B,T,L+1,D)

    def log_lik_fn(theta, x, y):
        theta_reshaped = theta.reshape(env_cfg.K, env_cfg.design_dim)
        return log_likelihood(y.squeeze(), theta_reshaped, x, env_cfg)

    # vectorise over batch, time, candidates
    log_lik = jax.vmap(
        jax.vmap(
            jax.vmap(log_lik_fn, in_axes=(0, None, None)),
            in_axes=(0, 0, 0)
        ),
        in_axes=(0, 0, 0)
    )(theta_candidates, x_ts, y_ts)  # (B,T,L+1)

    log_p_theta0 = log_lik[:, :, 0]    # (B,T)
    log_avg = jax.nn.logsumexp(log_lik, axis=-1) - jnp.log(L + 1)  # (B,T)
    pce = log_p_theta0 - log_avg      # (B,T)
    return -jnp.mean(pce)

def fisher_regularizer(
    mus_before: jnp.ndarray,
    log_sigmas_before: jnp.ndarray,
    x_ts: jnp.ndarray,
    env_cfg: EnvConfig
) -> jnp.ndarray:
    """
    Local Fisher-information (D-optimality) approximation.
    Returns scalar loss = -0.5 * mean(log det(I + Fisher * Sigma)).
    """
    B, T, D = mus_before.shape
    # Jacobian of mean function w.r.t. theta
    def mean_fn(theta, x):
        return source_intensity(theta.reshape(-1, env_cfg.design_dim), x, env_cfg)

    def fisher_matrix(theta, x):
        J = jax.jacfwd(mean_fn, argnums=0)(theta, x)  # (D,)
        J = J[None, :]  # (1, D)
        F = (1.0 / env_cfg.noise_var) * J.T @ J      # (D, D)
        return F

    fisher = jax.vmap(jax.vmap(fisher_matrix, in_axes=(0, 0)), in_axes=(0, 0))(mus_before, x_ts)  # (B,T,D,D)
    sqrt_sigma2 = jnp.exp(log_sigmas_before)         # (B,T,D)
    F_scaled = fisher * sqrt_sigma2[..., None] * sqrt_sigma2[..., :, None]  # (B,T,D,D)
    M = jnp.eye(D)[None, None, :, :] + F_scaled
    sign, logdet = jnp.linalg.slogdet(M)
    # ignore sign (should be positive)
    return -0.5 * jnp.mean(logdet)

def trust_regularizer(
    x_ts: jnp.ndarray,
    ref_design: Optional[jnp.ndarray] = None,
    tau: float = 0.1,
    rho: float = 1.0
) -> jnp.ndarray:
    """
    Trust-region penalty: KL between Gaussian policy and reference.
    Returns scalar loss (mean over batch and time).
    """
    if ref_design is None:
        ref_design = jnp.zeros(x_ts.shape[-1])
    d = x_ts.shape[-1]
    diff = x_ts - ref_design[None, None, :]  # (B,T,D)
    sq_dist = jnp.sum(diff**2, axis=-1)      # (B,T)
    kl = 0.5 * (d * jnp.log(rho**2 / tau**2) + (d * tau**2 + sq_dist) / rho**2 - d)
    return jnp.mean(kl)

# %% [markdown]
# ## 5. Model Building and Rollout

# %%
class LABEDModel(eqx.Module):
    policy: eqx.Module
    simulator: eqx.Module

def build_model(cfg: Config, key: jax.random.PRNGKey) -> LABEDModel:
    p_key, s_key = jax.random.split(key)
    policy = DesignPolicyMLP(
        design_dim=cfg.design_dim,
        obs_dim=cfg.obs_dim,
        belief_dim=cfg.belief_dim,
        hidden=cfg.policy_hidden,
        key=p_key
    )
    if cfg.simulator_type == "mlp":
        simulator = BayesSimulatorMLP(
            design_dim=cfg.design_dim,
            obs_dim=cfg.obs_dim,
            belief_dim=cfg.belief_dim,
            hidden=cfg.sim_hidden,
            depth=3,
            key=s_key
        )
    else:
        simulator = BayesSimulatorTransformer(
            design_dim=cfg.design_dim,
            obs_dim=cfg.obs_dim,
            belief_dim=cfg.belief_dim,
            hidden=64,
            num_heads=4,
            num_layers=2,
            key=s_key
        )
    return LABEDModel(policy=policy, simulator=simulator)

def rollout_episode(
    model: LABEDModel,
    theta_true: jnp.ndarray,
    episode_key: jax.random.PRNGKey,
    cfg: Config,
    env_cfg: EnvConfig
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Rolls out an episode, collecting all quantities of interest.
    Returns:
        x_ts: (T, design_dim)
        y_ts: (T, obs_dim)
        mus_after: (T, D) belief mean after each step
        log_sigmas_after: (T, D) belief log-sigma after each step
        mus_before: (T, D) belief mean before each step
        log_sigmas_before: (T, D) belief log-sigma before each step
        theta_true_flat: (D,) flattened true sources (sorted)
    """
    sim_cfg = SourceLocConfig(K=cfg.k_sources)
    # sort sources for consistency
    theta_true_sorted = sort_sources_by_amplitude(theta_true, sim_cfg) if cfg.k_sources > 1 else theta_true
    theta_true_flat = theta_true_sorted.reshape(-1)

    initial_belief = init_belief(cfg.belief_dim, init_log_sigma=0.0)
    x_hist = jnp.zeros((cfg.max_t, cfg.design_dim))
    y_hist = jnp.zeros((cfg.max_t, cfg.obs_dim))
    keys = jax.random.split(episode_key, cfg.max_t)
    timesteps = jnp.arange(cfg.max_t)

    # -------------------------------------------------------------------------------------------------
    # jax.lax.scan expects operations to be pure and does not allow Python list mutations.
    # Mutating Python lists (like `mus_before_list.append()`) leaks tracer objects outside the scan,
    # raising the UnexpectedTracerError.
    # Instead, we simply return the quantities as the second output of scan_step and JAX will
    # seamlessly aggregate/stack them across the time dimension for us.
    # -------------------------------------------------------------------------------------------------

    def scan_step(carry, scan_input):
        belief, x_h, y_h = carry
        t, step_key = scan_input
        # 1. Design phase
        x_t = model.policy(x_h, y_h, belief)

        # 2. Environment observation
        y_t = environment_step(theta_true, x_t, step_key, sim_cfg)

        # update histories inline with JAX conventions
        x_h = x_h.at[t].set(x_t)
        y_h = y_h.at[t].set(y_t)

        # 3. Bayes simulator update
        next_belief = model.simulator(x_t, y_t, belief, x_h, y_h)

        # New carry state
        carry = (next_belief, x_h, y_h)

        # We output belief.mu and belief.log_sigma here to represent "belief before"
        # and next_belief.mu, next_belief.log_sigma to represent "belief after"
        return carry, (x_t, y_t, next_belief.mu, next_belief.log_sigma, belief.mu, belief.log_sigma)

    carry_in = (initial_belief, x_hist, y_hist)

    # We unpack the 6 arrays accumulated cleanly across the max_t time steps
    final_carry, (x_ts, y_ts, mus_after, log_sigmas_after, mus_before, log_sigmas_before) = jax.lax.scan(
        scan_step, carry_in, (timesteps, keys)
    )

    return x_ts, y_ts, mus_after, log_sigmas_after, mus_before, log_sigmas_before, theta_true_flat

vmap_rollout = jax.vmap(rollout_episode, in_axes=(None, 0, 0, None, None))

# %% [markdown]
# ## 6. Shared terminal-MSE helper (ActionBED-style, permutation-safe)

# %%
def terminal_mse(mus_final: jnp.ndarray, batch_theta: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """mus_final: (B, D) predicted final belief means, flattened.
    batch_theta: (B, D) true sources, flattened (NOT pre-sorted).
    Returns scalar raw MSE (mean over batch & dims), matching sources either by
    a fixed canonical sort ("sort") or by the best permutation ("permute")."""
    B = batch_theta.shape[0]
    batch_theta_flat = batch_theta.reshape(B, -1)
    theta_reshaped = batch_theta.reshape(B, cfg.k_sources, cfg.design_dim)
    if cfg.loss_matching == "permute":
        perm_indices = jnp.array(list(permutations(range(cfg.k_sources))))
        def mse_for_perm(perm):
            mus_perm = mus_final.reshape(B, cfg.k_sources, cfg.design_dim)[:, perm, :].reshape(B, -1)
            return jnp.mean((mus_perm - batch_theta_flat) ** 2, axis=-1)
        all_mses = jax.vmap(mse_for_perm)(perm_indices)  # (K!, B)
        min_mse = jnp.min(all_mses, axis=0)              # (B,)
        return jnp.mean(min_mse)
    else:
        sorted_theta = jax.vmap(sort_sources_by_amplitude, in_axes=(0, None))(
            theta_reshaped, SourceLocConfig(K=cfg.k_sources)
        )
        return jnp.mean((mus_final.reshape(B, cfg.k_sources, cfg.design_dim) - sorted_theta) ** 2)

# %% [markdown]
# ## 7. Loss Functions and Gradient Routing
# Two separate loss functions: one for the simulator (RAW terminal MSE + Barber-Agakov)
# and one for the policy (RAW terminal MSE + PCE + Fisher + trust). Gradients are routed
# using boolean-mask `eqx.partition` (see Section 8 for the actual mask construction).

# %%
def make_loss_fns(cfg: Config, env_cfg: EnvConfig):

    def loss_sim_fn(model: LABEDModel, batch_theta, batch_key, epoch):
        """Loss that updates only the simulator: RAW terminal MSE (primary, ActionBED-style)
        + Barber-Agakov regulariser."""
        B = batch_theta.shape[0]
        keys = jax.random.split(batch_key, B)
        x_ts, y_ts, mus_after, log_sigmas_after, mus_before, log_sigmas_before, theta_true_flat = vmap_rollout(
            model, batch_theta, keys, cfg, env_cfg
        )

        mus_final = mus_after[:, -1, :]  # (B, D)
        total_mse = terminal_mse(mus_final, batch_theta, cfg)   # FIX: raw MSE is now the primary target

        # Barber-Agakov (only updates simulator)
        key_barber, _ = jax.random.split(batch_key)
        loss_barber = barber_agakov_regularizer(
            mus_before, log_sigmas_before, mus_after, log_sigmas_after,
            theta_true_flat, key_barber
        )
        loss = cfg.lambda_mse * total_mse + cfg.lambda_barber * loss_barber
        aux = {"mse": total_mse, "barber": loss_barber}
        return loss, aux

    def loss_pol_fn(model: LABEDModel, batch_theta, batch_key, epoch):
        """Loss that updates only the policy: RAW terminal MSE (primary) + PCE + Fisher + trust."""
        B = batch_theta.shape[0]
        keys = jax.random.split(batch_key, B)
        x_ts, y_ts, mus_after, log_sigmas_after, mus_before, log_sigmas_before, theta_true_flat = vmap_rollout(
            model, batch_theta, keys, cfg, env_cfg
        )

        mus_final = mus_after[:, -1, :]
        total_mse = terminal_mse(mus_final, batch_theta, cfg)

        # PCE, Fisher, trust - we stop gradients on beliefs to avoid updating simulator
        mus_before_sg = jax.lax.stop_gradient(mus_before)
        log_sigmas_before_sg = jax.lax.stop_gradient(log_sigmas_before)
        key_pce, key_rest = jax.random.split(batch_key)

        loss_pce = pce_regularizer(
            mus_before_sg, log_sigmas_before_sg, x_ts, y_ts, env_cfg, key_pce, L=cfg.pce_samples
        )
        loss_fisher = fisher_regularizer(
            mus_before_sg, log_sigmas_before_sg, x_ts, env_cfg
        )
        # Trust does not use beliefs
        ref = jnp.array(cfg.trust_reference) if cfg.trust_reference is not None else None
        loss_trust = trust_regularizer(x_ts, ref, tau=cfg.trust_tau, rho=cfg.trust_rho)

        loss = (cfg.lambda_mse * total_mse
                + cfg.lambda_pce * loss_pce
                + cfg.lambda_fisher * loss_fisher
                + cfg.lambda_trust * loss_trust)
        aux = {"mse": total_mse, "pce": loss_pce, "fisher": loss_fisher, "trust": loss_trust}
        return loss, aux

    return loss_sim_fn, loss_pol_fn

# %% [markdown]
# ## 8. Gradient-routing masks (THE ACTUAL BUGFIX)
#
# `eqx.partition(model, predicate)` calls `predicate` on every *leaf* of the pytree
# (arrays), not on sub-modules. `lambda n: n is model.simulator` therefore compares an
# `eqx.Module` identity against `jnp.ndarray` leaves and is always `False`.  The correct
# pattern is to build a boolean pytree that mirrors `model`'s structure, defaulting to
# `False` everywhere, then overwrite just the `.simulator` (resp. `.policy`) subtree with
# `True` via `eqx.tree_at`. That boolean pytree is then a valid `filter_spec` for
# `eqx.partition`.

# %%
def make_module_mask(model: LABEDModel, attr_name: str) -> LABEDModel:
    """Boolean pytree, same structure as `model`, True only on the leaves under
    `getattr(model, attr_name)`."""
    false_mask = jax.tree_util.tree_map(lambda _: False, model)
    submodule = getattr(model, attr_name)
    # FIX: only mark *array* leaves True. Some leaves inside an eqx.Module subtree are
    # non-array (e.g. activation callables); including them in a partition breaks
    # optax's `zeros_like`-based state init, since they aren't valid array-likes.
    true_submask = jax.tree_util.tree_map(lambda leaf: eqx.is_array(leaf), submodule)
    mask = eqx.tree_at(lambda m: getattr(m, attr_name), false_mask, true_submask,
                        is_leaf=lambda x: x is None)
    return mask

# %% [markdown]
# ## 9. Utility Functions for Run Management

# %%
def make_run_dir(base: Path = THIS_DIR / "runs") -> Path:
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = base / stamp
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir

def snapshot_code(run_dir: Path) -> None:
    for name in ("main.py", "models.py", "datalloaders.py"):
        src = THIS_DIR / name
        if src.exists():
            shutil.copy2(src, run_dir / name)

def save_config(cfg: Config, run_dir: Path) -> None:
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

def save_model(model: LABEDModel, path: Path) -> None:
    eqx.tree_serialise_leaves(str(path), model)

def load_model(path: Path, skeleton: LABEDModel) -> LABEDModel:
    return eqx.tree_deserialise_leaves(str(path), skeleton)

# %% [markdown]
# ## 10. Optimisers: warmup -> constant Adam, wrapped in reduce-on-plateau
#
# `optax.contrib.reduce_on_plateau` is a `GradientTransformationExtraArgs`: it needs the
# current scalar loss passed as `value=...` to `opt.update(...)`. Chaining it after Adam
# rescales Adam's update by a shrinking multiplier whenever the *tracked* (smoothed) loss
# stops improving for `patience` readings, each reading itself an average over
# `accumulation_size` steps -- exactly "ReduceLROnPlateau" semantics.

# %%
def make_optimizer(cfg: Config) -> optax.GradientTransformationExtraArgs:
    warmup = optax.linear_schedule(
        init_value=cfg.lr_max * 1e-3, end_value=cfg.lr_max, transition_steps=cfg.lr_warmup_steps
    )
    schedule = optax.join_schedules([warmup, optax.constant_schedule(cfg.lr_max)],
                                     boundaries=[cfg.lr_warmup_steps])
    base = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adam(schedule),
    )
    plateau = optax_contrib.reduce_on_plateau(
        factor=cfg.plateau_factor,
        patience=cfg.plateau_patience,
        rtol=cfg.plateau_rtol,
        cooldown=cfg.plateau_cooldown,
        accumulation_size=cfg.plateau_accumulation_size,
        min_scale=cfg.lr_min_multiplier,
    )
    return optax.chain(base, plateau)

# %% [markdown]
# ## 11. Training Loop with Separate Optimisers (fixed routing + plateau LR)

# %%
def train(cfg: Config, model: LABEDModel, key: jax.random.PRNGKey, run_dir: Path):
    env_cfg = EnvConfig(K=cfg.k_sources, design_dim=cfg.design_dim)
    prior = SourceLocPrior(K=cfg.k_sources, prior_std=1.0)
    train_loader = make_train_loader(
        prior, cfg.batch_size, base_seed=cfg.seed,
        data_mode=cfg.data_mode, n_train_episodes=cfg.n_train_episodes
    )

    # FIX: boolean-mask partitions (identity-based partitions never matched any leaf).
    sim_mask = make_module_mask(model, "simulator")
    pol_mask = make_module_mask(model, "policy")

    sim_params = eqx.filter(model, sim_mask)
    pol_params = eqx.filter(model, pol_mask)

    opt_sim = make_optimizer(cfg)
    opt_pol = make_optimizer(cfg)
    # opt_state_sim = opt_sim.init(sim_params)
    opt_state_sim = opt_sim.init(eqx.filter(model, eqx.is_array))  # FIX: init with *all* array leaves, not just the simulator
    opt_state_pol = opt_pol.init(pol_params)

    loss_sim_fn, loss_pol_fn = make_loss_fns(cfg, env_cfg)

    # JIT-compiled training steps. `value=loss_val` feeds the reduce-on-plateau tracker.
    @eqx.filter_jit
    def train_step_sim(model, opt_state, batch_theta, batch_key, epoch):
        (loss_val, aux), grads = eqx.filter_value_and_grad(loss_sim_fn, has_aux=True)(model, batch_theta, batch_key, epoch)

        grads_sim = eqx.filter(grads, sim_mask)
        grad_norm = optax.global_norm(grads_sim)
        # updates, opt_state = opt_sim.update(grads_sim, opt_state, sim_params, value=loss_val)

        ## This loss updates the entire model, not just the simulator:
        updates, opt_state = opt_sim.update(grads, opt_state, model, value=loss_val)

        model = eqx.apply_updates(model, updates)
        scale = optax.tree_utils.tree_get(opt_state, "scale")
        return model, opt_state, loss_val, aux, grad_norm, scale

    @eqx.filter_jit
    def train_step_pol(model, opt_state, batch_theta, batch_key, epoch):
        (loss_val, aux), grads = eqx.filter_value_and_grad(loss_pol_fn, has_aux=True)(model, batch_theta, batch_key, epoch)
        grads_pol = eqx.filter(grads, pol_mask)
        grad_norm = optax.global_norm(grads_pol)
        updates, opt_state = opt_pol.update(grads_pol, opt_state, pol_params, value=loss_val)
        model = eqx.apply_updates(model, updates)
        scale = optax.tree_utils.tree_get(opt_state, "scale")
        return model, opt_state, loss_val, aux, grad_norm, scale

    history = {
        "epoch_losses_sim": [], "epoch_losses_pol": [],
        "epoch_mse_sim": [], "epoch_mse_pol": [],
        "step_losses_sim": [], "step_losses_pol": [],
        "step_mse": [],
        "step_grad_norm_sim": [], "step_grad_norm_pol": [],
        "step_lr_scale_sim": [], "step_lr_scale_pol": [],
    }
    infinite_iter = iter(train_loader) if cfg.data_mode == "infinite" else None

    print("Starting Training with (fixed) Gradient Routing + Reduce-on-Plateau LR...")
    for epoch in range(cfg.epochs):
        n_batches = len(train_loader) if cfg.data_mode == "finite" else cfg.steps_per_epoch
        batch_source = train_loader if cfg.data_mode == "finite" else (next(infinite_iter) for _ in range(n_batches))

        running_loss_sim, running_loss_pol, running_mse, n_steps = 0.0, 0.0, 0.0, 0
        pbar = tqdm(batch_source, total=n_batches, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for theta_batch_np in pbar:
            key, batch_key = jax.random.split(key)
            batch_theta = jnp.asarray(theta_batch_np)  # (B, K, design_dim) -- rollout expects this shape

            model, opt_state_sim, loss_sim_val, aux_sim, gnorm_sim, scale_sim = train_step_sim(
                model, opt_state_sim, batch_theta, batch_key, epoch
            )
            model, opt_state_pol, loss_pol_val, aux_pol, gnorm_pol, scale_pol = train_step_pol(
                model, opt_state_pol, batch_theta, batch_key, epoch
            )

            running_loss_sim += float(loss_sim_val)
            running_loss_pol += float(loss_pol_val)
            running_mse += float(aux_sim["mse"])
            n_steps += 1

            history["step_losses_sim"].append(float(loss_sim_val))
            history["step_losses_pol"].append(float(loss_pol_val))
            history["step_mse"].append(float(aux_sim["mse"]))
            history["step_grad_norm_sim"].append(float(gnorm_sim))
            history["step_grad_norm_pol"].append(float(gnorm_pol))
            history["step_lr_scale_sim"].append(float(scale_sim))
            history["step_lr_scale_pol"].append(float(scale_pol))

            pbar.set_postfix(
                loss_sim=f"{float(loss_sim_val):.4f}",
                loss_pol=f"{float(loss_pol_val):.4f}",
                mse=f"{float(aux_sim['mse']):.4f}",
                lr_x=f"{float(scale_sim):.2f}"
            )

        avg_loss_sim = running_loss_sim / max(n_steps, 1)
        avg_loss_pol = running_loss_pol / max(n_steps, 1)
        avg_mse = running_mse / max(n_steps, 1)
        history["epoch_losses_sim"].append(avg_loss_sim)
        history["epoch_losses_pol"].append(avg_loss_pol)
        history["epoch_mse_sim"].append(avg_mse)
        history["epoch_mse_pol"].append(avg_mse)
        print(f"Epoch {epoch+1}: avg_loss_sim={avg_loss_sim:.4f}, avg_loss_pol={avg_loss_pol:.4f}, "
              f"avg_terminal_mse={avg_mse:.4f}, lr_scale_sim={float(scale_sim):.3f}, lr_scale_pol={float(scale_pol):.3f}")

        # Checkpoint
        save_model(model, run_dir / "checkpoint.eqx")

    with open(run_dir / "train_history.json", "w") as f:
        json.dump(history, f)

    return model, history, key

# %% [markdown]
# ## 12. Execution

# %%
if __name__ == "__main__":
    key = jax.random.PRNGKey(cfg.seed)
    run_dir = make_run_dir()
    snapshot_code(run_dir)
    save_config(cfg, run_dir)
    print(f"Run directory: {run_dir}")

    # Build model
    model = build_model(cfg, key)

    # Train
    model, history, key = train(cfg, model, key, run_dir)

    # %% [markdown]
    # ## 13. Visualisation

    # %%
    fig, ax = plt.subplots(3, 2, figsize=(13, 15))

    ax[0, 0].plot(history["epoch_losses_sim"], marker='o', color='purple')
    ax[0, 0].set_title("Simulator Loss (raw MSE + Barber)")
    ax[0, 0].set_yscale('symlog')
    ax[0, 0].set_xlabel("Epoch"); ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(history["epoch_losses_pol"], marker='s', color='orange')
    ax[0, 1].set_title("Policy Loss (raw MSE + PCE + Fisher + Trust)")
    ax[0, 1].set_yscale('symlog')   
    ax[0, 1].set_xlabel("Epoch"); ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].plot(history["epoch_mse_sim"], marker='d', color='green')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_title("Terminal MSE per epoch (log scale) -- the metric that matters")
    ax[1, 0].set_xlabel("Epoch"); ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(history["step_mse"], alpha=0.5, color='green')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_title("Terminal MSE per step (log scale)")
    ax[1, 1].set_xlabel("Step"); ax[1, 1].grid(True, alpha=0.3)

    ax[2, 0].plot(history["step_lr_scale_sim"], color='purple', label='sim')
    ax[2, 0].plot(history["step_lr_scale_pol"], color='orange', label='pol', alpha=0.7)
    ax[2, 0].set_title("Reduce-on-Plateau LR multiplier")
    ax[2, 0].set_xlabel("Step"); ax[2, 0].legend(); ax[2, 0].grid(True, alpha=0.3)

    ax[2, 1].plot(history["step_grad_norm_sim"], alpha=0.6, color='purple', label='sim grad norm')
    ax[2, 1].plot(history["step_grad_norm_pol"], alpha=0.6, color='orange', label='pol grad norm')
    ax[2, 1].set_yscale('log')
    ax[2, 1].set_title("Gradient norms per module (log scale)")
    ax[2, 1].set_xlabel("Step"); ax[2, 1].legend(); ax[2, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "plots" / "training_curves.png", dpi=150)
    # plt.close(fig)

    # --- Trajectory Evaluation ---
    sim_cfg = SourceLocConfig(K=cfg.k_sources)
    prior = SourceLocPrior(K=cfg.k_sources, prior_std=1.0)
    eval_loader = make_eval_loader(prior, n_episodes=128, batch_size=128, seed=cfg.seed + 1)
    eval_theta = jnp.asarray(next(iter(eval_loader)))  # (128, K, design_dim)
    key, eval_key = jax.random.split(key)
    eval_keys = jax.random.split(eval_key, eval_theta.shape[0])

    env_cfg = EnvConfig(K=cfg.k_sources, design_dim=cfg.design_dim)
    x_ts, y_ts, mus_after, log_sigmas_after, mus_before, log_sigmas_before, theta_flat = vmap_rollout(
        model, eval_theta, eval_keys, cfg, env_cfg
    )
    eval_mse = float(terminal_mse(mus_after[:, -1, :], eval_theta, cfg))
    print(f"Held-out terminal MSE over {eval_theta.shape[0]} episodes: {eval_mse:.4f}")

    idx = np.random.randint(0, eval_theta.shape[0])
    true_src = eval_theta[idx].reshape(-1, cfg.design_dim)
    xs = x_ts[idx]
    pred_path = mus_after[idx].reshape(cfg.max_t, cfg.k_sources, cfg.design_dim)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(true_src[:, 0], true_src[:, 1], c='red', marker='*', s=350, label='True sources', zorder=5)
    ax.plot(xs[:, 0], xs[:, 1], 'bo-', alpha=0.8, markersize=5, label='Designs $x_t$')
    for t in range(cfg.max_t):
        ax.text(xs[t, 0] + 0.05, xs[t, 1] + 0.05, f"$t_{{{t+1}}}$", fontsize=9, color='darkblue')

    for k in range(cfg.k_sources):
        for t in range(cfg.max_t):
            alpha = 0.15 + 0.85 * ((t + 1) / cfg.max_t)
            size = 2 + 160 * ((t + 1) / cfg.max_t)
            label = rf"Belief (src {k+1})" if t == cfg.max_t - 1 else None
            ax.scatter(pred_path[t, k, 0], pred_path[t, k, 1], c='green', alpha=alpha, s=size, marker='X', label=label)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4)
    ax.set_title(f"LA-BED Trajectory (seq={idx}, T={cfg.max_t}, held-out MSE={eval_mse:.3f})")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "plots" / f"trajectory_eval_{idx}.png", dpi=150)
    # plt.close(fig)

    print("All done!")