# VERSION: GP-TIME-VARYING-WAVE-PDE-BAYES-v1
# Controlled damped elastic string PDE with nominal REINFORCE/pathwise baselines,
# GP dynamics randomization, and Bayesian adaptive control.

# %% [markdown]
# # Controlled wave PDE: nominal vs GP-DR vs Bayesian adaptation
#
# PDE on x in (0, L), with fixed ends u(0,t)=u(L,t)=0:
#
#     u_tt = c(t)^2 u_xx - damping(t) u_t - stiffness(t) u + alpha b(x/L(t)) a_t
# on the changing physical domain x in (0,L(t)).
#
# The state is the full discretized displacement and velocity field. The hidden
# physical parameters, including the physical domain length, follow smooth
# functions of time drawn from Gaussian processes. Training and deployment use different GP functions; by default the
# deployment GP also has a shorter length scale (faster parameter changes).
#
# Four controllers are compared on the SAME held-out deployment physics path:
#   1) nominal REINFORCE
#   2) nominal pathwise policy gradient
#   3) GP dynamics-randomized (DR) robust pathwise policy
#   4) GP Bayesian-adaptive pathwise policy
#
# There are exactly two visual stages:
#   * START: PDE + GP environment preview
#   * END: a fixed-layout four-way GIF containing live hidden physics, Bayesian
#          tracking, and the training loss curves.

# %% Imports and config
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import optax

try:
    from IPython.display import display, Image
except ImportError:
    def display(*a, **k):
        pass

    def Image(*a, **k):
        return None

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlepad": 8.0})


OUT_DIR = Path("wave_pde_out")
OUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------
    # PDE discretization and nominal physical parameters
    # ------------------------------------------------------------------
    domain_length: float = 1.0       # nominal physical length L
    n_grid: int = 32                 # interior grid points; state dimension = 64
    dt: float = 0.0025

    # Nominal coefficients in u_tt = c^2 u_xx - gamma u_t - k u + alpha b a.
    wave_speed: float = 1.0          # c
    damping: float = 0.35            # gamma
    stiffness: float = 4.0           # k (elastic foundation / restoring term)
    actuator_gain: float = 8.0       # alpha is kept known/fixed

    # One scalar localized actuator. Keep force bounded so model knowledge matters.
    force_max: float = 4.0
    actuator_center: float = 0.50
    actuator_width: float = 0.10

    # ------------------------------------------------------------------
    # Hidden time-varying physics
    # ------------------------------------------------------------------
    uncertain_param: str = "all"
    # Allowed: "wave_speed", "damping", "stiffness", "domain_length", "all"

    # Independent +/- margins about each nominal coefficient.
    wave_speed_rel_range: float = 0.45
    damping_rel_range: float = 0.70
    stiffness_rel_range: float = 0.60
    domain_length_rel_range: float = 0.45

    # Small pointwise noise superposed on the smooth GP function. These values are
    # relative to nominal and deliberately smaller than the GP excursion itself.
    wave_speed_process_rel_std: float = 0.025
    damping_process_rel_std: float = 0.040
    stiffness_process_rel_std: float = 0.030
    domain_length_process_rel_std: float = 0.025

    # GP meta-distribution. DR and Bayes see exactly the same training bank.
    gp_train_function_bank_size: int = 31
    gp_train_length_scale_seconds: float = 0.22
    gp_deploy_length_scale_seconds: float = 0.075
    gp_latent_std: float = 1.0
    gp_jitter: float = 1e-5
    gp_point_noise_scale: float = 0.10

    # ------------------------------------------------------------------
    # Bayesian / EKF online system identification
    # ------------------------------------------------------------------
    # Observation model is the full N-dimensional acceleration field inferred
    # from consecutive velocity fields. Noise is in acceleration units.
    belief_accel_noise_std: float = 0.35
    belief_initial_rel_std: float = 0.45
    belief_process_rel_std_scale: float = 0.035
    stop_gradient_through_belief: bool = True

    # ------------------------------------------------------------------
    # Episode / initial vibration
    # ------------------------------------------------------------------
    horizon: int = 300
    init_mode1_amp: float = 0.55
    init_mode2_amp: float = 0.35
    init_velocity_amp: float = 0.20

    # Reward: stabilize the complete distributed field, not one coordinate.
    displacement_penalty: float = 2.0
    velocity_penalty: float = 0.20
    action_penalty: float = 0.004

    # Policy
    hidden: int = 96
    depth: int = 2
    init_log_std: float = -0.8

    # Optimisation
    updates: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-4
    grad_clip: Optional[float] = 1.0
    gamma: float = 0.995

    # REINFORCE variance reduction
    use_loo_baseline: bool = True
    normalize_advantage: bool = True
    entropy_coef: float = 0.0

    # Visualisation
    eval_every: int = 50             # print cadence only; no intermediate plots
    gif_fps: int = 30
    gif_stride: int = 2              # animate every 2nd PDE time step

    seed: int = 0


cfg = Config()
key = jax.random.PRNGKey(cfg.seed)


# %% Physical-parameter helpers
class PDEParams(NamedTuple):
    wave_speed: jax.Array
    damping: jax.Array
    stiffness: jax.Array
    domain_length: jax.Array


PARAM_NAMES = ("wave_speed", "damping", "stiffness", "domain_length")
PARAM_LABELS = {
    "wave_speed": "wave speed c",
    "damping": "damping gamma",
    "stiffness": "stiffness k",
    "domain_length": "physical length L",
}
NUM_PARAMS = len(PARAM_NAMES)


def nominal_parameter_vector(cfg: Config) -> jax.Array:
    return jnp.asarray([
        cfg.wave_speed,
        cfg.damping,
        cfg.stiffness,
        cfg.domain_length,
    ])


def parameter_rel_ranges(cfg: Config) -> jax.Array:
    return jnp.asarray([
        cfg.wave_speed_rel_range,
        cfg.damping_rel_range,
        cfg.stiffness_rel_range,
        cfg.domain_length_rel_range,
    ])


def parameter_process_rel_stds(cfg: Config) -> jax.Array:
    return jnp.asarray([
        cfg.wave_speed_process_rel_std,
        cfg.damping_process_rel_std,
        cfg.stiffness_process_rel_std,
        cfg.domain_length_process_rel_std,
    ])


def active_parameter_indices(cfg: Config) -> tuple[int, ...]:
    if cfg.uncertain_param == "all":
        return tuple(range(NUM_PARAMS))
    if cfg.uncertain_param in PARAM_NAMES:
        return (PARAM_NAMES.index(cfg.uncertain_param),)
    raise ValueError(
        f"uncertain_param={cfg.uncertain_param!r}; choose one of {PARAM_NAMES} or 'all'."
    )


def active_parameter_mask(cfg: Config) -> jax.Array:
    m = [False] * NUM_PARAMS
    for i in active_parameter_indices(cfg):
        m[i] = True
    return jnp.asarray(m)


def params_from_vector(v: jax.Array) -> PDEParams:
    return PDEParams(v[0], v[1], v[2], v[3])


def clip_parameter_vector(v: jax.Array, cfg: Config) -> jax.Array:
    nominal = nominal_parameter_vector(cfg)
    r = parameter_rel_ranges(cfg)
    lo, hi = nominal * (1.0 - r), nominal * (1.0 + r)
    return jnp.clip(v, lo, hi)


# %% PDE spatial grid and differentiable simulator

def material_grid(cfg: Config) -> jax.Array:
    """Fixed material coordinate xi in (0,1), independent of current length."""
    dxi = 1.0 / (cfg.n_grid + 1)
    return dxi * jnp.arange(1, cfg.n_grid + 1)


def spatial_grid(cfg: Config) -> jax.Array:
    """Nominal physical x-grid, used only for static initialization/labels."""
    return cfg.domain_length * material_grid(cfg)


def dxi_value(cfg: Config) -> float:
    return 1.0 / (cfg.n_grid + 1)


def actuator_shape(cfg: Config) -> jax.Array:
    # Actuator is fixed to a material location xi, so it stretches with the body.
    xi = material_grid(cfg)
    b = jnp.exp(-0.5 * ((xi - cfg.actuator_center) / cfg.actuator_width) ** 2)
    return b / (jnp.max(b) + 1e-8)


def laplacian_material(u: jax.Array, cfg: Config) -> jax.Array:
    """Second derivative d²u/dxi² on the fixed material coordinate."""
    padded = jnp.pad(u, (1, 1), mode="constant", constant_values=0.0)
    return (padded[:-2] - 2.0 * padded[1:-1] + padded[2:]) / dxi_value(cfg) ** 2


def pde_acceleration(
    u: jax.Array,
    v: jax.Array,
    action: jax.Array,
    params_vec: jax.Array,
    cfg: Config,
) -> jax.Array:
    """Evaluate u_tt on a time-varying physical domain.

    We represent the field in material coordinate xi=x/L(t). Since

        d²u/dx² = (1/L(t)²) d²u/dxi²,

    changing physical length changes the PDE operator and modal frequencies even
    when the material-coordinate displacement vector is identical. This is the
    key partial-observability mechanism we want to study.
    """
    p = params_from_vector(params_vec)
    physical_laplacian = laplacian_material(u, cfg) / (p.domain_length**2)
    return (
        p.wave_speed**2 * physical_laplacian
        - p.damping * v
        - p.stiffness * u
        + cfg.actuator_gain * actuator_shape(cfg) * action
    )


def wave_pde_step(
    state: jax.Array,
    action: jax.Array,
    cfg: Config,
    params_vec: Optional[jax.Array] = None,
) -> jax.Array:
    """One differentiable semi-implicit Euler step of the PDE discretization.

    state = concat([u_1..u_N, v_1..v_N]).  Semi-implicit integration updates
    velocity first and displacement second, which is substantially more stable
    for wave dynamics than explicit position-first Euler at the same dt.
    """
    if params_vec is None:
        params_vec = nominal_parameter_vector(cfg)
    n = cfg.n_grid
    u, v = state[:n], state[n:]
    acc = pde_acceleration(u, v, action, params_vec, cfg)
    v_new = v + cfg.dt * acc
    u_new = u + cfg.dt * v_new
    return jnp.concatenate([u_new, v_new])


def initial_state(key: jax.Array, cfg: Config) -> jax.Array:
    """Random combination of low spatial modes; all methods see this distribution."""
    k1, k2, kv = jax.random.split(key, 3)
    xi = material_grid(cfg)
    mode1 = jnp.sin(jnp.pi * xi)
    mode2 = jnp.sin(2.0 * jnp.pi * xi)
    mode3 = jnp.sin(3.0 * jnp.pi * xi)

    a1 = cfg.init_mode1_amp * (0.75 + 0.25 * jax.random.uniform(k1, ()))
    a2 = cfg.init_mode2_amp * jax.random.uniform(k2, (), minval=-1.0, maxval=1.0)
    u0 = a1 * mode1 + a2 * mode2
    v0 = cfg.init_velocity_amp * jax.random.normal(kv, ()) * mode3
    return jnp.concatenate([u0, v0])


def time_feature(t_index: jax.Array, cfg: Config) -> jax.Array:
    # Shared metadata: every controller receives normalized episode time.
    return 2.0 * t_index / jnp.maximum(cfg.horizon - 1, 1) - 1.0


def base_observation(state: jax.Array, t_index: jax.Array, cfg: Config) -> jax.Array:
    # Scale field magnitudes mildly so MLP input coordinates stay O(1).
    return jnp.concatenate([state, jnp.asarray([time_feature(t_index, cfg)])])


def reward_fn(state: jax.Array, action: jax.Array, cfg: Config) -> jax.Array:
    n = cfg.n_grid
    u, v = state[:n], state[n:]
    return -(
        cfg.displacement_penalty * jnp.mean(u**2)
        + cfg.velocity_penalty * jnp.mean(v**2)
        + cfg.action_penalty * action**2
    )


# %% Policy
class Policy(eqx.Module):
    mlp: eqx.nn.MLP
    raw_log_std: jax.Array

    def __init__(self, key: jax.Array, cfg: Config, in_size: int):
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=cfg.hidden,
            depth=cfg.depth,
            activation=jax.nn.tanh,
            key=key,
        )
        self.raw_log_std = jnp.asarray(cfg.init_log_std, dtype=jnp.float32)


def policy_stats(policy: Policy, obs: jax.Array):
    return policy.mlp(obs)[0], policy.raw_log_std


def squash_to_force(z: jax.Array, cfg: Config) -> jax.Array:
    return cfg.force_max * jnp.tanh(z)


def normal_log_prob(z, mu, log_std):
    inv_std = jnp.exp(-log_std)
    return -0.5 * ((z - mu) * inv_std) ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)


def normal_entropy(log_std):
    return log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)


# %% Returns / optimization helpers

def discounted_returns_to_go(rewards, gamma):
    def scan_fn(carry, r):
        c = r + gamma * carry
        return c, c
    _, rev = jax.lax.scan(scan_fn, 0.0, rewards[::-1])
    return rev[::-1]


def discounted_episode_return(rewards, gamma):
    return jnp.sum((gamma ** jnp.arange(rewards.shape[-1])) * rewards, axis=-1)


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    sq = [jnp.sum(x**2) for x in leaves if x is not None and eqx.is_inexact_array(x)]
    return jnp.sqrt(sum(sq))


def make_optimizer(cfg: Config):
    adam = optax.adam(cfg.learning_rate)
    if cfg.grad_clip is None:
        return adam
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), adam)


def make_train_step(loss_fn, optimizer):
    value_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    @eqx.filter_jit
    def step(policy, opt_state, keys):
        (loss, diag), grads = value_and_grad(policy, keys)
        grad_norm = tree_l2_norm(grads)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, updates)
        return policy, opt_state, {**diag, "loss": loss, "grad_norm": grad_norm}

    return step


# %% Nominal rollouts / losses
BASE_OBS_SIZE = 2 * cfg.n_grid + 1


def rollout_one_nominal(policy: Policy, key: jax.Array, cfg: Config):
    k_state, k_action = jax.random.split(key)
    state0 = initial_state(k_state, cfg)
    eps = jax.random.normal(k_action, (cfg.horizon,))

    def step(carry, inp):
        state, t_idx = carry
        eps_t = inp
        obs = base_observation(state, t_idx, cfg)
        mu, log_std = policy_stats(policy, obs)
        z = mu + jnp.exp(log_std) * eps_t
        action = squash_to_force(z, cfg)
        reward = reward_fn(state, action, cfg)
        next_state = wave_pde_step(state, action, cfg)
        return (next_state, t_idx + 1), {
            "obs": obs, "z": z, "reward": reward, "state": state, "action": action,
        }

    (_, _), traj = jax.lax.scan(step, (state0, jnp.asarray(0)), eps)
    return traj


def rollout_batch_nominal(policy, keys, cfg):
    return jax.vmap(lambda k: rollout_one_nominal(policy, k, cfg))(keys)


def make_reinforce_loss(cfg: Config):
    def loss_fn(policy: Policy, keys: jax.Array):
        traj = rollout_batch_nominal(policy, keys, cfg)
        rewards = traj["reward"]
        rtg = jax.vmap(lambda r: discounted_returns_to_go(r, cfg.gamma))(rewards)

        if cfg.use_loo_baseline and cfg.batch_size > 1:
            baseline = (jnp.sum(rtg, axis=0, keepdims=True) - rtg) / (cfg.batch_size - 1)
        else:
            baseline = jnp.zeros_like(rtg)
        advantage = rtg - baseline
        if cfg.normalize_advantage:
            advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

        obs_sg = jax.lax.stop_gradient(traj["obs"])
        z_sg = jax.lax.stop_gradient(traj["z"])

        def lp(o, z):
            mu, ls = policy_stats(policy, o)
            return normal_log_prob(z, mu, ls)

        logp = jax.vmap(jax.vmap(lp))(obs_sg, z_sg)
        discounts = cfg.gamma ** jnp.arange(cfg.horizon)
        surrogate = jnp.mean(jnp.sum(
            discounts[None, :] * jax.lax.stop_gradient(advantage) * logp, axis=1
        ))
        entropy_bonus = cfg.entropy_coef * cfg.horizon * normal_entropy(policy.raw_log_std)
        loss = -(surrogate + entropy_bonus)
        ep_ret = discounted_episode_return(rewards, cfg.gamma)
        return loss, {
            "return": jnp.mean(ep_ret),
            "field_rms": jnp.mean(jnp.sqrt(jnp.mean(traj["state"][:, -1, :cfg.n_grid]**2, axis=1))),
            "log_std": policy.raw_log_std,
        }
    return loss_fn


def make_nominal_pathwise_loss(cfg: Config):
    def loss_fn(policy: Policy, keys: jax.Array):
        traj = rollout_batch_nominal(policy, keys, cfg)
        ep_ret = discounted_episode_return(traj["reward"], cfg.gamma)
        entropy_bonus = cfg.entropy_coef * cfg.horizon * normal_entropy(policy.raw_log_std)
        return -(jnp.mean(ep_ret) + entropy_bonus), {
            "return": jnp.mean(ep_ret),
            "field_rms": jnp.mean(jnp.sqrt(jnp.mean(traj["state"][:, -1, :cfg.n_grid]**2, axis=1))),
            "log_std": policy.raw_log_std,
        }
    return loss_fn


# %% GP parameter-function sampling

def gp_time_grid(cfg: Config) -> jax.Array:
    return cfg.dt * jnp.arange(cfg.horizon + 1)


def rbf_gp_cholesky(cfg: Config, ell_seconds: float) -> jax.Array:
    t = gp_time_grid(cfg)
    d = t[:, None] - t[None, :]
    ell = jnp.maximum(jnp.asarray(ell_seconds), 1e-6)
    K = cfg.gp_latent_std**2 * jnp.exp(-0.5 * (d / ell) ** 2)
    K = K + cfg.gp_jitter * jnp.eye(t.shape[0])
    return jnp.linalg.cholesky(K)


def sample_gp_parameter_paths(
    key: jax.Array,
    n_paths: int,
    cfg: Config,
    *,
    length_scale_seconds: float,
) -> jax.Array:
    """Sample [B,T+1,4] bounded physical-parameter functions.

    A latent zero-mean GP z_p(t) is squashed through tanh and mapped into each
    parameter's own relative margin. Independent small pointwise noise is added.
    Inactive parameters remain exactly nominal.
    """
    k_gp, k_noise = jax.random.split(key)
    L = rbf_gp_cholesky(cfg, length_scale_seconds)
    z = jax.random.normal(k_gp, (n_paths, NUM_PARAMS, cfg.horizon + 1))
    latent = jnp.einsum("ij,bpj->bpi", L, z)                 # [B,P,T+1]
    latent = jnp.transpose(latent, (0, 2, 1))               # [B,T+1,P]

    nominal = nominal_parameter_vector(cfg)
    ranges = parameter_rel_ranges(cfg)
    active = active_parameter_mask(cfg)
    smooth_rel = ranges[None, None, :] * jnp.tanh(latent)

    point_eps = jax.random.normal(k_noise, smooth_rel.shape)
    point_rel = (
        cfg.gp_point_noise_scale
        * parameter_process_rel_stds(cfg)[None, None, :]
        * point_eps
    )
    rel = jnp.clip(smooth_rel + point_rel, -ranges, ranges)
    path = nominal[None, None, :] * (1.0 + rel)
    return jnp.where(active[None, None, :], path, nominal[None, None, :])


# %% Bayesian/EKF identifier

def initial_belief(cfg: Config):
    nominal = nominal_parameter_vector(cfg)
    ranges = parameter_rel_ranges(cfg)
    std = cfg.belief_initial_rel_std * ranges * nominal
    std = jnp.maximum(std, 1e-5)
    cov = jnp.diag(std**2)
    return nominal, cov


def belief_features(mean: jax.Array, cov: jax.Array, cfg: Config) -> jax.Array:
    """Normalized posterior mean offsets and marginal stds, active parameters only."""
    nominal = nominal_parameter_vector(cfg)
    half_width = jnp.maximum(parameter_rel_ranges(cfg) * nominal, 1e-6)
    idx = jnp.asarray(active_parameter_indices(cfg))
    mean_f = ((mean - nominal) / half_width)[idx]
    std_f = (jnp.sqrt(jnp.maximum(jnp.diag(cov), 1e-12)) / half_width)[idx]
    return jnp.stack([mean_f, std_f], axis=1).reshape(-1)


def adaptive_obs_size(cfg: Config) -> int:
    return BASE_OBS_SIZE + 2 * len(active_parameter_indices(cfg))


def bayes_ekf_update(
    mean: jax.Array,
    cov: jax.Array,
    state: jax.Array,
    action: jax.Array,
    next_state: jax.Array,
    cfg: Config,
):
    """EKF measurement update over the PDE coefficients.

    Measurement y is the full spatial acceleration field inferred from v_{t+1}.
    The nonlinear measurement function h(phi) is exactly the PDE acceleration.
    This uses N simultaneous spatial observations to identify only four latent
    coefficients, which is far better posed than CartPole's two accelerations.
    """
    n = cfg.n_grid
    u, v = state[:n], state[n:]
    next_v = next_state[n:]
    y = (next_v - v) / cfg.dt

    def h(phi):
        return pde_acceleration(u, v, action, phi, cfg)

    yhat = h(mean)
    H = jax.jacfwd(h)(mean)                                 # [N,4]
    R = (cfg.belief_accel_noise_std**2) * jnp.eye(n)
    S = H @ cov @ H.T + R
    PHt = cov @ H.T
    # Solve S X = H P, avoiding an explicit matrix inverse.
    K = jnp.linalg.solve(S, PHt.T).T                        # [4,N]
    innovation = y - yhat
    post_mean = clip_parameter_vector(mean + K @ innovation, cfg)

    I = jnp.eye(NUM_PARAMS)
    IKH = I - K @ H
    post_cov = IKH @ cov @ IKH.T + K @ R @ K.T             # Joseph form
    post_cov = 0.5 * (post_cov + post_cov.T) + 1e-9 * I
    return post_mean, post_cov


def predict_belief(mean: jax.Array, cov: jax.Array, cfg: Config):
    """Prediction step for unknown smooth parameter motion.

    We deliberately do not reveal the sampled GP function to the estimator. It
    only knows parameters can move, represented by small Gaussian process noise.
    """
    nominal = nominal_parameter_vector(cfg)
    ranges = parameter_rel_ranges(cfg)
    qstd = cfg.belief_process_rel_std_scale * ranges * nominal
    active = active_parameter_mask(cfg).astype(mean.dtype)
    Q = jnp.diag((qstd * active) ** 2)
    return mean, cov + Q


def normalized_id_error(mean, truth, cfg):
    nominal = nominal_parameter_vector(cfg)
    hw = jnp.maximum(parameter_rel_ranges(cfg) * nominal, 1e-6)
    idx = jnp.asarray(active_parameter_indices(cfg))
    return jnp.mean(jnp.abs(mean[idx] - truth[idx]) / hw[idx])


# %% GP-varying rollouts: DR and Bayes share exactly the same bank

def rollout_one_gp(
    policy: Policy,
    key: jax.Array,
    cfg: Config,
    training_gp_bank: jax.Array,
    *,
    adaptive: bool,
):
    k_state, k_action, k_bank = jax.random.split(key, 3)
    state0 = initial_state(k_state, cfg)
    eps = jax.random.normal(k_action, (cfg.horizon,))
    bank_idx = jax.random.randint(k_bank, (), 0, training_gp_bank.shape[0])
    true_path = training_gp_bank[bank_idx]
    mean0, cov0 = initial_belief(cfg)

    def step(carry, inp):
        state, mean, cov = carry
        eps_t, t_idx = inp
        base = base_observation(state, t_idx, cfg)
        if adaptive:
            obs = jnp.concatenate([base, belief_features(mean, cov, cfg)])
        else:
            obs = base

        mu, log_std = policy_stats(policy, obs)
        z = mu + jnp.exp(log_std) * eps_t
        action = squash_to_force(z, cfg)
        reward = reward_fn(state, action, cfg)
        params_t = true_path[t_idx]
        next_state = wave_pde_step(state, action, cfg, params_t)

        if adaptive:
            post_mean, post_cov = bayes_ekf_update(mean, cov, state, action, next_state, cfg)
            next_mean, next_cov = predict_belief(post_mean, post_cov, cfg)
            diag_mean, diag_cov = post_mean, post_cov
        else:
            next_mean, next_cov = mean, cov
            diag_mean, diag_cov = mean, cov

        if cfg.stop_gradient_through_belief:
            next_mean = jax.lax.stop_gradient(next_mean)
            next_cov = jax.lax.stop_gradient(next_cov)

        return (next_state, next_mean, next_cov), {
            "obs": obs,
            "z": z,
            "reward": reward,
            "state": state,
            "action": action,
            "true_params": params_t,
            "belief_mean": diag_mean,
            "belief_cov": diag_cov,
        }

    t_idx = jnp.arange(cfg.horizon)
    (_, final_mean, final_cov), traj = jax.lax.scan(
        step, (state0, mean0, cov0), (eps, t_idx)
    )
    return {**traj, "final_belief_mean": final_mean, "final_belief_cov": final_cov}


def rollout_batch_gp(policy, keys, cfg, training_gp_bank, *, adaptive: bool):
    return jax.vmap(
        lambda k: rollout_one_gp(policy, k, cfg, training_gp_bank, adaptive=adaptive)
    )(keys)


def make_gp_pathwise_loss(cfg: Config, training_gp_bank: jax.Array, *, adaptive: bool):
    def loss_fn(policy: Policy, keys: jax.Array):
        traj = rollout_batch_gp(policy, keys, cfg, training_gp_bank, adaptive=adaptive)
        ep_ret = discounted_episode_return(traj["reward"], cfg.gamma)
        entropy_bonus = cfg.entropy_coef * cfg.horizon * normal_entropy(policy.raw_log_std)
        loss = -(jnp.mean(ep_ret) + entropy_bonus)

        final_u = traj["state"][:, -1, :cfg.n_grid]
        if adaptive:
            errors = jax.vmap(jax.vmap(lambda m, p: normalized_id_error(m, p, cfg)))(
                traj["belief_mean"], traj["true_params"]
            )
            id_error = jnp.mean(errors)
        else:
            id_error = jnp.asarray(0.0)
        return loss, {
            "return": jnp.mean(ep_ret),
            "field_rms": jnp.mean(jnp.sqrt(jnp.mean(final_u**2, axis=1))),
            "log_std": policy.raw_log_std,
            "id_error": id_error,
        }
    return loss_fn


# %% Histories
HISTORY_KEYS = ("loss", "return", "field_rms", "log_std", "grad_norm", "id_error")


def init_history():
    return {k: [] for k in HISTORY_KEYS}


def append_history(history, diag):
    h = jax.device_get(diag)
    for name in HISTORY_KEYS:
        history[name].append(float(h.get(name, 0.0)))


# %% Build shared GP bank + held-out deployment path BEFORE training
key, k_bank, k_deploy, k_schedule = jax.random.split(key, 4)
training_gp_bank = sample_gp_parameter_paths(
    k_bank,
    cfg.gp_train_function_bank_size,
    cfg,
    length_scale_seconds=cfg.gp_train_length_scale_seconds,
)
deployment_parameter_path = sample_gp_parameter_paths(
    k_deploy,
    1,
    cfg,
    length_scale_seconds=cfg.gp_deploy_length_scale_seconds,
)[0]

# The DR and Bayesian policies see exactly the same rollout keys at every update.
training_keys = jax.random.split(k_schedule, cfg.updates * cfg.batch_size).reshape(
    cfg.updates, cfg.batch_size, 2
)


# %% START visualization: PDE + GP environment preview

def make_start_preview(training_bank, deploy_path, cfg, path):
    train = np.asarray(jax.device_get(training_bank))
    deploy = np.asarray(jax.device_get(deploy_path))
    t = np.asarray(jax.device_get(gp_time_grid(cfg)))
    x = np.asarray(jax.device_get(spatial_grid(cfg)))

    # Deterministic representative initial field for illustration.
    u0 = (
        cfg.init_mode1_amp * np.sin(np.pi * x / cfg.domain_length)
        + 0.5 * cfg.init_mode2_amp * np.sin(2 * np.pi * x / cfg.domain_length)
    )
    b = np.asarray(jax.device_get(actuator_shape(cfg)))

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax_pde = fig.add_subplot(gs[:, 0])
    ax_pde.plot(x, u0, linewidth=2, label="initial displacement u(x,0)")
    ax_pde.plot(x, 0.35 * b, linestyle="--", label="actuator footprint b(x)")
    ax_pde.axhline(0.0, linewidth=1)
    ax_pde.set_xlabel("position x")
    ax_pde.set_ylabel("field amplitude")
    ax_pde.set_title("Distributed PDE state")
    ax_pde.legend(fontsize=8)

    for j, i in enumerate(range(NUM_PARAMS)):
        ax = fig.add_subplot(gs[j // 2, 1 + j % 2])
        for k in range(min(8, train.shape[0])):
            ax.plot(t, train[k, :, i], alpha=0.22, linewidth=1)
        ax.plot(t, deploy[:, i], linewidth=2.2, label="held-out deployment")
        ax.set_title(PARAM_LABELS[PARAM_NAMES[i]])
        ax.set_xlabel("time [s]")
        ax.legend(fontsize=7)

    fig.suptitle(
        "START — controlled wave PDE and GP-varying hidden physics\n"
        f"train GP ell={cfg.gp_train_length_scale_seconds:g}s | "
        f"deployment ell={cfg.gp_deploy_length_scale_seconds:g}s",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160)
    plt.show()
    return path


start_path = make_start_preview(
    training_gp_bank,
    deployment_parameter_path,
    cfg,
    OUT_DIR / "START_wave_pde_gp_environment.png",
)


# %% Train nominal REINFORCE and nominal pathwise — no intermediate visualizations
key, k_init, k_nom_schedule = jax.random.split(key, 3)
init_policy = Policy(k_init, cfg, BASE_OBS_SIZE)
nominal_keys = jax.random.split(k_nom_schedule, cfg.updates * cfg.batch_size).reshape(
    cfg.updates, cfg.batch_size, 2
)

reinforce_policy = init_policy
reinforce_opt = make_optimizer(cfg)
reinforce_state = reinforce_opt.init(eqx.filter(reinforce_policy, eqx.is_inexact_array))
reinforce_step = make_train_step(make_reinforce_loss(cfg), reinforce_opt)
reinforce_history = init_history()

for update in range(cfg.updates):
    reinforce_policy, reinforce_state, diag = reinforce_step(
        reinforce_policy, reinforce_state, nominal_keys[update]
    )
    append_history(reinforce_history, diag)
    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(f"[nominal REINFORCE] {update:04d} return={reinforce_history['return'][-1]:9.3f}")

pathwise_policy = init_policy
pathwise_opt = make_optimizer(cfg)
pathwise_state = pathwise_opt.init(eqx.filter(pathwise_policy, eqx.is_inexact_array))
pathwise_step = make_train_step(make_nominal_pathwise_loss(cfg), pathwise_opt)
pathwise_history = init_history()

for update in range(cfg.updates):
    pathwise_policy, pathwise_state, diag = pathwise_step(
        pathwise_policy, pathwise_state, nominal_keys[update]
    )
    append_history(pathwise_history, diag)
    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(f"[nominal pathwise ] {update:04d} return={pathwise_history['return'][-1]:9.3f}")


# %% Train GP-DR robust and GP-Bayes adaptive on exactly the same functions/keys
dr_policy = init_policy
dr_opt = make_optimizer(cfg)
dr_state = dr_opt.init(eqx.filter(dr_policy, eqx.is_inexact_array))
dr_step = make_train_step(make_gp_pathwise_loss(cfg, training_gp_bank, adaptive=False), dr_opt)
dr_history = init_history()

for update in range(cfg.updates):
    dr_policy, dr_state, diag = dr_step(dr_policy, dr_state, training_keys[update])
    append_history(dr_history, diag)
    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(f"[GP-DR robust     ] {update:04d} return={dr_history['return'][-1]:9.3f}")

key, k_bayes_init = jax.random.split(key)
bayes_policy = Policy(k_bayes_init, cfg, adaptive_obs_size(cfg))
bayes_opt = make_optimizer(cfg)
bayes_state = bayes_opt.init(eqx.filter(bayes_policy, eqx.is_inexact_array))
bayes_step = make_train_step(make_gp_pathwise_loss(cfg, training_gp_bank, adaptive=True), bayes_opt)
bayes_history = init_history()

for update in range(cfg.updates):
    bayes_policy, bayes_state, diag = bayes_step(
        bayes_policy, bayes_state, training_keys[update]
    )
    append_history(bayes_history, diag)
    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(
            f"[GP-Bayes adaptive] {update:04d} return={bayes_history['return'][-1]:9.3f} "
            f"IDerr={bayes_history['id_error'][-1]:.3f}"
        )


# %% Deterministic evaluation: all four controllers, SAME initial state + GP path

def deterministic_deployment_rollout(
    policy: Policy,
    cfg: Config,
    parameter_path: jax.Array,
    state0: jax.Array,
    *,
    adaptive: bool,
):
    mean0, cov0 = initial_belief(cfg)

    def step(carry, t_idx):
        state, mean, cov = carry
        base = base_observation(state, t_idx, cfg)
        obs = jnp.concatenate([base, belief_features(mean, cov, cfg)]) if adaptive else base
        mu, _ = policy_stats(policy, obs)
        action = squash_to_force(mu, cfg)
        reward = reward_fn(state, action, cfg)
        params_t = parameter_path[t_idx]
        next_state = wave_pde_step(state, action, cfg, params_t)

        if adaptive:
            post_mean, post_cov = bayes_ekf_update(mean, cov, state, action, next_state, cfg)
            next_mean, next_cov = predict_belief(post_mean, post_cov, cfg)
            out_mean, out_cov = post_mean, post_cov
        else:
            next_mean, next_cov = mean, cov
            out_mean, out_cov = mean, cov

        return (next_state, next_mean, next_cov), {
            "state": state,
            "action": action,
            "reward": reward,
            "true_params": params_t,
            "belief_mean": out_mean,
            "belief_cov": out_cov,
        }

    (final_state, final_mean, final_cov), traj = jax.lax.scan(
        step, (state0, mean0, cov0), jnp.arange(cfg.horizon)
    )
    return {
        **traj,
        "states_with_final": jnp.concatenate([traj["state"], final_state[None]], axis=0),
        "params_with_final": parameter_path,
        "belief_mean_with_final": jnp.concatenate([traj["belief_mean"], final_mean[None]], axis=0),
        "belief_cov_with_final": jnp.concatenate([traj["belief_cov"], final_cov[None]], axis=0),
    }


key, k_eval_state = jax.random.split(key)
eval_state0 = initial_state(k_eval_state, cfg)

eval_reinforce = deterministic_deployment_rollout(
    reinforce_policy, cfg, deployment_parameter_path, eval_state0, adaptive=False
)
eval_pathwise = deterministic_deployment_rollout(
    pathwise_policy, cfg, deployment_parameter_path, eval_state0, adaptive=False
)
eval_dr = deterministic_deployment_rollout(
    dr_policy, cfg, deployment_parameter_path, eval_state0, adaptive=False
)
eval_bayes = deterministic_deployment_rollout(
    bayes_policy, cfg, deployment_parameter_path, eval_state0, adaptive=True
)


def host(x):
    return np.asarray(jax.device_get(x))


def eval_return(traj):
    return float(discounted_episode_return(traj["reward"], cfg.gamma))


print("\nHeld-out GP deployment return (same PDE path + initial field):")
for label, tr in [
    ("Nominal REINFORCE", eval_reinforce),
    ("Nominal pathwise", eval_pathwise),
    ("GP-DR robust", eval_dr),
    ("GP-Bayes adaptive", eval_bayes),
]:
    print(f"  {label:20s}: {eval_return(tr):10.3f}")


# %% FINAL visualization: fixed four-way PDE dashboard + live physics + loss plots

def make_final_dashboard_gif(
    evals,
    histories,
    cfg: Config,
    path: Path,
):
    """Create one stable, publication-style final comparison animation.

    No axis is cleared inside update(). All artists are created once and mutated,
    so subplot sizes, scales, titles and labels remain fixed across every frame.
    """
    labels = ["Nominal REINFORCE", "Nominal Pathwise", "GP-DR robust", "GP-Bayes adaptive"]
    xi = host(material_grid(cfg))
    xi_full = np.concatenate([[0.0], xi, [1.0]])
    t = host(gp_time_grid(cfg))
    param_path = host(evals[-1]["params_with_final"])
    bayes_mean = host(evals[-1]["belief_mean_with_final"])
    bayes_cov = host(evals[-1]["belief_cov_with_final"])
    bayes_std = np.sqrt(np.maximum(np.diagonal(bayes_cov, axis1=1, axis2=2), 0.0))

    states = [host(e["states_with_final"]) for e in evals]
    actions = [host(e["action"]) for e in evals]
    rewards = [host(e["reward"]) for e in evals]
    returns = [np.cumsum(r) for r in rewards]

    # Fixed y-range from all trajectories; add margin once, never auto-rescale.
    all_u = np.concatenate([s[:, :cfg.n_grid].ravel() for s in states])
    amp = max(0.7, 1.15 * np.max(np.abs(all_u)))

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.15, 1.15, 0.75, 0.80], hspace=0.52, wspace=0.32)

    controller_axes = [
        fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
    ]

    field_lines, info_texts = [], []
    for ax, label in zip(controller_axes, labels):
        line, = ax.plot(cfg.domain_length * xi_full, np.zeros_like(xi_full), linewidth=2.5)
        ax.axhline(0.0, linewidth=0.8)
        max_L = cfg.domain_length * (1.0 + cfg.domain_length_rel_range)
        ax.set_xlim(0.0, max_L)
        ax.set_ylim(-amp, amp)
        ax.set_xlabel("position x")
        ax.set_ylabel("displacement u(x,t)")
        ax.set_title(label, fontsize=11, fontweight="bold")
        txt = ax.text(
            0.015, 0.95, "", transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82),
        )
        field_lines.append(line)
        info_texts.append(txt)

    param_axes = [fig.add_subplot(gs[2, j]) for j in range(4)]
    cursor_lines = []
    for i, ax in enumerate(param_axes):
        ax.plot(t, param_path[:, i], linewidth=1.8, label="true")
        ax.plot(t, bayes_mean[:, i], linewidth=1.4, label="Bayes mean")
        ax.fill_between(
            t,
            bayes_mean[:, i] - 2.0 * bayes_std[:, i],
            bayes_mean[:, i] + 2.0 * bayes_std[:, i],
            alpha=0.16,
            label="+/- 2 std" if i == 0 else None,
        )
        cursor = ax.axvline(t[0], linewidth=1.2, linestyle="--")
        cursor_lines.append(cursor)
        ax.set_title(PARAM_LABELS[PARAM_NAMES[i]], fontsize=9)
        ax.set_xlabel("time [s]", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="best")

    # Loss plots are part of the final dashboard, so the user gets one final vis.
    ax_loss_nom = fig.add_subplot(gs[3, 0:2])
    ax_loss_adapt = fig.add_subplot(gs[3, 2:4])
    updates = np.arange(cfg.updates)
    ax_loss_nom.plot(updates, histories[0]["loss"], label="Nominal REINFORCE")
    ax_loss_nom.plot(updates, histories[1]["loss"], label="Nominal pathwise")
    ax_loss_nom.set_title("Nominal training losses")
    ax_loss_nom.set_xlabel("update")
    ax_loss_nom.set_ylabel("loss")
    ax_loss_nom.legend(fontsize=8)

    ax_loss_adapt.plot(updates, histories[2]["loss"], label="GP-DR robust")
    ax_loss_adapt.plot(updates, histories[3]["loss"], label="GP-Bayes adaptive")
    ax_loss_adapt.set_title("Uncertain-physics training losses")
    ax_loss_adapt.set_xlabel("update")
    ax_loss_adapt.set_ylabel("loss")
    ax_loss_adapt.legend(fontsize=8)

    suptitle = fig.suptitle("", fontsize=13, fontweight="bold")

    stride = max(1, cfg.gif_stride)
    frame_indices = np.arange(0, cfg.horizon + 1, stride)
    if frame_indices[-1] != cfg.horizon:
        frame_indices = np.append(frame_indices, cfg.horizon)

    def update(frame_number):
        k = int(frame_indices[frame_number])
        time_now = t[k]
        true_p = param_path[k]

        for j, (line, txt) in enumerate(zip(field_lines, info_texts)):
            u = states[j][k, :cfg.n_grid]
            line.set_xdata(true_p[3] * xi_full)
            line.set_ydata(np.concatenate([[0.0], u, [0.0]]))
            a = actions[j][min(k, cfg.horizon - 1)] if k < cfg.horizon else actions[j][-1]
            ret = returns[j][min(k, cfg.horizon - 1)] if k > 0 else 0.0
            rms = float(np.sqrt(np.mean(u**2)))
            extra = ""
            if j == 3:
                err = np.mean(
                    np.abs(bayes_mean[k] - true_p)
                    / np.maximum(host(parameter_rel_ranges(cfg) * nominal_parameter_vector(cfg)), 1e-8)
                )
                unc = np.mean(
                    bayes_std[k]
                    / np.maximum(host(parameter_rel_ranges(cfg) * nominal_parameter_vector(cfg)), 1e-8)
                )
                extra = f"\nID error={err:.3f}   uncertainty={unc:.3f}"
            txt.set_text(
                f"force={a:+.2f}   field RMS={rms:.3f}\n"
                f"cumulative reward={ret:.2f}{extra}"
            )

        for cursor in cursor_lines:
            cursor.set_xdata([time_now, time_now])

        suptitle.set_text(
            "Held-out GP deployment — same hidden PDE physics for all controllers\n"
            f"t={time_now:5.3f}s   "
            f"c={true_p[0]:.3f}   gamma={true_p[1]:.3f}   "
            f"k={true_p[2]:.3f}   L={true_p[3]:.3f}"
        )
        return field_lines + info_texts + cursor_lines + [suptitle]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / cfg.gif_fps,
        blit=False,
    )
    anim.save(path, writer=animation.PillowWriter(fps=cfg.gif_fps), dpi=110)
    plt.close(fig)
    return path


final_gif = make_final_dashboard_gif(
    [eval_reinforce, eval_pathwise, eval_dr, eval_bayes],
    [reinforce_history, pathwise_history, dr_history, bayes_history],
    cfg,
    OUT_DIR / "FINAL_wave_pde_four_way_with_losses.gif",
)
display(Image(filename=str(final_gif)))

print(f"\nSTART preview: {start_path}")
print(f"FINAL dashboard: {final_gif}")
