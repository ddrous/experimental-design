# VERSION: GP-TIME-VARYING-BAYES-v4
# v4: GP time-varying physics with explicit nominal/DR/Bayes baselines and a fixed-layout dashboard.
# Training uses a shared bank of GP functions; deployment uses an independent held-out GP
# function (with a deliberately different temporal length scale by default).

# %% [markdown]
# # CartPole v4: GP time-varying physics + explicit baselines + Bayesian adaptation
#
# Same two algorithms as the 1D tutorial, now on the classic 4-state CartPole
# task (cart position, cart velocity, pole angle, pole angular velocity).
# The physics simulator is written from scratch in JAX (no Gym/Gymnasium),
# so it is fully differentiable end-to-end — required for the pathwise
# gradient to even be defined.
#
# Run this file cell-by-cell (#%% blocks) in VS Code's Interactive Window,
# Jupytext, or `jupyter nbconvert --to notebook --execute`. GIFs are written
# to disk. v4 deliberately displays only one pre-training environment preview and one final comparison GIF.

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
import seaborn as sns

try:
    from IPython.display import display, Image
except ImportError:  # script still runs fine outside Jupyter
    def display(*a, **k): pass
    def Image(*a, **k): return None

sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlepad": 8.0})

OUT_DIR = Path("cartpole_out_v4")
OUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Config:
    # Physics (classic CartPole constants, Florian 2007 continuous-force form)
    gravity: float = 9.8
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    half_length: float = 0.5
    dt: float = 0.02
    force_max: float = 10.0

    # ------------------------------------------------------------------
    # Time-varying / uncertain physics experiment (v4: GP trajectories)
    # ------------------------------------------------------------------
    # Select which latent physical parameter(s) vary DURING an episode.
    # "all" makes cart mass, pole mass, gravity, and half-length vary together.
    uncertain_param: str = "all"
    # Allowed values:
    #   "cart_mass", "pole_mass", "gravity", "half_length", "all"

    # KEEPING YOUR v2 HYPERPARAMETERS: each parameter remains constrained to
    # nominal * (1 +/- rel_range).  v4 keeps the v3 GP mechanism for HOW values move inside that range:
    # the smooth component is now a sampled Gaussian-process function of time.
    cart_mass_rel_range: float = 0.65
    pole_mass_rel_range: float = 0.65
    gravity_rel_range: float = 0.65
    half_length_rel_range: float = 0.65

    # KEEPING YOUR v2 HYPERPARAMETERS: these numbers are retained as the
    # parameter-specific scale of the small, independent noise around the smooth
    # GP function. ``gp_point_noise_scale`` below controls how much of these
    # original values is actually injected at every time step.
    cart_mass_process_rel_std: float = 0.15
    pole_mass_process_rel_std: float = 0.15
    gravity_process_rel_std: float = 0.05
    half_length_process_rel_std: float = 0.10

    # Bayesian online system identification.  ``belief_grid_size`` is kept from
    # v2 for reproducibility/backwards compatibility.  v4 uses the same full-covariance
    # Gaussian belief (EKF-style) rather than four independent grid marginals;
    # this directly addresses the confounding that hurt the v2 Bayes controller
    # when all four physical parameters changed at once.
    belief_grid_size: int = 31
    belief_accel_noise_std: float = 0.75

    # Gaussian-process environment model.  A latent f_p(t) ~ GP(0, k) is sampled
    # for each active physical parameter p, then mapped into its allowed range by
    # tanh.  The training bank is shared by DR and Bayes, with exactly the same
    # rollout keys, so their training physics data are matched episode-for-episode.
    gp_train_function_bank_size: int = 31
    gp_train_length_scale_seconds: float = 0.55

    # Deployment is intentionally a HELD-OUT GP function.  The shorter default
    # length scale makes it a temporal sim-to-real shift, not merely another
    # realization the DR policy can average over.  Set this equal to the training
    # length scale if you want an in-distribution held-out GP test instead.
    gp_deploy_length_scale_seconds: float = 0.18
    gp_latent_std: float = 1.0
    gp_jitter: float = 1e-5

    # True parameters follow the sampled smooth GP function plus a smaller
    # pointwise noise term.  We retain your per-parameter process stds above and
    # multiply them by this common factor so the function, not white noise, is
    # the dominant source of nonstationarity.
    gp_point_noise_scale: float = 0.10

    # Process-noise inflation used by the Bayesian tracker between observations.
    # The tracker is deliberately NOT given the held-out deployment function; it
    # only gets a generic local-drift prior and must infer parameters online.
    bayes_process_noise_scale: float = 0.25
    bayes_stop_gradient_through_filter: bool = True

    # Episode
    horizon: int = 150
    x_init_noise: float = 0.05
    theta_init_noise: float = 0.35   # radians, ~8.5 deg
    veloc_init_noise: float = 0.05

    # Reward shaping (dense, differentiable — needed for the pathwise path)
    theta_penalty: float = 1.0
    x_penalty: float = 0.05
    action_penalty: float = 0.001

    # Policy
    hidden: int = 64
    depth: int = 2
    init_log_std: float = -0.7

    # Optimisation
    updates: int = 500
    batch_size: int = 128
    learning_rate: float = 3e-3
    grad_clip: Optional[float] = 1.0
    gamma: float = 0.99

    # REINFORCE variance reduction
    use_loo_baseline: bool = True
    normalize_advantage: bool = True
    entropy_coef: float = 0.0

    # Visualisation
    eval_every: int = 50
    gif_fps: int = 30

    seed: int = 0


cfg = Config()
key = jax.random.PRNGKey(cfg.seed)


class PhysicsParams(NamedTuple):
    """Physical parameters used by one transition of the simulator.

    Keeping them in a small JAX-friendly tuple lets us change physics at every
    time step without mutating ``cfg`` (which remains a static experiment
    configuration). It also keeps the pathwise simulator differentiable.
    """

    cart_mass: jax.Array
    pole_mass: jax.Array
    half_length: jax.Array
    gravity: jax.Array


def nominal_physics(cfg: Config) -> PhysicsParams:
    """Return the nominal simulator parameters as JAX scalars."""
    return PhysicsParams(
        cart_mass=jnp.asarray(cfg.cart_mass),
        pole_mass=jnp.asarray(cfg.pole_mass),
        half_length=jnp.asarray(cfg.half_length),
        gravity=jnp.asarray(cfg.gravity),
    )


# Canonical ordering used by all vector-valued latent-physics helpers below.
# Keeping one fixed order makes the JAX scan state compact and makes "all"
# almost as easy to simulate as the single-parameter cases.
PHYSICS_PARAM_NAMES = ("cart_mass", "pole_mass", "half_length", "gravity")
PHYSICS_PARAM_LABELS = {
    "cart_mass": "cart mass",
    "pole_mass": "pole mass",
    "half_length": "half length",
    "gravity": "gravity",
}
PHYSICS_PARAM_UNITS = {
    "cart_mass": "kg",
    "pole_mass": "kg",
    "half_length": "m",
    "gravity": "m/s^2",
}
NUM_PHYSICS_PARAMS = len(PHYSICS_PARAM_NAMES)

# v4 shared meta-data: every policy observes normalized episode time in addition
# to the original five CartPole state features.  Nobody observes the true GP
# function or physical parameters.
BASE_OBS_SIZE = 6


def nominal_parameter_vector(cfg: Config) -> jax.Array:
    """Return [cart_mass, pole_mass, half_length, gravity]."""
    return jnp.asarray([
        cfg.cart_mass,
        cfg.pole_mass,
        cfg.half_length,
        cfg.gravity,
    ])


def parameter_rel_ranges(cfg: Config) -> jax.Array:
    """Independent +/- relative uncertainty margins for all four parameters."""
    return jnp.asarray([
        cfg.cart_mass_rel_range,
        cfg.pole_mass_rel_range,
        cfg.half_length_rel_range,
        cfg.gravity_rel_range,
    ])


def parameter_process_rel_stds(cfg: Config) -> jax.Array:
    """Per-parameter relative scales retained from v2 for local/noise variation."""
    return jnp.asarray([
        cfg.cart_mass_process_rel_std,
        cfg.pole_mass_process_rel_std,
        cfg.half_length_process_rel_std,
        cfg.gravity_process_rel_std,
    ])


def active_parameter_indices(cfg: Config) -> tuple[int, ...]:
    """Indices of the parameters that are latent/time-varying in this run."""
    if cfg.uncertain_param == "all":
        return tuple(range(NUM_PHYSICS_PARAMS))
    if cfg.uncertain_param in PHYSICS_PARAM_NAMES:
        return (PHYSICS_PARAM_NAMES.index(cfg.uncertain_param),)
    raise ValueError(
        f"uncertain_param={cfg.uncertain_param!r} is invalid; choose "
        "'cart_mass', 'pole_mass', 'half_length', 'gravity', or 'all'."
    )


def active_parameter_names(cfg: Config) -> tuple[str, ...]:
    """Names corresponding to ``active_parameter_indices``."""
    return tuple(PHYSICS_PARAM_NAMES[i] for i in active_parameter_indices(cfg))


def active_parameter_mask(cfg: Config) -> jax.Array:
    """Boolean mask [4] used to freeze non-selected parameters at nominal."""
    mask = [False] * NUM_PHYSICS_PARAMS
    for i in active_parameter_indices(cfg):
        mask[i] = True
    return jnp.asarray(mask)


def adaptive_policy_input_size(cfg: Config) -> int:
    """Shared 6-D state/time metadata + (posterior mean, std) per active parameter."""
    return BASE_OBS_SIZE + 2 * len(active_parameter_indices(cfg))


def physics_from_parameter_vector(values: jax.Array) -> PhysicsParams:
    """Convert the canonical 4-vector into parameters for one transition."""
    return PhysicsParams(
        cart_mass=values[0],
        pole_mass=values[1],
        half_length=values[2],
        gravity=values[3],
    )


# %% CartPole physics simulator (pure JAX, fully differentiable)
#
# State = [x, x_dot, theta, theta_dot]. theta = 0 is upright; gravity pulls
# it away from 0. Semi-implicit ("symplectic") Euler integration — same
# scheme used by the classic Gym CartPole, but written as a plain function
# with no branching, so jax.grad can differentiate straight through it.


def cartpole_dynamics(
    state: jax.Array,
    force: jax.Array,
    cfg: Config,
    physics: Optional[PhysicsParams] = None,
) -> jax.Array:
    """Advance CartPole by one integration step.

    ``physics=None`` reproduces the ORIGINAL nominal simulator exactly. Passing
    a ``PhysicsParams`` object is the new hook used by the uncertain-physics
    experiments below, where the parameter can change as t evolves.
    """
    x, x_dot, theta, theta_dot = state
    if physics is None:
        physics = nominal_physics(cfg)
    mc, mp, l, g = (
        physics.cart_mass, physics.pole_mass, physics.half_length, physics.gravity
    )
    total_mass = mc + mp

    sin_th, cos_th = jnp.sin(theta), jnp.cos(theta)
    temp = (force + mp * l * theta_dot**2 * sin_th) / total_mass
    theta_acc = (g * sin_th - cos_th * temp) / (
        l * (4.0 / 3.0 - mp * cos_th**2 / total_mass)
    )
    x_acc = temp - mp * l * theta_acc * cos_th / total_mass

    x_dot_new = x_dot + cfg.dt * x_acc
    x_new = x + cfg.dt * x_dot_new
    theta_dot_new = theta_dot + cfg.dt * theta_acc
    theta_new = theta + cfg.dt * theta_dot_new
    return jnp.array([x_new, x_dot_new, theta_new, theta_dot_new])


def obs_from_state(state: jax.Array, tau: jax.Array = 0.0) -> jax.Array:
    """Return the shared policy observation.

    v4 appends normalized episode time ``tau in [-1, 1]`` to the original five
    CartPole features.  This is the SAME external meta-data for nominal, DR, and
    Bayesian policies.  Importantly, the latent GP value / physical parameters
    are still hidden; time only tells the controller where it is in the episode.
    """
    x, x_dot, theta, theta_dot = state
    return jnp.array([x, x_dot, jnp.sin(theta), jnp.cos(theta), theta_dot, tau])


def normalized_episode_times(cfg: Config) -> jax.Array:
    """One normalized time feature for every control action, shape [horizon]."""
    if cfg.horizon <= 1:
        return jnp.zeros((cfg.horizon,))
    return jnp.linspace(-1.0, 1.0, cfg.horizon)


def reward_fn(state: jax.Array, action: jax.Array, cfg: Config) -> jax.Array:
    x, _, theta, _ = state
    return (
        1.0
        - cfg.theta_penalty * theta**2
        - cfg.x_penalty * x**2
        - cfg.action_penalty * action**2
    )


# %% Cell: draw a single CartPole frame (static check of the renderer)


def draw_cartpole(
    ax, x: float, theta: float, cfg: Config, track_half_width=2.4, half_length=None
):
    ax.cla()
    cart_w, cart_h = 0.4, 0.22
    # ``half_length`` is optional so all original visualisations remain
    # unchanged, while the uncertain-length experiment can visibly resize the
    # pole from frame to frame.
    pole_len = 2 * (cfg.half_length if half_length is None else half_length)

    ax.plot([-track_half_width, track_half_width], [0, 0], color="0.6", linewidth=2, zorder=0)
    cart = plt.Rectangle((x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
                          facecolor="#3b6ea5", edgecolor="black", zorder=2)
    ax.add_patch(cart)
    pole_x = x + pole_len * np.sin(theta)
    pole_y = cart_h / 2 + pole_len * np.cos(theta)
    ax.plot([x, pole_x], [cart_h / 2, pole_y], color="#c0392b", linewidth=4, zorder=3,
            solid_capstyle="round")
    ax.plot(x, cart_h / 2, "o", color="black", ms=4, zorder=4)

    ax.set_xlim(-track_half_width, track_half_width)

    # IMPORTANT FOR ANIMATIONS: keep the y-limits fixed across every frame.
    # In v2 this limit used the *current* pole length, so when half_length varied
    # Matplotlib's equal-aspect axes physically changed height from frame to frame.
    # We instead reserve enough vertical room for the largest allowed pole.
    max_half_length = cfg.half_length * (1.0 + cfg.half_length_rel_range)
    max_pole_len = 2.0 * max_half_length
    ax.set_ylim(-0.5, max_pole_len + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


# v4 intentionally emits no renderer sanity plot or random-policy GIF here.
# The first visualization is the GP environment preview immediately before
# training; the second and final visualization is the four-controller dashboard.

# %% Policy: diagonal Gaussian over a continuous force, squashed by tanh


class Policy(eqx.Module):
    mlp: eqx.nn.MLP
    raw_log_std: jax.Array

    def __init__(self, key: jax.Array, cfg: Config, in_size: int = BASE_OBS_SIZE):
        # v4 gives EVERY policy the same 6-D base observation: the original five
        # state features plus normalized episode time.  The Bayesian adaptive
        # policy appends one posterior-mean and one posterior-std feature for each
        # active physical parameter (8 inputs for one parameter, 14 for "all").
        self.mlp = eqx.nn.MLP(
            in_size=in_size, out_size=1, width_size=cfg.hidden, depth=cfg.depth,
            activation=jax.nn.tanh, key=key,
        )
        self.raw_log_std = jnp.asarray(cfg.init_log_std, dtype=jnp.float32)


def policy_stats(policy: Policy, obs: jax.Array):
    mu = policy.mlp(obs)[0]
    log_std = policy.raw_log_std
    return mu, log_std


def squash_to_force(z: jax.Array, cfg: Config) -> jax.Array:
    return cfg.force_max * jnp.tanh(z)


def normal_log_prob(z, mu, log_std):
    inv_std = jnp.exp(-log_std)
    return -0.5 * ((z - mu) * inv_std) ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)


def normal_entropy(log_std):
    return log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)


def rollout_one(policy: Policy, key: jax.Array, cfg: Config):
    k_state, k_eps = jax.random.split(key)
    ks = jax.random.split(k_state, 3)
    x0 = cfg.x_init_noise * jax.random.normal(ks[0], ())
    xdot0 = cfg.veloc_init_noise * jax.random.normal(ks[1], ())
    theta0 = cfg.theta_init_noise * jax.random.normal(ks[2], ())
    state0 = jnp.array([x0, xdot0, theta0, 0.0])
    eps = jax.random.normal(k_eps, (cfg.horizon,))
    taus = normalized_episode_times(cfg)

    def step(state, scan_input):
        eps_t, tau_t = scan_input
        obs = obs_from_state(state, tau_t)
        mu, log_std = policy_stats(policy, obs)
        z = mu + jnp.exp(log_std) * eps_t
        force = squash_to_force(z, cfg)
        reward = reward_fn(state, force, cfg)
        next_state = cartpole_dynamics(state, force, cfg)
        data = {"obs": obs, "z": z, "reward": reward, "state": state}
        return next_state, data

    _, traj = jax.lax.scan(step, state0, (eps, taus))
    return traj


def rollout_batch(policy, keys, cfg):
    return jax.vmap(lambda k: rollout_one(policy, k, cfg))(keys)


def discounted_returns_to_go(rewards, gamma):
    def scan_fn(carry, r):
        c = r + gamma * carry
        return c, c
    _, rev = jax.lax.scan(scan_fn, 0.0, rewards[::-1])
    return rev[::-1]


def discounted_episode_return(rewards, gamma):
    discounts = gamma ** jnp.arange(rewards.shape[-1])
    return jnp.sum(discounts * rewards, axis=-1)


# %% REINFORCE loss — score function estimator, reward used only as a scalar weight
#
# Gradient path: theta -> mu,log_std -> log pi(z|s)   [differentiated]
#                theta -> mu,log_std -> z -> force -> state -> reward   [stop_gradient'd]


def make_reinforce_loss(cfg: Config):
    def loss_fn(policy: Policy, keys: jax.Array):
        traj = rollout_batch(policy, keys, cfg)
        rewards = traj["reward"]
        rtg = jax.vmap(lambda r: discounted_returns_to_go(r, cfg.gamma))(rewards)

        if cfg.use_loo_baseline and cfg.batch_size > 1:
            baseline = (jnp.sum(rtg, axis=0, keepdims=True) - rtg) / (cfg.batch_size - 1)
        else:
            baseline = jnp.zeros_like(rtg)
        advantage = rtg - baseline
        raw_adv_std = jnp.std(advantage)
        if cfg.normalize_advantage:
            advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

        obs_sg = jax.lax.stop_gradient(traj["obs"])
        z_sg = jax.lax.stop_gradient(traj["z"])

        def log_prob_one(o, z):
            mu, log_std = policy_stats(policy, o)
            return normal_log_prob(z, mu, log_std)

        logp = jax.vmap(jax.vmap(log_prob_one))(obs_sg, z_sg)
        discounts = cfg.gamma ** jnp.arange(cfg.horizon)
        weighted_adv = discounts[None, :] * jax.lax.stop_gradient(advantage)

        entropy = jax.vmap(
            jax.vmap(lambda o: normal_entropy(policy_stats(policy, o)[1]))
        )(obs_sg)

        surrogate = jnp.mean(jnp.sum(weighted_adv * logp, axis=1))
        entropy_bonus = cfg.entropy_coef * jnp.mean(jnp.sum(entropy, axis=1))
        loss = -(surrogate + entropy_bonus)

        diag = {
            "return": jnp.mean(discounted_episode_return(rewards, cfg.gamma)),
            "final_theta": jnp.mean(jnp.abs(traj["state"][:, -1, 2])),
            "log_std": policy.raw_log_std,
            "adv_std": raw_adv_std,
        }
        return loss, diag

    return loss_fn


# %% Pathwise loss — full backprop through the simulator, no stop_gradient anywhere
#
# Gradient path: theta -> mu,log_std -> z -> force -> state_{t+1} -> reward -> ... -> J
# Every arrow is live; this needs cartpole_dynamics to be differentiable, which it is.


def make_pathwise_loss(cfg: Config):
    def loss_fn(policy: Policy, keys: jax.Array):
        traj = rollout_batch(policy, keys, cfg)
        ep_return = discounted_episode_return(traj["reward"], cfg.gamma)
        entropy_bonus = cfg.entropy_coef * cfg.horizon * normal_entropy(policy.raw_log_std)
        loss = -(jnp.mean(ep_return) + entropy_bonus)

        diag = {
            "return": jnp.mean(ep_return),
            "final_theta": jnp.mean(jnp.abs(traj["state"][:, -1, 2])),
            "log_std": policy.raw_log_std,
            "adv_std": jnp.asarray(0.0),
        }
        return loss, diag

    return loss_fn


# %% Optimiser / train-step helper


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    sq = [jnp.sum(v**2) for v in leaves if v is not None and eqx.is_inexact_array(v)]
    return jnp.sqrt(sum(sq))


def make_optimizer(cfg: Config):
    if cfg.grad_clip is None:
        return optax.adam(cfg.learning_rate)
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.learning_rate))


def make_train_step(loss_fn, optimizer):
    value_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    @eqx.filter_jit
    def step(policy, opt_state, keys):
        (loss, diag), grads = value_and_grad(policy, keys)
        grad_norm = tree_l2_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(policy, eqx.is_inexact_array))
        policy = eqx.apply_updates(policy, updates)
        diag = {**diag, "loss": loss, "grad_norm": grad_norm}
        return policy, opt_state, diag

    return step


def deterministic_rollout(policy: Policy, cfg: Config):
    state0 = jnp.array([0.0, 0.0, cfg.theta_init_noise, 0.0])
    taus = normalized_episode_times(cfg)

    def step(state, tau_t):
        obs = obs_from_state(state, tau_t)
        mu, _ = policy_stats(policy, obs)
        force = squash_to_force(mu, cfg)
        next_state = cartpole_dynamics(state, force, cfg)
        return next_state, next_state

    _, states = jax.lax.scan(step, state0, taus)
    return jnp.concatenate([state0[None], states], axis=0)


METRICS = [
    ("return", "Discounted return (higher is better)"),
    ("loss", "Optimisation loss"),
    ("final_theta", "|theta| at episode end (lower is better)"),
    ("grad_norm", "Gradient norm"),
    ("log_std", "Learned log std (exploration)"),
]


def init_history():
    return {k: [] for k, _ in METRICS}


def append_diag(history, diag):
    host = jax.device_get(diag)
    for k in history:
        history[k].append(float(host[k]))


# %% Nominal baselines are trained later, after the one pre-training preview
#
# v4 keeps the original nominal REINFORCE and nominal pathwise algorithms, but
# delays their training until the GP train/deployment functions have been sampled
# and visualized. This makes the experiment produce exactly one visualization at
# the start, then no visualization during any of the four training loops.


# %% [markdown]
# # v4: Gaussian-process physical parameters + Bayesian adaptation
#
# WHY THIS VERSION IS DIFFERENT FROM v2
# -------------------------------------
# v2 used an independent bounded random-walk increment at every time step.  A
# sufficiently strong state-feedback policy can often absorb that disturbance
# without explicitly identifying the physics, which is exactly what happened in
# the user's experiment: the nominal and DR policies stayed upright while the
# more complicated Bayesian policy suffered from identification error.
#
# v4 keeps the hidden environment STRUCTURED rather than i.i.d.-like:
#
#       f_p(t) ~ GP(0, k_p(t,t'))
#       phi_p(t) = phi_p,nominal * [1 + clip(r_p tanh(f_p(t)) + eta_p(t), -r_p, r_p)]
#
# where eta_p(t) is a smaller pointwise noise term.  Thus mass / gravity / length
# are values of smooth UNKNOWN FUNCTIONS OF TIME.  The transition remains
#
#       s_{t+1} = dynamics(s_t, a_t ; phi(t)).
#
# TRAIN vs DEPLOY
# ----------------
# We sample a finite BANK of GP functions once for training.  DR and Bayes use
# exactly the SAME bank and exactly the SAME rollout keys at every optimizer
# update.  Deployment uses an independently sampled held-out GP function.  The
# default deployment length scale is shorter than training, deliberately making
# this a temporal sim-to-real shift; set the two length scales equal for an IID
# held-out-function experiment.
#
# SHARED META-DATA
# ----------------
# Every policy sees the same external meta-data: the usual CartPole state plus
# normalized episode time tau in [-1,1].  No method receives the true physical
# parameter values or the sampled GP function.  Bayesian adaptation gets only an
# INTERNAL belief inferred from observed (state_t, action_t, state_{t+1}).
#
# BAYESIAN TRACKER
# ----------------
# v2 used four independent 1-D categorical marginals.  With four unknown physical
# parameters but only two measured accelerations, that factorization is badly
# confounded.  v4 therefore keeps a small full-covariance Gaussian belief over
# [cart_mass, pole_mass, half_length, gravity] and an EKF-style measurement
# update.  This is still deliberately lightweight, but it preserves posterior
# cross-correlations and is considerably more appropriate in "all" mode.


class GaussianParameterBelief(NamedTuple):
    """Gaussian belief N(mean, cov) over the four current physical parameters."""

    mean: jax.Array       # [4]
    cov: jax.Array        # [4,4]


def gp_time_grid(cfg: Config) -> jax.Array:
    """Physical time points t_0,...,t_T used to evaluate sampled GP functions."""
    return jnp.arange(cfg.horizon + 1, dtype=jnp.float32) * cfg.dt


def rbf_gp_cholesky(cfg: Config, length_scale_seconds: float) -> jax.Array:
    """Cholesky factor of an RBF/squared-exponential GP covariance matrix.

    k(t,t') = sigma_f^2 exp(-(t-t')^2 / (2 ell^2)).

    We sample the GP only on the finite simulation time grid, which is exactly
    what is required because the simulator asks for a parameter value only at
    those control times.
    """
    t = gp_time_grid(cfg)
    delta = t[:, None] - t[None, :]
    ell = jnp.maximum(jnp.asarray(length_scale_seconds), 1e-4)
    K = (cfg.gp_latent_std ** 2) * jnp.exp(-0.5 * (delta / ell) ** 2)
    K = K + cfg.gp_jitter * jnp.eye(t.shape[0])
    return jnp.linalg.cholesky(K)


def sample_gp_parameter_paths(
    key: jax.Array,
    n_paths: int,
    cfg: Config,
    *,
    length_scale_seconds: float,
) -> jax.Array:
    """Sample hidden physical-parameter FUNCTIONS, shape [N,T+1,4].

    For every path and every active physical parameter we first sample a latent
    smooth GP function f_p(t).  We then map it through tanh into that parameter's
    own +/- margin.  Finally we add smaller pointwise noise and clip back to the
    same legal margin:

        relative_offset_p(t)
            = clip(r_p tanh(f_p(t)) + alpha sigma_p eps_t, -r_p, r_p)

        phi_p(t) = phi_p,nominal * (1 + relative_offset_p(t)).

    This exactly implements "evaluate a sampled function of time, with some
    noise, and use that value in the CartPole dynamics".  Inactive parameters
    remain identically nominal.
    """
    k_gp, k_noise = jax.random.split(key)
    L = rbf_gp_cholesky(cfg, length_scale_seconds)  # [T+1,T+1]

    # Independent GP draws for each physical parameter, but temporally correlated
    # within each parameter according to the RBF kernel.
    z = jax.random.normal(
        k_gp, (n_paths, NUM_PHYSICS_PARAMS, cfg.horizon + 1)
    )
    latent = jnp.einsum("ij,npj->npi", L, z)          # [N,4,T+1]
    latent = jnp.transpose(latent, (0, 2, 1))         # [N,T+1,4]

    nominal = nominal_parameter_vector(cfg)
    rel_range = parameter_rel_ranges(cfg)
    point_std = parameter_process_rel_stds(cfg)

    smooth_relative = rel_range[None, None, :] * jnp.tanh(latent)
    point_noise = (
        cfg.gp_point_noise_scale
        * point_std[None, None, :]
        * jax.random.normal(k_noise, smooth_relative.shape)
    )
    relative = jnp.clip(
        smooth_relative + point_noise,
        -rel_range[None, None, :],
        rel_range[None, None, :],
    )
    paths = nominal[None, None, :] * (1.0 + relative)

    active = active_parameter_mask(cfg)[None, None, :]
    return jnp.where(active, paths, nominal[None, None, :])


def initial_parameter_belief(cfg: Config) -> GaussianParameterBelief:
    """Broad Gaussian prior matching each allowed physical-parameter margin.

    A uniform distribution on [nominal-half_width, nominal+half_width] has
    variance half_width^2/3.  We use that variance as a sensible Gaussian prior
    scale, while inactive parameters are essentially known exactly.
    """
    nominal = nominal_parameter_vector(cfg)
    half_width = parameter_rel_ranges(cfg) * nominal
    active = active_parameter_mask(cfg)
    var = jnp.where(active, (half_width ** 2) / 3.0, 1e-12)
    return GaussianParameterBelief(mean=nominal, cov=jnp.diag(var))


def belief_moments(belief: GaussianParameterBelief, cfg: Config):
    """Posterior mean/std for all four physical parameters."""
    del cfg
    std = jnp.sqrt(jnp.maximum(jnp.diag(belief.cov), 1e-12))
    return belief.mean, std


def normalized_belief_features(
    belief: GaussianParameterBelief,
    cfg: Config,
) -> jax.Array:
    """Interleaved normalized (posterior mean offset, posterior std) features."""
    mean, std = belief_moments(belief, cfg)
    nominal = nominal_parameter_vector(cfg)
    half_width = jnp.maximum(parameter_rel_ranges(cfg) * nominal, 1e-8)
    idx = jnp.asarray(active_parameter_indices(cfg))

    mean_feature = ((mean - nominal) / half_width)[idx]
    std_feature = (std / half_width)[idx]
    return jnp.stack([mean_feature, std_feature], axis=1).reshape(-1)


def normalized_active_parameter_error(
    estimate: jax.Array,
    truth: jax.Array,
    cfg: Config,
) -> jax.Array:
    """Mean |estimate-truth| normalized by each active parameter's own margin."""
    nominal = nominal_parameter_vector(cfg)
    half_width = jnp.maximum(parameter_rel_ranges(cfg) * nominal, 1e-8)
    idx = jnp.asarray(active_parameter_indices(cfg))
    return jnp.mean(jnp.abs(estimate[idx] - truth[idx]) / half_width[idx])


def normalized_active_belief_std(std: jax.Array, cfg: Config) -> jax.Array:
    """Mean posterior std normalized by each active parameter's own margin."""
    nominal = nominal_parameter_vector(cfg)
    half_width = jnp.maximum(parameter_rel_ranges(cfg) * nominal, 1e-8)
    idx = jnp.asarray(active_parameter_indices(cfg))
    return jnp.mean(std[idx] / half_width[idx])


def acceleration_from_parameter_vector(
    state: jax.Array,
    force: jax.Array,
    parameter_vector: jax.Array,
    cfg: Config,
) -> jax.Array:
    """Two measured accelerations predicted by one candidate physical model."""
    next_state = cartpole_dynamics(
        state,
        force,
        cfg,
        physics_from_parameter_vector(parameter_vector),
    )
    return jnp.asarray([
        (next_state[1] - state[1]) / cfg.dt,
        (next_state[3] - state[3]) / cfg.dt,
    ])


def bayes_parameter_update(
    belief: GaussianParameterBelief,
    state: jax.Array,
    force: jax.Array,
    next_state_observed: jax.Array,
    cfg: Config,
) -> GaussianParameterBelief:
    """Full-covariance EKF-style Bayesian measurement update.

    Observation:
        y_t = [x_ddot, theta_ddot] measured from the observed state transition.

    Nonlinear measurement model:
        y_t = h(phi_t; state_t, force_t) + v_t.

    We linearize h around the current posterior mean and perform the standard
    Gaussian Bayes / Kalman update.  Unlike v2's factorized marginals, the 4x4
    covariance can express correlations such as mass-length or mass-gravity
    tradeoffs induced by the same acceleration observation.
    """
    mean, P = belief
    observed_accel = jnp.asarray([
        (next_state_observed[1] - state[1]) / cfg.dt,
        (next_state_observed[3] - state[3]) / cfg.dt,
    ])

    h = lambda params: acceleration_from_parameter_vector(state, force, params, cfg)
    predicted_accel = h(mean)
    H = jax.jacfwd(h)(mean)  # [2,4]

    # Inactive physical parameters are known, so remove their columns from the
    # measurement sensitivity rather than letting numerical noise move them.
    active = active_parameter_mask(cfg).astype(mean.dtype)
    H = H * active[None, :]

    R = (cfg.belief_accel_noise_std ** 2) * jnp.eye(2)
    S = H @ P @ H.T + R + 1e-8 * jnp.eye(2)
    K = jnp.linalg.solve(S, H @ P).T  # P H^T S^{-1}, shape [4,2]

    innovation = observed_accel - predicted_accel
    mean_post = mean + K @ innovation

    # Respect the same physical bounds used to generate the GP environment.
    nominal = nominal_parameter_vector(cfg)
    half_width = parameter_rel_ranges(cfg) * nominal
    lo, hi = nominal - half_width, nominal + half_width
    mean_post = jnp.clip(mean_post, lo, hi)
    mean_post = jnp.where(active.astype(bool), mean_post, nominal)

    # Joseph-form covariance update is numerically safer / keeps P PSD.
    I = jnp.eye(NUM_PHYSICS_PARAMS)
    IKH = I - K @ H
    P_post = IKH @ P @ IKH.T + K @ R @ K.T
    P_post = 0.5 * (P_post + P_post.T)

    active_outer = active[:, None] * active[None, :]
    P_post = P_post * active_outer + jnp.diag((1.0 - active) * 1e-12)
    P_post = P_post + jnp.diag(active * 1e-10)
    return GaussianParameterBelief(mean_post, P_post)


def predict_parameter_belief(
    posterior: GaussianParameterBelief,
    cfg: Config,
) -> GaussianParameterBelief:
    """Local process prediction between two observations.

    The controller is NOT handed the sampled GP function.  It only assumes that
    parameters can move locally, so we inflate covariance by Q.  This deliberately
    flexible model lets the tracker adapt when the deployment GP function differs
    from those used for training.
    """
    nominal = nominal_parameter_vector(cfg)
    active = active_parameter_mask(cfg).astype(posterior.mean.dtype)

    q_std = (
        cfg.bayes_process_noise_scale
        * parameter_process_rel_stds(cfg)
        * nominal
    )
    Q = jnp.diag(active * (q_std ** 2))
    P_next = posterior.cov + Q
    P_next = 0.5 * (P_next + P_next.T)

    mean_next = jnp.where(active.astype(bool), posterior.mean, nominal)
    return GaussianParameterBelief(mean_next, P_next)


def policy_belief_features(
    belief: GaussianParameterBelief,
    cfg: Config,
) -> jax.Array:
    """Features passed to the adaptive policy, optionally detached from gradients.

    Stopping gradients through the estimator is a practical adaptive-control
    baseline: the policy still CONDITIONS on the inferred belief at run time,
    but pathwise training does not require fragile second-order derivatives
    through the EKF Jacobian.  Set the config flag False to recover the fully
    differentiable/dual-control-style path.
    """
    features = normalized_belief_features(belief, cfg)
    if cfg.bayes_stop_gradient_through_filter:
        features = jax.lax.stop_gradient(features)
    return features


def rollout_one_gp_physics(
    policy: Policy,
    key: jax.Array,
    cfg: Config,
    training_gp_bank: jax.Array,
    *,
    adaptive: bool,
):
    """Training rollout under one hidden GP-parameter function from a shared bank.

    ``training_gp_bank`` has shape [B,T+1,4].  The rollout key chooses one bank
    member.  Because DR and Bayes receive the SAME keys at every update, they see
    exactly the same sequence of GP-function indices, initial states, and action-
    noise draws during training.  The policies of course generate different
    actions, but the exogenous training metadata is matched.
    """
    k_state, k_action, k_bank = jax.random.split(key, 3)

    ks = jax.random.split(k_state, 3)
    x0 = cfg.x_init_noise * jax.random.normal(ks[0], ())
    xdot0 = cfg.veloc_init_noise * jax.random.normal(ks[1], ())
    theta0 = cfg.theta_init_noise * jax.random.normal(ks[2], ())
    state0 = jnp.array([x0, xdot0, theta0, 0.0])

    action_eps = jax.random.normal(k_action, (cfg.horizon,))
    taus = normalized_episode_times(cfg)

    bank_idx = jax.random.randint(
        k_bank, (), 0, training_gp_bank.shape[0]
    )
    true_path = training_gp_bank[bank_idx]  # [T+1,4]

    belief0 = initial_parameter_belief(cfg)
    nominal = nominal_parameter_vector(cfg)
    _, prior_std = belief_moments(belief0, cfg)

    def step(carry, scan_input):
        state, belief = carry
        eps_action_t, tau_t, true_params_t = scan_input

        base_obs = obs_from_state(state, tau_t)
        if adaptive:
            policy_obs = jnp.concatenate([
                base_obs,
                policy_belief_features(belief, cfg),
            ])
        else:
            policy_obs = base_obs

        mu, log_std = policy_stats(policy, policy_obs)
        z = mu + jnp.exp(log_std) * eps_action_t
        force = squash_to_force(z, cfg)
        reward = reward_fn(state, force, cfg)

        # The GP is evaluated at the CURRENT time and that physical vector is
        # passed directly into the CartPole differential equation.
        next_state = cartpole_dynamics(
            state,
            force,
            cfg,
            physics_from_parameter_vector(true_params_t),
        )

        if adaptive:
            posterior_t = bayes_parameter_update(
                belief, state, force, next_state, cfg
            )
            belief_mean, belief_std = belief_moments(posterior_t, cfg)
            next_belief = predict_parameter_belief(posterior_t, cfg)
        else:
            belief_mean = nominal
            belief_std = prior_std
            next_belief = belief

        data = {
            "obs": policy_obs,
            "z": z,
            "reward": reward,
            "state": state,
            "force": force,
            "true_params": true_params_t,
            "belief_mean": belief_mean,
            "belief_std": belief_std,
        }
        return (next_state, next_belief), data

    (final_state, final_belief), traj = jax.lax.scan(
        step,
        (state0, belief0),
        (action_eps, taus, true_path[:-1]),
    )
    final_mean, final_std = belief_moments(final_belief, cfg)
    return {
        **traj,
        "final_state": final_state,
        "final_true_params": true_path[-1],
        "final_belief_mean": final_mean,
        "final_belief_std": final_std,
    }


def rollout_batch_gp_physics(
    policy,
    keys,
    cfg,
    training_gp_bank,
    *,
    adaptive: bool,
):
    """Vectorized GP-physics training rollouts."""
    return jax.vmap(
        lambda k: rollout_one_gp_physics(
            policy, k, cfg, training_gp_bank, adaptive=adaptive
        )
    )(keys)


def make_gp_pathwise_loss(
    cfg: Config,
    training_gp_bank: jax.Array,
    *,
    adaptive: bool,
):
    """Pathwise objective under latent GP physical functions."""
    def loss_fn(policy: Policy, keys: jax.Array):
        traj = rollout_batch_gp_physics(
            policy, keys, cfg, training_gp_bank, adaptive=adaptive
        )
        ep_return = discounted_episode_return(traj["reward"], cfg.gamma)
        entropy_bonus = (
            cfg.entropy_coef * cfg.horizon * normal_entropy(policy.raw_log_std)
        )
        loss = -(jnp.mean(ep_return) + entropy_bonus)

        normalized_errors = jax.vmap(
            jax.vmap(
                lambda est, truth: normalized_active_parameter_error(
                    est, truth, cfg
                )
            )
        )(traj["belief_mean"], traj["true_params"])
        normalized_stds = jax.vmap(
            jax.vmap(lambda std: normalized_active_belief_std(std, cfg))
        )(traj["belief_std"])

        diag = {
            "return": jnp.mean(ep_return),
            "final_theta": jnp.mean(jnp.abs(traj["state"][:, -1, 2])),
            "log_std": policy.raw_log_std,
            "adv_std": jnp.asarray(0.0),
            "param_abs_error": jnp.mean(normalized_errors),
            "belief_std": jnp.mean(normalized_stds),
        }
        return loss, diag

    return loss_fn


ADAPTIVE_METRICS = [
    ("return", "Discounted return (higher is better)"),
    ("loss", "Optimisation loss"),
    ("final_theta", "|theta| at episode end"),
    ("grad_norm", "Gradient norm"),
    ("log_std", "Learned action log std"),
    ("param_abs_error", "Mean normalized active-parameter ID error"),
    ("belief_std", "Mean normalized active posterior std"),
]


def init_adaptive_history():
    return {k: [] for k, _ in ADAPTIVE_METRICS}


def append_adaptive_diag(history, diag):
    host_diag = jax.device_get(diag)
    for k in history:
        history[k].append(float(host_diag[k]))


# %% Build ONE shared GP training-function bank and ONE shared rollout-key schedule
#
# This is the fairness mechanism for v4.  DR and Bayes do not merely
# draw from the same distribution; update u / episode b uses the exact same
# exogenous GP training function and random seeds for both methods.
key, k_gp_bank, k_training_schedule = jax.random.split(key, 3)
training_gp_bank = sample_gp_parameter_paths(
    k_gp_bank,
    cfg.gp_train_function_bank_size,
    cfg,
    length_scale_seconds=cfg.gp_train_length_scale_seconds,
)
training_key_schedule = jax.random.split(
    k_training_schedule,
    cfg.updates * cfg.batch_size,
).reshape(cfg.updates, cfg.batch_size, 2)

# Sample the deployment function BEFORE training only for evaluator visualization.
# It is never passed to an optimizer or policy, so this does not leak deployment
# physics to any method. fold_in creates a separate deterministic PRNG stream
# without perturbing the shared training schedule.
k_deploy_gp = jax.random.fold_in(k_gp_bank, 202603)
deployment_parameter_path = sample_gp_parameter_paths(
    k_deploy_gp,
    1,
    cfg,
    length_scale_seconds=cfg.gp_deploy_length_scale_seconds,
)[0]


# %% START VISUALIZATION — environment/function preview (the only pre-training vis)
def plot_gp_experiment_preview(training_gp_bank, deployment_parameter_path, cfg, path):
    """Show what changes between training and deployment before any policy trains.

    The policies do NOT receive these curves. This figure is evaluator-only and
    makes the train/deployment GP shift explicit. Several training-bank functions
    are shown faintly; the held-out deployment function is emphasized.
    """
    train = np.asarray(jax.device_get(training_gp_bank))
    deploy = np.asarray(jax.device_get(deployment_parameter_path))
    t = np.asarray(jax.device_get(gp_time_grid(cfg)))
    nominal = np.asarray(jax.device_get(nominal_parameter_vector(cfg)))
    rel = np.asarray(jax.device_get(parameter_rel_ranges(cfg)))
    active = active_parameter_indices(cfg)

    if len(active) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(9.5, 4.8))
        axes = np.asarray([axes])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8))
        axes = np.asarray(axes).ravel()

    n_show = min(8, train.shape[0])
    for ax, p_idx in zip(axes, active):
        name = PHYSICS_PARAM_NAMES[p_idx]
        unit = PHYSICS_PARAM_UNITS[name]
        lo = nominal[p_idx] * (1.0 - rel[p_idx])
        hi = nominal[p_idx] * (1.0 + rel[p_idx])

        for j in range(n_show):
            ax.plot(
                t,
                train[j, :, p_idx],
                linewidth=1.0,
                alpha=0.22,
                label="training GP samples" if j == 0 else None,
            )
        ax.plot(t, deploy[:, p_idx], linewidth=2.6, label="held-out deployment GP")
        ax.axhline(nominal[p_idx], linestyle="--", linewidth=1.2, label="nominal")
        ax.axhline(lo, linestyle=":", linewidth=0.9)
        ax.axhline(hi, linestyle=":", linewidth=0.9)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_title(f"{PHYSICS_PARAM_LABELS[name]}  [{unit}]")
        ax.set_xlabel("time [s]")
        ax.legend(fontsize=8, loc="best")

    for ax in axes[len(active):]:
        ax.axis("off")

    fig.suptitle(
        "START — latent physical functions used by the experiment\n"
        f"train GP length scale={cfg.gp_train_length_scale_seconds:g}s  |  "
        f"held-out deployment={cfg.gp_deploy_length_scale_seconds:g}s",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.show()
    return path


preview_path = plot_gp_experiment_preview(
    training_gp_bank,
    deployment_parameter_path,
    cfg,
    OUT_DIR / f"START_gp_environment_{cfg.uncertain_param}.png",
)


# %% Train ORIGINAL nominal-physics baselines — NO training-time visualization
#
# These are the original controls against which the GP methods are evaluated:
#   1) nominal REINFORCE
#   2) nominal pathwise policy gradient
# Both are trained only with the constant nominal simulator and both receive the
# same base state + normalized-time metadata as DR/Bayes. They never see the GP
# physical parameters or the Bayesian belief.
key, model_key = jax.random.split(key)
init_policy = Policy(model_key, cfg)

reinforce_policy = init_policy
reinforce_optimizer = make_optimizer(cfg)
reinforce_opt_state = reinforce_optimizer.init(
    eqx.filter(reinforce_policy, eqx.is_inexact_array)
)
reinforce_step = make_train_step(make_reinforce_loss(cfg), reinforce_optimizer)
reinforce_history = init_history()

for update in range(cfg.updates):
    key, k = jax.random.split(key)
    keys = jax.random.split(k, cfg.batch_size)
    reinforce_policy, reinforce_opt_state, diag = reinforce_step(
        reinforce_policy, reinforce_opt_state, keys
    )
    append_diag(reinforce_history, diag)
    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(
            f"[nominal REINFORCE] update={update:04d}  "
            f"return={reinforce_history['return'][-1]:8.3f}"
        )

pathwise_policy = init_policy
pathwise_optimizer = make_optimizer(cfg)
pathwise_opt_state = pathwise_optimizer.init(
    eqx.filter(pathwise_policy, eqx.is_inexact_array)
)
pathwise_step = make_train_step(make_pathwise_loss(cfg), pathwise_optimizer)
pathwise_history = init_history()

for update in range(cfg.updates):
    key, k = jax.random.split(key)
    keys = jax.random.split(k, cfg.batch_size)
    pathwise_policy, pathwise_opt_state, diag = pathwise_step(
        pathwise_policy, pathwise_opt_state, keys
    )
    append_diag(pathwise_history, diag)
    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(
            f"[nominal Pathwise]  update={update:04d}  "
            f"return={pathwise_history['return'][-1]:8.3f}"
        )


# %% Train GP dynamics-randomized robust policy (NO online identification)
#
# DR sees the same shared state/time metadata as Bayes and is trained over the
# shared bank of GP functions, but it has no memory/belief about which function
# is active.  It must compress that whole family into one feedback law.
dr_policy = init_policy
dr_optimizer = make_optimizer(cfg)
dr_opt_state = dr_optimizer.init(eqx.filter(dr_policy, eqx.is_inexact_array))
dr_step = make_train_step(
    make_gp_pathwise_loss(cfg, training_gp_bank, adaptive=False),
    dr_optimizer,
)
dr_history = init_adaptive_history()

for update in range(cfg.updates):
    keys = training_key_schedule[update]
    dr_policy, dr_opt_state, diag = dr_step(
        dr_policy, dr_opt_state, keys
    )
    append_adaptive_diag(dr_history, diag)

    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(
            f"[GP-DR robust] update={update:04d}  "
            f"return={dr_history['return'][-1]:8.3f}"
        )


# %% Train Bayesian adaptive policy on the EXACT SAME GP training functions/keys
key, adaptive_model_key = jax.random.split(key)
bayes_policy = Policy(
    adaptive_model_key,
    cfg,
    in_size=adaptive_policy_input_size(cfg),
)
bayes_optimizer = make_optimizer(cfg)
bayes_opt_state = bayes_optimizer.init(
    eqx.filter(bayes_policy, eqx.is_inexact_array)
)
bayes_step = make_train_step(
    make_gp_pathwise_loss(cfg, training_gp_bank, adaptive=True),
    bayes_optimizer,
)
bayes_history = init_adaptive_history()

for update in range(cfg.updates):
    keys = training_key_schedule[update]
    bayes_policy, bayes_opt_state, diag = bayes_step(
        bayes_policy, bayes_opt_state, keys
    )
    append_adaptive_diag(bayes_history, diag)

    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(
            f"[GP-Bayes adapt] update={update:04d}  "
            f"return={bayes_history['return'][-1]:8.3f}  "
            f"normalized ID-error={bayes_history['param_abs_error'][-1]:.4g}  "
            f"normalized posterior-std={bayes_history['belief_std'][-1]:.4g}"
        )


# %% Deterministic evaluation on ONE HELD-OUT DEPLOYMENT GP FUNCTION

def deterministic_gp_rollout(
    policy: Policy,
    cfg: Config,
    *,
    adaptive: bool,
    true_parameter_path: jax.Array,
    theta0: float = 0.20,
):
    """Evaluate a policy on a prescribed latent GP physical-parameter path.

    All controllers receive the exact same ``true_parameter_path`` and normalized
    time metadata.  Only Bayes also maintains an inferred internal belief.
    """
    state0 = jnp.array([0.0, 0.0, theta0, 0.0])
    belief0 = initial_parameter_belief(cfg)
    nominal = nominal_parameter_vector(cfg)
    _, prior_std = belief_moments(belief0, cfg)
    taus = normalized_episode_times(cfg)

    def step(carry, scan_input):
        state, belief = carry
        tau_t, true_params_t = scan_input

        base_obs = obs_from_state(state, tau_t)
        if adaptive:
            policy_obs = jnp.concatenate([
                base_obs,
                policy_belief_features(belief, cfg),
            ])
        else:
            policy_obs = base_obs

        mu, _ = policy_stats(policy, policy_obs)
        force = squash_to_force(mu, cfg)
        reward = reward_fn(state, force, cfg)
        next_state = cartpole_dynamics(
            state,
            force,
            cfg,
            physics_from_parameter_vector(true_params_t),
        )

        if adaptive:
            posterior_t = bayes_parameter_update(
                belief, state, force, next_state, cfg
            )
            mean, std = belief_moments(posterior_t, cfg)
            next_belief = predict_parameter_belief(posterior_t, cfg)
        else:
            mean = nominal
            std = prior_std
            next_belief = belief

        data = {
            "state": state,
            "reward": reward,
            "force": force,
            "true_params": true_params_t,
            "belief_mean": mean,
            "belief_std": std,
        }
        return (next_state, next_belief), data

    (final_state, final_belief), traj = jax.lax.scan(
        step,
        (state0, belief0),
        (taus, true_parameter_path[:-1]),
    )
    final_mean, final_std = belief_moments(final_belief, cfg)

    return {
        **traj,
        "states_with_final": jnp.concatenate(
            [traj["state"], final_state[None]], axis=0
        ),
        "params_with_final": true_parameter_path,
        "belief_mean_with_final": jnp.concatenate(
            [traj["belief_mean"], final_mean[None]], axis=0
        ),
        "belief_std_with_final": jnp.concatenate(
            [traj["belief_std"], final_std[None]], axis=0
        ),
    }


# %% Deterministic FINAL evaluation — all four methods, same held-out GP path
#
# Both original nominal baselines are now explicit.  DR and Bayes are evaluated
# beside them on exactly the same hidden deployment function and initial state.
# No method sees the true physical parameters.
eval_nominal_reinforce = deterministic_gp_rollout(
    reinforce_policy,
    cfg,
    adaptive=False,
    true_parameter_path=deployment_parameter_path,
)
eval_nominal_pathwise = deterministic_gp_rollout(
    pathwise_policy,
    cfg,
    adaptive=False,
    true_parameter_path=deployment_parameter_path,
)
eval_dr = deterministic_gp_rollout(
    dr_policy,
    cfg,
    adaptive=False,
    true_parameter_path=deployment_parameter_path,
)
eval_bayes = deterministic_gp_rollout(
    bayes_policy,
    cfg,
    adaptive=True,
    true_parameter_path=deployment_parameter_path,
)


def host(x):
    return np.asarray(jax.device_get(x))


def eval_return(eval_traj):
    return float(discounted_episode_return(eval_traj["reward"], cfg.gamma))


print("\nHeld-out GP deployment evaluation (IDENTICAL latent function for all):")
print(f"  uncertain parameter(s) : {cfg.uncertain_param}")
print(f"  train GP ell [s]        : {cfg.gp_train_length_scale_seconds:.3g}")
print(f"  deploy GP ell [s]       : {cfg.gp_deploy_length_scale_seconds:.3g}")
print(f"  nominal REINFORCE       : {eval_return(eval_nominal_reinforce):8.3f}")
print(f"  nominal Pathwise        : {eval_return(eval_nominal_pathwise):8.3f}")
print(f"  GP-DR robust            : {eval_return(eval_dr):8.3f}")
print(f"  GP-Bayes adaptive       : {eval_return(eval_bayes):8.3f}")

final_true = host(eval_bayes["params_with_final"][-1])
final_mean = host(eval_bayes["belief_mean_with_final"][-1])
final_std = host(eval_bayes["belief_std_with_final"][-1])
print("  final Bayesian estimates:")
for i in active_parameter_indices(cfg):
    name = PHYSICS_PARAM_NAMES[i]
    print(
        f"    {name:11s}: true={final_true[i]:.5g}, "
        f"mean={final_mean[i]:.5g}, std={final_std[i]:.3g}"
    )


# %% FINAL VISUALIZATION — four-way fixed-layout deployment dashboard
#
# This is the only visualization after training.  It compares BOTH original
# nominal baselines, the GP dynamics-randomized baseline, and the Bayesian
# adaptive controller on the exact same held-out physical-function realization.
#
# Unlike the old renderer, this dashboard NEVER clears an axes during animation.
# Every cart/pole is a persistent Matplotlib artist whose coordinates are updated
# in place.  Axis limits, subplot geometry, and the maximum possible pole length
# are fixed once before frame 0, so boxes cannot jump or change height.
def make_final_four_way_dashboard_gif(
    eval_reinforce,
    eval_pathwise,
    eval_dr,
    eval_bayes,
    cfg,
    path,
):
    """High-information deployment animation with fixed geometry.

    Top/middle: four controllers on the SAME hidden physics path.
      - Nominal REINFORCE
      - Nominal Pathwise
      - GP-DR robust
      - GP-Bayes adaptive

    Bottom: live true physical parameters plus the Bayesian estimate and +/-2 std.
    Thus the viewer can simultaneously see control quality, the actual changing
    dynamics, and whether Bayesian identification is keeping up.
    """
    evals = [eval_reinforce, eval_pathwise, eval_dr, eval_bayes]
    labels = [
        "Nominal REINFORCE",
        "Nominal Pathwise",
        "GP-DR robust",
        "GP-Bayes adaptive",
    ]
    training_notes = [
        "trained on nominal physics",
        "trained on nominal physics",
        "trained on GP physics; no identifier",
        "trained on GP physics; EKF belief",
    ]

    states = [host(e["states_with_final"]) for e in evals]
    forces = [host(e["force"]) for e in evals]
    rewards = [host(e["reward"]) for e in evals]
    params = host(eval_bayes["params_with_final"])
    means = host(eval_bayes["belief_mean_with_final"])
    stds = host(eval_bayes["belief_std_with_final"])

    n = min(len(x) for x in states)
    t = np.arange(n) * cfg.dt
    active = active_parameter_indices(cfg)
    nominal = host(nominal_parameter_vector(cfg))
    rel = host(parameter_rel_ranges(cfg))
    half_width = np.maximum(rel * nominal, 1e-8)

    # Running discounted return aligned to STATE frames: frame 0 has received no
    # reward yet; frame k contains rewards from transitions 0,...,k-1.
    running_returns = []
    for r in rewards:
        discounts = cfg.gamma ** np.arange(len(r))
        running_returns.append(np.concatenate([[0.0], np.cumsum(discounts * r)]))

    # Choose one fixed horizontal range large enough to show all four trajectories.
    max_abs_x = max(float(np.max(np.abs(s[:, 0]))) for s in states)
    track_half_width = max(2.4, max_abs_x + 0.55)

    # Choose one fixed vertical range from the LARGEST legal half-length, not the
    # current one. This directly prevents the changing-height bug from v2/v3.
    max_half_length = cfg.half_length * (1.0 + cfg.half_length_rel_range)
    max_pole_len = 2.0 * max_half_length
    cart_w, cart_h = 0.40, 0.22

    fig = plt.figure(figsize=(15.8, 10.4))
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=(1.0, 1.0, 0.72),
        hspace=0.34,
        wspace=0.20,
        left=0.045,
        right=0.985,
        bottom=0.075,
        top=0.90,
    )
    cart_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]

    # Persistent artists: no ax.cla(), no tight_layout() inside update().
    cart_artists = []
    metric_texts = []
    for ax, label, note in zip(cart_axes, labels, training_notes):
        ax.set_xlim(-track_half_width, track_half_width)
        ax.set_ylim(-0.55, max_pole_len + 0.55)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0.0, linewidth=2.0, color="0.65", zorder=0)
        ax.set_title(label, fontsize=12, fontweight="bold")

        cart = plt.Rectangle(
            (-cart_w / 2, -cart_h / 2),
            cart_w,
            cart_h,
            facecolor="#3b6ea5",
            edgecolor="black",
            linewidth=1.1,
            zorder=2,
        )
        ax.add_patch(cart)
        pole, = ax.plot(
            [0.0, 0.0],
            [cart_h / 2, cart_h / 2 + 2.0 * cfg.half_length],
            color="#c0392b",
            linewidth=4.2,
            solid_capstyle="round",
            zorder=3,
        )
        pivot, = ax.plot([0.0], [cart_h / 2], "o", color="black", ms=4.5, zorder=4)
        metrics = ax.text(
            0.018,
            0.975,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="0.8"),
        )
        ax.text(
            0.018,
            0.035,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="0.35",
        )
        cart_artists.append((cart, pole, pivot))
        metric_texts.append(metrics)

    # Parameter monitor axes. One active parameter uses the whole row; two split
    # it in half; three/four use equal-width cells. The geometry is fixed once.
    if len(active) == 1:
        param_axes = [fig.add_subplot(gs[2, :])]
    elif len(active) == 2:
        param_axes = [fig.add_subplot(gs[2, :2]), fig.add_subplot(gs[2, 2:])]
    else:
        param_axes = [fig.add_subplot(gs[2, i]) for i in range(len(active))]

    param_artists = []
    for ax_i, (ax, p_idx) in enumerate(zip(param_axes, active)):
        name = PHYSICS_PARAM_NAMES[p_idx]
        unit = PHYSICS_PARAM_UNITS[name]
        lo = nominal[p_idx] * (1.0 - rel[p_idx])
        hi = nominal[p_idx] * (1.0 + rel[p_idx])
        pad = max(0.04 * (hi - lo), 1e-6)

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(lo - pad, hi + pad)
        ax.axhline(nominal[p_idx], linestyle="--", linewidth=1.0, color="0.55")
        true_line, = ax.plot([], [], linewidth=2.0, label="true hidden GP")
        mean_line, = ax.plot([], [], linewidth=1.8, label="Bayes mean")
        lower_line, = ax.plot([], [], linestyle=":", linewidth=1.0, label="Bayes +/- 2 std")
        upper_line, = ax.plot([], [], linestyle=":", linewidth=1.0)
        true_dot, = ax.plot([], [], "o", ms=5.0)
        mean_dot, = ax.plot([], [], "o", ms=4.0)
        cursor = ax.axvline(0.0, linewidth=0.9, color="0.35", alpha=0.65)
        live = ax.text(
            0.02,
            0.96,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86, edgecolor="0.85"),
        )
        ax.set_title(f"{PHYSICS_PARAM_LABELS[name]} [{unit}]", fontsize=10)
        ax.set_xlabel("time [s]", fontsize=8.5)
        ax.tick_params(labelsize=8)
        if ax_i == 0:
            ax.legend(fontsize=7.2, loc="lower left", ncol=2)
        param_artists.append(
            (p_idx, true_line, mean_line, lower_line, upper_line, true_dot, mean_dot, cursor, live)
        )

    # Header/footer are persistent text objects whose contents change, not geometry.
    header = fig.text(
        0.5,
        0.965,
        "",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    physics_header = fig.text(
        0.5,
        0.932,
        "",
        ha="center",
        va="top",
        fontsize=10.5,
        family="monospace",
    )
    fig.text(
        0.5,
        0.018,
        "All four controllers see the same state + time metadata and the same held-out GP physics. "
        "Only Bayes receives an internally inferred belief.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.30",
    )

    def update(frame):
        p = params[frame]
        dynamic_half_length = p[PHYSICS_PARAM_NAMES.index("half_length")]
        frame_action = min(frame, cfg.horizon - 1)

        header.set_text(
            f"HELD-OUT GP DEPLOYMENT — same physics for every controller   |   t={t[frame]:.2f} s"
        )
        physics_header.set_text(
            "true physics:  "
            f"mc={p[0]:.4f} kg   |   mp={p[1]:.4f} kg   |   "
            f"half-length={p[2]:.4f} m   |   g={p[3]:.4f} m/s^2"
        )

        for method_i, (s, f, artists, metrics) in enumerate(
            zip(states, forces, cart_artists, metric_texts)
        ):
            x = float(s[frame, 0])
            theta = float(s[frame, 2])
            force = float(f[frame_action])
            pole_len = 2.0 * float(dynamic_half_length)
            pole_x = x + pole_len * np.sin(theta)
            pole_y = cart_h / 2 + pole_len * np.cos(theta)

            cart, pole, pivot = artists
            cart.set_xy((x - cart_w / 2, -cart_h / 2))
            pole.set_data([x, pole_x], [cart_h / 2, pole_y])
            pivot.set_data([x], [cart_h / 2])

            text = (
                f"x={x:+7.3f} m    theta={theta:+7.3f} rad\\n"
                f"force={force:+6.2f} N   discounted return={running_returns[method_i][frame]:+8.2f}"
            )
            if method_i == 3:
                idx = np.asarray(active, dtype=int)
                id_err = np.mean(np.abs(means[frame, idx] - p[idx]) / half_width[idx])
                unc = np.mean(stds[frame, idx] / half_width[idx])
                text += f"\\nnormalized ID error={id_err:.3f}   uncertainty={unc:.3f}"
            metrics.set_text(text)

        prefix = slice(0, frame + 1)
        for (
            p_idx,
            true_line,
            mean_line,
            lower_line,
            upper_line,
            true_dot,
            mean_dot,
            cursor,
            live,
        ) in param_artists:
            true_line.set_data(t[prefix], params[prefix, p_idx])
            mean_line.set_data(t[prefix], means[prefix, p_idx])
            lower_line.set_data(t[prefix], means[prefix, p_idx] - 2.0 * stds[prefix, p_idx])
            upper_line.set_data(t[prefix], means[prefix, p_idx] + 2.0 * stds[prefix, p_idx])
            true_dot.set_data([t[frame]], [params[frame, p_idx]])
            mean_dot.set_data([t[frame]], [means[frame, p_idx]])
            cursor.set_xdata([t[frame], t[frame]])
            unit = PHYSICS_PARAM_UNITS[PHYSICS_PARAM_NAMES[p_idx]]
            live.set_text(
                f"true={params[frame, p_idx]:.4g} {unit}   |   "
                f"Bayes={means[frame, p_idx]:.4g} +/- {stds[frame, p_idx]:.2g}"
            )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n,
        interval=1000 / cfg.gif_fps,
        blit=False,
    )
    anim.save(path, writer=animation.PillowWriter(fps=cfg.gif_fps))
    plt.close(fig)
    return path


final_dashboard_gif = make_final_four_way_dashboard_gif(
    eval_nominal_reinforce,
    eval_nominal_pathwise,
    eval_dr,
    eval_bayes,
    cfg,
    OUT_DIR / f"FINAL_gp_deployment_{cfg.uncertain_param}_four_way.gif",
)
display(Image(filename=str(final_dashboard_gif)))


# %% [markdown]
# ## v4 experiment summary
#
# Main data-generating process:
#
#     f_p(t) ~ GP(0, RBF kernel)
#     phi_p(t) = phi_p,nominal * [1 + clip(r_p*tanh(f_p(t)) + noise, -r_p, r_p)]
#     s_{t+1} = dynamics(s_t, a_t; phi(t))
#
# Training and deployment use different sampled functions.  By default they also
# use different GP temporal length scales (0.55 s training, 0.18 s deployment),
# creating a deliberately nontrivial temporal sim-to-real shift.  Set
# ``gp_deploy_length_scale_seconds = gp_train_length_scale_seconds`` for a pure
# held-out-function test from the same GP prior.
#
# Fairness:
#   * Every policy observes the same 6-D [state, normalized time] metadata.
#   * DR and Bayes train on the exact same finite bank of GP functions.
#   * DR and Bayes use the exact same rollout-key schedule at every update.
#   * Nobody sees the true physical parameters / GP function values.
#   * BOTH original nominal policies (REINFORCE and Pathwise) are retained as baselines.
#   * GP-DR is a separate robust baseline with no online identifier.
#
# Bayesian v4:
#   * full 4-D Gaussian mean + 4x4 covariance belief,
#   * EKF-style update from observed cart/pole accelerations,
#   * posterior mean/std are appended to the adaptive policy observation,
#   * cross-parameter correlations are retained instead of factorized away.
#
# The final GIF fixes its axes limits using the maximum allowed pole length, so
# changing half_length no longer changes subplot height from frame to frame.
