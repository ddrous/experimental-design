# %% [markdown]
# # CartPole: REINFORCE vs pathwise policy gradient
#
# Same two algorithms as the 1D tutorial, now on the classic 4-state CartPole
# task (cart position, cart velocity, pole angle, pole angular velocity).
# The physics simulator is written from scratch in JAX (no Gym/Gymnasium),
# so it is fully differentiable end-to-end — required for the pathwise
# gradient to even be defined.
#
# Run this file cell-by-cell (#%% blocks) in VS Code's Interactive Window,
# Jupytext, or `jupyter nbconvert --to notebook --execute`. GIFs are written
# to disk and displayed inline via IPython.display as training progresses.

# %% Imports and config
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

OUT_DIR = Path("cartpole_out")
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
    updates: int = 100
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

# %% CartPole physics simulator (pure JAX, fully differentiable)
#
# State = [x, x_dot, theta, theta_dot]. theta = 0 is upright; gravity pulls
# it away from 0. Semi-implicit ("symplectic") Euler integration — same
# scheme used by the classic Gym CartPole, but written as a plain function
# with no branching, so jax.grad can differentiate straight through it.


def cartpole_dynamics(state: jax.Array, force: jax.Array, cfg: Config) -> jax.Array:
    x, x_dot, theta, theta_dot = state
    mc, mp, l, g = cfg.cart_mass, cfg.pole_mass, cfg.half_length, cfg.gravity
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


def obs_from_state(state: jax.Array) -> jax.Array:
    """sin/cos encoding of theta avoids the angle-wraparound discontinuity."""
    x, x_dot, theta, theta_dot = state
    return jnp.array([x, x_dot, jnp.sin(theta), jnp.cos(theta), theta_dot])


def reward_fn(state: jax.Array, action: jax.Array, cfg: Config) -> jax.Array:
    x, _, theta, _ = state
    return (
        1.0
        - cfg.theta_penalty * theta**2
        - cfg.x_penalty * x**2
        - cfg.action_penalty * action**2
    )


# %% Cell: draw a single CartPole frame (static check of the renderer)


def draw_cartpole(ax, x: float, theta: float, cfg: Config, track_half_width=2.4):
    ax.cla()
    cart_w, cart_h = 0.4, 0.22
    pole_len = 2 * cfg.half_length

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
    ax.set_ylim(-0.5, pole_len + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


fig, ax = plt.subplots(figsize=(6, 4))
draw_cartpole(ax, x=0.3, theta=0.25, cfg=cfg)
ax.set_title("CartPole renderer sanity check (x=0.3, theta=0.25 rad)")
plt.show()

# %% Cell: animate a random (untrained) rollout as a GIF, shown inline


def simulate_rollout(state0: jax.Array, forces: jax.Array, cfg: Config) -> jax.Array:
    """forces: [T] array of raw force values -> returns states [T+1, 4]."""
    def step(state, force):
        next_state = cartpole_dynamics(state, force, cfg)
        return next_state, next_state
    _, states = jax.lax.scan(step, state0, forces)
    return jnp.concatenate([state0[None], states], axis=0)


def make_gif(states: np.ndarray, cfg: Config, path: Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 4))

    def update(frame):
        draw_cartpole(ax, x=states[frame, 0], theta=states[frame, 2], cfg=cfg)
        ax.set_title(f"{title}  (t={frame})")

    anim = animation.FuncAnimation(fig, update, frames=len(states), interval=1000 / cfg.gif_fps)
    anim.save(path, writer=animation.PillowWriter(fps=cfg.gif_fps))
    plt.close(fig)
    return path


key, subkey = jax.random.split(key)
random_forces = cfg.force_max * jax.random.uniform(subkey, (cfg.horizon,), minval=-1, maxval=1)
state0 = jnp.array([0.0, 0.0, 0.1, 0.0])
random_states = np.asarray(jax.device_get(simulate_rollout(state0, random_forces, cfg)))

gif_path = make_gif(random_states, cfg, OUT_DIR / "random_policy.gif", title="Random forces (untrained)")
display(Image(filename=str(gif_path)))

# %% Policy: diagonal Gaussian over a continuous force, squashed by tanh


class Policy(eqx.Module):
    mlp: eqx.nn.MLP
    raw_log_std: jax.Array

    def __init__(self, key: jax.Array, cfg: Config):
        self.mlp = eqx.nn.MLP(
            in_size=5, out_size=1, width_size=cfg.hidden, depth=cfg.depth,
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

    def step(state, eps_t):
        obs = obs_from_state(state)
        mu, log_std = policy_stats(policy, obs)
        z = mu + jnp.exp(log_std) * eps_t
        force = squash_to_force(z, cfg)
        reward = reward_fn(state, force, cfg)
        next_state = cartpole_dynamics(state, force, cfg)
        data = {"obs": obs, "z": z, "reward": reward, "state": state}
        return next_state, data

    _, traj = jax.lax.scan(step, state0, eps)
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

    def step(state, _):
        obs = obs_from_state(state)
        mu, _ = policy_stats(policy, obs)
        force = squash_to_force(mu, cfg)
        next_state = cartpole_dynamics(state, force, cfg)
        return next_state, next_state

    _, states = jax.lax.scan(step, state0, None, length=cfg.horizon)
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


# %% Train REINFORCE — periodically renders a GIF of the current deterministic policy

key, model_key = jax.random.split(key)
init_policy = Policy(model_key, cfg)

reinforce_policy = init_policy
reinforce_optimizer = make_optimizer(cfg)
reinforce_opt_state = reinforce_optimizer.init(eqx.filter(reinforce_policy, eqx.is_inexact_array))
reinforce_step = make_train_step(make_reinforce_loss(cfg), reinforce_optimizer)
reinforce_history = init_history()

for update in range(cfg.updates):
    key, k = jax.random.split(key)
    keys = jax.random.split(k, cfg.batch_size)
    reinforce_policy, reinforce_opt_state, diag = reinforce_step(reinforce_policy, reinforce_opt_state, keys)
    append_diag(reinforce_history, diag)

    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(f"[reinforce] update={update:04d}  return={reinforce_history['return'][-1]:8.3f}")
        states = np.asarray(jax.device_get(deterministic_rollout(reinforce_policy, cfg)))
        gif_path = make_gif(states, cfg, OUT_DIR / f"reinforce_update_{update:04d}.gif",
                             title=f"REINFORCE — update {update}")
        display(Image(filename=str(gif_path)))

# %% Train pathwise PG — same loop structure, different loss

pathwise_policy = init_policy
pathwise_optimizer = make_optimizer(cfg)
pathwise_opt_state = pathwise_optimizer.init(eqx.filter(pathwise_policy, eqx.is_inexact_array))
pathwise_step = make_train_step(make_pathwise_loss(cfg), pathwise_optimizer)
pathwise_history = init_history()

for update in range(cfg.updates):
    key, k = jax.random.split(key)
    keys = jax.random.split(k, cfg.batch_size)
    pathwise_policy, pathwise_opt_state, diag = pathwise_step(pathwise_policy, pathwise_opt_state, keys)
    append_diag(pathwise_history, diag)

    if update % cfg.eval_every == 0 or update == cfg.updates - 1:
        print(f"[pathwise ] update={update:04d}  return={pathwise_history['return'][-1]:8.3f}")
        states = np.asarray(jax.device_get(deterministic_rollout(pathwise_policy, cfg)))
        gif_path = make_gif(states, cfg, OUT_DIR / f"pathwise_update_{update:04d}.gif",
                             title=f"Pathwise — update {update}")
        display(Image(filename=str(gif_path)))

# %% Cell: loss curves and diagnostics, REINFORCE vs pathwise side by side

histories = {"reinforce": reinforce_history, "pathwise": pathwise_history}

fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6))
for (name, title), ax in zip(METRICS, axes):
    for algo in ("reinforce", "pathwise"):
        y = histories[algo][name]
        ax.plot(np.arange(len(y)), y, label=algo)
    ax.set_title(title)
    ax.set_xlabel("update")
    if name == "grad_norm":
        ax.set_yscale("log")
    ax.legend(fontsize=8)
fig.suptitle("CartPole: REINFORCE vs pathwise policy gradient — training diagnostics")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT_DIR / "diagnostics.png", dpi=150)
plt.show()

# %% Cell: final trained agents, side by side GIF

final_reinforce = np.asarray(jax.device_get(deterministic_rollout(reinforce_policy, cfg)))
final_pathwise = np.asarray(jax.device_get(deterministic_rollout(pathwise_policy, cfg)))


def make_side_by_side_gif(states_a, states_b, cfg, path, label_a="REINFORCE", label_b="Pathwise"):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4))

    def update(frame):
        draw_cartpole(ax_a, x=states_a[frame, 0], theta=states_a[frame, 2], cfg=cfg)
        ax_a.set_title(f"{label_a}  (t={frame})")
        draw_cartpole(ax_b, x=states_b[frame, 0], theta=states_b[frame, 2], cfg=cfg)
        ax_b.set_title(f"{label_b}  (t={frame})")

    n = min(len(states_a), len(states_b))
    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / cfg.gif_fps)
    anim.save(path, writer=animation.PillowWriter(fps=cfg.gif_fps))
    plt.close(fig)
    return path


final_gif = make_side_by_side_gif(final_reinforce, final_pathwise, cfg, OUT_DIR / "final_comparison.gif")
display(Image(filename=str(final_gif)))

print("\nFinal |theta| (0 = perfectly upright):")
print(f"  REINFORCE: {abs(float(final_reinforce[-1, 2])):.4f}")
print(f"  Pathwise:  {abs(float(final_pathwise[-1, 2])):.4f}")
