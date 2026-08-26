"""
Tiny continuous-control tutorial: REINFORCE vs pathwise policy gradient
using JAX + Equinox + Optax.

Task
----
A point starts near x=-1 and must move to TARGET=+1 in HORIZON steps.
The policy outputs a Gaussian latent action z. The environment receives
either tanh(z) (recommended) or clip(z, -1, 1) (a deliberate pitfall demo).

Why this design
----------------
Both REINFORCE and the pathwise gradient are computed from the *same*
rollout function. The only difference is which quantities we let
gradients flow through:

  * REINFORCE (score-function estimator): we treat the sampled action and
    the resulting trajectory as constants, and only differentiate
    log pi(a|s) through the policy network. The reward signal enters only
    as a scalar "advantage" that multiplies the score.

  * Pathwise (reparameterisation) gradient: we differentiate the reward
    itself, straight through the sampled action z = mu + std * eps and
    through the environment dynamics. This needs a *differentiable*
    environment (true here, rare in real RL).

Install:
    pip install "jax[cpu]" equinox optax matplotlib seaborn

Run:
    python reinforce_vs_pathwise_jax.py
This trains both algorithms and saves a dashboard PNG. Nothing here
depends on an interactive display, so it works the same in a terminal,
over SSH, or in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import seaborn as sns

sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})


# -----------------------------------------------------------------------------
# 1. Hyperparameters: edit these first
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # Environment
    horizon: int = 32
    target: float = 1.0
    start: float = -1.0
    start_noise: float = 0.05
    step_scale: float = 0.12
    action_cost: float = 0.01
    gamma: float = 1.0  # Keep 1.0 initially: easiest exact comparison.

    # Policy
    hidden: int = 32
    depth: int = 2
    init_log_std: float = -0.5
    log_std_min: float = -3.0
    log_std_max: float = 0.5

    # "smooth" is recommended. "hard" demonstrates dead-gradient clipping.
    log_std_mode: str = "smooth"  # {"smooth", "hard"}

    # "tanh" is recommended. "clip" is intentionally a useful failure-mode demo.
    action_mode: str = "tanh"  # {"tanh", "clip"}

    # Optimisation
    updates: int = 300
    batch_size: int = 64
    learning_rate: float = 3e-3
    grad_clip: Optional[float] = 1.0  # None disables global-norm clipping.

    # REINFORCE variance-reduction knobs
    use_loo_baseline: bool = True
    normalize_advantage: bool = True

    # Exploration bonus. Leave at zero first so the comparison is clean.
    entropy_coef: float = 0.0

    seed: int = 0


CFG = Config()


# -----------------------------------------------------------------------------
# 2. Policy
# -----------------------------------------------------------------------------

class Policy(eqx.Module):
    mlp: eqx.nn.MLP
    raw_log_std: jax.Array

    def __init__(self, key: jax.Array, cfg: Config):
        self.mlp = eqx.nn.MLP(
            in_size=3,
            out_size=1,
            width_size=cfg.hidden,
            depth=cfg.depth,
            activation=jax.nn.tanh,
            key=key,
        )
        # Initialise raw_log_std so the *effective* log std is approximately
        # cfg.init_log_std under whichever squashing mode is active.
        if cfg.log_std_mode == "smooth":
            p = (cfg.init_log_std - cfg.log_std_min) / (cfg.log_std_max - cfg.log_std_min)
            p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
            raw = np.log(p / (1.0 - p))
        else:
            raw = cfg.init_log_std
        self.raw_log_std = jnp.asarray(raw, dtype=jnp.float32)


def effective_log_std(policy: Policy, cfg: Config) -> jax.Array:
    """Bound policy scale, either smoothly (sigmoid) or with a hard clip."""
    if cfg.log_std_mode == "smooth":
        p = jax.nn.sigmoid(policy.raw_log_std)
        return cfg.log_std_min + (cfg.log_std_max - cfg.log_std_min) * p
    elif cfg.log_std_mode == "hard":
        # Pitfall: once raw_log_std moves outside the interval, jnp.clip has
        # zero derivative there and the parameter can get "stuck".
        return jnp.clip(policy.raw_log_std, cfg.log_std_min, cfg.log_std_max)
    raise ValueError(f"Unknown log_std_mode={cfg.log_std_mode!r}")


def policy_stats(policy: Policy, obs: jax.Array, cfg: Config):
    mu = policy.mlp(obs)[0]
    log_std = effective_log_std(policy, cfg)
    return mu, log_std


def squash_action(z: jax.Array, cfg: Config) -> jax.Array:
    if cfg.action_mode == "tanh":
        return jnp.tanh(z)
    elif cfg.action_mode == "clip":
        # Deliberate pitfall for pathwise gradients: d clip(z)/dz = 0
        # outside [-1, 1], so the pathwise gradient can vanish entirely.
        return jnp.clip(z, -1.0, 1.0)
    raise ValueError(f"Unknown action_mode={cfg.action_mode!r}")


def normal_log_prob(z, mu, log_std):
    inv_std = jnp.exp(-log_std)
    return -0.5 * ((z - mu) * inv_std) ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)


def normal_entropy(log_std):
    return log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)


# -----------------------------------------------------------------------------
# 3. Tiny differentiable environment
# -----------------------------------------------------------------------------

def rollout_one(policy: Policy, key: jax.Array, cfg: Config):
    """One stochastic trajectory. z = mu + std*eps is reparameterised, so
    this same rollout supports both algorithms: REINFORCE stop-gradients the
    sampled z and state; pathwise PG differentiates straight through them."""
    k_start, k_eps = jax.random.split(key)
    x0 = cfg.start + cfg.start_noise * jax.random.normal(k_start, ())
    eps = jax.random.normal(k_eps, (cfg.horizon,))
    ts = jnp.arange(cfg.horizon)

    def step(x, inp):
        t, eps_t = inp
        t_frac = t / max(cfg.horizon - 1, 1)
        obs = jnp.array([x, cfg.target - x, t_frac])
        mu, log_std = policy_stats(policy, obs, cfg)
        z = mu + jnp.exp(log_std) * eps_t
        action = squash_action(z, cfg)
        x_next = x + cfg.step_scale * action
        reward = -(x_next - cfg.target) ** 2 - cfg.action_cost * action**2
        data = {"obs": obs, "z": z, "action": action, "reward": reward, "x_next": x_next}
        return x_next, data

    _, traj = jax.lax.scan(step, x0, (ts, eps))
    traj["x0"] = x0
    return traj


def rollout_batch(policy: Policy, keys: jax.Array, cfg: Config):
    return jax.vmap(lambda k: rollout_one(policy, k, cfg))(keys)


def discounted_returns_to_go(rewards: jax.Array, gamma: float):
    """G_t = r_t + gamma r_{t+1} + ... for one trajectory."""
    def scan_fn(carry, reward):
        new_carry = reward + gamma * carry
        return new_carry, new_carry
    _, rev = jax.lax.scan(scan_fn, 0.0, rewards[::-1])
    return rev[::-1]


def discounted_episode_return(rewards: jax.Array, gamma: float):
    discounts = gamma ** jnp.arange(rewards.shape[-1])
    return jnp.sum(discounts * rewards, axis=-1)


# -----------------------------------------------------------------------------
# 4. REINFORCE (score-function / likelihood-ratio gradient)
# -----------------------------------------------------------------------------

def make_reinforce_loss(cfg: Config):
    def loss_fn(policy: Policy, keys: jax.Array):
        traj = rollout_batch(policy, keys, cfg)
        rewards = traj["reward"]  # [B, T]
        rtg = jax.vmap(lambda r: discounted_returns_to_go(r, cfg.gamma))(rewards)

        # Leave-one-out baseline: for trajectory i, average the *other*
        # trajectories' returns. This reduces variance without biasing the
        # gradient, because it doesn't depend on trajectory i's own actions.
        if cfg.use_loo_baseline and cfg.batch_size > 1:
            baseline = (jnp.sum(rtg, axis=0, keepdims=True) - rtg) / (cfg.batch_size - 1)
        else:
            baseline = jnp.zeros_like(rtg)

        advantage = rtg - baseline
        raw_adv_std = jnp.std(advantage)
        if cfg.normalize_advantage:
            advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)

        # Crucial REINFORCE detail: recompute log pi(a|s) while treating both
        # the state and the sampled z as constants. This prevents accidental
        # pathwise gradients leaking through the environment or through
        # z = mu + std*eps.
        obs_sg = jax.lax.stop_gradient(traj["obs"])
        z_sg = jax.lax.stop_gradient(traj["z"])

        def log_prob_one(o, z):
            mu, log_std = policy_stats(policy, o, cfg)
            return normal_log_prob(z, mu, log_std)

        logp = jax.vmap(jax.vmap(log_prob_one))(obs_sg, z_sg)

        discounts = cfg.gamma ** jnp.arange(cfg.horizon)
        weighted_adv = discounts[None, :] * jax.lax.stop_gradient(advantage)

        entropy = jax.vmap(
            jax.vmap(lambda o: normal_entropy(policy_stats(policy, o, cfg)[1]))
        )(obs_sg)

        surrogate = jnp.mean(jnp.sum(weighted_adv * logp, axis=1))
        entropy_bonus = cfg.entropy_coef * jnp.mean(jnp.sum(entropy, axis=1))
        loss = -(surrogate + entropy_bonus)

        diag = {
            "return": jnp.mean(discounted_episode_return(rewards, cfg.gamma)),
            "final_error": jnp.mean(jnp.abs(traj["x_next"][:, -1] - cfg.target)),
            "log_std": effective_log_std(policy, cfg),
            "action_sat": jnp.mean(jnp.abs(traj["action"]) > 0.95),
            "adv_std": raw_adv_std,
        }
        return loss, diag

    return loss_fn


# -----------------------------------------------------------------------------
# 5. Pathwise policy gradient
# -----------------------------------------------------------------------------

def make_pathwise_loss(cfg: Config):
    def loss_fn(policy: Policy, keys: jax.Array):
        traj = rollout_batch(policy, keys, cfg)
        # No stop_gradient anywhere: JAX differentiates
        # theta -> mu/std -> z -> action -> x_{t+1} -> reward directly.
        ep_return = discounted_episode_return(traj["reward"], cfg.gamma)
        entropy = normal_entropy(effective_log_std(policy, cfg))
        entropy_bonus = cfg.entropy_coef * cfg.horizon * entropy
        loss = -(jnp.mean(ep_return) + entropy_bonus)

        diag = {
            "return": jnp.mean(ep_return),
            "final_error": jnp.mean(jnp.abs(traj["x_next"][:, -1] - cfg.target)),
            "log_std": effective_log_std(policy, cfg),
            "action_sat": jnp.mean(jnp.abs(traj["action"]) > 0.95),
            "adv_std": jnp.asarray(0.0),  # not used by pathwise PG
        }
        return loss, diag

    return loss_fn


# -----------------------------------------------------------------------------
# 6. Optimiser + train step
# -----------------------------------------------------------------------------

def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    sq = [jnp.sum(x**2) for x in leaves if x is not None and eqx.is_inexact_array(x)]
    return jnp.sqrt(sum(sq))


def make_optimizer(cfg: Config):
    if cfg.grad_clip is None:
        return optax.adam(cfg.learning_rate)
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optax.adam(cfg.learning_rate))


def make_train_step(loss_fn, optimizer, cfg: Config):
    value_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    @eqx.filter_jit
    def step(policy, opt_state, keys):
        (loss, diag), grads = value_and_grad(policy, keys)
        grad_norm = tree_l2_norm(grads)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(policy, eqx.is_inexact_array)
        )
        policy = eqx.apply_updates(policy, updates)
        diag = {**diag, "loss": loss, "grad_norm": grad_norm}
        return policy, opt_state, diag

    return step


# -----------------------------------------------------------------------------
# 7. Deterministic evaluation (mean action, no sampling noise)
# -----------------------------------------------------------------------------

def deterministic_rollout(policy: Policy, cfg: Config):
    x0 = jnp.asarray(cfg.start)
    ts = jnp.arange(cfg.horizon)

    def step(x, t):
        t_frac = t / max(cfg.horizon - 1, 1)
        obs = jnp.array([x, cfg.target - x, t_frac])
        mu, _ = policy_stats(policy, obs, cfg)
        action = squash_action(mu, cfg)
        x_next = x + cfg.step_scale * action
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, ts)
    return jnp.concatenate([x0[None], xs], axis=0)


# -----------------------------------------------------------------------------
# 8. Training loop (headless: collect full histories first, no live plotting)
# -----------------------------------------------------------------------------

METRICS = [
    ("return", "Discounted return (higher is better)"),
    ("loss", "Optimisation loss"),
    ("final_error", "Final |x - target| (lower is better)"),
    ("grad_norm", "Gradient norm"),
    ("log_std", "Effective log std (exploration)"),
    ("action_sat", "Action saturation fraction"),
]


def init_history():
    return {k: [] for k, _ in METRICS}


def append_diag(history, diag):
    host = jax.device_get(diag)
    for k in history:
        history[k].append(float(host[k]))


def train(cfg: Config, algo: str):
    """algo in {'reinforce', 'pathwise'}"""
    key = jax.random.PRNGKey(cfg.seed)
    key, model_key = jax.random.split(key)
    policy = Policy(model_key, cfg)

    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_inexact_array))
    loss_fn = make_reinforce_loss(cfg) if algo == "reinforce" else make_pathwise_loss(cfg)
    step_fn = make_train_step(loss_fn, optimizer, cfg)

    history = init_history()
    for update in range(cfg.updates):
        key, k = jax.random.split(key)
        keys = jax.random.split(k, cfg.batch_size)
        policy, opt_state, diag = step_fn(policy, opt_state, keys)
        append_diag(history, diag)
        if update % 50 == 0 or update == cfg.updates - 1:
            print(f"  [{algo:>9s}] update={update:03d}  return={history['return'][-1]:8.3f}")

    return policy, history


# -----------------------------------------------------------------------------
# 9. Dashboard (static, saved to file — works in any environment)
# -----------------------------------------------------------------------------

def draw_dashboard(cfg, histories, trajectories, save_path="dashboard.png"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    # Panel 1: agent trajectories in the 1D environment.
    ax = axes[0, 0]
    ax.set_title("Trained agents in the environment")
    ax.axvline(cfg.target, linestyle="--", linewidth=1.5, label="target")
    ax.axvline(cfg.start, linestyle=":", linewidth=1.0, label="start")
    for algo, y_off in [("reinforce", 0.12), ("pathwise", -0.12)]:
        xs = np.asarray(trajectories[algo])
        ax.plot(xs, np.full_like(xs, y_off), "o-", ms=3, label=algo)
        ax.plot(xs[-1], y_off, "o", ms=12)
    margin = 0.3
    lo = min(cfg.start, cfg.target) - margin
    hi = max(cfg.start, cfg.target) + margin
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks([])
    ax.set_xlabel("position x")
    ax.legend(fontsize=8, loc="upper left")

    # Remaining panels: training curves for each metric.
    metric_axes = [axes[0, 1], axes[0, 2], axes[0, 3], axes[1, 0], axes[1, 1], axes[1, 2]]
    for (name, title), ax in zip(METRICS, metric_axes):
        for algo in ("reinforce", "pathwise"):
            y = histories[algo][name]
            ax.plot(np.arange(len(y)), y, label=algo)
        ax.set_title(title)
        ax.set_xlabel("update")
        if name == "grad_norm":
            ax.set_yscale("log")
            if cfg.grad_clip is not None:
                ax.axhline(cfg.grad_clip, linestyle="--", linewidth=1, color="grey", label="clip threshold")
        if name == "action_sat":
            ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)

    # Last panel: leave a short text summary of the run configuration.
    ax = axes[1, 3]
    ax.axis("off")
    summary = (
        f"action_mode = {cfg.action_mode!r}\n"
        f"log_std_mode = {cfg.log_std_mode!r}\n"
        f"grad_clip = {cfg.grad_clip}\n"
        f"batch_size = {cfg.batch_size}\n"
        f"updates = {cfg.updates}\n"
        f"lr = {cfg.learning_rate}\n"
    )
    ax.text(0.0, 0.9, "Run config", fontsize=12, fontweight="bold", va="top")
    ax.text(0.0, 0.75, summary, fontsize=10, va="top", family="monospace")

    fig.suptitle("REINFORCE vs pathwise policy gradient", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved dashboard to {save_path}")
    return fig


# -----------------------------------------------------------------------------
# 10. Main
# -----------------------------------------------------------------------------

def main(cfg: Config = CFG):
    print("Configuration:")
    print(cfg)
    print()

    print("Training REINFORCE...")
    reinforce_policy, reinforce_hist = train(cfg, "reinforce")
    print("Training pathwise PG...")
    pathwise_policy, pathwise_hist = train(cfg, "pathwise")

    histories = {"reinforce": reinforce_hist, "pathwise": pathwise_hist}
    trajectories = {
        "reinforce": np.asarray(jax.device_get(deterministic_rollout(reinforce_policy, cfg))),
        "pathwise": np.asarray(jax.device_get(deterministic_rollout(pathwise_policy, cfg))),
    }

    draw_dashboard(cfg, histories, trajectories)

    print("\nFinal deterministic positions:")
    print(f"  REINFORCE: x={trajectories['reinforce'][-1]:.4f}  (target={cfg.target})")
    print(f"  Pathwise:  x={trajectories['pathwise'][-1]:.4f}  (target={cfg.target})")


if __name__ == "__main__":
    main()