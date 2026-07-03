# %% [markdown]
# # Latent Action Bayesian Experimental Design (LA-BED)
# ### Production Pipeline: Training, Evaluation, Regularisation, and Visualisation
# 
# --------------------------------------------------------------------------------------
# ### Overview of Improvements
# 1. **Run Management:** Auto-generates timestamped folders, saves code snapshots, configs, and model checkpoints.
# 2. **Design Regularisation:** Explicitly forces the policy to explore and be informative via Coverage, Batch Entropy, and LangForce-style Contrastive LLR.
# 3. **Stable Training:** Uses Optax Cosine Decay scheduling and gradient clipping to break loss plateaus.
# 4. **Data Regimes:** Supports both `finite` (reproducible epochs) and `infinite` streaming.
# --------------------------------------------------------------------------------------

# %%
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Optional

import jax
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)

import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
except ImportError:
    pass

# Import custom modules
from models import (
    DesignPolicyTransformer, DesignPolicyMLP, BayesSimulatorMLP, BayesSimulatorTransformer, 
    SourceLocConfig, environment_step, init_belief, sort_sources_by_amplitude, expected_information_gain
)
from datalloaders import SourceLocPrior, make_train_loader, make_eval_loader

THIS_FILE = Path.cwd() / "main.py"
if "__file__" in dir():
    THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent

# %% [markdown]
# ## 1. Configuration Dataclass

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
    simulator_type: str = "mlp"   # "mlp" | "transformer"
    policy_hidden: int = 128
    sim_hidden: int = 128
    max_t: int = 30

    # --- Optimisation & Loss ---
    batch_size: int = 512
    epochs: int = 10
    lr_max: float = 5e-4
    lr_warmup_steps: int = 500
    gamma: float = 0.95           # Temporal discount
    loss_matching: str = "sort"   # "sort" (recommended) | "permute"
    
    # --- Design Regularisation ---
    # Options: "none", "coverage", "batch_entropy", "contrastive_llr"
    regularizer_kind: str = "none" 
    lambda_mse: float = 1.0
    lambda_reg: float = 0.5       # Weight for the chosen regulariser

    # --- Data Regime ---
    data_mode: str = "infinite"     # "finite" | "infinite"
    n_train_episodes: int = 100_000
    steps_per_epoch: int = 500    # Used if data_mode == "infinite"

    @property
    def belief_dim(self) -> int:
        return self.k_sources * self.design_dim

cfg = Config()

# %% [markdown]
# ## 2. Run Folder & Checkpoint Management

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

class LABEDModel(eqx.Module):
    policy: eqx.Module
    simulator: eqx.Module

def save_model(model: LABEDModel, path: Path) -> None:
    eqx.tree_serialise_leaves(str(path), model)

def load_model(path: Path, skeleton: LABEDModel) -> LABEDModel:
    return eqx.tree_deserialise_leaves(str(path), skeleton)

# %% [markdown]
# ## 3. Design Policy Regularisers

# %%
def design_regularizer(x_t: jnp.ndarray, kind: str, **kwargs) -> jnp.ndarray:
    """
    Penalties/bonuses to shape design behavior, preventing exploration collapse.
    Returns a SCALAR loss term (lower is better).
    """
    if kind == "none":
        return jnp.array(0.0)
        
    elif kind == "coverage":
        # Penalise designs being too close to each other across time (spatial repulsion).
        # x_hist is passed via kwargs. Shape: (T, D)
        x_hist = kwargs.get("x_hist")
        if x_hist is None:
            return jnp.array(0.0)
        # Calculate pairwise distances
        diffs = x_t[None, :] - x_hist # (T, D)
        sq_dists = jnp.sum(diffs**2, axis=-1)
        # Repulsion: exp(-dist^2 / tau). Ignore empty history (zeros).
        mask = jnp.any(x_hist != 0, axis=-1)
        repulsion = jnp.sum(jnp.exp(-sq_dists / 0.5) * mask)
        return repulsion  # Minimise this -> push points apart

    elif kind == "batch_entropy":
        # Proxy for entropy: Maximise spatial variance of x_t across the batch
        # x_t shape here is (B, D) if vmapped, but inside scan it's (D,).
        # We handle this at the batch level in the main loss_fn.
        pass 
        
    elif kind == "contrastive_llr":
        # LangForce-inspired Log-Likelihood Ratio (LLR) approximation.
        # We explicitly maximise the computed Expected Information Gain (EIG),
        # but penalise it if a *random* design achieves similar EIG.
        eig_t = kwargs.get("eig_t", 0.0)
        # We want to MAXIMISE eig_t, so we return -eig_t as the loss
        return -eig_t
    
    elif kind == "log_prob_max":
        ## The current design gave \theta_t (mean and std) as the result
        prob_ratio = kwargs.get("prob_ratio", 0.0)

        return -prob_ratio  # We want to maximise the log prob ratio, so return negative

    raise ValueError(f"Unknown regularizer kind: {kind}")

# %% [markdown]
# ## 4. Core Model & Rollout Logic

# %%
def build_model(cfg: Config, key: jax.random.PRNGKey) -> LABEDModel:
    p_key, s_key = jax.random.split(key)
    
    policy = DesignPolicyMLP(
        design_dim=cfg.design_dim, obs_dim=cfg.obs_dim, belief_dim=cfg.belief_dim, 
        hidden=cfg.policy_hidden, key=p_key
    )
    
    if cfg.simulator_type == "mlp":
        simulator = BayesSimulatorMLP(
            design_dim=cfg.design_dim, obs_dim=cfg.obs_dim, belief_dim=cfg.belief_dim, 
            hidden=cfg.sim_hidden, depth=3, key=s_key
        )
    else:
        simulator = BayesSimulatorTransformer(
            design_dim=cfg.design_dim, obs_dim=cfg.obs_dim, belief_dim=cfg.belief_dim, 
            hidden=64, num_heads=4, num_layers=2, key=s_key
        )
        
    return LABEDModel(policy=policy, simulator=simulator)


def rollout_episode(model: LABEDModel, theta_true: jnp.ndarray, episode_key: jax.random.PRNGKey, cfg: Config, sim_cfg: SourceLocConfig):
    initial_belief = init_belief(cfg.belief_dim, init_log_sigma=0.0)
    x_hist = jnp.zeros((cfg.max_t, cfg.design_dim))
    y_hist = jnp.zeros((cfg.max_t, cfg.obs_dim))
    
    keys = jax.random.split(episode_key, cfg.max_t)
    timesteps = jnp.arange(cfg.max_t)
    
    # Pre-sort true sources for stability
    theta_true_sorted = sort_sources_by_amplitude(theta_true, sim_cfg) if cfg.k_sources > 1 else theta_true
    theta_true_flat = theta_true_sorted.reshape(-1)

    def scan_step(carry, scan_input):
        belief, x_h, y_h = carry
        t, step_key = scan_input
        eig_key, env_key = jax.random.split(step_key)
        
        # 1. Generate Design
        x_t = model.policy(x_h, y_h, belief)
        
        # 2. Estimate EIG (LangForce Objective)
        eig_t = jnp.array(0.0)
        if cfg.belief_mode == "gaussian":
            eig_t = expected_information_gain(x_t, belief, x_h, y_h, model, sim_cfg, eig_key, num_samples=16)

        # 3. Environment Step
        y_t = environment_step(theta_true, x_t, env_key, sim_cfg)
        
        # Update History
        x_h = x_h.at[t].set(x_t)
        y_h = y_h.at[t].set(y_t)
        
        # 5. Bayes Update
        next_belief = model.simulator(x_t, y_t, belief, x_h, y_h)

        prob_ratio = jnp.array(0.0)
        if cfg.belief_mode == "gaussian":
            # Lets calculate the prob ratio of the current design vs a random design
            x_t_random = jax.random.uniform(step_key, shape=x_t.shape, minval=-model.policy.design_dim, maxval=model.policy.design_dim)
            y_t_random = environment_step(theta_true, x_t_random, env_key, sim_cfg)
            belief_random = model.simulator(x_t_random, y_t_random, belief, x_h, y_h)

            ## Next belief and belief random are both Gaussian, so we can compute their log ratios and maximise it 
            # log_prob = log (next_belief / belief_random)
            # log_prob_next = -0.5 * jnp.sum(((theta_true_flat - next_belief.mu) ** 2) / (jnp.exp(next_belief.log_sigma) ** 2)) - jnp.sum(next_belief.log_sigma)

            log_probs_next = []
            ## Let's permute the sources
            for perm in permutations(range(cfg.k_sources)):
                perm_array = jnp.array(perm)
                permuted_mu = next_belief.mu.reshape(cfg.k_sources, cfg.design_dim)[perm_array].reshape(-1)
                permuted_log_sigma = next_belief.log_sigma.reshape(cfg.k_sources, cfg.design_dim)[perm_array].reshape(-1)
                log_prob_next = -0.5 * jnp.sum(((theta_true_flat - permuted_mu) ** 2) / (jnp.exp(permuted_log_sigma) ** 2)) - jnp.sum(permuted_log_sigma)
                log_probs_next.append(log_prob_next)
            log_prob_next = jnp.max(jnp.array(log_probs_next))

            # log_prob_random = -0.5 * jnp.sum(((theta_true_flat - belief_random.mu) ** 2) / (jnp.exp(belief_random.log_sigma) ** 2)) - jnp.sum(belief_random.log_sigma)

            # prob_ratio = log_prob_next - log_prob_random
            # prob_ratio = - log_prob_random
            # prob_ratio = log_prob_next

            ## Let's maximise the KL divergence between the next belief and the previous belief.
            # kl = jnp.sum(next_belief.log_sigma - belief.log_sigma + (jnp.exp(belief.log_sigma) ** 2 + (belief.mu - next_belief.mu) ** 2) / (2 * jnp.exp(next_belief.log_sigma) ** 2) - 0.5)
            # prob_ratio = kl

        # 4. Compute Regularisation
        reg_t = design_regularizer(x_t, cfg.regularizer_kind, x_hist=x_h, eig_t=eig_t, prob_ratio=prob_ratio, rng=env_key)

        carry = (next_belief, x_h, y_h)
        return carry, (x_t, y_t, eig_t, reg_t, next_belief.mu, next_belief.log_sigma)

    carry_in = (initial_belief, x_hist, y_hist)
    final_carry, (x_ts, y_ts, eigs, regs, mus, log_sigmas) = jax.lax.scan(scan_step, carry_in, (timesteps, keys))
    
    return x_ts, y_ts, eigs, regs, mus, log_sigmas

vmap_rollout = jax.vmap(rollout_episode, in_axes=(None, 0, 0, None, None))

# %% [markdown]
# ## 5. Loss Function

# %%
def make_loss_fn(cfg: Config, sim_cfg: SourceLocConfig):
    
    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(model: LABEDModel, batch_theta: jnp.ndarray, batch_key: jax.random.PRNGKey, epoch: int):
        B = batch_theta.shape[0]
        keys = jax.random.split(batch_key, B)
        
        x_ts, y_ts, eigs, regs, mus, log_sigmas = vmap_rollout(model, batch_theta, keys, cfg, sim_cfg)
        
        # Reshape to (B, T, K, D)
        mus_reshaped = jnp.reshape(mus, (B, cfg.max_t, cfg.k_sources, cfg.design_dim))
        theta_reshaped = jnp.reshape(batch_theta, (B, cfg.k_sources, cfg.design_dim))

        # --- 1. Downstream MSE Loss ---
        if cfg.loss_matching == "permute":
            # Combinatorial matching (Harder, prone to plateaus but strictly invariant)
            perm_indices = jnp.array(list(permutations(range(cfg.k_sources))))
            def mse_for_perm(perm):
                mus_perm = mus_reshaped[:, -1, perm, :]
                return jnp.mean((mus_perm - theta_reshaped)**2, axis=(1, 2))
            
            all_mses = jax.vmap(mse_for_perm)(perm_indices) # (K!, B)
            min_mse_per_batch = jnp.min(all_mses, axis=0)   # (B,)
            total_mse = jnp.mean(min_mse_per_batch)
            
        else:
            # Sorted matching (Smoother gradients, prevents collapse)
            sorted_theta = jax.vmap(sort_sources_by_amplitude, in_axes=(0, None))(theta_reshaped, sim_cfg)
            total_mse = jnp.mean((mus_reshaped[:, -1, :, :] - sorted_theta)**2)

        # Log transform for numerical stability
        loss_mse = jnp.log(total_mse + 1e-8)

        # --- 2. Regularisation Loss ---
        if cfg.regularizer_kind == "batch_entropy":
            # Maximize variance of actions across the batch dimension
            batch_var = jnp.mean(jnp.var(x_ts, axis=0))
            loss_reg = -batch_var
        else:
            loss_reg = jnp.mean(jnp.sum(regs, axis=1))

        # Total Composite Loss
        loss = (cfg.lambda_mse * loss_mse) + (cfg.lambda_reg * loss_reg)
        
        return loss, {"mse": total_mse, "reg": loss_reg, "eig_sum": jnp.mean(jnp.sum(eigs, axis=1))}
        
    return loss_fn

# %% [markdown]
# ## 6. Training Loop

# %%
def train(cfg: Config, model: LABEDModel, key: jax.random.PRNGKey, run_dir: Path):
    sim_cfg = SourceLocConfig(K=cfg.k_sources)
    prior = SourceLocPrior(K=cfg.k_sources, prior_std=1.0)
    
    train_loader = make_train_loader(
        prior, cfg.batch_size, base_seed=cfg.seed, 
        data_mode=cfg.data_mode, n_train_episodes=cfg.n_train_episodes
    )

    # Cosine Decay Learning Rate with Warmup (Crucial for breaking plateaus)
    total_steps = (len(train_loader) if cfg.data_mode == "finite" else cfg.steps_per_epoch) * cfg.epochs
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=cfg.lr_max,
        warmup_steps=cfg.lr_warmup_steps,
        decay_steps=total_steps,
        end_value=1e-6
    )
    
    # Add gradient clipping to stabilize transformer/RNN scanning
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    loss_fn = make_loss_fn(cfg, sim_cfg)

    @eqx.filter_jit
    def train_step(model, opt_state, batch_theta, batch_key, epoch):
        (loss_val, aux), grads = loss_fn(model, batch_theta, batch_key, epoch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, aux

    history = {"epoch_losses": [], "step_losses": [], "step_mse": []}
    infinite_iter = iter(train_loader) if cfg.data_mode == "infinite" else None

    print("🚀 Starting Training...")
    for epoch in range(cfg.epochs):
        n_batches = len(train_loader) if cfg.data_mode == "finite" else cfg.steps_per_epoch
        batch_source = train_loader if cfg.data_mode == "finite" else (next(infinite_iter) for _ in range(n_batches))
        
        running_loss, n_steps = 0.0, 0
        pbar = tqdm(batch_source, total=n_batches, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        
        for theta_batch_np in pbar:
            key, batch_key = jax.random.split(key)
            batch_theta = jnp.asarray(theta_batch_np)
            
            model, opt_state, loss_val, aux = train_step(model, opt_state, batch_theta, batch_key, epoch)
            
            loss_f = float(loss_val)
            running_loss += loss_f
            n_steps += 1
            
            history["step_losses"].append(loss_f)
            history["step_mse"].append(float(aux["mse"]))
            pbar.set_postfix(loss=f"{loss_f:.4f}", mse=f"{float(aux['mse']):.4f}")

        avg_loss = running_loss / max(n_steps, 1)
        history["epoch_losses"].append(avg_loss)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # Checkpoint every epoch
        save_model(model, run_dir / "checkpoint.eqx")

    with open(run_dir / "train_history.json", "w") as f:
        json.dump(history, f)
        
    return model, history, key

# %% [markdown]
# ## 7. Execution & Visualisation 

# %%
key = jax.random.PRNGKey(cfg.seed)
run_dir = make_run_dir()
snapshot_code(run_dir)
save_config(cfg, run_dir)
print(f"📁 Run directory created: {run_dir}")

# Build Model & Train
skeleton = build_model(cfg, key)
model, history, key = train(cfg, skeleton, key, run_dir)


#%%
# --- Plot Training Curves ---
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(range(1, cfg.epochs + 1), history["epoch_losses"], marker='o', color='purple')
ax[0].set_title("Composite Loss Over Epochs")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].grid(True, alpha=0.3)

ax[1].plot(range(1, len(history["step_mse"]) + 1), history["step_mse"], color='orange', lw=0.8)
ax[1].set_title("Terminal MSE Loss Over Steps (Un-logged)")
ax[1].set_xlabel("Step"); ax[1].set_ylabel("MSE"); ax[1].set_yscale("log"); ax[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
fig.savefig(run_dir / "plots" / "training_curves.png", dpi=150)
plt.close(fig)

#%%
# --- Visualise Evaluation Trajectory ---
sim_cfg = SourceLocConfig(K=cfg.k_sources)
prior = SourceLocPrior(K=cfg.k_sources, prior_std=1.0)
eval_loader = make_eval_loader(prior, n_episodes=128, batch_size=128, seed=cfg.seed + 1)

eval_theta = jnp.asarray(next(iter(eval_loader)))
key, eval_key = jax.random.split(key)
eval_keys = jax.random.split(eval_key, eval_theta.shape[0])

x_ts, y_ts, eigs, regs, mus, log_sigmas = vmap_rollout(model, eval_theta, eval_keys, cfg, sim_cfg)

idx = np.random.randint(0, eval_theta.shape[0])
true_src = eval_theta[idx].reshape(-1, 2)
xs = x_ts[idx]
pred_path = mus[idx].reshape(cfg.max_t, cfg.k_sources, 2)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(true_src[:, 0], true_src[:, 1], c='red', marker='*', s=350, label='Source(s)', zorder=5)
ax.plot(xs[:, 0], xs[:, 1], 'bo-', alpha=0.8, markersize=5, label='Designs $x_t$')
for t in range(cfg.max_t):
    plt.text(xs[t, 0] + 0.05, xs[t, 1] + 0.05, f"$t_{{{t+1}}}$", fontsize=9, color='darkblue')


# Dynamic sizing and alpha for belief timeline
for k in range(cfg.k_sources):
    for t in range(cfg.max_t):
        alpha = 0.15 + 0.85 * ((t + 1) / cfg.max_t)
        size = 2 + 160 * ((t + 1) / cfg.max_t)
        label = rf"Belief (Src {k+1})" if t == cfg.max_t - 1 else None
        ax.scatter(pred_path[t, k, 0], pred_path[t, k, 1], c='green', alpha=alpha, s=size, marker='X', label=label)

ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
ax.axvline(0, color='gray', linestyle='--', alpha=0.4)
ax.set_title(f"LA-BED Trajectory (Seq={idx}, T={cfg.max_t})")

## Draw a horizontal legend instead of vertical
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=10)

ax.grid(True, alpha=0.3)
plt.show()
fig.savefig(run_dir / "plots" / f"trajectory_eval_{idx}.png", dpi=150)
plt.close(fig)
