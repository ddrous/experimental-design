# %% [markdown]
# # Latent Action Bayesian Experimental Design (LA-BED)
# Training, Evaluation and Visualisation Pipeline
# This file is structured in interactive blocks. Run cells sequentially.

# %%
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})

# Import our custom modules
from models import (
    DesignPolicy, BayesSimulatorMLP, BayesSimulatorTransformer, 
    SourceLocConfig, environment_step, init_belief, belief_entropy, sort_sources_by_amplitude, expected_information_gain
)
from datalloaders import SourceLocPrior, make_train_loader, make_eval_loader

# %% [markdown]
# ## 1. Configuration & Hyperparameters

# %%
# Experiment Config
# SEED = 2026
SEED = time.time_ns() % (2**32 - 1)  # Use current time as seed for variability
MAX_T = 25                  # Trajectory length (number of queries per episode)
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCH = 150        # Virtual epoch size for infinite streaming
LR = 1e-4

# Architecture Config
DESIGN_DIM = 2
OBS_DIM = 1
K_SOURCES = 1               # "Simplest version" with 1 source
BELIEF_DIM = K_SOURCES * 2  # Flattened coordinates of sources
BELIEF_MODE = "gaussian"    # "point" or "gaussian"
SIMULATOR_TYPE = "mlp" # "mlp" or "transformer"

# Loss Config
GAMMA = 0.95                # Discount factor for intermediate losses
LAMBDA_MSE = 1.0            # Weight of downstream NLL/MSE loss
LAMBDA_EIG = 1.0            # Weight of EIG loss

cfg = SourceLocConfig(K=K_SOURCES)
key = jax.random.PRNGKey(SEED)

# %% [markdown]
# ## 2. Model Initialisation

# %%
key, p_key, s_key = jax.random.split(key, 3)

# 1. Design Policy (Inverse Model)
policy = DesignPolicy(
    design_dim=DESIGN_DIM, obs_dim=OBS_DIM, belief_dim=BELIEF_DIM, 
    hidden=64, num_heads=4, num_layers=2, key=p_key
)

# 2. Bayes Simulator (Forward Model)
if SIMULATOR_TYPE == "mlp":
    simulator = BayesSimulatorMLP(
        design_dim=DESIGN_DIM, obs_dim=OBS_DIM, belief_dim=BELIEF_DIM, 
        hidden=64, depth=3, key=s_key
    )
else:
    simulator = BayesSimulatorTransformer(
        design_dim=DESIGN_DIM, obs_dim=OBS_DIM, belief_dim=BELIEF_DIM, 
        hidden=64, num_heads=4, num_layers=2, key=s_key
    )

# Combine for Equinox parameter updates
class LABEDModel(eqx.Module):
    policy: DesignPolicy
    simulator: eqx.Module

model = LABEDModel(policy=policy, simulator=simulator)

# Initialise Dataloaders
prior = SourceLocPrior(K=K_SOURCES, prior_std=1.0)
train_loader = make_train_loader(prior, BATCH_SIZE, base_seed=SEED)
eval_loader = make_eval_loader(prior, n_episodes=256, batch_size=BATCH_SIZE, seed=SEED+1)
train_iter = iter(train_loader)

# %% [markdown]
# ## 3. Rollout Logic with `jax.lax.scan`

# %%
def rollout_episode(model: LABEDModel, theta_true: jnp.ndarray, episode_key: jax.random.PRNGKey):
    """
    Unrolls a full BOED trajectory for a single episode (batch size 1).
    Leverages `jax.lax.scan` keeping static shapes by preallocating history.
    """
    # Initialize state
    initial_belief = init_belief(BELIEF_DIM, init_log_sigma=0.0)
    x_hist = jnp.zeros((MAX_T, DESIGN_DIM))
    y_hist = jnp.zeros((MAX_T, OBS_DIM))
    
    # Pre-split keys for the environment steps
    keys = jax.random.split(episode_key, MAX_T)
    timesteps = jnp.arange(MAX_T)
    
    # Sort true sources by amplitude to avoid permutation ambiguity (ActionBED strategy)
    if K_SOURCES > 1:
        theta_true_sorted = sort_sources_by_amplitude(theta_true, cfg)
    else:
        theta_true_sorted = theta_true
    theta_true_flat = theta_true_sorted.reshape(-1)

    # def scan_step(carry, scan_input):
    #     belief, x_h, y_h = carry
    #     t, step_key = scan_input
        
    #     # 1. Policy generates design
    #     x_t = model.policy(x_h, y_h, belief)
        
    #     # 2. Environment observes outcome
    #     y_t = environment_step(theta_true, x_t, step_key, cfg)
        
    #     # Update history inline
    #     x_h = x_h.at[t].set(x_t)
    #     y_h = y_h.at[t].set(y_t)
        
    #     # 3. Bayes Simulator updates belief
    #     next_belief = model.simulator(x_t, y_t, belief, x_h, y_h)
        
    #     # 4. Compute losses
    #     if BELIEF_MODE == "gaussian":
    #         eig_t = belief_entropy(belief) - belief_entropy(next_belief)
    #     else:
    #         eig_t = jnp.array(0.0)
            
    #     mse_t = jnp.mean((next_belief.mu - theta_true_flat)**2)
        
    #     carry = (next_belief, x_h, y_h)
    #     return carry, (x_t, y_t, eig_t, mse_t, next_belief.mu, next_belief.log_sigma)

    def scan_step(carry, scan_input):
        belief, x_h, y_h = carry
        t, step_key = scan_input
        
        # Dedicate separate keys for the EIG estimation and the actual environment step
        eig_key, env_key = jax.random.split(step_key)
        
        # 1. Policy generates design x_t based on history and current belief
        x_t = model.policy(x_h, y_h, belief)
        
        # 2. Calculate the Expected Information Gain strictly as a function of x_t
        # This provides the necessary gradients to train the policy.
        if BELIEF_MODE == "gaussian":
            eig_t = expected_information_gain(
                x_t, belief, x_h, y_h, model, cfg, eig_key, num_samples=16
            )
        else:
            eig_t = jnp.array(0.0)
            
        # 3. Environment observes the ACTUAL outcome using the true parameter
        y_t = environment_step(theta_true, x_t, env_key, cfg)
        
        # Update history inline
        x_h = x_h.at[t].set(x_t)
        y_h = y_h.at[t].set(y_t)
        
        # 4. Bayes Simulator updates the actual belief state based on the ACTUAL outcome
        next_belief = model.simulator(x_t, y_t, belief, x_h, y_h)
        
        # 5. Compute downstream task loss (MSE)
        mse_t = jnp.mean((next_belief.mu - theta_true_flat)**2)
        
        carry = (next_belief, x_h, y_h)
        return carry, (x_t, y_t, eig_t, mse_t, next_belief.mu, next_belief.log_sigma)

    carry_in = (initial_belief, x_hist, y_hist)
    final_carry, (x_ts, y_ts, eigs, mses, mus, log_sigma) = jax.lax.scan(scan_step, carry_in, (timesteps, keys))
    
    return x_ts, y_ts, eigs, mses, mus, log_sigma

# Vmap across the batch dimension
vmap_rollout = jax.vmap(rollout_episode, in_axes=(None, 0, 0))

# %% [markdown]
# ## 4. Loss Function and Optimizer

# %%
@eqx.filter_value_and_grad
def loss_fn(model: LABEDModel, batch_theta: jnp.ndarray, batch_key: jax.random.PRNGKey, epoch: int):
    B = batch_theta.shape[0]
    keys = jax.random.split(batch_key, B)
    
    x_ts, y_ts, eigs, mses, mus, log_sigma = vmap_rollout(model, batch_theta, keys)
    
    # Discounts: gamma^0, gamma^1, ..., gamma^{T-1}
    # discounts = GAMMA ** jnp.arange(MAX_T)
    discounts = GAMMA ** jnp.arange(MAX_T-1, -1, -1)  # Reverse discounting to emphasize later steps
    discounts = discounts.reshape(1, MAX_T) # broadcast over batch
    
    # Compute Expected Information Gain over time
    total_eig = jnp.mean(jnp.sum(eigs * discounts, axis=1))
    # total_eig = jnp.mean(jnp.sum(eigs * 1, axis=1))
    
    # Compute downstream task error over time
    # total_mse = jnp.mean(jnp.sum(mses * discounts, axis=1))
    # total_mse = jnp.mean(mses[:, -1])
    
    # Composite loss
    # loss = - (LAMBDA_EIG * total_eig) + (LAMBDA_MSE * total_mse)

    # print("SHape of mus:", mus.shape)       ## Shape of mus: (B, T, K*d)
    # ## We wand to order these mus by these MSE norms

    # ## Reshape back to (B, T, K*d)
    # mus_ordered_flat = mus_ordered.reshape(B, MAX_T, K_SOURCES * DESIGN_DIM)
    # print("Shape of mus_ordered_flat:", mus_ordered_flat.shape)

    # ## Compute MSE with respect to the true theta (which should also be reshaped and reordered)
    # theta_true_reshaped = batch_theta.reshape(B, K_SOURCES, DESIGN_DIM)
    # theta_true_ordered = jnp.take_along_axis(theta_true_reshaped, ordering_indices[..., None], axis=1)  # Shape: (B, K, d)
    # theta_true_ordered_flat = theta_true_ordered.reshape(B, K_SOURCES * DESIGN_DIM)
    # print("Shape of theta_true_ordered_flat:", theta_true_ordered_flat.shape)


    mus = jnp.reshape(mus, (B, MAX_T, K_SOURCES, DESIGN_DIM))  # Shape: (B, T, K, d)
    # total_mse = jnp.mean((mus[:, -1, :] - batch_theta) ** 2)
    total_mse = jnp.mean(jnp.sum(mses * discounts, axis=1))

    # total_mse = jnp.log(total_mse + 1e-8)  # Log-transform to stabilize training

    # loss = total_mse


    # # Let's use log-sigma and compute the negative log-likelihood loss for the final belief
    # batch_theta = batch_theta.reshape(B, K_SOURCES * DESIGN_DIM)  # Shape: (B, K*d)
    # final_mu = mus[:, -1, :]          # Shape: (B, K*d)
    # final_log_sigma = log_sigma[:, -1, :]  # Shape: (B, K*d)
    # final_sigma = jnp.exp(final_log_sigma)  # Shape: (B, K*d)
    # nll_loss = 0.5 * jnp.mean(((final_mu - batch_theta) ** 2) / (final_sigma ** 2) + 2 * final_log_sigma + jnp.log(2 * jnp.pi))
    # # loss = nll_loss

    # ## Let's use the NLL loss, but consider discounted NLL over time (final belief is more important, but we can also consider intermediate beliefs)
    # batch_theta = batch_theta.reshape(B, K_SOURCES * DESIGN_DIM)  # Shape: (B, K*d)
    # # print("Shape of batch_theta:", batch_theta.shape, mus.shape, log_sigma.shape, discounts.shape)  # Shape of batch_theta: (B, K*d) (B, T, K*d) (B, T, K*d) (1, T)
    # nll_losses = 0.5 * (((mus - batch_theta[:, None, :]) ** 2) / (jnp.exp(log_sigma) ** 2) + 2 * log_sigma + jnp.log(2 * jnp.pi))  # Shape: (B, T, K*d)
    # discounted_nll_losses = nll_losses * discounts[:, :, None]  # Shape: (B, T, K*d)
    # total_nll_loss = jnp.mean(jnp.sum(discounted_nll_losses, axis=(1, 2)))  # Average over batch and sum over time and sources
    # loss = total_nll_loss

    # loss = LAMBDA_MSE * nll_loss - LAMBDA_EIG * total_eig

    # loss = LAMBDA_MSE*total_mse - LAMBDA_EIG*total_eig
    # loss = - LAMBDA_EIG*total_eig

    ## Start optimising the EIG + MSE ofter eopoch 5, and just the MSE before that
    # return_eig = jnp.where(epoch >= 5, total_eig, 0.0)
    return_eig = total_eig

    # loss = LAMBDA_MSE*total_mse - LAMBDA_EIG*return_eig
    loss = LAMBDA_MSE*total_mse

    return loss

optimizer = optax.adam(LR)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(model, opt_state, batch_theta, batch_key, epoch):
    loss_val, grads = loss_fn(model, batch_theta, batch_key, epoch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

# %% [markdown]
# ## 5. Training Loop

# %%
train_losses = []
all_losses = []

print("Starting Training...")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for _ in tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        key, batch_key = jax.random.split(key)

        # Get next batch of true thetas (converted to JAX arrays)
        batch_theta_np = next(train_iter)
        batch_theta = jnp.asarray(batch_theta_np)

        model, opt_state, loss_val = train_step(model, opt_state, batch_theta, batch_key, epoch)
        epoch_loss += loss_val.item()
        all_losses.append(loss_val.item())

    avg_loss = epoch_loss / STEPS_PER_EPOCH
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# %% [markdown]
# ## 6. Evaluation and Visualisation

# %%
### PLot training curves in both eopchs and steps
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# Epoch-wise training loss
ax[0].plot(range(1, EPOCHS + 1), train_losses, marker='o', color='purple')
ax[0].set_title("Training Loss Over Epochs")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Terminal MSE Loss")
ax[0].set_yscale("symlog")
ax[0].grid(True, alpha=0.3)

# Step-wise training loss
ax[1].plot(range(1, len(all_losses) + 1), all_losses, marker='.', color='orange')
ax[1].set_title("Training Loss Over Steps")
ax[1].set_xlabel("Training Step")
ax[1].set_ylabel("Terminal MSE Loss")
ax[1].set_yscale("symlog")
ax[1].grid(True, alpha=0.3)


# %%
# Visualise a Single Evaluation Trajectory
eval_theta_np = next(iter(eval_loader))
eval_theta = jnp.asarray(eval_theta_np)

key, eval_key = jax.random.split(key)
eval_keys = jax.random.split(eval_key, eval_theta.shape[0])

# Rollout evaluation batch
x_ts, y_ts, eigs, mses, mus, log_sigmas = vmap_rollout(model, eval_theta, eval_keys)

# Pick the first episode in the batch to visualise
idx = np.random.randint(0, eval_theta.shape[0])
true_src = eval_theta[idx] # Shape (K, 2)
xs = x_ts[idx]             # Shape (T, 2)
predicted_mus = mus[idx]   # Shape (T, K*2)

plt.figure(figsize=(10, 6))

# Plot True Source
plt.scatter(true_src[:, 0], true_src[:, 1], c='red', marker='*', s=300, label='True Source', zorder=5)

# Plot Designs (Sensor Placements) over time
plt.plot(xs[:, 0], xs[:, 1], 'bo-', alpha=1.0, markersize=7, label='Designs $x_t$')
for t in range(MAX_T):
    plt.text(xs[t, 0] + 0.05, xs[t, 1] + 0.05, f"$t_{{{t+1}}}$", fontsize=9, color='darkblue')

# Plot Belief evolution
pred_path = predicted_mus.reshape(MAX_T, K_SOURCES, 2)
# plt.plot(pred_path[:, 0, 0], pred_path[:, 0, 1], 'gs--', alpha=0.95, markersize=6, label=r"Belief $\hat{\theta}_t$")
# for t in range(MAX_T):
#     plt.text(pred_path[t, 0, 0] - 0.1, pred_path[t, 0, 1] - 0.1, f"$t_{{{t+1}}}$", fontsize=9, color='green', alpha=0.5)

## Plot the beleif evolution, all sources in green. The earlier beliefs are plotted with lighter colors, and the later are plotted with darker colors. As T grows, the alpha grows, and the size of the maker increases as well.
for k in range(K_SOURCES):
    for t in range(MAX_T):
        alpha = 0.1 + 0.9 * ((t+1) / MAX_T)  # Gradually increase alpha
        size = 4 + 100 * ((t+1) / MAX_T)       # Gradually increase marker size

        plt.scatter(pred_path[t, k, 0], pred_path[t, k, 1], c='green', alpha=alpha, s=size, marker='X', label=None if t < MAX_T-1 else rf"Belief $\hat{{\theta}}_t$ (Source {k+1})")

# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
# plt.title(f"BOED Trajectory (T={MAX_T})")
plt.title(f"BOED Trajectory (Seq={idx}, T={MAX_T})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Visualise EIG per step
plt.figure(figsize=(8, 4))
plt.bar(range(1, MAX_T + 1), eigs[idx], color='teal', alpha=0.7)
plt.title("Expected Information Gain (EIG) per Step")
plt.xlabel("Time Step $t$")
plt.ylabel("Entropy Reduction")
plt.xticks(range(1, MAX_T + 1, 5))
plt.grid(True, alpha=0.3)
plt.show()
