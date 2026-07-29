# %% [markdown]
# # WeightTransformer: learning to write the weights of a density estimator
#
# We train a Transformer that maps a randomly-initialised tiny SIREN
# (`theta_0`) all the way to `theta_T`, conditioned on a sequence of samples
# `x_1 .. x_T` drawn from a target distribution. `theta_T` is decoded into a
# density `p_{theta_T}` that should match where the samples came from.
#
# Two architectures are implemented in `models.py` (see the figures below):
#   - `adaln`:     causal Transformer over the *weight trajectory*, AdaLN-conditioned on x_t.
#   - `sampleseq`: causal Transformer whose tokens *are* the samples themselves; no AdaLN.
#
# Both are trained with the same objective, maximum likelihood over the full
# conditioning sequence: L(phi) = sum_t -log p_{theta_t}(x_t).

# %% Imports and setup
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})

jax.print_environment_info()

from loaders import get_dataloaders
from models import WeightTransformerDensityModel, batch_nll, sequence_nll

# %% [markdown]
# ## Architecture reference figures
# Import these directly from a notebook to explain the two variants:
# ```python
# from PIL import Image
# Image.open("assets/adaln_weighttransformer.png")
# Image.open("assets/sampleseq_weighttransformer.png")
# ```

ASSETS_DIR = ROOT / "assets"
ADALN_FIGURE = ASSETS_DIR / "adaln_weighttransformer.png"
SAMPLESEQ_FIGURE = ASSETS_DIR / "sampleseq_weighttransformer.png"

# %% Config -- all model/training settings are flat variables, set here early.
SEED = 2030

# --- data / task -----------------------------------------------------------
TASK_NAME = "ring"          # "gaussian" | "ring" -- see loaders.TASK_REGISTRY
# TASK_KWARGS = dict(dim=2, mean_range=2.0, log_std_low=-1.5, log_std_high=0.4)
TASK_KWARGS = dict(dim=2)
SAMPLE_DIM = TASK_KWARGS["dim"]
SEQ_LEN = 32                    # T: number of conditioning samples per task
PERMUTE_INPUTS = True           # shuffle sample order per task/epoch/batch

BATCH_SIZE = 64
BATCHES_PER_EPOCH = 50
VAL_BATCHES_PER_EPOCH = 5

# --- model -------------------------------------------------------------
VARIANT = "adaln"               # "adaln" | "sampleseq"  -- models.VARIANTS
INR_WIDTH = 32
INR_DEPTH = 4
INR_BASE_OMEGA = 10.0

PREDICTOR_HIDDEN_DIM = 128
PREDICTOR_DEPTH = 4
PREDICTOR_HEADS = 4
PREDICTOR_MLP_DIM = 256
DROPOUT_RATE = 0.1

QUAD_DOMAIN = 4.0                # density is normalised over [-QUAD_DOMAIN, QUAD_DOMAIN]^2
QUAD_GRID_SIZE = 48

# --- optimisation --------------------------------------------------------
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60
GRAD_CLIP_NORM = 1.0
VIS_EVERY_FRAC = 0.1             # visualise every 1/10th of total epochs

CONFIG = dict(
    task_name=TASK_NAME,
    task_kwargs=TASK_KWARGS,
    sample_dim=SAMPLE_DIM,
    seq_len=SEQ_LEN,
    permute_inputs=PERMUTE_INPUTS,
    batch_size=BATCH_SIZE,
    batches_per_epoch=BATCHES_PER_EPOCH,
    val_batches_per_epoch=VAL_BATCHES_PER_EPOCH,
    seed=SEED,
    variant=VARIANT,
    inr_width=INR_WIDTH,
    inr_depth=INR_DEPTH,
    inr_base_omega=INR_BASE_OMEGA,
    predictor_hidden_dim=PREDICTOR_HIDDEN_DIM,
    predictor_depth=PREDICTOR_DEPTH,
    predictor_heads=PREDICTOR_HEADS,
    predictor_mlp_dim=PREDICTOR_MLP_DIM,
    dropout_rate=DROPOUT_RATE,
    quad_domain=QUAD_DOMAIN,
    quad_grid_size=QUAD_GRID_SIZE,
)

RUN_DIR = ROOT / "runs" / f"{VARIANT}_{TASK_NAME}_{int(time.time())}"
(RUN_DIR / "plots").mkdir(parents=True, exist_ok=True)
print(f"Run directory: {RUN_DIR}")

key = jax.random.PRNGKey(SEED)

# %% Data
train_loader, val_loader, task_dist = get_dataloaders(CONFIG)
sample_x, sample_params = next(iter(train_loader))
print(f"x_seq batch: {sample_x.shape}  task params: {sample_params.shape}")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for i, ax in enumerate(axes):
    pts = np.asarray(sample_x[i])
    ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.7)
    order = np.arange(len(pts))
    ax.plot(pts[order[:5], 0], pts[order[:5], 1], "r-", lw=0.5, alpha=0.5)
    ax.set_title(f"task {i}")
    ax.set_xlim(-QUAD_DOMAIN, QUAD_DOMAIN)
    ax.set_ylim(-QUAD_DOMAIN, QUAD_DOMAIN)
    ax.set_aspect("equal")
fig.suptitle("Example conditioning sample sequences (first 5 points connected)")
fig.tight_layout()
fig.savefig(RUN_DIR / "plots" / "sample_sequences.png", dpi=120)
plt.show()
plt.close(fig)

# %% Model and optimizer
key, init_key = jax.random.split(key)
model = WeightTransformerDensityModel(CONFIG, key=init_key)

num_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
print(f"Variant: {VARIANT}")
print(f"theta dim (INR param count): {model.theta0.shape[0]}")
print(f"Total trainable params: {num_params / 1e3:.1f}K")

optimizer = optax.chain(
    optax.clip_by_global_norm(GRAD_CLIP_NORM),
    optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

# %% Train / eval steps


@eqx.filter_jit
def train_step(model, opt_state, x_batch, key):
    loss_fn = lambda m: batch_nll(m, x_batch, key, inference=False)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model, x_batch, key):
    return batch_nll(model, x_batch, key, inference=True)


# %% [markdown]
# ## Visualisation helper
# Renders the ground-truth samples/density next to the model's predicted
# density at several points along the trajectory (t = T/4, T/2, T).


def plot_density_progress(model, x_seq, task_params, task_dist, epoch, save_path=None):
    key_vis = jax.random.PRNGKey(12345)
    log_probs, theta_traj = model.sequence_log_probs(x_seq, key=key_vis, inference=True)
    seq_len = x_seq.shape[0]
    checkpoints = sorted(set([max(1, seq_len // 4), max(1, seq_len // 2), seq_len]))

    grid_points = np.asarray(model.quad_points)
    grid_size = int(round(np.sqrt(grid_points.shape[0])))
    gx = grid_points[:, 0].reshape(grid_size, grid_size)
    gy = grid_points[:, 1].reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, len(checkpoints) + 1, figsize=(4 * (len(checkpoints) + 1), 4))

    axes[0].scatter(np.asarray(x_seq)[:, 0], np.asarray(x_seq)[:, 1], s=12, c="k", alpha=0.6)
    if hasattr(task_dist, "log_prob"):
        true_density = np.exp(task_dist.log_prob(np.asarray(task_params), grid_points)).reshape(grid_size, grid_size)
        axes[0].contourf(gx, gy, true_density, levels=20, cmap="viridis", alpha=0.6)
    axes[0].set_title("ground truth samples")
    axes[0].set_xlim(gx.min(), gx.max())
    axes[0].set_ylim(gy.min(), gy.max())
    axes[0].set_aspect("equal")

    for i, t in enumerate(checkpoints):
        density = np.asarray(model.density_grid(theta_traj[t - 1])).reshape(grid_size, grid_size)
        ax = axes[i + 1]
        ax.contourf(gx, gy, density, levels=20, cmap="viridis")
        ax.scatter(np.asarray(x_seq)[:t, 0], np.asarray(x_seq)[:t, 1], s=10, c="white", edgecolor="k", linewidth=0.3)
        ax.set_title(f"$p_{{\\theta_{{{t}}}}}$  (t={t}/{seq_len})")
        ax.set_aspect("equal")

    fig.suptitle(f"epoch {epoch}  |  mean seq NLL={-np.mean(np.asarray(log_probs)):.3f}")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    plt.show()
    plt.close(fig)


# %% Training loop -- collects loss every step AND every epoch for plotting.
step_losses = []          # (global_step, loss) every train step
epoch_train_losses = []   # (epoch, mean_loss) every epoch
epoch_val_losses = []     # (epoch, mean_loss) every epoch

vis_every = max(1, int(round(NUM_EPOCHS * VIS_EVERY_FRAC)))
global_step = 0

# Fixed batch used purely for visualisation, so progress is comparable across epochs.
vis_x_batch, vis_params_batch = next(iter(val_loader))

for epoch in range(1, NUM_EPOCHS + 1):
    train_loader.set_epoch(epoch, seed=SEED)
    epoch_losses = []
    t0 = time.time()

    for x_batch, _params_batch in train_loader:
        key, step_key = jax.random.split(key)
        model, opt_state, loss = train_step(model, opt_state, x_batch, step_key)
        loss_value = float(loss)
        step_losses.append((global_step, loss_value))
        epoch_losses.append(loss_value)
        global_step += 1

    train_loader_mean = float(np.mean(epoch_losses))
    epoch_train_losses.append((epoch, train_loader_mean))

    val_loader.set_epoch(epoch, seed=SEED + 1)
    val_losses = []
    for x_batch, _params_batch in val_loader:
        key, val_key = jax.random.split(key)
        val_losses.append(float(eval_step(model, x_batch, val_key)))
    val_mean = float(np.mean(val_losses))
    epoch_val_losses.append((epoch, val_mean))

    dt = time.time() - t0
    print(f"epoch {epoch:3d}/{NUM_EPOCHS} | train NLL {train_loader_mean:8.4f} | val NLL {val_mean:8.4f} | {dt:5.1f}s")

    if epoch % vis_every == 0 or epoch == NUM_EPOCHS:
        plot_density_progress(
            model,
            vis_x_batch[0],
            vis_params_batch[0],
            task_dist,
            epoch,
            save_path=RUN_DIR / "plots" / f"density_epoch_{epoch:04d}.png",
        )

# %% Loss curves (per-step and per-epoch)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

steps, losses = zip(*step_losses)
axes[0].plot(steps, losses, lw=0.6, alpha=0.8)
axes[0].set_xlabel("train step")
axes[0].set_ylabel("NLL")
axes[0].set_title("per-step training loss")

epochs_t, train_means = zip(*epoch_train_losses)
_, val_means = zip(*epoch_val_losses)
axes[1].plot(epochs_t, train_means, label="train")
axes[1].plot(epochs_t, val_means, label="val")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("mean NLL")
axes[1].set_title("per-epoch loss")
axes[1].legend()

fig.tight_layout()
fig.savefig(RUN_DIR / "plots" / "loss_curves.png", dpi=120)
plt.show()
plt.close(fig)

# %% Save final model
checkpoint_path = RUN_DIR / "model.eqx"
eqx.tree_serialise_leaves(checkpoint_path, model)
print(f"Saved model to {checkpoint_path}")