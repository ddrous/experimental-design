#%% 1) Imports, policy configuration, and frozen posterior-model loading
"""Train a Transformer policy for sequential source-location experimental design.

The frozen posterior updater learned by ``adaptive_posterior_particle_transformer.py``
implements

    q_phi(theta | B_t, x_t, y_t),

where B_t is any unweighted sample from the current belief. The policy implements

    x_t = pi_psi(B_t, t/T).

A differentiable simulation rollout is used during policy training:

    theta_true ~ p(theta)
    B_0 ~ p(theta)
    x_t = pi_psi(B_t)
    y_t = log s(theta_true, x_t) + sigma_y epsilon_t
    q_t = q_phi(theta | B_t, x_t, y_t)
    B_{t+1} ~ q_t.

The principal objective is the final conditional NLL of theta_true under q_T. This
is a simulation-based design objective because theta_true is available while training.
At deployment theta_true is never required: the policy sees only its current belief.

Important: with non-zero observation noise and finite T, a calibrated posterior should
usually become concentrated but not literally collapse to a Dirac delta. Forcing zero
variance would create overconfidence. The NLL is a proper scoring objective that rewards
concentration only when the true theta remains probable.
"""
from __future__ import annotations

from dataclasses import asdict
from itertools import islice
from pathlib import Path
import math
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm
import yaml

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

import bayes_simulator_common as bsc
from bayes_simulator_common import (
    DesignPolicyTransformer,
    PolicyConfig,
    PosteriorConfig,
    SourceLocPrior,
    build_source_permutations,
    dataclass_from_dict,
    find_latest_run,
    load_posterior_model,
    make_condition_jax,
    make_run_dir,
    make_theta_eval_loader,
    make_theta_train_loader,
    permutation_marginal_nll,
    relaxed_sample_gmm_single,
    sample_gmm_single,
    save_config_yaml,
    save_json,
    save_policy_model,
    snapshot_files,
    source_log_signal_jax,
)


POLICY_CFG = PolicyConfig(
    inference_run_dir=None,  # set an explicit run path here, or use newest posterior run
    horizon=6,
    num_belief_particles=64,
    epochs=30,
    n_train_episodes=20_000,
    n_eval_episodes=256,
    batch_size=64,
)

np.random.seed(POLICY_CFG.seed)
jax_key = jax.random.key(POLICY_CFG.seed)
print("JAX devices:", jax.devices())

if POLICY_CFG.inference_run_dir is None:
    INFERENCE_RUN_DIR = find_latest_run(
        POLICY_CFG.runs_base, POLICY_CFG.inference_env_name
    )
else:
    INFERENCE_RUN_DIR = Path(POLICY_CFG.inference_run_dir).expanduser().resolve()

with (INFERENCE_RUN_DIR / "config.yaml").open("r", encoding="utf-8") as handle:
    inference_config_payload = yaml.safe_load(handle)
POSTERIOR_CFG = dataclass_from_dict(PosteriorConfig, inference_config_payload)

inference_checkpoint = (
    INFERENCE_RUN_DIR
    / "artefacts"
    / POLICY_CFG.inference_checkpoint_name
)
if not inference_checkpoint.is_file():
    raise FileNotFoundError(f"Missing posterior checkpoint: {inference_checkpoint}")

# The one-step updater was trained on random histories H<=max_history_steps. A policy
# rollout reaches an input belief conditioned on t previous observations before step t,
# so horizon-1 should remain within the updater's training support.
if POLICY_CFG.horizon - 1 > POSTERIOR_CFG.max_history_steps:
    raise ValueError(
        "Policy horizon is outside the posterior updater's trained history support: "
        f"horizon-1={POLICY_CFG.horizon - 1} > "
        f"max_history_steps={POSTERIOR_CFG.max_history_steps}. Retrain the posterior "
        "model with a larger max_history_steps or shorten the policy horizon."
    )
if POLICY_CFG.num_belief_particles != POSTERIOR_CFG.num_particles:
    print(
        "WARNING: the set architecture permits variable particle counts, but the "
        "posterior updater was trained with",
        POSTERIOR_CFG.num_particles,
        "particles and the policy requests",
        POLICY_CFG.num_belief_particles,
        ". Matching them is recommended.",
    )

frozen_inference = load_posterior_model(
    inference_checkpoint,
    POSTERIOR_CFG,
    key=jax.random.key(0),
)

RUN_DIR = make_run_dir(POLICY_CFG.env_name, POLICY_CFG.runs_base_policy)
SCRIPT_PATH = Path(globals().get("__file__", "train_design_policy.py")).resolve()
COMMON_PATH = Path(bsc.__file__).resolve()
snapshot_files(RUN_DIR, [SCRIPT_PATH, COMMON_PATH])
shutil.copy2(INFERENCE_RUN_DIR / "config.yaml", RUN_DIR / "artefacts" / "inference_config.yaml")
shutil.copy2(inference_checkpoint, RUN_DIR / "artefacts" / "frozen_inference.eqx")
save_config_yaml(
    POLICY_CFG,
    RUN_DIR / "config.yaml",
    extra={
        "training_complete": False,
        "inference_run_dir_resolved": str(INFERENCE_RUN_DIR),
        "inference_checkpoint_resolved": str(inference_checkpoint),
    },
)

print("Frozen inference run:", INFERENCE_RUN_DIR)
print("Policy run directory:", RUN_DIR)
print("Policy configuration:\n", yaml.safe_dump(asdict(POLICY_CFG), sort_keys=False))
print("Posterior configuration:\n", yaml.safe_dump(asdict(POSTERIOR_CFG), sort_keys=False))


#%% 2) Theta data loaders and one fixed rollout ground truth
prior = SourceLocPrior(K=POSTERIOR_CFG.K, prior_std=POSTERIOR_CFG.prior_std)
train_loader = make_theta_train_loader(prior, POLICY_CFG)
eval_loader = make_theta_eval_loader(prior, POLICY_CFG)
fixed_theta = eval_loader.dataset[0]
np.save(RUN_DIR / "artefacts" / "fixed_theta.npy", fixed_theta)
steps_per_epoch = (
    len(train_loader)
    if POLICY_CFG.data_mode == "finite"
    else POLICY_CFG.steps_per_epoch
)
print("Optimiser steps per epoch:", steps_per_epoch)
print("Fixed ground truth theta:\n", fixed_theta)

source_permutations_np, permutations_are_exact = build_source_permutations(
    POSTERIOR_CFG.K,
    POSTERIOR_CFG.exact_permutation_max_k,
    POSTERIOR_CFG.sampled_permutations,
    POSTERIOR_CFG.seed + 313,
)
SOURCE_PERMUTATIONS = jnp.asarray(source_permutations_np)
print(
    f"Using {len(source_permutations_np)} source permutations; "
    f"exact marginalisation={permutations_are_exact}."
)

# The initial belief is the original prior. This pre-training plot makes the scale of
# the downstream rollouts explicit before policy learning starts.
initial_rng = np.random.default_rng(POLICY_CFG.seed + 1234)
initial_belief_np = prior.sample_n(initial_rng, POLICY_CFG.num_belief_particles)
fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
ax.scatter(
    initial_belief_np[..., 0].reshape(-1),
    initial_belief_np[..., 1].reshape(-1),
    s=11,
    alpha=0.28,
    label="initial prior belief",
)
ax.scatter(
    fixed_theta[:, 0],
    fixed_theta[:, 1],
    marker="*",
    s=190,
    label="fixed true sources",
)
ax.set_xlim(POSTERIOR_CFG.design_low, POSTERIOR_CFG.design_high)
ax.set_ylim(POSTERIOR_CFG.design_low, POSTERIOR_CFG.design_high)
ax.set_aspect("equal")
ax.grid(alpha=0.2)
ax.legend()
ax.set_title("Fixed policy-evaluation episode before any observations")
fig.savefig(RUN_DIR / "plots" / "fixed_policy_episode_initial_prior.png", dpi=160)
display(fig)
plt.close(fig)


#%% 3) Create the set-Transformer design policy
jax_key, policy_key = jax.random.split(jax_key)
policy = DesignPolicyTransformer(POSTERIOR_CFG, POLICY_CFG, key=policy_key)
eqx.tree_pprint(policy)

# For K=1, the two-dimensional design has the same geometry as theta and may be read as
# the policy's current best source-location guess. For K>1, one design cannot equal all
# K sources; the NLL objective learns where a measurement is useful, while the optional
# coverage auxiliary encourages the sequence to visit every source rather than repeatedly
# selecting only the easiest one.
example_design = policy(
    jnp.asarray(initial_belief_np),
    jnp.asarray(0.0),
)
print("Initial policy design for the fixed prior sample:", np.asarray(example_design))


#%% 4) Differentiable rollout, losses, train/eval utilities, and trajectory extraction
optimizer = optax.chain(
    optax.clip_by_global_norm(POLICY_CFG.grad_clip_norm),
    optax.adamw(
        learning_rate=POLICY_CFG.learning_rate,
        weight_decay=POLICY_CFG.weight_decay,
    ),
)
opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))


def assignment_invariant_belief_mean_rmse(belief: jax.Array, theta_true: jax.Array):
    """RMSE between mean belief sources and theta, minimised over source labels."""
    mean_theta = jnp.mean(belief, axis=0)
    permuted = theta_true[SOURCE_PERMUTATIONS]
    mse = jnp.mean((mean_theta[None, :, :] - permuted) ** 2, axis=(-1, -2))
    return jnp.sqrt(jnp.min(mse))


def differentiable_policy_loss(
    candidate_policy: DesignPolicyTransformer,
    theta_true: jax.Array,
    key: jax.Array,
    inference_model,
):
    """End-to-end differentiable expected design loss for a batch of true thetas.

    The frozen inference parameters are constants, but derivatives flow *through* the
    inference computations with respect to designs and belief particles. Gumbel-softmax
    mixture sampling provides a differentiable approximation to recursively drawing
    B_{t+1} ~ q_phi.
    """
    batch_size = theta_true.shape[0]
    key, initial_belief_key = jax.random.split(key)
    belief = POSTERIOR_CFG.prior_std * jax.random.normal(
        initial_belief_key,
        shape=(
            batch_size,
            POLICY_CFG.num_belief_particles,
            POSTERIOR_CFG.K,
            2,
        ),
    )

    nlls = []
    designs = []
    belief_spreads = []

    for step in range(POLICY_CFG.horizon):
        step_fraction = jnp.asarray(
            step / max(POLICY_CFG.horizon - 1, 1), dtype=jnp.float32
        )
        proposed_design = jax.vmap(
            lambda particles: candidate_policy(particles, step_fraction)
        )(belief)

        # Reparameterised exploration prevents a deterministic early policy from
        # visiting only a tiny region before useful gradients have developed.
        key, design_noise_key, observation_key, sample_master_key = jax.random.split(
            key, 4
        )
        design_noise = jax.random.normal(design_noise_key, proposed_design.shape)
        design = proposed_design + (
            POLICY_CFG.design_exploration_std
            * (POSTERIOR_CFG.design_high - POSTERIOR_CFG.design_low)
            * design_noise
        )
        design = jnp.clip(
            design,
            POSTERIOR_CFG.design_low,
            POSTERIOR_CFG.design_high,
        )

        observation_mean = source_log_signal_jax(theta_true, design, POSTERIOR_CFG)
        observation = observation_mean + (
            POSTERIOR_CFG.observation_noise_std
            * jax.random.normal(observation_key, observation_mean.shape)
        )
        condition = make_condition_jax(design, observation, POSTERIOR_CFG)

        posterior_params = jax.vmap(inference_model)(belief, condition)
        nll = permutation_marginal_nll(
            posterior_params, theta_true, SOURCE_PERMUTATIONS
        )

        sample_keys = jax.random.split(sample_master_key, batch_size)
        belief = jax.vmap(
            lambda params, sample_key: relaxed_sample_gmm_single(
                params,
                sample_key,
                POLICY_CFG.num_belief_particles,
                POSTERIOR_CFG.K,
                POLICY_CFG.relaxed_mixture_temperature,
            )
        )(posterior_params, sample_keys)

        nlls.append(nll)
        designs.append(design)
        belief_spreads.append(jnp.mean(jnp.var(belief.reshape(batch_size, -1), axis=1)))

    nll_by_step = jnp.stack(nlls, axis=1)          # (B,T)
    design_sequence = jnp.stack(designs, axis=1)  # (B,T,2)

    # Primary proper-scoring objective. Intermediate terms provide denser gradients;
    # the terminal term directly optimises the final posterior after budget T.
    intermediate_nll = jnp.mean(nll_by_step)
    terminal_nll = jnp.mean(nll_by_step[:, -1])

    # Simulation-only oracle shaping. For every true source, find a smooth minimum
    # squared distance to any design in the trajectory:
    #
    #   C = mean_k softmin_t ||x_t - theta_k||^2.
    #
    # For K=1 this formalises "design = best current theta guess". For K>1 it
    # discourages repeatedly probing only one source. The NLL remains the main loss.
    squared_design_to_source = jnp.sum(
        (
            design_sequence[:, :, None, :]
            - theta_true[:, None, :, :]
        )
        ** 2,
        axis=-1,
    )  # (B,T,K)
    softmin_temperature = jnp.asarray(0.10, dtype=jnp.float32)
    coverage_per_source = -softmin_temperature * (
        jax.scipy.special.logsumexp(
            -squared_design_to_source / softmin_temperature,
            axis=1,
        )
        - jnp.log(POLICY_CFG.horizon)
    )
    oracle_coverage = jnp.mean(coverage_per_source)

    if POLICY_CFG.horizon > 1:
        design_smoothness = jnp.mean(
            (design_sequence[:, 1:, :] - design_sequence[:, :-1, :]) ** 2
        )
    else:
        design_smoothness = jnp.asarray(0.0)

    total = (
        POLICY_CFG.intermediate_nll_weight * intermediate_nll
        + POLICY_CFG.terminal_nll_weight * terminal_nll
        + POLICY_CFG.oracle_coverage_weight * oracle_coverage
        + POLICY_CFG.design_smoothness_weight * design_smoothness
    )

    final_rmse = jnp.mean(
        jax.vmap(assignment_invariant_belief_mean_rmse)(belief, theta_true)
    )
    final_spread = jnp.mean(jnp.var(belief.reshape(batch_size, -1), axis=1))
    metrics = {
        "loss": total,
        "intermediate_nll": intermediate_nll,
        "terminal_nll": terminal_nll,
        "oracle_coverage": oracle_coverage,
        "design_smoothness": design_smoothness,
        "final_belief_mean_rmse": final_rmse,
        "final_belief_spread": final_spread,
        "first_step_nll": jnp.mean(nll_by_step[:, 0]),
    }
    return total, metrics


@eqx.filter_jit
def train_step(candidate_policy, candidate_opt_state, theta_true, key, inference_model):
    (_, metrics), grads = eqx.filter_value_and_grad(
        differentiable_policy_loss,
        has_aux=True,
    )(candidate_policy, theta_true, key, inference_model)
    params = eqx.filter(candidate_policy, eqx.is_array)
    updates, candidate_opt_state = optimizer.update(grads, candidate_opt_state, params)
    candidate_policy = eqx.apply_updates(candidate_policy, updates)
    grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
    return candidate_policy, candidate_opt_state, metrics, grad_norm


@eqx.filter_jit
def eval_step(candidate_policy, theta_true, key, inference_model):
    _, metrics = differentiable_policy_loss(
        candidate_policy, theta_true, key, inference_model
    )
    return metrics


def evaluate(candidate_policy, loader, evaluation_seed):
    collected: dict[str, list[float]] = {}
    root_key = jax.random.key(evaluation_seed)
    for batch_index, theta_np in enumerate(loader):
        batch_key = jax.random.fold_in(root_key, batch_index)
        metrics = jax.device_get(
            eval_step(
                candidate_policy,
                jnp.asarray(theta_np),
                batch_key,
                frozen_inference,
            )
        )
        for name, value in metrics.items():
            collected.setdefault(name, []).append(float(value))
    return {name: float(np.mean(values)) for name, values in collected.items()}


def rollout_single_discrete(candidate_policy, theta_true_np, seed):
    """Deployment-faithful single rollout using exact categorical GMM sampling.

    This utility returns arrays only. All diagnostic plotting is intentionally kept
    directly in the later notebook cells.
    """
    theta_true = jnp.asarray(theta_true_np)
    key = jax.random.key(seed)
    key, initial_key = jax.random.split(key)
    belief = POSTERIOR_CFG.prior_std * jax.random.normal(
        initial_key,
        shape=(
            POLICY_CFG.num_belief_particles,
            POSTERIOR_CFG.K,
            2,
        ),
    )
    beliefs = [np.asarray(jax.device_get(belief))]
    designs = []
    observations = []
    nlls = []

    for step in range(POLICY_CFG.horizon):
        step_fraction = jnp.asarray(
            step / max(POLICY_CFG.horizon - 1, 1), dtype=jnp.float32
        )
        design = candidate_policy(belief, step_fraction)
        key, observation_key, sample_key = jax.random.split(key, 3)
        observation_mean = source_log_signal_jax(theta_true, design, POSTERIOR_CFG)
        observation = observation_mean + (
            POSTERIOR_CFG.observation_noise_std
            * jax.random.normal(observation_key, ())
        )
        condition = make_condition_jax(design, observation, POSTERIOR_CFG)
        posterior_params = frozen_inference(belief, condition)
        nll = permutation_marginal_nll(
            bsc.GMMParams(
                logits=posterior_params.logits[None, ...],
                means=posterior_params.means[None, ...],
                log_scales=posterior_params.log_scales[None, ...],
            ),
            theta_true[None, ...],
            SOURCE_PERMUTATIONS,
        )[0]
        belief = sample_gmm_single(
            posterior_params,
            sample_key,
            POLICY_CFG.num_belief_particles,
            POSTERIOR_CFG.K,
        )

        designs.append(np.asarray(jax.device_get(design)))
        observations.append(float(jax.device_get(observation)))
        nlls.append(float(jax.device_get(nll)))
        beliefs.append(np.asarray(jax.device_get(belief)))

    return {
        "theta_true": np.asarray(theta_true_np),
        "beliefs": beliefs,
        "designs": np.stack(designs, axis=0),
        "observations": np.asarray(observations),
        "nlls": np.asarray(nlls),
        "spreads": np.asarray(
            [np.mean(np.var(b.reshape(b.shape[0], -1), axis=0)) for b in beliefs]
        ),
    }


initial_metrics = evaluate(policy, eval_loader, POLICY_CFG.seed + 100_000)
print("Initial policy validation metrics:", initial_metrics)


#%% 5) Policy training, evaluation every epoch, tenth-of-total-epochs visualisation, saving
history: dict[str, list[float]] = {
    "step_loss": [],
    "step_terminal_nll": [],
    "step_coverage": [],
    "step_final_rmse": [],
    "step_final_spread": [],
    "step_grad_norm": [],
    "epoch_train_loss": [],
    "epoch_val_loss": [],
    "epoch_val_terminal_nll": [],
    "epoch_val_final_rmse": [],
    "epoch_val_final_spread": [],
}

visualisation_epochs = sorted(
    set(
        int(math.ceil(fraction * POLICY_CFG.epochs / 10.0))
        for fraction in range(1, 11)
    )
)
print("Fixed-rollout visualisation epochs:", visualisation_epochs)

best_val_loss = float("inf")
best_epoch = 0
global_step = 0
training_started_at = time.time()
train_key = jax.random.key(POLICY_CFG.seed + 1)

for epoch in range(1, POLICY_CFG.epochs + 1):
    epoch_started_at = time.time()
    train_losses_this_epoch: list[float] = []

    if POLICY_CFG.data_mode == "finite":
        epoch_iterator = iter(train_loader)
    else:
        epoch_iterator = islice(iter(train_loader), POLICY_CFG.steps_per_epoch)

    progress = tqdm(
        enumerate(epoch_iterator, start=1),
        total=steps_per_epoch,
        desc=f"Policy epoch {epoch:03d}/{POLICY_CFG.epochs:03d}",
        dynamic_ncols=True,
        leave=True,
    )

    for step_in_epoch, theta_np in progress:
        train_key, step_key = jax.random.split(train_key)
        policy, opt_state, metrics, grad_norm = train_step(
            policy,
            opt_state,
            jnp.asarray(theta_np),
            step_key,
            frozen_inference,
        )
        host = {name: float(value) for name, value in jax.device_get(metrics).items()}
        host_grad_norm = float(jax.device_get(grad_norm))

        global_step += 1
        train_losses_this_epoch.append(host["loss"])
        history["step_loss"].append(host["loss"])
        history["step_terminal_nll"].append(host["terminal_nll"])
        history["step_coverage"].append(host["oracle_coverage"])
        history["step_final_rmse"].append(host["final_belief_mean_rmse"])
        history["step_final_spread"].append(host["final_belief_spread"])
        history["step_grad_norm"].append(host_grad_norm)

        progress.set_postfix(
            loss=f"{host['loss']:.4f}",
            terminal_nll=f"{host['terminal_nll']:.4f}",
            rmse=f"{host['final_belief_mean_rmse']:.3f}",
        )

    epoch_train_loss = float(np.mean(train_losses_this_epoch))
    val_metrics = evaluate(policy, eval_loader, POLICY_CFG.seed + 100_000)
    history["epoch_train_loss"].append(epoch_train_loss)
    history["epoch_val_loss"].append(val_metrics["loss"])
    history["epoch_val_terminal_nll"].append(val_metrics["terminal_nll"])
    history["epoch_val_final_rmse"].append(val_metrics["final_belief_mean_rmse"])
    history["epoch_val_final_spread"].append(val_metrics["final_belief_spread"])

    save_policy_model(RUN_DIR / "artefacts" / "model_last.eqx", policy)
    # Persist optimiser moments and the functional JAX PRNG key for exact continuation.
    eqx.tree_serialise_leaves(
        RUN_DIR / "artefacts" / "training_state_last.eqx",
        (policy, opt_state, train_key),
    )
    if epoch % POLICY_CFG.save_every_epochs == 0:
        save_policy_model(
            RUN_DIR / "artefacts" / f"model_epoch_{epoch:04d}.eqx", policy
        )
    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        best_epoch = epoch
        save_policy_model(RUN_DIR / "artefacts" / "model_best.eqx", policy)

    np.savez_compressed(
        RUN_DIR / "artefacts" / "history.npz",
        **{name: np.asarray(values, dtype=np.float64) for name, values in history.items()},
    )
    save_json(
        RUN_DIR / "artefacts" / "training_state.json",
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "elapsed_seconds": time.time() - training_started_at,
            "frozen_inference_run": str(INFERENCE_RUN_DIR),
        },
    )

    print(
        f"Epoch {epoch:03d}: train={epoch_train_loss:.6f} | "
        f"val={val_metrics['loss']:.6f} | "
        f"val terminal NLL={val_metrics['terminal_nll']:.6f} | "
        f"val final RMSE={val_metrics['final_belief_mean_rmse']:.5f} | "
        f"val spread={val_metrics['final_belief_spread']:.5f} | "
        f"{time.time() - epoch_started_at:.1f}s"
    )

    # Direct plotting at 10%,20%,...,100% of the total epoch budget.
    if epoch in visualisation_epochs:
        trajectory = rollout_single_discrete(
            policy,
            fixed_theta,
            seed=POLICY_CFG.seed + 700_000,
        )
        all_points = [fixed_theta.reshape(-1, 2), trajectory["designs"].reshape(-1, 2)]
        all_points.extend(belief.reshape(-1, 2) for belief in trajectory["beliefs"])
        shared_points = np.concatenate(all_points, axis=0)
        lim = max(
            3.0 * POSTERIOR_CFG.prior_std,
            1.15 * float(np.quantile(np.abs(shared_points), 0.995)),
        )

        fig, axes = plt.subplots(
            2,
            POLICY_CFG.horizon + 1,
            figsize=(3.15 * (POLICY_CFG.horizon + 1), 6.3),
            squeeze=False,
            constrained_layout=True,
        )
        for stage in range(POLICY_CFG.horizon + 1):
            belief = trajectory["beliefs"][stage]
            axes[0, stage].scatter(
                belief[..., 0].reshape(-1),
                belief[..., 1].reshape(-1),
                s=9,
                alpha=0.24,
            )
            axes[0, stage].scatter(
                fixed_theta[:, 0],
                fixed_theta[:, 1],
                marker="*",
                s=135,
            )
            if stage == 0:
                axes[0, stage].set_title("B0: prior")
            else:
                axes[0, stage].set_title(
                    f"B{stage}\nNLL={trajectory['nlls'][stage-1]:.2f}\n"
                    f"spread={trajectory['spreads'][stage]:.3f}"
                )

            axes[1, stage].scatter(
                fixed_theta[:, 0],
                fixed_theta[:, 1],
                marker="*",
                s=135,
                label="true sources" if stage == 0 else None,
            )
            if stage > 0:
                used_designs = trajectory["designs"][:stage]
                axes[1, stage].plot(
                    used_designs[:, 0],
                    used_designs[:, 1],
                    marker="X",
                    linewidth=1.2,
                )
                for design_index, design in enumerate(used_designs, start=1):
                    axes[1, stage].annotate(
                        str(design_index),
                        design,
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
            axes[1, stage].set_title(f"Designs available by stage {stage}")

            for row in range(2):
                axes[row, stage].set_xlim(-lim, lim)
                axes[row, stage].set_ylim(-lim, lim)
                axes[row, stage].set_aspect("equal")
                axes[row, stage].grid(alpha=0.15)

        fig.suptitle(
            f"Fixed sequential rollout at epoch {epoch}/{POLICY_CFG.epochs} "
            f"({100.0 * epoch / POLICY_CFG.epochs:.0f}% of total training)",
            fontsize=15,
        )
        fig.savefig(
            RUN_DIR / "plots" / f"fixed_rollout_epoch_{epoch:04d}.png",
            dpi=160,
        )
        display(fig)
        plt.close(fig)

save_config_yaml(
    POLICY_CFG,
    RUN_DIR / "config.yaml",
    extra={
        "training_complete": True,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "inference_run_dir_resolved": str(INFERENCE_RUN_DIR),
        "inference_checkpoint_resolved": str(inference_checkpoint),
    },
)
print("Policy training complete. Best epoch:", best_epoch, "best val loss:", best_val_loss)


#%% 6) Plot and save policy training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

axes[0, 0].plot(history["step_loss"], alpha=0.62)
axes[0, 0].set_title("Policy objective per optimiser step")
axes[0, 0].set_xlabel("Global train step")
axes[0, 0].set_yscale("symlog")
axes[0, 0].grid(alpha=0.25)

axes[0, 1].plot(history["step_terminal_nll"], alpha=0.62, label="terminal NLL")
axes[0, 1].plot(history["step_coverage"], alpha=0.62, label="oracle coverage")
axes[0, 1].set_title("Policy loss components")
axes[0, 1].set_xlabel("Global train step")
axes[0, 1].set_yscale("symlog")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.25)

epoch_axis = np.arange(1, len(history["epoch_train_loss"]) + 1)
axes[1, 0].plot(epoch_axis, history["epoch_train_loss"], marker="o", label="train")
axes[1, 0].plot(epoch_axis, history["epoch_val_loss"], marker="s", label="validation")
axes[1, 0].set_title("Epoch policy objective")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_yscale("symlog")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.25)

axes[1, 1].plot(history["step_final_rmse"], alpha=0.62, label="final belief-mean RMSE")
axes[1, 1].plot(history["step_final_spread"], alpha=0.62, label="final belief spread")
axes[1, 1].plot(history["step_grad_norm"], alpha=0.50, label="gradient norm")
axes[1, 1].set_title("Final-belief and optimisation diagnostics")
axes[1, 1].set_xlabel("Global train step")
axes[1, 1].set_yscale("log")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.25)

fig.savefig(RUN_DIR / "plots" / "training_curves.png", dpi=160)
display(fig)
plt.close(fig)


#%% 7) Direct giant multi-axis plot of multiple sequential location-finding rollouts
n_examples = min(POLICY_CFG.final_plot_examples, len(eval_loader.dataset))
trajectories = [
    rollout_single_discrete(
        policy,
        eval_loader.dataset[index],
        seed=POLICY_CFG.seed + 900_000 + index,
    )
    for index in range(n_examples)
]

all_points = []
for trajectory in trajectories:
    all_points.append(trajectory["theta_true"].reshape(-1, 2))
    all_points.append(trajectory["designs"].reshape(-1, 2))
    all_points.extend(belief.reshape(-1, 2) for belief in trajectory["beliefs"])
all_points_np = np.concatenate(all_points, axis=0)
shared_lim = max(
    3.0 * POSTERIOR_CFG.prior_std,
    1.15 * float(np.quantile(np.abs(all_points_np), 0.995)),
)

fig, axes = plt.subplots(
    n_examples,
    POLICY_CFG.horizon + 1,
    figsize=(3.2 * (POLICY_CFG.horizon + 1), 3.2 * n_examples),
    squeeze=False,
    constrained_layout=True,
)

for row, trajectory in enumerate(trajectories):
    theta_true = trajectory["theta_true"]
    for stage in range(POLICY_CFG.horizon + 1):
        belief = trajectory["beliefs"][stage]
        ax = axes[row, stage]
        ax.scatter(
            belief[..., 0].reshape(-1),
            belief[..., 1].reshape(-1),
            s=8,
            alpha=0.22,
        )
        ax.scatter(theta_true[:, 0], theta_true[:, 1], marker="*", s=125)
        if stage > 0:
            used_designs = trajectory["designs"][:stage]
            ax.plot(
                used_designs[:, 0],
                used_designs[:, 1],
                marker="X",
                linewidth=1.0,
                markersize=6,
            )
        if stage == 0:
            ax.set_title(f"Episode {row + 1}: prior")
        else:
            ax.set_title(
                f"t={stage}, NLL={trajectory['nlls'][stage-1]:.2f}\n"
                f"spread={trajectory['spreads'][stage]:.3f}"
            )
        ax.set_xlim(-shared_lim, shared_lim)
        ax.set_ylim(-shared_lim, shared_lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.13)
        ax.set_xlabel(r"$\theta_x$ / $x_x$")
        ax.set_ylabel(r"$\theta_y$ / $x_y$")

fig.suptitle(
    "Sequential policy rollouts: current beliefs, true sources, and cumulative designs",
    fontsize=16,
)
fig.savefig(RUN_DIR / "plots" / "many_policy_rollouts.png", dpi=160)
display(fig)
plt.close(fig)

print("Saved policy model and artefacts under:", RUN_DIR)
