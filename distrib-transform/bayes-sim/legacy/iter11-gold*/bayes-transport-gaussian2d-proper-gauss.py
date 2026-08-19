#%% 1) Imports, configuration, and experiment description
"""2-D Gaussian-Gaussian sanity check for single-cloud Bayesian particle transport.

This notebook-style script is a conjugate simplification of the attached source-localisation
experiment. It intentionally preserves the main training and evaluation structure:

  * one fixed low-dimensional latent parameter, theta in R^2,
  * one observation per amortized training row,
  * synthetic interpolated Gaussian input clouds C_tau,
  * the same AdaLN-conditioned particle Transformer,
  * exact empirical multivariate energy-score training,
  * fresh continuous iid training data,
  * evaluation-only repeated application of the learned one-step map,
  * per-train-step history collection, checkpointing, plateau LR reduction, and periodic plots.

The location-finding-specific pieces are removed: source labels, source/design geometry,
inverse-distance forward model, design coordinates, source canonicalisation, and SNIS.

The replacement observation model is configurable but deliberately restricted to LINEAR
Gaussian observations, because this is what guarantees Gaussian conjugacy:

    theta ~ N(mu, Sigma)
    y | theta ~ N(A theta, R),       R = sigma_y^2 I_M,

where theta is 2-D and y can have any configured dimension M.  The matrix A is selected by the
config.  Presets include identity observations, either coordinate alone, scalar sums/differences,
and a two-direction projection; a custom Mx2 matrix is also supported.

A genuinely nonlinear mean h(theta) is intentionally NOT implemented here.  In general,

    y | theta ~ N(h(theta), R)

makes the log likelihood non-quadratic in theta, so a Gaussian prior no longer has an exactly
Gaussian posterior.  Since this file is specifically a conjugacy sanity check, every allowed
observation operator is linear.

For any Gaussian prior N(mu, Sigma), one observation y gives the exact posterior

    S          = A Sigma A^T + R
    G          = Sigma A^T S^{-1}
    mu_post    = mu + G (y - A mu)
    Sigma_post = Sigma - G A Sigma.

For the evaluation-only sequence y_1,...,y_t starting from the fixed base prior N(mu_0,Sigma_0),
the closed-form information-form posterior is

    Sigma_t^{-1} = Sigma_0^{-1} + t A^T R^{-1} A
    mu_t          = Sigma_t [Sigma_0^{-1} mu_0 + A^T R^{-1} sum_{s=1}^t y_s].

This exact posterior is used throughout as a sanity-check reference. In particular:
  * every training step records learned-vs-exact posterior mean and covariance errors;
  * sequential evaluation records those errors at every repeated-Bayes step;
  * posterior-evolution plots overlay the learned cloud with exact Gaussian 1-sigma/2-sigma
    ellipses and the exact posterior mean;
  * before training, the fixed example is printed and plotted so theta, A theta, y, and their
    dimensions are immediately visible.

Training algorithm
------------------
For each fresh iid training row:
  1. Draw N iid z_n ~ N(mu_0,Sigma_0), one potential centre theta_tilde ~ N(mu_0,Sigma_0), and
     tau ~ Uniform[0,1].
  2. Form C_tau = (1-tau) z_n + tau theta_tilde.
  3. Conditional on (theta_tilde,tau), C_tau has the exact population law
         N(mu_tau, Sigma_tau),
         mu_tau = (1-tau)mu_0 + tau theta_tilde,
         Sigma_tau = (1-tau)^2 Sigma_0.
     Draw theta* independently from that same Gaussian law.
  4. Draw one y ~ N(A theta*, R).
  5. Transport C_tau with the AdaLN particle Transformer conditioned on y.
  6. Optimize the empirical multivariate energy score against theta*.

For one output cloud Q_hat = N^-1 sum_n delta_{theta_n}, the optimized score is

    ES(Q_hat, theta*) = N^-1 sum_n ||theta_n - theta*||
                        - (2N^2)^-1 sum_{n,m} ||theta_n - theta_m||.

Sequential evaluation
---------------------
Starting from a fresh base-prior cloud, the SAME learned one-observation transport is applied
repeatedly via jax.lax.scan. This recurrence remains evaluation-only and is never differentiated
through. At each prefix it is compared directly with the analytic Gaussian posterior above.

Array shapes
------------
B : batch size        N : particles per cloud        K=2 : theta dimension
M : observation dimension determined by A           T : evaluation horizon

theta_true        [B,K]
observation       [B,M]
prior_particles   [B,N,K]
prior_mean        [B,K]       exact population mean underlying the synthetic input cloud
prior_cov         [B,K,K]     exact population covariance underlying the synthetic input cloud
posterior_theta   [B,N,K]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
import json
import math
import shutil
from pathlib import Path
import time
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from IPython.display import display
from tqdm.auto import tqdm
import yaml

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

Array = jax.Array

# Execution switch: True trains a fresh model into a new run folder; False reloads the best
# checkpoint from the current run folder (cwd) and only regenerates the core diagnostic plots.
train_wm: bool = False

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "DejaVu Sans",
    "axes.titlepad": 8.0,
    "axes.labelpad": 6.0,
})


@dataclass(frozen=True)
class BayesTransportConfig:
    """Active hyperparameters for the 2-D Gaussian conjugate sanity check."""

    env_name: str = "gaussian2d"
    seed: int = 2030
    runs_base: str = "./runs"

    # Fixed 2-D Gaussian base prior. These replace the location-specific geometry parameters.
    prior_mean_x: float = 0.0
    prior_mean_y: float = 0.0
    # Defaults are isotropic on purpose: this is the cleanest conjugate control experiment.
    prior_std_x: float = 1.0
    prior_std_y: float = 1.0
    prior_correlation: float = 0.0

    # Conjugate Gaussian observation model y | theta ~ N(A theta, sigma_y^2 I_M).
    # Only LINEAR operators are allowed: nonlinear h(theta) generally breaks exact Gaussian
    # conjugacy.  Useful presets:
    #   "identity"          : A = [[1,0],[0,1]], M=2
    #   "first_coordinate" : A = [[1,0]],         M=1
    #   "second_coordinate": A = [[0,1]],         M=1
    #   "sum"               : A = [[1,1]]/sqrt(2), M=1
    #   "difference"        : A = [[1,-1]]/sqrt(2), M=1
    #   "two_projections"   : A has rows e1 and (1,1)/sqrt(2), M=2
    #   "custom"            : use custom_observation_matrix below, shape Mx2
    #
    # Nonlinear examples such as h(theta)=[theta_1^2, theta_2], h(theta)=||theta||, or a small
    # neural network are deliberately NOT options here: with Gaussian observation noise they
    # generally produce non-Gaussian posteriors and would defeat this exact-conjugacy check.
    #
    # The default deliberately observes only theta_1, making dim(y)=1 != dim(theta)=2.  This is
    # a useful sanity check: with an independent isotropic prior, the exact posterior contracts
    # only in theta_1 while theta_2 remains at its prior marginal.
    observation_operator: str = "two_projections"
    custom_observation_matrix: tuple[tuple[float, float], ...] = ((1.0, 0.0),)

    # Preserved from the original observation-noise hyperparameter.
    observation_noise_std: float = 0.30

    # Particle-cloud / training-stream sizes (preserved).
    num_particles: int = 64
    n_train_trajectories: int = 4096
    n_eval_trajectories: int = 256
    batch_size: int = 16 * 16
    train_dataloader_num_workers: int = 0
    train_dataloader_prefetch_factor: int = 2

    # Evaluation-only sequential rollout horizon (preserved).
    evaluation_trajectory_length: int = 16

    # Posterior Transformer (preserved, except theta width is K=2 and conditioning width is M).
    hidden_dim: int = 256
    heads: int = 8
    mlp_ratio: int = 4
    posterior_depth: int = 6
    max_theta_displacement: float = 6.0

    # Observation normalisation (preserved).
    y_center: float = 0.0
    y_scale: float = 3.0

    # Optimisation (preserved).
    epochs: int = 20000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1000.0
    lr_plateau_patience: int = 500
    lr_plateau_rtol: float = 1e-4

    # Persistence / visualisation cadence (preserved).
    save_every_epochs: int = 5000


K = 2


def observation_matrix_np(cfg: BayesTransportConfig) -> np.ndarray:
    """Return the configured Mx2 linear observation matrix A.

    Keeping this function linear is the conjugacy guarantee for this sanity-check notebook.
    """
    name = cfg.observation_operator.strip().lower()
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    presets = {
        "identity": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        "first_coordinate": np.asarray([[1.0, 0.0]], dtype=np.float64),
        "second_coordinate": np.asarray([[0.0, 1.0]], dtype=np.float64),
        "sum": np.asarray([[inv_sqrt2, inv_sqrt2]], dtype=np.float64),
        "difference": np.asarray([[inv_sqrt2, -inv_sqrt2]], dtype=np.float64),
        "two_projections": np.asarray([[1.0, 0.0], [inv_sqrt2, inv_sqrt2]], dtype=np.float64),
    }
    if name == "custom":
        A = np.asarray(cfg.custom_observation_matrix, dtype=np.float64)
    elif name in presets:
        A = presets[name]
    else:
        raise ValueError(
            f"Unknown observation_operator={cfg.observation_operator!r}. "
            f"Choose one of {sorted(presets)} or 'custom'. Nonlinear operators are intentionally unsupported."
        )
    if A.ndim != 2 or A.shape[1] != K or A.shape[0] < 1:
        raise ValueError(f"Observation matrix A must have shape [M,{K}] with M>=1; got {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError("Observation matrix A must contain only finite values.")
    return A


def observation_matrix_jax(cfg: BayesTransportConfig) -> Array:
    return jnp.asarray(observation_matrix_np(cfg), dtype=jnp.float32)


def observation_dim(cfg: BayesTransportConfig) -> int:
    return int(observation_matrix_np(cfg).shape[0])


CFG = BayesTransportConfig()
OBS_DIM = observation_dim(CFG)


#%% 2) Run directories and small persistence helpers
def make_run_dir(env_name: str, base: str | Path = "./runs") -> Path:
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = Path(base).expanduser().resolve() / f"{env_name}_{stamp}"
    (run_dir / "plots").mkdir(parents=True, exist_ok=False)
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def copy_running_script_to_run_dir(run_dir: Path) -> Path | None:
    """Archive the exact script into the run folder, when running as a real .py file."""
    if "__file__" not in globals():
        return None
    source = Path(__file__).expanduser().resolve()
    if not source.is_file():
        return None
    destination = run_dir / source.name
    if source != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def save_json(path: str | Path, payload: dict[str, Any]):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_model(path: str | Path, model: "SequentialBayesModel"):
    eqx.tree_serialise_leaves(Path(path), model)


def load_model(path: str | Path, cfg: BayesTransportConfig, *, key: Array | None = None) -> "SequentialBayesModel":
    skeleton = SequentialBayesModel(cfg, key=jax.random.key(0) if key is None else key)
    return eqx.tree_deserialise_leaves(Path(path), skeleton)


#%% 3) Gaussian prior, conjugate likelihood, synthetic interpolation, and exact posterior
def base_prior_mean_np(cfg: BayesTransportConfig = CFG) -> np.ndarray:
    return np.asarray([cfg.prior_mean_x, cfg.prior_mean_y], dtype=np.float64)


def base_prior_cov_np(cfg: BayesTransportConfig = CFG) -> np.ndarray:
    sx, sy, rho = float(cfg.prior_std_x), float(cfg.prior_std_y), float(cfg.prior_correlation)
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("prior_std_x and prior_std_y must be positive.")
    if not (-1.0 < rho < 1.0):
        raise ValueError("prior_correlation must lie strictly between -1 and 1.")
    return np.asarray([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]], dtype=np.float64)


def likelihood_cov_np(cfg: BayesTransportConfig = CFG) -> np.ndarray:
    variance = float(cfg.observation_noise_std) ** 2
    if variance <= 0.0:
        raise ValueError("observation_noise_std must be positive.")
    m = observation_matrix_np(cfg).shape[0]
    return variance * np.eye(m, dtype=np.float64)


def base_prior_mean_jax(cfg: BayesTransportConfig = CFG) -> Array:
    return jnp.asarray([cfg.prior_mean_x, cfg.prior_mean_y], dtype=jnp.float32)


def base_prior_cov_jax(cfg: BayesTransportConfig = CFG) -> Array:
    sx, sy, rho = cfg.prior_std_x, cfg.prior_std_y, cfg.prior_correlation
    return jnp.asarray([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]], dtype=jnp.float32)


def likelihood_cov_jax(cfg: BayesTransportConfig = CFG) -> Array:
    return (cfg.observation_noise_std**2) * jnp.eye(observation_matrix_np(cfg).shape[0], dtype=jnp.float32)


def observation_mean_np(theta: np.ndarray, cfg: BayesTransportConfig = CFG) -> np.ndarray:
    """Compute A theta for theta shaped [...,2], returning [...,M]."""
    theta = np.asarray(theta, dtype=np.float64)
    A = observation_matrix_np(cfg)
    return np.einsum("...k,mk->...m", theta, A)


def observation_mean_jax(theta: Array, cfg: BayesTransportConfig = CFG) -> Array:
    A = observation_matrix_jax(cfg)
    return jnp.einsum("...k,mk->...m", theta, A)


def sample_gaussian_np(
    rng: np.random.Generator, mean: np.ndarray, cov: np.ndarray, n: int
) -> np.ndarray:
    """Draw n iid K-dimensional Gaussian samples, including the degenerate-covariance limit."""
    mean = np.asarray(mean, dtype=np.float64)
    cov = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    root = evecs @ np.diag(np.sqrt(evals))
    standard = rng.normal(size=(int(n), K))
    return (mean[None, :] + standard @ root.T).astype(np.float32)


def sample_base_prior_np(rng: np.random.Generator, n: int, cfg: BayesTransportConfig = CFG) -> np.ndarray:
    """Draw n iid theta samples from the fixed Gaussian base prior N(mu_0, Sigma_0)."""
    return sample_gaussian_np(rng, base_prior_mean_np(cfg), base_prior_cov_np(cfg), int(n))


def gaussian_update_np(
    prior_mean: np.ndarray, prior_cov: np.ndarray, observation: np.ndarray, cfg: BayesTransportConfig = CFG
) -> tuple[np.ndarray, np.ndarray]:
    """Exact one-observation posterior for y | theta ~ N(A theta, R)."""
    mean = np.asarray(prior_mean, dtype=np.float64)
    cov = np.asarray(prior_cov, dtype=np.float64)
    y = np.asarray(observation, dtype=np.float64).reshape(-1)
    A = observation_matrix_np(cfg)
    R = likelihood_cov_np(cfg)
    innovation_cov = A @ cov @ A.T + R
    gain = np.linalg.solve(innovation_cov, A @ cov).T  # cov A^T (A cov A^T + R)^-1
    post_mean = mean + gain @ (y - A @ mean)
    post_cov = cov - gain @ A @ cov
    post_cov = 0.5 * (post_cov + post_cov.T)
    return post_mean.astype(np.float32), post_cov.astype(np.float32)


def gaussian_update_jax(prior_mean: Array, prior_cov: Array, observation: Array) -> tuple[Array, Array]:
    """JAX exact one-step conjugate update, with CFG fixed on the active path."""
    A = observation_matrix_jax(CFG)
    R = likelihood_cov_jax(CFG)
    innovation_cov = A @ prior_cov @ A.T + R
    gain = jnp.linalg.solve(innovation_cov, A @ prior_cov).T
    post_mean = prior_mean + gain @ (observation - A @ prior_mean)
    post_cov = prior_cov - gain @ A @ prior_cov
    post_cov = 0.5 * (post_cov + post_cov.T)
    return post_mean, post_cov


def exact_base_posterior_np(
    observations: np.ndarray, cfg: BayesTransportConfig = CFG
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form posterior after y_1,...,y_t under the fixed linear-Gaussian model.

    Sigma_t^{-1} = Sigma_0^{-1} + t A^T R^{-1} A
    mu_t = Sigma_t [Sigma_0^{-1} mu_0 + A^T R^{-1} sum_s y_s].
    """
    A = observation_matrix_np(cfg)
    observations = np.asarray(observations, dtype=np.float64).reshape(-1, A.shape[0])
    mu0 = base_prior_mean_np(cfg)
    Sigma0 = base_prior_cov_np(cfg)
    R = likelihood_cov_np(cfg)
    if len(observations) == 0:
        return mu0.astype(np.float32), Sigma0.astype(np.float32)
    P0 = np.linalg.inv(Sigma0)
    Rinv = np.linalg.inv(R)
    precision_increment = A.T @ Rinv @ A
    precision = P0 + len(observations) * precision_increment
    cov = np.linalg.inv(precision)
    rhs = P0 @ mu0 + A.T @ Rinv @ observations.sum(axis=0)
    mean = cov @ rhs
    return mean.astype(np.float32), cov.astype(np.float32)


def exact_base_posterior_sequence_np(
    observations: np.ndarray, cfg: BayesTransportConfig = CFG
) -> tuple[np.ndarray, np.ndarray]:
    A = observation_matrix_np(cfg)
    observations = np.asarray(observations, dtype=np.float64).reshape(-1, A.shape[0])
    means, covs = [], []
    for t in range(1, len(observations) + 1):
        mean, cov = exact_base_posterior_np(observations[:t], cfg)
        means.append(mean)
        covs.append(cov)
    return np.stack(means), np.stack(covs)


def exact_base_posterior_sequence_jax(observations: Array) -> tuple[Array, Array]:
    """Vectorised information-form posterior for every prefix of one observation sequence."""
    mu0 = base_prior_mean_jax(CFG)
    Sigma0 = base_prior_cov_jax(CFG)
    A = observation_matrix_jax(CFG)
    R = likelihood_cov_jax(CFG)
    P0 = jnp.linalg.inv(Sigma0)
    Rinv = jnp.linalg.inv(R)
    precision_increment = A.T @ Rinv @ A
    t = jnp.arange(1, observations.shape[0] + 1, dtype=observations.dtype)
    cumulative_y = jnp.cumsum(observations, axis=0)
    precisions = P0[None, :, :] + t[:, None, None] * precision_increment[None, :, :]
    covs = jnp.linalg.inv(precisions)
    eta0 = P0 @ mu0
    observation_information = jnp.einsum("km,mn,tn->tk", A.T, Rinv, cumulative_y)
    rhs = eta0[None, :] + observation_information
    means = jnp.einsum("tij,tj->ti", covs, rhs)
    return means, covs


def sample_interpolated_prior_and_truth_np(
    rng: np.random.Generator, n_particles: int, cfg: BayesTransportConfig = CFG
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one synthetic Gaussian training prior cloud and an iid truth from its exact law.

    z_n ~ N(mu_0,Sigma_0), theta_tilde ~ N(mu_0,Sigma_0), tau ~ U[0,1]
    C_tau = (1-tau) z_n + tau theta_tilde.

    Conditional on (theta_tilde,tau),
      C_tau ~ N((1-tau)mu_0 + tau theta_tilde, (1-tau)^2 Sigma_0).
    """
    base_particles = sample_base_prior_np(rng, n_particles, cfg)
    potential_theta = sample_base_prior_np(rng, 1, cfg)[0]
    tau = np.float32(rng.uniform(0.0, 1.0))
    one_minus_tau = np.float32(1.0) - tau

    prior_particles = (one_minus_tau * base_particles + tau * potential_theta[None, :]).astype(np.float32)
    prior_mean = (
        one_minus_tau * base_prior_mean_np(cfg).astype(np.float32) + tau * potential_theta
    ).astype(np.float32)
    prior_cov = ((float(one_minus_tau) ** 2) * base_prior_cov_np(cfg)).astype(np.float32)
    theta_true = sample_gaussian_np(rng, prior_mean, prior_cov, 1)[0]
    return prior_particles, theta_true, prior_mean, prior_cov


def simulate_observation_np(
    rng: np.random.Generator, theta_true: np.ndarray, cfg: BayesTransportConfig = CFG
) -> np.ndarray:
    """Draw one M-D observation y ~ N(A theta_true, sigma_y^2 I_M)."""
    mean = observation_mean_np(theta_true, cfg).reshape(-1)
    noise = cfg.observation_noise_std * rng.normal(size=(mean.shape[0],))
    return (mean + noise).astype(np.float32)


def simulate_trajectories(
    rng: np.random.Generator, n_trajectories: int, trajectory_length: int, cfg: BayesTransportConfig = CFG
) -> dict[str, np.ndarray]:
    """Evaluation-only trajectories: one theta* ~ base prior, reused for T iid linear-Gaussian observations."""
    n_trajectories, trajectory_length = int(n_trajectories), int(trajectory_length)
    theta_true = sample_base_prior_np(rng, n_trajectories, cfg)
    observations = np.zeros((n_trajectories, trajectory_length, observation_dim(cfg)), dtype=np.float32)
    for m in range(n_trajectories):
        for t in range(trajectory_length):
            observations[m, t] = simulate_observation_np(rng, theta_true[m], cfg)
    return {"theta_true": theta_true, "observations": observations}


def simulate_iid_joint_samples(
    rng: np.random.Generator, n_samples: int, cfg: BayesTransportConfig = CFG
) -> dict[str, np.ndarray]:
    """Fixed iid (C_tau, theta*, y) rows used for amortized validation, with exact prior parameters."""
    n_samples = int(n_samples)
    theta_true = np.zeros((n_samples, K), dtype=np.float32)
    observation = np.zeros((n_samples, observation_dim(cfg)), dtype=np.float32)
    prior_particles = np.zeros((n_samples, cfg.num_particles, K), dtype=np.float32)
    prior_mean = np.zeros((n_samples, K), dtype=np.float32)
    prior_cov = np.zeros((n_samples, K, K), dtype=np.float32)
    for m in range(n_samples):
        particles, theta, mean, cov = sample_interpolated_prior_and_truth_np(rng, cfg.num_particles, cfg)
        theta_true[m] = theta
        observation[m] = simulate_observation_np(rng, theta, cfg)
        prior_particles[m] = particles
        prior_mean[m] = mean
        prior_cov[m] = cov
    return {
        "theta_true": theta_true,
        "observation": observation,
        "prior_particles": prior_particles,
        "prior_mean": prior_mean,
        "prior_cov": prior_cov,
    }


class ContinuousJointDataset(IterableDataset):
    """Infinite CPU stream of fresh iid conjugate (C_tau, theta*, y) training rows."""

    def __init__(self, cfg: BayesTransportConfig, *, seed: int):
        super().__init__()
        self.cfg = cfg
        self.seed = int(seed)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        rng = np.random.default_rng(self.seed + 1_000_003 * worker_id)
        while True:
            particles, theta_true, prior_mean, prior_cov = sample_interpolated_prior_and_truth_np(
                rng, self.cfg.num_particles, self.cfg
            )
            observation = simulate_observation_np(rng, theta_true, self.cfg)
            yield {
                "theta_true": theta_true,
                "observation": observation,
                "prior_particles": particles,
                "prior_mean": prior_mean,
                "prior_cov": prior_cov,
            }


def _numpy_collate(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack([np.asarray(s[name]) for s in samples], axis=0) for name in samples[0]}


def make_continuous_train_loader(cfg: BayesTransportConfig = CFG, *, seed: int) -> DataLoader:
    dataset = ContinuousJointDataset(cfg, seed=seed)
    kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.train_dataloader_num_workers,
        collate_fn=_numpy_collate,
        drop_last=True,
    )
    if cfg.train_dataloader_num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.train_dataloader_prefetch_factor
    return DataLoader(**kwargs)


#%% 4) Small array/token helpers
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _modulate(x: Array, shift: Array, scale: Array) -> Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


#%% 5) AdaLN particle-conditioning block and the Posterior Transformer
class AdaLNParticleBlock(eqx.Module):
    """Particle self-attention conditioned on the single M-D observation via AdaLN-Zero."""

    self_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    modulation: eqx.nn.Linear

    def __init__(self, hidden_dim: int, conditioning_dim: int, heads: int, mlp_dim: int, *, key: Array):
        self_key, ff1_key, ff2_key, mod_key = jax.random.split(key, 4)
        self.self_norm = eqx.nn.LayerNorm(hidden_dim)
        self.ff_norm = eqx.nn.LayerNorm(hidden_dim)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=hidden_dim,
            key_size=hidden_dim,
            value_size=hidden_dim,
            output_size=hidden_dim,
            dropout_p=0.0,
            key=self_key,
        )
        self.ff_in = eqx.nn.Linear(hidden_dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, hidden_dim, key=ff2_key)
        modulation = eqx.nn.Linear(conditioning_dim, 6 * hidden_dim, key=mod_key)
        modulation = eqx.tree_at(lambda l: l.weight, modulation, jnp.zeros_like(modulation.weight))
        modulation = eqx.tree_at(lambda l: l.bias, modulation, jnp.zeros_like(modulation.bias))
        self.modulation = modulation

    def __call__(self, particles: Array, conditioning: Array) -> Array:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = jnp.split(
            self.modulation(jax.nn.silu(conditioning)), 6, axis=-1
        )
        h = _modulate(_layernorm_tokens(self.self_norm, particles), shift_a, scale_a)
        particles = particles + gate_a[None, :] * self.self_attention(h, h, h)
        h = _modulate(_layernorm_tokens(self.ff_norm, particles), shift_f, scale_f)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return particles + gate_f[None, :] * h


class ThetaParticleOutputHead(eqx.Module):
    """Map [N,hidden] posterior tokens to a bounded displacement of the 2-D theta cloud."""

    final_norm: eqx.nn.LayerNorm
    displacement_head: eqx.nn.Linear
    max_displacement: float = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.max_displacement = cfg.max_theta_displacement
        self.final_norm = eqx.nn.LayerNorm(cfg.hidden_dim)
        head = eqx.nn.Linear(cfg.hidden_dim, K, key=key)
        head = eqx.tree_at(lambda l: l.weight, head, jnp.zeros_like(head.weight))
        head = eqx.tree_at(lambda l: l.bias, head, jnp.zeros_like(head.bias))
        self.displacement_head = head

    def __call__(self, particle_tokens: Array, current_theta: Array) -> Array:
        particle_tokens = _layernorm_tokens(self.final_norm, particle_tokens)
        displacement = self.max_displacement * jnp.tanh(_linear_tokens(self.displacement_head, particle_tokens))
        return current_theta + displacement


class AdaLNPosteriorTransformer(eqx.Module):
    """Direct reference-cloud -> posterior transport, AdaLN-conditioned on one M-D observation y."""

    particle_in: eqx.nn.Linear
    blocks: tuple[AdaLNParticleBlock, ...]
    output_head: ThetaParticleOutputHead

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        keys = jax.random.split(key, cfg.posterior_depth + 2)
        self.particle_in = eqx.nn.Linear(K, cfg.hidden_dim, key=keys[0])
        self.blocks = tuple(
            AdaLNParticleBlock(cfg.hidden_dim, observation_dim(cfg), cfg.heads, cfg.mlp_ratio * cfg.hidden_dim, key=keys[1 + i])
            for i in range(cfg.posterior_depth)
        )
        self.output_head = ThetaParticleOutputHead(cfg, key=keys[-1])

    def __call__(self, particle_embeddings: Array, current_theta: Array, observation: Array) -> Array:
        particles = _linear_tokens(self.particle_in, particle_embeddings)
        for block in self.blocks:
            particles = block(particles, observation)
        return self.output_head(particles, current_theta)


#%% 6) End-to-end amortized model with an evaluation-only sequential rollout
class SequentialBayesModel(eqx.Module):
    """One-step amortized transport for training; repeated application only during evaluation."""

    posterior_transformer: AdaLNPosteriorTransformer
    theta_scale: float = eqx.field(static=True)
    y_center: float = eqx.field(static=True)
    y_scale: float = eqx.field(static=True)

    def __init__(self, cfg: BayesTransportConfig, *, key: Array):
        self.posterior_transformer = AdaLNPosteriorTransformer(cfg, key=key)
        self.theta_scale = max(cfg.prior_std_x, cfg.prior_std_y, 1e-6)
        self.y_center = cfg.y_center
        self.y_scale = max(cfg.y_scale, 1e-6)

    def _normalize_particles(self, theta: Array) -> Array:
        center = base_prior_mean_jax(CFG)
        return (theta - center[None, :]) / self.theta_scale

    def _normalize_observation(self, observation: Array) -> Array:
        return (observation - self.y_center) / self.y_scale

    def predict(self, prior_particles: Array, observation: Array) -> Array:
        """Training/amortized path: [N,2] prior cloud + one [M] observation -> [N,2] posterior cloud."""
        embeddings = self._normalize_particles(prior_particles)
        obs = self._normalize_observation(observation)
        return self.posterior_transformer(embeddings, prior_particles, obs)

    def __call__(self, prior_particles: Array, observations: Array) -> Array:
        """Evaluation-only rollout: observations [T,M] -> posterior_sequence [T,N,2]."""

        def scan_step(current_particles: Array, observation: Array):
            embeddings = self._normalize_particles(current_particles)
            obs = self._normalize_observation(observation)
            next_particles = self.posterior_transformer(embeddings, current_particles, obs)
            return next_particles, next_particles

        _, sequence = jax.lax.scan(scan_step, prior_particles, observations)
        return sequence


def count_parameters(module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array)))


def print_model_parameter_count(model: SequentialBayesModel):
    total = count_parameters(model.posterior_transformer)
    print(f"Total parameters: {total / 1e6:.3f} M  (Posterior Transformer, AdaLN-conditioned)")


#%% 7) Empirical-cloud energy score and exact-posterior diagnostics
def empirical_energy_score_terms_single(particle_theta: Array, target_theta: Array) -> tuple[Array, Array, Array]:
    """Exact multivariate energy score of one transported empirical particle measure."""
    target_sq = jnp.sum((particle_theta - target_theta[None, :]) ** 2, axis=-1)
    attraction = jnp.mean(jnp.sqrt(target_sq + 1e-12))

    differences = particle_theta[:, None, :] - particle_theta[None, :, :]
    pair_sq = jnp.sum(differences**2, axis=-1)
    off_diagonal = 1.0 - jnp.eye(particle_theta.shape[0], dtype=particle_theta.dtype)
    repulsion = jnp.sum(jnp.sqrt(pair_sq + 1e-12) * off_diagonal) / (particle_theta.shape[0] ** 2)
    energy_score = attraction - 0.5 * repulsion
    return energy_score, attraction, repulsion


def energy_score_single(particle_theta: Array, target_theta: Array) -> Array:
    energy_score, _, _ = empirical_energy_score_terms_single(particle_theta, target_theta)
    return energy_score


def posterior_mean_rmse_single(particle_theta: Array, target_theta: Array) -> Array:
    squared_error = (jnp.mean(particle_theta, axis=0) - target_theta) ** 2
    return jnp.sqrt(jnp.mean(squared_error))


def posterior_spread_single(particle_theta: Array) -> Array:
    return jnp.mean(jnp.var(particle_theta, axis=0))


def particle_covariance_single(particle_theta: Array) -> Array:
    centered = particle_theta - jnp.mean(particle_theta, axis=0, keepdims=True)
    denominator = jnp.maximum(particle_theta.shape[0] - 1, 1)
    return (centered.T @ centered) / denominator


def exact_mean_rmse_single(particle_theta: Array, exact_mean: Array) -> Array:
    return jnp.sqrt(jnp.mean((jnp.mean(particle_theta, axis=0) - exact_mean) ** 2))


def exact_cov_frobenius_single(particle_theta: Array, exact_cov: Array) -> Array:
    learned_cov = particle_covariance_single(particle_theta)
    return jnp.linalg.norm(learned_cov - exact_cov, ord="fro")


def batch_objective(model: SequentialBayesModel, batch: dict[str, Array]) -> tuple[Array, dict[str, Array]]:
    """Non-sequential iid energy-score objective plus exact conjugate posterior diagnostics."""
    predicted = jax.vmap(model.predict)(batch["prior_particles"], batch["observation"])  # [B,N,2]
    energy, attraction, repulsion = jax.vmap(empirical_energy_score_terms_single)(predicted, batch["theta_true"])
    rmse = jax.vmap(posterior_mean_rmse_single)(predicted, batch["theta_true"])
    spread = jax.vmap(posterior_spread_single)(predicted)

    exact_mean, exact_cov = jax.vmap(gaussian_update_jax)(
        batch["prior_mean"], batch["prior_cov"], batch["observation"]
    )
    exact_mean_error = jax.vmap(exact_mean_rmse_single)(predicted, exact_mean)
    exact_cov_error = jax.vmap(exact_cov_frobenius_single)(predicted, exact_cov)

    loss = jnp.mean(energy)
    metrics = {
        "loss": loss,
        "energy_score": loss,
        "posterior_mean_rmse": jnp.mean(rmse),
        "posterior_spread": jnp.mean(spread),
        "attraction": jnp.mean(attraction),
        "repulsion": jnp.mean(repulsion),
        "exact_mean_rmse": jnp.mean(exact_mean_error),
        "exact_cov_frobenius": jnp.mean(exact_cov_error),
    }
    return loss, metrics


@eqx.filter_jit
def predict_batch(model: SequentialBayesModel, prior_particles: Array, observation: Array) -> Array:
    return jax.vmap(model.predict)(prior_particles, observation)


@eqx.filter_jit
def sequential_predict_batch(model: SequentialBayesModel, prior_particles: Array, observations: Array) -> Array:
    return jax.vmap(model)(prior_particles, observations)


@eqx.filter_jit
def amortized_evaluation_batch(model: SequentialBayesModel, batch: dict[str, Array]) -> dict[str, Array]:
    _, metrics = batch_objective(model, batch)
    return metrics


@eqx.filter_jit
def sequential_evaluation_batch(model: SequentialBayesModel, batch: dict[str, Array]) -> dict[str, Array]:
    """Evaluate repeated learned Bayes updates and compare every prefix with the exact Gaussian posterior."""
    predicted = sequential_predict_batch(model, batch["prior_particles"], batch["observations"])  # [B,T,N,2]
    target = batch["theta_true"]  # [B,2]

    energy = jax.vmap(lambda seq, truth: jax.vmap(lambda p: energy_score_single(p, truth))(seq))(predicted, target)
    rmse = jax.vmap(lambda seq, truth: jax.vmap(lambda p: posterior_mean_rmse_single(p, truth))(seq))(
        predicted, target
    )

    exact_means, exact_covs = jax.vmap(exact_base_posterior_sequence_jax)(batch["observations"])
    exact_mean_error = jax.vmap(
        lambda seq, means: jax.vmap(lambda p, m: exact_mean_rmse_single(p, m))(seq, means)
    )(predicted, exact_means)
    exact_cov_error = jax.vmap(
        lambda seq, covs: jax.vmap(lambda p, c: exact_cov_frobenius_single(p, c))(seq, covs)
    )(predicted, exact_covs)

    return {
        "energy_score": jnp.mean(energy),
        "final_energy_score": jnp.mean(energy[:, -1]),
        "posterior_mean_rmse": jnp.mean(rmse),
        "final_mean_rmse": jnp.mean(rmse[:, -1]),
        "energy_by_t": jnp.mean(energy, axis=0),
        "rmse_by_t": jnp.mean(rmse, axis=0),
        "exact_mean_rmse": jnp.mean(exact_mean_error),
        "final_exact_mean_rmse": jnp.mean(exact_mean_error[:, -1]),
        "exact_cov_frobenius": jnp.mean(exact_cov_error),
        "final_exact_cov_frobenius": jnp.mean(exact_cov_error[:, -1]),
        "exact_mean_rmse_by_t": jnp.mean(exact_mean_error, axis=0),
        "exact_cov_frobenius_by_t": jnp.mean(exact_cov_error, axis=0),
    }


#%% 8) Amortized validation and sequential evaluation with reproducible data
def evaluate_amortized_model(
    model: SequentialBayesModel,
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    *,
    batch_size: int | None = None,
) -> dict[str, float]:
    n_total = len(dataset["theta_true"])
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    names = [
        "loss",
        "energy_score",
        "posterior_mean_rmse",
        "posterior_spread",
        "attraction",
        "repulsion",
        "exact_mean_rmse",
        "exact_cov_frobenius",
    ]
    values = {name: [] for name in names}
    weights = []
    batch_fields = ("theta_true", "observation", "prior_particles", "prior_mean", "prior_cov")
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = {name: jnp.asarray(dataset[name][start:stop]) for name in batch_fields}
        metrics = jax.device_get(amortized_evaluation_batch(model, batch))
        weights.append(stop - start)
        for name in names:
            values[name].append(float(metrics[name]))
    weight_array = np.asarray(weights, dtype=np.float64)
    return {name: float(np.average(values[name], weights=weight_array)) for name in names}


def evaluate_sequential_model(
    model: SequentialBayesModel,
    dataset: dict[str, np.ndarray],
    cfg: BayesTransportConfig = CFG,
    *,
    batch_size: int | None = None,
) -> dict[str, np.ndarray | float]:
    n_total = len(dataset["theta_true"])
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    scalar_names = [
        "energy_score",
        "final_energy_score",
        "posterior_mean_rmse",
        "final_mean_rmse",
        "exact_mean_rmse",
        "final_exact_mean_rmse",
        "exact_cov_frobenius",
        "final_exact_cov_frobenius",
    ]
    by_t_names = ["energy_by_t", "rmse_by_t", "exact_mean_rmse_by_t", "exact_cov_frobenius_by_t"]
    scalar_values = {name: [] for name in scalar_names}
    by_t_values = {name: [] for name in by_t_names}
    weights = []
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = {
            "theta_true": jnp.asarray(dataset["theta_true"][start:stop]),
            "observations": jnp.asarray(dataset["observations"][start:stop]),
            "prior_particles": jnp.asarray(dataset["prior_particles"][start:stop]),
        }
        metrics = jax.device_get(sequential_evaluation_batch(model, batch))
        weights.append(stop - start)
        for name in scalar_names:
            scalar_values[name].append(float(metrics[name]))
        for name in by_t_names:
            by_t_values[name].append(np.asarray(metrics[name], dtype=np.float64))
    weight_array = np.asarray(weights, dtype=np.float64)
    result: dict[str, np.ndarray | float] = {
        name: float(np.average(scalar_values[name], weights=weight_array)) for name in scalar_names
    }
    for name in by_t_names:
        result[name] = np.average(np.stack(by_t_values[name]), axis=0, weights=weight_array)
    return result


#%% 9) Exact Gaussian reference helpers (replaces SNIS)
def sample_exact_posterior_np(
    rng: np.random.Generator, observations: np.ndarray, n: int, cfg: BayesTransportConfig = CFG
) -> np.ndarray:
    """Optional exact posterior particle sample; mainly useful for additional diagnostics."""
    mean, cov = exact_base_posterior_np(observations, cfg)
    return sample_gaussian_np(rng, mean, cov, int(n))


#%% 10) Visualisation: dataset intuition + sequential posterior evolution with exact Gaussian overlays
def select_steps(trajectory_length: int, n_panels_after_prior: int = 5) -> list[int]:
    values = np.unique(np.rint(np.geomspace(1, trajectory_length, n_panels_after_prior)).astype(int))
    if values[-1] != trajectory_length:
        values = np.append(values, trajectory_length)
    while len(values) > n_panels_after_prior:
        values = np.delete(values, 1)
    return values.tolist()


def add_gaussian_ellipse(
    ax,
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float,
    *,
    color: str,
    linewidth: float = 1.8,
    linestyle: str = "-",
    alpha: float = 1.0,
    label: str | None = None,
):
    """Add an n-standard-deviation covariance ellipse for a 2-D Gaussian."""
    mean = np.asarray(mean, dtype=float)
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 0.0, None)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(evals)
    ellipse = Ellipse(
        xy=mean,
        width=max(width, 1e-10),
        height=max(height, 1e-10),
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
        zorder=4,
    )
    ax.add_patch(ellipse)
    return ellipse


def _theta_plot_limits(
    prior_particles: np.ndarray,
    theta_true: np.ndarray,
    extra_clouds: list[np.ndarray],
    exact_means: np.ndarray,
    cfg: BayesTransportConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Common theta-space limits that respect a nonzero/nonisotropic prior."""
    points = [np.asarray(prior_particles).reshape(-1, 2), np.asarray(theta_true).reshape(-1, 2)]
    points.extend(np.asarray(c).reshape(-1, 2) for c in extra_clouds)
    points.append(np.asarray(exact_means).reshape(-1, 2))
    all_points = np.concatenate(points, axis=0)
    mu0 = base_prior_mean_np(cfg)
    marginal_std = np.sqrt(np.diag(base_prior_cov_np(cfg)))
    lower = np.minimum(np.min(all_points, axis=0), mu0 - 3.4 * marginal_std)
    upper = np.maximum(np.max(all_points, axis=0), mu0 + 3.4 * marginal_std)
    span = np.maximum(upper - lower, 1e-6)
    lower -= 0.08 * span
    upper += 0.08 * span
    return (float(lower[0]), float(upper[0])), (float(lower[1]), float(upper[1]))


def show_dataset_example(
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
):
    """Print and plot the exact fixed example that is reused later during training diagnostics.

    The right panel lives in observation space and therefore remains meaningful even when
    dim(y) != dim(theta).
    """
    theta_true = np.asarray(trajectory["theta_true"], dtype=float)
    observations = np.asarray(trajectory["observations"], dtype=float)
    A = observation_matrix_np(cfg)
    obs_dim = observation_dim(cfg)
    noiseless_y = observation_mean_np(theta_true, cfg).reshape(-1)
    exact_final_mean, exact_final_cov = exact_base_posterior_np(observations, cfg)

    print("\nFixed example used for periodic sequential plots")
    print("------------------------------------------------")
    print(f"observation_operator = {cfg.observation_operator!r}")
    print(f"dim(theta) = {K}; dim(y) = {obs_dim}")
    print("A =\n", A)
    print("theta* =", np.array2string(theta_true, precision=4))
    print("A theta* =", np.array2string(noiseless_y, precision=4))
    print(f"ys shape = {observations.shape}")
    preview_count = min(8, len(observations))
    print(f"first {preview_count} ys =\n", np.array2string(observations[:preview_count], precision=4))
    print("exact posterior mean after all ys =", np.array2string(exact_final_mean, precision=4))
    print("exact posterior covariance after all ys =\n", np.array2string(exact_final_cov, precision=4))

    learned_blue = "#4C78A8"
    truth_color = "#111111"
    obs_color = "#2A6F97"
    mean_color = "#8B0000"
    sigma1_color = "#D95F02"
    sigma2_color = "#6F4C9B"

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)

    # Left: latent theta-space geometry.
    ax = axes[0]
    prior_particles = np.asarray(prior_particles)
    ax.scatter(
        prior_particles[:, 0], prior_particles[:, 1], s=18, alpha=0.28,
        color=learned_blue, edgecolors="none", label=r"prior particles $\theta^{(n)}$",
    )
    add_gaussian_ellipse(
        ax, base_prior_mean_np(cfg), base_prior_cov_np(cfg), 1.0,
        color=sigma1_color, linewidth=2.2, label=r"prior $1\sigma$",
    )
    add_gaussian_ellipse(
        ax, base_prior_mean_np(cfg), base_prior_cov_np(cfg), 2.0,
        color=sigma2_color, linewidth=1.8, linestyle="--", label=r"prior $2\sigma$",
    )
    ax.scatter(
        theta_true[0], theta_true[1], marker="*", s=220, color=truth_color,
        edgecolors="white", linewidths=0.7, zorder=7, label=r"$\theta^\star$",
    )
    xlim, ylim = _theta_plot_limits(prior_particles, theta_true, [], base_prior_mean_np(cfg)[None, :], cfg)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
    ax.set_title(r"Latent space: prior and sampled $\theta^\star$", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="best")

    # Right: observation-space trajectory.  Plot against t so arbitrary M is supported.
    ax = axes[1]
    t_axis = np.arange(1, len(observations) + 1)
    for j in range(obs_dim):
        component_label = r"$y_{t,%d}$" % (j + 1) if obs_dim > 1 else r"$y_t$"
        mean_label = r"$(A\theta^\star)_{%d}$" % (j + 1) if obs_dim > 1 else r"$A\theta^\star$"
        if obs_dim == 1:
            line_color = obs_color
            mean_line_color = mean_color
        else:
            line_color = None
            mean_line_color = None
        line = ax.plot(
            t_axis, observations[:, j], marker="o", markersize=4.2, linewidth=1.35,
            alpha=0.82, color=line_color, label=component_label,
        )[0]
        ax.axhline(
            noiseless_y[j], linewidth=2.0, linestyle="--", alpha=0.88,
            color=mean_line_color if mean_line_color is not None else line.get_color(),
            label=mean_label if j == 0 or obs_dim <= 3 else None,
        )
    ax.set_xlabel(r"observation index $t$")
    ax.set_ylabel(r"observed value $y$")
    ax.set_title(r"Observation space: $y_t=A\theta^\star+\varepsilon_t$", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=1 if obs_dim <= 2 else 2)

    for ax in axes:
        ax.grid(alpha=0.18)
    fig.suptitle(
        rf"Fixed conjugate example before training: $A\in\mathbb{{R}}^{{{obs_dim}\times 2}}$",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    display(fig)
    plt.close(fig)


def plot_posterior_evolution(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
    title: str = "Posterior evolution",
):
    """Plot learned repeated-Bayes clouds and the exact Gaussian posterior at each prefix."""
    observations = np.asarray(trajectory["observations"])
    posterior_sequence = np.asarray(
        jax.device_get(
            sequential_predict_batch(model, jnp.asarray(prior_particles)[None], jnp.asarray(observations)[None])[0]
        )
    )  # [T,N,2]
    theta_true = np.asarray(trajectory["theta_true"])
    exact_means, exact_covs = exact_base_posterior_sequence_np(observations, cfg)

    steps = select_steps(len(observations), 5)
    clouds = [np.asarray(prior_particles)] + [posterior_sequence[t - 1] for t in steps]
    exact_plot_means = [base_prior_mean_np(cfg)] + [exact_means[t - 1] for t in steps]
    exact_plot_covs = [base_prior_cov_np(cfg)] + [exact_covs[t - 1] for t in steps]
    labels = [r"base prior $p(\theta)$"] + [rf"$q_\phi(\theta\mid y_{{1:{t}}})$" for t in steps]

    learned_blue = "#4C78A8"
    prior_blue = "#72A0C1"
    sigma1_color = "#D95F02"   # burnt orange: clearly visible on white
    sigma2_color = "#6F4C9B"   # deep purple: distinct from 1 sigma and particles
    exact_mean_color = "#8B0000"  # dark red, intentionally far from observation/particle colors
    truth_color = "#111111"

    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.6), constrained_layout=True)
    axes = axes.ravel()
    xlim, ylim = _theta_plot_limits(
        prior_particles, theta_true, [posterior_sequence[t - 1] for t in steps], np.asarray(exact_plot_means), cfg
    )

    for panel_index, (ax, cloud, exact_mean, exact_cov, label) in enumerate(
        zip(axes, clouds, exact_plot_means, exact_plot_covs, labels)
    ):
        particle_color = prior_blue if panel_index == 0 else learned_blue
        particle_label = r"prior particles" if panel_index == 0 else r"learned particles"
        ax.scatter(
            cloud[:, 0], cloud[:, 1], s=18, alpha=0.30, color=particle_color,
            edgecolors="none", label=particle_label, zorder=2,
        )
        add_gaussian_ellipse(
            ax, exact_mean, exact_cov, 1.0, color=sigma1_color, linewidth=2.4,
            label=r"exact Gaussian $1\sigma$",
        )
        add_gaussian_ellipse(
            ax, exact_mean, exact_cov, 2.0, color=sigma2_color, linewidth=1.9,
            linestyle="--", label=r"exact Gaussian $2\sigma$",
        )
        ax.scatter(
            exact_mean[0], exact_mean[1], marker="X", s=88, color=exact_mean_color,
            edgecolors="white", linewidths=0.7, zorder=6, label=r"exact posterior mean $\mu_t$",
        )
        ax.scatter(
            theta_true[0], theta_true[1], marker="*", s=205, color=truth_color,
            edgecolors="white", linewidths=0.7, zorder=7, label=r"$\theta^\star$",
        )
        if panel_index > 0:
            t = steps[panel_index - 1]
            ybar = observations[:t].mean(axis=0)
            summary = np.array2string(ybar, precision=2, separator=", ")
            ax.text(
                0.03, 0.03, rf"$t={t}$\n$\bar{{y}}_{{1:t}}={summary}$",
                transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.82", alpha=0.88),
            )
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.set_title(label, fontsize=11.5)
        ax.legend(fontsize=7.1, loc="upper right", framealpha=0.92)

    fig.suptitle(
        rf"{title}: learned transport vs exact Gaussian $p(\theta\mid y_{{1:t}})$",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    display(fig)
    plt.close(fig)


def plot_sequential_reference_check(
    model: SequentialBayesModel,
    trajectory: dict[str, np.ndarray],
    prior_particles: np.ndarray,
    cfg: BayesTransportConfig = CFG,
    destination: Path | None = None,
):
    """Soundness check against the exact analytic posterior (no Monte Carlo reference needed)."""
    observations = np.asarray(trajectory["observations"])
    theta_true = np.asarray(trajectory["theta_true"])
    posterior_sequence = np.asarray(
        jax.device_get(
            sequential_predict_batch(model, jnp.asarray(prior_particles)[None], jnp.asarray(observations)[None])[0]
        )
    )  # [T,N,2]
    exact_means, exact_covs = exact_base_posterior_sequence_np(observations, cfg)

    learned_mean_rmse_to_truth = []
    exact_mean_rmse_to_truth = []
    learned_mean_rmse_to_exact = []
    covariance_frobenius = []
    for t in range(len(observations)):
        learned = posterior_sequence[t]
        learned_mean = learned.mean(axis=0)
        learned_cov = np.cov(learned, rowvar=False, ddof=1)
        learned_mean_rmse_to_truth.append(float(np.sqrt(np.mean((learned_mean - theta_true) ** 2))))
        exact_mean_rmse_to_truth.append(float(np.sqrt(np.mean((exact_means[t] - theta_true) ** 2))))
        learned_mean_rmse_to_exact.append(float(np.sqrt(np.mean((learned_mean - exact_means[t]) ** 2))))
        covariance_frobenius.append(float(np.linalg.norm(learned_cov - exact_covs[t], ord="fro")))

    t_axis = np.arange(1, len(observations) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)
    axes[0].plot(
        t_axis, np.maximum(learned_mean_rmse_to_truth, 1e-12), marker="o",
        label=r"learned mean vs $\theta^\star$",
    )
    axes[0].plot(
        t_axis, np.maximum(exact_mean_rmse_to_truth, 1e-12), marker="o",
        label=r"exact $\mu_t$ vs $\theta^\star$",
    )
    axes[0].plot(
        t_axis, np.maximum(learned_mean_rmse_to_exact, 1e-12), marker="o",
        label=r"learned mean vs exact $\mu_t$",
    )
    axes[0].set_title(r"Posterior-mean error", loc="left", fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        t_axis, np.maximum(covariance_frobenius, 1e-12), marker="o",
        label=r"$\|\widehat{\Sigma}_{\phi,t}-\Sigma_t\|_F$",
    )
    axes[1].set_title(r"Learned-vs-exact covariance error", loc="left", fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel(r"sequential Bayes step $t$")
        ax.grid(alpha=0.18)
    fig.suptitle(r"Sequential soundness check against exact linear-Gaussian Bayes", fontsize=13.5, fontweight="bold")
    if destination is not None:
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    display(fig)
    plt.close(fig)


#%% 11) Visualisation: training diagnostics
def _format_train_step_ticks(ax):
    """Use compact k notation for the thousands of train steps."""
    from matplotlib.ticker import FuncFormatter

    def formatter(value, _position):
        if abs(value) >= 1000:
            scaled = value / 1000.0
            return f"{scaled:g}k"
        if abs(value - round(value)) < 1e-8:
            return f"{int(round(value))}"
        return f"{value:g}"

    ax.xaxis.set_major_formatter(FuncFormatter(formatter))


def plot_training_diagnostics(
    history: dict[str, list], best_epoch: int, destination: Path | None = None, cfg: BayesTransportConfig = CFG
):
    train_steps = np.arange(1, len(history["step_loss"]) + 1)
    epochs = np.arange(1, len(history["epoch_train_loss"]) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(20.2, 9.2), constrained_layout=True)

    # Raw ES and its moving average intentionally use the SAME hue; smoothing is communicated
    # only by opacity and line weight, so the average visually reads as the same quantity.
    es_color = "#4C78A8"
    values = np.maximum(np.asarray(history["step_loss"], dtype=float), 1e-12)
    axes[0, 0].plot(
        train_steps, values, color=es_color, linewidth=0.70, alpha=0.24,
        label=r"empirical energy score",
    )
    if len(values) >= 20:
        window = max(5, len(values) // 100)
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(
            train_steps[window - 1 :], smoothed, color=es_color, linewidth=2.35, alpha=0.98,
            label=rf"moving average ({window} train steps)",
        )
    axes[0, 0].set_title(r"Training objective at every train step", loc="left", fontweight="bold")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("train step")
    axes[0, 0].legend(fontsize=8)
    _format_train_step_ticks(axes[0, 0])

    axes[0, 1].plot(train_steps, history["step_attraction"], linewidth=0.78, label=r"target attraction")
    axes[0, 1].plot(train_steps, history["step_repulsion"], linewidth=0.78, label=r"cloud pairwise spread")
    axes[0, 1].set_title(r"Energy-score decomposition", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("train step")
    axes[0, 1].legend(fontsize=8)
    _format_train_step_ticks(axes[0, 1])

    axes[0, 2].plot(train_steps, history["step_grad_norm"], linewidth=0.78)
    axes[0, 2].set_title(r"Gradient norm $\|\nabla_\phi\mathcal{L}\|_2$", loc="left", fontweight="bold")
    axes[0, 2].set_xlabel("train step")
    _format_train_step_ticks(axes[0, 2])

    axes[0, 3].plot(
        train_steps, np.maximum(history["step_exact_mean_rmse"], 1e-12), linewidth=0.82,
        label=r"RMSE$(\widehat\mu_\phi,\mu_{\rm exact})$",
    )
    axes[0, 3].plot(
        train_steps, np.maximum(history["step_exact_cov_frobenius"], 1e-12), linewidth=0.82,
        label=r"$\|\widehat\Sigma_\phi-\Sigma_{\rm exact}\|_F$",
    )
    axes[0, 3].set_title(r"Exact-posterior error at every train step", loc="left", fontweight="bold")
    axes[0, 3].set_xlabel("train step")
    axes[0, 3].set_yscale("log")
    axes[0, 3].legend(fontsize=8)
    _format_train_step_ticks(axes[0, 3])

    axes[1, 0].plot(epochs, np.maximum(history["epoch_train_loss"], 1e-12), marker="o", markersize=3, label=r"train ES")
    axes[1, 0].plot(epochs, np.maximum(history["epoch_val_loss"], 1e-12), marker="o", markersize=3, label=r"iid val ES")
    axes[1, 0].axvline(best_epoch, linestyle="--", linewidth=1.1, label=rf"best epoch ${best_epoch}$")
    lr_axis = axes[1, 0].twinx()
    lr_axis.plot(epochs, np.asarray(history["epoch_learning_rate"], dtype=float), linestyle=":", linewidth=1.5, alpha=0.78)
    lr_axis.set_yscale("log")
    lr_axis.set_ylabel(r"learning rate $\eta$")
    axes[1, 0].set_title(r"Model-selection objective", loc="left", fontweight="bold")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend(fontsize=8)

    seq_energy = np.asarray(history["epoch_val_energy_by_t"], dtype=float)
    seq_rmse = np.asarray(history["epoch_val_rmse_by_t"], dtype=float)
    seq_exact_mean = np.asarray(history["epoch_val_exact_mean_rmse_by_t"], dtype=float)
    seq_exact_cov = np.asarray(history["epoch_val_exact_cov_frobenius_by_t"], dtype=float)
    t_axis = np.arange(1, seq_energy.shape[1] + 1)
    selected_epochs = np.unique(
        np.clip(np.rint(np.linspace(0, len(seq_energy) - 1, 5)).astype(int), 0, len(seq_energy) - 1)
    )
    for epoch_index in selected_epochs:
        axes[1, 1].plot(t_axis, np.maximum(seq_energy[epoch_index], 1e-12), label=rf"epoch ${epoch_index + 1}$")
        axes[1, 2].plot(t_axis, np.maximum(seq_rmse[epoch_index], 1e-12), label=rf"epoch ${epoch_index + 1}$")
        axes[1, 3].plot(
            t_axis, np.maximum(seq_exact_mean[epoch_index], 1e-12),
            label=rf"$\mu$ error, epoch ${epoch_index + 1}$",
        )
        axes[1, 3].plot(
            t_axis, np.maximum(seq_exact_cov[epoch_index], 1e-12), linestyle="--",
            label=rf"$\Sigma$ error, epoch ${epoch_index + 1}$",
        )
    axes[1, 1].set_title(r"Repeated-Bayes energy score", loc="left", fontweight="bold")
    axes[1, 2].set_title(r"Repeated-Bayes RMSE to $\theta^\star$", loc="left", fontweight="bold")
    axes[1, 3].set_title(r"Repeated-Bayes error to exact posterior", loc="left", fontweight="bold")
    for ax in axes[1, 1:]:
        ax.set_xlabel(r"evaluation-only sequential step $t$")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.18)

    fig.suptitle(
        r"2-D conjugate linear-Gaussian transport: training and exact-posterior diagnostics",
        fontsize=14, fontweight="bold",
    )
    if destination is not None:
        fig.savefig(destination, dpi=180, bbox_inches="tight")
    display(fig)
    plt.close(fig)


#%% 12) Training function
def train_model(
    train_loader: DataLoader,
    amortized_eval_data: dict[str, np.ndarray],
    sequential_eval_data: dict[str, np.ndarray],
    fixed_trajectory: dict[str, np.ndarray],
    fixed_prior_particles: np.ndarray,
    run_dir: Path,
    cfg: BayesTransportConfig = CFG,
) -> dict[str, Any]:
    model = SequentialBayesModel(cfg, key=jax.random.key(cfg.seed))
    print("\namortized single-cloud energy transport: 2-D Gaussian conjugate sanity check (AdaLN)")
    print_model_parameter_count(model)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay),
    )
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    plateau = optax.contrib.reduce_on_plateau(
        factor=0.5,
        patience=cfg.lr_plateau_patience,
        rtol=cfg.lr_plateau_rtol,
        cooldown=0,
        accumulation_size=1,
        min_scale=0.01,
    )
    plateau_state = plateau.init(params)

    @eqx.filter_jit
    def train_step(candidate_model, candidate_opt_state, learning_rate_scale, batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(batch_objective, has_aux=True)(candidate_model, batch)
        params = eqx.filter(candidate_model, eqx.is_array)
        updates, candidate_opt_state = optimizer.update(grads, candidate_opt_state, params)
        updates = jax.tree_util.tree_map(lambda u: learning_rate_scale * u, updates)
        candidate_model = eqx.apply_updates(candidate_model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_array))
        return candidate_model, candidate_opt_state, loss, metrics, grad_norm

    history: dict[str, list] = {
        "step_loss": [],
        "step_attraction": [],
        "step_repulsion": [],
        "step_grad_norm": [],
        "step_exact_mean_rmse": [],
        "step_exact_cov_frobenius": [],
        "epoch_train_loss": [],
        "epoch_learning_rate": [],
        "epoch_val_loss": [],
        "epoch_val_mean_rmse": [],
        "epoch_val_exact_mean_rmse": [],
        "epoch_val_exact_cov_frobenius": [],
        "epoch_seq_final_energy_score": [],
        "epoch_seq_final_mean_rmse": [],
        "epoch_seq_final_exact_mean_rmse": [],
        "epoch_seq_final_exact_cov_frobenius": [],
        "epoch_val_energy_by_t": [],
        "epoch_val_rmse_by_t": [],
        "epoch_val_exact_mean_rmse_by_t": [],
        "epoch_val_exact_cov_frobenius_by_t": [],
    }

    plot_posterior_evolution(
        model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        run_dir / "plots" / "fixed_trajectory_before_training.png",
        "evaluation-only repeated Bayes: before training",
    )

    visualisation_epochs = sorted(set(max(1, int(math.ceil(f * cfg.epochs / 10.0))) for f in range(1, 11)))
    best_val_loss = float("inf")
    best_epoch = 0
    n_steps = cfg.n_train_trajectories // cfg.batch_size
    if n_steps < 1:
        raise ValueError("n_train_trajectories must be at least one batch_size.")
    train_iterator = iter(train_loader)
    training_started_at = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_started_at = time.time()
        epoch_lr_scale = plateau_state.scale
        epoch_learning_rate = cfg.learning_rate * float(jax.device_get(epoch_lr_scale))
        train_losses_this_epoch: list[float] = []
        progress = tqdm(
            range(n_steps),
            desc=f"amortized epoch {epoch:04d}/{cfg.epochs:04d}",
            dynamic_ncols=True,
            leave=True,
            mininterval=5.0,
        )

        for _ in progress:
            batch_np = next(train_iterator)
            batch = {name: jnp.asarray(value) for name, value in batch_np.items()}
            model, opt_state, loss, metrics, grad_norm = train_step(model, opt_state, epoch_lr_scale, batch)
            host = jax.device_get(metrics)
            host_loss = float(jax.device_get(loss))
            train_losses_this_epoch.append(host_loss)
            history["step_loss"].append(host_loss)
            history["step_attraction"].append(float(host["attraction"]))
            history["step_repulsion"].append(float(host["repulsion"]))
            history["step_grad_norm"].append(float(jax.device_get(grad_norm)))
            history["step_exact_mean_rmse"].append(float(host["exact_mean_rmse"]))
            history["step_exact_cov_frobenius"].append(float(host["exact_cov_frobenius"]))
            progress.set_postfix(
                ES=f"{host_loss:.4f}",
                exact_mu=f"{float(host['exact_mean_rmse']):.4f}",
                exact_cov=f"{float(host['exact_cov_frobenius']):.4f}",
                refresh=False,
            )

        epoch_train_loss = float(np.mean(train_losses_this_epoch))
        val_metrics = evaluate_amortized_model(model, amortized_eval_data, cfg)
        seq_metrics = evaluate_sequential_model(model, sequential_eval_data, cfg)

        _, plateau_state = plateau.update(
            updates=eqx.filter(model, eqx.is_array),
            state=plateau_state,
            value=jnp.asarray(val_metrics["loss"], dtype=jnp.float32),
        )
        next_learning_rate = cfg.learning_rate * float(jax.device_get(plateau_state.scale))

        history["epoch_train_loss"].append(epoch_train_loss)
        history["epoch_learning_rate"].append(epoch_learning_rate)
        history["epoch_val_loss"].append(float(val_metrics["loss"]))
        history["epoch_val_mean_rmse"].append(float(val_metrics["posterior_mean_rmse"]))
        history["epoch_val_exact_mean_rmse"].append(float(val_metrics["exact_mean_rmse"]))
        history["epoch_val_exact_cov_frobenius"].append(float(val_metrics["exact_cov_frobenius"]))
        history["epoch_seq_final_energy_score"].append(float(seq_metrics["final_energy_score"]))
        history["epoch_seq_final_mean_rmse"].append(float(seq_metrics["final_mean_rmse"]))
        history["epoch_seq_final_exact_mean_rmse"].append(float(seq_metrics["final_exact_mean_rmse"]))
        history["epoch_seq_final_exact_cov_frobenius"].append(float(seq_metrics["final_exact_cov_frobenius"]))
        history["epoch_val_energy_by_t"].append(np.asarray(seq_metrics["energy_by_t"], dtype=np.float64))
        history["epoch_val_rmse_by_t"].append(np.asarray(seq_metrics["rmse_by_t"], dtype=np.float64))
        history["epoch_val_exact_mean_rmse_by_t"].append(
            np.asarray(seq_metrics["exact_mean_rmse_by_t"], dtype=np.float64)
        )
        history["epoch_val_exact_cov_frobenius_by_t"].append(
            np.asarray(seq_metrics["exact_cov_frobenius_by_t"], dtype=np.float64)
        )

        save_model(run_dir / "artefacts" / "model_last.eqx", model)
        if epoch % cfg.save_every_epochs == 0:
            save_model(run_dir / "artefacts" / f"model_epoch_{epoch:04d}.eqx", model)
        if float(val_metrics["loss"]) < best_val_loss:
            best_val_loss, best_epoch = float(val_metrics["loss"]), epoch
            save_model(run_dir / "artefacts" / "model_best.eqx", model)

        np.savez_compressed(run_dir / "artefacts" / "history.npz", **{n: np.asarray(v) for n, v in history.items()})
        print(
            f"[amortized] epoch {epoch:04d}: train ES={epoch_train_loss:.6f} | "
            f"val ES={float(val_metrics['loss']):.6f} | val RMSE(theta*)={float(val_metrics['posterior_mean_rmse']):.5f} | "
            f"val exact-mean RMSE={float(val_metrics['exact_mean_rmse']):.5f} | "
            f"val exact-cov F={float(val_metrics['exact_cov_frobenius']):.5f} | "
            f"lr={epoch_learning_rate:.3e} -> {next_learning_rate:.3e} || "
            f"seq final ES={float(seq_metrics['final_energy_score']):.6f} | "
            f"seq final RMSE(theta*)={float(seq_metrics['final_mean_rmse']):.5f} | "
            f"seq exact-mean RMSE={float(seq_metrics['final_exact_mean_rmse']):.5f} | "
            f"seq exact-cov F={float(seq_metrics['final_exact_cov_frobenius']):.5f} | "
            f"{time.time() - epoch_started_at:.1f}s"
        )

        if epoch in visualisation_epochs:
            plot_posterior_evolution(
                model,
                fixed_trajectory,
                fixed_prior_particles,
                cfg,
                run_dir / "plots" / f"fixed_trajectory_epoch_{epoch:04d}.png",
                f"evaluation-only repeated Bayes after amortized epoch {epoch}",
            )

    best_model = load_model(run_dir / "artefacts" / "model_best.eqx", cfg)
    final_amortized_metrics = evaluate_amortized_model(best_model, amortized_eval_data, cfg)
    final_metrics = evaluate_sequential_model(best_model, sequential_eval_data, cfg)
    plot_posterior_evolution(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        run_dir / "plots" / "fixed_trajectory_best_model.png",
        f"best model (epoch {best_epoch}): evaluation-only repeated Bayes",
    )
    plot_sequential_reference_check(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        cfg,
        run_dir / "plots" / "sequential_exact_reference_check.png",
    )
    plot_training_diagnostics(history, best_epoch, run_dir / "plots" / "training_diagnostics.png", cfg)

    elapsed = int(time.time() - training_started_at)
    print(
        f"[amortized] training complete in {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}; "
        f"best epoch={best_epoch}, iid val ES={best_val_loss:.6f}"
    )
    return {
        "model": best_model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "amortized_final_metrics": final_amortized_metrics,
        "final_metrics": final_metrics,
    }


#%% 13) Build run, data, and either train a fresh model or reload the best checkpoint
# Notebook-style execution is intentional: there is no main() and no __name__ guard.
np.random.seed(CFG.seed)
print("JAX devices:", jax.devices())
print("Configuration:\n", yaml.safe_dump(asdict(CFG), sort_keys=False))
print("Closed-form base prior mean:", base_prior_mean_np(CFG))
print("Closed-form base prior covariance:\n", base_prior_cov_np(CFG))
print("Observation matrix A:\n", observation_matrix_np(CFG))
print(f"Observation dimension M: {OBS_DIM}")
print("Likelihood covariance R:\n", likelihood_cov_np(CFG))

if train_wm:
    run_dir = make_run_dir(CFG.env_name, CFG.runs_base)
    archived_script = copy_running_script_to_run_dir(run_dir)
    print("Run directory:", run_dir)
    if archived_script is not None:
        print("Archived training script:", archived_script)

    train_loader = make_continuous_train_loader(CFG, seed=CFG.seed + 1_000)
    amortized_eval_data = simulate_iid_joint_samples(
        np.random.default_rng(CFG.seed + 2_000), CFG.n_eval_trajectories, CFG
    )
    sequential_eval_data = simulate_trajectories(
        np.random.default_rng(CFG.seed + 2_100),
        CFG.n_eval_trajectories,
        CFG.evaluation_trajectory_length,
        CFG,
    )

    # Sequential eval starts each trajectory from a fresh iid base-prior particle cloud.
    prior_rng = np.random.default_rng(CFG.seed + 2_200)
    sequential_eval_data["prior_particles"] = np.stack(
        [sample_base_prior_np(prior_rng, CFG.num_particles, CFG) for _ in range(len(sequential_eval_data["theta_true"]))]
    )

    # One fixed problem kept for periodic-during-training and final exact-posterior diagnostic plots.
    fixed_data = simulate_trajectories(
        np.random.default_rng(CFG.seed + 2_500), 1, CFG.evaluation_trajectory_length, CFG
    )
    fixed_trajectory = {
        "theta_true": fixed_data["theta_true"][0],
        "observations": fixed_data["observations"][0],
    }
    fixed_prior_particles = sample_base_prior_np(
        np.random.default_rng(CFG.seed + 3_000), CFG.num_particles, CFG
    )
    exact_means_fixed, exact_covs_fixed = exact_base_posterior_sequence_np(fixed_trajectory["observations"], CFG)
    np.savez_compressed(
        run_dir / "artefacts" / "fixed_trajectory.npz",
        theta_true=fixed_trajectory["theta_true"],
        observations=fixed_trajectory["observations"],
        prior_particles=fixed_prior_particles,
        exact_posterior_means=exact_means_fixed,
        exact_posterior_covariances=exact_covs_fixed,
    )

    # Before any optimisation, inspect the exact fixed example that all periodic plots will reuse.
    show_dataset_example(
        fixed_trajectory,
        fixed_prior_particles,
        CFG,
        run_dir / "plots" / "fixed_dataset_example_before_training.png",
    )

    result = train_model(
        train_loader,
        amortized_eval_data,
        sequential_eval_data,
        fixed_trajectory,
        fixed_prior_particles,
        run_dir,
        CFG,
    )
else:
    run_dir = Path.cwd().expanduser().resolve()
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=True)
    print("Existing run directory:", run_dir)

    amortized_eval_data = simulate_iid_joint_samples(
        np.random.default_rng(CFG.seed + 2_000), CFG.n_eval_trajectories, CFG
    )
    sequential_eval_data = simulate_trajectories(
        np.random.default_rng(CFG.seed + 2_100),
        CFG.n_eval_trajectories,
        CFG.evaluation_trajectory_length,
        CFG,
    )
    prior_rng = np.random.default_rng(CFG.seed + 2_200)
    sequential_eval_data["prior_particles"] = np.stack(
        [sample_base_prior_np(prior_rng, CFG.num_particles, CFG) for _ in range(len(sequential_eval_data["theta_true"]))]
    )

    fixed = np.load(run_dir / "artefacts" / "fixed_trajectory.npz")
    fixed_trajectory = {"theta_true": fixed["theta_true"], "observations": fixed["observations"]}
    fixed_prior_particles = fixed["prior_particles"]
    show_dataset_example(
        fixed_trajectory,
        fixed_prior_particles,
        CFG,
        run_dir / "plots" / "fixed_dataset_example_reloaded.png",
    )

    best_model = load_model(run_dir / "artefacts" / "model_best.eqx", CFG)
    history_path = run_dir / "artefacts" / "history.npz"
    history = dict(np.load(history_path)) if history_path.is_file() else None

    final_amortized_metrics = evaluate_amortized_model(best_model, amortized_eval_data, CFG)
    final_metrics = evaluate_sequential_model(best_model, sequential_eval_data, CFG)
    print_model_parameter_count(best_model)
    if history is not None and len(history["epoch_val_loss"]):
        best_epoch = int(np.argmin(history["epoch_val_loss"])) + 1
        plot_training_diagnostics(history, best_epoch, run_dir / "plots" / "training_diagnostics.png", CFG)
    plot_posterior_evolution(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        CFG,
        run_dir / "plots" / "fixed_trajectory_reloaded.png",
        "reloaded best model: evaluation-only repeated Bayes",
    )
    plot_sequential_reference_check(
        best_model,
        fixed_trajectory,
        fixed_prior_particles,
        CFG,
        run_dir / "plots" / "sequential_exact_reference_check.png",
    )
    result = {
        "model": best_model,
        "history": history,
        "amortized_final_metrics": final_amortized_metrics,
        "final_metrics": final_metrics,
    }

model = result["model"]

summary = {
    "objective": "exact empirical multivariate energy score in 2-D theta space",
    "training_mode": "non-sequential amortized transport from synthetic interpolated Gaussian input clouds",
    "problem_shape": {"theta_dim": K, "observation_dim": OBS_DIM},
    "prior_distribution": "2-D Gaussian",
    "likelihood": "y | theta ~ N(A theta, sigma_y^2 I_M)",
    "observation_operator": CFG.observation_operator,
    "observation_matrix": observation_matrix_np(CFG).tolist(),
    "observations_per_training_step": 1,
    "posterior_conditioning": "adaln",
    "synthetic_truth_sampling_mode": "exact conditional Gaussian interpolation law",
    "sequential_evaluation": True,
    "sequential_reference": "closed-form Gaussian posterior",
    "closed_form": {
        "posterior_covariance": "Sigma_t = (Sigma_0^{-1} + t A^T R^{-1} A)^{-1}",
        "posterior_mean": "mu_t = Sigma_t (Sigma_0^{-1} mu_0 + A^T R^{-1} sum_{s<=t} y_s)",
    },
    "final_amortized_metrics": result["amortized_final_metrics"],
    "final_sequential_metrics": {k: float(v) for k, v in result["final_metrics"].items() if np.ndim(v) == 0},
}
save_json(run_dir / "artefacts" / "final_summary.json", summary)
print("\nFinal summary:")
print(json.dumps(summary, indent=2))
print("All artefacts saved under:", run_dir)
