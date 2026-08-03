"""
loaders.py
==========

Dataloaders for the WeightTransformer density-estimation problem.

Each "example" is a *task*: a freshly-sampled target distribution (e.g. a 2D
Gaussian with random mean/std), together with a sequence of T samples x_1..x_T
drawn i.i.d. from it. The WeightTransformer consumes the sequence of samples
(as conditioning, or as tokens directly) and is trained to make p_{theta_t}
assign high likelihood to x_t at every step t (MLE over the whole sequence).

Design:
    - Pure NumPy for the host-side sampling (fast, simple, avoids host<->device
      churn for tiny 2D data), converted to JAX arrays per-batch.
    - `TaskDistribution` subclasses define how to sample a random task
      (its parameters) and how to sample points given those parameters.
      This makes it trivial to swap in other easy 2D densities later
      (e.g. mixtures of Gaussians, rings, etc.) without touching the loader.
    - Every batch optionally has its sample order permuted independently per
      task, per epoch, per batch -- this is what "permute the inputs before
      exposing them to the Transformer" means: the model must be robust to
      the arbitrary order in which conditioning samples arrive (important
      for the non-AdaLN, "samples-are-tokens" variant especially).
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------------------
# Task distributions (host-side, NumPy)
# --------------------------------------------------------------------------------------


class TaskDistribution:
    """Interface for a family of 2D (or D-D) target densities.

    A "task" is one concrete member of the family (e.g. one Gaussian with a
    particular mean/std). ``sample_task_params`` draws a random task;
    ``sample_points`` draws i.i.d. points from a given task.
    """

    dim: int = 2

    def sample_task_params(self, rng: np.random.Generator) -> np.ndarray:
        """Return a flat float32 array describing one random task instance."""
        raise NotImplementedError

    def sample_points(self, params: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``(n, dim)`` i.i.d. samples from the task described by params."""
        raise NotImplementedError

    def log_prob(self, params: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Ground-truth log density (for evaluation/plotting only)."""
        raise NotImplementedError


@dataclasses.dataclass
class GaussianTask(TaskDistribution):
    """Isotropic-ish 2D Gaussian with random mean and per-axis std.

    params layout: [mean_x, mean_y, log_std_x, log_std_y]
    """

    dim: int = 2
    mean_range: float = 2.0
    log_std_low: float = -1.5   # std ~= 0.22
    log_std_high: float = 0.4   # std ~= 1.5

    def sample_task_params(self, rng: np.random.Generator) -> np.ndarray:
        mean = rng.uniform(-self.mean_range, self.mean_range, size=(self.dim,))
        log_std = rng.uniform(self.log_std_low, self.log_std_high, size=(self.dim,))
        return np.concatenate([mean, log_std]).astype(np.float32)

    def _unpack(self, params: np.ndarray):
        mean, log_std = params[: self.dim], params[self.dim:]
        return mean, np.exp(log_std)

    def sample_points(self, params: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        mean, std = self._unpack(params)
        return (mean + std * rng.standard_normal((n, self.dim))).astype(np.float32)

    def log_prob(self, params: np.ndarray, points: np.ndarray) -> np.ndarray:
        mean, std = self._unpack(params)
        z = (points - mean) / std
        return (
            -0.5 * np.sum(z ** 2, axis=-1)
            - np.sum(np.log(std))
            - self.dim * 0.5 * np.log(2 * np.pi)
        )


@dataclasses.dataclass
class RingTask(TaskDistribution):
    """A noisy ring/annulus in 2D -- an easy non-Gaussian family for later use.

    params layout: [center_x, center_y, radius, noise_std]
    """

    dim: int = 2
    center_range: float = 1.0
    radius_low: float = 0.6
    radius_high: float = 1.8
    noise_low: float = 0.05
    noise_high: float = 0.2

    def sample_task_params(self, rng: np.random.Generator) -> np.ndarray:
        center = rng.uniform(-self.center_range, self.center_range, size=(2,))
        radius = rng.uniform(self.radius_low, self.radius_high, size=(1,))
        noise = rng.uniform(self.noise_low, self.noise_high, size=(1,))
        return np.concatenate([center, radius, noise]).astype(np.float32)

    def sample_points(self, params: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        center, radius, noise = params[:2], params[2], params[3]
        angles = rng.uniform(0, 2 * np.pi, size=(n,))
        radii = radius + noise * rng.standard_normal((n,))
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radii[:, None]
        return (center + pts).astype(np.float32)

    def log_prob(self, params: np.ndarray, points: np.ndarray) -> np.ndarray:
        # Approximate: not needed for training, only informative for plotting.
        center, radius, noise = params[:2], params[2], params[3]
        r = np.linalg.norm(points - center, axis=-1)
        z = (r - radius) / noise
        return -0.5 * z ** 2 - np.log(noise) - 0.5 * np.log(2 * np.pi) - np.log(np.maximum(r, 1e-6))


TASK_REGISTRY = {
    "gaussian": GaussianTask,
    "ring": RingTask,
}


def make_task_distribution(name: str, **kwargs) -> TaskDistribution:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task distribution '{name}'. Options: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name](**kwargs)


# --------------------------------------------------------------------------------------
# Dataloader
# --------------------------------------------------------------------------------------


class TaskSequenceLoader:
    """Batches of (task_params, sample_sequence) with optional permutation.

    Each item in a batch is an independent random task (e.g. a fresh Gaussian
    mean/std), paired with ``seq_len`` i.i.d. samples drawn from it. Shapes:

        x_seq:       (batch_size, seq_len, dim)          float32
        task_params: (batch_size, param_dim)             float32

    ``permute_inputs=True`` independently shuffles the order of the seq_len
    samples for every task in every batch, every epoch -- this is what feeds
    the "permute before exposing to the Transformer" requirement. Since the
    points are i.i.d., permuting is exact (no information change), and it
    forces both architectures (especially the no-AdaLN token variant, which
    sees the samples as its own sequence) to not overfit to a spurious order.
    """

    def __init__(
        self,
        task_dist: TaskDistribution,
        seq_len: int,
        batch_size: int,
        num_batches_per_epoch: int,
        seed: int = 0,
        permute_inputs: bool = True,
    ):
        self.task_dist = task_dist
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.num_batches_per_epoch = int(num_batches_per_epoch)
        self.permute_inputs = bool(permute_inputs)
        self._rng = np.random.default_rng(seed)

    def set_epoch(self, epoch: int, seed: Optional[int] = None) -> None:
        """Re-seed deterministically per epoch (mirrors torch DataLoader semantics)."""
        base_seed = seed if seed is not None else 0
        self._rng = np.random.default_rng(base_seed + 1000 * epoch)

    def _sample_batch(self):
        dim = self.task_dist.dim
        param_dim = self.task_dist.sample_task_params(self._rng).shape[0]

        x_seq = np.empty((self.batch_size, self.seq_len, dim), dtype=np.float32)
        task_params = np.empty((self.batch_size, param_dim), dtype=np.float32)

        for b in range(self.batch_size):
            params = self.task_dist.sample_task_params(self._rng)
            points = self.task_dist.sample_points(params, self.seq_len, self._rng)
            if self.permute_inputs:
                order = self._rng.permutation(self.seq_len)
                points = points[order]
            task_params[b] = params
            x_seq[b] = points

        return jnp.asarray(x_seq), jnp.asarray(task_params)

    def __iter__(self) -> Iterator:
        for _ in range(self.num_batches_per_epoch):
            yield self._sample_batch()

    def __len__(self) -> int:
        return self.num_batches_per_epoch


def get_dataloaders(cfg: dict, task_dist: Optional[TaskDistribution] = None):
    """Build train/val TaskSequenceLoaders from a flat config dict.

    Expected keys (see train.py CONFIG): task_name, task_kwargs, seq_len,
    batch_size, batches_per_epoch, val_batches_per_epoch, seed, permute_inputs.
    """
    if task_dist is None:
        task_dist = make_task_distribution(cfg["task_name"], **cfg.get("task_kwargs", {}))

    train_loader = TaskSequenceLoader(
        task_dist,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        num_batches_per_epoch=cfg["batches_per_epoch"],
        seed=cfg["seed"],
        permute_inputs=cfg.get("permute_inputs", True),
    )
    val_loader = TaskSequenceLoader(
        task_dist,
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        num_batches_per_epoch=cfg.get("val_batches_per_epoch", 4),
        seed=cfg["seed"] + 1,
        permute_inputs=cfg.get("permute_inputs", True),
    )
    return train_loader, val_loader, task_dist