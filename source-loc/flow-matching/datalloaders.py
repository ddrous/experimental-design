"""
dataloader.py
=============
PyTorch `Dataset`/`DataLoader` for the source-localisation BED task. There is no fixed
dataset in the usual sense -- each "example" is simply a draw theta ~ p(theta) of the K
true source locations for one simulated episode. We still route this through
`torch.utils.data.DataLoader` (as requested) purely for convenient batching/shuffling/
num_workers parallelism; the collate function returns plain numpy arrays, which
`main.py` converts to `jax.numpy` arrays with a single `jnp.asarray` call (zero-copy on
CPU, cheap on GPU).

Two training-data regimes are supported, selected via `data_mode`:

  - "finite"   (default, recommended for reproducibility): `FiniteEpisodes` pre-generates
               a FIXED pool of `n_train_episodes` integer seeds at construction time
               (stored on the dataset as `self.seeds`), i.e. the complete list of every
               seed that will EVER be used by this dataset instance. `__getitem__(idx)`
               deterministically draws exactly one theta via
               `np.random.default_rng(self.seeds[idx]).` A full pass over the
               `DataLoader` therefore visits every one of those `n_train_episodes`
               episodes exactly once -- this is what "one epoch" means below, matching
               the usual definition (the entire dataset traversed once per epoch), and
               it makes `len(train_loader)` the natural (and only) source of truth for
               how many gradient steps happen per epoch (no separate/free
               `steps_per_epoch` knob needed). Shuffling is done through an explicit,
               seeded `torch.Generator` so the traversal order is itself reproducible.

  - "infinite": `TrainEpisodes` (unchanged from before) is an infinite stream of fresh
               theta draws, reseeded per DataLoader worker. Useful if you'd rather keep
               generating "as much data as possible" continuously instead of cycling a
               fixed pool; in this mode `Config.steps_per_epoch` is what defines an
               "epoch" (there is no natural dataset length to traverse).

A separate seeded, materialised dataset (`EvalEpisodes`) is used for evaluation/test so
that metrics are reproducible across runs and across model variants being compared.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info


class SourceLocPrior:
    """theta_k ~ N(0, prior_std^2 I_2), iid across the K sources."""

    def __init__(self, K: int = 1, prior_std: float = 1.0):
        self.K = K
        self.prior_std = prior_std

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.prior_std, size=(self.K, 2)).astype(np.float32)


class FiniteEpisodes(Dataset):
    """Map-style, reproducible dataset: pre-generates a fixed pool of `n_episodes` seeds
    at construction time. Every `__getitem__(idx)` call is a pure, deterministic function
    of `idx` (via `self.seeds[idx]`), so a full traversal of this dataset -- i.e. one
    epoch -- is exactly reproducible given the same `base_seed`/`n_episodes`, regardless
    of `num_workers` or how many epochs have already run."""

    def __init__(self, prior: SourceLocPrior, n_episodes: int, base_seed: int = 0):
        self.prior = prior
        self.n_episodes = n_episodes
        # The complete list of every seed this dataset instance will ever draw from.
        self.seeds = (np.arange(n_episodes, dtype=np.int64) + base_seed).tolist()

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int) -> np.ndarray:
        rng = np.random.default_rng(self.seeds[idx])
        return self.prior.sample(rng)


class TrainEpisodes(IterableDataset):
    """Infinite stream of theta draws for training. Each worker gets its own RNG stream so
    that parallel data loading does not duplicate episodes. Use `data_mode='infinite'` in
    `make_train_loader` to select this; there is no natural notion of dataset length /
    "one epoch" here -- see `Config.steps_per_epoch`."""

    def __init__(self, prior: SourceLocPrior, base_seed: int = 0):
        self.prior = prior
        self.base_seed = base_seed

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = np.random.default_rng(self.base_seed + worker_id)
        while True:
            yield self.prior.sample(rng)


class EvalEpisodes(Dataset):
    """Materialised, seeded evaluation set: same `n_episodes` thetas every run."""

    def __init__(self, prior: SourceLocPrior, n_episodes: int = 512, seed: int = 12345):
        rng = np.random.default_rng(seed)
        self.thetas = np.stack([prior.sample(rng) for _ in range(n_episodes)], axis=0)

    def __len__(self):
        return self.thetas.shape[0]

    def __getitem__(self, idx):
        return self.thetas[idx]


def collate_thetas(batch):
    """list of (K,2) arrays -> single (B,K,2) numpy array, ready for jnp.asarray()."""
    return np.stack(batch, axis=0)


def make_train_loader(prior: SourceLocPrior, batch_size: int, base_seed: int = 0,
                       num_workers: int = 0, data_mode: str = "finite",
                       n_train_episodes: int = 200_000, drop_last: bool = True) -> DataLoader:
    """Build the training DataLoader.

    data_mode='finite'   -> reproducible, fixed pool of `n_train_episodes` seeds; one
                             epoch = one full (seeded-shuffle) pass over the pool. This is
                             the recommended default.
    data_mode='infinite' -> unbounded stream of fresh draws (old behaviour); pair with
                             `Config.steps_per_epoch` to define an epoch length.

    `drop_last=True` (finite mode only) keeps every training batch the same shape, which
    matters for JAX/XLA: `eqx.filter_jit` retraces (recompiles) whenever the input shape
    changes, so a single ragged final batch per epoch would otherwise trigger an extra,
    wasteful compilation on every epoch.
    """
    if data_mode == "finite":
        ds = FiniteEpisodes(prior, n_episodes=n_train_episodes, base_seed=base_seed)
        generator = torch.Generator()
        generator.manual_seed(base_seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_thetas,
                           num_workers=num_workers, generator=generator, drop_last=drop_last)
    elif data_mode == "infinite":
        ds = TrainEpisodes(prior, base_seed=base_seed)
        return DataLoader(ds, batch_size=batch_size, collate_fn=collate_thetas,
                           num_workers=num_workers)
    else:
        raise ValueError(f"Unknown data_mode '{data_mode}'. Choose 'finite' or 'infinite'.")


def make_eval_loader(prior: SourceLocPrior, n_episodes: int, batch_size: int,
                      seed: int = 12345) -> DataLoader:
    ds = EvalEpisodes(prior, n_episodes=n_episodes, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_thetas)