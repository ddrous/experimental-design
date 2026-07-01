"""
dataloader.py
=============
PyTorch `Dataset`/`DataLoader` for the source-localisation BED task. There is no fixed dataset
in the usual sense -- each "example" is simply a draw theta ~ p(theta) of the K true source
locations for one simulated episode. We still route this through `torch.utils.data.DataLoader`
(as requested) purely for convenient batching/shuffling/num_workers parallelism; the collate
function returns plain numpy arrays, which `main.py` converts to `jax.numpy` arrays with a
single `jnp.asarray` call (zero-copy on CPU, cheap on GPU).

A fixed-size `IterableDataset` is used for training (infinite stream, reseeded per-worker), and
a separate seeded, *materialised* dataset is used for evaluation/test so that metrics are
reproducible across runs and across the two Bayes-simulator variants being compared.
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


class TrainEpisodes(IterableDataset):
    """Infinite stream of theta draws for training. Each worker gets its own RNG stream so that
    parallel data loading does not duplicate episodes."""

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

    ## Order the sources by their amplitude (norm) to avoid permutation ambiguity in the loss function (Allready done in the model)
    # batch = [np.array(sorted(theta, key=lambda x: np.linalg.norm(x), reverse=True)) for theta in batch] 

    return np.stack(batch, axis=0)


def make_train_loader(prior: SourceLocPrior, batch_size: int, base_seed: int = 0,
                       num_workers: int = 0) -> DataLoader:
    ds = TrainEpisodes(prior, base_seed=base_seed)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_thetas,
                       num_workers=num_workers)


def make_eval_loader(prior: SourceLocPrior, n_episodes: int, batch_size: int,
                      seed: int = 12345) -> DataLoader:
    ds = EvalEpisodes(prior, n_episodes=n_episodes, seed=seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_thetas)