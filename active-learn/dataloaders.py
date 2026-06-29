import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import jax
import jax.numpy as jnp
from typing import Tuple

class PoolDataset(Dataset):
    """Pool of unlabeled candidate points."""
    def __init__(self, x_min: float, x_max: float, n_pool: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.X = rng.uniform(x_min, x_max, (n_pool, 1)).astype(np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])

class ValidationDataset(Dataset):
    """Validation set generated from oracle."""
    def __init__(self, oracle_fn, x_min: float, x_max: float, n_val: int):
        val_X = np.linspace(x_min, x_max, n_val, dtype=np.float32).reshape(-1, 1)
        val_Y = oracle_fn(val_X)
        self.X = torch.from_numpy(val_X)
        self.Y = torch.from_numpy(val_Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class OracleFamily:
    """Family of random oracles (He-initialized neural networks)."""
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64, 
                 n_hidden: int = 2, x_min: float = -3.0, x_max: float = 3.0):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.x_min = x_min
        self.x_max = x_max
    
    def sample(self, key: jax.random.PRNGKey) -> Tuple:
        """Sample a new oracle (random He-initialized MLP)."""
        dims = [self.x_dim] + [self.hidden_dim] * self.n_hidden + [self.y_dim]
        keys = jax.random.split(key, len(dims) - 1)
        
        weights = []
        biases = []
        for i, (in_d, out_d, k) in enumerate(zip(dims[:-1], dims[1:], keys)):
            w = jax.random.normal(k, (out_d, in_d)) * jnp.sqrt(2.0 / in_d)
            b = jnp.zeros(out_d)
            weights.append(w)
            biases.append(b)
        
        return weights, biases
    
    def evaluate(self, oracle_params: Tuple, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate oracle at x. x: (..., x_dim) -> (..., y_dim)"""
        weights, biases = oracle_params
        h = x
        for i, (w, b) in enumerate(zip(weights, biases)):
            h = h @ w.T + b
            if i < len(weights) - 1:
                h = jax.nn.relu(h)
        return h
    
    def evaluate_numpy(self, oracle_params: Tuple, x: np.ndarray) -> np.ndarray:
        """NumPy interface for oracle evaluation."""
        return np.array(self.evaluate(oracle_params, jnp.array(x)))

def make_dataloaders(cfg, oracle_fn=None):
    """Create pool and validation dataloaders."""
    pool_ds = PoolDataset(cfg.x_min, cfg.x_max, cfg.al.n_pool, cfg.seed)
    
    if oracle_fn is not None:
        val_ds = ValidationDataset(oracle_fn, cfg.x_min, cfg.x_max, cfg.al.n_val)
        val_loader = DataLoader(val_ds, batch_size=cfg.al.n_val, shuffle=False)
    else:
        val_loader = None
        
    pool_loader = DataLoader(pool_ds, batch_size=cfg.al.n_pool, shuffle=False)
    return pool_loader, val_loader

def collate_to_jax(batch):
    """Convert a PyTorch batch to JAX arrays."""
    if isinstance(batch, (list, tuple)):
        return [jnp.array(b.numpy()) for b in batch]
    return jnp.array(batch.numpy())

def get_pool_jax(pool_loader) -> jnp.ndarray:
    """Get full pool as a JAX array."""
    data = next(iter(pool_loader))
    return jnp.array(data.numpy())

def get_val_jax(val_loader) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get full validation set as JAX arrays."""
    X, Y = next(iter(val_loader))
    return jnp.array(X.numpy()), jnp.array(Y.numpy())