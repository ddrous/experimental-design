from dataclasses import dataclass, field
from typing import List

@dataclass
class SurrogateConfig:
    in_dim: int = 1
    out_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [32, 32])
    lr: float = 5e-3  # SGD learning rate for inner loop

@dataclass
class PolicyConfig:
    context_dim: int = 16
    point_emb_dim: int = 8
    hidden_dim: int = 32
    temperature_init: float = 2.0
    temperature_final: float = 0.3

@dataclass
class ALConfig:
    T: int = 15        # Number of acquisitions per episode
    gamma: float = 0.99
    n_pool: int = 80   # Pool size
    n_val: int = 100   # Validation size

@dataclass
class TrainingConfig:
    n_episodes: int = 300
    lr: float = 1e-3
    n_train_seeds: int = 20    # Different He-init surrogate seeds per oracle
    n_test_seeds: int = 20     # Test initialization seeds

@dataclass
class Config:
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    al: ALConfig = field(default_factory=ALConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    x_min: float = -3.0
    x_max: float = 3.0
    seed: int = 42
    gamma_sweep: List[float] = field(default_factory=lambda: [1.0, 0.95, 0.9, 0.7])
    results_dir: str = "results"