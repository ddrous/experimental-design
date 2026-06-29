import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict
from training import run_episode_eval
from acquisition import random_acquire, greedy_acquire, albed_acquire

def make_random_acquire_fn():
    def fn(key, surrogate, pool_X, pool_mask, **kwargs):
        return random_acquire(key, pool_mask)
    return fn

def make_greedy_acquire_fn():
    def fn(key, surrogate, pool_X, pool_mask, **kwargs):
        return greedy_acquire(surrogate, pool_X, pool_mask)
    return fn

def make_albed_acquire_fn(policy, temperature=0.1):
    def fn(key, surrogate, pool_X, pool_mask, history_X, history_Y, n_history, **kwargs):
        return albed_acquire(policy, surrogate, history_X, history_Y, n_history, pool_X, pool_mask, key, temperature)
    return fn

def run_experiments(cfg, oracle_family, oracle_params, policy, key) -> Dict[str, np.ndarray]:
    """Evaluate Random, Greedy, and AL-BED strategies systematically."""
    test_keys = jax.random.split(key, cfg.training.n_test_seeds)

    # Establish environments
    pool_X_np = np.linspace(cfg.x_min, cfg.x_max, cfg.al.n_pool, dtype=np.float32).reshape(-1, 1)
    pool_X = jnp.array(pool_X_np)

    val_X_np = np.linspace(cfg.x_min, cfg.x_max, cfg.al.n_val, dtype=np.float32).reshape(-1, 1)
    val_Y_np = oracle_family.evaluate_numpy(oracle_params, val_X_np)
    val_X = jnp.array(val_X_np)
    val_Y = jnp.array(val_Y_np)

    results = {'Random': [], 'Greedy Magnitude': [], 'AL-BED': []}
    acquire_fns = {
        'Random': make_random_acquire_fn(),
        'Greedy Magnitude': make_greedy_acquire_fn(),
        'AL-BED': make_albed_acquire_fn(policy, temperature=0.1)
    }

    for name, acq_fn in acquire_fns.items():
        print(f"Running evaluation mapping for strategy: {name}...")
        for i in range(cfg.training.n_test_seeds):
            init_key, eval_key = jax.random.split(test_keys[i])
            val_losses, _ = run_episode_eval(
                acquire_fn=acq_fn,
                oracle_family=oracle_family,
                oracle_params=oracle_params,
                cfg=cfg,
                init_key=init_key,
                key=eval_key,
                pool_X=pool_X,
                val_X=val_X,
                val_Y=val_Y
            )
            results[name].append(np.array(val_losses))

    # Compile array structures
    return {k: np.array(v) for k, v in results.items()}