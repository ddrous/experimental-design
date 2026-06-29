import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from models import Surrogate, DesignPolicy

def compute_gradient_magnitudes(surrogate: Surrogate, pool_X: jnp.ndarray) -> jnp.ndarray:
    """Compute ||∂S_θ(x)/∂θ||_F for each x in pool."""
    params, static = eqx.partition(surrogate, eqx.is_array)
    
    def grad_norm_single(x):
        def output_fn(p):
            model = eqx.combine(p, static)
            return model(x).sum()
        
        grads = jax.grad(output_fn)(params)
        flat_grads = jnp.concatenate([g.ravel() for g in jax.tree_util.tree_leaves(grads)])
        return jnp.linalg.norm(flat_grads)
    
    return jax.vmap(grad_norm_single)(pool_X)

def gumbel_softmax(logits: jnp.ndarray, temperature: float, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Gumbel-Softmax trick for differentiable discrete selection."""
    gumbels = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-10) + 1e-10)
    y = (logits + gumbels) / temperature
    return jax.nn.softmax(y)

def gumbel_softmax_hard(logits: jnp.ndarray, temperature: float, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, int]:
    """Straight-through Gumbel-Softmax."""
    soft = gumbel_softmax(logits, temperature, key)
    idx = jnp.argmax(logits + gumbel_softmax(logits, 1e-8, key)) 
    hard = jax.nn.one_hot(idx, logits.shape[0])
    weights = hard + (soft - jax.lax.stop_gradient(soft))
    return weights, idx

def random_acquire(key: jax.random.PRNGKey, pool_mask: jnp.ndarray, **kwargs) -> int:
    """Random acquisition: select uniformly from unqueried pool points."""
    N = len(pool_mask)
    logits = jnp.where(pool_mask, -jnp.inf, 0.0)
    probs = jax.nn.softmax(logits)
    return jax.random.choice(key, N, p=probs)

def greedy_acquire(surrogate: Surrogate, pool_X: jnp.ndarray, pool_mask: jnp.ndarray, **kwargs) -> int:
    """Greedy gradient-magnitude acquisition."""
    grad_mags = compute_gradient_magnitudes(surrogate, pool_X)
    grad_mags = jnp.where(pool_mask, -jnp.inf, grad_mags)
    return jnp.argmax(grad_mags)

def albed_acquire(policy: DesignPolicy, surrogate: Surrogate, history_X: jnp.ndarray, 
                  history_Y: jnp.ndarray, n_history: int, pool_X: jnp.ndarray, 
                  pool_mask: jnp.ndarray, key: jax.random.PRNGKey, temperature: float = 0.1) -> int:
    """AL-BED acquisition: use trained policy to select next point."""
    grad_mags = compute_gradient_magnitudes(surrogate, pool_X)
    scores = policy(history_X, history_Y, jnp.array(n_history, dtype=float), pool_X, grad_mags)
    scores = jnp.where(pool_mask, -jnp.inf, scores)
    return jnp.argmax(scores)