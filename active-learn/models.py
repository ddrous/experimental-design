import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List

class Surrogate(eqx.Module):
    """Neural surrogate S_θ: X -> Y."""
    layers: List[eqx.nn.Linear]
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], key: jax.random.PRNGKey):
        dims = [in_dim] + hidden_dims + [out_dim]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [eqx.nn.Linear(dims[i], dims[i+1], key=keys[i]) 
                       for i in range(len(dims) - 1)]
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (in_dim,) -> (out_dim,)"""
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)

class PairEncoder(eqx.Module):
    """Encodes (x, y) pairs into embeddings for history context."""
    mlp: eqx.nn.MLP
    
    def __init__(self, x_dim: int, y_dim: int, context_dim: int, key: jax.random.PRNGKey):
        self.mlp = eqx.nn.MLP(
            in_size=x_dim + y_dim,
            out_size=context_dim,
            width_size=32,
            depth=2,
            key=key
        )
    
    def __call__(self, xy: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(xy)

class DesignPolicy(eqx.Module):
    """
    Policy π^φ_d that maps (history, pool, surrogate_state) -> acquisition scores.
    """
    pair_encoder: PairEncoder   
    point_encoder: eqx.nn.MLP   
    scorer: eqx.nn.MLP          
    
    def __init__(self, x_dim: int, y_dim: int, context_dim: int, 
                 point_emb_dim: int, hidden_dim: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 3)
        self.pair_encoder = PairEncoder(x_dim, y_dim, context_dim, keys[0])
        self.point_encoder = eqx.nn.MLP(
            in_size=x_dim,
            out_size=point_emb_dim,
            width_size=hidden_dim,
            depth=2,
            key=keys[1]
        )
        self.scorer = eqx.nn.MLP(
            in_size=context_dim + point_emb_dim + 1,
            out_size=1,
            width_size=hidden_dim,
            depth=2,
            key=keys[2]
        )
    
    def encode_history(self, history_X: jnp.ndarray, history_Y: jnp.ndarray, 
                       n_history: jnp.ndarray) -> jnp.ndarray:
        """Encode variable-length history into fixed-size context."""
        T_max = history_X.shape[0]
        pairs = jnp.concatenate([history_X, history_Y], axis=-1)
        embs = jax.vmap(self.pair_encoder)(pairs)
        
        mask = (jnp.arange(T_max) < n_history).astype(float)
        sum_embs = jnp.sum(embs * mask[:, None], axis=0)
        count = jnp.maximum(n_history.astype(float), 1.0)
        return sum_embs / count
    
    def __call__(self, history_X: jnp.ndarray, history_Y: jnp.ndarray,
                 n_history: jnp.ndarray, pool_X: jnp.ndarray, 
                 grad_mags: jnp.ndarray) -> jnp.ndarray:
        """Compute acquisition scores for each pool point."""
        context = self.encode_history(history_X, history_Y, n_history)
        point_embs = jax.vmap(self.point_encoder)(pool_X)
        
        N = pool_X.shape[0]
        ctx_expanded = jnp.broadcast_to(context[None], (N, context.shape[0]))
        features = jnp.concatenate([ctx_expanded, point_embs, grad_mags[:, None]], axis=-1)
        scores = jax.vmap(self.scorer)(features).squeeze(-1)
        return scores

def make_surrogate(cfg, key: jax.random.PRNGKey) -> Surrogate:
    return Surrogate(
        in_dim=cfg.surrogate.in_dim,
        out_dim=cfg.surrogate.out_dim,
        hidden_dims=cfg.surrogate.hidden_dims,
        key=key
    )

def make_policy(cfg, key: jax.random.PRNGKey) -> DesignPolicy:
    return DesignPolicy(
        x_dim=cfg.surrogate.in_dim,
        y_dim=cfg.surrogate.out_dim,
        context_dim=cfg.policy.context_dim,
        point_emb_dim=cfg.policy.point_emb_dim,
        hidden_dim=cfg.policy.hidden_dim,
        key=key
    )