import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from typing import Tuple, Callable
from models import Surrogate, DesignPolicy, make_surrogate, make_policy
from acquisition import compute_gradient_magnitudes, gumbel_softmax

def mse_loss(surrogate: Surrogate, X: jnp.ndarray, Y: jnp.ndarray) -> float:
    """Mean squared error loss."""
    pred = jax.vmap(surrogate)(X)
    return jnp.mean((pred - Y) ** 2)

def surrogate_step(surrogate: Surrogate, x: jnp.ndarray, y: jnp.ndarray, lr: float) -> Surrogate:
    """Single gradient descent step on the surrogate."""
    params, static = eqx.partition(surrogate, eqx.is_array)
    
    def loss_fn(p):
        model = eqx.combine(p, static)
        pred = model(x)
        return jnp.mean((pred - y) ** 2)
    
    grads = jax.grad(loss_fn)(params)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g if g is not None else p, params, grads)
    return eqx.combine(new_params, static)

def episode_loss_albed(
    policy: DesignPolicy,
    surrogate_init: Surrogate,
    oracle_eval: Callable,
    oracle_params: Tuple,
    pool_X: jnp.ndarray,
    val_X: jnp.ndarray,
    val_Y: jnp.ndarray,
    T: int,
    gamma: float,
    surrogate_lr: float,
    temperature: float,
    key: jax.random.PRNGKey
) -> Tuple[float, jnp.ndarray]:
    """Compute the discounted expected future loss for one AL episode."""
    x_dim = pool_X.shape[1]
    y_dim = val_Y.shape[1]
    N_pool = pool_X.shape[0]
    
    surrogate = surrogate_init
    history_X = jnp.zeros((T, x_dim))
    history_Y = jnp.zeros((T, y_dim))
    pool_mask = jnp.zeros(N_pool, dtype=bool)
    
    prev_val_loss = mse_loss(surrogate, val_X, val_Y)
    val_losses = [prev_val_loss]
    
    J = 0.0
    
    for t in range(T):
        key, subkey = jax.random.split(key)
        
        grad_mags = compute_gradient_magnitudes(surrogate, pool_X)
        n_history_jax = jnp.array(float(t))
        scores = policy(history_X, history_Y, n_history_jax, pool_X, grad_mags)
        scores = jnp.where(jax.lax.stop_gradient(pool_mask), -jnp.inf, scores)
        
        soft_weights = gumbel_softmax(scores, temperature, subkey)
        x_t = jnp.dot(soft_weights, pool_X) 
        y_t = oracle_eval(oracle_params, x_t) 
        
        surrogate = surrogate_step(surrogate, x_t, y_t, surrogate_lr)
        
        curr_val_loss = mse_loss(surrogate, val_X, val_Y)
        improvement = curr_val_loss - prev_val_loss
        J = J + (gamma ** t) * improvement
        prev_val_loss = curr_val_loss
        val_losses.append(curr_val_loss)
        
        hard_idx = jnp.argmax(jax.lax.stop_gradient(scores))
        history_X = history_X.at[t].set(jax.lax.stop_gradient(pool_X[hard_idx]))
        history_Y = history_Y.at[t].set(jax.lax.stop_gradient(y_t))
        pool_mask = pool_mask.at[hard_idx].set(True)
    
    return J, jnp.array(val_losses)

def train_policy(cfg, oracle_family, key: jax.random.PRNGKey):
    """Train the AL-BED design policy."""
    key, policy_key, pool_key, val_key, oracle_key = jax.random.split(key, 5)
    
    oracle_params = oracle_family.sample(oracle_key)
    
    pool_X_np = np.linspace(cfg.x_min, cfg.x_max, cfg.al.n_pool, dtype=np.float32).reshape(-1, 1)
    pool_X = jnp.array(pool_X_np)
    
    val_X_np = np.linspace(cfg.x_min, cfg.x_max, cfg.al.n_val, dtype=np.float32).reshape(-1, 1)
    val_Y_np = oracle_family.evaluate_numpy(oracle_params, val_X_np)
    val_X = jnp.array(val_X_np)
    val_Y = jnp.array(val_Y_np)
    
    policy = make_policy(cfg, policy_key)
    optimizer = optax.adam(cfg.training.lr)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))
    
    policy_params, policy_static = eqx.partition(policy, eqx.is_array)
    
    def loss_for_grad(p_params, s_init, eps_key, temp):
        p = eqx.combine(p_params, policy_static)
        loss_val, _ = episode_loss_albed(
            p, s_init, oracle_family.evaluate, oracle_params,
            pool_X, val_X, val_Y,
            cfg.al.T, cfg.al.gamma, cfg.surrogate.lr, temp, eps_key
        )
        return loss_val
    
    losses = []
    for episode in range(cfg.training.n_episodes):
        key, init_key, episode_key = jax.random.split(key, 3)
        surrogate_init = make_surrogate(cfg, init_key)
        
        frac = episode / cfg.training.n_episodes
        temperature = cfg.policy.temperature_init * (1 - frac) + cfg.policy.temperature_final * frac
        
        J, grads = jax.value_and_grad(loss_for_grad)(policy_params, surrogate_init, episode_key, temperature)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        policy_params = optax.apply_updates(policy_params, updates)
        losses.append(float(J))
        
        if episode % 50 == 0:
            print(f"Episode {episode}/{cfg.training.n_episodes}, J={J:.4f}, temp={temperature:.3f}")
            
    policy = eqx.combine(policy_params, policy_static)
    return policy, losses

def run_episode_eval(acquire_fn, oracle_family, oracle_params, cfg, 
                     init_key: jax.random.PRNGKey, key: jax.random.PRNGKey,
                     pool_X: jnp.ndarray, val_X: jnp.ndarray, val_Y: jnp.ndarray,
                     **acquire_kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run a single evaluation episode with a given acquisition strategy."""
    surrogate = make_surrogate(cfg, init_key)
    N_pool = pool_X.shape[0]
    x_dim = pool_X.shape[1]
    y_dim = val_Y.shape[1]
    T = cfg.al.T
    
    pool_mask = jnp.zeros(N_pool, dtype=bool)
    history_X = jnp.zeros((T, x_dim))
    history_Y = jnp.zeros((T, y_dim))
    
    val_losses = [float(mse_loss(surrogate, val_X, val_Y))]
    queried_X = []
    
    for t in range(T):
        key, subkey = jax.random.split(key)
        idx = acquire_fn(
            key=subkey,
            surrogate=surrogate,
            pool_X=pool_X,
            pool_mask=pool_mask,
            history_X=history_X,
            history_Y=history_Y,
            n_history=t,
            **acquire_kwargs
        )
        idx = int(idx)
        
        x_t = pool_X[idx]
        y_t = oracle_family.evaluate(oracle_params, x_t)
        
        surrogate = surrogate_step(surrogate, x_t, y_t, cfg.surrogate.lr)
        
        history_X = history_X.at[t].set(x_t)
        history_Y = history_Y.at[t].set(y_t)
        pool_mask = pool_mask.at[idx].set(True)
        
        val_losses.append(float(mse_loss(surrogate, val_X, val_Y)))
        queried_X.append(float(x_t[0]))
    
    return jnp.array(val_losses), jnp.array(queried_X)