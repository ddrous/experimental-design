"""
models.py
=========
Equinox + JAX model definitions for Latent-Action Bayesian Experimental Design (LA-BED).

Three actors, matching the whiteboard sketch:

  ENVIRONMENT          y_t ~ p(. | x_t, theta)              -- true oracle, fixed theta per episode
  DESIGN POLICY  (psi)  x_t  = pi_psi(h_{t-1}, hat_theta_t)  -- "inverse model": picks next design
  BAYES SIMULATOR (phi) hat_theta_{t+1} = f_phi(x_t, y_t, hat_theta_t [, h_{1:t}])
                                                              -- "forward model": implements Bayes' rule

The belief hat_theta_t is represented by a `Belief` pytree (mean + log-std). Two regimes are
supported through the *same* pytree structure (so jax.lax.scan carries a fixed-shape carry):

  - "point"    : only `mu` is used/trained on; `log_sigma` is present but ignored by the loss.
  - "gaussian" : both `mu` and `log_sigma` are used; EIG is computed in closed form as the
                 reduction in differential entropy of a diagonal Gaussian belief.

Two interchangeable Bayes-Simulator implementations are provided (`BayesSimulatorMLP` and
`BayesSimulatorTransformer`); both expose the same `__call__(history, belief) -> Belief` API
so they can be swapped in `main.py` without touching the training loop.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
from jax import random


# ----------------------------------------------------------------------------------
# Belief pytree
# ----------------------------------------------------------------------------------
class Belief(NamedTuple):
    mu: jnp.ndarray          # (D,)  point estimate / Gaussian mean
    log_sigma: jnp.ndarray   # (D,)  log std-dev (unused when belief_mode == "point")


def init_belief(dim: int, init_log_sigma: float = 0.0) -> Belief:
    return Belief(mu=jnp.zeros((dim,)), log_sigma=jnp.full((dim,), init_log_sigma))


def belief_entropy(belief: Belief) -> jnp.ndarray:
    """Differential entropy of an axis-aligned Gaussian belief, summed over dimensions."""
    return jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + belief.log_sigma)


def expected_information_gain(x_t: jnp.ndarray, belief: Belief, 
                            x_h: jnp.ndarray, y_h: jnp.ndarray,
                            model, cfg, key: jax.random.PRNGKey, 
                            num_samples: int = 32) -> jnp.ndarray:
    """
    Estimates the Expected Information Gain (EIG) for a proposed design x_t.
    EIG(x_t) = E_{theta ~ belief, y ~ p(y|theta, x_t)} [ H(belief) - H(next_belief) ]
    """
    # Split keys for the Monte Carlo samples
    keys = jax.random.split(key, num_samples)
    
    # Standard deviation derived from the belief's log_sigma
    std = jnp.exp(belief.log_sigma)
    
    def simulate_single_outcome(k):
        k_theta, k_env = jax.random.split(k)
        
        # 1. Sample a hypothetical theta from the current parametric belief
        noise = jax.random.normal(k_theta, belief.mu.shape)
        theta_sample_flat = belief.mu + std * noise
        
        # Reshape to (K, 2) dynamically based on flattened size
        theta_sample = theta_sample_flat.reshape((-1, 2)) 
        
        # 2. Simulate a hypothetical observation y for this design
        y_sample = environment_step(theta_sample, x_t, k_env, cfg)
        
        # 3. Simulate what the NEXT belief would be using the Bayes Simulator
        next_belief_sample = model.simulator(x_t, y_sample, belief, x_h, y_h)
        
        # 4. Calculate the reduction in entropy for this specific outcome
        return belief_entropy(belief) - belief_entropy(next_belief_sample)
        
    # Vectorize over the random keys to generate `num_samples` MC samples
    reductions = jax.vmap(simulate_single_outcome)(keys)
    
    # Return the expected (average) information gain
    return jnp.mean(reductions)


def belief_nll(belief: Belief, theta_true: jnp.ndarray) -> jnp.ndarray:
    """Gaussian NLL of the true parameter under the current belief (only meaningful if
    belief_mode == 'gaussian'; falls back gracefully otherwise since log_sigma still exists)."""
    var = jnp.exp(2.0 * belief.log_sigma)
    return jnp.sum(0.5 * jnp.log(2.0 * jnp.pi * var) + 0.5 * (theta_true - belief.mu) ** 2 / var)


# ----------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------
class AdaLN(eqx.Module):
    """Adaptive LayerNorm: normalises x then applies a (scale, shift) predicted from `cond`."""
    norm: eqx.nn.LayerNorm
    to_scale_shift: eqx.nn.Linear

    def __init__(self, dim: int, cond_dim: int, *, key):
        self.norm = eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False)
        self.to_scale_shift = eqx.nn.Linear(cond_dim, 2 * dim, key=key)
        # zero-init so AdaLN starts as identity-ish (standard DiT-style trick)
        zeros_w = jnp.zeros_like(self.to_scale_shift.weight)
        zeros_b = jnp.zeros_like(self.to_scale_shift.bias)
        self.to_scale_shift = eqx.tree_at(
            lambda l: (l.weight, l.bias), self.to_scale_shift, (zeros_w, zeros_b)
        )

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        # x: (T, dim) or (dim,); cond: (cond_dim,)
        gamma, beta = jnp.split(self.to_scale_shift(cond), 2, axis=-1)
        if x.ndim == 2:
            normed = jax.vmap(self.norm)(x)
        else:
            normed = self.norm(x)
        return normed * (1.0 + gamma) + beta


class FeedForward(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, hidden: int, *, key):
        k1, k2 = random.split(key)
        self.fc1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.fc2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.fc2)(jax.nn.gelu(jax.vmap(self.fc1)(x)))


class TransformerBlock(eqx.Module):
    """Pre-AdaLN transformer encoder block (self-attention + MLP), conditioned on `cond`."""
    ada1: AdaLN
    ada2: AdaLN
    attn: eqx.nn.MultiheadAttention
    ff: FeedForward

    def __init__(self, dim: int, cond_dim: int, num_heads: int, ff_mult: int, *, key):
        k1, k2, k3 = random.split(key, 3)
        self.ada1 = AdaLN(dim, cond_dim, key=k1)
        self.ada2 = AdaLN(dim, cond_dim, key=k2)
        self.attn = eqx.nn.MultiheadAttention(num_heads, dim, key=k3)
        self.ff = FeedForward(dim, dim * ff_mult, key=k3)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        # x: (T, dim)
        h = self.ada1(x, cond)
        x = x + self.attn(h, h, h)
        h = self.ada2(x, cond)
        x = x + self.ff(h)
        return x


def _pair_encode(x_seq: jnp.ndarray, y_seq: jnp.ndarray, encoder: eqx.nn.MLP) -> jnp.ndarray:
    """Encode a sequence of (design, outcome) pairs into tokens. x_seq: (T,dx), y_seq:(T,dy)."""
    pairs = jnp.concatenate([x_seq, y_seq], axis=-1)
    return jax.vmap(encoder)(pairs)


# ----------------------------------------------------------------------------------
# Design policy (inverse model): pi_psi(history, belief) -> next design
# ----------------------------------------------------------------------------------
class DesignPolicyTransformer(eqx.Module):
    """Transformer design policy conditioned via AdaLN on the current belief hat_theta_t."""
    pair_encoder: eqx.nn.MLP
    null_token: jnp.ndarray
    blocks: Tuple[TransformerBlock, ...]
    out_head: eqx.nn.MLP
    design_dim: int = eqx.field(static=True)
    design_bound: float = eqx.field(static=True)

    def __init__(self, design_dim, obs_dim, belief_dim, hidden=64, num_heads=4,
                 num_layers=2, ff_mult=2, design_bound=4.0, *, key):
        keys = random.split(key, num_layers + 3)
        self.pair_encoder = eqx.nn.MLP(design_dim + obs_dim, hidden, hidden, depth=2, key=keys[0])
        self.null_token = jnp.zeros((hidden,))
        cond_dim = 2 * belief_dim  # cond on [mu, log_sigma]
        self.blocks = tuple(
            TransformerBlock(hidden, cond_dim, num_heads, ff_mult, key=keys[i + 1])
            for i in range(num_layers)
        )
        self.out_head = eqx.nn.MLP(hidden, design_dim, hidden, depth=2, key=keys[-1])
        self.design_dim = design_dim
        self.design_bound = design_bound

    def __call__(self, x_hist: jnp.ndarray, y_hist: jnp.ndarray, belief: Belief) -> jnp.ndarray:
        """x_hist: (t, design_dim), y_hist: (t, obs_dim) history *before* this step (t may be 0)."""
        cond = jnp.concatenate([belief.mu, belief.log_sigma], axis=-1)
        if x_hist.shape[0] == 0:
            tokens = self.null_token[None, :]
        else:
            tokens = _pair_encode(x_hist, y_hist, self.pair_encoder)
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        summary = tokens[-1]
        raw = self.out_head(summary)
        return self.design_bound * jnp.tanh(raw)
    
# ----------------------------------------------------------------------------------
# Design policy MLP (inverse model): pi_psi(\cdot, belief) -> next design; it ignores the history and is a simple MLP conditioned on the belief.
# ----------------------------------------------------------------------------------
class DesignPolicyMLP(eqx.Module):
    """MLP design policy conditioned on the current belief hat_theta_t. Ignores history."""
    net: eqx.nn.MLP
    design_dim: int = eqx.field(static=True)
    design_bound: float = eqx.field(static=True)

    def __init__(self, design_dim, obs_dim, belief_dim, hidden=64, depth=3,
                 design_bound=4.0, *, key):
        in_dim = 2 * belief_dim  # cond on [mu, log_sigma]
        out_dim = design_dim
        self.net = eqx.nn.MLP(in_dim, out_dim, hidden, depth=depth, key=key)
        self.design_dim = design_dim
        self.design_bound = design_bound

    def __call__(self, x_hist: jnp.ndarray, y_hist: jnp.ndarray, belief: Belief) -> jnp.ndarray:
        """x_hist/y_hist are ignored; only the belief is used to produce the next design."""
        inp = jnp.concatenate([belief.mu, belief.log_sigma], axis=-1)
        raw = self.net(inp)
        # return self.design_bound * jnp.tanh(raw)
        ## Crop the design to be within the bounds [-design_bound, design_bound]
        return jnp.clip(raw, -self.design_bound, self.design_bound)



# ----------------------------------------------------------------------------------
# Bayes simulator (forward model): f_phi(history, belief) -> next belief
# Two interchangeable implementations sharing the call signature
#     __call__(x_t, y_t, belief, x_hist, y_hist) -> Belief
# ----------------------------------------------------------------------------------
class BayesSimulatorMLP(eqx.Module):
    """MLP variant: takes only (x_t, y_t, hat_theta_t) -- a *Markovian* belief update,
    i.e. exactly Bayes' rule applied to the running belief (no access to full history)."""
    net: eqx.nn.MLP
    belief_dim: int = eqx.field(static=True)

    def __init__(self, design_dim, obs_dim, belief_dim, hidden=64, depth=3, *, key):
        in_dim = design_dim + obs_dim + 2 * belief_dim
        out_dim = 2 * belief_dim  # delta_mu, delta_log_sigma
        self.net = eqx.nn.MLP(in_dim, out_dim, hidden, depth=depth, key=key)
        self.belief_dim = belief_dim

    def __call__(self, x_t, y_t, belief: Belief, x_hist=None, y_hist=None) -> Belief:
        inp = jnp.concatenate([x_t, y_t, belief.mu, belief.log_sigma], axis=-1)
        out = self.net(inp)
        d_mu, d_log_sigma = jnp.split(out, 2, axis=-1)
        new_mu = belief.mu + d_mu
        # new_log_sigma = belief.log_sigma + jnp.tanh(d_log_sigma) * 0.5  # bounded, stable update
        new_log_sigma = belief.log_sigma + d_log_sigma
        return Belief(mu=new_mu, log_sigma=new_log_sigma)


class BayesSimulatorTransformer(eqx.Module):
    """Transformer variant: re-reads the *entire* history (x_{1:t}, y_{1:t}), conditioned via
    AdaLN on the previous belief, and emits the updated belief. Strictly more expressive than
    the MLP variant (non-Markovian), at higher compute cost."""
    pair_encoder: eqx.nn.MLP
    blocks: Tuple[TransformerBlock, ...]
    out_head: eqx.nn.MLP
    belief_dim: int = eqx.field(static=True)

    def __init__(self, design_dim, obs_dim, belief_dim, hidden=64, num_heads=4,
                 num_layers=2, ff_mult=2, *, key):
        keys = random.split(key, num_layers + 2)
        self.pair_encoder = eqx.nn.MLP(design_dim + obs_dim, hidden, hidden, depth=2, key=keys[0])
        cond_dim = 2 * belief_dim
        self.blocks = tuple(
            TransformerBlock(hidden, cond_dim, num_heads, ff_mult, key=keys[i + 1])
            for i in range(num_layers)
        )
        self.out_head = eqx.nn.MLP(hidden, 2 * belief_dim, hidden, depth=2, key=keys[-1])
        self.belief_dim = belief_dim

    def __call__(self, x_t, y_t, belief: Belief, x_hist, y_hist) -> Belief:
        """x_hist/y_hist: full history INCLUDING the current step (t, dim), already concatenated
        by the caller; x_t, y_t passed separately only to keep the API symmetric with the MLP."""
        cond = jnp.concatenate([belief.mu, belief.log_sigma], axis=-1)
        tokens = _pair_encode(x_hist, y_hist, self.pair_encoder)
        for blk in self.blocks:
            tokens = blk(tokens, cond)
        summary = tokens[-1]
        out = self.out_head(summary)
        d_mu, d_log_sigma = jnp.split(out, 2, axis=-1)
        new_mu = belief.mu + d_mu
        new_log_sigma = belief.log_sigma + jnp.tanh(d_log_sigma) * 0.5
        return Belief(mu=new_mu, log_sigma=new_log_sigma)


# ----------------------------------------------------------------------------------
# Environment: 2D source-localisation oracle (ActionBED, Appendix G.1, simplified to K sources)
# ----------------------------------------------------------------------------------
class SourceLocConfig(NamedTuple):
    K: int = 1            # number of sources (K=1 is the "simplest version")
    b: float = 0.1         # background offset
    m: float = 1e-4        # stabilising constant
    A: float = 1.0         # signal amplitude
    sigma: float = 0.5     # measurement noise std (on log-intensity)


def source_log_intensity(theta_sources: jnp.ndarray, x: jnp.ndarray, cfg: SourceLocConfig):
    """theta_sources: (K, 2) source positions. x: (2,) sensor location. Returns log mu(theta, x)."""
    sq_dist = jnp.sum((x[None, :] - theta_sources) ** 2, axis=-1)  # (K,)
    mu = cfg.b + jnp.sum(cfg.A / (cfg.m + sq_dist))
    return jnp.log(mu)


def environment_step(theta_sources: jnp.ndarray, x_t: jnp.ndarray, key, cfg: SourceLocConfig):
    """Sample a noisy log-intensity observation y_t in R^1 for design x_t given the true sources."""
    log_mu = source_log_intensity(theta_sources, x_t, cfg)
    noise = cfg.sigma * random.normal(key, ())
    y_t = (log_mu + noise)[None]
    return y_t


def sort_sources_by_amplitude(theta_sources: jnp.ndarray, cfg: SourceLocConfig) -> jnp.ndarray:
    """Canonicalise an unordered set of K sources by sorting on a proxy 'amplitude' (here, the
    intensity contribution at the origin), matching the canonicalisation used by Action-BED to
    make the permutation-invariant downstream loss well defined. theta_sources: (K, 2)."""
    amplitude = cfg.A / (cfg.m + jnp.sum(theta_sources ** 2, axis=-1))
    order = jnp.argsort(-amplitude)
    return theta_sources[order]