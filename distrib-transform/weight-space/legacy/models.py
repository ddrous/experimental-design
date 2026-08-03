"""
models.py
=========

WeightTransformer: a Transformer whose *output* is the trajectory of weights
theta_1 ... theta_T of a small implicit density network (INR), starting from
a random initialisation theta_0. theta_T is decoded into a density p_{theta_T}
that should match the target distribution the conditioning samples x_1..x_T
were drawn from.

Two variants (matching the two attached architecture figures):

1. ``AdaLNWeightTransformer``
   ("AdaLN-conditioned WeightTransformer", assets/adaln_weighttransformer.png)
   A causal Transformer that operates on the weight trajectory itself
   (theta_0 -> theta_1 -> ... -> theta_T as the token sequence). Each block is
   AdaLN-modulated by an encoding of the *conditioning sample* x_t. The
   Transformer predicts weight increments Delta_theta_t, and
   theta_t = theta_0 + cumsum(Delta_theta_{1..t}).

2. ``SampleSeqWeightTransformer``
   ("Sample-sequence WeightTransformer (no AdaLN conditioning)",
   assets/sampleseq_weighttransformer.png)
   A causal Transformer whose *tokens are the samples themselves*
   (e1 = encoder(x1), ..., eT = encoder(xT)); theta_0 is only used as an
   additive/initial anchor. The Transformer directly regresses the sequence
   of predicted weight states theta_1 ... theta_T from the sample tokens,
   with no separate AdaLN conditioning pathway.

Both variants share:
  - the same tiny SIREN-style INR as the decoded density model,
  - the same grid-quadrature density normalisation (`GridDensityINR`),
  - the same MLE loss: L(phi) = sum_t -log p_{theta_t}(x_t).
"""

from __future__ import annotations

import math
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def _split_optional_key(key: Optional[jax.Array], n: int):
    if key is None:
        return (None,) * n
    return tuple(jax.random.split(key, n))


# --------------------------------------------------------------------------------------
# Implicit density network (SIREN INR + grid-quadrature normalisation)
# --------------------------------------------------------------------------------------


class SirenINR(eqx.Module):
    """Small SIREN MLP: R^dim -> R^1, used as an *unnormalised* log-density."""

    layers: list
    base_omega: float = eqx.field(static=True)

    def __init__(self, in_dim: int, width: int, depth: int, key: jax.Array, base_omega: float = 10.0):
        if depth < 1:
            raise ValueError("SIREN depth must be >= 1.")
        keys = jax.random.split(key, depth)
        dims = [in_dim] + [width] * (depth - 1) + [1]
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(depth)]
        self.base_omega = float(base_omega)

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jnp.sin(self.base_omega * x)
        return x[0]


def siren_init(model: SirenINR, key: jax.Array) -> SirenINR:
    """Principled SIREN weight initialisation (Sitzmann et al., 2020)."""
    layer_keys = jax.random.split(key, len(model.layers))
    new_layers = []
    for i, (layer, layer_key) in enumerate(zip(model.layers, layer_keys)):
        fan_in = int(layer.weight.shape[1])
        bound = 1.0 / fan_in if i == 0 else math.sqrt(6.0 / fan_in) / model.base_omega
        new_weight = jax.random.uniform(
            layer_key, layer.weight.shape, minval=-bound, maxval=bound, dtype=layer.weight.dtype
        )
        new_layers.append(eqx.tree_at(lambda lin: lin.weight, layer, new_weight))
    return eqx.tree_at(lambda siren: siren.layers, model, new_layers)


def make_quadrature_grid(domain: float, grid_size: int, dim: int = 2) -> tuple[jax.Array, float]:
    """Fixed regular grid over ``[-domain, domain]^dim`` for numerical normalisation.

    Returns (grid_points of shape (grid_size**dim, dim), cell volume).
    Only practical for small dim (2 here), which matches the 2D visualisation
    use case. Precomputed once and reused for every forward pass.
    """
    if dim != 2:
        raise NotImplementedError("Grid quadrature is implemented for dim=2.")
    axis = jnp.linspace(-domain, domain, grid_size)
    cell = (axis[1] - axis[0]).item() ** 2 if grid_size > 1 else (2 * domain) ** 2
    gx, gy = jnp.meshgrid(axis, axis, indexing="ij")
    points = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
    return points, cell


def log_density_from_params(
    flat_params: jax.Array,
    unravel_fn,
    x: jax.Array,
    quad_points: jax.Array,
    quad_cell: float,
) -> jax.Array:
    """log p_theta(x) for a single point x, given INR weights ``flat_params``.

    p_theta(x) = exp(f_theta(x)) / Z, Z = sum_grid exp(f_theta(g)) * cell_area,
    computed with the log-sum-exp trick for numerical stability.
    """
    inr = unravel_fn(flat_params)
    log_unnorm_x = inr(x)
    log_unnorm_grid = jax.vmap(inr)(quad_points)
    log_z = jax.scipy.special.logsumexp(log_unnorm_grid) + jnp.log(quad_cell)
    return log_unnorm_x - log_z


def density_on_grid(flat_params: jax.Array, unravel_fn, quad_points: jax.Array, quad_cell: float) -> jax.Array:
    """Evaluate the normalised density p_theta on the quadrature grid (for plotting)."""
    inr = unravel_fn(flat_params)
    log_unnorm_grid = jax.vmap(inr)(quad_points)
    log_z = jax.scipy.special.logsumexp(log_unnorm_grid) + jnp.log(quad_cell)
    return jnp.exp(log_unnorm_grid - log_z)


# --------------------------------------------------------------------------------------
# Shared building blocks: sample encoder + AdaLN transformer block
# --------------------------------------------------------------------------------------


class SampleEncoder(eqx.Module):
    """MLP encoder e_phi(x_t) turning a raw sample (R^dim) into a token (R^hidden)."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, dim: int, hidden_dim: int, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.gelu(self.fc1(x)))


def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1 + scale) + shift


class AdaLNBlock(eqx.Module):
    """Causal self-attention + MLP block, AdaLN-zero modulated by a condition vector."""

    attn: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    adaLN: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, key: jax.Array, dropout_rate: float = 0.1):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.attn = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=dim, dropout_p=dropout_rate, key=k1)
        self.norm1 = eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False, eps=1e-6)
        self.norm2 = eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False, eps=1e-6)
        self.fc1 = eqx.nn.Linear(dim, mlp_dim, key=k2)
        self.fc2 = eqx.nn.Linear(mlp_dim, dim, key=k3)
        adaln = eqx.nn.Linear(dim, 6 * dim, key=k4)
        adaln = eqx.tree_at(lambda lin: lin.weight, adaln, jnp.zeros_like(adaln.weight))
        adaln = eqx.tree_at(lambda lin: lin.bias, adaln, jnp.zeros_like(adaln.bias))
        self.adaLN = adaln
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x, c, causal_mask, key, inference):
        k_attn, k_drop = jax.random.split(key, 2)
        modulation = jax.vmap(self.adaLN)(jax.nn.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        hidden = jax.vmap(self.norm1)(x)
        hidden = modulate(hidden, shift_msa, scale_msa)
        attention = self.attn(hidden, hidden, hidden, mask=causal_mask, key=k_attn, inference=inference)
        x = x + gate_msa * attention

        hidden = jax.vmap(self.norm2)(x)
        hidden = modulate(hidden, shift_mlp, scale_mlp)
        hidden = jax.vmap(self.fc1)(hidden)
        hidden = jax.nn.gelu(hidden)
        hidden = self.dropout(hidden, key=k_drop, inference=inference)
        hidden = jax.vmap(self.fc2)(hidden)
        return x + gate_mlp * hidden


class PlainBlock(eqx.Module):
    """Ordinary pre-norm causal self-attention + MLP block (no external conditioning)."""

    attn: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, key: jax.Array, dropout_rate: float = 0.1):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attn = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=dim, dropout_p=dropout_rate, key=k1)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.fc1 = eqx.nn.Linear(dim, mlp_dim, key=k2)
        self.fc2 = eqx.nn.Linear(mlp_dim, dim, key=k3)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x, causal_mask, key, inference):
        k_attn, k_drop = jax.random.split(key, 2)
        hidden = jax.vmap(self.norm1)(x)
        attention = self.attn(hidden, hidden, hidden, mask=causal_mask, key=k_attn, inference=inference)
        x = x + attention

        hidden = jax.vmap(self.norm2)(x)
        hidden = jax.vmap(self.fc1)(hidden)
        hidden = jax.nn.gelu(hidden)
        hidden = self.dropout(hidden, key=k_drop, inference=inference)
        hidden = jax.vmap(self.fc2)(hidden)
        return x + hidden


def _causal_mask(max_len: int) -> jax.Array:
    return jnp.tril(jnp.ones((max_len, max_len), dtype=bool))


# --------------------------------------------------------------------------------------
# Variant 1: AdaLN-conditioned WeightTransformer  (assets/adaln_weighttransformer.png)
# --------------------------------------------------------------------------------------


class AdaLNWeightTransformer(eqx.Module):
    """Causal Transformer over the *weight trajectory*, AdaLN-conditioned on x_t.

    theta_0 (flattened random INR init) is the anchor. At every step t the
    Transformer, conditioned on e_phi(x_t), predicts an update Delta_theta_t;
    theta_t = theta_0 + cumsum(Delta_theta_{1..t}). theta_T decodes the density.
    """

    sample_encoder: SampleEncoder
    theta_in_proj: eqx.nn.Linear
    theta_out_proj: eqx.nn.Linear
    pos_emb: jax.Array
    blocks: list
    final_norm: eqx.nn.LayerNorm
    causal_mask: jax.Array
    theta_dim: int = eqx.field(static=True)

    def __init__(
        self,
        sample_dim: int,
        theta_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        max_len: int,
        dropout_rate: float,
        key: jax.Array,
    ):
        keys = jax.random.split(key, depth + 4)
        self.sample_encoder = SampleEncoder(sample_dim, hidden_dim, keys[0])
        self.theta_in_proj = eqx.nn.Linear(theta_dim, hidden_dim, key=keys[1])
        self.theta_out_proj = eqx.nn.Linear(hidden_dim, theta_dim, key=keys[2])
        self.pos_emb = jax.random.normal(keys[3], (max_len, hidden_dim)) * 0.02
        self.blocks = [
            AdaLNBlock(hidden_dim, num_heads, mlp_dim, keys[4 + i], dropout_rate=dropout_rate)
            for i in range(depth)
        ]
        self.final_norm = eqx.nn.LayerNorm(hidden_dim)
        self.causal_mask = _causal_mask(max_len)
        self.theta_dim = int(theta_dim)

    def __call__(
        self,
        theta0: jax.Array,
        x_seq: jax.Array,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> jax.Array:
        """Returns theta_traj of shape (T, theta_dim): theta_1 ... theta_T."""
        seq_len = x_seq.shape[0]
        if key is None:
            key = jax.random.PRNGKey(0)
        embed_key, *block_keys = jax.random.split(key, len(self.blocks) + 1)

        # Token stream: the *previous* weight state feeds token t (theta_{t-1});
        # AdaLN conditioning at position t comes from x_t.
        theta_prev = jnp.concatenate([theta0[None, :], jnp.zeros((seq_len - 1, self.theta_dim))], axis=0)
        tokens = jax.vmap(self.theta_in_proj)(theta_prev) + self.pos_emb[:seq_len]
        condition = jax.vmap(self.sample_encoder)(x_seq)
        mask = self.causal_mask[:seq_len, :seq_len]

        del embed_key  # no embedding dropout needed for this tiny model
        h = tokens
        for block, bkey in zip(self.blocks, block_keys):
            h = block(h, condition, mask, key=bkey, inference=inference)
        h = jax.vmap(self.final_norm)(h)
        deltas = jax.vmap(self.theta_out_proj)(h)

        # theta_t = theta0 + cumsum(delta_1..delta_t)  (Note box in the figure).
        theta_traj = theta0[None, :] + jnp.cumsum(deltas, axis=0)
        return theta_traj


# --------------------------------------------------------------------------------------
# Variant 2: Sample-sequence WeightTransformer, no AdaLN (assets/sampleseq_weighttransformer.png)
# --------------------------------------------------------------------------------------


class SampleSeqWeightTransformer(eqx.Module):
    """Causal Transformer whose tokens *are* the encoded samples e_1 .. e_T.

    theta_0 only serves as an initial weight anchor added to every predicted
    weight state; there is no separate AdaLN conditioning pathway.
    """

    sample_encoder: SampleEncoder
    theta0_proj: eqx.nn.Linear
    theta_out_proj: eqx.nn.Linear
    pos_emb: jax.Array
    blocks: list
    final_norm: eqx.nn.LayerNorm
    causal_mask: jax.Array
    theta_dim: int = eqx.field(static=True)

    def __init__(
        self,
        sample_dim: int,
        theta_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        max_len: int,
        dropout_rate: float,
        key: jax.Array,
    ):
        keys = jax.random.split(key, depth + 4)
        self.sample_encoder = SampleEncoder(sample_dim, hidden_dim, keys[0])
        self.theta0_proj = eqx.nn.Linear(theta_dim, hidden_dim, key=keys[1])
        self.theta_out_proj = eqx.nn.Linear(hidden_dim, theta_dim, key=keys[2])
        self.pos_emb = jax.random.normal(keys[3], (max_len, hidden_dim)) * 0.02
        self.blocks = [
            PlainBlock(hidden_dim, num_heads, mlp_dim, keys[4 + i], dropout_rate=dropout_rate)
            for i in range(depth)
        ]
        self.final_norm = eqx.nn.LayerNorm(hidden_dim)
        self.causal_mask = _causal_mask(max_len)
        self.theta_dim = int(theta_dim)

    def __call__(
        self,
        theta0: jax.Array,
        x_seq: jax.Array,
        key: Optional[jax.Array] = None,
        inference: bool = False,
    ) -> jax.Array:
        """Returns theta_traj of shape (T, theta_dim): theta_1 ... theta_T."""
        seq_len = x_seq.shape[0]
        if key is None:
            key = jax.random.PRNGKey(0)
        block_keys = jax.random.split(key, len(self.blocks))

        tokens = jax.vmap(self.sample_encoder)(x_seq) + self.pos_emb[:seq_len]
        mask = self.causal_mask[:seq_len, :seq_len]

        h = tokens
        for block, bkey in zip(self.blocks, block_keys):
            h = block(h, mask, key=bkey, inference=inference)
        h = jax.vmap(self.final_norm)(h)
        weight_updates = jax.vmap(self.theta_out_proj)(h)

        # theta0 used as an additive anchor for every predicted state (per the figure's
        # "used as initial weight anchor" annotation), not accumulated like variant 1.
        theta_traj = theta0[None, :] + weight_updates
        return theta_traj


# --------------------------------------------------------------------------------------
# Full model: wraps a WeightTransformer variant + the decoded density INR
# --------------------------------------------------------------------------------------

VARIANTS = {
    "adaln": AdaLNWeightTransformer,
    "sampleseq": SampleSeqWeightTransformer,
}


class WeightTransformerDensityModel(eqx.Module):
    """Ties a WeightTransformer variant to the SIREN density decoder + quadrature grid."""

    predictor: eqx.Module
    theta0: jax.Array
    unravel_fn: callable = eqx.field(static=True)
    quad_points: jax.Array
    quad_cell: float = eqx.field(static=True)
    variant: str = eqx.field(static=True)
    sample_dim: int = eqx.field(static=True)

    def __init__(self, cfg: dict, key: jax.Array):
        k_inr, k_init, k_pred = jax.random.split(key, 3)
        dim = int(cfg["sample_dim"])

        template_inr = SirenINR(
            in_dim=dim,
            width=int(cfg["inr_width"]),
            depth=int(cfg["inr_depth"]),
            key=k_inr,
            base_omega=float(cfg.get("inr_base_omega", 10.0)),
        )
        template_inr = siren_init(template_inr, k_init)
        flat_theta0, unravel_fn = ravel_pytree(template_inr)
        theta_dim = int(flat_theta0.shape[0])

        variant = cfg.get("variant", "adaln")
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Options: {list(VARIANTS)}")
        predictor_cls = VARIANTS[variant]

        self.predictor = predictor_cls(
            sample_dim=dim,
            theta_dim=theta_dim,
            hidden_dim=int(cfg["predictor_hidden_dim"]),
            depth=int(cfg["predictor_depth"]),
            num_heads=int(cfg["predictor_heads"]),
            mlp_dim=int(cfg["predictor_mlp_dim"]),
            max_len=int(cfg["seq_len"]),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
            key=k_pred,
        )
        self.theta0 = flat_theta0
        self.unravel_fn = unravel_fn

        quad_points, quad_cell = make_quadrature_grid(
            domain=float(cfg.get("quad_domain", 4.0)),
            grid_size=int(cfg.get("quad_grid_size", 48)),
            dim=dim,
        )
        self.quad_points = quad_points
        self.quad_cell = quad_cell
        self.variant = variant
        self.sample_dim = dim

    def theta_trajectory(self, x_seq: jax.Array, key: Optional[jax.Array] = None, inference: bool = False):
        """(seq_len, theta_dim) weight trajectory theta_1 ... theta_T."""
        return self.predictor(self.theta0, x_seq, key=key, inference=inference)

    def sequence_log_probs(self, x_seq: jax.Array, key: Optional[jax.Array] = None, inference: bool = False):
        """log p_{theta_t}(x_t) for every t -- the per-step MLE terms."""
        theta_traj = self.theta_trajectory(x_seq, key=key, inference=inference)

        def per_step(theta_t, x_t):
            return log_density_from_params(theta_t, self.unravel_fn, x_t, self.quad_points, self.quad_cell)

        return jax.vmap(per_step)(theta_traj, x_seq), theta_traj

    def density_grid(self, theta_t: jax.Array) -> jax.Array:
        """Normalised density values on the quadrature grid, for plotting."""
        return density_on_grid(theta_t, self.unravel_fn, self.quad_points, self.quad_cell)


def sequence_nll(model: WeightTransformerDensityModel, x_seq: jax.Array, key=None, inference: bool = False):
    """L(phi) = sum_t -log p_{theta_t}(x_t), averaged over the sequence."""
    log_probs, theta_traj = model.sequence_log_probs(x_seq, key=key, inference=inference)
    return -jnp.mean(log_probs), theta_traj


def batch_nll(model: WeightTransformerDensityModel, x_batch: jax.Array, key: jax.Array, inference: bool = False):
    """Mean NLL over a batch of (seq_len, dim) sample sequences."""
    keys = jax.random.split(key, x_batch.shape[0])
    losses, _ = jax.vmap(lambda x_seq, k: sequence_nll(model, x_seq, key=k, inference=inference))(x_batch, keys)
    return jnp.mean(losses)