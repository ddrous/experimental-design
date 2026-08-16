#%% 1) Imports, configuration, and experiment conventions
"""Version 5 — benchmark alternative ways to train the dimension-agnostic theta embedder.

As in version 2, every training phase owns exactly ONE tqdm progress bar for
its complete optimisation run.  The bar is refreshed in place across epoch boundaries and
uses the same display properties as sequential-bayes-transport.py: dynamic_ncols=True,
leave=True, and mininterval=5.0.  Epoch/validation information is written into the existing
bar instead of creating a fresh bar or printing one training-status line per epoch.

This file is deliberately notebook-style: execute cells from top to bottom and stop after
any method you do not want to run.  It isolates the theta/prior embedding from the full
sequential posterior model and compares four representation-learning objectives:

1. End-to-end reconstruction autoencoder.
2. Variational autoencoder (VAE; the prompt's "VEA") with a Gaussian latent prior.
3. Anchored INR renderer: the shared theta embedder feeds a hypernetwork that predicts
   residual INR weights around a learned anchor; the INR is rendered analytically.
4. LeWM-inspired JEPA two-stage training: first learn ONE shared encoder and predictor end-to-end
   from clean theta / globally rotated theta views using raw latent prediction MSE + SIGReg.
   There is no EMA target encoder and no stop-gradient.  Stage 2 freezes the learned encoder and
   trains the same Transformer reconstruction decoder used by the other abstract-latent models.

The core ThetaDimensionEmbedder below is copied from sequential-bayes-transport.py.  The
visualisation decoder block is also preserved.  One necessary change is made to the decoder:
the original decoder has a fixed number of flat coordinate queries and is explicitly trained
for ONE fixed (S,D).  That cannot test heterogeneous S and D.  Here the same decoder block,
latent cross-attention and scalar head are retained, but its query table is factorised into
(source, coordinate) queries and masked to the active S x D block.  Without this small change,
a single decoder cannot distinguish e.g. flat index 2 meaning (source 2, dim 1) at D=2 from
(source 1, dim 3) at D=3.

The benchmark trains on a subset of (S,D) combinations and reserves several combinations as
held-out SHAPES.  This is stricter than merely mixing heterogeneous padded arrays: it tests
whether the learned representation/decoder compositionality transfers to unseen S-D pairs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from tqdm.auto import tqdm

import seaborn as sns
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

Array = jax.Array


@dataclass(frozen=True)
class EmbeddingBenchmarkConfig:
    # Reproducibility / output.
    seed: int = 2030
    runs_base: str = "./runs"
    run_name: str = "embedding_benchmark"

    # Heterogeneous source configurations.  The original script defaults to only (2,2),
    # which exercises the implementation but does NOT test dimensionality-agnostic training.
    min_num_sources: int = 1
    max_num_sources: int = 6
    min_source_dim: int = 1
    max_source_dim: int = 6
    heldout_shapes: tuple[tuple[int, int], ...] = ((1, 6), (6, 1), (3, 3), (6, 6))
    prior_std: float = 1.0

    # Same TAMO-style theta embedder architecture as the supplied script.
    embedding_dim: int = 192
    dimension_embedder_depth: int = 4
    scalar_encoder_depth: int = 4
    embedding_heads: int = 8
    mlp_ratio: int = 4
    canonicalize_particle_sources: bool = False

    # Same decoder family/hyperparameters as the supplied script.
    decoder_hidden_dim: int = 128
    decoder_heads: int = 8
    decoder_depth: int = 4
    decoder_learning_rate: float = 1e-4
    decoder_batch_size: int = 128
    decoder_epochs: int = 500
    decoder_plateau_patience: int = 500
    decoder_plateau_factor: float = 0.5
    decoder_plateau_cooldown: int = 100
    decoder_plateau_rtol: float = 1e-3
    decoder_plateau_atol: float = 0.0
    decoder_plateau_min_scale: float = 1.0 / 64.0

    # Dataset.  Test data are balanced across ALL shapes, so per-shape comparisons are fair.
    n_train: int = 8192*2
    n_val: int = 2048
    n_test_per_shape: int = 256
    batch_size: int = 128*2

    # End-to-end AE / VAE / anchored-INR optimisation.
    representation_epochs: int = 600
    representation_learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1000.0

    # VAE.  KL is averaged over latent dimensions; a modest beta avoids making the
    # reconstruction benchmark mostly a posterior-collapse benchmark.
    vae_beta: float = 1e-2
    vae_kl_warmup_fraction: float = 0.20

    # Anchored INR renderer.  NOVA uses a deep/narrow 6-layer width-12 INR; we keep that
    # spirit.  The shared E-vector remains the representation because the experiment asks
    # us to keep the current Prior Embedder architecture; a hypernetwork maps E -> INR weights.
    inr_width: int = 12
    inr_hidden_layers: int = 6
    inr_fourier_bands: int = 4
    inr_residual_scale: float = 0.10
    hypernetwork_width: int = 256
    hypernetwork_depth: int = 2

    # LeWM-style JEPA stage 1: one shared encoder/predictor, no EMA and no stop-gradient.
    # The only anti-collapse loss is SIGReg; prediction uses raw latent-space MSE.
    jepa_epochs: int = 600
    jepa_learning_rate: float = 1e-5
    jepa_sigreg_weight: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sigreg_t_max: float = 3.0

    # Legacy Bayesian-augmentation hyperparameters retained verbatim so the experiment
    # configuration is otherwise untouched. The current JEPA geometric augmentation below
    # does not use these values; design_low/design_high still define the valid source grid.
    aug_local_prior_std: float = 0.35
    aug_init_noise_std: float = 0.12
    aug_map_step_size: float = 0.025
    aug_map_steps: int = 2
    aug_observations: int = 2
    design_low: float = -3.0
    design_high: float = 3.0
    background: float = 0.10
    source_strength: float = 1.0
    softening: float = 0.10
    observation_noise_std: float = 0.30

    # Notebook convenience.  Set True before executing expensive cells for a smoke test.
    quick_run: bool = False


CFG = EmbeddingBenchmarkConfig()

# Notebook execution selection. Edit only this list to skip complete methods while
# preserving the cell order below. JEPA means both stage 1 and its stage-2 decoder.
# MODELS_TO_TRAIN = ["AE", "VAE", "Anchored INR", "JEPA"]
MODELS_TO_TRAIN = ["JEPA"]
VALID_MODEL_NAMES = ("AE", "VAE", "Anchored INR", "JEPA")
unknown_models = sorted(set(MODELS_TO_TRAIN) - set(VALID_MODEL_NAMES))
if unknown_models:
    raise ValueError(f"Unknown MODELS_TO_TRAIN entries: {unknown_models}. Valid names: {VALID_MODEL_NAMES}")

if CFG.embedding_dim % CFG.embedding_heads != 0:
    raise ValueError("embedding_dim must be divisible by embedding_heads")
if CFG.decoder_hidden_dim % CFG.decoder_heads != 0:
    raise ValueError("decoder_hidden_dim must be divisible by decoder_heads")
if CFG.max_num_sources * CFG.max_source_dim > CFG.embedding_dim:
    raise ValueError("max S*D must not exceed embedding_dim for this benchmark")

ALL_SHAPES = tuple(
    (s, d)
    for s in range(CFG.min_num_sources, CFG.max_num_sources + 1)
    for d in range(CFG.min_source_dim, CFG.max_source_dim + 1)
)
HELDOUT_SHAPES = tuple(shape for shape in CFG.heldout_shapes if shape in ALL_SHAPES)
TRAIN_SHAPES = tuple(shape for shape in ALL_SHAPES if shape not in HELDOUT_SHAPES)
if not TRAIN_SHAPES:
    raise ValueError("heldout_shapes removed every training shape")

if CFG.quick_run:
    # Keep architecture unchanged; only reduce data/steps.  This block is intentionally
    # obvious and local so it is easy to delete once the smoke test passes.
    N_TRAIN = 512
    N_VAL = 256
    N_TEST_PER_SHAPE = 32
    REPRESENTATION_EPOCHS = 3
    JEPA_EPOCHS = 3
    DECODER_EPOCHS = 8
else:
    N_TRAIN = CFG.n_train
    N_VAL = CFG.n_val
    N_TEST_PER_SHAPE = CFG.n_test_per_shape
    REPRESENTATION_EPOCHS = CFG.representation_epochs
    JEPA_EPOCHS = CFG.jepa_epochs
    DECODER_EPOCHS = CFG.decoder_epochs

stamp = time.strftime("%y%m%d-%H%M%S")
RUN_DIR = Path(CFG.runs_base).expanduser().resolve() / f"{CFG.run_name}_{stamp}"
(RUN_DIR / "plots").mkdir(parents=True, exist_ok=False)
(RUN_DIR / "artefacts").mkdir(parents=True, exist_ok=True)
with (RUN_DIR / "config.json").open("w", encoding="utf-8") as handle:
    json.dump(asdict(CFG), handle, indent=2)

print("Run directory:", RUN_DIR)
print("Training shapes:", TRAIN_SHAPES)
print("Held-out shape combinations:", HELDOUT_SHAPES)
print("All test shapes:", ALL_SHAPES)


#%% 2) Fixed heterogeneous train/validation/test sets and shared visualisation examples
def canonicalize_sources_np(theta: np.ndarray, num_sources: int) -> np.ndarray:
    """Sort active source rows by first coordinate, matching the supplied embedder."""
    theta = np.asarray(theta)
    active = theta[:num_sources]
    order = np.argsort(active[:, 0])
    out = theta.copy()
    out[:num_sources] = active[order]
    return out


def make_theta_dataset(
    rng: np.random.Generator,
    n: int,
    shapes: tuple[tuple[int, int], ...],
    *,
    balanced: bool,
) -> dict[str, np.ndarray]:
    """Sample padded theta arrays; targets are canonicalised because sources are exchangeable."""
    if balanced:
        chosen = np.asarray([shapes[i % len(shapes)] for i in range(n)], dtype=np.int32)
        rng.shuffle(chosen)
    else:
        chosen = np.asarray([shapes[i] for i in rng.integers(0, len(shapes), size=n)], dtype=np.int32)

    theta = np.zeros((n, CFG.max_num_sources, CFG.max_source_dim), dtype=np.float32)
    target = np.zeros_like(theta)
    mask = np.zeros_like(theta, dtype=bool)
    for i, (s, d) in enumerate(chosen):
        active = rng.normal(0.0, CFG.prior_std, size=(int(s), int(d))).astype(np.float32)
        theta[i, :s, :d] = active
        mask[i, :s, :d] = True
        if CFG.canonicalize_particle_sources and s > 1:
            target[i] = canonicalize_sources_np(theta[i], int(s))
        else:
            target[i] = theta[i]

    num_sources = chosen[:, 0].astype(np.int32)
    source_dim = chosen[:, 1].astype(np.int32)
    return {
        "theta": theta,
        "target": target,
        "mask": mask,
        "num_sources": num_sources,
        "source_dim": source_dim,
        "theta_size": (num_sources * source_dim).astype(np.int32),
        "shape_seen": np.asarray(
            [(int(s), int(d)) not in HELDOUT_SHAPES for s, d in chosen], dtype=bool
        ),
    }


train_data = make_theta_dataset(
    np.random.default_rng(CFG.seed + 100), N_TRAIN, TRAIN_SHAPES, balanced=True
)
val_data = make_theta_dataset(
    np.random.default_rng(CFG.seed + 200), N_VAL, TRAIN_SHAPES, balanced=True
)
test_data = make_theta_dataset(
    np.random.default_rng(CFG.seed + 300),
    N_TEST_PER_SHAPE * len(ALL_SHAPES),
    ALL_SHAPES,
    balanced=True,
)

# One deterministic example per selected shape, reused by EVERY method.
# preferred_visual_shapes = ((1, 1), (2, 2), (3, 2), (2, 4), (4, 1), (4, 4))
preferred_visual_shapes = ((1, 1), (2, 2), (3, 2), (2, 4), (4, 1), (4, 4), (6, 6))
visual_shapes = tuple(shape for shape in preferred_visual_shapes if shape in ALL_SHAPES)
visual_indices = []
for shape in visual_shapes:
    hits = np.flatnonzero(
        (test_data["num_sources"] == shape[0]) & (test_data["source_dim"] == shape[1])
    )
    visual_indices.append(int(hits[0]))
visual_indices = np.asarray(visual_indices, dtype=np.int32)
np.save(RUN_DIR / "artefacts" / "visual_indices.npy", visual_indices)

print(f"Train/val/test sizes: {len(train_data['theta'])}/{len(val_data['theta'])}/{len(test_data['theta'])}")
print("Shared visualisation shapes:", visual_shapes)


#%% 3) Shared TAMO-style theta embedder (copied from the supplied script)
def _linear_tokens(layer: eqx.nn.Linear, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _mlp_tokens(layer: eqx.nn.MLP, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _layernorm_tokens(layer: eqx.nn.LayerNorm, x: Array) -> Array:
    return jax.vmap(layer)(x)


def _masked_mean(tokens: Array, valid: Array) -> Array:
    weights = valid.astype(tokens.dtype)[:, None]
    return jnp.sum(tokens * weights, axis=0) / jnp.maximum(jnp.sum(weights), 1.0)


def canonicalize_padded_sources_jax(theta: Array, num_sources: Array) -> Array:
    indices = jnp.arange(theta.shape[-2])
    key = jnp.where(indices < num_sources, theta[..., 0], jnp.inf)
    order = jnp.argsort(key, axis=-1)
    return jnp.take_along_axis(theta, order[..., None], axis=-2)


class DimensionSelfAttentionBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        attn_key, ff1_key, ff2_key = jax.random.split(key, 3)
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=heads,
            query_size=dim,
            key_size=dim,
            value_size=dim,
            output_size=dim,
            dropout_p=0.0,
            key=attn_key,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=ff2_key)

    def __call__(self, tokens: Array, valid: Array) -> Array:
        key_mask = jnp.broadcast_to(valid[None, :], (tokens.shape[0], tokens.shape[0]))
        h = _layernorm_tokens(self.norm1, tokens)
        tokens = tokens + self.attention(h, h, h, mask=key_mask)
        tokens = jnp.where(valid[:, None], tokens, 0.0)
        h = _layernorm_tokens(self.norm2, tokens)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        tokens = tokens + h
        return jnp.where(valid[:, None], tokens, 0.0)


class ThetaDimensionEmbedder(eqx.Module):
    """The supplied dimensionality-agnostic theta embedder, unchanged in substance.

    One caveat worth keeping visible: sorting exchangeable sources by their first coordinate
    makes the map permutation-invariant away from ties, but it is discontinuous when two
    sources exchange order.  That is acceptable for this benchmark because the point is to
    test the current implementation, not silently replace it with a different set encoder.
    """

    scalar_encoder: eqx.nn.MLP
    blocks: tuple[DimensionSelfAttentionBlock, ...]
    final_norm: eqx.nn.LayerNorm
    source_position_pool: Array
    coordinate_position_pool: Array

    max_num_sources: int = eqx.field(static=True)
    max_source_dim: int = eqx.field(static=True)
    max_theta_size: int = eqx.field(static=True)
    prior_std: float = eqx.field(static=True)
    canonicalize: bool = eqx.field(static=True)

    def __init__(self, cfg: EmbeddingBenchmarkConfig, *, key: Array):
        keys = jax.random.split(key, cfg.dimension_embedder_depth + 3)
        E = cfg.embedding_dim
        self.max_num_sources = cfg.max_num_sources
        self.max_source_dim = cfg.max_source_dim
        self.max_theta_size = cfg.max_num_sources * cfg.max_source_dim
        self.prior_std = max(cfg.prior_std, 1e-6)
        self.canonicalize = cfg.canonicalize_particle_sources
        self.scalar_encoder = eqx.nn.MLP(
            in_size=1,
            out_size=E,
            width_size=E,
            depth=cfg.scalar_encoder_depth,
            activation=jax.nn.silu,
            final_activation=jax.nn.silu,
            key=keys[0],
        )
        self.blocks = tuple(
            DimensionSelfAttentionBlock(E, cfg.embedding_heads, cfg.mlp_ratio * E, key=keys[1 + i])
            for i in range(cfg.dimension_embedder_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(E)
        self.source_position_pool = 1.0 + 0.02 * jax.random.normal(
            keys[-2], (self.max_num_sources, E)
        )
        self.coordinate_position_pool = 1.0 + 0.02 * jax.random.normal(
            keys[-1], (self.max_source_dim, E)
        )

    def __call__(self, theta: Array, num_sources: Array, theta_size: Array) -> Array:
        source_dim = theta_size // num_sources
        if self.canonicalize:
            theta = canonicalize_padded_sources_jax(theta, num_sources)

        k = jnp.arange(self.max_theta_size)
        source_index = jnp.clip(k // source_dim, 0, self.max_num_sources - 1)
        coordinate_index = jnp.clip(k % source_dim, 0, self.max_source_dim - 1)
        values = theta[source_index, coordinate_index] / self.prior_std
        valid = k < theta_size
        values = jnp.where(valid, values, 0.0)

        tokens = _mlp_tokens(self.scalar_encoder, values[:, None])
        tokens = jnp.where(valid[:, None], tokens, 0.0)
        for block in self.blocks:
            tokens = block(tokens, valid)
        tokens = _layernorm_tokens(self.final_norm, tokens)
        positions = self.source_position_pool[source_index] * self.coordinate_position_pool[coordinate_index]
        return _masked_mean(tokens * positions, valid)


#%% 4) Same Transformer decoder family, made shape-aware only where required
class VisualizationDecoderBlock(eqx.Module):
    self_norm: eqx.nn.LayerNorm
    cross_query_norm: eqx.nn.LayerNorm
    memory_norm: eqx.nn.LayerNorm
    ff_norm: eqx.nn.LayerNorm
    self_attention: eqx.nn.MultiheadAttention
    cross_attention: eqx.nn.MultiheadAttention
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear

    def __init__(self, dim: int, heads: int, mlp_dim: int, *, key: Array):
        self_key, cross_key, ff1_key, ff2_key = jax.random.split(key, 4)
        self.self_norm = eqx.nn.LayerNorm(dim)
        self.cross_query_norm = eqx.nn.LayerNorm(dim)
        self.memory_norm = eqx.nn.LayerNorm(dim)
        self.ff_norm = eqx.nn.LayerNorm(dim)
        self.self_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=self_key,
        )
        self.cross_attention = eqx.nn.MultiheadAttention(
            num_heads=heads, query_size=dim, key_size=dim, value_size=dim,
            output_size=dim, dropout_p=0.0, key=cross_key,
        )
        self.ff_in = eqx.nn.Linear(dim, mlp_dim, key=ff1_key)
        self.ff_out = eqx.nn.Linear(mlp_dim, dim, key=ff2_key)

    def __call__(self, queries: Array, memory: Array, valid: Array) -> Array:
        # The original fixed-shape decoder needs no mask.  A heterogeneous decoder does:
        # inactive source-coordinate queries must not become attention memory for active ones.
        key_mask = jnp.broadcast_to(valid[None, :], (queries.shape[0], queries.shape[0]))
        h = _layernorm_tokens(self.self_norm, queries)
        queries = queries + self.self_attention(h, h, h, mask=key_mask)
        queries = jnp.where(valid[:, None], queries, 0.0)

        q = _layernorm_tokens(self.cross_query_norm, queries)
        m = _layernorm_tokens(self.memory_norm, memory)
        queries = queries + self.cross_attention(q, m, m)
        queries = jnp.where(valid[:, None], queries, 0.0)

        h = _layernorm_tokens(self.ff_norm, queries)
        h = jax.nn.gelu(_linear_tokens(self.ff_in, h))
        h = _linear_tokens(self.ff_out, h)
        return jnp.where(valid[:, None], queries + h, 0.0)


class HeterogeneousThetaDecoder(eqx.Module):
    """The supplied post-hoc decoder architecture with compositional (source, dim) queries."""

    latent_in: eqx.nn.Linear
    source_queries: Array
    coordinate_queries: Array
    blocks: tuple[VisualizationDecoderBlock, ...]
    final_norm: eqx.nn.LayerNorm
    scalar_head: eqx.nn.Linear
    max_num_sources: int = eqx.field(static=True)
    max_source_dim: int = eqx.field(static=True)

    def __init__(self, cfg: EmbeddingBenchmarkConfig, *, key: Array):
        keys = jax.random.split(key, cfg.decoder_depth + 5)
        self.max_num_sources = cfg.max_num_sources
        self.max_source_dim = cfg.max_source_dim
        self.latent_in = eqx.nn.Linear(cfg.embedding_dim, cfg.decoder_hidden_dim, key=keys[0])
        self.source_queries = 0.02 * jax.random.normal(
            keys[1], (cfg.max_num_sources, cfg.decoder_hidden_dim)
        )
        self.coordinate_queries = 0.02 * jax.random.normal(
            keys[2], (cfg.max_source_dim, cfg.decoder_hidden_dim)
        )
        self.blocks = tuple(
            VisualizationDecoderBlock(
                cfg.decoder_hidden_dim,
                cfg.decoder_heads,
                cfg.mlp_ratio * cfg.decoder_hidden_dim,
                key=keys[3 + i],
            )
            for i in range(cfg.decoder_depth)
        )
        self.final_norm = eqx.nn.LayerNorm(cfg.decoder_hidden_dim)
        self.scalar_head = eqx.nn.Linear(cfg.decoder_hidden_dim, 1, key=keys[-1])

    def __call__(self, embedding: Array, num_sources: Array, theta_size: Array) -> Array:
        source_dim = theta_size // num_sources
        source_idx = jnp.repeat(jnp.arange(self.max_num_sources), self.max_source_dim)
        dim_idx = jnp.tile(jnp.arange(self.max_source_dim), self.max_num_sources)
        valid = (source_idx < num_sources) & (dim_idx < source_dim)

        # Separate source/dimension identities avoid the ambiguous flat-index semantics that
        # the supplied encoder itself warns about when D varies.
        queries = self.source_queries[source_idx] + self.coordinate_queries[dim_idx]
        queries = jnp.where(valid[:, None], queries, 0.0)
        memory = self.latent_in(embedding)[None, :]
        for block in self.blocks:
            queries = block(queries, memory, valid)
        queries = _layernorm_tokens(self.final_norm, queries)
        values = jax.vmap(self.scalar_head)(queries)[:, 0]
        values = jnp.where(valid, values, 0.0)
        return values.reshape(self.max_num_sources, self.max_source_dim)


class ReconstructionAutoencoder(eqx.Module):
    encoder: ThetaDimensionEmbedder
    decoder: HeterogeneousThetaDecoder

    def __init__(self, cfg: EmbeddingBenchmarkConfig, *, key: Array):
        enc_key, dec_key = jax.random.split(key)
        self.encoder = ThetaDimensionEmbedder(cfg, key=enc_key)
        self.decoder = HeterogeneousThetaDecoder(cfg, key=dec_key)

    def __call__(self, theta: Array, num_sources: Array, theta_size: Array) -> tuple[Array, Array]:
        z = self.encoder(theta, num_sources, theta_size)
        return self.decoder(z, num_sources, theta_size), z


class VariationalAutoencoder(eqx.Module):
    encoder: ThetaDimensionEmbedder
    mu_head: eqx.nn.Linear
    logvar_head: eqx.nn.Linear
    decoder: HeterogeneousThetaDecoder

    def __init__(self, cfg: EmbeddingBenchmarkConfig, *, key: Array):
        enc_key, mu_key, logvar_key, dec_key = jax.random.split(key, 4)
        self.encoder = ThetaDimensionEmbedder(cfg, key=enc_key)
        self.mu_head = eqx.nn.Linear(cfg.embedding_dim, cfg.embedding_dim, key=mu_key)
        self.logvar_head = eqx.nn.Linear(cfg.embedding_dim, cfg.embedding_dim, key=logvar_key)
        self.decoder = HeterogeneousThetaDecoder(cfg, key=dec_key)

    def __call__(
        self,
        theta: Array,
        num_sources: Array,
        theta_size: Array,
        key: Array,
        *,
        deterministic: bool = False,
    ) -> tuple[Array, Array, Array, Array]:
        h = self.encoder(theta, num_sources, theta_size)
        mu = self.mu_head(h)
        logvar = jnp.clip(self.logvar_head(h), -10.0, 8.0)
        z = mu if deterministic else mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        return self.decoder(z, num_sources, theta_size), mu, logvar, z


class SIGReg(eqx.Module):
    """Same Epps-Pulley normality regularizer as in the supplied script."""

    knots: int = eqx.field(static=True)
    num_proj: int = eqx.field(static=True)
    t_max: float = eqx.field(static=True)

    def __init__(self, knots: int = 17, num_proj: int = 1024, t_max: float = 3.0):
        self.knots = knots
        self.num_proj = num_proj
        self.t_max = t_max

    def __call__(self, z: Array, key: Array) -> Array:
        T, B, D = z.shape
        A = jax.random.normal(key, (D, self.num_proj))
        A = A / (jnp.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
        t = jnp.linspace(0.0, self.t_max, self.knots)
        dt = self.t_max / (self.knots - 1)
        weights = jnp.full((self.knots,), 2.0 * dt).at[0].set(dt).at[-1].set(dt)
        window = jnp.exp(-0.5 * t**2)
        weights = weights * window
        phi = window
        h = z @ A
        x_t = h[..., None] * t
        ecf_real = jnp.mean(jnp.cos(x_t), axis=1)
        ecf_imag = jnp.mean(jnp.sin(x_t), axis=1)
        err = (ecf_real - phi) ** 2 + ecf_imag**2
        statistic = jnp.einsum("tpk,k->tp", err, weights) * B
        return statistic.mean()


def count_parameters(module: Any) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(module, eqx.is_array)))


def masked_mse(predicted: Array, target: Array, mask: Array) -> Array:
    mask_f = mask.astype(predicted.dtype)
    return jnp.sum((predicted - target) ** 2 * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)


#%% 5) Shared evaluation / plotting helpers
# These are kept deliberately few: the training loops stay explicit in their own cells.
def evaluate_reconstruction(
    name: str,
    predict_batch,
    data: dict[str, np.ndarray],
    *,
    batch_size: int = 256,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    predictions = []
    for start in range(0, len(data["theta"]), batch_size):
        stop = min(start + batch_size, len(data["theta"]))
        predictions.append(
            np.asarray(
                jax.device_get(
                    predict_batch(
                        jnp.asarray(data["theta"][start:stop]),
                        jnp.asarray(data["num_sources"][start:stop]),
                        jnp.asarray(data["theta_size"][start:stop]),
                    )
                )
            )
        )
    pred = np.concatenate(predictions, axis=0)
    error = pred - data["target"]
    active_error = error[data["mask"]]
    active_target = data["target"][data["mask"]]
    mse = float(np.mean(active_error**2))
    mae = float(np.mean(np.abs(active_error)))
    rmse = math.sqrt(mse)
    max_abs = float(np.max(np.abs(active_error)))
    denom = float(np.sum((active_target - np.mean(active_target)) ** 2))
    r2 = 1.0 - float(np.sum(active_error**2)) / max(denom, 1e-12)

    rows = []
    for s, d in ALL_SHAPES:
        select = (data["num_sources"] == s) & (data["source_dim"] == d)
        e = error[select][data["mask"][select]]
        rows.append(
            {
                "method": name,
                "S": s,
                "D": d,
                "seen_shape": (s, d) not in HELDOUT_SHAPES,
                "rmse": float(np.sqrt(np.mean(e**2))),
                "mae": float(np.mean(np.abs(e))),
                "n_samples": int(np.sum(select)),
            }
        )
    per_shape = pd.DataFrame(rows)
    heldout = per_shape.loc[~per_shape["seen_shape"], "rmse"]
    seen = per_shape.loc[per_shape["seen_shape"], "rmse"]
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "r2": r2,
        "mean_seen_shape_rmse": float(seen.mean()) if len(seen) else np.nan,
        "mean_heldout_shape_rmse": float(heldout.mean()) if len(heldout) else np.nan,
    }
    return metrics, per_shape, pred


def plot_diagnostics(
    name: str,
    history: dict[str, list[float]],
    per_shape: pd.DataFrame,
    predictions: np.ndarray,
    *,
    extra_curves: tuple[str, ...] = (),
):
    """Immediate post-training diagnostics; every method uses identical test examples."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    step_loss = np.asarray(history.get("step_loss", []), dtype=float)
    if len(step_loss):
        axes[0, 0].plot(np.arange(1, len(step_loss) + 1), step_loss, linewidth=0.7, label="train step loss")
    if history.get("epoch_val_loss"):
        x = np.arange(1, len(history["epoch_val_loss"]) + 1)
        axes[0, 0].plot(x * max(len(step_loss) / max(len(x), 1), 1), history["epoch_val_loss"], label="validation loss")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Loss after every train step")
    axes[0, 0].set_xlabel("gradient step")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    if history.get("step_grad_norm"):
        axes[0, 1].plot(history["step_grad_norm"], linewidth=0.7, label="grad norm")
    for curve in extra_curves:
        values = history.get(curve, [])
        if len(values):
            axes[0, 1].plot(values, linewidth=0.8, label=curve)
    axes[0, 1].set_title("Optimisation diagnostics")
    axes[0, 1].set_xlabel("gradient step")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    heat = np.full((CFG.max_num_sources, CFG.max_source_dim), np.nan)
    for row in per_shape.itertuples(index=False):
        heat[row.S - 1, row.D - 1] = row.rmse
    im = axes[1, 0].imshow(heat, origin="lower", aspect="auto")
    axes[1, 0].set_xticks(np.arange(CFG.max_source_dim), np.arange(1, CFG.max_source_dim + 1))
    axes[1, 0].set_yticks(np.arange(CFG.max_num_sources), np.arange(1, CFG.max_num_sources + 1))
    axes[1, 0].set_xlabel("D")
    axes[1, 0].set_ylabel("S")
    axes[1, 0].set_title("Test RMSE by shape")
    fig.colorbar(im, ax=axes[1, 0], label="RMSE")
    for s, d in HELDOUT_SHAPES:
        axes[1, 0].text(d - 1, s - 1, "H", ha="center", va="center", fontweight="bold")

    target = test_data["target"]
    mask = test_data["mask"]
    axes[1, 1].scatter(target[mask], predictions[mask], s=7, alpha=0.18)
    lo = float(min(np.min(target[mask]), np.min(predictions[mask])))
    hi = float(max(np.max(target[mask]), np.max(predictions[mask])))
    axes[1, 1].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    axes[1, 1].set_xlabel("true canonical theta")
    axes[1, 1].set_ylabel("reconstruction")
    axes[1, 1].set_title("All active coordinates")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(name, fontsize=15, fontweight="bold")
    fig.savefig(RUN_DIR / "plots" / f"{name.lower().replace(' ', '_')}_diagnostics.png", dpi=170)
    display(fig)
    plt.close(fig)

    # Same exact samples and ordering for every method.
    fig, axes = plt.subplots(len(visual_indices), 1, figsize=(12, 2.4 * len(visual_indices)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, visual_indices):
        s = int(test_data["num_sources"][idx])
        d = int(test_data["source_dim"][idx])
        truth = test_data["target"][idx, :s, :d].reshape(-1)
        recon = predictions[idx, :s, :d].reshape(-1)
        x = np.arange(len(truth))
        ax.plot(x, truth, marker="o", label="true")
        ax.plot(x, recon, marker="x", label="reconstructed")
        ax.set_title(f"fixed test example: S={s}, D={d}" + (" (held-out shape)" if (s, d) in HELDOUT_SHAPES else ""))
        ax.set_ylabel("theta")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
    axes[-1].set_xlabel("canonical flattened active coordinate")
    fig.suptitle(f"{name} — identical reconstruction examples used for every method", fontweight="bold")
    fig.savefig(RUN_DIR / "plots" / f"{name.lower().replace(' ', '_')}_fixed_examples.png", dpi=170)
    display(fig)
    plt.close(fig)


def latent_diagnostics(embeddings: np.ndarray) -> dict[str, float]:
    """Simple collapse/Gaussianity diagnostics; not a substitute for a formal normality test."""
    z = np.asarray(embeddings, dtype=np.float64)
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    centered = z - mean
    covariance = centered.T @ centered / max(len(z) - 1, 1)
    eig = np.linalg.eigvalsh(covariance)
    eig = np.clip(eig, 0.0, None)
    p = eig / max(eig.sum(), 1e-12)
    entropy = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0))
    effective_rank = float(np.exp(entropy))
    offdiag = covariance - np.diag(np.diag(covariance))
    return {
        "latent_mean_abs": float(np.mean(np.abs(mean))),
        "latent_std_mean": float(np.mean(std)),
        "latent_std_abs_error_from_1": float(np.mean(np.abs(std - 1.0))),
        "latent_offdiag_cov_abs": float(np.mean(np.abs(offdiag))),
        "latent_effective_rank": effective_rank,
    }


# Common JAX arrays used in validation cells.
val_jax = {k: jnp.asarray(v) for k, v in val_data.items() if k in {"theta", "target", "mask", "num_sources", "theta_size"}}

RESULTS: dict[str, dict[str, Any]] = {}
PER_SHAPE: list[pd.DataFrame] = []


#%% 6) Method 1 — simple end-to-end reconstruction autoencoder
if "AE" not in MODELS_TO_TRAIN:
    print("[AE] skipped by MODELS_TO_TRAIN")
else:
    # Both encoder and decoder are trained jointly from physical-coordinate reconstruction MSE.
    ae = ReconstructionAutoencoder(CFG, key=jax.random.key(CFG.seed + 1_000))
    ae_optimizer = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip_norm),
        optax.adamw(CFG.representation_learning_rate, weight_decay=CFG.weight_decay),
    )
    ae_state = ae_optimizer.init(eqx.filter(ae, eqx.is_array))


    @eqx.filter_jit
    def ae_train_step(model, state, theta, target, mask, num_sources, theta_size):
        def loss_fn(candidate):
            predicted, _ = jax.vmap(candidate)(theta, num_sources, theta_size)
            return masked_mse(predicted, target, mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        params = eqx.filter(model, eqx.is_array)
        updates, state = ae_optimizer.update(grads, state, params)
        model = eqx.apply_updates(model, updates)
        return model, state, loss, optax.global_norm(eqx.filter(grads, eqx.is_array))


    @eqx.filter_jit
    def ae_predict_batch(theta, num_sources, theta_size):
        predicted, _ = jax.vmap(ae)(theta, num_sources, theta_size)
        return predicted


    ae_history = {"step_loss": [], "step_grad_norm": [], "epoch_val_loss": []}
    ae_rng = np.random.default_rng(CFG.seed + 9_000)
    ae_started = time.time()
    ae_steps_per_epoch = math.ceil(len(train_data["theta"]) / CFG.batch_size)
    ae_progress = tqdm(
        total=REPRESENTATION_EPOCHS * ae_steps_per_epoch,
        desc=f"AE 001/{REPRESENTATION_EPOCHS:03d}",
        dynamic_ncols=True,
        leave=True,
        mininterval=5.0,
    )
    for epoch in range(1, REPRESENTATION_EPOCHS + 1):
        order = ae_rng.permutation(len(train_data["theta"]))
        ae_progress.set_description(f"AE {epoch:03d}/{REPRESENTATION_EPOCHS:03d}", refresh=False)
        for start in range(0, len(order), CFG.batch_size):
            idx = order[start:start + CFG.batch_size]
            ae, ae_state, loss, grad_norm = ae_train_step(
                ae,
                ae_state,
                jnp.asarray(train_data["theta"][idx]),
                jnp.asarray(train_data["target"][idx]),
                jnp.asarray(train_data["mask"][idx]),
                jnp.asarray(train_data["num_sources"][idx]),
                jnp.asarray(train_data["theta_size"][idx]),
            )
            host_loss = float(jax.device_get(loss))
            host_grad_norm = float(jax.device_get(grad_norm))
            ae_history["step_loss"].append(host_loss)
            ae_history["step_grad_norm"].append(host_grad_norm)
            ae_progress.set_postfix(L=f"{host_loss:.4e}", grad=f"{host_grad_norm:.3f}", refresh=False)
            ae_progress.update(1)

        val_pred, _ = jax.vmap(ae)(val_jax["theta"], val_jax["num_sources"], val_jax["theta_size"])
        val_loss = float(jax.device_get(masked_mse(val_pred, val_jax["target"], val_jax["mask"])))
        ae_history["epoch_val_loss"].append(val_loss)
        ae_progress.set_postfix(L=f"{host_loss:.4e}", val=f"{val_loss:.4e}", grad=f"{host_grad_norm:.3f}", refresh=False)
    ae_progress.close()
    ae_time = time.time() - ae_started
    print(f"[AE] training complete | final val MSE={ae_history['epoch_val_loss'][-1]:.6e} | time={ae_time:.1f}s", flush=True)

    # Re-bind predictor after training because `ae` is a new immutable Equinox tree.
    @eqx.filter_jit
    def ae_predict_batch(theta, num_sources, theta_size):
        predicted, _ = jax.vmap(ae)(theta, num_sources, theta_size)
        return predicted


    ae_metrics, ae_shape, ae_predictions = evaluate_reconstruction("AE", ae_predict_batch, test_data)
    ae_embeddings = np.asarray(jax.device_get(jax.vmap(ae.encoder)(
        jnp.asarray(test_data["theta"]), jnp.asarray(test_data["num_sources"]), jnp.asarray(test_data["theta_size"])
    )))
    ae_latent = latent_diagnostics(ae_embeddings)
    print("\nAE test metrics:", {**ae_metrics, **ae_latent, "training_seconds": ae_time})
    plot_diagnostics("AE", ae_history, ae_shape, ae_predictions)
    RESULTS["AE"] = {
        **ae_metrics,
        **ae_latent,
        "training_seconds": ae_time,
        "parameter_count": count_parameters(ae),
        "representation_dim": CFG.embedding_dim,
    }
    PER_SHAPE.append(ae_shape)
    eqx.tree_serialise_leaves(RUN_DIR / "artefacts" / "ae.eqx", ae)

#%% 7) Method 2 — VAE: reconstruction + KL regularisation to N(0,I)
if "VAE" not in MODELS_TO_TRAIN:
    print("[VAE] skipped by MODELS_TO_TRAIN")
else:
    # The supplied theta embedder is the trunk; only small mu/logvar heads are added.
    vae = VariationalAutoencoder(CFG, key=jax.random.key(CFG.seed + 2_000))
    vae_optimizer = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip_norm),
        optax.adamw(CFG.representation_learning_rate, weight_decay=CFG.weight_decay),
    )
    vae_state = vae_optimizer.init(eqx.filter(vae, eqx.is_array))


    @eqx.filter_jit
    def vae_train_step(model, state, theta, target, mask, num_sources, theta_size, key, beta):
        keys = jax.random.split(key, theta.shape[0])

        def loss_fn(candidate):
            predicted, mu, logvar, _ = jax.vmap(candidate)(theta, num_sources, theta_size, keys)
            recon = masked_mse(predicted, target, mask)
            kl = 0.5 * jnp.mean(jnp.exp(logvar) + mu**2 - 1.0 - logvar)
            return recon + beta * kl, (recon, kl)

        (loss, (recon, kl)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        params = eqx.filter(model, eqx.is_array)
        updates, state = vae_optimizer.update(grads, state, params)
        model = eqx.apply_updates(model, updates)
        return model, state, loss, recon, kl, optax.global_norm(eqx.filter(grads, eqx.is_array))


    vae_history = {
        "step_loss": [], "step_reconstruction": [], "step_kl": [],
        "step_grad_norm": [], "epoch_val_loss": []
    }
    vae_rng = np.random.default_rng(CFG.seed + 9_000)
    vae_key = jax.random.key(CFG.seed + 2_020)
    vae_started = time.time()
    vae_global_step = 0
    vae_steps_per_epoch = math.ceil(len(train_data["theta"]) / CFG.batch_size)
    total_vae_steps = max(1, REPRESENTATION_EPOCHS * vae_steps_per_epoch)
    warmup_steps = max(1, int(CFG.vae_kl_warmup_fraction * total_vae_steps))
    vae_progress = tqdm(
        total=total_vae_steps,
        desc=f"VAE 001/{REPRESENTATION_EPOCHS:03d}",
        dynamic_ncols=True,
        leave=True,
        mininterval=5.0,
    )
    for epoch in range(1, REPRESENTATION_EPOCHS + 1):
        order = vae_rng.permutation(len(train_data["theta"]))
        vae_progress.set_description(f"VAE {epoch:03d}/{REPRESENTATION_EPOCHS:03d}", refresh=False)
        for start in range(0, len(order), CFG.batch_size):
            idx = order[start:start + CFG.batch_size]
            vae_key, step_key = jax.random.split(vae_key)
            beta = CFG.vae_beta * min(1.0, (vae_global_step + 1) / warmup_steps)
            vae, vae_state, loss, recon, kl, grad_norm = vae_train_step(
                vae, vae_state,
                jnp.asarray(train_data["theta"][idx]), jnp.asarray(train_data["target"][idx]),
                jnp.asarray(train_data["mask"][idx]), jnp.asarray(train_data["num_sources"][idx]),
                jnp.asarray(train_data["theta_size"][idx]), step_key, jnp.asarray(beta),
            )
            vae_global_step += 1
            host_loss = float(jax.device_get(loss))
            host_recon = float(jax.device_get(recon))
            host_kl = float(jax.device_get(kl))
            host_grad_norm = float(jax.device_get(grad_norm))
            vae_history["step_loss"].append(host_loss)
            vae_history["step_reconstruction"].append(host_recon)
            vae_history["step_kl"].append(host_kl)
            vae_history["step_grad_norm"].append(host_grad_norm)
            vae_progress.set_postfix(
                L=f"{host_loss:.4e}", recon=f"{host_recon:.4e}", KL=f"{host_kl:.3f}",
                grad=f"{host_grad_norm:.3f}", refresh=False,
            )
            vae_progress.update(1)

        val_keys = jax.random.split(jax.random.key(CFG.seed + 2_100), len(val_data["theta"]))
        val_pred, val_mu, val_logvar, _ = jax.vmap(lambda t, s, z, k: vae(t, s, z, k, deterministic=True))(
            val_jax["theta"], val_jax["num_sources"], val_jax["theta_size"], val_keys
        )
        val_recon = masked_mse(val_pred, val_jax["target"], val_jax["mask"])
        val_kl = 0.5 * jnp.mean(jnp.exp(val_logvar) + val_mu**2 - 1.0 - val_logvar)
        val_total = val_recon + CFG.vae_beta * val_kl
        host_val_total = float(jax.device_get(val_total))
        host_val_recon = float(jax.device_get(val_recon))
        host_val_kl = float(jax.device_get(val_kl))
        vae_history["epoch_val_loss"].append(host_val_total)
        vae_progress.set_postfix(
            L=f"{host_loss:.4e}", val=f"{host_val_total:.4e}",
            vRecon=f"{host_val_recon:.4e}", vKL=f"{host_val_kl:.3f}", refresh=False,
        )
    vae_progress.close()
    vae_time = time.time() - vae_started
    print(
        f"[VAE] training complete | final val total={vae_history['epoch_val_loss'][-1]:.6e} | "
        f"recon={host_val_recon:.6e} | KL={host_val_kl:.4f} | time={vae_time:.1f}s",
        flush=True,
    )


    @eqx.filter_jit
    def vae_predict_batch(theta, num_sources, theta_size):
        keys = jax.random.split(jax.random.key(0), theta.shape[0])
        predicted, _, _, _ = jax.vmap(lambda t, s, z, k: vae(t, s, z, k, deterministic=True))(
            theta, num_sources, theta_size, keys
        )
        return predicted


    vae_metrics, vae_shape, vae_predictions = evaluate_reconstruction("VAE", vae_predict_batch, test_data)
    vae_h = jax.vmap(vae.encoder)(
        jnp.asarray(test_data["theta"]), jnp.asarray(test_data["num_sources"]), jnp.asarray(test_data["theta_size"])
    )
    vae_mu = np.asarray(jax.device_get(jax.vmap(vae.mu_head)(vae_h)))
    vae_logvar = np.asarray(jax.device_get(jax.vmap(vae.logvar_head)(vae_h)))
    vae_logvar = np.clip(vae_logvar, -10.0, 8.0)
    vae_eps = np.random.default_rng(CFG.seed + 2_500).normal(size=vae_mu.shape)
    vae_samples = vae_mu + np.exp(0.5 * vae_logvar) * vae_eps
    # Gaussianity is assessed on posterior samples; deterministic reconstruction still uses mu.
    vae_latent = latent_diagnostics(vae_samples)
    vae_mu_rank = latent_diagnostics(vae_mu)["latent_effective_rank"]
    print("\nVAE test metrics:", {**vae_metrics, **vae_latent, "mean_code_effective_rank": vae_mu_rank, "training_seconds": vae_time})
    plot_diagnostics("VAE", vae_history, vae_shape, vae_predictions, extra_curves=("step_reconstruction", "step_kl"))
    RESULTS["VAE"] = {
        **vae_metrics,
        **vae_latent,
        "mean_code_effective_rank": vae_mu_rank,
        "training_seconds": vae_time,
        "parameter_count": count_parameters(vae),
        "representation_dim": CFG.embedding_dim,
    }
    PER_SHAPE.append(vae_shape)
    eqx.tree_serialise_leaves(RUN_DIR / "artefacts" / "vae.eqx", vae)

#%% 8) Method 3 — anchored weight-space / INR reconstruction
if "Anchored INR" not in MODELS_TO_TRAIN:
    print("[Anchored INR] skipped by MODELS_TO_TRAIN")
else:
    # NOVA's core idea is anchor + per-example residual weights and analytical rendering.  Because
    # the experiment explicitly asks to preserve the current E-dimensional Prior Embedder, this is
    # a hypernetwork decoder rather than a literal NOVA latent: z in R^E -> residual INR weights.
    # A literal weight-space latent would remove this hypernetwork and set the representation itself
    # to the INR parameter vector, which would no longer have the same E-dimensional interface.
    def inr_feature_dim() -> int:
        # Fourier features for (source coordinate, dimension coordinate), plus normalised S and D.
        return 4 * CFG.inr_fourier_bands + 2


    INR_LAYER_SIZES = (inr_feature_dim(),) + (CFG.inr_width,) * CFG.inr_hidden_layers + (1,)
    INR_PARAM_COUNT = sum(
        INR_LAYER_SIZES[i] * INR_LAYER_SIZES[i + 1] + INR_LAYER_SIZES[i + 1]
        for i in range(len(INR_LAYER_SIZES) - 1)
    )
    print("Anchored INR layer sizes:", INR_LAYER_SIZES, "| parameter count:", INR_PARAM_COUNT)


    def initialise_inr_anchor(key: Array) -> Array:
        """Flatten a normally initialised INR; the anchor starts as a usable network, not zero."""
        keys = jax.random.split(key, len(INR_LAYER_SIZES) - 1)
        parts = []
        for layer_key, fan_in, fan_out in zip(keys, INR_LAYER_SIZES[:-1], INR_LAYER_SIZES[1:]):
            scale = math.sqrt(2.0 / (fan_in + fan_out))
            weight = scale * jax.random.normal(layer_key, (fan_out, fan_in))
            bias = jnp.zeros((fan_out,))
            parts.extend([weight.reshape(-1), bias])
        return jnp.concatenate(parts)


    def inr_features(source_index: Array, dim_index: Array, num_sources: Array, source_dim: Array) -> Array:
        # Coordinates live in [-1,1], as in the paper; singleton axes map to 0 rather than -1.
        s = jnp.where(num_sources > 1, 2.0 * source_index / jnp.maximum(num_sources - 1, 1) - 1.0, 0.0)
        d = jnp.where(source_dim > 1, 2.0 * dim_index / jnp.maximum(source_dim - 1, 1) - 1.0, 0.0)
        coord = jnp.stack([s, d])
        frequencies = (2.0 ** jnp.arange(CFG.inr_fourier_bands)) * jnp.pi
        angles = coord[:, None] * frequencies[None, :]
        fourier = jnp.concatenate([jnp.sin(angles).reshape(-1), jnp.cos(angles).reshape(-1)])
        shape = jnp.asarray([
            num_sources / CFG.max_num_sources,
            source_dim / CFG.max_source_dim,
        ])
        return jnp.concatenate([fourier, shape])


    def render_flat_inr(flat_weights: Array, num_sources: Array, theta_size: Array) -> Array:
        source_dim = theta_size // num_sources
        params = []
        cursor = 0
        for fan_in, fan_out in zip(INR_LAYER_SIZES[:-1], INR_LAYER_SIZES[1:]):
            w_size = fan_in * fan_out
            weight = flat_weights[cursor:cursor + w_size].reshape(fan_out, fan_in)
            cursor += w_size
            bias = flat_weights[cursor:cursor + fan_out]
            cursor += fan_out
            params.append((weight, bias))

        source_idx = jnp.repeat(jnp.arange(CFG.max_num_sources), CFG.max_source_dim)
        dim_idx = jnp.tile(jnp.arange(CFG.max_source_dim), CFG.max_num_sources)
        valid = (source_idx < num_sources) & (dim_idx < source_dim)

        def render_coordinate(s_idx, d_idx):
            h = inr_features(s_idx, d_idx, num_sources, source_dim)
            for layer_index, (weight, bias) in enumerate(params):
                h = weight @ h + bias
                if layer_index < len(params) - 1:
                    h = jax.nn.silu(h)
            return h[0]

        values = jax.vmap(render_coordinate)(source_idx, dim_idx)
        values = jnp.where(valid, values, 0.0)
        return values.reshape(CFG.max_num_sources, CFG.max_source_dim)


    class AnchoredINRAutoencoder(eqx.Module):
        encoder: ThetaDimensionEmbedder
        hypernetwork: eqx.nn.MLP
        anchor: Array
        residual_scale: float = eqx.field(static=True)

        def __init__(self, cfg: EmbeddingBenchmarkConfig, *, key: Array):
            enc_key, hyper_key, anchor_key = jax.random.split(key, 3)
            self.encoder = ThetaDimensionEmbedder(cfg, key=enc_key)
            hypernetwork = eqx.nn.MLP(
                in_size=cfg.embedding_dim,
                out_size=INR_PARAM_COUNT,
                width_size=cfg.hypernetwork_width,
                depth=cfg.hypernetwork_depth,
                activation=jax.nn.silu,
                final_activation=lambda x: x,
                key=hyper_key,
            )
            # Start exactly at the shared anchor: the hypernetwork initially predicts zero residual.
            hypernetwork = eqx.tree_at(
                lambda model: model.layers[-1].weight,
                hypernetwork,
                jnp.zeros_like(hypernetwork.layers[-1].weight),
            )
            hypernetwork = eqx.tree_at(
                lambda model: model.layers[-1].bias,
                hypernetwork,
                jnp.zeros_like(hypernetwork.layers[-1].bias),
            )
            self.hypernetwork = hypernetwork
            # Learned dataset-level base weights, initialized as a valid Xavier-like INR.
            self.anchor = initialise_inr_anchor(anchor_key)
            self.residual_scale = cfg.inr_residual_scale

        def __call__(self, theta: Array, num_sources: Array, theta_size: Array):
            z = self.encoder(theta, num_sources, theta_size)
            raw_delta = self.hypernetwork(z)
            delta = self.residual_scale * jnp.tanh(raw_delta)
            weights = self.anchor + delta
            reconstruction = render_flat_inr(weights, num_sources, theta_size)
            return reconstruction, z, weights, delta


    wsae = AnchoredINRAutoencoder(CFG, key=jax.random.key(CFG.seed + 3_000))
    wsae_optimizer = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip_norm),
        optax.adamw(CFG.representation_learning_rate, weight_decay=CFG.weight_decay),
    )
    wsae_state = wsae_optimizer.init(eqx.filter(wsae, eqx.is_array))


    @eqx.filter_jit
    def wsae_train_step(model, state, theta, target, mask, num_sources, theta_size):
        def loss_fn(candidate):
            predicted, _, _, delta = jax.vmap(candidate)(theta, num_sources, theta_size)
            recon = masked_mse(predicted, target, mask)
            # Tiny residual penalty keeps the anchor meaningful without overpowering reconstruction.
            residual_penalty = jnp.mean(delta**2)
            return recon + 1e-4 * residual_penalty, (recon, residual_penalty, jnp.mean(jnp.linalg.norm(delta, axis=-1)))

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        params = eqx.filter(model, eqx.is_array)
        updates, state = wsae_optimizer.update(grads, state, params)
        model = eqx.apply_updates(model, updates)
        return model, state, loss, aux, optax.global_norm(eqx.filter(grads, eqx.is_array))


    wsae_history = {
        "step_loss": [], "step_reconstruction": [], "step_residual_norm": [],
        "step_grad_norm": [], "epoch_val_loss": []
    }
    wsae_rng = np.random.default_rng(CFG.seed + 9_000)
    wsae_started = time.time()
    wsae_steps_per_epoch = math.ceil(len(train_data["theta"]) / CFG.batch_size)
    wsae_progress = tqdm(
        total=REPRESENTATION_EPOCHS * wsae_steps_per_epoch,
        desc=f"Anchored INR 001/{REPRESENTATION_EPOCHS:03d}",
        dynamic_ncols=True,
        leave=True,
        mininterval=5.0,
    )
    for epoch in range(1, REPRESENTATION_EPOCHS + 1):
        order = wsae_rng.permutation(len(train_data["theta"]))
        wsae_progress.set_description(
            f"Anchored INR {epoch:03d}/{REPRESENTATION_EPOCHS:03d}", refresh=False
        )
        for start in range(0, len(order), CFG.batch_size):
            idx = order[start:start + CFG.batch_size]
            wsae, wsae_state, loss, aux, grad_norm = wsae_train_step(
                wsae, wsae_state,
                jnp.asarray(train_data["theta"][idx]), jnp.asarray(train_data["target"][idx]),
                jnp.asarray(train_data["mask"][idx]), jnp.asarray(train_data["num_sources"][idx]),
                jnp.asarray(train_data["theta_size"][idx]),
            )
            recon, residual_penalty, residual_norm = aux
            host_loss = float(jax.device_get(loss))
            host_recon = float(jax.device_get(recon))
            host_residual_norm = float(jax.device_get(residual_norm))
            host_grad_norm = float(jax.device_get(grad_norm))
            wsae_history["step_loss"].append(host_loss)
            wsae_history["step_reconstruction"].append(host_recon)
            wsae_history["step_residual_norm"].append(host_residual_norm)
            wsae_history["step_grad_norm"].append(host_grad_norm)
            wsae_progress.set_postfix(
                L=f"{host_loss:.4e}", recon=f"{host_recon:.4e}",
                resid=f"{host_residual_norm:.3f}", grad=f"{host_grad_norm:.3f}", refresh=False,
            )
            wsae_progress.update(1)

        val_pred, _, _, _ = jax.vmap(wsae)(val_jax["theta"], val_jax["num_sources"], val_jax["theta_size"])
        val_loss = float(jax.device_get(masked_mse(val_pred, val_jax["target"], val_jax["mask"])))
        wsae_history["epoch_val_loss"].append(val_loss)
        anchor_norm = float(jax.device_get(jnp.linalg.norm(wsae.anchor)))
        wsae_progress.set_postfix(
            L=f"{host_loss:.4e}", val=f"{val_loss:.4e}", anchor=f"{anchor_norm:.3f}", refresh=False
        )
    wsae_progress.close()
    wsae_time = time.time() - wsae_started
    print(
        f"[Anchored INR] training complete | final val MSE={wsae_history['epoch_val_loss'][-1]:.6e} | "
        f"anchor norm={anchor_norm:.3f} | time={wsae_time:.1f}s",
        flush=True,
    )


    @eqx.filter_jit
    def wsae_predict_batch(theta, num_sources, theta_size):
        predicted, _, _, _ = jax.vmap(wsae)(theta, num_sources, theta_size)
        return predicted


    wsae_metrics, wsae_shape, wsae_predictions = evaluate_reconstruction("Anchored INR", wsae_predict_batch, test_data)
    wsae_embeddings = np.asarray(jax.device_get(jax.vmap(wsae.encoder)(
        jnp.asarray(test_data["theta"]), jnp.asarray(test_data["num_sources"]), jnp.asarray(test_data["theta_size"])
    )))
    wsae_latent = latent_diagnostics(wsae_embeddings)
    print("\nAnchored-INR test metrics:", {**wsae_metrics, **wsae_latent, "training_seconds": wsae_time})
    plot_diagnostics(
        "Anchored INR", wsae_history, wsae_shape, wsae_predictions,
        extra_curves=("step_reconstruction", "step_residual_norm"),
    )
    RESULTS["Anchored INR"] = {
        **wsae_metrics,
        **wsae_latent,
        "training_seconds": wsae_time,
        "parameter_count": count_parameters(wsae),
        "representation_dim": CFG.embedding_dim,
        "inr_parameter_count": INR_PARAM_COUNT,
        "anchor_norm": float(jax.device_get(jnp.linalg.norm(wsae.anchor))),
    }
    PER_SHAPE.append(wsae_shape)
    eqx.tree_serialise_leaves(RUN_DIR / "artefacts" / "anchored_inr.eqx", wsae)

#%% 9) JEPA stage 1 — LeWM-style end-to-end joint embedding + SIGReg
# LeWorldModel's key simplification is used literally here: ONE encoder, ONE predictor,
# no EMA target encoder, no stop-gradient, and only two loss terms:
#
#     L = latent prediction MSE + lambda * SIGReg(embeddings)
#
# LeWM predicts the next temporal embedding.  Our benchmark has no action/time transition at
# this stage, so view 1 is the clean theta itself and view 2 is a rotated version of the SAME
# source configuration.  One rotation matrix acts on every active source.  If that rotation
# leaves the valid grid, its angle is repeatedly halved until the transformed cloud is valid.
# Prediction remains symmetric (view 1 -> view 2 and view 2 -> view 1), with gradients through
# BOTH sides; SIGReg remains the only anti-collapse regulariser.
# def augment_theta(
#     theta: Array,
#     num_sources: Array,
#     theta_size: Array,
#     key: Array,
# ) -> Array:
#     """Rotate all active sources by the same canonical rotation.

#     For D >= 2, rotation is always in the first two coordinate dimensions.
#     For D == 1, theta is returned unchanged.

#     If the proposed rotation moves any active source outside the valid grid,
#     halve the rotation angle repeatedly until the transformed theta is valid.
#     """
#     source_dim = theta_size // num_sources

#     # No meaningful rotation in one dimension.
#     def no_rotation(_):
#         return theta

#     def rotate(_):
#         # # One random initial rotation angle.
#         # angle = jax.random.uniform(
#         #     key,
#         #     (),
#         #     minval=-jnp.pi//8,
#         #     maxval=jnp.pi//8,
#         # )

#         angle = jnp.pi / 4.0 

#         # Masks identifying the actual S x D part of the padded tensor.
#         source_mask = jnp.arange(CFG.max_num_sources) < num_sources
#         dim_mask = jnp.arange(CFG.max_source_dim) < source_dim
#         active = source_mask[:, None] & dim_mask[None, :]

#         def transform(angle):
#             c = jnp.cos(angle)
#             s = jnp.sin(angle)

#             # Canonical rotation: dimensions 0 and 1 only.
#             x = theta[:, 0]
#             y = theta[:, 1]

#             transformed = theta
#             transformed = transformed.at[:, 0].set(c * x - s * y)
#             transformed = transformed.at[:, 1].set(s * x + c * y)

#             # Keep padded entries zero.
#             transformed = jnp.where(active, transformed, 0.0)

#             # Test only active source coordinates.
#             valid = jnp.all(
#                 jnp.where(
#                     active,
#                     (transformed >= CFG.design_low)
#                     & (transformed <= CFG.design_high),
#                     True,
#                 )
#             )

#             return transformed, valid

#         # Try the original angle first.
#         transformed, valid = transform(angle)

#         # If invalid:
#         # alpha -> alpha/2 -> alpha/4 -> ...
#         def cond_fn(state):
#             angle, transformed, valid, iteration = state
#             return (~valid) & (iteration < 20)

#         def body_fn(state):
#             angle, _, _, iteration = state

#             angle = angle / 2.0
#             transformed, valid = transform(angle)

#             return angle, transformed, valid, iteration + 1

#         _, transformed, valid, _ = jax.lax.while_loop(
#             cond_fn,
#             body_fn,
#             (
#                 angle,
#                 transformed,
#                 valid,
#                 jnp.asarray(0),
#             ),
#         )

#         # At angle -> 0 the result approaches the original theta.
#         # This fallback only matters if the original sample itself was
#         # already outside the configured valid grid.
#         return jax.lax.cond(
#             valid,
#             lambda _: transformed,
#             lambda _: theta,
#             operand=None,
#         )

#     return jax.lax.cond(
#         source_dim >= 2,
#         rotate,
#         no_rotation,
#         operand=None,
#     )


def augment_theta(
    theta: Array,
    num_sources: Array,
    theta_size: Array,
    key: Array,
) -> Array:
    """Translate all active sources by the same random D-dimensional vector.

    If the translation moves any active source outside the valid grid,
    repeatedly halve the translation vector until all sources are valid.
    """
    source_dim = theta_size // num_sources

    source_mask = jnp.arange(CFG.max_num_sources) < num_sources
    dim_mask = jnp.arange(CFG.max_source_dim) < source_dim
    active = source_mask[:, None] & dim_mask[None, :]

    # One shared D-dimensional translation vector for the whole source cloud.
    translation = (
        CFG.aug_init_noise_std
        * jax.random.normal(key, (CFG.max_source_dim,))
        * dim_mask
    )

    def transform(shift):
        augmented = theta + shift[None, :] * active

        valid = jnp.all(
            jnp.where(
                active,
                (augmented >= CFG.design_low)
                & (augmented <= CFG.design_high),
                True,
            )
        )

        return jnp.where(active, augmented, 0.0), valid

    augmented, valid = transform(translation)

    def cond_fn(state):
        shift, augmented, valid, iteration = state
        return (~valid) & (iteration < 20)

    def body_fn(state):
        shift, _, _, iteration = state

        shift = shift / 2.0
        augmented, valid = transform(shift)

        return shift, augmented, valid, iteration + 1

    _, augmented, valid, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (
            translation,
            augmented,
            valid,
            jnp.asarray(0),
        ),
    )

    # If theta itself is outside the valid grid, shrinking the translation
    # cannot necessarily produce a valid sample, so fall back to theta.
    return jax.lax.cond(
        valid,
        lambda _: augmented,
        lambda _: theta,
        operand=None,
    )


def plot_jepa_augmentation_examples(n_examples: int = 6) -> None:
    """Plot a few 2-D training thetas beside their deterministic-seed augmentations."""
    candidates = []
    for idx in range(len(train_data["theta"])):
        s = int(train_data["num_sources"][idx])
        d = int(train_data["source_dim"][idx])
        points = train_data["theta"][idx, :s, :d]
        if d == 2 and s >= 2 and np.all(points >= CFG.design_low) and np.all(points <= CFG.design_high):
            candidates.append(idx)
        if len(candidates) == n_examples:
            break

    if not candidates:
        print("[JEPA] no in-grid D=2 samples available for augmentation preview")
        return

    idx = np.asarray(candidates, dtype=np.int32)
    keys = jax.random.split(jax.random.key(CFG.seed + 4_001), len(idx))
    augmented = np.asarray(jax.device_get(jax.vmap(augment_theta)(
        jnp.asarray(train_data["theta"][idx]),
        jnp.asarray(train_data["num_sources"][idx]),
        jnp.asarray(train_data["theta_size"][idx]),
        keys,
    )))

    ncols = min(3, len(idx))
    nrows = math.ceil(len(idx) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.3 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, sample_idx, aug in zip(axes, idx, augmented):
        s = int(train_data["num_sources"][sample_idx])
        original = train_data["theta"][sample_idx, :s, :2]
        transformed = aug[:s, :2]
        ax.scatter(original[:, 0], original[:, 1], marker="o", label="theta")
        ax.scatter(transformed[:, 0], transformed[:, 1], marker="x", label="rotated")
        for p, q in zip(original, transformed):
            ax.plot([p[0], q[0]], [p[1], q[1]], linewidth=0.8, alpha=0.6)
        ax.axhline(0.0, linewidth=0.7, alpha=0.5)
        ax.axvline(0.0, linewidth=0.7, alpha=0.5)
        ax.set_xlim(CFG.design_low, CFG.design_high)
        ax.set_ylim(CFG.design_low, CFG.design_high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"S={s}, D=2")
        ax.grid(alpha=0.25)
        ax.legend()

    for ax in axes[len(idx):]:
        ax.axis("off")

    fig.suptitle("JEPA augmentation preview: clean theta vs shared rotation", fontweight="bold")
    fig.savefig(RUN_DIR / "plots" / "jepa_augmentation_preview.png", dpi=170)
    display(fig)
    plt.close(fig)

if "JEPA" not in MODELS_TO_TRAIN:
    print("[JEPA stage 1] skipped by MODELS_TO_TRAIN")
else:
    # Visual sanity check before any JEPA optimisation begins.
    plot_jepa_augmentation_examples()

    class LeWMStyleJEPA(eqx.Module):
        """Shared end-to-end encoder/predictor with SIGReg; no target network of any kind."""

        encoder: ThetaDimensionEmbedder
        predictor: eqx.nn.MLP
        sigreg: SIGReg

        def __init__(self, cfg: EmbeddingBenchmarkConfig, *, key: Array):
            enc_key, pred_key = jax.random.split(key)
            self.encoder = ThetaDimensionEmbedder(cfg, key=enc_key)
            self.predictor = eqx.nn.MLP(
                in_size=cfg.embedding_dim,
                out_size=cfg.embedding_dim,
                width_size=cfg.embedding_dim,
                depth=2,
                activation=jax.nn.gelu,
                final_activation=lambda x: x,
                key=pred_key,
            )
            self.sigreg = SIGReg(cfg.sigreg_knots, cfg.sigreg_num_proj, cfg.sigreg_t_max)


    jepa_model = LeWMStyleJEPA(CFG, key=jax.random.key(CFG.seed + 4_000))
    jepa_optimizer = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip_norm),
        optax.adamw(CFG.jepa_learning_rate, weight_decay=CFG.weight_decay),
    )
    jepa_state = jepa_optimizer.init(eqx.filter(jepa_model, eqx.is_inexact_array))


    @eqx.filter_jit
    def jepa_train_step(model, state, theta, num_sources, theta_size, key):
        key_view2, key_sigreg = jax.random.split(key)
        keys2 = jax.random.split(key_view2, theta.shape[0])

        # View 1 is exactly the clean sample. View 2 applies one shared geometric
        # rotation matrix to all active sources in that sample.
        view1 = theta
        view2 = jax.vmap(augment_theta)(theta, num_sources, theta_size, keys2)

        def loss_fn(candidate):
            # ONE shared encoder.  Crucially, both z1 and z2 remain inside the autodiff graph.
            z1 = jax.vmap(candidate.encoder)(view1, num_sources, theta_size)
            z2 = jax.vmap(candidate.encoder)(view2, num_sources, theta_size)

            # LeWM uses raw embedding MSE, not cosine/unit-normalised matching.  SIGReg fixes the
            # latent scale/distribution, so normalising here would unnecessarily discard that signal.
            pred12 = jax.vmap(candidate.predictor)(z1)
            pred21 = jax.vmap(candidate.predictor)(z2)
            prediction_loss = 0.5 * (
                jnp.mean((pred12 - z2) ** 2)
                + jnp.mean((pred21 - z1) ** 2)
            )

            # The only anti-collapse regulariser: both view populations should look N(0, I).
            sigreg = candidate.sigreg(jnp.stack([z1, z2], axis=0), key_sigreg)
            loss = prediction_loss + CFG.jepa_sigreg_weight * sigreg

            # Diagnostics only; neither term below contributes to the objective.
            embedding_rmse = jnp.sqrt(jnp.mean((z1 - z2) ** 2))
            source_dim = theta_size // num_sources
            active_mask = (
                (jnp.arange(CFG.max_num_sources)[None, :, None] < num_sources[:, None, None])
                & (jnp.arange(CFG.max_source_dim)[None, None, :] < source_dim[:, None, None])
            )
            view_rmse = jnp.sqrt(masked_mse(view2, theta, active_mask))
            return loss, (prediction_loss, sigreg, embedding_rmse, view_rmse)

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, state = jepa_optimizer.update(grads, state, params)
        model = eqx.apply_updates(model, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_inexact_array))
        return model, state, loss, aux, grad_norm


    jepa_history = {
        "step_loss": [],
        "step_prediction": [],
        "step_sigreg": [],
        "step_embedding_rmse": [],
        "step_view_rmse": [],
        "step_grad_norm": [],
        "epoch_val_loss": [],
        "epoch_val_prediction": [],
        "epoch_val_sigreg": [],
        "epoch_val_embedding_rmse": [],
    }
    jepa_rng = np.random.default_rng(CFG.seed + 9_000)
    jepa_key = jax.random.key(CFG.seed + 4_020)
    jepa_stage1_started = time.time()
    jepa_steps_per_epoch = math.ceil(len(train_data["theta"]) / CFG.batch_size)
    jepa_progress = tqdm(
        total=JEPA_EPOCHS * jepa_steps_per_epoch,
        desc=f"JEPA stage 1 001/{JEPA_EPOCHS:03d}",
        dynamic_ncols=True,
        leave=True,
        mininterval=5.0,
    )

    for epoch in range(1, JEPA_EPOCHS + 1):
        order = jepa_rng.permutation(len(train_data["theta"]))
        jepa_progress.set_description(
            f"JEPA stage 1 {epoch:03d}/{JEPA_EPOCHS:03d}", refresh=False
        )

        for start in range(0, len(order), CFG.batch_size):
            idx = order[start:start + CFG.batch_size]
            jepa_key, step_key = jax.random.split(jepa_key)
            jepa_model, jepa_state, loss, aux, grad_norm = jepa_train_step(
                jepa_model,
                jepa_state,
                jnp.asarray(train_data["theta"][idx]),
                jnp.asarray(train_data["num_sources"][idx]),
                jnp.asarray(train_data["theta_size"][idx]),
                step_key,
            )

            prediction_loss, sigreg, embedding_rmse, view_rmse = aux
            host_loss = float(jax.device_get(loss))
            host_prediction = float(jax.device_get(prediction_loss))
            host_sigreg = float(jax.device_get(sigreg))
            host_embedding_rmse = float(jax.device_get(embedding_rmse))
            host_view_rmse = float(jax.device_get(view_rmse))
            host_grad_norm = float(jax.device_get(grad_norm))

            jepa_history["step_loss"].append(host_loss)
            jepa_history["step_prediction"].append(host_prediction)
            jepa_history["step_sigreg"].append(host_sigreg)
            jepa_history["step_embedding_rmse"].append(host_embedding_rmse)
            jepa_history["step_view_rmse"].append(host_view_rmse)
            jepa_history["step_grad_norm"].append(host_grad_norm)

            jepa_progress.set_postfix(
                L=f"{host_loss:.4e}",
                pred=f"{host_prediction:.4e}",
                SIG=f"{host_sigreg:.2f}",
                zRMSE=f"{host_embedding_rmse:.3f}",
                view=f"{host_view_rmse:.3f}",
                grad=f"{host_grad_norm:.3f}",
                refresh=False,
            )
            jepa_progress.update(1)

        # Validation uses the same end-to-end objective but does not update parameters.
        val_subset = min(len(val_data["theta"]), 512)
        vtheta = jnp.asarray(val_data["theta"][:val_subset])
        vS = jnp.asarray(val_data["num_sources"][:val_subset])
        vsize = jnp.asarray(val_data["theta_size"][:val_subset])
        key2, key_sig = jax.random.split(
            jax.random.key(CFG.seed + 4_100 + epoch)
        )
        vv1 = vtheta
        vv2 = jax.vmap(augment_theta)(
            vtheta, vS, vsize, jax.random.split(key2, val_subset)
        )
        vz1 = jax.vmap(jepa_model.encoder)(vv1, vS, vsize)
        vz2 = jax.vmap(jepa_model.encoder)(vv2, vS, vsize)
        vp12 = jax.vmap(jepa_model.predictor)(vz1)
        vp21 = jax.vmap(jepa_model.predictor)(vz2)

        val_prediction = 0.5 * (
            jnp.mean((vp12 - vz2) ** 2)
            + jnp.mean((vp21 - vz1) ** 2)
        )
        val_sigreg = jepa_model.sigreg(jnp.stack([vz1, vz2], axis=0), key_sig)
        val_embedding_rmse = jnp.sqrt(jnp.mean((vz1 - vz2) ** 2))
        val_loss = val_prediction + CFG.jepa_sigreg_weight * val_sigreg

        host_val_loss = float(jax.device_get(val_loss))
        host_val_prediction = float(jax.device_get(val_prediction))
        host_val_sigreg = float(jax.device_get(val_sigreg))
        host_val_embedding_rmse = float(jax.device_get(val_embedding_rmse))
        jepa_history["epoch_val_loss"].append(host_val_loss)
        jepa_history["epoch_val_prediction"].append(host_val_prediction)
        jepa_history["epoch_val_sigreg"].append(host_val_sigreg)
        jepa_history["epoch_val_embedding_rmse"].append(host_val_embedding_rmse)

        jepa_progress.set_postfix(
            L=f"{host_loss:.4e}",
            val=f"{host_val_loss:.4e}",
            vPred=f"{host_val_prediction:.4e}",
            vSIG=f"{host_val_sigreg:.2f}",
            vzRMSE=f"{host_val_embedding_rmse:.3f}",
            refresh=False,
        )

    jepa_progress.close()
    jepa_stage1_time = time.time() - jepa_stage1_started
    print(
        f"[JEPA stage 1] training complete | final val total={jepa_history['epoch_val_loss'][-1]:.6e} | "
        f"prediction={host_val_prediction:.6e} | SIGReg={host_val_sigreg:.3f} | "
        f"embedding RMSE={host_val_embedding_rmse:.4f} | time={jepa_stage1_time:.1f}s",
        flush=True,
    )

    # Stage-1 diagnostics on CLEAN theta samples.  The decoder has not been trained yet.
    jepa_stage1_embeddings = np.asarray(
        jax.device_get(
            jax.vmap(jepa_model.encoder)(
                jnp.asarray(test_data["theta"]),
                jnp.asarray(test_data["num_sources"]),
                jnp.asarray(test_data["theta_size"]),
            )
        )
    )
    jepa_stage1_latent = latent_diagnostics(jepa_stage1_embeddings)
    print("\nJEPA stage-1 latent diagnostics:", jepa_stage1_latent)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), constrained_layout=True)
    axes[0].plot(jepa_history["step_loss"], linewidth=0.7, label="total")
    axes[0].plot(jepa_history["step_prediction"], linewidth=0.7, label="prediction")
    axes[0].set_yscale("log")
    axes[0].set_title("Stage-1 objective")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(jepa_history["step_sigreg"], linewidth=0.7)
    axes[1].set_title("SIGReg statistic")
    axes[1].grid(alpha=0.25)

    axes[2].plot(jepa_history["step_embedding_rmse"], linewidth=0.7, label="embedding RMSE")
    axes[2].plot(jepa_history["step_view_rmse"], linewidth=0.7, label="theta-view RMSE")
    axes[2].set_title("View invariance diagnostics")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    axes[3].hist(jepa_stage1_embeddings.reshape(-1), bins=80, density=True, alpha=0.75)
    x_grid = np.linspace(-4, 4, 300)
    axes[3].plot(
        x_grid,
        np.exp(-0.5 * x_grid**2) / np.sqrt(2 * np.pi),
        linestyle="--",
        label="N(0,1)",
    )
    axes[3].set_title("Marginal latent values")
    axes[3].legend()
    axes[3].grid(alpha=0.25)
    fig.suptitle(
        "JEPA stage 1 — LeWM-style joint training with SIGReg",
        fontweight="bold",
    )
    fig.savefig(RUN_DIR / "plots" / "jepa_stage1_diagnostics.png", dpi=170)
    display(fig)
    plt.close(fig)

    eqx.tree_serialise_leaves(RUN_DIR / "artefacts" / "jepa_stage1.eqx", jepa_model)

#%% 10) JEPA stage 2 — freeze the jointly trained encoder, train the shared decoder
if "JEPA" not in MODELS_TO_TRAIN:
    print("[JEPA stage 2] skipped by MODELS_TO_TRAIN")
else:
    # Stage 1 is now completely finished.  Stage 2 is intentionally separate: clean theta samples
    # are encoded once by the frozen stage-1 encoder, then ONLY the reconstruction decoder is trained.
    jepa_decoder = HeterogeneousThetaDecoder(CFG, key=jax.random.key(CFG.seed + 5_000))
    jepa_decoder_optimizer = optax.adamw(
        CFG.decoder_learning_rate, weight_decay=CFG.weight_decay
    )
    jepa_decoder_state = jepa_decoder_optimizer.init(
        eqx.filter(jepa_decoder, eqx.is_inexact_array)
    )
    jepa_plateau = optax.contrib.reduce_on_plateau(
        factor=CFG.decoder_plateau_factor,
        patience=CFG.decoder_plateau_patience,
        rtol=CFG.decoder_plateau_rtol,
        atol=CFG.decoder_plateau_atol,
        cooldown=CFG.decoder_plateau_cooldown,
        accumulation_size=1,
        min_scale=CFG.decoder_plateau_min_scale,
    )
    jepa_plateau_state = jepa_plateau.init(
        eqx.filter(jepa_decoder, eqx.is_inexact_array)
    )


    # Encode once because the JEPA encoder is frozen throughout stage 2.
    @eqx.filter_jit
    def jepa_encode_batch(theta, num_sources, theta_size):
        return jax.vmap(jepa_model.encoder)(theta, num_sources, theta_size)


    def encode_numpy_dataset(data: dict[str, np.ndarray]) -> np.ndarray:
        chunks = []
        for start in range(0, len(data["theta"]), 256):
            stop = min(start + 256, len(data["theta"]))
            chunks.append(
                np.asarray(
                    jax.device_get(
                        jepa_encode_batch(
                            jnp.asarray(data["theta"][start:stop]),
                            jnp.asarray(data["num_sources"][start:stop]),
                            jnp.asarray(data["theta_size"][start:stop]),
                        )
                    )
                )
            )
        return np.concatenate(chunks, axis=0)


    jepa_z_train = encode_numpy_dataset(train_data)
    jepa_z_val = encode_numpy_dataset(val_data)


    @eqx.filter_jit
    def jepa_decoder_step(decoder, state, lr_scale, z, target, mask, num_sources, theta_size):
        def loss_fn(candidate):
            predicted = jax.vmap(candidate)(z, num_sources, theta_size)
            return masked_mse(predicted, target, mask)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(decoder)
        params = eqx.filter(decoder, eqx.is_inexact_array)
        raw_updates, state = jepa_decoder_optimizer.update(grads, state, params)
        updates = jax.tree_util.tree_map(
            lambda update: None if update is None else update * lr_scale,
            raw_updates,
        )
        decoder = eqx.apply_updates(decoder, updates)
        grad_norm = optax.global_norm(eqx.filter(grads, eqx.is_inexact_array))
        return decoder, state, loss, grad_norm


    jepa_decoder_history = {
        "step_loss": [],
        "step_grad_norm": [],
        "epoch_val_loss": [],
        "epoch_lr": [],
    }
    jepa_dec_rng = np.random.default_rng(CFG.seed + 5_010)
    jepa_stage2_started = time.time()
    best_jepa_val = float("inf")
    best_jepa_decoder = jepa_decoder
    jepa_decoder_steps_per_epoch = math.ceil(len(jepa_z_train) / CFG.decoder_batch_size)
    jepa_decoder_progress = tqdm(
        total=DECODER_EPOCHS * jepa_decoder_steps_per_epoch,
        desc=f"JEPA decoder 0001/{DECODER_EPOCHS:04d}",
        dynamic_ncols=True,
        leave=True,
        mininterval=5.0,
    )

    for epoch in range(1, DECODER_EPOCHS + 1):
        order = jepa_dec_rng.permutation(len(jepa_z_train))
        lr_scale = jepa_plateau_state.scale
        jepa_decoder_progress.set_description(
            f"JEPA decoder {epoch:04d}/{DECODER_EPOCHS:04d}", refresh=False
        )

        for start in range(0, len(order), CFG.decoder_batch_size):
            idx = order[start:start + CFG.decoder_batch_size]
            jepa_decoder, jepa_decoder_state, loss, grad_norm = jepa_decoder_step(
                jepa_decoder,
                jepa_decoder_state,
                lr_scale,
                jnp.asarray(jepa_z_train[idx]),
                jnp.asarray(train_data["target"][idx]),
                jnp.asarray(train_data["mask"][idx]),
                jnp.asarray(train_data["num_sources"][idx]),
                jnp.asarray(train_data["theta_size"][idx]),
            )
            host_loss = float(jax.device_get(loss))
            host_grad_norm = float(jax.device_get(grad_norm))
            current_lr = CFG.decoder_learning_rate * float(jax.device_get(lr_scale))
            jepa_decoder_history["step_loss"].append(host_loss)
            jepa_decoder_history["step_grad_norm"].append(host_grad_norm)
            jepa_decoder_progress.set_postfix(
                L=f"{host_loss:.4e}",
                lr=f"{current_lr:.3e}",
                grad=f"{host_grad_norm:.3f}",
                refresh=False,
            )
            jepa_decoder_progress.update(1)

        val_pred = jax.vmap(jepa_decoder)(
            jnp.asarray(jepa_z_val), val_jax["num_sources"], val_jax["theta_size"]
        )
        val_loss = float(
            jax.device_get(masked_mse(val_pred, val_jax["target"], val_jax["mask"]))
        )
        jepa_decoder_history["epoch_val_loss"].append(val_loss)
        jepa_decoder_history["epoch_lr"].append(current_lr)

        _, jepa_plateau_state = jepa_plateau.update(
            updates=eqx.filter(jepa_decoder, eqx.is_inexact_array),
            state=jepa_plateau_state,
            value=jnp.asarray(val_loss),
        )
        if val_loss < best_jepa_val:
            best_jepa_val = val_loss
            best_jepa_decoder = jepa_decoder

        jepa_decoder_progress.set_postfix(
            L=f"{host_loss:.4e}",
            val=f"{val_loss:.4e}",
            best=f"{best_jepa_val:.4e}",
            lr=f"{current_lr:.3e}",
            refresh=False,
        )

    jepa_decoder_progress.close()
    jepa_decoder = best_jepa_decoder
    jepa_stage2_time = time.time() - jepa_stage2_started
    print(
        f"[JEPA stage 2] training complete | best val MSE={best_jepa_val:.6e} | "
        f"final lr={jepa_decoder_history['epoch_lr'][-1]:.3e} | time={jepa_stage2_time:.1f}s",
        flush=True,
    )
    jepa_total_time = jepa_stage1_time + jepa_stage2_time


    @eqx.filter_jit
    def jepa_predict_batch(theta, num_sources, theta_size):
        z = jax.vmap(jepa_model.encoder)(theta, num_sources, theta_size)
        return jax.vmap(jepa_decoder)(z, num_sources, theta_size)


    jepa_metrics, jepa_shape, jepa_predictions = evaluate_reconstruction(
        "JEPA", jepa_predict_batch, test_data
    )
    print(
        "\nJEPA final test metrics:",
        {
            **jepa_metrics,
            **jepa_stage1_latent,
            "stage1_seconds": jepa_stage1_time,
            "stage2_seconds": jepa_stage2_time,
            "training_seconds": jepa_total_time,
        },
    )
    plot_diagnostics(
        "JEPA",
        jepa_decoder_history,
        jepa_shape,
        jepa_predictions,
        extra_curves=("epoch_lr",),
    )
    RESULTS["JEPA"] = {
        **jepa_metrics,
        **jepa_stage1_latent,
        "training_seconds": jepa_total_time,
        "stage1_seconds": jepa_stage1_time,
        "stage2_seconds": jepa_stage2_time,
        "parameter_count": count_parameters(jepa_model) + count_parameters(jepa_decoder),
        "representation_dim": CFG.embedding_dim,
    }
    PER_SHAPE.append(jepa_shape)
    eqx.tree_serialise_leaves(RUN_DIR / "artefacts" / "jepa_decoder.eqx", jepa_decoder)

#%% 11) Final side-by-side comparative study
if not RESULTS:
    raise RuntimeError("No models were trained. Add at least one entry to MODELS_TO_TRAIN.")

comparison = pd.DataFrame(RESULTS).T.reset_index(names="method")
comparison = comparison.sort_values("rmse").reset_index(drop=True)
per_shape_comparison = pd.concat(PER_SHAPE, ignore_index=True)
comparison.to_csv(RUN_DIR / "artefacts" / "comparison.csv", index=False)
per_shape_comparison.to_csv(
    RUN_DIR / "artefacts" / "per_shape_comparison.csv", index=False
)
with (RUN_DIR / "artefacts" / "comparison.json").open("w", encoding="utf-8") as handle:
    json.dump(
        {
            name: {
                k: float(v) if np.isscalar(v) else v
                for k, v in values.items()
            }
            for name, values in RESULTS.items()
        },
        handle,
        indent=2,
    )

print("\nFinal comparison (lower reconstruction errors are better):")
display(
    comparison[
        [
            "method",
            "rmse",
            "mae",
            "r2",
            "mean_seen_shape_rmse",
            "mean_heldout_shape_rmse",
            "training_seconds",
            "parameter_count",
            "latent_std_abs_error_from_1",
            "latent_effective_rank",
        ]
    ]
)

# Error / time / held-out generalisation.
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
axes[0].bar(comparison["method"], comparison["rmse"])
axes[0].set_title("Overall reconstruction RMSE")
axes[0].tick_params(axis="x", rotation=25)
axes[0].grid(axis="y", alpha=0.25)
axes[1].bar(comparison["method"], comparison["mean_heldout_shape_rmse"])
axes[1].set_title("Held-out (S,D) shape RMSE")
axes[1].tick_params(axis="x", rotation=25)
axes[1].grid(axis="y", alpha=0.25)
axes[2].bar(comparison["method"], comparison["training_seconds"])
axes[2].set_title("Wall-clock training time")
axes[2].set_ylabel("seconds")
axes[2].tick_params(axis="x", rotation=25)
axes[2].grid(axis="y", alpha=0.25)
fig.suptitle("Dimensionality-agnostic embedding benchmark", fontweight="bold")
fig.savefig(RUN_DIR / "plots" / "final_comparison.png", dpi=180)
display(fig)
plt.close(fig)

# Per-shape comparison: one line per method across the same ordered shape grid.
fig, ax = plt.subplots(figsize=(14, 5.4), constrained_layout=True)
shape_labels = [f"{s}x{d}" for s, d in ALL_SHAPES]
x = np.arange(len(ALL_SHAPES))
for method in comparison["method"]:
    frame = per_shape_comparison[per_shape_comparison["method"] == method]
    rmse_by_shape = []
    for s, d in ALL_SHAPES:
        rmse_by_shape.append(
            float(frame[(frame["S"] == s) & (frame["D"] == d)]["rmse"].iloc[0])
        )
    ax.plot(x, rmse_by_shape, marker="o", label=method)
for i, shape in enumerate(ALL_SHAPES):
    if shape in HELDOUT_SHAPES:
        ax.axvspan(i - 0.4, i + 0.4, alpha=0.08)
ax.set_xticks(x, shape_labels, rotation=45)
ax.set_xlabel("shape SxD (shaded = never seen during representation training)")
ax.set_ylabel("RMSE")
ax.set_title("Reconstruction error across source count and coordinate dimension")
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(RUN_DIR / "plots" / "per_shape_comparison.png", dpi=180)
display(fig)
plt.close(fig)

# Same exact fixed examples in one cross-method plot for visual fairness.
method_predictions = {}
if "AE" in RESULTS:
    method_predictions["AE"] = ae_predictions
if "VAE" in RESULTS:
    method_predictions["VAE"] = vae_predictions
if "Anchored INR" in RESULTS:
    method_predictions["Anchored INR"] = wsae_predictions
if "JEPA" in RESULTS:
    method_predictions["JEPA"] = jepa_predictions
fig, axes = plt.subplots(
    len(visual_indices),
    1,
    figsize=(14, 2.7 * len(visual_indices)),
    constrained_layout=True,
)
axes = np.atleast_1d(axes)
for ax, idx in zip(axes, visual_indices):
    s = int(test_data["num_sources"][idx])
    d = int(test_data["source_dim"][idx])
    truth = test_data["target"][idx, :s, :d].reshape(-1)
    x = np.arange(len(truth))
    ax.plot(x, truth, marker="o", linewidth=2, label="true")
    for method, pred in method_predictions.items():
        ax.plot(x, pred[idx, :s, :d].reshape(-1), marker=".", label=method)
    ax.set_title(
        f"S={s}, D={d}"
        + (" — held-out shape" if (s, d) in HELDOUT_SHAPES else "")
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=8)
axes[-1].set_xlabel("canonical flattened active coordinate")
fig.suptitle(
    "Identical held-out examples: selected methods side by side",
    fontweight="bold",
)
fig.savefig(RUN_DIR / "plots" / "fixed_examples_all_methods.png", dpi=180)
display(fig)
plt.close(fig)

print("\nInterpretation reminders:")
print("- Overall reconstruction asks whether information survives the embedding.")
print("- Held-out-shape RMSE is the stronger test of dimensionality-agnostic compositionality.")
print("- JEPA view 1 is clean theta; view 2 is one shared source-cloud rotation with angle backoff.")
print("- Its only anti-collapse loss is SIGReg; the predictive term is raw latent-space MSE, LeWM-style.")
print("- JEPA stage 1 never sees reconstruction loss. Stage 2 therefore measures how much theta information its invariant latent retained.")
print("- The anchored-INR method is decoder-free only after the hypernetwork has produced INR weights; it is not a literal weight-space latent because E was held fixed by design.")
print("All artefacts saved under:", RUN_DIR)
