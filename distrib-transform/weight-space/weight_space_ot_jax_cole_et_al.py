# %% [markdown]
# Amortised optimal transport for Gaussian and MNIST generation
# JAX + Equinox + Optax
#
# This file is organised as a notebook-style Python script. Editors such as
# VS Code, Spyder, and PyCharm recognise ``# %%`` cell boundaries.
#
# Experiments
# -----------
# 1. Gaussian -> Gaussian. A finite-dimensional, weight-conditioned transport
#    network is trained against the exact diagonal-Gaussian Brenier map.
#
# 2. Gaussian -> MNIST generation. A convolutional autoencoder first learns a
#    four-dimensional latent representation of MNIST. A single conditional
#    transport model is then trained to push N(0, I_4) to each digit-class
#    latent distribution. In the default ``prompt`` mode, target examples from
#    a digit class form an empirical-measure prompt, closely following Cole et
#    al. In the optional ``class`` mode, a one-hot class vector is supplied
#    directly; this is the finite-dimensional pullback / vector-space version
#    motivated by the accompanying theory.
#
# The transport loss is the soft-constrained Monge objective
#
#     E ||T(x)-x||^2 + lambda * MMD^2(T#rho_0, rho_1),
#
# computed in the learned latent space. Generated latent samples are decoded
# into actual 28x28 MNIST digits. This is intentionally not a pixel-grid mass
# transport experiment.
#
# Install a JAX build appropriate for your hardware, then:
#     pip install equinox optax numpy matplotlib
#
# The paper-scale cross-attention width is 2048. The default below is smaller
# so that the code is usable on ordinary GPUs; set hidden_dim=2048 to match the
# reported architecture more closely.

# %%
# -----------------------------------------------------------------------------
# Configuration. All user-facing hyperparameters are collected here.
# -----------------------------------------------------------------------------
CONFIG = {
    "global": {
        "seed": 2026,
        "run": ["gaussian", "mnist_generation"],
        "output_dir": "./amortised_ot_outputs",
        "quick_test": False,
        "enable_x64": False,
        "jax_platform": None,            # None, "cpu", "gpu", or "tpu"
        "preallocate_gpu_memory": False,
        "print_model": True,
        "run_shape_smoke_tests": True,
    },
    "gaussian": {
        "dimension": 2,
        "epochs": 50,
        "steps_per_epoch": 100,
        "batch_tasks": 64,
        "samples_per_task": 64,
        "validation_batches": 8,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-6,
        "gradient_clip": 1.0,
        "source_mean_range": [-1.5, 1.5],
        "target_mean_range": [-1.5, 1.5],
        "source_std_range": [0.35, 1.25],
        "target_std_range": [0.35, 1.25],
        "context_dim": 64,
        "encoder_width": 128,
        "encoder_depth": 2,
        "fusion_width": 192,
        "fusion_depth": 3,
        "residual_scale": 1.0,
        "loss_weights": {"supervised_map": 1.0, "rff_mmd": 0.10, "displacement": 0.0},
        "rff": {"bandwidths": [0.25, 0.5, 1.0, 2.0], "features_per_bandwidth": 64},
        "log_every_steps": 100,
        "visualisation_tasks": 4,
        "visualisation_samples": 800,
    },
    "mnist": {
        "data_dir": "./data/mnist_raw",
        "digit_classes": list(range(10)),
        "max_train_tasks": 55000,
        "max_validation_tasks": 5000,
        "max_test_tasks": 10000,
        "autoencoder": {
            "latent_dim": 4,
            "epochs": 100,
            "batch_size": 128,
            "validation_batches": 20,
            "learning_rate": 1.0e-3,
            "end_learning_rate": 1.0e-5,
            "warmup_steps": 500,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "groups": 8,
            "checkpoint_every_epochs": 10,
            "reuse_checkpoint": True,
            "train_if_checkpoint_missing": True,
            "encode_chunk_size": 512,
        },
        "transport": {
            # "prompt" reproduces prompt-conditioned generation; "class" is
            # the finite-dimensional vector-space ablation.
            "conditioning_mode": "prompt",
            "epochs": 3000,
            "steps_per_epoch": 50,
            "prompt_size": 64,
            "samples_per_task": 128,
            "validation_repeats_per_class": 2,
            "hidden_dim": 256,            # Cole et al. report 2048
            "num_heads": 4,
            "self_attention_blocks": 2,
            "mlp_ratio": 2,
            "residual_output_scale": 0.10,
            "learning_rate": 3.0e-5,
            "end_learning_rate": 3.0e-6,
            "warmup_steps": 500,
            "weight_decay": 1.0e-6,
            "gradient_clip": 1.0,
            "mmd_estimator_for_training": "biased",  # stable and nonnegative
            "loss_weights": {
                "displacement": 1.0,
                "mmd": 50.0,
                "moments": 0.10,
            },
            "mmd": {
                "num_scales": 5,
                "scale_ratio": 2.0,
                "minimum_bandwidth": 1.0e-4,
            },
            "standardise_latents": True,
            "latent_std_floor": 1.0e-3,
            "checkpoint_every_epochs": 100,
            "reuse_checkpoint": True,
            "train_if_checkpoint_missing": True,
            "log_every_steps": 100,
        },
        "visualisation": {
            "reconstruction_examples": 16,
            "generated_rows": 4,
            "prompt_examples_per_class": 1,
            "latent_scatter_samples": 500,
            "selected_scatter_digits": [0, 1, 4, 7],
        },
    },
}

# %%
# -----------------------------------------------------------------------------
# Environment setup and imports.
# -----------------------------------------------------------------------------
import copy
import gzip
import hashlib
import json
import math
import os
import struct
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

if not CONFIG["global"]["preallocate_gpu_memory"]:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if CONFIG["global"]["jax_platform"] is not None:
    os.environ.setdefault("JAX_PLATFORM_NAME", str(CONFIG["global"]["jax_platform"]))

try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import optax
except ImportError as exc:
    raise ImportError(
        "Missing dependency. Install a JAX build appropriate for your hardware, "
        "then install equinox, optax, numpy, and matplotlib."
    ) from exc

jax.config.update("jax_enable_x64", bool(CONFIG["global"]["enable_x64"]))
Array = jax.Array
PyTree = Any

# %%
# -----------------------------------------------------------------------------
# General utilities.
# -----------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = value
    return result


def apply_quick_test_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    if not config["global"]["quick_test"]:
        return config
    return deep_update(
        config,
        {
            "gaussian": {
                "epochs": 2,
                "steps_per_epoch": 4,
                "batch_tasks": 8,
                "samples_per_task": 16,
                "validation_batches": 2,
                "visualisation_samples": 100,
            },
            "mnist": {
                "max_train_tasks": 1024,
                "max_validation_tasks": 256,
                "max_test_tasks": 256,
                "autoencoder": {
                    "epochs": 2,
                    "batch_size": 32,
                    "validation_batches": 2,
                    "warmup_steps": 2,
                    "encode_chunk_size": 64,
                    "reuse_checkpoint": False,
                },
                "transport": {
                    "epochs": 2,
                    "steps_per_epoch": 10,
                    "prompt_size": 16,
                    "samples_per_task": 32,
                    "validation_repeats_per_class": 1,
                    "hidden_dim": 64,
                    "self_attention_blocks": 1,
                    "warmup_steps": 2,
                    "reuse_checkpoint": False,
                    "log_every_steps": 1,
                },
                "visualisation": {
                    "generated_rows": 2,
                    "latent_scatter_samples": 50,
                },
            },
        },
    )


CONFIG = apply_quick_test_overrides(CONFIG)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, sort_keys=True)


def tree_parameter_count(model: PyTree) -> int:
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return int(sum(int(np.prod(x.shape)) for x in leaves if x is not None))


def differentiable_tree_global_norm(tree: PyTree) -> Array:
    """Global L2 norm over floating-point array leaves, ignoring static/None leaves."""
    leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def scalar_metrics_to_python(metrics: Mapping[str, Array]) -> Dict[str, float]:
    host_metrics = jax.device_get(metrics)
    return {name: float(np.asarray(value)) for name, value in host_metrics.items()}


def average_metric_dicts(metric_dicts: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not metric_dicts:
        raise ValueError("Cannot average an empty metric sequence.")
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}


def append_step_history(history: Dict[str, List[float]], step: int, metrics: Mapping[str, float]) -> None:
    history.setdefault("step", []).append(int(step))
    for name, value in metrics.items():
        history.setdefault(name, []).append(float(value))


def append_epoch_history(
    history: Dict[str, List[float]],
    epoch: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
) -> None:
    history.setdefault("epoch", []).append(int(epoch))
    for name, value in train_metrics.items():
        history.setdefault(f"train_{name}", []).append(float(value))
    for name, value in validation_metrics.items():
        history.setdefault(f"validation_{name}", []).append(float(value))


def save_histories(
    step_history: Mapping[str, Sequence[float]],
    epoch_history: Mapping[str, Sequence[float]],
    output_path: str | Path,
) -> None:
    arrays: Dict[str, np.ndarray] = {}
    arrays.update({f"step_{k}": np.asarray(v) for k, v in step_history.items()})
    arrays.update({f"epoch_{k}": np.asarray(v) for k, v in epoch_history.items()})
    np.savez_compressed(output_path, **arrays)


def make_optimizer(config: Mapping[str, Any]) -> optax.GradientTransformation:
    transforms: List[optax.GradientTransformation] = []
    clip = float(config.get("gradient_clip", 0.0))
    if clip > 0.0:
        transforms.append(optax.clip_by_global_norm(clip))
    transforms.append(
        optax.adamw(
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    )
    return optax.chain(*transforms)


def make_scheduled_optimizer(
    config: Mapping[str, Any], total_steps: int
) -> Tuple[optax.GradientTransformation, Any]:
    warmup_steps = min(int(config.get("warmup_steps", 0)), max(total_steps - 1, 0))
    peak = float(config["learning_rate"])
    end = float(config.get("end_learning_rate", peak * 0.1))
    if total_steps <= 1:
        schedule = optax.constant_schedule(peak)
    elif warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=end,
        )
    else:
        alpha = max(0.0, min(1.0, end / peak))
        schedule = optax.cosine_decay_schedule(peak, decay_steps=total_steps, alpha=alpha)
    transforms: List[optax.GradientTransformation] = []
    clip = float(config.get("gradient_clip", 0.0))
    if clip > 0.0:
        transforms.append(optax.clip_by_global_norm(clip))
    transforms.append(
        optax.adamw(
            learning_rate=schedule,
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    )
    return optax.chain(*transforms), schedule


def print_runtime_information() -> None:
    print("JAX version:", jax.__version__)
    print("Equinox version:", getattr(eqx, "__version__", "unknown"))
    print("Optax version:", getattr(optax, "__version__", "unknown"))
    print("JAX devices:", jax.devices())

# %%
# -----------------------------------------------------------------------------
# Model: branch encoders for source and target weights, point encoder, fusion MLP.
# -----------------------------------------------------------------------------
def _scaled_last_layer(mlp: eqx.nn.MLP, scale: float) -> eqx.nn.MLP:
    """Scale the final affine layer for a near-identity residual initialisation."""
    last = mlp.layers[-1]
    mlp = eqx.tree_at(
        lambda module: module.layers[-1].weight,
        mlp,
        last.weight * jnp.asarray(scale, dtype=last.weight.dtype),
    )
    if last.bias is not None:
        mlp = eqx.tree_at(
            lambda module: module.layers[-1].bias,
            mlp,
            jnp.zeros_like(last.bias),
        )
    return mlp


class WeightConditionedTransport(eqx.Module):
    """A finite-dimensional pullback of a transport operator.

    The model evaluates T(source_weights, target_weights, x). The target_weights
    can be Gaussian parameters, a neural-density parameter vector, or—as in the
    MNIST experiment—a 784-dimensional probability-mass vector.
    """

    source_encoder: eqx.nn.MLP
    target_encoder: eqx.nn.MLP
    point_encoder: eqx.nn.MLP
    fusion: eqx.nn.MLP
    output_mode: str = eqx.field(static=True)
    residual_scale: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_dim: int,
        target_dim: int,
        point_dim: int,
        output_dim: int,
        context_dim: int,
        source_width: int,
        target_width: int,
        point_width: int,
        encoder_depth: int,
        fusion_width: int,
        fusion_depth: int,
        output_mode: str,
        residual_scale: float,
        key: Array,
    ):
        if output_mode not in {"unbounded_residual", "tanh_residual", "tanh_direct"}:
            raise ValueError(f"Unsupported output_mode={output_mode!r}.")
        keys = jr.split(key, 4)
        activation = jax.nn.silu
        self.source_encoder = eqx.nn.MLP(
            in_size=source_dim,
            out_size=context_dim,
            width_size=source_width,
            depth=encoder_depth,
            activation=activation,
            key=keys[0],
        )
        self.target_encoder = eqx.nn.MLP(
            in_size=target_dim,
            out_size=context_dim,
            width_size=target_width,
            depth=encoder_depth,
            activation=activation,
            key=keys[1],
        )
        self.point_encoder = eqx.nn.MLP(
            in_size=point_dim,
            out_size=context_dim,
            width_size=point_width,
            depth=encoder_depth,
            activation=activation,
            key=keys[2],
        )
        fusion = eqx.nn.MLP(
            in_size=3 * context_dim,
            out_size=output_dim,
            width_size=fusion_width,
            depth=fusion_depth,
            activation=activation,
            key=keys[3],
        )
        self.fusion = _scaled_last_layer(fusion, scale=0.05)
        self.output_mode = output_mode
        self.residual_scale = float(residual_scale)

    def __call__(self, source_weights: Array, target_weights: Array, x: Array) -> Array:
        source_context = self.source_encoder(source_weights)
        target_context = self.target_encoder(target_weights)
        point_context = self.point_encoder(x)
        fused = jnp.concatenate((source_context, target_context, point_context), axis=0)
        delta = self.fusion(fused)
        if self.output_mode == "unbounded_residual":
            return x + self.residual_scale * delta
        if self.output_mode == "tanh_residual":
            return jnp.tanh(x + self.residual_scale * delta)
        return jnp.tanh(delta)


def apply_transport_batch(
    model: WeightConditionedTransport,
    source_weights: Array,
    target_weights: Array,
    points: Array,
) -> Array:
    """Apply a single-point Equinox model to [tasks, points, dimension]."""

    def apply_one_task(source: Array, target: Array, task_points: Array) -> Array:
        return jax.vmap(lambda point: model(source, target, point))(task_points)

    return jax.vmap(apply_one_task)(source_weights, target_weights, points)


# %%
# -----------------------------------------------------------------------------
# Random Fourier features and sliced Wasserstein utilities.
# -----------------------------------------------------------------------------
class RFFParameters(NamedTuple):
    omega: Array  # [features, dimension]
    phase: Array  # [features]


def make_rff_parameters(
    key: Array,
    dimension: int,
    bandwidths: Sequence[float],
    features_per_bandwidth: int,
) -> RFFParameters:
    if features_per_bandwidth <= 0:
        raise ValueError("features_per_bandwidth must be positive.")
    keys = jr.split(key, len(bandwidths) + 1)
    omega_parts = []
    for bandwidth, part_key in zip(bandwidths, keys[:-1]):
        if bandwidth <= 0:
            raise ValueError("Every RFF bandwidth must be positive.")
        omega_parts.append(
            jr.normal(part_key, (features_per_bandwidth, dimension)) / float(bandwidth)
        )
    omega = jnp.concatenate(omega_parts, axis=0)
    phase = jr.uniform(keys[-1], (omega.shape[0],), minval=0.0, maxval=2.0 * jnp.pi)
    return RFFParameters(omega=omega, phase=phase)


def rff_features(points: Array, rff: RFFParameters) -> Array:
    """Return features with the convention E[phi]^2 approximates a Gaussian MMD."""
    scale = jnp.sqrt(jnp.asarray(2.0 / rff.omega.shape[0], dtype=points.dtype))
    return scale * jnp.cos(jnp.einsum("...d,rd->...r", points, rff.omega) + rff.phase)


def gaussian_rff_mean(parameters: Array, rff: RFFParameters) -> Array:
    """Analytic E[phi(Y)] for diagonal Gaussian parameters [mean, log_std]."""
    dimension = parameters.shape[-1] // 2
    mean = parameters[..., :dimension]
    variance = jnp.exp(2.0 * parameters[..., dimension:])
    phase = jnp.einsum("...d,rd->...r", mean, rff.omega) + rff.phase
    damping = jnp.exp(
        -0.5 * jnp.einsum("...d,rd->...r", variance, jnp.square(rff.omega))
    )
    scale = jnp.sqrt(jnp.asarray(2.0 / rff.omega.shape[0], dtype=parameters.dtype))
    return scale * damping * jnp.cos(phase)


def rff_mmd_from_means(first_mean: Array, second_mean: Array) -> Array:
    return jnp.sum(jnp.square(first_mean - second_mean), axis=-1)


def make_projection_directions(key: Array, count: int, dimension: int) -> Array:
    directions = jr.normal(key, (count, dimension))
    norms = jnp.linalg.norm(directions, axis=-1, keepdims=True)
    return directions / jnp.maximum(norms, 1.0e-8)


def sliced_wasserstein_squared(predicted: Array, target: Array, directions: Array) -> Array:
    """Empirical sliced W2^2 per task for equal-size point clouds.

    predicted and target have shape [tasks, samples, dimension].
    """
    predicted_projection = jnp.einsum("bnd,pd->bnp", predicted, directions)
    target_projection = jnp.einsum("bnd,pd->bnp", target, directions)
    predicted_sorted = jnp.sort(predicted_projection, axis=1)
    target_sorted = jnp.sort(target_projection, axis=1)
    return jnp.mean(jnp.square(predicted_sorted - target_sorted), axis=(1, 2))


# %%
# -----------------------------------------------------------------------------
# Gaussian -> Gaussian experiment.
# -----------------------------------------------------------------------------
class GaussianBatch(NamedTuple):
    source_parameters: Array  # [B, 2d]
    target_parameters: Array  # [B, 2d]
    source_points: Array  # [B, N, d]
    oracle_points: Array  # [B, N, d]


def sample_diagonal_gaussian_parameters(
    key: Array,
    batch_size: int,
    dimension: int,
    mean_range: Sequence[float],
    std_range: Sequence[float],
) -> Array:
    if std_range[0] <= 0.0 or std_range[1] <= std_range[0]:
        raise ValueError("std_range must satisfy 0 < low < high.")
    mean_key, std_key = jr.split(key)
    means = jr.uniform(
        mean_key,
        (batch_size, dimension),
        minval=float(mean_range[0]),
        maxval=float(mean_range[1]),
    )
    log_stds = jr.uniform(
        std_key,
        (batch_size, dimension),
        minval=math.log(float(std_range[0])),
        maxval=math.log(float(std_range[1])),
    )
    return jnp.concatenate((means, log_stds), axis=-1)


def sample_from_diagonal_gaussian(key: Array, parameters: Array, samples: int) -> Array:
    dimension = parameters.shape[-1] // 2
    means = parameters[:, :dimension]
    stds = jnp.exp(parameters[:, dimension:])
    noise = jr.normal(key, (parameters.shape[0], samples, dimension))
    return means[:, None, :] + stds[:, None, :] * noise


def exact_diagonal_gaussian_transport(
    source_parameters: Array,
    target_parameters: Array,
    source_points: Array,
) -> Array:
    dimension = source_parameters.shape[-1] // 2
    source_mean = source_parameters[:, :dimension]
    source_log_std = source_parameters[:, dimension:]
    target_mean = target_parameters[:, :dimension]
    target_log_std = target_parameters[:, dimension:]
    scale = jnp.exp(target_log_std - source_log_std)
    return target_mean[:, None, :] + scale[:, None, :] * (
        source_points - source_mean[:, None, :]
    )


def make_gaussian_batch(key: Array, config: Mapping[str, Any]) -> GaussianBatch:
    keys = jr.split(key, 3)
    batch_size = int(config["batch_tasks"])
    dimension = int(config["dimension"])
    source_parameters = sample_diagonal_gaussian_parameters(
        keys[0],
        batch_size,
        dimension,
        config["source_mean_range"],
        config["source_std_range"],
    )
    target_parameters = sample_diagonal_gaussian_parameters(
        keys[1],
        batch_size,
        dimension,
        config["target_mean_range"],
        config["target_std_range"],
    )
    source_points = sample_from_diagonal_gaussian(
        keys[2], source_parameters, int(config["samples_per_task"])
    )
    oracle_points = exact_diagonal_gaussian_transport(
        source_parameters, target_parameters, source_points
    )
    return GaussianBatch(source_parameters, target_parameters, source_points, oracle_points)


def make_gaussian_loss(
    rff: RFFParameters,
    loss_weights: Mapping[str, float],
):
    def loss_fn(
        model: WeightConditionedTransport,
        batch: GaussianBatch,
    ) -> Tuple[Array, Dict[str, Array]]:
        predicted = apply_transport_batch(
            model,
            batch.source_parameters,
            batch.target_parameters,
            batch.source_points,
        )
        map_mse_per_task = jnp.mean(
            jnp.square(predicted - batch.oracle_points), axis=(1, 2)
        )
        predicted_rff_mean = jnp.mean(rff_features(predicted, rff), axis=1)
        target_rff_mean = gaussian_rff_mean(batch.target_parameters, rff)
        mmd_per_task = rff_mmd_from_means(predicted_rff_mean, target_rff_mean)
        displacement_per_task = jnp.mean(
            jnp.sum(jnp.square(predicted - batch.source_points), axis=-1), axis=1
        )

        dimension = predicted.shape[-1]
        predicted_mean = jnp.mean(predicted, axis=1)
        predicted_std = jnp.std(predicted, axis=1)
        target_mean = batch.target_parameters[:, :dimension]
        target_std = jnp.exp(batch.target_parameters[:, dimension:])
        mean_mse_per_task = jnp.mean(jnp.square(predicted_mean - target_mean), axis=-1)
        std_mse_per_task = jnp.mean(jnp.square(predicted_std - target_std), axis=-1)

        total_per_task = (
            float(loss_weights["supervised_map"]) * map_mse_per_task
            + float(loss_weights["rff_mmd"]) * mmd_per_task
            + float(loss_weights["displacement"]) * displacement_per_task
        )
        metrics = {
            "total": jnp.mean(total_per_task),
            "map_mse": jnp.mean(map_mse_per_task),
            "rff_mmd": jnp.mean(mmd_per_task),
            "displacement": jnp.mean(displacement_per_task),
            "mean_mse": jnp.mean(mean_mse_per_task),
            "std_mse": jnp.mean(std_mse_per_task),
        }
        return metrics["total"], metrics

    return loss_fn


def build_gaussian_model(key: Array, config: Mapping[str, Any]) -> WeightConditionedTransport:
    dimension = int(config["dimension"])
    return WeightConditionedTransport(
        source_dim=2 * dimension,
        target_dim=2 * dimension,
        point_dim=dimension,
        output_dim=dimension,
        context_dim=int(config["context_dim"]),
        source_width=int(config["encoder_width"]),
        target_width=int(config["encoder_width"]),
        point_width=int(config["encoder_width"]),
        encoder_depth=int(config["encoder_depth"]),
        fusion_width=int(config["fusion_width"]),
        fusion_depth=int(config["fusion_depth"]),
        output_mode="unbounded_residual",
        residual_scale=float(config["residual_scale"]),
        key=key,
    )


def plot_history(
    step_history: Mapping[str, Sequence[float]],
    epoch_history: Mapping[str, Sequence[float]],
    output_dir: Path,
    prefix: str,
) -> None:
    """Save step-level and epoch-level loss figures."""
    if step_history.get("step"):
        figure, axis = plt.subplots(figsize=(9, 5))
        steps = np.asarray(step_history["step"])
        for metric_name in ("total", "map_mse", "rff_mmd", "sliced_w2", "moments", "displacement"):
            if metric_name in step_history:
                values = np.maximum(np.asarray(step_history[metric_name]), 1.0e-12)
                axis.plot(steps, values, label=metric_name)
        axis.set_yscale("log")
        axis.set_xlabel("Optimisation step")
        axis.set_ylabel("Loss / diagnostic")
        axis.set_title(f"{prefix}: step-level training history")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{prefix}_step_history.png", dpi=180)
        plt.close(figure)

    if epoch_history.get("epoch"):
        figure, axis = plt.subplots(figsize=(9, 5))
        epochs = np.asarray(epoch_history["epoch"])
        for key, values in epoch_history.items():
            if key == "epoch" or not (key.endswith("_total") or key.endswith("_map_mse")):
                continue
            axis.plot(epochs, np.maximum(np.asarray(values), 1.0e-12), label=key)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title(f"{prefix}: epoch train/validation history")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{prefix}_epoch_history.png", dpi=180)
        plt.close(figure)


def plot_gaussian_qualitative(
    model: WeightConditionedTransport,
    key: Array,
    config: Mapping[str, Any],
    output_dir: Path,
) -> None:
    if int(config["dimension"]) != 2:
        print("Skipping Gaussian scatter visualisation because dimension != 2.")
        return
    task_count = int(config["visualisation_tasks"])
    sample_count = int(config["visualisation_samples"])
    keys = jr.split(key, 4)
    source_parameters = sample_diagonal_gaussian_parameters(
        keys[0], task_count, 2, config["source_mean_range"], config["source_std_range"]
    )
    target_parameters = sample_diagonal_gaussian_parameters(
        keys[1], task_count, 2, config["target_mean_range"], config["target_std_range"]
    )
    source_points = sample_from_diagonal_gaussian(keys[2], source_parameters, sample_count)
    exact_points = exact_diagonal_gaussian_transport(
        source_parameters, target_parameters, source_points
    )
    predicted_points = apply_transport_batch(
        model, source_parameters, target_parameters, source_points
    )
    source_np, exact_np, predicted_np = map(
        np.asarray, jax.device_get((source_points, exact_points, predicted_points))
    )

    figure, axes = plt.subplots(task_count, 2, figsize=(11, 4.2 * task_count), squeeze=False)
    for task in range(task_count):
        left, right = axes[task]
        left.scatter(source_np[task, :, 0], source_np[task, :, 1], s=8, alpha=0.25, label="source")
        left.scatter(exact_np[task, :, 0], exact_np[task, :, 1], s=8, alpha=0.25, label="exact target")
        left.set_title(f"Task {task}: source and exact target")
        left.set_aspect("equal", adjustable="box")
        left.grid(True, alpha=0.2)
        left.legend()

        right.scatter(exact_np[task, :, 0], exact_np[task, :, 1], s=8, alpha=0.25, label="exact target")
        right.scatter(predicted_np[task, :, 0], predicted_np[task, :, 1], s=8, alpha=0.25, label="network push-forward")
        line_indices = np.linspace(0, sample_count - 1, min(30, sample_count), dtype=int)
        for index in line_indices:
            right.plot(
                [source_np[task, index, 0], predicted_np[task, index, 0]],
                [source_np[task, index, 1], predicted_np[task, index, 1]],
                linewidth=0.5,
                alpha=0.15,
            )
        right.set_title(f"Task {task}: learned map")
        right.set_aspect("equal", adjustable="box")
        right.grid(True, alpha=0.2)
        right.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "gaussian_qualitative.png", dpi=180)
    plt.close(figure)


def train_gaussian_experiment(
    key: Array,
    config: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[WeightConditionedTransport, Dict[str, List[float]], Dict[str, List[float]]]:
    model_key, rff_key, validation_key, training_key, plot_key = jr.split(key, 5)
    model = build_gaussian_model(model_key, config)
    rff = make_rff_parameters(
        rff_key,
        int(config["dimension"]),
        config["rff"]["bandwidths"],
        int(config["rff"]["features_per_bandwidth"]),
    )
    loss_fn = make_gaussian_loss(rff, config["loss_weights"])
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(
        current_model: WeightConditionedTransport,
        current_opt_state: PyTree,
        batch: GaussianBatch,
    ):
        (loss_value, metrics), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(current_model, batch)
        updates, new_opt_state = optimizer.update(
            grads,
            current_opt_state,
            eqx.filter(current_model, eqx.is_inexact_array),
        )
        new_model = eqx.apply_updates(current_model, updates)
        return new_model, new_opt_state, loss_value, metrics

    @eqx.filter_jit
    def evaluation_step(current_model: WeightConditionedTransport, batch: GaussianBatch):
        _, metrics = loss_fn(current_model, batch)
        return metrics

    validation_keys = jr.split(validation_key, int(config["validation_batches"]))
    validation_batches = [make_gaussian_batch(k, config) for k in validation_keys]

    step_history: Dict[str, List[float]] = {}
    epoch_history: Dict[str, List[float]] = {}
    global_step = 0
    start_time = time.time()
    print(f"Gaussian model parameters: {tree_parameter_count(model):,}")
    if CONFIG["global"]["print_model"]:
        print(model)

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_metrics: List[Dict[str, float]] = []
        for _ in range(int(config["steps_per_epoch"])):
            training_key, batch_key = jr.split(training_key)
            batch = make_gaussian_batch(batch_key, config)
            model, opt_state, _, metrics = train_step(model, opt_state, batch)
            metrics_python = scalar_metrics_to_python(metrics)
            append_step_history(step_history, global_step, metrics_python)
            epoch_metrics.append(metrics_python)
            global_step += 1
            if global_step % int(config["log_every_steps"]) == 0:
                print(
                    f"[Gaussian] step={global_step:6d} "
                    f"total={metrics_python['total']:.6e} "
                    f"map_mse={metrics_python['map_mse']:.6e} "
                    f"mmd={metrics_python['rff_mmd']:.6e}"
                )

        train_average = average_metric_dicts(epoch_metrics)
        validation_metrics = [
            scalar_metrics_to_python(evaluation_step(model, batch))
            for batch in validation_batches
        ]
        validation_average = average_metric_dicts(validation_metrics)
        append_epoch_history(epoch_history, epoch, train_average, validation_average)
        elapsed = time.time() - start_time
        print(
            f"[Gaussian] epoch={epoch:3d}/{int(config['epochs'])} "
            f"train={train_average['total']:.6e} "
            f"validation={validation_average['total']:.6e} "
            f"validation_map_mse={validation_average['map_mse']:.6e} "
            f"elapsed={elapsed:.1f}s"
        )

    eqx.tree_serialise_leaves(output_dir / "gaussian_model.eqx", model)
    save_histories(step_history, epoch_history, output_dir / "gaussian_history.npz")
    plot_history(step_history, epoch_history, output_dir, prefix="gaussian")
    plot_gaussian_qualitative(model, plot_key, config, output_dir)
    return model, step_history, epoch_history


# %%
# -----------------------------------------------------------------------------
# MNIST download, parsing, and probability-measure representation.
# -----------------------------------------------------------------------------
MNIST_FILES = {
    "train_images": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    "train_labels": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    "test_images": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    "test_labels": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
}


@dataclass(frozen=True)
class MNISTData:
    train_images: np.ndarray
    train_labels: np.ndarray
    validation_images: np.ndarray
    validation_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()  # nosec B324 - used only for public dataset integrity
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected_md5: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and file_md5(destination) == expected_md5:
        return
    if destination.exists():
        destination.unlink()
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    print(f"Downloading {url} -> {destination}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                output.write(chunk)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    if file_md5(temporary) != expected_md5:
        temporary.unlink(missing_ok=True)
        raise IOError(f"MD5 mismatch after downloading {url}.")
    temporary.replace(destination)


def parse_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise IOError(f"Invalid IDX image header in {path}.")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise IOError(f"Unexpected image magic number {magic} in {path}.")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    expected = count * rows * columns
    if data.size != expected:
        raise IOError(f"Expected {expected} image bytes in {path}, found {data.size}.")
    return data.reshape(count, rows, columns)


def parse_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise IOError(f"Invalid IDX label header in {path}.")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise IOError(f"Unexpected label magic number {magic} in {path}.")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != count:
        raise IOError(f"Expected {count} labels in {path}, found {data.size}.")
    return data


def deterministic_subset(
    images: np.ndarray,
    labels: np.ndarray,
    maximum: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if maximum is None or maximum >= images.shape[0]:
        return images, labels
    rng = np.random.default_rng(seed)
    indices = rng.choice(images.shape[0], size=int(maximum), replace=False)
    return images[indices], labels[indices]


def load_mnist(config: Mapping[str, Any], seed: int) -> MNISTData:
    data_dir = ensure_dir(config["data_dir"])
    paths: Dict[str, Path] = {}
    for name, (url, md5) in MNIST_FILES.items():
        path = data_dir / Path(url).name
        download_file(url, path, md5)
        paths[name] = path

    all_train_images = parse_idx_images(paths["train_images"])
    all_train_labels = parse_idx_labels(paths["train_labels"])
    test_images = parse_idx_images(paths["test_images"])
    test_labels = parse_idx_labels(paths["test_labels"])

    classes = np.asarray(config["digit_classes"], dtype=np.uint8)
    train_mask = np.isin(all_train_labels, classes)
    test_mask = np.isin(test_labels, classes)
    all_train_images, all_train_labels = all_train_images[train_mask], all_train_labels[train_mask]
    test_images, test_labels = test_images[test_mask], test_labels[test_mask]

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(all_train_images.shape[0])
    validation_count = min(int(config["max_validation_tasks"]), all_train_images.shape[0] // 5)
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    train_images, train_labels = all_train_images[training_indices], all_train_labels[training_indices]
    validation_images, validation_labels = (
        all_train_images[validation_indices],
        all_train_labels[validation_indices],
    )

    train_images, train_labels = deterministic_subset(
        train_images, train_labels, int(config["max_train_tasks"]), seed + 1
    )
    validation_images, validation_labels = deterministic_subset(
        validation_images,
        validation_labels,
        int(config["max_validation_tasks"]),
        seed + 2,
    )
    test_images, test_labels = deterministic_subset(
        test_images, test_labels, int(config["max_test_tasks"]), seed + 3
    )

    return MNISTData(
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

# %%
# -----------------------------------------------------------------------------
# MNIST latent autoencoder.
# -----------------------------------------------------------------------------
def _group_count(channels: int, requested: int) -> int:
    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvResidualBlock(eqx.Module):
    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d

    def __init__(self, channels: int, groups: int, *, key: Array):
        k1, k2 = jr.split(key)
        group_count = _group_count(channels, groups)
        self.norm1 = eqx.nn.GroupNorm(group_count, channels)
        self.conv1 = eqx.nn.Conv2d(channels, channels, 3, padding=1, key=k1)
        self.norm2 = eqx.nn.GroupNorm(group_count, channels)
        self.conv2 = eqx.nn.Conv2d(channels, channels, 3, padding=1, key=k2)

    def __call__(self, x: Array) -> Array:
        y = self.conv1(jax.nn.silu(self.norm1(x)))
        y = self.conv2(jax.nn.silu(self.norm2(y)))
        return x + y


class MNISTAutoencoder(eqx.Module):
    enc_conv1: eqx.nn.Conv2d
    enc_res1: ConvResidualBlock
    enc_conv2: eqx.nn.Conv2d
    enc_res2: ConvResidualBlock
    enc_conv3: eqx.nn.Conv2d
    enc_res3: ConvResidualBlock
    to_latent: eqx.nn.Linear
    from_latent: eqx.nn.Linear
    dec_res3: ConvResidualBlock
    dec_up2: eqx.nn.ConvTranspose2d
    dec_res2: ConvResidualBlock
    dec_up1: eqx.nn.ConvTranspose2d
    dec_res1: ConvResidualBlock
    output_conv: eqx.nn.Conv2d
    latent_dim: int = eqx.field(static=True)

    def __init__(self, latent_dim: int, groups: int, *, key: Array):
        keys = iter(jr.split(key, 14))
        self.enc_conv1 = eqx.nn.Conv2d(1, 64, 4, stride=2, padding=1, key=next(keys))
        self.enc_res1 = ConvResidualBlock(64, groups, key=next(keys))
        self.enc_conv2 = eqx.nn.Conv2d(64, 128, 4, stride=2, padding=1, key=next(keys))
        self.enc_res2 = ConvResidualBlock(128, groups, key=next(keys))
        self.enc_conv3 = eqx.nn.Conv2d(128, 256, 3, padding=1, key=next(keys))
        self.enc_res3 = ConvResidualBlock(256, groups, key=next(keys))
        self.to_latent = eqx.nn.Linear(256 * 7 * 7, latent_dim, key=next(keys))
        self.from_latent = eqx.nn.Linear(latent_dim, 256 * 7 * 7, key=next(keys))
        self.dec_res3 = ConvResidualBlock(256, groups, key=next(keys))
        self.dec_up2 = eqx.nn.ConvTranspose2d(
            256, 128, 4, stride=2, padding=1, key=next(keys)
        )
        self.dec_res2 = ConvResidualBlock(128, groups, key=next(keys))
        self.dec_up1 = eqx.nn.ConvTranspose2d(
            128, 64, 4, stride=2, padding=1, key=next(keys)
        )
        self.dec_res1 = ConvResidualBlock(64, groups, key=next(keys))
        self.output_conv = eqx.nn.Conv2d(64, 1, 3, padding=1, key=next(keys))
        self.latent_dim = int(latent_dim)

    def encode(self, x: Array) -> Array:
        x = jax.nn.silu(self.enc_conv1(x))
        x = self.enc_res1(x)
        x = jax.nn.silu(self.enc_conv2(x))
        x = self.enc_res2(x)
        x = jax.nn.silu(self.enc_conv3(x))
        x = self.enc_res3(x)
        return self.to_latent(x.reshape(-1))

    def decode(self, z: Array) -> Array:
        x = self.from_latent(z).reshape(256, 7, 7)
        x = self.dec_res3(x)
        x = jax.nn.silu(self.dec_up2(x))
        x = self.dec_res2(x)
        x = jax.nn.silu(self.dec_up1(x))
        x = self.dec_res1(x)
        return jax.nn.sigmoid(self.output_conv(x))

    def __call__(self, x: Array) -> Array:
        return self.decode(self.encode(x))


def prepare_images(images: np.ndarray) -> np.ndarray:
    return (images.astype(np.float32) / 255.0)[:, None, :, :]


def fixed_size_batches(
    size: int, batch_size: int, rng: np.random.Generator, shuffle: bool
) -> Iterable[np.ndarray]:
    if size <= 0:
        raise ValueError("Cannot batch an empty dataset.")
    indices = rng.permutation(size) if shuffle else np.arange(size)
    if size < batch_size:
        yield rng.choice(indices, size=batch_size, replace=True)
        return
    usable = (size // batch_size) * batch_size
    for start in range(0, usable, batch_size):
        yield indices[start : start + batch_size]


def autoencoder_loss(model: MNISTAutoencoder, images: Array) -> Tuple[Array, Dict[str, Array]]:
    reconstructions = jax.vmap(lambda image: model(image))(images)
    mse = jnp.mean(jnp.square(reconstructions - images))
    return mse, {"total": mse, "mse": mse}


def make_autoencoder_steps(optimizer: optax.GradientTransformation):
    @eqx.filter_jit
    def train_step(model, opt_state, images):
        (loss, metrics), grads = eqx.filter_value_and_grad(
            autoencoder_loss, has_aux=True
        )(model, images)
        grad_norm = differentiable_tree_global_norm(grads)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        metrics = {**metrics, "grad_norm": grad_norm}
        return model, opt_state, metrics

    @eqx.filter_jit
    def eval_step(model, images):
        return autoencoder_loss(model, images)[1]

    return train_step, eval_step


def plot_autoencoder_history(
    step_history: Mapping[str, Sequence[float]],
    epoch_history: Mapping[str, Sequence[float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(step_history.get("step", []), step_history.get("total", []))
    axes[0].set_yscale("log")
    axes[0].set_title("Autoencoder step loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("MSE")
    epochs = epoch_history.get("epoch", [])
    axes[1].plot(epochs, epoch_history.get("train_total", []), label="train")
    axes[1].plot(epochs, epoch_history.get("validation_total", []), label="validation")
    axes[1].set_yscale("log")
    axes[1].set_title("Autoencoder epoch loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstructions(
    model: MNISTAutoencoder,
    images: np.ndarray,
    count: int,
    output_path: Path,
) -> None:
    count = min(count, images.shape[0])
    batch = jnp.asarray(prepare_images(images[:count]))
    recon = np.asarray(jax.device_get(jax.vmap(lambda x: model(x))(batch)))[:, 0]
    originals = images[:count].astype(np.float32) / 255.0
    fig, axes = plt.subplots(2, count, figsize=(1.4 * count, 3.0), squeeze=False)
    for col in range(count):
        axes[0, col].imshow(originals[col], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].imshow(recon[col], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("recon")
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train_or_load_autoencoder(
    data: MNISTData,
    config: Mapping[str, Any],
    key: Array,
    output_dir: Path,
) -> MNISTAutoencoder:
    checkpoint = output_dir / "mnist_autoencoder.eqx"
    model = MNISTAutoencoder(
        latent_dim=int(config["latent_dim"]),
        groups=int(config["groups"]),
        key=key,
    )
    if checkpoint.exists() and bool(config.get("reuse_checkpoint", True)):
        print(f"[AE] loading checkpoint: {checkpoint}")
        return eqx.tree_deserialise_leaves(checkpoint, model)
    if not bool(config.get("train_if_checkpoint_missing", True)):
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint}")

    train_images = prepare_images(data.train_images)
    val_images = prepare_images(data.validation_images)
    batch_size = int(config["batch_size"])
    steps_per_epoch = max(1, train_images.shape[0] // batch_size)
    total_steps = int(config["epochs"]) * steps_per_epoch
    optimizer, schedule = make_scheduled_optimizer(config, total_steps)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step, eval_step = make_autoencoder_steps(optimizer)
    rng = np.random.default_rng(int(CONFIG["global"]["seed"]) + 101)
    step_history: Dict[str, List[float]] = {}
    epoch_history: Dict[str, List[float]] = {}
    global_step = 0
    best_validation = float("inf")
    best_model = model
    start_time = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_metrics = []
        for batch_indices in fixed_size_batches(
            train_images.shape[0], batch_size, rng, shuffle=True
        ):
            images = jnp.asarray(train_images[batch_indices])
            model, opt_state, metrics_jax = train_step(model, opt_state, images)
            metrics = scalar_metrics_to_python(metrics_jax)
            metrics["learning_rate"] = float(np.asarray(schedule(global_step)))
            append_step_history(step_history, global_step, metrics)
            epoch_metrics.append(metrics)
            global_step += 1

        validation_metrics = []
        for index, batch_indices in enumerate(
            fixed_size_batches(val_images.shape[0], batch_size, rng, shuffle=False)
        ):
            if index >= int(config["validation_batches"]):
                break
            validation_metrics.append(
                scalar_metrics_to_python(eval_step(model, jnp.asarray(val_images[batch_indices])))
            )
        train_average = average_metric_dicts(epoch_metrics)
        validation_average = average_metric_dicts(validation_metrics)
        append_epoch_history(epoch_history, epoch, train_average, validation_average)
        if validation_average["total"] < best_validation:
            best_validation = validation_average["total"]
            best_model = model
            eqx.tree_serialise_leaves(checkpoint, best_model)
        if epoch % int(config["checkpoint_every_epochs"]) == 0:
            eqx.tree_serialise_leaves(output_dir / f"mnist_autoencoder_epoch_{epoch}.eqx", model)
        elapsed = time.time() - start_time
        print(
            f"[AE] epoch={epoch:4d}/{int(config['epochs'])} "
            f"train={train_average['total']:.6e} "
            f"validation={validation_average['total']:.6e} elapsed={elapsed:.1f}s"
        )

    model = best_model
    eqx.tree_serialise_leaves(checkpoint, model)
    save_histories(step_history, epoch_history, output_dir / "autoencoder_history.npz")
    plot_autoencoder_history(
        step_history, epoch_history, output_dir / "autoencoder_losses.png"
    )
    return model


@dataclass(frozen=True)
class LatentDataset:
    train: np.ndarray
    train_labels: np.ndarray
    validation: np.ndarray
    validation_labels: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def encode_image_array(
    model: MNISTAutoencoder, images: np.ndarray, chunk_size: int
) -> np.ndarray:
    prepared = prepare_images(images)

    @eqx.filter_jit
    def encode_batch(current_model, batch):
        return jax.vmap(lambda x: current_model.encode(x))(batch)

    outputs: List[np.ndarray] = []
    for start in range(0, prepared.shape[0], chunk_size):
        batch = prepared[start : start + chunk_size]
        actual = batch.shape[0]
        if actual < chunk_size:
            pad = np.repeat(batch[-1:], chunk_size - actual, axis=0)
            batch = np.concatenate([batch, pad], axis=0)
        encoded = np.asarray(jax.device_get(encode_batch(model, jnp.asarray(batch))))[:actual]
        outputs.append(encoded)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def make_latent_dataset(
    model: MNISTAutoencoder,
    data: MNISTData,
    config: Mapping[str, Any],
    output_dir: Path,
) -> LatentDataset:
    cache = output_dir / "mnist_latents.npz"
    if cache.exists() and bool(config.get("reuse_checkpoint", True)):
        loaded = np.load(cache)
        return LatentDataset(
            train=loaded["train"],
            train_labels=loaded["train_labels"],
            validation=loaded["validation"],
            validation_labels=loaded["validation_labels"],
            test=loaded["test"],
            test_labels=loaded["test_labels"],
            mean=loaded["mean"],
            std=loaded["std"],
        )

    chunk = int(config["encode_chunk_size"])
    train_raw = encode_image_array(model, data.train_images, chunk)
    validation_raw = encode_image_array(model, data.validation_images, chunk)
    test_raw = encode_image_array(model, data.test_images, chunk)
    if bool(CONFIG["mnist"]["transport"]["standardise_latents"]):
        mean = train_raw.mean(axis=0)
        std = train_raw.std(axis=0)
        std = np.maximum(std, float(CONFIG["mnist"]["transport"]["latent_std_floor"]))
    else:
        mean = np.zeros(train_raw.shape[1], dtype=np.float32)
        std = np.ones(train_raw.shape[1], dtype=np.float32)
    train = ((train_raw - mean) / std).astype(np.float32)
    validation = ((validation_raw - mean) / std).astype(np.float32)
    test = ((test_raw - mean) / std).astype(np.float32)
    np.savez_compressed(
        cache,
        train=train,
        train_labels=data.train_labels,
        validation=validation,
        validation_labels=data.validation_labels,
        test=test,
        test_labels=data.test_labels,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    return LatentDataset(
        train=train,
        train_labels=data.train_labels,
        validation=validation,
        validation_labels=data.validation_labels,
        test=test,
        test_labels=data.test_labels,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


# %%
# -----------------------------------------------------------------------------
# Prompt-conditioned transport architecture.
# -----------------------------------------------------------------------------
def vmap_module(module, x: Array) -> Array:
    return jax.vmap(lambda row: module(row))(x)


class MultiHeadAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    hidden_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, hidden_dim: int, num_heads: int, *, key: Array):
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        kq, kk, kv, ko = jr.split(key, 4)
        self.q_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=kq)
        self.k_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=kk)
        self.v_proj = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=kv)
        self.out_proj = eqx.nn.Linear(hidden_dim, hidden_dim, key=ko)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads

    def __call__(self, query: Array, context: Array) -> Array:
        q = vmap_module(self.q_proj, query).reshape(
            query.shape[0], self.num_heads, self.head_dim
        )
        k = vmap_module(self.k_proj, context).reshape(
            context.shape[0], self.num_heads, self.head_dim
        )
        v = vmap_module(self.v_proj, context).reshape(
            context.shape[0], self.num_heads, self.head_dim
        )
        scores = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(self.head_dim)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("hqk,khd->qhd", weights, v).reshape(
            query.shape[0], self.hidden_dim
        )
        return vmap_module(self.out_proj, attended)


class AttentionBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attention: MultiHeadAttention
    norm2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int, *, key: Array):
        ka, km = jr.split(key)
        self.norm1 = eqx.nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads, key=ka)
        self.norm2 = eqx.nn.LayerNorm(hidden_dim)
        self.mlp = eqx.nn.MLP(
            hidden_dim,
            hidden_dim,
            width_size=hidden_dim * mlp_ratio,
            depth=1,
            activation=jax.nn.gelu,
            key=km,
        )

    def __call__(self, x: Array) -> Array:
        normalised = vmap_module(self.norm1, x)
        x = x + self.attention(normalised, normalised)
        return x + vmap_module(self.mlp, vmap_module(self.norm2, x))


class CrossAttentionBlock(eqx.Module):
    query_norm: eqx.nn.LayerNorm
    context_norm: eqx.nn.LayerNorm
    attention: MultiHeadAttention
    output_norm: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int, *, key: Array):
        ka, km = jr.split(key)
        self.query_norm = eqx.nn.LayerNorm(hidden_dim)
        self.context_norm = eqx.nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads, key=ka)
        self.output_norm = eqx.nn.LayerNorm(hidden_dim)
        self.mlp = eqx.nn.MLP(
            hidden_dim,
            hidden_dim,
            width_size=hidden_dim * mlp_ratio,
            depth=1,
            activation=jax.nn.gelu,
            key=km,
        )

    def __call__(self, query: Array, context: Array) -> Array:
        q = vmap_module(self.query_norm, query)
        c = vmap_module(self.context_norm, context)
        query = query + self.attention(q, c)
        return query + vmap_module(self.mlp, vmap_module(self.output_norm, query))


class PromptTransport(eqx.Module):
    source_embed: eqx.nn.MLP
    target_embed: eqx.nn.MLP
    query_embed: eqx.nn.MLP
    source_type: Array
    target_type: Array
    self_blocks: Tuple[AttentionBlock, ...]
    cross_block: CrossAttentionBlock
    output_mlp: eqx.nn.MLP
    residual_output_scale: float = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        self_attention_blocks: int,
        mlp_ratio: int,
        residual_output_scale: float,
        *,
        key: Array,
    ):
        keys = jr.split(key, 7 + self_attention_blocks)
        self.source_embed = eqx.nn.MLP(
            latent_dim, hidden_dim, hidden_dim, depth=2, activation=jax.nn.gelu, key=keys[0]
        )
        self.target_embed = eqx.nn.MLP(
            latent_dim, hidden_dim, hidden_dim, depth=2, activation=jax.nn.gelu, key=keys[1]
        )
        self.query_embed = eqx.nn.MLP(
            latent_dim, hidden_dim, hidden_dim, depth=2, activation=jax.nn.gelu, key=keys[2]
        )
        self.source_type = 0.02 * jr.normal(keys[3], (hidden_dim,))
        self.target_type = 0.02 * jr.normal(keys[4], (hidden_dim,))
        self.self_blocks = tuple(
            AttentionBlock(hidden_dim, num_heads, mlp_ratio, key=keys[5 + i])
            for i in range(self_attention_blocks)
        )
        self.cross_block = CrossAttentionBlock(
            hidden_dim, num_heads, mlp_ratio, key=keys[5 + self_attention_blocks]
        )
        output = eqx.nn.MLP(
            hidden_dim,
            latent_dim,
            width_size=hidden_dim,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[6 + self_attention_blocks],
        )
        self.output_mlp = _scaled_last_layer(output, 0.05)
        self.residual_output_scale = float(residual_output_scale)

    def __call__(
        self, source_prompt: Array, target_prompt: Array, queries: Array
    ) -> Array:
        source_tokens = vmap_module(self.source_embed, source_prompt) + self.source_type
        target_tokens = vmap_module(self.target_embed, target_prompt) + self.target_type
        context = jnp.concatenate([source_tokens, target_tokens], axis=0)
        for block in self.self_blocks:
            context = block(context)
        query_tokens = vmap_module(self.query_embed, queries)
        query_tokens = self.cross_block(query_tokens, context)
        delta = vmap_module(self.output_mlp, query_tokens)
        return queries + self.residual_output_scale * delta


class ClassConditionedTransport(eqx.Module):
    network: eqx.nn.MLP
    residual_output_scale: float = eqx.field(static=True)

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        residual_output_scale: float,
        *,
        key: Array,
    ):
        mlp = eqx.nn.MLP(
            latent_dim + 10,
            latent_dim,
            width_size=hidden_dim,
            depth=4,
            activation=jax.nn.gelu,
            key=key,
        )
        self.network = _scaled_last_layer(mlp, 0.05)
        self.residual_output_scale = float(residual_output_scale)

    def __call__(self, class_one_hot: Array, queries: Array) -> Array:
        repeated = jnp.broadcast_to(class_one_hot, (queries.shape[0], 10))
        inputs = jnp.concatenate([queries, repeated], axis=-1)
        delta = vmap_module(self.network, inputs)
        return queries + self.residual_output_scale * delta


class LatentTaskBatch(NamedTuple):
    source_prompt: Array
    target_prompt: Array
    source_queries: Array
    target_samples: Array
    class_one_hot: Array


def apply_latent_transport(model, batch: LatentTaskBatch, conditioning_mode: str) -> Array:
    if conditioning_mode == "prompt":
        return model(batch.source_prompt, batch.target_prompt, batch.source_queries)
    if conditioning_mode == "class":
        return model(batch.class_one_hot, batch.source_queries)
    raise ValueError(f"Unknown conditioning_mode={conditioning_mode!r}")


# %%
# -----------------------------------------------------------------------------
# Distributional losses in latent space.
# -----------------------------------------------------------------------------
def pairwise_squared_distances(x: Array, y: Array) -> Array:
    x2 = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    y2 = jnp.sum(jnp.square(y), axis=-1, keepdims=True).T
    return jnp.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def adaptive_multiscale_bandwidths(
    x: Array, y: Array, num_scales: int, scale_ratio: float, minimum: float
) -> Array:
    pooled = jnp.concatenate([x, y], axis=0)
    distances = pairwise_squared_distances(pooled, pooled)
    count = pooled.shape[0]
    off_diagonal_mean = jnp.sum(distances) / jnp.maximum(count * (count - 1), 1)
    base = jax.lax.stop_gradient(jnp.maximum(off_diagonal_mean, minimum))
    centre = (num_scales - 1) / 2.0
    exponents = jnp.arange(num_scales, dtype=x.dtype) - centre
    return base * jnp.power(jnp.asarray(scale_ratio, dtype=x.dtype), exponents)


def multiscale_rbf_kernel(x: Array, y: Array, bandwidths: Array) -> Array:
    distances = pairwise_squared_distances(x, y)
    kernels = jnp.exp(-distances[None, :, :] / (2.0 * bandwidths[:, None, None]))
    return jnp.mean(kernels, axis=0)


def multiscale_mmd_squared(
    x: Array,
    y: Array,
    *,
    num_scales: int,
    scale_ratio: float,
    minimum_bandwidth: float,
    estimator: str,
) -> Array:
    bandwidths = adaptive_multiscale_bandwidths(
        x, y, num_scales, scale_ratio, minimum_bandwidth
    )
    kxx = multiscale_rbf_kernel(x, x, bandwidths)
    kyy = multiscale_rbf_kernel(y, y, bandwidths)
    kxy = multiscale_rbf_kernel(x, y, bandwidths)
    if estimator == "biased":
        return jnp.maximum(jnp.mean(kxx) + jnp.mean(kyy) - 2.0 * jnp.mean(kxy), 0.0)
    if estimator == "unbiased":
        nx = x.shape[0]
        ny = y.shape[0]
        xx = (jnp.sum(kxx) - jnp.trace(kxx)) / jnp.maximum(nx * (nx - 1), 1)
        yy = (jnp.sum(kyy) - jnp.trace(kyy)) / jnp.maximum(ny * (ny - 1), 1)
        return xx + yy - 2.0 * jnp.mean(kxy)
    raise ValueError(f"Unknown MMD estimator={estimator!r}")


def mean_and_covariance(samples: Array) -> Tuple[Array, Array]:
    mean = jnp.mean(samples, axis=0)
    centered = samples - mean
    denominator = jnp.maximum(samples.shape[0] - 1, 1)
    covariance = centered.T @ centered / denominator
    return mean, covariance


def latent_transport_loss(
    model,
    batch: LatentTaskBatch,
    conditioning_mode: str,
    config: Mapping[str, Any],
) -> Tuple[Array, Dict[str, Array]]:
    generated = apply_latent_transport(model, batch, conditioning_mode)
    mmd_cfg = config["mmd"]
    mmd_train = multiscale_mmd_squared(
        generated,
        batch.target_samples,
        num_scales=int(mmd_cfg["num_scales"]),
        scale_ratio=float(mmd_cfg["scale_ratio"]),
        minimum_bandwidth=float(mmd_cfg["minimum_bandwidth"]),
        estimator=str(config["mmd_estimator_for_training"]),
    )
    mmd_unbiased = multiscale_mmd_squared(
        generated,
        batch.target_samples,
        num_scales=int(mmd_cfg["num_scales"]),
        scale_ratio=float(mmd_cfg["scale_ratio"]),
        minimum_bandwidth=float(mmd_cfg["minimum_bandwidth"]),
        estimator="unbiased",
    )
    displacement = jnp.mean(jnp.sum(jnp.square(generated - batch.source_queries), axis=-1))
    generated_mean, generated_cov = mean_and_covariance(generated)
    target_mean, target_cov = mean_and_covariance(batch.target_samples)
    moments = jnp.mean(jnp.square(generated_mean - target_mean)) + jnp.mean(
        jnp.square(generated_cov - target_cov)
    )
    weights = config["loss_weights"]
    total = (
        float(weights["displacement"]) * displacement
        + float(weights["mmd"]) * mmd_train
        + float(weights["moments"]) * moments
    )
    metrics = {
        "total": total,
        "mmd_train": mmd_train,
        "mmd_unbiased": mmd_unbiased,
        "displacement": displacement,
        "moments": moments,
        "generated_norm": jnp.mean(jnp.linalg.norm(generated, axis=-1)),
    }
    return total, metrics


def make_transport_steps(
    optimizer: optax.GradientTransformation,
    conditioning_mode: str,
    config: Mapping[str, Any],
):
    def loss_fn(model, batch):
        return latent_transport_loss(model, batch, conditioning_mode, config)

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch)
        grad_norm = differentiable_tree_global_norm(grads)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        metrics = {**metrics, "grad_norm": grad_norm}
        return model, opt_state, metrics

    @eqx.filter_jit
    def eval_step(model, batch):
        return loss_fn(model, batch)[1]

    return train_step, eval_step


# %%
# -----------------------------------------------------------------------------
# Task sampling and latent transport training.
# -----------------------------------------------------------------------------
def class_indices(labels: np.ndarray, digits: Sequence[int]) -> Dict[int, np.ndarray]:
    result = {int(digit): np.flatnonzero(labels == digit) for digit in digits}
    missing = [digit for digit, indices in result.items() if indices.size == 0]
    if missing:
        raise ValueError(f"No samples found for digit classes {missing}.")
    return result


def sample_rows(
    values: np.ndarray,
    indices: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chosen = rng.choice(indices, size=count, replace=indices.size < count)
    return values[chosen]


def make_latent_task_batch(
    latents: np.ndarray,
    indices_by_class: Mapping[int, np.ndarray],
    digit: int,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    key: Array,
) -> LatentTaskBatch:
    prompt_size = int(config["prompt_size"])
    sample_count = int(config["samples_per_task"])
    source_key, query_key = jr.split(key)
    source_prompt = jr.normal(source_key, (prompt_size, latents.shape[1]))
    source_queries = jr.normal(query_key, (sample_count, latents.shape[1]))
    target_prompt = sample_rows(
        latents, indices_by_class[int(digit)], prompt_size, rng
    ).astype(np.float32)
    target_samples = sample_rows(
        latents, indices_by_class[int(digit)], sample_count, rng
    ).astype(np.float32)
    one_hot = np.zeros(10, dtype=np.float32)
    one_hot[int(digit)] = 1.0
    return LatentTaskBatch(
        source_prompt=source_prompt,
        target_prompt=jnp.asarray(target_prompt),
        source_queries=source_queries,
        target_samples=jnp.asarray(target_samples),
        class_one_hot=jnp.asarray(one_hot),
    )


def make_validation_batches(
    latents: np.ndarray,
    labels: np.ndarray,
    digits: Sequence[int],
    config: Mapping[str, Any],
    seed: int,
) -> List[LatentTaskBatch]:
    rng = np.random.default_rng(seed)
    indices = class_indices(labels, digits)
    key = jr.PRNGKey(seed)
    batches = []
    for digit in digits:
        for _ in range(int(config["validation_repeats_per_class"])):
            key, subkey = jr.split(key)
            batches.append(
                make_latent_task_batch(latents, indices, int(digit), config, rng, subkey)
            )
    return batches


def plot_transport_history(
    step_history: Mapping[str, Sequence[float]],
    epoch_history: Mapping[str, Sequence[float]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(step_history.get("step", []), step_history.get("total", []), linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Transport loss at every step")
    axes[0].set_xlabel("step")
    epochs = epoch_history.get("epoch", [])
    axes[1].plot(epochs, epoch_history.get("train_total", []), label="train")
    axes[1].plot(epochs, epoch_history.get("validation_total", []), label="validation")
    axes[1].set_yscale("log")
    axes[1].set_title("Epoch total loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[2].plot(epochs, epoch_history.get("train_mmd_unbiased", []), label="train")
    axes[2].plot(epochs, epoch_history.get("validation_mmd_unbiased", []), label="validation")
    axes[2].set_title("Unbiased MMD diagnostic")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def initialise_transport_model(
    latent_dim: int, config: Mapping[str, Any], key: Array
):
    mode = str(config["conditioning_mode"])
    if mode == "prompt":
        return PromptTransport(
            latent_dim=latent_dim,
            hidden_dim=int(config["hidden_dim"]),
            num_heads=int(config["num_heads"]),
            self_attention_blocks=int(config["self_attention_blocks"]),
            mlp_ratio=int(config["mlp_ratio"]),
            residual_output_scale=float(config["residual_output_scale"]),
            key=key,
        )
    if mode == "class":
        return ClassConditionedTransport(
            latent_dim=latent_dim,
            hidden_dim=int(config["hidden_dim"]),
            residual_output_scale=float(config["residual_output_scale"]),
            key=key,
        )
    raise ValueError("conditioning_mode must be 'prompt' or 'class'.")


def train_or_load_transport(
    latent_data: LatentDataset,
    config: Mapping[str, Any],
    key: Array,
    output_dir: Path,
):
    mode = str(config["conditioning_mode"])
    checkpoint = output_dir / f"mnist_transport_{mode}.eqx"
    model = initialise_transport_model(latent_data.train.shape[1], config, key)
    if checkpoint.exists() and bool(config.get("reuse_checkpoint", True)):
        print(f"[Transport] loading checkpoint: {checkpoint}")
        return eqx.tree_deserialise_leaves(checkpoint, model), {}, {}
    if not bool(config.get("train_if_checkpoint_missing", True)):
        raise FileNotFoundError(f"Transport checkpoint not found: {checkpoint}")

    total_steps = int(config["epochs"]) * int(config["steps_per_epoch"])
    optimizer, schedule = make_scheduled_optimizer(config, total_steps)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step, eval_step = make_transport_steps(optimizer, mode, config)
    digits = [int(x) for x in CONFIG["mnist"]["digit_classes"]]
    train_indices = class_indices(latent_data.train_labels, digits)
    validation_batches = make_validation_batches(
        latent_data.validation,
        latent_data.validation_labels,
        digits,
        config,
        seed=int(CONFIG["global"]["seed"]) + 707,
    )
    rng = np.random.default_rng(int(CONFIG["global"]["seed"]) + 303)
    jax_key = jr.PRNGKey(int(CONFIG["global"]["seed"]) + 404)
    step_history: Dict[str, List[float]] = {}
    epoch_history: Dict[str, List[float]] = {}
    global_step = 0
    best_validation = float("inf")
    best_model = model
    start_time = time.time()

    for epoch in range(1, int(config["epochs"]) + 1):
        repetitions = math.ceil(int(config["steps_per_epoch"]) / len(digits))
        task_order = np.tile(np.asarray(digits), repetitions)[: int(config["steps_per_epoch"])]
        rng.shuffle(task_order)
        epoch_metrics = []
        for digit in task_order:
            jax_key, batch_key = jr.split(jax_key)
            batch = make_latent_task_batch(
                latent_data.train,
                train_indices,
                int(digit),
                config,
                rng,
                batch_key,
            )
            model, opt_state, metrics_jax = train_step(model, opt_state, batch)
            metrics = scalar_metrics_to_python(metrics_jax)
            metrics["learning_rate"] = float(np.asarray(schedule(global_step)))
            metrics["digit"] = float(digit)
            append_step_history(step_history, global_step, metrics)
            epoch_metrics.append(metrics)
            global_step += 1
            if global_step % int(config["log_every_steps"]) == 0:
                print(
                    f"[Transport] step={global_step:7d} digit={int(digit)} "
                    f"loss={metrics['total']:.6e} mmd={metrics['mmd_unbiased']:.6e}"
                )

        validation_metrics = [
            scalar_metrics_to_python(eval_step(model, batch)) for batch in validation_batches
        ]
        train_average = average_metric_dicts(epoch_metrics)
        validation_average = average_metric_dicts(validation_metrics)
        append_epoch_history(epoch_history, epoch, train_average, validation_average)
        if validation_average["total"] < best_validation:
            best_validation = validation_average["total"]
            best_model = model
            eqx.tree_serialise_leaves(checkpoint, best_model)
        if epoch % int(config["checkpoint_every_epochs"]) == 0:
            eqx.tree_serialise_leaves(
                output_dir / f"mnist_transport_{mode}_epoch_{epoch}.eqx", model
            )
        elapsed = time.time() - start_time
        print(
            f"[Transport] epoch={epoch:4d}/{int(config['epochs'])} "
            f"train={train_average['total']:.6e} "
            f"validation={validation_average['total']:.6e} "
            f"validation_mmd={validation_average['mmd_unbiased']:.6e} "
            f"elapsed={elapsed:.1f}s"
        )

    model = best_model
    eqx.tree_serialise_leaves(checkpoint, model)
    save_histories(step_history, epoch_history, output_dir / f"transport_{mode}_history.npz")
    plot_transport_history(
        step_history, epoch_history, output_dir / f"transport_{mode}_losses.png"
    )
    return model, step_history, epoch_history


# %%
# -----------------------------------------------------------------------------
# Decoding, evaluation, and visualisation of generated MNIST digits.
# -----------------------------------------------------------------------------
def decode_standardised_latents(
    autoencoder: MNISTAutoencoder,
    standardised_latents: np.ndarray,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    raw_latents = standardised_latents * latent_std + latent_mean

    @eqx.filter_jit
    def decode_batch(model, batch):
        return jax.vmap(lambda z: model.decode(z))(batch)

    outputs = []
    for start in range(0, raw_latents.shape[0], chunk_size):
        batch = raw_latents[start : start + chunk_size].astype(np.float32)
        actual = batch.shape[0]
        if actual < chunk_size:
            batch = np.concatenate(
                [batch, np.repeat(batch[-1:], chunk_size - actual, axis=0)], axis=0
            )
        decoded = np.asarray(jax.device_get(decode_batch(autoencoder, jnp.asarray(batch))))[:actual]
        outputs.append(decoded[:, 0])
    return np.concatenate(outputs, axis=0)


def generate_for_digit(
    model,
    digit: int,
    target_latents: np.ndarray,
    target_labels: np.ndarray,
    config: Mapping[str, Any],
    source_queries: Array,
    source_prompt: Array,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.flatnonzero(target_labels == digit)
    prompt = sample_rows(
        target_latents, indices, int(config["prompt_size"]), rng
    ).astype(np.float32)
    one_hot = np.zeros(10, dtype=np.float32)
    one_hot[digit] = 1.0
    batch = LatentTaskBatch(
        source_prompt=source_prompt,
        target_prompt=jnp.asarray(prompt),
        source_queries=source_queries,
        target_samples=jnp.asarray(prompt[: min(prompt.shape[0], source_queries.shape[0])]),
        class_one_hot=jnp.asarray(one_hot),
    )
    generated = apply_latent_transport(model, batch, str(config["conditioning_mode"]))
    return np.asarray(jax.device_get(generated))


def plot_prompt_and_generated_grid(
    model,
    autoencoder: MNISTAutoencoder,
    latent_data: LatentDataset,
    raw_test_images: np.ndarray,
    config: Mapping[str, Any],
    visual_config: Mapping[str, Any],
    output_path: Path,
) -> None:
    digits = [int(x) for x in CONFIG["mnist"]["digit_classes"]]
    rows = int(visual_config["generated_rows"])
    rng = np.random.default_rng(int(CONFIG["global"]["seed"]) + 909)
    key = jr.PRNGKey(int(CONFIG["global"]["seed"]) + 1001)
    source_prompt_key, query_key = jr.split(key)
    source_prompt = jr.normal(source_prompt_key, (int(config["prompt_size"]), latent_data.test.shape[1]))
    fixed_queries = jr.normal(query_key, (rows, latent_data.test.shape[1]))
    generated_all = []
    prompt_images = []
    for digit in digits:
        generated_all.append(
            generate_for_digit(
                model,
                digit,
                latent_data.test,
                latent_data.test_labels,
                config,
                fixed_queries,
                source_prompt,
                rng,
            )
        )
        digit_indices = np.flatnonzero(latent_data.test_labels == digit)
        prompt_images.append(raw_test_images[rng.choice(digit_indices)])
    generated_latents = np.stack(generated_all, axis=1).reshape(rows * len(digits), -1)
    decoded = decode_standardised_latents(
        autoencoder, generated_latents, latent_data.mean, latent_data.std
    ).reshape(rows, len(digits), 28, 28)

    fig, axes = plt.subplots(rows + 1, len(digits), figsize=(1.45 * len(digits), 1.45 * (rows + 1)))
    for col, digit in enumerate(digits):
        axes[0, col].imshow(prompt_images[col], cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(str(digit))
        axes[0, col].axis("off")
        for row in range(rows):
            axes[row + 1, col].imshow(decoded[row, col], cmap="gray", vmin=0.0, vmax=1.0)
            axes[row + 1, col].axis("off")
    axes[0, 0].set_ylabel("prompt")
    axes[1, 0].set_ylabel("generated")
    fig.suptitle("Gaussian to MNIST: prompt-conditioned latent transport", y=0.995)
    fig.tight_layout(pad=0.15)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_latent_scatter(
    model,
    latent_data: LatentDataset,
    config: Mapping[str, Any],
    visual_config: Mapping[str, Any],
    output_path: Path,
) -> None:
    digits = [int(x) for x in visual_config["selected_scatter_digits"]]
    count = int(visual_config["latent_scatter_samples"])
    rng = np.random.default_rng(int(CONFIG["global"]["seed"]) + 1212)
    key = jr.PRNGKey(int(CONFIG["global"]["seed"]) + 1313)
    fig, axes = plt.subplots(1, len(digits), figsize=(4 * len(digits), 4), squeeze=False)
    for column, digit in enumerate(digits):
        key, kp, kq = jr.split(key, 3)
        source_prompt = jr.normal(kp, (int(config["prompt_size"]), latent_data.test.shape[1]))
        source_queries = jr.normal(kq, (count, latent_data.test.shape[1]))
        generated = generate_for_digit(
            model,
            digit,
            latent_data.test,
            latent_data.test_labels,
            config,
            source_queries,
            source_prompt,
            rng,
        )
        real = sample_rows(
            latent_data.test,
            np.flatnonzero(latent_data.test_labels == digit),
            count,
            rng,
        )
        ax = axes[0, column]
        ax.scatter(real[:, 0], real[:, 1], s=7, alpha=0.45, label="real")
        ax.scatter(generated[:, 0], generated[:, 1], s=7, alpha=0.45, label="generated")
        ax.set_title(f"digit {digit}")
        ax.set_xlabel("latent 1")
        ax.set_ylabel("latent 2")
        if column == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_per_class_mmd(
    model,
    latent_data: LatentDataset,
    config: Mapping[str, Any],
) -> Dict[str, float]:
    digits = [int(x) for x in CONFIG["mnist"]["digit_classes"]]
    batches = make_validation_batches(
        latent_data.test,
        latent_data.test_labels,
        digits,
        config,
        seed=int(CONFIG["global"]["seed"]) + 1414,
    )
    mode = str(config["conditioning_mode"])
    values: Dict[int, List[float]] = {digit: [] for digit in digits}
    for index, batch in enumerate(batches):
        generated = apply_latent_transport(model, batch, mode)
        mmd = multiscale_mmd_squared(
            generated,
            batch.target_samples,
            num_scales=int(config["mmd"]["num_scales"]),
            scale_ratio=float(config["mmd"]["scale_ratio"]),
            minimum_bandwidth=float(config["mmd"]["minimum_bandwidth"]),
            estimator="unbiased",
        )
        digit = digits[index // int(config["validation_repeats_per_class"])]
        values[digit].append(float(np.asarray(jax.device_get(mmd))))
    result = {str(digit): float(np.mean(values[digit])) for digit in digits}
    result["overall"] = float(np.mean([value for group in values.values() for value in group]))
    return result


def run_mnist_generation_experiment(
    config: Mapping[str, Any], key: Array, output_dir: Path
) -> None:
    output_dir = ensure_dir(output_dir)
    data = load_mnist(config, int(CONFIG["global"]["seed"]))
    print(
        "[MNIST] sizes:",
        data.train_images.shape[0],
        data.validation_images.shape[0],
        data.test_images.shape[0],
    )
    ae_key, transport_key = jr.split(key)
    autoencoder = train_or_load_autoencoder(
        data, config["autoencoder"], ae_key, output_dir
    )
    print("[AE] parameters:", tree_parameter_count(autoencoder))
    plot_reconstructions(
        autoencoder,
        data.test_images,
        int(config["visualisation"]["reconstruction_examples"]),
        output_dir / "autoencoder_reconstructions.png",
    )
    latent_data = make_latent_dataset(
        autoencoder, data, config["autoencoder"], output_dir
    )
    model, _, _ = train_or_load_transport(
        latent_data, config["transport"], transport_key, output_dir
    )
    print("[Transport] parameters:", tree_parameter_count(model))
    test_metrics = evaluate_per_class_mmd(model, latent_data, config["transport"])
    save_json(test_metrics, output_dir / "mnist_test_mmd.json")
    print("[MNIST] test MMD:", test_metrics)
    plot_prompt_and_generated_grid(
        model,
        autoencoder,
        latent_data,
        data.test_images,
        config["transport"],
        config["visualisation"],
        output_dir / "mnist_prompt_conditioned_generations.png",
    )
    plot_latent_scatter(
        model,
        latent_data,
        config["transport"],
        config["visualisation"],
        output_dir / "mnist_latent_real_vs_generated.png",
    )


# %%
# -----------------------------------------------------------------------------
# Smoke tests and entry point.
# -----------------------------------------------------------------------------
def run_shape_smoke_tests() -> None:
    key = jr.PRNGKey(0)
    ae_key, prompt_key, class_key = jr.split(key, 3)
    ae = MNISTAutoencoder(latent_dim=4, groups=8, key=ae_key)
    image = jnp.zeros((1, 28, 28), dtype=jnp.float32)
    latent = ae.encode(image)
    reconstruction = ae.decode(latent)
    if latent.shape != (4,) or reconstruction.shape != (1, 28, 28):
        raise RuntimeError(
            f"Autoencoder shape test failed: latent={latent.shape}, recon={reconstruction.shape}"
        )
    prompt_model = PromptTransport(
        latent_dim=4,
        hidden_dim=64,
        num_heads=4,
        self_attention_blocks=1,
        mlp_ratio=2,
        residual_output_scale=0.1,
        key=prompt_key,
    )
    output = prompt_model(
        jnp.zeros((8, 4)), jnp.ones((8, 4)), jnp.zeros((16, 4))
    )
    if output.shape != (16, 4):
        raise RuntimeError(f"Prompt transport shape test failed: {output.shape}")
    class_model = ClassConditionedTransport(4, 64, 0.1, key=class_key)
    output = class_model(jax.nn.one_hot(3, 10), jnp.zeros((16, 4)))
    if output.shape != (16, 4):
        raise RuntimeError(f"Class transport shape test failed: {output.shape}")
    print("Shape smoke tests passed.")


def main() -> None:
    print_runtime_information()
    root = ensure_dir(CONFIG["global"]["output_dir"])
    save_json(CONFIG, root / "config.json")
    if bool(CONFIG["global"]["run_shape_smoke_tests"]):
        run_shape_smoke_tests()
    key = jr.PRNGKey(int(CONFIG["global"]["seed"]))
    gaussian_key, mnist_key = jr.split(key)
    requested = set(CONFIG["global"]["run"])
    if "gaussian" in requested:
        gaussian_dir = ensure_dir(root / "gaussian")
        train_gaussian_experiment(CONFIG["gaussian"], gaussian_key, gaussian_dir)
    if "mnist_generation" in requested:
        mnist_dir = ensure_dir(root / "mnist_generation")
        run_mnist_generation_experiment(CONFIG["mnist"], mnist_key, mnist_dir)


if __name__ == "__main__":
    main()
