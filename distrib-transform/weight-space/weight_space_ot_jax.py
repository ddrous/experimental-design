# %% [markdown]
# Weight-space amortised optimal transport with JAX + Equinox + Optax
#
# This script is intentionally organised like a Python notebook. Editors such as
# VS Code, Spyder, and PyCharm recognise ``# %%`` as cell boundaries.
#
# Experiments
# -----------
# 1. Gaussian -> Gaussian:
#    A weight-conditioned network receives source Gaussian parameters, target
#    Gaussian parameters, and a source point. It is trained against the exact
#    diagonal-Gaussian Brenier map and is also evaluated distributionally.
#
# 2. Gaussian -> MNIST spatial measures:
#    Each 28x28 MNIST image is interpreted as a probability measure on the 2-D
#    pixel grid. The image's 784 normalised pixel masses are the target "weight
#    vector". A single network receives source Gaussian parameters, these target
#    weights, and a source point, and outputs a transported 2-D point.
#
# The MNIST objective is label-free: a random-Fourier-feature MMD penalty and a
# sliced-Wasserstein penalty enforce push-forward matching, whilst a small
# displacement penalty favours low-cost maps. The discrete MNIST target should
# be viewed as an experimental approximation to a dequantised image density.
#
# Tested design target: Python >= 3.10, recent JAX, Equinox, and Optax.
# Install the JAX build appropriate for your hardware, then install Equinox,
# Optax, NumPy, and Matplotlib.

# %%
# -----------------------------------------------------------------------------
# Configuration: all user-facing hyperparameters live here.
# -----------------------------------------------------------------------------
CONFIG = {
    "global": {
        "seed": 2026,
        "run": ["gaussian", "mnist"],  # any subset of {"gaussian", "mnist"}
        "output_dir": "./weight_space_ot_outputs",
        "quick_test": False,
        "enable_x64": False,
        "jax_platform": None,  # None, "cpu", "gpu", or "tpu"
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
        "loss_weights": {
            "supervised_map": 1.0,
            "rff_mmd": 0.10,
            "displacement": 0.0,
        },
        "rff": {
            "bandwidths": [0.25, 0.5, 1.0, 2.0],
            "features_per_bandwidth": 64,
        },
        "log_every_steps": 100,
        "visualisation_tasks": 4,
        "visualisation_samples": 800,
    },
    "mnist": {
        "data_dir": "./data/mnist_raw",
        "digit_classes": list(range(10)),
        "max_train_tasks": 10000,
        "max_validation_tasks": 1000,
        "max_test_tasks": 1000,
        "epochs": 25,
        "batch_tasks": 16,
        "samples_per_task": 128,
        "validation_batch_tasks": 16,
        "learning_rate": 2.0e-4,
        "weight_decay": 1.0e-6,
        "gradient_clip": 1.0,
        "source_mean_range": [-0.20, 0.20],
        "source_std_range": [0.35, 0.70],
        "pixel_mass_floor": 1.0e-4,
        "target_context_mode": "sqrt_scaled",  # weights, scaled, sqrt_scaled
        "target_jitter": True,
        "context_dim": 96,
        "source_encoder_width": 128,
        "target_encoder_width": 256,
        "point_encoder_width": 128,
        "encoder_depth": 2,
        "fusion_width": 256,
        "fusion_depth": 3,
        "residual_scale": 1.0,
        "loss_weights": {
            "rff_mmd": 10.0,
            "sliced_w2": 1.0,
            "moments": 1.0,
            "displacement": 0.01,
        },
        "rff": {
            "bandwidths": [0.05, 0.10, 0.20, 0.40, 0.80],
            "features_per_bandwidth": 64,
        },
        "sliced_wasserstein_projections": 32,
        "log_every_steps": 100,
        "visualisation_examples": 6,
        "visualisation_samples": 3000,
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
    import matplotlib.pyplot as plt
    import numpy as np
    import optax
except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
    raise ImportError(
        "Missing dependency. Install a JAX build appropriate for your hardware, "
        "then install equinox, optax, numpy, and matplotlib."
    ) from exc

jax.config.update("jax_enable_x64", bool(CONFIG["global"]["enable_x64"]))

Array = jax.Array
PyTree = Any


# %%
# -----------------------------------------------------------------------------
# Small utility functions.
# -----------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary and return a new dictionary."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = value
    return result


def apply_quick_test_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce every experiment to a short smoke test without changing the API."""
    if not config["global"]["quick_test"]:
        return config
    overrides = {
        "gaussian": {
            "epochs": 2,
            "steps_per_epoch": 4,
            "batch_tasks": 8,
            "samples_per_task": 16,
            "validation_batches": 2,
            "visualisation_samples": 200,
        },
        "mnist": {
            "max_train_tasks": 64,
            "max_validation_tasks": 32,
            "max_test_tasks": 32,
            "epochs": 2,
            "batch_tasks": 8,
            "validation_batch_tasks": 8,
            "samples_per_task": 32,
            "visualisation_examples": 3,
            "visualisation_samples": 500,
        },
    }
    return deep_update(config, overrides)


CONFIG = apply_quick_test_overrides(CONFIG)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    """Convert common NumPy/JAX/Python objects into JSON-serialisable values."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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


def make_mnist_grid() -> np.ndarray:
    """Pixel-centre coordinates in [-1, 1]^2, flattened in image row-major order."""
    pixel = 2.0 / 28.0
    x_coordinates = np.linspace(-1.0 + pixel / 2.0, 1.0 - pixel / 2.0, 28)
    y_coordinates = np.linspace(1.0 - pixel / 2.0, -1.0 + pixel / 2.0, 28)
    xx, yy = np.meshgrid(x_coordinates, y_coordinates, indexing="xy")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1).astype(np.float32)


def images_to_probability_weights(
    images: np.ndarray,
    mass_floor: float,
) -> np.ndarray:
    intensities = images.astype(np.float32).reshape(images.shape[0], -1) / 255.0
    weights = intensities + float(mass_floor)
    normalisers = weights.sum(axis=1, keepdims=True)
    if np.any(normalisers <= 0.0):
        raise ValueError("Every MNIST target must have positive total mass.")
    return weights / normalisers


def target_context_from_weights(weights: Array, mode: str) -> Array:
    if mode == "weights":
        return weights
    if mode == "scaled":
        return weights * weights.shape[-1]
    if mode == "sqrt_scaled":
        return jnp.sqrt(jnp.maximum(weights * weights.shape[-1], 0.0))
    raise ValueError(f"Unsupported target_context_mode={mode!r}.")


# %%
# -----------------------------------------------------------------------------
# Gaussian -> MNIST experiment.
# -----------------------------------------------------------------------------
class MNISTBatch(NamedTuple):
    source_parameters: Array  # [B, 4]
    target_context: Array  # [B, 784]
    target_weights: Array  # [B, 784]
    source_points: Array  # [B, N, 2]
    target_points: Array  # [B, N, 2]


def sample_weighted_grid_points(
    key: Array,
    weights: Array,
    grid: Array,
    count: int,
    add_jitter: bool,
) -> Array:
    batch_size = weights.shape[0]
    categorical_key, jitter_key = jr.split(key)
    task_keys = jr.split(categorical_key, batch_size)

    def sample_indices(task_key: Array, task_weights: Array) -> Array:
        logits = jnp.log(jnp.maximum(task_weights, 1.0e-12))
        return jr.categorical(task_key, logits, shape=(count,))

    indices = jax.vmap(sample_indices)(task_keys, weights)
    points = grid[indices]
    if add_jitter:
        half_pixel = 1.0 / 28.0
        jitter = jr.uniform(
            jitter_key,
            points.shape,
            minval=-half_pixel,
            maxval=half_pixel,
        )
        points = jnp.clip(points + jitter, -1.0, 1.0)
    return points


def make_mnist_batch(
    key: Array,
    images: np.ndarray,
    config: Mapping[str, Any],
    grid: Array,
) -> MNISTBatch:
    if images.shape[0] != int(config["batch_tasks"]):
        raise ValueError(
            f"Expected a fixed batch of {int(config['batch_tasks'])} images, "
            f"received {images.shape[0]}. Drop incomplete batches to avoid recompilation."
        )
    keys = jr.split(key, 3)
    weights_np = images_to_probability_weights(images, float(config["pixel_mass_floor"]))
    weights = jnp.asarray(weights_np)
    context = target_context_from_weights(weights, str(config["target_context_mode"]))
    source_parameters = sample_diagonal_gaussian_parameters(
        keys[0],
        int(config["batch_tasks"]),
        2,
        config["source_mean_range"],
        config["source_std_range"],
    )
    source_points = sample_from_diagonal_gaussian(
        keys[1], source_parameters, int(config["samples_per_task"])
    )
    target_points = sample_weighted_grid_points(
        keys[2],
        weights,
        grid,
        int(config["samples_per_task"]),
        bool(config["target_jitter"]),
    )
    return MNISTBatch(source_parameters, context, weights, source_points, target_points)


def make_fixed_mnist_validation_batches(
    key: Array,
    images: np.ndarray,
    config: Mapping[str, Any],
    grid: Array,
) -> List[MNISTBatch]:
    validation_config = dict(config)
    validation_config["batch_tasks"] = int(config["validation_batch_tasks"])
    batch_size = int(validation_config["batch_tasks"])
    batch_count = images.shape[0] // batch_size
    keys = jr.split(key, batch_count)
    return [
        make_mnist_batch(
            keys[index],
            images[index * batch_size : (index + 1) * batch_size],
            validation_config,
            grid,
        )
        for index in range(batch_count)
    ]


def build_mnist_model(key: Array, config: Mapping[str, Any]) -> WeightConditionedTransport:
    return WeightConditionedTransport(
        source_dim=4,
        target_dim=28 * 28,
        point_dim=2,
        output_dim=2,
        context_dim=int(config["context_dim"]),
        source_width=int(config["source_encoder_width"]),
        target_width=int(config["target_encoder_width"]),
        point_width=int(config["point_encoder_width"]),
        encoder_depth=int(config["encoder_depth"]),
        fusion_width=int(config["fusion_width"]),
        fusion_depth=int(config["fusion_depth"]),
        output_mode="tanh_residual",
        residual_scale=float(config["residual_scale"]),
        key=key,
    )


def make_mnist_loss(
    rff: RFFParameters,
    grid: Array,
    grid_rff_features: Array,
    projection_directions: Array,
    loss_weights: Mapping[str, float],
):
    grid_squared = jnp.square(grid)

    def loss_fn(
        model: WeightConditionedTransport,
        batch: MNISTBatch,
    ) -> Tuple[Array, Dict[str, Array]]:
        predicted = apply_transport_batch(
            model,
            batch.source_parameters,
            batch.target_context,
            batch.source_points,
        )
        predicted_feature_mean = jnp.mean(rff_features(predicted, rff), axis=1)
        target_feature_mean = jnp.einsum("bk,kr->br", batch.target_weights, grid_rff_features)
        mmd_per_task = rff_mmd_from_means(predicted_feature_mean, target_feature_mean)

        sliced_per_task = sliced_wasserstein_squared(
            predicted, batch.target_points, projection_directions
        )
        displacement_per_task = jnp.mean(
            jnp.sum(jnp.square(predicted - batch.source_points), axis=-1), axis=1
        )

        predicted_mean = jnp.mean(predicted, axis=1)
        predicted_second = jnp.mean(jnp.square(predicted), axis=1)
        target_mean = jnp.einsum("bk,kd->bd", batch.target_weights, grid)
        target_second = jnp.einsum("bk,kd->bd", batch.target_weights, grid_squared)
        moment_per_task = jnp.mean(jnp.square(predicted_mean - target_mean), axis=-1)
        moment_per_task += jnp.mean(
            jnp.square(predicted_second - target_second), axis=-1
        )

        total_per_task = (
            float(loss_weights["rff_mmd"]) * mmd_per_task
            + float(loss_weights["sliced_w2"]) * sliced_per_task
            + float(loss_weights["moments"]) * moment_per_task
            + float(loss_weights["displacement"]) * displacement_per_task
        )
        metrics = {
            "total": jnp.mean(total_per_task),
            "rff_mmd": jnp.mean(mmd_per_task),
            "sliced_w2": jnp.mean(sliced_per_task),
            "moments": jnp.mean(moment_per_task),
            "displacement": jnp.mean(displacement_per_task),
        }
        return metrics["total"], metrics

    return loss_fn


def iterate_fixed_numpy_batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Images and labels must have the same leading dimension.")
    indices = np.arange(images.shape[0])
    if shuffle:
        rng.shuffle(indices)
    usable = (indices.size // batch_size) * batch_size
    for start in range(0, usable, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]


def plot_mnist_qualitative(
    model: WeightConditionedTransport,
    key: Array,
    images: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
    grid: Array,
    output_dir: Path,
) -> None:
    count = min(int(config["visualisation_examples"]), images.shape[0])
    sample_count = int(config["visualisation_samples"])
    keys = jr.split(key, 4)
    selected_images = images[:count]
    selected_labels = labels[:count]
    weights_np = images_to_probability_weights(
        selected_images, float(config["pixel_mass_floor"])
    )
    weights = jnp.asarray(weights_np)
    target_context = target_context_from_weights(weights, str(config["target_context_mode"]))
    source_parameters = sample_diagonal_gaussian_parameters(
        keys[0],
        count,
        2,
        config["source_mean_range"],
        config["source_std_range"],
    )
    source_points = sample_from_diagonal_gaussian(keys[1], source_parameters, sample_count)
    predicted = apply_transport_batch(model, source_parameters, target_context, source_points)
    source_np, predicted_np = map(np.asarray, jax.device_get((source_points, predicted)))

    figure, axes = plt.subplots(count, 3, figsize=(12, 3.8 * count), squeeze=False)
    for row in range(count):
        axes[row, 0].imshow(selected_images[row], cmap="gray", origin="upper")
        axes[row, 0].set_title(f"Target digit {int(selected_labels[row])}")
        axes[row, 0].axis("off")

        axes[row, 1].scatter(
            source_np[row, :, 0], source_np[row, :, 1], s=4, alpha=0.25
        )
        axes[row, 1].set_xlim(-2.5, 2.5)
        axes[row, 1].set_ylim(-2.5, 2.5)
        axes[row, 1].set_aspect("equal", adjustable="box")
        axes[row, 1].set_title("Source Gaussian samples")
        axes[row, 1].grid(True, alpha=0.2)

        axes[row, 2].hist2d(
            predicted_np[row, :, 0],
            predicted_np[row, :, 1],
            bins=28,
            range=[[-1.0, 1.0], [-1.0, 1.0]],
            density=True,
        )
        axes[row, 2].scatter(
            predicted_np[row, :: max(1, sample_count // 400), 0],
            predicted_np[row, :: max(1, sample_count // 400), 1],
            s=3,
            alpha=0.15,
        )
        axes[row, 2].set_xlim(-1.0, 1.0)
        axes[row, 2].set_ylim(-1.0, 1.0)
        axes[row, 2].set_aspect("equal", adjustable="box")
        axes[row, 2].set_title("Learned push-forward density")
    figure.tight_layout()
    figure.savefig(output_dir / "mnist_qualitative.png", dpi=180)
    plt.close(figure)


def evaluate_model_over_batches(
    model: WeightConditionedTransport,
    batches: Sequence[MNISTBatch],
    evaluation_step,
) -> Dict[str, float]:
    metrics = [scalar_metrics_to_python(evaluation_step(model, batch)) for batch in batches]
    return average_metric_dicts(metrics)


def train_mnist_experiment(
    key: Array,
    config: Mapping[str, Any],
    output_dir: Path,
) -> Tuple[WeightConditionedTransport, Dict[str, List[float]], Dict[str, List[float]]]:
    model_key, rff_key, projection_key, validation_key, training_key, plot_key = jr.split(key, 6)
    data = load_mnist(config, seed=int(CONFIG["global"]["seed"]))
    print(
        "MNIST task counts:",
        data.train_images.shape[0],
        data.validation_images.shape[0],
        data.test_images.shape[0],
    )

    grid = jnp.asarray(make_mnist_grid())
    rff = make_rff_parameters(
        rff_key,
        2,
        config["rff"]["bandwidths"],
        int(config["rff"]["features_per_bandwidth"]),
    )
    grid_features = rff_features(grid, rff)
    directions = make_projection_directions(
        projection_key, int(config["sliced_wasserstein_projections"]), 2
    )
    model = build_mnist_model(model_key, config)
    loss_fn = make_mnist_loss(
        rff,
        grid,
        grid_features,
        directions,
        config["loss_weights"],
    )
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(
        current_model: WeightConditionedTransport,
        current_opt_state: PyTree,
        batch: MNISTBatch,
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
    def evaluation_step(current_model: WeightConditionedTransport, batch: MNISTBatch):
        _, metrics = loss_fn(current_model, batch)
        return metrics

    validation_batches = make_fixed_mnist_validation_batches(
        validation_key, data.validation_images, config, grid
    )
    if not validation_batches:
        raise ValueError("The validation set is too small for one fixed validation batch.")

    step_history: Dict[str, List[float]] = {}
    epoch_history: Dict[str, List[float]] = {}
    global_step = 0
    start_time = time.time()
    numpy_rng = np.random.default_rng(int(CONFIG["global"]["seed"]) + 100)
    print(f"MNIST model parameters: {tree_parameter_count(model):,}")
    if CONFIG["global"]["print_model"]:
        print(model)

    for epoch in range(1, int(config["epochs"]) + 1):
        epoch_metrics: List[Dict[str, float]] = []
        batch_iterator = iterate_fixed_numpy_batches(
            data.train_images,
            data.train_labels,
            int(config["batch_tasks"]),
            numpy_rng,
            shuffle=True,
        )
        for images, _labels in batch_iterator:
            training_key, batch_key = jr.split(training_key)
            batch = make_mnist_batch(batch_key, images, config, grid)
            model, opt_state, _, metrics = train_step(model, opt_state, batch)
            metrics_python = scalar_metrics_to_python(metrics)
            append_step_history(step_history, global_step, metrics_python)
            epoch_metrics.append(metrics_python)
            global_step += 1
            if global_step % int(config["log_every_steps"]) == 0:
                print(
                    f"[MNIST] step={global_step:6d} "
                    f"total={metrics_python['total']:.6e} "
                    f"mmd={metrics_python['rff_mmd']:.6e} "
                    f"sliced_w2={metrics_python['sliced_w2']:.6e}"
                )

        if not epoch_metrics:
            raise ValueError("Training set is smaller than one fixed-size batch.")
        train_average = average_metric_dicts(epoch_metrics)
        validation_average = evaluate_model_over_batches(
            model, validation_batches, evaluation_step
        )
        append_epoch_history(epoch_history, epoch, train_average, validation_average)
        elapsed = time.time() - start_time
        print(
            f"[MNIST] epoch={epoch:3d}/{int(config['epochs'])} "
            f"train={train_average['total']:.6e} "
            f"validation={validation_average['total']:.6e} "
            f"validation_mmd={validation_average['rff_mmd']:.6e} "
            f"elapsed={elapsed:.1f}s"
        )

    # A fixed test evaluation uses the validation batch size to avoid recompilation.
    test_batches = make_fixed_mnist_validation_batches(
        jr.fold_in(validation_key, 1), data.test_images, config, grid
    )
    if not test_batches:
        raise ValueError("The test set is too small for one fixed test batch.")
    test_metrics = evaluate_model_over_batches(model, test_batches, evaluation_step)
    print("[MNIST] final test metrics:", test_metrics)
    save_json(test_metrics, output_dir / "mnist_test_metrics.json")

    eqx.tree_serialise_leaves(output_dir / "mnist_model.eqx", model)
    save_histories(step_history, epoch_history, output_dir / "mnist_history.npz")
    plot_history(step_history, epoch_history, output_dir, prefix="mnist")
    plot_mnist_qualitative(
        model,
        plot_key,
        data.test_images,
        data.test_labels,
        config,
        grid,
        output_dir,
    )
    return model, step_history, epoch_history


# %%
# -----------------------------------------------------------------------------
# Fast shape-and-finiteness smoke tests. These do not train the models.
# -----------------------------------------------------------------------------
def run_shape_smoke_tests(key: Array) -> None:
    gaussian_config = dict(CONFIG["gaussian"])
    gaussian_config["batch_tasks"] = 2
    gaussian_config["samples_per_task"] = 4
    gaussian_keys = jr.split(key, 4)
    gaussian_model = build_gaussian_model(gaussian_keys[0], gaussian_config)
    gaussian_batch = make_gaussian_batch(gaussian_keys[1], gaussian_config)
    gaussian_rff = make_rff_parameters(
        gaussian_keys[2],
        int(gaussian_config["dimension"]),
        gaussian_config["rff"]["bandwidths"],
        4,
    )
    gaussian_loss, gaussian_metrics = make_gaussian_loss(
        gaussian_rff, gaussian_config["loss_weights"]
    )(gaussian_model, gaussian_batch)
    if not bool(np.asarray(jnp.isfinite(gaussian_loss))):
        raise FloatingPointError("Gaussian smoke-test loss is non-finite.")
    expected_gaussian_shape = (2, 4, int(gaussian_config["dimension"]))
    gaussian_prediction = apply_transport_batch(
        gaussian_model,
        gaussian_batch.source_parameters,
        gaussian_batch.target_parameters,
        gaussian_batch.source_points,
    )
    if gaussian_prediction.shape != expected_gaussian_shape:
        raise AssertionError(
            f"Gaussian prediction shape {gaussian_prediction.shape} != {expected_gaussian_shape}."
        )

    mnist_config = dict(CONFIG["mnist"])
    mnist_config["batch_tasks"] = 2
    mnist_config["samples_per_task"] = 4
    mnist_keys = jr.split(gaussian_keys[3], 5)
    mnist_model = build_mnist_model(mnist_keys[0], mnist_config)
    grid = jnp.asarray(make_mnist_grid())
    synthetic_images = np.zeros((2, 28, 28), dtype=np.uint8)
    synthetic_images[0, 5:23, 12:16] = 255
    synthetic_images[1, 12:16, 5:23] = 255
    mnist_batch = make_mnist_batch(mnist_keys[1], synthetic_images, mnist_config, grid)
    mnist_rff = make_rff_parameters(mnist_keys[2], 2, [0.1, 0.4], 4)
    grid_features = rff_features(grid, mnist_rff)
    directions = make_projection_directions(mnist_keys[3], 4, 2)
    mnist_loss, mnist_metrics = make_mnist_loss(
        mnist_rff,
        grid,
        grid_features,
        directions,
        mnist_config["loss_weights"],
    )(mnist_model, mnist_batch)
    if not bool(np.asarray(jnp.isfinite(mnist_loss))):
        raise FloatingPointError("MNIST smoke-test loss is non-finite.")
    mnist_prediction = apply_transport_batch(
        mnist_model,
        mnist_batch.source_parameters,
        mnist_batch.target_context,
        mnist_batch.source_points,
    )
    if mnist_prediction.shape != (2, 4, 2):
        raise AssertionError(f"Unexpected MNIST prediction shape {mnist_prediction.shape}.")
    print(
        "Shape smoke tests passed:",
        f"Gaussian loss={float(np.asarray(gaussian_loss)):.4g},",
        f"MNIST loss={float(np.asarray(mnist_loss)):.4g}.",
    )


# %%
# -----------------------------------------------------------------------------
# Main entry point.
# -----------------------------------------------------------------------------
def main() -> None:
    print_runtime_information()
    output_root = ensure_dir(CONFIG["global"]["output_dir"])
    save_json(CONFIG, output_root / "config.json")
    master_key = jr.PRNGKey(int(CONFIG["global"]["seed"]))
    if bool(CONFIG["global"]["run_shape_smoke_tests"]):
        master_key, smoke_key = jr.split(master_key)
        run_shape_smoke_tests(smoke_key)
    requested = {str(name).lower() for name in CONFIG["global"]["run"]}
    unknown = requested.difference({"gaussian", "mnist"})
    if unknown:
        raise ValueError(f"Unknown experiments requested: {sorted(unknown)}")

    if "gaussian" in requested:
        master_key, experiment_key = jr.split(master_key)
        gaussian_dir = ensure_dir(output_root / "gaussian")
        save_json(CONFIG["gaussian"], gaussian_dir / "config.json")
        train_gaussian_experiment(experiment_key, CONFIG["gaussian"], gaussian_dir)

    if "mnist" in requested:
        master_key, experiment_key = jr.split(master_key)
        mnist_dir = ensure_dir(output_root / "mnist")
        save_json(CONFIG["mnist"], mnist_dir / "config.json")
        train_mnist_experiment(experiment_key, CONFIG["mnist"], mnist_dir)

    print(f"Finished. Outputs written under: {output_root.resolve()}")


# %%
if __name__ == "__main__":
    main()
