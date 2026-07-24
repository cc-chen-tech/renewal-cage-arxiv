#!/usr/bin/env python3
"""Test whether microscopic ``L2p`` closes the relative Mori bath."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_collective_memory import discrete_mori_zwanzig_operators  # noqa: E402
from ka_markov_bath import (  # noqa: E402
    fit_stationary_vector_autoregression,
    simulate_mori_with_var_bath,
    vector_autoregressive_residual,
)
from ka_relative_mori import finite_memory_innovation_series  # noqa: E402


COORDINATE_NAMES = ("u", "p", "Lp", "L2p")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def output_path(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def _scalar_string(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("metadata fields must be scalar")
    return str(array.item())


def load_matched_second_generator_clones(
    drift_cache_directory: Path,
    second_generator_cache_directories: list[Path],
    *,
    maximum_frame_count: int,
) -> list[dict[str, np.ndarray | float | str]]:
    """Pair drift and ``L2p`` caches by immutable trajectory hash."""

    if maximum_frame_count < 0 or not second_generator_cache_directories:
        raise ValueError("second-generator cache controls are invalid")
    drift_paths = sorted(Path(drift_cache_directory).glob("clone_*_decomposed_drift.npz"))
    second_paths = sorted(
        path
        for directory in second_generator_cache_directories
        for path in Path(directory).glob("clone_*_relative_second_generator.npz")
    )
    if not drift_paths or not second_paths:
        raise ValueError("drift and second-generator caches are required")

    second_by_hash: dict[str, dict[str, np.ndarray | float | str]] = {}
    for path in second_paths:
        with np.load(path, allow_pickle=False) as cache:
            source_hash = _scalar_string(cache["trajectory_sha256"])
            if source_hash in second_by_hash:
                raise ValueError(f"duplicate second-generator trajectory hash: {source_hash}")
            claim = float(cache["thermodynamic_claim_allowed"])
            completed = int(float(cache["completed_frame_count"]))
            requested = int(float(cache["requested_frame_count"]))
            second = np.asarray(cache["second_relative_generator"], dtype=float)
            targets = np.asarray(cache["target_indices"], dtype=int)
            if (
                claim != 0.0
                or completed < 1
                or completed > requested
                or second.ndim != 3
                or second.shape[1:] != (len(targets), 3)
                or completed > len(second)
                or np.any(~np.isfinite(second[:completed]))
            ):
                raise ValueError(f"invalid second-generator cache: {path}")
            second_by_hash[source_hash] = {
                "second_relative_generator": second[:completed].copy(),
                "target_indices": targets.copy(),
                "completed_frame_count": float(completed),
                "potential_protocol": _scalar_string(cache["potential_protocol"]),
                "trace_probe_count": float(cache["trace_probe_count"]),
                "second_generator_cache_path": str(path.resolve()),
            }

    clones: list[dict[str, np.ndarray | float | str]] = []
    drift_hashes: set[str] = set()
    for path in drift_paths:
        with np.load(path, allow_pickle=False) as cache:
            source_hash = _scalar_string(cache["trajectory_sha256"])
            if source_hash in drift_hashes or source_hash not in second_by_hash:
                raise ValueError(f"drift/second-generator hash mismatch: {source_hash}")
            drift_hashes.add(source_hash)
            claim = float(cache["thermodynamic_claim_allowed"])
            targets = np.asarray(cache["target_indices"], dtype=int)
            position = np.asarray(cache["relative_position"], dtype=float)
            velocity = np.asarray(cache["relative_velocity"], dtype=float)
            drift = np.asarray(cache["relative_drift"], dtype=float)
        second = second_by_hash[source_hash]
        second_values = np.asarray(second["second_relative_generator"], dtype=float)
        frame_count = min(len(position), len(velocity), len(drift), len(second_values))
        if maximum_frame_count:
            frame_count = min(frame_count, maximum_frame_count)
        if (
            claim != 0.0
            or not np.array_equal(targets, np.asarray(second["target_indices"]))
            or position.shape != velocity.shape
            or position.shape != drift.shape
            or position.shape[1:] != (len(targets), 3)
            or frame_count < 2
            or np.any(~np.isfinite(position[:frame_count]))
            or np.any(~np.isfinite(velocity[:frame_count]))
            or np.any(~np.isfinite(drift[:frame_count]))
        ):
            raise ValueError(f"invalid matched drift cache: {path}")
        clones.append(
            {
                "relative_position": position[:frame_count].copy(),
                "relative_velocity": velocity[:frame_count].copy(),
                "relative_drift": drift[:frame_count].copy(),
                "second_relative_generator": second_values[:frame_count].copy(),
                "target_indices": targets.copy(),
                "trajectory_sha256": source_hash,
                "potential_protocol": str(second["potential_protocol"]),
                "trace_probe_count": float(second["trace_probe_count"]),
                "drift_cache_path": str(path.resolve()),
                "second_generator_cache_path": str(
                    second["second_generator_cache_path"]
                ),
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    if drift_hashes != set(second_by_hash):
        raise ValueError("second-generator caches contain unmatched trajectories")
    protocols = {str(clone["potential_protocol"]) for clone in clones}
    probe_counts = {float(clone["trace_probe_count"]) for clone in clones}
    if len(protocols) != 1 or len(probe_counts) != 1:
        raise ValueError("matched clones must use one frozen numerical protocol")
    return clones


def second_generator_resolved_state(
    clone: dict[str, np.ndarray | float | str],
    *,
    bias: np.ndarray,
    include_l2p: bool,
) -> np.ndarray:
    """Return matched ``(u-bias,p,Lp[,L2p])`` component paths."""

    position = np.asarray(clone["relative_position"], dtype=float)
    velocity = np.asarray(clone["relative_velocity"], dtype=float)
    drift = np.asarray(clone["relative_drift"], dtype=float)
    second = np.asarray(clone["second_relative_generator"], dtype=float)
    cage_bias = np.asarray(bias, dtype=float)
    if (
        position.ndim != 3
        or position.shape[-1] != 3
        or velocity.shape != position.shape
        or drift.shape != position.shape
        or second.shape != position.shape
        or cage_bias.shape != position.shape[1:]
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(drift))
        or np.any(~np.isfinite(second))
        or np.any(~np.isfinite(cage_bias))
    ):
        raise ValueError("relative generator coordinates must be finite and aligned")
    fields = [
        (position - cage_bias[None]).reshape(len(position), -1),
        velocity.reshape(len(position), -1),
        drift.reshape(len(position), -1),
    ]
    if include_l2p:
        fields.append(second.reshape(len(position), -1))
    return np.stack(fields, axis=2)


def white_residual_shape_diagnostic(
    residual: np.ndarray,
    *,
    maximum_lag: int,
) -> dict[str, np.ndarray]:
    """Measure linear, volatility, and marginal non-Gaussian residual defects."""

    values = np.asarray(residual, dtype=float)
    if (
        values.ndim != 3
        or values.shape[2] < 1
        or maximum_lag < 1
        or maximum_lag >= len(values)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("residual and maximum_lag must be finite and aligned")
    centered = values - np.mean(values, axis=(0, 1), keepdims=True)
    variance = np.mean(centered**2, axis=(0, 1))
    if np.any(variance <= 1e-30):
        raise ValueError("every residual component needs nonzero variance")
    squared = centered**2
    centered_squared = squared - np.mean(squared, axis=(0, 1), keepdims=True)
    squared_variance = np.mean(centered_squared**2, axis=(0, 1))
    if np.any(squared_variance <= 1e-30):
        raise ValueError("every squared residual component needs nonzero variance")

    correlation = np.empty((maximum_lag, values.shape[2]), dtype=float)
    squared_correlation = np.empty_like(correlation)
    for slot, lag in enumerate(range(1, maximum_lag + 1)):
        correlation[slot] = np.mean(
            centered[lag:] * centered[:-lag], axis=(0, 1)
        ) / variance
        squared_correlation[slot] = np.mean(
            centered_squared[lag:] * centered_squared[:-lag], axis=(0, 1)
        ) / squared_variance
    excess_kurtosis = np.mean(centered**4, axis=(0, 1)) / variance**2 - 3.0
    positive_kurtosis = np.maximum(excess_kurtosis, 0.0)
    return {
        "correlation_by_lag": correlation,
        "squared_correlation_by_lag": squared_correlation,
        "maximum_absolute_correlation": np.max(np.abs(correlation), axis=0),
        "maximum_absolute_squared_correlation": np.max(
            np.abs(squared_correlation), axis=0
        ),
        "excess_kurtosis": excess_kurtosis,
        "gaussian_scale_mixture_squared_correlation_bound": (
            positive_kurtosis / (3.0 * (positive_kurtosis + 2.0))
        ),
    }


def stationary_correlation(state: np.ndarray, maximum_lag: int) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    if (
        values.ndim != 3
        or maximum_lag < 0
        or maximum_lag >= len(values)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("state and maximum_lag must be finite and aligned")
    output = np.empty((maximum_lag + 1, values.shape[2], values.shape[2]))
    for lag in range(maximum_lag + 1):
        left = values[lag:]
        right = values[: len(values) - lag]
        output[lag] = np.einsum("tni,tnj->ij", left, right) / (
            left.shape[0] * left.shape[1]
        )
    return output


def extract_l2p_white_innovation_fold(
    training: list[dict[str, np.ndarray | float | str]],
    held: dict[str, np.ndarray | float | str],
    *,
    memory_order: int,
    bath_order: int,
    ridge_regularization: float,
    var_ridge_regularization: float,
    include_l2p: bool = True,
) -> dict[str, np.ndarray | float | dict[str, np.ndarray | float]]:
    """Fit one microscopic Mori/VAR fold and retain exact frame alignment."""

    if len(training) < 2 or memory_order < 1 or bath_order < 1:
        raise ValueError("innovation extraction requires training clones and positive orders")
    frame_count = min(
        len(np.asarray(clone["relative_position"])) for clone in [*training, held]
    )
    if frame_count <= memory_order + bath_order + 2:
        raise ValueError("matched paths are too short for frozen Mori and VAR controls")
    bias = np.mean(
        np.concatenate(
            [np.asarray(clone["relative_position"])[:frame_count] for clone in training],
            axis=0,
        ),
        axis=0,
    )

    def truncate(clone: dict[str, np.ndarray | float | str]) -> dict[str, object]:
        return {
            key: (
                np.asarray(value)[:frame_count]
                if key
                in {
                    "relative_position",
                    "relative_velocity",
                    "relative_drift",
                    "second_relative_generator",
                }
                else value
            )
            for key, value in clone.items()
        }

    training_state = np.concatenate(
        [
            second_generator_resolved_state(
                truncate(clone),
                bias=bias,
                include_l2p=include_l2p,
            )
            for clone in training
        ],
        axis=1,
    )
    held_state = second_generator_resolved_state(
        truncate(held),
        bias=bias,
        include_l2p=include_l2p,
    )
    mean = np.mean(training_state, axis=(0, 1), keepdims=True)
    scale = np.std(training_state, axis=(0, 1), keepdims=True)
    if np.any(scale <= 1e-12):
        raise ValueError("resolved generator coordinates need nonzero variance")
    training_state = (training_state - mean) / scale
    held_state = (held_state - mean) / scale
    mori = discrete_mori_zwanzig_operators(
        training_state,
        memory_order=memory_order,
        ridge_regularization=ridge_regularization,
    )
    operators = np.asarray(mori["operators"])
    training_innovation = finite_memory_innovation_series(training_state, operators)
    held_innovation = finite_memory_innovation_series(held_state, operators)
    var = fit_stationary_vector_autoregression(
        training_innovation,
        order=bath_order,
        ridge_regularization=var_ridge_regularization,
    )
    coefficients = np.asarray(var["coefficients"])
    bath_mean = np.asarray(var["mean"])
    training_white = vector_autoregressive_residual(
        training_innovation, coefficients, mean=bath_mean
    )
    held_white = vector_autoregressive_residual(
        held_innovation, coefficients, mean=bath_mean
    )
    first_source_frame = (len(operators) - 1) + 1 + len(coefficients)
    held_source_frames = np.arange(first_source_frame, frame_count, dtype=int)
    if len(held_source_frames) != len(held_white):
        raise ValueError("white-innovation frame alignment is inconsistent")
    target_count = np.asarray(held["relative_position"]).shape[1]
    if held_white.shape[1] != 3 * target_count:
        raise ValueError("white innovations do not preserve target-component alignment")
    held_l2p_vector = (
        held_white[:, :, 3].reshape(len(held_white), target_count, 3)
        if include_l2p
        else np.empty((len(held_white), target_count, 0))
    )
    return {
        "frame_count": float(frame_count),
        "bias": bias,
        "normalization_mean": mean,
        "normalization_scale": scale,
        "training_state": training_state,
        "held_state": held_state,
        "operators": operators,
        "training_innovation": training_innovation,
        "held_innovation": held_innovation,
        "var": var,
        "var_coefficients": coefficients,
        "var_mean": bath_mean,
        "training_white_innovation": training_white,
        "held_white_innovation": held_white,
        "held_l2p_vector_innovation": held_l2p_vector,
        "held_target_count": float(target_count),
        "held_source_frame_indices": held_source_frames,
        "thermodynamic_claim_allowed": 0.0,
    }


def score_fold_model(
    training: list[dict[str, np.ndarray | float | str]],
    held: dict[str, np.ndarray | float | str],
    *,
    include_l2p: bool,
    fold_index: int,
    protocol: str,
    memory_order: int,
    bath_order: int,
    maximum_lag: int,
    simulation_count: int,
    ridge_regularization: float,
    var_ridge_regularization: float,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    model = "relative_phase_generator_l2p" if include_l2p else "relative_phase_generator"
    extracted = extract_l2p_white_innovation_fold(
        training,
        held,
        memory_order=memory_order,
        bath_order=bath_order,
        ridge_regularization=ridge_regularization,
        var_ridge_regularization=var_ridge_regularization,
        include_l2p=include_l2p,
    )
    frame_count = int(extracted["frame_count"])
    if frame_count <= memory_order + bath_order + maximum_lag + 2:
        raise ValueError("matched paths are too short for frozen Mori and VAR controls")
    training_state = np.asarray(extracted["training_state"])
    held_state = np.asarray(extracted["held_state"])
    operators = np.asarray(extracted["operators"])
    training_innovation = np.asarray(extracted["training_innovation"])
    var = extracted["var"]
    coefficients = np.asarray(extracted["var_coefficients"])
    bath_mean = np.asarray(extracted["var_mean"])
    training_white = np.asarray(extracted["training_white_innovation"])
    held_white = np.asarray(extracted["held_white_innovation"])
    training_shape = white_residual_shape_diagnostic(
        training_white, maximum_lag=maximum_lag
    )
    held_shape = white_residual_shape_diagnostic(
        held_white, maximum_lag=maximum_lag
    )

    rng = np.random.default_rng(seed + fold_index)
    effective_memory_order = len(operators) - 1
    minimum_current = effective_memory_order + bath_order
    current = rng.integers(minimum_current, frame_count - 1, size=simulation_count)
    source = rng.integers(training_state.shape[1], size=simulation_count)
    state_history = np.stack(
        [
            training_state[time - effective_memory_order : time + 1, particle]
            for time, particle in zip(current, source, strict=True)
        ]
    )
    innovation_history = np.stack(
        [
            training_innovation[
                time - effective_memory_order - bath_order : time - effective_memory_order,
                particle,
            ]
            for time, particle in zip(current, source, strict=True)
        ]
    )
    simulated, _ = simulate_mori_with_var_bath(
        state_history,
        operators,
        innovation_history,
        coefficients,
        innovation_mean=bath_mean,
        white_noise_covariance=np.asarray(var["white_noise_covariance"]),
        output_count=maximum_lag + 1,
        rng=np.random.default_rng(seed + 1000 + fold_index),
    )
    held_correlation = stationary_correlation(held_state, maximum_lag)
    simulated_correlation = np.einsum(
        "nti,nj->tij", simulated, simulated[:, 0]
    ) / simulation_count
    simulated_covariance = np.einsum(
        "nti,ntj->tij", simulated, simulated
    ) / simulation_count
    stationary_covariance_error = simulated_covariance - held_correlation[0]
    target_correlation_error = (
        simulated_correlation[:, :2, :2] - held_correlation[:, :2, :2]
    )

    detail_rows: list[dict[str, object]] = []
    dimension = held_state.shape[2]
    for coordinate in range(dimension):
        detail_rows.append(
            {
                "record": "coordinate_fold",
                "protocol": protocol,
                "model": model,
                "fold_index": float(fold_index),
                "coordinate": COORDINATE_NAMES[coordinate],
                "coordinate_index": float(coordinate),
                "frame_count": float(frame_count),
                "training_source_count": float(training_state.shape[1]),
                "held_source_count": float(held_state.shape[1]),
                "memory_order": float(memory_order),
                "bath_order": float(bath_order),
                "potential_protocol": str(held["potential_protocol"]),
                "trace_probe_count": float(held["trace_probe_count"]),
                "var_spectral_radius": float(var["spectral_radius"]),
                "minimum_white_noise_covariance_eigenvalue": float(
                    var["minimum_noise_covariance_eigenvalue"]
                ),
                "training_maximum_absolute_white_residual_correlation": float(
                    training_shape["maximum_absolute_correlation"][coordinate]
                ),
                "training_maximum_absolute_squared_white_residual_correlation": float(
                    training_shape["maximum_absolute_squared_correlation"][coordinate]
                ),
                "training_white_residual_excess_kurtosis": float(
                    training_shape["excess_kurtosis"][coordinate]
                ),
                "held_maximum_absolute_white_residual_correlation": float(
                    held_shape["maximum_absolute_correlation"][coordinate]
                ),
                "held_maximum_absolute_squared_white_residual_correlation": float(
                    held_shape["maximum_absolute_squared_correlation"][coordinate]
                ),
                "held_white_residual_excess_kurtosis": float(
                    held_shape["excess_kurtosis"][coordinate]
                ),
                "held_gaussian_scale_mixture_squared_correlation_bound": float(
                    held_shape[
                        "gaussian_scale_mixture_squared_correlation_bound"
                    ][coordinate]
                ),
                "thermodynamic_claim_allowed": 0.0,
            }
        )

    correlation_rows: list[dict[str, object]] = []
    for lag in range(maximum_lag + 1):
        for row_coordinate in range(2):
            for column_coordinate in range(2):
                correlation_rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "fold_index": float(fold_index),
                        "lag_frames": float(lag),
                        "row_coordinate": COORDINATE_NAMES[row_coordinate],
                        "column_coordinate": COORDINATE_NAMES[column_coordinate],
                        "held_correlation": float(
                            held_correlation[lag, row_coordinate, column_coordinate]
                        ),
                        "simulated_correlation": float(
                            simulated_correlation[lag, row_coordinate, column_coordinate]
                        ),
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )

    lp_coordinate = 2
    white_gate = (
        float(var["spectral_radius"]) < 1.0
        and float(var["minimum_noise_covariance_eigenvalue"]) >= -1e-10
        and float(np.max(held_shape["maximum_absolute_correlation"])) <= 0.05
        and float(np.max(held_shape["maximum_absolute_squared_correlation"])) <= 0.05
        and float(np.max(np.abs(held_shape["excess_kurtosis"]))) <= 0.35
    )
    stationary_covariance_maximum_error = float(
        np.max(np.abs(stationary_covariance_error))
    )
    target_correlation_maximum_error = float(
        np.max(np.abs(target_correlation_error))
    )
    summary = {
        "record": "model_fold",
        "protocol": protocol,
        "model": model,
        "fold_index": float(fold_index),
        "include_l2p": float(include_l2p),
        "frame_count": float(frame_count),
        "memory_order": float(memory_order),
        "bath_order": float(bath_order),
        "potential_protocol": str(held["potential_protocol"]),
        "trace_probe_count": float(held["trace_probe_count"]),
        "var_spectral_radius": float(var["spectral_radius"]),
        "minimum_white_noise_covariance_eigenvalue": float(
            var["minimum_noise_covariance_eigenvalue"]
        ),
        "maximum_held_white_residual_correlation": float(
            np.max(held_shape["maximum_absolute_correlation"])
        ),
        "maximum_held_squared_white_residual_correlation": float(
            np.max(held_shape["maximum_absolute_squared_correlation"])
        ),
        "maximum_absolute_held_white_residual_excess_kurtosis": float(
            np.max(np.abs(held_shape["excess_kurtosis"]))
        ),
        "lp_held_white_residual_correlation": float(
            held_shape["maximum_absolute_correlation"][lp_coordinate]
        ),
        "lp_held_squared_white_residual_correlation": float(
            held_shape["maximum_absolute_squared_correlation"][lp_coordinate]
        ),
        "lp_held_white_residual_excess_kurtosis": float(
            held_shape["excess_kurtosis"][lp_coordinate]
        ),
        "stationary_covariance_maximum_error": stationary_covariance_maximum_error,
        "target_correlation_maximum_error": target_correlation_maximum_error,
        "maximum_simulated_absolute_state": float(np.max(np.abs(simulated))),
        "white_residual_gate_pass": float(white_gate),
        "gaussian_closure_gate_pass": float(
            white_gate
            and stationary_covariance_maximum_error <= 0.25
            and target_correlation_maximum_error <= 0.25
        ),
        "thermodynamic_claim_allowed": 0.0,
    }
    return summary, detail_rows, correlation_rows


def aggregate_summary(fold_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = (
        "var_spectral_radius",
        "minimum_white_noise_covariance_eigenvalue",
        "maximum_held_white_residual_correlation",
        "maximum_held_squared_white_residual_correlation",
        "maximum_absolute_held_white_residual_excess_kurtosis",
        "lp_held_white_residual_correlation",
        "lp_held_squared_white_residual_correlation",
        "lp_held_white_residual_excess_kurtosis",
        "stationary_covariance_maximum_error",
        "target_correlation_maximum_error",
        "maximum_simulated_absolute_state",
    )
    aggregates: dict[str, dict[str, object]] = {}
    for model in ("relative_phase_generator", "relative_phase_generator_l2p"):
        selected = [row for row in fold_rows if row["model"] == model]
        aggregate: dict[str, object] = {
            "record": "model_aggregate",
            "protocol": selected[0]["protocol"],
            "model": model,
            "held_clone_count": float(len(selected)),
            "memory_order": selected[0]["memory_order"],
            "bath_order": selected[0]["bath_order"],
            "potential_protocol": selected[0]["potential_protocol"],
            "trace_probe_count": selected[0]["trace_probe_count"],
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            aggregate[metric] = float(np.mean(values))
            aggregate[f"maximum_{metric}"] = float(np.max(values))
            aggregate[f"minimum_{metric}"] = float(np.min(values))
        aggregate[
            "maximum_absolute_lp_held_white_residual_excess_kurtosis"
        ] = float(
            np.max(
                np.abs(
                    [
                        float(row["lp_held_white_residual_excess_kurtosis"])
                        for row in selected
                    ]
                )
            )
        )
        aggregate["every_fold_white_residual_gate_pass"] = float(
            all(float(row["white_residual_gate_pass"]) == 1.0 for row in selected)
        )
        aggregate["every_fold_gaussian_closure_gate_pass"] = float(
            all(float(row["gaussian_closure_gate_pass"]) == 1.0 for row in selected)
        )
        aggregates[model] = aggregate
        output.append(aggregate)

    baseline = aggregates["relative_phase_generator"]
    extension = aggregates["relative_phase_generator_l2p"]
    improves_lp_shape = (
        float(extension["maximum_lp_held_squared_white_residual_correlation"])
        < float(baseline["maximum_lp_held_squared_white_residual_correlation"])
        and float(
            extension[
                "maximum_absolute_lp_held_white_residual_excess_kurtosis"
            ]
        )
        < float(
            baseline[
                "maximum_absolute_lp_held_white_residual_excess_kurtosis"
            ]
        )
    )
    output.append(
        {
            "record": "verdict",
            "protocol": extension["protocol"],
            "model": "relative_phase_generator_l2p",
            "held_clone_count": extension["held_clone_count"],
            "memory_order": extension["memory_order"],
            "bath_order": extension["bath_order"],
            "potential_protocol": extension["potential_protocol"],
            "trace_probe_count": extension["trace_probe_count"],
            "baseline_maximum_lp_squared_residual_correlation": baseline[
                "maximum_lp_held_squared_white_residual_correlation"
            ],
            "extension_maximum_lp_squared_residual_correlation": extension[
                "maximum_lp_held_squared_white_residual_correlation"
            ],
            "baseline_maximum_absolute_lp_residual_excess_kurtosis": baseline[
                "maximum_absolute_lp_held_white_residual_excess_kurtosis"
            ],
            "extension_maximum_absolute_lp_residual_excess_kurtosis": extension[
                "maximum_absolute_lp_held_white_residual_excess_kurtosis"
            ],
            "baseline_maximum_absolute_residual_excess_kurtosis": baseline[
                "maximum_maximum_absolute_held_white_residual_excess_kurtosis"
            ],
            "extension_maximum_absolute_residual_excess_kurtosis": extension[
                "maximum_maximum_absolute_held_white_residual_excess_kurtosis"
            ],
            "l2p_improves_lp_shape_on_aggregate": float(improves_lp_shape),
            "finite_discrete_gaussian_l2p_closure_supported": float(
                improves_lp_shape
                and float(extension["every_fold_gaussian_closure_gate_pass"]) == 1.0
            ),
            "continuous_gaussian_langevin_bath_allowed": 0.0,
            "autonomous_single_particle_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-drift-cache-directory", type=Path, required=True)
    parser.add_argument(
        "--training-second-generator-cache-directories",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--validation-drift-cache-directory", type=Path)
    parser.add_argument(
        "--validation-second-generator-cache-directories", type=Path, nargs="+"
    )
    parser.add_argument("--memory-order", type=int, default=40)
    parser.add_argument("--bath-order", type=int, default=16)
    parser.add_argument("--maximum-frame-count", type=int, default=0)
    parser.add_argument("--maximum-lag", type=int, default=40)
    parser.add_argument("--simulation-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--ridge-regularization", type=float, default=1e-8)
    parser.add_argument("--var-ridge-regularization", type=float, default=1e-6)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.memory_order < 1
        or args.bath_order < 1
        or args.maximum_frame_count < 0
        or args.maximum_lag < 1
        or args.simulation_count < 100
        or args.ridge_regularization < 0.0
        or args.var_ridge_regularization < 0.0
        or (args.validation_drift_cache_directory is None)
        != (args.validation_second_generator_cache_directories is None)
    ):
        raise ValueError("invalid relative-second-generator closure controls")
    training = load_matched_second_generator_clones(
        args.training_drift_cache_directory,
        args.training_second_generator_cache_directories,
        maximum_frame_count=args.maximum_frame_count,
    )
    validation_mode = args.validation_drift_cache_directory is not None
    if validation_mode:
        held_clones = load_matched_second_generator_clones(
            args.validation_drift_cache_directory,
            args.validation_second_generator_cache_directories,
            maximum_frame_count=args.maximum_frame_count,
        )
        if {
            str(clone["potential_protocol"]) for clone in training
        } != {str(clone["potential_protocol"]) for clone in held_clones} or {
            float(clone["trace_probe_count"]) for clone in training
        } != {float(clone["trace_probe_count"]) for clone in held_clones}:
            raise ValueError("validation must use the frozen discovery protocol")
        folds = [(training, held, index) for index, held in enumerate(held_clones, start=1)]
        protocol = "fixed_l2p_independent_clone_validation"
    else:
        if len(training) < 3:
            raise ValueError("discovery requires at least three matched clones")
        folds = [
            (
                [clone for index, clone in enumerate(training) if index != held_index],
                held,
                held_index + 1,
            )
            for held_index, held in enumerate(training)
        ]
        protocol = "l2p_discovery_leave_one_clone_out"

    fold_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for training_fold, held, fold_index in folds:
        for include_l2p in (False, True):
            summary, details, correlations = score_fold_model(
                training_fold,
                held,
                include_l2p=include_l2p,
                fold_index=fold_index,
                protocol=protocol,
                memory_order=args.memory_order,
                bath_order=args.bath_order,
                maximum_lag=args.maximum_lag,
                simulation_count=args.simulation_count,
                ridge_regularization=args.ridge_regularization,
                var_ridge_regularization=args.var_ridge_regularization,
                seed=args.seed,
            )
            fold_rows.append(summary)
            detail_rows.extend(details)
            correlation_rows.extend(correlations)
    summary_rows = [*fold_rows, *aggregate_summary(fold_rows)]
    write_rows(output_path(args.output_prefix, "_details.csv"), detail_rows)
    write_rows(output_path(args.output_prefix, "_summary.csv"), summary_rows)
    write_rows(output_path(args.output_prefix, "_correlation.csv"), correlation_rows)


if __name__ == "__main__":
    main()
