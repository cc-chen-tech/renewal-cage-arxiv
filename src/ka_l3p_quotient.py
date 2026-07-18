"""Held quotient tests for the microscopic ``L^3p`` generator coordinate."""

from __future__ import annotations

import math

import numpy as np

from ka_collective_memory import discrete_mori_zwanzig_operators
from ka_l2p_conditional_diffusion import replicate_first_t_interval
from ka_markov_bath import (
    fit_stationary_vector_autoregression,
    vector_autoregressive_residual,
)
from ka_relative_mori import finite_memory_innovation_series


L3P_QUOTIENT_MODELS = (
    "l2p_exact_q_baseline",
    "l3p_generator",
    "l3p_time_permuted",
    "l2p_backward_difference",
)

_CLOSED_CLAIMS = {
    "finite_l3p_gaussian_closure_supported": 0.0,
    "microscopic_environment_coordinate_z_allowed": 0.0,
    "continuous_gaussian_langevin_bath_allowed": 0.0,
    "autonomous_single_particle_gle_allowed": 0.0,
    "complete_event_clock_closure_allowed": 0.0,
    "kramers_escape_claim_allowed": 0.0,
    "spatial_facilitation_claim_allowed": 0.0,
    "thermodynamic_claim_allowed": 0.0,
}


def quotient_fifth_coordinate(
    l2p: np.ndarray,
    l3p: np.ndarray,
    *,
    model: str,
    frame_time: float,
    seed: int,
) -> np.ndarray | None:
    """Construct a frozen fifth coordinate after dropping the first frame."""

    second = np.asarray(l2p, dtype=float)
    third = np.asarray(l3p, dtype=float)
    if (
        second.ndim != 3
        or second.shape[-1] != 3
        or third.shape != second.shape
        or len(second) < 2
        or np.any(~np.isfinite(second))
        or np.any(~np.isfinite(third))
    ):
        raise ValueError("L2p and L3p paths must be aligned finite target vectors")
    if model not in L3P_QUOTIENT_MODELS:
        raise ValueError("unknown L3p quotient model")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if model == "l2p_exact_q_baseline":
        return None
    if model == "l3p_generator":
        return third[1:].copy()
    if model == "l2p_backward_difference":
        return np.diff(second, axis=0) / float(frame_time)

    values = third[1:]
    permuted = np.empty_like(values)
    rng = np.random.default_rng(seed)
    for target in range(values.shape[1]):
        permutation = rng.permutation(len(values))
        permuted[:, target] = values[permutation, target]
    return permuted


def augment_l3p_quotient_clone(
    clone: dict[str, object],
    *,
    model: str,
    frame_time: float,
    permutation_seed: int,
) -> dict[str, object]:
    """Attach one frozen fifth-coordinate path without changing base fields."""

    required = (
        "relative_position",
        "relative_velocity",
        "relative_drift",
        "second_relative_generator",
        "l3p_generator",
    )
    if any(key not in clone for key in required):
        raise ValueError("L3p quotient clone is incomplete")
    position = np.asarray(clone["relative_position"], dtype=float)
    arrays = [np.asarray(clone[key], dtype=float) for key in required[1:]]
    if (
        position.ndim != 3
        or position.shape[-1] != 3
        or any(value.shape != position.shape for value in arrays)
        or np.any(~np.isfinite(position))
        or any(np.any(~np.isfinite(value)) for value in arrays)
    ):
        raise ValueError("L3p quotient clone fields must be finite and aligned")
    augmented = dict(clone)
    augmented["quotient_model"] = model
    augmented["quotient_fifth_coordinate"] = quotient_fifth_coordinate(
        np.asarray(clone["second_relative_generator"]),
        np.asarray(clone["l3p_generator"]),
        model=model,
        frame_time=frame_time,
        seed=permutation_seed,
    )
    augmented["quotient_first_source_frame"] = 1.0
    augmented["thermodynamic_claim_allowed"] = 0.0
    return augmented


def _base_generator_state(
    clone: dict[str, object],
    *,
    frame_count: int,
    bias: np.ndarray,
) -> np.ndarray:
    fields = []
    for key in (
        "relative_position",
        "relative_velocity",
        "relative_drift",
        "second_relative_generator",
    ):
        value = np.asarray(clone[key], dtype=float)[:frame_count]
        if key == "relative_position":
            value = value - bias[None]
        fields.append(value[1:].reshape(frame_count - 1, -1))
    return np.stack(fields, axis=2)


def extract_l3p_quotient_fold(
    training: list[dict[str, object]],
    held: dict[str, object],
    *,
    memory_order: int,
    bath_order: int,
    ridge_regularization: float,
    var_ridge_regularization: float,
) -> dict[str, object]:
    """Fit one paired Mori/VAR fold and retain exact original-frame indices."""

    if len(training) < 2 or memory_order < 1 or bath_order < 1:
        raise ValueError("L3p quotient extraction needs training clones and orders")
    clones = [*training, held]
    models = {str(clone.get("quotient_model")) for clone in clones}
    if len(models) != 1 or models == {"None"}:
        raise ValueError("all quotient clones in a fold must use one model")
    frame_count = min(
        len(np.asarray(clone["relative_position"])) for clone in clones
    )
    if frame_count - 1 <= memory_order + bath_order + 2:
        raise ValueError("matched quotient paths are too short")
    target_shape = np.asarray(held["relative_position"]).shape[1:]
    if len(target_shape) != 2 or target_shape[1] != 3:
        raise ValueError("held quotient targets must be three-vectors")
    bias = np.mean(
        np.concatenate(
            [
                np.asarray(clone["relative_position"], dtype=float)[1:frame_count]
                for clone in training
            ],
            axis=0,
        ),
        axis=0,
    )
    base_training_paths = [
        _base_generator_state(clone, frame_count=frame_count, bias=bias)
        for clone in training
    ]
    held_base = _base_generator_state(held, frame_count=frame_count, bias=bias)
    base_training = np.concatenate(base_training_paths, axis=1)
    base_mean = np.mean(base_training, axis=(0, 1), keepdims=True)
    base_scale = np.std(base_training, axis=(0, 1), keepdims=True)
    if np.any(base_scale <= 1e-12):
        raise ValueError("base generator coordinates need nonzero variance")
    training_paths = [(path - base_mean) / base_scale for path in base_training_paths]
    held_state = (held_base - base_mean) / base_scale

    fifth_training_raw = [
        clone.get("quotient_fifth_coordinate") for clone in training
    ]
    held_fifth_raw = held.get("quotient_fifth_coordinate")
    if all(value is None for value in fifth_training_raw) and held_fifth_raw is None:
        mean = base_mean
        scale = base_scale
    else:
        if any(value is None for value in fifth_training_raw) or held_fifth_raw is None:
            raise ValueError("fifth quotient coordinate is missing within a fold")
        fifth_training_paths = [
            np.asarray(value, dtype=float)[: frame_count - 1].reshape(
                frame_count - 1,
                -1,
                1,
            )
            for value in fifth_training_raw
        ]
        held_fifth = np.asarray(held_fifth_raw, dtype=float)[: frame_count - 1].reshape(
            frame_count - 1,
            -1,
            1,
        )
        expected_fifth_shape = training_paths[0].shape[:2] + (1,)
        if (
            any(path.shape != expected_fifth_shape for path in fifth_training_paths)
            or held_fifth.shape != held_state.shape[:2] + (1,)
            or any(np.any(~np.isfinite(path)) for path in fifth_training_paths)
            or np.any(~np.isfinite(held_fifth))
        ):
            raise ValueError("fifth quotient coordinate is not frame-target aligned")
        fifth_training = np.concatenate(fifth_training_paths, axis=1)
        fifth_mean = np.mean(fifth_training, axis=(0, 1), keepdims=True)
        fifth_scale = np.std(fifth_training, axis=(0, 1), keepdims=True)
        if np.any(fifth_scale <= 1e-12):
            raise ValueError("fifth quotient coordinate needs nonzero variance")
        training_paths = [
            np.concatenate(
                (base, (fifth - fifth_mean) / fifth_scale),
                axis=2,
            )
            for base, fifth in zip(
                training_paths,
                fifth_training_paths,
                strict=True,
            )
        ]
        held_state = np.concatenate(
            (held_state, (held_fifth - fifth_mean) / fifth_scale),
            axis=2,
        )
        mean = np.concatenate((base_mean, fifth_mean), axis=2)
        scale = np.concatenate((base_scale, fifth_scale), axis=2)

    training_state = np.concatenate(training_paths, axis=1)
    mori = discrete_mori_zwanzig_operators(
        training_state,
        memory_order=memory_order,
        ridge_regularization=ridge_regularization,
    )
    operators = np.asarray(mori["operators"])
    training_innovation = finite_memory_innovation_series(
        training_state,
        operators,
    )
    held_innovation = finite_memory_innovation_series(held_state, operators)
    var = fit_stationary_vector_autoregression(
        training_innovation,
        order=bath_order,
        ridge_regularization=var_ridge_regularization,
    )
    coefficients = np.asarray(var["coefficients"])
    bath_mean = np.asarray(var["mean"])
    training_white = vector_autoregressive_residual(
        training_innovation,
        coefficients,
        mean=bath_mean,
    )
    held_white = vector_autoregressive_residual(
        held_innovation,
        coefficients,
        mean=bath_mean,
    )
    first_internal_frame = (len(operators) - 1) + 1 + len(coefficients)
    original_frames = np.arange(
        first_internal_frame + 1,
        frame_count,
        dtype=int,
    )
    if len(original_frames) != len(held_white):
        raise ValueError("L3p white innovation frame alignment is inconsistent")
    target_count = target_shape[0]
    if held_white.shape[1] != 3 * target_count:
        raise ValueError("L3p innovations do not preserve target components")
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
        "held_l2p_vector_innovation": held_white[:, :, 3].reshape(
            len(held_white),
            target_count,
            3,
        ),
        "held_source_frame_indices": original_frames,
        "thermodynamic_claim_allowed": 0.0,
    }


def _absolute_gate(row: dict[str, object]) -> bool:
    return (
        float(row["maximum_absolute_whitened_correlation"]) <= 0.05
        and float(row["maximum_absolute_squared_whitened_correlation"]) <= 0.05
        and float(row["maximum_absolute_component_excess_kurtosis"]) <= 0.35
        and float(row["maximum_absolute_whitened_covariance_error"]) <= 0.10
        and 0.8
        <= float(row["mean_squared_mahalanobis_per_dimension"])
        <= 1.2
    )


def classify_l3p_quotient(
    detail_rows: list[dict[str, object]],
    numerical_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the frozen paired held-clone and claim-boundary gates."""

    expected = {
        (fold, model)
        for fold in range(1, 5)
        for model in L3P_QUOTIENT_MODELS
    }
    actual = {
        (int(float(row["fold_index"])), str(row["model"]))
        for row in detail_rows
    }
    numerical_by_fold = {
        int(float(row["fold_index"])): row for row in numerical_rows
    }
    if actual != expected or set(numerical_by_fold) != {1, 2, 3, 4}:
        raise ValueError("four-fold four-model L3p quotient grid is not complete")
    by_cell = {
        (int(float(row["fold_index"])), str(row["model"])): row
        for row in detail_rows
    }
    baseline = [by_cell[(fold, "l2p_exact_q_baseline")] for fold in range(1, 5)]
    real = [by_cell[(fold, "l3p_generator")] for fold in range(1, 5)]
    permuted = [by_cell[(fold, "l3p_time_permuted")] for fold in range(1, 5)]
    backward = [
        by_cell[(fold, "l2p_backward_difference")] for fold in range(1, 5)
    ]
    nll_improvement = np.asarray(
        [
            float(base["negative_log_likelihood"])
            - float(candidate["negative_log_likelihood"])
            for base, candidate in zip(baseline, real, strict=True)
        ]
    )
    interval = replicate_first_t_interval(nll_improvement)
    every_fold_nll = bool(np.all(nll_improvement > 0.0))
    every_fold_squared_memory = all(
        float(candidate["maximum_absolute_squared_whitened_correlation"])
        <= 0.75 * float(base["maximum_absolute_squared_whitened_correlation"])
        for base, candidate in zip(baseline, real, strict=True)
    )
    every_fold_kurtosis = all(
        abs(float(candidate["maximum_absolute_component_excess_kurtosis"]))
        <= 0.75 * abs(float(base["maximum_absolute_component_excess_kurtosis"]))
        for base, candidate in zip(baseline, real, strict=True)
    )
    beats_permuted = all(
        float(candidate["negative_log_likelihood"])
        < float(null["negative_log_likelihood"])
        for candidate, null in zip(real, permuted, strict=True)
    )
    beats_backward = all(
        float(candidate["negative_log_likelihood"])
        < float(null["negative_log_likelihood"])
        for candidate, null in zip(real, backward, strict=True)
    )
    numerical_pass = all(
        float(numerical_by_fold[fold]["l3p_numerical_gate_pass"]) == 1.0
        for fold in range(1, 5)
    )
    informative = (
        numerical_pass
        and every_fold_nll
        and float(interval["ci95_low"]) > 0.0
        and every_fold_squared_memory
        and every_fold_kurtosis
        and beats_permuted
        and beats_backward
    )
    real_absolute = all(_absolute_gate(row) for row in real)
    floor_pass = all(
        float(row["isotropic_floor_variance_fraction"]) <= 0.25 for row in real
    )
    closed = informative and real_absolute and floor_pass
    return {
        "record": "verdict",
        "model": "microscopic_l3p_generator_quotient",
        "held_clone_count": 4.0,
        "mean_baseline_to_l3p_nll_improvement": interval["mean"],
        "baseline_to_l3p_nll_improvement_standard_error": interval[
            "standard_error"
        ],
        "baseline_to_l3p_nll_improvement_ci95_low": interval["ci95_low"],
        "baseline_to_l3p_nll_improvement_ci95_high": interval["ci95_high"],
        "every_fold_nll_improves": float(every_fold_nll),
        "every_fold_squared_memory_reduction_25pct": float(
            every_fold_squared_memory
        ),
        "every_fold_absolute_kurtosis_reduction_25pct": float(
            every_fold_kurtosis
        ),
        "every_fold_beats_time_permuted_null": float(beats_permuted),
        "every_fold_beats_backward_difference_null": float(beats_backward),
        "every_fold_l3p_numerical_gate_pass": float(numerical_pass),
        "every_fold_l3p_absolute_gate_pass": float(real_absolute),
        "every_fold_l3p_floor_fraction_pass": float(floor_pass),
        "l3p_generator_coordinate_informative": float(informative),
        "l2p_residual_closed_by_l3p": float(closed),
        "l3p_diffusion_derivation_authorized": float(informative),
        **_CLOSED_CLAIMS,
    }
