"""State assembly and kinematic diagnostics for a smooth-cage force bath."""

from __future__ import annotations

import math

import numpy as np


def assemble_augmented_hankel_state(
    velocities: np.ndarray,
    force_modes: np.ndarray,
    *,
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    include_position: bool,
    include_relative_velocity: bool,
) -> np.ndarray:
    """Align exact-force history modes with microscopic cage vector modes."""

    velocities = np.asarray(velocities, dtype=float)
    force_modes = np.asarray(force_modes, dtype=float)
    relative_position = np.asarray(relative_position, dtype=float)
    relative_velocity = np.asarray(relative_velocity, dtype=float)
    if (
        velocities.ndim != 3
        or velocities.shape[2] != 3
        or relative_position.shape != velocities.shape
        or relative_velocity.shape != velocities.shape
        or np.any(~np.isfinite(velocities))
        or np.any(~np.isfinite(relative_position))
        or np.any(~np.isfinite(relative_velocity))
    ):
        raise ValueError("velocity and cage paths must be aligned finite vector arrays")
    if (
        force_modes.ndim != 4
        or force_modes.shape[1] != velocities.shape[1]
        or force_modes.shape[2] < 1
        or force_modes.shape[3] != 3
        or len(force_modes) > len(velocities)
        or np.any(~np.isfinite(force_modes))
    ):
        raise ValueError("force_modes must be a finite aligned history projection")
    if not isinstance(include_position, (bool, np.bool_)) or not isinstance(
        include_relative_velocity, (bool, np.bool_)
    ):
        raise ValueError("cage-mode selectors must be Boolean")
    offset = len(velocities) - len(force_modes)
    parts = [velocities[offset:, :, None, :], force_modes]
    if include_position:
        parts.append(relative_position[offset:, :, None, :])
    if include_relative_velocity:
        parts.append(relative_velocity[offset:, :, None, :])
    return np.concatenate(parts, axis=2)


def cage_kinematic_diagnostic(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    frame_time: float,
) -> dict[str, float]:
    """Measure the saved-grid trapezoidal defect in the exact identity du=p dt."""

    position = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    if (
        position.ndim != 3
        or position.shape[2] != 3
        or velocity.shape != position.shape
        or len(position) < 2
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
    ):
        raise ValueError("cage position and velocity must be aligned finite paths")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be positive and finite")
    increment = position[1:] - position[:-1]
    trapezoid = 0.5 * frame_time * (velocity[:-1] + velocity[1:])
    residual = increment - trapezoid
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    increment_rms = float(np.sqrt(np.mean(increment**2)))
    normalized = residual_rms / max(increment_rms, np.finfo(float).tiny)
    return {
        "kinematic_residual_rms": residual_rms,
        "kinematic_increment_rms": increment_rms,
        "normalized_trapezoidal_kinematic_error": float(normalized),
        "thermodynamic_claim_allowed": 0.0,
    }


def heldout_residual_lag_profile(
    model: dict[str, np.ndarray | float],
    held_state: np.ndarray,
    *,
    maximum_lag: int = 16,
) -> dict[str, np.ndarray | float]:
    """Resolve held residual temporal correlation by predicted state mode."""

    state = np.asarray(held_state, dtype=float)
    transition = np.asarray(model["transition_matrix"], dtype=float)
    state_mean = np.asarray(model["state_mean"], dtype=float)
    mode_count = len(state_mean)
    if (
        state.ndim != 4
        or state.shape[2:] != (mode_count, 3)
        or len(state) < 3
        or np.any(~np.isfinite(state))
        or transition.shape != (mode_count, mode_count)
    ):
        raise ValueError("held_state and model must define aligned finite vector modes")
    if (
        isinstance(maximum_lag, bool)
        or not isinstance(maximum_lag, (int, np.integer))
        or maximum_lag < 1
    ):
        raise ValueError("maximum_lag must be a positive integer")
    current = state[:-1] - state_mean[None, None, :, None]
    target = state[1:] - state_mean[None, None, :, None]
    residual = target - np.einsum("ab,tpbc->tpac", transition, current)
    maximum_by_mode = np.zeros(mode_count, dtype=float)
    lag_at_maximum = np.zeros(mode_count, dtype=int)
    for lag in range(1, min(int(maximum_lag), len(residual) - 1) + 1):
        left = np.transpose(residual[lag:], (0, 1, 3, 2)).reshape(-1, mode_count)
        right = np.transpose(residual[:-lag], (0, 1, 3, 2)).reshape(-1, mode_count)
        left -= np.mean(left, axis=0)
        right -= np.mean(right, axis=0)
        denominator = np.sqrt(np.sum(left**2, axis=0))[:, None] * np.sqrt(
            np.sum(right**2, axis=0)
        )[None, :]
        correlation = np.divide(
            left.T @ right,
            denominator,
            out=np.zeros((mode_count, mode_count), dtype=float),
            where=denominator > np.finfo(float).tiny,
        )
        by_mode = np.max(np.abs(correlation), axis=1)
        improved = by_mode > maximum_by_mode
        maximum_by_mode[improved] = by_mode[improved]
        lag_at_maximum[improved] = lag
    return {
        "maximum_by_residual_mode": maximum_by_mode,
        "lag_at_maximum_by_residual_mode": lag_at_maximum,
        "maximum_overall_residual_lag_correlation": float(np.max(maximum_by_mode)),
        "thermodynamic_claim_allowed": 0.0,
    }


def heldout_linear_velocity_diagnostic(
    model: dict[str, np.ndarray | float],
    held_state: np.ndarray,
    *,
    velocity_weights: np.ndarray,
) -> dict[str, float]:
    """Evaluate a tagged velocity reconstructed as a fixed state-mode sum."""

    state = np.asarray(held_state, dtype=float)
    transition = np.asarray(model["transition_matrix"], dtype=float)
    state_mean = np.asarray(model["state_mean"], dtype=float)
    weights = np.asarray(velocity_weights, dtype=float)
    mode_count = len(state_mean)
    if (
        state.ndim != 4
        or state.shape[2:] != (mode_count, 3)
        or len(state) < 2
        or np.any(~np.isfinite(state))
        or transition.shape != (mode_count, mode_count)
        or weights.shape != (mode_count,)
        or np.any(~np.isfinite(weights))
    ):
        raise ValueError("model, state, and velocity weights must be finite and aligned")
    current = state[:-1] - state_mean[None, None, :, None]
    target = state[1:] - state_mean[None, None, :, None]
    predicted = np.einsum("ab,tpbc->tpac", transition, current)
    residual = target - predicted
    tagged_target = np.einsum("m,tpmc->tpc", weights, target)
    tagged_residual = np.einsum("m,tpmc->tpc", weights, residual)
    total = float(np.sum(tagged_target**2))
    return {
        "heldout_linear_velocity_r_squared": 1.0
        - float(np.sum(tagged_residual**2)) / max(total, np.finfo(float).tiny),
        "heldout_linear_velocity_residual_mean_squared": float(
            np.mean(tagged_residual**2)
        ),
        "thermodynamic_claim_allowed": 0.0,
    }
