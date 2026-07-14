"""Continuous bilinear diagnostics for state-dependent microscopic memory."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


INVARIANT_NAMES = ("bath_energy", "velocity_bath_power")


def _validate_states(state_series: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int]:
    if not state_series:
        raise ValueError("at least one state trajectory is required")
    states = [np.asarray(state, dtype=float) for state in state_series]
    mode_count = states[0].shape[2] if states[0].ndim == 4 else -1
    for state in states:
        if (
            state.ndim != 4
            or state.shape[2] != mode_count
            or state.shape[3] != 3
            or len(state) < 2
            or not np.all(np.isfinite(state))
        ):
            raise ValueError("states must be finite aligned (frames, particles, modes, 3) arrays")
    if mode_count < 2:
        raise ValueError("state-dependent memory requires velocity and at least one bath mode")
    return states, mode_count


def _raw_state_invariants(state: np.ndarray, mode_rms: np.ndarray) -> np.ndarray:
    normalized = np.asarray(state, dtype=float) / mode_rms[None, None, :, None]
    bath_energy = np.mean(normalized[:, :, 1:] ** 2, axis=(2, 3))
    velocity_bath_power = np.mean(normalized[:, :, 0] * normalized[:, :, 1], axis=2)
    return np.stack([bath_energy, velocity_bath_power], axis=2)


def fit_state_invariant_scaling(state_series: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Fit rotation-invariant bath-energy and velocity-power scales."""

    states, mode_count = _validate_states(state_series)
    pooled = np.concatenate([state.reshape(-1, mode_count, 3) for state in states], axis=0)
    mode_rms = np.sqrt(np.mean(pooled**2, axis=(0, 2)))
    mode_rms = np.maximum(mode_rms, np.finfo(float).eps)
    raw = np.concatenate([_raw_state_invariants(state, mode_rms).reshape(-1, 2) for state in states], axis=0)
    invariant_mean = np.mean(raw, axis=0)
    invariant_scale = np.std(raw, axis=0)
    invariant_scale = np.maximum(invariant_scale, np.finfo(float).eps)
    return {
        "mode_rms": mode_rms,
        "invariant_mean": invariant_mean,
        "invariant_scale": invariant_scale,
    }


def state_invariants(state: np.ndarray, scaling: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate standardized continuous invariants on a microscopic bath state."""

    state = np.asarray(state, dtype=float)
    mode_rms = np.asarray(scaling["mode_rms"], dtype=float)
    invariant_mean = np.asarray(scaling["invariant_mean"], dtype=float)
    invariant_scale = np.asarray(scaling["invariant_scale"], dtype=float)
    if (
        state.ndim != 4
        or state.shape[2] != len(mode_rms)
        or state.shape[3] != 3
        or not np.all(np.isfinite(state))
    ):
        raise ValueError("state must be a finite aligned microscopic bath path")
    raw = _raw_state_invariants(state, mode_rms)
    return (raw - invariant_mean[None, None, :]) / invariant_scale[None, None, :]


def _bilinear_design(
    centered_state: np.ndarray,
    invariants: np.ndarray,
    invariant_indices: tuple[int, ...],
) -> np.ndarray:
    mode_count = centered_state.shape[2]
    blocks = centered_state.reshape(-1, mode_count, 3)
    invariant_blocks = invariants.reshape(-1, invariants.shape[2])
    base = np.transpose(blocks, (0, 2, 1))
    parts = [base]
    parts.extend(base * invariant_blocks[:, None, index, None] for index in invariant_indices)
    return np.concatenate(parts, axis=2).reshape(-1, mode_count * (1 + len(invariant_indices)))


def fit_bilinear_state_dependent_model(
    state_series: Sequence[np.ndarray],
    *,
    invariant_names: tuple[str, ...] = INVARIANT_NAMES,
    ridge_relative: float = 1.0e-6,
) -> dict[str, np.ndarray | float | tuple[str, ...] | dict[str, np.ndarray]]:
    """Fit a continuous bilinear transition without using macro observables."""

    states, mode_count = _validate_states(state_series)
    if not invariant_names or len(set(invariant_names)) != len(invariant_names):
        raise ValueError("invariant_names must be nonempty and unique")
    if any(name not in INVARIANT_NAMES for name in invariant_names):
        raise ValueError("unknown state invariant")
    if ridge_relative <= 0.0:
        raise ValueError("ridge_relative must be positive")
    invariant_indices = tuple(INVARIANT_NAMES.index(name) for name in invariant_names)
    scaling = fit_state_invariant_scaling(states)
    pooled = np.concatenate([state.reshape(-1, mode_count, 3) for state in states], axis=0)
    state_mean = np.mean(np.transpose(pooled, (0, 2, 1)).reshape(-1, mode_count), axis=0)
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for state in states:
        centered_current = state[:-1] - state_mean[None, None, :, None]
        centered_target = state[1:] - state_mean[None, None, :, None]
        invariants = state_invariants(state[:-1], scaling)
        designs.append(_bilinear_design(centered_current, invariants, invariant_indices))
        targets.append(np.transpose(centered_target, (0, 1, 3, 2)).reshape(-1, mode_count))
    design = np.concatenate(designs, axis=0)
    target = np.concatenate(targets, axis=0)
    gram = design.T @ design
    ridge = ridge_relative * float(np.trace(gram)) / len(gram)
    coefficient = np.linalg.solve(gram + ridge * np.eye(len(gram)), design.T @ target)
    return {
        "coefficient_matrix": coefficient,
        "state_mean": state_mean,
        "invariant_scaling": scaling,
        "invariant_names": invariant_names,
        "invariant_indices": np.asarray(invariant_indices, dtype=int),
        "ridge": float(ridge),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def predict_bilinear_state(
    model: dict[str, np.ndarray | float | tuple[str, ...] | dict[str, np.ndarray]],
    current_state: np.ndarray,
) -> np.ndarray:
    """Predict one step from the observed current microscopic bath state."""

    state = np.asarray(current_state, dtype=float)
    state_mean = np.asarray(model["state_mean"], dtype=float)
    scaling = model["invariant_scaling"]
    if not isinstance(scaling, dict):
        raise ValueError("model invariant scaling is invalid")
    invariant_indices = tuple(int(index) for index in np.asarray(model["invariant_indices"]))
    invariants = state_invariants(state, scaling)
    centered = state - state_mean[None, None, :, None]
    design = _bilinear_design(centered, invariants, invariant_indices)
    coefficient = np.asarray(model["coefficient_matrix"], dtype=float)
    mode_count = len(state_mean)
    prediction = design @ coefficient
    prediction = prediction.reshape(state.shape[0], state.shape[1], 3, mode_count)
    return np.transpose(prediction, (0, 1, 3, 2)) + state_mean[None, None, :, None]


def _maximum_cross_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left -= np.mean(left, axis=0)
    right -= np.mean(right, axis=0)
    numerator = left.T @ right
    denominator = np.sqrt(np.sum(left**2, axis=0))[:, None] * np.sqrt(
        np.sum(right**2, axis=0)
    )[None, :]
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).tiny,
    )
    return float(np.max(np.abs(correlations)))


def bilinear_heldout_diagnostics(
    model: dict[str, np.ndarray | float | tuple[str, ...] | dict[str, np.ndarray]],
    held_state: np.ndarray,
) -> dict[str, float]:
    """Evaluate held one-step prediction and residual memory."""

    state = np.asarray(held_state, dtype=float)
    if state.ndim != 4 or len(state) < 3 or not np.all(np.isfinite(state)):
        raise ValueError("held_state must be a finite state path")
    predicted = predict_bilinear_state(model, state[:-1])
    target = state[1:]
    residual = target - predicted
    state_mean = np.asarray(model["state_mean"], dtype=float)
    centered_target = target - state_mean[None, None, :, None]
    total = float(np.sum(centered_target**2))
    velocity_total = float(np.sum(centered_target[:, :, 0] ** 2))
    mode_count = state.shape[2]
    residual_flat = np.transpose(residual, (0, 1, 3, 2)).reshape(-1, mode_count)
    current_flat = np.transpose(
        state[:-1] - state_mean[None, None, :, None],
        (0, 1, 3, 2),
    ).reshape(-1, mode_count)
    maximum_state_correlation = _maximum_cross_correlation(residual_flat, current_flat)
    maximum_lag_correlation = 0.0
    for lag in range(1, min(16, len(residual) - 1) + 1):
        left = np.transpose(residual[lag:], (0, 1, 3, 2)).reshape(-1, mode_count)
        right = np.transpose(residual[:-lag], (0, 1, 3, 2)).reshape(-1, mode_count)
        maximum_lag_correlation = max(
            maximum_lag_correlation,
            _maximum_cross_correlation(left, right),
        )
    return {
        "heldout_state_r_squared": 1.0 - float(np.sum(residual**2)) / max(total, np.finfo(float).tiny),
        "heldout_velocity_r_squared": 1.0
        - float(np.sum(residual[:, :, 0] ** 2)) / max(velocity_total, np.finfo(float).tiny),
        "heldout_velocity_residual_mean_squared": float(np.mean(residual[:, :, 0] ** 2)),
        "maximum_held_residual_state_correlation": maximum_state_correlation,
        "maximum_held_residual_lag_correlation": maximum_lag_correlation,
    }
