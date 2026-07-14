"""Stable auxiliary baths derived from exact tagged-particle force histories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _symmetric_sqrt(matrix: np.ndarray, *, floor: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(values)), 1.0)
    clipped = np.maximum(values, floor * scale)
    square_root = (vectors * np.sqrt(clipped)) @ vectors.T
    inverse_square_root = (vectors * (1.0 / np.sqrt(clipped))) @ vectors.T
    return square_root, inverse_square_root


def assemble_force_hankel(force: np.ndarray, *, history_length: int) -> np.ndarray:
    """Return newest-to-oldest force histories on a uniform saved-frame grid."""

    force = np.asarray(force, dtype=float)
    if force.ndim != 3 or force.shape[2] != 3:
        raise ValueError("force must have shape (frames, particles, 3)")
    if not np.all(np.isfinite(force)):
        raise ValueError("force must be finite")
    if history_length < 1 or len(force) < history_length:
        raise ValueError("history_length must fit inside the force trajectory")
    windows = np.lib.stride_tricks.sliding_window_view(
        force,
        window_shape=history_length,
        axis=0,
    )
    return np.asarray(windows[..., ::-1])


def fit_force_hankel_basis(
    force_series: Sequence[np.ndarray],
    *,
    history_length: int,
    mode_count: int,
) -> dict[str, np.ndarray | float]:
    """Fit temporal PCA modes from only the supplied microscopic force paths."""

    if not force_series:
        raise ValueError("at least one force trajectory is required")
    if mode_count < 1 or mode_count > history_length:
        raise ValueError("mode_count must lie between one and history_length")
    histories = [assemble_force_hankel(force, history_length=history_length) for force in force_series]
    samples = np.concatenate([history.reshape(-1, history_length) for history in histories], axis=0)
    history_mean = np.mean(samples, axis=0)
    centered = samples - history_mean
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order[:mode_count]].copy()
    for column in range(mode_count):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    total = float(np.sum(eigenvalues))
    captured = float(np.sum(eigenvalues[:mode_count]) / total) if total > 0.0 else 0.0
    return {
        "history_length": float(history_length),
        "mode_count": float(mode_count),
        "history_mean": history_mean,
        "basis": basis,
        "eigenvalues": eigenvalues,
        "captured_variance_fraction": captured,
    }


def project_force_hankel(
    force: np.ndarray,
    basis_fit: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Project one exact force trajectory onto a training-only temporal basis."""

    history_length = int(basis_fit["history_length"])
    basis = np.asarray(basis_fit["basis"], dtype=float)
    history_mean = np.asarray(basis_fit["history_mean"], dtype=float)
    if basis.ndim != 2 or basis.shape[0] != history_length:
        raise ValueError("basis must align with history_length")
    if history_mean.shape != (history_length,):
        raise ValueError("history_mean must align with history_length")
    hankel = assemble_force_hankel(force, history_length=history_length)
    projected = np.einsum("tfcp,pr->tfcr", hankel - history_mean, basis)
    return np.transpose(projected, (0, 1, 3, 2))


def fit_covariance_contracted_model(
    state_series: Sequence[np.ndarray],
    *,
    maximum_singular_value: float = 0.999,
    covariance_floor: float = 1.0e-10,
) -> dict[str, np.ndarray | float]:
    """Fit a stationary linear bath with covariance-consistent innovations.

    States use shape ``(frames, particles, modes, 3)``.  One isotropic mode
    transition is fitted while full three-component innovation blocks remain
    available for non-Gaussian autonomous sampling.
    """

    if not state_series:
        raise ValueError("at least one state trajectory is required")
    if not 0.0 < maximum_singular_value <= 1.0:
        raise ValueError("maximum_singular_value must lie in (0, 1]")
    if covariance_floor <= 0.0:
        raise ValueError("covariance_floor must be positive")
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

    all_blocks = np.concatenate([state.reshape(-1, mode_count, 3) for state in states], axis=0)
    state_mean = np.mean(np.transpose(all_blocks, (0, 2, 1)).reshape(-1, mode_count), axis=0)
    current_blocks = np.concatenate([state[:-1].reshape(-1, mode_count, 3) for state in states], axis=0)
    target_blocks = np.concatenate([state[1:].reshape(-1, mode_count, 3) for state in states], axis=0)
    current_blocks = current_blocks - state_mean[None, :, None]
    target_blocks = target_blocks - state_mean[None, :, None]
    current = np.transpose(current_blocks, (0, 2, 1)).reshape(-1, mode_count)
    target = np.transpose(target_blocks, (0, 2, 1)).reshape(-1, mode_count)

    stationary_covariance = current.T @ current / len(current)
    covariance_sqrt, covariance_inverse_sqrt = _symmetric_sqrt(
        stationary_covariance,
        floor=covariance_floor,
    )
    lagged_covariance = target.T @ current / len(current)
    whitened_transition = covariance_inverse_sqrt @ lagged_covariance @ covariance_inverse_sqrt
    left, singular_values, right = np.linalg.svd(whitened_transition, full_matrices=False)
    clipped_singular_values = np.minimum(singular_values, maximum_singular_value)
    whitened_transition = (left * clipped_singular_values) @ right
    transition = covariance_sqrt @ whitened_transition @ covariance_inverse_sqrt

    innovation_covariance = covariance_sqrt @ (
        np.eye(mode_count) - whitened_transition @ whitened_transition.T
    ) @ covariance_sqrt
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
    innovation_values, innovation_vectors = np.linalg.eigh(innovation_covariance)
    innovation_values = np.maximum(innovation_values, 0.0)
    innovation_covariance = (innovation_vectors * innovation_values) @ innovation_vectors.T
    innovation_sqrt = (innovation_vectors * np.sqrt(innovation_values)) @ innovation_vectors.T

    residual_blocks = target_blocks - np.einsum("ab,sbc->sac", transition, current_blocks)
    residual_blocks -= np.mean(residual_blocks, axis=(0, 2))[None, :, None]
    residual = np.transpose(residual_blocks, (0, 2, 1)).reshape(-1, mode_count)
    residual_covariance = residual.T @ residual / len(residual)
    _, residual_inverse_sqrt = _symmetric_sqrt(residual_covariance, floor=covariance_floor)
    standardized_residual_blocks = np.einsum(
        "ab,sbc->sac",
        residual_inverse_sqrt,
        residual_blocks,
    )

    reconstructed = transition @ stationary_covariance @ transition.T + innovation_covariance
    covariance_error = float(
        np.linalg.norm(reconstructed - stationary_covariance)
        / max(np.linalg.norm(stationary_covariance), np.finfo(float).tiny)
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    return {
        "transition_matrix": transition,
        "state_mean": state_mean,
        "stationary_covariance": stationary_covariance,
        "innovation_covariance": innovation_covariance,
        "innovation_square_root": innovation_sqrt,
        "standardized_residual_blocks": standardized_residual_blocks,
        "training_state_pool": all_blocks,
        "maximum_unclipped_singular_value": float(np.max(singular_values)),
        "maximum_clipped_singular_value": float(np.max(clipped_singular_values)),
        "spectral_radius": spectral_radius,
        "stationary_covariance_relative_error": covariance_error,
    }


def simulate_slow_bath(
    model: dict[str, np.ndarray | float],
    *,
    step_count: int,
    simulation_count: int,
    seed: int,
) -> np.ndarray:
    """Generate autonomous state paths from a covariance-contracted bath."""

    if step_count < 1 or simulation_count < 1:
        raise ValueError("step_count and simulation_count must be positive")
    transition = np.asarray(model["transition_matrix"], dtype=float)
    state_mean = np.asarray(model["state_mean"], dtype=float)
    innovation_sqrt = np.asarray(model["innovation_square_root"], dtype=float)
    residual_pool = np.asarray(model["standardized_residual_blocks"], dtype=float)
    state_pool = np.asarray(model["training_state_pool"], dtype=float)
    mode_count = len(state_mean)
    if transition.shape != (mode_count, mode_count) or innovation_sqrt.shape != transition.shape:
        raise ValueError("model matrices must align with state_mean")
    rng = np.random.default_rng(seed)
    current = state_pool[rng.integers(len(state_pool), size=simulation_count)].copy()
    paths = np.empty((step_count + 1, simulation_count, mode_count, 3), dtype=float)
    paths[0] = current
    for step in range(1, step_count + 1):
        centered = current - state_mean[None, :, None]
        current = state_mean[None, :, None] + np.einsum("ab,sbc->sac", transition, centered)
        standardized = residual_pool[rng.integers(len(residual_pool), size=simulation_count)]
        current += np.einsum("ab,sbc->sac", innovation_sqrt, standardized)
        paths[step] = current
    return paths


def heldout_state_diagnostics(
    model: dict[str, np.ndarray | float],
    held_state: np.ndarray,
) -> dict[str, float]:
    """Measure one-step prediction and orthogonality on an unseen state path."""

    state = np.asarray(held_state, dtype=float)
    transition = np.asarray(model["transition_matrix"], dtype=float)
    state_mean = np.asarray(model["state_mean"], dtype=float)
    mode_count = len(state_mean)
    if (
        state.ndim != 4
        or state.shape[2:] != (mode_count, 3)
        or len(state) < 2
        or not np.all(np.isfinite(state))
    ):
        raise ValueError("held_state must be a finite aligned state trajectory")
    current = state[:-1] - state_mean[None, None, :, None]
    target = state[1:] - state_mean[None, None, :, None]
    predicted = np.einsum("ab,tpbc->tpac", transition, current)
    residual = target - predicted
    total = float(np.sum(target**2))
    velocity_total = float(np.sum(target[:, :, 0] ** 2))
    state_r_squared = 1.0 - float(np.sum(residual**2)) / max(total, np.finfo(float).tiny)
    velocity_r_squared = 1.0 - float(np.sum(residual[:, :, 0] ** 2)) / max(
        velocity_total,
        np.finfo(float).tiny,
    )
    current_flat = np.transpose(current, (0, 1, 3, 2)).reshape(-1, mode_count)
    residual_flat = np.transpose(residual, (0, 1, 3, 2)).reshape(-1, mode_count)
    current_flat -= np.mean(current_flat, axis=0)
    residual_flat -= np.mean(residual_flat, axis=0)
    numerator = residual_flat.T @ current_flat
    residual_norm = np.sqrt(np.sum(residual_flat**2, axis=0))
    current_norm = np.sqrt(np.sum(current_flat**2, axis=0))
    denominator = residual_norm[:, None] * current_norm[None, :]
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(float).tiny,
    )
    maximum_lag_correlation = 0.0
    for lag in range(1, min(16, len(residual) - 1) + 1):
        left = np.transpose(residual[lag:], (0, 1, 3, 2)).reshape(-1, mode_count)
        right = np.transpose(residual[:-lag], (0, 1, 3, 2)).reshape(-1, mode_count)
        left -= np.mean(left, axis=0)
        right -= np.mean(right, axis=0)
        numerator = left.T @ right
        left_norm = np.sqrt(np.sum(left**2, axis=0))
        right_norm = np.sqrt(np.sum(right**2, axis=0))
        denominator = left_norm[:, None] * right_norm[None, :]
        lag_correlations = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > np.finfo(float).tiny,
        )
        maximum_lag_correlation = max(
            maximum_lag_correlation,
            float(np.max(np.abs(lag_correlations))),
        )
    return {
        "heldout_state_r_squared": float(state_r_squared),
        "heldout_velocity_r_squared": float(velocity_r_squared),
        "maximum_held_residual_state_correlation": float(np.max(np.abs(correlations))),
        "maximum_held_residual_lag_correlation": maximum_lag_correlation,
    }


def state_paths_to_displacements(state_paths: np.ndarray, *, frame_time: float) -> np.ndarray:
    """Integrate the first state mode as velocity by the trapezoid rule."""

    paths = np.asarray(state_paths, dtype=float)
    if (
        paths.ndim != 4
        or paths.shape[2] < 1
        or paths.shape[3] != 3
        or len(paths) < 2
        or not np.all(np.isfinite(paths))
    ):
        raise ValueError("state_paths must be finite (frames, samples, modes, 3)")
    if not np.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be positive and finite")
    velocity = paths[:, :, 0]
    increment = 0.5 * frame_time * (velocity[:-1] + velocity[1:])
    displacement = np.zeros((len(paths), paths.shape[1], 3), dtype=float)
    displacement[1:] = np.cumsum(increment, axis=0)
    return displacement


def simulate_slow_bath_displacements(
    model: dict[str, np.ndarray | float],
    *,
    step_count: int,
    simulation_count: int,
    frame_time: float,
    seed: int,
) -> np.ndarray:
    """Generate displacements without retaining every auxiliary-state frame."""

    if step_count < 1 or simulation_count < 1:
        raise ValueError("step_count and simulation_count must be positive")
    if not np.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be positive and finite")
    transition = np.asarray(model["transition_matrix"], dtype=float)
    state_mean = np.asarray(model["state_mean"], dtype=float)
    innovation_sqrt = np.asarray(model["innovation_square_root"], dtype=float)
    residual_pool = np.asarray(model["standardized_residual_blocks"], dtype=float)
    state_pool = np.asarray(model["training_state_pool"], dtype=float)
    rng = np.random.default_rng(seed)
    current = state_pool[rng.integers(len(state_pool), size=simulation_count)].copy()
    displacement = np.zeros((step_count + 1, simulation_count, 3), dtype=float)
    for step in range(1, step_count + 1):
        previous_velocity = current[:, 0].copy()
        centered = current - state_mean[None, :, None]
        current = state_mean[None, :, None] + np.einsum("ab,sbc->sac", transition, centered)
        standardized = residual_pool[rng.integers(len(residual_pool), size=simulation_count)]
        current += np.einsum("ab,sbc->sac", innovation_sqrt, standardized)
        displacement[step] = displacement[step - 1] + 0.5 * frame_time * (
            previous_velocity + current[:, 0]
        )
    return displacement
