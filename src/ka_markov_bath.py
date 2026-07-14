"""Finite-dimensional Markov models for projected Mori innovations."""

from __future__ import annotations

import math

import numpy as np


def vector_autoregressive_residual(
    series: np.ndarray,
    coefficients: np.ndarray,
    *,
    mean: np.ndarray,
) -> np.ndarray:
    """Return the white residual of a fitted stationary vector AR process."""

    values = np.asarray(series, dtype=float)
    memory = np.asarray(coefficients, dtype=float)
    location = np.asarray(mean, dtype=float)
    if (
        values.ndim != 3
        or memory.ndim != 3
        or memory.shape[1] != memory.shape[2]
        or values.shape[2] != memory.shape[1]
        or location.shape != (values.shape[2],)
        or len(memory) < 1
        or len(values) <= len(memory)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(memory))
        or np.any(~np.isfinite(location))
    ):
        raise ValueError("series, coefficients, and mean must be finite and aligned")
    centered = values - location
    order = len(memory)
    prediction = np.zeros_like(centered[order:])
    for lag, coefficient in enumerate(memory, start=1):
        prediction += centered[order - lag : len(centered) - lag] @ coefficient.T
    return centered[order:] - prediction


def fit_stationary_vector_autoregression(
    series: np.ndarray,
    *,
    order: int,
    ridge_regularization: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Fit a pooled stationary VAR by multivariate Yule--Walker equations."""

    values = np.asarray(series, dtype=float)
    if (
        values.ndim != 3
        or values.shape[2] < 1
        or order < 1
        or len(values) <= order + 1
        or not math.isfinite(ridge_regularization)
        or ridge_regularization < 0.0
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("series and vector autoregression controls must be valid")
    mean = np.mean(values, axis=(0, 1))
    centered = values - mean
    dimension = values.shape[2]
    covariance = np.empty((order + 1, dimension, dimension), dtype=float)
    for lag in range(order + 1):
        left = centered[lag:]
        right = centered[: len(centered) - lag]
        covariance[lag] = np.einsum("tni,tnj->ij", left, right) / (
            left.shape[0] * left.shape[1]
        )

    lag_covariance = np.empty(
        (order * dimension, order * dimension), dtype=float
    )
    for row in range(order):
        for column in range(order):
            lag = column - row
            block = covariance[lag] if lag >= 0 else covariance[-lag].T
            lag_covariance[
                row * dimension : (row + 1) * dimension,
                column * dimension : (column + 1) * dimension,
            ] = block
    cross_covariance = np.concatenate(covariance[1:], axis=1)
    scale = float(np.trace(lag_covariance) / len(lag_covariance))
    regularized = lag_covariance + ridge_regularization * max(scale, 1e-30) * np.eye(
        len(lag_covariance)
    )
    flattened = np.linalg.solve(regularized.T, cross_covariance.T).T
    coefficients = flattened.reshape(dimension, order, dimension).transpose(1, 0, 2)
    residual = vector_autoregressive_residual(values, coefficients, mean=mean)
    white_covariance = np.einsum("tni,tnj->ij", residual, residual) / (
        residual.shape[0] * residual.shape[1]
    )
    white_covariance = 0.5 * (white_covariance + white_covariance.T)

    companion = np.zeros((order * dimension, order * dimension), dtype=float)
    companion[:dimension] = flattened
    if order > 1:
        companion[dimension:, :-dimension] = np.eye((order - 1) * dimension)
    eigenvalues = np.linalg.eigvals(companion)
    noise_eigenvalues = np.linalg.eigvalsh(white_covariance)
    return {
        "mean": mean,
        "coefficients": coefficients,
        "white_noise_covariance": white_covariance,
        "spectral_radius": float(np.max(np.abs(eigenvalues))),
        "minimum_noise_covariance_eigenvalue": float(np.min(noise_eigenvalues)),
        "maximum_noise_covariance_eigenvalue": float(np.max(noise_eigenvalues)),
    }


def simulate_mori_with_var_bath(
    initial_state_history: np.ndarray,
    operators: np.ndarray,
    initial_innovation_history: np.ndarray,
    var_coefficients: np.ndarray,
    *,
    innovation_mean: np.ndarray,
    white_noise_covariance: np.ndarray,
    white_noise_pool: np.ndarray | None = None,
    output_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a Mori recurrence with a white-driven finite VAR bath."""

    state_history = np.asarray(initial_state_history, dtype=float).copy()
    memory = np.asarray(operators, dtype=float)
    innovation_history = np.asarray(initial_innovation_history, dtype=float).copy()
    bath = np.asarray(var_coefficients, dtype=float)
    mean = np.asarray(innovation_mean, dtype=float)
    covariance = np.asarray(white_noise_covariance, dtype=float)
    pool = None if white_noise_pool is None else np.asarray(white_noise_pool, dtype=float)
    if (
        state_history.ndim != 3
        or innovation_history.ndim != 3
        or memory.ndim != 3
        or bath.ndim != 3
        or memory.shape[1] != memory.shape[2]
        or bath.shape[1] != bath.shape[2]
        or state_history.shape[0] != innovation_history.shape[0]
        or state_history.shape[1] != len(memory)
        or innovation_history.shape[1] != len(bath)
        or state_history.shape[2] != memory.shape[1]
        or state_history.shape[2] != bath.shape[1]
        or mean.shape != (memory.shape[1],)
        or covariance.shape != memory.shape[1:]
        or output_count < 1
        or np.any(~np.isfinite(state_history))
        or np.any(~np.isfinite(innovation_history))
        or np.any(~np.isfinite(memory))
        or np.any(~np.isfinite(bath))
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(covariance))
        or (
            pool is not None
            and (
                pool.ndim not in (2, 3)
                or pool.shape[-1] != memory.shape[1]
                or not len(pool)
                or np.any(~np.isfinite(pool))
            )
        )
    ):
        raise ValueError("Mori and VAR bath inputs must be finite and aligned")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValueError("white-noise covariance must be positive semidefinite")
    noise_factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    state = np.empty(
        (len(state_history), output_count, state_history.shape[2]), dtype=float
    )
    innovation = np.empty_like(state)
    state[:, 0] = state_history[:, -1]
    innovation[:, 0] = innovation_history[:, -1]
    for output_index in range(1, output_count):
        next_innovation = mean + sum(
            (innovation_history[:, -1 - lag] - mean) @ coefficient.T
            for lag, coefficient in enumerate(bath)
        )
        if pool is None:
            driving = rng.normal(size=next_innovation.shape) @ noise_factor.T
        elif pool.ndim == 2:
            driving = pool[rng.integers(len(pool), size=len(next_innovation))]
        else:
            times = rng.integers(len(pool), size=len(next_innovation))
            sources = rng.integers(pool.shape[1], size=len(next_innovation))
            driving = pool[times, sources]
        next_innovation += driving
        next_state = sum(
            state_history[:, -1 - lag] @ coefficient.T
            for lag, coefficient in enumerate(memory)
        )
        next_state += next_innovation
        state_history[:, :-1] = state_history[:, 1:]
        state_history[:, -1] = next_state
        innovation_history[:, :-1] = innovation_history[:, 1:]
        innovation_history[:, -1] = next_innovation
        state[:, output_index] = next_state
        innovation[:, output_index] = next_innovation
    return state, innovation
