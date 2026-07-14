"""Finite-frame diagnostics for projected microscopic Langevin innovations."""

from __future__ import annotations

import math

import numpy as np


def block_projected_ito_increments(
    state: np.ndarray,
    drift: np.ndarray,
    covariance_rate: np.ndarray,
    *,
    frame_time: float,
    stride: int,
    scheme: str,
) -> dict[str, np.ndarray | float]:
    """Build nonoverlapping drift-subtracted increments and covariances."""

    values = np.asarray(state, dtype=float)
    deterministic = np.asarray(drift, dtype=float)
    covariance = np.asarray(covariance_rate, dtype=float)
    if (
        values.ndim != 3
        or values.shape[0] < 2
        or deterministic.shape != values.shape
        or covariance.shape != (*values.shape, values.shape[-1])
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(deterministic))
        or np.any(~np.isfinite(covariance))
    ):
        raise ValueError("state, drift, and covariance rate must be finite aligned paths")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be finite and positive")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if scheme not in {"left", "trapezoid", "adams_bashforth2"}:
        raise ValueError("scheme must be left, trapezoid, or adams_bashforth2")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError("covariance rate must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = np.max(np.abs(eigenvalues), axis=-1)
    if np.any(eigenvalues[..., 0] < -1e-10 * np.maximum(scale, 1.0)):
        raise ValueError("covariance rate must be positive semidefinite")

    if scheme == "left":
        difference = values[1:] - values[:-1]
        base_residual = difference - frame_time * deterministic[:-1]
        base_covariance = frame_time * covariance[:-1]
        base_start = 0
    elif scheme == "trapezoid":
        difference = values[1:] - values[:-1]
        base_residual = difference - 0.5 * frame_time * (
            deterministic[:-1] + deterministic[1:]
        )
        base_covariance = 0.5 * frame_time * (covariance[:-1] + covariance[1:])
        base_start = 0
    else:
        difference = values[2:] - values[1:-1]
        base_residual = difference - frame_time * (
            1.5 * deterministic[1:-1] - 0.5 * deterministic[:-2]
        )
        base_covariance = frame_time * covariance[1:-1]
        base_start = 1

    block_count = len(base_residual) // stride
    if block_count < 1:
        raise ValueError("stride exceeds the available interval count")
    retained = block_count * stride
    residual = base_residual[:retained].reshape(
        block_count, stride, values.shape[1], values.shape[2]
    ).sum(axis=1)
    integrated_covariance = base_covariance[:retained].reshape(
        block_count,
        stride,
        values.shape[1],
        values.shape[2],
        values.shape[2],
    ).sum(axis=1)
    starting_frame_indices = base_start + np.arange(block_count) * stride
    starting_state = values[starting_frame_indices]
    return {
        "residual": residual,
        "integrated_covariance": integrated_covariance,
        "starting_state": starting_state,
        "starting_frame_indices": starting_frame_indices,
        "block_count": float(block_count),
        "thermodynamic_claim_allowed": 0.0,
    }


def _maximum_absolute_cross_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left, axis=0, keepdims=True)
    right_centered = right - np.mean(right, axis=0, keepdims=True)
    numerator = left_centered.T @ right_centered
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=0)[:, None]
        * np.sum(right_centered**2, axis=0)[None, :]
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return float(np.max(np.abs(correlation)))


def multivariate_noise_covariance_diagnostic(
    residual: np.ndarray,
    integrated_covariance: np.ndarray,
    *,
    starting_state: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Whiten heteroscedastic vector increments and report calibration metrics."""

    values = np.asarray(residual, dtype=float)
    covariance = np.asarray(integrated_covariance, dtype=float)
    if (
        values.ndim != 3
        or values.shape[0] < 2
        or values.shape[1] < 2
        or covariance.shape != (*values.shape, values.shape[-1])
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(covariance))
    ):
        raise ValueError("residual and covariance must be finite aligned member paths")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError("integrated covariance must be symmetric")
    if starting_state is not None:
        physical_state = np.asarray(starting_state, dtype=float)
        if (
            physical_state.ndim != 3
            or physical_state.shape[:2] != values.shape[:2]
            or np.any(~np.isfinite(physical_state))
        ):
            raise ValueError("starting_state must be a finite aligned member path")
    else:
        physical_state = None

    flat_values = values.reshape(-1, values.shape[-1])
    flat_covariance = covariance.reshape(-1, values.shape[-1], values.shape[-1])
    eigenvalues, eigenvectors = np.linalg.eigh(flat_covariance)
    scale = np.max(eigenvalues, axis=1, keepdims=True)
    if np.any(scale <= 0.0) or np.any(eigenvalues < -1e-10 * scale):
        raise ValueError("integrated covariance must be positive semidefinite and nonzero")
    regularized = np.maximum(eigenvalues, 1e-12 * scale)
    projected = np.einsum("nrc,nr->nc", eigenvectors, flat_values)
    whitened_eigenbasis = projected / np.sqrt(regularized)
    whitened = np.einsum("nrc,nc->nr", eigenvectors, whitened_eigenbasis)
    squared_mahalanobis = np.sum(whitened_eigenbasis**2, axis=1)

    observed_energy = np.sum(flat_values**2, axis=1)
    predicted_energy = np.trace(flat_covariance, axis1=1, axis2=2)
    trace_ratio = float(np.sum(observed_energy) / np.sum(predicted_energy))
    whitened_mean = np.mean(whitened, axis=0)
    whitened_covariance = np.cov(whitened, rowvar=False, bias=True)
    covariance_error = whitened_covariance - np.eye(values.shape[-1])

    component_excess = np.empty(values.shape[-1], dtype=float)
    for component in range(values.shape[-1]):
        centered = whitened[:, component] - whitened_mean[component]
        variance = float(np.mean(centered**2))
        component_excess[component] = (
            float(np.mean(centered**4) / variance**2 - 3.0)
            if variance > 0.0
            else math.nan
        )

    whitened_paths = whitened.reshape(values.shape)
    lag_source = whitened_paths[:, :-1].reshape(-1, values.shape[-1])
    lag_target = whitened_paths[:, 1:].reshape(-1, values.shape[-1])
    lag_correlation = _maximum_absolute_cross_correlation(lag_source, lag_target)
    if physical_state is None:
        state_correlation = math.nan
    else:
        state_correlation = _maximum_absolute_cross_correlation(
            whitened,
            physical_state.reshape(-1, physical_state.shape[-1]),
        )

    return {
        "trace_variance_ratio": trace_ratio,
        "mean_squared_mahalanobis_per_dimension": float(
            np.mean(squared_mahalanobis) / values.shape[-1]
        ),
        "maximum_absolute_whitened_mean": float(np.max(np.abs(whitened_mean))),
        "maximum_absolute_whitened_covariance_error": float(
            np.max(np.abs(covariance_error))
        ),
        "maximum_absolute_whitened_component_excess_kurtosis": float(
            np.nanmax(np.abs(component_excess))
        ),
        "maximum_absolute_whitened_lag1_correlation": lag_correlation,
        "maximum_absolute_whitened_state_correlation": state_correlation,
        "minimum_integrated_covariance_eigenvalue": float(np.min(eigenvalues)),
        "sample_count": float(len(flat_values)),
        "whitened_residual": whitened_paths,
        "thermodynamic_claim_allowed": 0.0,
    }
