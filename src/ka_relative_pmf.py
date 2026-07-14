"""Equilibrium mean-force and Markov-OU diagnostics for a relative cage coordinate."""

from __future__ import annotations

import math

import numpy as np


def fit_gaussian_relative_pmf(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    relative_noise_variance_rate: float,
    friction: float,
) -> dict[str, np.ndarray | float]:
    """Infer constant-metric Gaussian PMF parameters from microscopic states."""

    position = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    if (
        position.shape != velocity.shape
        or position.ndim < 2
        or position.shape[-1] != 3
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
    ):
        raise ValueError("relative position and velocity must align as finite 3-vectors")
    if not math.isfinite(relative_noise_variance_rate) or relative_noise_variance_rate <= 0.0:
        raise ValueError("relative noise variance rate must be finite and positive")
    if not math.isfinite(friction) or friction <= 0.0:
        raise ValueError("friction must be finite and positive")

    flattened_position = position.reshape(-1, 3)
    flattened_velocity = velocity.reshape(-1, 3)
    centered_position = flattened_position - np.mean(flattened_position, axis=0)
    centered_velocity = flattened_velocity - np.mean(flattened_velocity, axis=0)
    position_component_variance = np.mean(centered_position**2, axis=0)
    velocity_component_variance = np.mean(centered_velocity**2, axis=0)
    position_variance = float(np.mean(position_component_variance))
    velocity_variance = float(np.mean(velocity_component_variance))
    fdt_velocity_variance = relative_noise_variance_rate / (2.0 * friction)
    if position_variance <= 0.0 or velocity_variance <= 0.0:
        raise ValueError("relative coordinate variances must be positive")

    cross_covariance = np.mean(
        centered_position[:, :, None] * centered_velocity[:, None, :], axis=0
    )
    cross_scale = np.sqrt(position_variance * velocity_variance)
    return {
        "relative_position_variance": position_variance,
        "relative_velocity_variance": velocity_variance,
        "fdt_velocity_variance": float(fdt_velocity_variance),
        "fdt_velocity_variance_relative_error": abs(
            velocity_variance / fdt_velocity_variance - 1.0
        ),
        "harmonic_acceleration_stiffness": float(
            fdt_velocity_variance / position_variance
        ),
        "observed_variance_ratio_stiffness": float(
            velocity_variance / position_variance
        ),
        "position_variance_isotropy_error": float(
            np.max(np.abs(position_component_variance / position_variance - 1.0))
        ),
        "velocity_variance_isotropy_error": float(
            np.max(np.abs(velocity_component_variance / velocity_variance - 1.0))
        ),
        "maximum_normalized_position_velocity_covariance": float(
            np.max(np.abs(cross_covariance)) / cross_scale
        ),
        "thermodynamic_claim_allowed": 0.0,
    }


def underdamped_ou_propagator(
    harmonic_acceleration_stiffness: float,
    friction: float,
    time: float,
) -> np.ndarray:
    """Return exp([[0,1],[-kappa,-gamma]] time) without SciPy."""

    kappa = float(harmonic_acceleration_stiffness)
    gamma = float(friction)
    elapsed = float(time)
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("harmonic acceleration stiffness must be finite and positive")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("time must be finite and nonnegative")
    if elapsed == 0.0:
        return np.eye(2)
    generator = np.array([[0.0, 1.0], [-kappa, -gamma]])
    eigenvalues, eigenvectors = np.linalg.eig(generator)
    propagator = eigenvectors @ np.diag(np.exp(eigenvalues * elapsed)) @ np.linalg.inv(
        eigenvectors
    )
    return np.real_if_close(propagator, tol=1000).astype(float)
