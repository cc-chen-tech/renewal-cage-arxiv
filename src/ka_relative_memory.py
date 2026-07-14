"""Harmonic relative-coordinate Volterra kernels and FDT diagnostics."""

from __future__ import annotations

import math

import numpy as np


def estimate_isoconfigurational_bias(relative_position_paths: np.ndarray) -> np.ndarray:
    """Average a clone ensemble over clone and time, retaining particle identity."""

    paths = np.asarray(relative_position_paths, dtype=float)
    if (
        paths.ndim != 4
        or paths.shape[0] < 1
        or paths.shape[1] < 2
        or paths.shape[-1] != 3
        or np.any(~np.isfinite(paths))
    ):
        raise ValueError("relative position paths must be finite clone-time-particle 3-vectors")
    return np.mean(paths, axis=(0, 1))


def _validate_harmonic_controls(
    harmonic_acceleration_stiffness: float,
    friction: float,
    frame_time: float,
) -> tuple[float, float, float]:
    kappa = float(harmonic_acceleration_stiffness)
    gamma = float(friction)
    dt = float(frame_time)
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("harmonic acceleration stiffness must be finite and positive")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("frame_time must be finite and positive")
    return kappa, gamma, dt


def invert_harmonic_velocity_memory_kernel(
    velocity_velocity_correlation: np.ndarray,
    position_velocity_correlation: np.ndarray,
    *,
    harmonic_acceleration_stiffness: float,
    friction: float,
    frame_time: float,
    kernel_count: int,
) -> np.ndarray:
    """Invert the harmonic GLE correlation equation by triangular recursion."""

    velocity = np.asarray(velocity_velocity_correlation, dtype=float)
    position_velocity = np.asarray(position_velocity_correlation, dtype=float)
    kappa, gamma, dt = _validate_harmonic_controls(
        harmonic_acceleration_stiffness, friction, frame_time
    )
    if (
        velocity.ndim != 1
        or position_velocity.shape != velocity.shape
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(position_velocity))
        or kernel_count < 1
        or len(velocity) < kernel_count + 1
        or velocity[0] <= 0.0
    ):
        raise ValueError("correlations and kernel_count must define a finite causal inversion")
    kernel = np.empty(kernel_count, dtype=float)
    kernel[0] = -2.0 * (
        (velocity[1] - velocity[0]) / dt**2
        + (kappa * position_velocity[0] + gamma * velocity[0]) / dt
    ) / velocity[0]
    for index in range(1, kernel_count):
        known = 0.5 * kernel[0] * velocity[index]
        if index > 1:
            known += float(
                np.dot(kernel[1:index], velocity[index - 1 : 0 : -1])
            )
        kernel[index] = -2.0 * (
            (velocity[index + 1] - velocity[index]) / dt**2
            + (kappa * position_velocity[index] + gamma * velocity[index]) / dt
            + known
        ) / velocity[0]
    return kernel


def propagate_harmonic_gle_correlations(
    kernel: np.ndarray,
    *,
    harmonic_acceleration_stiffness: float,
    friction: float,
    frame_time: float,
    output_count: int,
    position_variance: float,
    velocity_variance: float,
) -> dict[str, np.ndarray | float]:
    """Propagate harmonic-GLE correlations with causal trapezoid memory."""

    memory = np.asarray(kernel, dtype=float)
    kappa, gamma, dt = _validate_harmonic_controls(
        harmonic_acceleration_stiffness, friction, frame_time
    )
    if (
        memory.ndim != 1
        or not len(memory)
        or np.any(~np.isfinite(memory))
        or output_count < 2
        or not math.isfinite(position_variance)
        or position_variance <= 0.0
        or not math.isfinite(velocity_variance)
        or velocity_variance <= 0.0
    ):
        raise ValueError("kernel, output count, and variances must be finite and positive")
    response = np.zeros((output_count, 2, 2), dtype=float)
    response[0] = np.eye(2)
    for index in range(output_count - 1):
        active = min(index, len(memory) - 1)
        convolution = 0.5 * memory[0] * response[index, 1]
        for lag in range(1, active):
            convolution += memory[lag] * response[index - lag, 1]
        if active >= 1:
            convolution += 0.5 * memory[active] * response[index - active, 1]
        next_velocity = (
            response[index, 1]
            - dt * kappa * response[index, 0]
            - dt * gamma * response[index, 1]
            - dt**2 * convolution
        )
        response[index + 1, 1] = next_velocity
        response[index + 1, 0] = response[index, 0] + 0.5 * dt * (
            response[index, 1] + next_velocity
        )
    return {
        "position_position_correlation": position_variance * response[:, 0, 0],
        "velocity_velocity_correlation": velocity_variance * response[:, 1, 1],
        "position_velocity_correlation": velocity_variance * response[:, 0, 1],
        "velocity_position_correlation": position_variance * response[:, 1, 0],
        "response": response,
        "thermodynamic_claim_allowed": 0.0,
    }


def reconstruct_harmonic_random_force(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    relative_drift: np.ndarray,
    *,
    isoconfigurational_bias: np.ndarray,
    harmonic_acceleration_stiffness: float,
    friction: float,
    frame_time: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """Reconstruct the finite-support GLE random force from microscopic drift."""

    position = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    drift = np.asarray(relative_drift, dtype=float)
    bias = np.asarray(isoconfigurational_bias, dtype=float)
    memory = np.asarray(kernel, dtype=float)
    kappa, gamma, dt = _validate_harmonic_controls(
        harmonic_acceleration_stiffness, friction, frame_time
    )
    if (
        position.shape != velocity.shape
        or drift.shape != position.shape
        or position.ndim != 3
        or position.shape[-1] != 3
        or bias.shape != position.shape[1:]
        or memory.ndim != 1
        or not len(memory)
        or len(position) <= len(memory)
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(drift))
        or np.any(~np.isfinite(bias))
        or np.any(~np.isfinite(memory))
    ):
        raise ValueError("paths, bias, and kernel must be finite and aligned")
    relative_mean_force = drift + gamma * velocity + kappa * (
        position - bias[None, :, :]
    )
    random_force = np.empty((len(position) - len(memory) + 1, *position.shape[1:]))
    active = len(memory) - 1
    for output_index, frame in enumerate(range(active, len(position))):
        convolution = 0.5 * memory[0] * velocity[frame]
        for lag in range(1, active):
            convolution += memory[lag] * velocity[frame - lag]
        if active >= 1:
            convolution += 0.5 * memory[active] * velocity[frame - active]
        random_force[output_index] = relative_mean_force[frame] + dt * convolution
    return random_force
