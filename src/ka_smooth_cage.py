"""Differentiable microscopic cage coordinates for KA Langevin dynamics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ka_local_cage import ka_lj_force_and_isotropic_curvature
from ka_replicates import load_lammps_custom_trajectory


_SIGMA = np.array([[1.0, 0.8], [0.8, 0.88]], dtype=float)
_CUTOFF_SCALE = 2.5


def wendland_c4_weight(scaled_distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the compact Wendland C4 weight and its derivative in ``s``."""

    scaled = np.asarray(scaled_distance, dtype=float)
    if np.any(~np.isfinite(scaled)) or np.any(scaled < 0.0):
        raise ValueError("scaled_distance must be finite and nonnegative")
    active = scaled < 1.0
    value = np.where(active, scaled, 1.0)
    weight = np.where(
        active,
        (1.0 - value) ** 6 * (35.0 * value**2 + 18.0 * value + 3.0),
        0.0,
    )
    derivative = np.where(
        active,
        -56.0 * value * (5.0 * value + 1.0) * (1.0 - value) ** 5,
        0.0,
    )
    return weight, derivative


def smooth_force_support_cage(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_index: int,
) -> dict[str, np.ndarray | float]:
    """Return a smooth force-support cage coordinate and analytic Jacobian."""

    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or np.any(~np.isfinite(positions)):
        raise ValueError("positions must be a finite (particles, 3) array")
    if particle_types.shape != (len(positions),) or np.any(
        (particle_types < 0) | (particle_types > 1)
    ):
        raise ValueError("particle_types must be aligned KA 0/1 labels")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(
        box_lengths <= 0.0
    ):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if (
        isinstance(target_index, bool)
        or not isinstance(target_index, (int, np.integer))
        or target_index < 0
        or target_index >= len(positions)
    ):
        raise ValueError("target_index must select one particle")

    displacement = positions - positions[target_index]
    displacement -= box_lengths * np.rint(displacement / box_lengths)
    distance = np.linalg.norm(displacement, axis=1)
    other = np.arange(len(positions)) != int(target_index)
    if np.any(distance[other] <= 1e-12):
        raise ValueError("distinct particles must not overlap")

    support_radius = _CUTOFF_SCALE * _SIGMA[particle_types[target_index], particle_types]
    scaled_distance = distance / support_radius
    weight, scaled_derivative = wendland_c4_weight(scaled_distance)
    weight[target_index] = 0.0
    scaled_derivative[target_index] = 0.0
    total_weight = float(np.sum(weight))
    if not total_weight > 0.0:
        raise ValueError("target particle has no neighbor inside the KA force support")

    mean_offset = np.sum(weight[:, None] * displacement, axis=0) / total_weight
    unit = displacement / np.maximum(distance[:, None], 1e-12)
    radial_gradient = scaled_derivative[:, None] * unit / support_radius[:, None]
    block = (
        weight[:, None, None] * np.eye(3)
        + (displacement - mean_offset)[:, :, None] * radial_gradient[:, None, :]
    ) / total_weight
    block[target_index] = 0.0
    jacobian = -block
    jacobian[target_index] = np.sum(block, axis=0)
    cage_position = positions[target_index] + mean_offset
    return {
        "cage_position": cage_position,
        "relative_position": -mean_offset,
        "mean_neighbor_offset": mean_offset,
        "jacobian": jacobian,
        "weights": weight,
        "support": weight > 0.0,
        "support_radius": support_radius,
        "total_weight": total_weight,
        "thermodynamic_claim_allowed": 0.0,
    }


def smooth_cage_projected_observables(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_index: int,
    friction: float,
    temperature: float,
    directional_step: float,
    potential_protocol: str = "ka_lj_c3_switch",
) -> dict[str, np.ndarray | float]:
    """Return the exact instantaneous SDE coefficients of the cage coordinate."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if velocities.shape != positions.shape or np.any(~np.isfinite(velocities)):
        raise ValueError("velocities must be finite and align with positions")
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(directional_step) or directional_step <= 0.0:
        raise ValueError("directional_step must be finite and positive")
    if potential_protocol not in {"ka_lj_cut", "ka_lj_c3_switch"}:
        raise ValueError("unsupported KA pair-potential protocol")

    coordinate = smooth_force_support_cage(
        positions,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_index=target_index,
    )
    jacobian = np.asarray(coordinate["jacobian"], dtype=float)
    relative_velocity = np.einsum("nab,nb->a", jacobian, velocities)
    active = np.flatnonzero(np.any(np.abs(jacobian) > 0.0, axis=(1, 2)))
    active_force, _ = ka_lj_force_and_isotropic_curvature(
        positions,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=active,
        potential_protocol=potential_protocol,
    )
    force_drift = np.einsum("nab,nb->a", jacobian[active], active_force)

    plus = smooth_force_support_cage(
        positions + directional_step * velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_index=target_index,
    )
    minus = smooth_force_support_cage(
        positions - directional_step * velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_index=target_index,
    )
    plus_velocity = np.einsum("nab,nb->a", plus["jacobian"], velocities)
    minus_velocity = np.einsum("nab,nb->a", minus["jacobian"], velocities)
    geometric_drift = (plus_velocity - minus_velocity) / (2.0 * directional_step)
    projected_drift = force_drift + geometric_drift - friction * relative_velocity

    gram = np.einsum("nab,ncb->ac", jacobian, jacobian)
    eigenvalues = np.linalg.eigvalsh(gram)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("smooth cage Jacobian Gram matrix must be positive definite")
    effective_mass = np.linalg.inv(gram)
    return {
        **coordinate,
        "relative_velocity": relative_velocity,
        "force_drift": force_drift,
        "geometric_drift": geometric_drift,
        "projected_drift": projected_drift,
        "jacobian_gram": gram,
        "effective_mass": effective_mass,
        "noise_covariance_rate": 2.0 * friction * temperature * gram,
        "jacobian_gram_minimum_eigenvalue": float(np.min(eigenvalues)),
        "jacobian_gram_condition_number": float(np.max(eigenvalues) / np.min(eigenvalues)),
        "friction": float(friction),
        "temperature": float(temperature),
        "directional_step": float(directional_step),
        "potential_protocol": np.asarray(potential_protocol),
        "thermodynamic_claim_allowed": 0.0,
    }


def extract_smooth_cage_path(
    path: Path,
    *,
    target_id: int,
    friction: float,
    temperature: float,
    integration_time_step: float,
    directional_step: float,
    potential_protocol: str = "ka_lj_c3_switch",
) -> dict[str, np.ndarray | float]:
    """Extract the smooth cage SDE coefficients from one full-state dump."""

    if isinstance(target_id, bool) or not isinstance(target_id, int) or target_id < 1:
        raise ValueError("target_id must be a positive one-based particle id")
    if not math.isfinite(integration_time_step) or integration_time_step <= 0.0:
        raise ValueError("integration_time_step must be finite and positive")
    trajectory = load_lammps_custom_trajectory(Path(path))
    if "velocities" not in trajectory:
        raise ValueError("trajectory must contain full particle velocities")
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    velocities = np.asarray(trajectory["velocities"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    timesteps = np.asarray(trajectory["timesteps"], dtype=np.int64)
    target_index = target_id - 1
    if target_index >= positions.shape[1]:
        raise ValueError("target_id lies outside the trajectory atom table")
    intervals = np.diff(timesteps)
    if len(intervals) == 0 or not np.all(intervals == intervals[0]):
        raise ValueError("saved trajectory timesteps must be uniform")

    frame_count, particle_count = positions.shape[:2]
    relative_position = np.empty((frame_count, 3), dtype=float)
    relative_velocity = np.empty_like(relative_position)
    projected_drift = np.empty_like(relative_position)
    jacobian = np.empty((frame_count, particle_count, 3, 3), dtype=float)
    gram = np.empty((frame_count, 3, 3), dtype=float)
    covariance_rate = np.empty_like(gram)
    effective_mass = np.empty_like(gram)
    condition_number = np.empty(frame_count, dtype=float)
    for frame, (frame_positions, frame_velocities) in enumerate(zip(positions, velocities)):
        observable = smooth_cage_projected_observables(
            frame_positions,
            velocities=frame_velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_index=target_index,
            friction=friction,
            temperature=temperature,
            directional_step=directional_step,
            potential_protocol=potential_protocol,
        )
        relative_position[frame] = observable["relative_position"]
        relative_velocity[frame] = observable["relative_velocity"]
        projected_drift[frame] = observable["projected_drift"]
        jacobian[frame] = observable["jacobian"]
        gram[frame] = observable["jacobian_gram"]
        covariance_rate[frame] = observable["noise_covariance_rate"]
        effective_mass[frame] = observable["effective_mass"]
        condition_number[frame] = observable["jacobian_gram_condition_number"]
    frame_time = float(intervals[0]) * integration_time_step
    return {
        "time": (timesteps - timesteps[0]).astype(float) * integration_time_step,
        "relative_position": relative_position,
        "relative_velocity": relative_velocity,
        "projected_drift": projected_drift,
        "jacobian": jacobian,
        "jacobian_gram": gram,
        "noise_covariance_rate": covariance_rate,
        "effective_mass": effective_mass,
        "jacobian_gram_condition_number": condition_number,
        "frame_time": frame_time,
        "target_id": float(target_id),
        "friction": float(friction),
        "temperature": float(temperature),
        "directional_step": float(directional_step),
        "potential_protocol": np.asarray(potential_protocol),
        "thermodynamic_claim_allowed": 0.0,
    }


def matched_smooth_cage_tangent(
    positive: dict[str, np.ndarray | float],
    negative: dict[str, np.ndarray | float],
    *,
    epsilon: float,
) -> dict[str, np.ndarray | float]:
    """Reduce matched cage paths to a common-noise tangent process."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    positive_time = np.asarray(positive["time"], dtype=float)
    negative_time = np.asarray(negative["time"], dtype=float)
    if positive_time.ndim != 1 or len(positive_time) < 2 or not np.array_equal(
        positive_time, negative_time
    ):
        raise ValueError("matched paths must have identical one-dimensional time grids")
    response: dict[str, np.ndarray] = {}
    for key in ("relative_position", "relative_velocity", "projected_drift", "jacobian"):
        plus = np.asarray(positive[key], dtype=float)
        minus = np.asarray(negative[key], dtype=float)
        if plus.shape != minus.shape or plus.shape[0] != len(positive_time):
            raise ValueError(f"matched {key} arrays must align with the time grid")
        if np.any(~np.isfinite(plus)) or np.any(~np.isfinite(minus)):
            raise ValueError(f"matched {key} arrays must be finite")
        response[key] = (plus - minus) / (2.0 * epsilon)
    delta_jacobian = response["jacobian"]
    friction = float(positive["friction"])
    temperature = float(positive["temperature"])
    frame_time = float(positive["frame_time"])
    for key, left, right in (
        ("friction", friction, float(negative["friction"])),
        ("temperature", temperature, float(negative["temperature"])),
        ("frame_time", frame_time, float(negative["frame_time"])),
    ):
        if not np.isclose(left, right, rtol=0.0, atol=1e-15):
            raise ValueError(f"matched paths must have identical {key}")
    covariance_rate = 2.0 * friction * temperature * np.einsum(
        "tnab,tncb->tac", delta_jacobian, delta_jacobian
    )
    return {
        "time": positive_time,
        "relative_position_response": response["relative_position"],
        "relative_velocity_response": response["relative_velocity"],
        "projected_drift_response": response["projected_drift"],
        "jacobian_response": delta_jacobian,
        "tangent_noise_covariance_rate": covariance_rate,
        "frame_time": frame_time,
        "friction": friction,
        "temperature": temperature,
        "epsilon": float(epsilon),
        "thermodynamic_claim_allowed": 0.0,
    }


def integrated_smooth_cage_tangent_covariance(
    tangent: dict[str, np.ndarray | float],
    *,
    stride: int,
) -> dict[str, np.ndarray | float]:
    """Integrate drift and covariance on nonoverlapping strided intervals."""

    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    time = np.asarray(tangent["time"], dtype=float)
    velocity = np.asarray(tangent["relative_velocity_response"], dtype=float)
    drift = np.asarray(tangent["projected_drift_response"], dtype=float)
    covariance_rate = np.asarray(tangent["tangent_noise_covariance_rate"], dtype=float)
    if time.ndim != 1 or len(time) < 2 or (len(time) - 1) % stride:
        raise ValueError("time grid must contain a positive number of complete strided intervals")
    if velocity.shape != (len(time), 3) or drift.shape != velocity.shape:
        raise ValueError("velocity and drift responses must have shape (frames, 3)")
    if covariance_rate.shape != (len(time), 3, 3):
        raise ValueError("covariance rate must have shape (frames, 3, 3)")
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(velocity)) or np.any(
        ~np.isfinite(drift)
    ) or np.any(~np.isfinite(covariance_rate)):
        raise ValueError("tangent arrays must be finite")
    intervals = np.diff(time)
    frame_time = float(tangent["frame_time"])
    if not np.allclose(intervals, frame_time, rtol=1e-10, atol=1e-12):
        raise ValueError("time grid must be uniform and match frame_time")

    residual = []
    integrated_covariance = []
    for start in range(0, len(time) - 1, stride):
        stop = start + stride
        drift_integral = frame_time * (
            0.5 * drift[start]
            + np.sum(drift[start + 1 : stop], axis=0)
            + 0.5 * drift[stop]
        )
        covariance_integral = frame_time * (
            0.5 * covariance_rate[start]
            + np.sum(covariance_rate[start + 1 : stop], axis=0)
            + 0.5 * covariance_rate[stop]
        )
        residual.append(velocity[stop] - velocity[start] - drift_integral)
        integrated_covariance.append(covariance_integral)
    return {
        "residual": np.asarray(residual),
        "integrated_covariance": np.asarray(integrated_covariance),
        "stride": float(stride),
        "interval_time": frame_time * stride,
        "thermodynamic_claim_allowed": 0.0,
    }
