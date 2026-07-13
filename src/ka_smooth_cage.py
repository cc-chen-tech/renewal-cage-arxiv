"""Differentiable microscopic cage coordinates for KA Langevin dynamics."""

from __future__ import annotations

import math

import numpy as np

from ka_local_cage import ka_lj_force_and_isotropic_curvature


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
