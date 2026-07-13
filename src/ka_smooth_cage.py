"""Differentiable microscopic cage coordinates for KA Langevin dynamics."""

from __future__ import annotations

import numpy as np


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
