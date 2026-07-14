"""Microscopic structural coordinates for KA isoconfigurational dynamics."""

from __future__ import annotations

import math

import numpy as np


def species_resolved_radial_features(
    positions: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    *,
    radii: np.ndarray,
    width: float,
    cutoff: float,
    block_size: int = 128,
) -> np.ndarray:
    """Return smooth A/B radial shell densities around selected particles."""

    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target_indices = np.asarray(target_indices)
    radii = np.asarray(radii, dtype=float)
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
        target_indices.ndim != 1
        or len(target_indices) < 1
        or np.any(target_indices != target_indices.astype(int))
    ):
        raise ValueError("target_indices must be a nonempty integer vector")
    target_indices = target_indices.astype(int)
    if (
        np.any(target_indices < 0)
        or np.any(target_indices >= len(positions))
        or len(np.unique(target_indices)) != len(target_indices)
    ):
        raise ValueError("target_indices must select distinct particles")
    if (
        radii.ndim != 1
        or len(radii) < 1
        or np.any(~np.isfinite(radii))
        or np.any(radii <= 0.0)
        or np.any(np.diff(radii) <= 0.0)
    ):
        raise ValueError("radii must be finite, positive, and strictly increasing")
    if (
        not math.isfinite(width)
        or width <= 0.0
        or not math.isfinite(cutoff)
        or cutoff <= radii[-1]
    ):
        raise ValueError("width and cutoff must be finite with cutoff above all radii")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, (int, np.integer))
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer")

    output = np.empty((len(target_indices), 2 * len(radii)), dtype=float)
    wrapped = np.mod(positions, box_lengths)
    for start in range(0, len(target_indices), int(block_size)):
        stop = min(start + int(block_size), len(target_indices))
        selected = target_indices[start:stop]
        displacement = wrapped[selected, None, :] - wrapped[None, :, :]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        distance = np.linalg.norm(displacement, axis=2)
        cutoff_weight = np.where(
            distance < cutoff,
            0.5 * (np.cos(np.pi * distance / cutoff) + 1.0),
            0.0,
        )
        cutoff_weight[np.arange(len(selected)), selected] = 0.0
        for species in (0, 1):
            species_weight = cutoff_weight * (particle_types[None, :] == species)
            shell_weight = np.exp(
                -0.5
                * ((distance[:, :, None] - radii[None, None, :]) / width) ** 2
            )
            output[
                start:stop, species * len(radii) : (species + 1) * len(radii)
            ] = np.einsum("ij,ijn->in", species_weight, shell_weight)
    return output


def expand_isoconfigurational_structural_rows(
    features: np.ndarray,
    first_passage: np.ndarray,
    escaped: np.ndarray,
    *,
    horizon: float,
) -> dict[str, np.ndarray]:
    """Expand parent-target structure over clones and aggregate propensity."""

    features = np.asarray(features, dtype=float)
    first_passage = np.asarray(first_passage, dtype=float)
    escaped_raw = np.asarray(escaped)
    if (
        features.ndim != 3
        or features.shape[0] < 2
        or features.shape[1] < 1
        or features.shape[2] < 1
        or np.any(~np.isfinite(features))
    ):
        raise ValueError(
            "features must be finite (parents, targets, dimensions) with at least two parents"
        )
    parent_count, target_count, feature_count = features.shape
    if (
        first_passage.ndim != 3
        or first_passage.shape[0] != parent_count
        or first_passage.shape[1] < 2
        or first_passage.shape[2] != target_count
        or np.any(~np.isfinite(first_passage))
    ):
        raise ValueError(
            "first_passage must align as (parents, clones, targets) with at least two clones"
        )
    if escaped_raw.shape != first_passage.shape or not np.all(
        np.isin(escaped_raw, (False, True, 0, 1))
    ):
        raise ValueError("escaped must be an aligned Boolean tensor")
    escaped = escaped_raw.astype(bool)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if np.any(first_passage <= 0.0) or np.any(first_passage > horizon):
        raise ValueError("first_passage must be positive and no larger than horizon")
    if np.any(
        (~escaped)
        & ~np.isclose(first_passage, horizon, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("unescaped rows must use exact horizon censoring")

    clone_count = first_passage.shape[1]
    clone_features = np.broadcast_to(
        features[:, None, :, :],
        (parent_count, clone_count, target_count, feature_count),
    ).reshape(-1, feature_count)
    groups = np.broadcast_to(
        np.arange(parent_count)[:, None, None],
        (parent_count, clone_count, target_count),
    ).reshape(-1)
    configuration_groups = np.broadcast_to(
        np.arange(parent_count)[:, None], (parent_count, target_count)
    ).reshape(-1)
    return {
        "features": clone_features.copy(),
        "first_passage": first_passage.reshape(-1),
        "escaped": escaped.reshape(-1),
        "groups": groups,
        "configuration_features": features.reshape(-1, feature_count),
        "successes": np.sum(escaped, axis=1).reshape(-1).astype(float),
        "trials": np.full(parent_count * target_count, float(clone_count)),
        "configuration_groups": configuration_groups,
    }
