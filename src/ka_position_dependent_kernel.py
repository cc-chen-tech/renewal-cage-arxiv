"""Finite-basis position-dependent kernels for microscopic KA cage dynamics."""

from __future__ import annotations

import math

import numpy as np


def _finite_vectors(position: np.ndarray) -> np.ndarray:
    vectors = np.asarray(position, dtype=float)
    if (
        vectors.ndim < 1
        or vectors.shape[-1] != 3
        or vectors.size < 3
        or np.any(~np.isfinite(vectors))
    ):
        raise ValueError("positions must be finite 3-vectors")
    return vectors


def fit_radial_basis_scale(position: np.ndarray) -> dict[str, float]:
    """Fit radial normalization using training positions only."""

    vectors = _finite_vectors(position)
    radial_square = np.sum(vectors.reshape(-1, 3) ** 2, axis=-1)
    if len(radial_square) < 2:
        raise ValueError("radial basis scale requires at least two training vectors")
    positive = radial_square[radial_square > 0.0]
    sigma = float(np.std(radial_square))
    if len(positive) < 1 or not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("radial basis scale requires nonzero finite variance")
    return {
        "mu_r2": float(np.mean(radial_square)),
        "sigma_r2": sigma,
        "epsilon_r2": float(np.percentile(positive, 1.0)),
    }


def _validated_scale(scale: dict[str, float]) -> tuple[float, float, float]:
    try:
        mu = float(scale["mu_r2"])
        sigma = float(scale["sigma_r2"])
        epsilon = float(scale["epsilon_r2"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("radial basis scale is incomplete") from error
    if (
        not math.isfinite(mu)
        or not math.isfinite(sigma)
        or not math.isfinite(epsilon)
        or mu < 0.0
        or sigma <= 0.0
        or epsilon <= 0.0
    ):
        raise ValueError("radial basis scale must be finite and physical")
    return mu, sigma, epsilon


def radial_vector_basis(
    position: np.ndarray,
    scale: dict[str, float],
) -> np.ndarray:
    """Evaluate ``u``, ``u*s``, and ``u*(s^2-1)`` radial vector functions."""

    vectors = _finite_vectors(position)
    mu, sigma, _ = _validated_scale(scale)
    radial_square = np.sum(vectors**2, axis=-1)
    normalized = (radial_square - mu) / sigma
    return np.stack(
        (
            vectors,
            vectors * normalized[..., None],
            vectors * (normalized**2 - 1.0)[..., None],
        ),
        axis=-2,
    )


def radial_vector_basis_jacobian(
    position: np.ndarray,
    scale: dict[str, float],
) -> np.ndarray:
    """Evaluate exact coordinate Jacobians of the three radial vector bases."""

    vectors = _finite_vectors(position)
    mu, sigma, _ = _validated_scale(scale)
    radial_square = np.sum(vectors**2, axis=-1)
    normalized = (radial_square - mu) / sigma
    identity = np.broadcast_to(np.eye(3), (*vectors.shape[:-1], 3, 3))
    outer = vectors[..., :, None] * vectors[..., None, :]
    first = identity
    second = normalized[..., None, None] * identity + 2.0 * outer / sigma
    third = (
        (normalized**2 - 1.0)[..., None, None] * identity
        + 4.0 * normalized[..., None, None] * outer / sigma
    )
    return np.stack((first, second, third), axis=-3)


def _finite_clone_paths(
    position: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(position, dtype=float)
    velocities = np.asarray(velocity, dtype=float)
    if (
        positions.ndim != 4
        or positions.shape != velocities.shape
        or positions.shape[0] < 1
        or positions.shape[1] < 2
        or positions.shape[2] < 1
        or positions.shape[3] != 3
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(velocities))
    ):
        raise ValueError(
            "kernel paths must be finite aligned clone-time-particle 3-vectors"
        )
    return positions, velocities


def assemble_mz_volterra_system(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    scale: dict[str, float],
    support: int,
) -> dict[str, object]:
    """Assemble one joint causal MZ regression without crossing clone edges."""

    positions, velocities = _finite_clone_paths(position, velocity)
    accelerations = np.asarray(acceleration, dtype=float)
    if (
        accelerations.shape != positions.shape
        or np.any(~np.isfinite(accelerations))
        or isinstance(support, bool)
        or not isinstance(support, (int, np.integer))
        or support < 1
        or support > positions.shape[1]
    ):
        raise ValueError("acceleration and support must align with kernel paths")
    basis = radial_vector_basis(positions, scale)
    jacobian = radial_vector_basis_jacobian(positions, scale)
    jacobian_velocity = np.einsum(
        "ctpbik,ctpk->ctpbi",
        jacobian,
        velocities,
    )
    first = support - 1
    feature_blocks = [basis[:, first:]]
    for lag in range(support):
        stop = positions.shape[1] - lag
        feature_blocks.append(-jacobian_velocity[:, first - lag : stop])
    features = np.concatenate(feature_blocks, axis=-2)
    column_count = features.shape[-2]
    design = np.moveaxis(features, -1, -2).reshape(-1, column_count)
    target = accelerations[:, first:].reshape(-1)
    if np.linalg.matrix_rank(design) < 1:
        raise ValueError("MZ Volterra design has zero numerical rank")
    return {
        "design": design,
        "target": target,
        "basis_count": 3,
        "support": int(support),
        "training_clone_count": int(positions.shape[0]),
        "valid_time_count": int(positions.shape[1] - first),
        "particle_count": int(positions.shape[2]),
        "fit_uses_held_clone": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def solve_regularized_mz_kernel(
    system: dict[str, object],
    ridge: float,
) -> dict[str, object]:
    """Solve a normalized augmented least-squares MZ system."""

    design = np.asarray(system.get("design"), dtype=float)
    target = np.asarray(system.get("target"), dtype=float)
    try:
        basis_count = int(system["basis_count"])
        support = int(system["support"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MZ Volterra system metadata is incomplete") from error
    ridge_value = float(ridge)
    expected_columns = basis_count * (support + 1)
    if (
        design.ndim != 2
        or target.ndim != 1
        or design.shape[0] != len(target)
        or design.shape[1] != expected_columns
        or design.shape[0] < design.shape[1]
        or basis_count != 3
        or support < 1
        or np.any(~np.isfinite(design))
        or np.any(~np.isfinite(target))
        or not math.isfinite(ridge_value)
        or ridge_value < 0.0
    ):
        raise ValueError("MZ Volterra system and ridge must be finite and resolved")
    column_scale = np.linalg.norm(design, axis=0) / math.sqrt(design.shape[0])
    if np.any(~np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        raise ValueError("MZ Volterra design contains unresolved columns")
    normalized = design / column_scale
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    numerical_rank = int(np.linalg.matrix_rank(normalized))
    if ridge_value == 0.0 and numerical_rank != normalized.shape[1]:
        raise ValueError("zero-ridge MZ solve requires full column rank")
    penalty = np.ones(expected_columns, dtype=float)
    penalty[:basis_count] = 0.0
    augmented_design = np.concatenate(
        (
            normalized,
            np.diag(np.sqrt(ridge_value * penalty)),
        ),
        axis=0,
    )
    augmented_target = np.concatenate((target, np.zeros(expected_columns)))
    normalized_coefficients = np.linalg.lstsq(
        augmented_design,
        augmented_target,
        rcond=None,
    )[0]
    coefficients = normalized_coefficients / column_scale
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    return {
        "mean_force_coefficients": coefficients[:basis_count],
        "memory_coefficients": coefficients[basis_count:].reshape(
            support,
            basis_count,
        ),
        "ridge": ridge_value,
        "numerical_rank": float(numerical_rank),
        "condition_number": condition_number,
        "singular_values": singular_values,
        "column_scale": column_scale,
        "fit_uses_held_clone": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def predict_mz_drift(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    scale: dict[str, float],
    mean_force_coefficients: np.ndarray,
    memory_coefficients: np.ndarray,
) -> np.ndarray:
    """Predict exact-generator drift on each clone without crossing boundaries."""

    positions, velocities = _finite_clone_paths(position, velocity)
    mean_force = np.asarray(mean_force_coefficients, dtype=float)
    memory = np.asarray(memory_coefficients, dtype=float)
    if (
        mean_force.shape != (3,)
        or memory.ndim != 2
        or memory.shape[1] != 3
        or memory.shape[0] < 1
        or memory.shape[0] > positions.shape[1]
        or np.any(~np.isfinite(mean_force))
        or np.any(~np.isfinite(memory))
    ):
        raise ValueError("MZ coefficients must be finite and aligned")
    basis = radial_vector_basis(positions, scale)
    jacobian = radial_vector_basis_jacobian(positions, scale)
    jacobian_velocity = np.einsum(
        "ctpbik,ctpk->ctpbi",
        jacobian,
        velocities,
    )
    support = memory.shape[0]
    first = support - 1
    prediction = np.einsum(
        "b,ctpbi->ctpi",
        mean_force,
        basis[:, first:],
    )
    for lag in range(support):
        stop = positions.shape[1] - lag
        prediction -= np.einsum(
            "b,ctpbi->ctpi",
            memory[lag],
            jacobian_velocity[:, first - lag : stop],
        )
    return prediction
