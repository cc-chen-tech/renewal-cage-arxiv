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
