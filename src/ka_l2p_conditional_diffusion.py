"""Microscopic conditional-diffusion diagnostics for the ``L^2 p`` coordinate."""

from __future__ import annotations

import math

import numpy as np


def rademacher_velocity_probes(
    *,
    probe_count: int,
    particle_count: int,
    seed: int,
) -> np.ndarray:
    """Return a reproducible nested sequence of full velocity-space probes."""

    if (
        isinstance(probe_count, bool)
        or not isinstance(probe_count, (int, np.integer))
        or probe_count < 1
    ):
        raise ValueError("probe_count must be a positive integer")
    if (
        isinstance(particle_count, bool)
        or not isinstance(particle_count, (int, np.integer))
        or particle_count < 1
    ):
        raise ValueError("particle_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return np.random.default_rng(seed).choice(
        (-1.0, 1.0),
        size=(int(probe_count), int(particle_count), 3),
    )


def nested_diffusion_estimates(
    directional_responses: np.ndarray,
    *,
    prefix_counts: tuple[int, ...] | list[int] | np.ndarray,
    friction: float,
    temperature: float,
) -> dict[str, np.ndarray | float]:
    """Reduce nested directional responses to ``2 gamma T A A^T`` estimates."""

    responses = np.asarray(directional_responses, dtype=float)
    prefixes = np.asarray(prefix_counts, dtype=int)
    if (
        responses.ndim != 3
        or responses.shape[-1] != 3
        or len(responses) < 1
        or responses.shape[1] < 1
        or np.any(~np.isfinite(responses))
    ):
        raise ValueError("directional_responses must be finite (probes, targets, 3)")
    if (
        prefixes.ndim != 1
        or len(prefixes) < 1
        or np.any(prefixes < 1)
        or np.any(np.diff(prefixes) <= 0)
        or prefixes[-1] > len(responses)
    ):
        raise ValueError("prefix counts must be increasing and fit the probe axis")
    if not math.isfinite(friction) or friction <= 0.0:
        raise ValueError("friction must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    outer = np.einsum("pti,ptj->ptij", responses, responses)
    cumulative = np.cumsum(outer, axis=0)
    scale = 2.0 * friction * temperature
    estimates = np.asarray(
        [scale * cumulative[prefix - 1] / prefix for prefix in prefixes]
    )
    estimates = 0.5 * (estimates + np.swapaxes(estimates, -1, -2))
    return {
        "diffusion_prefixes": estimates,
        "prefix_counts": prefixes,
        "primary_probe_count": float(prefixes[-1]),
        "thermodynamic_claim_allowed": 0.0,
    }


def diffusion_convergence_summary(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Summarize pointwise relative Frobenius differences between Q paths."""

    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (
        candidate.shape != reference.shape
        or candidate.ndim < 2
        or candidate.shape[-2:] != (3, 3)
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(reference))
    ):
        raise ValueError("candidate and reference must be aligned finite 3x3 paths")
    reference_norm = np.linalg.norm(reference, axis=(-2, -1))
    if np.any(reference_norm <= 0.0):
        raise ValueError("reference diffusion matrices must have nonzero norm")
    relative = np.linalg.norm(candidate - reference, axis=(-2, -1)) / reference_norm
    return {
        "relative_frobenius_error": relative,
        "median_relative_frobenius_error": float(np.median(relative)),
        "p95_relative_frobenius_error": float(np.quantile(relative, 0.95)),
        "sample_count": float(relative.size),
        "thermodynamic_claim_allowed": 0.0,
    }
