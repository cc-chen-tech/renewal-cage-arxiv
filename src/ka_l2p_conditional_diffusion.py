"""Microscopic conditional-diffusion diagnostics for the ``L^2 p`` coordinate."""

from __future__ import annotations

import math

import numpy as np


def deterministic_conditional_diffusion(
    velocity_jacobian: np.ndarray,
    *,
    friction: float,
    temperature: float,
) -> dict[str, np.ndarray | float | str]:
    """Return ``2 gamma T A A^T`` from the full microscopic matrix ``A``."""

    jacobian = np.asarray(velocity_jacobian, dtype=float)
    if (
        jacobian.ndim != 4
        or jacobian.shape[1] != 3
        or jacobian.shape[-1] != 3
        or len(jacobian) < 1
        or jacobian.shape[2] < 1
        or np.any(~np.isfinite(jacobian))
    ):
        raise ValueError(
            "velocity_jacobian must have finite shape (targets, 3, particles, 3)"
        )
    if not math.isfinite(friction) or friction <= 0.0:
        raise ValueError("friction must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    diffusion = 2.0 * friction * temperature * np.einsum(
        "tanb,tcnb->tac", jacobian, jacobian
    )
    diffusion = 0.5 * (diffusion + np.swapaxes(diffusion, -1, -2))
    return {
        "conditional_diffusion": diffusion,
        "estimator": "deterministic_velocity_jacobian",
        "thermodynamic_claim_allowed": 0.0,
    }


def classify_deterministic_numerical_canary(
    *,
    a_primary_reference_error: np.ndarray,
    q_primary_reference_error: np.ndarray,
    directional_response_error: np.ndarray,
    a_coarse_reference_error: np.ndarray,
    q_coarse_reference_error: np.ndarray,
    q_minimum_eigenvalue: np.ndarray,
    q_trace: np.ndarray,
) -> dict[str, float | str]:
    """Apply the frozen numerical-only gate before held closure scoring."""

    arrays = {
        "a_primary_reference_error": np.asarray(a_primary_reference_error, dtype=float),
        "q_primary_reference_error": np.asarray(q_primary_reference_error, dtype=float),
        "directional_response_error": np.asarray(directional_response_error, dtype=float),
        "a_coarse_reference_error": np.asarray(a_coarse_reference_error, dtype=float),
        "q_coarse_reference_error": np.asarray(q_coarse_reference_error, dtype=float),
        "q_minimum_eigenvalue": np.asarray(q_minimum_eigenvalue, dtype=float),
        "q_trace": np.asarray(q_trace, dtype=float),
    }
    if any(
        value.ndim != 1 or not len(value) or np.any(~np.isfinite(value))
        for value in arrays.values()
    ):
        raise ValueError("deterministic canary inputs must be finite nonempty vectors")
    if any(
        np.any(arrays[name] < 0.0)
        for name in (
            "a_primary_reference_error",
            "q_primary_reference_error",
            "directional_response_error",
            "a_coarse_reference_error",
            "q_coarse_reference_error",
        )
    ):
        raise ValueError("deterministic canary errors must be nonnegative")
    if arrays["q_minimum_eigenvalue"].shape != arrays["q_trace"].shape:
        raise ValueError("Q eigenvalue and trace vectors must align")

    def distribution(name: str) -> tuple[float, float]:
        values = arrays[name]
        return float(np.median(values)), float(np.quantile(values, 0.95))

    a_median, a_p95 = distribution("a_primary_reference_error")
    q_median, q_p95 = distribution("q_primary_reference_error")
    directional_median, directional_p95 = distribution("directional_response_error")
    coarse_a_median, _ = distribution("a_coarse_reference_error")
    coarse_q_median, _ = distribution("q_coarse_reference_error")
    a_pass = a_median <= 0.02 and a_p95 <= 0.10
    q_pass = q_median <= 0.02 and q_p95 <= 0.10
    directional_pass = directional_median <= 0.02 and directional_p95 <= 0.10
    monotonic_pass = a_median <= coarse_a_median and q_median <= coarse_q_median
    psd_tolerance = 1e-10 * np.maximum(arrays["q_trace"], 1.0)
    psd_pass = bool(
        np.all(arrays["q_minimum_eigenvalue"] >= -psd_tolerance)
    )
    passed = a_pass and q_pass and directional_pass and monotonic_pass and psd_pass
    return {
        "numerical_state": (
            "deterministic_jacobian_numerically_resolved"
            if passed
            else "deterministic_jacobian_numerically_unresolved"
        ),
        "deterministic_numerical_gate_pass": float(passed),
        "a_step_gate_pass": float(a_pass),
        "q_step_gate_pass": float(q_pass),
        "directional_identity_gate_pass": float(directional_pass),
        "step_monotonicity_gate_pass": float(monotonic_pass),
        "positive_semidefinite_gate_pass": float(psd_pass),
        "a_primary_reference_median_relative_error": a_median,
        "a_primary_reference_p95_relative_error": a_p95,
        "q_primary_reference_median_relative_error": q_median,
        "q_primary_reference_p95_relative_error": q_p95,
        "directional_median_relative_error": directional_median,
        "directional_p95_relative_error": directional_p95,
        "a_coarse_reference_median_relative_error": coarse_a_median,
        "q_coarse_reference_median_relative_error": coarse_q_median,
        "microscopic_environment_coordinate_z_allowed": 0.0,
        "continuous_gaussian_langevin_bath_allowed": 0.0,
        "autonomous_single_particle_gle_allowed": 0.0,
        "complete_event_clock_closure_allowed": 0.0,
        "kramers_escape_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


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


def _validate_residual_covariance_paths(
    residual: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(residual, dtype=float)
    matrices = np.asarray(covariance, dtype=float)
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or matrices.shape != (*values.shape, 3)
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(matrices))
        or not np.allclose(matrices, np.swapaxes(matrices, -1, -2), atol=1e-12)
    ):
        raise ValueError("residual and covariance must be aligned finite 3-vector paths")
    return values, matrices


def fit_constant_covariance(residual: np.ndarray) -> np.ndarray:
    """Fit one full constant covariance to training residuals."""

    values = np.asarray(residual, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 3 or np.any(~np.isfinite(values)):
        raise ValueError("residual must be a finite (time, targets, 3) path")
    flat = values.reshape(-1, 3)
    covariance = flat.T @ flat / len(flat)
    covariance = 0.5 * (covariance + covariance.T)
    floor = max(float(np.trace(covariance)) / 3.0, 1e-30) * 1e-12
    minimum = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum < floor:
        covariance += (floor - minimum) * np.eye(3)
    return covariance


def _conditional_gaussian_nll_from_eigensystem(
    projected_residual: np.ndarray,
    eigenvalues: np.ndarray,
    scale: float,
    isotropic_floor: float,
) -> float:
    variance = scale * eigenvalues + isotropic_floor
    reference = max(float(np.mean(eigenvalues)), 1e-30)
    if scale < 0.0 or isotropic_floor < 0.0 or np.any(variance <= 1e-12 * reference):
        return math.inf
    return float(
        0.5
        * np.mean(
            np.sum(
                np.log(2.0 * math.pi * variance)
                + projected_residual**2 / variance,
                axis=1,
            )
        )
    )


def fit_scaled_conditional_covariance(
    residual: np.ndarray,
    conditional_covariance: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Fit nonnegative ``a Q + delta I`` by training Gaussian likelihood."""

    values, matrices = _validate_residual_covariance_paths(
        residual, conditional_covariance
    )
    flat_values = values.reshape(-1, 3)
    flat_matrices = matrices.reshape(-1, 3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(flat_matrices)
    if np.any(
        eigenvalues
        < -1e-10 * np.maximum(np.max(eigenvalues, axis=1), 1.0)[:, None]
    ):
        raise ValueError("conditional covariance must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = np.einsum("nji,nj->ni", eigenvectors, flat_values)

    observed_outer = np.einsum("ni,nj->nij", flat_values, flat_values)
    x11 = float(np.einsum("nij,nij->", flat_matrices, flat_matrices))
    x12 = float(np.sum(np.trace(flat_matrices, axis1=1, axis2=2)))
    x22 = float(3 * len(flat_values))
    y1 = float(np.einsum("nij,nij->", flat_matrices, observed_outer))
    y2 = float(np.sum(flat_values**2))
    design = np.array([[x11, x12], [x12, x22]])
    response = np.array([y1, y2])
    candidates = [(0.0, max(y2 / x22, 0.0)), (max(y1 / max(x11, 1e-30), 0.0), 0.0)]
    if np.linalg.cond(design) < 1e14:
        unconstrained = np.linalg.solve(design, response)
        candidates.append((max(float(unconstrained[0]), 0.0), max(float(unconstrained[1]), 0.0)))

    def objective(pair: tuple[float, float]) -> float:
        return _conditional_gaussian_nll_from_eigensystem(
            projected, eigenvalues, pair[0], pair[1]
        )

    current = min(candidates, key=objective)
    current_value = objective(current)
    residual_variance = max(float(np.mean(flat_values**2)), 1e-30)
    q_variance = max(float(np.mean(eigenvalues)), 1e-30)
    step_scale = max(current[0], residual_variance / q_variance) * 0.5
    step_floor = max(current[1], residual_variance) * 0.5
    for _ in range(100):
        trial_pairs = []
        for scale_direction in (-1.0, 0.0, 1.0):
            for floor_direction in (-1.0, 0.0, 1.0):
                if scale_direction == 0.0 and floor_direction == 0.0:
                    continue
                trial_pairs.append(
                    (
                        max(current[0] + scale_direction * step_scale, 0.0),
                        max(current[1] + floor_direction * step_floor, 0.0),
                    )
                )
        best = min(trial_pairs, key=objective)
        best_value = objective(best)
        if best_value + 1e-12 < current_value:
            current, current_value = best, best_value
        else:
            step_scale *= 0.55
            step_floor *= 0.55
        if step_scale <= 1e-8 * max(current[0], residual_variance / q_variance) and step_floor <= 1e-8 * max(current[1], residual_variance):
            break
    mean_conditional_variance = current[0] * float(np.mean(eigenvalues))
    floor_fraction = current[1] / max(mean_conditional_variance + current[1], 1e-30)
    return {
        "scale": float(current[0]),
        "isotropic_floor": float(current[1]),
        "negative_log_likelihood": float(current_value),
        "isotropic_floor_variance_fraction": float(floor_fraction),
        "fit_uses_held_samples": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def _maximum_lag_cross_correlation(values: np.ndarray, maximum_lag: int) -> float:
    maximum = 0.0
    for lag in range(1, maximum_lag + 1):
        left = values[lag:].reshape(-1, values.shape[-1])
        right = values[:-lag].reshape(-1, values.shape[-1])
        left -= np.mean(left, axis=0, keepdims=True)
        right -= np.mean(right, axis=0, keepdims=True)
        denominator = np.sqrt(
            np.sum(left**2, axis=0)[:, None] * np.sum(right**2, axis=0)[None, :]
        )
        correlation = np.divide(
            left.T @ right,
            denominator,
            out=np.zeros((values.shape[-1], values.shape[-1])),
            where=denominator > 0.0,
        )
        maximum = max(maximum, float(np.max(np.abs(correlation))))
    return maximum


def _gaussian_energy_distance(values: np.ndarray, *, seed: int, sample_count: int = 512) -> float:
    flat = values.reshape(-1, values.shape[-1])
    count = min(sample_count, len(flat))
    rng = np.random.default_rng(seed)
    selected = flat[rng.choice(len(flat), size=count, replace=False)]
    reference = rng.normal(size=(count, values.shape[-1]))

    def mean_pair_distance(left: np.ndarray, right: np.ndarray) -> float:
        squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        return float(np.mean(np.sqrt(squared)))

    return (
        2.0 * mean_pair_distance(selected, reference)
        - mean_pair_distance(selected, selected)
        - mean_pair_distance(reference, reference)
    )


def conditional_covariance_diagnostic(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    maximum_lag: int,
    gaussian_seed: int,
) -> dict[str, np.ndarray | float]:
    """Whiten held vector innovations and report Gaussian closure diagnostics."""

    values, matrices = _validate_residual_covariance_paths(residual, covariance)
    if maximum_lag < 1 or len(values) <= maximum_lag:
        raise ValueError("maximum_lag must fit the residual path")
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    scale = np.maximum(np.max(eigenvalues, axis=-1, keepdims=True), 1e-30)
    if np.any(eigenvalues <= 1e-12 * scale):
        raise ValueError("held covariance matrices must be positive definite")
    projected = np.einsum("...ji,...j->...i", eigenvectors, values)
    whitened_eigenbasis = projected / np.sqrt(eigenvalues)
    whitened = np.einsum("...ij,...j->...i", eigenvectors, whitened_eigenbasis)
    nll = float(
        0.5
        * np.mean(
            np.sum(
                np.log(2.0 * math.pi * eigenvalues)
                + whitened_eigenbasis**2,
                axis=-1,
            )
        )
    )
    flat = whitened.reshape(-1, 3)
    mean = np.mean(flat, axis=0)
    covariance_error = flat.T @ flat / len(flat) - np.eye(3)
    centered = flat - mean
    variance = np.mean(centered**2, axis=0)
    excess = np.mean(centered**4, axis=0) / variance**2 - 3.0
    squared = whitened**2
    return {
        "negative_log_likelihood": nll,
        "mean_squared_mahalanobis_per_dimension": float(np.mean(flat**2)),
        "maximum_absolute_whitened_mean": float(np.max(np.abs(mean))),
        "maximum_absolute_whitened_covariance_error": float(
            np.max(np.abs(covariance_error))
        ),
        "maximum_absolute_component_excess_kurtosis": float(np.max(np.abs(excess))),
        "maximum_absolute_whitened_correlation": _maximum_lag_cross_correlation(
            whitened, maximum_lag
        ),
        "maximum_absolute_squared_whitened_correlation": _maximum_lag_cross_correlation(
            squared, maximum_lag
        ),
        "gaussian_energy_distance": _gaussian_energy_distance(
            whitened, seed=gaussian_seed
        ),
        "sample_count": float(len(flat)),
        "thermodynamic_claim_allowed": 0.0,
    }


def replicate_first_t_interval(values: np.ndarray) -> dict[str, float]:
    """Return the frozen four-replicate Student-t confidence interval."""

    samples = np.asarray(values, dtype=float)
    if samples.shape != (4,) or np.any(~np.isfinite(samples)):
        raise ValueError("replicate-first inference requires exactly four finite clones")
    mean = float(np.mean(samples))
    standard_error = float(np.std(samples, ddof=1) / math.sqrt(4.0))
    critical = 3.182446305284263
    return {
        "mean": mean,
        "standard_error": standard_error,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "replicate_count": 4.0,
        "thermodynamic_claim_allowed": 0.0,
    }
