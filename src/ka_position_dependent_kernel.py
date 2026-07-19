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


def _validated_decay(
    decay_rates: np.ndarray,
    frame_time: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    rates = np.asarray(decay_rates, dtype=float)
    dt = float(frame_time)
    if (
        rates.ndim != 1
        or len(rates) < 1
        or np.any(~np.isfinite(rates))
        or np.any(rates <= 0.0)
        or not math.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("decay rates and frame time must be finite and positive")
    rho = np.exp(-rates * dt)
    integrated_forcing = -np.expm1(-rates * dt) / rates
    return rates, dt, rho, integrated_forcing


def _isotropic_radial_coupling(
    position: np.ndarray,
    *,
    scale: dict[str, float],
    coupling_coefficients: np.ndarray,
) -> np.ndarray:
    vectors = _finite_vectors(position)
    mu, sigma, epsilon = _validated_scale(scale)
    coefficients = np.asarray(coupling_coefficients, dtype=float)
    if (
        coefficients.ndim != 2
        or coefficients.shape[1] != 3
        or coefficients.shape[0] < 1
        or np.any(~np.isfinite(coefficients))
    ):
        raise ValueError("coupling coefficients must be finite rank-by-three values")
    radial_square = np.sum(vectors**2, axis=-1)
    normalized = (radial_square - mu) / sigma
    identity = np.broadcast_to(np.eye(3), (*vectors.shape[:-1], 3, 3))
    outer = vectors[..., :, None] * vectors[..., None, :]
    radial_projector = outer / np.maximum(radial_square, epsilon)[..., None, None]
    return (
        identity[..., None, :, :] * coefficients[:, 0, None, None]
        + normalized[..., None, None, None]
        * identity[..., None, :, :]
        * coefficients[:, 1, None, None]
        + radial_projector[..., None, :, :] * coefficients[:, 2, None, None]
    )


def real_pole_history_features(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    frame_time: float,
) -> dict[str, object]:
    """Build historical MZ features with positive real temporal poles."""

    positions, velocities = _finite_clone_paths(position, velocity)
    rates, _, rho, forcing = _validated_decay(decay_rates, frame_time)
    jacobian = radial_vector_basis_jacobian(positions, scale)
    jacobian_velocity = np.einsum(
        "ctpbik,ctpk->ctpbi",
        jacobian,
        velocities,
    )
    history = np.zeros(
        (*positions.shape[:-1], len(rates), 3, 3),
        dtype=float,
    )
    for time_index in range(positions.shape[1] - 1):
        history[:, time_index + 1] = (
            rho[None, None, :, None, None] * history[:, time_index]
            + forcing[None, None, :, None, None]
            * jacobian_velocity[:, time_index, :, None, :, :]
        )
    return {
        "history_features": history,
        "decay_rates": rates.copy(),
        "positive_real_poles": 1.0,
        "positive_prony_factorization": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def two_position_auxiliary_features(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    coupling_coefficients: np.ndarray,
    frame_time: float,
) -> dict[str, object]:
    """Propagate a positive two-position auxiliary-bath kernel realization."""

    positions, velocities = _finite_clone_paths(position, velocity)
    rates, _, rho, forcing = _validated_decay(decay_rates, frame_time)
    coefficients = np.asarray(coupling_coefficients, dtype=float)
    if coefficients.shape != (len(rates), 3):
        raise ValueError("one radial coupling row is required per decay rate")
    coupling = _isotropic_radial_coupling(
        positions,
        scale=scale,
        coupling_coefficients=coefficients,
    )
    auxiliary = np.zeros(
        (*positions.shape[:-1], len(rates), 3),
        dtype=float,
    )
    force = np.zeros_like(positions)
    for time_index in range(positions.shape[1]):
        force[:, time_index] = np.einsum(
            "cpaij,cpaj->cpi",
            coupling[:, time_index],
            auxiliary[:, time_index],
        )
        if time_index + 1 < positions.shape[1]:
            projected_velocity = np.einsum(
                "cpaij,cpj->cpai",
                np.swapaxes(coupling[:, time_index], -1, -2),
                velocities[:, time_index],
            )
            auxiliary[:, time_index + 1] = (
                rho[None, None, :, None] * auxiliary[:, time_index]
                - forcing[None, None, :, None] * projected_velocity
            )
    return {
        "force": force,
        "auxiliary": auxiliary,
        "coupling": coupling,
        "decay_rates": rates.copy(),
        "positive_real_poles": 1.0,
        "positive_prony_factorization": 1.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def reconstruct_auxiliary_innovations(
    auxiliary: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    coupling_coefficients: np.ndarray,
    frame_time: float,
) -> np.ndarray:
    """Reconstruct discrete innovations after removing exact auxiliary drift."""

    positions, velocities = _finite_clone_paths(position, velocity)
    rates, _, rho, forcing = _validated_decay(decay_rates, frame_time)
    states = np.asarray(auxiliary, dtype=float)
    if (
        states.shape != (*positions.shape[:-1], len(rates), 3)
        or np.any(~np.isfinite(states))
    ):
        raise ValueError("auxiliary states must be finite and path aligned")
    coupling = _isotropic_radial_coupling(
        positions,
        scale=scale,
        coupling_coefficients=coupling_coefficients,
    )
    projected_velocity = np.einsum(
        "ctpaij,ctpj->ctpai",
        np.swapaxes(coupling[:, :-1], -1, -2),
        velocities[:, :-1],
    )
    return (
        states[:, 1:]
        - rho[None, None, None, :, None] * states[:, :-1]
        + forcing[None, None, None, :, None] * projected_velocity
    )


_KERNEL_MODELS = (
    "stationary_scalar_nonparametric_volterra",
    "finite_basis_mz_position_kernel",
    "past_position_real_pole",
    "two_position_positive_prony",
    "time_permuted_position_null",
)

_BROAD_CLAIMS = (
    "autonomous_single_particle_gle_allowed",
    "complete_event_clock_closure_allowed",
    "kramers_escape_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)


def _absolute_correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float).reshape(-1)
    second = np.asarray(right, dtype=float).reshape(-1)
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("correlation arrays must be aligned and nontrivial")
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = math.sqrt(float(np.dot(first, first) * np.dot(second, second)))
    if denominator == 0.0:
        return 0.0
    return abs(float(np.dot(first, second) / denominator))


def held_kernel_diagnostics(
    observed_acceleration: np.ndarray,
    predicted_acceleration: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    scale: dict[str, float],
    training_residual_variance: float,
    maximum_lag: int,
    auxiliary_innovation: np.ndarray | None = None,
    observed_fdt_covariance: np.ndarray | None = None,
    target_fdt_covariance: np.ndarray | None = None,
) -> dict[str, float]:
    """Score one held path using scales and residual variance fixed on training."""

    positions, velocities = _finite_clone_paths(position, velocity)
    observed = np.asarray(observed_acceleration, dtype=float)
    predicted = np.asarray(predicted_acceleration, dtype=float)
    variance = float(training_residual_variance)
    if (
        observed.shape != positions.shape
        or predicted.shape != positions.shape
        or np.any(~np.isfinite(observed))
        or np.any(~np.isfinite(predicted))
        or not math.isfinite(variance)
        or variance <= 0.0
        or isinstance(maximum_lag, bool)
        or not isinstance(maximum_lag, (int, np.integer))
        or maximum_lag < 0
        or maximum_lag >= positions.shape[1]
    ):
        raise ValueError("held paths, training variance, and lag must be finite")
    residual = observed - predicted
    observed_scale = math.sqrt(float(np.mean(observed**2)))
    if observed_scale <= 0.0:
        raise ValueError("held observed acceleration must have positive scale")
    drift_rmse = math.sqrt(float(np.mean(residual**2))) / observed_scale
    drift_nll = 0.5 * residual.size * math.log(2.0 * math.pi * variance) + (
        0.5 * float(np.sum(residual**2)) / variance
    )

    basis = radial_vector_basis(positions, scale)
    jacobian = radial_vector_basis_jacobian(positions, scale)
    jacobian_velocity = np.einsum(
        "ctpbik,ctpk->ctpbi",
        jacobian,
        velocities,
    )
    resolved_correlations = []
    for lag in range(maximum_lag + 1):
        later_residual = residual[:, lag:]
        stop = positions.shape[1] - lag
        for feature in (basis[:, :stop], jacobian_velocity[:, :stop]):
            for basis_index in range(feature.shape[-2]):
                resolved_correlations.append(
                    _absolute_correlation(
                        later_residual,
                        feature[..., basis_index, :],
                    )
                )

    auxiliary_items = (
        auxiliary_innovation,
        observed_fdt_covariance,
        target_fdt_covariance,
    )
    auxiliary_applicable = all(item is not None for item in auxiliary_items)
    if any(item is not None for item in auxiliary_items) and not auxiliary_applicable:
        raise ValueError("auxiliary innovation and both FDT arrays are jointly required")
    innovation_correlation = 0.0
    fdt_error = 0.0
    if auxiliary_applicable:
        innovation = np.asarray(auxiliary_innovation, dtype=float)
        observed_fdt = np.asarray(observed_fdt_covariance, dtype=float)
        target_fdt = np.asarray(target_fdt_covariance, dtype=float)
        if (
            innovation.ndim != 5
            or innovation.shape[0] != positions.shape[0]
            or innovation.shape[2] != positions.shape[2]
            or innovation.shape[-1] != 3
            or innovation.shape[1] < 2
            or np.any(~np.isfinite(innovation))
            or observed_fdt.shape != target_fdt.shape
            or observed_fdt.ndim != 1
            or len(observed_fdt) < 1
            or np.any(~np.isfinite(observed_fdt))
            or np.any(~np.isfinite(target_fdt))
        ):
            raise ValueError("auxiliary and FDT diagnostics must be finite and aligned")
        innovation_correlations = []
        active_lag = min(maximum_lag, innovation.shape[1] - 1)
        for lag in range(1, active_lag + 1):
            for mode in range(innovation.shape[-2]):
                innovation_correlations.append(
                    _absolute_correlation(
                        innovation[:, lag:, :, mode],
                        innovation[:, :-lag, :, mode],
                    )
                )
        innovation_correlation = max(innovation_correlations, default=0.0)
        target_scale = math.sqrt(float(np.mean(target_fdt**2)))
        if target_scale <= 0.0:
            raise ValueError("FDT target must have positive scale")
        fdt_error = math.sqrt(float(np.mean((observed_fdt - target_fdt) ** 2))) / (
            target_scale
        )
    return {
        "drift_rmse": drift_rmse,
        "drift_nll": drift_nll,
        "held_scalar_component_count": float(residual.size),
        "maximum_normalized_resolved_basis_residual_correlation": max(
            resolved_correlations,
            default=0.0,
        ),
        "maximum_normalized_auxiliary_innovation_autocorrelation": (
            innovation_correlation
        ),
        "second_fdt_covariance_normalized_rmse": fdt_error,
        "auxiliary_diagnostics_applicable": float(auxiliary_applicable),
        "thermodynamic_claim_allowed": 0.0,
    }


def _finite_row_value(row: dict[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"kernel gate row lacks finite {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"kernel gate row lacks finite {key}")
    return value


def classify_position_dependent_kernel_gate(
    rows: list[dict[str, object]],
) -> dict[str, float | str]:
    """Classify the exact four-fold frozen kernel model grid."""

    expected = {(clone, model) for clone in range(1, 5) for model in _KERNEL_MODELS}
    indexed: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        clone_value = _finite_row_value(row, "held_clone_index")
        clone = int(clone_value)
        model = str(row.get("model", ""))
        key = (clone, model)
        if clone_value != clone or key not in expected or key in indexed:
            raise ValueError("kernel gate rows must form one exact four-fold model grid")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("kernel gate rows must form one exact four-fold model grid")

    ratios = []
    mz_fold_pass = []
    for clone in range(1, 5):
        scalar = indexed[(clone, _KERNEL_MODELS[0])]
        mz = indexed[(clone, _KERNEL_MODELS[1])]
        permuted = indexed[(clone, _KERNEL_MODELS[4])]
        scalar_rmse = _finite_row_value(scalar, "drift_rmse")
        mz_rmse = _finite_row_value(mz, "drift_rmse")
        permuted_rmse = _finite_row_value(permuted, "drift_rmse")
        scalar_nll = _finite_row_value(scalar, "drift_nll")
        mz_nll = _finite_row_value(mz, "drift_nll")
        permuted_nll = _finite_row_value(permuted, "drift_nll")
        if scalar_rmse <= 0.0:
            raise ValueError("stationary scalar RMSE must be positive")
        ratio = mz_rmse / scalar_rmse
        ratios.append(ratio)
        mz_fold_pass.append(
            ratio <= 0.90
            and mz_nll < scalar_nll
            and mz_rmse < permuted_rmse
            and mz_nll < permuted_nll
            and _finite_row_value(
                mz,
                "maximum_normalized_resolved_basis_residual_correlation",
            )
            <= 0.20
            and _finite_row_value(mz, "all_fitted_arrays_finite") == 1.0
        )
    ratio_array = np.asarray(ratios)
    ratio_mean = float(np.mean(ratio_array))
    ratio_se = float(np.std(ratio_array, ddof=1) / math.sqrt(len(ratio_array)))
    t95 = 3.182446305284263
    ratio_low = ratio_mean - t95 * ratio_se
    ratio_high = ratio_mean + t95 * ratio_se
    mz_identified = all(mz_fold_pass) and ratio_high < 1.0

    def realization_pass(model: str) -> tuple[bool, list[int]]:
        fold_pass = []
        ranks = []
        for clone in range(1, 5):
            mz = indexed[(clone, _KERNEL_MODELS[1])]
            row = indexed[(clone, model)]
            mz_rmse = _finite_row_value(mz, "drift_rmse")
            candidate_rmse = _finite_row_value(row, "drift_rmse")
            mz_nll = _finite_row_value(mz, "drift_nll")
            candidate_nll = _finite_row_value(row, "drift_nll")
            scalar_count = _finite_row_value(
                row,
                "held_scalar_component_count",
            )
            rank_value = _finite_row_value(row, "selected_auxiliary_rank")
            rank = int(rank_value)
            if scalar_count < 1.0 or rank_value != rank or rank < 1:
                raise ValueError("realization rows require positive count and rank")
            ranks.append(rank)
            fold_pass.append(
                candidate_rmse <= 1.10 * mz_rmse
                and candidate_nll <= mz_nll + 0.05 * scalar_count
                and _finite_row_value(
                    row,
                    "maximum_normalized_resolved_basis_residual_correlation",
                )
                <= 0.20
                and _finite_row_value(
                    row,
                    "maximum_normalized_auxiliary_innovation_autocorrelation",
                )
                <= 0.20
                and _finite_row_value(
                    row,
                    "second_fdt_covariance_normalized_rmse",
                )
                <= 0.30
                and _finite_row_value(row, "auxiliary_diagnostics_applicable")
                == 1.0
                and _finite_row_value(row, "all_selected_decay_rates_positive")
                == 1.0
                and _finite_row_value(row, "all_fitted_arrays_finite") == 1.0
            )
        return bool(mz_identified and all(fold_pass)), ranks

    real_pole_identified, _ = realization_pass(_KERNEL_MODELS[2])
    positive_prony_identified, positive_ranks = realization_pass(_KERNEL_MODELS[3])
    rank_identified = positive_prony_identified and len(set(positive_ranks)) == 1
    oscillatory_authorized = (
        mz_identified
        and not real_pole_identified
        and not positive_prony_identified
    )
    return {
        "record": "verdict",
        "real_ka_kernel_identifiability_test_required": 1.0,
        "real_ka_position_dependent_mz_kernel_identified": float(mz_identified),
        "past_position_real_pole_identified_in_ka": float(real_pole_identified),
        "two_position_positive_prony_identified_in_ka": float(
            positive_prony_identified
        ),
        "positive_prony_kernel_identified_in_ka": float(
            positive_prony_identified
        ),
        "finite_auxiliary_rank_identified_in_ka": float(rank_identified),
        "selected_auxiliary_rank": float(positive_ranks[0])
        if rank_identified
        else 0.0,
        "oscillatory_matrix_bath_authorized": float(oscillatory_authorized),
        "mz_rmse_ratio_mean": ratio_mean,
        "mz_rmse_ratio_standard_error": ratio_se,
        "mz_rmse_ratio_t95_low": ratio_low,
        "mz_rmse_ratio_t95_high": ratio_high,
        **{claim: 0.0 for claim in _BROAD_CLAIMS},
    }
