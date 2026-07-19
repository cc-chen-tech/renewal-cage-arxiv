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
    basis_indices: tuple[int, ...] = (0, 1, 2),
) -> dict[str, object]:
    """Assemble one joint causal MZ regression without crossing clone edges."""

    positions, velocities = _finite_clone_paths(position, velocity)
    accelerations = np.asarray(acceleration, dtype=float)
    indices = tuple(basis_indices)
    if (
        accelerations.shape != positions.shape
        or np.any(~np.isfinite(accelerations))
        or isinstance(support, bool)
        or not isinstance(support, (int, np.integer))
        or support < 1
        or support > positions.shape[1]
        or len(indices) < 1
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or index < 0
            or index > 2
            for index in indices
        )
    ):
        raise ValueError("acceleration and support must align with kernel paths")
    basis = radial_vector_basis(positions, scale)[..., indices, :]
    jacobian = radial_vector_basis_jacobian(positions, scale)[..., indices, :, :]
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
        "basis_count": len(indices),
        "basis_indices": indices,
        "support": int(support),
        "training_clone_count": int(positions.shape[0]),
        "valid_time_count": int(positions.shape[1] - first),
        "particle_count": int(positions.shape[2]),
        "fit_uses_held_clone": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def assemble_mz_correlation_system(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    scale: dict[str, float],
    support: int,
    basis_indices: tuple[int, ...] = (0, 1, 2),
    memory_position: np.ndarray | None = None,
    projection_lag_count: int | None = None,
) -> dict[str, object]:
    """Assemble the finite-basis MZ projected-correlation identity."""

    positions, velocities = _finite_clone_paths(position, velocity)
    memory_positions = positions
    if memory_position is not None:
        memory_positions, memory_velocities = _finite_clone_paths(
            memory_position,
            velocity,
        )
        if memory_velocities.shape != velocities.shape:
            raise ValueError("memory positions must align with kernel paths")
    accelerations = np.asarray(acceleration, dtype=float)
    indices = tuple(basis_indices)
    lag_count = (
        positions.shape[1] - support + 1
        if projection_lag_count is None
        else projection_lag_count
    )
    if (
        accelerations.shape != positions.shape
        or np.any(~np.isfinite(accelerations))
        or isinstance(support, bool)
        or not isinstance(support, (int, np.integer))
        or support < 1
        or support > positions.shape[1]
        or isinstance(lag_count, bool)
        or not isinstance(lag_count, (int, np.integer))
        or lag_count < 1
        or support - 1 + lag_count > positions.shape[1]
        or len(indices) < 1
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or index < 0
            or index > 2
            for index in indices
        )
    ):
        raise ValueError("acceleration and support must align with kernel paths")
    basis = radial_vector_basis(positions, scale)[..., indices, :]
    jacobian = radial_vector_basis_jacobian(memory_positions, scale)[
        ..., indices, :, :
    ]
    jacobian_velocity = np.einsum(
        "ctpbik,ctpk->ctpbi",
        jacobian,
        velocities,
    )
    design_rows = []
    target_rows = []
    time_count = positions.shape[1]
    particle_count = positions.shape[2]
    maximum_projection_lag = support - 1 + int(lag_count) - 1
    for clone in range(positions.shape[0]):
        mean_correlations = []
        memory_correlations = []
        target_correlations = []
        origin_count = time_count - maximum_projection_lag
        scale_factor = float(origin_count * particle_count * 3)
        left = basis[clone, :origin_count]
        for lag in range(maximum_projection_lag + 1):
            mean_correlations.append(
                np.einsum(
                    "opli,opji->lj",
                    left,
                    basis[clone, lag : lag + origin_count],
                )
                / scale_factor
            )
            memory_correlations.append(
                -np.einsum(
                    "opli,opji->lj",
                    left,
                    jacobian_velocity[clone, lag : lag + origin_count],
                )
                / scale_factor
            )
            target_correlations.append(
                np.einsum(
                    "opli,opi->l",
                    left,
                    accelerations[clone, lag : lag + origin_count],
                )
                / scale_factor
            )
        for time_lag in range(
            support - 1,
            support - 1 + int(lag_count),
        ):
            target_rows.append(target_correlations[time_lag])
            design_rows.append(
                np.concatenate(
                    (
                        mean_correlations[time_lag],
                        *(memory_correlations[time_lag - lag] for lag in range(support)),
                    ),
                    axis=1,
                )
            )
    design = np.concatenate(design_rows, axis=0)
    target = np.concatenate(target_rows, axis=0)
    if np.linalg.matrix_rank(design) < 1:
        raise ValueError("MZ correlation design has zero numerical rank")
    return {
        "design": design,
        "target": target,
        "basis_count": len(indices),
        "basis_indices": indices,
        "support": int(support),
        "training_clone_count": int(positions.shape[0]),
        "correlation_lag_count": int(lag_count),
        "particle_count": int(particle_count),
        "correlation_identity": 1.0,
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
        basis_indices = tuple(system["basis_indices"])
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
        or basis_count < 1
        or basis_count > 3
        or len(basis_indices) != basis_count
        or len(set(basis_indices)) != basis_count
        or any(index < 0 or index > 2 for index in basis_indices)
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
        "basis_indices": basis_indices,
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
    basis_indices: tuple[int, ...] | None = None,
    memory_position: np.ndarray | None = None,
) -> np.ndarray:
    """Predict exact-generator drift on each clone without crossing boundaries."""

    positions, velocities = _finite_clone_paths(position, velocity)
    memory_positions = positions
    if memory_position is not None:
        memory_positions, memory_velocities = _finite_clone_paths(
            memory_position,
            velocity,
        )
        if memory_velocities.shape != velocities.shape:
            raise ValueError("memory positions must align with kernel paths")
    mean_force = np.asarray(mean_force_coefficients, dtype=float)
    memory = np.asarray(memory_coefficients, dtype=float)
    indices = (
        tuple(range(len(mean_force)))
        if basis_indices is None
        else tuple(basis_indices)
    )
    if (
        mean_force.ndim != 1
        or len(mean_force) < 1
        or len(mean_force) > 3
        or memory.ndim != 2
        or memory.shape[1] != len(mean_force)
        or memory.shape[0] < 1
        or memory.shape[0] > positions.shape[1]
        or np.any(~np.isfinite(mean_force))
        or np.any(~np.isfinite(memory))
        or len(indices) != len(mean_force)
        or len(set(indices)) != len(indices)
        or any(index < 0 or index > 2 for index in indices)
    ):
        raise ValueError("MZ coefficients must be finite and aligned")
    basis = radial_vector_basis(positions, scale)[..., indices, :]
    jacobian = radial_vector_basis_jacobian(memory_positions, scale)[
        ..., indices, :, :
    ]
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


def select_decay_rates_from_memory(
    memory_coefficients: np.ndarray,
    *,
    frame_time: float,
    decay_grid: np.ndarray,
    rank: int,
) -> dict[str, object]:
    """Select positive exponential poles from a training M2 memory kernel."""

    memory = np.asarray(memory_coefficients, dtype=float)
    rates = np.asarray(decay_grid, dtype=float)
    dt = float(frame_time)
    if (
        memory.ndim != 2
        or memory.shape[0] < 2
        or memory.shape[1] < 1
        or np.any(~np.isfinite(memory))
        or not math.isfinite(dt)
        or dt <= 0.0
        or rates.ndim != 1
        or len(rates) < 1
        or np.any(~np.isfinite(rates))
        or np.any(rates <= 0.0)
        or np.any(np.diff(rates) <= 0.0)
        or isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or rank < 1
        or rank > len(rates)
    ):
        raise ValueError(
            "training memory, frame time, decay grid, and rank must be resolved"
        )
    memory_scale = math.sqrt(float(np.mean(memory**2)))
    if not math.isfinite(memory_scale) or memory_scale <= 0.0:
        raise ValueError("training memory must have positive finite scale")

    time = np.arange(memory.shape[0], dtype=float) * dt
    dictionary = np.exp(-time[:, None] * rates[None, :])
    selected: list[int] = []
    amplitudes = np.empty((0, memory.shape[1]), dtype=float)
    residual_square = float(np.sum(memory**2))
    for _ in range(int(rank)):
        best: tuple[float, int, np.ndarray] | None = None
        for candidate in range(len(rates)):
            if candidate in selected:
                continue
            trial = selected + [candidate]
            trial_amplitudes = np.linalg.lstsq(
                dictionary[:, trial],
                memory,
                rcond=None,
            )[0]
            residual = memory - dictionary[:, trial] @ trial_amplitudes
            trial_square = float(np.sum(residual**2))
            if best is None or trial_square < best[0]:
                best = (trial_square, candidate, trial_amplitudes)
        if best is None:
            raise RuntimeError("positive-pole OMP exhausted its fixed dictionary")
        residual_square, candidate, amplitudes = best
        selected.append(candidate)

    normalized_rmse = math.sqrt(
        residual_square / memory.size
    ) / memory_scale
    return {
        "selected_decay_rates": rates[selected].copy(),
        "exponential_amplitudes": amplitudes,
        "normalized_reconstruction_rmse": normalized_rmse,
        "selection_uses_held_clone": 0.0,
        "positive_decay_grid_only": 1.0,
        "autonomous_single_particle_gle_allowed": 0.0,
        "complete_event_clock_closure_allowed": 0.0,
        "kramers_escape_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


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


def force_autocovariance(
    force: np.ndarray,
    maximum_lag: int,
) -> np.ndarray:
    """Return the component-averaged pathwise force autocovariance."""

    values = np.asarray(force, dtype=float)
    if (
        values.ndim != 4
        or values.shape[0] < 1
        or values.shape[1] < 2
        or values.shape[2] < 1
        or values.shape[3] != 3
        or np.any(~np.isfinite(values))
        or isinstance(maximum_lag, bool)
        or not isinstance(maximum_lag, (int, np.integer))
        or maximum_lag < 0
        or maximum_lag >= values.shape[1]
    ):
        raise ValueError("force paths and covariance lag must be finite and aligned")
    covariance = np.empty(maximum_lag + 1, dtype=float)
    covariance[0] = float(np.mean(values**2))
    for lag in range(1, maximum_lag + 1):
        covariance[lag] = float(np.mean(values[:, lag:] * values[:, :-lag]))
    return covariance


def two_position_fdt_covariance(
    position: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    coupling_coefficients: np.ndarray,
    temperature: float,
    frame_time: float,
    maximum_lag: int,
) -> np.ndarray:
    """Evaluate the positive-Prony second-FDT covariance on an observed path."""

    positions = _finite_vectors(position)
    if positions.ndim != 4:
        raise ValueError("FDT positions must be clone-time-particle vectors")
    rates, _, rho, _ = _validated_decay(decay_rates, frame_time)
    thermal_energy = float(temperature)
    if (
        not math.isfinite(thermal_energy)
        or thermal_energy <= 0.0
        or isinstance(maximum_lag, bool)
        or not isinstance(maximum_lag, (int, np.integer))
        or maximum_lag < 0
        or maximum_lag >= positions.shape[1]
    ):
        raise ValueError("FDT temperature and lag must be finite and physical")
    coupling = _isotropic_radial_coupling(
        positions,
        scale=scale,
        coupling_coefficients=coupling_coefficients,
    )
    if coupling.shape[-3] != len(rates):
        raise ValueError("FDT coupling rank must align with positive poles")
    covariance = np.empty(maximum_lag + 1, dtype=float)
    for lag in range(maximum_lag + 1):
        later = coupling[:, lag:]
        earlier = coupling[:, : positions.shape[1] - lag]
        pole_overlap = np.mean(
            np.sum(later * earlier, axis=(-2, -1)) / 3.0,
            axis=(0, 1, 2),
        )
        covariance[lag] = thermal_energy * float(
            np.sum((rho**lag) * pole_overlap)
        )
    return covariance


def _prepare_linear_memory_statistics(
    mean_features: np.ndarray,
    memory_features: np.ndarray,
    acceleration: np.ndarray,
) -> dict[str, object]:
    mean = np.asarray(mean_features, dtype=float)
    memory = np.asarray(memory_features, dtype=float)
    target_array = np.asarray(acceleration, dtype=float)
    if (
        mean.ndim != 5
        or memory.ndim != 5
        or mean.shape[:3] != memory.shape[:3]
        or mean.shape[-1] != 3
        or memory.shape[-1] != 3
        or target_array.shape != (*mean.shape[:3], 3)
        or mean.shape[-2] < 1
        or memory.shape[-2] < 1
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(memory))
        or np.any(~np.isfinite(target_array))
    ):
        raise ValueError("linear memory features must be finite and aligned")
    features = np.concatenate((mean, memory), axis=-2)
    design = np.moveaxis(features, -1, -2).reshape(-1, features.shape[-2])
    target = target_array.reshape(-1)
    column_scale = np.linalg.norm(design, axis=0) / math.sqrt(len(target))
    if np.any(column_scale <= 0.0) or np.any(~np.isfinite(column_scale)):
        raise ValueError("linear memory feature design has unresolved columns")
    normalized = design / column_scale
    gram = np.einsum("ni,nj->ij", normalized, normalized, optimize=False)
    target_projection = np.einsum("ni,n->i", normalized, target, optimize=False)
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular_values = np.sqrt(eigenvalues[::-1])
    tolerance = (
        max(normalized.shape)
        * np.finfo(float).eps
        * max(float(singular_values[0]), np.finfo(float).tiny)
    )
    rank = int(np.sum(singular_values > tolerance))
    return {
        "normalized_gram": gram,
        "normalized_target_projection": target_projection,
        "target_square": float(np.dot(target, target)),
        "column_scale": column_scale,
        "mean_count": int(mean.shape[-2]),
        "feature_count": int(normalized.shape[1]),
        "row_count": int(normalized.shape[0]),
        "numerical_rank": float(rank),
        "singular_values": singular_values,
    }


def _solve_linear_memory_statistics(
    statistics: dict[str, object],
    *,
    ridge: float,
) -> dict[str, object]:
    gram = np.asarray(statistics.get("normalized_gram"), dtype=float)
    target_projection = np.asarray(
        statistics.get("normalized_target_projection"),
        dtype=float,
    )
    column_scale = np.asarray(statistics.get("column_scale"), dtype=float)
    singular_values = np.asarray(statistics.get("singular_values"), dtype=float)
    try:
        mean_count = int(statistics["mean_count"])
        feature_count = int(statistics["feature_count"])
        rank = int(float(statistics["numerical_rank"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("linear memory statistics are incomplete") from error
    ridge_value = float(ridge)
    if (
        gram.shape != (feature_count, feature_count)
        or target_projection.shape != (feature_count,)
        or column_scale.shape != (feature_count,)
        or singular_values.shape != (feature_count,)
        or mean_count < 1
        or mean_count >= feature_count
        or np.any(~np.isfinite(gram))
        or np.any(~np.isfinite(target_projection))
        or np.any(~np.isfinite(column_scale))
        or np.any(column_scale <= 0.0)
        or np.any(~np.isfinite(singular_values))
        or not math.isfinite(ridge_value)
        or ridge_value < 0.0
    ):
        raise ValueError("linear memory statistics and ridge must be resolved")
    if ridge_value == 0.0 and rank != feature_count:
        raise ValueError("zero-ridge memory fit requires full column rank")
    penalty = np.ones(feature_count)
    penalty[:mean_count] = 0.0
    normalized_coefficients = np.linalg.lstsq(
        gram + np.diag(ridge_value * penalty),
        target_projection,
        rcond=None,
    )[0]
    coefficients = normalized_coefficients / column_scale
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    return {
        "mean_coefficients": coefficients[:mean_count],
        "memory_coefficients": coefficients[mean_count:],
        "condition_number": condition,
        "numerical_rank": float(rank),
        "singular_values": singular_values,
        "ridge": ridge_value,
    }


def _fit_linear_memory_features(
    mean_features: np.ndarray,
    memory_features: np.ndarray,
    acceleration: np.ndarray,
    *,
    ridge: float,
) -> dict[str, object]:
    statistics = _prepare_linear_memory_statistics(
        mean_features,
        memory_features,
        acceleration,
    )
    fitted = _solve_linear_memory_statistics(statistics, ridge=ridge)
    features = np.concatenate((mean_features, memory_features), axis=-2)
    fitted["linear_prediction"] = np.einsum(
        "...fi,f->...i",
        features,
        np.concatenate(
            (fitted["mean_coefficients"], fitted["memory_coefficients"])
        ),
    )
    return fitted


def prepare_real_pole_model(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    frame_time: float,
) -> dict[str, object]:
    """Build one reusable signed real-pole sufficient-statistics system."""

    positions, velocities = _finite_clone_paths(position, velocity)
    target = np.asarray(acceleration, dtype=float)
    if target.shape != positions.shape or np.any(~np.isfinite(target)):
        raise ValueError("real-pole acceleration must be finite and aligned")
    history_result = real_pole_history_features(
        positions,
        velocities,
        scale=scale,
        decay_rates=decay_rates,
        frame_time=frame_time,
    )
    history = np.asarray(history_result["history_features"])
    memory_features = -history.reshape(*history.shape[:3], -1, 3)
    mean_features = radial_vector_basis(positions, scale)
    statistics = _prepare_linear_memory_statistics(
        mean_features,
        memory_features,
        target,
    )
    return {
        "statistics": statistics,
        "pole_count": float(len(np.asarray(decay_rates))),
        "decay_rates": np.asarray(decay_rates, dtype=float).copy(),
        "fit_uses_held_clone": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def solve_prepared_real_pole_model(
    prepared: dict[str, object],
    *,
    ridge: float,
) -> dict[str, object]:
    """Solve one ridge value from a reusable signed real-pole system."""

    try:
        pole_count = int(float(prepared["pole_count"]))
        rates = np.asarray(prepared["decay_rates"], dtype=float)
        statistics = prepared["statistics"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prepared real-pole model is incomplete") from error
    if pole_count < 1 or rates.shape != (pole_count,) or np.any(rates <= 0.0):
        raise ValueError("prepared real-pole model has invalid positive poles")
    fitted = _solve_linear_memory_statistics(statistics, ridge=ridge)
    return {
        "mean_force_coefficients": fitted["mean_coefficients"],
        "pole_coefficients": np.asarray(fitted["memory_coefficients"]).reshape(
            pole_count,
            3,
        ),
        "condition_number": fitted["condition_number"],
        "numerical_rank": fitted["numerical_rank"],
        "ridge": fitted["ridge"],
        "decay_rates": rates.copy(),
        "positive_prony_factorization": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def fit_real_pole_model(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    frame_time: float,
    ridge: float,
) -> dict[str, object]:
    """Fit a signed past-position real-pole MZ realization."""

    prepared = prepare_real_pole_model(
        position,
        velocity,
        acceleration,
        scale=scale,
        decay_rates=decay_rates,
        frame_time=frame_time,
    )
    fitted = solve_prepared_real_pole_model(prepared, ridge=ridge)
    prediction = predict_real_pole_drift(
        position,
        velocity,
        scale=scale,
        decay_rates=decay_rates,
        frame_time=frame_time,
        mean_force_coefficients=fitted["mean_force_coefficients"],
        pole_coefficients=fitted["pole_coefficients"],
    )
    target = np.asarray(acceleration, dtype=float)
    return {
        **fitted,
        "prediction": prediction,
        "training_residual_variance": float(np.mean((target - prediction) ** 2)),
    }


def predict_real_pole_drift(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    frame_time: float,
    mean_force_coefficients: np.ndarray,
    pole_coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted signed past-position real-pole drift."""

    positions, velocities = _finite_clone_paths(position, velocity)
    rates, _, _, _ = _validated_decay(decay_rates, frame_time)
    mean = np.asarray(mean_force_coefficients, dtype=float)
    coefficients = np.asarray(pole_coefficients, dtype=float)
    if (
        mean.shape != (3,)
        or coefficients.shape != (len(rates), 3)
        or np.any(~np.isfinite(mean))
        or np.any(~np.isfinite(coefficients))
    ):
        raise ValueError("real-pole prediction coefficients must be finite and aligned")
    history = np.asarray(
        real_pole_history_features(
            positions,
            velocities,
            scale=scale,
            decay_rates=rates,
            frame_time=frame_time,
        )["history_features"]
    )
    return np.einsum(
        "b,ctpbi->ctpi",
        mean,
        radial_vector_basis(positions, scale),
    ) - np.einsum("ctpabi,ab->ctpi", history, coefficients)


def _radial_coupling_basis(
    position: np.ndarray,
    scale: dict[str, float],
) -> np.ndarray:
    vectors = _finite_vectors(position)
    mu, sigma, epsilon = _validated_scale(scale)
    radial_square = np.sum(vectors**2, axis=-1)
    normalized = (radial_square - mu) / sigma
    identity = np.broadcast_to(np.eye(3), (*vectors.shape[:-1], 3, 3))
    outer = vectors[..., :, None] * vectors[..., None, :]
    radial_projector = outer / np.maximum(radial_square, epsilon)[..., None, None]
    return np.stack(
        (
            identity,
            normalized[..., None, None] * identity,
            radial_projector,
        ),
        axis=-3,
    )


def prepare_two_position_prony_model(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    frame_time: float,
) -> dict[str, object]:
    """Build one reusable symmetric two-position Prony fit system."""

    positions, velocities = _finite_clone_paths(position, velocity)
    target = np.asarray(acceleration, dtype=float)
    rates, _, rho, forcing = _validated_decay(decay_rates, frame_time)
    if target.shape != positions.shape or np.any(~np.isfinite(target)):
        raise ValueError("two-position acceleration must be finite and aligned")
    coupling_basis = _radial_coupling_basis(positions, scale)
    history = np.zeros(
        (*positions.shape[:-1], len(rates), 3, 3),
        dtype=float,
    )
    for time_index in range(positions.shape[1] - 1):
        projected = np.einsum(
            "cpmij,cpj->cpmi",
            np.swapaxes(coupling_basis[:, time_index], -1, -2),
            velocities[:, time_index],
        )
        history[:, time_index + 1] = (
            rho[None, None, :, None, None] * history[:, time_index]
            + forcing[None, None, :, None, None] * projected[:, :, None]
        )
    gram_pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    feature_rows = []
    for pole in range(len(rates)):
        for left, right in gram_pairs:
            feature = np.einsum(
                "ctpij,ctpj->ctpi",
                coupling_basis[..., left, :, :],
                history[..., pole, right, :],
            )
            if left != right:
                feature += np.einsum(
                    "ctpij,ctpj->ctpi",
                    coupling_basis[..., right, :, :],
                    history[..., pole, left, :],
                )
            feature_rows.append(-feature)
    memory_features = np.stack(feature_rows, axis=-2)
    mean_features = radial_vector_basis(positions, scale)
    statistics = _prepare_linear_memory_statistics(
        mean_features,
        memory_features,
        target,
    )
    return {
        "statistics": statistics,
        "decay_rates": rates.copy(),
        "gram_pairs": gram_pairs,
        "fit_uses_held_clone": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def solve_prepared_two_position_prony_model(
    prepared: dict[str, object],
    *,
    ridge: float,
) -> dict[str, object]:
    """Solve and project one ridge value from a reusable Prony system."""

    try:
        rates = np.asarray(prepared["decay_rates"], dtype=float)
        gram_pairs = tuple(tuple(pair) for pair in prepared["gram_pairs"])
        statistics = prepared["statistics"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prepared two-position Prony model is incomplete") from error
    if rates.ndim != 1 or len(rates) < 1 or np.any(rates <= 0.0):
        raise ValueError("prepared two-position Prony model has invalid poles")
    if gram_pairs != ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        raise ValueError("prepared two-position Prony Gram basis is invalid")
    fitted = _solve_linear_memory_statistics(statistics, ridge=ridge)
    raw_grams = np.zeros((len(rates), 3, 3), dtype=float)
    offset = 0
    for pole in range(len(rates)):
        for left, right in gram_pairs:
            value = float(fitted["memory_coefficients"][offset])
            raw_grams[pole, left, right] = value
            raw_grams[pole, right, left] = value
            offset += 1
    projected_grams = np.zeros_like(raw_grams)
    coupling_coefficients = np.zeros((len(rates), 3), dtype=float)
    for pole, gram in enumerate(raw_grams):
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        largest = float(eigenvalues[-1])
        if largest > 0.0:
            coupling_coefficients[pole] = math.sqrt(largest) * eigenvectors[:, -1]
            projected_grams[pole] = np.outer(
                coupling_coefficients[pole],
                coupling_coefficients[pole],
            )
    minimum_projected_eigenvalue = float(
        min(np.min(np.linalg.eigvalsh(gram)) for gram in projected_grams)
    )
    return {
        "mean_force_coefficients": fitted["mean_coefficients"],
        "raw_gram_matrices": raw_grams,
        "projected_gram_matrices": projected_grams,
        "coupling_coefficients": coupling_coefficients,
        "condition_number": fitted["condition_number"],
        "numerical_rank": fitted["numerical_rank"],
        "ridge": fitted["ridge"],
        "decay_rates": rates.copy(),
        "minimum_projected_gram_eigenvalue": minimum_projected_eigenvalue,
        "all_projected_gram_matrices_psd": float(
            minimum_projected_eigenvalue >= -1e-12
        ),
        "positive_prony_factorization": 1.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def fit_two_position_prony_model(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    frame_time: float,
    ridge: float,
) -> dict[str, object]:
    """Fit symmetric pole Grams and project each to a positive rank-one bath."""

    prepared = prepare_two_position_prony_model(
        position,
        velocity,
        acceleration,
        scale=scale,
        decay_rates=decay_rates,
        frame_time=frame_time,
    )
    fitted = solve_prepared_two_position_prony_model(prepared, ridge=ridge)
    prediction = predict_two_position_prony_drift(
        position,
        velocity,
        scale=scale,
        decay_rates=decay_rates,
        coupling_coefficients=fitted["coupling_coefficients"],
        frame_time=frame_time,
        mean_force_coefficients=fitted["mean_force_coefficients"],
    )
    target = np.asarray(acceleration, dtype=float)
    return {
        **fitted,
        "prediction": prediction,
        "training_residual_variance": float(np.mean((target - prediction) ** 2)),
    }


def predict_two_position_prony_drift(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    scale: dict[str, float],
    decay_rates: np.ndarray,
    coupling_coefficients: np.ndarray,
    frame_time: float,
    mean_force_coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate a fitted positive two-position Prony force-memory drift."""

    positions, velocities = _finite_clone_paths(position, velocity)
    mean = np.asarray(mean_force_coefficients, dtype=float)
    if mean.shape != (3,) or np.any(~np.isfinite(mean)):
        raise ValueError("positive-Prony mean-force coefficients must be finite")
    auxiliary = two_position_auxiliary_features(
        positions,
        velocities,
        scale=scale,
        decay_rates=decay_rates,
        coupling_coefficients=coupling_coefficients,
        frame_time=frame_time,
    )
    return np.einsum(
        "b,ctpbi->ctpi",
        mean,
        radial_vector_basis(positions, scale),
    ) + np.asarray(auxiliary["force"])


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

    innovation_applicable = auxiliary_innovation is not None
    fdt_applicable = (
        observed_fdt_covariance is not None and target_fdt_covariance is not None
    )
    if (observed_fdt_covariance is None) != (target_fdt_covariance is None):
        raise ValueError("observed and target FDT arrays are jointly required")
    innovation_correlation = 0.0
    fdt_error = 0.0
    if innovation_applicable:
        innovation = np.asarray(auxiliary_innovation, dtype=float)
        if (
            innovation.ndim != 5
            or innovation.shape[0] != positions.shape[0]
            or innovation.shape[2] != positions.shape[2]
            or innovation.shape[-1] != 3
            or innovation.shape[1] < 2
            or np.any(~np.isfinite(innovation))
        ):
            raise ValueError("auxiliary innovations must be finite and aligned")
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
    if fdt_applicable:
        observed_fdt = np.asarray(observed_fdt_covariance, dtype=float)
        target_fdt = np.asarray(target_fdt_covariance, dtype=float)
        if (
            observed_fdt.shape != target_fdt.shape
            or observed_fdt.ndim != 1
            or len(observed_fdt) < 1
            or np.any(~np.isfinite(observed_fdt))
            or np.any(~np.isfinite(target_fdt))
        ):
            raise ValueError("FDT diagnostics must be finite and aligned")
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
        "auxiliary_diagnostics_applicable": float(
            innovation_applicable or fdt_applicable
        ),
        "auxiliary_innovation_diagnostics_applicable": float(
            innovation_applicable
        ),
        "second_fdt_diagnostics_applicable": float(fdt_applicable),
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
            deterministic_pass = (
                candidate_rmse <= 1.10 * mz_rmse
                and candidate_nll <= mz_nll + 0.05 * scalar_count
                and _finite_row_value(
                    row,
                    "maximum_normalized_resolved_basis_residual_correlation",
                )
                <= 0.20
                and _finite_row_value(row, "all_selected_decay_rates_positive")
                == 1.0
                and _finite_row_value(row, "all_fitted_arrays_finite") == 1.0
            )
            if model == _KERNEL_MODELS[3]:
                deterministic_pass = (
                    deterministic_pass
                    and _finite_row_value(
                        row,
                        "second_fdt_covariance_normalized_rmse",
                    )
                    <= 0.30
                    and _finite_row_value(
                        row,
                        "second_fdt_diagnostics_applicable",
                    )
                    == 1.0
                )
            fold_pass.append(deterministic_pass)
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
        "latent_auxiliary_innovation_identified_in_ka": 0.0,
        "stochastic_auxiliary_bath_identified_in_ka": 0.0,
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
