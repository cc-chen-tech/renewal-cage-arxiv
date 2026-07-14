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


def smooth_force_support_cage_batch(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    target_batch_size: int = 16,
) -> dict[str, np.ndarray | float]:
    """Evaluate exact smooth-cage vector coordinates for fixed targets."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    targets_raw = np.asarray(target_indices)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or velocities.shape != positions.shape
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(velocities))
    ):
        raise ValueError("positions and velocities must be aligned finite (particles, 3) arrays")
    if particle_types.shape != (len(positions),) or np.any(
        (particle_types < 0) | (particle_types > 1)
    ):
        raise ValueError("particle_types must be aligned KA 0/1 labels")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(
        box_lengths <= 0.0
    ):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if (
        targets_raw.ndim != 1
        or len(targets_raw) < 1
        or np.any(targets_raw != targets_raw.astype(int))
    ):
        raise ValueError("target_indices must be a nonempty integer vector")
    targets = targets_raw.astype(int)
    if (
        np.any(targets < 0)
        or np.any(targets >= len(positions))
        or len(np.unique(targets)) != len(targets)
    ):
        raise ValueError("target_indices must select distinct particles")
    if (
        isinstance(target_batch_size, bool)
        or not isinstance(target_batch_size, (int, np.integer))
        or target_batch_size < 1
    ):
        raise ValueError("target_batch_size must be a positive integer")

    relative_position = np.empty((len(targets), 3), dtype=float)
    relative_velocity = np.empty_like(relative_position)
    jacobian_gram = np.empty((len(targets), 3, 3), dtype=float)
    total_weight = np.empty(len(targets), dtype=float)
    support_count = np.empty(len(targets), dtype=int)
    identity = np.eye(3)
    for start in range(0, len(targets), int(target_batch_size)):
        stop = min(start + int(target_batch_size), len(targets))
        selected = targets[start:stop]
        displacement = positions[None, :, :] - positions[selected, None, :]
        displacement -= box_lengths[None, None, :] * np.rint(
            displacement / box_lengths[None, None, :]
        )
        distance = np.linalg.norm(displacement, axis=2)
        support_radius = _CUTOFF_SCALE * _SIGMA[
            particle_types[selected, None], particle_types[None, :]
        ]
        weight, scaled_derivative = wendland_c4_weight(distance / support_radius)
        local_axis = np.arange(len(selected))
        weight[local_axis, selected] = 0.0
        scaled_derivative[local_axis, selected] = 0.0
        chunk_weight = np.sum(weight, axis=1)
        if np.any(chunk_weight <= 0.0):
            raise ValueError("a target particle has no neighbor inside the KA force support")
        mean_offset = np.einsum("tn,tna->ta", weight, displacement) / chunk_weight[:, None]
        unit = displacement / np.maximum(distance[:, :, None], 1e-12)
        radial_gradient = (
            scaled_derivative[:, :, None]
            * unit
            / support_radius[:, :, None]
        )
        centered = displacement - mean_offset[:, None, :]
        centered_gradient = np.einsum(
            "tna,tnb->tab", centered, radial_gradient
        )
        target_block = identity[None, :, :] + centered_gradient / chunk_weight[:, None, None]
        velocity_difference = velocities[selected, None, :] - velocities[None, :, :]
        projected_velocity = np.einsum(
            "tn,tnb->tb", weight, velocity_difference
        )
        projected_velocity += np.einsum(
            "tna,tn->ta",
            centered,
            np.einsum("tnb,tnb->tn", radial_gradient, velocity_difference),
        )
        projected_velocity /= chunk_weight[:, None]
        gram = np.einsum("tac,tbc->tab", target_block, target_block)
        weight_squared = np.sum(weight**2, axis=1)
        weighted_cross = np.einsum(
            "tn,tna,tnb->tab", weight, centered, radial_gradient
        )
        gradient_squared = np.einsum("tna,tna->tn", radial_gradient, radial_gradient)
        neighbor_gram = weight_squared[:, None, None] * identity[None, :, :]
        neighbor_gram += weighted_cross + np.transpose(weighted_cross, (0, 2, 1))
        neighbor_gram += np.einsum(
            "tn,tna,tnb->tab", gradient_squared, centered, centered
        )
        gram += neighbor_gram / chunk_weight[:, None, None] ** 2
        gram = 0.5 * (gram + np.transpose(gram, (0, 2, 1)))
        eigenvalues = np.linalg.eigvalsh(gram)
        if np.any(eigenvalues[:, 0] <= 0.0):
            raise ValueError("smooth cage Jacobian Gram matrices must be positive definite")

        relative_position[start:stop] = -mean_offset
        relative_velocity[start:stop] = projected_velocity
        jacobian_gram[start:stop] = gram
        total_weight[start:stop] = chunk_weight
        support_count[start:stop] = np.sum(weight > 0.0, axis=1)

    return {
        "relative_position": relative_position,
        "relative_velocity": relative_velocity,
        "jacobian_gram": jacobian_gram,
        "total_weight": total_weight,
        "support_count": support_count,
        "target_indices": targets,
        "thermodynamic_claim_allowed": 0.0,
    }


def smooth_cage_projected_observables(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    forces: np.ndarray | None = None,
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
    if forces is not None:
        forces = np.asarray(forces, dtype=float)
        if forces.shape != positions.shape or np.any(~np.isfinite(forces)):
            raise ValueError("forces must be finite and align with positions")
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
    if forces is None:
        active = np.flatnonzero(np.any(np.abs(jacobian) > 0.0, axis=(1, 2)))
        active_force, _ = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=active,
            potential_protocol=potential_protocol,
        )
        force_drift = np.einsum("nab,nb->a", jacobian[active], active_force)
    else:
        force_drift = np.einsum("nab,nb->a", jacobian, forces)

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


def smooth_cage_invariant_features(
    observable: dict[str, np.ndarray | float],
) -> dict[str, np.ndarray]:
    """Map one projected cage state to fixed rotationally invariant features."""

    relative_position = np.asarray(observable["relative_position"], dtype=float)
    relative_velocity = np.asarray(observable["relative_velocity"], dtype=float)
    projected_drift = np.asarray(observable["projected_drift"], dtype=float)
    gram = np.asarray(observable["jacobian_gram"], dtype=float)
    for name, vector in (
        ("relative_position", relative_position),
        ("relative_velocity", relative_velocity),
        ("projected_drift", projected_drift),
    ):
        if vector.shape != (3,) or np.any(~np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite three-vector")
    if gram.shape != (3, 3) or np.any(~np.isfinite(gram)) or not np.allclose(
        gram, gram.T, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("jacobian_gram must be a finite symmetric 3x3 matrix")
    eigenvalues = np.linalg.eigvalsh(gram)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("jacobian_gram must be positive definite")

    floor = np.finfo(float).tiny
    position_squared = float(relative_position @ relative_position)
    velocity_squared = float(relative_velocity @ relative_velocity)
    drift_squared = float(projected_drift @ projected_drift)

    def cosine(left: np.ndarray, right: np.ndarray, norm_product: float) -> float:
        return float(left @ right) / math.sqrt(max(norm_product, floor))

    log_position = math.log(max(position_squared, floor))
    log_velocity = math.log(max(velocity_squared, floor))
    log_drift = math.log(max(drift_squared, floor))
    cosine_position_velocity = cosine(
        relative_position, relative_velocity, position_squared * velocity_squared
    )
    cosine_position_drift = cosine(
        relative_position, projected_drift, position_squared * drift_squared
    )
    cosine_velocity_drift = cosine(
        relative_velocity, projected_drift, velocity_squared * drift_squared
    )
    log_eigenvalues = np.log(eigenvalues)
    geometry = np.array([log_position, *log_eigenvalues], dtype=float)
    kinematic = np.array(
        [log_position, log_velocity, cosine_position_velocity, *log_eigenvalues],
        dtype=float,
    )
    full = np.array(
        [
            log_position,
            log_velocity,
            cosine_position_velocity,
            *log_eigenvalues,
            log_drift,
            cosine_position_drift,
            cosine_velocity_drift,
        ],
        dtype=float,
    )
    return {"geometry": geometry, "kinematic": kinematic, "full": full}


def smooth_cage_geometry_features(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Return the configuration-only part of the projected cage state."""

    target_indices = np.asarray(target_indices)
    if (
        target_indices.ndim != 1
        or len(target_indices) < 1
        or np.any(target_indices != target_indices.astype(int))
    ):
        raise ValueError("target_indices must be a nonempty integer vector")
    target_indices = target_indices.astype(int)
    if len(np.unique(target_indices)) != len(target_indices):
        raise ValueError("target_indices must select distinct particles")

    floor = np.finfo(float).tiny
    rows: list[np.ndarray] = []
    for target_index in target_indices:
        coordinate = smooth_force_support_cage(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_index=int(target_index),
        )
        relative_position = np.asarray(coordinate["relative_position"], dtype=float)
        jacobian = np.asarray(coordinate["jacobian"], dtype=float)
        gram = np.einsum("nab,ncb->ac", jacobian, jacobian)
        gram = 0.5 * (gram + gram.T)
        eigenvalues = np.linalg.eigvalsh(gram)
        if np.min(eigenvalues) <= 0.0:
            raise ValueError("jacobian_gram must be positive definite")
        position_squared = float(relative_position @ relative_position)
        rows.append(
            np.array(
                [math.log(max(position_squared, floor)), *np.log(eigenvalues)],
                dtype=float,
            )
        )
    return np.asarray(rows)


def grouped_exponential_escape_diagnostic(
    features: np.ndarray,
    first_passage: np.ndarray,
    escaped: np.ndarray,
    groups: np.ndarray,
    *,
    horizon: float,
    survival_times: np.ndarray,
    l2_regularization: float = 1.0,
    maximum_iterations: int = 100,
) -> dict[str, np.ndarray | float]:
    """Fit a censored exponential reaction coordinate across held-out parents."""

    features = np.asarray(features, dtype=float)
    first_passage = np.asarray(first_passage, dtype=float)
    escaped_raw = np.asarray(escaped)
    groups = np.asarray(groups)
    survival_times = np.asarray(survival_times, dtype=float)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1 or np.any(
        ~np.isfinite(features)
    ):
        raise ValueError("features must be a finite nonempty two-dimensional array")
    sample_count = len(features)
    if first_passage.shape != (sample_count,) or np.any(~np.isfinite(first_passage)):
        raise ValueError("first_passage must be finite and align with features")
    if escaped_raw.shape != (sample_count,) or not np.all(
        np.isin(escaped_raw, (False, True, 0, 1))
    ):
        raise ValueError("escaped must be an aligned Boolean array")
    escaped = escaped_raw.astype(bool)
    if groups.shape != (sample_count,):
        raise ValueError("groups must align with feature rows")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if np.any(first_passage <= 0.0) or np.any(first_passage > horizon) or np.any(
        (~escaped) & ~np.isclose(first_passage, horizon, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "first_passage must contain positive observed times or horizon censoring"
        )
    if (
        survival_times.ndim != 1
        or len(survival_times) < 1
        or np.any(~np.isfinite(survival_times))
        or np.any(survival_times <= 0.0)
        or np.any(survival_times > horizon)
        or np.any(np.diff(survival_times) <= 0.0)
    ):
        raise ValueError("survival_times must increase inside the common horizon")
    if (
        not math.isfinite(l2_regularization)
        or l2_regularization < 0.0
        or isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, (int, np.integer))
        or maximum_iterations < 1
    ):
        raise ValueError("regularization and iteration controls must be valid")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two parent groups are required")

    out_rate = np.full(sample_count, np.nan, dtype=float)
    out_probability = np.full(sample_count, np.nan, dtype=float)
    out_baseline_rate = np.full(sample_count, np.nan, dtype=float)
    out_baseline_probability = np.full(sample_count, np.nan, dtype=float)
    group_brier_skill: list[float] = []
    group_log_likelihood_gain: list[float] = []
    group_log_likelihood_gain_per_observation: list[float] = []
    group_survival_error: list[float] = []
    group_baseline_survival_error: list[float] = []
    group_iteration_count: list[float] = []

    for held_group in unique_groups:
        held = groups == held_group
        train = ~held
        train_event = escaped[train].astype(float)
        held_event = escaped[held].astype(float)
        training_event_count = float(np.sum(train_event))
        training_exposure = float(np.sum(first_passage[train]))
        if training_event_count <= 0.0 or training_exposure <= 0.0:
            raise ValueError("every training fold must contain event exposure")
        if not np.any(held_event) or np.all(held_event):
            raise ValueError("every held parent must contain escaped and censored rows")

        mean = np.mean(features[train], axis=0)
        scale = np.std(features[train], axis=0)
        scale[scale < 1e-12] = 1.0
        train_x = np.column_stack(
            [np.ones(np.sum(train)), (features[train] - mean) / scale]
        )
        held_x = np.column_stack(
            [np.ones(np.sum(held)), (features[held] - mean) / scale]
        )
        baseline_rate = training_event_count / training_exposure
        coefficient = np.zeros(train_x.shape[1], dtype=float)
        coefficient[0] = math.log(baseline_rate)
        penalty = np.diag(
            np.concatenate(
                [[0.0], np.full(train_x.shape[1] - 1, l2_regularization)]
            )
        )
        penalty_diagonal = np.diag(penalty)

        def objective(value: np.ndarray) -> float:
            eta = np.clip(np.einsum("ij,j->i", train_x, value), -30.0, 30.0)
            rate = np.exp(eta)
            return float(
                np.sum(rate * first_passage[train] - train_event * eta)
                + 0.5 * np.sum(penalty_diagonal * value**2)
            )

        iteration_count = maximum_iterations
        for iteration in range(maximum_iterations):
            eta = np.clip(
                np.einsum("ij,j->i", train_x, coefficient), -30.0, 30.0
            )
            rate = np.exp(eta)
            weighted_exposure = rate * first_passage[train]
            gradient = np.einsum(
                "ij,i->j", train_x, weighted_exposure - train_event
            ) + penalty_diagonal * coefficient
            hessian = np.einsum(
                "ia,i,ib->ab", train_x, weighted_exposure, train_x
            ) + penalty
            step = np.linalg.solve(hessian, gradient)
            if float(np.max(np.abs(step))) < 1e-9:
                iteration_count = iteration + 1
                break
            current_objective = objective(coefficient)
            step_scale = 1.0
            while step_scale >= 2.0**-20:
                candidate = coefficient - step_scale * step
                if objective(candidate) <= current_objective:
                    coefficient = candidate
                    break
                step_scale *= 0.5
            else:
                raise ValueError("censored exponential Newton step failed to decrease")

        held_rate = np.exp(
            np.clip(np.einsum("ij,j->i", held_x, coefficient), -30.0, 30.0)
        )
        held_probability = -np.expm1(-held_rate * horizon)
        baseline_probability = -math.expm1(-baseline_rate * horizon)
        out_rate[held] = held_rate
        out_probability[held] = held_probability
        out_baseline_rate[held] = baseline_rate
        out_baseline_probability[held] = baseline_probability

        model_brier = float(np.mean((held_probability - held_event) ** 2))
        baseline_brier = float(np.mean((baseline_probability - held_event) ** 2))
        model_log_likelihood = float(
            np.sum(held_event * np.log(held_rate) - held_rate * first_passage[held])
        )
        baseline_log_likelihood = float(
            np.sum(
                held_event * math.log(baseline_rate)
                - baseline_rate * first_passage[held]
            )
        )
        likelihood_gain = model_log_likelihood - baseline_log_likelihood
        held_first_passage = first_passage[held]
        held_escaped = escaped[held]
        observed_survival = np.array(
            [
                np.mean(
                    (held_escaped & (held_first_passage > time))
                    | (~held_escaped & (held_first_passage >= time))
                )
                for time in survival_times
            ],
            dtype=float,
        )
        predicted_survival = np.array(
            [np.mean(np.exp(-held_rate * time)) for time in survival_times],
            dtype=float,
        )
        baseline_survival = np.exp(-baseline_rate * survival_times)
        group_brier_skill.append(
            1.0 - model_brier / baseline_brier if baseline_brier > 0.0 else math.nan
        )
        group_log_likelihood_gain.append(likelihood_gain)
        group_log_likelihood_gain_per_observation.append(
            likelihood_gain / float(np.sum(held))
        )
        group_survival_error.append(
            float(np.max(np.abs(predicted_survival - observed_survival)))
        )
        group_baseline_survival_error.append(
            float(np.max(np.abs(baseline_survival - observed_survival)))
        )
        group_iteration_count.append(float(iteration_count))

    if np.any(~np.isfinite(out_rate)) or np.any(~np.isfinite(out_probability)):
        raise ValueError("every observation must receive an out-of-group prediction")
    return {
        "parent_groups": unique_groups,
        "out_of_group_rate": out_rate,
        "out_of_group_event_probability": out_probability,
        "out_of_group_baseline_rate": out_baseline_rate,
        "out_of_group_baseline_event_probability": out_baseline_probability,
        "group_brier_skill": np.asarray(group_brier_skill),
        "group_log_likelihood_gain": np.asarray(group_log_likelihood_gain),
        "group_log_likelihood_gain_per_observation": np.asarray(
            group_log_likelihood_gain_per_observation
        ),
        "group_survival_calibration_error": np.asarray(group_survival_error),
        "group_baseline_survival_calibration_error": np.asarray(
            group_baseline_survival_error
        ),
        "group_iteration_count": np.asarray(group_iteration_count),
        "mean_heldout_brier_skill": float(np.mean(group_brier_skill)),
        "mean_heldout_log_likelihood_gain_per_observation": float(
            np.mean(group_log_likelihood_gain_per_observation)
        ),
        "minimum_group_log_likelihood_gain": float(
            np.min(group_log_likelihood_gain)
        ),
        "maximum_heldout_survival_calibration_error": float(
            np.max(group_survival_error)
        ),
        "maximum_baseline_survival_calibration_error": float(
            np.max(group_baseline_survival_error)
        ),
        "parent_group_count": float(len(unique_groups)),
        "observation_count": float(sample_count),
        "event_count": float(np.sum(escaped)),
        "horizon": float(horizon),
        "l2_regularization": float(l2_regularization),
        "fit_parameters_from_macro_observables": 0.0,
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
