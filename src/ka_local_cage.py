"""Microscopic local-cage coordinates derived from the Kob-Andersen potential."""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from renewal_cage import event_space_correlated_diffusion, extract_nonrecrossing_phop_events


_EPSILON = np.array([[1.0, 1.5], [1.5, 0.5]], dtype=float)
_SIGMA = np.array([[1.0, 0.8], [0.8, 0.88]], dtype=float)
_CUTOFF_SCALE = 2.5
_C3_SWITCH_ON_SCALE = 2.0


def ka_lj_radial_derivatives(
    distance: np.ndarray,
    *,
    epsilon: np.ndarray | float,
    sigma: np.ndarray | float,
    protocol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `U`, `U'`, `U''`, and `U'''` for one KA pair protocol."""

    radius, pair_epsilon, pair_sigma = np.broadcast_arrays(
        np.asarray(distance, dtype=float),
        np.asarray(epsilon, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("distance must be finite and nonnegative")
    if np.any(~np.isfinite(pair_epsilon)) or np.any(pair_epsilon < 0.0):
        raise ValueError("epsilon must be finite and nonnegative")
    if np.any(~np.isfinite(pair_sigma)) or np.any(pair_sigma <= 0.0):
        raise ValueError("sigma must be finite and positive")
    if protocol not in {"ka_lj_cut", "ka_lj_c3_switch"}:
        raise ValueError("unsupported KA pair-potential protocol")

    safe_radius = np.maximum(radius, 1e-12)
    sigma_over_r = pair_sigma / safe_radius
    sigma_over_r6 = sigma_over_r**6
    sigma_over_r12 = sigma_over_r6**2
    lj = 4.0 * pair_epsilon * (sigma_over_r12 - sigma_over_r6)
    lj_first = 24.0 * pair_epsilon * (-2.0 * sigma_over_r12 + sigma_over_r6) / safe_radius
    lj_second = (
        24.0 * pair_epsilon * (26.0 * sigma_over_r12 - 7.0 * sigma_over_r6) / safe_radius**2
    )
    lj_third = (
        24.0 * pair_epsilon * (-364.0 * sigma_over_r12 + 56.0 * sigma_over_r6)
        / safe_radius**3
    )
    active = (radius > 1e-10) & (radius < _CUTOFF_SCALE * pair_sigma)
    if protocol == "ka_lj_cut":
        return tuple(value * active for value in (lj, lj_first, lj_second, lj_third))

    inner = _C3_SWITCH_ON_SCALE * pair_sigma
    width = (_CUTOFF_SCALE - _C3_SWITCH_ON_SCALE) * pair_sigma
    switch_region = active & (radius > inner)
    x = np.where(switch_region, (radius - inner) / width, 0.0)
    switch = 1.0 - 35.0 * x**4 + 84.0 * x**5 - 70.0 * x**6 + 20.0 * x**7
    switch_first = (-140.0 * x**3 + 420.0 * x**4 - 420.0 * x**5 + 140.0 * x**6) / width
    switch_second = (-420.0 * x**2 + 1680.0 * x**3 - 2100.0 * x**4 + 840.0 * x**5) / width**2
    switch_third = (-840.0 * x + 5040.0 * x**2 - 8400.0 * x**3 + 4200.0 * x**4) / width**3

    energy = lj * switch
    first = lj_first * switch + lj * switch_first
    second = lj_second * switch + 2.0 * lj_first * switch_first + lj * switch_second
    third = (
        lj_third * switch
        + 3.0 * lj_second * switch_first
        + 3.0 * lj_first * switch_second
        + lj * switch_third
    )
    return tuple(value * active for value in (energy, first, second, third))


def static_neighbor_cage_displacement(
    positions: np.ndarray,
    *,
    target_indices: np.ndarray,
    box_lengths: np.ndarray,
    cutoff: float = 1.5,
) -> dict[str, np.ndarray]:
    """Decompose tagged displacement into a fixed-neighbor cage and residual.

    For the initial neighbor graph ``N_i(0)``, the microscopic cage coordinate
    is ``Delta C_i(t)=|N_i|^-1 sum_{j in N_i(0)} Delta r_j(t)``.  The relative
    coordinate is ``Delta u_i=Delta r_i-Delta C_i``.  The graph is fixed at
    the initial configuration, so the definition has no event labels or
    time-dependent threshold.
    """

    positions = np.asarray(positions, dtype=float)
    target_indices = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[0] < 1 or positions.shape[2] != 3 or np.any(~np.isfinite(positions)):
        raise ValueError("positions must be a finite (frames, particles, 3) array")
    if target_indices.ndim != 1 or not len(target_indices) or np.any(target_indices < 0) or np.any(target_indices >= positions.shape[1]):
        raise ValueError("target_indices must be a nonempty valid particle-index vector")
    if len(np.unique(target_indices)) != len(target_indices):
        raise ValueError("target_indices must be unique")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")

    initial = np.mod(positions[0], box_lengths)
    target_initial = initial[target_indices]
    separation = target_initial[:, None, :] - initial[None, :, :]
    separation -= box_lengths * np.rint(separation / box_lengths)
    distance = np.sqrt(np.sum(separation**2, axis=2))
    neighbor_mask = (distance < cutoff) & (distance > 1e-12)
    neighbor_count = np.sum(neighbor_mask, axis=1)
    if np.any(neighbor_count == 0):
        raise ValueError("each target must have at least one initial neighbor inside cutoff")

    displacement = positions - positions[0:1]
    cage_displacement = np.empty((len(positions), len(target_indices), 3), dtype=float)
    for slot, neighbors in enumerate(neighbor_mask):
        cage_displacement[:, slot] = np.mean(displacement[:, neighbors], axis=1)
    tagged_displacement = displacement[:, target_indices]
    return {
        "cage_displacement": cage_displacement,
        "relative_displacement": tagged_displacement - cage_displacement,
        "tagged_displacement": tagged_displacement,
        "neighbor_count": neighbor_count.astype(int),
    }


def dynamic_neighbor_cage_displacement(
    positions: np.ndarray,
    *,
    target_indices: np.ndarray,
    box_lengths: np.ndarray,
    cutoff: float = 1.5,
) -> dict[str, np.ndarray]:
    """Decompose motion using the instantaneous geometric-neighbor cage.

    At every frame, the cage center is the mean of minimum-image neighbor
    positions inside ``cutoff``.  The target anchors the image convention, so
    ``tagged_displacement = cage_displacement + relative_displacement`` holds
    exactly even when the neighbor graph changes.
    """

    positions = np.asarray(positions, dtype=float)
    target_indices = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[0] < 1 or positions.shape[2] != 3 or np.any(~np.isfinite(positions)):
        raise ValueError("positions must be a finite (frames, particles, 3) array")
    if target_indices.ndim != 1 or not len(target_indices) or np.any(target_indices < 0) or np.any(target_indices >= positions.shape[1]):
        raise ValueError("target_indices must be a nonempty valid particle-index vector")
    if len(np.unique(target_indices)) != len(target_indices):
        raise ValueError("target_indices must be unique")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")

    cage_position = np.empty((len(positions), len(target_indices), 3), dtype=float)
    neighbor_count = np.empty((len(positions), len(target_indices)), dtype=int)
    for frame, configuration in enumerate(positions):
        target = configuration[target_indices]
        separation = configuration[None, :, :] - target[:, None, :]
        separation -= box_lengths * np.rint(separation / box_lengths)
        distance = np.linalg.norm(separation, axis=2)
        neighbor_mask = (distance < cutoff) & (distance > 1e-12)
        count = np.sum(neighbor_mask, axis=1)
        if np.any(count == 0):
            raise ValueError("each target must have at least one current neighbor inside cutoff")
        cage_position[frame] = target + np.sum(separation * neighbor_mask[:, :, None], axis=1) / count[:, None]
        neighbor_count[frame] = count
    tagged_position = positions[:, target_indices]
    cage_displacement = cage_position - cage_position[0:1]
    tagged_displacement = tagged_position - tagged_position[0:1]
    relative_position = tagged_position - cage_position
    return {
        "cage_displacement": cage_displacement,
        "relative_displacement": relative_position - relative_position[0:1],
        "tagged_displacement": tagged_displacement,
        "neighbor_count": neighbor_count,
    }


def hysteretic_neighbor_cage_displacement(
    positions: np.ndarray,
    *,
    target_indices: np.ndarray,
    box_lengths: np.ndarray,
    inner_cutoff: float = 1.4,
    outer_cutoff: float = 1.6,
) -> dict[str, np.ndarray]:
    """Decompose motion using a deterministic hysteretic geometric-neighbor graph.

    A non-member enters only inside ``inner_cutoff``; a current member remains
    until it exits ``outer_cutoff``.  The graph is consequently a microscopic
    finite-memory state determined by the preceding graph and current particle
    configuration, with no event labels or fitted switching rate.
    """

    positions = np.asarray(positions, dtype=float)
    target_indices = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[0] < 1 or positions.shape[2] != 3 or np.any(~np.isfinite(positions)):
        raise ValueError("positions must be a finite (frames, particles, 3) array")
    if target_indices.ndim != 1 or not len(target_indices) or np.any(target_indices < 0) or np.any(target_indices >= positions.shape[1]):
        raise ValueError("target_indices must be a nonempty valid particle-index vector")
    if len(np.unique(target_indices)) != len(target_indices):
        raise ValueError("target_indices must be unique")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if not math.isfinite(inner_cutoff) or not math.isfinite(outer_cutoff) or not (0.0 < inner_cutoff <= outer_cutoff):
        raise ValueError("cutoffs must satisfy 0 < inner_cutoff <= outer_cutoff")

    cage_position = np.empty((len(positions), len(target_indices), 3), dtype=float)
    neighbor_count = np.empty((len(positions), len(target_indices)), dtype=int)
    membership_change_count = np.empty((len(positions), len(target_indices)), dtype=int)
    graph: np.ndarray | None = None
    for frame, configuration in enumerate(positions):
        target = configuration[target_indices]
        separation = configuration[None, :, :] - target[:, None, :]
        separation -= box_lengths * np.rint(separation / box_lengths)
        distance = np.linalg.norm(separation, axis=2)
        self_mask = distance <= 1e-12
        if graph is None:
            graph = (distance < inner_cutoff) & ~self_mask
            membership_change_count[frame] = 0
        else:
            previous_graph = graph
            graph = ((graph & (distance < outer_cutoff)) | (~graph & (distance < inner_cutoff))) & ~self_mask
            membership_change_count[frame] = np.sum(graph != previous_graph, axis=1)
        count = np.sum(graph, axis=1)
        if np.any(count == 0):
            raise ValueError("each target must retain at least one hysteretic neighbor")
        cage_position[frame] = target + np.sum(separation * graph[:, :, None], axis=1) / count[:, None]
        neighbor_count[frame] = count
    tagged_position = positions[:, target_indices]
    cage_displacement = cage_position - cage_position[0:1]
    tagged_displacement = tagged_position - tagged_position[0:1]
    relative_position = tagged_position - cage_position
    return {
        "cage_displacement": cage_displacement,
        "relative_displacement": relative_position - relative_position[0:1],
        "tagged_displacement": tagged_displacement,
        "neighbor_count": neighbor_count,
        "neighbor_membership_change_count": membership_change_count,
    }


def local_bond_orientational_features(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    cutoff: float = 1.5,
    orders: tuple[int, ...] = (4, 6, 8),
) -> dict[str, np.ndarray | tuple[str, ...]]:
    """Return local species coordination and rotationally invariant bond order.

    For a target's initial neighbors, ``Q_l^2`` is evaluated through the
    Legendre addition theorem, ``N^-2 sum_jk P_l(rhat_ij dot rhat_ik)``.
    This avoids a coordinate-frame choice and uses only initial geometry.
    """

    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or np.any(~np.isfinite(positions)):
        raise ValueError("positions must be a finite (particles, 3) array")
    if particle_types.shape != (len(positions),) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be aligned 0/1 labels")
    if target_indices.ndim != 1 or not len(target_indices) or np.any(target_indices < 0) or np.any(target_indices >= len(positions)):
        raise ValueError("target_indices must be a nonempty valid index vector")
    if len(np.unique(target_indices)) != len(target_indices):
        raise ValueError("target_indices must be unique")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    if not orders or any(order < 1 or order % 2 for order in orders) or len(set(orders)) != len(orders):
        raise ValueError("orders must be unique positive even integers")

    wrapped = np.mod(positions, box_lengths)
    output = np.empty((len(target_indices), 2 + len(orders)), dtype=float)
    for slot, target in enumerate(target_indices):
        displacement = wrapped - wrapped[target]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        distance = np.linalg.norm(displacement, axis=1)
        neighbor = (distance > 1e-12) & (distance < cutoff)
        if not np.any(neighbor):
            raise ValueError("each target must have at least one neighbor inside cutoff")
        vectors = displacement[neighbor] / distance[neighbor, None]
        cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
        output[slot, 0] = float(np.sum(particle_types[neighbor] == 0))
        output[slot, 1] = float(np.sum(particle_types[neighbor] == 1))
        for index, order in enumerate(orders, start=2):
            coefficients = np.zeros(order + 1)
            coefficients[-1] = 1.0
            invariant_squared = float(np.mean(np.polynomial.legendre.legval(cosine, coefficients)))
            output[slot, index] = math.sqrt(max(invariant_squared, 0.0))
    return {
        "features": output,
        "feature_names": ("coordination_A", "coordination_B", *tuple(f"Q{order}" for order in orders)),
    }


def ka_lj_local_energy_force_hessian(
    candidates: np.ndarray,
    *,
    environment_positions: np.ndarray,
    target_indices: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the frozen-neighbor tagged-particle KA potential and Hessian.

    ``target_indices`` removes the tagged particle's own original position
    from the environment.  This explicit exclusion remains necessary after a
    trial point moves away from its saved coordinate.
    """

    candidates = np.asarray(candidates, dtype=float)
    environment_positions = np.asarray(environment_positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidates must have shape (targets, 3)")
    if environment_positions.ndim != 2 or environment_positions.shape[1] != 3:
        raise ValueError("environment_positions must have shape (particles, 3)")
    if particle_types.shape != (len(environment_positions),) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be 0/1 and align with environment_positions")
    if target.shape != (len(candidates),) or np.any(target < 0) or np.any(target >= len(environment_positions)):
        raise ValueError("target_indices must align with candidates and environment_positions")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")

    wrapped_candidates = np.mod(candidates, box_lengths)
    wrapped_environment = np.mod(environment_positions, box_lengths)
    displacement = wrapped_candidates[:, None, :] - wrapped_environment[None, :, :]
    displacement -= box_lengths * np.rint(displacement / box_lengths)
    squared_distance = np.sum(displacement**2, axis=2)
    distance = np.sqrt(np.maximum(squared_distance, 1e-24))
    epsilon = _EPSILON[particle_types[target, None], particle_types[None, :]]
    sigma = _SIGMA[particle_types[target, None], particle_types[None, :]]
    active = (distance > 1e-10) & (distance < _CUTOFF_SCALE * sigma)
    active[np.arange(len(candidates)), target] = False
    sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
    sigma_over_r6 = sigma_over_r2**3
    sigma_over_r12 = sigma_over_r6**2
    energy = np.sum(4.0 * epsilon * (sigma_over_r12 - sigma_over_r6) * active, axis=1)
    force_coefficient = (
        24.0 * epsilon * (2.0 * sigma_over_r12 - sigma_over_r6) / np.maximum(squared_distance, 1e-24)
    )
    force = np.sum((force_coefficient * active)[:, :, None] * displacement, axis=1)
    potential_prime_over_r = (
        24.0 * epsilon * (-2.0 * sigma_over_r12 + sigma_over_r6) / np.maximum(squared_distance, 1e-24)
    ) * active
    potential_second = (
        24.0 * epsilon * (26.0 * sigma_over_r12 - 7.0 * sigma_over_r6) / np.maximum(squared_distance, 1e-24)
    ) * active
    unit = displacement / np.maximum(distance[:, :, None], 1e-12)
    identity = np.eye(3)
    hessian = np.sum(
        (potential_second - potential_prime_over_r)[:, :, None, None]
        * unit[:, :, :, None]
        * unit[:, :, None, :]
        + potential_prime_over_r[:, :, None, None] * identity,
        axis=1,
    )
    return energy, force, hessian


def ka_lj_force_and_isotropic_curvature(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray | None = None,
    potential_protocol: str = "ka_lj_cut",
) -> tuple[np.ndarray, np.ndarray]:
    """Return KA forces and ``tr(H_ii)/3`` for selected particles.

    For a central pair potential, the diagonal Hessian trace is
    ``V''(r) + 2 V'(r)/r`` in three dimensions.  The returned curvature is
    therefore the local isotropic harmonic stiffness entering
    ``F_i = -kappa_i (r_i-C_i)``.
    """

    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (particles, 3)")
    if particle_types.shape != (len(positions),) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be 0/1 and align with positions")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if target_indices is None:
        target = np.arange(len(positions), dtype=int)
    else:
        target = np.asarray(target_indices, dtype=int)
        if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= len(positions)):
            raise ValueError("target_indices must select valid particles")

    wrapped = np.mod(positions, box_lengths)
    displacement = wrapped[target, None, :] - wrapped[None, :, :]
    displacement -= box_lengths * np.rint(displacement / box_lengths)
    squared_distance = np.sum(displacement**2, axis=2)
    distance = np.sqrt(np.maximum(squared_distance, 1e-24))
    epsilon = _EPSILON[particle_types[target, None], particle_types[None, :]]
    sigma = _SIGMA[particle_types[target, None], particle_types[None, :]]
    _, potential_first, potential_second, _ = ka_lj_radial_derivatives(
        distance,
        epsilon=epsilon,
        sigma=sigma,
        protocol=potential_protocol,
    )
    potential_first_over_r = potential_first / np.maximum(distance, 1e-12)
    force = np.sum((-potential_first_over_r)[:, :, None] * displacement, axis=1)
    curvature = np.sum(potential_second + 2.0 * potential_first_over_r, axis=1) / 3.0
    return force, curvature


def _ka_lj_target_pair_geometry(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    potential_protocol: str = "ka_lj_cut",
) -> dict[str, np.ndarray]:
    """Return exact KA pair forces and Hessian blocks for selected targets."""

    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if positions.ndim != 2 or positions.shape[1] != 3 or np.any(~np.isfinite(positions)):
        raise ValueError("positions must have finite shape (particles, 3)")
    if particle_types.shape != (len(positions),) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be 0/1 and align with positions")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= len(positions)):
        raise ValueError("target_indices must select valid particles")

    wrapped = np.mod(positions, box_lengths)
    displacement = wrapped[target, None, :] - wrapped[None, :, :]
    displacement -= box_lengths * np.rint(displacement / box_lengths)
    squared_distance = np.sum(displacement**2, axis=2)
    distance = np.sqrt(np.maximum(squared_distance, 1e-24))
    epsilon = _EPSILON[particle_types[target, None], particle_types[None, :]]
    sigma = _SIGMA[particle_types[target, None], particle_types[None, :]]
    _, potential_first, potential_second, _ = ka_lj_radial_derivatives(
        distance,
        epsilon=epsilon,
        sigma=sigma,
        protocol=potential_protocol,
    )
    active = (distance > 1e-10) & (distance < _CUTOFF_SCALE * sigma)
    potential_prime_over_r = potential_first / np.maximum(distance, 1e-12)
    pair_force = (-potential_prime_over_r)[:, :, None] * displacement
    unit = displacement / np.maximum(distance[:, :, None], 1e-12)
    pair_hessian = (
        (potential_second - potential_prime_over_r)[:, :, None, None]
        * unit[:, :, :, None]
        * unit[:, :, None, :]
        + potential_prime_over_r[:, :, None, None] * np.eye(3)
    )
    return {
        "pair_force": pair_force,
        "pair_hessian": pair_hessian,
        "active": active,
        "distance": distance,
        "sigma": sigma,
    }


def ka_lj_force_generator_observables(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    friction: float,
    temperature: float,
    potential_protocol: str = "ka_lj_cut",
) -> dict[str, np.ndarray]:
    """Return ``F``, ``L F``, and the conditional noise covariance rate."""

    velocities = np.asarray(velocities, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if velocities.shape != np.asarray(positions).shape or np.any(~np.isfinite(velocities)):
        raise ValueError("velocities must be finite and align with positions")
    if not math.isfinite(friction) or friction < 0.0 or not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("friction and temperature must be finite and nonnegative")
    geometry = _ka_lj_target_pair_geometry(
        positions,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        potential_protocol=potential_protocol,
    )
    pair_hessian = geometry["pair_hessian"]
    relative_velocity = velocities[target, None, :] - velocities[None, :, :]
    force_generator = -np.sum(np.einsum("pnab,pnb->pna", pair_hessian, relative_velocity), axis=1)
    diagonal_hessian = np.sum(pair_hessian, axis=1)
    covariance_geometry = np.einsum("pab,pcb->pac", diagonal_hessian, diagonal_hessian)
    covariance_geometry += np.einsum("pnab,pncb->pac", pair_hessian, pair_hessian)
    cutoff_gap = geometry["distance"] - _CUTOFF_SCALE * geometry["sigma"]
    valid_pair = geometry["distance"] > 1e-10
    nearest_cutoff_particle = np.argmin(np.where(valid_pair, np.abs(cutoff_gap), np.inf), axis=1)
    return {
        "force": np.sum(geometry["pair_force"], axis=1),
        "force_generator": force_generator,
        "force_generator_noise_covariance_rate": 2.0 * friction * temperature * covariance_geometry,
        "target_pair_active": geometry["active"],
        "target_pair_hessian": pair_hessian,
        "nearest_cutoff_particle_index": nearest_cutoff_particle,
        "nearest_cutoff_signed_gap": cutoff_gap[np.arange(len(target)), nearest_cutoff_particle],
    }


def ka_lj_sparse_force_generator_observables(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    potential_protocol: str = "ka_lj_cut",
) -> dict[str, np.ndarray | float]:
    """Return selected KA forces and ``L F`` using periodic linked cells."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    targets = np.asarray(target_indices, dtype=int)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or velocities.shape != positions.shape
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(velocities))
    ):
        raise ValueError("positions and velocities must be aligned finite arrays")
    if particle_types.shape != (len(positions),) or np.any(
        (particle_types < 0) | (particle_types > 1)
    ):
        raise ValueError("particle_types must be aligned KA 0/1 labels")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(
        box_lengths <= 0.0
    ):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if (
        targets.ndim != 1
        or len(targets) < 1
        or np.any(targets < 0)
        or np.any(targets >= len(positions))
    ):
        raise ValueError("target_indices must select one or more particles")
    if potential_protocol not in {"ka_lj_cut", "ka_lj_c3_switch"}:
        raise ValueError("unsupported KA pair-potential protocol")

    wrapped = np.mod(positions, box_lengths)
    maximum_cutoff = _CUTOFF_SCALE * float(np.max(_SIGMA))
    cell_count = np.maximum(np.floor(box_lengths / maximum_cutoff).astype(int), 1)
    cell_width = box_lengths / cell_count
    particle_cells = np.floor(wrapped / cell_width).astype(int)
    particle_cells = np.minimum(particle_cells, cell_count - 1)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for particle, cell in enumerate(particle_cells):
        key = tuple(int(value) for value in cell)
        buckets.setdefault(key, []).append(particle)

    force = np.empty((len(targets), 3), dtype=float)
    force_generator = np.empty_like(force)
    candidate_count = np.empty(len(targets), dtype=int)
    identity = np.eye(3)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    for slot, target in enumerate(targets):
        center = particle_cells[target]
        neighbor_cells = {
            tuple(int(value) for value in np.mod(center + offset, cell_count))
            for offset in offsets
        }
        candidates = np.asarray(
            sorted(
                particle
                for cell in neighbor_cells
                for particle in buckets.get(cell, ())
            ),
            dtype=int,
        )
        candidate_count[slot] = len(candidates)
        displacement = wrapped[target] - wrapped[candidates]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        distance = np.linalg.norm(displacement, axis=1)
        epsilon = _EPSILON[particle_types[target], particle_types[candidates]]
        sigma = _SIGMA[particle_types[target], particle_types[candidates]]
        _, first, second, _ = ka_lj_radial_derivatives(
            distance,
            epsilon=epsilon,
            sigma=sigma,
            protocol=potential_protocol,
        )
        safe_distance = np.maximum(distance, 1e-12)
        first_over_r = first / safe_distance
        unit = displacement / safe_distance[:, None]
        pair_hessian = (
            (second - first_over_r)[:, None, None]
            * unit[:, :, None]
            * unit[:, None, :]
            + first_over_r[:, None, None] * identity
        )
        force[slot] = np.sum((-first_over_r)[:, None] * displacement, axis=0)
        relative_velocity = velocities[target] - velocities[candidates]
        force_generator[slot] = -np.sum(
            np.einsum("nab,nb->na", pair_hessian, relative_velocity), axis=0
        )

    return {
        "force": force,
        "force_generator": force_generator,
        "candidate_count": candidate_count,
        "cell_count": cell_count,
        "minimum_cell_width": float(np.min(cell_width)),
        "thermodynamic_claim_allowed": 0.0,
    }


def ka_lj_sparse_force_generator_multi(
    positions: np.ndarray,
    *,
    velocity_fields: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    potential_protocol: str = "ka_lj_cut",
) -> dict[str, np.ndarray | float]:
    """Contract one sparse KA pair-Hessian geometry with many velocity fields."""

    positions = np.asarray(positions, dtype=float)
    fields = np.asarray(velocity_fields, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    targets = np.asarray(target_indices, dtype=int)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or fields.ndim != 3
        or fields.shape[1:] != positions.shape
        or len(fields) < 1
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(fields))
    ):
        raise ValueError("positions and velocity_fields must be aligned finite arrays")
    if particle_types.shape != (len(positions),) or np.any(
        (particle_types < 0) | (particle_types > 1)
    ):
        raise ValueError("particle_types must be aligned KA 0/1 labels")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(
        box_lengths <= 0.0
    ):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if (
        targets.ndim != 1
        or len(targets) < 1
        or np.any(targets < 0)
        or np.any(targets >= len(positions))
    ):
        raise ValueError("target_indices must select one or more particles")
    if potential_protocol not in {"ka_lj_cut", "ka_lj_c3_switch"}:
        raise ValueError("unsupported KA pair-potential protocol")

    wrapped = np.mod(positions, box_lengths)
    maximum_cutoff = _CUTOFF_SCALE * float(np.max(_SIGMA))
    cell_count = np.maximum(np.floor(box_lengths / maximum_cutoff).astype(int), 1)
    cell_width = box_lengths / cell_count
    particle_cells = np.floor(wrapped / cell_width).astype(int)
    particle_cells = np.minimum(particle_cells, cell_count - 1)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for particle, cell in enumerate(particle_cells):
        key = tuple(int(value) for value in cell)
        buckets.setdefault(key, []).append(particle)

    force_generator = np.empty((len(fields), len(targets), 3), dtype=float)
    candidate_count = np.empty(len(targets), dtype=int)
    identity = np.eye(3)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    for slot, target in enumerate(targets):
        center = particle_cells[target]
        neighbor_cells = {
            tuple(int(value) for value in np.mod(center + offset, cell_count))
            for offset in offsets
        }
        candidates = np.asarray(
            sorted(
                particle
                for cell in neighbor_cells
                for particle in buckets.get(cell, ())
            ),
            dtype=int,
        )
        candidate_count[slot] = len(candidates)
        displacement = wrapped[target] - wrapped[candidates]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        distance = np.linalg.norm(displacement, axis=1)
        epsilon = _EPSILON[particle_types[target], particle_types[candidates]]
        sigma = _SIGMA[particle_types[target], particle_types[candidates]]
        _, first, second, _ = ka_lj_radial_derivatives(
            distance,
            epsilon=epsilon,
            sigma=sigma,
            protocol=potential_protocol,
        )
        safe_distance = np.maximum(distance, 1e-12)
        first_over_r = first / safe_distance
        unit = displacement / safe_distance[:, None]
        pair_hessian = (
            (second - first_over_r)[:, None, None]
            * unit[:, :, None]
            * unit[:, None, :]
            + first_over_r[:, None, None] * identity
        )
        relative_velocity = fields[:, target, None, :] - fields[:, candidates, :]
        force_generator[:, slot] = -np.sum(
            np.einsum("nab,pnb->pna", pair_hessian, relative_velocity), axis=1
        )
    return {
        "force_generator": force_generator,
        "candidate_count": candidate_count,
        "cell_count": cell_count,
        "minimum_cell_width": float(np.min(cell_width)),
        "thermodynamic_claim_allowed": 0.0,
    }


def ka_lj_second_force_generator(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    friction: float,
    directional_step: float = 1e-5,
    potential_protocol: str = "ka_lj_cut",
) -> np.ndarray:
    """Return the deterministic drift ``L^2 F`` for selected KA particles."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if not math.isfinite(directional_step) or directional_step <= 0.0:
        raise ValueError("directional_step must be finite and positive")
    geometry = _ka_lj_target_pair_geometry(
        positions,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        potential_protocol=potential_protocol,
    )
    plus = ka_lj_force_generator_observables(
        positions + directional_step * velocities,
        velocities=velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=friction,
        temperature=0.0,
        potential_protocol=potential_protocol,
    )["force_generator"]
    minus = ka_lj_force_generator_observables(
        positions - directional_step * velocities,
        velocities=velocities,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=target,
        friction=friction,
        temperature=0.0,
        potential_protocol=potential_protocol,
    )["force_generator"]
    position_drift = (plus - minus) / (2.0 * directional_step)

    velocity_drift = np.empty_like(position_drift)
    pair_hessian = geometry["pair_hessian"]
    for slot, particle in enumerate(target):
        neighbors = np.flatnonzero(geometry["active"][slot])
        selected = np.concatenate(([particle], neighbors))
        selected_force = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=selected,
            potential_protocol=potential_protocol,
        )[0]
        acceleration = selected_force - friction * velocities[selected]
        velocity_drift[slot] = -np.sum(
            np.einsum("jab,jb->ja", pair_hessian[slot, neighbors], acceleration[0] - acceleration[1:]),
            axis=0,
        )
    return position_drift + velocity_drift


def force_generator_increment_diagnostic(
    force: np.ndarray,
    force_generator: np.ndarray,
    second_force_generator: np.ndarray,
    force_generator_noise_covariance_rate: np.ndarray,
    *,
    frame_time: float,
) -> dict[str, float]:
    """Compare finite force increments with exact KA generator predictions."""

    force = np.asarray(force, dtype=float)
    generator = np.asarray(force_generator, dtype=float)
    second = np.asarray(second_force_generator, dtype=float)
    covariance_rate = np.asarray(force_generator_noise_covariance_rate, dtype=float)
    if force.ndim != 2 or force.shape[0] < 3 or force.shape[1] != 3:
        raise ValueError("force must have shape (at least 3 frames, 3)")
    if generator.shape != force.shape or second.shape != force.shape or np.any(~np.isfinite(force)):
        raise ValueError("force and generator arrays must be finite and aligned")
    if np.any(~np.isfinite(generator)) or np.any(~np.isfinite(second)):
        raise ValueError("force and generator arrays must be finite and aligned")
    if covariance_rate.shape != (len(force), 3, 3) or np.any(~np.isfinite(covariance_rate)):
        raise ValueError("covariance rates must be finite and align with force frames")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be finite and positive")

    centered_force_derivative = (force[2:] - force[:-2]) / (2.0 * frame_time)
    centered_generator = generator[1:-1]
    derivative_difference = centered_force_derivative - centered_generator
    derivative_norm = float(np.linalg.norm(centered_generator))
    if derivative_norm == 0.0:
        raise ValueError("force_generator must have nonzero centered norm")

    innovation = generator[1:] - generator[:-1] - frame_time * second[:-1]
    predicted_covariance = frame_time * covariance_rate[:-1]
    squared_mahalanobis = np.asarray(
        [value @ np.linalg.solve(covariance, value) for value, covariance in zip(innovation, predicted_covariance)]
    )
    innovation_rms = math.sqrt(float(np.mean(np.sum(innovation**2, axis=1))))
    if innovation_rms == 0.0:
        raise ValueError("generator increments must have nonzero innovation")
    return {
        "force_derivative_relative_l2": float(np.linalg.norm(derivative_difference) / derivative_norm),
        "force_derivative_correlation": float(
            np.corrcoef(centered_force_derivative.reshape(-1), centered_generator.reshape(-1))[0, 1]
        ),
        "innovation_trace_variance_ratio": float(
            np.sum(innovation**2) / np.sum(np.trace(predicted_covariance, axis1=1, axis2=2))
        ),
        "innovation_mean_squared_mahalanobis": float(np.mean(squared_mahalanobis)),
        "innovation_normalized_mean": float(np.linalg.norm(np.mean(innovation, axis=0)) / innovation_rms),
        "thermodynamic_claim_allowed": 0.0,
    }


def ka_lj_shell_forces(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    shell_edges: np.ndarray,
) -> np.ndarray:
    """Resolve exact KA pair force on targets into fixed radial shells."""

    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    shell_edges = np.asarray(shell_edges, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or particle_types.shape != (len(positions),):
        raise ValueError("positions and particle_types must align")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= len(positions)):
        raise ValueError("target_indices must select valid particles")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be positive")
    if shell_edges.ndim != 1 or len(shell_edges) < 2 or shell_edges[0] != 0.0 or np.any(np.diff(shell_edges) <= 0.0):
        raise ValueError("shell_edges must start at zero and increase strictly")
    if shell_edges[-1] < _CUTOFF_SCALE * float(np.max(_SIGMA)):
        raise ValueError("shell_edges must cover the full KA pair cutoff")
    wrapped = np.mod(positions, box_lengths)
    displacement = wrapped[target, None, :] - wrapped[None, :, :]
    displacement -= box_lengths * np.rint(displacement / box_lengths)
    squared_distance = np.sum(displacement**2, axis=2)
    distance = np.sqrt(np.maximum(squared_distance, 1e-24))
    epsilon = _EPSILON[particle_types[target, None], particle_types[None, :]]
    sigma = _SIGMA[particle_types[target, None], particle_types[None, :]]
    active = (distance > 1e-10) & (distance < _CUTOFF_SCALE * sigma)
    sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
    sigma_over_r6 = sigma_over_r2**3
    sigma_over_r12 = sigma_over_r6**2
    coefficient = 24.0 * epsilon * (2.0 * sigma_over_r12 - sigma_over_r6) / np.maximum(squared_distance, 1e-24)
    pair_force = (coefficient * active)[:, :, None] * displacement
    shell = np.empty((len(target), len(shell_edges) - 1, 3), dtype=float)
    for shell_index, (left, right) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        include_right = shell_index == len(shell_edges) - 2
        member = (distance >= left) & ((distance <= right) if include_right else (distance < right))
        shell[:, shell_index] = np.sum(pair_force * member[:, :, None], axis=1)
    return shell


def ka_local_cluster_hessian(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_index: int,
    cluster_cutoff: float,
) -> dict[str, np.ndarray]:
    """Return the exact KA Hessian for a target-centered cluster.

    Cluster particles are dynamical coordinates while all particles outside the
    cluster remain pinned.  Their pair interactions still enter diagonal
    blocks, so the resulting local spectrum is the harmonic landscape seen by
    that cluster in its instantaneous many-particle environment.
    """

    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (particles, 3)")
    if particle_types.shape != (len(positions),) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be 0/1 and align with positions")
    if not isinstance(target_index, (int, np.integer)) or target_index < 0 or target_index >= len(positions):
        raise ValueError("target_index must select a valid particle")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0) or cluster_cutoff <= 0.0:
        raise ValueError("box_lengths and cluster_cutoff must be positive")
    wrapped = np.mod(positions, box_lengths)
    target_displacement = wrapped - wrapped[target_index]
    target_displacement -= box_lengths * np.rint(target_displacement / box_lengths)
    cluster_indices = np.flatnonzero(np.linalg.norm(target_displacement, axis=1) < cluster_cutoff)
    local_index = {int(particle): index for index, particle in enumerate(cluster_indices)}
    hessian = np.zeros((3 * len(cluster_indices), 3 * len(cluster_indices)), dtype=float)
    identity = np.eye(3)
    for local_i, particle_i in enumerate(cluster_indices):
        displacement = wrapped[particle_i] - wrapped
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        squared_distance = np.sum(displacement**2, axis=1)
        distance = np.sqrt(np.maximum(squared_distance, 1e-24))
        epsilon = _EPSILON[particle_types[particle_i], particle_types]
        sigma = _SIGMA[particle_types[particle_i], particle_types]
        active = (distance > 1e-10) & (distance < _CUTOFF_SCALE * sigma)
        sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
        sigma_over_r6 = sigma_over_r2**3
        sigma_over_r12 = sigma_over_r6**2
        potential_prime_over_r = (
            24.0 * epsilon * (-2.0 * sigma_over_r12 + sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        ) * active
        potential_second = (
            24.0 * epsilon * (26.0 * sigma_over_r12 - 7.0 * sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        ) * active
        unit = displacement / np.maximum(distance[:, None], 1e-12)
        pair_hessian = (
            (potential_second - potential_prime_over_r)[:, None, None] * unit[:, :, None] * unit[:, None, :]
            + potential_prime_over_r[:, None, None] * identity
        )
        diagonal = np.sum(pair_hessian, axis=0)
        row = slice(3 * local_i, 3 * local_i + 3)
        hessian[row, row] += diagonal
        for particle_j in cluster_indices:
            if particle_j == particle_i:
                continue
            local_j = local_index[int(particle_j)]
            column = slice(3 * local_j, 3 * local_j + 3)
            hessian[row, column] -= pair_hessian[particle_j]
    return {"cluster_indices": cluster_indices, "hessian": hessian}


def ka_local_cluster_soft_mode_features(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    cluster_cutoff: float,
    ranks: tuple[int, ...] = (0, 3),
    eigenvalue_floor: float = 1e-6,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return target participation in positive local KA Hessian modes.

    The supplied configuration can be an inherent structure or an instantaneous
    liquid configuration.  Every target has its own target-centered active
    cluster, with the exterior retained in the Hessian diagonal as a pinned
    many-particle environment.
    """

    target = np.asarray(target_indices, dtype=int)
    if target.ndim != 1 or not len(target) or len(np.unique(target)) != len(target):
        raise ValueError("target_indices must be a nonempty vector of unique particle indices")
    if not ranks or any(rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be unique nonnegative integers")
    if not math.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be finite and positive")
    names = tuple(
        name
        for rank in ranks
        for name in (f"inverse_eigenvalue_rank_{rank}", f"target_weighted_softness_rank_{rank}")
    )
    features = np.empty((len(target), len(names)), dtype=float)
    for target_slot, target_index in enumerate(target):
        cluster = ka_local_cluster_hessian(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_index=int(target_index),
            cluster_cutoff=cluster_cutoff,
        )
        eigenvalue, eigenvector = np.linalg.eigh(cluster["hessian"])
        positive = np.flatnonzero(eigenvalue > eigenvalue_floor)
        if len(positive) <= max(ranks):
            raise ValueError("cluster has too few positive modes for requested ranks")
        local_target = int(np.flatnonzero(cluster["cluster_indices"] == target_index)[0])
        column = 0
        for rank in ranks:
            mode = int(positive[rank])
            stiffness = float(eigenvalue[mode])
            weight = float(np.sum(eigenvector[3 * local_target : 3 * local_target + 3, mode] ** 2))
            features[target_slot, column] = 1.0 / stiffness
            features[target_slot, column + 1] = weight / stiffness
            column += 2
    return features, names


def ka_local_cluster_anharmonic_barrier_features(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    cluster_cutoff: float,
    ranks: tuple[int, ...] = (0, 3),
    eigenvalue_floor: float = 1e-6,
    finite_difference_step: float = 2e-3,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Estimate cubic landscape barriers along target-localized KA soft modes.

    For a normalized cluster mode ``e`` with stiffness ``kappa`` we obtain the
    signed cubic coefficient by a centered second difference of the exact KA
    generalized force ``dU/dq=-F.dot(e)`` with the exterior held fixed.  This
    avoids subtracting nearly equal extensive local energies.  The cubic truncation has its
    nonzero stationary point at ``q=-2*kappa/tau`` and barrier
    ``2*kappa**3/(3*tau**2)``.  This is an anharmonic barrier proxy, not a
    fully optimized nonlinear quasilocalized excitation.
    """

    positions = np.asarray(positions, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("positions must be a finite particle-by-three array")
    if particle_types.shape != (len(positions),) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must align with positions and use 0/1 labels")
    if box_lengths.shape != (3,) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be positive")
    if target.ndim != 1 or not len(target) or len(np.unique(target)) != len(target) or np.any(target < 0) or np.any(target >= len(positions)):
        raise ValueError("target_indices must be unique valid particle indices")
    if not ranks or any(rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("ranks must be unique nonnegative integers")
    if not math.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0 or not math.isfinite(finite_difference_step) or finite_difference_step <= 0.0:
        raise ValueError("eigenvalue_floor and finite_difference_step must be finite and positive")
    names = tuple(
        name
        for rank in ranks
        for name in (f"cubic_magnitude_rank_{rank}", f"anharmonic_barrier_proxy_rank_{rank}")
    )
    features = np.empty((len(target), len(names)), dtype=float)
    for target_slot, target_index in enumerate(target):
        cluster = ka_local_cluster_hessian(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_index=int(target_index),
            cluster_cutoff=cluster_cutoff,
        )
        cluster_indices = cluster["cluster_indices"]
        eigenvalue, eigenvector = np.linalg.eigh(cluster["hessian"])
        positive = np.flatnonzero(eigenvalue > eigenvalue_floor)
        if len(positive) <= max(ranks):
            raise ValueError("cluster has too few positive modes for requested ranks")
        column = 0
        for rank in ranks:
            mode = int(positive[rank])
            direction = eigenvector[:, mode].reshape((-1, 3))
            generalized_force = []
            for multiple in (-1.0, 0.0, 1.0):
                displaced = positions.copy()
                displaced[cluster_indices] += multiple * finite_difference_step * direction
                force, _ = ka_lj_force_and_isotropic_curvature(
                    displaced,
                    particle_types=particle_types,
                    box_lengths=box_lengths,
                    target_indices=cluster_indices,
                )
                generalized_force.append(
                    -float(np.sum(force * direction))
                )
            cubic = (generalized_force[2] - 2.0 * generalized_force[1] + generalized_force[0]) / finite_difference_step**2
            if not math.isfinite(cubic) or abs(cubic) <= 1e-14:
                raise ValueError("mode cubic coefficient is not resolvable at this finite-difference step")
            stiffness = float(eigenvalue[mode])
            features[target_slot, column] = abs(cubic)
            features[target_slot, column + 1] = 2.0 * stiffness**3 / (3.0 * cubic**2)
            column += 2
    return features, names


def ka_local_harmonic_cage_coordinates(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    curvature_floor: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Infer ``C_i=r_i+F_i/kappa_i`` from a trajectory of KA configurations."""

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (frames, particles, 3)")
    if curvature_floor <= 0.0 or not math.isfinite(curvature_floor):
        raise ValueError("curvature_floor must be positive and finite")
    target = np.asarray(target_indices, dtype=int)
    centers = np.full((len(positions), len(target), 3), np.nan, dtype=float)
    forces = np.empty_like(centers)
    curvatures = np.empty((len(positions), len(target)), dtype=float)
    for frame_index, frame in enumerate(positions):
        force, curvature = ka_lj_force_and_isotropic_curvature(
            frame,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
        )
        valid = curvature > curvature_floor
        centers[frame_index, valid] = frame[target[valid]] + force[valid] / curvature[valid, None]
        forces[frame_index] = force
        curvatures[frame_index] = curvature
    residual = positions[:, target] - centers
    return {
        "target_indices": target,
        "centers": centers,
        "residual": residual,
        "forces": forces,
        "curvatures": curvatures,
        "valid": np.isfinite(residual).all(axis=2),
    }


def ka_frozen_neighbor_minima(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    curvature_floor: float = 1e-4,
    maximum_step: float = 0.15,
    maximum_iterations: int = 8,
) -> dict[str, np.ndarray]:
    """Find a tagged particle's local KA minimum with each frame's neighbors fixed."""

    positions = np.asarray(positions, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (frames, particles, 3)")
    if curvature_floor <= 0.0 or maximum_step <= 0.0 or maximum_iterations < 1:
        raise ValueError("minimum-search controls must be positive")
    centers = np.full((len(positions), len(target), 3), np.nan, dtype=float)
    valid = np.zeros((len(positions), len(target)), dtype=bool)
    iterations = np.zeros(len(positions), dtype=int)
    isotropic_curvatures = np.full((len(positions), len(target)), np.nan, dtype=float)
    basin_depths = np.full((len(positions), len(target)), np.nan, dtype=float)
    for frame_index, frame in enumerate(positions):
        candidate = frame[target].copy()
        initial_energy, _, _ = ka_lj_local_energy_force_hessian(
            candidate,
            environment_positions=frame,
            target_indices=target,
            particle_types=particle_types,
            box_lengths=box_lengths,
        )
        for iteration in range(maximum_iterations):
            energy, force, hessian = ka_lj_local_energy_force_hessian(
                candidate,
                environment_positions=frame,
                target_indices=target,
                particle_types=particle_types,
                box_lengths=box_lengths,
            )
            stable = np.min(np.linalg.eigvalsh(hessian), axis=1) > curvature_floor
            displacement = np.zeros_like(candidate)
            if np.any(stable):
                displacement[stable] = np.linalg.solve(
                    hessian[stable], force[stable, :, None]
                )[:, :, 0]
            norm = np.linalg.norm(displacement, axis=1)
            displacement *= np.minimum(1.0, maximum_step / np.maximum(norm, 1e-12))[:, None]
            trial = candidate + displacement
            trial_energy, _, _ = ka_lj_local_energy_force_hessian(
                trial,
                environment_positions=frame,
                target_indices=target,
                particle_types=particle_types,
                box_lengths=box_lengths,
            )
            improved = stable & (trial_energy < energy - 1e-10)
            candidate[improved] = trial[improved]
            if not np.any(improved):
                iterations[frame_index] = iteration + 1
                break
        else:
            iterations[frame_index] = maximum_iterations
        final_energy, _, final_hessian = ka_lj_local_energy_force_hessian(
            candidate,
            environment_positions=frame,
            target_indices=target,
            particle_types=particle_types,
            box_lengths=box_lengths,
        )
        stable = np.min(np.linalg.eigvalsh(final_hessian), axis=1) > curvature_floor
        valid[frame_index] = stable & (final_energy < initial_energy - 1e-8)
        centers[frame_index, valid[frame_index]] = candidate[valid[frame_index]]
        isotropic_curvatures[frame_index, valid[frame_index]] = (
            np.trace(final_hessian[valid[frame_index]], axis1=1, axis2=2) / 3.0
        )
        basin_depths[frame_index, valid[frame_index]] = (
            initial_energy[valid[frame_index]] - final_energy[valid[frame_index]]
        )
    residual = positions[:, target] - centers
    return {
        "target_indices": target,
        "centers": centers,
        "residual": residual,
        "valid": valid,
        "iterations": iterations,
        "isotropic_curvatures": isotropic_curvatures,
        "basin_depths": basin_depths,
    }


def ka_frozen_neighbor_multistart_minima(
    environment_positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_index: int,
    seed_offsets: np.ndarray,
    curvature_floor: float = 1e-4,
    maximum_step: float = 0.15,
    maximum_iterations: int = 24,
    uniqueness_radius: float = 0.05,
) -> dict[str, np.ndarray | int]:
    """Enumerate stable minima of a frozen KA one-particle potential from fixed seeds.

    The environment is held fixed and the tagged particle's original entry is
    excluded from the pair sum.  Distinct minima are identified modulo the
    periodic box by their final positions, not by their seed labels.
    """

    environment = np.asarray(environment_positions, dtype=float)
    types = np.asarray(particle_types, dtype=int)
    box = np.asarray(box_lengths, dtype=float)
    offsets = np.asarray(seed_offsets, dtype=float)
    if environment.ndim != 2 or environment.shape[1] != 3:
        raise ValueError("environment_positions must have shape (particles, 3)")
    if types.shape != (len(environment),) or np.any((types < 0) | (types > 1)):
        raise ValueError("particle_types must align with environment_positions and contain KA types")
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_lengths must be finite and positive")
    if not isinstance(target_index, (int, np.integer)) or target_index < 0 or target_index >= len(environment):
        raise ValueError("target_index must select an environment particle")
    if offsets.ndim != 2 or offsets.shape[1] != 3 or not len(offsets) or np.any(~np.isfinite(offsets)):
        raise ValueError("seed_offsets must be a nonempty finite (seeds, 3) array")
    if curvature_floor <= 0.0 or maximum_step <= 0.0 or maximum_iterations < 1 or uniqueness_radius <= 0.0:
        raise ValueError("minimization controls must be positive")

    candidate = environment[target_index] + offsets
    target = np.full(len(candidate), target_index, dtype=int)
    for _ in range(maximum_iterations):
        energy, force, hessian = ka_lj_local_energy_force_hessian(
            candidate,
            environment_positions=environment,
            target_indices=target,
            particle_types=types,
            box_lengths=box,
        )
        stable = np.min(np.linalg.eigvalsh(hessian), axis=1) > curvature_floor
        displacement = np.zeros_like(candidate)
        if np.any(stable):
            displacement[stable] = np.linalg.solve(hessian[stable], force[stable, :, None])[:, :, 0]
        norm = np.linalg.norm(displacement, axis=1)
        displacement *= np.minimum(1.0, maximum_step / np.maximum(norm, 1e-12))[:, None]
        trial = candidate + displacement
        trial_energy, _, _ = ka_lj_local_energy_force_hessian(
            trial,
            environment_positions=environment,
            target_indices=target,
            particle_types=types,
            box_lengths=box,
        )
        improved = stable & (trial_energy < energy - 1e-10)
        candidate[improved] = trial[improved]
        if not np.any(improved):
            break

    energy, _, hessian = ka_lj_local_energy_force_hessian(
        candidate,
        environment_positions=environment,
        target_indices=target,
        particle_types=types,
        box_lengths=box,
    )
    stable = np.min(np.linalg.eigvalsh(hessian), axis=1) > curvature_floor
    seed_minimum_index = np.full(len(candidate), -1, dtype=int)
    unique_centers: list[np.ndarray] = []
    unique_energies: list[float] = []
    unique_hessians: list[np.ndarray] = []
    for seed in np.argsort(energy):
        if not stable[seed]:
            continue
        assigned = -1
        for index, center in enumerate(unique_centers):
            displacement = candidate[seed] - center
            displacement -= box * np.rint(displacement / box)
            if np.linalg.norm(displacement) <= uniqueness_radius:
                assigned = index
                break
        if assigned < 0:
            assigned = len(unique_centers)
            unique_centers.append(candidate[seed].copy())
            unique_energies.append(float(energy[seed]))
            unique_hessians.append(hessian[seed].copy())
        seed_minimum_index[seed] = assigned
    return {
        "centers": np.asarray(unique_centers, dtype=float).reshape((-1, 3)),
        "energies": np.asarray(unique_energies, dtype=float),
        "hessians": np.asarray(unique_hessians, dtype=float).reshape((-1, 3, 3)),
        "seed_minimum_index": seed_minimum_index,
        "seed_stable": stable,
        "minimum_count": len(unique_centers),
    }


def driven_tagged_langevin_trajectory(
    bath_positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_index: int,
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    saved_frame_time: float,
    integration_time_step: float,
    mass: float,
    friction: float,
    temperature: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Integrate a tagged particle in the recorded time-dependent KA bath field.

    ``bath_positions`` supplies the external neighbor trajectory. The tagged
    particle is removed from every pair sum, so this is a nonautonomous
    one-particle Langevin experiment with bath feedback deliberately omitted.
    A BAOAB step is used between linearly interpolated saved bath frames.
    """

    bath = np.asarray(bath_positions, dtype=float)
    types = np.asarray(particle_types, dtype=int)
    box = np.asarray(box_lengths, dtype=float)
    position = np.asarray(initial_positions, dtype=float).copy()
    velocity = np.asarray(initial_velocities, dtype=float).copy()
    if bath.ndim != 3 or bath.shape[0] < 2 or bath.shape[2] != 3 or np.any(~np.isfinite(bath)):
        raise ValueError("bath_positions must be finite with shape (at least 2 frames, particles, 3)")
    if types.shape != (bath.shape[1],) or np.any((types < 0) | (types > 1)):
        raise ValueError("particle_types must align with bath_positions and contain KA types")
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_lengths must be finite and positive")
    if not isinstance(target_index, (int, np.integer)) or target_index < 0 or target_index >= bath.shape[1]:
        raise ValueError("target_index must select a bath particle")
    if position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape or not len(position):
        raise ValueError("initial positions and velocities must align as a nonempty (replicas, 3) array")
    if np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity)):
        raise ValueError("initial positions and velocities must be finite")
    if saved_frame_time <= 0.0 or integration_time_step <= 0.0 or mass <= 0.0 or friction < 0.0 or temperature < 0.0:
        raise ValueError("time, mass, friction, and temperature controls must be nonnegative with positive time and mass")
    substeps = int(round(saved_frame_time / integration_time_step))
    if substeps < 1 or not np.isclose(substeps * integration_time_step, saved_frame_time, rtol=1e-10, atol=1e-12):
        raise ValueError("saved_frame_time must be an integer multiple of integration_time_step")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")

    trajectory = np.empty((len(bath), len(position), 3), dtype=float)
    velocities = np.empty_like(trajectory)
    trajectory[0] = position
    velocities[0] = velocity
    target = np.full(len(position), target_index, dtype=int)
    damping = math.exp(-friction * integration_time_step / mass)
    noise_scale = math.sqrt(temperature / mass * max(1.0 - damping * damping, 0.0))
    for frame in range(len(bath) - 1):
        left = bath[frame]
        right = bath[frame + 1]
        for substep in range(substeps):
            fraction = substep / substeps
            next_fraction = (substep + 1) / substeps
            environment = left + fraction * (right - left)
            _, force, _ = ka_lj_local_energy_force_hessian(
                position,
                environment_positions=environment,
                target_indices=target,
                particle_types=types,
                box_lengths=box,
            )
            velocity += 0.5 * integration_time_step * force / mass
            position += 0.5 * integration_time_step * velocity
            velocity = damping * velocity + noise_scale * rng.normal(size=velocity.shape)
            position += 0.5 * integration_time_step * velocity
            next_environment = left + next_fraction * (right - left)
            _, next_force, _ = ka_lj_local_energy_force_hessian(
                position,
                environment_positions=next_environment,
                target_indices=target,
                particle_types=types,
                box_lengths=box,
            )
            velocity += 0.5 * integration_time_step * next_force / mass
        trajectory[frame + 1] = position
        velocities[frame + 1] = velocity
    return {"positions": trajectory, "velocities": velocities}


def _ka_active_cluster_force(
    active_positions: np.ndarray,
    external_positions: np.ndarray,
    *,
    active_particle_types: np.ndarray,
    external_particle_types: np.ndarray,
    box_lengths: np.ndarray,
) -> np.ndarray:
    """Return KA forces on active particles from active and external particles."""

    active = np.asarray(active_positions, dtype=float)
    external = np.asarray(external_positions, dtype=float)
    active_types = np.asarray(active_particle_types, dtype=int)
    external_types = np.asarray(external_particle_types, dtype=int)
    box = np.asarray(box_lengths, dtype=float)
    if active.ndim != 3 or active.shape[2] != 3:
        raise ValueError("active_positions must have shape (replicas, particles, 3)")
    if external.ndim == 2 and external.shape[1:] == (3,):
        external = np.broadcast_to(external, (len(active), *external.shape))
    elif external.ndim != 3 or external.shape[0] != len(active) or external.shape[2] != 3:
        raise ValueError("external_positions must be (particles, 3) or align with active replicas")
    if active_types.shape != (active.shape[1],) or external_types.shape != (external.shape[1],):
        raise ValueError("particle types must align with active and external particle axes")
    force = np.zeros_like(active)
    if len(external):
        displacement = active[:, :, None, :] - external[:, None, :, :]
        displacement -= box * np.rint(displacement / box)
        squared_distance = np.sum(displacement**2, axis=3)
        distance = np.sqrt(np.maximum(squared_distance, 1e-24))
        epsilon = _EPSILON[active_types[:, None], external_types[None, :]]
        sigma = _SIGMA[active_types[:, None], external_types[None, :]]
        active_pair = distance < _CUTOFF_SCALE * sigma
        sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
        sigma_over_r6 = sigma_over_r2**3
        sigma_over_r12 = sigma_over_r6**2
        coefficient = 24.0 * epsilon * (2.0 * sigma_over_r12 - sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        force += np.sum((coefficient * active_pair)[..., None] * displacement, axis=2)
    if active.shape[1] > 1:
        displacement = active[:, :, None, :] - active[:, None, :, :]
        displacement -= box * np.rint(displacement / box)
        squared_distance = np.sum(displacement**2, axis=3)
        distance = np.sqrt(np.maximum(squared_distance, 1e-24))
        epsilon = _EPSILON[active_types[:, None], active_types[None, :]]
        sigma = _SIGMA[active_types[:, None], active_types[None, :]]
        active_pair = (distance < _CUTOFF_SCALE * sigma) & ~np.eye(active.shape[1], dtype=bool)[None, :, :]
        sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
        sigma_over_r6 = sigma_over_r2**3
        sigma_over_r12 = sigma_over_r6**2
        coefficient = 24.0 * epsilon * (2.0 * sigma_over_r12 - sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        force += np.sum((coefficient * active_pair)[..., None] * displacement, axis=2)
    return force


def active_cluster_langevin_residual(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    active_indices: np.ndarray,
    external_indices: np.ndarray,
    frame_time: float,
    friction: float,
) -> dict[str, np.ndarray]:
    """Reconstruct the retained-bath Langevin residual on an active cluster.

    The raw trajectory is partitioned into active particles, an explicitly
    retained external bath, and the omitted complement.  The reported
    ``omitted_force`` is an algebraic force-decomposition error, while
    ``full_residual`` uses every non-active particle and is the discrete
    Langevin residual ``Delta v/dt - mean(F) + gamma mean(v)``.
    """

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box = np.asarray(box_lengths, dtype=float)
    active = np.asarray(active_indices, dtype=int)
    external = np.asarray(external_indices, dtype=int)
    if positions.ndim != 3 or positions.shape[2] != 3 or len(positions) < 2:
        raise ValueError("positions must have shape (at least 2 frames, particles, 3)")
    if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must be finite and match positions")
    if particle_types.shape != (positions.shape[1],) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must align with positions and contain KA types")
    if box.shape != (3,) or np.any(box <= 0.0) or frame_time <= 0.0 or friction < 0.0:
        raise ValueError("box_lengths, frame_time, and friction must be valid")
    for indices, name in ((active, "active_indices"), (external, "external_indices")):
        if indices.ndim != 1 or len(indices) != len(np.unique(indices)) or np.any(indices < 0) or np.any(indices >= positions.shape[1]):
            raise ValueError(f"{name} must be unique valid particle indices")
    if not len(active) or np.intersect1d(active, external).size:
        raise ValueError("the active set must be nonempty and disjoint from the external set")
    all_indices = np.arange(positions.shape[1], dtype=int)
    nonactive = all_indices[~np.isin(all_indices, active)]
    omitted = all_indices[~np.isin(all_indices, np.concatenate([active, external]))]
    active_position = positions[:, active]
    full_force = _ka_active_cluster_force(
        active_position,
        positions[:, nonactive],
        active_particle_types=particle_types[active],
        external_particle_types=particle_types[nonactive],
        box_lengths=box,
    )
    retained_force = _ka_active_cluster_force(
        active_position,
        positions[:, external],
        active_particle_types=particle_types[active],
        external_particle_types=particle_types[external],
        box_lengths=box,
    )
    mean_velocity = 0.5 * (velocities[1:, active] + velocities[:-1, active])
    acceleration = (velocities[1:, active] - velocities[:-1, active]) / frame_time
    full_residual = acceleration - 0.5 * (full_force[1:] + full_force[:-1]) + friction * mean_velocity
    retained_residual = acceleration - 0.5 * (retained_force[1:] + retained_force[:-1]) + friction * mean_velocity
    return {
        "full_force": full_force,
        "retained_force": retained_force,
        "omitted_force": full_force - retained_force,
        "full_residual": full_residual,
        "retained_residual": retained_residual,
        "omitted_indices": omitted,
    }


def driven_active_cluster_langevin_trajectory(
    external_bath_positions: np.ndarray,
    *,
    active_particle_types: np.ndarray,
    external_particle_types: np.ndarray,
    box_lengths: np.ndarray,
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    saved_frame_time: float,
    integration_time_step: float,
    mass: float,
    friction: float,
    temperature: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Integrate an active KA cluster in a recorded external bath trajectory."""

    bath = np.asarray(external_bath_positions, dtype=float)
    active_types = np.asarray(active_particle_types, dtype=int)
    external_types = np.asarray(external_particle_types, dtype=int)
    box = np.asarray(box_lengths, dtype=float)
    position = np.asarray(initial_positions, dtype=float).copy()
    velocity = np.asarray(initial_velocities, dtype=float).copy()
    if bath.ndim != 3 or bath.shape[0] < 2 or bath.shape[2] != 3 or np.any(~np.isfinite(bath)):
        raise ValueError("external_bath_positions must be finite with shape (at least 2 frames, particles, 3)")
    if active_types.shape != (position.shape[1],) or external_types.shape != (bath.shape[1],):
        raise ValueError("active and external particle types must align with their coordinates")
    if np.any((active_types < 0) | (active_types > 1)) or np.any((external_types < 0) | (external_types > 1)):
        raise ValueError("particle types must contain KA types")
    if position.ndim != 3 or position.shape[2] != 3 or velocity.shape != position.shape or not len(position):
        raise ValueError("initial positions and velocities must align as nonempty (replicas, active particles, 3) arrays")
    if np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity)):
        raise ValueError("initial positions and velocities must be finite")
    if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_lengths must be finite and positive")
    if saved_frame_time <= 0.0 or integration_time_step <= 0.0 or mass <= 0.0 or friction < 0.0 or temperature < 0.0:
        raise ValueError("time, mass, friction, and temperature controls must be valid")
    substeps = int(round(saved_frame_time / integration_time_step))
    if substeps < 1 or not np.isclose(substeps * integration_time_step, saved_frame_time, rtol=1e-10, atol=1e-12):
        raise ValueError("saved_frame_time must be an integer multiple of integration_time_step")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")

    trajectory = np.empty((len(bath), *position.shape), dtype=float)
    velocities = np.empty_like(trajectory)
    trajectory[0] = position
    velocities[0] = velocity
    damping = math.exp(-friction * integration_time_step / mass)
    noise_scale = math.sqrt(temperature / mass * max(1.0 - damping * damping, 0.0))
    for frame in range(len(bath) - 1):
        left = bath[frame]
        right = bath[frame + 1]
        for substep in range(substeps):
            fraction = substep / substeps
            next_fraction = (substep + 1) / substeps
            force = _ka_active_cluster_force(
                position,
                left + fraction * (right - left),
                active_particle_types=active_types,
                external_particle_types=external_types,
                box_lengths=box,
            )
            velocity += 0.5 * integration_time_step * force / mass
            position += 0.5 * integration_time_step * velocity
            velocity = damping * velocity + noise_scale * rng.normal(size=velocity.shape)
            position += 0.5 * integration_time_step * velocity
            next_force = _ka_active_cluster_force(
                position,
                left + next_fraction * (right - left),
                active_particle_types=active_types,
                external_particle_types=external_types,
                box_lengths=box,
            )
            velocity += 0.5 * integration_time_step * next_force / mass
        trajectory[frame + 1] = position
        velocities[frame + 1] = velocity
    return {"positions": trajectory, "velocities": velocities}


def active_cluster_source_velocity_ensemble(
    source_velocities: np.ndarray,
    active_indices: np.ndarray,
    *,
    replicas: int,
) -> np.ndarray:
    """Copy a source phase point's selected velocities to each cluster replica."""

    velocity = np.asarray(source_velocities, dtype=float)
    active = np.asarray(active_indices, dtype=int)
    if velocity.ndim != 2 or velocity.shape[1] != 3 or not len(velocity) or not np.all(np.isfinite(velocity)):
        raise ValueError("source_velocities must be a finite (particles, 3) array")
    if active.ndim != 1 or not len(active) or np.any(active < 0) or np.any(active >= len(velocity)) or len(np.unique(active)) != len(active):
        raise ValueError("active_indices must be unique valid source-velocity indices")
    if not isinstance(replicas, (int, np.integer)) or replicas < 1:
        raise ValueError("replicas must be a positive integer")
    return np.broadcast_to(velocity[active], (int(replicas), len(active), 3)).copy()


def ensemble_displacement_observables(displacements: np.ndarray, *, wave_numbers: np.ndarray) -> dict[str, float]:
    """Measure MSD, NGP, and axis-averaged Fs from an ensemble of vectors."""

    displacement = np.asarray(displacements, dtype=float)
    wave = np.asarray(wave_numbers, dtype=float)
    if displacement.ndim < 2 or displacement.shape[-1] != 3 or displacement.size == 0 or not np.all(np.isfinite(displacement)):
        raise ValueError("displacements must be a finite nonempty (..., 3) array")
    if int(np.prod(displacement.shape[:-1])) < 2:
        raise ValueError("at least two displacement samples are required for ensemble observables")
    if wave.ndim != 1 or not len(wave) or not np.all(np.isfinite(wave)) or np.any(wave <= 0.0):
        raise ValueError("wave_numbers must be a nonempty positive finite vector")
    squared = np.sum(displacement**2, axis=-1)
    second = float(np.mean(squared))
    result = {
        "msd": second,
        "ngp": 3.0 * float(np.mean(squared**2)) / (5.0 * second**2) - 1.0 if second > 0.0 else 0.0,
    }
    for wave_number in wave:
        result[f"fs_k_{wave_number:g}"] = float(np.mean(np.cos(wave_number * displacement)))
    return result


def nonlinear_core_harmonic_bath_langevin_trajectory(
    fixed_outer_positions: np.ndarray,
    *,
    core_particle_types: np.ndarray,
    bath_particle_types: np.ndarray,
    fixed_outer_particle_types: np.ndarray,
    box_lengths: np.ndarray,
    initial_core_positions: np.ndarray,
    initial_bath_positions: np.ndarray,
    initial_core_velocities: np.ndarray,
    initial_bath_velocities: np.ndarray,
    bath_initial_forces: np.ndarray,
    bath_core_hessian: np.ndarray,
    bath_hessian: np.ndarray,
    frame_count: int = 3,
    saved_frame_time: float,
    integration_time_step: float,
    mass: float,
    friction: float,
    temperature: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Evolve a nonlinear KA core coupled to a feedback-bearing harmonic shell.

    The core force is evaluated from the complete KA pair potential against the
    moving shell and fixed outer particles.  The shell force is its microscopic
    initial force plus the Hessian linearization in core and shell displacements.
    This realizes a local, feedback-bearing harmonic bath rather than a
    prescribed recorded bath trajectory.
    """

    fixed = np.asarray(fixed_outer_positions, dtype=float)
    core_types = np.asarray(core_particle_types, dtype=int)
    bath_types = np.asarray(bath_particle_types, dtype=int)
    fixed_types = np.asarray(fixed_outer_particle_types, dtype=int)
    box = np.asarray(box_lengths, dtype=float)
    core_position = np.asarray(initial_core_positions, dtype=float).copy()
    bath_position = np.asarray(initial_bath_positions, dtype=float).copy()
    core_velocity = np.asarray(initial_core_velocities, dtype=float).copy()
    bath_velocity = np.asarray(initial_bath_velocities, dtype=float).copy()
    bath_force_zero = np.asarray(bath_initial_forces, dtype=float)
    coupling = np.asarray(bath_core_hessian, dtype=float)
    bath_matrix = np.asarray(bath_hessian, dtype=float)
    if fixed.ndim != 2 or fixed.shape[1:] != (3,) or fixed_types.shape != (len(fixed),):
        raise ValueError("fixed outer positions and particle types must align as (particles, 3)")
    if core_position.ndim != 3 or bath_position.ndim != 3 or core_position.shape[0] != bath_position.shape[0]:
        raise ValueError("core and bath positions must align on the replica axis")
    if core_position.shape[2] != 3 or bath_position.shape[2] != 3:
        raise ValueError("core and bath positions must have Cartesian coordinates")
    if core_velocity.shape != core_position.shape or bath_velocity.shape != bath_position.shape:
        raise ValueError("initial velocities must align with their positions")
    if core_types.shape != (core_position.shape[1],) or bath_types.shape != (bath_position.shape[1],):
        raise ValueError("core and bath particle types must align with their positions")
    bath_dimension = 3 * bath_position.shape[1]
    core_dimension = 3 * core_position.shape[1]
    if bath_force_zero.shape != (bath_position.shape[1], 3):
        raise ValueError("bath_initial_forces must align with bath particles")
    if coupling.shape != (bath_dimension, core_dimension) or bath_matrix.shape != (bath_dimension, bath_dimension):
        raise ValueError("Hessian blocks must align with flattened bath and core coordinates")
    if not np.allclose(bath_matrix, bath_matrix.T, rtol=1e-10, atol=1e-10):
        raise ValueError("bath_hessian must be symmetric")
    if box.shape != (3,) or np.any(box <= 0.0) or np.any(~np.isfinite(box)):
        raise ValueError("box_lengths must be finite and positive")
    if not isinstance(frame_count, (int, np.integer)) or frame_count < 2:
        raise ValueError("frame_count must be an integer of at least two")
    if saved_frame_time <= 0.0 or integration_time_step <= 0.0 or mass <= 0.0 or friction < 0.0 or temperature < 0.0:
        raise ValueError("time, mass, friction, and temperature controls must be valid")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    substeps = int(round(saved_frame_time / integration_time_step))
    if substeps < 1 or not np.isclose(substeps * integration_time_step, saved_frame_time, rtol=1e-10, atol=1e-12):
        raise ValueError("saved_frame_time must be an integer multiple of integration_time_step")

    initial_core = core_position.copy()
    initial_bath = bath_position.copy()
    core_trajectory = np.empty((frame_count, *core_position.shape), dtype=float)
    bath_trajectory = np.empty((frame_count, *bath_position.shape), dtype=float)
    core_velocities = np.empty_like(core_trajectory)
    bath_velocities = np.empty_like(bath_trajectory)
    core_trajectory[0] = core_position
    bath_trajectory[0] = bath_position
    core_velocities[0] = core_velocity
    bath_velocities[0] = bath_velocity
    damping = math.exp(-friction * integration_time_step / mass)
    noise_scale = math.sqrt(temperature / mass * max(1.0 - damping * damping, 0.0))
    external_types = np.concatenate([bath_types, fixed_types])

    def forces() -> tuple[np.ndarray, np.ndarray]:
        fixed_by_replica = np.broadcast_to(fixed, (len(core_position), *fixed.shape))
        external = np.concatenate([bath_position, fixed_by_replica], axis=1)
        core_force = _ka_active_cluster_force(
            core_position,
            external,
            active_particle_types=core_types,
            external_particle_types=external_types,
            box_lengths=box,
        )
        core_displacement = (core_position - initial_core).reshape(len(core_position), core_dimension)
        bath_displacement = (bath_position - initial_bath).reshape(len(bath_position), bath_dimension)
        bath_force = bath_force_zero.reshape(1, bath_dimension)
        bath_force = bath_force - np.einsum("ij,rj->ri", coupling, core_displacement)
        bath_force = bath_force - np.einsum("ij,rj->ri", bath_matrix, bath_displacement)
        return core_force, bath_force.reshape(bath_position.shape)

    for frame in range(frame_count - 1):
        for _ in range(substeps):
            core_force, bath_force = forces()
            core_velocity += 0.5 * integration_time_step * core_force / mass
            bath_velocity += 0.5 * integration_time_step * bath_force / mass
            core_position += 0.5 * integration_time_step * core_velocity
            bath_position += 0.5 * integration_time_step * bath_velocity
            core_velocity = damping * core_velocity + noise_scale * rng.normal(size=core_velocity.shape)
            bath_velocity = damping * bath_velocity + noise_scale * rng.normal(size=bath_velocity.shape)
            core_position += 0.5 * integration_time_step * core_velocity
            bath_position += 0.5 * integration_time_step * bath_velocity
            core_force, bath_force = forces()
            core_velocity += 0.5 * integration_time_step * core_force / mass
            bath_velocity += 0.5 * integration_time_step * bath_force / mass
        core_trajectory[frame + 1] = core_position
        bath_trajectory[frame + 1] = bath_position
        core_velocities[frame + 1] = core_velocity
        bath_velocities[frame + 1] = bath_velocity
    return {
        "core_positions": core_trajectory,
        "bath_positions": bath_trajectory,
        "core_velocities": core_velocities,
        "bath_velocities": bath_velocities,
    }


def local_harmonic_displacement_memory_kernel(
    hessian: np.ndarray,
    *,
    target_local_index: int,
    times: np.ndarray,
    mass: float,
    damping: float,
    bath_eigenvalue_floor: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Eliminate harmonic bath coordinates to obtain a tagged displacement kernel.

    For the block Hessian ``H=[[Hxx,HxB],[HBx,HBB]]``, the kernel is
    ``Lambda(t)=HxB G_B(t) HBx`` where ``G_B`` is the impulse Green function
    of the damped bath. It acts on the tagged displacement in the exact
    harmonic block dynamics before nonlinear basin changes are considered.
    """

    matrix = np.asarray(hessian, dtype=float)
    evaluation_times = np.asarray(times, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 6 or matrix.shape[0] % 3:
        raise ValueError("hessian must be a square 3n-by-3n matrix with at least one bath particle")
    particle_count = matrix.shape[0] // 3
    if not isinstance(target_local_index, (int, np.integer)) or target_local_index < 0 or target_local_index >= particle_count:
        raise ValueError("target_local_index must select a Hessian particle block")
    if evaluation_times.ndim != 1 or not len(evaluation_times) or np.any(~np.isfinite(evaluation_times)) or np.any(evaluation_times < 0.0):
        raise ValueError("times must be a nonempty nonnegative finite vector")
    if mass <= 0.0 or damping < 0.0 or bath_eigenvalue_floor <= 0.0:
        raise ValueError("mass, damping, and bath_eigenvalue_floor must be valid")
    if np.any(~np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10):
        raise ValueError("hessian must be finite and symmetric")

    target_block = np.arange(3 * target_local_index, 3 * target_local_index + 3)
    bath_block = np.setdiff1d(np.arange(matrix.shape[0]), target_block)
    coupling = matrix[np.ix_(target_block, bath_block)]
    bath_hessian = matrix[np.ix_(bath_block, bath_block)]
    eigenvalue, eigenvector = np.linalg.eigh(bath_hessian)
    if np.any(eigenvalue <= bath_eigenvalue_floor):
        raise ValueError("bath Hessian must be positive definite above bath_eigenvalue_floor")

    green = np.empty((len(evaluation_times), len(eigenvalue)), dtype=float)
    decay = damping / (2.0 * mass)
    natural_squared = eigenvalue / mass
    for mode, value in enumerate(natural_squared):
        discriminant = value - decay * decay
        if discriminant > 1e-14:
            frequency = math.sqrt(discriminant)
            green[:, mode] = np.exp(-decay * evaluation_times) * np.sin(frequency * evaluation_times) / (mass * frequency)
        elif discriminant < -1e-14:
            root = math.sqrt(-discriminant)
            rate_plus = -decay + root
            rate_minus = -decay - root
            green[:, mode] = (np.exp(rate_plus * evaluation_times) - np.exp(rate_minus * evaluation_times)) / (mass * (rate_plus - rate_minus))
        else:
            green[:, mode] = evaluation_times * np.exp(-decay * evaluation_times) / mass
    projected_coupling = coupling @ eigenvector
    kernel = np.einsum("am,tm,bm->tab", projected_coupling, green, projected_coupling)
    adiabatic_effective_hessian = matrix[np.ix_(target_block, target_block)] - coupling @ np.linalg.solve(bath_hessian, coupling.T)
    return {
        "memory_kernel": kernel,
        "bath_eigenvalues": eigenvalue,
        "target_hessian_block": matrix[np.ix_(target_block, target_block)],
        "adiabatic_effective_hessian": adiabatic_effective_hessian,
    }


def symmetric_finite_difference_response(
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    displacement: float,
) -> dict[str, np.ndarray]:
    """Return the odd linear response from matched ``+/-`` perturbations."""

    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if positive.shape != negative.shape or positive.ndim < 1:
        raise ValueError("positive and negative observables must have identical non-scalar shapes")
    if np.any(~np.isfinite(positive)) or np.any(~np.isfinite(negative)):
        raise ValueError("positive and negative observables must be finite")
    if not math.isfinite(displacement) or displacement <= 0.0:
        raise ValueError("displacement must be finite and positive")
    return {
        "response": (positive - negative) / (2.0 * displacement),
        "even_component": 0.5 * (positive + negative),
    }


def ensemble_symmetric_response_summary(responses: np.ndarray) -> dict[str, np.ndarray | int]:
    """Return the isoconfigurational mean response and its member standard error."""

    values = np.asarray(responses, dtype=float)
    if values.ndim < 2 or len(values) < 2:
        raise ValueError("responses must contain at least two equally shaped ensemble members")
    if np.any(~np.isfinite(values)):
        raise ValueError("responses must be finite")
    member_count = len(values)
    return {
        "mean": np.mean(values, axis=0),
        "standard_error": np.std(values, axis=0, ddof=1) / math.sqrt(member_count),
        "member_count": member_count,
    }


def shared_response_prefix_length(step_grids: list[np.ndarray]) -> int:
    """Return the longest identical leading timestep grid across members."""

    if len(step_grids) < 2:
        raise ValueError("at least two timestep grids are required")
    grids = [np.asarray(grid, dtype=float) for grid in step_grids]
    if any(grid.ndim != 1 or len(grid) < 2 or np.any(~np.isfinite(grid)) for grid in grids):
        raise ValueError("timestep grids must be finite vectors with at least two frames")
    frame_count = min(len(grid) for grid in grids)
    reference = grids[0][:frame_count]
    if any(not np.array_equal(grid[:frame_count], reference) for grid in grids[1:]):
        raise ValueError("timestep grids do not share an identical leading prefix")
    return frame_count


def infer_causal_position_memory_kernel(
    position_response: np.ndarray,
    force_response: np.ndarray,
    *,
    frame_time: float,
) -> dict[str, np.ndarray | float]:
    """Infer a causal position-memory kernel from matched linear responses.

    The discrete convention is
    ``F_n = -kappa chi_n + dt sum_{ell=1}^n Lambda_ell chi_{n-ell}``.
    The zero-lag memory is excluded, as it belongs to the instantaneous
    curvature term.  Since ``chi_0`` is nonzero for a displacement response,
    the resulting Volterra system is triangular.
    """

    response = np.asarray(position_response, dtype=float)
    force = np.asarray(force_response, dtype=float)
    if response.ndim != 1 or force.shape != response.shape or len(response) < 2:
        raise ValueError("position_response and force_response must be matching vectors with at least two frames")
    if np.any(~np.isfinite(response)) or np.any(~np.isfinite(force)):
        raise ValueError("responses must be finite")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be finite and positive")
    if abs(float(response[0])) <= 1e-14:
        raise ValueError("initial position response must be nonzero")

    curvature = -float(force[0] / response[0])
    memory = np.zeros(len(response), dtype=float)
    source = force + curvature * response
    for frame in range(1, len(response)):
        previous = float(np.dot(memory[1:frame], response[frame - 1 : 0 : -1]))
        memory[frame] = (source[frame] / frame_time - previous) / response[0]
    return {
        "instantaneous_curvature": curvature,
        "memory_kernel": memory,
        "reconstructed_force_response": -curvature * response
        + np.asarray(
            [
                0.0
                if frame == 0
                else frame_time * np.dot(memory[1 : frame + 1], response[frame - 1 :: -1])
                for frame in range(len(response))
            ]
        ),
    }


def propagate_causal_position_memory_response(
    *,
    instantaneous_curvature: float,
    memory_kernel: np.ndarray,
    frame_time: float,
    mass: float,
    friction: float,
    initial_position_response: float = 1.0,
    initial_velocity_response: float = 0.0,
    frame_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Propagate the discrete single-coordinate position-memory equation."""

    kernel = np.asarray(memory_kernel, dtype=float)
    if kernel.ndim != 1 or len(kernel) < 2 or np.any(~np.isfinite(kernel)):
        raise ValueError("memory_kernel must be a finite vector with at least two frames")
    if not all(math.isfinite(value) for value in (instantaneous_curvature, frame_time, mass, friction, initial_position_response, initial_velocity_response)):
        raise ValueError("propagator parameters must be finite")
    if frame_time <= 0.0 or mass <= 0.0 or friction < 0.0:
        raise ValueError("frame_time and mass must be positive and friction nonnegative")
    if frame_count is None:
        frame_count = len(kernel)
    if not isinstance(frame_count, (int, np.integer)) or frame_count < 2:
        raise ValueError("frame_count must be an integer of at least two")

    position = np.empty(frame_count, dtype=float)
    velocity = np.empty(frame_count, dtype=float)
    acceleration = np.empty(frame_count, dtype=float)
    memory_force = np.zeros(frame_count, dtype=float)
    conservative_force = np.empty(frame_count, dtype=float)
    position[0] = initial_position_response
    velocity[0] = initial_velocity_response
    conservative_force[0] = -instantaneous_curvature * position[0]
    acceleration[0] = (conservative_force[0] - friction * velocity[0]) / mass
    for frame in range(frame_count - 1):
        next_frame = frame + 1
        position[next_frame] = position[frame] + frame_time * velocity[frame] + 0.5 * frame_time**2 * acceleration[frame]
        available_lags = min(next_frame, len(kernel) - 1)
        memory_force[next_frame] = frame_time * np.dot(kernel[1 : available_lags + 1], position[frame::-1][:available_lags])
        conservative_force[next_frame] = -instantaneous_curvature * position[next_frame] + memory_force[next_frame]
        velocity[next_frame] = (
            velocity[frame] + 0.5 * frame_time * (acceleration[frame] + conservative_force[next_frame] / mass)
        ) / (1.0 + 0.5 * frame_time * friction / mass)
        acceleration[next_frame] = (conservative_force[next_frame] - friction * velocity[next_frame]) / mass
    return {
        "position_response": position,
        "velocity_response": velocity,
        "acceleration_response": acceleration,
        "memory_force_response": memory_force,
        "conservative_force_response": conservative_force,
    }


def finite_memory_position_response_holdout(
    position_response: np.ndarray,
    pair_force_response: np.ndarray,
    *,
    frame_time: float,
    mass: float,
    friction: float,
    fit_frames: int,
) -> dict[str, np.ndarray | float]:
    """Fit a finite causal memory on an initial window and predict its tail."""

    position = np.asarray(position_response, dtype=float)
    pair_force = np.asarray(pair_force_response, dtype=float)
    if position.ndim != 1 or pair_force.shape != position.shape or len(position) < 3:
        raise ValueError("position_response and pair_force_response must be matching vectors with at least three frames")
    if not isinstance(fit_frames, (int, np.integer)) or not (2 <= fit_frames < len(position)):
        raise ValueError("fit_frames must leave a nonempty temporal holdout")
    inferred = infer_causal_position_memory_kernel(position[:fit_frames], pair_force[:fit_frames], frame_time=frame_time)
    predicted = propagate_causal_position_memory_response(
        instantaneous_curvature=float(inferred["instantaneous_curvature"]),
        memory_kernel=np.asarray(inferred["memory_kernel"]),
        frame_time=frame_time,
        mass=mass,
        friction=friction,
        initial_position_response=float(position[0]),
        initial_velocity_response=0.0,
        frame_count=len(position),
    )["position_response"]
    held = slice(fit_frames, None)
    held_difference = predicted[held] - position[held]
    held_norm = float(np.linalg.norm(position[held]))
    if held_norm <= 0.0:
        raise ValueError("held position response must be nonzero")
    return {
        "instantaneous_curvature": float(inferred["instantaneous_curvature"]),
        "memory_kernel": np.asarray(inferred["memory_kernel"]),
        "predicted_position_response": predicted,
        "training_relative_l2_error": float(np.linalg.norm(predicted[:fit_frames] - position[:fit_frames]) / np.linalg.norm(position[:fit_frames])),
        "heldout_relative_l2_error": float(np.linalg.norm(held_difference) / held_norm),
        "heldout_maximum_absolute_error": float(np.max(np.abs(held_difference))),
    }


def fit_linear_response_auxiliary_embedding(
    state_response: np.ndarray,
    *,
    fit_frames: int,
) -> dict[str, np.ndarray | float]:
    """Fit and propagate a finite-dimensional linear response embedding.

    The discrete hypothesis is ``s_(n+1) = M s_n``.  Its first coordinate is
    the response position; any remaining coordinates are explicit bath modes.
    The fit is performed on an initial segment and propagated without
    refitting over the remaining frames.
    """

    state = np.asarray(state_response, dtype=float)
    if state.ndim != 2 or state.shape[1] < 2 or len(state) < 5:
        raise ValueError("state_response must be a finite (frames, coordinate_count) array with at least two coordinates")
    if np.any(~np.isfinite(state)):
        raise ValueError("state_response must be finite")
    if not isinstance(fit_frames, (int, np.integer)) or not (4 <= fit_frames < len(state)):
        raise ValueError("fit_frames must leave an out-of-sample response interval")

    training = state[:fit_frames]
    scale = np.sqrt(np.mean(training**2, axis=0))
    if np.any(scale <= 1e-14):
        raise ValueError("each response coordinate must vary on the fitting interval")
    scaled_training = training / scale
    coefficient, _, rank, _ = np.linalg.lstsq(scaled_training[:-1], scaled_training[1:], rcond=None)
    if rank < state.shape[1]:
        raise ValueError("training response does not identify the requested auxiliary transition")
    scaled_transition = coefficient.T
    transition = np.diag(scale) @ scaled_transition @ np.diag(1.0 / scale)
    predicted = np.empty_like(state)
    predicted[0] = state[0]
    for frame in range(1, len(state)):
        predicted[frame] = transition @ predicted[frame - 1]
    training_norm = float(np.linalg.norm(state[:fit_frames, 0]))
    held_norm = float(np.linalg.norm(state[fit_frames:, 0]))
    if training_norm <= 0.0 or held_norm <= 0.0:
        raise ValueError("position response must be nonzero in training and held-out windows")
    return {
        "transition_matrix": transition,
        "predicted_state_response": predicted,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
        "training_position_relative_l2_error": float(
            np.linalg.norm(predicted[:fit_frames, 0] - state[:fit_frames, 0]) / training_norm
        ),
        "heldout_position_relative_l2_error": float(
            np.linalg.norm(predicted[fit_frames:, 0] - state[fit_frames:, 0]) / held_norm
        ),
        "heldout_position_maximum_absolute_error": float(
            np.max(np.abs(predicted[fit_frames:, 0] - state[fit_frames:, 0]))
        ),
    }


def fit_pair_force_response_auxiliary_embedding(
    state_response: np.ndarray,
    *,
    fit_frames: int,
) -> dict[str, np.ndarray | float]:
    """Fit the physical ``(position, velocity, pair-force)`` response state."""

    state = np.asarray(state_response, dtype=float)
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError("state_response must have shape (frames, 3): position, velocity, pair force")
    return fit_linear_response_auxiliary_embedding(state, fit_frames=fit_frames)


def frozen_minimum_response_jacobian(
    positions: np.ndarray,
    *,
    centers: np.ndarray,
    valid: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    conditioning_maximum: float = 1e8,
) -> dict[str, np.ndarray]:
    """Return ``dC_i/dr_j`` for the KA frozen-minimum coordinate.

    At a stable local minimum, implicit differentiation of the exact pair
    potential gives ``J_ij = H_ii^-1 H_ij^(pair)``.  The tagged-particle
    block is explicitly zero because the frozen-minimum coordinate depends
    only on the environment.  This full tensor is intentionally separate from
    the lower-memory increment response routine.
    """

    positions = np.asarray(positions, dtype=float)
    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (at least 2 frames, particles, 3)")
    if centers.shape != (positions.shape[0], len(target), 3) or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align with positions and target_indices")
    if particle_types.shape != (positions.shape[1],) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be 0/1 and align with positions")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= positions.shape[1]):
        raise ValueError("target_indices must select valid particles")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if not math.isfinite(conditioning_maximum) or conditioning_maximum <= 1.0:
        raise ValueError("conditioning_maximum must be finite and greater than one")

    jacobian = np.zeros(
        (positions.shape[0] - 1, len(target), positions.shape[1], 3, 3), dtype=float
    )
    response_valid = np.zeros(jacobian.shape[:2], dtype=bool)
    hessian_condition = np.full(jacobian.shape[:2], np.nan, dtype=float)
    identity = np.eye(3)
    for frame_index in range(positions.shape[0] - 1):
        usable = valid[frame_index] & valid[frame_index + 1] & np.isfinite(centers[frame_index]).all(axis=1)
        if not np.any(usable):
            continue
        frame = positions[frame_index]
        candidate = centers[frame_index]
        displacement = np.mod(candidate[:, None, :], box_lengths) - np.mod(frame[None, :, :], box_lengths)
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        squared_distance = np.sum(displacement**2, axis=2)
        distance = np.sqrt(np.maximum(squared_distance, 1e-24))
        epsilon = _EPSILON[particle_types[target, None], particle_types[None, :]]
        sigma = _SIGMA[particle_types[target, None], particle_types[None, :]]
        active = (distance > 1e-10) & (distance < _CUTOFF_SCALE * sigma)
        active[np.arange(len(target)), target] = False
        sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
        sigma_over_r6 = sigma_over_r2**3
        sigma_over_r12 = sigma_over_r6**2
        potential_prime_over_r = (
            24.0 * epsilon * (-2.0 * sigma_over_r12 + sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        ) * active
        potential_second = (
            24.0 * epsilon * (26.0 * sigma_over_r12 - 7.0 * sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        ) * active
        unit = displacement / np.maximum(distance[:, :, None], 1e-12)
        pair_hessian = (
            (potential_second - potential_prime_over_r)[:, :, None, None]
            * unit[:, :, :, None]
            * unit[:, :, None, :]
            + potential_prime_over_r[:, :, None, None] * identity
        )
        local_hessian = np.sum(pair_hessian, axis=1)
        for local_index in np.flatnonzero(usable):
            condition = float(np.linalg.cond(local_hessian[local_index]))
            hessian_condition[frame_index, local_index] = condition
            if math.isfinite(condition) and condition <= conditioning_maximum:
                jacobian[frame_index, local_index] = np.linalg.solve(
                    local_hessian[local_index], pair_hessian[local_index]
                )
                response_valid[frame_index, local_index] = True
    return {
        "jacobian": jacobian,
        "response_valid": response_valid,
        "hessian_condition": hessian_condition,
    }


def projected_overdamped_diffusion_tensor(
    jacobian: np.ndarray,
    *,
    temperature: float,
    bare_mobility: float = 1.0,
) -> dict[str, np.ndarray]:
    """Apply the exact Ito diffusion map for an overdamped many-particle bath.

    If every microscopic particle follows
    ``dr_j = -mu grad_j U dt + sqrt(2 mu T) dW_j``, then the frozen-minimum
    coordinate has instantaneous diffusion tensor
    ``D_C(R) = mu T sum_j J_ij(R) J_ij(R)^T``.  This routine deliberately
    performs only that local kinematic projection; it does not claim that the
    tensor is a closed function of ``C_i`` alone.
    """

    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim < 3 or jacobian.shape[-2:] != (3, 3):
        raise ValueError("jacobian must end in (particles, 3, 3)")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(bare_mobility) or bare_mobility <= 0.0:
        raise ValueError("bare_mobility must be positive and finite")
    diffusion = bare_mobility * temperature * np.einsum(
        "...pab,...pcb->...ac", jacobian, jacobian
    )
    eigenvalues = np.linalg.eigvalsh(diffusion)
    isotropic = np.trace(diffusion, axis1=-2, axis2=-1) / 3.0
    smallest = eigenvalues[..., 0]
    anisotropy = np.where(smallest > 0.0, eigenvalues[..., -1] / smallest, math.inf)
    return {
        "diffusion_tensor": diffusion,
        "isotropic_diffusion": isotropic,
        "anisotropy_ratio": anisotropy,
        "diffusion_eigenvalues": eigenvalues,
    }


def projected_overdamped_mixture_ngp(diffusion_tensor: np.ndarray) -> dict[str, float]:
    """Return the infinitesimal NGP implied by a frozen diffusion-tensor mixture.

    Conditional on a configuration, an overdamped projected coordinate has a
    Gaussian increment with covariance ``2 D_C(R) dt``.  Averaging over a
    frozen ensemble of such tensors yields the exact short-time mixture value
    reported here.  Any larger observed NGP requires temporal correlations,
    non-Gaussian conditional noise, or a failure of the frozen-mixture limit.
    """

    tensor = np.asarray(diffusion_tensor, dtype=float)
    if tensor.ndim < 3 or tensor.shape[-2:] != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("diffusion_tensor must be finite and end in (3, 3)")
    flattened = tensor.reshape(-1, 3, 3)
    trace = np.trace(flattened, axis1=1, axis2=2)
    trace_square = np.einsum("nij,nji->n", flattened, flattened)
    mean_trace = float(np.mean(trace))
    if mean_trace <= 0.0:
        raise ValueError("diffusion_tensor must have positive mean trace")
    fourth_moment_core = float(np.mean(trace**2 + 2.0 * trace_square))
    predicted_ngp = 3.0 * fourth_moment_core / (5.0 * mean_trace**2) - 1.0
    isotropic = trace / 3.0
    return {
        "predicted_infinitesimal_ngp": float(predicted_ngp),
        "mean_isotropic_diffusion": float(np.mean(isotropic)),
        "isotropic_diffusion_cv2": float(np.var(isotropic) / np.mean(isotropic) ** 2),
    }


def projected_diffusion_increment_association(
    diffusion_tensor: np.ndarray,
    center_increment: np.ndarray,
    *,
    tail_fraction: float = 0.05,
) -> dict[str, float]:
    """Test the unfit conditional-mobility prediction of a projected bath.

    The overdamped Ito projection predicts that configurations with larger
    ``tr(D_C)`` have larger conditional mean-squared increments.  This
    diagnostic deliberately uses only a rank association and a fixed upper
    tail ratio, so it does not estimate a mobility scale from the observed
    displacement.
    """

    tensor = np.asarray(diffusion_tensor, dtype=float)
    increment = np.asarray(center_increment, dtype=float)
    if tensor.ndim < 3 or tensor.shape[-2:] != (3, 3) or not np.all(np.isfinite(tensor)):
        raise ValueError("diffusion_tensor must be finite and end in (3, 3)")
    if increment.shape != tensor.shape[:-2] + (3,) or not np.all(np.isfinite(increment)):
        raise ValueError("center_increment must align with diffusion_tensor leading dimensions")
    if not math.isfinite(tail_fraction) or not (0.0 < tail_fraction <= 0.5):
        raise ValueError("tail_fraction must lie in (0, 0.5]")
    isotropic = np.trace(tensor, axis1=-2, axis2=-1).reshape(-1) / 3.0
    squared_increment = np.sum(increment.reshape(-1, 3) ** 2, axis=1)
    if len(isotropic) < 2 or np.ptp(isotropic) <= 0.0 or np.ptp(squared_increment) <= 0.0:
        rank_correlation = math.nan
    else:
        rank_diffusion = np.empty(len(isotropic), dtype=float)
        rank_increment = np.empty(len(squared_increment), dtype=float)
        rank_diffusion[np.argsort(isotropic, kind="stable")] = np.arange(len(isotropic), dtype=float)
        rank_increment[np.argsort(squared_increment, kind="stable")] = np.arange(len(squared_increment), dtype=float)
        rank_correlation = float(np.corrcoef(rank_diffusion, rank_increment)[0, 1])
    tail_count = max(1, int(math.ceil(tail_fraction * len(isotropic))))
    upper_tail = np.argsort(isotropic, kind="stable")[-tail_count:]
    mean_increment = float(np.mean(squared_increment))
    return {
        "rank_correlation": rank_correlation,
        "upper_tail_increment_ratio": float(np.mean(squared_increment[upper_tail]) / mean_increment)
        if mean_increment > 0.0
        else math.nan,
        "tail_fraction": float(tail_fraction),
    }


def frozen_minimum_environment_response(
    positions: np.ndarray,
    *,
    centers: np.ndarray,
    valid: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    conditioning_maximum: float = 1e8,
) -> dict[str, np.ndarray]:
    """Differentiate a frozen-neighbor KA minimum with respect to its environment.

    For ``C_i(R) = argmin_x U_KA(x | R_{-i})``, stationarity gives

    ``H_ii dC_i = sum_j H_ij^(pair) dr_j``.

    The returned increment is this parameter-free, infinitesimal response
    evaluated at each saved frame.  ``positions`` must be unwrapped so that
    frame-to-frame environmental displacements are continuous.
    """

    positions = np.asarray(positions, dtype=float)
    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (at least 2 frames, particles, 3)")
    if centers.shape != (positions.shape[0], len(target), 3) or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align with positions and target_indices")
    if particle_types.shape != (positions.shape[1],) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be 0/1 and align with positions")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= positions.shape[1]):
        raise ValueError("target_indices must select valid particles")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if not math.isfinite(conditioning_maximum) or conditioning_maximum <= 1.0:
        raise ValueError("conditioning_maximum must be finite and greater than one")

    predicted = np.full((positions.shape[0] - 1, len(target), 3), np.nan, dtype=float)
    response_valid = np.zeros(predicted.shape[:2], dtype=bool)
    hessian_condition = np.full(predicted.shape[:2], np.nan, dtype=float)
    identity = np.eye(3)
    for frame_index in range(positions.shape[0] - 1):
        usable = valid[frame_index] & valid[frame_index + 1] & np.isfinite(centers[frame_index]).all(axis=1)
        if not np.any(usable):
            continue
        frame = positions[frame_index]
        candidate = centers[frame_index]
        displacement = np.mod(candidate[:, None, :], box_lengths) - np.mod(frame[None, :, :], box_lengths)
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        squared_distance = np.sum(displacement**2, axis=2)
        distance = np.sqrt(np.maximum(squared_distance, 1e-24))
        epsilon = _EPSILON[particle_types[target, None], particle_types[None, :]]
        sigma = _SIGMA[particle_types[target, None], particle_types[None, :]]
        active = (distance > 1e-10) & (distance < _CUTOFF_SCALE * sigma)
        active[np.arange(len(target)), target] = False
        sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
        sigma_over_r6 = sigma_over_r2**3
        sigma_over_r12 = sigma_over_r6**2
        potential_prime_over_r = (
            24.0 * epsilon * (-2.0 * sigma_over_r12 + sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        ) * active
        potential_second = (
            24.0 * epsilon * (26.0 * sigma_over_r12 - 7.0 * sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        ) * active
        unit = displacement / np.maximum(distance[:, :, None], 1e-12)
        pair_hessian = (
            (potential_second - potential_prime_over_r)[:, :, None, None]
            * unit[:, :, :, None]
            * unit[:, :, None, :]
            + potential_prime_over_r[:, :, None, None] * identity
        )
        local_hessian = np.sum(pair_hessian, axis=1)
        environment_increment = positions[frame_index + 1] - positions[frame_index]
        numerator = np.einsum("tnab,nb->ta", pair_hessian, environment_increment)
        for local_index in np.flatnonzero(usable):
            condition = float(np.linalg.cond(local_hessian[local_index]))
            hessian_condition[frame_index, local_index] = condition
            if math.isfinite(condition) and condition <= conditioning_maximum:
                predicted[frame_index, local_index] = np.linalg.solve(
                    local_hessian[local_index], numerator[local_index]
                )
                response_valid[frame_index, local_index] = True
    return {
        "predicted_center_increment": predicted,
        "response_valid": response_valid,
        "hessian_condition": hessian_condition,
    }


def frozen_minimum_shell_environment_response(
    positions: np.ndarray,
    *,
    centers: np.ndarray,
    valid: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    shell_edges: np.ndarray,
    conditioning_maximum: float = 1e8,
) -> dict[str, np.ndarray]:
    """Decompose the exact frozen-minimum response into radial pair shells."""

    positions = np.asarray(positions, dtype=float)
    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    target = np.asarray(target_indices, dtype=int)
    shell_edges = np.asarray(shell_edges, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (at least 2 frames, particles, 3)")
    if centers.shape != (positions.shape[0], len(target), 3) or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align with positions and target_indices")
    if particle_types.shape != (positions.shape[1],) or np.any((particle_types < 0) | (particle_types > 1)):
        raise ValueError("particle_types must be 0/1 and align with positions")
    if target.ndim != 1 or not len(target) or np.any(target < 0) or np.any(target >= positions.shape[1]):
        raise ValueError("target_indices must select valid particles")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be a finite positive three-vector")
    if shell_edges.ndim != 1 or len(shell_edges) < 2 or shell_edges[0] != 0.0 or np.any(~np.isfinite(shell_edges)) or np.any(np.diff(shell_edges) <= 0.0):
        raise ValueError("shell_edges must start at zero and increase strictly")
    if not math.isfinite(conditioning_maximum) or conditioning_maximum <= 1.0:
        raise ValueError("conditioning_maximum must be finite and greater than one")

    shell_response = np.full((len(positions) - 1, len(target), len(shell_edges) - 1, 3), np.nan, dtype=float)
    response_valid = np.zeros(shell_response.shape[:2], dtype=bool)
    hessian_condition = np.full(shell_response.shape[:2], np.nan, dtype=float)
    identity = np.eye(3)
    for frame_index, frame in enumerate(positions[:-1]):
        usable = valid[frame_index] & valid[frame_index + 1] & np.isfinite(centers[frame_index]).all(axis=1)
        if not np.any(usable):
            continue
        candidate = centers[frame_index]
        displacement = np.mod(candidate[:, None, :], box_lengths) - np.mod(frame[None, :, :], box_lengths)
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        squared_distance = np.sum(displacement**2, axis=2)
        distance = np.sqrt(np.maximum(squared_distance, 1e-24))
        epsilon = _EPSILON[particle_types[target, None], particle_types[None, :]]
        sigma = _SIGMA[particle_types[target, None], particle_types[None, :]]
        active = (distance > 1e-10) & (distance < _CUTOFF_SCALE * sigma)
        active[np.arange(len(target)), target] = False
        sigma_over_r2 = (sigma / np.maximum(distance, 1e-12)) ** 2
        sigma_over_r6 = sigma_over_r2**3
        sigma_over_r12 = sigma_over_r6**2
        potential_prime_over_r = (
            24.0 * epsilon * (-2.0 * sigma_over_r12 + sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        ) * active
        potential_second = (
            24.0 * epsilon * (26.0 * sigma_over_r12 - 7.0 * sigma_over_r6) / np.maximum(squared_distance, 1e-24)
        ) * active
        unit = displacement / np.maximum(distance[:, :, None], 1e-12)
        pair_hessian = (
            (potential_second - potential_prime_over_r)[:, :, None, None]
            * unit[:, :, :, None]
            * unit[:, :, None, :]
            + potential_prime_over_r[:, :, None, None] * identity
        )
        local_hessian = np.sum(pair_hessian, axis=1)
        environment_increment = positions[frame_index + 1] - frame
        for local_index in np.flatnonzero(usable):
            condition = float(np.linalg.cond(local_hessian[local_index]))
            hessian_condition[frame_index, local_index] = condition
            if not (math.isfinite(condition) and condition <= conditioning_maximum):
                continue
            pair_drive = np.einsum("nab,nb->na", pair_hessian[local_index], environment_increment)
            pair_response = np.linalg.solve(local_hessian[local_index], pair_drive.T).T
            shell_index = np.searchsorted(shell_edges, distance[local_index], side="right") - 1
            for shell in range(len(shell_edges) - 1):
                in_shell = active[local_index] & (shell_index == shell)
                shell_response[frame_index, local_index, shell] = np.sum(pair_response[in_shell], axis=0)
            response_valid[frame_index, local_index] = True
    return {
        "shell_response": shell_response,
        "response_valid": response_valid,
        "hessian_condition": hessian_condition,
    }


def minimum_response_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    predicted_center_increment: np.ndarray,
    response_valid: np.ndarray,
    *,
    train_stop: int,
) -> dict[str, float]:
    """Evaluate the parameter-free minimum-response formula on a time holdout."""

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    predicted = np.asarray(predicted_center_increment, dtype=float)
    response_valid = np.asarray(response_valid, dtype=bool)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if predicted.shape != (centers.shape[0] - 1, centers.shape[1], 3) or response_valid.shape != predicted.shape[:2]:
        raise ValueError("response arrays must contain one increment per center interval")
    if not (2 < train_stop < centers.shape[0] - 2):
        raise ValueError("train_stop must leave at least two increments in each segment")

    actual = centers[1:] - centers[:-1]
    usable = valid[:-1] & valid[1:] & response_valid
    usable &= np.isfinite(actual).all(axis=2) & np.isfinite(predicted).all(axis=2)
    if not np.any(usable[: train_stop - 1]) or not np.any(usable[train_stop - 1 :]):
        raise ValueError("both time segments require valid response increments")

    def metrics(
        observed_vectors: np.ndarray,
        response_vectors: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[float, float, float, float]:
        observed = observed_vectors[mask].reshape(-1)
        response = response_vectors[mask].reshape(-1)
        residual = observed - response
        total = float(np.sum((observed - np.mean(observed)) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else math.nan
        response_norm = float(np.sum(response**2))
        scale = float(np.dot(response, observed) / response_norm) if response_norm > 0.0 else math.nan
        squared = np.sum((observed_vectors[mask] - response_vectors[mask]) ** 2, axis=1)
        residual_ngp = (
            3.0 * float(np.mean(squared**2)) / (5.0 * float(np.mean(squared)) ** 2) - 1.0
            if float(np.mean(squared)) > 0.0
            else 0.0
        )
        return r_squared, scale, residual_ngp, float(np.mean(mask))

    train = usable[: train_stop - 1]
    held = usable[train_stop - 1 :]
    split = train_stop - 1
    train_r2, train_scale, train_ngp, train_fraction = metrics(
        actual[:split], predicted[:split], train
    )
    held_r2, held_scale, held_ngp, held_fraction = metrics(
        actual[split:], predicted[split:], held
    )
    return {
        "training_response_r_squared": train_r2,
        "training_response_scale": train_scale,
        "training_response_residual_ngp": train_ngp,
        "training_response_valid_fraction": train_fraction,
        "heldout_response_r_squared": held_r2,
        "heldout_response_scale": held_scale,
        "heldout_response_residual_ngp": held_ngp,
        "heldout_response_valid_fraction": held_fraction,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def response_residual_memory_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    predicted_center_increment: np.ndarray,
    response_valid: np.ndarray,
    *,
    frame_time: float,
    white_noise_correlation_maximum: float = 0.05,
) -> dict[str, float]:
    """Diagnose memory and NGP after subtracting the local-potential response."""

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    predicted = np.asarray(predicted_center_increment, dtype=float)
    response_valid = np.asarray(response_valid, dtype=bool)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if predicted.shape != (centers.shape[0] - 1, centers.shape[1], 3) or response_valid.shape != predicted.shape[:2]:
        raise ValueError("response arrays must contain one increment per center interval")
    response_residual = centers[1:] - centers[:-1] - predicted
    usable = valid[:-1] & valid[1:] & response_valid
    usable &= np.isfinite(response_residual).all(axis=2)
    response_residual[~usable] = np.nan
    diagnostic = residual_ar1_memory_diagnostic(
        response_residual,
        frame_time=frame_time,
        white_noise_correlation_maximum=white_noise_correlation_maximum,
    )
    squared = np.sum(response_residual[usable] ** 2, axis=1)
    second = float(np.mean(squared))
    diagnostic.update(
        {
            "response_residual_ngp": float(3.0 * np.mean(squared**2) / (5.0 * second**2) - 1.0)
            if second > 0.0
            else 0.0,
            "response_residual_valid_fraction": float(np.mean(usable)),
        }
    )
    return diagnostic


def time_split_driven_response_empirical_innovation_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    predicted_center_increment: np.ndarray,
    response_valid: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    seed: int = 20260806,
) -> dict[str, float | np.ndarray]:
    """Predict held center statistics from exact bath response plus trained residual draws.

    The held response increment is supplied by the actual bath path through
    the parameter-free Hessian formula. Only the residual increment law is
    estimated from the training half; no macroscopic observable is fitted.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    response = np.asarray(predicted_center_increment, dtype=float)
    response_valid = np.asarray(response_valid, dtype=bool)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if response.shape != (centers.shape[0] - 1, centers.shape[1], 3) or response_valid.shape != response.shape[:2]:
        raise ValueError("response arrays must contain one increment per center interval")
    if not (2 < train_stop < len(centers) - 2) or frame_time <= 0.0:
        raise ValueError("train_stop and frame_time must leave train and held intervals")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(centers) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be positive")

    actual = centers[1:] - centers[:-1]
    interval_valid = valid[:-1] & valid[1:] & response_valid
    interval_valid &= np.isfinite(actual).all(axis=2) & np.isfinite(response).all(axis=2)
    residual = actual - response
    train_valid = interval_valid[: train_stop - 1]
    train_residual = residual[: train_stop - 1][train_valid]
    if len(train_residual) < 10:
        raise ValueError("training segment has insufficient valid residual increments")

    held_centers = centers[train_stop:]
    held_response = response[train_stop:]
    held_interval_valid = interval_valid[train_stop:]
    prefix_invalid = np.concatenate(
        [np.zeros((1, centers.shape[1]), dtype=int), np.cumsum(~held_interval_valid, axis=0)], axis=0
    )
    response_prefix = np.concatenate(
        [np.zeros((1, centers.shape[1], 3), dtype=float), np.cumsum(held_response, axis=0)], axis=0
    )
    rng = np.random.default_rng(seed)
    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    predicted_ngp: list[float] = []
    observed_ngp: list[float] = []
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        path_valid = prefix_invalid[lag:] - prefix_invalid[:-lag] == 0
        if not np.any(path_valid):
            raise ValueError("held-out segment has no fully valid bath-driven paths")
        driving = response_prefix[lag:] - response_prefix[:-lag]
        observed = held_centers[lag:] - held_centers[:-lag]
        selected_driving = driving[path_valid]
        selected_observed = observed[path_valid]
        noise = train_residual[rng.integers(len(train_residual), size=(len(selected_driving), int(lag)))].sum(axis=1)
        predicted = selected_driving + noise
        predicted_squared = np.sum(predicted**2, axis=1)
        observed_squared = np.sum(selected_observed**2, axis=1)
        predicted_msd.append(float(np.mean(predicted_squared)))
        observed_msd.append(float(np.mean(observed_squared)))
        predicted_ngp.append(float(3.0 * np.mean(predicted_squared**2) / (5.0 * np.mean(predicted_squared) ** 2) - 1.0))
        observed_ngp.append(float(3.0 * np.mean(observed_squared**2) / (5.0 * np.mean(observed_squared) ** 2) - 1.0))
        for wave_number in wave_numbers:
            predicted_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * predicted))))
            observed_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * selected_observed))))
    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    predicted_ngp_array = np.asarray(predicted_ngp)
    observed_ngp_array = np.asarray(observed_ngp)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_errors = [
        abs(predicted / observed - 1.0)
        for wave_number in wave_numbers
        for predicted, observed in zip(predicted_fs[float(wave_number)], observed_fs[float(wave_number)])
        if abs(observed) > 1e-12
    ]
    return {
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "driven_response_fs_max_relative_error": float(max(fs_errors)) if fs_errors else math.nan,
        "driven_response_ngp_max_absolute_error": float(np.max(np.abs(predicted_ngp_array - observed_ngp_array))),
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "predicted_ngp": predicted_ngp_array,
        "observed_ngp": observed_ngp_array,
        "training_residual_sample_count": float(len(train_residual)),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_graph_conditioned_increment_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    graph_state: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    seed: int = 20260812,
) -> dict[str, float | np.ndarray]:
    """Held-out coordinate closure from empirical increments conditioned on graph state.

    ``graph_state[t]`` is a causal state available at the beginning of the
    increment from ``t`` to ``t+1``.  Only its binary zero/nonzero class is
    used, making this a minimal test of whether graph reconnections supply the
    missing orthogonal-process state.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    state = np.asarray(graph_state, dtype=float)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2] or state.shape != centers.shape[:2]:
        raise ValueError("centers, valid, and graph_state must align as (frames, particles, 3)/(frames, particles)")
    if not (2 < train_stop < len(centers) - 2) or frame_time <= 0.0:
        raise ValueError("train_stop and frame_time must leave train and held intervals")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(centers) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be positive")
    increments = centers[1:] - centers[:-1]
    interval_valid = valid[:-1] & valid[1:] & np.isfinite(increments).all(axis=2) & np.isfinite(state[:-1])
    state_class = state[:-1] > 0.0
    train_valid = interval_valid[: train_stop - 1]
    train_increment = increments[: train_stop - 1]
    samples = [train_increment[train_valid & (state_class[: train_stop - 1] == value)] for value in (False, True)]
    if any(len(value) < 10 for value in samples):
        raise ValueError("each graph-state class needs at least ten training increments")
    held_centers = centers[train_stop:]
    held_valid = interval_valid[train_stop:]
    held_state = state_class[train_stop:]
    prefix_invalid = np.concatenate(
        [np.zeros((1, centers.shape[1]), dtype=int), np.cumsum(~held_valid, axis=0)], axis=0
    )
    rng = np.random.default_rng(seed)
    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    predicted_ngp: list[float] = []
    observed_ngp: list[float] = []
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        path_valid = prefix_invalid[lag:] - prefix_invalid[:-lag] == 0
        start, particle = np.nonzero(path_valid)
        if len(start) == 0:
            raise ValueError("held-out segment has no fully valid paths")
        time_index = start[:, None] + np.arange(int(lag))[None, :]
        classes = held_state[time_index, particle[:, None]].reshape(-1)
        noise = np.empty((len(classes), 3), dtype=float)
        for class_index, value in enumerate((False, True)):
            selected = classes == value
            noise[selected] = samples[class_index][rng.integers(len(samples[class_index]), size=np.count_nonzero(selected))]
        predicted = np.sum(noise.reshape(len(start), int(lag), 3), axis=1)
        observed = held_centers[start + int(lag), particle] - held_centers[start, particle]
        predicted_squared = np.sum(predicted**2, axis=1)
        observed_squared = np.sum(observed**2, axis=1)
        predicted_msd.append(float(np.mean(predicted_squared)))
        observed_msd.append(float(np.mean(observed_squared)))
        predicted_ngp.append(float(3.0 * np.mean(predicted_squared**2) / (5.0 * np.mean(predicted_squared) ** 2) - 1.0))
        observed_ngp.append(float(3.0 * np.mean(observed_squared**2) / (5.0 * np.mean(observed_squared) ** 2) - 1.0))
        for wave_number in wave_numbers:
            predicted_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * predicted))))
            observed_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * observed))))
    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    predicted_ngp_array = np.asarray(predicted_ngp)
    observed_ngp_array = np.asarray(observed_ngp)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_errors = [
        abs(predicted / observed - 1.0)
        for wave_number in wave_numbers
        for predicted, observed in zip(predicted_fs[float(wave_number)], observed_fs[float(wave_number)])
        if abs(observed) > 1e-12
    ]
    return {
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "graph_conditioned_fs_max_relative_error": float(max(fs_errors)) if fs_errors else math.nan,
        "graph_conditioned_ngp_max_absolute_error": float(np.max(np.abs(predicted_ngp_array - observed_ngp_array))),
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "predicted_ngp": predicted_ngp_array,
        "observed_ngp": observed_ngp_array,
        "training_graph_change_fraction": float(np.mean(state_class[: train_stop - 1][train_valid])),
        "training_nochange_sample_count": float(len(samples[0])),
        "training_change_sample_count": float(len(samples[1])),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def lagged_coordinate_increment_history(centers: np.ndarray, *, order: int) -> np.ndarray:
    """Return a causal recent-first increment history for each coordinate frame.

    Frame ``t`` contains ``[C(t)-C(t-1), ..., C(t-order+1)-C(t-order)]``.
    Frames without a complete history are non-finite so downstream held-out
    diagnostics exclude them rather than silently using future information.
    """

    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or len(centers) < 2:
        raise ValueError("centers must have shape (at least 2 frames, particles, 3)")
    if order < 1 or order >= len(centers):
        raise ValueError("order must fit inside the coordinate history")
    increments = centers[1:] - centers[:-1]
    history = np.full((len(centers), centers.shape[1], 3 * order), np.nan, dtype=float)
    for frame_index in range(order, len(centers)):
        recent = increments[frame_index - order : frame_index][::-1]
        history[frame_index] = recent.transpose(1, 0, 2).reshape(centers.shape[1], -1)
    return history


def time_split_knn_state_conditioned_increment_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    state_features: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    neighbor_count: int = 24,
    seed: int = 20260813,
) -> dict[str, float | np.ndarray]:
    """Held-out increment closure from a finite local microscopic state vector.

    The model uses standardized Euclidean KNN only to sample a training
    increment conditional on the current supplied state.  It tests state
    sufficiency; it is not an autonomous or fitted macroscopic model.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    features = np.asarray(state_features, dtype=float)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if features.ndim != 3 or features.shape[:2] != centers.shape[:2] or features.shape[2] < 1:
        raise ValueError("state_features must align with centers on frame and particle axes")
    if not (2 < train_stop < len(centers) - 2) or frame_time <= 0.0:
        raise ValueError("train_stop and frame_time must leave train and held intervals")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(centers) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0) or neighbor_count < 1:
        raise ValueError("wave numbers and neighbor_count must be valid")
    increments = centers[1:] - centers[:-1]
    interval_valid = valid[:-1] & valid[1:] & np.isfinite(increments).all(axis=2) & np.isfinite(features[:-1]).all(axis=2)
    train_valid = interval_valid[: train_stop - 1]
    train_feature = features[: train_stop - 1][train_valid]
    train_increment = increments[: train_stop - 1][train_valid]
    if len(train_increment) < max(10, neighbor_count):
        raise ValueError("training segment has insufficient state-conditioned increments")
    mean = np.mean(train_feature, axis=0)
    scale = np.std(train_feature, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_feature = (train_feature - mean) / scale
    effective_neighbors = min(neighbor_count, len(train_increment))
    train_squared_norm = np.sum(train_feature**2, axis=1)

    def sample_increment(query: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sampled = np.empty((len(query), 3), dtype=float)
        for start in range(0, len(query), 512):
            stop = min(start + 512, len(query))
            batch = query[start:stop]
            squared_distance = np.sum(batch**2, axis=1, keepdims=True) + train_squared_norm[None, :]
            squared_distance -= 2.0 * batch @ train_feature.T
            np.maximum(squared_distance, 0.0, out=squared_distance)
            nearest = np.argpartition(squared_distance, effective_neighbors - 1, axis=1)[:, :effective_neighbors]
            selected = nearest[np.arange(len(nearest)), rng.integers(effective_neighbors, size=len(nearest))]
            sampled[start:stop] = train_increment[selected]
        return sampled

    held_centers = centers[train_stop:]
    held_valid = interval_valid[train_stop:]
    held_feature = (features[train_stop:-1] - mean) / scale
    prefix_invalid = np.concatenate(
        [np.zeros((1, centers.shape[1]), dtype=int), np.cumsum(~held_valid, axis=0)], axis=0
    )
    rng = np.random.default_rng(seed)
    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    predicted_ngp: list[float] = []
    observed_ngp: list[float] = []
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        path_valid = prefix_invalid[lag:] - prefix_invalid[:-lag] == 0
        start, particle = np.nonzero(path_valid)
        if len(start) == 0:
            raise ValueError("held-out segment has no fully valid paths")
        time_index = start[:, None] + np.arange(int(lag))[None, :]
        query = held_feature[time_index, particle[:, None]].reshape(-1, features.shape[2])
        predicted = np.sum(sample_increment(query, rng).reshape(len(start), int(lag), 3), axis=1)
        observed = held_centers[start + int(lag), particle] - held_centers[start, particle]
        predicted_squared = np.sum(predicted**2, axis=1)
        observed_squared = np.sum(observed**2, axis=1)
        predicted_msd.append(float(np.mean(predicted_squared)))
        observed_msd.append(float(np.mean(observed_squared)))
        predicted_ngp.append(float(3.0 * np.mean(predicted_squared**2) / (5.0 * np.mean(predicted_squared) ** 2) - 1.0))
        observed_ngp.append(float(3.0 * np.mean(observed_squared**2) / (5.0 * np.mean(observed_squared) ** 2) - 1.0))
        for wave_number in wave_numbers:
            predicted_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * predicted))))
            observed_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * observed))))
    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    predicted_ngp_array = np.asarray(predicted_ngp)
    observed_ngp_array = np.asarray(observed_ngp)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_errors = [
        abs(predicted / observed - 1.0)
        for wave_number in wave_numbers
        for predicted, observed in zip(predicted_fs[float(wave_number)], observed_fs[float(wave_number)])
        if abs(observed) > 1e-12
    ]
    return {
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "knn_state_fs_max_relative_error": float(max(fs_errors)) if fs_errors else math.nan,
        "knn_state_ngp_max_absolute_error": float(np.max(np.abs(predicted_ngp_array - observed_ngp_array))),
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "predicted_ngp": predicted_ngp_array,
        "observed_ngp": observed_ngp_array,
        "state_feature_count": float(features.shape[2]),
        "state_neighbor_count": float(effective_neighbors),
        "training_increment_sample_count": float(len(train_increment)),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def _response_aligned_frames(response: np.ndarray) -> np.ndarray:
    """Return deterministic orthonormal frames whose third axis follows response."""

    vectors = np.asarray(response, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("response vectors must have shape (samples, 3)")
    magnitude = np.linalg.norm(vectors, axis=1)
    frames = np.broadcast_to(np.eye(3), (len(vectors), 3, 3)).copy()
    nonzero = magnitude > 1e-12
    if not np.any(nonzero):
        return frames
    axis = vectors[nonzero] / magnitude[nonzero, None]
    reference = np.eye(3)[np.argmin(np.abs(axis), axis=1)]
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(axis, first)
    frames[nonzero] = np.stack([first, second, axis], axis=1)
    return frames


def time_split_driven_response_conditioned_innovation_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    predicted_center_increment: np.ndarray,
    response_valid: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    bin_count: int = 4,
    seed: int = 20260807,
) -> dict[str, float | np.ndarray]:
    """Held-out exact-bath closure with rotationally covariant conditional noise.

    The residual law is learned only from the training half.  Each residual is
    expressed in a local frame whose third axis is the exact bath response,
    then binned by the response magnitude.  Held residuals are sampled from
    the matching training bin and rotated into their held response frames.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    response = np.asarray(predicted_center_increment, dtype=float)
    response_valid = np.asarray(response_valid, dtype=bool)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if response.shape != (centers.shape[0] - 1, centers.shape[1], 3) or response_valid.shape != response.shape[:2]:
        raise ValueError("response arrays must contain one increment per center interval")
    if not (2 < train_stop < len(centers) - 2) or frame_time <= 0.0:
        raise ValueError("train_stop and frame_time must leave train and held intervals")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(centers) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0) or bin_count < 2:
        raise ValueError("wave numbers must be positive and bin_count must be at least two")

    actual = centers[1:] - centers[:-1]
    interval_valid = valid[:-1] & valid[1:] & response_valid
    interval_valid &= np.isfinite(actual).all(axis=2) & np.isfinite(response).all(axis=2)
    residual = actual - response
    train_valid = interval_valid[: train_stop - 1]
    train_response = response[: train_stop - 1][train_valid]
    train_residual = residual[: train_stop - 1][train_valid]
    if len(train_residual) < 10 * bin_count:
        raise ValueError("training segment has insufficient valid residual increments")
    train_magnitude = np.linalg.norm(train_response, axis=1)
    bin_edges = np.quantile(train_magnitude, np.linspace(0.0, 1.0, bin_count + 1))
    if np.any(np.diff(bin_edges) <= 1e-14):
        raise ValueError("training response magnitudes do not support distinct conditional bins")
    train_bins = np.searchsorted(bin_edges[1:-1], train_magnitude, side="right")
    train_frames = _response_aligned_frames(train_response)
    train_local_residual = np.einsum("nij,nj->ni", train_frames, train_residual)
    residual_by_bin = [train_local_residual[train_bins == index] for index in range(bin_count)]
    if any(len(samples) < 10 for samples in residual_by_bin):
        raise ValueError("each conditional response bin needs at least ten residual samples")

    held_centers = centers[train_stop:]
    held_response = response[train_stop:]
    held_interval_valid = interval_valid[train_stop:]
    prefix_invalid = np.concatenate(
        [np.zeros((1, centers.shape[1]), dtype=int), np.cumsum(~held_interval_valid, axis=0)], axis=0
    )
    rng = np.random.default_rng(seed)
    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    predicted_ngp: list[float] = []
    observed_ngp: list[float] = []
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        path_valid = prefix_invalid[lag:] - prefix_invalid[:-lag] == 0
        start, particle = np.nonzero(path_valid)
        if len(start) == 0:
            raise ValueError("held-out segment has no fully valid bath-driven paths")
        time_index = start[:, None] + np.arange(int(lag))[None, :]
        response_path = held_response[time_index, particle[:, None]]
        flat_response = response_path.reshape(-1, 3)
        held_bins = np.searchsorted(bin_edges[1:-1], np.linalg.norm(flat_response, axis=1), side="right")
        local_noise = np.empty_like(flat_response)
        for index, samples in enumerate(residual_by_bin):
            selected = held_bins == index
            local_noise[selected] = samples[rng.integers(len(samples), size=np.count_nonzero(selected))]
        held_frames = _response_aligned_frames(flat_response)
        noise = np.einsum("nij,ni->nj", held_frames, local_noise).reshape(response_path.shape)
        predicted = np.sum(response_path + noise, axis=1)
        observed = held_centers[start + int(lag), particle] - held_centers[start, particle]
        predicted_squared = np.sum(predicted**2, axis=1)
        observed_squared = np.sum(observed**2, axis=1)
        predicted_msd.append(float(np.mean(predicted_squared)))
        observed_msd.append(float(np.mean(observed_squared)))
        predicted_ngp.append(float(3.0 * np.mean(predicted_squared**2) / (5.0 * np.mean(predicted_squared) ** 2) - 1.0))
        observed_ngp.append(float(3.0 * np.mean(observed_squared**2) / (5.0 * np.mean(observed_squared) ** 2) - 1.0))
        for wave_number in wave_numbers:
            predicted_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * predicted))))
            observed_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * observed))))
    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    predicted_ngp_array = np.asarray(predicted_ngp)
    observed_ngp_array = np.asarray(observed_ngp)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_errors = [
        abs(predicted / observed - 1.0)
        for wave_number in wave_numbers
        for predicted, observed in zip(predicted_fs[float(wave_number)], observed_fs[float(wave_number)])
        if abs(observed) > 1e-12
    ]
    residual_squared = np.sum(train_residual**2, axis=1)
    return {
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "driven_response_fs_max_relative_error": float(max(fs_errors)) if fs_errors else math.nan,
        "driven_response_ngp_max_absolute_error": float(np.max(np.abs(predicted_ngp_array - observed_ngp_array))),
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "predicted_ngp": predicted_ngp_array,
        "observed_ngp": observed_ngp_array,
        "training_residual_sample_count": float(len(train_residual)),
        "conditional_response_bin_count": float(bin_count),
        "training_response_residual_square_correlation": float(np.corrcoef(train_magnitude**2, residual_squared)[0, 1]),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_lagged_response_embedding_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    response_increment: np.ndarray,
    response_valid: np.ndarray,
    *,
    train_stop: int,
) -> dict[str, float]:
    """Test whether a past microscopic response field closes tagged-center memory.

    The added predictor is ``u_(t-1)=J(R_(t-1)) Delta R_(t-1)``.  It is
    causal at time ``t`` but requires the measured environment, so a positive
    result proves missing collective information rather than an autonomous
    single-coordinate closure.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    response_increment = np.asarray(response_increment, dtype=float)
    response_valid = np.asarray(response_valid, dtype=bool)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if response_increment.shape != (centers.shape[0] - 1, centers.shape[1], 3):
        raise ValueError("response_increment must contain one vector per center interval")
    if response_valid.shape != response_increment.shape[:2]:
        raise ValueError("response_valid must align with response_increment")
    if not (4 < train_stop < centers.shape[0] - 3):
        raise ValueError("train_stop must leave lagged response samples in both segments")

    velocity = centers[1:] - centers[:-1]
    index = np.arange(1, len(velocity) - 1)
    usable = (
        valid[index]
        & valid[index + 1]
        & valid[index + 2]
        & response_valid[index - 1]
        & np.isfinite(velocity[index]).all(axis=2)
        & np.isfinite(velocity[index + 1]).all(axis=2)
        & np.isfinite(response_increment[index - 1]).all(axis=2)
    )
    train_mask = usable[index < train_stop - 2]
    held_mask = usable[index >= train_stop - 1]
    if not np.any(train_mask) or not np.any(held_mask):
        raise ValueError("both time segments require valid lagged response samples")
    predictor_velocity = velocity[index]
    predictor_response = response_increment[index - 1]
    target_velocity = velocity[index + 1]

    x_train = np.column_stack(
        [
            predictor_velocity[index < train_stop - 2][train_mask].reshape(-1),
            predictor_response[index < train_stop - 2][train_mask].reshape(-1),
        ]
    )
    y_train = target_velocity[index < train_stop - 2][train_mask].reshape(-1)
    coefficient, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    ar_coefficient = float(np.dot(x_train[:, 0], y_train) / np.dot(x_train[:, 0], x_train[:, 0]))

    held_selector = index >= train_stop - 1
    x_held = np.column_stack(
        [
            predictor_velocity[held_selector][held_mask].reshape(-1),
            predictor_response[held_selector][held_mask].reshape(-1),
        ]
    )
    y_held = target_velocity[held_selector][held_mask].reshape(-1)
    prediction = x_held @ coefficient
    ar_prediction = x_held[:, 0] * ar_coefficient
    total = float(np.sum((y_held - np.mean(y_held)) ** 2))
    response_r_squared = 1.0 - float(np.sum((y_held - prediction) ** 2)) / total if total > 0.0 else math.nan
    ar_r_squared = 1.0 - float(np.sum((y_held - ar_prediction) ** 2)) / total if total > 0.0 else math.nan
    return {
        "center_velocity_coefficient": float(coefficient[0]),
        "lagged_response_coefficient": float(coefficient[1]),
        "heldout_response_r_squared": float(response_r_squared),
        "heldout_ar_only_r_squared": float(ar_r_squared),
        "heldout_response_r_squared_gain": float(response_r_squared - ar_r_squared),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def residual_ar1_memory_diagnostic(
    residual: np.ndarray,
    *,
    frame_time: float,
    white_noise_correlation_maximum: float = 0.05,
) -> dict[str, float]:
    """Fit a discrete OU residual and test whether its innovations are white."""

    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 3 or residual.shape[0] < 3 or residual.shape[2] < 1:
        raise ValueError("residual must have shape (at least 3 frames, particles, dimensions)")
    if frame_time <= 0.0 or not math.isfinite(frame_time):
        raise ValueError("frame_time must be positive and finite")
    if white_noise_correlation_maximum <= 0.0:
        raise ValueError("white_noise_correlation_maximum must be positive")
    valid = np.isfinite(residual).all(axis=2)
    pair = valid[:-1] & valid[1:]
    pair_mask = np.repeat(pair[:, :, None], residual.shape[2], axis=2)
    left = residual[:-1]
    right = residual[1:]
    denominator = float(np.sum(left[pair_mask] ** 2))
    if denominator <= 0.0:
        raise ValueError("residual must contain nonzero consecutive values")
    coefficient = float(np.sum(left[pair_mask] * right[pair_mask]) / denominator)
    innovation = right - coefficient * left
    triple = valid[:-2] & valid[1:-1] & valid[2:]
    triple_mask = np.repeat(triple[:, :, None], residual.shape[2], axis=2)
    innovation_left = innovation[:-1][triple_mask]
    innovation_right = innovation[1:][triple_mask]
    innovation_denominator = float(np.sum(innovation_left**2))
    correlation = (
        float(np.sum(innovation_left * innovation_right) / innovation_denominator)
        if innovation_denominator > 0.0
        else math.nan
    )
    ou_time = -frame_time / math.log(coefficient) if 0.0 < coefficient < 1.0 else math.nan
    memory_time = -frame_time / math.log(abs(correlation)) if 0.0 < abs(correlation) < 1.0 else math.nan
    return {
        "ar1_coefficient": coefficient,
        "ou_relaxation_time": ou_time,
        "innovation_lag1_correlation": correlation,
        "innovation_memory_time": memory_time,
        "white_noise_correlation_maximum": float(white_noise_correlation_maximum),
        "white_noise_candidate": float(abs(correlation) <= white_noise_correlation_maximum),
        "valid_pair_fraction": float(np.mean(pair)),
    }


def segmented_cage_center_event_statistics(
    centers: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float,
    half_window: int,
    frame_time: float,
    max_correlation_lag: int = 2,
) -> dict[str, float]:
    """Extract a p_hop clock without treating invalid local minima as exposure.

    Each continuous valid segment is independently segmented, so neither a
    waiting interval nor a jump-vector correlation crosses a failed local
    minimization frame.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if threshold <= 0.0 or half_window < 1 or frame_time <= 0.0 or max_correlation_lag < 0:
        raise ValueError("event controls must be positive and finite")

    jump_vectors: list[np.ndarray] = []
    correlation_values: dict[int, list[float]] = {
        lag: [] for lag in range(1, max_correlation_lag + 1)
    }
    exposure = 0.0
    segment_count = 0
    for particle in range(centers.shape[1]):
        usable = valid[:, particle] & np.isfinite(centers[:, particle]).all(axis=1)
        indices = np.flatnonzero(usable)
        if not len(indices):
            continue
        boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
        for segment_indices in np.split(indices, boundaries):
            if len(segment_indices) < 2 * half_window + 1:
                continue
            segment_count += 1
            exposure += (len(segment_indices) - 1) * frame_time
            segment = centers[segment_indices, particle : particle + 1]
            events = extract_nonrecrossing_phop_events(
                segment,
                threshold=threshold,
                half_window=half_window,
                recrossing_radius=math.sqrt(threshold),
            )
            jumps = events["jump_vector"]
            if not len(jumps):
                continue
            jump_vectors.extend(jumps)
            for lag in correlation_values:
                if len(jumps) > lag:
                    correlation_values[lag].extend(
                        np.sum(jumps[:-lag] * jumps[lag:], axis=1).tolist()
                    )
    if exposure <= 0.0 or not jump_vectors:
        raise ValueError("valid segments must contain exposure and at least one p_hop event")
    jumps = np.asarray(jump_vectors, dtype=float)
    squared = np.sum(jumps**2, axis=1)
    jump_squared_mean = float(np.mean(squared))
    correlations = [
        float(np.mean(correlation_values[lag])) if correlation_values[lag] else 0.0
        for lag in range(1, max_correlation_lag + 1)
    ]
    rate = len(jumps) / exposure
    uncorrelated = rate * jump_squared_mean / 6.0
    correlated = event_space_correlated_diffusion(rate, jump_squared_mean, correlations, dimension=3)
    return {
        "segment_count": float(segment_count),
        "valid_exposure_time": float(exposure),
        "event_count": float(len(jumps)),
        "event_rate": float(rate),
        "jump_squared_mean": jump_squared_mean,
        "jump_correlation_lag1_over_q": correlations[0] / jump_squared_mean if correlations else 0.0,
        "jump_correlation_lag2_over_q": correlations[1] / jump_squared_mean if len(correlations) > 1 else 0.0,
        "uncorrelated_diffusion": float(uncorrelated),
        "correlated_diffusion": float(correlated),
    }


def precursor_event_hazard_diagnostic(
    state: np.ndarray,
    event_particles: np.ndarray,
    event_times: np.ndarray,
    *,
    train_stop: int,
    precursor_lag: int,
    frame_time: float,
    bin_count: int = 4,
    held_source_start: int | None = None,
    state_valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Time-split conditional event hazard from a positive microscopic state.

    An event at frame ``t`` is forecast only from ``state[t-precursor_lag]``.
    Quantile bins and their discrete hazards are fitted on the training time
    interval; held-out rates, calibration, and readiness statistics are then
    predictions.  The event detector is intentionally external so a fixed
    p_hop protocol can be audited independently from this precursor test.
    """

    state = np.asarray(state, dtype=float)
    event_particles = np.asarray(event_particles, dtype=int)
    event_times = np.asarray(event_times, dtype=int)
    if state.ndim != 2:
        raise ValueError("state must be a two-dimensional (frames, particles) array")
    if state_valid is None:
        valid = np.ones(state.shape, dtype=bool)
    else:
        valid = np.asarray(state_valid, dtype=bool)
        if valid.shape != state.shape:
            raise ValueError("state_valid must align with state")
    if not np.any(valid) or np.any(~np.isfinite(state[valid])) or np.any(state[valid] <= 0.0):
        raise ValueError("valid state entries must be finite and positive")
    if event_particles.ndim != 1 or event_times.shape != event_particles.shape:
        raise ValueError("event particle and time arrays must be aligned one-dimensional arrays")
    if not (0 < precursor_lag < train_stop < len(state) - precursor_lag):
        raise ValueError("train_stop and precursor_lag must leave train and held-out forecast windows")
    if held_source_start is None:
        held_source_start = train_stop
    if not (train_stop <= held_source_start < len(state) - precursor_lag):
        raise ValueError("held_source_start must follow training and leave held-out forecasts")
    if frame_time <= 0.0 or bin_count < 2:
        raise ValueError("frame_time and bin_count must be positive")
    if np.any(event_particles < 0) or np.any(event_particles >= state.shape[1]):
        raise ValueError("event_particles must select state columns")
    if np.any(event_times < precursor_lag) or np.any(event_times >= len(state)):
        raise ValueError("event_times must permit a causal precursor and fit inside state")
    if len(event_times) and len(np.unique(np.column_stack([event_times, event_particles]), axis=0)) != len(event_times):
        raise ValueError("event labels must contain at most one event per particle and frame")

    label = np.zeros(state.shape, dtype=bool)
    event_source = event_times - precursor_lag
    event_usable = valid[event_source, event_particles]
    label[event_source[event_usable], event_particles[event_usable]] = True
    train_source = np.arange(0, train_stop - precursor_lag)
    held_source = np.arange(held_source_start, len(state) - precursor_lag)
    log_state = np.full(state.shape, np.nan, dtype=float)
    log_state[valid] = np.log(state[valid])
    train_valid = valid[train_source]
    held_valid = valid[held_source]
    train_values = log_state[train_source][train_valid]
    edges = np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) != bin_count + 1:
        raise ValueError("state has insufficient variation for the requested precursor bins")

    def bin_index(values: np.ndarray) -> np.ndarray:
        return np.searchsorted(edges[1:-1], values, side="right")

    train_bins_full = np.full(train_valid.shape, -1, dtype=int)
    held_bins_full = np.full(held_valid.shape, -1, dtype=int)
    train_bins_full[train_valid] = bin_index(log_state[train_source][train_valid])
    held_bins_full[held_valid] = bin_index(log_state[held_source][held_valid])
    train_bins = train_bins_full[train_valid]
    held_bins = held_bins_full[held_valid]
    train_labels = label[train_source][train_valid]
    held_labels = label[held_source][held_valid]
    hazard = np.empty(bin_count, dtype=float)
    train_counts = np.empty(bin_count, dtype=float)
    held_counts = np.empty(bin_count, dtype=float)
    held_hazard = np.empty(bin_count, dtype=float)
    for index in range(bin_count):
        train_mask = train_bins == index
        held_mask = held_bins == index
        if not np.any(train_mask) or not np.any(held_mask):
            raise ValueError("each precursor bin must have train and held-out exposure")
        train_counts[index] = float(np.sum(train_labels[train_mask]))
        held_counts[index] = float(np.sum(held_labels[held_mask]))
        hazard[index] = train_counts[index] / float(np.sum(train_mask))
        held_hazard[index] = held_counts[index] / float(np.sum(held_mask))

    prediction = hazard[held_bins]
    observed = held_labels.astype(float)
    baseline = float(np.mean(train_labels))
    brier = float(np.mean((prediction - observed) ** 2))
    baseline_brier = float(np.mean((baseline - observed) ** 2))
    clipped_prediction = np.clip(prediction, 1e-15, 1.0 - 1e-15)
    clipped_baseline = min(max(baseline, 1e-15), 1.0 - 1e-15)
    held_log_likelihood = float(
        np.sum(observed * np.log(clipped_prediction) + (1.0 - observed) * np.log1p(-clipped_prediction))
    )
    baseline_log_likelihood = float(
        np.sum(observed * math.log(clipped_baseline) + (1.0 - observed) * math.log1p(-clipped_baseline))
    )
    log_likelihood_gain = held_log_likelihood - baseline_log_likelihood
    held_event_count = float(np.sum(observed))
    ready = held_bins == bin_count - 1
    unready = ~ready
    ready_hazard = float(np.mean(held_labels[ready])) if np.any(ready) else math.nan
    unready_hazard = float(np.mean(held_labels[unready])) if np.any(unready) else math.nan
    training_ready = train_bins == bin_count - 1
    training_unready = ~training_ready
    train_ready_hazard = float(np.mean(train_labels[training_ready]))
    train_unready_hazard = float(np.mean(train_labels[training_unready]))

    first_passage: list[float] = []
    ready_full = held_bins_full == bin_count - 1
    for particle in range(state.shape[1]):
        valid_indices = np.flatnonzero(held_valid[:, particle])
        boundaries = np.flatnonzero(np.diff(valid_indices) > 1) + 1
        for run in np.split(valid_indices, boundaries):
            nonready_run = 0
            for is_ready in ready_full[run, particle]:
                if is_ready:
                    if nonready_run:
                        first_passage.append(nonready_run * frame_time)
                    nonready_run = 0
                else:
                    nonready_run += 1
    first_passage_array = np.asarray(first_passage, dtype=float)
    if len(first_passage_array):
        mean_first_passage = float(np.mean(first_passage_array))
        empirical_cdf = np.arange(1, len(first_passage_array) + 1) / len(first_passage_array)
        exponential_cdf = 1.0 - np.exp(-np.sort(first_passage_array) / mean_first_passage)
        first_passage_ks = float(np.max(np.abs(empirical_cdf - exponential_cdf)))
        first_passage_cv = float(np.std(first_passage_array) / mean_first_passage)
    else:
        mean_first_passage = math.nan
        first_passage_ks = math.nan
        first_passage_cv = math.nan

    result: dict[str, float] = {
        "training_event_hazard": baseline / frame_time,
        "heldout_event_hazard": float(np.mean(held_labels)) / frame_time,
        "heldout_brier_score": brier,
        "heldout_brier_skill": 1.0 - brier / baseline_brier if baseline_brier > 0.0 else math.nan,
        "heldout_log_likelihood_gain": log_likelihood_gain,
        "heldout_log_likelihood_gain_per_label": log_likelihood_gain / len(observed) if len(observed) else math.nan,
        "heldout_log_likelihood_gain_per_event": log_likelihood_gain / held_event_count if held_event_count > 0.0 else math.nan,
        "training_ready_hazard": train_ready_hazard / frame_time,
        "training_unready_hazard": train_unready_hazard / frame_time,
        "heldout_ready_hazard": ready_hazard / frame_time,
        "heldout_unready_hazard": unready_hazard / frame_time,
        "training_ready_to_unready_hazard_ratio": train_ready_hazard / train_unready_hazard if train_unready_hazard > 0.0 else math.nan,
        "heldout_ready_to_unready_hazard_ratio": ready_hazard / unready_hazard if unready_hazard > 0.0 else math.nan,
        "heldout_readiness_first_passage_count": float(len(first_passage_array)),
        "heldout_readiness_first_passage_mean": mean_first_passage,
        "heldout_readiness_first_passage_cv": first_passage_cv,
        "heldout_readiness_first_passage_exponential_ks": first_passage_ks,
        "training_state_valid_fraction": float(np.mean(train_valid)),
        "heldout_state_valid_fraction": float(np.mean(held_valid)),
        "usable_event_fraction": float(np.mean(event_usable)) if len(event_usable) else 1.0,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for index in range(bin_count):
        result[f"training_hazard_bin_{index + 1}"] = hazard[index] / frame_time
        result[f"heldout_hazard_bin_{index + 1}"] = held_hazard[index] / frame_time
        result[f"training_event_count_bin_{index + 1}"] = train_counts[index]
        result[f"heldout_event_count_bin_{index + 1}"] = held_counts[index]
    return result


def isoconfigurational_first_passage_diagnostic(
    first_passage: np.ndarray,
    *,
    horizon: float,
    survival_times: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Estimate a finite-time exit clock from right-censored noise clones.

    Rows are independent Langevin-noise clones of one configuration and
    columns are tagged particles.  ``inf`` denotes no detected first exit
    before ``horizon``.  The exponential rate is the censored-likelihood MLE;
    agreement of its survival curve is evidence for a memoryless exit clock,
    not by itself an Eyring-Kramers barrier proof.
    """

    first_passage = np.asarray(first_passage, dtype=float)
    survival_times = np.asarray(survival_times, dtype=float)
    if first_passage.ndim != 2 or not first_passage.shape[0] >= 2 or not first_passage.shape[1] >= 1:
        raise ValueError("first_passage must be a (clones, particles) array with at least two clones")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be positive and finite")
    if survival_times.ndim != 1 or not len(survival_times) or np.any(~np.isfinite(survival_times)) or np.any(survival_times < 0.0) or np.any(survival_times > horizon):
        raise ValueError("survival_times must be finite values inside the observation horizon")
    if np.any(np.isnan(first_passage)) or np.any(first_passage[np.isfinite(first_passage)] <= 0.0):
        raise ValueError("first_passage must contain positive finite times or inf censoring")
    if np.any(first_passage[np.isfinite(first_passage)] > horizon):
        raise ValueError("finite first-passage times must lie inside the observation horizon")

    escaped = np.isfinite(first_passage)
    event_count = int(np.sum(escaped))
    exposure = float(np.sum(np.minimum(first_passage, horizon)))
    if exposure <= 0.0:
        raise ValueError("first-passage exposure must be positive")
    rate = event_count / exposure
    observed_survival = np.array([float(np.mean(first_passage > time)) for time in survival_times])
    predicted_survival = np.exp(-rate * survival_times)
    committor = np.mean(escaped, axis=0)
    return {
        "escape_rate_mle": float(rate),
        "event_count": float(event_count),
        "clone_count": float(first_passage.shape[0]),
        "particle_count": float(first_passage.shape[1]),
        "total_exposure": exposure,
        "committor_mean": float(np.mean(committor)),
        "committor_standard_deviation": float(np.std(committor, ddof=1)) if len(committor) > 1 else 0.0,
        "committor_minimum": float(np.min(committor)),
        "committor_maximum": float(np.max(committor)),
        "survival_max_absolute_error": float(np.max(np.abs(observed_survival - predicted_survival))),
        "exponential_exit_clock_candidate": float(np.max(np.abs(observed_survival - predicted_survival)) <= 0.05),
        "survival_times": survival_times,
        "observed_survival": observed_survival,
        "predicted_exponential_survival": predicted_survival,
        "committor": committor,
        "thermodynamic_claim_allowed": 0.0,
    }


def isoconfigurational_state_rate_diagnostic(
    state: np.ndarray,
    first_passage: np.ndarray,
    *,
    train_mask: np.ndarray,
    horizon: float,
    bin_count: int = 5,
) -> dict[str, float]:
    """Predict clone-resolved exits from a particle's initial microstate.

    State bins and their censored-likelihood exit rates are fitted using only
    ``train_mask`` particles.  The resulting finite-horizon escape
    probabilities are then scored on the held-out particles across all noise
    clones.  No trajectory-scale observable enters this calculation.
    """

    state = np.asarray(state, dtype=float)
    first_passage = np.asarray(first_passage, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)
    if state.ndim != 1 or not len(state) or np.any(~np.isfinite(state)) or np.any(state <= 0.0):
        raise ValueError("state must be a nonempty finite positive vector")
    if first_passage.ndim != 2 or first_passage.shape[1] != len(state) or first_passage.shape[0] < 2:
        raise ValueError("first_passage must align as (clones, particles) with at least two clones")
    if train_mask.shape != state.shape or np.all(train_mask) or not np.any(train_mask):
        raise ValueError("train_mask must split the particle axis")
    if not math.isfinite(horizon) or horizon <= 0.0 or bin_count < 2:
        raise ValueError("horizon and bin_count must be positive")
    if np.any(np.isnan(first_passage)) or np.any(first_passage[np.isfinite(first_passage)] <= 0.0) or np.any(first_passage[np.isfinite(first_passage)] > horizon):
        raise ValueError("first_passage must contain positive in-horizon times or inf censoring")

    log_state = np.log(state)
    edges = np.unique(np.quantile(log_state[train_mask], np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) != bin_count + 1:
        raise ValueError("training state has insufficient variation for requested bins")
    bin_index = np.searchsorted(edges[1:-1], log_state, side="right")
    escaped = np.isfinite(first_passage)
    exposure = np.minimum(first_passage, horizon)
    rate = np.empty(bin_count, dtype=float)
    bin_center = np.empty(bin_count, dtype=float)
    for index in range(bin_count):
        selected = train_mask & (bin_index == index)
        event_count = float(np.sum(escaped[:, selected]))
        total_exposure = float(np.sum(exposure[:, selected]))
        if total_exposure <= 0.0:
            raise ValueError("every training state bin must have first-passage exposure")
        rate[index] = event_count / total_exposure
        bin_center[index] = float(np.mean(log_state[selected]))

    held_mask = ~train_mask
    held_rate = rate[bin_index[held_mask]]
    predicted_escape = 1.0 - np.exp(-held_rate * horizon)
    observed_escape = escaped[:, held_mask].astype(float)
    prediction_matrix = np.broadcast_to(predicted_escape, observed_escape.shape)
    baseline = float(np.mean(escaped[:, train_mask]))
    brier = float(np.mean((prediction_matrix - observed_escape) ** 2))
    baseline_brier = float(np.mean((baseline - observed_escape) ** 2))
    low = held_mask & (bin_index == 0)
    high = held_mask & (bin_index == bin_count - 1)
    low_rate = float(np.sum(escaped[:, low]) / np.sum(exposure[:, low]))
    high_rate = float(np.sum(escaped[:, high]) / np.sum(exposure[:, high]))
    positive = rate > 0.0
    slope = float(np.polyfit(bin_center[positive], np.log(rate[positive]), 1)[0]) if np.sum(positive) >= 2 else math.nan
    result: dict[str, float] = {
        "heldout_brier_score": brier,
        "heldout_brier_skill": 1.0 - brier / baseline_brier if baseline_brier > 0.0 else math.nan,
        "heldout_low_state_rate": low_rate,
        "heldout_high_state_rate": high_rate,
        "heldout_high_to_low_rate_ratio": high_rate / low_rate if low_rate > 0.0 else math.nan,
        "state_rate_log_slope": slope,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for index in range(bin_count):
        result[f"training_rate_bin_{index + 1}"] = float(rate[index])
    return result


def grouped_binomial_logistic_committor_diagnostic(
    features: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    groups: np.ndarray,
    *,
    l2_regularization: float = 1.0,
    maximum_iterations: int = 50,
) -> dict[str, float | np.ndarray]:
    """Leave-one-group-out structural prediction of clone escape outcomes.

    Each row is one particle in one parent configuration, with a binomial
    finite-time committor estimate ``successes / trials``.  A regularized
    logistic reaction coordinate is re-fitted after withholding every parent
    group, so its reported score is not a within-configuration propensity fit.
    """

    features = np.asarray(features, dtype=float)
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    groups = np.asarray(groups)
    if features.ndim != 2 or not len(features) or np.any(~np.isfinite(features)):
        raise ValueError("features must be a nonempty finite two-dimensional array")
    if successes.shape != (len(features),) or trials.shape != (len(features),) or groups.shape != (len(features),):
        raise ValueError("successes, trials, and groups must align with feature rows")
    if np.any(~np.isfinite(successes)) or np.any(~np.isfinite(trials)) or np.any(trials <= 0.0) or np.any(successes < 0.0) or np.any(successes > trials):
        raise ValueError("successes must lie in [0, trials] for positive finite trials")
    if not math.isfinite(l2_regularization) or l2_regularization < 0.0 or maximum_iterations < 1:
        raise ValueError("l2_regularization and maximum_iterations must be valid")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two parent groups are required")

    prediction = np.full(len(features), np.nan, dtype=float)
    baseline_prediction = np.full(len(features), np.nan, dtype=float)
    brier_skills: list[float] = []
    likelihood_gains: list[float] = []
    for group in unique_groups:
        held = groups == group
        train = ~held
        mean = np.mean(features[train], axis=0)
        scale = np.std(features[train], axis=0)
        scale[scale < 1e-12] = 1.0
        train_x = (features[train] - mean) / scale
        held_x = (features[held] - mean) / scale
        train_x = np.column_stack([np.ones(len(train_x)), train_x])
        held_x = np.column_stack([np.ones(len(held_x)), held_x])
        baseline = float(np.sum(successes[train]) / np.sum(trials[train]))
        baseline = min(max(baseline, 1e-8), 1.0 - 1e-8)
        coefficient = np.zeros(train_x.shape[1], dtype=float)
        coefficient[0] = math.log(baseline / (1.0 - baseline))
        penalty = np.diag(np.concatenate([[0.0], np.full(train_x.shape[1] - 1, l2_regularization)]))
        for _ in range(maximum_iterations):
            eta = np.clip(train_x @ coefficient, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-eta))
            weight = np.maximum(trials[train] * probability * (1.0 - probability), 1e-10)
            gradient = train_x.T @ (successes[train] - trials[train] * probability) - penalty @ coefficient
            hessian = train_x.T @ (weight[:, None] * train_x) + penalty
            step = np.linalg.solve(hessian, gradient)
            coefficient += step
            if float(np.max(np.abs(step))) < 1e-9:
                break
        held_probability = 1.0 / (1.0 + np.exp(-np.clip(held_x @ coefficient, -30.0, 30.0)))
        prediction[held] = held_probability
        baseline_prediction[held] = baseline
        held_success = successes[held]
        held_trials = trials[held]
        model_brier = float(np.sum(held_success * (1.0 - held_probability) ** 2 + (held_trials - held_success) * held_probability**2) / np.sum(held_trials))
        base_brier = float(np.sum(held_success * (1.0 - baseline) ** 2 + (held_trials - held_success) * baseline**2) / np.sum(held_trials))
        model_log_likelihood = float(np.sum(held_success * np.log(held_probability) + (held_trials - held_success) * np.log1p(-held_probability)))
        base_log_likelihood = float(np.sum(held_success * math.log(baseline) + (held_trials - held_success) * math.log1p(-baseline)))
        brier_skills.append(1.0 - model_brier / base_brier if base_brier > 0.0 else math.nan)
        likelihood_gains.append((model_log_likelihood - base_log_likelihood) / float(np.sum(held_trials)))
    return {
        "mean_heldout_brier_skill": float(np.nanmean(brier_skills)),
        "mean_heldout_log_likelihood_gain_per_trial": float(np.nanmean(likelihood_gains)),
        "parent_group_count": float(len(unique_groups)),
        "out_of_group_prediction": prediction,
        "out_of_group_baseline_prediction": baseline_prediction,
        "thermodynamic_claim_allowed": 0.0,
    }


def heterogeneous_marked_poisson_prediction(
    rates: np.ndarray,
    jump_vectors: np.ndarray,
    *,
    times: np.ndarray,
    wave_numbers: np.ndarray,
) -> dict[str, np.ndarray | dict[float, np.ndarray] | float]:
    """Analytic marked-Poisson observables for particle-resolved exit rates.

    This is the stationary independent-jump null implied by a structural
    reaction-coordinate rate.  It intentionally has no cage-vibration,
    jump-correlation, or rate-evolution correction, so any mismatch localizes
    which effective-theory ingredient remains missing.
    """

    rates = np.asarray(rates, dtype=float)
    jump_vectors = np.asarray(jump_vectors, dtype=float)
    times = np.asarray(times, dtype=float)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if rates.ndim != 1 or not len(rates) or np.any(~np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("rates must be a nonempty finite nonnegative vector")
    if jump_vectors.ndim != 2 or jump_vectors.shape[1] != 3 or not len(jump_vectors) or np.any(~np.isfinite(jump_vectors)):
        raise ValueError("jump_vectors must be a nonempty finite (events, 3) array")
    if times.ndim != 1 or not len(times) or np.any(~np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("times must be a nonempty positive finite vector")
    if wave_numbers.ndim != 1 or not len(wave_numbers) or np.any(~np.isfinite(wave_numbers)) or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be a nonempty positive finite vector")

    squared = np.sum(jump_vectors**2, axis=1)
    second = float(np.mean(squared))
    fourth = float(np.mean(squared**2))
    mean_rate = float(np.mean(rates))
    second_rate = float(np.mean(rates**2))
    predicted_msd = times * second * mean_rate
    predicted_fourth = times * fourth * mean_rate + (5.0 / 3.0) * times**2 * second**2 * second_rate
    predicted_ngp = 3.0 * predicted_fourth / (5.0 * predicted_msd**2) - 1.0
    predicted_fs: dict[float, np.ndarray] = {}
    for wave_number in wave_numbers:
        characteristic = float(np.mean(np.cos(float(wave_number) * jump_vectors.reshape(-1))))
        predicted_fs[float(wave_number)] = np.mean(
            np.exp(np.outer(times, rates) * (characteristic - 1.0)), axis=1
        )
    return {
        "predicted_msd": predicted_msd,
        "predicted_fourth": predicted_fourth,
        "predicted_ngp": predicted_ngp,
        "predicted_fs": predicted_fs,
        "jump_squared_mean": second,
        "jump_fourth_mean": fourth,
        "mean_rate": mean_rate,
        "rate_second_moment": second_rate,
        "thermodynamic_claim_allowed": 0.0,
    }


def underdamped_ou_cage_prediction(
    times: np.ndarray,
    *,
    temperature: float,
    mass: float,
    stiffness: float,
    damping: float,
) -> dict[str, np.ndarray | float]:
    """Position-displacement statistics of an isotropic projected cage mode.

    The mode obeys ``m y'' + gamma y' + kappa y = xi`` with equilibrium
    white noise of variance ``2 gamma T``.  The returned MSD is three
    dimensional, whereas ``coordinate_displacement_variance`` is for one
    Cartesian component.  This is a dynamical projection, not a statement
    that the fitted stiffness equals a bare local Hessian eigenvalue.
    """

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times must be a nonempty finite nonnegative vector")
    if not all(math.isfinite(value) and value > 0.0 for value in (temperature, mass, stiffness)):
        raise ValueError("temperature, mass, and stiffness must be finite and positive")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and nonnegative")

    alpha = damping / (2.0 * mass)
    natural_squared = stiffness / mass
    discriminant = alpha**2 - natural_squared
    if abs(discriminant) < 1e-12 * max(alpha**2, natural_squared, 1.0):
        correlation_fraction = np.exp(-alpha * times) * (1.0 + alpha * times)
    elif discriminant > 0.0:
        relaxation = math.sqrt(discriminant)
        slow_rate = -alpha + relaxation
        fast_rate = -alpha - relaxation
        correlation_fraction = (
            (alpha + relaxation) / (2.0 * relaxation) * np.exp(slow_rate * times)
            + (relaxation - alpha) / (2.0 * relaxation) * np.exp(fast_rate * times)
        )
    else:
        frequency = math.sqrt(-discriminant)
        correlation_fraction = np.exp(-alpha * times) * (
            np.cos(frequency * times) + alpha / frequency * np.sin(frequency * times)
        )
    coordinate_variance = 2.0 * temperature / stiffness * (1.0 - correlation_fraction)
    coordinate_variance = np.maximum(coordinate_variance, 0.0)
    return {
        "coordinate_displacement_variance": coordinate_variance,
        "predicted_msd": 3.0 * coordinate_variance,
        "temperature": float(temperature),
        "mass": float(mass),
        "stiffness": float(stiffness),
        "damping": float(damping),
        "thermodynamic_claim_allowed": 0.0,
    }


def fit_underdamped_ou_cage_from_msd(
    times: np.ndarray,
    observed_msd: np.ndarray,
    *,
    temperature: float,
    mass: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Identify projected OU cage parameters from a fixed short-time MSD set.

    A deterministic log-grid search avoids importing an optimizer and makes
    the fitted training-only procedure reproducible.  It is deliberately a
    two-parameter short-time identification step; it does not use a held-out
    macro observable or infer a free-energy landscape.
    """

    times = np.asarray(times, dtype=float)
    observed_msd = np.asarray(observed_msd, dtype=float)
    if (
        times.ndim != 1
        or observed_msd.shape != times.shape
        or len(times) < 3
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(observed_msd))
        or np.any(times <= 0.0)
        or np.any(observed_msd < 0.0)
    ):
        raise ValueError("times and observed_msd must be aligned finite vectors with at least three positive times")
    if not all(math.isfinite(value) and value > 0.0 for value in (temperature, mass)):
        raise ValueError("temperature and mass must be finite and positive")

    normalization = max(float(np.mean(observed_msd**2)), 1e-16)
    best_score = math.inf
    best_stiffness = math.nan
    best_damping = math.nan
    log_stiffness = np.linspace(-4.0, 4.0, 49)
    log_damping = np.linspace(-4.0, 4.0, 49)
    for _ in range(3):
        for log_kappa in log_stiffness:
            stiffness = 10.0**float(log_kappa)
            for log_gamma in log_damping:
                damping = 10.0**float(log_gamma)
                prediction = underdamped_ou_cage_prediction(
                    times,
                    temperature=temperature,
                    mass=mass,
                    stiffness=stiffness,
                    damping=damping,
                )["predicted_msd"]
                score = float(np.mean((prediction - observed_msd) ** 2) / normalization)
                if score < best_score:
                    best_score = score
                    best_stiffness = stiffness
                    best_damping = damping
        stiffness_step = float(log_stiffness[1] - log_stiffness[0])
        damping_step = float(log_damping[1] - log_damping[0])
        log_stiffness = np.linspace(math.log10(best_stiffness) - stiffness_step, math.log10(best_stiffness) + stiffness_step, 49)
        log_damping = np.linspace(math.log10(best_damping) - damping_step, math.log10(best_damping) + damping_step, 49)
    result = underdamped_ou_cage_prediction(
        times,
        temperature=temperature,
        mass=mass,
        stiffness=best_stiffness,
        damping=best_damping,
    )
    return {
        **result,
        "relative_mse": best_score,
        "fit_scope": "short_time_training_msd_only",
        "thermodynamic_claim_allowed": 0.0,
    }


def free_exponential_memory_gle_prediction(
    times: np.ndarray,
    *,
    temperature: float,
    mass: float,
    memory_amplitude: float,
    memory_rate: float,
) -> dict[str, np.ndarray | float]:
    """Free-particle GLE statistics for ``K(t)=A exp(-b t)``.

    The projected cage velocity obeys ``M vdot=-int K(t-s)v(s)ds+eta`` and
    the second fluctuation-dissipation relation.  Its normalized VACF has
    Laplace transform ``(s+b)/(s^2+b s+A/M)``.  The returned displacement
    variance is obtained by analytically integrating that VACF twice.
    """

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("times must be a nonempty finite nonnegative vector")
    if not all(math.isfinite(value) and value > 0.0 for value in (temperature, mass, memory_amplitude, memory_rate)):
        raise ValueError("temperature, mass, memory_amplitude, and memory_rate must be finite and positive")

    roots = np.roots([1.0, memory_rate, memory_amplitude / mass])
    first, second = complex(roots[0]), complex(roots[1])
    if abs(first - second) < 1e-8 * max(1.0, abs(first), abs(second)):
        decay = 0.5 * memory_rate
        exponent = -decay
        exponential = np.exp(exponent * times)
        integral = (np.expm1(exponent * times) - exponent * times) / exponent**2
        derivative = times * (exponential - 1.0) / exponent**2 - 2.0 * (np.expm1(exponent * times) - exponent * times) / exponent**3
        vacf_integral = integral + decay * derivative
    else:
        first_weight = (first + memory_rate) / (first - second)
        second_weight = (second + memory_rate) / (second - first)

        def integrated_exponential(root: complex) -> np.ndarray:
            return (np.exp(root * times) - 1.0 - root * times) / root**2

        vacf_integral = first_weight * integrated_exponential(first) + second_weight * integrated_exponential(second)
    coordinate_variance = np.maximum(2.0 * temperature / mass * np.real(vacf_integral), 0.0)
    return {
        "coordinate_displacement_variance": coordinate_variance,
        "predicted_msd": 3.0 * coordinate_variance,
        "long_time_diffusion": float(temperature * memory_rate / memory_amplitude),
        "temperature": float(temperature),
        "mass": float(mass),
        "memory_amplitude": float(memory_amplitude),
        "memory_rate": float(memory_rate),
        "thermodynamic_claim_allowed": 0.0,
    }


def fit_free_exponential_memory_gle_from_msd(
    times: np.ndarray,
    observed_msd: np.ndarray,
    *,
    temperature: float,
) -> dict[str, np.ndarray | float]:
    """Training-only deterministic identification of a one-kernel free GLE."""

    times = np.asarray(times, dtype=float)
    observed_msd = np.asarray(observed_msd, dtype=float)
    if (
        times.ndim != 1
        or observed_msd.shape != times.shape
        or len(times) < 4
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(observed_msd))
        or np.any(times <= 0.0)
        or np.any(observed_msd < 0.0)
    ):
        raise ValueError("times and observed_msd must be aligned finite vectors with at least four positive times")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    def search(log_mass: np.ndarray, log_amplitude: np.ndarray, log_rate: np.ndarray) -> list[tuple[float, float, float, float]]:
        candidates: list[tuple[float, float, float, float]] = []
        for log_m in log_mass:
            mass = 10.0**float(log_m)
            for log_a in log_amplitude:
                amplitude = 10.0**float(log_a)
                for log_b in log_rate:
                    rate = 10.0**float(log_b)
                    predicted = free_exponential_memory_gle_prediction(
                        times,
                        temperature=temperature,
                        mass=mass,
                        memory_amplitude=amplitude,
                        memory_rate=rate,
                    )["predicted_msd"]
                    score = float(np.mean(((predicted - observed_msd) / np.maximum(observed_msd, 1e-16)) ** 2))
                    candidates.append((score, mass, amplitude, rate))
        return candidates

    mass_step = 5.0 / 16.0
    amplitude_step = 10.0 / 16.0
    rate_step = 7.0 / 16.0
    candidates = sorted(
        search(
            np.linspace(-2.0, 3.0, 17),
            np.linspace(-4.0, 6.0, 17),
            np.linspace(-3.0, 4.0, 17),
        )
    )[:8]
    for _ in range(2):
        refined: list[tuple[float, float, float, float]] = []
        for _, mass, amplitude, rate in candidates:
            refined.extend(
                search(
                    np.linspace(math.log10(mass) - mass_step, math.log10(mass) + mass_step, 11),
                    np.linspace(math.log10(amplitude) - amplitude_step, math.log10(amplitude) + amplitude_step, 11),
                    np.linspace(math.log10(rate) - rate_step, math.log10(rate) + rate_step, 11),
                )
            )
        candidates = sorted(refined)[:8]
        mass_step /= 10.0
        amplitude_step /= 10.0
        rate_step /= 10.0
    best_score, best_mass, best_amplitude, best_rate = candidates[0]
    result = free_exponential_memory_gle_prediction(
        times,
        temperature=temperature,
        mass=best_mass,
        memory_amplitude=best_amplitude,
        memory_rate=best_rate,
    )
    return {
        **result,
        "relative_mse": best_score,
        "fit_scope": "training_cage_msd_only",
        "thermodynamic_claim_allowed": 0.0,
    }


def independent_gaussian_cage_marked_poisson_prediction(
    rates: np.ndarray,
    jump_vectors: np.ndarray,
    *,
    times: np.ndarray,
    wave_numbers: np.ndarray,
    cage_coordinate_variance: np.ndarray,
) -> dict[str, np.ndarray | dict[float, np.ndarray] | float]:
    """Compose a measured Gaussian cage field with an independent jump clock.

    ``cage_coordinate_variance`` is the one-Cartesian-component variance of
    the collective cage displacement at each requested time.  It may be
    supplied by a training-only GLE fit or by a deliberately labelled
    empirical benchmark; it is never inferred from held observables here.
    """

    times = np.asarray(times, dtype=float)
    cage_coordinate_variance = np.asarray(cage_coordinate_variance, dtype=float)
    if cage_coordinate_variance.shape != times.shape or np.any(~np.isfinite(cage_coordinate_variance)) or np.any(cage_coordinate_variance < 0.0):
        raise ValueError("cage_coordinate_variance must align with times and be finite nonnegative")
    jump = heterogeneous_marked_poisson_prediction(rates, jump_vectors, times=times, wave_numbers=wave_numbers)
    jump_msd = np.asarray(jump["predicted_msd"], dtype=float)
    jump_fourth = np.asarray(jump["predicted_fourth"], dtype=float)
    predicted_msd = jump_msd + 3.0 * cage_coordinate_variance
    predicted_fourth = jump_fourth + 10.0 * cage_coordinate_variance * jump_msd + 15.0 * cage_coordinate_variance**2
    predicted_ngp = 3.0 * predicted_fourth / (5.0 * predicted_msd**2) - 1.0
    predicted_fs = {
        float(wave_number): np.asarray(jump["predicted_fs"][float(wave_number)], dtype=float)
        * np.exp(-0.5 * float(wave_number) ** 2 * cage_coordinate_variance)
        for wave_number in np.asarray(wave_numbers, dtype=float)
    }
    return {
        "predicted_msd": predicted_msd,
        "predicted_fourth": predicted_fourth,
        "predicted_ngp": predicted_ngp,
        "predicted_fs": predicted_fs,
        "cage_coordinate_displacement_variance": cage_coordinate_variance,
        "jump_prediction": jump,
        "thermodynamic_claim_allowed": 0.0,
    }


def hybrid_structural_clock_underdamped_cage_prediction(
    rates: np.ndarray,
    jump_vectors: np.ndarray,
    *,
    times: np.ndarray,
    wave_numbers: np.ndarray,
    temperature: float,
    mass: float,
    stiffness: float,
    damping: float,
) -> dict[str, np.ndarray | dict[float, np.ndarray] | float]:
    """Combine an independent structural exit clock with projected cage OU motion.

    The additive displacement is ``Delta r = Delta r_jump + Delta r_cage``.
    The cage term is isotropic Gaussian at every fixed time, so its fourth
    moment and scattering factor are fixed by the same fitted two parameters.
    """

    cage = underdamped_ou_cage_prediction(
        times,
        temperature=temperature,
        mass=mass,
        stiffness=stiffness,
        damping=damping,
    )
    coordinate_variance = np.asarray(cage["coordinate_displacement_variance"], dtype=float)
    combined = independent_gaussian_cage_marked_poisson_prediction(
        rates,
        jump_vectors,
        times=times,
        wave_numbers=wave_numbers,
        cage_coordinate_variance=coordinate_variance,
    )
    return {
        **combined,
        "cage_prediction": cage,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_center_gle_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Predict held-out center transport from a training discrete velocity kernel.

    The Green--Kubo sum is exact for the measured discrete center increments.
    The accompanying scattering prediction is the stricter additive-Gaussian
    GLE closure and is therefore expected to fail when slow center increments
    remain non-Gaussian.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if not (3 < train_stop < len(centers) - 2) or frame_time <= 0.0:
        raise ValueError("train_stop and frame_time must leave train/held-out trajectories")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(centers) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be positive")

    train_increment = centers[1:train_stop] - centers[: train_stop - 1]
    train_valid = valid[:train_stop]
    increment_valid = train_valid[:-1] & train_valid[1:]
    if not np.any(increment_valid):
        raise ValueError("training segment has no valid center increments")
    gamma: dict[int, float] = {}
    maximum_lag = int(np.max(lags))
    invalid_prefix = np.concatenate(
        [np.zeros((1, centers.shape[1]), dtype=int), np.cumsum(~train_valid, axis=0)],
        axis=0,
    )
    for lag in range(maximum_lag):
        if lag == 0:
            mask = increment_valid
            left = train_increment
            right = train_increment
        else:
            length = train_increment.shape[0] - lag
            if length <= 0:
                raise ValueError("training segment is too short for requested Green-Kubo lag")
            # A velocity pair only contributes when all centers between it are valid.
            fully_valid = invalid_prefix[lag + 2 : lag + 2 + length] - invalid_prefix[:length] == 0
            mask = fully_valid
            left = train_increment[:length]
            right = train_increment[lag : lag + length]
        if not np.any(mask):
            gamma[lag] = 0.0
        else:
            gamma[lag] = float(np.mean(np.sum(left[mask] * right[mask], axis=1)))

    held_centers = centers[train_stop:]
    held_valid = valid[train_stop:]
    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    observed_ngp: list[float] = []
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        predicted = lag * gamma[0] + 2.0 * sum((lag - memory_lag) * gamma[memory_lag] for memory_lag in range(1, lag))
        predicted_msd.append(float(max(predicted, 0.0)))
        mask = held_valid[:-lag] & held_valid[lag:]
        if not np.any(mask):
            raise ValueError("held-out segment has no valid center displacement pairs")
        displacement = held_centers[lag:] - held_centers[:-lag]
        squared = np.sum(displacement[mask] ** 2, axis=1)
        observed_msd.append(float(np.mean(squared)))
        fourth = float(np.mean(squared**2))
        second = float(np.mean(squared))
        observed_ngp.append(3.0 * fourth / (5.0 * second**2) - 1.0 if second > 0.0 else 0.0)
        for wave_number in wave_numbers:
            observed_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * displacement[mask]))))
            predicted_fs[float(wave_number)].append(float(math.exp(-wave_number**2 * predicted / 6.0)))

    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_relative_errors = [
        abs(predicted / observed - 1.0)
        for wave_number in wave_numbers
        for predicted, observed in zip(predicted_fs[float(wave_number)], observed_fs[float(wave_number)])
        if abs(observed) > 1e-12
    ]
    increment_squared = np.sum(train_increment[increment_valid] ** 2, axis=1)
    increment_ngp = (
        3.0 * float(np.mean(increment_squared**2)) / (5.0 * float(np.mean(increment_squared)) ** 2) - 1.0
        if np.mean(increment_squared) > 0.0
        else 0.0
    )
    return {
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "velocity_lag1_correlation_over_gamma0": gamma[1] / gamma[0] if gamma[0] > 0.0 else math.nan,
        "training_increment_ngp": float(increment_ngp),
        "observed_heldout_ngp_peak": float(np.max(observed_ngp)),
        "gaussian_fs_max_relative_error": float(max(fs_relative_errors)) if fs_relative_errors else math.nan,
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "observed_ngp": np.asarray(observed_ngp),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_empirical_increment_levy_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Test an iid non-Gaussian increment closure on a time holdout.

    This is the discrete-time one-particle white-noise limit.  Its only
    training inputs are the microscopic one-step center increments: their
    second and fourth moments and characteristic function.  No macro curve is
    fitted.  A failure therefore isolates memory or hidden state dependence,
    rather than a Gaussian-noise approximation.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if not (3 < train_stop < len(centers) - 2) or frame_time <= 0.0:
        raise ValueError("train_stop and frame_time must leave train/held-out trajectories")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(centers) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be positive")

    train_increment = centers[1:train_stop] - centers[: train_stop - 1]
    increment_valid = valid[1:train_stop] & valid[: train_stop - 1]
    if not np.any(increment_valid):
        raise ValueError("training segment has no valid center increments")
    one_step = train_increment[increment_valid]
    squared_one_step = np.sum(one_step**2, axis=1)
    mean_squared = float(np.mean(squared_one_step))
    if mean_squared <= 0.0:
        raise ValueError("training increments must have positive variance")
    one_step_ngp = 3.0 * float(np.mean(squared_one_step**2)) / (5.0 * mean_squared**2) - 1.0
    characteristic = {
        float(k): float(np.mean(np.cos(k * one_step))) for k in wave_numbers
    }

    held_centers = centers[train_stop:]
    held_valid = valid[train_stop:]
    invalid_prefix = np.concatenate(
        [np.zeros((1, centers.shape[1]), dtype=int), np.cumsum(~held_valid, axis=0)],
        axis=0,
    )
    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    predicted_ngp: list[float] = []
    observed_ngp: list[float] = []
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        path_valid = invalid_prefix[lag + 1 :] - invalid_prefix[: -lag - 1] == 0
        if not np.any(path_valid):
            raise ValueError("held-out segment has no fully valid displacement paths")
        displacement = held_centers[lag:] - held_centers[:-lag]
        squared = np.sum(displacement[path_valid] ** 2, axis=1)
        predicted_msd.append(float(lag * mean_squared))
        observed_msd.append(float(np.mean(squared)))
        predicted_ngp.append(float(one_step_ngp / lag))
        observed_ngp.append(
            float(3.0 * np.mean(squared**2) / (5.0 * np.mean(squared) ** 2) - 1.0)
            if float(np.mean(squared)) > 0.0
            else 0.0
        )
        for wave_number in wave_numbers:
            observed_fs[float(wave_number)].append(
                float(np.mean(np.cos(wave_number * displacement[path_valid])))
            )
            predicted_fs[float(wave_number)].append(characteristic[float(wave_number)] ** lag)

    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    predicted_ngp_array = np.asarray(predicted_ngp)
    observed_ngp_array = np.asarray(observed_ngp)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_relative_errors = [
        abs(predicted / observed - 1.0)
        for wave_number in wave_numbers
        for predicted, observed in zip(predicted_fs[float(wave_number)], observed_fs[float(wave_number)])
        if abs(observed) > 1e-12
    ]
    return {
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "training_one_step_ngp": float(one_step_ngp),
        "observed_heldout_ngp_peak": float(np.max(observed_ngp_array)),
        "levy_ngp_max_absolute_error": float(np.max(np.abs(predicted_ngp_array - observed_ngp_array))),
        "levy_fs_max_relative_error": float(max(fs_relative_errors)) if fs_relative_errors else math.nan,
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "predicted_ngp": predicted_ngp_array,
        "observed_ngp": observed_ngp_array,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_ar1_empirical_innovation_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    simulation_count: int = 20000,
    seed: int = 20260721,
) -> dict[str, float | np.ndarray]:
    """Test a velocity-augmented non-Gaussian Markov Langevin closure.

    The scalar AR(1) coefficient and empirical innovation vectors are measured
    only from training center increments.  Bootstrap propagation then predicts
    held-out displacement moments and characteristic functions without fitting
    any macroscopic observable.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if not (3 < train_stop < len(centers) - 2) or frame_time <= 0.0:
        raise ValueError("train_stop and frame_time must leave train/held-out trajectories")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(centers) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be positive")
    if simulation_count < 100:
        raise ValueError("simulation_count must be at least 100")

    train_velocity = centers[1:train_stop] - centers[: train_stop - 1]
    increment_valid = valid[1:train_stop] & valid[: train_stop - 1]
    pair_valid = valid[: train_stop - 2] & valid[1 : train_stop - 1] & valid[2:train_stop]
    if not np.any(pair_valid) or not np.any(increment_valid):
        raise ValueError("training segment has insufficient valid center increments")
    previous = train_velocity[:-1][pair_valid]
    response = train_velocity[1:][pair_valid]
    denominator = float(np.sum(previous**2))
    if denominator <= 0.0:
        raise ValueError("training velocity must have positive variance")
    coefficient = float(np.sum(previous * response) / denominator)
    innovation = response - coefficient * previous
    initial_pool = train_velocity[increment_valid]

    held_centers = centers[train_stop:]
    held_valid = valid[train_stop:]
    invalid_prefix = np.concatenate(
        [np.zeros((1, centers.shape[1]), dtype=int), np.cumsum(~held_valid, axis=0)],
        axis=0,
    )
    maximum_lag = int(np.max(lags))
    rng = np.random.default_rng(seed)
    velocity = initial_pool[rng.integers(len(initial_pool), size=simulation_count)].copy()
    position = np.zeros_like(velocity)
    predicted_by_lag: dict[int, np.ndarray] = {}
    for lag in range(1, maximum_lag + 1):
        position += velocity
        if lag in set(int(value) for value in lags):
            predicted_by_lag[lag] = position.copy()
        velocity = coefficient * velocity + innovation[rng.integers(len(innovation), size=simulation_count)]

    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    predicted_ngp: list[float] = []
    observed_ngp: list[float] = []
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        path_valid = invalid_prefix[lag + 1 :] - invalid_prefix[: -lag - 1] == 0
        if not np.any(path_valid):
            raise ValueError("held-out segment has no fully valid displacement paths")
        observed = held_centers[lag:] - held_centers[:-lag]
        predicted = predicted_by_lag[int(lag)]
        observed_squared = np.sum(observed[path_valid] ** 2, axis=1)
        predicted_squared = np.sum(predicted**2, axis=1)
        observed_msd.append(float(np.mean(observed_squared)))
        predicted_msd.append(float(np.mean(predicted_squared)))
        observed_ngp.append(
            float(3.0 * np.mean(observed_squared**2) / (5.0 * np.mean(observed_squared) ** 2) - 1.0)
            if float(np.mean(observed_squared)) > 0.0
            else 0.0
        )
        predicted_ngp.append(
            float(3.0 * np.mean(predicted_squared**2) / (5.0 * np.mean(predicted_squared) ** 2) - 1.0)
            if float(np.mean(predicted_squared)) > 0.0
            else 0.0
        )
        for wave_number in wave_numbers:
            observed_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * observed[path_valid]))))
            predicted_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * predicted))))

    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    predicted_ngp_array = np.asarray(predicted_ngp)
    observed_ngp_array = np.asarray(observed_ngp)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_relative_errors = [
        abs(predicted / observed - 1.0)
        for wave_number in wave_numbers
        for predicted, observed in zip(predicted_fs[float(wave_number)], observed_fs[float(wave_number)])
        if abs(observed) > 1e-12
    ]
    return {
        "ar1_coefficient": coefficient,
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "ar1_fs_max_relative_error": float(max(fs_relative_errors)) if fs_relative_errors else math.nan,
        "ar1_ngp_max_absolute_error": float(np.max(np.abs(predicted_ngp_array - observed_ngp_array))),
        "observed_heldout_ngp_peak": float(np.max(observed_ngp_array)),
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "predicted_ngp": predicted_ngp_array,
        "observed_ngp": observed_ngp_array,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def time_split_arp_empirical_innovation_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    memory_order: int,
    simulation_count: int = 20000,
    seed: int = 20260722,
) -> dict[str, float | np.ndarray]:
    """Fit a finite-memory non-Gaussian discrete GLE and predict a holdout.

    ``memory_order`` is the number of past tagged-center velocities retained
    in the Mori-kernel approximation.  Innovation vectors are resampled
    empirically, retaining non-Gaussian noise without macro fitting.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2]:
        raise ValueError("centers and valid must align as (frames, particles, 3)/(frames, particles)")
    if memory_order < 1 or simulation_count < 100:
        raise ValueError("memory_order and simulation_count must be positive")
    if not (memory_order + 3 < train_stop < len(centers) - 2) or frame_time <= 0.0:
        raise ValueError("train_stop must leave enough training and held-out velocity histories")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(centers) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be positive")

    velocity = centers[1:] - centers[:-1]
    train_velocity = velocity[: train_stop - 1]
    train_rows: list[np.ndarray] = []
    train_target: list[np.ndarray] = []
    history_pool: list[np.ndarray] = []
    for time_index in range(memory_order, len(train_velocity)):
        needed_centers = valid[time_index - memory_order : time_index + 2]
        if not np.all(needed_centers, axis=0).any():
            continue
        usable = np.all(needed_centers, axis=0)
        for particle in np.flatnonzero(usable):
            history = np.stack([train_velocity[time_index - lag, particle] for lag in range(1, memory_order + 1)])
            train_rows.append(history)
            train_target.append(train_velocity[time_index, particle])
            history_pool.append(history)
    if not train_rows:
        raise ValueError("training segment has no complete velocity histories")
    histories = np.asarray(train_rows)
    targets = np.asarray(train_target)
    x_train = np.column_stack([histories[:, lag].reshape(-1) for lag in range(memory_order)])
    y_train = targets.reshape(-1)
    kernel, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    innovation = targets - np.einsum("hlc,l->hc", histories, kernel)

    held_rows: list[np.ndarray] = []
    held_target: list[np.ndarray] = []
    for time_index in range(max(memory_order, train_stop - 1), len(velocity)):
        if time_index + 1 >= len(valid):
            break
        needed_centers = valid[time_index - memory_order : time_index + 2]
        usable = np.all(needed_centers, axis=0)
        for particle in np.flatnonzero(usable):
            held_rows.append(
                np.stack([velocity[time_index - lag, particle] for lag in range(1, memory_order + 1)])
            )
            held_target.append(velocity[time_index, particle])
    if not held_rows:
        raise ValueError("held-out segment has no complete velocity histories")
    held_histories = np.asarray(held_rows)
    held_targets = np.asarray(held_target)
    held_prediction = np.einsum("hlc,l->hc", held_histories, kernel)
    total = float(np.sum((held_targets.reshape(-1) - np.mean(held_targets)) ** 2))
    one_step_r_squared = 1.0 - float(np.sum((held_targets - held_prediction) ** 2)) / total if total > 0.0 else math.nan

    rng = np.random.default_rng(seed)
    history_pool_array = np.asarray(history_pool)
    history = history_pool_array[rng.integers(len(history_pool_array), size=simulation_count)].copy()
    position = np.zeros((simulation_count, 3), dtype=float)
    maximum_lag = int(np.max(lags))
    predicted_by_lag: dict[int, np.ndarray] = {}
    for lag in range(1, maximum_lag + 1):
        next_velocity = np.einsum("slc,l->sc", history, kernel)
        next_velocity += innovation[rng.integers(len(innovation), size=simulation_count)]
        position += next_velocity
        history[:, 1:] = history[:, :-1]
        history[:, 0] = next_velocity
        if lag in set(int(value) for value in lags):
            predicted_by_lag[lag] = position.copy()

    held_centers = centers[train_stop:]
    held_valid = valid[train_stop:]
    invalid_prefix = np.concatenate([np.zeros((1, centers.shape[1]), dtype=int), np.cumsum(~held_valid, axis=0)], axis=0)
    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    predicted_ngp: list[float] = []
    observed_ngp: list[float] = []
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        path_valid = invalid_prefix[lag + 1 :] - invalid_prefix[: -lag - 1] == 0
        observed = held_centers[lag:] - held_centers[:-lag]
        if not np.any(path_valid):
            raise ValueError("held-out segment has no fully valid displacement paths")
        predicted = predicted_by_lag[int(lag)]
        observed_squared = np.sum(observed[path_valid] ** 2, axis=1)
        predicted_squared = np.sum(predicted**2, axis=1)
        observed_msd.append(float(np.mean(observed_squared)))
        predicted_msd.append(float(np.mean(predicted_squared)))
        observed_ngp.append(float(3.0 * np.mean(observed_squared**2) / (5.0 * np.mean(observed_squared) ** 2) - 1.0))
        predicted_ngp.append(float(3.0 * np.mean(predicted_squared**2) / (5.0 * np.mean(predicted_squared) ** 2) - 1.0))
        for wave_number in wave_numbers:
            observed_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * observed[path_valid]))))
            predicted_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * predicted))))
    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    predicted_ngp_array = np.asarray(predicted_ngp)
    observed_ngp_array = np.asarray(observed_ngp)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_errors = [abs(predicted / observed - 1.0) for k in wave_numbers for predicted, observed in zip(predicted_fs[float(k)], observed_fs[float(k)]) if abs(observed) > 1e-12]
    result: dict[str, float | np.ndarray] = {
        "heldout_one_step_r_squared": float(one_step_r_squared),
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "arp_fs_max_relative_error": float(max(fs_errors)) if fs_errors else math.nan,
        "arp_ngp_max_absolute_error": float(np.max(np.abs(predicted_ngp_array - observed_ngp_array))),
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "predicted_ngp": predicted_ngp_array,
        "observed_ngp": observed_ngp_array,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for lag, coefficient in enumerate(kernel, start=1):
        result[f"kernel_lag_{lag}"] = float(coefficient)
    return result


def time_split_force_auxiliary_markov_diagnostic(
    positions: np.ndarray,
    velocity: np.ndarray,
    force: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    simulation_count: int = 20000,
    seed: int = 20260820,
) -> dict[str, float | np.ndarray]:
    """Test a measured pair-force auxiliary Markov embedding for one particle.

    The resolved state is ``(v_i, F_i^KA)``.  Its isotropic linear transition
    and joint empirical innovation are measured from a training trajectory
    segment, then propagated autonomously to held-out displacement statistics.
    It is a finite-dimensional test of whether the physical pair force can
    represent the eliminated bath, rather than an assumed renewal clock.
    """

    positions = np.asarray(positions, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    force = np.asarray(force, dtype=float)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or velocity.shape != positions.shape or force.shape != positions.shape:
        raise ValueError("positions, velocity, and force must have matching (frames, particles, 3) shapes")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(force)):
        raise ValueError("positions, velocity, and force must be finite")
    if not (2 < train_stop < len(positions) - 2) or frame_time <= 0.0 or simulation_count < 100:
        raise ValueError("train_stop, frame_time, and simulation_count must be valid")
    if len(lags) == 0 or np.any(lags < 1) or np.any(lags >= len(positions) - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be positive")

    state = np.stack([velocity, force], axis=2)
    train_current = state[: train_stop - 1]
    train_next = state[1:train_stop]
    x_train = train_current.transpose(0, 1, 3, 2).reshape(-1, 2)
    y_train = train_next.transpose(0, 1, 3, 2).reshape(-1, 2)
    coefficient, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    transition = coefficient.T
    innovation = train_next - np.einsum("ab,tpbc->tpac", transition, train_current)

    held_current = state[train_stop:-1]
    held_next = state[train_stop + 1 :]
    held_prediction = np.einsum("ab,tpbc->tpac", transition, held_current)
    total = float(np.sum((held_next - np.mean(held_next)) ** 2))
    held_r_squared = 1.0 - float(np.sum((held_next - held_prediction) ** 2)) / total if total > 0.0 else math.nan

    rng = np.random.default_rng(seed)
    initial_pool = train_current.reshape(-1, 2, 3)
    innovation_pool = innovation.reshape(-1, 2, 3)
    simulated_state = initial_pool[rng.integers(len(initial_pool), size=simulation_count)].copy()
    displacement = np.zeros((simulation_count, 3), dtype=float)
    maximum_lag = int(np.max(lags))
    predicted_by_lag: dict[int, np.ndarray] = {}
    lag_set = set(int(lag) for lag in lags)
    for lag in range(1, maximum_lag + 1):
        displacement += frame_time * simulated_state[:, 0]
        if lag in lag_set:
            predicted_by_lag[lag] = displacement.copy()
        simulated_state = np.einsum("ab,sbc->sac", transition, simulated_state)
        simulated_state += innovation_pool[rng.integers(len(innovation_pool), size=simulation_count)]

    held_positions = positions[train_stop:]
    predicted_msd: list[float] = []
    observed_msd: list[float] = []
    predicted_ngp: list[float] = []
    observed_ngp: list[float] = []
    predicted_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    observed_fs: dict[float, list[float]] = {float(k): [] for k in wave_numbers}
    for lag in lags:
        predicted = predicted_by_lag[int(lag)]
        observed = held_positions[lag:] - held_positions[:-lag]
        predicted_squared = np.sum(predicted**2, axis=1)
        observed_squared = np.sum(observed**2, axis=2).reshape(-1)
        predicted_msd.append(float(np.mean(predicted_squared)))
        observed_msd.append(float(np.mean(observed_squared)))
        predicted_ngp.append(float(3.0 * np.mean(predicted_squared**2) / (5.0 * np.mean(predicted_squared) ** 2) - 1.0))
        observed_ngp.append(float(3.0 * np.mean(observed_squared**2) / (5.0 * np.mean(observed_squared) ** 2) - 1.0))
        for wave_number in wave_numbers:
            predicted_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * predicted))))
            observed_fs[float(wave_number)].append(float(np.mean(np.cos(wave_number * observed))))
    predicted_msd_array = np.asarray(predicted_msd)
    observed_msd_array = np.asarray(observed_msd)
    predicted_ngp_array = np.asarray(predicted_ngp)
    observed_ngp_array = np.asarray(observed_ngp)
    terminal_lag = int(lags[-1])
    predicted_diffusion = predicted_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    observed_diffusion = observed_msd_array[-1] / (6.0 * terminal_lag * frame_time)
    fs_errors = [
        abs(predicted / observed - 1.0)
        for wave_number in wave_numbers
        for predicted, observed in zip(predicted_fs[float(wave_number)], observed_fs[float(wave_number)])
        if abs(observed) > 1e-12
    ]
    return {
        "transition_matrix": transition,
        "heldout_state_r_squared": float(held_r_squared),
        "predicted_heldout_diffusion": float(predicted_diffusion),
        "observed_heldout_diffusion": float(observed_diffusion),
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "force_auxiliary_fs_max_relative_error": float(max(fs_errors)) if fs_errors else math.nan,
        "force_auxiliary_ngp_max_absolute_error": float(np.max(np.abs(predicted_ngp_array - observed_ngp_array))),
        "predicted_msd": predicted_msd_array,
        "observed_msd": observed_msd_array,
        "predicted_ngp": predicted_ngp_array,
        "observed_ngp": observed_ngp_array,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def state_dependent_increment_diagnostic(
    centers: np.ndarray,
    valid: np.ndarray,
    state: np.ndarray,
    *,
    train_stop: int,
    frame_time: float,
    bin_count: int = 5,
) -> dict[str, float]:
    """Test whether a positive microscopic state predicts center mobility.

    A power law ``q(state)=A state**p`` is inferred only from training
    increment second moments.  The held-out diffusion and conditional-Gaussian
    diffusivity-mixture NGP are then predictions, not macro fit inputs.
    """

    centers = np.asarray(centers, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    state = np.asarray(state, dtype=float)
    if centers.ndim != 3 or centers.shape[2] != 3 or valid.shape != centers.shape[:2] or state.shape != centers.shape[:2]:
        raise ValueError("centers, valid, and state must align on frame/particle axes")
    if not (3 < train_stop < len(centers) - 2) or frame_time <= 0.0 or bin_count < 2:
        raise ValueError("train_stop, frame_time, and bin_count must be valid")
    increments = centers[1:] - centers[:-1]
    increment_valid = valid[1:] & valid[:-1] & np.isfinite(state[:-1]) & (state[:-1] > 0.0)
    train_mask = increment_valid[: train_stop - 1]
    train_state = state[: train_stop - 1][train_mask]
    train_q = np.sum(increments[: train_stop - 1][train_mask] ** 2, axis=1)
    if len(train_state) < bin_count * 10:
        raise ValueError("training data are insufficient for state-mobility bins")
    log_state = np.log(train_state)
    edges = np.unique(np.quantile(log_state, np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) < 3:
        raise ValueError("microscopic state has insufficient variation")
    centers_x: list[float] = []
    centers_y: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (log_state >= left) & (log_state <= right if right == edges[-1] else log_state < right)
        if np.sum(selected) < 5:
            continue
        mean_q = float(np.mean(train_q[selected]))
        if mean_q > 0.0:
            centers_x.append(float(np.mean(log_state[selected])))
            centers_y.append(math.log(mean_q))
    if len(centers_x) < 2:
        raise ValueError("state-mobility bins have no positive second moments")
    slope, intercept = np.polyfit(np.asarray(centers_x), np.asarray(centers_y), 1)

    held_mask = increment_valid[train_stop:]
    held_state = state[train_stop:-1][held_mask]
    held_q = np.sum(increments[train_stop:][held_mask] ** 2, axis=1)
    if len(held_state) == 0:
        raise ValueError("held-out data contain no valid increments")
    predicted_q = np.exp(intercept + slope * np.log(held_state))
    predicted_mean_q = float(np.mean(predicted_q))
    observed_mean_q = float(np.mean(held_q))
    predicted_ngp = float(np.mean(predicted_q**2) / predicted_mean_q**2 - 1.0)
    observed_ngp = float(3.0 * np.mean(held_q**2) / (5.0 * observed_mean_q**2) - 1.0)
    return {
        "state_mobility_log_slope": float(slope),
        "state_mobility_log_intercept": float(intercept),
        "predicted_heldout_diffusion": predicted_mean_q / (6.0 * frame_time),
        "observed_heldout_diffusion": observed_mean_q / (6.0 * frame_time),
        "diffusion_relative_error": abs(predicted_mean_q / observed_mean_q - 1.0),
        "predicted_heldout_mixture_ngp": predicted_ngp,
        "observed_heldout_increment_ngp": observed_ngp,
        "mixture_ngp_absolute_error": abs(predicted_ngp - observed_ngp),
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
