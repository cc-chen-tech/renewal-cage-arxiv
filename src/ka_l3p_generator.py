"""Microscopic third-generator quotient for the smooth KA cage coordinate."""

from __future__ import annotations

import math

import numpy as np

from ka_smooth_cage import smooth_force_support_cage_batch


_CLOSED_CLAIMS = {
    "finite_l3p_gaussian_closure_supported": 0.0,
    "microscopic_environment_coordinate_z_allowed": 0.0,
    "continuous_gaussian_langevin_bath_allowed": 0.0,
    "autonomous_single_particle_gle_allowed": 0.0,
    "complete_event_clock_closure_allowed": 0.0,
    "kramers_escape_claim_allowed": 0.0,
    "spatial_facilitation_claim_allowed": 0.0,
    "thermodynamic_claim_allowed": 0.0,
}


def _validated_prefix_counts(
    prefix_counts: tuple[int, ...],
    probe_count: int,
) -> tuple[int, ...]:
    raw = np.asarray(prefix_counts)
    if (
        raw.ndim != 1
        or len(raw) < 1
        or np.any(raw != raw.astype(int))
    ):
        raise ValueError("prefix_counts must be a nonempty integer sequence")
    prefixes = tuple(int(value) for value in raw)
    if (
        any(value < 1 for value in prefixes)
        or tuple(sorted(set(prefixes))) != prefixes
        or prefixes[-1] > probe_count
    ):
        raise ValueError("prefix_counts must be sorted unique available prefixes")
    return prefixes


def smooth_cage_laplacian_prefixes(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    trace_probes: np.ndarray,
    directional_step: float,
    prefix_counts: tuple[int, ...] = (4, 8, 16, 32),
    target_batch_size: int = 16,
) -> dict[str, np.ndarray | float]:
    """Estimate ``Delta_R u`` from paired cage-Hessian trace actions."""

    positions = np.asarray(positions, dtype=float)
    probes = np.asarray(trace_probes, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or np.any(~np.isfinite(positions))
    ):
        raise ValueError("positions must be a finite (particles, 3) array")
    if (
        probes.ndim != 3
        or probes.shape[1:] != positions.shape
        or len(probes) < 1
        or np.any(~np.isfinite(probes))
    ):
        raise ValueError("trace_probes must be finite full configuration vectors")
    if not math.isfinite(directional_step) or directional_step <= 0.0:
        raise ValueError("directional_step must be finite and positive")
    prefixes = _validated_prefix_counts(prefix_counts, len(probes))
    common = {
        "particle_types": np.asarray(particle_types, dtype=int),
        "box_lengths": np.asarray(box_lengths, dtype=float),
        "target_indices": np.asarray(target_indices, dtype=int),
        "target_batch_size": target_batch_size,
        "compute_gram": False,
    }
    step = float(directional_step)
    actions = []
    for probe in probes:
        plus = smooth_force_support_cage_batch(
            positions + step * probe,
            velocities=probe,
            **common,
        )["relative_velocity"]
        minus = smooth_force_support_cage_batch(
            positions - step * probe,
            velocities=probe,
            **common,
        )["relative_velocity"]
        actions.append((np.asarray(plus) - np.asarray(minus)) / (2.0 * step))
    hessian_actions = np.asarray(actions)
    laplacian_prefixes = np.asarray(
        [np.mean(hessian_actions[:prefix], axis=0) for prefix in prefixes]
    )
    return {
        "hessian_trace_actions": hessian_actions,
        "laplacian_prefixes": laplacian_prefixes,
        "prefix_counts": np.asarray(prefixes, dtype=int),
        "directional_step": step,
        **_CLOSED_CLAIMS,
    }


def assemble_l3p_generator(
    *,
    position_transport_term: np.ndarray,
    l2p_velocity_jacobian: np.ndarray,
    acceleration: np.ndarray,
    laplacian_prefixes: np.ndarray,
    laplacian_velocity_derivative_prefixes: np.ndarray,
    friction: float,
    temperature: float,
) -> dict[str, np.ndarray | float]:
    """Assemble the exact ``L^3p`` quotient from validated components."""

    position = np.asarray(position_transport_term, dtype=float)
    jacobian = np.asarray(l2p_velocity_jacobian, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    laplacian = np.asarray(laplacian_prefixes, dtype=float)
    laplacian_gradient = np.asarray(
        laplacian_velocity_derivative_prefixes,
        dtype=float,
    )
    if (
        position.ndim != 2
        or position.shape[1] != 3
        or jacobian.ndim != 4
        or jacobian.shape[:2] != position.shape
        or jacobian.shape[3] != 3
        or acceleration.shape != (jacobian.shape[2], 3)
        or laplacian.ndim != 3
        or laplacian.shape[1:] != position.shape
        or laplacian_gradient.shape != laplacian.shape
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(jacobian))
        or np.any(~np.isfinite(acceleration))
        or np.any(~np.isfinite(laplacian))
        or np.any(~np.isfinite(laplacian_gradient))
    ):
        raise ValueError("L3p components must be aligned finite arrays")
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and nonnegative")

    acceleration_term = np.einsum("tanb,nb->ta", jacobian, acceleration)
    thermal_gradient = (
        8.0 * float(friction) * float(temperature) * laplacian_gradient
    )
    thermal_friction = (
        -6.0 * float(friction) ** 2 * float(temperature) * laplacian
    )
    l3p_prefixes = (
        position[None]
        + acceleration_term[None]
        + thermal_gradient
        + thermal_friction
    )
    return {
        "l3p": l3p_prefixes[-1],
        "l3p_prefixes": l3p_prefixes,
        "position_transport_term": position,
        "acceleration_response_term": acceleration_term,
        "thermal_gradient_prefixes": thermal_gradient,
        "thermal_friction_prefixes": thermal_friction,
        "laplacian_prefixes": laplacian,
        "laplacian_velocity_derivative_prefixes": laplacian_gradient,
        "friction": float(friction),
        "temperature": float(temperature),
        **_CLOSED_CLAIMS,
    }
