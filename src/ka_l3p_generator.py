"""Microscopic third-generator quotient for the smooth KA cage coordinate."""

from __future__ import annotations

import math

import numpy as np

from ka_local_cage import ka_lj_sparse_force_generator_observables
from ka_smooth_cage import (
    smooth_cage_l2p_velocity_jacobian_batch,
    smooth_cage_second_generator_batch,
    smooth_force_support_cage_batch,
)


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


def _error_summary(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 1 or np.any(~np.isfinite(array)):
        return math.nan, math.nan
    return float(np.median(array)), float(np.quantile(array, 0.95))


def classify_l3p_numerical_canary(
    *,
    prefix_16_32_error: np.ndarray,
    position_primary_reference_error: np.ndarray,
    position_coarse_reference_error: np.ndarray,
    cage_primary_reference_error: np.ndarray,
    cage_coarse_reference_error: np.ndarray,
    acceleration_directional_error: np.ndarray,
) -> dict[str, float | str]:
    """Apply the frozen fail-closed numerical gates for microscopic ``L^3p``."""

    arrays = {
        "prefix_16_32": np.asarray(prefix_16_32_error, dtype=float),
        "position_primary_reference": np.asarray(
            position_primary_reference_error,
            dtype=float,
        ),
        "position_coarse_reference": np.asarray(
            position_coarse_reference_error,
            dtype=float,
        ),
        "cage_primary_reference": np.asarray(
            cage_primary_reference_error,
            dtype=float,
        ),
        "cage_coarse_reference": np.asarray(
            cage_coarse_reference_error,
            dtype=float,
        ),
        "acceleration_directional": np.asarray(
            acceleration_directional_error,
            dtype=float,
        ),
    }
    finite = all(value.size > 0 and np.all(np.isfinite(value)) for value in arrays.values())
    summaries = {key: _error_summary(value) for key, value in arrays.items()}
    prefix_gate = finite and summaries["prefix_16_32"][0] <= 0.10 and summaries[
        "prefix_16_32"
    ][1] <= 0.25
    position_gate = finite and summaries["position_primary_reference"][0] <= 0.02 and summaries[
        "position_primary_reference"
    ][1] <= 0.10
    cage_gate = finite and summaries["cage_primary_reference"][0] <= 0.02 and summaries[
        "cage_primary_reference"
    ][1] <= 0.10
    directional_gate = finite and summaries["acceleration_directional"][0] <= 0.02 and summaries[
        "acceleration_directional"
    ][1] <= 0.10
    monotonic_gate = finite and (
        summaries["position_primary_reference"][0]
        <= summaries["position_coarse_reference"][0]
        and summaries["cage_primary_reference"][0]
        <= summaries["cage_coarse_reference"][0]
    )
    passed = all(
        (
            finite,
            prefix_gate,
            position_gate,
            cage_gate,
            directional_gate,
            monotonic_gate,
        )
    )
    result: dict[str, float | str] = {
        "numerical_state": (
            "l3p_generator_numerically_resolved"
            if passed
            else "l3p_generator_numerically_unresolved"
        ),
        "l3p_numerical_gate_pass": float(passed),
        "finite_component_gate_pass": float(finite),
        "trace_prefix_gate_pass": float(prefix_gate),
        "position_step_gate_pass": float(position_gate),
        "cage_step_gate_pass": float(cage_gate),
        "step_monotonicity_gate_pass": float(monotonic_gate),
        "acceleration_directional_gate_pass": float(directional_gate),
        **_CLOSED_CLAIMS,
    }
    for key, (median, p95) in summaries.items():
        result[f"{key}_median_relative_error"] = median
        result[f"{key}_p95_relative_error"] = p95
    return result


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


def smooth_cage_l3p_generator_batch(
    positions: np.ndarray,
    *,
    velocities: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
    friction: float,
    temperature: float,
    trace_probes: np.ndarray,
    prefix_counts: tuple[int, ...] = (4, 8, 16, 32),
    position_step: float,
    cage_hessian_step: float,
    jacobian_step: float,
    l2p_directional_step: float,
    l2p_phase_space_step: float,
    potential_protocol: str = "ka_lj_cut",
    target_batch_size: int = 16,
) -> dict[str, np.ndarray | float | str]:
    """Evaluate ``L^3p`` from one full microscopic KA phase-space frame."""

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    particle_types = np.asarray(particle_types, dtype=int)
    box_lengths = np.asarray(box_lengths, dtype=float)
    targets = np.asarray(target_indices, dtype=int)
    probes = np.asarray(trace_probes, dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or velocities.shape != positions.shape
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(velocities))
    ):
        raise ValueError("positions and velocities must be aligned finite arrays")
    if particle_types.shape != (len(positions),):
        raise ValueError("particle_types must align with the phase-space frame")
    if box_lengths.shape != (3,) or np.any(~np.isfinite(box_lengths)):
        raise ValueError("box_lengths must be a finite three-vector")
    if (
        probes.ndim != 3
        or probes.shape[1:] != positions.shape
        or len(probes) < 1
        or np.any(~np.isfinite(probes))
    ):
        raise ValueError("trace_probes must be finite full configuration vectors")
    prefixes = _validated_prefix_counts(prefix_counts, len(probes))
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and nonnegative")
    steps = (
        position_step,
        cage_hessian_step,
        jacobian_step,
        l2p_directional_step,
        l2p_phase_space_step,
    )
    if any(not math.isfinite(step) or step <= 0.0 for step in steps):
        raise ValueError("all L3p numerical steps must be finite and positive")
    if potential_protocol not in {"ka_lj_cut", "ka_lj_c3_switch"}:
        raise ValueError("unsupported KA pair-potential protocol")

    all_particles = np.arange(len(positions), dtype=int)

    def force_state(
        configuration: np.ndarray,
        current_velocities: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        return ka_lj_sparse_force_generator_observables(
            configuration,
            velocities=current_velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=all_particles,
            potential_protocol=potential_protocol,
        )

    def zero_temperature_l2p(
        configuration: np.ndarray,
        current_velocities: np.ndarray,
    ) -> np.ndarray:
        microscopic = force_state(configuration, current_velocities)
        return np.asarray(
            smooth_cage_second_generator_batch(
                configuration,
                velocities=current_velocities,
                forces=np.asarray(microscopic["force"]),
                force_generator=np.asarray(microscopic["force_generator"]),
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=targets,
                friction=friction,
                temperature=0.0,
                directional_step=l2p_directional_step,
                phase_space_step=l2p_phase_space_step,
                trace_probes=None,
                target_batch_size=target_batch_size,
            )["second_relative_generator"],
            dtype=float,
        )

    outer_step = float(position_step)
    position_transport = (
        zero_temperature_l2p(
            positions + outer_step * velocities,
            velocities,
        )
        - zero_temperature_l2p(
            positions - outer_step * velocities,
            velocities,
        )
    ) / (2.0 * outer_step)

    base_force_state = force_state(positions, velocities)
    forces = np.asarray(base_force_state["force"], dtype=float)
    acceleration = forces - float(friction) * velocities
    jacobian_result = smooth_cage_l2p_velocity_jacobian_batch(
        positions,
        velocities=velocities,
        forces=forces,
        particle_types=particle_types,
        box_lengths=box_lengths,
        target_indices=targets,
        friction=friction,
        jacobian_step=jacobian_step,
        potential_protocol=potential_protocol,
        target_batch_size=target_batch_size,
    )
    l2p_velocity_jacobian = np.asarray(
        jacobian_result["l2p_velocity_jacobian"],
        dtype=float,
    )

    if temperature == 0.0:
        prefix_shape = (len(prefixes), len(targets), 3)
        laplacian = np.zeros(prefix_shape)
        laplacian_gradient = np.zeros(prefix_shape)
    else:
        laplacian_arguments = {
            "particle_types": particle_types,
            "box_lengths": box_lengths,
            "target_indices": targets,
            "trace_probes": probes,
            "directional_step": cage_hessian_step,
            "prefix_counts": prefixes,
            "target_batch_size": target_batch_size,
        }
        laplacian = np.asarray(
            smooth_cage_laplacian_prefixes(
                positions,
                **laplacian_arguments,
            )["laplacian_prefixes"],
            dtype=float,
        )
        plus_laplacian = np.asarray(
            smooth_cage_laplacian_prefixes(
                positions + outer_step * velocities,
                **laplacian_arguments,
            )["laplacian_prefixes"],
            dtype=float,
        )
        minus_laplacian = np.asarray(
            smooth_cage_laplacian_prefixes(
                positions - outer_step * velocities,
                **laplacian_arguments,
            )["laplacian_prefixes"],
            dtype=float,
        )
        laplacian_gradient = (
            plus_laplacian - minus_laplacian
        ) / (2.0 * outer_step)

    assembled = assemble_l3p_generator(
        position_transport_term=position_transport,
        l2p_velocity_jacobian=l2p_velocity_jacobian,
        acceleration=acceleration,
        laplacian_prefixes=laplacian,
        laplacian_velocity_derivative_prefixes=laplacian_gradient,
        friction=friction,
        temperature=temperature,
    )
    return {
        **assembled,
        "l2p_velocity_jacobian": l2p_velocity_jacobian,
        "acceleration": acceleration,
        "force": forces,
        "force_generator": np.asarray(base_force_state["force_generator"]),
        "prefix_counts": np.asarray(prefixes, dtype=int),
        "position_step": float(position_step),
        "cage_hessian_step": float(cage_hessian_step),
        "jacobian_step": float(jacobian_step),
        "l2p_directional_step": float(l2p_directional_step),
        "l2p_phase_space_step": float(l2p_phase_space_step),
        "pair_count": float(jacobian_result["pair_count"]),
        "potential_protocol": potential_protocol,
        **_CLOSED_CLAIMS,
    }
