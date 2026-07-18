#!/usr/bin/env python3
"""Cache deterministic microscopic conditional diffusion of ``L^2 p``."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_projected_ito_innovations import load_drift_cache  # noqa: E402
from cache_ka_l2p_conditional_diffusion import (  # noqa: E402
    atomic_savez,
    file_sha256,
    load_second_generator_cache,
    validate_cache_alignment,
)
from ka_l2p_conditional_diffusion import (  # noqa: E402
    classify_deterministic_numerical_canary,
    deterministic_conditional_diffusion,
)
from ka_local_cage import (  # noqa: E402
    ka_lj_sparse_force_generator_multi,
    ka_lj_sparse_force_generator_observables,
)
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_smooth_cage import (  # noqa: E402
    smooth_cage_l2p_velocity_directional_derivative_batch,
    smooth_cage_l2p_velocity_jacobian_batch,
)


_PRIMARY_STEP = 1e-5
_REFERENCE_STEP = 3e-6
_COARSE_STEP = 3e-5
_DIRECTION_SEEDS = (20260721, 20260722, 20260723, 20260724)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-directory", type=Path, required=True)
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--second-generator-cache-directory", type=Path, required=True)
    parser.add_argument("--output-cache-directory", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--clone-indices", type=int, nargs="+")
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--maximum-frame-count", type=int, default=200)
    parser.add_argument("--primary-jacobian-step", type=float, default=_PRIMARY_STEP)
    parser.add_argument("--reference-jacobian-step", type=float, default=_REFERENCE_STEP)
    parser.add_argument("--coarse-jacobian-step", type=float, default=_COARSE_STEP)
    parser.add_argument("--sensitivity-frame-count", type=int, default=1)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--phase-space-step", type=float, default=3e-6)
    parser.add_argument("--velocity-step", type=float, default=2e-5)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.expected_clone_count < 1
        or args.target_count < 1
        or args.maximum_frame_count < 1
        or args.sensitivity_frame_count < 1
        or args.sensitivity_frame_count > args.maximum_frame_count
        or args.friction <= 0.0
        or not math.isfinite(args.friction)
        or args.temperature <= 0.0
        or not math.isfinite(args.temperature)
        or args.target_batch_size < 1
        or args.checkpoint_interval < 1
    ):
        raise ValueError("invalid deterministic L2p cache controls")
    frozen = (
        (args.primary_jacobian_step, _PRIMARY_STEP),
        (args.reference_jacobian_step, _REFERENCE_STEP),
        (args.coarse_jacobian_step, _COARSE_STEP),
        (args.directional_step, 1e-5),
        (args.phase_space_step, 3e-6),
        (args.velocity_step, 2e-5),
    )
    if any(not math.isclose(value, expected, rel_tol=0.0, abs_tol=0.0) for value, expected in frozen):
        raise ValueError("deterministic L2p numerical steps must match the frozen canary")


def relative_frobenius(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim < 2:
        raise ValueError("candidate and reference tensors must align")
    axes = tuple(range(1, candidate.ndim))
    numerator = np.sqrt(np.sum((candidate - reference) ** 2, axis=axes))
    denominator = np.sqrt(np.sum(reference**2, axis=axes))
    return numerator / np.maximum(denominator, 1e-30)


def _unresolved_verdict() -> dict[str, float | str]:
    return {
        "numerical_state": "deterministic_jacobian_numerically_unresolved",
        "deterministic_numerical_gate_pass": 0.0,
        "a_step_gate_pass": 0.0,
        "q_step_gate_pass": 0.0,
        "directional_identity_gate_pass": 0.0,
        "step_monotonicity_gate_pass": 0.0,
        "positive_semidefinite_gate_pass": 0.0,
        "microscopic_environment_coordinate_z_allowed": 0.0,
        "continuous_gaussian_langevin_bath_allowed": 0.0,
        "autonomous_single_particle_gle_allowed": 0.0,
        "complete_event_clock_closure_allowed": 0.0,
        "kramers_escape_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def cache_clone(
    trajectory_path: Path,
    drift_path: Path,
    second_path: Path,
    output_path: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    trajectory_hash = file_sha256(trajectory_path)
    drift = load_drift_cache(drift_path)
    second = load_second_generator_cache(second_path)
    validate_cache_alignment(
        trajectory_sha256=trajectory_hash,
        drift=drift,
        second=second,
        target_count=args.target_count,
    )
    trajectory = load_lammps_custom_trajectory(
        trajectory_path,
        maximum_frame_count=args.maximum_frame_count,
    )
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    velocities = np.asarray(trajectory.get("velocities"), dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    targets = np.asarray(drift["target_indices"], dtype=int)
    second_values = np.asarray(second["second_relative_generator"], dtype=float)
    frame_count = min(args.maximum_frame_count, len(positions), len(second_values))
    if positions.shape != velocities.shape or frame_count < 1:
        raise ValueError("trajectory and second-generator cache are not aligned")
    positions = positions[:frame_count]
    velocities = velocities[:frame_count]
    protocol = str(second["potential_protocol"])
    sensitivity_count = min(args.sensitivity_frame_count, frame_count)
    direction_count = len(_DIRECTION_SEEDS)

    primary_q = np.full((frame_count, len(targets), 3, 3), np.nan)
    a_primary_error = np.full((sensitivity_count, len(targets)), np.nan)
    q_primary_error = np.full_like(a_primary_error, np.nan)
    a_coarse_error = np.full_like(a_primary_error, np.nan)
    q_coarse_error = np.full_like(a_primary_error, np.nan)
    directional_error = np.full(
        (sensitivity_count, direction_count, len(targets)), np.nan
    )
    q_minimum_eigenvalue = np.full((sensitivity_count, len(targets)), np.nan)
    q_trace = np.full_like(q_minimum_eigenvalue, np.nan)
    completed = 0

    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as saved:
            valid = (
                str(saved["trajectory_sha256"]) == trajectory_hash
                and np.array_equal(saved["target_indices"], targets)
                and str(saved["potential_protocol"]) == protocol
                and str(saved["estimator"]) == "deterministic_velocity_jacobian"
                and int(saved["requested_frame_count"]) == frame_count
                and math.isclose(float(saved["primary_jacobian_step"]), _PRIMARY_STEP)
                and math.isclose(float(saved["reference_jacobian_step"]), _REFERENCE_STEP)
                and math.isclose(float(saved["coarse_jacobian_step"]), _COARSE_STEP)
                and int(saved["sensitivity_frame_count"]) == sensitivity_count
                and int(saved["target_batch_size"]) == args.target_batch_size
                and np.array_equal(saved["direction_seeds"], _DIRECTION_SEEDS)
            )
            if not valid:
                raise ValueError("existing deterministic diffusion cache has mismatched provenance")
            completed = int(saved["completed_frame_count"])
            primary_q = np.asarray(saved["l2p_conditional_diffusion"]).copy()
            a_primary_error = np.asarray(saved["a_primary_reference_error"]).copy()
            q_primary_error = np.asarray(saved["q_primary_reference_error"]).copy()
            a_coarse_error = np.asarray(saved["a_coarse_reference_error"]).copy()
            q_coarse_error = np.asarray(saved["q_coarse_reference_error"]).copy()
            directional_error = np.asarray(saved["directional_response_error"]).copy()
            q_minimum_eigenvalue = np.asarray(saved["q_minimum_eigenvalue"]).copy()
            q_trace = np.asarray(saved["q_trace"]).copy()
            if completed < 0 or completed > frame_count:
                raise ValueError("existing deterministic cache has invalid completion count")

    def numerical_verdict() -> dict[str, float | str]:
        finite = (
            np.all(np.isfinite(a_primary_error))
            and np.all(np.isfinite(q_primary_error))
            and np.all(np.isfinite(a_coarse_error))
            and np.all(np.isfinite(q_coarse_error))
            and np.all(np.isfinite(directional_error))
            and np.all(np.isfinite(q_minimum_eigenvalue))
            and np.all(np.isfinite(q_trace))
        )
        if not finite:
            return _unresolved_verdict()
        return classify_deterministic_numerical_canary(
            a_primary_reference_error=a_primary_error.ravel(),
            q_primary_reference_error=q_primary_error.ravel(),
            directional_response_error=directional_error.ravel(),
            a_coarse_reference_error=a_coarse_error.ravel(),
            q_coarse_reference_error=q_coarse_error.ravel(),
            q_minimum_eigenvalue=q_minimum_eigenvalue.ravel(),
            q_trace=q_trace.ravel(),
        )

    def save(done: int) -> None:
        verdict = numerical_verdict()
        atomic_savez(
            output_path,
            l2p_conditional_diffusion=primary_q,
            a_primary_reference_error=a_primary_error,
            q_primary_reference_error=q_primary_error,
            a_coarse_reference_error=a_coarse_error,
            q_coarse_reference_error=q_coarse_error,
            directional_response_error=directional_error,
            q_minimum_eigenvalue=q_minimum_eigenvalue,
            q_trace=q_trace,
            target_indices=targets,
            trajectory_sha256=np.asarray(trajectory_hash),
            potential_protocol=np.asarray(protocol),
            estimator=np.asarray("deterministic_velocity_jacobian"),
            primary_jacobian_step=_PRIMARY_STEP,
            reference_jacobian_step=_REFERENCE_STEP,
            coarse_jacobian_step=_COARSE_STEP,
            direction_seeds=np.asarray(_DIRECTION_SEEDS, dtype=int),
            directional_step=float(args.directional_step),
            phase_space_step=float(args.phase_space_step),
            velocity_step=float(args.velocity_step),
            friction=float(args.friction),
            temperature=float(args.temperature),
            completed_frame_count=float(done),
            requested_frame_count=float(frame_count),
            sensitivity_frame_count=float(sensitivity_count),
            target_batch_size=float(args.target_batch_size),
            **verdict,
        )

    directions = np.asarray(
        [
            np.random.default_rng(seed).normal(size=velocities.shape[1:])
            for seed in _DIRECTION_SEEDS
        ]
    )
    all_particles = np.arange(len(particle_types), dtype=int)
    for frame in range(completed, frame_count):
        base_force = ka_lj_sparse_force_generator_observables(
            positions[frame],
            velocities=velocities[frame],
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=all_particles,
            potential_protocol=protocol,
        )
        forces = np.asarray(base_force["force"])
        matrices: dict[float, np.ndarray] = {}
        diffusions: dict[float, np.ndarray] = {}
        steps = (_PRIMARY_STEP,)
        if frame < sensitivity_count:
            steps = (_COARSE_STEP, _PRIMARY_STEP, _REFERENCE_STEP)
        for step in steps:
            matrix = np.asarray(
                smooth_cage_l2p_velocity_jacobian_batch(
                    positions[frame],
                    velocities=velocities[frame],
                    forces=forces,
                    particle_types=particle_types,
                    box_lengths=box_lengths,
                    target_indices=targets,
                    friction=args.friction,
                    jacobian_step=step,
                    potential_protocol=protocol,
                    target_batch_size=args.target_batch_size,
                )["l2p_velocity_jacobian"]
            )
            matrices[step] = matrix
            diffusions[step] = np.asarray(
                deterministic_conditional_diffusion(
                    matrix,
                    friction=args.friction,
                    temperature=args.temperature,
                )["conditional_diffusion"]
            )
        primary_q[frame] = diffusions[_PRIMARY_STEP]

        if frame < sensitivity_count:
            a_primary_error[frame] = relative_frobenius(
                matrices[_PRIMARY_STEP], matrices[_REFERENCE_STEP]
            )
            q_primary_error[frame] = relative_frobenius(
                diffusions[_PRIMARY_STEP], diffusions[_REFERENCE_STEP]
            )
            a_coarse_error[frame] = relative_frobenius(
                matrices[_COARSE_STEP], matrices[_REFERENCE_STEP]
            )
            q_coarse_error[frame] = relative_frobenius(
                diffusions[_COARSE_STEP], diffusions[_REFERENCE_STEP]
            )
            q_minimum_eigenvalue[frame] = np.linalg.eigvalsh(
                diffusions[_PRIMARY_STEP]
            )[:, 0]
            q_trace[frame] = np.trace(
                diffusions[_PRIMARY_STEP], axis1=-2, axis2=-1
            )
            force_directions = np.asarray(
                ka_lj_sparse_force_generator_multi(
                    positions[frame],
                    velocity_fields=directions,
                    particle_types=particle_types,
                    box_lengths=box_lengths,
                    target_indices=all_particles,
                    potential_protocol=protocol,
                )["force_generator"]
            )
            for slot, (direction, force_direction) in enumerate(
                zip(directions, force_directions)
            ):
                expected = np.asarray(
                    smooth_cage_l2p_velocity_directional_derivative_batch(
                        positions[frame],
                        velocities=velocities[frame],
                        velocity_direction=direction,
                        forces=forces,
                        force_generator=np.asarray(base_force["force_generator"]),
                        force_generator_direction=force_direction,
                        particle_types=particle_types,
                        box_lengths=box_lengths,
                        target_indices=targets,
                        friction=args.friction,
                        directional_step=args.directional_step,
                        phase_space_step=args.phase_space_step,
                        velocity_step=args.velocity_step,
                        target_batch_size=args.target_batch_size,
                    )["l2p_velocity_directional_derivative"]
                )
                actual = np.einsum(
                    "tanb,nb->ta", matrices[_PRIMARY_STEP], direction
                )
                directional_error[frame, slot] = np.linalg.norm(
                    actual - expected, axis=1
                ) / np.maximum(np.linalg.norm(expected, axis=1), 1e-30)

        done = frame + 1
        if done % args.checkpoint_interval == 0 or done == frame_count:
            save(done)
        print(f"{output_path.name}: frame {done}/{frame_count}", flush=True)

    return {
        "trajectory_sha256": trajectory_hash,
        "completed_frame_count": frame_count,
        "cache": str(output_path.resolve()),
        **numerical_verdict(),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    trajectories = sorted(
        path / "trajectory.lammpstrj"
        for path in args.clone_directory.glob("clone_*")
        if (path / "trajectory.lammpstrj").is_file()
    )
    if len(trajectories) != args.expected_clone_count:
        raise ValueError("completed clone count does not match the frozen protocol")
    selected = (
        tuple(range(1, len(trajectories) + 1))
        if args.clone_indices is None
        else tuple(args.clone_indices)
    )
    if len(set(selected)) != len(selected) or any(
        index < 1 or index > len(trajectories) for index in selected
    ):
        raise ValueError("clone indices must be unique completed clone ordinals")
    for index in selected:
        cache_clone(
            trajectories[index - 1],
            args.drift_cache_directory / f"clone_{index:03d}_decomposed_drift.npz",
            args.second_generator_cache_directory
            / f"clone_{index:03d}_relative_second_generator.npz",
            args.output_cache_directory
            / f"clone_{index:03d}_l2p_deterministic_diffusion.npz",
            args=args,
        )


if __name__ == "__main__":
    main()
