#!/usr/bin/env python3
"""Cache probe-converged conditional diffusion of the microscopic ``L^2 p`` coordinate."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_projected_ito_innovations import load_drift_cache  # noqa: E402
from ka_l2p_conditional_diffusion import (  # noqa: E402
    nested_diffusion_estimates,
    rademacher_velocity_probes,
)
from ka_local_cage import (  # noqa: E402
    ka_lj_sparse_force_generator_multi,
    ka_lj_sparse_force_generator_observables,
)
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_smooth_cage import (  # noqa: E402
    smooth_cage_l2p_velocity_directional_derivative_batch,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: np.ndarray | float | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_second_generator_cache(path: Path) -> dict[str, np.ndarray | float | str]:
    required = (
        "second_relative_generator",
        "target_indices",
        "trajectory_sha256",
        "potential_protocol",
        "trace_probe_count",
        "thermodynamic_claim_allowed",
    )
    with np.load(path, allow_pickle=False) as cache:
        if any(key not in cache for key in required):
            raise ValueError(f"incomplete second-generator cache: {path}")
        return {key: np.asarray(cache[key]) for key in required}


def validate_cache_alignment(
    *,
    trajectory_sha256: str,
    drift: dict[str, np.ndarray | float | str],
    second: dict[str, np.ndarray | float | str],
    target_count: int,
) -> None:
    if (
        str(drift["trajectory_sha256"]) != trajectory_sha256
        or str(second["trajectory_sha256"]) != trajectory_sha256
    ):
        raise ValueError("trajectory SHA256 does not match microscopic caches")
    drift_targets = np.asarray(drift["target_indices"], dtype=int)
    second_targets = np.asarray(second["target_indices"], dtype=int)
    if (
        len(drift_targets) != target_count
        or not np.array_equal(drift_targets, second_targets)
    ):
        raise ValueError("target indices do not match the conditional-diffusion protocol")
    if str(second["potential_protocol"]) not in {"ka_lj_cut", "ka_lj_c3_switch"}:
        raise ValueError("potential protocol is unsupported")
    if float(second["thermodynamic_claim_allowed"]) != 0.0:
        raise ValueError("claim boundary is open in the second-generator cache")


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
    parser.add_argument("--probe-prefix-counts", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--probe-seed", type=int, default=20260719)
    parser.add_argument("--velocity-step", type=float, default=1e-5)
    parser.add_argument(
        "--sensitivity-velocity-steps",
        type=float,
        nargs="+",
        default=[3e-6, 3e-5],
    )
    parser.add_argument("--sensitivity-frame-count", type=int, default=4)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--phase-space-step", type=float, default=1e-5)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    prefixes = tuple(sorted(set(args.probe_prefix_counts)))
    if (
        args.expected_clone_count < 1
        or args.target_count < 1
        or args.maximum_frame_count < 2
        or not prefixes
        or prefixes[0] < 1
        or any(step <= 0.0 or not math.isfinite(step) for step in args.sensitivity_velocity_steps)
        or args.velocity_step <= 0.0
        or args.friction <= 0.0
        or args.temperature <= 0.0
        or args.directional_step <= 0.0
        or args.phase_space_step <= 0.0
        or args.sensitivity_frame_count < 0
        or args.target_batch_size < 1
        or args.checkpoint_interval < 1
    ):
        raise ValueError("invalid L2p conditional-diffusion cache controls")
    return prefixes


def relative_frobenius(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(reference, axis=(-2, -1))
    return np.linalg.norm(candidate - reference, axis=(-2, -1)) / np.maximum(
        denominator, 1e-30
    )


def cache_clone(
    trajectory_path: Path,
    drift_path: Path,
    second_path: Path,
    output_path: Path,
    *,
    args: argparse.Namespace,
    prefixes: tuple[int, ...],
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
    trajectory = load_lammps_custom_trajectory(trajectory_path)
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    if "velocities" not in trajectory:
        raise ValueError("trajectory must contain microscopic velocities")
    velocities = np.asarray(trajectory["velocities"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    targets = np.asarray(drift["target_indices"], dtype=int)
    second_values = np.asarray(second["second_relative_generator"], dtype=float)
    frame_count = min(args.maximum_frame_count, len(positions), len(second_values))
    if positions.shape != velocities.shape or frame_count < 2:
        raise ValueError("trajectory and second-generator cache are not aligned")

    probes = rademacher_velocity_probes(
        probe_count=prefixes[-1],
        particle_count=len(particle_types),
        seed=args.probe_seed,
    )
    primary = np.full((frame_count, len(targets), 3, 3), np.nan)
    prefix_error = np.full((frame_count, len(prefixes) - 1, len(targets)), np.nan)
    sensitivity_error = np.full(
        (min(args.sensitivity_frame_count, frame_count), len(args.sensitivity_velocity_steps), len(targets)),
        np.nan,
    )
    completed = 0
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as saved:
            valid = (
                str(saved["trajectory_sha256"]) == trajectory_hash
                and np.array_equal(saved["target_indices"], targets)
                and np.array_equal(saved["probe_prefix_counts"], prefixes)
                and int(saved["probe_seed"]) == args.probe_seed
                and math.isclose(float(saved["velocity_step"]), args.velocity_step)
                and int(saved["requested_frame_count"]) == frame_count
            )
            if not valid:
                raise ValueError("existing conditional-diffusion cache has mismatched provenance")
            completed = int(saved["completed_frame_count"])
            primary = np.asarray(saved["l2p_conditional_diffusion"]).copy()
            prefix_error = np.asarray(saved["prefix_relative_frobenius_error"]).copy()
            sensitivity_error = np.asarray(saved["step_relative_frobenius_error"]).copy()

    protocol = str(second["potential_protocol"])
    def save(done: int) -> None:
        atomic_savez(
            output_path,
            l2p_conditional_diffusion=primary,
            prefix_relative_frobenius_error=prefix_error,
            step_relative_frobenius_error=sensitivity_error,
            target_indices=targets,
            trajectory_sha256=np.asarray(trajectory_hash),
            potential_protocol=np.asarray(protocol),
            probe_prefix_counts=np.asarray(prefixes, dtype=int),
            probe_seed=float(args.probe_seed),
            velocity_step=float(args.velocity_step),
            sensitivity_velocity_steps=np.asarray(args.sensitivity_velocity_steps),
            friction=float(args.friction),
            temperature=float(args.temperature),
            directional_step=float(args.directional_step),
            phase_space_step=float(args.phase_space_step),
            completed_frame_count=float(done),
            requested_frame_count=float(frame_count),
            thermodynamic_claim_allowed=0.0,
        )

    for frame in range(completed, frame_count):
        steps = [args.velocity_step]
        if frame < len(sensitivity_error):
            steps.extend(args.sensitivity_velocity_steps)
        responses_by_step = {
            step: np.empty((len(probes), len(targets), 3), dtype=float)
            for step in steps
        }
        all_particles = np.arange(len(particle_types), dtype=int)
        base = ka_lj_sparse_force_generator_observables(
            positions[frame],
            velocities=velocities[frame],
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=all_particles,
            potential_protocol=protocol,
        )
        directional_force_generators = np.asarray(
            ka_lj_sparse_force_generator_multi(
                positions[frame],
                velocity_fields=probes,
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=all_particles,
                potential_protocol=protocol,
            )["force_generator"]
        )
        for probe_index, probe in enumerate(probes):
            for step in steps:
                responses_by_step[step][probe_index] = (
                    smooth_cage_l2p_velocity_directional_derivative_batch(
                        positions[frame],
                        velocities=velocities[frame],
                        velocity_direction=probe,
                        forces=np.asarray(base["force"]),
                        force_generator=np.asarray(base["force_generator"]),
                        force_generator_direction=directional_force_generators[
                            probe_index
                        ],
                        particle_types=particle_types,
                        box_lengths=box_lengths,
                        target_indices=targets,
                        friction=args.friction,
                        directional_step=args.directional_step,
                        phase_space_step=args.phase_space_step,
                        velocity_step=step,
                        target_batch_size=args.target_batch_size,
                    )["l2p_velocity_directional_derivative"]
                )

        estimates = nested_diffusion_estimates(
            responses_by_step[args.velocity_step],
            prefix_counts=prefixes,
            friction=args.friction,
            temperature=args.temperature,
        )["diffusion_prefixes"]
        primary[frame] = estimates[-1]
        for slot, estimate in enumerate(estimates[:-1]):
            prefix_error[frame, slot] = relative_frobenius(estimate, estimates[-1])
        if frame < len(sensitivity_error):
            for slot, step in enumerate(args.sensitivity_velocity_steps):
                sensitivity = nested_diffusion_estimates(
                    responses_by_step[step],
                    prefix_counts=(prefixes[-1],),
                    friction=args.friction,
                    temperature=args.temperature,
                )["diffusion_prefixes"][0]
                sensitivity_error[frame, slot] = relative_frobenius(
                    sensitivity, estimates[-1]
                )
        if (frame + 1) % args.checkpoint_interval == 0 or frame + 1 == frame_count:
            save(frame + 1)
        print(f"{output_path.name}: frame {frame + 1}/{frame_count}", flush=True)

    return {
        "trajectory_sha256": trajectory_hash,
        "completed_frame_count": frame_count,
        "cache": str(output_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    prefixes = validate_args(args)
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
    if len(set(selected)) != len(selected) or any(index < 1 or index > len(trajectories) for index in selected):
        raise ValueError("clone indices must be unique completed clone ordinals")
    for index in selected:
        cache_clone(
            trajectories[index - 1],
            args.drift_cache_directory / f"clone_{index:03d}_decomposed_drift.npz",
            args.second_generator_cache_directory / f"clone_{index:03d}_relative_second_generator.npz",
            args.output_cache_directory / f"clone_{index:03d}_l2p_conditional_diffusion.npz",
            args=args,
            prefixes=prefixes,
        )


if __name__ == "__main__":
    main()
