#!/usr/bin/env python3
"""Cache the exact smooth-cage ``L2p`` observable from full KA trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_projected_ito_innovations import load_drift_cache  # noqa: E402
from ka_local_cage import ka_lj_sparse_force_generator_observables  # noqa: E402
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_smooth_cage import smooth_cage_second_generator_batch  # noqa: E402


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


def relative_l2_error(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(value - reference) / max(denominator, 1e-30))


def load_second_generator_drift_cache(
    path: Path,
) -> dict[str, np.ndarray | float | str]:
    """Load the base drift plus fields needed for independent reconstruction."""

    result = load_drift_cache(path)
    with np.load(path, allow_pickle=False) as cache:
        if "projected_force" not in cache:
            raise ValueError(f"drift cache lacks projected_force: {path}")
        projected_force = np.asarray(cache["projected_force"], dtype=float)
    if projected_force.shape != np.asarray(result["relative_drift"]).shape or np.any(
        ~np.isfinite(projected_force)
    ):
        raise ValueError(f"invalid projected_force in drift cache: {path}")
    result["projected_force"] = projected_force
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-directory", type=Path, required=True)
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--output-cache-directory", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, required=True)
    parser.add_argument("--clone-indices", type=int, nargs="+")
    parser.add_argument("--maximum-frame-count", type=int, default=0)
    parser.add_argument(
        "--potential-protocol",
        choices=("ka_lj_cut", "ka_lj_c3_switch"),
        default="ka_lj_c3_switch",
    )
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--phase-space-step", type=float, default=1e-5)
    parser.add_argument("--trace-probe-count", type=int, default=32)
    parser.add_argument("--trace-probe-seed", type=int, default=20260718)
    parser.add_argument(
        "--probe-prefix-counts", type=int, nargs="+", default=[4, 8, 16, 32]
    )
    parser.add_argument(
        "--sensitivity-directional-steps",
        type=float,
        nargs="+",
        default=[5e-6, 2e-5],
    )
    parser.add_argument(
        "--sensitivity-phase-space-steps",
        type=float,
        nargs="+",
        default=[3e-6, 3e-5],
    )
    parser.add_argument("--sensitivity-frame-count", type=int, default=4)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    prefixes = tuple(sorted(set(args.probe_prefix_counts)))
    if (
        args.expected_clone_count < 1
        or args.maximum_frame_count < 0
        or not math.isfinite(args.friction)
        or args.friction <= 0.0
        or not math.isfinite(args.temperature)
        or args.temperature <= 0.0
        or args.directional_step <= 0.0
        or args.phase_space_step <= 0.0
        or args.trace_probe_count < 1
        or not prefixes
        or prefixes[0] < 1
        or prefixes[-1] != args.trace_probe_count
        or any(step <= 0.0 for step in args.sensitivity_directional_steps)
        or any(step <= 0.0 for step in args.sensitivity_phase_space_steps)
        or args.sensitivity_frame_count < 0
        or args.target_batch_size < 1
        or args.checkpoint_interval < 1
    ):
        raise ValueError("invalid relative-second-generator cache controls")
    return prefixes


def initialize_arrays(
    frame_count: int, target_count: int, prefix_count: int
) -> dict[str, np.ndarray]:
    vector_shape = (frame_count, target_count, 3)
    return {
        "relative_drift": np.full(vector_shape, np.nan),
        "second_relative_generator": np.full(vector_shape, np.nan),
        "position_generator_term": np.full(vector_shape, np.nan),
        "velocity_generator_term": np.full(vector_shape, np.nan),
        "ito_trace_term": np.full(vector_shape, np.nan),
        "probe_prefix_second_generator": np.full(
            (frame_count, prefix_count, target_count, 3), np.nan
        ),
        "dump_force_protocol_relative_l2_difference": np.full(frame_count, np.nan),
        "dump_force_protocol_maximum_absolute_difference": np.full(frame_count, np.nan),
        "projected_force_cache_maximum_absolute_error": np.full(frame_count, np.nan),
        "relative_drift_cache_maximum_absolute_error": np.full(frame_count, np.nan),
        "sparse_candidate_mean": np.full(frame_count, np.nan),
        "sparse_candidate_maximum": np.full(frame_count, np.nan),
    }


def cache_clone(
    trajectory_path: Path,
    drift_path: Path,
    output_path: Path,
    *,
    args: argparse.Namespace,
    prefixes: tuple[int, ...],
) -> dict[str, object]:
    source_hash = file_sha256(trajectory_path)
    trajectory = load_lammps_custom_trajectory(trajectory_path)
    drift = load_second_generator_drift_cache(drift_path)
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    velocities = np.asarray(trajectory["velocities"], dtype=float)
    forces = np.asarray(trajectory["forces"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    targets = np.asarray(drift["target_indices"], dtype=int)
    if str(drift["trajectory_sha256"]) != source_hash:
        raise ValueError("trajectory SHA256 does not match the decomposed-drift cache")
    available_frames = min(len(positions), len(np.asarray(drift["relative_drift"])))
    frame_count = (
        available_frames
        if args.maximum_frame_count == 0
        else min(available_frames, args.maximum_frame_count)
    )
    if (
        positions.shape != velocities.shape
        or positions.shape != forces.shape
        or positions.shape[1:] != (len(particle_types), 3)
        or frame_count < 2
    ):
        raise ValueError("trajectory and drift cache arrays are not aligned")

    rng = np.random.default_rng(args.trace_probe_seed)
    probes = rng.choice(
        (-1.0, 1.0),
        size=(args.trace_probe_count, positions.shape[1], 3),
    )
    arrays = initialize_arrays(frame_count, len(targets), len(prefixes))
    completed = 0
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as saved:
            if (
                str(saved["trajectory_sha256"]) != source_hash
                or not np.array_equal(saved["target_indices"], targets)
                or int(saved["trace_probe_seed"]) != args.trace_probe_seed
                or int(saved["trace_probe_count"]) != args.trace_probe_count
                or str(saved["potential_protocol"]) != args.potential_protocol
                or not math.isclose(float(saved["directional_step"]), args.directional_step)
                or not math.isclose(float(saved["phase_space_step"]), args.phase_space_step)
                or int(saved["requested_frame_count"]) != frame_count
            ):
                raise ValueError("existing cache does not match requested protocol")
            completed = int(saved["completed_frame_count"])
            for key in arrays:
                arrays[key] = np.asarray(saved[key]).copy()
    if completed >= frame_count:
        return {
            "trajectory": str(trajectory_path.resolve()),
            "cache": str(output_path.resolve()),
            "trajectory_sha256": source_hash,
            "completed_frame_count": completed,
            "skipped_complete_cache": True,
        }

    all_particles = np.arange(positions.shape[1], dtype=int)
    sensitivity_rows: list[dict[str, float]] = []
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as saved:
            if "sensitivity_json" in saved:
                sensitivity_rows = json.loads(str(saved["sensitivity_json"]))

    def save_checkpoint(done: int) -> None:
        atomic_savez(
            output_path,
            **arrays,
            target_indices=targets,
            probe_prefix_counts=np.asarray(prefixes, dtype=int),
            completed_frame_count=float(done),
            requested_frame_count=float(frame_count),
            trajectory_sha256=np.asarray(source_hash),
            trace_probe_seed=float(args.trace_probe_seed),
            trace_probe_count=float(args.trace_probe_count),
            directional_step=float(args.directional_step),
            phase_space_step=float(args.phase_space_step),
            friction=float(args.friction),
            temperature=float(args.temperature),
            potential_protocol=np.asarray(args.potential_protocol),
            sensitivity_json=np.asarray(json.dumps(sensitivity_rows, sort_keys=True)),
            thermodynamic_claim_allowed=0.0,
        )

    for frame in range(completed, frame_count):
        sparse = ka_lj_sparse_force_generator_observables(
            positions[frame],
            velocities=velocities[frame],
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=all_particles,
            potential_protocol=args.potential_protocol,
        )
        reconstructed_force = np.asarray(sparse["force"])
        force_difference = reconstructed_force - forces[frame]
        force_generator = np.asarray(sparse["force_generator"])
        result = smooth_cage_second_generator_batch(
            positions[frame],
            velocities=velocities[frame],
            forces=reconstructed_force,
            force_generator=force_generator,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=targets,
            friction=args.friction,
            temperature=args.temperature,
            directional_step=args.directional_step,
            phase_space_step=args.phase_space_step,
            trace_probes=probes,
            target_batch_size=args.target_batch_size,
        )
        for key in (
            "relative_drift",
            "second_relative_generator",
            "position_generator_term",
            "velocity_generator_term",
            "ito_trace_term",
        ):
            arrays[key][frame] = result[key]
        trace_terms = np.asarray(result["trace_probe_velocity_laplacian_terms"])
        deterministic = (
            np.asarray(result["position_generator_term"])
            + np.asarray(result["velocity_generator_term"])
        )
        for prefix_slot, prefix in enumerate(prefixes):
            arrays["probe_prefix_second_generator"][frame, prefix_slot] = (
                deterministic
                + args.friction
                * args.temperature
                * np.mean(trace_terms[:prefix], axis=0)
            )
        arrays["dump_force_protocol_relative_l2_difference"][frame] = relative_l2_error(
            reconstructed_force, forces[frame]
        )
        arrays["dump_force_protocol_maximum_absolute_difference"][frame] = float(
            np.max(np.abs(force_difference))
        )
        arrays["projected_force_cache_maximum_absolute_error"][frame] = float(
            np.max(
                np.abs(
                    np.asarray(result["projected_force"])
                    - np.asarray(drift["projected_force"])[frame]
                )
            )
        )
        arrays["relative_drift_cache_maximum_absolute_error"][frame] = float(
            np.max(
                np.abs(
                    np.asarray(result["relative_drift"])
                    - np.asarray(drift["relative_drift"])[frame]
                )
            )
        )
        arrays["sparse_candidate_mean"][frame] = float(
            np.mean(sparse["candidate_count"])
        )
        arrays["sparse_candidate_maximum"][frame] = float(
            np.max(sparse["candidate_count"])
        )

        if frame < args.sensitivity_frame_count:
            reference = np.asarray(result["second_relative_generator"])
            controls = [
                (step, args.phase_space_step, "directional_step")
                for step in args.sensitivity_directional_steps
            ] + [
                (args.directional_step, step, "phase_space_step")
                for step in args.sensitivity_phase_space_steps
            ]
            for directional_step, phase_space_step, varied in controls:
                alternate = smooth_cage_second_generator_batch(
                    positions[frame],
                    velocities=velocities[frame],
                    forces=reconstructed_force,
                    force_generator=force_generator,
                    particle_types=particle_types,
                    box_lengths=box_lengths,
                    target_indices=targets,
                    friction=args.friction,
                    temperature=args.temperature,
                    directional_step=directional_step,
                    phase_space_step=phase_space_step,
                    trace_probes=probes,
                    target_batch_size=args.target_batch_size,
                )["second_relative_generator"]
                sensitivity_rows.append(
                    {
                        "frame": float(frame),
                        "varied_control": varied,
                        "directional_step": float(directional_step),
                        "phase_space_step": float(phase_space_step),
                        "relative_l2_error": relative_l2_error(
                            np.asarray(alternate), reference
                        ),
                    }
                )

        done = frame + 1
        if done % args.checkpoint_interval == 0 or done == frame_count:
            save_checkpoint(done)
            print(
                f"{trajectory_path.parent.name}: cached {done}/{frame_count} frames",
                flush=True,
            )

    reference_drift = np.asarray(drift["relative_drift"][:frame_count])
    return {
        "trajectory": str(trajectory_path.resolve()),
        "cache": str(output_path.resolve()),
        "trajectory_sha256": source_hash,
        "completed_frame_count": frame_count,
        "maximum_dump_force_protocol_relative_l2_difference": float(
            np.max(arrays["dump_force_protocol_relative_l2_difference"])
        ),
        "maximum_dump_force_protocol_absolute_difference": float(
            np.max(arrays["dump_force_protocol_maximum_absolute_difference"])
        ),
        "maximum_projected_force_cache_absolute_error": float(
            np.max(arrays["projected_force_cache_maximum_absolute_error"])
        ),
        "maximum_relative_drift_cache_frame_error": float(
            np.max(arrays["relative_drift_cache_maximum_absolute_error"])
        ),
        "maximum_relative_drift_absolute_error": float(
            np.max(np.abs(arrays["relative_drift"] - reference_drift))
        ),
        "maximum_step_sensitivity_relative_l2_error": float(
            max((row["relative_l2_error"] for row in sensitivity_rows), default=0.0)
        ),
        "skipped_complete_cache": False,
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
        raise ValueError("completed trajectory count does not match expected clone count")
    selected = (
        set(args.clone_indices)
        if args.clone_indices is not None
        else set(range(1, len(trajectories) + 1))
    )
    if not selected or min(selected) < 1 or max(selected) > len(trajectories):
        raise ValueError("clone_indices must select available one-based clones")

    rows = []
    for clone_index, trajectory_path in enumerate(trajectories, start=1):
        if clone_index not in selected:
            continue
        drift_path = (
            args.drift_cache_directory
            / f"clone_{clone_index:03d}_decomposed_drift.npz"
        )
        if not drift_path.is_file():
            raise ValueError(f"missing drift cache: {drift_path}")
        output_path = (
            args.output_cache_directory
            / f"clone_{clone_index:03d}_relative_second_generator.npz"
        )
        rows.append(
            {
                "clone_index": clone_index,
                **cache_clone(
                    trajectory_path,
                    drift_path,
                    output_path,
                    args=args,
                    prefixes=prefixes,
                ),
            }
        )

    args.output_cache_directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "microscopic_smooth_cage_relative_second_generator",
        "source_clone_directory": str(args.clone_directory.resolve()),
        "expected_clone_count": args.expected_clone_count,
        "processed_clone_indices": sorted(selected),
        "directional_step": args.directional_step,
        "phase_space_step": args.phase_space_step,
        "trace_probe_count": args.trace_probe_count,
        "trace_probe_seed": args.trace_probe_seed,
        "potential_protocol": args.potential_protocol,
        "probe_prefix_counts": list(prefixes),
        "rows": rows,
        "thermodynamic_claim_allowed": False,
    }
    (args.output_cache_directory / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
