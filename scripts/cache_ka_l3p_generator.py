#!/usr/bin/env python3
"""Cache the microscopic ``L^3p`` generator quotient on frozen KA clones."""

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

from ka_l2p_conditional_diffusion import rademacher_velocity_probes  # noqa: E402
from ka_l3p_generator import (  # noqa: E402
    classify_l3p_numerical_canary,
    smooth_cage_l3p_generator_batch,
)
from ka_local_cage import ka_lj_sparse_force_generator_observables  # noqa: E402
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_smooth_cage import (  # noqa: E402
    smooth_cage_l2p_velocity_directional_derivative_batch,
)


PRIMARY_STEP = 1e-5
REFERENCE_STEP = 3e-6
COARSE_STEP = 3e-5
L2P_DIRECTIONAL_STEP = 1e-5
L2P_PHASE_SPACE_STEP = 3e-6
TRACE_SEED = 20260731
ESTIMATOR = "microscopic_l3p_generator_quotient"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_second_generator_cache(path: Path) -> dict[str, np.ndarray]:
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


def load_drift_cache(path: Path) -> dict[str, np.ndarray]:
    """Load only the frozen fields needed to align the L3p target set."""

    required = (
        "relative_position",
        "relative_velocity",
        "center_velocity",
        "relative_drift",
        "center_drift",
        "target_indices",
        "trajectory_sha256",
        "thermodynamic_claim_allowed",
    )
    with np.load(path, allow_pickle=False) as cache:
        if any(key not in cache for key in required):
            raise ValueError(f"incomplete decomposed-drift cache: {path}")
        result = {key: np.asarray(cache[key]) for key in required}
    shape = result["relative_velocity"].shape
    if (
        len(shape) != 3
        or shape[-1] != 3
        or any(result[key].shape != shape for key in required[:5])
        or float(result["thermodynamic_claim_allowed"]) != 0.0
    ):
        raise ValueError(f"invalid decomposed-drift cache: {path}")
    return result


def validate_cache_alignment(
    *,
    trajectory_sha256: str,
    drift: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
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
        raise ValueError("target indices do not match the L3p protocol")
    if str(second["potential_protocol"]) not in {"ka_lj_cut", "ka_lj_c3_switch"}:
        raise ValueError("potential protocol is unsupported")
    if float(second["thermodynamic_claim_allowed"]) != 0.0:
        raise ValueError("claim boundary is open in the second-generator cache")


def _provenance_equal(actual: object, expected: object) -> bool:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    return actual_array.shape == expected_array.shape and np.array_equal(
        actual_array,
        expected_array,
    )


def validate_existing_checkpoint(
    saved: dict[str, object],
    expected_provenance: dict[str, object],
    expected_shapes: dict[str, tuple[int, ...]],
) -> int:
    """Validate a resumable cache before any saved values are reused."""

    for key, expected in expected_provenance.items():
        if key not in saved or not _provenance_equal(saved[key], expected):
            raise ValueError(
                f"existing L3p cache has mismatched provenance: {key}"
            )
    for key, shape in expected_shapes.items():
        if key not in saved or np.asarray(saved[key]).shape != shape:
            raise ValueError(
                f"existing L3p cache has mismatched array shape: {key}"
            )
    if "completed_frame_count" not in saved:
        raise ValueError("existing L3p cache has no completion count")
    completed_raw = float(np.asarray(saved["completed_frame_count"]))
    completed = int(completed_raw)
    frame_count = int(expected_provenance["requested_frame_count"])
    if completed_raw != completed or completed < 0 or completed > frame_count:
        raise ValueError("existing L3p cache has invalid completion count")
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clone-directory", type=Path, required=True)
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--second-generator-cache-directory", type=Path, required=True)
    parser.add_argument("--output-cache-directory", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--clone-indices", type=int, nargs="+")
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--maximum-frame-count", type=int, default=200)
    parser.add_argument(
        "--probe-prefix-counts",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32],
        help="nested trace prefixes: 4 8 16 32",
    )
    parser.add_argument(
        "--trace-seed",
        type=int,
        default=20260731,
        help="frozen nested trace-probe seed 20260731",
    )
    parser.add_argument("--primary-position-step", type=float, default=PRIMARY_STEP)
    parser.add_argument("--reference-position-step", type=float, default=REFERENCE_STEP)
    parser.add_argument("--coarse-position-step", type=float, default=COARSE_STEP)
    parser.add_argument(
        "--primary-cage-hessian-step",
        type=float,
        default=PRIMARY_STEP,
    )
    parser.add_argument(
        "--reference-cage-hessian-step",
        type=float,
        default=REFERENCE_STEP,
    )
    parser.add_argument(
        "--coarse-cage-hessian-step",
        type=float,
        default=COARSE_STEP,
    )
    parser.add_argument("--jacobian-step", type=float, default=PRIMARY_STEP)
    parser.add_argument(
        "--l2p-directional-step",
        type=float,
        default=L2P_DIRECTIONAL_STEP,
    )
    parser.add_argument(
        "--l2p-phase-space-step",
        type=float,
        default=L2P_PHASE_SPACE_STEP,
    )
    parser.add_argument("--directional-velocity-step", type=float, default=2e-5)
    parser.add_argument("--sensitivity-frame-count", type=int, default=1)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    prefixes = tuple(args.probe_prefix_counts)
    frozen = (
        (args.primary_position_step, PRIMARY_STEP),
        (args.reference_position_step, REFERENCE_STEP),
        (args.coarse_position_step, COARSE_STEP),
        (args.primary_cage_hessian_step, PRIMARY_STEP),
        (args.reference_cage_hessian_step, REFERENCE_STEP),
        (args.coarse_cage_hessian_step, COARSE_STEP),
        (args.jacobian_step, PRIMARY_STEP),
        (args.l2p_directional_step, L2P_DIRECTIONAL_STEP),
        (args.l2p_phase_space_step, L2P_PHASE_SPACE_STEP),
        (args.directional_velocity_step, 2e-5),
    )
    if (
        prefixes != (4, 8, 16, 32)
        or args.trace_seed != TRACE_SEED
        or any(
            not math.isclose(value, expected, rel_tol=0.0, abs_tol=0.0)
            for value, expected in frozen
        )
        or args.expected_clone_count < 1
        or args.target_count < 1
        or args.maximum_frame_count < 1
        or args.sensitivity_frame_count < 1
        or args.sensitivity_frame_count > args.maximum_frame_count
        or not math.isfinite(args.friction)
        or args.friction <= 0.0
        or not math.isfinite(args.temperature)
        or args.temperature <= 0.0
        or args.target_batch_size < 1
        or args.checkpoint_interval < 1
    ):
        raise ValueError("L3p cache controls must match the frozen protocol")
    return prefixes


def relative_vector_error(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 2 or candidate.shape[1] != 3:
        raise ValueError("candidate and reference must align as target vectors")
    return np.linalg.norm(candidate - reference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1),
        1e-30,
    )


def cache_clone(
    trajectory_path: Path,
    drift_path: Path,
    second_path: Path,
    output_path: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    prefixes = validate_args(args)
    trajectory_hash = file_sha256(trajectory_path)
    drift_hash = file_sha256(drift_path)
    second_hash = file_sha256(second_path)
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
    velocities_raw = trajectory.get("velocities")
    if velocities_raw is None:
        raise ValueError("L3p cache requires microscopic velocities")
    velocities = np.asarray(velocities_raw, dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    targets = np.asarray(drift["target_indices"], dtype=int)
    second_values = np.asarray(second["second_relative_generator"], dtype=float)
    frame_count = min(args.maximum_frame_count, len(positions), len(second_values))
    if (
        positions.shape != velocities.shape
        or positions.ndim != 3
        or positions.shape[2] != 3
        or frame_count < 1
        or box_lengths.shape != (3,)
        or np.any(~np.isfinite(box_lengths))
    ):
        raise ValueError("trajectory and second-generator cache are not aligned")
    positions = positions[:frame_count]
    velocities = velocities[:frame_count]
    protocol = str(second["potential_protocol"])
    sensitivity_count = min(args.sensitivity_frame_count, frame_count)
    target_count = len(targets)
    prefix_count = len(prefixes)
    probes = rademacher_velocity_probes(
        probe_count=prefixes[-1],
        particle_count=positions.shape[1],
        seed=args.trace_seed,
    )

    arrays: dict[str, np.ndarray] = {
        "l3p": np.full((frame_count, target_count, 3), np.nan),
        "l3p_prefixes": np.full(
            (frame_count, prefix_count, target_count, 3),
            np.nan,
        ),
        "position_transport_term": np.full((frame_count, target_count, 3), np.nan),
        "acceleration_response_term": np.full((frame_count, target_count, 3), np.nan),
        "thermal_gradient_prefixes": np.full(
            (frame_count, prefix_count, target_count, 3),
            np.nan,
        ),
        "thermal_friction_prefixes": np.full(
            (frame_count, prefix_count, target_count, 3),
            np.nan,
        ),
        "laplacian_prefixes": np.full(
            (frame_count, prefix_count, target_count, 3),
            np.nan,
        ),
        "laplacian_velocity_derivative_prefixes": np.full(
            (frame_count, prefix_count, target_count, 3),
            np.nan,
        ),
        "prefix_16_32_error": np.full((sensitivity_count, target_count), np.nan),
        "position_primary_reference_error": np.full(
            (sensitivity_count, target_count),
            np.nan,
        ),
        "position_coarse_reference_error": np.full(
            (sensitivity_count, target_count),
            np.nan,
        ),
        "cage_primary_reference_error": np.full(
            (sensitivity_count, target_count),
            np.nan,
        ),
        "cage_coarse_reference_error": np.full(
            (sensitivity_count, target_count),
            np.nan,
        ),
        "acceleration_directional_error": np.full(
            (sensitivity_count, target_count),
            np.nan,
        ),
    }
    expected_shapes = {key: value.shape for key, value in arrays.items()}
    provenance: dict[str, object] = {
        "trajectory_sha256": trajectory_hash,
        "drift_cache_sha256": drift_hash,
        "second_generator_cache_sha256": second_hash,
        "target_indices": targets,
        "potential_protocol": protocol,
        "estimator": ESTIMATOR,
        "probe_prefix_counts": np.asarray(prefixes, dtype=int),
        "trace_seed": int(args.trace_seed),
        "primary_position_step": float(args.primary_position_step),
        "reference_position_step": float(args.reference_position_step),
        "coarse_position_step": float(args.coarse_position_step),
        "primary_cage_hessian_step": float(args.primary_cage_hessian_step),
        "reference_cage_hessian_step": float(args.reference_cage_hessian_step),
        "coarse_cage_hessian_step": float(args.coarse_cage_hessian_step),
        "jacobian_step": float(args.jacobian_step),
        "l2p_directional_step": float(args.l2p_directional_step),
        "l2p_phase_space_step": float(args.l2p_phase_space_step),
        "directional_velocity_step": float(args.directional_velocity_step),
        "friction": float(args.friction),
        "temperature": float(args.temperature),
        "requested_frame_count": int(frame_count),
        "sensitivity_frame_count": int(sensitivity_count),
        "target_batch_size": int(args.target_batch_size),
    }
    completed = 0
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as checkpoint:
            saved = {key: np.asarray(checkpoint[key]) for key in checkpoint.files}
        completed = validate_existing_checkpoint(saved, provenance, expected_shapes)
        for key in arrays:
            arrays[key] = np.asarray(saved[key], dtype=float).copy()

    def numerical_verdict() -> dict[str, float | str]:
        return classify_l3p_numerical_canary(
            prefix_16_32_error=arrays["prefix_16_32_error"].ravel(),
            position_primary_reference_error=arrays[
                "position_primary_reference_error"
            ].ravel(),
            position_coarse_reference_error=arrays[
                "position_coarse_reference_error"
            ].ravel(),
            cage_primary_reference_error=arrays[
                "cage_primary_reference_error"
            ].ravel(),
            cage_coarse_reference_error=arrays[
                "cage_coarse_reference_error"
            ].ravel(),
            acceleration_directional_error=arrays[
                "acceleration_directional_error"
            ].ravel(),
        )

    def save(done: int) -> None:
        atomic_savez(
            output_path,
            **arrays,
            **provenance,
            completed_frame_count=float(done),
            checkpoint_interval=float(args.checkpoint_interval),
            **numerical_verdict(),
        )

    all_particles = np.arange(len(particle_types), dtype=int)
    common = {
        "particle_types": particle_types,
        "target_indices": targets,
        "friction": args.friction,
        "temperature": args.temperature,
        "trace_probes": probes,
        "prefix_counts": prefixes,
        "jacobian_step": args.jacobian_step,
        "l2p_directional_step": args.l2p_directional_step,
        "l2p_phase_space_step": args.l2p_phase_space_step,
        "potential_protocol": protocol,
        "target_batch_size": args.target_batch_size,
    }
    component_keys = (
        "l3p",
        "l3p_prefixes",
        "position_transport_term",
        "acceleration_response_term",
        "thermal_gradient_prefixes",
        "thermal_friction_prefixes",
        "laplacian_prefixes",
        "laplacian_velocity_derivative_prefixes",
    )
    for frame in range(completed, frame_count):
        frame_common = {
            **common,
            "positions": positions[frame],
            "velocities": velocities[frame],
            "box_lengths": box_lengths,
        }
        primary = smooth_cage_l3p_generator_batch(
            **frame_common,
            position_step=args.primary_position_step,
            cage_hessian_step=args.primary_cage_hessian_step,
        )
        for key in component_keys:
            arrays[key][frame] = np.asarray(primary[key], dtype=float)

        if frame < sensitivity_count:
            position_reference = smooth_cage_l3p_generator_batch(
                **frame_common,
                position_step=args.reference_position_step,
                cage_hessian_step=args.primary_cage_hessian_step,
            )
            position_coarse = smooth_cage_l3p_generator_batch(
                **frame_common,
                position_step=args.coarse_position_step,
                cage_hessian_step=args.primary_cage_hessian_step,
            )
            cage_reference = smooth_cage_l3p_generator_batch(
                **frame_common,
                position_step=args.primary_position_step,
                cage_hessian_step=args.reference_cage_hessian_step,
            )
            cage_coarse = smooth_cage_l3p_generator_batch(
                **frame_common,
                position_step=args.primary_position_step,
                cage_hessian_step=args.coarse_cage_hessian_step,
            )
            arrays["prefix_16_32_error"][frame] = relative_vector_error(
                np.asarray(primary["l3p_prefixes"])[-2],
                np.asarray(primary["l3p_prefixes"])[-1],
            )
            arrays["position_primary_reference_error"][frame] = relative_vector_error(
                np.asarray(primary["l3p"]),
                np.asarray(position_reference["l3p"]),
            )
            arrays["position_coarse_reference_error"][frame] = relative_vector_error(
                np.asarray(position_coarse["l3p"]),
                np.asarray(position_reference["l3p"]),
            )
            arrays["cage_primary_reference_error"][frame] = relative_vector_error(
                np.asarray(primary["l3p"]),
                np.asarray(cage_reference["l3p"]),
            )
            arrays["cage_coarse_reference_error"][frame] = relative_vector_error(
                np.asarray(cage_coarse["l3p"]),
                np.asarray(cage_reference["l3p"]),
            )

            acceleration = np.asarray(primary["acceleration"], dtype=float)
            acceleration_force = ka_lj_sparse_force_generator_observables(
                positions[frame],
                velocities=acceleration,
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=all_particles,
                potential_protocol=protocol,
            )
            directional = smooth_cage_l2p_velocity_directional_derivative_batch(
                positions[frame],
                velocities=velocities[frame],
                velocity_direction=acceleration,
                forces=np.asarray(primary["force"]),
                force_generator=np.asarray(primary["force_generator"]),
                force_generator_direction=np.asarray(
                    acceleration_force["force_generator"]
                ),
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=targets,
                friction=args.friction,
                directional_step=args.l2p_directional_step,
                phase_space_step=args.l2p_phase_space_step,
                velocity_step=args.directional_velocity_step,
                target_batch_size=args.target_batch_size,
            )
            arrays["acceleration_directional_error"][frame] = relative_vector_error(
                np.asarray(primary["acceleration_response_term"]),
                np.asarray(directional["l2p_velocity_directional_derivative"]),
            )

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
            args.output_cache_directory / f"clone_{index:03d}_l3p_generator.npz",
            args=args,
        )


if __name__ == "__main__":
    main()
