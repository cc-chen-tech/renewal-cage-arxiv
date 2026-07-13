#!/usr/bin/env python3
"""Run low-disk C3 common-noise responses for the smooth cage coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_generator_response import generator_response_lammps_input  # noqa: E402
from ka_smooth_cage import extract_smooth_cage_path, matched_smooth_cage_tangent  # noqa: E402


VELOCITY_SEEDS = (82101, 82139, 82157, 82181, 82203, 82239, 82277, 82301)
LANGEVIN_SEEDS = (83101, 83139, 83157, 83181, 83203, 83239, 83277, 83301)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_epsilon(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def validate_c3_parent(parent_restart: Path) -> tuple[Path, str]:
    manifest_path = parent_restart.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("C3 parent requires a sibling preparation manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("potential_protocol") != "ka_lj_c3_switch":
        raise ValueError("parent manifest must declare ka_lj_c3_switch")
    if manifest.get("output_restart_sha256") != file_sha256(parent_restart):
        raise ValueError("parent restart does not match its preparation manifest")
    return manifest_path, file_sha256(manifest_path)


def verify_tangent_artifact(path: Path, *, expected_frames: int) -> None:
    expected_shapes = {
        "time": (expected_frames,),
        "relative_position_response": (expected_frames, 3),
        "relative_velocity_response": (expected_frames, 3),
        "projected_drift_response": (expected_frames, 3),
        "tangent_noise_covariance_rate": (expected_frames, 3, 3),
        "positive_condition_number": (expected_frames,),
        "negative_condition_number": (expected_frames,),
    }
    with np.load(path, allow_pickle=False) as payload:
        for key, shape in expected_shapes.items():
            if key not in payload or payload[key].shape != shape or np.any(~np.isfinite(payload[key])):
                raise ValueError(f"{path}: invalid tangent array {key}")
        if str(payload["potential_protocol"]) != "ka_lj_c3_switch":
            raise ValueError(f"{path}: potential protocol mismatch")
        if float(payload["thermodynamic_claim_allowed"]) != 0.0:
            raise ValueError(f"{path}: thermodynamic claim boundary was not preserved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-restart", type=Path, required=True)
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.001, 0.002])
    parser.add_argument("--run-steps", type=int, default=200)
    parser.add_argument("--dump-interval", type=int, default=1)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--integration-time-step", type=float, default=0.001)
    parser.add_argument("--target-id", type=int, default=821)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--retain-audit-raw", action="store_true")
    args = parser.parse_args()

    parent_restart = args.parent_restart.resolve()
    lammps = args.lammps.resolve()
    output_directory = args.output_directory.resolve()
    if not parent_restart.is_file() or not lammps.is_file():
        raise ValueError("parent restart and LAMMPS binary must exist")
    if args.members < 1 or args.members > len(VELOCITY_SEEDS):
        raise ValueError(f"members must lie between 1 and {len(VELOCITY_SEEDS)}")
    if isinstance(args.run_steps, bool) or args.run_steps < 1:
        raise ValueError("run_steps must be a positive integer")
    if isinstance(args.dump_interval, bool) or args.dump_interval < 1:
        raise ValueError("dump_interval must be a positive integer")
    if args.run_steps % args.dump_interval:
        raise ValueError("run_steps must be divisible by dump_interval")
    if not args.epsilons or len(set(args.epsilons)) != len(args.epsilons) or any(
        not math.isfinite(value) or value <= 0.0 for value in args.epsilons
    ):
        raise ValueError("epsilons must be unique finite positive values")
    for name, value in (
        ("directional_step", args.directional_step),
        ("integration_time_step", args.integration_time_step),
        ("temperature", args.temperature),
        ("friction", args.friction),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if output_directory.exists():
        raise ValueError("output directory must not already exist")

    parent_manifest, parent_manifest_hash = validate_c3_parent(parent_restart)
    parent_hash = file_sha256(parent_restart)
    binary_hash = file_sha256(lammps)
    output_directory.mkdir(parents=True)
    expected_frames = args.run_steps // args.dump_interval + 1
    records: list[dict[str, object]] = []
    retained_raw = False
    for member_index in range(1, args.members + 1):
        velocity_seed = VELOCITY_SEEDS[member_index - 1]
        langevin_seed = LANGEVIN_SEEDS[member_index - 1]
        for epsilon in args.epsilons:
            pair_directory = (
                output_directory
                / f"member_{member_index:03d}"
                / f"epsilon_{encode_epsilon(epsilon)}"
            )
            paths: dict[int, Path] = {}
            for sign_name, sign in (("plus", 1), ("minus", -1)):
                run_directory = pair_directory / sign_name
                run_directory.mkdir(parents=True)
                trajectory_path = run_directory / "trajectory.lammpstrj"
                input_path = run_directory / "in.response"
                input_path.write_text(
                    generator_response_lammps_input(
                        parent_restart=parent_restart,
                        target_id=args.target_id,
                        displacement=sign * epsilon,
                        temperature=args.temperature,
                        friction=args.friction,
                        velocity_seed=velocity_seed,
                        langevin_seed=langevin_seed,
                        run_steps=args.run_steps,
                        dump_interval_steps=args.dump_interval,
                        trajectory_name=trajectory_path.name,
                        potential_protocol="ka_lj_c3_switch",
                    )
                )
                subprocess.run(
                    [
                        str(lammps),
                        "-in",
                        input_path.name,
                        "-log",
                        "log.lammps",
                        "-screen",
                        "none",
                    ],
                    cwd=run_directory,
                    check=True,
                )
                paths[sign] = trajectory_path

            positive = extract_smooth_cage_path(
                paths[1],
                target_id=args.target_id,
                friction=args.friction,
                temperature=args.temperature,
                integration_time_step=args.integration_time_step,
                directional_step=args.directional_step,
                potential_protocol="ka_lj_c3_switch",
            )
            negative = extract_smooth_cage_path(
                paths[-1],
                target_id=args.target_id,
                friction=args.friction,
                temperature=args.temperature,
                integration_time_step=args.integration_time_step,
                directional_step=args.directional_step,
                potential_protocol="ka_lj_c3_switch",
            )
            tangent = matched_smooth_cage_tangent(positive, negative, epsilon=epsilon)
            output_path = pair_directory / "smooth_cage_tangent.npz"
            temporary = pair_directory / "smooth_cage_tangent.tmp.npz"
            np.savez_compressed(
                temporary,
                time=tangent["time"],
                relative_position_response=tangent["relative_position_response"],
                relative_velocity_response=tangent["relative_velocity_response"],
                projected_drift_response=tangent["projected_drift_response"],
                tangent_noise_covariance_rate=tangent["tangent_noise_covariance_rate"],
                positive_condition_number=positive["jacobian_gram_condition_number"],
                negative_condition_number=negative["jacobian_gram_condition_number"],
                frame_time=tangent["frame_time"],
                friction=args.friction,
                temperature=args.temperature,
                directional_step=args.directional_step,
                target_id=args.target_id,
                member_index=member_index,
                epsilon=epsilon,
                velocity_seed=velocity_seed,
                langevin_seed=langevin_seed,
                parent_restart_sha256=parent_hash,
                lammps_binary_sha256=binary_hash,
                potential_protocol=np.asarray("ka_lj_c3_switch"),
                fit_parameters_from_macro_observables=False,
                thermodynamic_claim_allowed=0.0,
            )
            verify_tangent_artifact(temporary, expected_frames=expected_frames)
            temporary.replace(output_path)
            keep_pair = bool(args.retain_audit_raw and not retained_raw)
            if keep_pair:
                retained_raw = True
            else:
                for trajectory_path in paths.values():
                    trajectory_path.unlink()
            records.append(
                {
                    "member_index": member_index,
                    "epsilon": epsilon,
                    "velocity_seed": velocity_seed,
                    "langevin_seed": langevin_seed,
                    "path": str(output_path.relative_to(output_directory)),
                    "path_sha256": file_sha256(output_path),
                    "raw_pair_retained": keep_pair,
                    "frame_count": expected_frames,
                    "frame_time_tau": args.dump_interval * args.integration_time_step,
                    "fit_parameters_from_macro_observables": False,
                    "thermodynamic_claim_allowed": False,
                }
            )
            print(f"member={member_index} epsilon={epsilon:g} reduced")

    manifest = {
        "protocol": "smooth_force_support_cage_common_noise_tangent",
        "parent_restart_path": str(parent_restart),
        "parent_restart_sha256": parent_hash,
        "parent_manifest_path": str(parent_manifest),
        "parent_manifest_sha256": parent_manifest_hash,
        "lammps_binary": str(lammps),
        "lammps_binary_sha256": binary_hash,
        "potential_protocol": "ka_lj_c3_switch",
        "cage_weight": "wendland_c4",
        "cage_support_sigma": 2.5,
        "target_id": args.target_id,
        "temperature": args.temperature,
        "friction": args.friction,
        "integration_time_step_tau": args.integration_time_step,
        "duration_tau": args.run_steps * args.integration_time_step,
        "saved_frame_interval_tau": args.dump_interval * args.integration_time_step,
        "directional_step": args.directional_step,
        "member_count": args.members,
        "epsilons": args.epsilons,
        "record_count": len(records),
        "retained_raw_pair_count": sum(bool(row["raw_pair_retained"]) for row in records),
        "fit_parameters_from_macro_observables": False,
        "thermodynamic_claim_allowed": False,
        "records": records,
    }
    temporary_manifest = output_directory / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(output_directory / "manifest.json")


if __name__ == "__main__":
    main()
