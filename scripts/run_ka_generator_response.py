#!/usr/bin/env python3
"""Run sequential low-disk matched KA force-generator responses."""

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

from ka_generator_response import extract_generator_response_path, generator_response_lammps_input  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_epsilon(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def verify_extraction(path: Path, *, expected_frames: int) -> None:
    expected_shapes = {
        "time": (expected_frames,),
        "position": (expected_frames, 3),
        "velocity": (expected_frames, 3),
        "force": (expected_frames, 3),
        "force_generator": (expected_frames, 3),
        "second_force_generator": (expected_frames, 3),
        "force_generator_noise_covariance_rate": (expected_frames, 3, 3),
    }
    with np.load(path, allow_pickle=False) as payload:
        for key, shape in expected_shapes.items():
            if key not in payload or payload[key].shape != shape or np.any(~np.isfinite(payload[key])):
                raise ValueError(f"{path}: invalid extracted array {key}")
        if float(payload["thermodynamic_claim_allowed"]) != 0.0:
            raise ValueError(f"{path}: thermodynamic claim boundary was not preserved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lammps-binary", type=Path, required=True)
    parser.add_argument("--parent-restart", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--velocity-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--langevin-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.001, 0.002])
    parser.add_argument("--target-id", type=int, default=821)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--integration-time-step", type=float, default=0.001)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--dump-interval", type=float, default=0.005)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--retain-audit-raw", action="store_true")
    args = parser.parse_args()

    lammps_binary = args.lammps_binary.resolve()
    parent_restart = args.parent_restart.resolve()
    output_directory = args.output_directory.resolve()
    if not lammps_binary.is_file() or not parent_restart.is_file():
        raise ValueError("LAMMPS binary and parent restart must exist")
    if len(args.velocity_seeds) != len(args.langevin_seeds) or not args.velocity_seeds:
        raise ValueError("velocity and Langevin seed vectors must have equal nonzero length")
    if len(set(args.velocity_seeds)) != len(args.velocity_seeds) or len(set(args.langevin_seeds)) != len(args.langevin_seeds):
        raise ValueError("seeds must be unique within each vector")
    if any(seed < 1 for seed in args.velocity_seeds + args.langevin_seeds):
        raise ValueError("seeds must be positive")
    if not args.epsilons or any(not math.isfinite(value) or value <= 0.0 for value in args.epsilons):
        raise ValueError("epsilons must be finite and positive")
    if len(set(args.epsilons)) != len(args.epsilons):
        raise ValueError("epsilons must be unique")
    for name, value in (
        ("temperature", args.temperature),
        ("friction", args.friction),
        ("integration_time_step", args.integration_time_step),
        ("duration", args.duration),
        ("dump_interval", args.dump_interval),
        ("directional_step", args.directional_step),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    run_steps = int(round(args.duration / args.integration_time_step))
    dump_interval_steps = int(round(args.dump_interval / args.integration_time_step))
    if not math.isclose(run_steps * args.integration_time_step, args.duration):
        raise ValueError("duration must be an integer multiple of the integration time step")
    if not math.isclose(dump_interval_steps * args.integration_time_step, args.dump_interval):
        raise ValueError("dump interval must be an integer multiple of the integration time step")
    if run_steps % dump_interval_steps:
        raise ValueError("duration must contain an integer number of dump intervals")
    if output_directory.exists():
        raise ValueError("output directory must not already exist")
    output_directory.mkdir(parents=True)

    parent_hash = file_sha256(parent_restart)
    expected_frames = run_steps // dump_interval_steps + 1
    records: list[dict[str, object]] = []
    audit_selected = False
    for member_index, (velocity_seed, langevin_seed) in enumerate(
        zip(args.velocity_seeds, args.langevin_seeds), start=1
    ):
        for epsilon in args.epsilons:
            for sign_name, sign in (("plus", 1), ("minus", -1)):
                run_directory = (
                    output_directory
                    / f"member_{member_index:03d}"
                    / f"epsilon_{encode_epsilon(epsilon)}"
                    / sign_name
                )
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
                        run_steps=run_steps,
                        dump_interval_steps=dump_interval_steps,
                        trajectory_name=trajectory_path.name,
                    )
                )
                subprocess.run(
                    [
                        str(lammps_binary),
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
                extracted = extract_generator_response_path(
                    trajectory_path,
                    target_id=args.target_id,
                    temperature=args.temperature,
                    friction=args.friction,
                    integration_time_step=args.integration_time_step,
                    directional_step=args.directional_step,
                )
                extraction_path = run_directory / "generator_path.npz"
                np.savez_compressed(
                    extraction_path,
                    **extracted,
                    member_index=member_index,
                    epsilon=epsilon,
                    sign=sign,
                    velocity_seed=velocity_seed,
                    langevin_seed=langevin_seed,
                    parent_restart_sha256=parent_hash,
                    fit_parameters_from_macro_observables=False,
                )
                verify_extraction(extraction_path, expected_frames=expected_frames)
                retain_raw = bool(args.retain_audit_raw and not audit_selected)
                if retain_raw:
                    audit_selected = True
                else:
                    trajectory_path.unlink()
                records.append(
                    {
                        "member_index": member_index,
                        "epsilon": epsilon,
                        "sign": sign,
                        "velocity_seed": velocity_seed,
                        "langevin_seed": langevin_seed,
                        "path": str(extraction_path.relative_to(output_directory)),
                        "path_sha256": file_sha256(extraction_path),
                        "raw_path": str(trajectory_path.relative_to(output_directory)) if retain_raw else None,
                        "raw_retained": retain_raw,
                        "frame_count": expected_frames,
                        "frame_time_tau": args.dump_interval,
                        "fit_parameters_from_macro_observables": False,
                        "thermodynamic_claim_allowed": False,
                    }
                )

    manifest = {
        "protocol": "full_KA_common_noise_generator_response",
        "parent_restart_path": str(parent_restart),
        "parent_restart_sha256": parent_hash,
        "lammps_binary": str(lammps_binary),
        "target_id": args.target_id,
        "temperature": args.temperature,
        "friction": args.friction,
        "integration_time_step_tau": args.integration_time_step,
        "duration_tau": args.duration,
        "saved_frame_interval_tau": args.dump_interval,
        "directional_step": args.directional_step,
        "member_count": len(args.velocity_seeds),
        "epsilons": args.epsilons,
        "record_count": len(records),
        "retained_raw_count": sum(bool(row["raw_retained"]) for row in records),
        "axis_semantics": "matched_common_noise_tangent_response",
        "fit_parameters_from_macro_observables": False,
        "thermodynamic_claim_allowed": False,
        "records": records,
    }
    temporary_manifest = output_directory / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(output_directory / "manifest.json")


if __name__ == "__main__":
    main()
