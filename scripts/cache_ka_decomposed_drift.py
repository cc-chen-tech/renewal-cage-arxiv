#!/usr/bin/env python3
"""Cache exact smooth-cage projected drifts without fitting a bath model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_decomposed_cage_drift_bath import load_or_extract_drift  # noqa: E402
from analyze_ka_hankel_slow_force_bath import load_or_reduce_clone  # noqa: E402
from analyze_ka_smooth_cage_hankel_bath import load_or_extract_cage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--reduced-cache-directory", type=Path, required=True)
    parser.add_argument("--cage-cache-directory", type=Path, required=True)
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--rerun-directory", type=Path, required=True)
    parser.add_argument("--lammps-binary", type=Path, required=True)
    parser.add_argument("--parent-restart", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, required=True)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--target-seed", type=int, default=20260714)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--sensitivity-steps", type=float, nargs="+", default=[5e-6, 2e-5])
    parser.add_argument("--retain-rerun-dumps", action="store_true")
    args = parser.parse_args()
    if (
        args.expected_clone_count < 1
        or args.target_count < 1
        or args.target_batch_size < 1
        or args.friction <= 0.0
        or args.temperature <= 0.0
        or args.directional_step <= 0.0
        or any(step <= 0.0 for step in args.sensitivity_steps)
        or not args.lammps_binary.is_file()
        or not args.parent_restart.is_file()
    ):
        raise ValueError("invalid microscopic drift-cache controls")

    manifest = json.loads((args.clone_directory / "manifest.json").read_text())
    if (
        int(manifest["clone_count"]) != args.expected_clone_count
        or manifest["dynamics"] != "nve_plus_langevin"
        or not bool(manifest["dump_velocity_force"])
        or bool(manifest["thermodynamic_claim_allowed"])
        or not math.isclose(float(manifest["saved_frame_interval_tau"]), 0.01)
        or float(manifest["duration_tau"]) < 10.0
        or not math.isclose(float(manifest["langevin_damping"]), args.friction)
        or not math.isclose(float(manifest["temperature"]), args.temperature)
    ):
        raise ValueError("clone manifest does not match the microscopic drift protocol")
    trajectories = sorted(
        path / "trajectory.lammpstrj"
        for path in args.clone_directory.glob("clone_*")
        if (path / "trajectory.lammpstrj").is_file()
    )
    if len(trajectories) != args.expected_clone_count:
        raise ValueError("completed trajectory count does not match the manifest")

    targets: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    for clone_index, trajectory in enumerate(trajectories, start=1):
        reduced = load_or_reduce_clone(
            trajectory,
            clone_index=clone_index,
            cache_directory=args.reduced_cache_directory,
            target_count=args.target_count,
            target_seed=args.target_seed,
            expected_target_indices=targets,
        )
        if targets is None:
            targets = np.asarray(reduced["target_indices"], dtype=int)
        cage_path = args.cage_cache_directory / f"clone_{clone_index:03d}_smooth_cage.npz"
        load_or_extract_cage(
            trajectory,
            cage_path,
            reduced=reduced,
            target_batch_size=args.target_batch_size,
        )
        drift_path = args.drift_cache_directory / f"clone_{clone_index:03d}_decomposed_drift.npz"
        drift = load_or_extract_drift(
            trajectory,
            reduced=reduced,
            smooth_cache_path=cage_path,
            drift_cache_path=drift_path,
            rerun_directory=args.rerun_directory,
            lammps_binary=args.lammps_binary.resolve(),
            parent_restart=args.parent_restart.resolve(),
            friction=args.friction,
            temperature=args.temperature,
            directional_step=args.directional_step,
            sensitivity_steps=np.asarray(args.sensitivity_steps, dtype=float),
            target_batch_size=args.target_batch_size,
            retain_rerun_dump=args.retain_rerun_dumps,
        )
        rows.append(
            {
                "clone_index": clone_index,
                "trajectory_sha256": str(drift["trajectory_sha256"]),
                "maximum_force_cache_absolute_error": float(
                    drift["maximum_force_cache_absolute_error"]
                ),
                "force_cache_relative_rms_error": float(
                    drift["force_cache_relative_rms_error"]
                ),
                "force_cache_correlation": float(drift["force_cache_correlation"]),
                "thermodynamic_claim_allowed": 0,
            }
        )
        print(f"cached clone {clone_index}/{len(trajectories)}", flush=True)

    args.drift_cache_directory.mkdir(parents=True, exist_ok=True)
    cache_manifest = {
        "protocol": "exact_many_particle_smooth_cage_projected_drift_cache",
        "source_clone_manifest": str((args.clone_directory / "manifest.json").resolve()),
        "clone_count": len(rows),
        "target_count": args.target_count,
        "target_seed": args.target_seed,
        "directional_step": args.directional_step,
        "rows": rows,
        "thermodynamic_claim_allowed": False,
    }
    (args.drift_cache_directory / "cache_manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
