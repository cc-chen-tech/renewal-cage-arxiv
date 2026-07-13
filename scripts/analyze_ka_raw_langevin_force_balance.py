#!/usr/bin/env python3
"""Verify the raw tagged KA Langevin equation from recorded x, v, and f."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import ka_lj_force_and_isotropic_curvature  # noqa: E402
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def conservative_force_series(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    forces: list[np.ndarray] = []
    curvatures: list[np.ndarray] = []
    for frame in positions:
        force, curvature = ka_lj_force_and_isotropic_curvature(
            frame,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target_indices,
        )
        forces.append(force)
        curvatures.append(curvature)
    return np.asarray(forces), np.asarray(curvatures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--damping", type=float, default=1.0)
    args = parser.parse_args()
    if args.target_count < 1 or args.temperature <= 0.0 or args.damping <= 0.0:
        raise ValueError("sampling, temperature, and damping must be positive")
    clones = sorted(path for path in args.clone_directory.glob("clone_*") if (path / "trajectory.lammpstrj").is_file())
    if len(clones) < 2:
        raise ValueError("at least two completed force/velocity clones are required")

    target: np.ndarray | None = None
    details: list[dict[str, object]] = []
    for clone_index, clone in enumerate(clones, start=1):
        trajectory = load_lammps_custom_trajectory(clone / "trajectory.lammpstrj")
        if "velocities" not in trajectory or "forces" not in trajectory:
            raise ValueError("trajectory must use the x,v,f LAMMPS dump schema")
        time_steps = np.asarray(trajectory["timesteps"])
        interval = np.diff(time_steps)
        if len(interval) == 0 or not np.all(interval == interval[0]):
            raise ValueError("trajectory must have a uniform saved-frame interval")
        frame_time = float(interval[0]) * 0.001
        types = np.asarray(trajectory["particle_types"])
        if target is None:
            target = np.random.default_rng(args.seed).choice(np.flatnonzero(types == 0), size=args.target_count, replace=False)
        positions = np.asarray(trajectory["unwrapped_positions"])
        conservative_force, curvature = conservative_force_series(
            positions,
            particle_types=types,
            box_lengths=np.asarray(trajectory["box_lengths"]),
            target_indices=target,
        )
        velocity = np.asarray(trajectory["velocities"])[:, target]
        recorded_total_force = np.asarray(trajectory["forces"])[:, target]
        bath_residual = (
            (velocity[1:] - velocity[:-1]) / frame_time
            - 0.5 * (conservative_force[1:] + conservative_force[:-1])
            + args.damping * 0.5 * (velocity[1:] + velocity[:-1])
        )
        conservative_rms = float(np.sqrt(np.mean(conservative_force**2)))
        total_force_mismatch = float(np.sqrt(np.mean((recorded_total_force - conservative_force) ** 2)))
        residual_variance = float(np.mean(bath_residual**2))
        lag1 = float(np.mean(bath_residual[:-1] * bath_residual[1:]) / residual_variance)
        curvature_series = 0.5 * (curvature[1:] + curvature[:-1])
        residual_squared = np.mean(bath_residual**2, axis=2)
        curvature_flat = curvature_series.reshape(-1)
        residual_flat = residual_squared.reshape(-1)
        curvature_residual_correlation = (
            float(np.corrcoef(curvature_flat, residual_flat)[0, 1])
            if np.std(curvature_flat) > 0.0 and np.std(residual_flat) > 0.0
            else math.nan
        )
        details.append(
            {
                "clone_index": float(clone_index),
                "target_count": float(len(target)),
                "frame_time_tau": frame_time,
                "conservative_force_rms": conservative_rms,
                "recorded_total_to_conservative_force_rms_ratio": total_force_mismatch / conservative_rms,
                "bath_residual_mean": float(np.mean(bath_residual)),
                "bath_residual_fdt_variance_ratio": residual_variance * frame_time / (2.0 * args.damping * args.temperature),
                "bath_residual_lag1_correlation": lag1,
                "curvature_bath_variance_correlation": curvature_residual_correlation,
                "coordinate_definition": "raw_tagged_A_particle",
                "force_definition": "KA_pair_force_recomputed_from_recorded_many_particle_positions",
                "residual_definition": "Delta_v/dt-mean_pair_force+damping*mean_velocity",
                "model": "exact_raw_many_particle_Langevin_force_balance",
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    summary: dict[str, object] = {
        "clone_count": float(len(details)),
        "target_count": float(args.target_count),
        "temperature": args.temperature,
        "damping": args.damping,
        "coordinate_definition": "raw_tagged_A_particle",
        "force_definition": "KA_pair_force_recomputed_from_recorded_many_particle_positions",
        "residual_definition": "Delta_v/dt-mean_pair_force+damping*mean_velocity",
        "model": "exact_raw_many_particle_Langevin_force_balance",
        "thermodynamic_claim_allowed": 0.0,
    }
    for metric in (
        "recorded_total_to_conservative_force_rms_ratio",
        "bath_residual_mean",
        "bath_residual_fdt_variance_ratio",
        "bath_residual_lag1_correlation",
        "curvature_bath_variance_correlation",
    ):
        values = np.asarray([float(row[metric]) for row in details])
        summary[metric] = float(np.mean(values))
        summary[f"{metric}_standard_error"] = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    summary["raw_langevin_noise_candidate"] = float(
        abs(summary["bath_residual_mean"]) <= 0.05
        and abs(summary["bath_residual_lag1_correlation"]) <= 0.10
        and 0.75 <= summary["bath_residual_fdt_variance_ratio"] <= 1.25
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), [summary])


if __name__ == "__main__":
    main()
