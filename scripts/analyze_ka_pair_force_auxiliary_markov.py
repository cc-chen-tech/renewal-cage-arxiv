#!/usr/bin/env python3
"""Held-out test of a measured KA pair-force auxiliary Markov bath."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import (  # noqa: E402
    ka_lj_force_and_isotropic_curvature,
    time_split_force_auxiliary_markov_diagnostic,
)
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
) -> np.ndarray:
    return np.asarray(
        [
            ka_lj_force_and_isotropic_curvature(
                frame,
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=target_indices,
            )[0]
            for frame in positions
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--simulation-count", type=int, default=8000)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    if args.target_count < 1 or args.simulation_count < 100:
        raise ValueError("target-count and simulation-count must be positive")
    clones = sorted(path for path in args.clone_directory.glob("clone_*") if (path / "trajectory.lammpstrj").is_file())
    if len(clones) < 2:
        raise ValueError("at least two completed force/velocity clones are required")
    target: np.ndarray | None = None
    details: list[dict[str, object]] = []
    for clone_index, clone in enumerate(clones, start=1):
        trajectory = load_lammps_custom_trajectory(clone / "trajectory.lammpstrj")
        if "velocities" not in trajectory:
            raise ValueError("trajectory must use the x,v,f LAMMPS dump schema")
        timesteps = np.asarray(trajectory["timesteps"])
        interval = np.diff(timesteps)
        if len(interval) == 0 or not np.all(interval == interval[0]):
            raise ValueError("trajectory must have a uniform saved-frame interval")
        frame_time = float(interval[0]) * 0.001
        types = np.asarray(trajectory["particle_types"])
        if target is None:
            target = np.random.default_rng(args.seed).choice(np.flatnonzero(types == 0), size=args.target_count, replace=False)
        positions = np.asarray(trajectory["unwrapped_positions"])[:, target]
        conservative_force = conservative_force_series(
            np.asarray(trajectory["unwrapped_positions"]),
            particle_types=types,
            box_lengths=np.asarray(trajectory["box_lengths"]),
            target_indices=target,
        )
        result = time_split_force_auxiliary_markov_diagnostic(
            positions,
            np.asarray(trajectory["velocities"])[:, target],
            conservative_force,
            train_stop=len(positions) // 2,
            frame_time=frame_time,
            lags=np.asarray(args.lags, dtype=int),
            wave_numbers=np.array([1.0, 3.0, 7.25]),
            simulation_count=args.simulation_count,
            seed=args.seed + clone_index,
        )
        transition = np.asarray(result["transition_matrix"])
        details.append(
            {
                "clone_index": float(clone_index),
                "target_count": float(len(target)),
                **{key: value for key, value in result.items() if not isinstance(value, np.ndarray)},
                "transition_v_from_v": float(transition[0, 0]),
                "transition_v_from_force": float(transition[0, 1]),
                "transition_force_from_v": float(transition[1, 0]),
                "transition_force_from_force": float(transition[1, 1]),
                "coordinate_definition": "raw_tagged_A_particle",
                "auxiliary_variable": "exact_KA_pair_force_from_full_recorded_configuration",
                "model": "two_variable_linear_Markov_embedding_with_joint_empirical_innovation",
                "prediction_scope": "first_half_v_pairforce_to_second_half_raw_MSD_NGP_Fs",
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    summary: dict[str, object] = {
        "clone_count": float(len(details)),
        "target_count": float(args.target_count),
        "coordinate_definition": "raw_tagged_A_particle",
        "auxiliary_variable": "exact_KA_pair_force_from_full_recorded_configuration",
        "model": "two_variable_linear_Markov_embedding_with_joint_empirical_innovation",
        "prediction_scope": "first_half_v_pairforce_to_second_half_raw_MSD_NGP_Fs",
        "thermodynamic_claim_allowed": 0.0,
    }
    for metric in (
        "heldout_state_r_squared",
        "diffusion_relative_error",
        "force_auxiliary_fs_max_relative_error",
        "force_auxiliary_ngp_max_absolute_error",
    ):
        values = np.asarray([float(row[metric]) for row in details])
        summary[metric] = float(np.mean(values))
        summary[f"{metric}_standard_error"] = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    summary["single_pair_force_auxiliary_relative_closure_allowed"] = float(
        summary["diffusion_relative_error"] <= 0.2
        and summary["force_auxiliary_fs_max_relative_error"] <= 0.2
        and summary["force_auxiliary_ngp_max_absolute_error"] <= 0.1
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), [summary])


if __name__ == "__main__":
    main()
