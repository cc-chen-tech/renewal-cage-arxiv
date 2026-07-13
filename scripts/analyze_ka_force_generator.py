#!/usr/bin/env python3
"""Validate KA force-generator observables on full-state Langevin paths."""

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
    force_generator_increment_diagnostic,
    ka_lj_force_generator_observables,
    ka_lj_second_force_generator,
)
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--target-id", type=int, default=821)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--integration-time-step", type=float, default=0.001)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--stochastic-frame-limit", type=int, default=102)
    args = parser.parse_args()
    if args.target_id < 1 or args.temperature < 0.0 or args.friction < 0.0:
        raise ValueError("target-id, temperature, and friction must be physical")
    if args.integration_time_step <= 0.0 or args.directional_step <= 0.0 or args.stochastic_frame_limit < 4:
        raise ValueError("time steps and stochastic-frame-limit must be positive")

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    metric_names = (
        "force_derivative_relative_l2",
        "force_derivative_correlation",
        "innovation_trace_variance_ratio",
        "innovation_mean_squared_mahalanobis",
        "innovation_normalized_mean",
    )
    for clone_index, path in enumerate(args.trajectories, start=1):
        trajectory = load_lammps_custom_trajectory(path)
        if "velocities" not in trajectory:
            raise ValueError(f"{path}: trajectory must contain velocities")
        positions = np.asarray(trajectory["unwrapped_positions"])
        velocities = np.asarray(trajectory["velocities"])
        particle_types = np.asarray(trajectory["particle_types"])
        box_lengths = np.asarray(trajectory["box_lengths"])
        timesteps = np.asarray(trajectory["timesteps"])
        target_index = args.target_id - 1
        if target_index >= positions.shape[1]:
            raise ValueError(f"{path}: target id is outside the atom table")
        intervals = np.diff(timesteps)
        if len(intervals) == 0 or not np.all(intervals == intervals[0]):
            raise ValueError(f"{path}: saved timesteps must be uniform")
        frame_time = float(intervals[0]) * args.integration_time_step
        target = np.array([target_index])
        force: list[np.ndarray] = []
        generator: list[np.ndarray] = []
        covariance_rate: list[np.ndarray] = []
        for frame_positions, frame_velocities in zip(positions, velocities):
            observables = ka_lj_force_generator_observables(
                frame_positions,
                velocities=frame_velocities,
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=target,
                friction=args.friction,
                temperature=args.temperature,
            )
            force.append(observables["force"][0])
            generator.append(observables["force_generator"][0])
            covariance_rate.append(observables["force_generator_noise_covariance_rate"][0])
        force_array = np.asarray(force)
        generator_array = np.asarray(generator)
        covariance_array = np.asarray(covariance_rate)
        stochastic_frames = min(args.stochastic_frame_limit, len(positions))
        second_array = np.asarray(
            [
                ka_lj_second_force_generator(
                    positions[frame],
                    velocities=velocities[frame],
                    particle_types=particle_types,
                    box_lengths=box_lengths,
                    target_indices=target,
                    friction=args.friction,
                    directional_step=args.directional_step,
                )[0]
                for frame in range(stochastic_frames)
            ]
        )
        diagnostic = force_generator_increment_diagnostic(
            force_array[:stochastic_frames],
            generator_array[:stochastic_frames],
            second_array,
            covariance_array[:stochastic_frames],
            frame_time=frame_time,
        )
        summary_rows.append(
            {
                "record": "clone",
                "clone_index": clone_index,
                "trajectory": str(path),
                "target_id": args.target_id,
                "frame_time": frame_time,
                "frame_count": len(positions),
                "stochastic_frame_count": stochastic_frames,
                "directional_step": args.directional_step,
                **diagnostic,
            }
        )
        for frame, (timestep, force_value, generator_value) in enumerate(zip(timesteps, force_array, generator_array)):
            second_value: np.ndarray | None = second_array[frame] if frame < stochastic_frames else None
            curve_rows.append(
                {
                    "clone_index": clone_index,
                    "time": float(timestep * args.integration_time_step),
                    **{f"force_{axis}": float(force_value[index]) for index, axis in enumerate("xyz")},
                    **{f"force_generator_{axis}": float(generator_value[index]) for index, axis in enumerate("xyz")},
                    **{
                        f"second_force_generator_{axis}": "" if second_value is None else float(second_value[index])
                        for index, axis in enumerate("xyz")
                    },
                    "thermodynamic_claim_allowed": 0,
                }
            )
    aggregate: dict[str, object] = {
        "record": "aggregate",
        "clone_index": "",
        "trajectory": "",
        "target_id": args.target_id,
        "frame_time": summary_rows[0]["frame_time"],
        "frame_count": summary_rows[0]["frame_count"],
        "stochastic_frame_count": summary_rows[0]["stochastic_frame_count"],
        "directional_step": args.directional_step,
        "thermodynamic_claim_allowed": 0,
    }
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in summary_rows])
        aggregate[metric] = float(np.mean(values))
        aggregate[f"{metric}_standard_error"] = (
            float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        )
    summary_rows.append(aggregate)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)


if __name__ == "__main__":
    main()
