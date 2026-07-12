#!/usr/bin/env python3
"""Run preregistered waiting-law diagnostics on Debye-Waller cage jumps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_replicates import summarize_waiting_diagnostic_rows  # noqa: E402
from ka_replicates import (  # noqa: E402
    extract_debye_waller_cage_jumps,
    load_lammps_custom_trajectory,
    position_fluctuation_values,
)
from renewal_cage import waiting_time_shuffle_diagnostics  # noqa: E402


def parse_windows(value: str) -> np.ndarray:
    return np.array([float(item) for item in value.split(",")], dtype=float)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--heldout-replicates", type=Path, required=True)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument(
        "--count-windows",
        type=parse_windows,
        default=parse_windows("50,100,200,500,1000"),
    )
    parser.add_argument("--shuffle-replicates", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=94531)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    input_rows = {
        int(float(row["replicate"])): row for row in read_rows(args.heldout_replicates)
    }
    rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = args.ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        if replicate_index not in input_rows:
            raise ValueError(f"replicate {replicate_index} lacks a Debye-Waller calibration row")
        calibration_row = input_rows[replicate_index]
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        calibration = positions[: args.calibration_time + 1]
        times, fluctuation = position_fluctuation_values(
            calibration,
            half_window=args.fluctuation_half_window,
        )
        events = extract_debye_waller_cage_jumps(
            calibration,
            debye_waller_factor=float(calibration_row["debye_waller_factor"]),
            half_window=args.fluctuation_half_window,
            activity_times=times,
            activity_values=fluctuation,
        )
        for window_index, count_window in enumerate(args.count_windows):
            if count_window > args.calibration_time / 2:
                raise ValueError("count windows must leave at least two complete windows")
            result = waiting_time_shuffle_diagnostics(
                events,
                duration=float(args.calibration_time),
                particle_count=calibration.shape[1],
                count_window=float(count_window),
                shuffle_replicates=args.shuffle_replicates,
                random_seed=args.random_seed + 1000 * replicate_index + window_index,
            )
            result.update(
                {
                    "replicate": float(replicate_index),
                    "temperature": float(manifest["temperature"]),
                    "event_definition": "rolling_position_variance_above_calibration_debye_waller_factor",
                    "debye_waller_factor": float(calibration_row["debye_waller_factor"]),
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
            rows.append(result)
    replicate_rows, verdict = summarize_waiting_diagnostic_rows(rows)
    for row in replicate_rows:
        row["temperature"] = float(manifest["temperature"])
        row["event_definition"] = "debye_waller_finite_duration_cage_jump"
        row["thermodynamic_claim_allowed"] = 0.0
    verdict.update(
        {
            "temperature": float(manifest["temperature"]),
            "calibration_time": float(args.calibration_time),
            "count_windows": ";".join(f"{value:g}" for value in args.count_windows),
            "shuffle_replicates": float(args.shuffle_replicates),
            "event_definition": "debye_waller_finite_duration_cage_jump",
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_windows.csv"), rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicate_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
