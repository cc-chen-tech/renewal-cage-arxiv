#!/usr/bin/env python3
"""Measure persistence and exchange of particle mobility identities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    extract_debye_waller_cage_jumps,
    independent_sample_ci95,
    load_lammps_custom_trajectory,
    particle_event_count_correlation_curve,
    particle_event_count_cross_window_correlation,
    position_fluctuation_values,
)


def parse_block_sizes(value: str) -> np.ndarray:
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
    parser.add_argument("--block-sizes", type=parse_block_sizes, required=True)
    parser.add_argument("--maximum-lag-time", type=float, required=True)
    parser.add_argument("--permutation-replicates", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=76131)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    production_time = int(round(float(manifest["production_time_tau"])))
    if production_time != 2 * args.calibration_time:
        raise ValueError("environment exchange audit requires equal trajectory halves")
    calibration_rows = {
        int(float(row["replicate"])): row for row in read_rows(args.heldout_replicates)
    }
    curve_rows: list[dict[str, object]] = []
    cross_rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = args.ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        threshold = float(calibration_rows[replicate_index]["debye_waller_factor"])
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        calibration = positions[: args.calibration_time + 1]
        heldout = positions[args.calibration_time :]

        def events_for(window: np.ndarray) -> dict[str, np.ndarray]:
            times, fluctuation = position_fluctuation_values(
                window,
                half_window=args.fluctuation_half_window,
            )
            return extract_debye_waller_cage_jumps(
                window,
                debye_waller_factor=threshold,
                half_window=args.fluctuation_half_window,
                activity_times=times,
                activity_values=fluctuation,
            )

        calibration_events = events_for(calibration)
        heldout_events = events_for(heldout)
        for block_size in args.block_sizes:
            block_count = int(args.calibration_time // block_size)
            maximum_lag = min(
                int(args.maximum_lag_time // block_size),
                block_count - 1,
            )
            local = particle_event_count_correlation_curve(
                calibration_events,
                duration=float(args.calibration_time),
                particle_count=calibration.shape[1],
                block_size=float(block_size),
                maximum_lag=maximum_lag,
            )
            for row in local:
                row.update(
                    {
                        "replicate": float(replicate_index),
                        "temperature": float(manifest["temperature"]),
                        "debye_waller_factor": threshold,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
            curve_rows.extend(local)

        cross = particle_event_count_cross_window_correlation(
            calibration_events,
            heldout_events,
            particle_count=calibration.shape[1],
        )
        rng = np.random.default_rng(args.random_seed + replicate_index)
        null = []
        for _ in range(args.permutation_replicates):
            mapping = rng.permutation(calibration.shape[1])
            permuted = {"particle": mapping[heldout_events["particle"]]}
            null.append(
                float(
                    particle_event_count_cross_window_correlation(
                        calibration_events,
                        permuted,
                        particle_count=calibration.shape[1],
                    )["particle_identity_correlation"]
                )
            )
        null_mean = float(np.mean(null))
        null_sd = float(np.std(null, ddof=1))
        cross.update(
            {
                "replicate": float(replicate_index),
                "temperature": float(manifest["temperature"]),
                "calibration_time": float(args.calibration_time),
                "heldout_time": float(production_time - args.calibration_time),
                "calibration_event_count": float(len(calibration_events["time"])),
                "heldout_event_count": float(len(heldout_events["time"])),
                "permutation_null_mean": null_mean,
                "permutation_null_standard_deviation": null_sd,
                "permutation_z_score": (
                    (float(cross["particle_identity_correlation"]) - null_mean) / null_sd
                    if null_sd > 0.0
                    else math.inf
                ),
                "permutation_replicates": float(args.permutation_replicates),
                "debye_waller_factor": threshold,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
        cross_rows.append(cross)

    values = np.array([float(row["particle_identity_correlation"]) for row in cross_rows])
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    ci_low, ci_high, critical = independent_sample_ci95(
        mean=mean,
        standard_error=standard_error,
        sample_count=len(values),
    )
    verdict = {
        "temperature": float(manifest["temperature"]),
        "independent_replicate_count": float(len(values)),
        "mean_cross_half_particle_identity_correlation": mean,
        "standard_error_cross_half_particle_identity_correlation": standard_error,
        "ci95_low_cross_half_particle_identity_correlation": ci_low,
        "ci95_high_cross_half_particle_identity_correlation": ci_high,
        "ci95_critical_value": critical,
        "ci95_method": "student_t_independent_replicates",
        "cross_half_identity_persistence_detected": float(ci_low > 0.0),
        "mean_permutation_z_score": float(
            np.mean([float(row["permutation_z_score"]) for row in cross_rows])
        ),
        "static_disorder_claim_allowed": 0.0,
        "finite_exchange_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_cross_half.csv"), cross_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
