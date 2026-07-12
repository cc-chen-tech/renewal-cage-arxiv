#!/usr/bin/env python3
"""Predict held-out KA diffusion from calibration-window p_hop events."""

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
    load_lammps_custom_trajectory,
    summarize_heldout_event_transport,
    trajectory_diffusion_estimate,
)
from renewal_cage import (  # noqa: E402
    event_clock_statistics,
    extract_nonrecrossing_phop_events,
    phop_values,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--calibration-time", type=int, default=5000)
    parser.add_argument("--heldout-diffusion-lag", type=int, default=4096)
    parser.add_argument("--origin-stride", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--maximum-coverage", type=float, default=1.2)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    production_time = int(round(float(manifest["production_time_tau"])))
    if production_time != 2 * args.calibration_time:
        raise ValueError("held-out protocol requires equal calibration and prediction windows")
    rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = args.ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"]
        mask = trajectory["particle_types"] == 0
        a_positions = positions[:, mask]
        calibration = a_positions[: args.calibration_time + 1]
        heldout = a_positions[args.calibration_time :]
        activity_times, activity = phop_values(calibration, half_window=args.half_window)
        events = extract_nonrecrossing_phop_events(
            calibration,
            threshold=args.threshold,
            half_window=args.half_window,
            recrossing_radius=math.sqrt(args.threshold),
            activity_times=activity_times,
            activity_values=activity,
        )
        event_stats = event_clock_statistics(
            events,
            duration=float(args.calibration_time),
            particle_count=calibration.shape[1],
            dimension=calibration.shape[2],
        )
        observed = trajectory_diffusion_estimate(
            heldout,
            lag=args.heldout_diffusion_lag,
            origin_stride=args.origin_stride,
        )
        rows.append(
            {
                "replicate": float(replicate_index),
                "temperature": float(manifest["temperature"]),
                "calibration_time": float(args.calibration_time),
                "heldout_time": float(production_time - args.calibration_time),
                "heldout_diffusion_lag": float(args.heldout_diffusion_lag),
                "threshold": args.threshold,
                "half_window": float(args.half_window),
                "calibration_event_count": event_stats["event_count"],
                "calibration_exchange_interval_count": event_stats["exchange_interval_count"],
                "observed_diffusion": observed,
                "uncorrelated_event_diffusion": event_stats["uncorrelated_diffusion"],
                "correlated_event_diffusion": event_stats["correlated_diffusion"],
                "uncorrelated_coverage": event_stats["uncorrelated_diffusion"] / observed,
                "correlated_coverage": event_stats["correlated_diffusion"] / observed,
                "macro_fit_parameter_count": 0.0,
                "independence_class": str(manifest["independence_class"]),
                "thermodynamic_claim_allowed": 0.0,
            }
        )

    summary, verdict = summarize_heldout_event_transport(
        rows,
        minimum_coverage=args.minimum_coverage,
        maximum_coverage=args.maximum_coverage,
    )
    for row in summary:
        row["temperature"] = float(manifest["temperature"])
        row["calibration_time"] = float(args.calibration_time)
        row["heldout_time"] = float(production_time - args.calibration_time)
        row["thermodynamic_claim_allowed"] = 0.0
    verdict["temperature"] = float(manifest["temperature"])
    verdict["calibration_time"] = float(args.calibration_time)
    verdict["heldout_time"] = float(production_time - args.calibration_time)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
