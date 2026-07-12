#!/usr/bin/env python3
"""Measure event-conditioned cooperative displacement halos in KA replicas."""

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
    event_conditioned_neighbor_displacement,
    load_lammps_custom_trajectory,
    summarize_neighbor_halo_replicates,
)
from renewal_cage import extract_nonrecrossing_phop_events, phop_values  # noqa: E402


def parse_edges(value: str) -> np.ndarray:
    return np.array([float(item) for item in value.split(",")], dtype=float)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--heldout-start", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--sample-events", type=int, default=1200)
    parser.add_argument("--random-seed", type=int, default=4517)
    parser.add_argument(
        "--distance-edges",
        type=parse_edges,
        default=parse_edges("0,1.4,2,3,4,5,7,10,13.1"),
    )
    parser.add_argument("--integration-max-distance", type=float, default=4.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    shell_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = args.ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][args.heldout_start :]
        positions = positions[:, trajectory["particle_types"] == 0]
        activity_times, activity = phop_values(positions, half_window=args.half_window)
        events = extract_nonrecrossing_phop_events(
            positions,
            threshold=args.threshold,
            half_window=args.half_window,
            recrossing_radius=math.sqrt(args.threshold),
            activity_times=activity_times,
            activity_values=activity,
        )
        valid = np.flatnonzero(
            (events["time"] >= args.half_window)
            & (events["time"] + args.half_window <= len(positions))
        )
        if len(valid) < args.sample_events:
            raise ValueError(f"replicate {replicate_index} has too few complete events")
        rng = np.random.default_rng(args.random_seed + replicate_index)
        selected = np.sort(rng.choice(valid, size=args.sample_events, replace=False))
        event_particles = events["particle"][selected]
        controls = rng.integers(0, positions.shape[1] - 1, size=len(selected))
        controls += controls >= event_particles
        rows, summary = event_conditioned_neighbor_displacement(
            positions,
            events,
            box_lengths=trajectory["box_lengths"],
            distance_edges=args.distance_edges,
            half_window=args.half_window,
            event_indices=selected,
            control_particles=controls,
            integration_max_distance=args.integration_max_distance,
        )
        for row in rows:
            row.update(
                {
                    "replicate": float(replicate_index),
                    "temperature": float(manifest["temperature"]),
                    "threshold": args.threshold,
                    "half_window": float(args.half_window),
                    "heldout_start": float(args.heldout_start),
                    "control_definition": "same_time_uniform_random_non_event_particle",
                    "spatial_measurement_claim_allowed": 1.0,
                    "spatial_model_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        shell_rows.extend(rows)
        summary.update(
            {
                "replicate": float(replicate_index),
                "temperature": float(manifest["temperature"]),
                "threshold": args.threshold,
                "half_window": float(args.half_window),
                "heldout_start": float(args.heldout_start),
                "total_event_count": float(len(events["time"])),
                "random_seed": float(args.random_seed + replicate_index),
                "spatial_measurement_claim_allowed": 1.0,
                "spatial_model_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
        replicate_rows.append(summary)

    curve, verdict = summarize_neighbor_halo_replicates(shell_rows, replicate_rows)
    for row in curve:
        row.update(
            {
                "temperature": float(manifest["temperature"]),
                "threshold": args.threshold,
                "half_window": float(args.half_window),
                "heldout_start": float(args.heldout_start),
                "spatial_measurement_claim_allowed": 1.0,
                "spatial_model_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    verdict.update(
        {
            "temperature": float(manifest["temperature"]),
            "threshold": args.threshold,
            "half_window": float(args.half_window),
            "heldout_start": float(args.heldout_start),
            "sampled_events_per_replicate": float(args.sample_events),
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_shell_replicates.csv"), shell_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicate_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve_summary.csv"), curve)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
