#!/usr/bin/env python3
"""Measure distance-resolved event-count covariance in a completed KA replicate."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    distance_resolved_event_count_covariance,
    fit_spatial_covariance_length,
    load_lammps_custom_trajectory,
    summarize_spatial_covariance_replicates,
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


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def aggregate_block_rows(
    rows: list[dict[str, object]],
    *,
    minimum_distance: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    replicate_rows, summary_rows, fits, fit_summary = summarize_spatial_covariance_replicates(
        rows,
        minimum_distance=minimum_distance,
    )
    for row in replicate_rows:
        row.update(
            {
                "uncertainty_scope": "within_trajectory_block_average",
                "spatial_measurement_claim_allowed": 1.0,
                "spatial_model_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    for row in summary_rows:
        row.update(
            {
                "uncertainty_scope": "independent_replicates",
                "spatial_measurement_claim_allowed": 1.0,
                "spatial_model_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    for row in fits:
        row.update(
            {
                "fit_status": "independent_replicate_fit",
                "spatial_measurement_claim_allowed": 1.0,
                "spatial_model_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    fit_summary.update(
        {
            "fit_status": "between_replicate_uncertainty",
            "uncertainty_scope": "independent_replicates",
            "spatial_measurement_claim_allowed": 1.0,
            "spatial_model_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    return replicate_rows, summary_rows, fits, fit_summary


def analyze(
    trajectory_path: Path,
    *,
    temperature: float,
    spatial_block: int,
    count_window: int,
    distance_edges: np.ndarray,
    threshold: float,
    half_window: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trajectory = load_lammps_custom_trajectory(trajectory_path)
    positions = trajectory["unwrapped_positions"]
    particle_mask = trajectory["particle_types"] == 0
    a_positions = positions[:, particle_mask]
    times, activity = phop_values(a_positions, half_window=half_window)
    events = extract_nonrecrossing_phop_events(
        a_positions,
        threshold=threshold,
        half_window=half_window,
        recrossing_radius=math.sqrt(threshold),
        activity_times=times,
        activity_values=activity,
    )
    production_time = len(positions) - 1
    block_count = production_time // spatial_block
    if block_count < 3:
        raise ValueError("at least three complete spatial blocks are required")

    block_rows: list[dict[str, object]] = []
    for block_index in range(block_count):
        start = block_index * spatial_block
        stop = start + spatial_block
        selected = (events["time"] >= start) & (events["time"] < stop)
        local_events = {
            "particle": events["particle"][selected],
            "time": events["time"][selected] - start,
        }
        rows = distance_resolved_event_count_covariance(
            local_events,
            a_positions[start],
            trajectory["box_lengths"],
            duration=float(spatial_block),
            count_window=float(count_window),
            distance_edges=distance_edges,
        )
        for row in rows:
            row["temperature"] = temperature
            row["block_index"] = float(block_index)
            row["spatial_block"] = float(spatial_block)
            row["threshold"] = threshold
            row["half_window"] = float(half_window)
            row["uncertainty_scope"] = "single_trajectory_nonoverlapping_time_blocks"
            row["independent_replicate_count"] = 1.0
            row["spatial_model_claim_allowed"] = 0.0
            row["thermodynamic_claim_allowed"] = 0.0
        block_rows.extend(rows)

    summary_rows: list[dict[str, object]] = []
    for distance_low in sorted({float(row["distance_low"]) for row in block_rows}):
        selected = [row for row in block_rows if float(row["distance_low"]) == distance_low]
        values = np.array([float(row["covariance_excess_over_all_pairs"]) for row in selected])
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(values))
        summary_rows.append(
            {
                "temperature": temperature,
                "distance_low": distance_low,
                "distance_high": float(selected[0]["distance_high"]),
                "distance_midpoint": float(selected[0]["distance_midpoint"]),
                "mean_covariance_excess": mean,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "ci95_low": mean - 1.96 * standard_error,
                "ci95_high": mean + 1.96 * standard_error,
                "mean_pair_count": float(np.mean([float(row["pair_count"]) for row in selected])),
                "block_count": float(len(selected)),
                "count_window": float(count_window),
                "spatial_block": float(spatial_block),
                "threshold": threshold,
                "uncertainty_scope": "single_trajectory_nonoverlapping_time_blocks",
                "independent_replicate_count": 1.0,
                "spatial_model_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return block_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path, nargs="?")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--spatial-block", type=int, default=1000)
    parser.add_argument("--count-window", type=int, default=100)
    parser.add_argument("--distance-edges", type=parse_edges, default=parse_edges("0,1.1,1.4,1.8,2.5,3.5,5,7.5,13.1"))
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--fit-minimum-distance", type=float, default=1.1)
    parser.add_argument("--aggregate-block-files", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.aggregate_block_files:
        if args.trajectory is not None:
            parser.error("trajectory and --aggregate-block-files are mutually exclusive")
        combined: list[dict[str, object]] = []
        for replicate, path in enumerate(args.aggregate_block_files, start=1):
            for row in read_rows(path):
                row["replicate"] = float(replicate)
                combined.append(row)
        replicate_rows, summary, fits, fit_summary = aggregate_block_rows(
            combined,
            minimum_distance=args.fit_minimum_distance,
        )
        write_rows(
            args.output_prefix.with_name(args.output_prefix.name + "_replicate_curves.csv"),
            replicate_rows,
        )
        write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
        write_rows(args.output_prefix.with_name(args.output_prefix.name + "_fits.csv"), fits)
        write_rows(
            args.output_prefix.with_name(args.output_prefix.name + "_fit_summary.csv"),
            [fit_summary],
        )
        return
    if args.trajectory is None or args.temperature is None:
        parser.error("trajectory and --temperature are required unless --aggregate-block-files is used")

    blocks, summary = analyze(
        args.trajectory,
        temperature=args.temperature,
        spatial_block=args.spatial_block,
        count_window=args.count_window,
        distance_edges=args.distance_edges,
        threshold=args.threshold,
        half_window=args.half_window,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_blocks.csv"), blocks)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    fit: dict[str, object] = fit_spatial_covariance_length(
        summary,
        minimum_distance=args.fit_minimum_distance,
    )
    fit.update(
        {
            "temperature": args.temperature,
            "fit_status": "exploratory_single_trajectory_pilot",
            "independent_replicate_count": 1.0,
            "spatial_model_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_fit.csv"), [fit])


if __name__ == "__main__":
    main()
