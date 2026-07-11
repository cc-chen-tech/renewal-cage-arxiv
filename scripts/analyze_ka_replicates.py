#!/usr/bin/env python3
"""Reduce completed KA replicate dumps to independent-sample observables."""

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

from ka_replicates import load_lammps_custom_trajectory, summarize_replicate_curves  # noqa: E402
from renewal_cage import (  # noqa: E402
    block_trajectory_observables,
    event_clock_statistics,
    extract_nonrecrossing_phop_events,
    phop_values,
    summarize_block_trajectory_observables,
)


def parse_numbers(value: str, dtype: type[float] | type[int]) -> np.ndarray:
    return np.array([dtype(item) for item in value.split(",")], dtype=dtype)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_event_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "event_rate",
        "exchange_mean",
        "exchange_cv2",
        "stationary_persistence_mean",
        "persistence_exchange_ratio",
        "jump_squared_mean",
        "count_fano",
        "correlated_diffusion",
    )
    summary: list[dict[str, object]] = []
    for metric in metrics:
        values = np.array([float(row[metric]) for row in rows])
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(values))
        summary.append(
            {
                "metric": metric,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "ci95_low": mean - 1.96 * standard_error,
                "ci95_high": mean + 1.96 * standard_error,
                "independent_replicate_count": float(len(values)),
                "temperature": float(rows[0]["temperature"]),
                "threshold": float(rows[0]["threshold"]),
                "half_window": float(rows[0]["half_window"]),
                "independence_class": str(rows[0]["independence_class"]),
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return summary


def analyze_ensemble(
    ensemble_directory: Path,
    *,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    diffusion_lag: int,
    overlap_radius: float,
    origin_stride: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    manifest = json.loads((ensemble_directory / "ensemble_manifest.json").read_text())
    expected_replicates = int(manifest["replicate_count"])
    curve_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        timesteps = trajectory["timesteps"]
        if len(timesteps) < 2 or not np.all(np.diff(timesteps) == 1000):
            raise ValueError("replicate frames must be separated by exactly one tau")
        production_time = float(manifest["production_time_tau"])
        if timesteps[0] != 0 or timesteps[-1] != int(round(production_time * 1000.0)):
            raise ValueError("replicate does not cover the preregistered production window")
        positions = trajectory["unwrapped_positions"]
        particle_mask = trajectory["particle_types"] == 0
        rows = block_trajectory_observables(
            positions,
            lags=lags,
            block_size=len(positions) - 1,
            wave_numbers=wave_numbers,
            overlap_radius=overlap_radius,
            particle_mask=particle_mask,
            origin_stride=origin_stride,
        )
        for row in rows:
            row["block_index"] = float(replicate_index - 1)
            row["replicate"] = float(replicate_index)
            row["temperature"] = float(manifest["temperature"])
            row["independence_class"] = str(manifest["independence_class"])
            row["thermodynamic_claim_allowed"] = 0.0
        curve_rows.extend(rows)
        a_positions = positions[:, particle_mask]
        activity_times, activity_values = phop_values(a_positions, half_window=5)
        events = extract_nonrecrossing_phop_events(
            a_positions,
            threshold=0.2,
            half_window=5,
            recrossing_radius=math.sqrt(0.2),
            activity_times=activity_times,
            activity_values=activity_values,
        )
        event_row: dict[str, object] = {
            "replicate": float(replicate_index),
            "temperature": float(manifest["temperature"]),
            "threshold": 0.2,
            "half_window": 5.0,
            "event_definition": "candelier_phop_contiguous_peak_recursive_ABA_removal",
            "independence_class": str(manifest["independence_class"]),
            "thermodynamic_claim_allowed": 0.0,
        }
        event_row.update(
            event_clock_statistics(
                events,
                duration=production_time,
                particle_count=int(np.sum(particle_mask)),
                dimension=3,
            )
        )
        event_rows.append(event_row)

    if len({int(row["replicate"]) for row in curve_rows}) != expected_replicates:
        raise ValueError("replicate count does not match the ensemble manifest")
    fs_keys = [f"fs_k{wave_number:g}".replace(".", "p") for wave_number in wave_numbers]
    curve_summary_rows = summarize_replicate_curves(
        curve_rows,
        metric_keys=["msd", "ngp_3d", "overlap_mean", "overlap_chi4", *fs_keys],
    )
    for row in curve_summary_rows:
        row["temperature"] = float(manifest["temperature"])
        row["independence_class"] = str(manifest["independence_class"])
        row["thermodynamic_claim_allowed"] = 0.0
    fs_key = f"fs_k{wave_numbers[np.argmin(abs(wave_numbers - 7.25))]:g}".replace(".", "p")
    replicate_rows, summary_rows = summarize_block_trajectory_observables(
        curve_rows,
        fs_key=fs_key,
        diffusion_lag=diffusion_lag,
    )
    for row in replicate_rows:
        row["replicate"] = float(row.pop("block_index") + 1.0)
        row["temperature"] = float(manifest["temperature"])
        row["independence_class"] = str(manifest["independence_class"])
        row["thermodynamic_claim_allowed"] = 0.0
    for row in summary_rows:
        row["temperature"] = float(manifest["temperature"])
        row["independent_replicate_count"] = float(expected_replicates)
        row["independence_class"] = str(manifest["independence_class"])
        row["maximum_absolute_initial_fs"] = float(manifest["maximum_absolute_fs_observed"])
        row["thermodynamic_claim_allowed"] = 0.0
    return (
        curve_rows,
        curve_summary_rows,
        replicate_rows,
        summary_rows,
        event_rows,
        summarize_event_rows(event_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--lags", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--wave-numbers", default="5,7.25,9")
    parser.add_argument("--diffusion-lag", type=int, default=512)
    parser.add_argument("--overlap-radius", type=float, default=0.3)
    parser.add_argument("--origin-stride", type=int, default=4)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    curves, curve_summary, replicates, summary, events, event_summary = analyze_ensemble(
        args.ensemble_directory,
        lags=parse_numbers(args.lags, int),
        wave_numbers=parse_numbers(args.wave_numbers, float),
        diffusion_lag=args.diffusion_lag,
        overlap_radius=args.overlap_radius,
        origin_stride=args.origin_stride,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curves.csv"), curves)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_curve_summary.csv"),
        curve_summary,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicates)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_event_replicates.csv"), events)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_event_summary.csv"),
        event_summary,
    )


if __name__ == "__main__":
    main()
