#!/usr/bin/env python3
"""Run the fixed event-clock threshold grid across independent KA replicates."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_prl_event_clock_closure import load_a_positions, parse_numbers  # noqa: E402
from prl_event_clock_closure import summarize_threshold_ensemble, time_split_event_clock_closure  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.08,0.12,0.16,0.20,0.24")
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--lags", default="4,8,16,32,64,128,256,512")
    parser.add_argument("--wave-numbers", default="5,7.25,9")
    parser.add_argument("--overlap-radius", type=float, default=0.3)
    parser.add_argument("--origin-stride", type=int, default=4)
    args = parser.parse_args()

    thresholds = parse_numbers(args.thresholds, float)
    lags = parse_numbers(args.lags, int)
    wave_numbers = parse_numbers(args.wave_numbers, float)
    details: list[dict[str, object]] = []
    for directory in sorted(args.ensemble_directory.glob("replicate_*")):
        trajectory = directory / "trajectory.lammpstrj"
        if not trajectory.is_file():
            continue
        positions = load_a_positions(trajectory)
        for threshold in thresholds:
            result = time_split_event_clock_closure(
                positions,
                train_stop=(len(positions) - 1) // 2,
                threshold=float(threshold),
                half_window=args.half_window,
                lags=lags,
                wave_numbers=wave_numbers,
                overlap_radius=args.overlap_radius,
                origin_stride=args.origin_stride,
            )
            score = result["variant_scores"]["full_event_clock"]
            details.append(
                {
                    "replicate": directory.name,
                    "threshold": float(threshold),
                    "event_definition": result["event_definition"],
                    "train_event_count": result["train_event_count"],
                    "D_relerr": score["diffusion_relative_error"],
                    "ngp_rmse": score["ngp"]["relative_rmse"],
                    "fs5_rmse": score["fs"]["fs_k5"]["relative_rmse"],
                    "fs7p25_rmse": score["fs"]["fs_k7.25"]["relative_rmse"],
                    "fs9_rmse": score["fs"]["fs_k9"]["relative_rmse"],
                    "fit_parameters_from_macro_observables": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    if not details:
        raise ValueError("no completed replicate trajectory.lammpstrj files were found")
    metrics = ("D_relerr", "ngp_rmse", "fs5_rmse", "fs7p25_rmse", "fs9_rmse")
    summary = summarize_threshold_ensemble(details, metric_keys=metrics)
    for row in summary:
        row.update(
            {
                "event_definition": "candelier_phop_contiguous_peak_recursive_ABA_removal",
                "source_class": "real_reproduced_KA_independent_velocity_replicates",
                "prediction_scope": "time_split_train_microstatistics_to_heldout_macro_observables",
                "threshold_selection": "complete_preregistered_grid_no_posthoc_selection",
                "fit_parameters_from_macro_observables": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)


if __name__ == "__main__":
    main()
