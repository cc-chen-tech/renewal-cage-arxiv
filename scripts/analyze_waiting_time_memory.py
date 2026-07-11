#!/usr/bin/env python3
"""Compare empirical, shuffled, and gamma waiting-time count fluctuations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renewal_cage import waiting_time_shuffle_diagnostics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_csv", type=Path)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--particle-count", type=int, required=True)
    parser.add_argument("--count-windows", default="512,1024,2048,4096,8192")
    parser.add_argument("--shuffle-replicates", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.event_csv.open() as handle:
        source_rows = list(csv.DictReader(handle))
    events = {
        "particle": np.array([int(float(row["particle"])) for row in source_rows]),
        "time": np.array([float(row["global_time"]) for row in source_rows]),
    }
    rows = []
    for index, count_window in enumerate(float(item) for item in args.count_windows.split(",")):
        result = waiting_time_shuffle_diagnostics(
            events,
            duration=args.duration,
            particle_count=args.particle_count,
            count_window=count_window,
            shuffle_replicates=args.shuffle_replicates,
            random_seed=args.random_seed + index,
        )
        result.update(
            {
                "temperature": float(source_rows[0]["temperature"]),
                "threshold": float(source_rows[0]["threshold"]),
                "half_window": float(source_rows[0]["half_window"]),
                "event_definition": "candelier_phop_contiguous_peak_recursive_ABA_removal",
                "source_doi": "10.5281/zenodo.7469766",
                "thermodynamic_claim_allowed": 0.0,
            }
        )
        rows.append(result)

    median_empirical_error = float(np.median([row["empirical_iid_relative_error"] for row in rows]))
    median_gamma_error = float(np.median([row["gamma_iid_relative_error"] for row in rows]))
    median_memory_excess = float(np.median([row["temporal_memory_excess_fraction"] for row in rows]))
    median_environment_excess = float(
        np.median([row["persistent_environment_excess_fraction"] for row in rows])
    )
    correlation = float(rows[0]["waiting_lag1_correlation"])
    correlation_null_sigma = float(rows[0]["shuffle_lag1_correlation_standard_deviation"])
    correlation_z = abs(
        (correlation - float(rows[0]["shuffle_lag1_correlation_mean"]))
        / correlation_null_sigma
    )
    if median_empirical_error <= 0.15:
        verdict = "empirical_iid_waiting_law_sufficient"
    elif median_memory_excess > 0.20 and correlation_z >= 2.0:
        verdict = "temporal_waiting_memory_required"
    elif median_environment_excess > 0.20:
        verdict = "persistent_particle_environment_required"
    elif median_gamma_error - median_empirical_error > 0.15:
        verdict = "gamma_shape_failure_empirical_law_improves"
    else:
        verdict = "waiting_failure_unresolved"
    for row in rows:
        row["median_empirical_iid_relative_error"] = median_empirical_error
        row["median_gamma_iid_relative_error"] = median_gamma_error
        row["median_temporal_memory_excess_fraction"] = median_memory_excess
        row["median_persistent_environment_excess_fraction"] = median_environment_excess
        row["waiting_correlation_z_vs_shuffle"] = correlation_z
        row["waiting_failure_verdict"] = verdict

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
