#!/usr/bin/env python3
"""Select finite exchange from the temperature dependence of mobility identity."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import correlation_efold_crossing, independent_group_ratio  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-curve", type=Path, required=True)
    parser.add_argument("--low-curve", type=Path, required=True)
    parser.add_argument("--high-cross-half", type=Path, required=True)
    parser.add_argument("--low-cross-half", type=Path, required=True)
    parser.add_argument("--high-waiting-verdict", type=Path, required=True)
    parser.add_argument("--low-waiting-verdict", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    curves = {
        "high": read_rows(args.high_curve),
        "low": read_rows(args.low_curve),
    }
    common_blocks = sorted(
        {float(row["block_size"]) for row in curves["high"]}
        & {float(row["block_size"]) for row in curves["low"]}
    )
    if not common_blocks:
        raise ValueError("temperature curves require at least one common block size")
    crossing_rows: list[dict[str, object]] = []
    crossing_values: dict[tuple[str, float], list[float]] = {}
    for temperature_group, rows in curves.items():
        for block_size in common_blocks:
            selected = [row for row in rows if float(row["block_size"]) == block_size]
            for replicate in sorted({float(row["replicate"]) for row in selected}):
                local = [row for row in selected if float(row["replicate"]) == replicate]
                crossing = correlation_efold_crossing(local)
                crossing.update(
                    {
                        "temperature_group": temperature_group,
                        "replicate": replicate,
                        "block_size": block_size,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
                crossing_rows.append(crossing)
                crossing_values.setdefault((temperature_group, block_size), []).append(
                    float(crossing["efold_crossing_time"])
                )
    growth_rows: list[dict[str, object]] = []
    for block_size in common_blocks:
        comparison = independent_group_ratio(
            np.array(crossing_values[("low", block_size)]),
            np.array(crossing_values[("high", block_size)]),
            relative_equivalence_margin=0.2,
        )
        comparison.update(
            {
                "block_size": block_size,
                "cooling_exchange_time_growth_detected": comparison["growth_detected"],
                "thermodynamic_claim_allowed": 0.0,
            }
        )
        growth_rows.append(comparison)

    high_cross = np.array(
        [float(row["particle_identity_correlation"]) for row in read_rows(args.high_cross_half)]
    )
    low_cross = np.array(
        [float(row["particle_identity_correlation"]) for row in read_rows(args.low_cross_half)]
    )
    cross_growth = independent_group_ratio(
        low_cross,
        high_cross,
        relative_equivalence_margin=0.2,
    )
    high_waiting = read_one(args.high_waiting_verdict)
    low_waiting = read_one(args.low_waiting_verdict)
    high_iid = high_waiting["consensus_verdict"] == "empirical_iid_waiting_law_sufficient"
    low_environment = (
        low_waiting["consensus_verdict"] == "persistent_particle_environment_required"
    )
    all_growth = all(float(row["growth_detected"]) == 1.0 for row in growth_rows)
    finite_exchange = bool(high_iid and low_environment and all_growth)
    verdict = {
        "common_block_sizes": ";".join(f"{value:g}" for value in common_blocks),
        "high_temperature_waiting_verdict": high_waiting["consensus_verdict"],
        "low_temperature_waiting_verdict": low_waiting["consensus_verdict"],
        "waiting_mechanism_crossover_detected": float(high_iid and low_environment),
        "exchange_time_growth_detected_all_block_sizes": float(all_growth),
        "minimum_exchange_time_growth_ratio": min(
            float(row["mean_ratio"]) for row in growth_rows
        ),
        "minimum_exchange_time_growth_ci95_low": min(
            float(row["ci95_low_ratio"]) for row in growth_rows
        ),
        "cross_half_identity_correlation_growth_ratio": cross_growth["mean_ratio"],
        "cross_half_identity_correlation_growth_ci95_low": cross_growth["ci95_low_ratio"],
        "cross_half_identity_correlation_growth_ci95_high": cross_growth["ci95_high_ratio"],
        "pure_static_particle_rate_disorder_rejected": float(all_growth),
        "finite_exchange_environment_claim_allowed": float(finite_exchange),
        "finite_waiting_sequence_memory_required": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "source_dois": "10.1063/1.2803062;10.3390/ijms23073556",
        "source_alignment": "cooling_decouples_exchange_and_persistence_and_strengthens_dynamic_heterogeneity",
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_crossings.csv"),
        crossing_rows,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_growth.csv"),
        growth_rows,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
