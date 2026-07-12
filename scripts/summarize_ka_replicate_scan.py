#!/usr/bin/env python3
"""Compare independent KA restarts across temperature and against parent blocks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import temperature_scan_verdict  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def independent_replicate_count(rows: list[dict[str, str]]) -> float:
    counts = {float(row["independent_replicate_count"]) for row in rows}
    if len(counts) != 1:
        raise ValueError("summary rows disagree on independent replicate count")
    return counts.pop()


def protocol_comparison(
    temperature: float,
    replicate_rows: list[dict[str, str]],
    parent_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    replicate = {row["metric"]: row for row in replicate_rows}
    parent = {row["metric"]: row for row in parent_rows}
    rows: list[dict[str, object]] = []
    for metric in (
        "diffusion",
        "alpha_relaxation_time",
        "diffusion_alpha_product",
        "ngp_peak",
        "overlap_chi4_peak",
    ):
        new = replicate[metric]
        old = parent[metric]
        new_low, new_high = float(new["ci95_low"]), float(new["ci95_high"])
        old_low, old_high = float(old["ci95_low"]), float(old["ci95_high"])
        rows.append(
            {
                "temperature": temperature,
                "metric": metric,
                "replicate_mean": float(new["mean"]),
                "parent_block_mean": float(old["mean"]),
                "relative_protocol_shift": float(new["mean"]) / float(old["mean"]) - 1.0,
                "ci95_overlap": max(new_low, old_low) <= min(new_high, old_high),
                "replicate_ci95_low": new_low,
                "replicate_ci95_high": new_high,
                "parent_block_ci95_low": old_low,
                "parent_block_ci95_high": old_high,
                "interpretation": "statistical_and_protocol_uncertainty_must_be_reported_separately",
                "thermodynamic_claim_allowed": False,
            }
        )
    return rows


def provenance_rows(manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        rows.append(
            {
                "temperature": float(manifest["temperature"]),
                "replicate": int(replicate["replicate"]),
                "source_doi": manifest["source_doi"],
                "source_sha256": manifest["source_sha256"],
                "source_frame_index": int(replicate["source_frame_index"]),
                "velocity_seed": int(replicate["velocity_seed"]),
                "equilibration_time_tau": float(manifest["equilibration_time_tau"]),
                "production_time_tau": float(manifest["production_time_tau"]),
                "independence_wave_number": float(manifest["independence_wave_number"]),
                "maximum_absolute_initial_fs": float(manifest["maximum_absolute_fs_observed"]),
                "independence_class": manifest["independence_class"],
                "independently_prepared_parent_samples": bool(
                    manifest["independently_prepared_parent_samples"]
                ),
                "simulation_engine": "LAMMPS_22Jul2025_update4",
                "ensemble": "NVT",
                "thermostat": "Nose_Hoover_tau_10",
                "timestep_tau": 0.001,
                "saved_frame_interval_tau": 1.0,
                "potential": "standard_80_20_Kob_Andersen_LJ_shifted_at_2p5_sigma",
                "thermodynamic_claim_allowed": False,
            }
        )
    return rows


def event_temperature_trends(
    high_rows: list[dict[str, str]],
    low_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    high = {row["metric"]: row for row in high_rows}
    low = {row["metric"]: row for row in low_rows}
    high_count = independent_replicate_count(high_rows)
    low_count = independent_replicate_count(low_rows)
    directions = {
        "event_rate": "decrease",
        "exchange_mean": "increase",
        "stationary_persistence_mean": "increase",
        "persistence_exchange_ratio": "increase",
        "count_fano": "increase",
        "correlated_diffusion": "decrease",
    }
    rows: list[dict[str, object]] = []
    for metric, direction in directions.items():
        high_mean, low_mean = float(high[metric]["mean"]), float(low[metric]["mean"])
        if direction == "increase":
            ratio = low_mean / high_mean
            separated = float(low[metric]["ci95_low"]) > float(high[metric]["ci95_high"])
        else:
            ratio = high_mean / low_mean
            separated = float(low[metric]["ci95_high"]) < float(high[metric]["ci95_low"])
        rows.append(
            {
                "metric": metric,
                "cooling_direction": direction,
                "high_temperature_mean": high_mean,
                "high_temperature_ci95_low": float(high[metric]["ci95_low"]),
                "high_temperature_ci95_high": float(high[metric]["ci95_high"]),
                "low_temperature_mean": low_mean,
                "low_temperature_ci95_low": float(low[metric]["ci95_low"]),
                "low_temperature_ci95_high": float(low[metric]["ci95_high"]),
                "effect_ratio": ratio,
                "directional_ci95_separated": separated,
                "trend_pass": ratio > 1.0 and separated,
                "high_temperature_replicate_count": high_count,
                "low_temperature_replicate_count": low_count,
                "thermodynamic_claim_allowed": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-summary", type=Path, required=True)
    parser.add_argument("--low-summary", type=Path, required=True)
    parser.add_argument("--high-parent-summary", type=Path, required=True)
    parser.add_argument("--low-parent-summary", type=Path, required=True)
    parser.add_argument("--high-event-summary", type=Path, required=True)
    parser.add_argument("--low-event-summary", type=Path, required=True)
    parser.add_argument("--high-temperature", type=float, required=True)
    parser.add_argument("--low-temperature", type=float, required=True)
    parser.add_argument("--high-ensemble-manifest", type=Path, required=True)
    parser.add_argument("--low-ensemble-manifest", type=Path, required=True)
    parser.add_argument("--trend-output", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--event-trend-output", type=Path, required=True)
    args = parser.parse_args()

    high = read_rows(args.high_summary)
    low = read_rows(args.low_summary)
    trend = temperature_scan_verdict(high, low)
    high_count = independent_replicate_count(high)
    low_count = independent_replicate_count(low)
    for row in trend:
        row["high_temperature"] = args.high_temperature
        row["low_temperature"] = args.low_temperature
        row["high_temperature_replicate_count"] = high_count
        row["low_temperature_replicate_count"] = low_count
        row["independence_class"] = "decorrelated_parent_frames_plus_velocity_seeds"
    protocol = protocol_comparison(
        args.high_temperature,
        high,
        read_rows(args.high_parent_summary),
    ) + protocol_comparison(
        args.low_temperature,
        low,
        read_rows(args.low_parent_summary),
    )
    write_rows(args.trend_output, trend)
    write_rows(args.protocol_output, protocol)
    write_rows(
        args.provenance_output,
        provenance_rows(args.high_ensemble_manifest) + provenance_rows(args.low_ensemble_manifest),
    )
    write_rows(
        args.event_trend_output,
        event_temperature_trends(
            read_rows(args.high_event_summary),
            read_rows(args.low_event_summary),
        ),
    )


if __name__ == "__main__":
    main()
