#!/usr/bin/env python3
"""Summarize the temperature crossover in Debye-Waller cage-jump transport."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    independent_group_ratio,
    independent_sample_ci95,
    signed_temperature_separation,
)


SOURCES = (
    "10.1038/srep11770",
    "10.3390/ijms23073556",
)


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
    parser.add_argument("--high-replicates", type=Path, required=True)
    parser.add_argument("--low-replicates", type=Path, required=True)
    parser.add_argument("--high-verdict", type=Path, required=True)
    parser.add_argument("--low-verdict", type=Path, required=True)
    parser.add_argument("--high-correlations", type=Path, required=True)
    parser.add_argument("--low-correlations", type=Path, required=True)
    parser.add_argument("--primary-correlation-lag", type=int, default=2)
    parser.add_argument("--diagnostic-correlation-lag", type=int, default=10)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    high = read_rows(args.high_replicates)
    low = read_rows(args.low_replicates)

    def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
        return np.array([float(row[key]) for row in rows])

    lag1 = signed_temperature_separation(
        values(high, "dw_jump_correlation_lag1_over_q"),
        values(low, "dw_jump_correlation_lag1_over_q"),
    )
    lag2 = signed_temperature_separation(
        values(high, "dw_jump_correlation_lag2_over_q"),
        values(low, "dw_jump_correlation_lag2_over_q"),
    )
    high_factor = values(high, "debye_waller_correlated_diffusion") / values(
        high, "debye_waller_uncorrelated_diffusion"
    )
    low_factor = values(low, "debye_waller_correlated_diffusion") / values(
        low, "debye_waller_uncorrelated_diffusion"
    )
    correction_change = independent_group_ratio(
        low_factor,
        high_factor,
        relative_equivalence_margin=0.2,
    )
    high_verdict = read_one(args.high_verdict)
    low_verdict = read_one(args.low_verdict)
    high_pass = float(high_verdict["heldout_transport_pass"])
    low_pass = float(low_verdict["heldout_transport_pass"])
    low_uncorrelated = values(low, "debye_waller_uncorrelated_coverage")
    low_correlated = values(low, "debye_waller_correlated_coverage")
    low_backtracking_required = float(
        float(np.mean(low_uncorrelated)) > 1.2
        and abs(float(np.mean(low_correlated)) - 1.0)
        < abs(float(np.mean(low_uncorrelated)) - 1.0)
    )
    directional_crossover = float(
        lag1["positive_high_negative_low_reversal"] == 1.0
        and correction_change["decrease_detected"] == 1.0
        and low_backtracking_required == 1.0
    )
    convergence_rows = []
    convergence_by_temperature: dict[str, dict[str, float | str]] = {}
    for label, path in (
        ("high", args.high_correlations),
        ("low", args.low_correlations),
    ):
        correlations = read_rows(path)
        factors: dict[float, dict[int, float]] = {}
        for row in correlations:
            factors.setdefault(float(row["replicate"]), {})[int(float(row["event_lag"]))] = float(
                row["cumulative_green_kubo_factor"]
            )
        ratios = np.array(
            [
                values_by_lag[args.diagnostic_correlation_lag]
                / values_by_lag[args.primary_correlation_lag]
                for values_by_lag in factors.values()
            ]
        )
        mean = float(np.mean(ratios))
        standard_error = float(np.std(ratios, ddof=1) / np.sqrt(len(ratios)))
        ci_low, ci_high, critical = independent_sample_ci95(
            mean=mean,
            standard_error=standard_error,
            sample_count=len(ratios),
        )
        row = {
            "temperature_group": label,
            "primary_correlation_lag": float(args.primary_correlation_lag),
            "diagnostic_correlation_lag": float(args.diagnostic_correlation_lag),
            "mean_diagnostic_over_primary_green_kubo_factor": mean,
            "standard_error_ratio": standard_error,
            "ci95_low_ratio": ci_low,
            "ci95_high_ratio": ci_high,
            "ci95_critical_value": critical,
            "ci95_method": "student_t_independent_replicates",
            "independent_replicate_count": float(len(ratios)),
            "equivalent_within_twenty_percent": float(ci_low >= 0.8 and ci_high <= 1.2),
            "thermodynamic_claim_allowed": 0.0,
        }
        convergence_rows.append(row)
        convergence_by_temperature[label] = row
    metric_rows = [
        {
            "metric_id": "jump_vector_lag1_over_q",
            **lag1,
            "source_dois": ";".join(SOURCES),
            "sota_interpretation": "successive_jump_correlation_causes_low_temperature_subdiffusive_jump_count_motion",
            "thermodynamic_claim_allowed": 0.0,
        },
        {
            "metric_id": "jump_vector_lag2_over_q",
            **lag2,
            "source_dois": ";".join(SOURCES),
            "sota_interpretation": "short_event_lag_direction_memory",
            "thermodynamic_claim_allowed": 0.0,
        },
    ]
    verdict = {
        "high_temperature": 0.58,
        "low_temperature": 0.45,
        "high_temperature_replicate_count": float(len(high)),
        "low_temperature_replicate_count": float(len(low)),
        "lag1_positive_to_negative_reversal": lag1[
            "positive_high_negative_low_reversal"
        ],
        "lag1_high_mean_over_q": lag1["high_mean"],
        "lag1_high_ci95_low_over_q": lag1["high_ci95_low"],
        "lag1_high_ci95_high_over_q": lag1["high_ci95_high"],
        "lag1_low_mean_over_q": lag1["low_mean"],
        "lag1_low_ci95_low_over_q": lag1["low_ci95_low"],
        "lag1_low_ci95_high_over_q": lag1["low_ci95_high"],
        "high_mean_green_kubo_correction_factor": float(np.mean(high_factor)),
        "low_mean_green_kubo_correction_factor": float(np.mean(low_factor)),
        "cooling_correction_factor_ratio": correction_change["mean_ratio"],
        "cooling_correction_factor_ci95_low": correction_change["ci95_low_ratio"],
        "cooling_correction_factor_ci95_high": correction_change["ci95_high_ratio"],
        "low_temperature_uncorrelated_mean_coverage": float(np.mean(low_uncorrelated)),
        "low_temperature_correlated_mean_coverage": float(np.mean(low_correlated)),
        "low_temperature_backtracking_required": low_backtracking_required,
        "high_temperature_heldout_transport_pass": high_pass,
        "low_temperature_heldout_transport_pass": low_pass,
        "directional_crossover_claim_allowed": directional_crossover,
        "high_lag_truncation_equivalent_within_twenty_percent": convergence_by_temperature[
            "high"
        ]["equivalent_within_twenty_percent"],
        "low_lag_truncation_equivalent_within_twenty_percent": convergence_by_temperature[
            "low"
        ]["equivalent_within_twenty_percent"],
        "cross_temperature_transport_closure_claim_allowed": float(
            high_pass == 1.0 and low_pass == 1.0
        ),
        "primary_remaining_failure": (
            "low_temperature_replicate_transport_inconsistency"
            if low_pass == 0.0
            else "none"
        ),
        "source_dois": ";".join(SOURCES),
        "source_alignment": "same_3d_KALJ_low_temperature_successive_jump_subdiffusion",
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv"), metric_rows)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_convergence.csv"),
        convergence_rows,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
