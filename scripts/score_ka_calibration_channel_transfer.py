#!/usr/bin/env python3
"""Score calibration jump/cage channels against held-out trajectory observables."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import independent_sample_ci95  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(values: np.ndarray) -> dict[str, float | str]:
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    ci_low, ci_high, critical = independent_sample_ci95(
        mean=mean,
        standard_error=standard_error,
        sample_count=len(values),
    )
    return {
        "mean_paired_difference": mean,
        "standard_error_paired_difference": standard_error,
        "ci95_low_paired_difference": ci_low,
        "ci95_high_paired_difference": ci_high,
        "ci95_critical_value": critical,
        "ci95_method": "student_t_paired_independent_replicates",
        "paired_difference_ci_includes_zero": float(ci_low <= 0.0 <= ci_high),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-factorization", type=Path, required=True)
    parser.add_argument("--heldout-factorization", type=Path, required=True)
    parser.add_argument("--maximum-msd-relative-error", type=float, default=0.1)
    parser.add_argument("--maximum-ngp-absolute-error", type=float, default=0.3)
    parser.add_argument("--maximum-fs-absolute-error", type=float, default=0.03)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    calibration = {
        (int(float(row["replicate"])), int(float(row["lag"]))): row
        for row in read_rows(args.calibration_factorization)
    }
    heldout = {
        (int(float(row["replicate"])), int(float(row["lag"]))): row
        for row in read_rows(args.heldout_factorization)
    }
    if calibration.keys() != heldout.keys():
        raise ValueError("calibration and held-out factorization rows must share keys")
    fs_keys = [
        key.removeprefix("factorized_")
        for key in next(iter(calibration.values()))
        if key.startswith("factorized_fs_k")
    ]
    rows: list[dict[str, object]] = []
    for replicate, lag in sorted(calibration):
        predicted = calibration[(replicate, lag)]
        observed = heldout[(replicate, lag)]
        row: dict[str, object] = {
            "replicate": float(replicate),
            "temperature": float(predicted["temperature"]),
            "lag": float(lag),
            "predicted_msd": float(predicted["factorized_msd"]),
            "observed_msd": float(observed["observed_msd"]),
            "msd_relative_error": abs(
                float(predicted["factorized_msd"]) / float(observed["observed_msd"]) - 1.0
            ),
            "msd_log_error": math.log(
                float(predicted["factorized_msd"]) / float(observed["observed_msd"])
            ),
            "predicted_ngp": float(predicted["factorized_ngp"]),
            "observed_ngp": float(observed["observed_ngp"]),
            "ngp_absolute_error": abs(
                float(predicted["factorized_ngp"]) - float(observed["observed_ngp"])
            ),
            "ngp_signed_error": float(predicted["factorized_ngp"])
            - float(observed["observed_ngp"]),
            "calibration_only_microchannel_input": 1.0,
            "heldout_events_used_in_prediction": 0.0,
            "macro_fit_parameter_count": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for fs_key in fs_keys:
            predicted_fs = float(predicted[f"factorized_{fs_key}"])
            observed_fs = float(observed[f"observed_{fs_key}"])
            row[f"predicted_{fs_key}"] = predicted_fs
            row[f"observed_{fs_key}"] = observed_fs
            row[f"absolute_error_{fs_key}"] = abs(predicted_fs - observed_fs)
            row[f"signed_error_{fs_key}"] = predicted_fs - observed_fs
        rows.append(row)

    summary_rows: list[dict[str, object]] = []
    for lag in sorted({float(row["lag"]) for row in rows}):
        selected = [row for row in rows if float(row["lag"]) == lag]
        predicted_msd_mean = float(np.mean([float(row["predicted_msd"]) for row in selected]))
        observed_msd_mean = float(np.mean([float(row["observed_msd"]) for row in selected]))
        summary: dict[str, object] = {
            "lag": lag,
            "independent_replicate_count": float(len(selected)),
            "predicted_msd_mean": predicted_msd_mean,
            "observed_msd_mean": observed_msd_mean,
            "ensemble_msd_relative_error": abs(predicted_msd_mean / observed_msd_mean - 1.0),
            "predicted_ngp_mean": float(
                np.mean([float(row["predicted_ngp"]) for row in selected])
            ),
            "observed_ngp_mean": float(
                np.mean([float(row["observed_ngp"]) for row in selected])
            ),
        }
        summary["ensemble_ngp_absolute_error"] = abs(
            float(summary["predicted_ngp_mean"]) - float(summary["observed_ngp_mean"])
        )
        msd_interval = paired_interval(
            np.array([float(row["msd_log_error"]) for row in selected])
        )
        ngp_interval = paired_interval(
            np.array([float(row["ngp_signed_error"]) for row in selected])
        )
        for key, value in msd_interval.items():
            summary[f"msd_{key}"] = value
        for key, value in ngp_interval.items():
            summary[f"ngp_{key}"] = value
        for fs_key in fs_keys:
            predicted_mean = float(
                np.mean([float(row[f"predicted_{fs_key}"]) for row in selected])
            )
            observed_mean = float(
                np.mean([float(row[f"observed_{fs_key}"]) for row in selected])
            )
            summary[f"predicted_{fs_key}_mean"] = predicted_mean
            summary[f"observed_{fs_key}_mean"] = observed_mean
            summary[f"ensemble_absolute_error_{fs_key}"] = abs(predicted_mean - observed_mean)
            interval = paired_interval(
                np.array([float(row[f"signed_error_{fs_key}"]) for row in selected])
            )
            for key, value in interval.items():
                summary[f"{fs_key}_{key}"] = value
        summary_rows.append(summary)

    max_ensemble_msd = max(float(row["ensemble_msd_relative_error"]) for row in summary_rows)
    max_ensemble_ngp = max(float(row["ensemble_ngp_absolute_error"]) for row in summary_rows)
    max_ensemble_fs = max(
        float(row[f"ensemble_absolute_error_{fs_key}"])
        for row in summary_rows
        for fs_key in fs_keys
    )
    retrospective_pass = (
        max_ensemble_msd <= args.maximum_msd_relative_error
        and max_ensemble_ngp <= args.maximum_ngp_absolute_error
        and max_ensemble_fs <= args.maximum_fs_absolute_error
    )
    verdict = {
        "temperature": rows[0]["temperature"],
        "independent_replicate_count": float(len({float(row["replicate"]) for row in rows})),
        "lag_count": float(len(summary_rows)),
        "wave_number_count": float(len(fs_keys)),
        "maximum_ensemble_msd_relative_error": max_ensemble_msd,
        "maximum_ensemble_ngp_absolute_error": max_ensemble_ngp,
        "maximum_ensemble_fs_absolute_error": max_ensemble_fs,
        "maximum_individual_msd_relative_error": max(
            float(row["msd_relative_error"]) for row in rows
        ),
        "maximum_individual_ngp_absolute_error": max(
            float(row["ngp_absolute_error"]) for row in rows
        ),
        "msd_tolerance": args.maximum_msd_relative_error,
        "ngp_tolerance": args.maximum_ngp_absolute_error,
        "fs_tolerance": args.maximum_fs_absolute_error,
        "retrospective_ensemble_transfer_candidate_pass": float(retrospective_pass),
        "individual_trajectory_forecast_pass": 0.0,
        "early_ngp_significant_mismatch_lag_count": float(
            sum(
                float(row["ngp_paired_difference_ci_includes_zero"]) == 0.0
                for row in summary_rows
            )
        ),
        "calibration_only_microchannel_input": 1.0,
        "heldout_events_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "preregistered_heldout_prediction_claim_allowed": 0.0,
        "next_required_test": "new_independent_trajectory_preregistered_channel_transfer",
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_rows.csv"), rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
