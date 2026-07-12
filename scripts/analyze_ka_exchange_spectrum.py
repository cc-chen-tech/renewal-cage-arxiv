#!/usr/bin/env python3
"""Fit calibration-only finite exchange spectra and score held-out identity decay."""

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
    exponential_correlation_spectrum,
    fit_exponential_correlation_spectrum,
    independent_sample_ci95,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_spectrum_transfer(row: dict[str, object]) -> dict[str, float | str]:
    """Gate finite-spectrum selection and calibration-to-heldout transfer."""

    selected = float(row["calibration_bic_gain_two_over_one"]) >= 6.0
    finite = float(row["slow_time"]) < float(row["maximum_candidate_time"])
    checks = [
        ("calibration_model_selection", selected),
        ("finite_slow_mode", finite),
        ("identity_curve", float(row["two_mode_heldout_rmse"]) <= 0.05),
        (
            "no_heldout_improvement",
            float(row["two_mode_heldout_rmse"])
            <= float(row["single_mode_heldout_rmse"]),
        ),
        ("late_identity_decay", float(row["two_mode_late_absolute_error"]) <= 0.05),
    ]
    failure = next((name for name, passed in checks if not passed), "none")
    return {
        "minimum_bic_gain": 6.0,
        "maximum_heldout_rmse": 0.05,
        "maximum_late_absolute_error": 0.05,
        "two_mode_calibration_selected": float(selected and finite),
        "two_mode_heldout_transfer_pass": float(failure == "none"),
        "primary_failure": failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-curve", type=Path, required=True)
    parser.add_argument("--heldout-curve", type=Path, required=True)
    parser.add_argument("--calibration-horizon", type=float, required=True)
    parser.add_argument("--scoring-horizon", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.calibration_horizon < args.scoring_horizon:
        raise ValueError("calibration horizon must be positive and shorter than scoring horizon")

    calibration_rows = read_rows(args.calibration_curve)
    heldout_rows = read_rows(args.heldout_curve)
    candidate_blocks = sorted(
        {float(row["block_size"]) for row in calibration_rows}
        & {float(row["block_size"]) for row in heldout_rows}
    )
    common_replicates = sorted(
        {int(float(row["replicate"])) for row in calibration_rows}
        & {int(float(row["replicate"])) for row in heldout_rows}
    )
    temperature_values = {float(row["temperature"]) for row in heldout_rows}
    if len(temperature_values) != 1 or not candidate_blocks or not common_replicates:
        raise ValueError("curves must define one temperature and common blocks and replicates")
    temperature = temperature_values.pop()
    common_blocks = []
    excluded_blocks = []
    for block_size in candidate_blocks:
        minimum_points = min(
            sum(
                int(float(row["replicate"])) == replicate
                and float(row["block_size"]) == block_size
                and float(row["lag_time"]) <= args.calibration_horizon
                for row in calibration_rows
            )
            for replicate in common_replicates
        )
        if minimum_points >= 5:
            common_blocks.append(block_size)
        else:
            excluded_blocks.append(block_size)
    if not common_blocks:
        raise ValueError("no block size has enough calibration points for a two-mode spectrum")
    curve_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    for replicate in common_replicates:
        for block_size in common_blocks:
            calibration = sorted(
                [
                    row
                    for row in calibration_rows
                    if int(float(row["replicate"])) == replicate
                    and float(row["block_size"]) == block_size
                    and float(row["lag_time"]) <= args.calibration_horizon
                ],
                key=lambda row: float(row["lag_time"]),
            )
            heldout = sorted(
                [
                    row
                    for row in heldout_rows
                    if int(float(row["replicate"])) == replicate
                    and float(row["block_size"]) == block_size
                    and float(row["lag_time"]) <= args.scoring_horizon
                ],
                key=lambda row: float(row["lag_time"]),
            )
            if len(calibration) < 3 or len(heldout) < 3:
                raise ValueError("each replica-block pair requires at least three curve points")
            calibration_time = np.array([float(row["lag_time"]) for row in calibration])
            calibration_correlation = np.array(
                [float(row["particle_identity_correlation"]) for row in calibration]
            )
            heldout_time = np.array([float(row["lag_time"]) for row in heldout])
            heldout_correlation = np.array(
                [float(row["observed_identity_correlation"]) for row in heldout]
            )
            single = fit_exponential_correlation_spectrum(
                calibration_time,
                calibration_correlation,
                component_count=1,
            )
            broad = fit_exponential_correlation_spectrum(
                calibration_time,
                calibration_correlation,
                component_count=2,
            )
            single_prediction = exponential_correlation_spectrum(heldout_time, single)
            broad_prediction = exponential_correlation_spectrum(heldout_time, broad)
            hmm_prediction = np.array(
                [float(row["hmm_identity_correlation"]) for row in heldout]
            )
            static_prediction = np.array(
                [float(row["static_identity_correlation"]) for row in heldout]
            )
            poisson_prediction = np.zeros_like(heldout_correlation)

            def rmse(prediction: np.ndarray) -> float:
                return float(np.sqrt(np.mean((heldout_correlation - prediction) ** 2)))

            row: dict[str, object] = {
                "replicate": float(replicate),
                "temperature": temperature,
                "block_size": block_size,
                "calibration_horizon": float(args.calibration_horizon),
                "scoring_horizon": float(args.scoring_horizon),
                "calibration_point_count": float(len(calibration)),
                "heldout_point_count": float(len(heldout)),
                "calibration_bic_single": single["bic"],
                "calibration_bic_two_mode": broad["bic"],
                "calibration_bic_gain_two_over_one": single["bic"] - broad["bic"],
                "fast_amplitude": broad["fast_amplitude"],
                "fast_time": broad["fast_time"],
                "slow_amplitude": broad["slow_amplitude"],
                "slow_time": broad["slow_time"],
                "slow_amplitude_fraction": broad["slow_amplitude"] / broad["total_amplitude"],
                "time_scale_ratio": broad["time_scale_ratio"],
                "maximum_candidate_time": broad["maximum_candidate_time"],
                "single_mode_heldout_rmse": rmse(single_prediction),
                "two_mode_heldout_rmse": rmse(broad_prediction),
                "hmm_heldout_rmse": rmse(hmm_prediction),
                "static_heldout_rmse": rmse(static_prediction),
                "poisson_heldout_rmse": rmse(poisson_prediction),
                "two_mode_late_signed_error": float(
                    heldout_correlation[-1] - broad_prediction[-1]
                ),
                "two_mode_late_absolute_error": float(
                    abs(heldout_correlation[-1] - broad_prediction[-1])
                ),
                "finite_recovery_enforced": 1.0,
                "thermodynamic_claim_allowed": 0.0,
            }
            row.update(classify_spectrum_transfer(row))
            replicate_rows.append(row)
            for index, heldout_row in enumerate(heldout):
                curve_rows.append(
                    {
                        "replicate": float(replicate),
                        "temperature": temperature,
                        "block_size": block_size,
                        "block_lag": heldout_row["block_lag"],
                        "lag_time": heldout_time[index],
                        "heldout_identity_correlation": heldout_correlation[index],
                        "single_mode_prediction": single_prediction[index],
                        "two_mode_prediction": broad_prediction[index],
                        "hmm_prediction": hmm_prediction[index],
                        "static_prediction": static_prediction[index],
                        "poisson_prediction": poisson_prediction[index],
                    }
                )

    block_rows: list[dict[str, object]] = []
    for block_size in common_blocks:
        selected = [row for row in replicate_rows if float(row["block_size"]) == block_size]
        late = np.array([float(row["two_mode_late_signed_error"]) for row in selected])
        late_se = float(np.std(late, ddof=1) / math.sqrt(len(late)))
        late_low, late_high, critical = independent_sample_ci95(
            mean=float(np.mean(late)),
            standard_error=late_se,
            sample_count=len(late),
        )
        block_rows.append(
            {
                "block_size": block_size,
                "independent_replicate_count": float(len(selected)),
                "two_mode_selection_fraction": float(
                    np.mean([float(row["two_mode_calibration_selected"]) for row in selected])
                ),
                "two_mode_transfer_pass_fraction": float(
                    np.mean([float(row["two_mode_heldout_transfer_pass"]) for row in selected])
                ),
                "mean_bic_gain_two_over_one": float(
                    np.mean([float(row["calibration_bic_gain_two_over_one"]) for row in selected])
                ),
                "mean_time_scale_ratio": float(
                    np.mean([float(row["time_scale_ratio"]) for row in selected])
                ),
                "mean_slow_amplitude_fraction": float(
                    np.mean([float(row["slow_amplitude_fraction"]) for row in selected])
                ),
                "mean_two_mode_heldout_rmse": float(
                    np.mean([float(row["two_mode_heldout_rmse"]) for row in selected])
                ),
                "mean_single_mode_heldout_rmse": float(
                    np.mean([float(row["single_mode_heldout_rmse"]) for row in selected])
                ),
                "mean_late_signed_error": float(np.mean(late)),
                "ci95_low_late_signed_error": late_low,
                "ci95_high_late_signed_error": late_high,
                "ci95_critical_value": critical,
            }
        )
    selection_fraction = float(
        np.mean([float(row["two_mode_calibration_selected"]) for row in replicate_rows])
    )
    transfer_fraction = float(
        np.mean([float(row["two_mode_heldout_transfer_pass"]) for row in replicate_rows])
    )
    if selection_fraction == 1.0 and transfer_fraction == 1.0:
        outcome = "two_mode_finite_exchange_spectrum_closure"
    elif selection_fraction == 1.0 and transfer_fraction >= 0.8:
        outcome = "two_mode_finite_exchange_spectrum_near_closure"
    elif selection_fraction <= 0.2:
        outcome = "single_mode_exchange_sufficient"
    else:
        outcome = "exchange_spectrum_unresolved"
    verdict = {
        "temperature": temperature,
        "calibration_horizon": float(args.calibration_horizon),
        "scoring_horizon": float(args.scoring_horizon),
        "independent_replicate_count": float(len(common_replicates)),
        "common_block_sizes": ";".join(f"{value:g}" for value in common_blocks),
        "excluded_underidentified_block_sizes": (
            ";".join(f"{value:g}" for value in excluded_blocks) if excluded_blocks else "none"
        ),
        "full_resolution_scope": float(not excluded_blocks),
        "replica_block_count": float(len(replicate_rows)),
        "two_mode_calibration_selection_fraction": selection_fraction,
        "two_mode_heldout_transfer_pass_fraction": transfer_fraction,
        "event_level_outcome": outcome,
        "finite_exchange_spectrum_claim_allowed": float(
            selection_fraction == 1.0 and transfer_fraction >= 0.8
        ),
        "identity_spectrum_closure_claim_allowed": float(
            selection_fraction == 1.0 and transfer_fraction == 1.0
        ),
        "full_event_clock_closure_claim_allowed": 0.0,
        "heldout_macro_prediction_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "source_dois": (
            "10.1103/PhysRevE.67.030501;10.1063/1.2001629;"
            "10.1063/1.2803062;10.1103/PhysRevX.1.021013"
        ),
        "source_alignment": "structured_exchange_times_and_hierarchical_relaxation",
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicate_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_blocks.csv"), block_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
