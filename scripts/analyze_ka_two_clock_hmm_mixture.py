#!/usr/bin/env python3
"""Test HMM emissions combined with two finite exchange-rate classes."""

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
    two_clock_hmm_mixture_count_predictions,
    two_clock_hmm_mixture_pair_pmf,
    two_clock_hmm_mixture_parameters,
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


def classify_hybrid_transfer(row: dict[str, object]) -> dict[str, float | str]:
    """Separate stationary shape closure from absolute held-out rate drift."""

    shape_checks = [
        float(row["heldout_fano_relative_error"]) <= 0.20,
        float(row["heldout_count_tv_distance"]) <= 0.03,
        float(row["identity_rmse"]) <= 0.05,
        float(row["late_identity_absolute_error"]) <= 0.05,
        float(row["pair_distribution_pass"]) == 1.0,
    ]
    shape_pass = all(shape_checks)
    mean_pass = float(row["heldout_mean_relative_error"]) <= 0.10
    if not shape_pass:
        failure = "shape_or_distribution"
    elif not mean_pass:
        failure = "mean_count_drift"
    else:
        failure = "none"
    return {
        "maximum_mean_relative_error": 0.10,
        "maximum_fano_relative_error": 0.20,
        "maximum_count_tv_distance": 0.03,
        "maximum_identity_rmse": 0.05,
        "maximum_late_identity_absolute_error": 0.05,
        "conditional_shape_distribution_pass": float(shape_pass),
        "full_hybrid_transfer_pass": float(shape_pass and mean_pass),
        "primary_failure": failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectrum-replicates", type=Path, required=True)
    parser.add_argument("--hmm-replicates", type=Path, required=True)
    parser.add_argument("--heldout-curve", type=Path, required=True)
    parser.add_argument("--count-histogram", type=Path, required=True)
    parser.add_argument("--count-pair-histogram", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    spectrum_rows = read_rows(args.spectrum_replicates)
    hmm_rows = read_rows(args.hmm_replicates)
    curve_input = read_rows(args.heldout_curve)
    histogram_input = read_rows(args.count_histogram)
    pair_input = read_rows(args.count_pair_histogram)
    temperature_values = {float(row["temperature"]) for row in spectrum_rows}
    if len(temperature_values) != 1:
        raise ValueError("spectrum rows must contain one temperature")
    temperature = temperature_values.pop()
    curve_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    for spectrum in spectrum_rows:
        replicate = int(float(spectrum["replicate"]))
        block_size = float(spectrum["block_size"])
        hmm = next(
            row
            for row in hmm_rows
            if int(float(row["replicate"])) == replicate
            and float(row["block_size"]) == block_size
        )
        parameters = two_clock_hmm_mixture_parameters(
            hmm,
            spectrum,
            block_size=block_size,
        )
        heldout_curve = sorted(
            [
                row
                for row in curve_input
                if int(float(row["replicate"])) == replicate
                and float(row["block_size"]) == block_size
            ],
            key=lambda row: float(row["block_lag"]),
        )
        predictions = two_clock_hmm_mixture_count_predictions(
            parameters,
            maximum_lag=len(heldout_curve),
        )
        observed_identity = np.array(
            [float(row["observed_identity_correlation"]) for row in heldout_curve]
        )
        predicted_identity = np.array(
            [float(row["predicted_identity_correlation"]) for row in predictions]
        )
        heldout_histogram = sorted(
            [
                row
                for row in histogram_input
                if int(float(row["replicate"])) == replicate
                and float(row["block_size"]) == block_size
                and row["window"] == "heldout"
            ],
            key=lambda row: float(row["count_value"]),
        )
        maximum_count = int(max(float(row["count_value"]) for row in heldout_histogram))
        observed_pmf = np.zeros(maximum_count + 1, dtype=float)
        for row in heldout_histogram:
            observed_pmf[int(float(row["count_value"]))] = float(row["probability"])

        def poisson(mean: float) -> np.ndarray:
            values = np.zeros(maximum_count + 1, dtype=float)
            values[0] = math.exp(-mean)
            for count in range(1, maximum_count + 1):
                values[count] = values[count - 1] * mean / count
            return values

        predicted_pmf = (
            float(parameters["stationary_slow_probability"])
            * poisson(float(parameters["slow_mean_count"]))
            + float(parameters["stationary_fast_probability"])
            * poisson(float(parameters["fast_mean_count"]))
        )
        marginal_tail = max(0.0, 1.0 - float(np.sum(predicted_pmf)))
        marginal_tv = 0.5 * (
            float(np.sum(np.abs(observed_pmf - predicted_pmf))) + marginal_tail
        )
        pair_histograms = [
            row
            for row in pair_input
            if int(float(row["replicate"])) == replicate
            and float(row["block_size"]) == block_size
        ]
        maximum_pair_count = max(
            max(int(float(row["first_count"])), int(float(row["second_count"])))
            for row in pair_histograms
        )

        def observed_pair(window: str) -> np.ndarray:
            values = np.zeros(
                (maximum_pair_count + 1, maximum_pair_count + 1),
                dtype=float,
            )
            for row in pair_histograms:
                if row["window"] != window:
                    continue
                values[
                    int(float(row["first_count"])),
                    int(float(row["second_count"])),
                ] = float(row["probability"])
            return values

        calibration_pair = observed_pair("calibration")
        heldout_pair = observed_pair("heldout")
        predicted_pair = two_clock_hmm_mixture_pair_pmf(
            parameters,
            maximum_count=maximum_pair_count,
            block_lag=1,
        )
        empirical_pair_tv = 0.5 * float(np.sum(np.abs(calibration_pair - heldout_pair)))
        pair_tail = max(0.0, 1.0 - float(np.sum(predicted_pair)))
        pair_tv = 0.5 * (float(np.sum(np.abs(heldout_pair - predicted_pair))) + pair_tail)
        pair_tolerance = max(0.03, 1.5 * empirical_pair_tv)
        heldout_mean = float(hmm["heldout_mean_count"])
        heldout_fano = float(hmm["heldout_fano_factor"])
        identity_rmse = float(
            np.sqrt(np.mean((observed_identity - predicted_identity) ** 2))
        )
        row: dict[str, object] = {
            "replicate": float(replicate),
            "temperature": temperature,
            "block_size": block_size,
            "calibration_horizon": spectrum["calibration_horizon"],
            "scoring_horizon": spectrum["scoring_horizon"],
            "fast_clock_weight": parameters["fast_clock_weight"],
            "fast_clock_time": parameters["fast_clock_time"],
            "slow_clock_time": parameters["slow_clock_time"],
            "time_scale_ratio": float(parameters["slow_clock_time"])
            / float(parameters["fast_clock_time"]),
            "calibration_bic_gain_two_over_one": spectrum[
                "calibration_bic_gain_two_over_one"
            ],
            "second_clock_calibration_selected": float(
                float(spectrum["calibration_bic_gain_two_over_one"]) >= 6.0
                and float(spectrum["slow_time"])
                < float(spectrum["maximum_candidate_time"])
            ),
            "predicted_mean_count": parameters["mean_count"],
            "heldout_mean_count": heldout_mean,
            "heldout_mean_relative_error": abs(
                float(parameters["mean_count"]) / heldout_mean - 1.0
            ),
            "predicted_fano_factor": parameters["fano_factor"],
            "heldout_fano_factor": heldout_fano,
            "heldout_fano_relative_error": abs(
                float(parameters["fano_factor"]) / heldout_fano - 1.0
            ),
            "heldout_count_tv_distance": marginal_tv,
            "identity_rmse": identity_rmse,
            "late_identity_signed_error": float(
                observed_identity[-1] - predicted_identity[-1]
            ),
            "late_identity_absolute_error": float(
                abs(observed_identity[-1] - predicted_identity[-1])
            ),
            "empirical_split_half_pair_tv_distance": empirical_pair_tv,
            "pair_tv_tolerance": pair_tolerance,
            "pair_tv_distance": pair_tv,
            "pair_distribution_pass": float(pair_tv <= pair_tolerance),
            "positive_poisson_emissions": 1.0,
            "finite_exchange_recovery": 1.0,
            "static_rate_disorder": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        row.update(classify_hybrid_transfer(row))
        replicate_rows.append(row)
        for observed_row, predicted_row in zip(heldout_curve, predictions):
            curve_rows.append(
                {
                    "replicate": float(replicate),
                    "temperature": temperature,
                    "block_size": block_size,
                    "block_lag": observed_row["block_lag"],
                    "lag_time": observed_row["lag_time"],
                    "heldout_identity_correlation": observed_row[
                        "observed_identity_correlation"
                    ],
                    "hybrid_prediction": predicted_row[
                        "predicted_identity_correlation"
                    ],
                    "markov_hmm_prediction": observed_row["hmm_identity_correlation"],
                }
            )
        for count_value in range(maximum_count + 1):
            histogram_rows.append(
                {
                    "replicate": float(replicate),
                    "temperature": temperature,
                    "block_size": block_size,
                    "count_value": float(count_value),
                    "heldout_probability": observed_pmf[count_value],
                    "hybrid_probability": predicted_pmf[count_value],
                }
            )
        for first in range(maximum_pair_count + 1):
            for second in range(maximum_pair_count + 1):
                pair_rows.append(
                    {
                        "replicate": float(replicate),
                        "temperature": temperature,
                        "block_size": block_size,
                        "first_count": float(first),
                        "second_count": float(second),
                        "heldout_probability": heldout_pair[first, second],
                        "hybrid_probability": predicted_pair[first, second],
                    }
                )

    full_fraction = float(
        np.mean([float(row["full_hybrid_transfer_pass"]) for row in replicate_rows])
    )
    shape_fraction = float(
        np.mean(
            [float(row["conditional_shape_distribution_pass"]) for row in replicate_rows]
        )
    )
    selection_fraction = float(
        np.mean(
            [float(row["second_clock_calibration_selected"]) for row in replicate_rows]
        )
    )
    if shape_fraction == 1.0 and full_fraction == 1.0:
        outcome = "two_clock_hmm_full_event_shape_and_rate_closure"
    elif shape_fraction == 1.0:
        outcome = "two_clock_hmm_shape_closure_rate_drift_unresolved"
    else:
        outcome = "two_clock_hmm_hybrid_rejected"
    verdict = {
        "temperature": temperature,
        "calibration_horizon": spectrum_rows[0]["calibration_horizon"],
        "scoring_horizon": spectrum_rows[0]["scoring_horizon"],
        "independent_replicate_count": float(
            len({float(row["replicate"]) for row in replicate_rows})
        ),
        "common_block_sizes": ";".join(
            f"{value:g}" for value in sorted({float(row["block_size"]) for row in replicate_rows})
        ),
        "replica_block_count": float(len(replicate_rows)),
        "second_clock_calibration_selection_fraction": selection_fraction,
        "conditional_shape_distribution_pass_fraction": shape_fraction,
        "full_hybrid_transfer_pass_fraction": full_fraction,
        "maximum_mean_relative_error": max(
            float(row["heldout_mean_relative_error"]) for row in replicate_rows
        ),
        "maximum_fano_relative_error": max(
            float(row["heldout_fano_relative_error"]) for row in replicate_rows
        ),
        "maximum_count_tv_distance": max(
            float(row["heldout_count_tv_distance"]) for row in replicate_rows
        ),
        "maximum_identity_rmse": max(
            float(row["identity_rmse"]) for row in replicate_rows
        ),
        "maximum_pair_tv_distance": max(
            float(row["pair_tv_distance"]) for row in replicate_rows
        ),
        "event_level_outcome": outcome,
        "conditional_event_shape_closure_claim_allowed": float(shape_fraction == 1.0),
        "absolute_rate_transfer_claim_allowed": float(full_fraction == 1.0),
        "macro_observable_prediction_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicate_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_histogram.csv"), histogram_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_pair_histogram.csv"), pair_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
