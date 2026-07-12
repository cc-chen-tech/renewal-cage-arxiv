#!/usr/bin/env python3
"""Test a calibration-only two-clock gamma-refresh Cox event generator."""

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
    fit_anchored_exponential_correlation_spectrum,
    gamma_refresh_cox_count_pmf,
    gamma_refresh_cox_count_predictions,
    gamma_refresh_cox_pair_pmf,
    gamma_refresh_cox_parameters,
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


def classify_cox_transfer(row: dict[str, object]) -> dict[str, float | str]:
    """Gate calibration model selection and held-out count-moment transfer."""

    checks = [
        ("calibration_model_selection", float(row["calibration_bic_gain_two_over_one"]) >= 6.0),
        ("mean_count", float(row["heldout_mean_relative_error"]) <= 0.10),
        ("count_fano", float(row["heldout_fano_relative_error"]) <= 0.20),
        ("count_distribution", float(row["heldout_count_tv_distance"]) <= 0.03),
        ("identity_curve", float(row["two_clock_identity_rmse"]) <= 0.05),
        (
            "no_identity_improvement",
            float(row["two_clock_identity_rmse"])
            <= float(row["single_clock_identity_rmse"]),
        ),
        ("late_identity_decay", float(row["two_clock_late_absolute_error"]) <= 0.05),
        (
            "finite_slow_clock",
            float(row["slow_time"]) < float(row["maximum_candidate_time"]),
        ),
    ]
    failure = next((name for name, passed in checks if not passed), "none")
    return {
        "minimum_bic_gain": 6.0,
        "maximum_mean_relative_error": 0.10,
        "maximum_fano_relative_error": 0.20,
        "maximum_count_tv_distance": 0.03,
        "maximum_identity_rmse": 0.05,
        "maximum_late_absolute_error": 0.05,
        "two_clock_cox_transfer_pass": float(failure == "none"),
        "primary_failure": failure,
    }


def classify_pair_transfer(
    *,
    gamma_pair_tv: float,
    hmm_pair_tv: float,
    empirical_split_half_tv: float,
) -> dict[str, float]:
    """Resolve pair-distribution predictions against empirical split-half resolution."""

    tolerance = max(0.03, 1.5 * empirical_split_half_tv)
    return {
        "pair_tv_tolerance": tolerance,
        "gamma_pair_distribution_pass": float(gamma_pair_tv <= tolerance),
        "hmm_pair_distribution_pass": float(hmm_pair_tv <= tolerance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-curve", type=Path, required=True)
    parser.add_argument("--hmm-replicates", type=Path, required=True)
    parser.add_argument("--heldout-curve", type=Path, required=True)
    parser.add_argument("--count-histogram", type=Path, required=True)
    parser.add_argument("--count-pair-histogram", type=Path, required=True)
    parser.add_argument("--calibration-horizon", type=float, required=True)
    parser.add_argument("--scoring-horizon", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.calibration_horizon < args.scoring_horizon:
        raise ValueError("calibration horizon must be positive and shorter than scoring horizon")

    calibration_rows = read_rows(args.calibration_curve)
    moment_rows = read_rows(args.hmm_replicates)
    heldout_rows = read_rows(args.heldout_curve)
    histogram_input_rows = read_rows(args.count_histogram)
    pair_histogram_input_rows = read_rows(args.count_pair_histogram)
    candidate_blocks = sorted(
        {float(row["block_size"]) for row in calibration_rows}
        & {float(row["block_size"]) for row in moment_rows}
        & {float(row["block_size"]) for row in heldout_rows}
    )
    common_replicates = sorted(
        {int(float(row["replicate"])) for row in calibration_rows}
        & {int(float(row["replicate"])) for row in moment_rows}
        & {int(float(row["replicate"])) for row in heldout_rows}
    )
    temperature_values = {float(row["temperature"]) for row in heldout_rows}
    if len(temperature_values) != 1 or not candidate_blocks or not common_replicates:
        raise ValueError("inputs must define one temperature and common blocks and replicates")
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
        (common_blocks if minimum_points >= 5 else excluded_blocks).append(block_size)
    if not common_blocks:
        raise ValueError("no block size identifies a two-clock spectrum")

    curve_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    pair_histogram_rows: list[dict[str, object]] = []
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
            moments = next(
                row
                for row in moment_rows
                if int(float(row["replicate"])) == replicate
                and float(row["block_size"]) == block_size
            )
            calibration_mean = float(moments["calibration_mean_count"])
            calibration_fano = float(moments["calibration_fano_factor"])
            heldout_mean = float(moments["heldout_mean_count"])
            heldout_fano = float(moments["heldout_fano_factor"])
            zero_lag_amplitude = 1.0 - 1.0 / calibration_fano
            calibration_time = np.array([float(row["lag_time"]) for row in calibration])
            calibration_correlation = np.array(
                [float(row["particle_identity_correlation"]) for row in calibration]
            )
            heldout_time = np.array([float(row["lag_time"]) for row in heldout])
            heldout_correlation = np.array(
                [float(row["observed_identity_correlation"]) for row in heldout]
            )
            single = fit_anchored_exponential_correlation_spectrum(
                calibration_time,
                calibration_correlation,
                total_amplitude=zero_lag_amplitude,
                component_count=1,
            )
            broad = fit_anchored_exponential_correlation_spectrum(
                calibration_time,
                calibration_correlation,
                total_amplitude=zero_lag_amplitude,
                component_count=2,
            )
            parameters = gamma_refresh_cox_parameters(
                mean_count=calibration_mean,
                fano_factor=calibration_fano,
                block_size=block_size,
                fitted_spectrum=broad,
            )
            predictions = gamma_refresh_cox_count_predictions(
                parameters,
                maximum_lag=len(heldout),
            )
            two_clock_prediction = np.array(
                [float(row["predicted_identity_correlation"]) for row in predictions]
            )
            single_clock_prediction = exponential_correlation_spectrum(heldout_time, single)
            hmm_prediction = np.array(
                [float(row["hmm_identity_correlation"]) for row in heldout]
            )
            static_prediction = np.array(
                [float(row["static_identity_correlation"]) for row in heldout]
            )
            poisson_prediction = np.zeros_like(heldout_correlation)

            heldout_histogram = sorted(
                [
                    row
                    for row in histogram_input_rows
                    if int(float(row["replicate"])) == replicate
                    and float(row["block_size"]) == block_size
                    and row["window"] == "heldout"
                ],
                key=lambda row: float(row["count_value"]),
            )
            maximum_count = int(max(float(row["count_value"]) for row in heldout_histogram))
            observed_pmf = np.zeros(maximum_count + 1, dtype=float)
            frequencies = np.zeros(maximum_count + 1, dtype=float)
            for histogram_row in heldout_histogram:
                count_value = int(float(histogram_row["count_value"]))
                observed_pmf[count_value] = float(histogram_row["probability"])
                frequencies[count_value] = float(histogram_row["frequency"])
            two_clock_pmf = gamma_refresh_cox_count_pmf(
                parameters,
                maximum_count=maximum_count,
            )
            single_parameters = gamma_refresh_cox_parameters(
                mean_count=calibration_mean,
                fano_factor=calibration_fano,
                block_size=block_size,
                fitted_spectrum=single,
            )
            single_clock_pmf = gamma_refresh_cox_count_pmf(
                single_parameters,
                maximum_count=maximum_count,
            )

            def poisson_pmf(mean: float, limit: int = maximum_count) -> np.ndarray:
                values = np.zeros(limit + 1, dtype=float)
                values[0] = math.exp(-mean)
                for count in range(1, limit + 1):
                    values[count] = values[count - 1] * mean / count
                return values

            poisson_count_pmf = poisson_pmf(calibration_mean)
            hmm_count_pmf = (
                float(moments["stationary_slow_probability"])
                * poisson_pmf(float(moments["slow_mean_count"]))
                + float(moments["stationary_fast_probability"])
                * poisson_pmf(float(moments["fast_mean_count"]))
            )

            pair_histograms = [
                row
                for row in pair_histogram_input_rows
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
                for pair_row in pair_histograms:
                    if pair_row["window"] != window:
                        continue
                    values[
                        int(float(pair_row["first_count"])),
                        int(float(pair_row["second_count"])),
                    ] = float(pair_row["probability"])
                return values

            calibration_pair_pmf = observed_pair("calibration")
            heldout_pair_pmf = observed_pair("heldout")
            two_clock_pair_pmf = gamma_refresh_cox_pair_pmf(
                parameters,
                maximum_count=maximum_pair_count,
                block_lag=1,
            )
            single_clock_pair_pmf = gamma_refresh_cox_pair_pmf(
                single_parameters,
                maximum_count=maximum_pair_count,
                block_lag=1,
            )
            hmm_transition = np.array(
                [
                    [
                        1.0 - float(moments["slow_to_fast_probability"]),
                        float(moments["slow_to_fast_probability"]),
                    ],
                    [
                        float(moments["fast_to_slow_probability"]),
                        1.0 - float(moments["fast_to_slow_probability"]),
                    ],
                ]
            )
            hmm_stationary = np.array(
                [
                    float(moments["stationary_slow_probability"]),
                    float(moments["stationary_fast_probability"]),
                ]
            )
            hmm_emissions = [
                poisson_pmf(float(moments["slow_mean_count"]), maximum_pair_count),
                poisson_pmf(float(moments["fast_mean_count"]), maximum_pair_count),
            ]
            hmm_pair_pmf = sum(
                hmm_stationary[first_state]
                * hmm_transition[first_state, second_state]
                * np.outer(hmm_emissions[first_state], hmm_emissions[second_state])
                for first_state in range(2)
                for second_state in range(2)
            )
            poisson_pair_marginal = poisson_pmf(calibration_mean, maximum_pair_count)
            poisson_pair_pmf = np.outer(poisson_pair_marginal, poisson_pair_marginal)

            def total_variation(predicted: np.ndarray) -> float:
                tail = max(0.0, 1.0 - float(np.sum(predicted)))
                return 0.5 * (float(np.sum(np.abs(observed_pmf - predicted))) + tail)

            def mean_log_likelihood(predicted: np.ndarray) -> float:
                return float(
                    np.dot(frequencies, np.log(np.maximum(predicted, 1e-300)))
                    / np.sum(frequencies)
                )

            def pair_total_variation(predicted: np.ndarray, observed: np.ndarray) -> float:
                tail = max(0.0, 1.0 - float(np.sum(predicted)))
                return 0.5 * (float(np.sum(np.abs(observed - predicted))) + tail)

            empirical_pair_tv = 0.5 * float(
                np.sum(np.abs(calibration_pair_pmf - heldout_pair_pmf))
            )
            gamma_pair_tv = pair_total_variation(two_clock_pair_pmf, heldout_pair_pmf)
            hmm_pair_tv = pair_total_variation(hmm_pair_pmf, heldout_pair_pmf)
            single_pair_tv = pair_total_variation(
                single_clock_pair_pmf,
                heldout_pair_pmf,
            )
            poisson_pair_tv = pair_total_variation(poisson_pair_pmf, heldout_pair_pmf)
            pair_gate = classify_pair_transfer(
                gamma_pair_tv=gamma_pair_tv,
                hmm_pair_tv=hmm_pair_tv,
                empirical_split_half_tv=empirical_pair_tv,
            )

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
                "calibration_mean_count": calibration_mean,
                "heldout_mean_count": heldout_mean,
                "heldout_mean_relative_error": abs(calibration_mean / heldout_mean - 1.0),
                "calibration_fano_factor": calibration_fano,
                "heldout_fano_factor": heldout_fano,
                "heldout_fano_relative_error": abs(calibration_fano / heldout_fano - 1.0),
                "heldout_count_tv_distance": total_variation(two_clock_pmf),
                "single_clock_count_tv_distance": total_variation(single_clock_pmf),
                "hmm_count_tv_distance": total_variation(hmm_count_pmf),
                "poisson_count_tv_distance": total_variation(poisson_count_pmf),
                "two_clock_mean_log_likelihood": mean_log_likelihood(two_clock_pmf),
                "single_clock_mean_log_likelihood": mean_log_likelihood(single_clock_pmf),
                "hmm_mean_log_likelihood": mean_log_likelihood(hmm_count_pmf),
                "poisson_mean_log_likelihood": mean_log_likelihood(poisson_count_pmf),
                "empirical_split_half_pair_tv_distance": empirical_pair_tv,
                "two_clock_pair_tv_distance": gamma_pair_tv,
                "single_clock_pair_tv_distance": single_pair_tv,
                "hmm_pair_tv_distance": hmm_pair_tv,
                "poisson_pair_tv_distance": poisson_pair_tv,
                "zero_lag_amplitude_from_fano": zero_lag_amplitude,
                "calibration_bic_single_clock": single["bic"],
                "calibration_bic_two_clock": broad["bic"],
                "calibration_bic_gain_two_over_one": single["bic"] - broad["bic"],
                "fast_amplitude_fraction": broad["fast_amplitude_fraction"],
                "fast_time": broad["fast_time"],
                "slow_time": broad["slow_time"],
                "time_scale_ratio": broad["time_scale_ratio"],
                "maximum_candidate_time": broad["maximum_candidate_time"],
                "fast_gamma_shape": parameters["fast_gamma_shape"],
                "slow_gamma_shape": parameters["slow_gamma_shape"],
                "fast_retention_probability": parameters["fast_retention_probability"],
                "slow_retention_probability": parameters["slow_retention_probability"],
                "single_clock_identity_rmse": rmse(single_clock_prediction),
                "two_clock_identity_rmse": rmse(two_clock_prediction),
                "hmm_identity_rmse": rmse(hmm_prediction),
                "static_identity_rmse": rmse(static_prediction),
                "poisson_identity_rmse": rmse(poisson_prediction),
                "two_clock_late_signed_error": float(
                    heldout_correlation[-1] - two_clock_prediction[-1]
                ),
                "two_clock_late_absolute_error": float(
                    abs(heldout_correlation[-1] - two_clock_prediction[-1])
                ),
                "positive_intensity_generator": 1.0,
                "finite_recovery_enforced": 1.0,
                "thermodynamic_claim_allowed": 0.0,
            }
            row.update(classify_cox_transfer(row))
            row.update(pair_gate)
            row["conditional_moment_marginal_memory_pass"] = float(
                row["heldout_fano_relative_error"] <= 0.20
                and row["heldout_count_tv_distance"] <= 0.03
                and row["two_clock_identity_rmse"] <= 0.05
                and row["two_clock_identity_rmse"] <= row["single_clock_identity_rmse"]
                and row["two_clock_late_absolute_error"] <= 0.05
                and row["slow_time"] < row["maximum_candidate_time"]
            )
            single_pass = bool(
                row["heldout_mean_relative_error"] <= 0.10
                and row["heldout_fano_relative_error"] <= 0.20
                and row["single_clock_count_tv_distance"] <= 0.03
                and row["single_clock_identity_rmse"] <= 0.05
                and abs(heldout_correlation[-1] - single_clock_prediction[-1]) <= 0.05
            )
            row["single_clock_cox_transfer_pass"] = float(single_pass)
            replicate_rows.append(row)
            for count_value in range(maximum_count + 1):
                histogram_rows.append(
                    {
                        "replicate": float(replicate),
                        "temperature": temperature,
                        "block_size": block_size,
                        "count_value": float(count_value),
                        "heldout_probability": observed_pmf[count_value],
                        "single_clock_probability": single_clock_pmf[count_value],
                        "two_clock_probability": two_clock_pmf[count_value],
                        "hmm_probability": hmm_count_pmf[count_value],
                        "poisson_probability": poisson_count_pmf[count_value],
                    }
                )
            for first_count in range(maximum_pair_count + 1):
                for second_count in range(maximum_pair_count + 1):
                    pair_histogram_rows.append(
                        {
                            "replicate": float(replicate),
                            "temperature": temperature,
                            "block_size": block_size,
                            "first_count": float(first_count),
                            "second_count": float(second_count),
                            "heldout_probability": heldout_pair_pmf[first_count, second_count],
                            "two_clock_probability": two_clock_pair_pmf[first_count, second_count],
                            "single_clock_probability": single_clock_pair_pmf[first_count, second_count],
                            "hmm_probability": hmm_pair_pmf[first_count, second_count],
                            "poisson_probability": poisson_pair_pmf[first_count, second_count],
                        }
                    )
            for index, heldout_row in enumerate(heldout):
                curve_rows.append(
                    {
                        "replicate": float(replicate),
                        "temperature": temperature,
                        "block_size": block_size,
                        "block_lag": heldout_row["block_lag"],
                        "lag_time": heldout_time[index],
                        "heldout_identity_correlation": heldout_correlation[index],
                        "single_clock_prediction": single_clock_prediction[index],
                        "two_clock_prediction": two_clock_prediction[index],
                        "hmm_prediction": hmm_prediction[index],
                        "static_prediction": static_prediction[index],
                        "poisson_prediction": poisson_prediction[index],
                    }
                )

    block_rows: list[dict[str, object]] = []
    for block_size in common_blocks:
        selected = [row for row in replicate_rows if float(row["block_size"]) == block_size]
        fano_errors = np.array([float(row["heldout_fano_relative_error"]) for row in selected])
        identity_errors = np.array([float(row["two_clock_identity_rmse"]) for row in selected])
        mean_identity = float(np.mean(identity_errors))
        identity_se = float(np.std(identity_errors, ddof=1) / math.sqrt(len(identity_errors)))
        identity_low, identity_high, critical = independent_sample_ci95(
            mean=mean_identity,
            standard_error=identity_se,
            sample_count=len(identity_errors),
        )
        block_rows.append(
            {
                "block_size": block_size,
                "independent_replicate_count": float(len(selected)),
                "two_clock_selection_fraction": float(
                    np.mean(
                        [float(row["calibration_bic_gain_two_over_one"]) >= 6.0 for row in selected]
                    )
                ),
                "two_clock_transfer_pass_fraction": float(
                    np.mean([float(row["two_clock_cox_transfer_pass"]) for row in selected])
                ),
                "single_clock_transfer_pass_fraction": float(
                    np.mean([float(row["single_clock_cox_transfer_pass"]) for row in selected])
                ),
                "maximum_fano_relative_error": float(np.max(fano_errors)),
                "maximum_count_tv_distance": max(
                    float(row["heldout_count_tv_distance"]) for row in selected
                ),
                "gamma_pair_distribution_pass_fraction": float(
                    np.mean([float(row["gamma_pair_distribution_pass"]) for row in selected])
                ),
                "hmm_pair_distribution_pass_fraction": float(
                    np.mean([float(row["hmm_pair_distribution_pass"]) for row in selected])
                ),
                "maximum_two_clock_pair_tv_distance": max(
                    float(row["two_clock_pair_tv_distance"]) for row in selected
                ),
                "mean_two_clock_identity_rmse": mean_identity,
                "ci95_low_two_clock_identity_rmse": identity_low,
                "ci95_high_two_clock_identity_rmse": identity_high,
                "ci95_critical_value": critical,
                "mean_time_scale_ratio": float(
                    np.mean([float(row["time_scale_ratio"]) for row in selected])
                ),
            }
        )
    selection_fraction = float(
        np.mean([float(row["calibration_bic_gain_two_over_one"]) >= 6.0 for row in replicate_rows])
    )
    two_clock_transfer = float(
        np.mean([float(row["two_clock_cox_transfer_pass"]) for row in replicate_rows])
    )
    single_clock_transfer = float(
        np.mean([float(row["single_clock_cox_transfer_pass"]) for row in replicate_rows])
    )
    gamma_pair_transfer = float(
        np.mean([float(row["gamma_pair_distribution_pass"]) for row in replicate_rows])
    )
    hmm_pair_transfer = float(
        np.mean([float(row["hmm_pair_distribution_pass"]) for row in replicate_rows])
    )
    conditional_transfer = float(
        np.mean(
            [
                float(row["conditional_moment_marginal_memory_pass"])
                for row in replicate_rows
            ]
        )
    )
    if selection_fraction == 1.0 and two_clock_transfer == 1.0 and gamma_pair_transfer == 1.0:
        outcome = "two_clock_gamma_refresh_count_moment_closure"
    elif selection_fraction == 1.0 and conditional_transfer == 1.0:
        outcome = "two_clock_gamma_refresh_conditional_shape_closure"
    elif selection_fraction <= 0.2 and single_clock_transfer == 1.0:
        outcome = "single_clock_gamma_refresh_count_moment_closure"
    else:
        outcome = "gamma_refresh_count_moment_closure_failed"
    verdict = {
        "temperature": temperature,
        "calibration_horizon": float(args.calibration_horizon),
        "scoring_horizon": float(args.scoring_horizon),
        "independent_replicate_count": float(len(common_replicates)),
        "common_block_sizes": ";".join(f"{value:g}" for value in common_blocks),
        "excluded_underidentified_block_sizes": (
            ";".join(f"{value:g}" for value in excluded_blocks) if excluded_blocks else "none"
        ),
        "replica_block_count": float(len(replicate_rows)),
        "two_clock_selection_fraction": selection_fraction,
        "two_clock_transfer_pass_fraction": two_clock_transfer,
        "conditional_moment_marginal_memory_pass_fraction": conditional_transfer,
        "single_clock_transfer_pass_fraction": single_clock_transfer,
        "maximum_heldout_fano_relative_error": max(
            float(row["heldout_fano_relative_error"]) for row in replicate_rows
        ),
        "maximum_two_clock_identity_rmse": max(
            float(row["two_clock_identity_rmse"]) for row in replicate_rows
        ),
        "maximum_heldout_count_tv_distance": max(
            float(row["heldout_count_tv_distance"]) for row in replicate_rows
        ),
        "gamma_pair_distribution_pass_fraction": gamma_pair_transfer,
        "hmm_pair_distribution_pass_fraction": hmm_pair_transfer,
        "maximum_two_clock_pair_tv_distance": max(
            float(row["two_clock_pair_tv_distance"]) for row in replicate_rows
        ),
        "event_level_outcome": outcome,
        "count_moment_closure_claim_allowed": float(
            two_clock_transfer == 1.0
            or (selection_fraction <= 0.2 and single_clock_transfer == 1.0)
        ),
        "marginal_count_distribution_claim_allowed": float(
            max(float(row["heldout_count_tv_distance"]) for row in replicate_rows) <= 0.03
        ),
        "full_count_distribution_claim_allowed": 0.0,
        "full_count_sequence_likelihood_claim_allowed": 0.0,
        "joint_count_pair_distribution_claim_allowed": float(
            gamma_pair_transfer == 1.0
        ),
        "hybrid_semimarkov_emission_model_required": float(
            conditional_transfer == 1.0
            and gamma_pair_transfer < 1.0
            and hmm_pair_transfer == 1.0
        ),
        "heldout_macro_prediction_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "source_dois": "10.1063/1.2001629;10.1063/1.2803062;10.1103/PhysRevX.1.021013",
        "source_alignment": "finite_structured_exchange_clock_with_hierarchical_low_temperature_relaxation",
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicate_rows)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_histogram.csv"),
        histogram_rows,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_pair_histogram.csv"),
        pair_histogram_rows,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_blocks.csv"), block_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
