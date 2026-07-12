#!/usr/bin/env python3
"""Fit calibration-only finite-exchange clocks and score held-out event counts."""

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

from ka_replicates import (  # noqa: E402
    extract_debye_waller_cage_jumps,
    fit_two_state_poisson_hmm,
    independent_sample_ci95,
    load_lammps_custom_trajectory,
    particle_event_count_correlation_curve,
    particle_event_count_matrix,
    position_fluctuation_values,
    score_two_state_poisson_hmm,
    two_state_poisson_hmm_count_predictions,
)


def parse_block_sizes(value: str) -> np.ndarray:
    return np.array([float(item) for item in value.split(",")], dtype=float)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_hmm_transfer(row: dict[str, object]) -> dict[str, float | str]:
    """Apply fixed event-level transfer tolerances in failure-priority order."""

    checks = [
        ("mean_count", float(row["heldout_mean_relative_error"]) <= 0.10),
        ("count_fano", float(row["heldout_fano_relative_error"]) <= 0.20),
        ("identity_curve", float(row["heldout_identity_correlation_rmse"]) <= 0.05),
        ("late_identity_decay", float(row["heldout_late_identity_absolute_error"]) <= 0.05),
        (
            "heldout_likelihood",
            float(row["heldout_hmm_log_likelihood_gain_per_observation"]) > 0.0,
        ),
    ]
    failure = next((name for name, passed in checks if not passed), "none")
    return {
        "maximum_mean_relative_error": 0.10,
        "maximum_fano_relative_error": 0.20,
        "maximum_identity_correlation_rmse": 0.05,
        "maximum_late_identity_absolute_error": 0.05,
        "finite_exchange_hmm_transfer_pass": float(failure == "none"),
        "primary_failure": failure,
    }


def poisson_log_likelihood(counts: np.ndarray, mean_count: float) -> float:
    if mean_count <= 0.0:
        raise ValueError("Poisson mean must be positive")
    return float(
        sum(
            value * math.log(mean_count) - mean_count - math.lgamma(value + 1.0)
            for value in counts.ravel()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--heldout-replicates", type=Path, required=True)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--block-sizes", type=parse_block_sizes, required=True)
    parser.add_argument("--maximum-lag-time", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    production_time = int(round(float(manifest["production_time_tau"])))
    if production_time != 2 * args.calibration_time:
        raise ValueError("HMM transfer audit requires equal calibration and held-out halves")
    thresholds = {
        int(float(row["replicate"])): float(row["debye_waller_factor"])
        for row in read_rows(args.heldout_replicates)
    }
    curve_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    temperature = float(manifest["temperature"])
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = args.ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        threshold = thresholds[replicate_index]

        def events_for(window: np.ndarray) -> dict[str, np.ndarray]:
            times, fluctuation = position_fluctuation_values(
                window,
                half_window=args.fluctuation_half_window,
            )
            return extract_debye_waller_cage_jumps(
                window,
                debye_waller_factor=threshold,
                half_window=args.fluctuation_half_window,
                activity_times=times,
                activity_values=fluctuation,
            )

        calibration_events = events_for(positions[: args.calibration_time + 1])
        heldout_events = events_for(positions[args.calibration_time :])
        for block_size in args.block_sizes:
            calibration_counts = particle_event_count_matrix(
                calibration_events,
                duration=float(args.calibration_time),
                particle_count=positions.shape[1],
                block_size=float(block_size),
            )
            heldout_counts = particle_event_count_matrix(
                heldout_events,
                duration=float(args.calibration_time),
                particle_count=positions.shape[1],
                block_size=float(block_size),
            )
            maximum_lag = min(
                int(args.maximum_lag_time // block_size),
                calibration_counts.shape[1] - 1,
            )
            fitted = fit_two_state_poisson_hmm(
                calibration_counts,
                block_size=float(block_size),
            )
            predicted = two_state_poisson_hmm_count_predictions(
                fitted,
                maximum_lag=maximum_lag,
            )
            observed = particle_event_count_correlation_curve(
                heldout_events,
                duration=float(args.calibration_time),
                particle_count=positions.shape[1],
                block_size=float(block_size),
                maximum_lag=maximum_lag,
            )
            heldout_score = score_two_state_poisson_hmm(heldout_counts, fitted)
            calibration_mean = float(np.mean(calibration_counts))
            heldout_mean = float(np.mean(heldout_counts))
            heldout_variance = float(np.var(heldout_counts))
            heldout_fano = heldout_variance / heldout_mean
            predicted_mean = float(predicted[0]["predicted_mean_count"])
            predicted_fano = float(predicted[0]["predicted_fano_factor"])
            hmm_correlations = np.array(
                [float(row["predicted_particle_identity_correlation"]) for row in predicted]
            )
            observed_correlations = np.array(
                [float(row["particle_identity_correlation"]) for row in observed]
            )
            particle_means = np.mean(calibration_counts, axis=1)
            static_environment_variance = max(
                float(np.var(particle_means)) - calibration_mean / calibration_counts.shape[1],
                0.0,
            )
            static_correlation = static_environment_variance / (
                calibration_mean + static_environment_variance
            )
            poisson_ll = poisson_log_likelihood(heldout_counts, calibration_mean)
            row: dict[str, object] = {
                "replicate": float(replicate_index),
                "temperature": temperature,
                "block_size": float(block_size),
                "block_count": float(calibration_counts.shape[1]),
                "maximum_lag": float(maximum_lag),
                "calibration_event_count": float(np.sum(calibration_counts)),
                "heldout_event_count": float(np.sum(heldout_counts)),
                **fitted,
                "heldout_mean_count": heldout_mean,
                "heldout_count_variance": heldout_variance,
                "heldout_fano_factor": heldout_fano,
                "predicted_mean_count": predicted_mean,
                "predicted_fano_factor": predicted_fano,
                "heldout_mean_relative_error": abs(predicted_mean / heldout_mean - 1.0),
                "heldout_fano_relative_error": abs(predicted_fano / heldout_fano - 1.0),
                "heldout_identity_correlation_rmse": float(
                    np.sqrt(np.mean((hmm_correlations - observed_correlations) ** 2))
                ),
                "heldout_early_identity_signed_error": float(
                    observed_correlations[0] - hmm_correlations[0]
                ),
                "heldout_late_identity_signed_excess": float(
                    observed_correlations[-1] - hmm_correlations[-1]
                ),
                "heldout_late_identity_absolute_error": float(
                    abs(observed_correlations[-1] - hmm_correlations[-1])
                ),
                "heldout_hmm_log_likelihood": float(heldout_score["log_likelihood"]),
                "heldout_poisson_log_likelihood": poisson_ll,
                "heldout_hmm_log_likelihood_gain_per_observation": (
                    float(heldout_score["log_likelihood"]) - poisson_ll
                ) / heldout_counts.size,
                "poisson_identity_correlation_rmse": float(
                    np.sqrt(np.mean(observed_correlations**2))
                ),
                "static_identity_correlation": static_correlation,
                "static_identity_correlation_rmse": float(
                    np.sqrt(np.mean((observed_correlations - static_correlation) ** 2))
                ),
                "thermodynamic_claim_allowed": 0.0,
            }
            row.update(classify_hmm_transfer(row))
            replicate_rows.append(row)
            for observed_row, predicted_row in zip(observed, predicted):
                curve_rows.append(
                    {
                        "replicate": float(replicate_index),
                        "temperature": temperature,
                        "block_size": float(block_size),
                        "block_lag": observed_row["block_lag"],
                        "lag_time": observed_row["lag_time"],
                        "observed_identity_correlation": observed_row[
                            "particle_identity_correlation"
                        ],
                        "hmm_identity_correlation": predicted_row[
                            "predicted_particle_identity_correlation"
                        ],
                        "poisson_identity_correlation": 0.0,
                        "static_identity_correlation": static_correlation,
                    }
                )

    block_rows: list[dict[str, object]] = []
    for block_size in args.block_sizes:
        selected = [row for row in replicate_rows if float(row["block_size"]) == block_size]
        late = np.array([float(row["heldout_late_identity_signed_excess"]) for row in selected])
        early = np.array([float(row["heldout_early_identity_signed_error"]) for row in selected])
        late_se = float(np.std(late, ddof=1) / math.sqrt(len(late)))
        late_low, late_high, critical = independent_sample_ci95(
            mean=float(np.mean(late)),
            standard_error=late_se,
            sample_count=len(late),
        )
        block_rows.append(
            {
                "block_size": float(block_size),
                "independent_replicate_count": float(len(selected)),
                "mean_early_identity_signed_error": float(np.mean(early)),
                "mean_late_identity_signed_excess": float(np.mean(late)),
                "standard_error_late_identity_signed_excess": late_se,
                "ci95_low_late_identity_signed_excess": late_low,
                "ci95_high_late_identity_signed_excess": late_high,
                "ci95_critical_value": critical,
                "late_positive_excess_detected": float(late_low > 0.0),
                "early_identity_transfer_within_tolerance": float(
                    np.max(np.abs(early)) <= 0.05
                ),
                "replica_block_pass_fraction": float(
                    np.mean([float(row["finite_exchange_hmm_transfer_pass"]) for row in selected])
                ),
            }
        )
    all_pass = all(float(row["finite_exchange_hmm_transfer_pass"]) == 1.0 for row in replicate_rows)
    broad_exchange = any(
        float(row["late_positive_excess_detected"]) == 1.0
        and float(row["early_identity_transfer_within_tolerance"]) == 1.0
        for row in block_rows
    )
    verdict = {
        "temperature": temperature,
        "scoring_horizon": float(args.maximum_lag_time),
        "independent_replicate_count": float(len(manifest["replicates"])),
        "block_sizes": ";".join(f"{value:g}" for value in args.block_sizes),
        "replica_block_count": float(len(replicate_rows)),
        "replica_block_pass_fraction": float(
            np.mean([float(row["finite_exchange_hmm_transfer_pass"]) for row in replicate_rows])
        ),
        "two_state_poisson_hmm_sufficient": float(all_pass),
        "non_single_exponential_exchange_required": float(broad_exchange and not all_pass),
        "event_level_outcome": (
            "two_state_poisson_hmm_sufficient"
            if all_pass
            else (
                "non_single_exponential_exchange_required"
                if broad_exchange
                else "two_state_poisson_hmm_rejected_without_unique_replacement"
            )
        ),
        "heldout_macro_prediction_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicate_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_blocks.csv"), block_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
