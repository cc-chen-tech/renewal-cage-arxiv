#!/usr/bin/env python3
"""Propagate calibration-only two-clock event statistics to held-out observables."""

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
    compound_jump_cage_observables,
    correlated_jump_propagator,
    extract_debye_waller_cage_jumps,
    jump_vector_correlation_curve,
    load_lammps_custom_trajectory,
    position_fluctuation_values,
    two_clock_hmm_mixture_parameters,
    two_clock_hmm_mixture_total_count_pmf,
    two_clock_hmm_mixture_total_count_statistics,
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


def classify_macro_transfer(
    curve_errors: dict[str, float],
    scalar_errors: dict[str, float],
    *,
    alpha_crossing_ready: bool,
) -> dict[str, float | str]:
    curve_pass = (
        float(curve_errors["maximum_ensemble_msd_relative_error"]) <= 0.10
        and float(curve_errors["maximum_ensemble_ngp_absolute_error"]) <= 0.30
        and float(curve_errors["maximum_ensemble_fs_absolute_error"]) <= 0.03
        and float(curve_errors.get("maximum_count_tail_probability", 0.0)) <= 0.01
    )
    scalar_pass = (
        alpha_crossing_ready
        and float(scalar_errors["diffusion_relative_error"]) <= 0.15
        and float(scalar_errors["alpha_relaxation_relative_error"]) <= 0.20
        and float(scalar_errors["diffusion_alpha_product_relative_error"]) <= 0.25
    )
    if not curve_pass:
        failure = "heldout_curve_shape"
    elif not alpha_crossing_ready:
        failure = "alpha_crossing_not_identifiable"
    elif not scalar_pass:
        failure = "heldout_derived_scalars"
    else:
        failure = "none"
    return {
        "msd_relative_error_tolerance": 0.10,
        "ngp_absolute_error_tolerance": 0.30,
        "fs_absolute_error_tolerance": 0.03,
        "count_tail_probability_tolerance": 0.01,
        "diffusion_relative_error_tolerance": 0.15,
        "alpha_relaxation_relative_error_tolerance": 0.20,
        "diffusion_alpha_product_relative_error_tolerance": 0.25,
        "alpha_crossing_ready": float(alpha_crossing_ready),
        "curve_transfer_pass": float(curve_pass),
        "derived_scalar_transfer_pass": float(scalar_pass),
        "joint_macro_transfer_pass": float(curve_pass and scalar_pass),
        "primary_failure": failure,
        "heldout_events_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def alpha_crossing_time(lags: np.ndarray, values: np.ndarray) -> float:
    threshold = math.exp(-1.0)
    for index in range(1, len(lags)):
        if values[index - 1] > threshold >= values[index]:
            fraction = (threshold - values[index - 1]) / (
                values[index] - values[index - 1]
            )
            return float(
                math.exp(
                    math.log(float(lags[index - 1]))
                    + fraction
                    * (
                        math.log(float(lags[index]))
                        - math.log(float(lags[index - 1]))
                    )
                )
            )
    return math.nan


def wave_number_from_key(key: str) -> float:
    return float(key.removeprefix("residual_fs_k").replace("p", "."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--heldout-replicates", type=Path, required=True)
    parser.add_argument("--spectrum-replicates", type=Path, required=True)
    parser.add_argument("--hmm-replicates", type=Path, required=True)
    parser.add_argument("--calibration-factorization", type=Path, required=True)
    parser.add_argument("--heldout-factorization", type=Path, required=True)
    parser.add_argument("--block-size", type=float, default=20.0)
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--maximum-event-correlation-lag", type=int, default=8)
    parser.add_argument("--maximum-propagator-count", type=int, default=60)
    parser.add_argument("--minimum-propagator-samples", type=int, default=500)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    threshold_rows = {
        int(float(row["replicate"])): row for row in read_rows(args.heldout_replicates)
    }
    spectrum_rows = read_rows(args.spectrum_replicates)
    hmm_rows = read_rows(args.hmm_replicates)
    calibration = {
        (int(float(row["replicate"])), int(float(row["lag"]))): row
        for row in read_rows(args.calibration_factorization)
    }
    heldout = {
        (int(float(row["replicate"])), int(float(row["lag"]))): row
        for row in read_rows(args.heldout_factorization)
    }
    common_keys = sorted(
        key
        for key in calibration.keys() & heldout.keys()
        if key[1] % int(args.block_size) == 0
    )
    if not common_keys:
        raise ValueError("no factorization lags are divisible by the count block size")
    fs_keys = sorted(
        key for key in next(iter(calibration.values())) if key.startswith("residual_fs_k")
    )
    rows: list[dict[str, object]] = []
    propagator_rows: list[dict[str, object]] = []
    for replicate_spec in manifest["replicates"]:
        replicate = int(replicate_spec["replicate"])
        local_keys = [key for key in common_keys if key[0] == replicate]
        if not local_keys:
            raise ValueError(f"replicate {replicate} lacks common macro lags")
        spectrum = next(
            row
            for row in spectrum_rows
            if int(float(row["replicate"])) == replicate
            and float(row["block_size"]) == args.block_size
        )
        hmm = next(
            row
            for row in hmm_rows
            if int(float(row["replicate"])) == replicate
            and float(row["block_size"]) == args.block_size
        )
        parameters = two_clock_hmm_mixture_parameters(
            hmm,
            spectrum,
            block_size=args.block_size,
        )
        directory = args.ensemble_directory / str(replicate_spec["directory"])
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        calibration_positions = positions[: args.calibration_time + 1]
        times, fluctuations = position_fluctuation_values(
            calibration_positions,
            half_window=args.fluctuation_half_window,
        )
        events = extract_debye_waller_cage_jumps(
            calibration_positions,
            debye_waller_factor=float(threshold_rows[replicate]["debye_waller_factor"]),
            half_window=args.fluctuation_half_window,
            activity_times=times,
            activity_values=fluctuations,
        )
        jumps = np.asarray(events["jump_vector"], dtype=float)
        correlation = jump_vector_correlation_curve(
            events,
            maximum_lag=args.maximum_event_correlation_lag,
        )
        green_kubo_factor = float(correlation[-1]["cumulative_green_kubo_factor"])
        wave_numbers = np.array([wave_number_from_key(key) for key in fs_keys])
        propagator = correlated_jump_propagator(
            events,
            maximum_count=args.maximum_propagator_count,
            wave_numbers=wave_numbers,
            minimum_sample_count=args.minimum_propagator_samples,
        )
        maximum_kernel_count = int(propagator[-1]["jump_count"])
        for propagator_row in propagator:
            propagator_rows.append(
                {
                    "replicate": float(replicate),
                    "temperature": float(manifest["temperature"]),
                    **propagator_row,
                }
            )
        raw_squared = np.sum(jumps**2, axis=1)
        for key in local_keys:
            lag = key[1]
            calibration_row = calibration[key]
            heldout_row = heldout[key]
            block_count = int(round(lag / args.block_size))
            count_moments = two_clock_hmm_mixture_total_count_statistics(
                parameters,
                block_count=block_count,
                pgf_argument=1.0,
            )
            count_distribution = two_clock_hmm_mixture_total_count_pmf(
                parameters,
                block_count=block_count,
                maximum_count=maximum_kernel_count,
            )
            count_pmf = np.asarray(count_distribution["count_pmf"], dtype=float)
            conditional_msd = np.array(
                [float(value["conditional_msd"]) for value in propagator]
            )
            conditional_fourth = np.array(
                [float(value["conditional_fourth_moment"]) for value in propagator]
            )
            event_msd = float(np.sum(count_pmf * conditional_msd))
            event_fourth = float(np.sum(count_pmf * conditional_fourth))
            row: dict[str, object] = {
                "replicate": float(replicate),
                "temperature": float(manifest["temperature"]),
                "lag": float(lag),
                "block_size": args.block_size,
                "block_count": float(block_count),
                "calibration_jump_count": float(len(jumps)),
                "raw_calibration_jump_msd": float(np.mean(raw_squared)),
                "raw_calibration_jump_fourth_moment": float(
                    np.mean(raw_squared**2)
                ),
                "calibration_green_kubo_factor": green_kubo_factor,
                "event_correlation_lag_count": float(
                    args.maximum_event_correlation_lag
                ),
                "correlated_jump_kernel_maximum_count": float(
                    maximum_kernel_count
                ),
                "correlated_jump_kernel_minimum_samples": float(
                    args.minimum_propagator_samples
                ),
                "count_tail_probability": count_distribution[
                    "tail_probability"
                ],
                "predicted_event_msd": event_msd,
                "predicted_event_fourth_moment": event_fourth,
                "predicted_mean_count": count_moments["mean_count"],
                "predicted_count_variance": count_moments["count_variance"],
                "heldout_events_used_in_prediction": 0.0,
                "macro_fit_parameter_count": 0.0,
                "jump_direction_correlation_included": 1.0,
                "thermodynamic_claim_allowed": 0.0,
            }
            predicted_channels: dict[str, dict[str, float]] = {}
            for fs_key in fs_keys:
                characteristic_key = (
                    f"conditional_characteristic_k{wave_number_from_key(fs_key):g}".replace(
                        ".", "p"
                    )
                )
                event_fs = float(
                    np.sum(
                        count_pmf
                        * np.array(
                            [float(value[characteristic_key]) for value in propagator]
                        )
                    )
                )
                predicted_channels[fs_key] = compound_jump_cage_observables(
                    mean_count=1.0,
                    factorial_second_count=0.0,
                    jump_msd=event_msd,
                    jump_fourth_moment=event_fourth,
                    count_pgf=event_fs,
                    cage_msd=float(calibration_row["residual_msd"]),
                    cage_ngp=float(calibration_row["residual_ngp"]),
                    cage_fs=float(calibration_row[fs_key]),
                    dimension=jumps.shape[1],
                )
                observable_key = fs_key.removeprefix("residual_")
                predicted_fs = predicted_channels[fs_key]["factorized_fs"]
                observed_fs = float(heldout_row[f"observed_{observable_key}"])
                row[f"predicted_{observable_key}"] = predicted_fs
                row[f"observed_{observable_key}"] = observed_fs
                row[f"absolute_error_{observable_key}"] = abs(predicted_fs - observed_fs)
                row[f"predicted_event_{observable_key}"] = event_fs
            reference = predicted_channels[fs_keys[0]]
            row.update(
                {
                    "predicted_msd": reference["factorized_msd"],
                    "observed_msd": float(heldout_row["observed_msd"]),
                    "msd_relative_error": abs(
                        reference["factorized_msd"]
                        / float(heldout_row["observed_msd"])
                        - 1.0
                    ),
                    "predicted_ngp": reference["factorized_ngp"],
                    "observed_ngp": float(heldout_row["observed_ngp"]),
                    "ngp_absolute_error": abs(
                        reference["factorized_ngp"]
                        - float(heldout_row["observed_ngp"])
                    ),
                }
            )
            rows.append(row)

    summary_rows: list[dict[str, object]] = []
    for lag in sorted({float(row["lag"]) for row in rows}):
        selected = [row for row in rows if float(row["lag"]) == lag]
        predicted_msd = float(np.mean([float(row["predicted_msd"]) for row in selected]))
        observed_msd = float(np.mean([float(row["observed_msd"]) for row in selected]))
        predicted_ngp = float(np.mean([float(row["predicted_ngp"]) for row in selected]))
        observed_ngp = float(np.mean([float(row["observed_ngp"]) for row in selected]))
        summary: dict[str, object] = {
            "lag": lag,
            "independent_replicate_count": float(len(selected)),
            "predicted_msd": predicted_msd,
            "observed_msd": observed_msd,
            "ensemble_msd_relative_error": abs(predicted_msd / observed_msd - 1.0),
            "predicted_ngp": predicted_ngp,
            "observed_ngp": observed_ngp,
            "ensemble_ngp_absolute_error": abs(predicted_ngp - observed_ngp),
        }
        for fs_key in fs_keys:
            observable_key = fs_key.removeprefix("residual_")
            predicted_fs = float(
                np.mean([float(row[f"predicted_{observable_key}"]) for row in selected])
            )
            observed_fs = float(
                np.mean([float(row[f"observed_{observable_key}"]) for row in selected])
            )
            summary[f"predicted_{observable_key}"] = predicted_fs
            summary[f"observed_{observable_key}"] = observed_fs
            summary[f"ensemble_absolute_error_{observable_key}"] = abs(
                predicted_fs - observed_fs
            )
        summary_rows.append(summary)

    maximum_msd = max(float(row["ensemble_msd_relative_error"]) for row in summary_rows)
    maximum_ngp = max(float(row["ensemble_ngp_absolute_error"]) for row in summary_rows)
    maximum_fs = max(
        float(row[f"ensemble_absolute_error_{key.removeprefix('residual_')}"])
        for row in summary_rows
        for key in fs_keys
    )
    maximum_tail = max(float(row["count_tail_probability"]) for row in rows)
    largest = summary_rows[-1]
    predicted_diffusion = float(largest["predicted_msd"]) / (6.0 * float(largest["lag"]))
    observed_diffusion = float(largest["observed_msd"]) / (6.0 * float(largest["lag"]))
    alpha_key = min(fs_keys, key=lambda key: abs(wave_number_from_key(key) - 7.25))
    alpha_observable = alpha_key.removeprefix("residual_")
    lag_values = np.array([float(row["lag"]) for row in summary_rows])
    predicted_alpha = alpha_crossing_time(
        lag_values,
        np.array([float(row[f"predicted_{alpha_observable}"]) for row in summary_rows]),
    )
    observed_alpha = alpha_crossing_time(
        lag_values,
        np.array([float(row[f"observed_{alpha_observable}"]) for row in summary_rows]),
    )
    alpha_ready = math.isfinite(predicted_alpha) and math.isfinite(observed_alpha)
    scalar_row = {
        "temperature": float(manifest["temperature"]),
        "diffusion_lag": float(largest["lag"]),
        "predicted_diffusion": predicted_diffusion,
        "observed_diffusion": observed_diffusion,
        "diffusion_relative_error": abs(predicted_diffusion / observed_diffusion - 1.0),
        "predicted_alpha_relaxation_time": predicted_alpha,
        "observed_alpha_relaxation_time": observed_alpha,
        "alpha_relaxation_relative_error": (
            abs(predicted_alpha / observed_alpha - 1.0) if alpha_ready else math.nan
        ),
        "predicted_diffusion_alpha_product": predicted_diffusion * predicted_alpha,
        "observed_diffusion_alpha_product": observed_diffusion * observed_alpha,
        "diffusion_alpha_product_relative_error": (
            abs(
                predicted_diffusion
                * predicted_alpha
                / (observed_diffusion * observed_alpha)
                - 1.0
            )
            if alpha_ready
            else math.nan
        ),
    }
    curve_errors = {
        "maximum_ensemble_msd_relative_error": maximum_msd,
        "maximum_ensemble_ngp_absolute_error": maximum_ngp,
        "maximum_ensemble_fs_absolute_error": maximum_fs,
        "maximum_count_tail_probability": maximum_tail,
    }
    verdict: dict[str, object] = {
        "temperature": float(manifest["temperature"]),
        "independent_replicate_count": float(len(manifest["replicates"])),
        "lag_count": float(len(summary_rows)),
        "wave_number_count": float(len(fs_keys)),
        **curve_errors,
        **scalar_row,
        "calibration_only_two_clock_count_law": 1.0,
        "calibration_only_jump_distribution": 1.0,
        "calibration_correlated_jump_kernel": 1.0,
        "calibration_cage_residual_transfer": 1.0,
        "jump_direction_correlation_included": 1.0,
        "preregistered_heldout_prediction_claim_allowed": 0.0,
    }
    verdict.update(
        classify_macro_transfer(
            curve_errors,
            scalar_row,
            alpha_crossing_ready=alpha_ready,
        )
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_rows.csv"), rows)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_propagator.csv"),
        propagator_rows,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
        summary_rows,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_scalars.csv"),
        [scalar_row],
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"),
        [verdict],
    )


if __name__ == "__main__":
    main()
