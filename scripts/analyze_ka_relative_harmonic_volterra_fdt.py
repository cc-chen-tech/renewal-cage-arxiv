#!/usr/bin/env python3
"""Test a bias-conditioned harmonic relative Volterra kernel and generalized FDT."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_projected_ito_innovations import load_drift_cache  # noqa: E402
from ka_projected_innovation import fit_constant_joint_covariance  # noqa: E402
from ka_relative_memory import (  # noqa: E402
    estimate_isoconfigurational_bias,
    invert_harmonic_velocity_memory_kernel,
    propagate_harmonic_gle_correlations,
    reconstruct_harmonic_random_force,
)
from ka_relative_pmf import fit_gaussian_relative_pmf  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stationary_correlations(
    paths: list[dict[str, np.ndarray | float | str]],
    *,
    bias: np.ndarray,
    maximum_lag: int,
) -> dict[str, np.ndarray]:
    clone_correlations: list[np.ndarray] = []
    for path in paths:
        position = np.asarray(path["relative_position"]) - bias[None, :, :]
        velocity = np.asarray(path["relative_velocity"])
        velocity = velocity - np.mean(velocity, axis=(0, 1), keepdims=True)
        rows = np.empty((maximum_lag + 1, 3), dtype=float)
        for lag in range(maximum_lag + 1):
            if lag == 0:
                earlier_position = later_position = position
                earlier_velocity = later_velocity = velocity
            else:
                earlier_position = position[:-lag]
                later_position = position[lag:]
                earlier_velocity = velocity[:-lag]
                later_velocity = velocity[lag:]
            rows[lag] = (
                np.mean(earlier_position * later_position),
                np.mean(earlier_velocity * later_velocity),
                np.mean(later_position * earlier_velocity),
            )
        clone_correlations.append(rows)
    mean = np.mean(clone_correlations, axis=0)
    return {
        "position_position": mean[:, 0],
        "velocity_velocity": mean[:, 1],
        "position_velocity": mean[:, 2],
    }


def force_correlation(force: np.ndarray, maximum_lag: int) -> np.ndarray:
    values = np.asarray(force, dtype=float)
    result = np.empty(maximum_lag + 1, dtype=float)
    for lag in range(maximum_lag + 1):
        result[lag] = float(
            np.mean(values * values)
            if lag == 0
            else np.mean(values[:-lag] * values[lag:])
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--covariance-cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--frame-time", type=float, default=0.01)
    parser.add_argument("--kernel-count", type=int, default=100)
    parser.add_argument("--maximum-lag", type=int, default=800)
    args = parser.parse_args()
    if (
        args.expected_clone_count < 2
        or args.target_count < 1
        or args.friction <= 0.0
        or args.frame_time <= 0.0
        or args.kernel_count < 2
        or args.maximum_lag <= args.kernel_count
    ):
        raise ValueError("invalid relative Volterra/FDT controls")

    clones: list[dict[str, np.ndarray | float | str]] = []
    fixed_targets: np.ndarray | None = None
    for index in range(1, args.expected_clone_count + 1):
        drift = load_drift_cache(
            args.drift_cache_directory / f"clone_{index:03d}_decomposed_drift.npz"
        )
        targets = np.asarray(drift["target_indices"], dtype=int)
        if len(targets) != args.target_count:
            raise ValueError("target count does not match relative-memory protocol")
        if fixed_targets is None:
            fixed_targets = targets
        elif not np.array_equal(targets, fixed_targets):
            raise ValueError("all clones must use the same fixed targets")
        with np.load(
            args.covariance_cache_directory
            / f"clone_{index:03d}_projected_covariance.npz"
        ) as cache:
            joint = np.asarray(cache["joint_noise_covariance_rate"], dtype=float)
            if str(cache["trajectory_sha256"]) != str(drift["trajectory_sha256"]):
                raise ValueError("drift and covariance caches do not align")
        clones.append({**drift, "joint_noise_covariance_rate": joint})

    details: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    kernels: list[dict[str, object]] = []
    for held_index, held in enumerate(clones):
        training = [clone for index, clone in enumerate(clones) if index != held_index]
        bias = estimate_isoconfigurational_bias(
            np.asarray([clone["relative_position"] for clone in training])
        )
        held_position = np.asarray(held["relative_position"])
        held_own_bias = np.mean(held_position, axis=0)
        centered_global = held_position - np.mean(
            held_position, axis=(0, 1), keepdims=True
        )
        raw_variance = float(np.mean(centered_global**2))
        held_bias_correlation = float(
            np.corrcoef(bias.ravel(), held_own_bias.ravel())[0, 1]
        )
        held_bias_normalized_rms_error = float(
            np.sqrt(np.mean((bias - held_own_bias) ** 2)) / np.sqrt(raw_variance)
        )
        training_bias_variance_fraction_removed = float(
            1.0 - np.mean((held_position - bias[None, :, :]) ** 2) / raw_variance
        )

        training_position = np.concatenate(
            [np.asarray(clone["relative_position"]) - bias[None, :, :] for clone in training]
        )
        training_velocity = np.concatenate(
            [np.asarray(clone["relative_velocity"]) for clone in training]
        )
        training_covariance = np.concatenate(
            [
                np.asarray(clone["joint_noise_covariance_rate"]).reshape(-1, 6, 6)
                for clone in training
            ]
        )
        constant_covariance = fit_constant_joint_covariance(
            training_covariance, model="block_isotropic"
        )
        relative_noise_rate = float(np.trace(constant_covariance[3:, 3:]) / 3.0)
        pmf = fit_gaussian_relative_pmf(
            training_position,
            training_velocity,
            relative_noise_variance_rate=relative_noise_rate,
            friction=args.friction,
        )
        kappa = float(pmf["harmonic_acceleration_stiffness"])
        training_correlation = stationary_correlations(
            training,
            bias=bias,
            maximum_lag=args.kernel_count,
        )
        kernel = invert_harmonic_velocity_memory_kernel(
            training_correlation["velocity_velocity"],
            training_correlation["position_velocity"],
            harmonic_acceleration_stiffness=kappa,
            friction=args.friction,
            frame_time=args.frame_time,
            kernel_count=args.kernel_count,
        )
        prediction = propagate_harmonic_gle_correlations(
            kernel,
            harmonic_acceleration_stiffness=kappa,
            friction=args.friction,
            frame_time=args.frame_time,
            output_count=args.maximum_lag + 1,
            position_variance=float(training_correlation["position_position"][0]),
            velocity_variance=float(training_correlation["velocity_velocity"][0]),
        )
        held_correlation = stationary_correlations(
            [held],
            bias=bias,
            maximum_lag=args.maximum_lag,
        )
        observed = np.stack(
            [
                held_correlation["position_position"]
                / held_correlation["position_position"][0],
                held_correlation["velocity_velocity"]
                / held_correlation["velocity_velocity"][0],
                held_correlation["position_velocity"]
                / np.sqrt(
                    held_correlation["position_position"][0]
                    * held_correlation["velocity_velocity"][0]
                ),
            ]
        )
        predicted = np.stack(
            [
                prediction["position_position_correlation"]
                / float(training_correlation["position_position"][0]),
                prediction["velocity_velocity_correlation"]
                / float(training_correlation["velocity_velocity"][0]),
                prediction["position_velocity_correlation"]
                / np.sqrt(
                    float(training_correlation["position_position"][0])
                    * float(training_correlation["velocity_velocity"][0])
                ),
            ]
        )
        fit_slice = slice(0, args.kernel_count + 1)
        extrapolation_slice = slice(args.kernel_count + 1, None)
        fit_rmse = float(np.sqrt(np.mean((observed[:, fit_slice] - predicted[:, fit_slice]) ** 2)))
        extrapolation_rmse = float(
            np.sqrt(
                np.mean(
                    (observed[:, extrapolation_slice] - predicted[:, extrapolation_slice])
                    ** 2
                )
            )
        )
        extrapolation_maximum_error = float(
            np.max(np.abs(observed[:, extrapolation_slice] - predicted[:, extrapolation_slice]))
        )

        random_force = reconstruct_harmonic_random_force(
            held_position,
            np.asarray(held["relative_velocity"]),
            np.asarray(held["relative_drift"]),
            isoconfigurational_bias=bias,
            harmonic_acceleration_stiffness=kappa,
            friction=args.friction,
            frame_time=args.frame_time,
            kernel=kernel,
        )
        observed_force_correlation = force_correlation(
            random_force, args.kernel_count - 1
        )
        target_force_correlation = float(pmf["fdt_velocity_variance"]) * kernel
        fdt_variance_ratio = float(
            observed_force_correlation[0] / target_force_correlation[0]
        )
        fdt_shape_correlation = float(
            np.corrcoef(observed_force_correlation, target_force_correlation)[0, 1]
        )
        fdt_normalized_rmse = float(
            np.sqrt(
                np.mean((observed_force_correlation - target_force_correlation) ** 2)
            )
            / np.sqrt(np.mean(target_force_correlation**2))
        )
        toeplitz = kernel[np.abs(np.subtract.outer(np.arange(len(kernel)), np.arange(len(kernel))))]
        kernel_toeplitz_minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(toeplitz)))

        details.append(
            {
                "record": "held_clone",
                "held_clone_index": float(held_index + 1),
                "held_bias_correlation": held_bias_correlation,
                "held_bias_normalized_rms_error": held_bias_normalized_rms_error,
                "training_bias_variance_fraction_removed": training_bias_variance_fraction_removed,
                "harmonic_acceleration_stiffness": kappa,
                "kernel_support_tau": args.kernel_count * args.frame_time,
                "kernel_tail_to_peak_ratio": float(abs(kernel[-1]) / np.max(np.abs(kernel))),
                "kernel_toeplitz_minimum_eigenvalue": kernel_toeplitz_minimum_eigenvalue,
                "held_fit_window_correlation_rmse": fit_rmse,
                "held_extrapolation_correlation_rmse": extrapolation_rmse,
                "held_extrapolation_maximum_correlation_error": extrapolation_maximum_error,
                "fdt_random_force_variance_ratio": fdt_variance_ratio,
                "fdt_random_force_shape_correlation": fdt_shape_correlation,
                "fdt_random_force_normalized_rmse": fdt_normalized_rmse,
                "fit_uses_held_clone": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
        for lag in range(args.maximum_lag + 1):
            curves.append(
                {
                    "held_clone_index": float(held_index + 1),
                    "lag_frames": float(lag),
                    "lag_time": lag * args.frame_time,
                    "observed_uu": float(observed[0, lag]),
                    "predicted_uu": float(predicted[0, lag]),
                    "observed_pp": float(observed[1, lag]),
                    "predicted_pp": float(predicted[1, lag]),
                    "observed_up": float(observed[2, lag]),
                    "predicted_up": float(predicted[2, lag]),
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        for lag, value in enumerate(kernel):
            kernels.append(
                {
                    "held_clone_index": float(held_index + 1),
                    "lag_frames": float(lag),
                    "lag_time": lag * args.frame_time,
                    "kernel": float(value),
                    "observed_random_force_correlation": float(
                        observed_force_correlation[lag]
                    ),
                    "fdt_target_force_correlation": float(
                        target_force_correlation[lag]
                    ),
                    "thermodynamic_claim_allowed": 0.0,
                }
            )

    metric_names = tuple(
        key
        for key in details[0]
        if key
        not in {
            "record",
            "held_clone_index",
            "fit_uses_held_clone",
            "thermodynamic_claim_allowed",
        }
    )
    summary: dict[str, object] = {
        "record": "aggregate",
        "held_clone_count": float(len(details)),
        "fit_uses_held_clone": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in details])
        summary[metric] = float(np.mean(values))
        summary[f"maximum_{metric}"] = float(np.max(values))
        summary[f"minimum_{metric}"] = float(np.min(values))
    bias_supported = (
        float(summary["minimum_held_bias_correlation"]) >= 0.75
        and float(summary["minimum_training_bias_variance_fraction_removed"]) >= 0.15
    )
    correlation_allowed = (
        float(summary["maximum_held_extrapolation_correlation_rmse"]) <= 0.08
        and float(summary["maximum_held_extrapolation_maximum_correlation_error"])
        <= 0.20
    )
    fdt_allowed = (
        0.8 <= float(summary["minimum_fdt_random_force_variance_ratio"])
        and float(summary["maximum_fdt_random_force_variance_ratio"]) <= 1.2
        and float(summary["minimum_fdt_random_force_shape_correlation"]) >= 0.8
        and float(summary["maximum_fdt_random_force_normalized_rmse"]) <= 0.20
        and float(summary["minimum_kernel_toeplitz_minimum_eigenvalue"]) >= -1e-10
    )
    summaries = [
        summary,
        {
            "record": "verdict",
            "isoconfigurational_cage_bias_supported": float(bias_supported),
            "relative_correlation_volterra_allowed": float(correlation_allowed),
            "relative_mori_fdt_closure_allowed": float(fdt_allowed),
            "physical_scalar_relative_gle_allowed": float(
                bias_supported and correlation_allowed and fdt_allowed
            ),
            "relative_orthogonal_force_closure_required": float(not fdt_allowed),
            "autonomous_single_particle_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        },
    ]
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curves)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_kernel.csv"), kernels)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)


if __name__ == "__main__":
    main()
