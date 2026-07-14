#!/usr/bin/env python3
"""Test microscopic relative-coordinate PMF closure and Markov-OU dynamics."""

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
from ka_relative_pmf import (  # noqa: E402
    fit_gaussian_relative_pmf,
    underdamped_ou_propagator,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def radial_density_quadratic_coefficient(
    relative_position: np.ndarray,
    *,
    radial_bin_count: int,
    lower_quantile: float,
    upper_quantile: float,
    minimum_bin_count: int,
) -> float:
    position = np.asarray(relative_position, dtype=float).reshape(-1, 3)
    radius = np.linalg.norm(position, axis=1)
    variance = float(np.mean((position - np.mean(position, axis=0)) ** 2))
    lower, upper = np.quantile(radius, [lower_quantile, upper_quantile])
    edges = np.linspace(lower, upper, radial_bin_count + 1)
    count, _ = np.histogram(radius, edges)
    shell_volume = 4.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    center = 0.5 * (edges[1:] + edges[:-1])
    valid = count >= minimum_bin_count
    if np.sum(valid) < 3:
        raise ValueError("too few populated radial bins for state counting")
    design = np.stack([np.ones_like(center), center**2 / variance], axis=1)
    log_density = np.log(count / shell_volume)
    weight = np.sqrt(count)
    coefficient = np.linalg.lstsq(
        design[valid] * weight[valid, None],
        log_density[valid] * weight[valid],
        rcond=None,
    )[0]
    return float(coefficient[1])


def conditional_radial_curve(
    relative_position: np.ndarray,
    relative_mean_force: np.ndarray,
    *,
    edges: np.ndarray,
    harmonic_stiffness: float,
    temperature_naive_stiffness: float,
    minimum_bin_count: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    position = np.asarray(relative_position, dtype=float).reshape(-1, 3)
    force = np.asarray(relative_mean_force, dtype=float).reshape(-1, 3)
    radius = np.linalg.norm(position, axis=1)
    radial_force = np.sum(force * position, axis=1) / np.maximum(
        radius, np.finfo(float).tiny
    )
    assignment = np.digitize(radius, edges) - 1
    rows: list[dict[str, float]] = []
    for index in range(len(edges) - 1):
        selected = assignment == index
        count = int(np.sum(selected))
        if count < minimum_bin_count:
            continue
        mean_radius = float(np.mean(radius[selected]))
        rows.append(
            {
                "radial_bin_index": float(index),
                "sample_count": float(count),
                "mean_radius": mean_radius,
                "observed_conditional_radial_force": float(
                    np.mean(radial_force[selected])
                ),
                "pmf_predicted_radial_force": -harmonic_stiffness * mean_radius,
                "temperature_naive_radial_force": -temperature_naive_stiffness
                * mean_radius,
            }
        )
    if len(rows) < 3:
        raise ValueError("too few held radial bins for conditional-force validation")
    observed = np.asarray([row["observed_conditional_radial_force"] for row in rows])
    predicted = np.asarray([row["pmf_predicted_radial_force"] for row in rows])
    naive = np.asarray([row["temperature_naive_radial_force"] for row in rows])
    weight = np.asarray([row["sample_count"] for row in rows])
    observed_rms = float(np.sqrt(np.sum(weight * observed**2) / np.sum(weight)))

    def normalized_rmse(values: np.ndarray) -> float:
        return float(
            np.sqrt(np.sum(weight * (observed - values) ** 2) / np.sum(weight))
            / observed_rms
        )

    return {
        "conditional_radial_force_correlation": float(
            np.corrcoef(observed, predicted)[0, 1]
        ),
        "conditional_radial_force_normalized_rmse": normalized_rmse(predicted),
        "temperature_naive_force_normalized_rmse": normalized_rmse(naive),
        "held_radial_bin_count": float(len(rows)),
    }, rows


def maximum_component_lag_correlation(values: np.ndarray) -> float:
    path = np.asarray(values, dtype=float)
    source = path[:-1].reshape(-1, 3)
    target = path[1:].reshape(-1, 3)
    correlations = [
        np.corrcoef(source[:, component], target[:, component])[0, 1]
        for component in range(3)
    ]
    return float(np.max(np.abs(correlations)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--covariance-cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--frame-time", type=float, default=0.01)
    parser.add_argument("--radial-bin-count", type=int, default=100)
    parser.add_argument("--radial-lower-quantile", type=float, default=0.005)
    parser.add_argument("--radial-upper-quantile", type=float, default=0.995)
    parser.add_argument("--minimum-bin-count", type=int, default=100)
    parser.add_argument(
        "--correlation-lags", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 100]
    )
    args = parser.parse_args()
    lags = tuple(sorted(set(args.correlation_lags)))
    if (
        args.expected_clone_count < 2
        or args.target_count < 1
        or args.friction <= 0.0
        or args.temperature <= 0.0
        or args.frame_time <= 0.0
        or args.radial_bin_count < 10
        or not 0.0 <= args.radial_lower_quantile < args.radial_upper_quantile <= 1.0
        or args.minimum_bin_count < 10
        or not lags
        or any(lag < 1 for lag in lags)
    ):
        raise ValueError("invalid relative-PMF controls")

    clones: list[dict[str, np.ndarray | float | str]] = []
    fixed_targets: np.ndarray | None = None
    for index in range(1, args.expected_clone_count + 1):
        drift = load_drift_cache(
            args.drift_cache_directory / f"clone_{index:03d}_decomposed_drift.npz"
        )
        targets = np.asarray(drift["target_indices"], dtype=int)
        if len(targets) != args.target_count:
            raise ValueError("target count does not match the relative-PMF protocol")
        if fixed_targets is None:
            fixed_targets = targets
        elif not np.array_equal(targets, fixed_targets):
            raise ValueError("all clones must use the same fixed targets")
        with np.load(
            args.covariance_cache_directory
            / f"clone_{index:03d}_projected_covariance.npz"
        ) as cache:
            joint = np.asarray(cache["joint_noise_covariance_rate"], dtype=float)
            if (
                str(cache["trajectory_sha256"]) != str(drift["trajectory_sha256"])
                or float(cache["thermodynamic_claim_allowed"]) != 0.0
            ):
                raise ValueError("covariance and drift caches do not align")
        clones.append({**drift, "joint_noise_covariance_rate": joint})

    details: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for held_index, held in enumerate(clones):
        training = [clone for index, clone in enumerate(clones) if index != held_index]
        training_position = np.concatenate(
            [np.asarray(clone["relative_position"]).reshape(-1, 3) for clone in training]
        )
        training_velocity = np.concatenate(
            [np.asarray(clone["relative_velocity"]).reshape(-1, 3) for clone in training]
        )
        training_covariance = np.concatenate(
            [
                np.asarray(clone["joint_noise_covariance_rate"]).reshape(-1, 6, 6)
                for clone in training
            ]
        )
        constant_covariance = fit_constant_joint_covariance(
            training_covariance,
            model="block_isotropic",
        )
        relative_noise_rate = float(np.trace(constant_covariance[3:, 3:]) / 3.0)
        pmf = fit_gaussian_relative_pmf(
            training_position,
            training_velocity,
            relative_noise_variance_rate=relative_noise_rate,
            friction=args.friction,
        )
        density_quadratic = radial_density_quadratic_coefficient(
            training_position,
            radial_bin_count=args.radial_bin_count,
            lower_quantile=args.radial_lower_quantile,
            upper_quantile=args.radial_upper_quantile,
            minimum_bin_count=args.minimum_bin_count,
        )
        training_radius = np.linalg.norm(training_position, axis=1)
        radial_edges = np.linspace(
            *np.quantile(
                training_radius,
                [args.radial_lower_quantile, args.radial_upper_quantile],
            ),
            args.radial_bin_count + 1,
        )
        held_position = np.asarray(held["relative_position"])
        held_velocity = np.asarray(held["relative_velocity"])
        held_force = np.asarray(held["relative_drift"]) + args.friction * held_velocity
        harmonic_stiffness = float(pmf["harmonic_acceleration_stiffness"])
        radial_metrics, radial_rows = conditional_radial_curve(
            held_position,
            held_force,
            edges=radial_edges,
            harmonic_stiffness=harmonic_stiffness,
            temperature_naive_stiffness=args.temperature
            / float(pmf["relative_position_variance"]),
            minimum_bin_count=args.minimum_bin_count,
        )
        mean_force_residual = held_force + harmonic_stiffness * held_position
        residual_lag = maximum_component_lag_correlation(mean_force_residual)
        maximum_ou_error = 0.0
        training_u_variance = float(pmf["relative_position_variance"])
        training_p_variance = float(pmf["fdt_velocity_variance"])
        for lag in lags:
            propagator = underdamped_ou_propagator(
                harmonic_stiffness,
                args.friction,
                lag * args.frame_time,
            )
            observed_uu = float(
                np.mean(held_position[:-lag] * held_position[lag:])
                / np.mean(held_position**2)
            )
            observed_pp = float(
                np.mean(held_velocity[:-lag] * held_velocity[lag:])
                / np.mean(held_velocity**2)
            )
            observed_up = float(
                np.mean(held_position[:-lag] * held_velocity[lag:])
                / np.sqrt(np.mean(held_position**2) * np.mean(held_velocity**2))
            )
            predicted_uu = float(propagator[0, 0])
            predicted_pp = float(propagator[1, 1])
            predicted_up = float(
                propagator[1, 0]
                * np.sqrt(training_u_variance / training_p_variance)
            )
            maximum_ou_error = max(
                maximum_ou_error,
                abs(observed_uu - predicted_uu),
                abs(observed_pp - predicted_pp),
                abs(observed_up - predicted_up),
            )
            curves.append(
                {
                    "record": "correlation",
                    "held_clone_index": float(held_index + 1),
                    "lag_frames": float(lag),
                    "lag_time": lag * args.frame_time,
                    "observed_uu_correlation": observed_uu,
                    "predicted_uu_correlation": predicted_uu,
                    "observed_pp_correlation": observed_pp,
                    "predicted_pp_correlation": predicted_pp,
                    "observed_up_correlation": observed_up,
                    "predicted_up_correlation": predicted_up,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        for row in radial_rows:
            curves.append(
                {
                    "record": "radial_force",
                    "held_clone_index": float(held_index + 1),
                    **row,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        details.append(
            {
                "record": "held_clone",
                "held_clone_index": float(held_index + 1),
                **{key: float(value) for key, value in pmf.items() if key != "thermodynamic_claim_allowed"},
                "radial_log_density_quadratic_coefficient": density_quadratic,
                **radial_metrics,
                "mean_force_residual_lag_correlation": residual_lag,
                "maximum_markov_ou_correlation_error": maximum_ou_error,
                "fit_uses_held_clone": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )

    metrics = tuple(
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
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in details])
        summary[metric] = float(np.mean(values))
        summary[f"maximum_{metric}"] = float(np.max(values))
        summary[f"minimum_{metric}"] = float(np.min(values))

    static_allowed = (
        float(summary["maximum_fdt_velocity_variance_relative_error"]) <= 0.03
        and max(
            abs(float(summary["minimum_radial_log_density_quadratic_coefficient"]) + 0.5),
            abs(float(summary["maximum_radial_log_density_quadratic_coefficient"]) + 0.5),
        )
        <= 0.03
        and float(summary["minimum_conditional_radial_force_correlation"]) >= 0.98
        and float(summary["maximum_conditional_radial_force_normalized_rmse"]) <= 0.10
        and float(summary["minimum_temperature_naive_force_normalized_rmse"]) >= 1.0
    )
    markov_allowed = (
        float(summary["maximum_maximum_markov_ou_correlation_error"]) <= 0.20
        and float(summary["maximum_mean_force_residual_lag_correlation"]) <= 0.20
    )
    summaries = [
        summary,
        {
            "record": "verdict",
            "relative_pmf_static_closure_allowed": float(static_allowed),
            "markovian_relative_ou_allowed": float(markov_allowed),
            "relative_force_memory_required": float(static_allowed and not markov_allowed),
            "autonomous_relative_dynamics_allowed": float(static_allowed and markov_allowed),
            "autonomous_single_particle_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        },
    ]
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curves)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)


if __name__ == "__main__":
    main()
