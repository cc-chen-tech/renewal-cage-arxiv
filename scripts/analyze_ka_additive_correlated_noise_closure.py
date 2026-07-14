#!/usr/bin/env python3
"""Test held-clone constant closures of projected center-relative noise."""

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

from analyze_ka_projected_ito_innovations import (  # noqa: E402
    clone_diagnostic,
    load_drift_cache,
)
from ka_projected_innovation import (  # noqa: E402
    fit_constant_joint_covariance,
    multivariate_noise_covariance_diagnostic,
)


MODEL_MAP = {
    "exact_configuration_dependent": None,
    "constant_full": "full",
    "constant_block_isotropic": "block_isotropic",
    "constant_block_uncorrelated": "block_isotropic_uncorrelated",
    "constant_single_scalar": "single_scalar",
}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def passes_local_gate(row: dict[str, object]) -> bool:
    return (
        0.8 <= float(row["trace_variance_ratio"]) <= 1.2
        and 0.8 <= float(row["mean_squared_mahalanobis_per_dimension"]) <= 1.2
        and float(row["maximum_absolute_whitened_mean"]) <= 0.05
        and float(row["maximum_absolute_whitened_covariance_error"]) <= 0.10
        and float(row["maximum_absolute_whitened_lag1_correlation"]) <= 0.05
        and float(row["maximum_absolute_whitened_state_correlation"]) <= 0.10
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--covariance-cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--frame-time", type=float, default=0.01)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--scheme", default="adams_bashforth2")
    parser.add_argument("--maximum-exact-metric-difference", type=float, default=0.02)
    args = parser.parse_args()
    if (
        args.expected_clone_count < 2
        or args.target_count < 1
        or args.frame_time <= 0.0
        or args.stride < 1
        or args.scheme != "adams_bashforth2"
        or args.maximum_exact_metric_difference <= 0.0
    ):
        raise ValueError("invalid additive-noise closure controls")

    clones: list[dict[str, np.ndarray | float | str]] = []
    fixed_targets: np.ndarray | None = None
    for index in range(1, args.expected_clone_count + 1):
        drift = load_drift_cache(
            args.drift_cache_directory / f"clone_{index:03d}_decomposed_drift.npz"
        )
        targets = np.asarray(drift["target_indices"], dtype=int)
        if len(targets) != args.target_count:
            raise ValueError("target count does not match the noise-closure protocol")
        if fixed_targets is None:
            fixed_targets = targets
        elif not np.array_equal(targets, fixed_targets):
            raise ValueError("all clones must use the same fixed targets")
        covariance_path = (
            args.covariance_cache_directory
            / f"clone_{index:03d}_projected_covariance.npz"
        )
        with np.load(covariance_path) as cache:
            if (
                not np.array_equal(np.asarray(cache["target_indices"], dtype=int), targets)
                or str(cache["trajectory_sha256"]) != str(drift["trajectory_sha256"])
                or float(cache["thermodynamic_claim_allowed"]) != 0.0
            ):
                raise ValueError("projected covariance cache does not align with drift cache")
            joint = np.asarray(cache["joint_noise_covariance_rate"], dtype=float)
        expected_shape = (*np.asarray(drift["relative_velocity"]).shape[:2], 6, 6)
        if joint.shape != expected_shape:
            raise ValueError("projected covariance path has the wrong shape")
        clones.append({**drift, "joint_noise_covariance_rate": joint})

    metric_names = (
        "trace_variance_ratio",
        "mean_squared_mahalanobis_per_dimension",
        "maximum_absolute_whitened_mean",
        "maximum_absolute_whitened_covariance_error",
        "maximum_absolute_whitened_component_excess_kurtosis",
        "maximum_absolute_whitened_lag1_correlation",
        "maximum_absolute_whitened_state_correlation",
        "minimum_integrated_covariance_eigenvalue",
        "sample_count",
    )
    details: list[dict[str, object]] = []
    pooled_inputs: dict[str, dict[str, list[np.ndarray]]] = {
        model: {"residual": [], "covariance": [], "state": []} for model in MODEL_MAP
    }
    for held_index, held in enumerate(clones):
        training_covariance = np.concatenate(
            [
                np.asarray(clone["joint_noise_covariance_rate"]).reshape(-1, 6, 6)
                for index, clone in enumerate(clones)
                if index != held_index
            ]
        )
        for model_name, fit_name in MODEL_MAP.items():
            if fit_name is None:
                fitted = None
                covariance_path = np.asarray(held["joint_noise_covariance_rate"])
            else:
                fitted = fit_constant_joint_covariance(
                    training_covariance,
                    model=fit_name,
                )
                covariance_path = np.broadcast_to(
                    fitted,
                    np.asarray(held["joint_noise_covariance_rate"]).shape,
                )
            diagnostic, residual, covariance, starting = clone_diagnostic(
                {**held, "joint_noise_covariance_rate": covariance_path},
                frame_time=args.frame_time,
                stride=args.stride,
                scheme=args.scheme,
            )
            pooled_inputs[model_name]["residual"].append(residual)
            pooled_inputs[model_name]["covariance"].append(covariance)
            pooled_inputs[model_name]["state"].append(starting)
            details.append(
                {
                    "record": "held_clone",
                    "model": model_name,
                    "held_clone_index": float(held_index + 1),
                    **{key: float(diagnostic[key]) for key in metric_names},
                    "local_noise_gate_pass": float(passes_local_gate(diagnostic)),
                    "center_variance_rate": (
                        math.nan if fitted is None else float(np.trace(fitted[:3, :3]) / 3.0)
                    ),
                    "center_relative_cross_rate": (
                        math.nan if fitted is None else float(np.trace(fitted[:3, 3:]) / 3.0)
                    ),
                    "relative_variance_rate": (
                        math.nan if fitted is None else float(np.trace(fitted[3:, 3:]) / 3.0)
                    ),
                    "fit_uses_held_clone": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )

    summaries: list[dict[str, object]] = []
    for model_name in MODEL_MAP:
        inputs = pooled_inputs[model_name]
        diagnostic = multivariate_noise_covariance_diagnostic(
            np.concatenate(inputs["residual"], axis=0),
            np.concatenate(inputs["covariance"], axis=0),
            starting_state=np.concatenate(inputs["state"], axis=0),
        )
        held_rows = [row for row in details if row["model"] == model_name]
        summaries.append(
            {
                "record": "pooled_model",
                "model": model_name,
                "held_clone_count": float(len(held_rows)),
                **{key: float(diagnostic[key]) for key in metric_names},
                "every_held_clone_local_noise_gate_pass": float(
                    all(float(row["local_noise_gate_pass"]) == 1.0 for row in held_rows)
                ),
                "mean_center_variance_rate": float(
                    np.nanmean([float(row["center_variance_rate"]) for row in held_rows])
                )
                if model_name != "exact_configuration_dependent"
                else math.nan,
                "mean_center_relative_cross_rate": float(
                    np.nanmean(
                        [float(row["center_relative_cross_rate"]) for row in held_rows]
                    )
                )
                if model_name != "exact_configuration_dependent"
                else math.nan,
                "mean_relative_variance_rate": float(
                    np.nanmean([float(row["relative_variance_rate"]) for row in held_rows])
                )
                if model_name != "exact_configuration_dependent"
                else math.nan,
                "thermodynamic_claim_allowed": 0.0,
            }
        )

    all_covariance = np.concatenate(
        [np.asarray(clone["joint_noise_covariance_rate"]).reshape(-1, 6, 6) for clone in clones]
    )
    mean_covariance = np.mean(all_covariance, axis=0)
    relative_frobenius_rms = float(
        np.sqrt(np.mean(np.sum((all_covariance - mean_covariance) ** 2, axis=(1, 2))))
        / np.linalg.norm(mean_covariance)
    )
    trace = np.trace(all_covariance, axis1=1, axis2=2)
    trace_cv = float(np.std(trace) / np.mean(trace))
    by_model = {row["model"]: row for row in summaries}
    exact = by_model["exact_configuration_dependent"]
    primary = by_model["constant_block_isotropic"]
    uncorrelated = by_model["constant_block_uncorrelated"]
    comparison_metrics = (
        "trace_variance_ratio",
        "mean_squared_mahalanobis_per_dimension",
        "maximum_absolute_whitened_covariance_error",
        "maximum_absolute_whitened_lag1_correlation",
        "maximum_absolute_whitened_state_correlation",
    )
    maximum_exact_difference = max(
        abs(float(primary[key]) - float(exact[key])) for key in comparison_metrics
    )
    allowed = (
        passes_local_gate(primary)
        and float(primary["every_held_clone_local_noise_gate_pass"]) == 1.0
        and maximum_exact_difference <= args.maximum_exact_metric_difference
        and float(uncorrelated["maximum_absolute_whitened_covariance_error"]) > 0.50
    )
    summaries.append(
        {
            "record": "verdict",
            "model": "constant_block_isotropic",
            "exact_covariance_relative_frobenius_rms": relative_frobenius_rms,
            "exact_covariance_trace_coefficient_of_variation": trace_cv,
            "maximum_primary_to_exact_metric_difference": maximum_exact_difference,
            "uncorrelated_null_covariance_error": float(
                uncorrelated["maximum_absolute_whitened_covariance_error"]
            ),
            "constant_correlated_projected_noise_allowed": float(allowed),
            "configuration_dependent_noise_required": float(not allowed),
            "center_relative_cross_noise_required": float(allowed),
            "autonomous_drift_closure_allowed": 0.0,
            "autonomous_single_particle_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)


if __name__ == "__main__":
    main()
