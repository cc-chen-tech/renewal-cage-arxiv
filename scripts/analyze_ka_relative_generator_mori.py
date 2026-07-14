#!/usr/bin/env python3
"""Test a bias-centered generator-augmented discrete Mori closure."""

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
from ka_collective_memory import discrete_mori_zwanzig_operators  # noqa: E402
from ka_relative_memory import estimate_isoconfigurational_bias  # noqa: E402
from ka_relative_mori import (  # noqa: E402
    bias_centered_phase_state,
    discrete_mori_gfd_diagnostic,
    propagate_discrete_mori_correlation,
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


def load_clones(directory: Path) -> list[dict[str, np.ndarray | float | str]]:
    paths = sorted(directory.glob("clone_*_decomposed_drift.npz"))
    if not paths:
        raise ValueError(f"no decomposed-drift caches found in {directory}")
    clones = [load_drift_cache(path) for path in paths]
    targets = np.asarray(clones[0]["target_indices"], dtype=int)
    for clone in clones[1:]:
        if not np.array_equal(np.asarray(clone["target_indices"], dtype=int), targets):
            raise ValueError("all drift caches must use the same fixed targets")
    return clones


def resolved_state(
    clone: dict[str, np.ndarray | float | str],
    *,
    bias: np.ndarray,
    basis: str,
) -> np.ndarray:
    state = bias_centered_phase_state(
        np.asarray(clone["relative_position"]),
        np.asarray(clone["relative_velocity"]),
        bias=bias,
    )
    if basis == "relative_phase":
        return state
    if basis != "relative_phase_generator":
        raise ValueError(f"unknown resolved basis: {basis}")
    generator = np.asarray(clone["relative_drift"], dtype=float).reshape(
        len(state), -1, 1
    )
    return np.concatenate([state, generator], axis=2)


def stationary_correlation(state: np.ndarray, maximum_lag: int) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    correlation = np.empty((maximum_lag + 1, values.shape[2], values.shape[2]))
    for lag in range(maximum_lag + 1):
        left = values[lag:]
        right = values[: len(values) - lag]
        correlation[lag] = np.einsum("tni,tnj->ij", left, right) / (
            left.shape[0] * left.shape[1]
        )
    return correlation


def score_model(
    training: list[dict[str, np.ndarray | float | str]],
    held: dict[str, np.ndarray | float | str],
    *,
    basis: str,
    memory_order: int,
    maximum_lag: int,
    ridge_regularization: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    bias = estimate_isoconfigurational_bias(
        np.asarray([clone["relative_position"] for clone in training])
    )
    training_state = np.concatenate(
        [resolved_state(clone, bias=bias, basis=basis) for clone in training],
        axis=1,
    )
    held_state = resolved_state(held, bias=bias, basis=basis)
    mean = np.mean(training_state, axis=(0, 1), keepdims=True)
    scale = np.std(training_state, axis=(0, 1), keepdims=True)
    if np.any(scale <= 1e-12):
        raise ValueError("resolved coordinates need nonzero training variance")
    normalized_training = (training_state - mean) / scale
    normalized_held = (held_state - mean) / scale
    fit = discrete_mori_zwanzig_operators(
        normalized_training,
        memory_order=memory_order,
        ridge_regularization=ridge_regularization,
    )
    operators = np.asarray(fit["operators"])
    gfd = discrete_mori_gfd_diagnostic(normalized_held, operators)
    held_correlation = stationary_correlation(normalized_held, maximum_lag)
    predicted_correlation = propagate_discrete_mori_correlation(
        operators,
        initial_correlation=np.asarray(fit["correlation"]),
        output_count=maximum_lag + 1,
    )
    target_dimensions = 2
    extrapolation = slice(memory_order + 2, None)
    error = (
        predicted_correlation[extrapolation, :target_dimensions, :target_dimensions]
        - held_correlation[extrapolation, :target_dimensions, :target_dimensions]
    )
    held_own_bias = np.mean(np.asarray(held["relative_position"]), axis=0)
    return (
        {
            "held_bias_correlation": float(
                np.corrcoef(bias.ravel(), held_own_bias.ravel())[0, 1]
            ),
            "maximum_noise_initial_state_correlation": float(
                gfd["maximum_noise_initial_state_correlation"]
            ),
            "gfd_operator_normalized_rmse": float(
                gfd["gfd_operator_normalized_rmse"]
            ),
            "gfd_operator_shape_correlation": float(
                gfd["gfd_operator_shape_correlation"]
            ),
            "gfd_operator_maximum_absolute_error": float(
                gfd["gfd_operator_maximum_absolute_error"]
            ),
            "held_target_correlation_extrapolation_rmse": float(
                np.sqrt(np.mean(error**2))
            ),
            "held_target_correlation_extrapolation_maximum_error": float(
                np.max(np.abs(error))
            ),
            "lag_minus_correlation_condition_number": float(
                gfd["lag_minus_correlation_condition_number"]
            ),
        },
        held_correlation[:, :target_dimensions, :target_dimensions],
        predicted_correlation[:, :target_dimensions, :target_dimensions],
    )


def passes_physical_representation_gate(row: dict[str, object]) -> bool:
    return (
        float(row["maximum_maximum_noise_initial_state_correlation"]) <= 0.10
        and float(row["maximum_gfd_operator_normalized_rmse"]) <= 0.20
        and float(row["minimum_gfd_operator_shape_correlation"]) >= 0.80
        and float(row["maximum_held_target_correlation_extrapolation_rmse"])
        <= 0.08
        and float(row["maximum_held_target_correlation_extrapolation_maximum_error"])
        <= 0.20
    )


def aggregate_rows(details: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "held_bias_correlation",
        "maximum_noise_initial_state_correlation",
        "gfd_operator_normalized_rmse",
        "gfd_operator_shape_correlation",
        "gfd_operator_maximum_absolute_error",
        "held_target_correlation_extrapolation_rmse",
        "held_target_correlation_extrapolation_maximum_error",
        "lag_minus_correlation_condition_number",
    )
    summaries: list[dict[str, object]] = []
    keys = sorted(
        {(str(row["basis"]), int(float(row["memory_order"]))) for row in details}
    )
    for basis, memory_order in keys:
        selected = [
            row
            for row in details
            if row["basis"] == basis and int(float(row["memory_order"])) == memory_order
        ]
        summary: dict[str, object] = {
            "record": "model_aggregate",
            "protocol": selected[0]["protocol"],
            "basis": basis,
            "memory_order": float(memory_order),
            "held_clone_count": float(len(selected)),
            "hyperparameter_selection_uses_held_folds": selected[0][
                "hyperparameter_selection_uses_held_folds"
            ],
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            summary[metric] = float(np.mean(values))
            summary[f"maximum_{metric}"] = float(np.max(values))
            summary[f"minimum_{metric}"] = float(np.min(values))
        summary["physical_representation_gate_pass"] = float(
            passes_physical_representation_gate(summary)
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-drift-cache-directory", type=Path, required=True)
    parser.add_argument("--validation-drift-cache-directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--memory-orders", type=int, nargs="+", default=[1, 4, 16, 32, 40])
    parser.add_argument("--fixed-memory-order", type=int, default=40)
    parser.add_argument("--maximum-lag", type=int, default=800)
    parser.add_argument("--ridge-regularization", type=float, default=1e-8)
    args = parser.parse_args()
    orders = sorted(set(args.memory_orders))
    if (
        not orders
        or any(order < 1 for order in orders)
        or args.fixed_memory_order < 1
        or args.maximum_lag <= max(max(orders), args.fixed_memory_order) + 1
        or args.ridge_regularization < 0.0
    ):
        raise ValueError("invalid matrix-Mori controls")

    training_clones = load_clones(args.training_drift_cache_directory)
    validation_mode = args.validation_drift_cache_directory is not None
    if validation_mode:
        held_clones = load_clones(args.validation_drift_cache_directory)
        jobs = [
            (
                "relative_phase_generator",
                args.fixed_memory_order,
                training_clones,
                held,
                held_index,
            )
            for held_index, held in enumerate(held_clones, start=1)
        ]
        protocol = "fixed_order_independent_clone_validation"
    else:
        if len(training_clones) < 3:
            raise ValueError("discovery LOOCV requires at least three clones")
        jobs = []
        for basis in ("relative_phase", "relative_phase_generator"):
            for memory_order in orders:
                for held_index, held in enumerate(training_clones, start=1):
                    training = [
                        clone
                        for index, clone in enumerate(training_clones, start=1)
                        if index != held_index
                    ]
                    jobs.append(
                        (basis, memory_order, training, held, held_index)
                    )
        protocol = "discovery_leave_one_clone_out_memory_scan"

    details: list[dict[str, object]] = []
    stored_curves: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for basis, memory_order, training, held, held_index in jobs:
        metrics, observed, predicted = score_model(
            training,
            held,
            basis=basis,
            memory_order=memory_order,
            maximum_lag=args.maximum_lag,
            ridge_regularization=args.ridge_regularization,
        )
        details.append(
            {
                "record": "held_clone",
                "protocol": protocol,
                "basis": basis,
                "memory_order": float(memory_order),
                "held_clone_index": float(held_index),
                "training_clone_count": float(len(training)),
                "fit_uses_held_clone": 0.0,
                "hyperparameter_selection_uses_held_folds": float(not validation_mode),
                "resolved_variables": "delta_u,p,Lp" if basis.endswith("generator") else "delta_u,p",
                "generator_definition": "exact_projected_relative_Ito_drift",
                "thermodynamic_claim_allowed": 0.0,
                **metrics,
            }
        )
        stored_curves[(basis, memory_order, held_index)] = (observed, predicted)

    summaries = aggregate_rows(details)
    passing = [
        row
        for row in summaries
        if row["basis"] == "relative_phase_generator"
        and float(row["physical_representation_gate_pass"]) == 1.0
    ]
    selected = min(passing, key=lambda row: float(row["memory_order"])) if passing else None
    selected_order = (
        int(float(selected["memory_order"]))
        if selected is not None
        else args.fixed_memory_order
    )
    verdict = {
        "record": "verdict",
        "protocol": protocol,
        "selected_basis": "relative_phase_generator",
        "selected_memory_order": float(selected_order),
        "hyperparameter_selection_uses_held_folds": float(not validation_mode),
        "generator_coordinate_is_microscopic": 1.0,
        "generator_coordinate_memory_reduction_supported": float(selected is not None),
        "discovery_matrix_mori_gfd_closure_supported": float(
            selected is not None and not validation_mode
        ),
        "independent_validation_available": float(validation_mode),
        "confirmatory_matrix_mori_gfd_closure_supported": float(
            selected is not None and validation_mode
        ),
        "projected_relative_generator_mori_representation_allowed": float(
            selected is not None and validation_mode
        ),
        "thermal_fdt_adjoint_audit_pass": 0.0,
        "physical_relative_generator_gle_allowed": 0.0,
        "orthogonal_noise_generation_closed": 0.0,
        "autonomous_single_particle_gle_allowed": 0.0,
        "complete_event_clock_closure_allowed": 0.0,
        "kramers_escape_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    summaries.append(verdict)

    curve_rows: list[dict[str, object]] = []
    selected_key_order = selected_order
    for held_index in sorted(
        key[2]
        for key in stored_curves
        if key[0] == "relative_phase_generator" and key[1] == selected_key_order
    ):
        observed, predicted = stored_curves[
            ("relative_phase_generator", selected_key_order, held_index)
        ]
        for lag in range(args.maximum_lag + 1):
            for left, right, label in (
                (0, 0, "uu"),
                (0, 1, "up"),
                (1, 0, "pu"),
                (1, 1, "pp"),
            ):
                curve_rows.append(
                    {
                        "protocol": protocol,
                        "held_clone_index": float(held_index),
                        "basis": "relative_phase_generator",
                        "memory_order": float(selected_key_order),
                        "lag_frames": float(lag),
                        "correlation_component": label,
                        "observed": float(observed[lag, left, right]),
                        "predicted": float(predicted[lag, left, right]),
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )

    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)


if __name__ == "__main__":
    main()
