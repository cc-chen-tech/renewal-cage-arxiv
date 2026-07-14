#!/usr/bin/env python3
"""Test white-driven Markov baths for relative Mori innovations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_relative_generator_mori import load_clones, resolved_state  # noqa: E402
from analyze_ka_relative_generator_noise_closure import stationary_correlation  # noqa: E402
from ka_collective_memory import discrete_mori_zwanzig_operators  # noqa: E402
from ka_markov_bath import (  # noqa: E402
    fit_stationary_vector_autoregression,
    simulate_mori_with_var_bath,
    vector_autoregressive_residual,
)
from ka_relative_memory import estimate_isoconfigurational_bias  # noqa: E402
from ka_relative_mori import finite_memory_innovation_series  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_fold(
    training: list[dict[str, np.ndarray | float | str]],
    held: dict[str, np.ndarray | float | str],
    *,
    memory_order: int,
    ridge_regularization: float,
) -> dict[str, np.ndarray]:
    bias = estimate_isoconfigurational_bias(
        np.asarray([clone["relative_position"] for clone in training])
    )
    training_state = np.concatenate(
        [
            resolved_state(clone, bias=bias, basis="relative_phase_generator")
            for clone in training
        ],
        axis=1,
    )
    held_state = resolved_state(held, bias=bias, basis="relative_phase_generator")
    mean = np.mean(training_state, axis=(0, 1), keepdims=True)
    scale = np.std(training_state, axis=(0, 1), keepdims=True)
    normalized_training = (training_state - mean) / scale
    normalized_held = (held_state - mean) / scale
    fit = discrete_mori_zwanzig_operators(
        normalized_training,
        memory_order=memory_order,
        ridge_regularization=ridge_regularization,
    )
    operators = np.asarray(fit["operators"])
    return {
        "training_state": normalized_training,
        "held_state": normalized_held,
        "operators": operators,
        "innovation": finite_memory_innovation_series(
            normalized_training, operators
        ),
        "held_innovation": finite_memory_innovation_series(
            normalized_held, operators
        ),
    }


def score_var_bath(
    prepared: dict[str, np.ndarray],
    *,
    bath_order: int,
    maximum_lag: int,
    simulation_count: int,
    fit_source_count: int,
    var_ridge_regularization: float,
    white_driving_distribution: str,
    fit_source_seed: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    training_state = prepared["training_state"]
    held_state = prepared["held_state"]
    operators = prepared["operators"]
    innovation = prepared["innovation"]
    held_innovation = prepared["held_innovation"]
    fit_rng = np.random.default_rng(fit_source_seed)
    rng = np.random.default_rng(seed)
    source_count = min(fit_source_count, innovation.shape[1])
    fit_sources = np.sort(
        fit_rng.choice(innovation.shape[1], size=source_count, replace=False)
    )
    fitted = fit_stationary_vector_autoregression(
        innovation[:, fit_sources],
        order=bath_order,
        ridge_regularization=var_ridge_regularization,
    )
    coefficients = np.asarray(fitted["coefficients"])
    bath_mean = np.asarray(fitted["mean"])
    white_covariance = np.asarray(fitted["white_noise_covariance"])
    white_residual = vector_autoregressive_residual(
        innovation[:, fit_sources], coefficients, mean=bath_mean
    )
    held_white_residual = vector_autoregressive_residual(
        held_innovation, coefficients, mean=bath_mean
    )
    white_variance = np.sqrt(
        np.maximum(np.diag(white_covariance), 1e-30)
    )
    white_lag_one = np.einsum(
        "tni,tnj->ij", white_residual[1:], white_residual[:-1]
    ) / ((len(white_residual) - 1) * white_residual.shape[1])
    white_lag_one_correlation = white_lag_one / np.outer(
        white_variance, white_variance
    )
    white_excess_kurtosis = (
        np.mean(white_residual**4, axis=(0, 1)) / white_variance**4 - 3.0
    )
    maximum_white_residual_correlation = 0.0
    held_white_variance = np.sqrt(
        np.maximum(np.mean(held_white_residual**2, axis=(0, 1)), 1e-30)
    )
    maximum_held_white_residual_correlation = 0.0
    maximum_white_lag = min(40, len(white_residual) - 1)
    for lag in range(1, maximum_white_lag + 1):
        covariance = np.einsum(
            "tni,tnj->ij", white_residual[lag:], white_residual[:-lag]
        ) / ((len(white_residual) - lag) * white_residual.shape[1])
        correlation = covariance / np.outer(white_variance, white_variance)
        maximum_white_residual_correlation = max(
            maximum_white_residual_correlation,
            float(np.max(np.abs(correlation))),
        )
    maximum_held_white_lag = min(40, len(held_white_residual) - 1)
    for lag in range(1, maximum_held_white_lag + 1):
        covariance = np.einsum(
            "tni,tnj->ij",
            held_white_residual[lag:],
            held_white_residual[:-lag],
        ) / ((len(held_white_residual) - lag) * held_white_residual.shape[1])
        correlation = covariance / np.outer(
            held_white_variance, held_white_variance
        )
        maximum_held_white_residual_correlation = max(
            maximum_held_white_residual_correlation,
            float(np.max(np.abs(correlation))),
        )

    memory_order = len(operators) - 1
    minimum_current = memory_order + bath_order
    if minimum_current >= len(training_state) - 1:
        raise ValueError("Mori and bath histories leave no valid initial state")
    current = rng.integers(
        minimum_current, len(training_state) - 1, size=simulation_count
    )
    source = rng.integers(training_state.shape[1], size=simulation_count)
    state_history = np.stack(
        [
            training_state[time - memory_order : time + 1, particle]
            for time, particle in zip(current, source, strict=True)
        ]
    )
    innovation_history = np.stack(
        [
            innovation[
                time - memory_order - bath_order : time - memory_order,
                particle,
            ]
            for time, particle in zip(current, source, strict=True)
        ]
    )
    simulated, _ = simulate_mori_with_var_bath(
        state_history,
        operators,
        innovation_history,
        coefficients,
        innovation_mean=bath_mean,
        white_noise_covariance=white_covariance,
        white_noise_pool=(
            white_residual
            if white_driving_distribution == "empirical"
            else None
        ),
        output_count=maximum_lag + 1,
        rng=np.random.default_rng(seed + 1),
    )

    held_correlation = stationary_correlation(held_state, maximum_lag)
    simulated_correlation = np.einsum(
        "nti,nj->tij", simulated, simulated[:, 0]
    ) / simulation_count
    held_covariance = held_correlation[0]
    simulated_covariance = np.einsum(
        "nti,ntj->tij", simulated, simulated
    ) / simulation_count
    covariance_error = simulated_covariance - held_covariance
    target_error = (
        simulated_correlation[:, :2, :2] - held_correlation[:, :2, :2]
    )
    held_variance = np.mean(held_state**2, axis=(0, 1))
    held_excess = np.mean(held_state**4, axis=(0, 1)) / held_variance**2 - 3.0
    simulated_variance = np.mean(simulated**2, axis=(0, 1))
    simulated_excess = (
        np.mean(simulated**4, axis=(0, 1)) / simulated_variance**2 - 3.0
    )
    metrics = {
        "var_spectral_radius": float(fitted["spectral_radius"]),
        "minimum_white_noise_covariance_eigenvalue": float(
            fitted["minimum_noise_covariance_eigenvalue"]
        ),
        "maximum_white_residual_lag_one_correlation": float(
            np.max(np.abs(white_lag_one_correlation))
        ),
        "maximum_white_residual_correlation_through_40_frames": (
            maximum_white_residual_correlation
        ),
        "maximum_absolute_white_residual_excess_kurtosis": float(
            np.max(np.abs(white_excess_kurtosis))
        ),
        "maximum_held_white_residual_correlation_through_40_frames": (
            maximum_held_white_residual_correlation
        ),
        "stationary_covariance_rmse": float(
            np.sqrt(np.mean(covariance_error**2))
        ),
        "stationary_covariance_maximum_error": float(
            np.max(np.abs(covariance_error))
        ),
        "target_correlation_rmse": float(np.sqrt(np.mean(target_error**2))),
        "target_correlation_maximum_error": float(np.max(np.abs(target_error))),
        "marginal_excess_kurtosis_maximum_error": float(
            np.max(np.abs(simulated_excess - held_excess))
        ),
        "minimum_terminal_variance_ratio": float(
            np.min(np.diag(simulated_covariance[-1]) / np.diag(held_covariance))
        ),
        "maximum_simulated_absolute_state": float(np.max(np.abs(simulated))),
        "var_fit_source_count": float(source_count),
        "innovation_time_count": float(len(innovation)),
    }
    return metrics, held_correlation[:, :2, :2], simulated_correlation[:, :2, :2], simulated_covariance


def aggregate(details: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "var_spectral_radius",
        "minimum_white_noise_covariance_eigenvalue",
        "maximum_white_residual_lag_one_correlation",
        "maximum_white_residual_correlation_through_40_frames",
        "maximum_absolute_white_residual_excess_kurtosis",
        "maximum_held_white_residual_correlation_through_40_frames",
        "stationary_covariance_rmse",
        "stationary_covariance_maximum_error",
        "target_correlation_rmse",
        "target_correlation_maximum_error",
        "marginal_excess_kurtosis_maximum_error",
        "minimum_terminal_variance_ratio",
        "maximum_simulated_absolute_state",
    )
    rows: list[dict[str, object]] = []
    for order in sorted({int(float(row["bath_order"])) for row in details}):
        selected = [
            row for row in details if int(float(row["bath_order"])) == order
        ]
        output: dict[str, object] = {
            "record": "model_aggregate",
            "protocol": selected[0]["protocol"],
            "bath_order": float(order),
            "held_clone_count": float(len(selected)),
            "hyperparameter_selection_uses_held_folds": selected[0][
                "hyperparameter_selection_uses_held_folds"
            ],
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            output[metric] = float(np.mean(values))
            output[f"maximum_{metric}"] = float(np.max(values))
            output[f"minimum_{metric}"] = float(np.min(values))
        gate = (
            float(output["maximum_var_spectral_radius"]) < 1.0
            and float(output["minimum_minimum_white_noise_covariance_eigenvalue"])
            >= -1e-10
            and float(output["maximum_maximum_white_residual_lag_one_correlation"])
            <= 0.10
            and float(
                output[
                    "maximum_maximum_white_residual_correlation_through_40_frames"
                ]
            )
            <= 0.05
            and float(
                output[
                    "maximum_maximum_held_white_residual_correlation_through_40_frames"
                ]
            )
            <= 0.05
            and float(output["maximum_stationary_covariance_rmse"]) <= 0.08
            and float(output["maximum_stationary_covariance_maximum_error"])
            <= 0.25
            and float(output["maximum_target_correlation_rmse"]) <= 0.08
            and float(output["maximum_target_correlation_maximum_error"])
            <= 0.25
            and float(output["maximum_marginal_excess_kurtosis_maximum_error"])
            <= 0.35
            and float(output["maximum_maximum_simulated_absolute_state"]) <= 20.0
        )
        output["markov_bath_gate_pass"] = float(gate)
        rows.append(output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-drift-cache-directory", type=Path, required=True)
    parser.add_argument("--validation-drift-cache-directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--memory-order", type=int, default=40)
    parser.add_argument("--bath-orders", type=int, nargs="+", default=[1, 4, 8, 16, 40])
    parser.add_argument("--fixed-bath-order", type=int, default=40)
    parser.add_argument("--maximum-lag", type=int, default=800)
    parser.add_argument("--simulation-count", type=int, default=2000)
    parser.add_argument("--fit-source-count", type=int, default=8192)
    parser.add_argument("--ridge-regularization", type=float, default=1e-8)
    parser.add_argument("--var-ridge-regularization", type=float, default=1e-6)
    parser.add_argument(
        "--white-driving-distribution",
        choices=("gaussian", "empirical"),
        default="gaussian",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    orders = sorted(set(args.bath_orders))
    if (
        args.memory_order < 1
        or not orders
        or any(order < 1 for order in orders)
        or args.fixed_bath_order < 1
        or args.maximum_lag <= args.memory_order + 1
        or args.simulation_count < 100
        or args.fit_source_count < 100
        or args.ridge_regularization < 0.0
        or args.var_ridge_regularization < 0.0
    ):
        raise ValueError("invalid Markov-bath controls")

    training_clones = load_clones(args.training_drift_cache_directory)
    validation_mode = args.validation_drift_cache_directory is not None
    if validation_mode:
        held_clones = load_clones(args.validation_drift_cache_directory)
        folds = [(training_clones, held, index) for index, held in enumerate(held_clones, start=1)]
        active_orders = [args.fixed_bath_order]
        protocol = (
            f"fixed_{args.white_driving_distribution}_markov_bath_"
            "independent_clone_validation"
        )
    else:
        folds = []
        for held_index, held in enumerate(training_clones, start=1):
            training = [
                clone
                for index, clone in enumerate(training_clones, start=1)
                if index != held_index
            ]
            folds.append((training, held, held_index))
        active_orders = orders
        protocol = (
            "discovery_leave_one_clone_out_"
            f"{args.white_driving_distribution}_markov_bath_scan"
        )

    details: list[dict[str, object]] = []
    stored: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold_index, (training, held, held_index) in enumerate(folds):
        prepared = prepare_fold(
            training,
            held,
            memory_order=args.memory_order,
            ridge_regularization=args.ridge_regularization,
        )
        for order_index, bath_order in enumerate(active_orders):
            fit_source_seed = args.seed + 10000 * fold_index
            simulation_seed = args.seed + 10000 * fold_index + 1000 * order_index
            metrics, observed, simulated, covariance = score_var_bath(
                prepared,
                bath_order=bath_order,
                maximum_lag=args.maximum_lag,
                simulation_count=args.simulation_count,
                fit_source_count=args.fit_source_count,
                var_ridge_regularization=args.var_ridge_regularization,
                white_driving_distribution=args.white_driving_distribution,
                fit_source_seed=fit_source_seed,
                seed=simulation_seed,
            )
            details.append(
                {
                    "record": "held_clone",
                    "protocol": protocol,
                    "held_clone_index": float(held_index),
                    "training_clone_count": float(len(training)),
                    "memory_order": float(args.memory_order),
                    "bath_order": float(bath_order),
                    "simulation_count": float(args.simulation_count),
                    "fit_source_seed": float(fit_source_seed),
                    "simulation_seed": float(simulation_seed),
                    "fit_uses_held_clone": 0.0,
                    "hyperparameter_selection_uses_held_folds": float(
                        not validation_mode
                    ),
                    "white_driving_distribution": args.white_driving_distribution,
                    "thermodynamic_claim_allowed": 0.0,
                    **metrics,
                }
            )
            stored[(bath_order, held_index)] = (observed, simulated, covariance)

    summaries = aggregate(details)
    passing = [
        row for row in summaries if float(row["markov_bath_gate_pass"]) == 1.0
    ]
    selected = min(passing, key=lambda row: float(row["bath_order"])) if passing else None
    selected_order = (
        int(float(selected["bath_order"]))
        if selected is not None
        else (args.fixed_bath_order if validation_mode else max(active_orders))
    )
    summaries.append(
        {
            "record": "verdict",
            "protocol": protocol,
            "selected_bath_order": float(selected_order),
            "hyperparameter_selection_uses_held_folds": float(not validation_mode),
            "white_driving_distribution": args.white_driving_distribution,
            "stable_white_driven_markov_bath_found": float(selected is not None),
            "finite_dimensional_discrete_markov_bath_supported": float(
                selected is not None
            ),
            "autonomous_relative_matrix_mori_simulation_allowed": float(
                selected is not None and validation_mode
            ),
            "empirical_non_gaussian_white_driving_supported": float(
                selected is not None
                and args.white_driving_distribution == "empirical"
            ),
            "gaussian_markov_bath_generation_closed": float(
                selected is not None
                and validation_mode
                and args.white_driving_distribution == "gaussian"
            ),
            "empirical_white_markov_bath_generation_closed": float(
                selected is not None
                and validation_mode
                and args.white_driving_distribution == "empirical"
            ),
            "continuous_time_ou_embedding_audited": 0.0,
            "thermal_fdt_adjoint_audit_pass": 0.0,
            "microscopic_thermal_noise_model_closed": 0.0,
            "autonomous_single_particle_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )

    curves: list[dict[str, object]] = []
    for held_index in sorted(key[1] for key in stored if key[0] == selected_order):
        observed, simulated, covariance = stored[(selected_order, held_index)]
        for lag in range(args.maximum_lag + 1):
            for left, right, label in (
                (0, 0, "uu"),
                (0, 1, "up"),
                (1, 0, "pu"),
                (1, 1, "pp"),
            ):
                curves.append(
                    {
                        "held_clone_index": float(held_index),
                        "bath_order": float(selected_order),
                        "lag_frames": float(lag),
                        "correlation_component": label,
                        "observed_correlation": float(observed[lag, left, right]),
                        "simulated_correlation": float(simulated[lag, left, right]),
                        "simulated_marginal_covariance": float(
                            covariance[lag, left, right]
                        ),
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )

    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curves)


if __name__ == "__main__":
    main()
