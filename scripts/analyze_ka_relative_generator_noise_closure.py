#!/usr/bin/env python3
"""Test autonomous block-noise generation for the relative generator Mori model."""

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

from analyze_ka_relative_generator_mori import (  # noqa: E402
    load_clones,
    resolved_state,
)
from ka_collective_memory import (  # noqa: E402
    discrete_mori_zwanzig_operators,
    simulate_discrete_mz_block_innovations,
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


def stationary_correlation(state: np.ndarray, maximum_lag: int) -> np.ndarray:
    output = np.empty((maximum_lag + 1, state.shape[2], state.shape[2]))
    for lag in range(maximum_lag + 1):
        left = state[lag:]
        right = state[: len(state) - lag]
        output[lag] = np.einsum("tni,tnj->ij", left, right) / (
            left.shape[0] * left.shape[1]
        )
    return output


def score_noise_generator(
    training: list[dict[str, np.ndarray | float | str]],
    held: dict[str, np.ndarray | float | str],
    *,
    memory_order: int,
    innovation_block_length: int,
    maximum_lag: int,
    simulation_count: int,
    ridge_regularization: float,
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
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
    held_state = resolved_state(
        held, bias=bias, basis="relative_phase_generator"
    )
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
    innovation = finite_memory_innovation_series(
        normalized_training, operators
    )
    if innovation_block_length > len(innovation):
        raise ValueError("innovation block exceeds the training residual series")

    rng = np.random.default_rng(seed)
    start = rng.integers(memory_order, len(normalized_training), size=simulation_count)
    source = rng.integers(normalized_training.shape[1], size=simulation_count)
    history = np.stack(
        [
            normalized_training[time - memory_order : time + 1, particle]
            for time, particle in zip(start, source, strict=True)
        ]
    )
    simulated = simulate_discrete_mz_block_innovations(
        history,
        operators,
        innovation,
        output_count=maximum_lag + 1,
        block_length=innovation_block_length,
        rng=np.random.default_rng(seed + 1),
    )
    held_correlation = stationary_correlation(normalized_held, maximum_lag)
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

    held_variance = np.mean(normalized_held**2, axis=(0, 1))
    held_excess = (
        np.mean(normalized_held**4, axis=(0, 1)) / held_variance**2 - 3.0
    )
    simulated_variance = np.mean(simulated**2, axis=(0, 1))
    simulated_excess = (
        np.mean(simulated**4, axis=(0, 1)) / simulated_variance**2 - 3.0
    )
    return (
        {
            "stationary_covariance_rmse": float(
                np.sqrt(np.mean(covariance_error**2))
            ),
            "stationary_covariance_maximum_error": float(
                np.max(np.abs(covariance_error))
            ),
            "target_correlation_rmse": float(np.sqrt(np.mean(target_error**2))),
            "target_correlation_maximum_error": float(
                np.max(np.abs(target_error))
            ),
            "marginal_excess_kurtosis_maximum_error": float(
                np.max(np.abs(simulated_excess - held_excess))
            ),
            "minimum_terminal_variance_ratio": float(
                np.min(np.diag(simulated_covariance[-1]) / np.diag(held_covariance))
            ),
            "maximum_simulated_absolute_state": float(np.max(np.abs(simulated))),
            "innovation_time_count": float(len(innovation)),
        },
        held_correlation[:, :2, :2],
        simulated_correlation[:, :2, :2],
        simulated_covariance,
    )


def aggregate(details: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "stationary_covariance_rmse",
        "stationary_covariance_maximum_error",
        "target_correlation_rmse",
        "target_correlation_maximum_error",
        "marginal_excess_kurtosis_maximum_error",
        "minimum_terminal_variance_ratio",
        "maximum_simulated_absolute_state",
        "innovation_time_count",
    )
    rows: list[dict[str, object]] = []
    for block_length in sorted(
        {int(float(row["innovation_block_length"])) for row in details}
    ):
        selected = [
            row
            for row in details
            if int(float(row["innovation_block_length"])) == block_length
        ]
        output: dict[str, object] = {
            "record": "model_aggregate",
            "protocol": selected[0]["protocol"],
            "innovation_block_length": float(block_length),
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
            float(output["maximum_stationary_covariance_rmse"]) <= 0.08
            and float(output["maximum_stationary_covariance_maximum_error"])
            <= 0.25
            and float(output["maximum_target_correlation_rmse"]) <= 0.08
            and float(output["maximum_target_correlation_maximum_error"])
            <= 0.25
            and float(output["maximum_marginal_excess_kurtosis_maximum_error"])
            <= 0.35
            and float(output["maximum_maximum_simulated_absolute_state"]) <= 20.0
        )
        output["empirical_block_noise_gate_pass"] = float(gate)
        rows.append(output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-drift-cache-directory", type=Path, required=True)
    parser.add_argument("--validation-drift-cache-directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--memory-order", type=int, default=40)
    parser.add_argument("--innovation-block-lengths", type=int, nargs="+", default=[1, 4, 16, 40, 100, 200, 400])
    parser.add_argument("--fixed-innovation-block-length", type=int, default=100)
    parser.add_argument("--maximum-lag", type=int, default=800)
    parser.add_argument("--simulation-count", type=int, default=2000)
    parser.add_argument("--ridge-regularization", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    blocks = sorted(set(args.innovation_block_lengths))
    if (
        args.memory_order < 1
        or not blocks
        or any(block < 1 for block in blocks)
        or args.fixed_innovation_block_length < 1
        or args.maximum_lag <= args.memory_order + 1
        or args.simulation_count < 100
        or args.ridge_regularization < 0.0
    ):
        raise ValueError("invalid empirical block-noise controls")

    training_clones = load_clones(args.training_drift_cache_directory)
    validation_mode = args.validation_drift_cache_directory is not None
    if validation_mode:
        held_clones = load_clones(args.validation_drift_cache_directory)
        jobs = [
            (
                training_clones,
                held,
                held_index,
                args.fixed_innovation_block_length,
            )
            for held_index, held in enumerate(held_clones, start=1)
        ]
        protocol = "fixed_block_independent_clone_validation"
    else:
        jobs = []
        for block_length in blocks:
            for held_index, held in enumerate(training_clones, start=1):
                training = [
                    clone
                    for index, clone in enumerate(training_clones, start=1)
                    if index != held_index
                ]
                jobs.append((training, held, held_index, block_length))
        protocol = "discovery_leave_one_clone_out_block_scan"

    details: list[dict[str, object]] = []
    stored: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for job_index, (training, held, held_index, block_length) in enumerate(jobs):
        simulation_seed = args.seed + 1000 * job_index
        metrics, observed, simulated, covariance = score_noise_generator(
            training,
            held,
            memory_order=args.memory_order,
            innovation_block_length=block_length,
            maximum_lag=args.maximum_lag,
            simulation_count=args.simulation_count,
            ridge_regularization=args.ridge_regularization,
            seed=simulation_seed,
        )
        details.append(
            {
                "record": "held_clone",
                "protocol": protocol,
                "held_clone_index": float(held_index),
                "training_clone_count": float(len(training)),
                "memory_order": float(args.memory_order),
                "innovation_block_length": float(block_length),
                "simulation_count": float(args.simulation_count),
                "simulation_seed": float(simulation_seed),
                "fit_uses_held_clone": 0.0,
                "hyperparameter_selection_uses_held_folds": float(
                    not validation_mode
                ),
                "thermodynamic_claim_allowed": 0.0,
                **metrics,
            }
        )
        stored[(block_length, held_index)] = (observed, simulated, covariance)

    summaries = aggregate(details)
    passing = [
        row for row in summaries if float(row["empirical_block_noise_gate_pass"]) == 1.0
    ]
    selected = min(passing, key=lambda row: float(row["innovation_block_length"])) if passing else None
    selected_block = (
        int(float(selected["innovation_block_length"]))
        if selected is not None
        else args.fixed_innovation_block_length
    )
    verdict = {
        "record": "verdict",
        "protocol": protocol,
        "selected_innovation_block_length": float(selected_block),
        "hyperparameter_selection_uses_held_folds": float(not validation_mode),
        "iid_innovation_noise_allowed": float(
            any(
                float(row["empirical_block_noise_gate_pass"]) == 1.0
                and int(float(row["innovation_block_length"])) == 1
                for row in summaries
            )
        ),
        "colored_orthogonal_noise_required": 1.0,
        "empirical_block_noise_generation_closed": float(
            selected is not None and validation_mode
        ),
        "autonomous_relative_matrix_mori_simulation_allowed": float(
            selected is not None and validation_mode
        ),
        "thermal_fdt_adjoint_audit_pass": 0.0,
        "microscopic_thermal_noise_model_closed": 0.0,
        "autonomous_single_particle_gle_allowed": 0.0,
        "complete_event_clock_closure_allowed": 0.0,
        "kramers_escape_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    summaries.append(verdict)

    curves: list[dict[str, object]] = []
    held_indices = sorted(
        key[1] for key in stored if key[0] == selected_block
    )
    for held_index in held_indices:
        observed, simulated, covariance = stored[(selected_block, held_index)]
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
                        "innovation_block_length": float(selected_block),
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
