#!/usr/bin/env python3
"""Test continuous state-dependent memory on exact KA force histories."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_slow_force_bath import (  # noqa: E402
    fit_covariance_contracted_model,
    fit_force_hankel_basis,
    heldout_state_diagnostics,
    project_force_hankel,
)
from ka_state_dependent_memory import (  # noqa: E402
    bilinear_heldout_diagnostics,
    fit_bilinear_state_dependent_model,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_reduced_clones(
    cache_directory: Path,
    *,
    expected_clone_count: int,
    target_count: int,
    target_seed: int,
) -> list[dict[str, np.ndarray | float | str]]:
    paths = sorted(cache_directory.glob("clone_*_reduced.npz"))
    if len(paths) != expected_clone_count:
        raise ValueError("reduced cache count does not match the preregistered clone count")
    clones: list[dict[str, np.ndarray | float | str]] = []
    fixed_targets: np.ndarray | None = None
    frame_time: float | None = None
    for path in paths:
        with np.load(path) as cache:
            positions = np.asarray(cache["positions"], dtype=float)
            velocities = np.asarray(cache["velocities"], dtype=float)
            forces = np.asarray(cache["forces"], dtype=float)
            targets = np.asarray(cache["target_indices"], dtype=int)
            current_frame_time = float(cache["frame_time"])
            valid = (
                positions.shape == velocities.shape == forces.shape
                and positions.ndim == 3
                and positions.shape[1:] == (target_count, 3)
                and len(positions) >= 3
                and targets.shape == (target_count,)
                and int(cache["target_count"]) == target_count
                and int(cache["target_seed"]) == target_seed
                and float(cache["thermodynamic_claim_allowed"]) == 0.0
                and np.all(np.isfinite(positions))
                and np.all(np.isfinite(velocities))
                and np.all(np.isfinite(forces))
                and math.isclose(current_frame_time, 0.01)
            )
            if not valid:
                raise ValueError(f"invalid reduced exact-force cache: {path}")
            if fixed_targets is not None and not np.array_equal(targets, fixed_targets):
                raise ValueError("all reduced clones must use the same fixed A particles")
            if frame_time is not None and not math.isclose(current_frame_time, frame_time):
                raise ValueError("all reduced clones must share one saved-frame grid")
            fixed_targets = targets
            frame_time = current_frame_time
            clones.append(
                {
                    "positions": positions,
                    "velocities": velocities,
                    "forces": forces,
                    "target_indices": targets,
                    "frame_time": current_frame_time,
                    "trajectory_sha256": str(cache["trajectory_sha256"]),
                }
            )
    return clones


def standard_error(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--target-seed", type=int, default=20260714)
    parser.add_argument("--history-length", type=int, default=64)
    parser.add_argument("--mode-count", type=int, default=16)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-6)
    parser.add_argument("--maximum-singular-value", type=float, default=0.999)
    parser.add_argument("--minimum-velocity-r-squared", type=float, default=0.97)
    parser.add_argument("--maximum-state-correlation", type=float, default=0.20)
    parser.add_argument("--maximum-lag-correlation", type=float, default=0.70)
    parser.add_argument("--lag-improvement-fraction", type=float, default=0.80)
    args = parser.parse_args()

    if (
        args.expected_clone_count < 2
        or args.target_count < 1
        or args.history_length < 2
        or args.mode_count < 1
        or args.mode_count > args.history_length
        or args.ridge_relative <= 0.0
        or not 0.0 < args.maximum_singular_value <= 1.0
        or not 0.0 < args.lag_improvement_fraction < 1.0
    ):
        raise ValueError("invalid state-dependent memory controls")

    clones = load_reduced_clones(
        args.cache_directory,
        expected_clone_count=args.expected_clone_count,
        target_count=args.target_count,
        target_seed=args.target_seed,
    )
    model_definitions = (
        ("stationary_rank16", ()),
        ("bilinear_energy", ("bath_energy",)),
        ("bilinear_energy_power", ("bath_energy", "velocity_bath_power")),
    )
    details: list[dict[str, object]] = []
    for held_index, held in enumerate(clones):
        training = [clone for index, clone in enumerate(clones) if index != held_index]
        basis_fit = fit_force_hankel_basis(
            [np.asarray(clone["forces"]) for clone in training],
            history_length=args.history_length,
            mode_count=args.mode_count,
        )
        training_modes = [
            project_force_hankel(np.asarray(clone["forces"]), basis_fit) for clone in training
        ]
        held_modes = project_force_hankel(np.asarray(held["forces"]), basis_fit)
        training_states = [
            np.concatenate(
                [
                    np.asarray(clone["velocities"])[args.history_length - 1 :, :, None, :],
                    modes,
                ],
                axis=2,
            )
            for clone, modes in zip(training, training_modes, strict=True)
        ]
        held_state = np.concatenate(
            [
                np.asarray(held["velocities"])[args.history_length - 1 :, :, None, :],
                held_modes,
            ],
            axis=2,
        )
        for model_name, invariant_names in model_definitions:
            if invariant_names:
                model = fit_bilinear_state_dependent_model(
                    training_states,
                    invariant_names=invariant_names,
                    ridge_relative=args.ridge_relative,
                )
                diagnostic = bilinear_heldout_diagnostics(model, held_state)
                fitted_ridge = float(model["ridge"])
            else:
                model = fit_covariance_contracted_model(
                    training_states,
                    maximum_singular_value=args.maximum_singular_value,
                )
                diagnostic = heldout_state_diagnostics(model, held_state)
                fitted_ridge = 0.0
            details.append(
                {
                    "record": "held_clone",
                    "model": model_name,
                    "held_clone_index": float(held_index + 1),
                    "history_length": float(args.history_length),
                    "history_time_tau": float(
                        (args.history_length - 1) * float(held["frame_time"])
                    ),
                    "slow_mode_count": float(args.mode_count),
                    "state_mode_count": float(args.mode_count + 1),
                    "captured_force_history_variance_fraction": float(
                        basis_fit["captured_variance_fraction"]
                    ),
                    "ridge": fitted_ridge,
                    "trajectory_sha256": str(held["trajectory_sha256"]),
                    **diagnostic,
                    "fit_parameters_from_macro_observables": 0.0,
                    "autonomous_state_dependent_gle_allowed": 0.0,
                    "complete_event_clock_closure_allowed": 0.0,
                    "kramers_escape_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )

    metrics = (
        "heldout_state_r_squared",
        "heldout_velocity_r_squared",
        "heldout_velocity_residual_mean_squared",
        "maximum_held_residual_state_correlation",
        "maximum_held_residual_lag_correlation",
        "captured_force_history_variance_fraction",
    )
    summaries: list[dict[str, object]] = []
    for model_name, _ in model_definitions:
        rows = [row for row in details if row["model"] == model_name]
        summary: dict[str, object] = {
            "record": "aggregate_model",
            "model": model_name,
            "held_clone_count": float(len(rows)),
            "fit_parameters_from_macro_observables": 0.0,
            "autonomous_state_dependent_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_standard_error"] = standard_error(values)
            summary[f"maximum_{metric}"] = float(np.max(values))
        summaries.append(summary)

    baseline_rows = {
        int(float(row["held_clone_index"])): row
        for row in details
        if row["model"] == "stationary_rank16"
    }
    full_rows = {
        int(float(row["held_clone_index"])): row
        for row in details
        if row["model"] == "bilinear_energy_power"
    }
    baseline_summary = next(row for row in summaries if row["model"] == "stationary_rank16")
    full_summary = next(row for row in summaries if row["model"] == "bilinear_energy_power")
    every_fold_improves = all(
        float(full_rows[index]["maximum_held_residual_lag_correlation"])
        < float(baseline_rows[index]["maximum_held_residual_lag_correlation"])
        for index in baseline_rows
    )
    velocity_gate = float(full_summary["heldout_velocity_r_squared"]) >= args.minimum_velocity_r_squared
    state_gate = (
        float(full_summary["maximum_maximum_held_residual_state_correlation"])
        <= args.maximum_state_correlation
    )
    full_lag = float(full_summary["maximum_maximum_held_residual_lag_correlation"])
    baseline_lag = float(
        baseline_summary["maximum_maximum_held_residual_lag_correlation"]
    )
    lag_gate = (
        full_lag <= args.maximum_lag_correlation
        and full_lag <= args.lag_improvement_fraction * baseline_lag
        and every_fold_improves
    )
    promoted = velocity_gate and state_gate and lag_gate
    summaries.append(
        {
            "record": "verdict",
            "model": "bilinear_energy_power",
            "held_clone_count": float(len(clones)),
            "integrity_gate_pass": 1.0,
            "velocity_prediction_gate_pass": float(velocity_gate),
            "residual_state_gate_pass": float(state_gate),
            "residual_lag_gate_pass": float(lag_gate),
            "every_fold_lag_improves": float(every_fold_improves),
            "baseline_maximum_residual_lag_correlation": baseline_lag,
            "bilinear_maximum_residual_lag_correlation": full_lag,
            "bilinear_to_baseline_lag_ratio": full_lag / baseline_lag,
            "teacher_forced_state_dependent_memory_gate_pass": float(promoted),
            "autonomous_state_dependent_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)


if __name__ == "__main__":
    main()
