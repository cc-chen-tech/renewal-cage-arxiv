#!/usr/bin/env python3
"""Identify position-dependent memory kernels on frozen microscopic KA clones."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_projected_ito_innovations import load_drift_cache  # noqa: E402
from ka_position_dependent_kernel import (  # noqa: E402
    assemble_mz_correlation_system,
    classify_position_dependent_kernel_gate,
    fit_radial_basis_scale,
    fit_real_pole_model,
    fit_two_position_prony_model,
    force_autocovariance,
    held_kernel_diagnostics,
    predict_mz_drift,
    predict_real_pole_drift,
    predict_two_position_prony_drift,
    select_decay_rates_from_memory,
    solve_regularized_mz_kernel,
    two_position_fdt_covariance,
)


MEMORY_SUPPORTS = (4, 16, 40, 100)
AUXILIARY_RANKS = (1, 2, 4, 8)
RIDGE_GRID = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
DECAY_GRID_COUNT = 32
DECAY_MINIMUM = 0.05
DECAY_MAXIMUM = 50.0
TEMPERATURE = 0.58
MAXIMUM_DIAGNOSTIC_LAG = 40
PERMUTATION_SEED = 20260719

NONPARAMETRIC_MODELS = (
    "stationary_scalar_nonparametric_volterra",
    "finite_basis_mz_position_kernel",
    "time_permuted_position_null",
)

AUXILIARY_MODELS = (
    "past_position_real_pole",
    "two_position_positive_prony",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_kernel_clones(
    drift_cache_directory: Path,
    trajectory_directory: Path,
    *,
    expected_clone_count: int,
    target_count: int,
) -> list[dict[str, object]]:
    """Load one exact four-clone grid and recompute every upstream hash."""

    if (
        isinstance(expected_clone_count, bool)
        or not isinstance(expected_clone_count, (int, np.integer))
        or expected_clone_count < 2
        or isinstance(target_count, bool)
        or not isinstance(target_count, (int, np.integer))
        or target_count < 1
    ):
        raise ValueError("clone and target counts must be positive integers")
    drift_directory = Path(drift_cache_directory)
    trajectory_root = Path(trajectory_directory)
    expected_drift_paths = [
        drift_directory / f"clone_{index:03d}_decomposed_drift.npz"
        for index in range(1, expected_clone_count + 1)
    ]
    actual_drift_paths = sorted(drift_directory.glob("clone_*_decomposed_drift.npz"))
    if actual_drift_paths != expected_drift_paths:
        raise ValueError("drift caches must form the exact frozen clone grid")
    clones: list[dict[str, object]] = []
    fixed_targets: np.ndarray | None = None
    fixed_shape: tuple[int, ...] | None = None
    for index, drift_path in enumerate(expected_drift_paths, start=1):
        trajectory_path = (
            trajectory_root / f"clone_{index:03d}" / "trajectory.lammpstrj"
        )
        if not trajectory_path.is_file():
            raise ValueError("raw trajectory grid is incomplete")
        drift = load_drift_cache(drift_path)
        trajectory_hash = file_sha256(trajectory_path)
        cached_hash = str(np.asarray(drift["trajectory_sha256"]).item())
        if trajectory_hash != cached_hash:
            raise ValueError("raw trajectory hash does not match drift cache")
        targets = np.asarray(drift["target_indices"], dtype=int)
        positions = np.asarray(drift["relative_position"], dtype=float)
        if (
            targets.shape != (target_count,)
            or len(np.unique(targets)) != target_count
            or np.any(targets < 0)
            or positions.shape[1:] != (target_count, 3)
            or np.any(~np.isfinite(positions))
            or any(
                np.any(~np.isfinite(np.asarray(drift[key], dtype=float)))
                for key in (
                    "relative_velocity",
                    "center_velocity",
                    "relative_drift",
                    "center_drift",
                )
            )
        ):
            raise ValueError("drift cache target or finite-state grid is invalid")
        if fixed_targets is None:
            fixed_targets = targets.copy()
            fixed_shape = positions.shape
        elif not np.array_equal(targets, fixed_targets) or positions.shape != fixed_shape:
            raise ValueError("all drift caches must share targets and frame grid")
        clones.append(
            {
                **drift,
                "target_indices": targets,
                "trajectory_sha256": trajectory_hash,
                "trajectory_path": str(trajectory_path.resolve()),
                "drift_cache_sha256": file_sha256(drift_path),
                "drift_cache_path": str(drift_path.resolve()),
                "clone_index": float(index),
                "fit_uses_held_clone": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return clones


def _kernel_clone_arrays(
    clones: list[dict[str, object]],
    clone_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [clones[index] for index in clone_indices]
    position = np.stack(
        [np.asarray(clone["relative_position"], dtype=float) for clone in selected]
    )
    velocity = np.stack(
        [np.asarray(clone["relative_velocity"], dtype=float) for clone in selected]
    )
    acceleration = np.stack(
        [np.asarray(clone["relative_drift"], dtype=float) for clone in selected]
    )
    if (
        position.ndim != 4
        or position.shape != velocity.shape
        or position.shape != acceleration.shape
        or position.shape[-1] != 3
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(acceleration))
    ):
        raise ValueError("kernel clone arrays must be finite and aligned")
    return position, velocity, acceleration


def _permuted_memory_positions(
    clones: list[dict[str, object]],
    clone_indices: tuple[int, ...],
    *,
    permutation_seed: int,
) -> np.ndarray:
    permuted = []
    for index in clone_indices:
        clone = clones[index]
        position = np.asarray(clone["relative_position"], dtype=float)
        clone_label = int(float(clone.get("clone_index", index + 1)))
        rng = np.random.default_rng(permutation_seed + 1009 * clone_label)
        permuted.append(position[rng.permutation(position.shape[0])])
    return np.stack(permuted)


def _normalized_drift_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    target = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if target.shape != estimate.shape or np.any(~np.isfinite(target - estimate)):
        raise ValueError("validation drift arrays must be finite and aligned")
    scale = math.sqrt(float(np.mean(target**2)))
    if scale <= 0.0:
        raise ValueError("validation drift must have positive scale")
    return math.sqrt(float(np.mean((target - estimate) ** 2))) / scale


def select_nonparametric_hierarchy(
    clones: list[dict[str, object]],
    *,
    supports: tuple[int, ...],
    ridge_grid: tuple[float, ...],
    permutation_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Select M1, M2, and M5 using nested whole-clone folds only."""

    if len(clones) != 4:
        raise ValueError("nonparametric hierarchy requires exactly four clones")
    full_position, _, _ = _kernel_clone_arrays(clones, (0, 1, 2, 3))
    support_grid = tuple(int(value) for value in supports)
    ridges = tuple(float(value) for value in ridge_grid)
    if (
        len(support_grid) < 1
        or len(set(support_grid)) != len(support_grid)
        or any(value < 1 or 2 * value - 1 > full_position.shape[1] for value in support_grid)
        or len(ridges) < 1
        or len(set(ridges)) != len(ridges)
        or any(not math.isfinite(value) or value < 0.0 for value in ridges)
        or isinstance(permutation_seed, bool)
        or not isinstance(permutation_seed, (int, np.integer))
    ):
        raise ValueError("nested nonparametric grids must be finite and resolved")

    candidates: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for outer_held in range(4):
        outer_training = tuple(index for index in range(4) if index != outer_held)
        for model in NONPARAMETRIC_MODELS:
            basis_indices = (0,) if model == NONPARAMETRIC_MODELS[0] else (0, 1, 2)
            score_grid: dict[tuple[int, float], list[float]] = {
                (support, ridge): []
                for support in support_grid
                for ridge in ridges
            }
            fold_sets = []
            for inner_held in outer_training:
                inner_training = tuple(
                    index for index in outer_training if index != inner_held
                )
                fold_sets.append(
                    f"{','.join(str(index + 1) for index in inner_training)}>{inner_held + 1}"
                )
                fit_position, fit_velocity, fit_acceleration = _kernel_clone_arrays(
                    clones,
                    inner_training,
                )
                held_position, held_velocity, held_acceleration = _kernel_clone_arrays(
                    clones,
                    (inner_held,),
                )
                scale = fit_radial_basis_scale(fit_position)
                fit_memory_position = None
                held_memory_position = None
                if model == NONPARAMETRIC_MODELS[2]:
                    fit_memory_position = _permuted_memory_positions(
                        clones,
                        inner_training,
                        permutation_seed=permutation_seed,
                    )
                    held_memory_position = _permuted_memory_positions(
                        clones,
                        (inner_held,),
                        permutation_seed=permutation_seed,
                    )
                for support in support_grid:
                    system = assemble_mz_correlation_system(
                        fit_position,
                        fit_velocity,
                        fit_acceleration,
                        scale=scale,
                        support=support,
                        basis_indices=basis_indices,
                        memory_position=fit_memory_position,
                        projection_lag_count=support,
                    )
                    for ridge in ridges:
                        try:
                            fitted = solve_regularized_mz_kernel(system, ridge=ridge)
                            prediction = predict_mz_drift(
                                held_position,
                                held_velocity,
                                scale=scale,
                                mean_force_coefficients=fitted[
                                    "mean_force_coefficients"
                                ],
                                memory_coefficients=fitted["memory_coefficients"],
                                basis_indices=basis_indices,
                                memory_position=held_memory_position,
                            )
                            score = _normalized_drift_rmse(
                                held_acceleration[:, support - 1 :],
                                prediction,
                            )
                        except ValueError:
                            continue
                        score_grid[(support, ridge)].append(score)

            model_rows = []
            for support in support_grid:
                for ridge_index, ridge in enumerate(ridges):
                    scores = score_grid[(support, ridge)]
                    valid = len(scores) == len(outer_training)
                    row = {
                        "record": "inner_candidate",
                        "held_clone_index": float(outer_held + 1),
                        "model": model,
                        "memory_support": float(support),
                        "ridge": ridge,
                        "mean_inner_normalized_rmse": float(np.mean(scores))
                        if valid
                        else math.inf,
                        "maximum_inner_normalized_rmse": float(np.max(scores))
                        if valid
                        else math.inf,
                        "inner_validation_fold_count": float(len(scores)),
                        "candidate_valid": float(valid),
                        "inner_fit_to_validation_clone_sets": ";".join(fold_sets),
                        "fit_uses_outer_held_clone": 0.0,
                        "fit_uses_event_or_macro_observable": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                        "_ridge_grid_index": ridge_index,
                    }
                    candidates.append(row)
                    model_rows.append(row)
            valid_rows = [row for row in model_rows if row["candidate_valid"] == 1.0]
            if not valid_rows:
                raise ValueError("no resolved nested candidate remains")
            selected = min(
                valid_rows,
                key=lambda row: (
                    float(row["mean_inner_normalized_rmse"]),
                    float(row["maximum_inner_normalized_rmse"]),
                    float(row["memory_support"]),
                    int(row["_ridge_grid_index"]),
                ),
            )
            selections.append(
                {
                    "record": "outer_selection",
                    "held_clone_index": float(outer_held + 1),
                    "model": model,
                    "selected_memory_support": selected["memory_support"],
                    "selected_ridge": selected["ridge"],
                    "mean_inner_normalized_rmse": selected[
                        "mean_inner_normalized_rmse"
                    ],
                    "maximum_inner_normalized_rmse": selected[
                        "maximum_inner_normalized_rmse"
                    ],
                    "inner_validation_fold_count": float(len(outer_training)),
                    "fit_uses_outer_held_clone": 0.0,
                    "fit_uses_event_or_macro_observable": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    for row in candidates:
        row.pop("_ridge_grid_index")
    return candidates, selections


def select_auxiliary_hierarchy(
    clones: list[dict[str, object]],
    nonparametric_selections: list[dict[str, object]],
    *,
    ranks: tuple[int, ...],
    ridge_grid: tuple[float, ...],
    decay_grid: np.ndarray,
    frame_time: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Select M3/M4 while deriving every pole inside its inner fit fold."""

    if len(clones) != 4:
        raise ValueError("auxiliary hierarchy requires exactly four clones")
    rank_grid = tuple(int(value) for value in ranks)
    ridges = tuple(float(value) for value in ridge_grid)
    rates = np.asarray(decay_grid, dtype=float)
    dt = float(frame_time)
    if (
        len(rank_grid) < 1
        or len(set(rank_grid)) != len(rank_grid)
        or any(value < 1 or value > len(rates) for value in rank_grid)
        or len(ridges) < 1
        or len(set(ridges)) != len(ridges)
        or any(not math.isfinite(value) or value < 0.0 for value in ridges)
        or rates.ndim != 1
        or len(rates) < 1
        or np.any(~np.isfinite(rates))
        or np.any(rates <= 0.0)
        or np.any(np.diff(rates) <= 0.0)
        or not math.isfinite(dt)
        or dt <= 0.0
    ):
        raise ValueError("auxiliary grids and frame time must be finite and resolved")
    mz_selection = {
        int(float(row["held_clone_index"])) - 1: row
        for row in nonparametric_selections
        if row.get("model") == "finite_basis_mz_position_kernel"
    }
    if set(mz_selection) != set(range(4)):
        raise ValueError("one M2 selection is required for every outer fold")

    candidates: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for outer_held in range(4):
        outer_training = tuple(index for index in range(4) if index != outer_held)
        support = int(float(mz_selection[outer_held]["selected_memory_support"]))
        mz_ridge = float(mz_selection[outer_held]["selected_ridge"])
        score_grid = {
            (model, rank, ridge): []
            for model in AUXILIARY_MODELS
            for rank in rank_grid
            for ridge in ridges
        }
        pole_grid: dict[tuple[str, int, float], list[str]] = {
            key: [] for key in score_grid
        }
        for inner_held in outer_training:
            inner_training = tuple(
                index for index in outer_training if index != inner_held
            )
            fit_position, fit_velocity, fit_acceleration = _kernel_clone_arrays(
                clones,
                inner_training,
            )
            held_position, held_velocity, held_acceleration = _kernel_clone_arrays(
                clones,
                (inner_held,),
            )
            scale = fit_radial_basis_scale(fit_position)
            mz_system = assemble_mz_correlation_system(
                fit_position,
                fit_velocity,
                fit_acceleration,
                scale=scale,
                support=support,
                projection_lag_count=support,
            )
            mz_fit = solve_regularized_mz_kernel(mz_system, ridge=mz_ridge)
            for rank in rank_grid:
                if rank > support:
                    continue
                pole_selection = select_decay_rates_from_memory(
                    np.asarray(mz_fit["memory_coefficients"]),
                    frame_time=dt,
                    decay_grid=rates,
                    rank=rank,
                )
                selected_rates = np.asarray(
                    pole_selection["selected_decay_rates"],
                    dtype=float,
                )
                pole_text = ",".join(f"{value:.17g}" for value in selected_rates)
                for model in AUXILIARY_MODELS:
                    for ridge in ridges:
                        try:
                            if model == AUXILIARY_MODELS[0]:
                                fitted = fit_real_pole_model(
                                    fit_position,
                                    fit_velocity,
                                    fit_acceleration,
                                    scale=scale,
                                    decay_rates=selected_rates,
                                    frame_time=dt,
                                    ridge=ridge,
                                )
                                prediction = predict_real_pole_drift(
                                    held_position,
                                    held_velocity,
                                    scale=scale,
                                    decay_rates=selected_rates,
                                    frame_time=dt,
                                    mean_force_coefficients=fitted[
                                        "mean_force_coefficients"
                                    ],
                                    pole_coefficients=fitted["pole_coefficients"],
                                )
                            else:
                                fitted = fit_two_position_prony_model(
                                    fit_position,
                                    fit_velocity,
                                    fit_acceleration,
                                    scale=scale,
                                    decay_rates=selected_rates,
                                    frame_time=dt,
                                    ridge=ridge,
                                )
                                prediction = predict_two_position_prony_drift(
                                    held_position,
                                    held_velocity,
                                    scale=scale,
                                    decay_rates=selected_rates,
                                    coupling_coefficients=fitted[
                                        "coupling_coefficients"
                                    ],
                                    frame_time=dt,
                                    mean_force_coefficients=fitted[
                                        "mean_force_coefficients"
                                    ],
                                )
                            score = _normalized_drift_rmse(
                                held_acceleration[:, support - 1 :],
                                prediction[:, support - 1 :],
                            )
                        except (ValueError, np.linalg.LinAlgError):
                            continue
                        key = (model, rank, ridge)
                        score_grid[key].append(score)
                        pole_grid[key].append(pole_text)

        for model in AUXILIARY_MODELS:
            model_rows = []
            for rank in rank_grid:
                for ridge_index, ridge in enumerate(ridges):
                    key = (model, rank, ridge)
                    scores = score_grid[key]
                    valid = len(scores) == len(outer_training)
                    row = {
                        "record": "inner_candidate",
                        "held_clone_index": float(outer_held + 1),
                        "model": model,
                        "selected_m2_support": float(support),
                        "selected_m2_ridge": mz_ridge,
                        "auxiliary_rank": float(rank),
                        "ridge": ridge,
                        "mean_inner_normalized_rmse": float(np.mean(scores))
                        if valid
                        else math.inf,
                        "maximum_inner_normalized_rmse": float(np.max(scores))
                        if valid
                        else math.inf,
                        "inner_validation_fold_count": float(len(scores)),
                        "inner_selected_decay_rates": ";".join(pole_grid[key]),
                        "all_selected_decay_rates_positive": float(
                            valid and all(float(value) > 0.0 for text in pole_grid[key] for value in text.split(","))
                        ),
                        "candidate_valid": float(valid),
                        "fit_uses_outer_held_clone": 0.0,
                        "pole_selection_uses_outer_held_clone": 0.0,
                        "fit_uses_event_or_macro_observable": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                        "_ridge_grid_index": ridge_index,
                    }
                    candidates.append(row)
                    model_rows.append(row)
            valid_rows = [row for row in model_rows if row["candidate_valid"] == 1.0]
            if not valid_rows:
                raise ValueError("no resolved nested auxiliary candidate remains")
            selected = min(
                valid_rows,
                key=lambda row: (
                    float(row["mean_inner_normalized_rmse"]),
                    float(row["maximum_inner_normalized_rmse"]),
                    float(row["auxiliary_rank"]),
                    int(row["_ridge_grid_index"]),
                ),
            )
            selections.append(
                {
                    "record": "outer_selection",
                    "held_clone_index": float(outer_held + 1),
                    "model": model,
                    "selected_m2_support": float(support),
                    "selected_m2_ridge": mz_ridge,
                    "selected_auxiliary_rank": selected["auxiliary_rank"],
                    "selected_ridge": selected["ridge"],
                    "mean_inner_normalized_rmse": selected[
                        "mean_inner_normalized_rmse"
                    ],
                    "maximum_inner_normalized_rmse": selected[
                        "maximum_inner_normalized_rmse"
                    ],
                    "all_selected_decay_rates_positive": 1.0,
                    "inner_validation_fold_count": float(len(outer_training)),
                    "fit_uses_outer_held_clone": 0.0,
                    "pole_selection_uses_outer_held_clone": 0.0,
                    "fit_uses_event_or_macro_observable": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    for row in candidates:
        row.pop("_ridge_grid_index")
    return candidates, selections


def _training_residual_variance(
    observed: np.ndarray,
    predicted: np.ndarray,
    start: int,
) -> float:
    residual = np.asarray(observed)[:, start:] - np.asarray(predicted)[:, start:]
    variance = float(np.mean(residual**2))
    if not math.isfinite(variance):
        raise ValueError("training residual variance must be finite")
    return max(variance, np.finfo(float).tiny)


def fit_and_score_outer_hierarchy(
    clones: list[dict[str, object]],
    selections: list[dict[str, object]],
    *,
    decay_grid: np.ndarray,
    frame_time: float,
    temperature: float,
    maximum_diagnostic_lag: int,
    permutation_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Refit selected models on three clones and score each outer held clone once."""

    indexed = {
        (int(float(row["held_clone_index"])) - 1, str(row["model"])): row
        for row in selections
    }
    expected = {
        (held, model)
        for held in range(4)
        for model in (*NONPARAMETRIC_MODELS, *AUXILIARY_MODELS)
    }
    if len(clones) != 4 or set(indexed) != expected:
        raise ValueError("outer scoring requires one selected row per fold and model")
    rates = np.asarray(decay_grid, dtype=float)
    dt = float(frame_time)
    thermal_energy = float(temperature)
    if (
        rates.ndim != 1
        or len(rates) < 1
        or np.any(~np.isfinite(rates))
        or np.any(rates <= 0.0)
        or np.any(np.diff(rates) <= 0.0)
        or not math.isfinite(dt)
        or dt <= 0.0
        or not math.isfinite(thermal_energy)
        or thermal_energy <= 0.0
        or isinstance(maximum_diagnostic_lag, bool)
        or not isinstance(maximum_diagnostic_lag, (int, np.integer))
        or maximum_diagnostic_lag < 0
    ):
        raise ValueError("outer scoring controls must be finite and physical")

    details: list[dict[str, object]] = []
    kernels: list[dict[str, object]] = []
    for outer_held in range(4):
        training_indices = tuple(index for index in range(4) if index != outer_held)
        train_position, train_velocity, train_acceleration = _kernel_clone_arrays(
            clones,
            training_indices,
        )
        held_position, held_velocity, held_acceleration = _kernel_clone_arrays(
            clones,
            (outer_held,),
        )
        scale = fit_radial_basis_scale(train_position)
        selected_supports = [
            int(float(indexed[(outer_held, model)]["selected_memory_support"]))
            for model in NONPARAMETRIC_MODELS
        ]
        common_start = max(selected_supports) - 1
        if held_position.shape[1] - common_start <= maximum_diagnostic_lag:
            raise ValueError("held path is too short for the common diagnostic horizon")
        active_lag = int(maximum_diagnostic_lag)
        held_source = clones[outer_held]
        fitted_models: dict[str, dict[str, object]] = {}

        for model in NONPARAMETRIC_MODELS:
            selected = indexed[(outer_held, model)]
            support = int(float(selected["selected_memory_support"]))
            ridge = float(selected["selected_ridge"])
            basis_indices = (0,) if model == NONPARAMETRIC_MODELS[0] else (0, 1, 2)
            train_memory_position = None
            held_memory_position = None
            if model == NONPARAMETRIC_MODELS[2]:
                train_memory_position = _permuted_memory_positions(
                    clones,
                    training_indices,
                    permutation_seed=permutation_seed,
                )
                held_memory_position = _permuted_memory_positions(
                    clones,
                    (outer_held,),
                    permutation_seed=permutation_seed,
                )
            system = assemble_mz_correlation_system(
                train_position,
                train_velocity,
                train_acceleration,
                scale=scale,
                support=support,
                basis_indices=basis_indices,
                memory_position=train_memory_position,
                projection_lag_count=support,
            )
            fitted = solve_regularized_mz_kernel(system, ridge=ridge)
            train_prediction = predict_mz_drift(
                train_position,
                train_velocity,
                scale=scale,
                mean_force_coefficients=fitted["mean_force_coefficients"],
                memory_coefficients=fitted["memory_coefficients"],
                basis_indices=basis_indices,
                memory_position=train_memory_position,
            )
            held_prediction = predict_mz_drift(
                held_position,
                held_velocity,
                scale=scale,
                mean_force_coefficients=fitted["mean_force_coefficients"],
                memory_coefficients=fitted["memory_coefficients"],
                basis_indices=basis_indices,
                memory_position=held_memory_position,
            )
            train_prediction_full = np.zeros_like(train_acceleration)
            train_prediction_full[:, support - 1 :] = train_prediction
            held_prediction_full = np.zeros_like(held_acceleration)
            held_prediction_full[:, support - 1 :] = held_prediction
            variance = _training_residual_variance(
                train_acceleration,
                train_prediction_full,
                common_start,
            )
            diagnostics = held_kernel_diagnostics(
                held_acceleration[:, common_start:],
                held_prediction_full[:, common_start:],
                held_position[:, common_start:],
                held_velocity[:, common_start:],
                scale=scale,
                training_residual_variance=variance,
                maximum_lag=active_lag,
            )
            fitted_models[model] = fitted
            details.append(
                {
                    "record": "held_model",
                    "held_clone_index": float(outer_held + 1),
                    "model": model,
                    **diagnostics,
                    "selected_memory_support": float(support),
                    "selected_ridge": ridge,
                    "selected_auxiliary_rank": 0.0,
                    "selected_decay_rates": "",
                    "all_selected_decay_rates_positive": 1.0,
                    "condition_number": float(fitted["condition_number"]),
                    "all_fitted_arrays_finite": float(
                        np.all(np.isfinite(fitted["mean_force_coefficients"]))
                        and np.all(np.isfinite(fitted["memory_coefficients"]))
                        and math.isfinite(float(fitted["condition_number"]))
                    ),
                    "positive_prony_factorization": 0.0,
                    "common_held_start_frame": float(common_start),
                    "fit_uses_outer_held_clone": 0.0,
                    "fit_uses_event_or_macro_observable": 0.0,
                    "trajectory_sha256": str(held_source.get("trajectory_sha256", "")),
                    "drift_cache_sha256": str(held_source.get("drift_cache_sha256", "")),
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
            for lag, coefficients in enumerate(fitted["memory_coefficients"]):
                for basis_offset, value in enumerate(coefficients):
                    kernels.append(
                        {
                            "held_clone_index": float(outer_held + 1),
                            "model": model,
                            "kernel_index": float(lag),
                            "basis_index": float(basis_indices[basis_offset]),
                            "coefficient": float(value),
                            "decay_rate": 0.0,
                            "fit_uses_outer_held_clone": 0.0,
                            "thermodynamic_claim_allowed": 0.0,
                        }
                    )

        mz_fit = fitted_models["finite_basis_mz_position_kernel"]
        for model in AUXILIARY_MODELS:
            selected = indexed[(outer_held, model)]
            rank = int(float(selected["selected_auxiliary_rank"]))
            ridge = float(selected["selected_ridge"])
            pole_selection = select_decay_rates_from_memory(
                np.asarray(mz_fit["memory_coefficients"]),
                frame_time=dt,
                decay_grid=rates,
                rank=rank,
            )
            selected_rates = np.asarray(
                pole_selection["selected_decay_rates"],
                dtype=float,
            )
            if model == AUXILIARY_MODELS[0]:
                fitted = fit_real_pole_model(
                    train_position,
                    train_velocity,
                    train_acceleration,
                    scale=scale,
                    decay_rates=selected_rates,
                    frame_time=dt,
                    ridge=ridge,
                )
                train_prediction = predict_real_pole_drift(
                    train_position,
                    train_velocity,
                    scale=scale,
                    decay_rates=selected_rates,
                    frame_time=dt,
                    mean_force_coefficients=fitted["mean_force_coefficients"],
                    pole_coefficients=fitted["pole_coefficients"],
                )
                held_prediction = predict_real_pole_drift(
                    held_position,
                    held_velocity,
                    scale=scale,
                    decay_rates=selected_rates,
                    frame_time=dt,
                    mean_force_coefficients=fitted["mean_force_coefficients"],
                    pole_coefficients=fitted["pole_coefficients"],
                )
                diagnostics_kwargs: dict[str, object] = {}
                kernel_values = np.asarray(fitted["pole_coefficients"])
                positive_factorization = 0.0
            else:
                fitted = fit_two_position_prony_model(
                    train_position,
                    train_velocity,
                    train_acceleration,
                    scale=scale,
                    decay_rates=selected_rates,
                    frame_time=dt,
                    ridge=ridge,
                )
                train_prediction = predict_two_position_prony_drift(
                    train_position,
                    train_velocity,
                    scale=scale,
                    decay_rates=selected_rates,
                    coupling_coefficients=fitted["coupling_coefficients"],
                    frame_time=dt,
                    mean_force_coefficients=fitted["mean_force_coefficients"],
                )
                held_prediction = predict_two_position_prony_drift(
                    held_position,
                    held_velocity,
                    scale=scale,
                    decay_rates=selected_rates,
                    coupling_coefficients=fitted["coupling_coefficients"],
                    frame_time=dt,
                    mean_force_coefficients=fitted["mean_force_coefficients"],
                )
                observed_fdt = force_autocovariance(
                    held_acceleration[:, common_start:]
                    - held_prediction[:, common_start:],
                    active_lag,
                )
                target_fdt = two_position_fdt_covariance(
                    held_position[:, common_start:],
                    scale=scale,
                    decay_rates=selected_rates,
                    coupling_coefficients=fitted["coupling_coefficients"],
                    temperature=thermal_energy,
                    frame_time=dt,
                    maximum_lag=active_lag,
                )
                diagnostics_kwargs = (
                    {
                        "observed_fdt_covariance": observed_fdt,
                        "target_fdt_covariance": target_fdt,
                    }
                    if float(np.mean(target_fdt**2)) > np.finfo(float).tiny
                    else {}
                )
                kernel_values = np.asarray(fitted["coupling_coefficients"])
                positive_factorization = 1.0
            variance = _training_residual_variance(
                train_acceleration,
                train_prediction,
                common_start,
            )
            diagnostics = held_kernel_diagnostics(
                held_acceleration[:, common_start:],
                held_prediction[:, common_start:],
                held_position[:, common_start:],
                held_velocity[:, common_start:],
                scale=scale,
                training_residual_variance=variance,
                maximum_lag=active_lag,
                **diagnostics_kwargs,
            )
            details.append(
                {
                    "record": "held_model",
                    "held_clone_index": float(outer_held + 1),
                    "model": model,
                    **diagnostics,
                    "selected_memory_support": float(common_start + 1),
                    "selected_ridge": ridge,
                    "selected_auxiliary_rank": float(rank),
                    "selected_decay_rates": ",".join(
                        f"{value:.17g}" for value in selected_rates
                    ),
                    "all_selected_decay_rates_positive": float(
                        np.all(selected_rates > 0.0)
                    ),
                    "condition_number": float(fitted["condition_number"]),
                    "all_fitted_arrays_finite": float(
                        np.all(np.isfinite(fitted["mean_force_coefficients"]))
                        and np.all(np.isfinite(kernel_values))
                        and math.isfinite(float(fitted["condition_number"]))
                    ),
                    "positive_prony_factorization": positive_factorization,
                    "common_held_start_frame": float(common_start),
                    "fit_uses_outer_held_clone": 0.0,
                    "pole_selection_uses_outer_held_clone": 0.0,
                    "fit_uses_event_or_macro_observable": 0.0,
                    "trajectory_sha256": str(held_source.get("trajectory_sha256", "")),
                    "drift_cache_sha256": str(held_source.get("drift_cache_sha256", "")),
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
            for pole, rate in enumerate(selected_rates):
                for coefficient_index, value in enumerate(kernel_values[pole]):
                    kernels.append(
                        {
                            "held_clone_index": float(outer_held + 1),
                            "model": model,
                            "kernel_index": float(pole),
                            "basis_index": float(coefficient_index),
                            "coefficient": float(value),
                            "decay_rate": float(rate),
                            "fit_uses_outer_held_clone": 0.0,
                            "thermodynamic_claim_allowed": 0.0,
                        }
                    )
    return details, kernels


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty kernel artifact")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if isinstance(value, (float, np.floating))
                    and not math.isfinite(float(value))
                    else value
                    for key, value in row.items()
                }
            )


def _render_kernel_svg(details: list[dict[str, object]]) -> str:
    models = (
        "stationary_scalar_nonparametric_volterra",
        "finite_basis_mz_position_kernel",
        "past_position_real_pole",
        "two_position_positive_prony",
        "time_permuted_position_null",
    )
    labels = ("M1", "M2", "M3", "M4", "M5")
    metrics = (
        ("drift_rmse", "normalized drift RMSE", None),
        ("drift_nll_per_component", "Gaussian NLL / component", None),
        (
            "maximum_normalized_resolved_basis_residual_correlation",
            "residual correlation",
            0.20,
        ),
        (
            "second_fdt_covariance_normalized_rmse",
            "second-FDT normalized RMSE",
            0.30,
        ),
    )
    plot_rows = []
    for row in details:
        enriched = dict(row)
        count = float(row["held_scalar_component_count"])
        enriched["drift_nll_per_component"] = float(row["drift_nll"]) / count
        plot_rows.append(enriched)
    width = 1240
    height = 720
    panel_width = 270
    panel_height = 480
    panel_top = 105
    panel_left = 85
    panel_gap = 25
    colors = ("#1565c0", "#c62828", "#2e7d32", "#6a1b9a")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="42" font-family="Arial" font-size="24" font-weight="700">Real-KA position-dependent kernel gate</text>',
        '<text x="48" y="70" font-family="Arial" font-size="15">unclipped held diagnostics; each point is one outer held clone</text>',
    ]
    for panel_index, (metric, title, tolerance) in enumerate(metrics):
        left = panel_left + panel_index * (panel_width + panel_gap)
        values = [float(row[metric]) for row in plot_rows]
        maximum = max(values + ([tolerance] if tolerance is not None else [0.0]))
        minimum = min(values + [0.0])
        span = maximum - minimum
        y_min = minimum - 0.08 * (span if span > 0.0 else 1.0)
        y_max = maximum + 0.12 * (span if span > 0.0 else 1.0)

        def y_position(value: float) -> float:
            return panel_top + panel_height * (y_max - value) / (y_max - y_min)

        svg.extend(
            [
                f'<rect x="{left}" y="{panel_top}" width="{panel_width}" height="{panel_height}" fill="#fafafa" stroke="#bdbdbd"/>',
                f'<text x="{left + panel_width / 2:.1f}" y="{panel_top - 18}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700">{html.escape(title)}</text>',
                f'<text x="{left - 8}" y="{panel_top + 5}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.3g}</text>',
                f'<text x="{left - 8}" y="{panel_top + panel_height}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.3g}</text>',
            ]
        )
        if tolerance is not None:
            y_tol = y_position(tolerance)
            tolerance_label = (
                "residual-correlation tolerance 0.20"
                if metric.startswith("maximum_normalized")
                else "second-FDT tolerance 0.30"
            )
            svg.extend(
                [
                    f'<line x1="{left}" y1="{y_tol:.2f}" x2="{left + panel_width}" y2="{y_tol:.2f}" stroke="#555" stroke-dasharray="5 4"/>',
                    f'<text x="{left + 4}" y="{y_tol - 6:.2f}" font-family="Arial" font-size="10">{tolerance_label}</text>',
                ]
            )
        for model_index, (model, label) in enumerate(zip(models, labels)):
            x = left + 28 + model_index * (panel_width - 56) / 4
            svg.append(
                f'<text x="{x:.2f}" y="{panel_top + panel_height + 22}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>'
            )
            model_rows = [row for row in plot_rows if row["model"] == model]
            for row in model_rows:
                clone = int(float(row["held_clone_index"]))
                x_point = x + (clone - 2.5) * 3.5
                svg.append(
                    f'<circle cx="{x_point:.2f}" cy="{y_position(float(row[metric])):.2f}" r="4.5" fill="{colors[clone - 1]}"/>'
                )
    svg.extend(
        [
            '<text x="48" y="665" font-family="Arial" font-size="12">M1 stationary scalar; M2 finite-basis MZ; M3 signed real poles; M4 positive two-position Prony; M5 time-permuted null</text>',
            '<text x="48" y="690" font-family="Arial" font-size="12">M3 has no positive second-FDT target; a zero plotted value is marked non-applicable in CSV, not a pass.</text>',
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def write_kernel_artifacts(
    output_prefix: Path,
    *,
    details: list[dict[str, object]],
    selections: list[dict[str, object]],
    kernels: list[dict[str, object]],
    verdict: dict[str, object],
    source_paths: list[Path],
) -> dict[str, Path]:
    """Write deterministic CSV/SVG outputs and a recomputed SHA manifest."""

    prefix = Path(output_prefix)
    paths = {
        "details": prefix.with_name(prefix.name + "_details.csv"),
        "selection": prefix.with_name(prefix.name + "_selection.csv"),
        "kernel": prefix.with_name(prefix.name + "_kernel.csv"),
        "summary": prefix.with_name(prefix.name + "_summary.csv"),
        "svg": prefix.with_suffix(".svg"),
        "sha256": prefix.with_name(prefix.name + "_sha256.csv"),
    }
    write_rows(paths["details"], details)
    write_rows(paths["selection"], selections)
    write_rows(paths["kernel"], kernels)
    write_rows(paths["summary"], [verdict])
    paths["svg"].parent.mkdir(parents=True, exist_ok=True)
    paths["svg"].write_text(_render_kernel_svg(details))
    manifest_rows = [
        {
            "record": "artifact",
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
        for key, path in paths.items()
        if key != "sha256"
    ]
    for path in source_paths:
        source = Path(path)
        if not source.is_file():
            raise ValueError("kernel artifact source path is missing")
        manifest_rows.append(
            {
                "record": "source",
                "path": str(source.resolve()),
                "sha256": file_sha256(source),
            }
        )
    write_rows(paths["sha256"], manifest_rows)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--trajectory-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--frame-time", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--maximum-diagnostic-lag",
        type=int,
        default=MAXIMUM_DIAGNOSTIC_LAG,
    )
    parser.add_argument(
        "--permutation-seed",
        type=int,
        default=PERMUTATION_SEED,
    )
    parser.add_argument(
        "--memory-supports",
        type=int,
        nargs="+",
        default=list(MEMORY_SUPPORTS),
        help="frozen support grid: 4 16 40 100",
    )
    parser.add_argument(
        "--auxiliary-ranks",
        type=int,
        nargs="+",
        default=list(AUXILIARY_RANKS),
        help="frozen rank grid: 1 2 4 8",
    )
    parser.add_argument(
        "--ridge-grid",
        type=float,
        nargs="+",
        default=list(RIDGE_GRID),
        help="frozen ridge grid: 0.0 1e-10 1e-08 1e-06 0.0001 0.01",
    )
    parser.add_argument(
        "--decay-grid-count",
        type=int,
        default=DECAY_GRID_COUNT,
        help="frozen positive decay grid count 32",
    )
    parser.add_argument(
        "--decay-minimum",
        type=float,
        default=DECAY_MINIMUM,
        help="frozen minimum positive decay 0.05",
    )
    parser.add_argument(
        "--decay-maximum",
        type=float,
        default=DECAY_MAXIMUM,
        help="frozen maximum positive decay 50.0",
    )
    return parser.parse_args(argv)


def validate_frozen_args(args: argparse.Namespace) -> None:
    if (
        tuple(args.memory_supports) != MEMORY_SUPPORTS
        or tuple(args.auxiliary_ranks) != AUXILIARY_RANKS
        or tuple(args.ridge_grid) != RIDGE_GRID
        or args.decay_grid_count != DECAY_GRID_COUNT
        or not math.isclose(
            args.decay_minimum,
            DECAY_MINIMUM,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            args.decay_maximum,
            DECAY_MAXIMUM,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or args.expected_clone_count != 4
        or args.target_count != 64
        or not math.isclose(args.frame_time, 0.01, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            args.temperature,
            TEMPERATURE,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or args.maximum_diagnostic_lag != MAXIMUM_DIAGNOSTIC_LAG
        or args.permutation_seed != PERMUTATION_SEED
    ):
        raise ValueError("kernel analysis controls must match the frozen protocol")


def main() -> None:
    args = parse_args()
    validate_frozen_args(args)
    clones = load_frozen_kernel_clones(
        args.drift_cache_directory,
        args.trajectory_directory,
        expected_clone_count=args.expected_clone_count,
        target_count=args.target_count,
    )
    decay_grid = np.logspace(
        math.log10(args.decay_minimum),
        math.log10(args.decay_maximum),
        args.decay_grid_count,
    )
    nonparametric_candidates, nonparametric_selections = (
        select_nonparametric_hierarchy(
            clones,
            supports=tuple(args.memory_supports),
            ridge_grid=tuple(args.ridge_grid),
            permutation_seed=args.permutation_seed,
        )
    )
    auxiliary_candidates, auxiliary_selections = select_auxiliary_hierarchy(
        clones,
        nonparametric_selections,
        ranks=tuple(args.auxiliary_ranks),
        ridge_grid=tuple(args.ridge_grid),
        decay_grid=decay_grid,
        frame_time=args.frame_time,
    )
    details, kernels = fit_and_score_outer_hierarchy(
        clones,
        nonparametric_selections + auxiliary_selections,
        decay_grid=decay_grid,
        frame_time=args.frame_time,
        temperature=args.temperature,
        maximum_diagnostic_lag=args.maximum_diagnostic_lag,
        permutation_seed=args.permutation_seed,
    )
    verdict = classify_position_dependent_kernel_gate(details)
    source_paths = [
        Path(__file__),
        ROOT / "src" / "ka_position_dependent_kernel.py",
        ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-07-19-real-ka-position-dependent-kernel-identifiability-design.md",
    ]
    source_paths.extend(Path(str(clone["trajectory_path"])) for clone in clones)
    source_paths.extend(Path(str(clone["drift_cache_path"])) for clone in clones)
    paths = write_kernel_artifacts(
        args.output_prefix,
        details=details,
        selections=(
            nonparametric_candidates
            + auxiliary_candidates
            + nonparametric_selections
            + auxiliary_selections
        ),
        kernels=kernels,
        verdict=verdict,
        source_paths=source_paths,
    )
    for key, value in verdict.items():
        print(f"{key}={value}")
    for key, path in paths.items():
        print(f"{key}_path={path.resolve()}")


if __name__ == "__main__":
    main()
