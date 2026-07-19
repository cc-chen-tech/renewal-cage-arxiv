#!/usr/bin/env python3
"""Identify position-dependent memory kernels on frozen microscopic KA clones."""

from __future__ import annotations

import argparse
import hashlib
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
    fit_radial_basis_scale,
    predict_mz_drift,
    solve_regularized_mz_kernel,
)


MEMORY_SUPPORTS = (4, 16, 40, 100)
AUXILIARY_RANKS = (1, 2, 4, 8)
RIDGE_GRID = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
DECAY_GRID_COUNT = 32
DECAY_MINIMUM = 0.05
DECAY_MAXIMUM = 50.0

NONPARAMETRIC_MODELS = (
    "stationary_scalar_nonparametric_volterra",
    "finite_basis_mz_position_kernel",
    "time_permuted_position_null",
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
    ):
        raise ValueError("kernel analysis controls must match the frozen protocol")


def main() -> None:
    args = parse_args()
    validate_frozen_args(args)
    load_frozen_kernel_clones(
        args.drift_cache_directory,
        args.trajectory_directory,
        expected_clone_count=args.expected_clone_count,
        target_count=args.target_count,
    )
    raise RuntimeError("frozen kernel analysis implementation is incomplete")


if __name__ == "__main__":
    main()
