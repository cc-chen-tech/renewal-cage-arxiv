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


MEMORY_SUPPORTS = (4, 16, 40, 100)
AUXILIARY_RANKS = (1, 2, 4, 8)
RIDGE_GRID = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
DECAY_GRID_COUNT = 32
DECAY_MINIMUM = 0.05
DECAY_MAXIMUM = 50.0


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
