#!/usr/bin/env python3
"""Extract calibration-only Debye-Waller cage-jump geometry from KA trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    extract_debye_waller_cage_jumps,
    load_lammps_custom_trajectory,
    position_fluctuation_values,
)


STRONG_ZERO_FLAGS = (
    "blind_prediction_claim_allowed",
    "finite_exchange_resolved",
    "static_environment_resolved",
    "spatial_facilitation_resolved",
    "activated_cage_geometry_resolved",
    "microdynamic_closure_claim_allowed",
    "thermodynamic_claim_allowed",
)


def wave_key(wave_number: float) -> str:
    value = float(wave_number)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("wave numbers must be positive and finite")
    return f"jump_characteristic_k{format(value, 'g').replace('.', 'p')}"


def jump_geometry_statistics(
    jump_vectors: np.ndarray,
    wave_numbers: np.ndarray,
) -> dict[str, float]:
    jumps = np.asarray(jump_vectors, dtype=float)
    waves = np.asarray(wave_numbers, dtype=float)
    if (
        jumps.ndim != 2
        or jumps.shape[1] != 3
        or len(jumps) == 0
        or np.any(~np.isfinite(jumps))
    ):
        raise ValueError("jump_vectors must be a nonempty finite (events, 3) array")
    if (
        waves.ndim != 1
        or len(waves) == 0
        or np.any(~np.isfinite(waves))
        or np.any(waves <= 0.0)
    ):
        raise ValueError("wave_numbers must be a nonempty positive finite vector")

    squared = np.sum(jumps**2, axis=1)
    components = jumps.reshape(-1)
    result = {
        "event_count": float(len(jumps)),
        "jump_msd": float(np.mean(squared)),
        "jump_radial_fourth_moment": float(np.mean(squared**2)),
        "jump_component_fourth_moment": float(np.mean(components**4)),
    }
    for wave_number in waves:
        result[wave_key(float(wave_number))] = float(
            np.mean(np.cos(float(wave_number) * components))
        )
    return result


def extract_calibration_jump_geometry(
    positions: np.ndarray,
    *,
    calibration_time: int,
    debye_waller_factor: float,
    half_window: int,
    wave_numbers: np.ndarray,
) -> dict[str, float]:
    positions = np.asarray(positions, dtype=float)
    if (
        positions.ndim != 3
        or positions.shape[2] != 3
        or np.any(~np.isfinite(positions))
    ):
        raise ValueError("positions must be a finite (frames, particles, 3) array")
    if (
        isinstance(calibration_time, bool)
        or int(calibration_time) != calibration_time
        or calibration_time < 2
        or calibration_time >= len(positions)
    ):
        raise ValueError("calibration_time must select a strict nontrivial prefix")
    if not math.isfinite(debye_waller_factor) or debye_waller_factor <= 0.0:
        raise ValueError("debye_waller_factor must be positive and finite")
    if isinstance(half_window, bool) or int(half_window) != half_window or half_window < 1:
        raise ValueError("half_window must be a positive integer")

    calibration = positions[: int(calibration_time) + 1]
    activity_times, activity = position_fluctuation_values(
        calibration,
        half_window=int(half_window),
    )
    events = extract_debye_waller_cage_jumps(
        calibration,
        debye_waller_factor=float(debye_waller_factor),
        half_window=int(half_window),
        activity_times=activity_times,
        activity_values=activity,
    )
    result = jump_geometry_statistics(events["jump_vector"], wave_numbers)
    result.update(
        {
            "calibration_time": float(calibration_time),
            "calibration_frame_count": float(len(calibration)),
            "debye_waller_factor": float(debye_waller_factor),
            "fluctuation_half_window": float(half_window),
            "calibration_events_only": 1.0,
            "heldout_events_used": 0.0,
        }
    )
    result.update({flag: 0.0 for flag in STRONG_ZERO_FLAGS})
    return result


def load_calibration_type_a_positions(
    trajectory_path: Path,
    *,
    calibration_time: int,
) -> np.ndarray:
    if (
        isinstance(calibration_time, bool)
        or not isinstance(calibration_time, int)
        or calibration_time < 2
    ):
        raise ValueError("calibration_time must be an integer of at least two")
    trajectory = load_lammps_custom_trajectory(
        Path(trajectory_path),
        maximum_frame_count=calibration_time + 1,
    )
    return trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty geometry table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--threshold-table", type=Path, required=True)
    parser.add_argument("--calibration-time", type=int, default=5000)
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--wave-numbers", type=float, nargs="+", default=(2.0, 4.0, 7.25))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.ensemble_directory / "ensemble_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    thresholds = {
        int(float(row["replicate"])): float(row["debye_waller_factor"])
        for row in read_rows(args.threshold_table)
    }
    rows: list[dict[str, object]] = []
    for specification in manifest["replicates"]:
        replicate = int(specification["replicate"])
        if replicate not in thresholds:
            raise ValueError(f"replicate {replicate} lacks a Debye-Waller threshold")
        trajectory_path = (
            args.ensemble_directory
            / str(specification["directory"])
            / "trajectory.lammpstrj"
        )
        positions = load_calibration_type_a_positions(
            trajectory_path,
            calibration_time=args.calibration_time,
        )
        local = extract_calibration_jump_geometry(
            positions,
            calibration_time=args.calibration_time,
            debye_waller_factor=thresholds[replicate],
            half_window=args.fluctuation_half_window,
            wave_numbers=np.asarray(args.wave_numbers, dtype=float),
        )
        rows.append(
            {
                "temperature": float(manifest["temperature"]),
                "replicate": float(replicate),
                **local,
                "event_definition": "debye_waller_finite_duration_cage_jump",
                "source_manifest": str(manifest_path),
                "source_trajectory": str(trajectory_path),
                "source_doi": str(manifest.get("source_doi", "")),
                "source_sha256": str(manifest.get("source_sha256", "")),
                "threshold_source": str(args.threshold_table),
            }
        )
    if set(thresholds) != {int(row["replicate"]) for row in rows}:
        raise ValueError("threshold table and manifest replicate grids differ")
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
