#!/usr/bin/env python3
"""Cross-parent structural prediction of KA isoconfigurational committors."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import grouped_binomial_logistic_committor_diagnostic  # noqa: E402
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_structural_precursor import species_resolved_radial_features  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_committor(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no committor rows")
    indices = np.array([int(float(row["a_particle_index"])) for row in rows])
    probability = np.array([float(row["finite_time_escape_committor"]) for row in rows])
    summary_path = path.with_name(path.name.replace("_committor.csv", "_summary.csv"))
    with summary_path.open(newline="") as handle:
        clone_count = int(float(next(csv.DictReader(handle))["clone_count"]))
    return indices, probability, clone_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directories", type=Path, nargs="+")
    parser.add_argument("--committor-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.8, 1.05, 1.3, 1.55, 1.8, 2.05, 2.3])
    parser.add_argument("--width", type=float, default=0.12)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()
    if len(args.clone_directories) != len(args.committor_csv) or len(args.clone_directories) < 2:
        raise ValueError("provide aligned clone directories and committor CSVs for at least two parents")
    radii = np.asarray(args.radii, dtype=float)
    if np.any(radii <= 0.0) or np.any(np.diff(radii) <= 0.0) or args.width <= 0.0 or args.cutoff <= radii[-1]:
        raise ValueError("radii, width, and cutoff must define valid radial features")

    feature_rows: list[np.ndarray] = []
    success_rows: list[np.ndarray] = []
    trial_rows: list[np.ndarray] = []
    group_rows: list[np.ndarray] = []
    metadata: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for group, (directory, committor_path) in enumerate(zip(args.clone_directories, args.committor_csv)):
        trajectory = load_lammps_custom_trajectory(directory / "clone_001" / "trajectory.lammpstrj")
        index, probability, clone_count = load_committor(committor_path)
        features = species_resolved_radial_features(
            np.asarray(trajectory["unwrapped_positions"])[0],
            np.asarray(trajectory["particle_types"]),
            np.asarray(trajectory["box_lengths"]),
            index,
            radii=radii,
            width=args.width,
            cutoff=args.cutoff,
        )
        feature_rows.append(features)
        success_rows.append(np.rint(probability * clone_count).astype(float))
        trial_rows.append(np.full(len(index), float(clone_count)))
        group_rows.append(np.full(len(index), group, dtype=int))
        metadata.append((group, index, probability, clone_count))
    features = np.vstack(feature_rows)
    successes = np.concatenate(success_rows)
    trials = np.concatenate(trial_rows)
    groups = np.concatenate(group_rows)
    result = grouped_binomial_logistic_committor_diagnostic(
        features,
        successes,
        trials,
        groups,
        l2_regularization=args.l2,
    )
    detail: list[dict[str, object]] = []
    offset = 0
    for group, index, probability, clone_count in metadata:
        for particle_index, committor, prediction, baseline in zip(
            index,
            probability,
            result["out_of_group_prediction"][offset : offset + len(index)],
            result["out_of_group_baseline_prediction"][offset : offset + len(index)],
        ):
            detail.append(
                {
                    "parent_group": float(group),
                    "a_particle_index": float(particle_index),
                    "clone_count": float(clone_count),
                    "observed_finite_time_committor": float(committor),
                    "heldout_radial_structural_prediction": float(prediction),
                    "heldout_parent_baseline_prediction": float(baseline),
                    "descriptor": "species_resolved_smooth_radial_density",
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        offset += len(index)
    summary = {
        "parent_group_count": float(len(metadata)),
        "feature_count": float(features.shape[1]),
        "radii": ";".join(f"{radius:g}" for radius in radii),
        "radial_width": args.width,
        "radial_cutoff": args.cutoff,
        "l2_regularization": args.l2,
        "descriptor": "species_resolved_smooth_radial_density",
        "prediction_scope": "leave_one_parent_out_isoconfigurational_committor",
        **{key: value for key, value in result.items() if not isinstance(value, np.ndarray)},
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), detail)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), [summary])


if __name__ == "__main__":
    main()
