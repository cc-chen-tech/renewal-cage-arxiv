#!/usr/bin/env python3
"""Test held-parent first escape from the exact smooth-cage projected state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_smooth_cage import (  # noqa: E402
    grouped_exponential_escape_diagnostic,
    smooth_cage_invariant_features,
    smooth_cage_projected_observables,
    smooth_force_support_cage,
)
from renewal_cage import extract_nonrecrossing_phop_events  # noqa: E402


TARGET_SEED = 20260714
PHOP_THRESHOLD = 0.08
HALF_WINDOW = 8
TARGET_COUNT = 64
CLONE_COUNT = 8
PARENT_COUNT = 5
STRUCTURAL_BRIER_REFERENCE = 0.026964
L2_REGULARIZATION = 1.0
SURVIVAL_TIMES = np.array([1, 2, 4, 8, 12, 16, 20], dtype=float)
FEATURE_NAMES = (
    "log_u_squared",
    "log_p_squared",
    "cos_u_p",
    "log_gram_eigenvalue_1",
    "log_gram_eigenvalue_2",
    "log_gram_eigenvalue_3",
    "log_b_squared",
    "cos_u_b",
    "cos_p_b",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def smooth_center_first_passage(
    trajectory_path: Path,
    *,
    target_indices: np.ndarray,
    threshold: float,
    half_window: int,
    cache_path: Path,
) -> dict[str, np.ndarray | float]:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as payload:
            if not np.array_equal(payload["target_indices"], target_indices):
                raise ValueError(f"{cache_path}: cached target indices differ")
            if not np.isclose(float(payload["threshold"]), threshold) or int(
                payload["half_window"]
            ) != half_window:
                raise ValueError(f"{cache_path}: cached event protocol differs")
            return {key: payload[key].copy() for key in payload.files}

    trajectory = load_lammps_custom_trajectory(trajectory_path)
    timesteps = np.asarray(trajectory["timesteps"], dtype=np.int64)
    intervals = np.diff(timesteps)
    if len(intervals) == 0 or not np.all(intervals == intervals[0]):
        raise ValueError(f"{trajectory_path}: saved timestep grid must be uniform")
    frame_time = float(intervals[0]) * 0.001
    horizon = float(timesteps[-1] - timesteps[0]) * 0.001
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    if np.any(particle_types[target_indices] != 0):
        raise ValueError("every fixed target must remain an A particle")
    centers = np.empty((len(positions), len(target_indices), 3), dtype=np.float32)
    for frame_index, frame_positions in enumerate(positions):
        for target_slot, target_index in enumerate(target_indices):
            coordinate = smooth_force_support_cage(
                frame_positions,
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_index=int(target_index),
            )
            centers[frame_index, target_slot] = coordinate["cage_position"]
    events = extract_nonrecrossing_phop_events(
        centers,
        threshold=threshold,
        half_window=half_window,
        recrossing_radius=math.sqrt(threshold),
    )
    first_passage = np.full(len(target_indices), horizon, dtype=float)
    escaped = np.zeros(len(target_indices), dtype=bool)
    if len(events["time"]):
        event_time = np.asarray(events["time"], dtype=float) * frame_time
        np.minimum.at(first_passage, events["particle"], event_time)
        escaped[np.unique(events["particle"])] = True
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        centers=centers,
        first_passage=first_passage,
        escaped=escaped,
        target_indices=target_indices,
        particle_types=particle_types,
        box_lengths=box_lengths,
        timesteps=timesteps,
        frame_time=np.asarray(frame_time),
        horizon=np.asarray(horizon),
        threshold=np.asarray(threshold),
        half_window=np.asarray(half_window),
        event_count=np.asarray(len(events["time"])),
        thermodynamic_claim_allowed=np.asarray(0.0),
    )
    temporary_path.replace(cache_path)
    return {
        "centers": centers,
        "first_passage": first_passage,
        "escaped": escaped,
        "target_indices": target_indices,
        "particle_types": particle_types,
        "box_lengths": box_lengths,
        "timesteps": timesteps,
        "frame_time": np.asarray(frame_time),
        "horizon": np.asarray(horizon),
        "threshold": np.asarray(threshold),
        "half_window": np.asarray(half_window),
        "event_count": np.asarray(len(events["time"])),
        "thermodynamic_claim_allowed": np.asarray(0.0),
    }


def projected_features(
    initial_state_path: Path,
    *,
    target_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(initial_state_path, allow_pickle=False) as payload:
        positions = np.asarray(payload["positions"], dtype=float)
        velocities = np.asarray(payload["velocities"], dtype=float)
        forces = np.asarray(payload["forces"], dtype=float)
        particle_types = np.asarray(payload["particle_types"], dtype=int)
        box_lengths = np.asarray(payload["box_lengths"], dtype=float)
        temperature = float(payload["temperature"])
        damping = float(payload["damping"])
        thermodynamic_claim_allowed = float(payload["thermodynamic_claim_allowed"])
    if thermodynamic_claim_allowed != 0.0:
        raise ValueError("initial-state cache violates thermodynamic claim boundary")
    rows = {"geometry": [], "kinematic": [], "full": []}
    for target_index in target_indices:
        observable = smooth_cage_projected_observables(
            positions,
            velocities=velocities,
            forces=forces,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_index=int(target_index),
            friction=damping,
            temperature=temperature,
            directional_step=1e-5,
            potential_protocol="ka_lj_cut",
        )
        feature = smooth_cage_invariant_features(observable)
        for model in rows:
            rows[model].append(feature[model])
    return {model: np.asarray(values) for model, values in rows.items()}


def observed_survival(
    first_passage: np.ndarray,
    escaped: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            np.mean(
                (escaped & (first_passage > time))
                | (~escaped & (first_passage >= time))
            )
            for time in times
        ],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-directories", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--initial-state-directories", type=Path, nargs="+", required=True
    )
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument("--clone-count", type=int, default=CLONE_COUNT)
    args = parser.parse_args()
    if len(args.clone_directories) != len(args.initial_state_directories) or len(
        args.clone_directories
    ) < 2:
        raise ValueError("aligned clone and initial-state directories need at least two parents")
    if args.target_count < 1 or args.clone_count < 2:
        raise ValueError("target-count and clone-count must be positive")

    clone_directories = [path.resolve() for path in args.clone_directories]
    initial_directories = [path.resolve() for path in args.initial_state_directories]
    cache_directory = args.cache_directory.resolve()
    output_prefix = args.output_prefix.resolve()
    reference_path = clone_directories[0] / "clone_001" / "trajectory.lammpstrj"
    reference = load_lammps_custom_trajectory(reference_path)
    reference_types = np.asarray(reference["particle_types"], dtype=int)
    a_indices = np.flatnonzero(reference_types == 0)
    if args.target_count > len(a_indices):
        raise ValueError("target-count exceeds available A particles")
    target_indices = np.sort(
        np.random.default_rng(TARGET_SEED).choice(
            a_indices, size=args.target_count, replace=False
        )
    )

    metadata_rows: list[dict[str, object]] = []
    first_passage_rows: list[np.ndarray] = []
    escaped_rows: list[np.ndarray] = []
    group_rows: list[np.ndarray] = []
    feature_rows: dict[str, list[np.ndarray]] = {
        "geometry": [],
        "kinematic": [],
        "full": [],
    }
    parent_hashes: list[str] = []
    maximum_reconstruction_error = 0.0
    reference_horizon: float | None = None
    reference_frame_time: float | None = None
    start_time = time.perf_counter()
    for parent_index, (clone_directory, initial_directory) in enumerate(
        zip(clone_directories, initial_directories)
    ):
        clone_manifest = json.loads((clone_directory / "manifest.json").read_text())
        initial_manifest = json.loads((initial_directory / "manifest.json").read_text())
        if int(initial_manifest["clone_count"]) < args.clone_count:
            raise ValueError("initial-state directory has too few reconstructed clones")
        if clone_manifest["parent_restart_sha256"] != initial_manifest["parent_restart_sha256"]:
            raise ValueError("clone and initial-state parent hashes differ")
        parent_hashes.append(str(initial_manifest["parent_restart_sha256"]))
        maximum_reconstruction_error = max(
            maximum_reconstruction_error,
            float(initial_manifest["maximum_position_reconstruction_error"]),
        )
        for clone_index in range(1, args.clone_count + 1):
            trajectory_path = (
                clone_directory / f"clone_{clone_index:03d}" / "trajectory.lammpstrj"
            )
            cache_path = (
                cache_directory
                / f"parent_{parent_index + 1:02d}_clone_{clone_index:03d}_labels.npz"
            )
            label = smooth_center_first_passage(
                trajectory_path,
                target_indices=target_indices,
                threshold=PHOP_THRESHOLD,
                half_window=HALF_WINDOW,
                cache_path=cache_path,
            )
            horizon = float(label["horizon"])
            frame_time = float(label["frame_time"])
            if reference_horizon is None:
                reference_horizon = horizon
                reference_frame_time = frame_time
            elif not np.isclose(horizon, reference_horizon) or not np.isclose(
                frame_time, reference_frame_time
            ):
                raise ValueError("all parent clones must share frame time and horizon")
            initial_state_path = (
                initial_directory / f"clone_{clone_index:03d}_initial.npz"
            )
            feature = projected_features(
                initial_state_path, target_indices=target_indices
            )
            first_passage_rows.append(np.asarray(label["first_passage"], dtype=float))
            escaped_rows.append(np.asarray(label["escaped"], dtype=bool))
            group_rows.append(np.full(args.target_count, parent_index, dtype=int))
            for model in feature_rows:
                feature_rows[model].append(feature[model])
            metadata_rows.extend(
                {
                    "parent_group": parent_index + 1,
                    "clone_index": clone_index,
                    "target_slot": target_slot,
                    "a_particle_index": int(target_index),
                    "source_trajectory": str(trajectory_path),
                    "parent_restart_sha256": parent_hashes[-1],
                }
                for target_slot, target_index in enumerate(target_indices)
            )
            print(
                f"parent {parent_index + 1}/{len(clone_directories)} "
                f"clone {clone_index}/{args.clone_count}: "
                f"events={int(np.sum(label['escaped']))} "
                f"elapsed={time.perf_counter() - start_time:.1f}s",
                flush=True,
            )

    assert reference_horizon is not None and reference_frame_time is not None
    first_passage = np.concatenate(first_passage_rows)
    escaped = np.concatenate(escaped_rows)
    groups = np.concatenate(group_rows)
    features = {model: np.vstack(rows) for model, rows in feature_rows.items()}
    results = {
        model: grouped_exponential_escape_diagnostic(
            values,
            first_passage,
            escaped,
            groups,
            horizon=reference_horizon,
            survival_times=SURVIVAL_TIMES,
            l2_regularization=L2_REGULARIZATION,
        )
        for model, values in features.items()
    }

    detail_rows: list[dict[str, object]] = []
    for observation, metadata in enumerate(metadata_rows):
        detail_rows.append(
            {
                **metadata,
                "first_passage_tau": float(first_passage[observation]),
                "escaped_within_horizon": bool(escaped[observation]),
                **{
                    FEATURE_NAMES[index]: float(features["full"][observation, index])
                    for index in range(len(FEATURE_NAMES))
                },
                **{
                    f"{model}_heldout_rate": float(
                        results[model]["out_of_group_rate"][observation]
                    )
                    for model in results
                },
                **{
                    f"{model}_heldout_event_probability": float(
                        results[model]["out_of_group_event_probability"][observation]
                    )
                    for model in results
                },
                "thermodynamic_claim_allowed": False,
            }
        )

    model_rows: list[dict[str, object]] = []
    survival_rows: list[dict[str, object]] = []
    for model, result in results.items():
        model_rows.append(
            {
                "record": "model",
                "model": model,
                "feature_count": features[model].shape[1],
                "mean_heldout_brier_skill": result["mean_heldout_brier_skill"],
                "mean_heldout_log_likelihood_gain_per_observation": result[
                    "mean_heldout_log_likelihood_gain_per_observation"
                ],
                "minimum_group_log_likelihood_gain": result[
                    "minimum_group_log_likelihood_gain"
                ],
                "maximum_heldout_survival_calibration_error": result[
                    "maximum_heldout_survival_calibration_error"
                ],
                "maximum_baseline_survival_calibration_error": result[
                    "maximum_baseline_survival_calibration_error"
                ],
                "l2_regularization": L2_REGULARIZATION,
                "fit_parameters_from_macro_observables": False,
                "thermodynamic_claim_allowed": False,
            }
        )
        for group_slot, group in enumerate(result["parent_groups"]):
            held = groups == group
            model_rows.append(
                {
                    "record": "held_parent",
                    "model": model,
                    "parent_group": int(group) + 1,
                    "feature_count": features[model].shape[1],
                    "held_observation_count": int(np.sum(held)),
                    "held_event_count": int(np.sum(escaped[held])),
                    "held_brier_skill": float(result["group_brier_skill"][group_slot]),
                    "held_log_likelihood_gain": float(
                        result["group_log_likelihood_gain"][group_slot]
                    ),
                    "held_log_likelihood_gain_per_observation": float(
                        result["group_log_likelihood_gain_per_observation"][group_slot]
                    ),
                    "held_survival_calibration_error": float(
                        result["group_survival_calibration_error"][group_slot]
                    ),
                    "held_baseline_survival_calibration_error": float(
                        result["group_baseline_survival_calibration_error"][group_slot]
                    ),
                    "fit_parameters_from_macro_observables": False,
                    "thermodynamic_claim_allowed": False,
                }
            )
            observed = observed_survival(
                first_passage[held], escaped[held], SURVIVAL_TIMES
            )
            rate = np.asarray(result["out_of_group_rate"])[held]
            baseline_rate = float(np.asarray(result["out_of_group_baseline_rate"])[held][0])
            for survival_time, observed_value in zip(SURVIVAL_TIMES, observed):
                survival_rows.append(
                    {
                        "model": model,
                        "parent_group": int(group) + 1,
                        "time_tau": float(survival_time),
                        "observed_survival": float(observed_value),
                        "predicted_survival": float(np.mean(np.exp(-rate * survival_time))),
                        "constant_rate_survival": float(
                            math.exp(-baseline_rate * survival_time)
                        ),
                        "thermodynamic_claim_allowed": False,
                    }
                )

    full = results["full"]
    geometry = results["geometry"]
    kinematic = results["kinematic"]
    every_group_mixed = all(
        np.any(escaped[groups == group]) and np.any(~escaped[groups == group])
        for group in np.unique(groups)
    )
    exact_protocol = (
        len(clone_directories) == PARENT_COUNT
        and args.clone_count == CLONE_COUNT
        and args.target_count == TARGET_COUNT
        and len(set(parent_hashes)) == PARENT_COUNT
        and np.isclose(reference_horizon, 20.0)
        and np.isclose(reference_frame_time, 0.05)
    )
    integrity_gate = bool(
        exact_protocol
        and every_group_mixed
        and int(np.sum(escaped)) >= 64
        and maximum_reconstruction_error <= 2e-5
    )
    brier_reference_gate = bool(
        float(full["mean_heldout_brier_skill"]) > STRUCTURAL_BRIER_REFERENCE
    )
    likelihood_gate = bool(
        float(full["mean_heldout_log_likelihood_gain_per_observation"]) > 0.0
        and float(full["minimum_group_log_likelihood_gain"]) >= 0.0
    )
    geometry_increment_gate = bool(
        float(full["mean_heldout_brier_skill"])
        >= float(geometry["mean_heldout_brier_skill"]) + 0.01
    )
    kinematic_increment_gate = bool(
        float(full["mean_heldout_brier_skill"])
        >= float(kinematic["mean_heldout_brier_skill"])
    )
    survival_gate = bool(
        float(full["maximum_heldout_survival_calibration_error"]) <= 0.10
    )
    microscopic_initial_escape_state_allowed = bool(
        integrity_gate
        and brier_reference_gate
        and likelihood_gate
        and geometry_increment_gate
        and kinematic_increment_gate
        and survival_gate
    )
    summary = {
        "potential_protocol": "ka_lj_cut",
        "parent_axis_semantics": "five_independent_isoconfigurational_parent_groups",
        "parent_count": len(clone_directories),
        "distinct_parent_restart_hash_count": len(set(parent_hashes)),
        "clone_count_per_parent": args.clone_count,
        "target_count": args.target_count,
        "observation_count": len(first_passage),
        "event_count": int(np.sum(escaped)),
        "censored_count": int(np.sum(~escaped)),
        "frame_time_tau": reference_frame_time,
        "horizon_tau": reference_horizon,
        "event_coordinate": "wendland_c4_force_support_smooth_cage_center",
        "event_definition": "candelier_phop_contiguous_peak_recursive_ABA_removal",
        "phop_threshold": PHOP_THRESHOLD,
        "half_window_frames": HALF_WINDOW,
        "target_seed": TARGET_SEED,
        "l2_regularization": L2_REGULARIZATION,
        "structural_brier_reference": STRUCTURAL_BRIER_REFERENCE,
        "maximum_position_reconstruction_error": maximum_reconstruction_error,
        "full_mean_heldout_brier_skill": full["mean_heldout_brier_skill"],
        "geometry_mean_heldout_brier_skill": geometry["mean_heldout_brier_skill"],
        "kinematic_mean_heldout_brier_skill": kinematic["mean_heldout_brier_skill"],
        "full_mean_heldout_log_likelihood_gain_per_observation": full[
            "mean_heldout_log_likelihood_gain_per_observation"
        ],
        "full_minimum_group_log_likelihood_gain": full[
            "minimum_group_log_likelihood_gain"
        ],
        "full_maximum_heldout_survival_calibration_error": full[
            "maximum_heldout_survival_calibration_error"
        ],
        "integrity_gate_pass": integrity_gate,
        "brier_reference_gate_pass": brier_reference_gate,
        "likelihood_gate_pass": likelihood_gate,
        "geometry_increment_gate_pass": geometry_increment_gate,
        "kinematic_increment_gate_pass": kinematic_increment_gate,
        "survival_gate_pass": survival_gate,
        "microscopic_initial_escape_state_allowed": microscopic_initial_escape_state_allowed,
        "event_clock_claim_allowed": False,
        "autonomous_single_particle_gle_claim_allowed": False,
        "kramers_escape_claim_allowed": False,
        "fit_parameters_from_macro_observables": False,
        "thermodynamic_claim_allowed": False,
    }
    write_rows(output_prefix.with_name(output_prefix.name + "_details.csv"), detail_rows)
    write_rows(output_prefix.with_name(output_prefix.name + "_models.csv"), model_rows)
    write_rows(output_prefix.with_name(output_prefix.name + "_survival.csv"), survival_rows)
    write_rows(output_prefix.with_name(output_prefix.name + "_summary.csv"), [summary])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
