#!/usr/bin/env python3
"""Test species-resolved structure beyond smooth-cage escape geometry."""

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

from ka_local_cage import grouped_binomial_logistic_committor_diagnostic  # noqa: E402
from ka_smooth_cage import (  # noqa: E402
    grouped_exponential_escape_diagnostic,
    smooth_cage_geometry_features,
)
from ka_structural_precursor import (  # noqa: E402
    expand_isoconfigurational_structural_rows,
    species_resolved_radial_features,
)


RADII = np.array([0.8, 1.05, 1.3, 1.55, 1.8, 2.05, 2.3], dtype=float)
RADIAL_WIDTH = 0.12
RADIAL_CUTOFF = 2.5
SURVIVAL_TIMES = np.array([1, 2, 4, 8, 12, 16, 20], dtype=float)
L2_REGULARIZATION = 1.0
EXPECTED_PARENT_COUNT = 5
EXPECTED_CLONE_COUNT = 8
EXPECTED_TARGET_COUNT = 64
EXPECTED_EVENT_COUNT = 1731
EXPECTED_CENSORED_COUNT = 829
STRUCTURAL_BRIER_REFERENCE = 0.026964
GEOMETRY_REFERENCE = {
    "mean_heldout_brier_skill": 0.008357453921058,
    "mean_heldout_log_likelihood_gain_per_observation": 0.0064617896947432914,
    "minimum_group_log_likelihood_gain": 1.2946834014508113,
    "maximum_heldout_survival_calibration_error": 0.06742767815935347,
}
GEOMETRY_REPRODUCTION_TOLERANCE = 1e-12
CLONE_INVARIANCE_TOLERANCE = 1e-12


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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


def evaluate_radial_precursor_gates(
    metrics: dict[str, float | bool],
) -> dict[str, bool]:
    """Apply the preregistered residual-information decision gates."""

    combined_brier = float(metrics["geometry_radial_mean_heldout_brier_skill"])
    geometry_brier = float(metrics["geometry_mean_heldout_brier_skill"])
    radial_brier = float(metrics["radial_mean_heldout_brier_skill"])
    brier_increment_gate = combined_brier >= max(geometry_brier, radial_brier) + 0.01
    brier_reference_gate = combined_brier > STRUCTURAL_BRIER_REFERENCE
    likelihood_gate = (
        float(
            metrics[
                "geometry_radial_mean_heldout_log_likelihood_gain_per_observation"
            ]
        )
        > 0.0
        and float(metrics["geometry_radial_minimum_group_log_likelihood_gain"])
        >= 0.0
    )
    survival_gate = (
        float(
            metrics[
                "geometry_radial_maximum_heldout_survival_calibration_error"
            ]
        )
        <= 0.10
    )
    binomial_gate = (
        float(metrics["geometry_radial_binomial_mean_heldout_brier_skill"]) > 0.0
        and float(metrics["geometry_radial_binomial_mean_heldout_brier_skill"])
        > float(metrics["geometry_binomial_mean_heldout_brier_skill"])
    )
    static_radial_precursor_allowed = bool(
        metrics["integrity_gate_pass"]
        and metrics["geometry_reproduction_gate_pass"]
        and metrics["clone_invariance_gate_pass"]
        and brier_increment_gate
        and brier_reference_gate
        and likelihood_gate
        and survival_gate
        and binomial_gate
    )
    return {
        "brier_increment_gate_pass": bool(brier_increment_gate),
        "brier_reference_gate_pass": bool(brier_reference_gate),
        "likelihood_gate_pass": bool(likelihood_gate),
        "survival_gate_pass": bool(survival_gate),
        "binomial_gate_pass": bool(binomial_gate),
        "static_radial_precursor_allowed": static_radial_precursor_allowed,
    }


def load_parent_state_and_labels(
    initial_directory: Path,
    cache_directory: Path,
    *,
    parent_slot: int,
    clone_count: int,
    reference_target_indices: np.ndarray | None,
) -> dict[str, object]:
    manifest = json.loads((initial_directory / "manifest.json").read_text())
    if bool(manifest["thermodynamic_claim_allowed"]):
        raise ValueError("initial-state manifest violates thermodynamic claim boundary")
    if int(manifest["clone_count"]) < clone_count:
        raise ValueError("initial-state manifest has too few clones")

    parent_positions: np.ndarray | None = None
    parent_types: np.ndarray | None = None
    parent_box: np.ndarray | None = None
    first_passage_rows: list[np.ndarray] = []
    escaped_rows: list[np.ndarray] = []
    target_indices: np.ndarray | None = None
    maximum_position_difference = 0.0
    frame_time: float | None = None
    for clone_index in range(1, clone_count + 1):
        label_path = (
            cache_directory
            / f"parent_{parent_slot + 1:02d}_clone_{clone_index:03d}_labels.npz"
        )
        state_path = initial_directory / f"clone_{clone_index:03d}_initial.npz"
        with np.load(label_path, allow_pickle=False) as label:
            current_targets = np.asarray(label["target_indices"], dtype=int)
            label_first_passage = np.asarray(label["first_passage"], dtype=float)
            label_escaped = np.asarray(label["escaped"], dtype=bool)
            label_types = np.asarray(label["particle_types"], dtype=int)
            label_box = np.asarray(label["box_lengths"], dtype=float)
            if not np.isclose(float(label["threshold"]), 0.08):
                raise ValueError("label cache p_hop threshold differs from 0.08")
            if int(label["half_window"]) != 8:
                raise ValueError("label cache half-window differs from 8")
            if not np.isclose(float(label["horizon"]), 20.0):
                raise ValueError("label cache horizon differs from 20 tau")
            if float(label["thermodynamic_claim_allowed"]) != 0.0:
                raise ValueError("label cache violates thermodynamic claim boundary")
            current_frame_time = float(label["frame_time"])
        if target_indices is None:
            target_indices = current_targets
            frame_time = current_frame_time
        elif not np.array_equal(current_targets, target_indices):
            raise ValueError("target indices vary across clone label caches")
        if reference_target_indices is not None and not np.array_equal(
            current_targets, reference_target_indices
        ):
            raise ValueError("target indices vary across parents")
        if not np.isclose(current_frame_time, frame_time):
            raise ValueError("frame time varies across clone label caches")

        with np.load(state_path, allow_pickle=False) as state:
            positions = np.asarray(state["positions"], dtype=float)
            particle_types = np.asarray(state["particle_types"], dtype=int)
            box_lengths = np.asarray(state["box_lengths"], dtype=float)
            state_parent_hash = str(state["parent_restart_sha256"])
            if float(state["thermodynamic_claim_allowed"]) != 0.0:
                raise ValueError("initial state violates thermodynamic claim boundary")
        if state_parent_hash != str(manifest["parent_restart_sha256"]):
            raise ValueError("initial state and manifest parent hashes differ")
        if not np.array_equal(particle_types, label_types) or not np.allclose(
            box_lengths, label_box, rtol=0.0, atol=1e-12
        ):
            raise ValueError("initial state and label particle metadata differ")
        if np.any(particle_types[current_targets] != 0):
            raise ValueError("every fixed target must be an A particle")
        if parent_positions is None:
            parent_positions = positions
            parent_types = particle_types
            parent_box = box_lengths
        else:
            maximum_position_difference = max(
                maximum_position_difference,
                float(np.max(np.abs(positions - parent_positions))),
            )
            if not np.array_equal(particle_types, parent_types) or not np.allclose(
                box_lengths, parent_box, rtol=0.0, atol=1e-12
            ):
                raise ValueError("initial microscopic metadata varies across clones")
        first_passage_rows.append(label_first_passage)
        escaped_rows.append(label_escaped)

    assert parent_positions is not None
    assert parent_types is not None
    assert parent_box is not None
    assert target_indices is not None
    assert frame_time is not None
    geometry = smooth_cage_geometry_features(
        parent_positions,
        particle_types=parent_types,
        box_lengths=parent_box,
        target_indices=target_indices,
    )
    radial = species_resolved_radial_features(
        parent_positions,
        parent_types,
        parent_box,
        target_indices,
        radii=RADII,
        width=RADIAL_WIDTH,
        cutoff=RADIAL_CUTOFF,
    )
    return {
        "parent_restart_sha256": str(manifest["parent_restart_sha256"]),
        "target_indices": target_indices,
        "first_passage": np.asarray(first_passage_rows),
        "escaped": np.asarray(escaped_rows),
        "geometry": geometry,
        "radial": radial,
        "maximum_clone_position_difference": maximum_position_difference,
        "frame_time": frame_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-state-directories", type=Path, nargs="+", required=True
    )
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--clone-count", type=int, default=EXPECTED_CLONE_COUNT)
    args = parser.parse_args()
    if len(args.initial_state_directories) < 2 or args.clone_count < 2:
        raise ValueError("at least two parents and two clones are required")

    initial_directories = [path.resolve() for path in args.initial_state_directories]
    cache_directory = args.cache_directory.resolve()
    output_prefix = args.output_prefix.resolve()
    parents: list[dict[str, object]] = []
    reference_targets: np.ndarray | None = None
    for parent_slot, initial_directory in enumerate(initial_directories):
        parent = load_parent_state_and_labels(
            initial_directory,
            cache_directory,
            parent_slot=parent_slot,
            clone_count=args.clone_count,
            reference_target_indices=reference_targets,
        )
        if reference_targets is None:
            reference_targets = np.asarray(parent["target_indices"], dtype=int)
        parents.append(parent)
        print(
            f"parent {parent_slot + 1}/{len(initial_directories)}: "
            f"events={int(np.sum(parent['escaped']))}",
            flush=True,
        )
    assert reference_targets is not None

    first_passage_tensor = np.stack(
        [np.asarray(parent["first_passage"], dtype=float) for parent in parents]
    )
    escaped_tensor = np.stack(
        [np.asarray(parent["escaped"], dtype=bool) for parent in parents]
    )
    parent_features = {
        "geometry": np.stack(
            [np.asarray(parent["geometry"], dtype=float) for parent in parents]
        ),
        "radial": np.stack(
            [np.asarray(parent["radial"], dtype=float) for parent in parents]
        ),
    }
    parent_features["geometry_radial"] = np.concatenate(
        [parent_features["geometry"], parent_features["radial"]], axis=2
    )
    expanded = {
        model: expand_isoconfigurational_structural_rows(
            feature,
            first_passage_tensor,
            escaped_tensor,
            horizon=20.0,
        )
        for model, feature in parent_features.items()
    }
    common = expanded["geometry"]
    first_passage = np.asarray(common["first_passage"], dtype=float)
    escaped = np.asarray(common["escaped"], dtype=bool)
    groups = np.asarray(common["groups"], dtype=int)

    survival_results = {
        model: grouped_exponential_escape_diagnostic(
            np.asarray(rows["features"], dtype=float),
            first_passage,
            escaped,
            groups,
            horizon=20.0,
            survival_times=SURVIVAL_TIMES,
            l2_regularization=L2_REGULARIZATION,
        )
        for model, rows in expanded.items()
    }
    binomial_results = {
        model: grouped_binomial_logistic_committor_diagnostic(
            np.asarray(rows["configuration_features"], dtype=float),
            np.asarray(rows["successes"], dtype=float),
            np.asarray(rows["trials"], dtype=float),
            np.asarray(rows["configuration_groups"], dtype=int),
            l2_regularization=L2_REGULARIZATION,
        )
        for model, rows in expanded.items()
    }

    geometry_result = survival_results["geometry"]
    geometry_reference_errors = {
        key: abs(float(geometry_result[key]) - expected)
        for key, expected in GEOMETRY_REFERENCE.items()
    }
    geometry_reproduction_gate_pass = bool(
        max(geometry_reference_errors.values()) <= GEOMETRY_REPRODUCTION_TOLERANCE
    )
    maximum_clone_position_difference = max(
        float(parent["maximum_clone_position_difference"]) for parent in parents
    )
    clone_invariance_gate_pass = bool(
        maximum_clone_position_difference <= CLONE_INVARIANCE_TOLERANCE
    )
    parent_hashes = [str(parent["parent_restart_sha256"]) for parent in parents]
    observation_count = len(first_passage)
    event_count = int(np.sum(escaped))
    censored_count = int(np.sum(~escaped))
    configuration_count = len(common["configuration_features"])
    integrity_gate_pass = bool(
        len(parents) == EXPECTED_PARENT_COUNT
        and args.clone_count == EXPECTED_CLONE_COUNT
        and len(reference_targets) == EXPECTED_TARGET_COUNT
        and observation_count
        == EXPECTED_PARENT_COUNT * EXPECTED_CLONE_COUNT * EXPECTED_TARGET_COUNT
        and configuration_count == EXPECTED_PARENT_COUNT * EXPECTED_TARGET_COUNT
        and event_count == EXPECTED_EVENT_COUNT
        and censored_count == EXPECTED_CENSORED_COUNT
        and len(set(parent_hashes)) == EXPECTED_PARENT_COUNT
        and all(np.isclose(float(parent["frame_time"]), 0.05) for parent in parents)
    )

    metrics: dict[str, float | bool] = {
        "integrity_gate_pass": integrity_gate_pass,
        "geometry_reproduction_gate_pass": geometry_reproduction_gate_pass,
        "clone_invariance_gate_pass": clone_invariance_gate_pass,
    }
    for model, result in survival_results.items():
        metrics[f"{model}_mean_heldout_brier_skill"] = float(
            result["mean_heldout_brier_skill"]
        )
        metrics[
            f"{model}_mean_heldout_log_likelihood_gain_per_observation"
        ] = float(result["mean_heldout_log_likelihood_gain_per_observation"])
        metrics[f"{model}_minimum_group_log_likelihood_gain"] = float(
            result["minimum_group_log_likelihood_gain"]
        )
        metrics[
            f"{model}_maximum_heldout_survival_calibration_error"
        ] = float(result["maximum_heldout_survival_calibration_error"])
        metrics[f"{model}_binomial_mean_heldout_brier_skill"] = float(
            binomial_results[model]["mean_heldout_brier_skill"]
        )
        metrics[
            f"{model}_binomial_mean_heldout_log_likelihood_gain_per_trial"
        ] = float(
            binomial_results[model][
                "mean_heldout_log_likelihood_gain_per_trial"
            ]
        )
    gates = evaluate_radial_precursor_gates(metrics)

    model_rows: list[dict[str, object]] = []
    survival_rows: list[dict[str, object]] = []
    for model, result in survival_results.items():
        model_rows.append(
            {
                "record": "censored_model",
                "model": model,
                "feature_count": expanded[model]["features"].shape[1],
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
                "l2_regularization": L2_REGULARIZATION,
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
                    "thermodynamic_claim_allowed": False,
                }
            )
            held_first_passage = first_passage[held]
            held_escaped = escaped[held]
            rate = np.asarray(result["out_of_group_rate"])[held]
            baseline_rate = float(
                np.asarray(result["out_of_group_baseline_rate"])[held][0]
            )
            for time, observed in zip(
                SURVIVAL_TIMES,
                observed_survival(held_first_passage, held_escaped, SURVIVAL_TIMES),
            ):
                survival_rows.append(
                    {
                        "model": model,
                        "parent_group": int(group) + 1,
                        "time_tau": float(time),
                        "observed_survival": float(observed),
                        "predicted_survival": float(np.mean(np.exp(-rate * time))),
                        "constant_rate_survival": math.exp(-baseline_rate * time),
                        "thermodynamic_claim_allowed": False,
                    }
                )

    detail_rows: list[dict[str, object]] = []
    radial_names = [
        f"radial_{species}_{radius:g}"
        for species in ("A", "B")
        for radius in RADII
    ]
    clone_index = np.tile(
        np.repeat(np.arange(1, args.clone_count + 1), len(reference_targets)),
        len(parents),
    )
    target_slot = np.tile(
        np.arange(len(reference_targets)), len(parents) * args.clone_count
    )
    for row in range(observation_count):
        radial_values = expanded["radial"]["features"][row]
        detail_rows.append(
            {
                "parent_group": int(groups[row]) + 1,
                "clone_index": int(clone_index[row]),
                "target_slot": int(target_slot[row]),
                "a_particle_index": int(reference_targets[target_slot[row]]),
                "first_passage_tau": float(first_passage[row]),
                "escaped_within_horizon": bool(escaped[row]),
                **{
                    f"geometry_{index}": float(
                        expanded["geometry"]["features"][row, index]
                    )
                    for index in range(4)
                },
                **{
                    name: float(value)
                    for name, value in zip(radial_names, radial_values)
                },
                **{
                    f"{model}_heldout_rate": float(
                        survival_results[model]["out_of_group_rate"][row]
                    )
                    for model in survival_results
                },
                **{
                    f"{model}_heldout_event_probability": float(
                        survival_results[model]["out_of_group_event_probability"][row]
                    )
                    for model in survival_results
                },
                "thermodynamic_claim_allowed": False,
            }
        )

    committor_rows: list[dict[str, object]] = []
    configuration_groups = np.asarray(common["configuration_groups"], dtype=int)
    successes = np.asarray(common["successes"], dtype=float)
    trials = np.asarray(common["trials"], dtype=float)
    for row in range(configuration_count):
        slot = row % len(reference_targets)
        committor_rows.append(
            {
                "parent_group": int(configuration_groups[row]) + 1,
                "target_slot": slot,
                "a_particle_index": int(reference_targets[slot]),
                "escape_successes": int(successes[row]),
                "clone_trials": int(trials[row]),
                "observed_committor": float(successes[row] / trials[row]),
                **{
                    f"{model}_heldout_committor": float(
                        binomial_results[model]["out_of_group_prediction"][row]
                    )
                    for model in binomial_results
                },
                **{
                    f"{model}_heldout_baseline_committor": float(
                        binomial_results[model]["out_of_group_baseline_prediction"][row]
                    )
                    for model in binomial_results
                },
                "thermodynamic_claim_allowed": False,
            }
        )

    summary: dict[str, object] = {
        "temperature": 0.58,
        "parent_count": len(parents),
        "distinct_parent_restart_hash_count": len(set(parent_hashes)),
        "clone_count_per_parent": args.clone_count,
        "target_count": len(reference_targets),
        "observation_count": observation_count,
        "configuration_count": configuration_count,
        "event_count": event_count,
        "censored_count": censored_count,
        "horizon_tau": 20.0,
        "phop_threshold": 0.08,
        "half_window_frames": 8,
        "l2_regularization": L2_REGULARIZATION,
        "radial_radii": ";".join(f"{radius:g}" for radius in RADII),
        "radial_width": RADIAL_WIDTH,
        "radial_cutoff": RADIAL_CUTOFF,
        "structural_brier_reference": STRUCTURAL_BRIER_REFERENCE,
        "maximum_clone_position_difference": maximum_clone_position_difference,
        "maximum_geometry_reference_error": max(geometry_reference_errors.values()),
        **metrics,
        **gates,
        "event_clock_claim_allowed": False,
        "autonomous_single_particle_gle_claim_allowed": False,
        "kramers_escape_claim_allowed": False,
        "fit_parameters_from_macro_observables": False,
        "thermodynamic_claim_allowed": False,
    }
    write_rows(output_prefix.with_name(output_prefix.name + "_details.csv"), detail_rows)
    write_rows(output_prefix.with_name(output_prefix.name + "_models.csv"), model_rows)
    write_rows(output_prefix.with_name(output_prefix.name + "_survival.csv"), survival_rows)
    write_rows(
        output_prefix.with_name(output_prefix.name + "_committor.csv"), committor_rows
    )
    write_rows(output_prefix.with_name(output_prefix.name + "_summary.csv"), [summary])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
