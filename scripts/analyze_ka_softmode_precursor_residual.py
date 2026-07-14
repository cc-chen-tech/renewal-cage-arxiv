#!/usr/bin/env python3
"""Test local KA Hessian modes on frozen smooth-cage escape labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_radial_precursor_residual import (  # noqa: E402
    CLONE_INVARIANCE_TOLERANCE,
    GEOMETRY_REFERENCE,
    GEOMETRY_REPRODUCTION_TOLERANCE,
    L2_REGULARIZATION,
    STRUCTURAL_BRIER_REFERENCE,
    SURVIVAL_TIMES,
    load_parent_state_and_labels,
    write_rows,
)
from ka_local_cage import grouped_binomial_logistic_committor_diagnostic  # noqa: E402
from ka_smooth_cage import grouped_exponential_escape_diagnostic  # noqa: E402
from ka_structural_precursor import (  # noqa: E402
    expand_isoconfigurational_structural_rows,
    instantaneous_local_soft_mode_features,
)


CLUSTER_CUTOFF = 1.5
MODE_RANKS = (0, 3)
EIGENVALUE_FLOOR = 1e-6
EXPECTED_PARENT_COUNT = 5
EXPECTED_CLONE_COUNT = 8
EXPECTED_TARGET_COUNT = 64
EXPECTED_EVENT_COUNT = 1731
EXPECTED_CENSORED_COUNT = 829


def evaluate_softmode_precursor_gates(
    metrics: dict[str, float | bool],
) -> dict[str, bool]:
    combined_brier = float(metrics["geometry_softmode_mean_heldout_brier_skill"])
    brier_increment_gate = combined_brier >= max(
        float(metrics["geometry_mean_heldout_brier_skill"]),
        float(metrics["softmode_mean_heldout_brier_skill"]),
    ) + 0.01
    brier_reference_gate = combined_brier > STRUCTURAL_BRIER_REFERENCE
    likelihood_gate = (
        float(
            metrics[
                "geometry_softmode_mean_heldout_log_likelihood_gain_per_observation"
            ]
        )
        > 0.0
        and float(metrics["geometry_softmode_minimum_group_log_likelihood_gain"])
        >= 0.0
    )
    survival_gate = (
        float(
            metrics[
                "geometry_softmode_maximum_heldout_survival_calibration_error"
            ]
        )
        <= 0.10
    )
    binomial_gate = (
        float(metrics["geometry_softmode_binomial_mean_heldout_brier_skill"])
        > 0.0
        and float(metrics["geometry_softmode_binomial_mean_heldout_brier_skill"])
        > float(metrics["geometry_binomial_mean_heldout_brier_skill"])
    )
    allowed = bool(
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
        "instantaneous_local_softmode_precursor_allowed": allowed,
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

    cache_directory = args.cache_directory.resolve()
    output_prefix = args.output_prefix.resolve()
    parents: list[dict[str, object]] = []
    reference_targets: np.ndarray | None = None
    softmode_names: tuple[str, ...] | None = None
    for parent_slot, path in enumerate(args.initial_state_directories):
        parent = load_parent_state_and_labels(
            path.resolve(),
            cache_directory,
            parent_slot=parent_slot,
            clone_count=args.clone_count,
            reference_target_indices=reference_targets,
        )
        if reference_targets is None:
            reference_targets = np.asarray(parent["target_indices"], dtype=int)
        feature, names = instantaneous_local_soft_mode_features(
            np.asarray(parent["positions"], dtype=float),
            np.asarray(parent["particle_types"], dtype=int),
            np.asarray(parent["box_lengths"], dtype=float),
            reference_targets,
            cluster_cutoff=CLUSTER_CUTOFF,
            ranks=MODE_RANKS,
            eigenvalue_floor=EIGENVALUE_FLOOR,
        )
        if softmode_names is None:
            softmode_names = names
        elif names != softmode_names:
            raise ValueError("soft-mode feature names vary across parents")
        parent["softmode"] = feature
        parents.append(parent)
        print(
            f"parent {parent_slot + 1}/{len(args.initial_state_directories)}: "
            f"events={int(np.sum(parent['escaped']))}",
            flush=True,
        )
    assert reference_targets is not None
    assert softmode_names is not None

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
        "softmode": np.stack(
            [np.asarray(parent["softmode"], dtype=float) for parent in parents]
        ),
    }
    parent_features["geometry_softmode"] = np.concatenate(
        [parent_features["geometry"], parent_features["softmode"]], axis=2
    )
    expanded = {
        model: expand_isoconfigurational_structural_rows(
            values,
            first_passage_tensor,
            escaped_tensor,
            horizon=20.0,
        )
        for model, values in parent_features.items()
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

    geometry_reference_errors = {
        key: abs(float(survival_results["geometry"][key]) - expected)
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
    configuration_count = len(common["configuration_features"])
    event_count = int(np.sum(escaped))
    censored_count = int(np.sum(~escaped))
    integrity_gate_pass = bool(
        len(parents) == EXPECTED_PARENT_COUNT
        and args.clone_count == EXPECTED_CLONE_COUNT
        and len(reference_targets) == EXPECTED_TARGET_COUNT
        and observation_count == 2560
        and configuration_count == 320
        and event_count == EXPECTED_EVENT_COUNT
        and censored_count == EXPECTED_CENSORED_COUNT
        and len(set(parent_hashes)) == EXPECTED_PARENT_COUNT
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
    gates = evaluate_softmode_precursor_gates(metrics)

    model_rows: list[dict[str, object]] = []
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
                "thermodynamic_claim_allowed": False,
            }
        )
        for slot, group in enumerate(result["parent_groups"]):
            held = groups == group
            model_rows.append(
                {
                    "record": "held_parent",
                    "model": model,
                    "parent_group": int(group) + 1,
                    "held_observation_count": int(np.sum(held)),
                    "held_event_count": int(np.sum(escaped[held])),
                    "held_brier_skill": float(result["group_brier_skill"][slot]),
                    "held_log_likelihood_gain": float(
                        result["group_log_likelihood_gain"][slot]
                    ),
                    "held_survival_calibration_error": float(
                        result["group_survival_calibration_error"][slot]
                    ),
                    "thermodynamic_claim_allowed": False,
                }
            )

    configuration_groups = np.asarray(common["configuration_groups"], dtype=int)
    successes = np.asarray(common["successes"], dtype=float)
    trials = np.asarray(common["trials"], dtype=float)
    committor_rows: list[dict[str, object]] = []
    for row in range(configuration_count):
        target_slot = row % len(reference_targets)
        committor_rows.append(
            {
                "parent_group": int(configuration_groups[row]) + 1,
                "target_slot": target_slot,
                "a_particle_index": int(reference_targets[target_slot]),
                "escape_successes": int(successes[row]),
                "clone_trials": int(trials[row]),
                "observed_committor": float(successes[row] / trials[row]),
                **{
                    f"{model}_heldout_committor": float(
                        binomial_results[model]["out_of_group_prediction"][row]
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
        "cluster_cutoff": CLUSTER_CUTOFF,
        "mode_ranks": ";".join(str(rank) for rank in MODE_RANKS),
        "eigenvalue_floor": EIGENVALUE_FLOOR,
        "softmode_feature_names": ";".join(softmode_names),
        "maximum_clone_position_difference": maximum_clone_position_difference,
        "maximum_geometry_reference_error": max(geometry_reference_errors.values()),
        **metrics,
        **gates,
        "event_clock_claim_allowed": False,
        "autonomous_single_particle_gle_claim_allowed": False,
        "kramers_escape_claim_allowed": False,
        "thermodynamic_claim_allowed": False,
    }
    write_rows(output_prefix.with_name(output_prefix.name + "_models.csv"), model_rows)
    write_rows(
        output_prefix.with_name(output_prefix.name + "_committor.csv"), committor_rows
    )
    write_rows(output_prefix.with_name(output_prefix.name + "_summary.csv"), [summary])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
