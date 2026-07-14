#!/usr/bin/env python3
"""Audit time-reversal parity in the microscopic relative-generator basis."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_relative_generator_mori import load_clones  # noqa: E402
from ka_relative_memory import estimate_isoconfigurational_bias  # noqa: E402
from ka_relative_mori import (  # noqa: E402
    bias_centered_phase_state,
    parity_detailed_balance_diagnostic,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reversible_generator_state(
    clone: dict[str, np.ndarray | float | str],
    *,
    bias: np.ndarray,
    friction: float,
) -> np.ndarray:
    phase = bias_centered_phase_state(
        np.asarray(clone["relative_position"]),
        np.asarray(clone["relative_velocity"]),
        bias=bias,
    )
    reversible_acceleration = (
        np.asarray(clone["relative_drift"])
        + friction * np.asarray(clone["relative_velocity"])
    ).reshape(len(phase), -1, 1)
    return np.concatenate([phase, reversible_acceleration], axis=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-drift-cache-directory", type=Path, required=True)
    parser.add_argument("--validation-drift-cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--maximum-lag", type=int, default=800)
    parser.add_argument("--friction", type=float, default=1.0)
    args = parser.parse_args()
    if args.maximum_lag < 1 or args.friction <= 0.0:
        raise ValueError("maximum lag and friction must be positive")

    training = load_clones(args.training_drift_cache_directory)
    validation = load_clones(args.validation_drift_cache_directory)
    bias = estimate_isoconfigurational_bias(
        np.asarray([clone["relative_position"] for clone in training])
    )
    training_state = np.concatenate(
        [
            reversible_generator_state(clone, bias=bias, friction=args.friction)
            for clone in training
        ],
        axis=1,
    )
    mean = np.mean(training_state, axis=(0, 1), keepdims=True)
    scale = np.std(training_state, axis=(0, 1), keepdims=True)
    if np.any(scale <= 1e-12):
        raise ValueError("parity coordinates need nonzero training variance")

    correct_parity = np.array([1.0, -1.0, 1.0])
    wrong_parity = np.ones(3)
    details: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for held_index, clone in enumerate(validation, start=1):
        state = (
            reversible_generator_state(clone, bias=bias, friction=args.friction)
            - mean
        ) / scale
        correct = parity_detailed_balance_diagnostic(
            state, parity=correct_parity, maximum_lag=args.maximum_lag
        )
        wrong = parity_detailed_balance_diagnostic(
            state, parity=wrong_parity, maximum_lag=args.maximum_lag
        )
        details.append(
            {
                "record": "held_clone",
                "held_clone_index": float(held_index),
                "training_clone_count": float(len(training)),
                "fit_uses_held_clone": 0.0,
                "resolved_variables": "delta_u,p,Lp_plus_gamma_p",
                "time_reversal_parity": "+1,-1,+1",
                "parity_defect_normalized_rmse": float(
                    correct["parity_defect_normalized_rmse"]
                ),
                "parity_defect_maximum_absolute_error": float(
                    correct["parity_defect_maximum_absolute_error"]
                ),
                "equal_time_maximum_forbidden_parity_correlation": float(
                    correct["equal_time_maximum_forbidden_parity_correlation"]
                ),
                "wrong_all_even_parity_defect_normalized_rmse": float(
                    wrong["parity_defect_normalized_rmse"]
                ),
                "wrong_all_even_parity_defect_maximum_absolute_error": float(
                    wrong["parity_defect_maximum_absolute_error"]
                ),
                "thermodynamic_claim_allowed": 0.0,
            }
        )
        observed = np.asarray(correct["correlation"])
        reversed_correlation = np.asarray(correct["parity_reversed_correlation"])
        labels = ("u", "p", "a_rev")
        for lag in range(args.maximum_lag + 1):
            for left in range(3):
                for right in range(3):
                    curves.append(
                        {
                            "held_clone_index": float(held_index),
                            "lag_frames": float(lag),
                            "left_variable": labels[left],
                            "right_variable": labels[right],
                            "observed_correlation": float(observed[lag, left, right]),
                            "parity_reversed_correlation": float(
                                reversed_correlation[lag, left, right]
                            ),
                            "thermodynamic_claim_allowed": 0.0,
                        }
                    )

    metric_names = tuple(
        key
        for key in details[0]
        if key
        not in {
            "record",
            "held_clone_index",
            "training_clone_count",
            "fit_uses_held_clone",
            "resolved_variables",
            "time_reversal_parity",
            "thermodynamic_claim_allowed",
        }
    )
    aggregate: dict[str, object] = {
        "record": "aggregate",
        "held_clone_count": float(len(details)),
        "fit_uses_held_clone": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for metric in metric_names:
        values = np.asarray([float(row[metric]) for row in details])
        aggregate[metric] = float(np.mean(values))
        aggregate[f"maximum_{metric}"] = float(np.max(values))
        aggregate[f"minimum_{metric}"] = float(np.min(values))
    parity_supported = (
        float(aggregate["maximum_parity_defect_normalized_rmse"]) <= 0.03
        and float(aggregate["maximum_parity_defect_maximum_absolute_error"]) <= 0.01
        and float(
            aggregate["maximum_equal_time_maximum_forbidden_parity_correlation"]
        )
        <= 0.01
    )
    wrong_rejected = (
        float(aggregate["minimum_wrong_all_even_parity_defect_normalized_rmse"])
        >= 0.50
    )
    verdict = {
        "record": "verdict",
        "parity_definite_generator_basis_allowed": 1.0,
        "resolved_generalized_detailed_balance_supported": float(parity_supported),
        "wrong_all_even_parity_rejected": float(wrong_rejected),
        "thermal_fdt_adjoint_audit_pass": 0.0,
        "physical_relative_generator_gle_allowed": 0.0,
        "orthogonal_noise_generation_closed": 0.0,
        "autonomous_single_particle_gle_allowed": 0.0,
        "complete_event_clock_closure_allowed": 0.0,
        "kramers_escape_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curves)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
        [aggregate, verdict],
    )


if __name__ == "__main__":
    main()
