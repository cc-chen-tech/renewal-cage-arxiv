#!/usr/bin/env python3
"""Audit isoconfigurational full-KA pair-force response closures.

Each member is a common-random-number ``+/-`` displacement experiment.  The
script first averages the *linear responses* over members, then tests whether
a scalar causal position-memory kernel inferred on an early window predicts
the later ensemble-mean response.  No macroscopic observable is fit here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import (  # noqa: E402
    ensemble_symmetric_response_summary,
    finite_memory_position_response_holdout,
    fit_linear_response_auxiliary_embedding,
    fit_pair_force_response_auxiliary_embedding,
    shared_response_prefix_length,
    symmetric_finite_difference_response,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def load_pair(directory: Path, epsilon: float) -> dict[str, np.ndarray]:
    positive = np.loadtxt(directory / f"plus_eps{epsilon:.3f}.dat", comments="#")
    negative = np.loadtxt(directory / f"minus_eps{epsilon:.3f}.dat", comments="#")
    if positive.ndim != 2 or positive.shape != negative.shape or positive.shape[1] != 13:
        raise ValueError(f"{directory}: expected matched 13-column decomposed response files")
    if not np.array_equal(positive[:, 0], negative[:, 0]):
        raise ValueError(f"{directory}: plus/minus timesteps differ")

    def response(columns: slice) -> np.ndarray:
        return symmetric_finite_difference_response(
            positive[:, columns], negative[:, columns], displacement=epsilon
        )["response"]

    return {
        "steps": positive[:, 0],
        "position": response(slice(1, 4)),
        "velocity": response(slice(4, 7)),
        "total_force": response(slice(7, 10)),
        "pair_force": response(slice(10, 13)),
    }


def relative_l2_prefix(current: np.ndarray, reference: np.ndarray, stop: int) -> float:
    denominator = float(np.linalg.norm(reference[:stop]))
    if denominator <= 0.0:
        raise ValueError("reference response is zero on a requested linearity horizon")
    return float(np.linalg.norm(current[:stop] - reference[:stop]) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("member_directories", type=Path, nargs="+", help="one directory per isoconfigurational member")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.001, 0.002])
    parser.add_argument("--time-step", type=float, default=0.001)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--horizons", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0])
    parser.add_argument("--fit-times", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    args = parser.parse_args()
    if len(args.member_directories) < 2 or not args.epsilons or min(args.epsilons) <= 0.0:
        raise ValueError("at least two members and positive epsilons are required")
    if args.time_step <= 0.0 or args.mass <= 0.0 or args.friction < 0.0:
        raise ValueError("time-step and mass must be positive and friction nonnegative")

    raw = {
        epsilon: [load_pair(directory, epsilon) for directory in args.member_directories]
        for epsilon in args.epsilons
    }
    reference_epsilon = min(args.epsilons)
    frame_count = shared_response_prefix_length(
        [member["steps"] for members in raw.values() for member in members]
    )
    raw = {
        epsilon: [{key: value[:frame_count] for key, value in member.items()} for member in members]
        for epsilon, members in raw.items()
    }
    reference_steps = raw[reference_epsilon][0]["steps"]

    summary: dict[float, dict[str, dict[str, np.ndarray | int]]] = {}
    for epsilon, members in raw.items():
        summary[epsilon] = {
            key: ensemble_symmetric_response_summary(np.stack([member[key] for member in members]))
            for key in ("position", "velocity", "total_force", "pair_force")
        }

    times = reference_steps * args.time_step
    reference_position = np.asarray(summary[reference_epsilon]["position"]["mean"])
    reference_velocity = np.asarray(summary[reference_epsilon]["velocity"]["mean"])
    reference_pair_force = np.asarray(summary[reference_epsilon]["pair_force"]["mean"])
    reference_pair_force_rate = np.gradient(reference_pair_force[:, 0], args.time_step, edge_order=2)
    member_count = int(summary[reference_epsilon]["position"]["member_count"])
    summary_rows: list[dict[str, object]] = []
    for epsilon in args.epsilons:
        position = np.asarray(summary[epsilon]["position"]["mean"])
        pair_force = np.asarray(summary[epsilon]["pair_force"]["mean"])
        position_sem = np.asarray(summary[epsilon]["position"]["standard_error"])
        pair_sem = np.asarray(summary[epsilon]["pair_force"]["standard_error"])
        for horizon in args.horizons:
            stop = int(np.searchsorted(times, horizon, side="right"))
            if stop < 2 or stop > len(times):
                raise ValueError(f"horizon {horizon} lies outside the response grid")
            summary_rows.append(
                {
                    "record": "cross_epsilon_linearity",
                    "epsilon": epsilon,
                    "member_count": member_count,
                    "horizon_time": float(times[stop - 1]),
                    "horizon_frames": stop,
                    "position_reference_relative_l2": relative_l2_prefix(position[:, 0], reference_position[:, 0], stop),
                    "pair_force_reference_relative_l2": relative_l2_prefix(pair_force[:, 0], reference_pair_force[:, 0], stop),
                    "position_response_x_standard_error": float(position_sem[stop - 1, 0]),
                    "pair_force_response_x_standard_error": float(pair_sem[stop - 1, 0]),
                    "fit_time": "",
                    "fit_frames": "",
                    "training_relative_l2_error": "",
                    "heldout_relative_l2_error": "",
                    "heldout_maximum_absolute_error": "",
                    "response_scope": "full_KA_Langevin_isoconfigurational_pair_force_response",
                    "thermodynamic_claim_allowed": 0,
                }
            )
    for fit_time in args.fit_times:
        fit_frames = int(np.searchsorted(times, fit_time, side="right"))
        if not (2 <= fit_frames < len(times)):
            raise ValueError(f"fit time {fit_time} must leave a temporal holdout")
        result = finite_memory_position_response_holdout(
            reference_position[:, 0],
            reference_pair_force[:, 0],
            frame_time=args.time_step,
            mass=args.mass,
            friction=args.friction,
            fit_frames=fit_frames,
        )
        summary_rows.append(
            {
                "record": "scalar_finite_memory_temporal_holdout",
                "epsilon": reference_epsilon,
                "member_count": member_count,
                "horizon_time": float(times[-1]),
                "horizon_frames": len(times),
                "position_reference_relative_l2": "",
                "pair_force_reference_relative_l2": "",
                "position_response_x_standard_error": float(np.asarray(summary[reference_epsilon]["position"]["standard_error"])[-1, 0]),
                "pair_force_response_x_standard_error": float(np.asarray(summary[reference_epsilon]["pair_force"]["standard_error"])[-1, 0]),
                "fit_time": float(times[fit_frames - 1]),
                "fit_frames": fit_frames,
                "training_relative_l2_error": float(result["training_relative_l2_error"]),
                "heldout_relative_l2_error": float(result["heldout_relative_l2_error"]),
                "heldout_maximum_absolute_error": float(result["heldout_maximum_absolute_error"]),
                "response_scope": "ensemble_mean_scalar_causal_memory_with_explicit_Langevin_friction",
                "thermodynamic_claim_allowed": 0,
            }
        )
        auxiliary = fit_pair_force_response_auxiliary_embedding(
            np.column_stack(
                [reference_position[:, 0], reference_velocity[:, 0], reference_pair_force[:, 0]]
            ),
            fit_frames=fit_frames,
        )
        predicted_position = np.asarray(auxiliary["predicted_state_response"])[:, 0]
        for horizon in args.horizons:
            stop = int(np.searchsorted(times, horizon, side="right"))
            if stop <= fit_frames:
                continue
            held_difference = predicted_position[fit_frames:stop] - reference_position[fit_frames:stop, 0]
            held_norm = float(np.linalg.norm(reference_position[fit_frames:stop, 0]))
            if held_norm <= 0.0:
                raise ValueError("auxiliary held-out position response is zero")
            summary_rows.append(
                {
                    "record": "pair_force_auxiliary_markov_temporal_holdout",
                    "epsilon": reference_epsilon,
                    "member_count": member_count,
                    "horizon_time": float(times[stop - 1]),
                    "horizon_frames": stop,
                    "position_reference_relative_l2": "",
                    "pair_force_reference_relative_l2": "",
                    "position_response_x_standard_error": float(np.asarray(summary[reference_epsilon]["position"]["standard_error"])[stop - 1, 0]),
                    "pair_force_response_x_standard_error": float(np.asarray(summary[reference_epsilon]["pair_force"]["standard_error"])[stop - 1, 0]),
                    "fit_time": float(times[fit_frames - 1]),
                    "fit_frames": fit_frames,
                    "training_relative_l2_error": float(auxiliary["training_position_relative_l2_error"]),
                    "heldout_relative_l2_error": float(np.linalg.norm(held_difference) / held_norm),
                    "heldout_maximum_absolute_error": float(np.max(np.abs(held_difference))),
                    "spectral_radius": float(auxiliary["spectral_radius"]),
                    "response_scope": "ensemble_mean_position_velocity_exact_pair_force_auxiliary_response",
                    "thermodynamic_claim_allowed": 0,
                }
            )
        rate_auxiliary = fit_linear_response_auxiliary_embedding(
            np.column_stack(
                [
                    reference_position[:, 0],
                    reference_velocity[:, 0],
                    reference_pair_force[:, 0],
                    reference_pair_force_rate,
                ]
            ),
            fit_frames=fit_frames,
        )
        rate_predicted_position = np.asarray(rate_auxiliary["predicted_state_response"])[:, 0]
        for horizon in args.horizons:
            stop = int(np.searchsorted(times, horizon, side="right"))
            if stop <= fit_frames:
                continue
            held_difference = rate_predicted_position[fit_frames:stop] - reference_position[fit_frames:stop, 0]
            held_norm = float(np.linalg.norm(reference_position[fit_frames:stop, 0]))
            if held_norm <= 0.0:
                raise ValueError("force-rate auxiliary held-out position response is zero")
            summary_rows.append(
                {
                    "record": "pair_force_rate_auxiliary_markov_temporal_holdout",
                    "epsilon": reference_epsilon,
                    "member_count": member_count,
                    "horizon_time": float(times[stop - 1]),
                    "horizon_frames": stop,
                    "position_reference_relative_l2": "",
                    "pair_force_reference_relative_l2": "",
                    "position_response_x_standard_error": float(np.asarray(summary[reference_epsilon]["position"]["standard_error"])[stop - 1, 0]),
                    "pair_force_response_x_standard_error": float(np.asarray(summary[reference_epsilon]["pair_force"]["standard_error"])[stop - 1, 0]),
                    "fit_time": float(times[fit_frames - 1]),
                    "fit_frames": fit_frames,
                    "training_relative_l2_error": float(rate_auxiliary["training_position_relative_l2_error"]),
                    "heldout_relative_l2_error": float(np.linalg.norm(held_difference) / held_norm),
                    "heldout_maximum_absolute_error": float(np.max(np.abs(held_difference))),
                    "spectral_radius": float(rate_auxiliary["spectral_radius"]),
                    "response_scope": "ensemble_mean_position_velocity_pair_force_and_pair_force_rate_auxiliary_response",
                    "thermodynamic_claim_allowed": 0,
                }
            )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)

    curve_rows: list[dict[str, object]] = []
    for epsilon in args.epsilons:
        position = np.asarray(summary[epsilon]["position"]["mean"])
        pair_force = np.asarray(summary[epsilon]["pair_force"]["mean"])
        position_sem = np.asarray(summary[epsilon]["position"]["standard_error"])
        pair_sem = np.asarray(summary[epsilon]["pair_force"]["standard_error"])
        for time, position_value, pair_value, position_error, pair_error in zip(
            times, position[:, 0], pair_force[:, 0], position_sem[:, 0], pair_sem[:, 0]
        ):
            curve_rows.append(
                {
                    "epsilon": epsilon,
                    "time": float(time),
                    "position_response_x_mean": float(position_value),
                    "position_response_x_standard_error": float(position_error),
                    "pair_force_response_x_mean": float(pair_value),
                    "pair_force_response_x_standard_error": float(pair_error),
                    "member_count": member_count,
                    "thermodynamic_claim_allowed": 0,
                }
            )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)


if __name__ == "__main__":
    main()
