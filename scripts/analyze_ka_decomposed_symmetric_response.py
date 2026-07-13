#!/usr/bin/env python3
"""Infer a scalar conservative-bath memory kernel from full KA responses."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import (  # noqa: E402
    infer_causal_position_memory_kernel,
    propagate_causal_position_memory_response,
    symmetric_finite_difference_response,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pair(directory: Path, epsilon: float) -> dict[str, np.ndarray]:
    positive = np.loadtxt(directory / f"plus_eps{epsilon:.3f}.dat", comments="#")
    negative = np.loadtxt(directory / f"minus_eps{epsilon:.3f}.dat", comments="#")
    if positive.ndim != 2 or positive.shape != negative.shape or positive.shape[1] != 13:
        raise ValueError("each matched pair must contain step, position, velocity, total-force, and pair-force columns")
    if not np.array_equal(positive[:, 0], negative[:, 0]):
        raise ValueError("matched plus/minus outputs must have identical timesteps")
    response = lambda columns: symmetric_finite_difference_response(positive[:, columns], negative[:, columns], displacement=epsilon)["response"]
    return {
        "steps": positive[:, 0],
        "position": response(slice(1, 4)),
        "velocity": response(slice(4, 7)),
        "total_force": response(slice(7, 10)),
        "pair_force": response(slice(10, 13)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.001, 0.002, 0.004])
    parser.add_argument("--time-step", type=float, default=0.001)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=1.0)
    args = parser.parse_args()
    if not args.epsilons or min(args.epsilons) <= 0.0 or args.time_step <= 0.0 or args.mass <= 0.0 or args.friction < 0.0:
        raise ValueError("epsilons, time-step, mass, and friction must be valid")

    responses = {epsilon: load_pair(args.input_directory, epsilon) for epsilon in args.epsilons}
    reference_epsilon = min(args.epsilons)
    reference = responses[reference_epsilon]
    inferred = infer_causal_position_memory_kernel(reference["position"][:, 0], reference["pair_force"][:, 0], frame_time=args.time_step)
    prediction = propagate_causal_position_memory_response(
        instantaneous_curvature=float(inferred["instantaneous_curvature"]),
        memory_kernel=np.asarray(inferred["memory_kernel"]),
        frame_time=args.time_step,
        mass=args.mass,
        friction=args.friction,
    )
    reference_position_norm = float(np.linalg.norm(reference["position"][:, 0]))
    reference_pair_norm = float(np.linalg.norm(reference["pair_force"][:, 0]))
    if reference_position_norm <= 0.0 or reference_pair_norm <= 0.0:
        raise ValueError("reference responses must be nonzero")

    summary_rows: list[dict[str, object]] = []
    for epsilon, current in responses.items():
        if not np.array_equal(current["steps"], reference["steps"]):
            raise ValueError("all epsilon pairs must share the same timestep grid")
        centered_velocity = np.vstack([current["velocity"][0], 0.5 * (current["velocity"][1:] + current["velocity"][:-1])])
        thermostat_residual = current["total_force"] - current["pair_force"] + args.friction * centered_velocity
        observed = current["position"][:, 0]
        summary_rows.append(
            {
                "epsilon": epsilon,
                "time_points": len(current["steps"]),
                "position_response_reference_relative_l2": float(np.linalg.norm(observed - reference["position"][:, 0]) / reference_position_norm),
                "pair_force_response_reference_relative_l2": float(np.linalg.norm(current["pair_force"][:, 0] - reference["pair_force"][:, 0]) / reference_pair_norm),
                "thermostat_centered_balance_relative_l2": float(np.linalg.norm(thermostat_residual) / np.linalg.norm(current["total_force"])),
                "scalar_position_prediction_relative_l2": float(np.linalg.norm(prediction["position_response"] - observed) / np.linalg.norm(observed)),
                "scalar_position_prediction_max_absolute_error": float(np.max(np.abs(prediction["position_response"] - observed))),
                "instantaneous_pair_curvature_x": float(inferred["instantaneous_curvature"]),
                "response_scope": "full_KA_Langevin_pair_force_plus_explicit_thermostat_friction",
                "thermodynamic_claim_allowed": 0,
            }
        )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)

    curve_rows = [
        {
            "time": float(step * args.time_step),
            "position_response_x": float(position),
            "velocity_response_x": float(velocity),
            "total_force_response_x": float(total_force),
            "pair_force_response_x": float(pair_force),
            "inferred_memory_kernel_x": float(kernel),
            "predicted_position_response_x": float(predicted_position),
            "reference_epsilon": reference_epsilon,
            "thermodynamic_claim_allowed": 0,
        }
        for step, position, velocity, total_force, pair_force, kernel, predicted_position in zip(
            reference["steps"],
            reference["position"][:, 0],
            reference["velocity"][:, 0],
            reference["total_force"][:, 0],
            reference["pair_force"][:, 0],
            np.asarray(inferred["memory_kernel"]),
            prediction["position_response"],
        )
    ]
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)


if __name__ == "__main__":
    main()
