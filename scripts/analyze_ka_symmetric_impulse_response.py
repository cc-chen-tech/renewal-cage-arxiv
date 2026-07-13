#!/usr/bin/env python3
"""Analyze matched plus/minus full-KA Langevin perturbation trajectories."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import symmetric_finite_difference_response  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pair(directory: Path, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.loadtxt(directory / f"plus_eps{epsilon:.3f}.dat", comments="#")
    negative = np.loadtxt(directory / f"minus_eps{epsilon:.3f}.dat", comments="#")
    if positive.ndim != 2 or positive.shape != negative.shape or positive.shape[1] != 7:
        raise ValueError("each matched pair must contain aligned step, position, and force columns")
    if not np.array_equal(positive[:, 0], negative[:, 0]):
        raise ValueError("matched plus/minus outputs must have identical timesteps")
    force = symmetric_finite_difference_response(positive[:, 4:7], negative[:, 4:7], displacement=epsilon)["response"]
    position = symmetric_finite_difference_response(positive[:, 1:4], negative[:, 1:4], displacement=epsilon)["response"]
    return positive[:, 0], force, position


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.001, 0.002, 0.004])
    parser.add_argument("--time-step", type=float, default=0.001)
    args = parser.parse_args()
    if not args.epsilons or min(args.epsilons) <= 0.0 or args.time_step <= 0.0:
        raise ValueError("epsilons and time-step must be positive")

    responses = {epsilon: load_pair(args.input_directory, epsilon) for epsilon in args.epsilons}
    reference_epsilon = min(args.epsilons)
    reference_steps, reference_force, reference_position = responses[reference_epsilon]
    force_norm = float(np.linalg.norm(reference_force))
    position_norm = float(np.linalg.norm(reference_position))
    if force_norm <= 0.0 or position_norm <= 0.0:
        raise ValueError("reference response must be nonzero")

    summary_rows: list[dict[str, object]] = []
    for epsilon, (steps, force, position) in responses.items():
        if not np.array_equal(steps, reference_steps):
            raise ValueError("all epsilon pairs must share the same timestep grid")
        summary_rows.append(
            {
                "epsilon": epsilon,
                "time_points": len(steps),
                "force_response_reference_relative_l2": float(np.linalg.norm(force - reference_force) / force_norm),
                "position_response_reference_relative_l2": float(np.linalg.norm(position - reference_position) / position_norm),
                "initial_force_response_x": float(force[0, 0]),
                "final_force_response_norm": float(np.linalg.norm(force[-1])),
                "initial_position_response_x": float(position[0, 0]),
                "initial_position_response_transverse_norm": float(np.linalg.norm(position[0, 1:])),
                "response_scope": "full_KA_Langevin_matched_common_random_number_linear_response",
                "thermodynamic_claim_allowed": 0,
            }
        )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)

    curve_rows = [
        {
            "time": float(step * args.time_step),
            "force_response_x": float(force[0]),
            "force_response_y": float(force[1]),
            "force_response_z": float(force[2]),
            "position_response_x": float(position[0]),
            "position_response_y": float(position[1]),
            "position_response_z": float(position[2]),
            "reference_epsilon": reference_epsilon,
            "thermodynamic_claim_allowed": 0,
        }
        for step, force, position in zip(reference_steps, reference_force, reference_position)
    ]
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)


if __name__ == "__main__":
    main()
