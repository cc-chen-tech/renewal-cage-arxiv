#!/usr/bin/env python3
"""Test a generator-constrained tagged response on matched full-KA paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_generator_response import fit_generator_constrained_response, matched_generator_response  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    keys = ("time", "position", "velocity", "force", "force_generator", "second_force_generator")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in keys}


def propagate(transition: np.ndarray, initial_state: np.ndarray, frame_count: int) -> np.ndarray:
    predicted = np.empty((frame_count, 12), dtype=float)
    predicted[0] = initial_state
    for frame in range(1, frame_count):
        predicted[frame] = transition @ predicted[frame - 1]
    return predicted


def relative_l2(predicted: np.ndarray, observed: np.ndarray) -> float:
    norm = float(np.linalg.norm(observed))
    return float(np.linalg.norm(predicted - observed) / norm) if norm > 0.0 else float("nan")


def free_transition_fit(states: np.ndarray, *, fit_frames: int) -> dict[str, np.ndarray | float]:
    source = states[:, : fit_frames - 1].reshape(-1, 12)
    target = states[:, 1:fit_frames].reshape(-1, 12)
    scale = np.sqrt(np.mean(source**2, axis=0))
    if np.any(scale <= 1e-14):
        raise ValueError("free transition has a zero-variation response coordinate")
    coefficient, _, rank, _ = np.linalg.lstsq(source / scale, target, rcond=None)
    transition = coefficient.T / scale[None, :]
    return {
        "transition_matrix": transition,
        "design_rank": float(rank),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--fit-times", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--horizons", type=float, nargs="+", default=[0.2, 1.0])
    parser.add_argument("--linearity-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise ValueError("manifest must exist")
    if not args.fit_times or not args.horizons or min(args.fit_times) <= 0.0 or min(args.horizons) <= 0.0:
        raise ValueError("fit times and horizons must be positive")
    if args.linearity_tolerance <= 0.0:
        raise ValueError("linearity tolerance must be positive")

    manifest_path = args.manifest.resolve()
    response_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("thermodynamic_claim_allowed") is not False:
        raise ValueError("manifest must preserve the thermodynamic claim boundary")
    records = manifest.get("records", [])
    if len(records) != int(manifest.get("record_count", -1)):
        raise ValueError("manifest record count is inconsistent")
    grouped: dict[tuple[int, float, int], tuple[dict[str, object], dict[str, np.ndarray]]] = {}
    for row in records:
        path = response_root / str(row["path"])
        if not path.is_file() or file_sha256(path) != row["path_sha256"]:
            raise ValueError(f"missing or modified response path: {path}")
        key = (int(row["member_index"]), float(row["epsilon"]), int(row["sign"]))
        if key in grouped:
            raise ValueError(f"duplicate response record {key}")
        grouped[key] = (row, load_npz_arrays(path))

    members = sorted({key[0] for key in grouped})
    epsilons = sorted({key[1] for key in grouped})
    if len(members) < 2 or len(epsilons) < 2:
        raise ValueError("closure analysis requires at least two members and two epsilon values")
    responses: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
    for member in members:
        for epsilon in epsilons:
            plus_row, plus = grouped[(member, epsilon, 1)]
            minus_row, minus = grouped[(member, epsilon, -1)]
            if plus_row["velocity_seed"] != minus_row["velocity_seed"] or plus_row["langevin_seed"] != minus_row["langevin_seed"]:
                raise ValueError("matched paths do not use common random numbers")
            responses[(member, epsilon)] = matched_generator_response(plus, minus, epsilon=epsilon)

    reference_epsilon = epsilons[0]
    reference_time = np.asarray(responses[(members[0], reference_epsilon)]["time"])
    if any(not np.array_equal(np.asarray(value["time"]), reference_time) for value in responses.values()):
        raise ValueError("all response paths must share one time grid")
    frame_time = float(manifest["saved_frame_interval_tau"])
    friction = float(manifest["friction"])
    if not np.allclose(np.diff(reference_time), frame_time):
        raise ValueError("manifest frame time does not match extracted responses")

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    linearity: dict[tuple[int, float], bool] = {}
    comparison_epsilon = epsilons[1]
    for member in members:
        reference_state = np.asarray(responses[(member, reference_epsilon)]["state_response"])
        comparison_state = np.asarray(responses[(member, comparison_epsilon)]["state_response"])
        for horizon in args.horizons:
            stop = int(np.searchsorted(reference_time, horizon, side="right"))
            if stop < 2 or stop > len(reference_time):
                raise ValueError(f"horizon {horizon} is outside the response grid")
            position_mismatch = relative_l2(comparison_state[:stop, :3], reference_state[:stop, :3])
            full_state_mismatch = relative_l2(comparison_state[:stop], reference_state[:stop])
            identified = bool(position_mismatch <= args.linearity_tolerance)
            linearity[(member, float(reference_time[stop - 1]))] = identified
            summary_rows.append(
                {
                    "record": "cross_epsilon_linearity",
                    "model": "none",
                    "member_index": member,
                    "fit_time": "",
                    "evaluation_epsilon": comparison_epsilon,
                    "horizon_time": float(reference_time[stop - 1]),
                    "position_relative_l2_error": position_mismatch,
                    "full_state_relative_l2_error": full_state_mismatch,
                    "linearity_tolerance": args.linearity_tolerance,
                    "linearity_identified": float(identified),
                    "fit_parameters_from_macro_observables": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )

    model_rows: list[dict[str, object]] = []
    for fit_time in args.fit_times:
        fit_frames = int(np.searchsorted(reference_time, fit_time, side="right"))
        if not (4 <= fit_frames < len(reference_time)):
            raise ValueError(f"fit time {fit_time} must leave a temporal holdout")
        for held_member in members:
            training_members = [member for member in members if member != held_member]
            training_states = np.stack(
                [np.asarray(responses[(member, reference_epsilon)]["state_response"]) for member in training_members]
            )
            training_second = np.stack(
                [np.asarray(responses[(member, reference_epsilon)]["second_force_response"]) for member in training_members]
            )
            fits: dict[str, dict[str, np.ndarray | float] | None] = {}
            failures: dict[str, str] = {}
            try:
                fits["generator_constrained"] = fit_generator_constrained_response(
                    training_states,
                    training_second,
                    frame_time=frame_time,
                    friction=friction,
                    fit_frames=fit_frames,
                )
            except ValueError as error:
                fits["generator_constrained"] = None
                failures["generator_constrained"] = str(error)
            try:
                fits["free_transition"] = free_transition_fit(training_states, fit_frames=fit_frames)
            except ValueError as error:
                fits["free_transition"] = None
                failures["free_transition"] = str(error)

            for model, fit in fits.items():
                transition = None if fit is None else np.asarray(fit["transition_matrix"])
                spectral_radius = float("nan") if fit is None else float(fit["spectral_radius"])
                design_rank = float("nan") if fit is None else float(fit["design_rank"])
                stable = bool(fit is not None and spectral_radius <= 1.0 + 1e-6)
                for epsilon in epsilons:
                    observed = np.asarray(responses[(held_member, epsilon)]["state_response"])
                    predicted = None if transition is None else propagate(transition, observed[0], len(observed))
                    for horizon in args.horizons:
                        stop = int(np.searchsorted(reference_time, horizon, side="right"))
                        actual_horizon = float(reference_time[stop - 1])
                        identified = linearity[(held_member, actual_horizon)]
                        error = float("nan") if predicted is None else relative_l2(
                            predicted[:stop, :3], observed[:stop, :3]
                        )
                        threshold = 0.20 if actual_horizon <= 0.2 + 1e-12 else 0.30
                        gate_evaluable = bool(identified and fit is not None)
                        gate_pass = bool(gate_evaluable and stable and error <= threshold)
                        row = {
                            "record": "leave_one_member_out",
                            "model": model,
                            "member_index": held_member,
                            "fit_time": float(reference_time[fit_frames - 1]),
                            "evaluation_epsilon": epsilon,
                            "horizon_time": actual_horizon,
                            "position_relative_l2_error": error,
                            "position_error_threshold": threshold,
                            "linearity_tolerance": args.linearity_tolerance,
                            "linearity_identified": float(identified),
                            "model_identified": float(fit is not None),
                            "gate_evaluable": float(gate_evaluable),
                            "transition_stable": float(stable),
                            "gate_pass": float(gate_pass),
                            "spectral_radius": spectral_radius,
                            "design_rank": design_rank,
                            "fit_failure": failures.get(model, ""),
                            "training_second_generator_relative_l2_error": (
                                "" if fit is None else fit.get("training_second_generator_relative_l2_error", "")
                            ),
                            "maximum_residual_state_correlation": (
                                ""
                                if fit is None or "residual_state_correlation" not in fit
                                else float(np.nanmax(np.abs(np.asarray(fit["residual_state_correlation"]))))
                            ),
                            "fit_parameters_from_macro_observables": 0.0,
                            "thermodynamic_claim_allowed": 0.0,
                        }
                        summary_rows.append(row)
                        model_rows.append(row)
                    if predicted is not None:
                        for time, observed_value, predicted_value in zip(reference_time, observed[:, :3], predicted[:, :3]):
                            curve_rows.append(
                                {
                                    "model": model,
                                    "member_index": held_member,
                                    "fit_time": float(reference_time[fit_frames - 1]),
                                    "evaluation_epsilon": epsilon,
                                    "time": float(time),
                                    **{f"observed_position_{axis}": float(observed_value[index]) for index, axis in enumerate("xyz")},
                                    **{f"predicted_position_{axis}": float(predicted_value[index]) for index, axis in enumerate("xyz")},
                                    "thermodynamic_claim_allowed": 0.0,
                                }
                            )

    group_keys = sorted(
        {
            (str(row["model"]), float(row["fit_time"]), float(row["evaluation_epsilon"]), float(row["horizon_time"]))
            for row in model_rows
        }
    )
    for model, fit_time, epsilon, horizon in group_keys:
        rows = [
            row
            for row in model_rows
            if (row["model"], row["fit_time"], row["evaluation_epsilon"], row["horizon_time"])
            == (model, fit_time, epsilon, horizon)
        ]
        evaluable = [row for row in rows if float(row["gate_evaluable"]) == 1.0]
        errors = np.asarray([float(row["position_relative_l2_error"]) for row in evaluable])
        summary_rows.append(
            {
                "record": "aggregate_gate",
                "model": model,
                "member_index": "",
                "fit_time": fit_time,
                "evaluation_epsilon": epsilon,
                "horizon_time": horizon,
                "position_relative_l2_error": float(np.mean(errors)) if len(errors) else float("nan"),
                "position_relative_l2_error_standard_error": (
                    float(np.std(errors, ddof=1) / math.sqrt(len(errors))) if len(errors) > 1 else 0.0
                ),
                "maximum_position_relative_l2_error": float(np.max(errors)) if len(errors) else float("nan"),
                "linearity_identified_fold_count": sum(float(row["linearity_identified"]) for row in rows),
                "evaluable_fold_count": len(evaluable),
                "all_identified_folds_pass": float(
                    len(evaluable) == len(rows) and all(float(row["gate_pass"]) == 1.0 for row in rows)
                ),
                "fit_parameters_from_macro_observables": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )

    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)


if __name__ == "__main__":
    main()
