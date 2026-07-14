#!/usr/bin/env python3
"""Test a second-generator Krylov closure on matched C3 KA responses."""

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

from ka_generator_krylov import (  # noqa: E402
    assemble_second_generator_state,
    fit_free_second_generator_transition,
    fit_second_generator_constrained_response,
    propagate_linear_response,
    second_generator_residual_diagnostic,
)
from ka_generator_response import (  # noqa: E402
    fit_generator_constrained_response,
    matched_generator_response,
)


FORBIDDEN_CLAIMS = {
    "autonomous_stochastic_single_particle_gle_allowed": 0.0,
    "event_clock_claim_allowed": 0.0,
    "kramers_escape_claim_allowed": 0.0,
    "thermodynamic_claim_allowed": 0.0,
}


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


def load_response_path(path: Path) -> dict[str, np.ndarray]:
    keys = (
        "time",
        "position",
        "velocity",
        "force",
        "force_generator",
        "second_force_generator",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        protocol = str(np.asarray(payload["potential_protocol"]).item())
        claim = float(np.asarray(payload["thermodynamic_claim_allowed"]).item())
        if protocol != "ka_lj_c3_switch" or claim != 0.0:
            raise ValueError(f"{path}: path must preserve the C3 and thermodynamic boundaries")
        return {key: np.asarray(payload[key]) for key in keys}


def relative_l2(predicted: np.ndarray, observed: np.ndarray) -> float:
    norm = float(np.linalg.norm(observed))
    return float(np.linalg.norm(predicted - observed) / norm) if norm > 0.0 else float("nan")


def response_threshold(horizon: float) -> float:
    if horizon <= 0.2 + 1e-12:
        return 0.20
    if horizon <= 0.5 + 1e-12:
        return 0.35
    return 0.50


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--fit-times", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--horizons", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    parser.add_argument("--linearity-tolerance", type=float, default=0.02)
    parser.add_argument("--expected-member-count", type=int, default=8)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    if not args.manifest.is_file():
        raise ValueError("manifest must exist")
    if args.expected_member_count < 2:
        raise ValueError("expected-member-count must be at least two")
    if not args.fit_times or min(args.fit_times) <= 0.0:
        raise ValueError("fit-times must be positive")
    if not args.horizons or min(args.horizons) <= 0.0:
        raise ValueError("horizons must be positive")
    if args.linearity_tolerance <= 0.0:
        raise ValueError("linearity-tolerance must be positive")

    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("potential_protocol") != "ka_lj_c3_switch":
        raise ValueError("manifest must use potential_protocol=ka_lj_c3_switch")
    if manifest.get("thermodynamic_claim_allowed") is not False:
        raise ValueError("manifest must preserve the thermodynamic claim boundary")
    if manifest.get("fit_parameters_from_macro_observables") is not False:
        raise ValueError("manifest must not fit macroscopic observables")
    if int(manifest.get("member_count", -1)) != args.expected_member_count:
        raise ValueError("manifest member count does not match expected-member-count")
    epsilons = sorted(float(value) for value in manifest.get("epsilons", []))
    if epsilons != [0.001, 0.002]:
        raise ValueError("manifest must contain epsilon=0.001 and 0.002")
    frame_time = float(manifest.get("saved_frame_interval_tau", float("nan")))
    duration = float(manifest.get("duration_tau", float("nan")))
    friction = float(manifest.get("friction", float("nan")))
    if not math.isclose(frame_time, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("manifest saved interval must be 0.005 tau")
    if not math.isclose(duration, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("manifest duration must be 1.0 tau")
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("manifest friction must be finite and nonnegative")
    records = manifest.get("records", [])
    if len(records) != int(manifest.get("record_count", -1)):
        raise ValueError("manifest record count is inconsistent")

    grouped: dict[tuple[int, float, int], tuple[dict[str, object], dict[str, np.ndarray]]] = {}
    for record in records:
        path = root / str(record["path"])
        if not path.is_file() or file_sha256(path) != record["path_sha256"]:
            raise ValueError(f"missing or modified response path: {path}")
        if record.get("potential_protocol") != "ka_lj_c3_switch":
            raise ValueError("every manifest record must use the C3 potential")
        if record.get("fit_parameters_from_macro_observables") is not False:
            raise ValueError("response records must not fit macroscopic observables")
        if record.get("thermodynamic_claim_allowed") is not False:
            raise ValueError("response records must preserve the thermodynamic boundary")
        key = (int(record["member_index"]), float(record["epsilon"]), int(record["sign"]))
        if key in grouped:
            raise ValueError(f"duplicate response record {key}")
        grouped[key] = (record, load_response_path(path))

    members = sorted({key[0] for key in grouped})
    if members != list(range(1, args.expected_member_count + 1)):
        raise ValueError("manifest member indices must be consecutive from one")
    responses: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
    second_states: dict[tuple[int, float], np.ndarray] = {}
    for member in members:
        for epsilon in epsilons:
            plus_record, plus = grouped[(member, epsilon, 1)]
            minus_record, minus = grouped[(member, epsilon, -1)]
            if (
                plus_record["velocity_seed"] != minus_record["velocity_seed"]
                or plus_record["langevin_seed"] != minus_record["langevin_seed"]
            ):
                raise ValueError("matched paths must use common random seeds")
            response = matched_generator_response(plus, minus, epsilon=epsilon)
            responses[(member, epsilon)] = response
            second_states[(member, epsilon)] = assemble_second_generator_state(
                np.asarray(response["state_response"]),
                np.asarray(response["second_force_response"]),
            )

    reference_time = np.asarray(responses[(members[0], epsilons[0])]["time"])
    expected_frames = int(round(duration / frame_time)) + 1
    if len(reference_time) != expected_frames or not np.allclose(
        np.diff(reference_time), frame_time, rtol=0.0, atol=1e-12
    ):
        raise ValueError("response time grid does not match the manifest")
    if any(
        not np.array_equal(np.asarray(response["time"]), reference_time)
        for response in responses.values()
    ):
        raise ValueError("all response paths must share one time grid")
    if max(args.horizons) > reference_time[-1] + 1e-12:
        raise ValueError("requested horizon exceeds the response grid")

    linearity: dict[tuple[int, float], tuple[bool, float]] = {}
    for member in members:
        reference = np.asarray(responses[(member, epsilons[0])]["state_response"])
        comparison = np.asarray(responses[(member, epsilons[1])]["state_response"])
        for horizon in args.horizons:
            stop = int(np.searchsorted(reference_time, horizon, side="right"))
            actual_horizon = float(reference_time[stop - 1])
            mismatch = relative_l2(comparison[:stop, :3], reference[:stop, :3])
            linearity[(member, actual_horizon)] = (
                bool(mismatch <= args.linearity_tolerance),
                mismatch,
            )

    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for fit_time in args.fit_times:
        fit_frames = int(np.searchsorted(reference_time, fit_time, side="right"))
        if not (4 <= fit_frames < len(reference_time)):
            raise ValueError(f"fit time {fit_time} must leave a temporal holdout")
        actual_fit_time = float(reference_time[fit_frames - 1])
        for held_member in members:
            training_members = [member for member in members if member != held_member]
            training_first = np.stack(
                [
                    np.asarray(responses[(member, epsilons[0])]["state_response"])
                    for member in training_members
                ]
            )
            training_second_force = np.stack(
                [
                    np.asarray(responses[(member, epsilons[0])]["second_force_response"])
                    for member in training_members
                ]
            )
            training_second = np.stack(
                [second_states[(member, epsilons[0])] for member in training_members]
            )
            fits: dict[str, dict[str, np.ndarray | float] | None] = {}
            failures: dict[str, str] = {}
            fit_calls = {
                "first_generator_constrained": lambda: fit_generator_constrained_response(
                    training_first,
                    training_second_force,
                    frame_time=frame_time,
                    friction=friction,
                    fit_frames=fit_frames,
                ),
                "second_generator_constrained": lambda: fit_second_generator_constrained_response(
                    training_second,
                    frame_time=frame_time,
                    friction=friction,
                    fit_frames=fit_frames,
                ),
                "free_second_generator_transition": lambda: fit_free_second_generator_transition(
                    training_second,
                    fit_frames=fit_frames,
                ),
            }
            for model, fit_call in fit_calls.items():
                try:
                    fits[model] = fit_call()
                except (ValueError, np.linalg.LinAlgError) as error:
                    fits[model] = None
                    failures[model] = str(error)

            for epsilon in epsilons:
                observed_first = np.asarray(responses[(held_member, epsilon)]["state_response"])
                observed_second = second_states[(held_member, epsilon)]
                predictions: dict[str, np.ndarray | None] = {}
                errors: dict[tuple[str, float], float] = {}
                residual_correlations: dict[tuple[str, float], float] = {}
                for model, fit in fits.items():
                    observed = observed_first if model == "first_generator_constrained" else observed_second
                    predictions[model] = (
                        None
                        if fit is None
                        else propagate_linear_response(
                            np.asarray(fit["transition_matrix"]), observed[0], len(observed)
                        )
                    )
                for horizon in args.horizons:
                    stop = int(np.searchsorted(reference_time, horizon, side="right"))
                    actual_horizon = float(reference_time[stop - 1])
                    for model, predicted in predictions.items():
                        observed = (
                            observed_first
                            if model == "first_generator_constrained"
                            else observed_second
                        )
                        errors[(model, actual_horizon)] = (
                            float("nan")
                            if predicted is None
                            else relative_l2(predicted[:stop, :3], observed[:stop, :3])
                        )
                        if model == "second_generator_constrained" and fits[model] is not None:
                            diagnostic = second_generator_residual_diagnostic(
                                observed_second[:stop],
                                np.asarray(fits[model]["fitted_third_generator_block"]),
                                frame_time=frame_time,
                            )
                            residual_correlations[(model, actual_horizon)] = float(
                                diagnostic["maximum_abs_residual_state_correlation"]
                            )

                for model, fit in fits.items():
                    predicted = predictions[model]
                    spectral_radius = float("nan") if fit is None else float(fit["spectral_radius"])
                    stable = bool(fit is not None and spectral_radius <= 1.0 + 1e-6)
                    for horizon in args.horizons:
                        stop = int(np.searchsorted(reference_time, horizon, side="right"))
                        actual_horizon = float(reference_time[stop - 1])
                        identified, mismatch = linearity[(held_member, actual_horizon)]
                        error = errors[(model, actual_horizon)]
                        baseline_error = errors[("first_generator_constrained", actual_horizon)]
                        improvement = (
                            (baseline_error - error) / baseline_error
                            if model == "second_generator_constrained"
                            and math.isfinite(error)
                            and math.isfinite(baseline_error)
                            and baseline_error > 0.0
                            else float("nan")
                        )
                        residual_correlation = residual_correlations.get(
                            (model, actual_horizon), float("nan")
                        )
                        threshold = response_threshold(actual_horizon)
                        gate_pass = bool(
                            model == "second_generator_constrained"
                            and identified
                            and stable
                            and math.isfinite(error)
                            and error <= threshold
                            and math.isfinite(residual_correlation)
                            and residual_correlation <= 0.20
                        )
                        row = {
                            "record": "leave_one_member_out",
                            "model": model,
                            "member_index": held_member,
                            "fit_time": actual_fit_time,
                            "evaluation_epsilon": epsilon,
                            "horizon_time": actual_horizon,
                            "position_relative_l2_error": error,
                            "paired_improvement_fraction": improvement,
                            "position_error_threshold": threshold,
                            "linearity_mismatch": mismatch,
                            "linearity_identified": float(identified),
                            "model_identified": float(fit is not None),
                            "transition_stable": float(stable),
                            "spectral_radius": spectral_radius,
                            "design_rank": "" if fit is None else fit["design_rank"],
                            "maximum_held_residual_state_correlation": residual_correlation,
                            "gate_pass": float(gate_pass),
                            "fit_failure": failures.get(model, ""),
                            "fit_parameters_from_macro_observables": 0.0,
                            "second_generator_response_allowed": 0.0,
                            "one_tau_generator_response_allowed": 0.0,
                            **FORBIDDEN_CLAIMS,
                        }
                        summary_rows.append(row)
                        fold_rows.append(row)
                    if predicted is not None:
                        for time, observed_position, predicted_position in zip(
                            reference_time, observed_first[:, :3], predicted[:, :3]
                        ):
                            curve_rows.append(
                                {
                                    "model": model,
                                    "member_index": held_member,
                                    "fit_time": actual_fit_time,
                                    "evaluation_epsilon": epsilon,
                                    "time": float(time),
                                    **{
                                        f"observed_position_{axis}": float(observed_position[index])
                                        for index, axis in enumerate("xyz")
                                    },
                                    **{
                                        f"predicted_position_{axis}": float(predicted_position[index])
                                        for index, axis in enumerate("xyz")
                                    },
                                    **FORBIDDEN_CLAIMS,
                                }
                            )

    aggregate_rows: list[dict[str, object]] = []
    group_keys = sorted(
        {
            (
                str(row["model"]),
                float(row["fit_time"]),
                float(row["evaluation_epsilon"]),
                float(row["horizon_time"]),
            )
            for row in fold_rows
        }
    )
    for model, fit_time, epsilon, horizon in group_keys:
        rows = [
            row
            for row in fold_rows
            if (
                row["model"],
                row["fit_time"],
                row["evaluation_epsilon"],
                row["horizon_time"],
            )
            == (model, fit_time, epsilon, horizon)
        ]
        identified = [row for row in rows if float(row["linearity_identified"]) == 1.0]
        evaluable = [row for row in identified if float(row["model_identified"]) == 1.0]
        errors = np.asarray([float(row["position_relative_l2_error"]) for row in evaluable])
        improvements = np.asarray(
            [
                float(row["paired_improvement_fraction"])
                for row in evaluable
                if math.isfinite(float(row["paired_improvement_fraction"]))
            ]
        )
        aggregate = {
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
            "paired_improvement_fraction": (
                float(np.mean(improvements)) if len(improvements) else float("nan")
            ),
            "identified_fold_count": len(identified),
            "evaluable_fold_count": len(evaluable),
            "all_identified_folds_pass": float(
                len(evaluable) == len(identified)
                and len(evaluable) > 0
                and all(float(row["gate_pass"]) == 1.0 for row in evaluable)
            ),
            "fit_parameters_from_macro_observables": 0.0,
            "second_generator_response_allowed": 0.0,
            "one_tau_generator_response_allowed": 0.0,
            **FORBIDDEN_CLAIMS,
        }
        aggregate_rows.append(aggregate)
        summary_rows.append(aggregate)

    primary_fit = max(float(value) for value in args.fit_times)
    minimum_long_folds = min(6, args.expected_member_count)

    def selected_aggregate(epsilon: float, horizon: float) -> dict[str, object] | None:
        candidates = [
            row
            for row in aggregate_rows
            if row["model"] == "second_generator_constrained"
            and math.isclose(float(row["fit_time"]), primary_fit, abs_tol=1e-12)
            and math.isclose(float(row["evaluation_epsilon"]), epsilon, abs_tol=1e-12)
            and math.isclose(float(row["horizon_time"]), horizon, abs_tol=1e-12)
        ]
        return candidates[0] if candidates else None

    short_rows = [selected_aggregate(epsilon, 0.2) for epsilon in epsilons]
    short_allowed = bool(
        all(row is not None for row in short_rows)
        and all(int(row["identified_fold_count"]) == args.expected_member_count for row in short_rows if row)
        and all(float(row["all_identified_folds_pass"]) == 1.0 for row in short_rows if row)
        and all(float(row["paired_improvement_fraction"]) >= 0.20 for row in short_rows if row)
    )
    long_rows = [selected_aggregate(epsilon, 1.0) for epsilon in epsilons]
    long_allowed = bool(
        short_allowed
        and all(row is not None for row in long_rows)
        and all(int(row["identified_fold_count"]) >= minimum_long_folds for row in long_rows if row)
        and all(float(row["all_identified_folds_pass"]) == 1.0 for row in long_rows if row)
        and all(float(row["paired_improvement_fraction"]) >= 0.20 for row in long_rows if row)
    )
    summary_rows.append(
        {
            "record": "verdict",
            "model": "second_generator_constrained",
            "member_index": "",
            "fit_time": primary_fit,
            "evaluation_epsilon": "",
            "horizon_time": "",
            "integrity_gate_pass": 1.0,
            "primary_fit_time": primary_fit,
            "minimum_one_tau_identified_folds": minimum_long_folds,
            "second_generator_response_allowed": float(short_allowed),
            "one_tau_generator_response_allowed": float(long_allowed),
            "fit_parameters_from_macro_observables": 0.0,
            **FORBIDDEN_CLAIMS,
        }
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
        summary_rows,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"),
        curve_rows,
    )


if __name__ == "__main__":
    main()
