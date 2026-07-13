#!/usr/bin/env python3
"""Resolve numerical versus physical color in KA tangent-force residuals."""

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

from ka_generator_response import generator_response_tangent_diagnostic, matched_generator_response  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_path(path: Path) -> dict[str, np.ndarray]:
    keys = ("time", "position", "velocity", "force", "force_generator", "second_force_generator")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in keys}


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    norm = float(np.linalg.norm(right))
    return float(np.linalg.norm(left - right) / norm) if norm > 0.0 else float("nan")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).reshape(-1)
    right_flat = np.asarray(right, dtype=float).reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise ValueError("correlation arrays must have matching shapes")
    if np.std(left_flat) == 0.0 or np.std(right_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty resolution table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--maximum-time", type=float)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise ValueError("manifest must exist")
    if not args.strides or any(isinstance(stride, bool) or stride < 1 for stride in args.strides):
        raise ValueError("strides must be positive integers")
    if len(set(args.strides)) != len(args.strides):
        raise ValueError("strides must be unique")
    if args.maximum_time is not None and (not math.isfinite(args.maximum_time) or args.maximum_time <= 0.0):
        raise ValueError("maximum time must be finite and positive")

    manifest_path = args.manifest.resolve()
    response_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("thermodynamic_claim_allowed") is not False:
        raise ValueError("manifest must preserve the thermodynamic claim boundary")
    base_frame_time = float(manifest["saved_frame_interval_tau"])
    friction = float(manifest["friction"])
    grouped: dict[tuple[int, float, int], dict[str, np.ndarray]] = {}
    precision_evidence: list[bool] = []
    for row in manifest["records"]:
        path = response_root / str(row["path"])
        if not path.is_file() or file_sha256(path) != row["path_sha256"]:
            raise ValueError(f"missing or modified response path: {path}")
        key = (int(row["member_index"]), float(row["epsilon"]), int(row["sign"]))
        if key in grouped:
            raise ValueError(f"duplicate response path {key}")
        grouped[key] = load_path(path)
        input_path = path.parent / "in.response"
        precision_evidence.append(
            input_path.is_file() and "dump_modify trajectory sort id format float %.17g" in input_path.read_text()
        )

    members = sorted({key[0] for key in grouped})
    epsilons = sorted({key[1] for key in grouped})
    if len(members) < 2 or len(epsilons) < 2:
        raise ValueError("resolution scan requires at least two members and two epsilons")
    responses: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
    for member in members:
        for epsilon in epsilons:
            responses[(member, epsilon)] = matched_generator_response(
                grouped[(member, epsilon, 1)],
                grouped[(member, epsilon, -1)],
                epsilon=epsilon,
            )

    reference_time = np.asarray(responses[(members[0], epsilons[0])]["time"], dtype=float)
    if args.maximum_time is not None:
        stop = int(np.searchsorted(reference_time, args.maximum_time, side="right"))
        if stop < 3:
            raise ValueError("maximum time must retain at least three frames")
    else:
        stop = len(reference_time)
    precision_verified = float(bool(precision_evidence) and all(precision_evidence))
    rows: list[dict[str, object]] = []
    symmetric_by_key: dict[tuple[int, float, int], np.ndarray] = {}
    metric_names = (
        "tangent_innovation_rms",
        "tangent_innovation_lag1_correlation",
        "tangent_innovation_squared_norm_lag1_correlation",
        "symmetric_tangent_residual_rms",
        "symmetric_tangent_residual_lag1_correlation",
        "symmetric_tangent_residual_squared_norm_lag1_correlation",
        "symmetric_tangent_residual_max_abs_component_excess_kurtosis",
        "symmetric_tangent_residual_normalized_mean",
    )
    for stride in sorted(args.strides):
        if (stop - 1) % stride:
            raise ValueError(f"stride {stride} must divide the retained interval count")
        frame_time = base_frame_time * stride
        for epsilon in epsilons:
            states = np.stack(
                [np.asarray(responses[(member, epsilon)]["state_response"])[:stop:stride] for member in members]
            )
            second = np.stack(
                [np.asarray(responses[(member, epsilon)]["second_force_response"])[:stop:stride] for member in members]
            )
            aggregate = generator_response_tangent_diagnostic(
                states,
                second,
                frame_time=frame_time,
                friction=friction,
            )
            rows.append(
                {
                    "record": "resolution_aggregate",
                    "member_index": "",
                    "epsilon": epsilon,
                    "stride": stride,
                    "frame_time": frame_time,
                    "maximum_time": float(reference_time[stop - 1]),
                    "position_velocity_relative_l2_error": aggregate["position_velocity_relative_l2_error"],
                    "velocity_force_relative_l2_error": aggregate["velocity_force_relative_l2_error"],
                    "force_generator_relative_l2_error": aggregate["force_generator_relative_l2_error"],
                    "generator_second_relative_l2_error": aggregate["generator_second_relative_l2_error"],
                    **{name: aggregate[name] for name in metric_names},
                    "dump_float_format_verified": precision_verified,
                    "fit_parameters_from_macro_observables": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
            for member_index, member in enumerate(members):
                diagnostic = generator_response_tangent_diagnostic(
                    states[member_index],
                    second[member_index],
                    frame_time=frame_time,
                    friction=friction,
                )
                symmetric_by_key[(member, epsilon, stride)] = np.asarray(
                    diagnostic["symmetric_tangent_residual"]
                )[0]
                rows.append(
                    {
                        "record": "resolution_member",
                        "member_index": member,
                        "epsilon": epsilon,
                        "stride": stride,
                        "frame_time": frame_time,
                        "maximum_time": float(reference_time[stop - 1]),
                        **{name: diagnostic[name] for name in metric_names},
                        "dump_float_format_verified": precision_verified,
                        "fit_parameters_from_macro_observables": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )

        reference_epsilon, comparison_epsilon = epsilons[:2]
        cross_rows: list[dict[str, object]] = []
        for member in members:
            reference = symmetric_by_key[(member, reference_epsilon, stride)]
            comparison = symmetric_by_key[(member, comparison_epsilon, stride)]
            row = {
                "record": "resolution_cross_epsilon",
                "member_index": member,
                "epsilon": comparison_epsilon,
                "reference_epsilon": reference_epsilon,
                "stride": stride,
                "frame_time": frame_time,
                "maximum_time": float(reference_time[stop - 1]),
                "symmetric_residual_cross_epsilon_relative_l2_error": relative_l2(comparison, reference),
                "symmetric_residual_cross_epsilon_correlation": correlation(comparison, reference),
                "dump_float_format_verified": precision_verified,
                "fit_parameters_from_macro_observables": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
            rows.append(row)
            cross_rows.append(row)
        rows.append(
            {
                "record": "resolution_cross_epsilon_aggregate",
                "member_index": "",
                "epsilon": comparison_epsilon,
                "reference_epsilon": reference_epsilon,
                "stride": stride,
                "frame_time": frame_time,
                "maximum_time": float(reference_time[stop - 1]),
                "symmetric_residual_cross_epsilon_relative_l2_error": float(
                    np.mean([row["symmetric_residual_cross_epsilon_relative_l2_error"] for row in cross_rows])
                ),
                "maximum_symmetric_residual_cross_epsilon_relative_l2_error": float(
                    np.max([row["symmetric_residual_cross_epsilon_relative_l2_error"] for row in cross_rows])
                ),
                "symmetric_residual_cross_epsilon_correlation": float(
                    np.mean([row["symmetric_residual_cross_epsilon_correlation"] for row in cross_rows])
                ),
                "minimum_symmetric_residual_cross_epsilon_correlation": float(
                    np.min([row["symmetric_residual_cross_epsilon_correlation"] for row in cross_rows])
                ),
                "dump_float_format_verified": precision_verified,
                "fit_parameters_from_macro_observables": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )

    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
