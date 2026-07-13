#!/usr/bin/env python3
"""Diagnose non-differentiable target-pair crossings of the KA LJ cutoff."""

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

from ka_generator_response import (  # noqa: E402
    generator_response_tangent_diagnostic,
    matched_generator_response,
    right_censored_tangent_interval_mask,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_path(path: Path) -> dict[str, np.ndarray]:
    keys = (
        "time",
        "position",
        "velocity",
        "force",
        "force_generator",
        "second_force_generator",
        "target_pair_active",
        "nearest_cutoff_signed_gap",
        "nearest_cutoff_particle_index",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f"{path}: missing cutoff arrays {missing}")
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
        raise ValueError("cannot write an empty cutoff table")
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
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--maximum-time", type=float)
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise ValueError("manifest must exist")
    if isinstance(args.stride, bool) or args.stride < 1:
        raise ValueError("stride must be a positive integer")
    if args.maximum_time is not None and (not math.isfinite(args.maximum_time) or args.maximum_time <= 0.0):
        raise ValueError("maximum time must be finite and positive")

    manifest_path = args.manifest.resolve()
    response_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("thermodynamic_claim_allowed") is not False:
        raise ValueError("manifest must preserve the thermodynamic claim boundary")
    grouped: dict[tuple[int, float, int], tuple[dict[str, object], dict[str, np.ndarray]]] = {}
    for row in manifest["records"]:
        path = response_root / str(row["path"])
        if not path.is_file() or file_sha256(path) != row["path_sha256"]:
            raise ValueError(f"missing or modified response path: {path}")
        key = (int(row["member_index"]), float(row["epsilon"]), int(row["sign"]))
        grouped[key] = (row, load_path(path))

    members = sorted({key[0] for key in grouped})
    epsilons = sorted({key[1] for key in grouped})
    if len(epsilons) < 2:
        raise ValueError("cutoff analysis requires two epsilon values")
    reference_epsilon, comparison_epsilon = epsilons[:2]
    base_frame_time = float(manifest["saved_frame_interval_tau"])
    friction = float(manifest["friction"])
    rows: list[dict[str, object]] = []
    for member in members:
        responses: dict[float, dict[str, np.ndarray | float]] = {}
        mismatch_by_epsilon: dict[float, np.ndarray] = {}
        metadata: dict[float, dict[str, object]] = {}
        stop: int | None = None
        for epsilon in (reference_epsilon, comparison_epsilon):
            plus_row, plus = grouped[(member, epsilon, 1)]
            minus_row, minus = grouped[(member, epsilon, -1)]
            time = np.asarray(plus["time"], dtype=float)
            current_stop = (
                len(time)
                if args.maximum_time is None
                else int(np.searchsorted(time, args.maximum_time, side="right"))
            )
            if current_stop < 3:
                raise ValueError("maximum time must retain at least three frames")
            if stop is None:
                stop = current_stop
            elif stop != current_stop:
                raise ValueError("matched paths do not share one retained horizon")
            mismatch = np.logical_xor(
                plus["target_pair_active"][:stop],
                minus["target_pair_active"][:stop],
            )
            mismatch_by_epsilon[epsilon] = np.any(mismatch, axis=1)
            crossing = np.flatnonzero(mismatch_by_epsilon[epsilon])
            first_crossing = int(crossing[0]) if len(crossing) else None
            responses[epsilon] = matched_generator_response(plus, minus, epsilon=epsilon)
            metadata[epsilon] = plus_row
            rows.append(
                {
                    "record": "cutoff_epsilon",
                    "member_index": member,
                    "velocity_seed": plus_row["velocity_seed"],
                    "langevin_seed": plus_row["langevin_seed"],
                    "epsilon": epsilon,
                    "stride": args.stride,
                    "frame_time": base_frame_time * args.stride,
                    "maximum_time": float(time[stop - 1]),
                    "active_mask_mismatch_count": int(np.sum(mismatch)),
                    "active_mask_mismatch_frame_count": int(np.sum(np.any(mismatch, axis=1))),
                    "active_mask_mismatch_pair_count": int(np.sum(np.any(mismatch, axis=0))),
                    "first_active_mask_mismatch_frame": "" if first_crossing is None else first_crossing,
                    "first_active_mask_mismatch_time": "" if first_crossing is None else float(time[first_crossing]),
                    "minimum_absolute_cutoff_gap_plus": float(
                        np.min(np.abs(plus["nearest_cutoff_signed_gap"][:stop]))
                    ),
                    "minimum_absolute_cutoff_gap_minus": float(
                        np.min(np.abs(minus["nearest_cutoff_signed_gap"][:stop]))
                    ),
                    "cutoff_nonsmoothness_detected": float(np.any(mismatch)),
                    "fit_parameters_from_macro_observables": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )

        assert stop is not None
        if (stop - 1) % args.stride:
            raise ValueError("stride must divide the retained interval count")
        symmetric: dict[float, np.ndarray] = {}
        pre_crossing: dict[float, np.ndarray] = {}
        for epsilon in (reference_epsilon, comparison_epsilon):
            response = responses[epsilon]
            diagnostic = generator_response_tangent_diagnostic(
                np.asarray(response["state_response"])[:stop:args.stride],
                np.asarray(response["second_force_response"])[:stop:args.stride],
                frame_time=base_frame_time * args.stride,
                friction=friction,
            )
            symmetric[epsilon] = np.asarray(diagnostic["symmetric_tangent_residual"])[0]
            pre_crossing[epsilon] = right_censored_tangent_interval_mask(
                mismatch_by_epsilon[epsilon],
                stride=args.stride,
                interval_count=len(symmetric[epsilon]),
            )
        keep = pre_crossing[reference_epsilon] & pre_crossing[comparison_epsilon]
        unfiltered_reference = symmetric[reference_epsilon]
        unfiltered_comparison = symmetric[comparison_epsilon]
        censored_reference = unfiltered_reference[keep]
        censored_comparison = unfiltered_comparison[keep]
        rows.append(
            {
                "record": "cutoff_cross_epsilon",
                "member_index": member,
                "velocity_seed": metadata[reference_epsilon]["velocity_seed"],
                "langevin_seed": metadata[reference_epsilon]["langevin_seed"],
                "epsilon": comparison_epsilon,
                "reference_epsilon": reference_epsilon,
                "stride": args.stride,
                "frame_time": base_frame_time * args.stride,
                "maximum_time": float(
                    np.asarray(responses[reference_epsilon]["time"])[stop - 1]
                ),
                "interval_count": len(keep),
                "right_censored_interval_count": int(np.sum(~keep)),
                "pre_crossing_interval_count": int(np.sum(keep)),
                "unfiltered_cross_epsilon_relative_l2_error": relative_l2(
                    unfiltered_comparison, unfiltered_reference
                ),
                "unfiltered_cross_epsilon_correlation": correlation(
                    unfiltered_comparison, unfiltered_reference
                ),
                "pre_crossing_cross_epsilon_relative_l2_error": relative_l2(
                    censored_comparison, censored_reference
                ),
                "pre_crossing_cross_epsilon_correlation": (
                    correlation(censored_comparison, censored_reference)
                    if len(censored_reference) >= 2
                    else float("nan")
                ),
                "cutoff_nonsmoothness_detected": float(np.any(~keep)),
                "fit_parameters_from_macro_observables": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )

    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
