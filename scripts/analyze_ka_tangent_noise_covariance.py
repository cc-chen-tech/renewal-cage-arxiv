#!/usr/bin/env python3
"""Calibrate KA common-noise tangent residuals against exact Hessian FDT."""

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
    tangent_force_generator_noise_covariance_rate,
    tangent_noise_covariance_diagnostic,
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
        "target_pair_hessian",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload]
        if missing:
            raise ValueError(f"{path}: missing tangent covariance arrays {missing}")
        return {key: np.asarray(payload[key]) for key in keys}


def integrate_covariance(covariance_rate: np.ndarray, *, stride: int, frame_time: float) -> np.ndarray:
    rate = np.asarray(covariance_rate, dtype=float)
    if rate.ndim != 3 or rate.shape[1:] != (3, 3) or (len(rate) - 1) % stride:
        raise ValueError("covariance rate must have divisible shape (frames, 3, 3)")
    integrated = []
    for start in range(0, len(rate) - 1, stride):
        stop = start + stride
        integrated.append(
            frame_time
            * (0.5 * rate[start] + np.sum(rate[start + 1 : stop], axis=0) + 0.5 * rate[stop])
        )
    return np.asarray(integrated)


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    norm = float(np.linalg.norm(right))
    return float(np.linalg.norm(left - right) / norm) if norm > 0.0 else float("nan")


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).reshape(-1)
    right_flat = np.asarray(right, dtype=float).reshape(-1)
    if np.std(left_flat) == 0.0 or np.std(right_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty covariance table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_row(
    *,
    record: str,
    member_index: int | str,
    epsilon: float,
    stride: int,
    frame_time: float,
    maximum_time: float,
    valid_interval_count: int,
    contaminated_interval_count: int,
    result: dict[str, np.ndarray | float],
) -> dict[str, object]:
    keys = (
        "trace_variance_ratio",
        "mean_squared_mahalanobis",
        "summed_mean_squared_mahalanobis",
        "observed_predicted_energy_correlation",
        "whitened_max_abs_component_excess_kurtosis",
        "whitened_lag1_correlation",
        "minimum_covariance_eigenvalue",
        "valid_sample_count",
    )
    return {
        "record": record,
        "member_index": member_index,
        "epsilon": epsilon,
        "stride": stride,
        "frame_time": frame_time,
        "maximum_time": maximum_time,
        "valid_interval_count": valid_interval_count,
        "cutoff_contaminated_interval_count": contaminated_interval_count,
        **{key: result[key] for key in keys},
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def unavailable_diagnostic(valid_sample_count: int) -> dict[str, float]:
    """Represent a member that is too short after rigorous right censoring."""

    return {
        "trace_variance_ratio": float("nan"),
        "mean_squared_mahalanobis": float("nan"),
        "summed_mean_squared_mahalanobis": float("nan"),
        "observed_predicted_energy_correlation": float("nan"),
        "whitened_max_abs_component_excess_kurtosis": float("nan"),
        "whitened_lag1_correlation": float("nan"),
        "minimum_covariance_eigenvalue": float("nan"),
        "valid_sample_count": float(valid_sample_count),
    }


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
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("thermodynamic_claim_allowed") is not False:
        raise ValueError("manifest must preserve the thermodynamic claim boundary")
    base_frame_time = float(manifest["saved_frame_interval_tau"])
    friction = float(manifest["friction"])
    temperature = float(manifest["temperature"])
    grouped: dict[tuple[int, float, int], dict[str, np.ndarray]] = {}
    for row in manifest["records"]:
        path = root / str(row["path"])
        if not path.is_file() or file_sha256(path) != row["path_sha256"]:
            raise ValueError(f"missing or modified response path: {path}")
        grouped[(int(row["member_index"]), float(row["epsilon"]), int(row["sign"]))] = load_path(path)
    members = sorted({key[0] for key in grouped})
    epsilons = sorted({key[1] for key in grouped})
    if len(members) < 2 or len(epsilons) < 2:
        raise ValueError("covariance analysis requires at least two members and epsilons")

    residuals: dict[tuple[int, float], np.ndarray] = {}
    covariance: dict[tuple[int, float], np.ndarray] = {}
    valid_masks: dict[tuple[int, float], np.ndarray] = {}
    maximum_time: float | None = None
    rows: list[dict[str, object]] = []
    for member in members:
        for epsilon in epsilons:
            plus = grouped[(member, epsilon, 1)]
            minus = grouped[(member, epsilon, -1)]
            time = np.asarray(plus["time"], dtype=float)
            stop = len(time) if args.maximum_time is None else int(np.searchsorted(time, args.maximum_time, side="right"))
            if stop < 3 or (stop - 1) % args.stride:
                raise ValueError("retained interval count must be positive and divisible by stride")
            maximum_time = float(time[stop - 1])
            response = matched_generator_response(plus, minus, epsilon=epsilon)
            tangent = generator_response_tangent_diagnostic(
                np.asarray(response["state_response"])[:stop:args.stride],
                np.asarray(response["second_force_response"])[:stop:args.stride],
                frame_time=base_frame_time * args.stride,
                friction=friction,
            )
            residual = np.asarray(tangent["symmetric_tangent_residual"])[0]
            rate = tangent_force_generator_noise_covariance_rate(
                plus["target_pair_hessian"][:stop],
                minus["target_pair_hessian"][:stop],
                epsilon=epsilon,
                friction=friction,
                temperature=temperature,
            )
            integrated = integrate_covariance(rate, stride=args.stride, frame_time=base_frame_time)
            mismatch = np.any(
                np.logical_xor(plus["target_pair_active"][:stop], minus["target_pair_active"][:stop]),
                axis=1,
            )
            valid = right_censored_tangent_interval_mask(
                mismatch,
                stride=args.stride,
                interval_count=len(residual),
            )
            valid_count = int(np.sum(valid))
            result = (
                tangent_noise_covariance_diagnostic(residual, integrated, valid_mask=valid)
                if valid_count >= 4
                else unavailable_diagnostic(valid_count)
            )
            residuals[(member, epsilon)] = residual
            covariance[(member, epsilon)] = integrated
            valid_masks[(member, epsilon)] = valid
            rows.append(
                diagnostic_row(
                    record="covariance_member",
                    member_index=member,
                    epsilon=epsilon,
                    stride=args.stride,
                    frame_time=base_frame_time * args.stride,
                    maximum_time=maximum_time,
                    valid_interval_count=int(np.sum(valid)),
                    contaminated_interval_count=int(np.sum(~valid)),
                    result=result,
                )
            )

    assert maximum_time is not None
    for epsilon in epsilons:
        stacked_residual = np.stack([residuals[(member, epsilon)] for member in members])
        stacked_covariance = np.stack([covariance[(member, epsilon)] for member in members])
        stacked_valid = np.stack([valid_masks[(member, epsilon)] for member in members])
        result = tangent_noise_covariance_diagnostic(
            stacked_residual,
            stacked_covariance,
            valid_mask=stacked_valid,
        )
        rows.append(
            diagnostic_row(
                record="covariance_aggregate",
                member_index="",
                epsilon=epsilon,
                stride=args.stride,
                frame_time=base_frame_time * args.stride,
                maximum_time=maximum_time,
                valid_interval_count=int(np.sum(stacked_valid)),
                contaminated_interval_count=int(np.sum(~stacked_valid)),
                result=result,
            )
        )

    reference_epsilon, comparison_epsilon = epsilons[:2]
    for member in members:
        valid = valid_masks[(member, reference_epsilon)] & valid_masks[(member, comparison_epsilon)]
        reference = covariance[(member, reference_epsilon)][valid]
        comparison = covariance[(member, comparison_epsilon)][valid]
        rows.append(
            {
                "record": "covariance_cross_epsilon",
                "member_index": member,
                "epsilon": comparison_epsilon,
                "reference_epsilon": reference_epsilon,
                "stride": args.stride,
                "frame_time": base_frame_time * args.stride,
                "maximum_time": maximum_time,
                "valid_interval_count": int(np.sum(valid)),
                "covariance_cross_epsilon_relative_l2_error": relative_l2(comparison, reference),
                "covariance_cross_epsilon_correlation": correlation(comparison, reference),
                "fit_parameters_from_macro_observables": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )

    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
