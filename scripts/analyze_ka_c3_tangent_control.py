#!/usr/bin/env python3
"""Summarize preregistered full-horizon C3 tangent-covariance gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty diagnostic table: {path}")
    return rows


def write_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--covariance", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--linearity-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("stride must be positive")
    if args.linearity_tolerance <= 0.0:
        raise ValueError("linearity tolerance must be positive")
    for path in (args.manifest, args.covariance, args.resolution):
        if not path.is_file():
            raise ValueError(f"required input does not exist: {path}")

    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("potential_protocol") != "ka_lj_c3_switch":
        raise ValueError("control summary requires a C3-switched response manifest")
    if manifest.get("thermodynamic_claim_allowed") is not False:
        raise ValueError("manifest must preserve the thermodynamic claim boundary")
    records = manifest.get("records", [])
    if len(records) != int(manifest.get("record_count", -1)):
        raise ValueError("manifest response record count is inconsistent")
    paths_verified = True
    for record in records:
        path = root / str(record["path"])
        paths_verified = paths_verified and path.is_file() and file_sha256(path) == record["path_sha256"]

    interval_count = int(round(float(manifest["duration_tau"]) / float(manifest["saved_frame_interval_tau"])))
    if interval_count % args.stride:
        raise ValueError("stride must divide the full response horizon")
    expected_per_epsilon = int(manifest["member_count"]) * interval_count // args.stride

    covariance_rows = read_rows(args.covariance)
    covariance_aggregate = [
        row
        for row in covariance_rows
        if row["record"] == "covariance_aggregate" and int(row["stride"]) == args.stride
    ]
    covariance_cross = [
        row
        for row in covariance_rows
        if row["record"] == "covariance_cross_epsilon" and int(row["stride"]) == args.stride
    ]
    if len(covariance_aggregate) != len(manifest["epsilons"]):
        raise ValueError("covariance table lacks one aggregate per epsilon")
    if len(covariance_cross) != int(manifest["member_count"]):
        raise ValueError("covariance table lacks one cross-epsilon row per member")

    valid_counts = [int(row["valid_interval_count"]) for row in covariance_aggregate]
    censored_counts = [int(row["cutoff_contaminated_interval_count"]) for row in covariance_aggregate]
    trace_ratios = [float(row["trace_variance_ratio"]) for row in covariance_aggregate]
    mahalanobis = [float(row["mean_squared_mahalanobis"]) for row in covariance_aggregate]
    lag1 = [abs(float(row["whitened_lag1_correlation"])) for row in covariance_aggregate]
    support_crossings = [int(row["pair_support_crossing_interval_count"]) for row in covariance_aggregate]
    cross_covariance_relative = [float(row["covariance_cross_epsilon_relative_l2_error"]) for row in covariance_cross]
    cross_covariance_correlation = [float(row["covariance_cross_epsilon_correlation"]) for row in covariance_cross]

    resolution_rows = read_rows(args.resolution)
    resolution_aggregate = [
        row
        for row in resolution_rows
        if row["record"] == "resolution_aggregate" and int(row["stride"]) == args.stride
    ]
    resolution_cross = [
        row
        for row in resolution_rows
        if row["record"] == "resolution_cross_epsilon_aggregate" and int(row["stride"]) == args.stride
    ]
    if len(resolution_aggregate) != len(manifest["epsilons"]) or len(resolution_cross) != 1:
        raise ValueError("resolution table lacks required aggregate rows")
    deterministic_identity_errors = [
        float(row[key])
        for row in resolution_aggregate
        for key in (
            "position_velocity_relative_l2_error",
            "velocity_force_relative_l2_error",
            "force_generator_relative_l2_error",
        )
    ]
    generator_second_errors = [float(row["generator_second_relative_l2_error"]) for row in resolution_aggregate]
    resolution_cross_relative = float(resolution_cross[0]["symmetric_residual_cross_epsilon_relative_l2_error"])
    resolution_cross_correlation = float(resolution_cross[0]["symmetric_residual_cross_epsilon_correlation"])

    integrity_pass = bool(paths_verified and len(records) == 32)
    full_horizon_pass = all(value == expected_per_epsilon for value in valid_counts) and not any(censored_counts)
    trace_pass = all(0.8 <= value <= 1.2 for value in trace_ratios)
    mahalanobis_pass = all(2.4 <= value <= 3.6 for value in mahalanobis)
    lag1_pass = all(value < 0.1 for value in lag1)
    cross_epsilon_pass = bool(
        max(cross_covariance_relative) <= args.linearity_tolerance
        and resolution_cross_relative <= args.linearity_tolerance
    )
    covariance_gate_pass = bool(
        integrity_pass
        and full_horizon_pass
        and trace_pass
        and mahalanobis_pass
        and lag1_pass
        and cross_epsilon_pass
    )
    row = {
        "potential_protocol": "ka_lj_c3_switch",
        "stride": args.stride,
        "frame_time_tau": args.stride * float(manifest["saved_frame_interval_tau"]),
        "member_count": int(manifest["member_count"]),
        "epsilon_count": len(manifest["epsilons"]),
        "response_record_count": len(records),
        "paths_hash_verified": paths_verified,
        "expected_intervals_per_epsilon": expected_per_epsilon,
        "minimum_valid_intervals_per_epsilon": min(valid_counts),
        "maximum_right_censored_intervals_per_epsilon": max(censored_counts),
        "maximum_pair_support_crossing_intervals": max(support_crossings),
        "minimum_trace_variance_ratio": min(trace_ratios),
        "maximum_trace_variance_ratio": max(trace_ratios),
        "minimum_mean_squared_mahalanobis": min(mahalanobis),
        "maximum_mean_squared_mahalanobis": max(mahalanobis),
        "maximum_abs_whitened_lag1_correlation": max(lag1),
        "maximum_memberwise_covariance_cross_epsilon_relative_l2_error": max(cross_covariance_relative),
        "minimum_memberwise_covariance_cross_epsilon_correlation": min(cross_covariance_correlation),
        "resolution_cross_epsilon_relative_l2_error": resolution_cross_relative,
        "resolution_cross_epsilon_correlation": resolution_cross_correlation,
        "maximum_deterministic_identity_relative_l2_error": max(deterministic_identity_errors),
        "minimum_generator_second_relative_l2_error": min(generator_second_errors),
        "maximum_generator_second_relative_l2_error": max(generator_second_errors),
        "integrity_gate_pass": integrity_pass,
        "full_horizon_gate_pass": full_horizon_pass,
        "trace_variance_gate_pass": trace_pass,
        "mahalanobis_gate_pass": mahalanobis_pass,
        "whitened_lag1_gate_pass": lag1_pass,
        "cross_epsilon_gate_pass": cross_epsilon_pass,
        "microscopic_tangent_covariance_gate_pass": covariance_gate_pass,
        "original_four_deterministic_identity_gate_evaluable": False,
        "gate_interpretation": "first_three_deterministic_identities_plus_stochastic_LF_covariance",
        "all_original_design_gates_pass": False,
        "fit_parameters_from_macro_observables": False,
        "thermodynamic_claim_allowed": False,
    }
    write_row(args.output, row)
    print(
        f"C3 tangent covariance pass={covariance_gate_pass}; full={full_horizon_pass}; "
        f"trace=[{min(trace_ratios):.4g},{max(trace_ratios):.4g}]"
    )


if __name__ == "__main__":
    main()
