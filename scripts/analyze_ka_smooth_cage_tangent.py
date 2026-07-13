#!/usr/bin/env python3
"""Apply preregistered gates to smooth-cage common-noise tangent paths."""

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

from ka_generator_response import tangent_noise_covariance_diagnostic  # noqa: E402
from ka_smooth_cage import integrated_smooth_cage_tangent_covariance  # noqa: E402


PRIMARY_STRIDE = 5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tangent(path: Path) -> dict[str, np.ndarray | float]:
    required = (
        "time",
        "relative_velocity_response",
        "projected_drift_response",
        "tangent_noise_covariance_rate",
        "positive_condition_number",
        "negative_condition_number",
        "frame_time",
        "friction",
        "temperature",
        "epsilon",
        "member_index",
        "thermodynamic_claim_allowed",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        output = {key: np.asarray(payload[key]) for key in required}
    if float(output["thermodynamic_claim_allowed"]) != 0.0:
        raise ValueError(f"{path}: thermodynamic claim boundary was not preserved")
    return output


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(right))
    return float(np.linalg.norm(left - right) / denominator) if denominator > 0.0 else math.nan


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=float).reshape(-1)
    right_flat = np.asarray(right, dtype=float).reshape(-1)
    if np.std(left_flat) <= 0.0 or np.std(right_flat) <= 0.0:
        return math.nan
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def diagnostic_fields(result: dict[str, np.ndarray | float]) -> dict[str, float]:
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
    return {key: float(result[key]) for key in keys}


def stride_output_path(prefix: Path, stride: int, temperature_label: str) -> Path:
    return prefix.with_name(f"{prefix.name}_stride{stride}_{temperature_label}.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--stride-output-prefix", type=Path, required=True)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--temperature-label", default="T058")
    args = parser.parse_args()

    input_directory = args.input_directory.resolve()
    manifest_path = input_directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("input directory must contain manifest.json")
    if not args.strides or len(set(args.strides)) != len(args.strides) or any(
        isinstance(value, bool) or value < 1 for value in args.strides
    ):
        raise ValueError("strides must be unique positive integers")
    if PRIMARY_STRIDE not in args.strides:
        raise ValueError(f"strides must include preregistered primary stride {PRIMARY_STRIDE}")

    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records", [])
    manifest_protocol_pass = bool(
        manifest.get("protocol") == "smooth_force_support_cage_common_noise_tangent"
        and manifest.get("potential_protocol") == "ka_lj_c3_switch"
        and manifest.get("cage_weight") == "wendland_c4"
        and float(manifest.get("cage_support_sigma", math.nan)) == 2.5
        and manifest.get("thermodynamic_claim_allowed") is False
        and len(records) == int(manifest.get("record_count", -1))
    )
    grouped: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
    path_hashes_pass = True
    for record in records:
        path = input_directory / str(record["path"])
        path_hashes_pass = bool(
            path_hashes_pass
            and path.is_file()
            and file_sha256(path) == str(record["path_sha256"])
        )
        if not path.is_file():
            raise ValueError(f"missing tangent path: {path}")
        payload = load_tangent(path)
        key = (int(record["member_index"]), float(record["epsilon"]))
        if key in grouped:
            raise ValueError("manifest contains duplicate member/epsilon records")
        grouped[key] = payload
    members = sorted({key[0] for key in grouped})
    epsilons = sorted({key[1] for key in grouped})
    if len(members) < 2 or len(epsilons) != 2:
        raise ValueError("analysis requires at least two members and exactly two epsilons")
    if set(grouped) != {(member, epsilon) for member in members for epsilon in epsilons}:
        raise ValueError("manifest does not contain a complete member/epsilon rectangle")

    aggregate_by_stride: dict[int, dict[float, dict[str, float]]] = {}
    cross_by_stride: dict[int, list[dict[str, object]]] = {}
    integrated_by_stride: dict[
        int, dict[tuple[int, float], dict[str, np.ndarray | float]]
    ] = {}
    for stride in args.strides:
        rows: list[dict[str, object]] = []
        integrated: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
        for member in members:
            for epsilon in epsilons:
                reduced = integrated_smooth_cage_tangent_covariance(
                    grouped[(member, epsilon)], stride=stride
                )
                integrated[(member, epsilon)] = reduced
                diagnostic = tangent_noise_covariance_diagnostic(
                    np.asarray(reduced["residual"]),
                    np.asarray(reduced["integrated_covariance"]),
                )
                payload = grouped[(member, epsilon)]
                rows.append(
                    {
                        "record": "member",
                        "member_index": member,
                        "epsilon": epsilon,
                        "stride": stride,
                        "frame_time_tau": float(payload["frame_time"]) * stride,
                        "interval_count": len(reduced["residual"]),
                        "maximum_jacobian_gram_condition_number": float(
                            max(
                                np.max(payload["positive_condition_number"]),
                                np.max(payload["negative_condition_number"]),
                            )
                        ),
                        **diagnostic_fields(diagnostic),
                        "fit_parameters_from_macro_observables": False,
                        "thermodynamic_claim_allowed": False,
                    }
                )

        aggregate_by_stride[stride] = {}
        for epsilon in epsilons:
            residual = np.stack(
                [np.asarray(integrated[(member, epsilon)]["residual"]) for member in members]
            )
            covariance = np.stack(
                [
                    np.asarray(integrated[(member, epsilon)]["integrated_covariance"])
                    for member in members
                ]
            )
            diagnostic = tangent_noise_covariance_diagnostic(residual, covariance)
            fields = diagnostic_fields(diagnostic)
            aggregate_by_stride[stride][epsilon] = fields
            rows.append(
                {
                    "record": "aggregate",
                    "member_index": "all",
                    "epsilon": epsilon,
                    "stride": stride,
                    "frame_time_tau": float(manifest["saved_frame_interval_tau"]) * stride,
                    "interval_count": residual.shape[0] * residual.shape[1],
                    **fields,
                    "fit_parameters_from_macro_observables": False,
                    "thermodynamic_claim_allowed": False,
                }
            )

        cross_rows: list[dict[str, object]] = []
        for member in members:
            left = np.asarray(integrated[(member, epsilons[0])]["integrated_covariance"])
            right = np.asarray(integrated[(member, epsilons[1])]["integrated_covariance"])
            cross_epsilon_covariance_relative_l2 = relative_l2(left, right)
            cross_epsilon_covariance_correlation = correlation(left, right)
            row = {
                "record": "cross_epsilon",
                "member_index": member,
                "epsilon": "both",
                "stride": stride,
                "frame_time_tau": float(manifest["saved_frame_interval_tau"]) * stride,
                "interval_count": len(left),
                "cross_epsilon_covariance_relative_l2": cross_epsilon_covariance_relative_l2,
                "cross_epsilon_covariance_correlation": cross_epsilon_covariance_correlation,
                "fit_parameters_from_macro_observables": False,
                "thermodynamic_claim_allowed": False,
            }
            rows.append(row)
            cross_rows.append(row)
        cross_by_stride[stride] = cross_rows
        integrated_by_stride[stride] = integrated
        write_rows(
            stride_output_path(args.stride_output_prefix, stride, args.temperature_label),
            rows,
        )

    primary = aggregate_by_stride[PRIMARY_STRIDE]
    primary_cross = cross_by_stride[PRIMARY_STRIDE]
    primary_integrated = integrated_by_stride[PRIMARY_STRIDE]
    expected_records = len(members) * len(epsilons)
    expected_intervals_per_epsilon = len(members) * int(
        round(float(manifest["duration_tau"]) / float(manifest["saved_frame_interval_tau"]))
    ) // PRIMARY_STRIDE
    actual_intervals = [int(primary[epsilon]["valid_sample_count"]) for epsilon in epsilons]
    trace_values = [float(primary[epsilon]["trace_variance_ratio"]) for epsilon in epsilons]
    mahalanobis_values = [
        float(primary[epsilon]["mean_squared_mahalanobis"]) for epsilon in epsilons
    ]
    lag_values = [
        abs(float(primary[epsilon]["whitened_lag1_correlation"])) for epsilon in epsilons
    ]
    cross_relative = [float(row["cross_epsilon_covariance_relative_l2"]) for row in primary_cross]
    cross_correlation = [float(row["cross_epsilon_covariance_correlation"]) for row in primary_cross]
    integrity_gate_pass = bool(
        manifest_protocol_pass
        and path_hashes_pass
        and len(records) == expected_records
        and len(members) == int(manifest["member_count"])
    )
    full_horizon_gate_pass = all(value == expected_intervals_per_epsilon for value in actual_intervals)
    cross_epsilon_gate_pass = bool(
        max(cross_relative) <= 0.05 and min(cross_correlation) >= 0.995
    )
    trace_variance_gate_pass = all(0.8 <= value <= 1.2 for value in trace_values)
    mahalanobis_gate_pass = all(2.4 <= value <= 3.6 for value in mahalanobis_values)
    whitened_lag1_gate_pass = all(value <= 0.1 for value in lag_values)
    microscopic_smooth_cage_tangent_gate_pass = bool(
        integrity_gate_pass
        and full_horizon_gate_pass
        and cross_epsilon_gate_pass
        and trace_variance_gate_pass
        and mahalanobis_gate_pass
        and whitened_lag1_gate_pass
    )
    observed_residual_energy = float(
        sum(
            np.sum(np.asarray(reduced["residual"]) ** 2)
            for reduced in primary_integrated.values()
        )
    )
    predicted_delta_j_covariance_trace = float(
        sum(
            np.sum(
                np.trace(
                    np.asarray(reduced["integrated_covariance"]), axis1=1, axis2=2
                )
            )
            for reduced in primary_integrated.values()
        )
    )
    without_delta_j_null_rejected = bool(
        microscopic_smooth_cage_tangent_gate_pass
        and observed_residual_energy > 0.0
        and predicted_delta_j_covariance_trace > 0.0
    )
    maximum_condition_number = max(
        float(
            max(
                np.max(payload["positive_condition_number"]),
                np.max(payload["negative_condition_number"]),
            )
        )
        for payload in grouped.values()
    )
    summary = {
        "potential_protocol": "ka_lj_c3_switch",
        "cage_coordinate": "wendland_c4_force_support_relative_position",
        "primary_stride": PRIMARY_STRIDE,
        "frame_time_tau": PRIMARY_STRIDE * float(manifest["saved_frame_interval_tau"]),
        "member_count": len(members),
        "epsilon_count": len(epsilons),
        "response_pair_count": len(records),
        "expected_intervals_per_epsilon": expected_intervals_per_epsilon,
        "minimum_valid_intervals_per_epsilon": min(actual_intervals),
        "minimum_trace_variance_ratio": min(trace_values),
        "maximum_trace_variance_ratio": max(trace_values),
        "minimum_mean_squared_mahalanobis": min(mahalanobis_values),
        "maximum_mean_squared_mahalanobis": max(mahalanobis_values),
        "maximum_abs_whitened_lag1_correlation": max(lag_values),
        "maximum_cross_epsilon_covariance_relative_l2": max(cross_relative),
        "minimum_cross_epsilon_covariance_correlation": min(cross_correlation),
        "maximum_jacobian_gram_condition_number": maximum_condition_number,
        "observed_tangent_residual_energy": observed_residual_energy,
        "predicted_delta_j_covariance_trace": predicted_delta_j_covariance_trace,
        "without_delta_j_predicted_covariance_trace": 0.0,
        "without_delta_j_null_rejected": without_delta_j_null_rejected,
        "integrity_gate_pass": integrity_gate_pass,
        "full_horizon_gate_pass": full_horizon_gate_pass,
        "cross_epsilon_gate_pass": cross_epsilon_gate_pass,
        "trace_variance_gate_pass": trace_variance_gate_pass,
        "mahalanobis_gate_pass": mahalanobis_gate_pass,
        "whitened_lag1_gate_pass": whitened_lag1_gate_pass,
        "microscopic_smooth_cage_tangent_gate_pass": microscopic_smooth_cage_tangent_gate_pass,
        "event_clock_claim_allowed": False,
        "autonomous_single_particle_gle_claim_allowed": False,
        "fit_parameters_from_macro_observables": False,
        "thermodynamic_claim_allowed": False,
    }
    write_rows(args.summary_output, [summary])
    print(
        f"smooth cage tangent pass={microscopic_smooth_cage_tangent_gate_pass}; "
        f"trace=[{min(trace_values):.4g},{max(trace_values):.4g}]; "
        f"mahal=[{min(mahalanobis_values):.4g},{max(mahalanobis_values):.4g}]"
    )


if __name__ == "__main__":
    main()
