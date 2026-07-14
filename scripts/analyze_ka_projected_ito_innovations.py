#!/usr/bin/env python3
"""Audit exact smooth-cage Ito innovations on microscopic KA trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_ka_hankel_slow_force_bath import file_sha256  # noqa: E402
from ka_projected_innovation import (  # noqa: E402
    block_projected_ito_increments,
    multivariate_noise_covariance_diagnostic,
)
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_smooth_cage import (  # noqa: E402
    smooth_cage_joint_noise_covariance_rate,
    smooth_force_support_cage_batch,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_drift_cache(path: Path) -> dict[str, np.ndarray | float | str]:
    required = (
        "relative_position",
        "relative_velocity",
        "center_velocity",
        "relative_drift",
        "center_drift",
        "target_indices",
        "trajectory_sha256",
        "thermodynamic_claim_allowed",
    )
    with np.load(path) as cache:
        if any(key not in cache for key in required):
            raise ValueError(f"incomplete decomposed-drift cache: {path}")
        result = {key: np.asarray(cache[key]) for key in required}
    shape = np.asarray(result["relative_velocity"]).shape
    if (
        len(shape) != 3
        or shape[-1] != 3
        or any(np.asarray(result[key]).shape != shape for key in required[:5])
        or float(result["thermodynamic_claim_allowed"]) != 0.0
    ):
        raise ValueError(f"invalid decomposed-drift cache: {path}")
    return result


def load_or_extract_covariance(
    trajectory_path: Path,
    *,
    drift: dict[str, np.ndarray | float | str],
    cache_path: Path,
    friction: float,
    temperature: float,
    target_batch_size: int,
) -> dict[str, np.ndarray | float | str]:
    trajectory_hash = file_sha256(trajectory_path)
    targets = np.asarray(drift["target_indices"], dtype=int)
    expected_shape = np.asarray(drift["relative_velocity"]).shape
    if trajectory_hash != str(drift["trajectory_sha256"]):
        raise ValueError("trajectory hash does not match decomposed-drift cache")
    if cache_path.is_file():
        with np.load(cache_path) as cache:
            valid = (
                str(cache["trajectory_sha256"]) == trajectory_hash
                and np.array_equal(np.asarray(cache["target_indices"], dtype=int), targets)
                and math.isclose(float(cache["friction"]), friction)
                and math.isclose(float(cache["temperature"]), temperature)
                and float(cache["thermodynamic_claim_allowed"]) == 0.0
            )
            joint = np.asarray(cache["joint_noise_covariance_rate"], dtype=float)
            if valid and joint.shape == (*expected_shape[:2], 6, 6):
                return {
                    "joint_noise_covariance_rate": joint,
                    "maximum_relative_position_error": float(
                        cache["maximum_relative_position_error"]
                    ),
                    "maximum_relative_velocity_error": float(
                        cache["maximum_relative_velocity_error"]
                    ),
                    "trajectory_sha256": trajectory_hash,
                }

    trajectory = load_lammps_custom_trajectory(trajectory_path)
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    velocities = np.asarray(trajectory["velocities"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    if (
        positions.shape != velocities.shape
        or positions[:, targets].shape != expected_shape
        or np.any(particle_types[targets] != 0)
    ):
        raise ValueError("trajectory does not align with decomposed-drift cache")

    joint = np.empty((*expected_shape[:2], 6, 6), dtype=float)
    maximum_position_error = 0.0
    maximum_velocity_error = 0.0
    for frame in range(len(positions)):
        cage = smooth_force_support_cage_batch(
            positions[frame],
            velocities=velocities[frame],
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=targets,
            target_batch_size=target_batch_size,
        )
        noise = smooth_cage_joint_noise_covariance_rate(
            np.asarray(cage["jacobian_gram"]),
            np.asarray(cage["target_jacobian_block"]),
            friction=friction,
            temperature=temperature,
        )
        joint[frame] = noise["joint_noise_covariance_rate"]
        maximum_position_error = max(
            maximum_position_error,
            float(
                np.max(
                    np.abs(
                        np.asarray(cage["relative_position"])
                        - np.asarray(drift["relative_position"])[frame]
                    )
                )
            ),
        )
        maximum_velocity_error = max(
            maximum_velocity_error,
            float(
                np.max(
                    np.abs(
                        np.asarray(cage["relative_velocity"])
                        - np.asarray(drift["relative_velocity"])[frame]
                    )
                )
            ),
        )
    if maximum_position_error > 2e-5 or maximum_velocity_error > 2e-5:
        raise ValueError("covariance projection does not reproduce cached cage paths")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        joint_noise_covariance_rate=joint,
        target_indices=targets,
        trajectory_sha256=np.asarray(trajectory_hash),
        friction=np.asarray(friction),
        temperature=np.asarray(temperature),
        maximum_relative_position_error=np.asarray(maximum_position_error),
        maximum_relative_velocity_error=np.asarray(maximum_velocity_error),
        thermodynamic_claim_allowed=np.asarray(0.0),
    )
    return {
        "joint_noise_covariance_rate": joint,
        "maximum_relative_position_error": maximum_position_error,
        "maximum_relative_velocity_error": maximum_velocity_error,
        "trajectory_sha256": trajectory_hash,
    }


def clone_diagnostic(
    clone: dict[str, np.ndarray | float | str],
    *,
    frame_time: float,
    stride: int,
    scheme: str,
) -> tuple[dict[str, np.ndarray | float], np.ndarray, np.ndarray, np.ndarray]:
    state = np.concatenate(
        [np.asarray(clone["center_velocity"]), np.asarray(clone["relative_velocity"])],
        axis=2,
    )
    drift = np.concatenate(
        [np.asarray(clone["center_drift"]), np.asarray(clone["relative_drift"])],
        axis=2,
    )
    physical_state = np.concatenate(
        [state, np.asarray(clone["relative_position"])], axis=2
    )
    blocks = block_projected_ito_increments(
        state,
        drift,
        np.asarray(clone["joint_noise_covariance_rate"]),
        frame_time=frame_time,
        stride=stride,
        scheme=scheme,
    )
    residual = np.transpose(np.asarray(blocks["residual"]), (1, 0, 2))
    covariance = np.transpose(
        np.asarray(blocks["integrated_covariance"]), (1, 0, 2, 3)
    )
    block_count = residual.shape[1]
    starting_indices = np.asarray(blocks["starting_frame_indices"], dtype=int)
    starting = np.transpose(physical_state[starting_indices], (1, 0, 2))
    diagnostic = multivariate_noise_covariance_diagnostic(
        residual,
        covariance,
        starting_state=starting,
    )
    return diagnostic, residual, covariance, starting


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--covariance-cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["left", "adams_bashforth2", "trapezoid"],
    )
    args = parser.parse_args()
    strides = tuple(sorted(set(args.strides)))
    schemes = tuple(dict.fromkeys(args.schemes))
    if (
        args.expected_clone_count < 2
        or args.target_count < 1
        or args.target_batch_size < 1
        or args.friction <= 0.0
        or args.temperature <= 0.0
        or not strides
        or any(stride < 1 for stride in strides)
        or not schemes
        or any(
            scheme not in {"left", "adams_bashforth2", "trapezoid"}
            for scheme in schemes
        )
    ):
        raise ValueError("invalid projected-Ito audit controls")

    manifest = json.loads((args.clone_directory / "manifest.json").read_text())
    frame_time = float(manifest["saved_frame_interval_tau"])
    if (
        int(manifest["clone_count"]) != args.expected_clone_count
        or manifest["dynamics"] != "nve_plus_langevin"
        or bool(manifest["thermodynamic_claim_allowed"])
        or not math.isclose(float(manifest["langevin_damping"]), args.friction)
        or not math.isclose(float(manifest["temperature"]), args.temperature)
    ):
        raise ValueError("manifest does not match the projected-Ito protocol")
    trajectory_paths = sorted(
        path / "trajectory.lammpstrj"
        for path in args.clone_directory.glob("clone_*")
        if (path / "trajectory.lammpstrj").is_file()
    )
    if len(trajectory_paths) != args.expected_clone_count:
        raise ValueError("completed clone count does not match manifest")

    clones: list[dict[str, np.ndarray | float | str]] = []
    fixed_targets: np.ndarray | None = None
    for index, trajectory_path in enumerate(trajectory_paths, start=1):
        drift = load_drift_cache(
            args.drift_cache_directory / f"clone_{index:03d}_decomposed_drift.npz"
        )
        targets = np.asarray(drift["target_indices"], dtype=int)
        if len(targets) != args.target_count:
            raise ValueError("target count does not match the audit protocol")
        if fixed_targets is None:
            fixed_targets = targets
        elif not np.array_equal(targets, fixed_targets):
            raise ValueError("all clones must use the same fixed targets")
        covariance = load_or_extract_covariance(
            trajectory_path,
            drift=drift,
            cache_path=args.covariance_cache_directory
            / f"clone_{index:03d}_projected_covariance.npz",
            friction=args.friction,
            temperature=args.temperature,
            target_batch_size=args.target_batch_size,
        )
        clones.append({**drift, **covariance})

    details: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    metric_names = (
        "trace_variance_ratio",
        "mean_squared_mahalanobis_per_dimension",
        "maximum_absolute_whitened_mean",
        "maximum_absolute_whitened_covariance_error",
        "maximum_absolute_whitened_component_excess_kurtosis",
        "maximum_absolute_whitened_lag1_correlation",
        "maximum_absolute_whitened_state_correlation",
        "minimum_integrated_covariance_eigenvalue",
        "sample_count",
    )
    for scheme in schemes:
        for stride in strides:
            pooled_residual: list[np.ndarray] = []
            pooled_covariance: list[np.ndarray] = []
            pooled_state: list[np.ndarray] = []
            for clone_index, clone in enumerate(clones, start=1):
                diagnostic, residual, covariance, starting = clone_diagnostic(
                    clone,
                    frame_time=frame_time,
                    stride=stride,
                    scheme=scheme,
                )
                details.append(
                    {
                        "record": "clone",
                        "scheme": scheme,
                        "stride": float(stride),
                        "clone_index": float(clone_index),
                        **{key: float(diagnostic[key]) for key in metric_names},
                        "maximum_relative_position_error": float(
                            clone["maximum_relative_position_error"]
                        ),
                        "maximum_relative_velocity_error": float(
                            clone["maximum_relative_velocity_error"]
                        ),
                        "trajectory_sha256": str(clone["trajectory_sha256"]),
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
                pooled_residual.append(residual)
                pooled_covariance.append(covariance)
                pooled_state.append(starting)
            pooled = multivariate_noise_covariance_diagnostic(
                np.concatenate(pooled_residual, axis=0),
                np.concatenate(pooled_covariance, axis=0),
                starting_state=np.concatenate(pooled_state, axis=0),
            )
            summaries.append(
                {
                    "record": "pooled",
                    "scheme": scheme,
                    "stride": float(stride),
                    "clone_count": float(len(clones)),
                    **{key: float(pooled[key]) for key in metric_names},
                    "thermodynamic_claim_allowed": 0.0,
                }
            )

    primary = next(
        row for row in summaries if row["scheme"] == "left" and row["stride"] == 1.0
    )
    adapted = next(
        row
        for row in summaries
        if row["scheme"] == "adams_bashforth2" and row["stride"] == 1.0
    )
    trapezoid = next(
        row
        for row in summaries
        if row["scheme"] == "trapezoid" and row["stride"] == 1.0
    )

    def passes_local_gate(row: dict[str, object]) -> bool:
        return (
            0.8 <= float(row["trace_variance_ratio"]) <= 1.2
            and 0.8 <= float(row["mean_squared_mahalanobis_per_dimension"]) <= 1.2
            and float(row["maximum_absolute_whitened_mean"]) <= 0.05
            and float(row["maximum_absolute_whitened_covariance_error"]) <= 0.10
            and float(row["maximum_absolute_whitened_lag1_correlation"]) <= 0.05
            and float(row["maximum_absolute_whitened_state_correlation"]) <= 0.10
        )

    local_gate = passes_local_gate(primary)
    adapted_gate = passes_local_gate(adapted)
    trapezoid_gate = passes_local_gate(trapezoid)
    summaries.append(
        {
            "record": "verdict",
            "scheme": "left",
            "stride": 1.0,
            "projected_ito_local_gate_pass": float(local_gate),
            "adapted_second_order_consistency_gate_pass": float(adapted_gate),
            "trapezoid_sensitivity_gate_pass": float(trapezoid_gate),
            "projected_sde_numerically_supported": float(adapted_gate and trapezoid_gate),
            "microscopic_projected_sde_allowed": float(local_gate),
            "autonomous_single_particle_gle_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)


if __name__ == "__main__":
    main()
