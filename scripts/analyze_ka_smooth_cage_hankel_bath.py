#!/usr/bin/env python3
"""Test an exact-force Hankel bath augmented by smooth cage coordinates."""

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

from analyze_ka_hankel_slow_force_bath import (  # noqa: E402
    evaluate_displacements,
    event_rate,
    file_sha256,
)
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_slow_force_bath import (  # noqa: E402
    fit_covariance_contracted_model,
    fit_force_hankel_basis,
    heldout_state_diagnostics,
    project_force_hankel,
    simulate_slow_bath_displacements,
)
from ka_smooth_cage import smooth_force_support_cage_batch  # noqa: E402
from ka_smooth_cage_bath import (  # noqa: E402
    assemble_augmented_hankel_state,
    cage_kinematic_diagnostic,
    heldout_residual_lag_profile,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def standard_error(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def load_force_cache(
    path: Path,
    *,
    target_count: int,
    target_seed: int,
    expected_targets: np.ndarray | None,
) -> dict[str, np.ndarray | float | str]:
    with np.load(path) as cache:
        positions = np.asarray(cache["positions"], dtype=float)
        velocities = np.asarray(cache["velocities"], dtype=float)
        forces = np.asarray(cache["forces"], dtype=float)
        targets = np.asarray(cache["target_indices"], dtype=int)
        frame_time = float(cache["frame_time"])
        valid = (
            positions.shape == velocities.shape == forces.shape
            and positions.ndim == 3
            and positions.shape[1:] == (target_count, 3)
            and targets.shape == (target_count,)
            and int(cache["target_count"]) == target_count
            and int(cache["target_seed"]) == target_seed
            and float(cache["thermodynamic_claim_allowed"]) == 0.0
            and math.isclose(frame_time, 0.01)
            and np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
            and np.all(np.isfinite(forces))
        )
        if not valid:
            raise ValueError(f"invalid exact-force reduced cache: {path}")
        if expected_targets is not None and not np.array_equal(targets, expected_targets):
            raise ValueError("all exact-force caches must use the same fixed A particles")
        return {
            "positions": positions,
            "velocities": velocities,
            "forces": forces,
            "target_indices": targets,
            "frame_time": frame_time,
            "trajectory_sha256": str(cache["trajectory_sha256"]),
        }


def load_or_extract_cage(
    trajectory_path: Path,
    cage_cache_path: Path,
    *,
    reduced: dict[str, np.ndarray | float | str],
    target_batch_size: int,
) -> dict[str, np.ndarray | float | str]:
    trajectory_hash = file_sha256(trajectory_path)
    targets = np.asarray(reduced["target_indices"], dtype=int)
    if trajectory_hash != str(reduced["trajectory_sha256"]):
        raise ValueError("full trajectory hash does not match the exact-force cache")
    if cage_cache_path.is_file():
        with np.load(cage_cache_path) as cache:
            cached_targets = np.asarray(cache["target_indices"], dtype=int)
            valid = (
                str(cache["trajectory_sha256"]) == trajectory_hash
                and np.array_equal(cached_targets, targets)
                and math.isclose(float(cache["frame_time"]), float(reduced["frame_time"]))
                and float(cache["thermodynamic_claim_allowed"]) == 0.0
            )
            if valid:
                relative_position = np.asarray(cache["relative_position"], dtype=float)
                relative_velocity = np.asarray(cache["relative_velocity"], dtype=float)
                jacobian_gram = np.asarray(cache["jacobian_gram"], dtype=float)
                support_count = np.asarray(cache["support_count"], dtype=int)
                expected = np.asarray(reduced["positions"]).shape
                if (
                    relative_position.shape == expected
                    and relative_velocity.shape == expected
                    and jacobian_gram.shape == (*expected[:2], 3, 3)
                    and support_count.shape == expected[:2]
                    and np.all(np.isfinite(relative_position))
                    and np.all(np.isfinite(relative_velocity))
                    and np.all(np.isfinite(jacobian_gram))
                ):
                    return {
                        "relative_position": relative_position,
                        "relative_velocity": relative_velocity,
                        "jacobian_gram": jacobian_gram,
                        "support_count": support_count,
                        "trajectory_sha256": trajectory_hash,
                    }

    trajectory = load_lammps_custom_trajectory(trajectory_path)
    if "velocities" not in trajectory:
        raise ValueError("full trajectory must contain particle velocities")
    timesteps = np.asarray(trajectory["timesteps"], dtype=int)
    intervals = np.diff(timesteps)
    if len(intervals) == 0 or not np.all(intervals == intervals[0]):
        raise ValueError("full trajectory must use a uniform saved-frame grid")
    frame_time = float(intervals[0]) * 0.001
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    velocities = np.asarray(trajectory["velocities"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    if (
        not math.isclose(frame_time, float(reduced["frame_time"]))
        or np.any(particle_types[targets] != 0)
        or positions.shape != velocities.shape
        or positions[:, targets].shape != np.asarray(reduced["positions"]).shape
        or not np.allclose(positions[:, targets], reduced["positions"], rtol=0.0, atol=1e-10)
        or not np.allclose(velocities[:, targets], reduced["velocities"], rtol=0.0, atol=1e-10)
    ):
        raise ValueError("full trajectory does not align with the exact-force reduced cache")

    relative_position = np.empty_like(np.asarray(reduced["positions"]), dtype=float)
    relative_velocity = np.empty_like(relative_position)
    jacobian_gram = np.empty((*relative_position.shape[:2], 3, 3), dtype=float)
    support_count = np.empty(relative_position.shape[:2], dtype=int)
    for frame in range(len(positions)):
        result = smooth_force_support_cage_batch(
            positions[frame],
            velocities=velocities[frame],
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=targets,
            target_batch_size=target_batch_size,
        )
        relative_position[frame] = result["relative_position"]
        relative_velocity[frame] = result["relative_velocity"]
        jacobian_gram[frame] = result["jacobian_gram"]
        support_count[frame] = result["support_count"]
    cage_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cage_cache_path,
        relative_position=relative_position,
        relative_velocity=relative_velocity,
        jacobian_gram=jacobian_gram,
        support_count=support_count,
        target_indices=targets,
        frame_time=np.asarray(frame_time),
        trajectory_sha256=np.asarray(trajectory_hash),
        thermodynamic_claim_allowed=np.asarray(0.0),
    )
    return {
        "relative_position": relative_position,
        "relative_velocity": relative_velocity,
        "jacobian_gram": jacobian_gram,
        "support_count": support_count,
        "trajectory_sha256": trajectory_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--cage-cache-directory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--target-seed", type=int, default=20260714)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--history-length", type=int, default=64)
    parser.add_argument("--mode-count", type=int, default=16)
    parser.add_argument("--simulation-count", type=int, default=4000)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 400])
    parser.add_argument("--wave-numbers", type=float, nargs="+", default=[1.0, 3.0, 7.25])
    parser.add_argument("--half-window", type=int, default=40)
    parser.add_argument("--phop-threshold", type=float, default=0.08)
    parser.add_argument("--maximum-singular-value", type=float, default=0.999)
    parser.add_argument("--fs-denominator-floor", type=float, default=0.05)
    parser.add_argument("--maximum-lag-correlation", type=float, default=0.70)
    parser.add_argument("--lag-improvement-fraction", type=float, default=0.80)
    args = parser.parse_args()

    if (
        args.expected_clone_count < 2
        or args.target_count < 1
        or args.target_batch_size < 1
        or args.history_length < 2
        or args.mode_count < 1
        or args.mode_count > args.history_length
        or args.simulation_count < 100
        or args.half_window < 1
        or args.phop_threshold <= 0.0
        or args.fs_denominator_floor <= 0.0
        or not 0.0 < args.lag_improvement_fraction < 1.0
    ):
        raise ValueError("invalid smooth-cage Hankel controls")
    lags = np.asarray(sorted(set(args.lags)), dtype=int)
    wave_numbers = np.asarray(args.wave_numbers, dtype=float)
    if np.any(lags < 1) or np.any(~np.isfinite(wave_numbers)):
        raise ValueError("lags and wave numbers must be finite and positive")

    manifest = json.loads((args.clone_directory / "manifest.json").read_text())
    if (
        int(manifest["clone_count"]) != args.expected_clone_count
        or manifest["dynamics"] != "nve_plus_langevin"
        or not bool(manifest["dump_velocity_force"])
        or bool(manifest["thermodynamic_claim_allowed"])
        or not math.isclose(float(manifest["saved_frame_interval_tau"]), 0.01)
        or float(manifest["duration_tau"]) < 10.0
    ):
        raise ValueError("manifest does not match the preregistered long KA protocol")
    trajectory_paths = sorted(
        path / "trajectory.lammpstrj"
        for path in args.clone_directory.glob("clone_*")
        if (path / "trajectory.lammpstrj").is_file()
    )
    if len(trajectory_paths) != args.expected_clone_count:
        raise ValueError("completed clone count does not match the manifest")

    clones: list[dict[str, np.ndarray | float | str]] = []
    targets: np.ndarray | None = None
    for clone_index, trajectory_path in enumerate(trajectory_paths, start=1):
        reduced = load_force_cache(
            args.cache_directory / f"clone_{clone_index:03d}_reduced.npz",
            target_count=args.target_count,
            target_seed=args.target_seed,
            expected_targets=targets,
        )
        if targets is None:
            targets = np.asarray(reduced["target_indices"], dtype=int)
        cage = load_or_extract_cage(
            trajectory_path,
            args.cage_cache_directory / f"clone_{clone_index:03d}_smooth_cage.npz",
            reduced=reduced,
            target_batch_size=args.target_batch_size,
        )
        clones.append({**reduced, **cage})
    frame_time = float(clones[0]["frame_time"])
    if int(lags[-1]) >= min(len(np.asarray(clone["positions"])) for clone in clones) - args.history_length:
        raise ValueError("maximum lag leaves no held displacement origins")

    model_definitions = (
        (f"hankel_slow_{args.mode_count}", False, False),
        (f"hankel_slow_{args.mode_count}_position", True, False),
        (f"hankel_slow_{args.mode_count}_position_velocity", True, True),
    )
    details: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for held_index, held in enumerate(clones):
        training = [clone for index, clone in enumerate(clones) if index != held_index]
        basis_fit = fit_force_hankel_basis(
            [np.asarray(clone["forces"]) for clone in training],
            history_length=args.history_length,
            mode_count=args.mode_count,
        )
        training_modes = [
            project_force_hankel(np.asarray(clone["forces"]), basis_fit) for clone in training
        ]
        held_modes = project_force_hankel(np.asarray(held["forces"]), basis_fit)
        kinematic = cage_kinematic_diagnostic(
            np.asarray(held["relative_position"]),
            np.asarray(held["relative_velocity"]),
            frame_time=frame_time,
        )
        gram = np.asarray(held["jacobian_gram"])
        gram_eigenvalues = np.linalg.eigvalsh(gram)
        maximum_gram_condition = float(
            np.max(gram_eigenvalues[:, :, -1] / gram_eigenvalues[:, :, 0])
        )
        for model_index, (model_name, include_position, include_cage_velocity) in enumerate(
            model_definitions
        ):
            training_states = [
                assemble_augmented_hankel_state(
                    np.asarray(clone["velocities"]),
                    modes,
                    relative_position=np.asarray(clone["relative_position"]),
                    relative_velocity=np.asarray(clone["relative_velocity"]),
                    include_position=include_position,
                    include_relative_velocity=include_cage_velocity,
                )
                for clone, modes in zip(training, training_modes, strict=True)
            ]
            held_state = assemble_augmented_hankel_state(
                np.asarray(held["velocities"]),
                held_modes,
                relative_position=np.asarray(held["relative_position"]),
                relative_velocity=np.asarray(held["relative_velocity"]),
                include_position=include_position,
                include_relative_velocity=include_cage_velocity,
            )
            model = fit_covariance_contracted_model(
                training_states,
                maximum_singular_value=args.maximum_singular_value,
            )
            diagnostic = heldout_state_diagnostics(model, held_state)
            lag_profile = heldout_residual_lag_profile(model, held_state)
            lag_by_mode = np.asarray(lag_profile["maximum_by_residual_mode"])
            resolved_lag = {
                "maximum_held_velocity_residual_lag_correlation": float(lag_by_mode[0]),
                "maximum_held_force_residual_lag_correlation": float(
                    np.max(lag_by_mode[1 : 1 + args.mode_count])
                ),
            }
            next_mode = 1 + args.mode_count
            if include_position:
                resolved_lag["maximum_held_cage_position_residual_lag_correlation"] = float(
                    lag_by_mode[next_mode]
                )
                next_mode += 1
            if include_cage_velocity:
                resolved_lag["maximum_held_cage_velocity_residual_lag_correlation"] = float(
                    lag_by_mode[next_mode]
                )
            predicted_displacements = simulate_slow_bath_displacements(
                model,
                step_count=int(lags[-1]),
                simulation_count=args.simulation_count,
                frame_time=frame_time,
                seed=args.target_seed + 1000 * (held_index + 1) + model_index,
            )
            held_positions = np.asarray(held["positions"])[args.history_length - 1 :]
            macro, model_curves = evaluate_displacements(
                held_positions,
                predicted_displacements,
                lags=lags,
                wave_numbers=wave_numbers,
                fs_denominator_floor=args.fs_denominator_floor,
            )
            observed_rate, observed_count = event_rate(
                held_positions,
                frame_time=frame_time,
                threshold=args.phop_threshold,
                half_window=args.half_window,
            )
            predicted_rate, predicted_count = event_rate(
                predicted_displacements,
                frame_time=frame_time,
                threshold=args.phop_threshold,
                half_window=args.half_window,
            )
            event_error = abs(predicted_rate / observed_rate - 1.0) if observed_rate > 0.0 else math.inf
            common = {
                "record": "held_clone",
                "model": model_name,
                "held_clone_index": float(held_index + 1),
                "history_length": float(args.history_length),
                "history_time_tau": float((args.history_length - 1) * frame_time),
                "slow_mode_count": float(args.mode_count),
                "state_mode_count": float(held_state.shape[2]),
                "include_smooth_cage_position": float(include_position),
                "include_smooth_cage_velocity": float(include_cage_velocity),
                "captured_force_history_variance_fraction": float(
                    basis_fit["captured_variance_fraction"]
                ),
                "trajectory_sha256": str(held["trajectory_sha256"]),
            }
            details.append(
                {
                    **common,
                    **diagnostic,
                    **resolved_lag,
                    **kinematic,
                    **macro,
                    "observed_event_rate": observed_rate,
                    "predicted_event_rate": predicted_rate,
                    "event_rate_relative_error": event_error,
                    "observed_event_count": float(observed_count),
                    "predicted_event_count": float(predicted_count),
                    "maximum_jacobian_gram_condition_number": maximum_gram_condition,
                    "minimum_support_count": float(np.min(held["support_count"])),
                    "spectral_radius": float(model["spectral_radius"]),
                    "stationary_covariance_relative_error": float(
                        model["stationary_covariance_relative_error"]
                    ),
                    "fit_parameters_from_macro_observables": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
            for curve in model_curves:
                curves.append({**common, **curve})

    metrics = (
        "heldout_state_r_squared",
        "heldout_velocity_r_squared",
        "maximum_held_residual_state_correlation",
        "maximum_held_residual_lag_correlation",
        "maximum_held_velocity_residual_lag_correlation",
        "maximum_held_force_residual_lag_correlation",
        "terminal_diffusion_relative_error",
        "multi_k_fs_max_relative_error",
        "ngp_max_absolute_error",
        "event_rate_relative_error",
        "normalized_trapezoidal_kinematic_error",
        "maximum_jacobian_gram_condition_number",
        "spectral_radius",
        "stationary_covariance_relative_error",
        "captured_force_history_variance_fraction",
    )
    summaries: list[dict[str, object]] = []
    for model_name, include_position, include_cage_velocity in model_definitions:
        rows = [row for row in details if row["model"] == model_name]
        summary: dict[str, object] = {
            "record": "aggregate_model",
            "model": model_name,
            "held_clone_count": float(len(rows)),
            "include_smooth_cage_position": float(include_position),
            "include_smooth_cage_velocity": float(include_cage_velocity),
            "fit_parameters_from_macro_observables": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_standard_error"] = standard_error(values)
            summary[f"maximum_{metric}"] = float(np.max(values))
        optional_metrics = []
        if include_position:
            optional_metrics.append(
                "maximum_held_cage_position_residual_lag_correlation"
            )
        if include_cage_velocity:
            optional_metrics.append(
                "maximum_held_cage_velocity_residual_lag_correlation"
            )
        for metric in optional_metrics:
            values = [float(row[metric]) for row in rows]
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_standard_error"] = standard_error(values)
            summary[f"maximum_{metric}"] = float(np.max(values))
        summaries.append(summary)

    baseline_name = model_definitions[0][0]
    primary_name = model_definitions[-1][0]
    baseline_rows = {
        int(float(row["held_clone_index"])): row
        for row in details
        if row["model"] == baseline_name
    }
    primary_rows = {
        int(float(row["held_clone_index"])): row
        for row in details
        if row["model"] == primary_name
    }
    baseline_summary = next(row for row in summaries if row["model"] == baseline_name)
    primary_summary = next(row for row in summaries if row["model"] == primary_name)
    every_fold_improves = all(
        float(primary_rows[index]["maximum_held_residual_lag_correlation"])
        < float(baseline_rows[index]["maximum_held_residual_lag_correlation"])
        for index in baseline_rows
    )
    baseline_lag = float(
        baseline_summary["maximum_maximum_held_residual_lag_correlation"]
    )
    primary_lag = float(
        primary_summary["maximum_maximum_held_residual_lag_correlation"]
    )
    numerical_gate = (
        float(primary_summary["maximum_spectral_radius"]) <= 1.0 + 1e-8
        and float(primary_summary["maximum_stationary_covariance_relative_error"]) <= 1e-8
    )
    residual_gate = (
        float(primary_summary["heldout_velocity_r_squared"]) >= 0.97
        and float(primary_summary["maximum_maximum_held_residual_state_correlation"]) <= 0.20
        and primary_lag <= args.maximum_lag_correlation
        and primary_lag <= args.lag_improvement_fraction * baseline_lag
        and every_fold_improves
    )
    macro_gate = (
        float(primary_summary["terminal_diffusion_relative_error"]) <= 0.20
        and float(primary_summary["multi_k_fs_max_relative_error"]) <= 0.20
        and float(primary_summary["ngp_max_absolute_error"]) <= 0.10
        and float(primary_summary["event_rate_relative_error"]) <= 0.20
    )
    allowed = numerical_gate and residual_gate and macro_gate
    summaries.append(
        {
            "record": "verdict",
            "model": primary_name,
            "held_clone_count": float(len(clones)),
            "integrity_gate_pass": 1.0,
            "numerical_gate_pass": float(numerical_gate),
            "residual_memory_gate_pass": float(residual_gate),
            "macro_event_gate_pass": float(macro_gate),
            "every_fold_lag_improves": float(every_fold_improves),
            "baseline_maximum_residual_lag_correlation": baseline_lag,
            "primary_maximum_residual_lag_correlation": primary_lag,
            "primary_to_baseline_lag_ratio": primary_lag / baseline_lag,
            "smooth_cage_hankel_bath_allowed": float(allowed),
            "autonomous_single_particle_gle_allowed": float(allowed),
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curves)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)


if __name__ == "__main__":
    main()
