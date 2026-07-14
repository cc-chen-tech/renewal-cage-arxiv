#!/usr/bin/env python3
"""Test a stable slow bath extracted from exact KA pair-force histories."""

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

from ka_local_cage import ka_lj_force_and_isotropic_curvature  # noqa: E402
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_slow_force_bath import (  # noqa: E402
    assemble_force_hankel,
    fit_covariance_contracted_model,
    fit_force_hankel_basis,
    heldout_state_diagnostics,
    project_force_hankel,
    simulate_slow_bath_displacements,
)
from renewal_cage import extract_nonrecrossing_phop_events  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def conservative_force_series(
    positions: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    target_indices: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            ka_lj_force_and_isotropic_curvature(
                frame,
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=target_indices,
            )[0]
            for frame in positions
        ]
    )


def _cache_path(cache_directory: Path, clone_index: int) -> Path:
    return cache_directory / f"clone_{clone_index:03d}_reduced.npz"


def load_or_reduce_clone(
    trajectory_path: Path,
    *,
    clone_index: int,
    cache_directory: Path,
    target_count: int,
    target_seed: int,
    expected_target_indices: np.ndarray | None,
) -> dict[str, np.ndarray | float | str]:
    trajectory_hash = file_sha256(trajectory_path)
    cache_path = _cache_path(cache_directory, clone_index)
    if cache_path.is_file():
        with np.load(cache_path) as cache:
            cached_target = np.asarray(cache["target_indices"], dtype=int)
            valid = (
                str(cache["trajectory_sha256"]) == trajectory_hash
                and int(cache["target_count"]) == target_count
                and int(cache["target_seed"]) == target_seed
                and (expected_target_indices is None or np.array_equal(cached_target, expected_target_indices))
            )
            if valid:
                return {
                    "positions": np.asarray(cache["positions"]),
                    "velocities": np.asarray(cache["velocities"]),
                    "forces": np.asarray(cache["forces"]),
                    "target_indices": cached_target,
                    "frame_time": float(cache["frame_time"]),
                    "trajectory_sha256": trajectory_hash,
                }

    trajectory = load_lammps_custom_trajectory(trajectory_path)
    if "velocities" not in trajectory:
        raise ValueError("trajectory must contain particle velocities")
    timesteps = np.asarray(trajectory["timesteps"], dtype=int)
    intervals = np.diff(timesteps)
    if len(intervals) == 0 or not np.all(intervals == intervals[0]):
        raise ValueError("trajectory must use a uniform saved-frame interval")
    frame_time = float(intervals[0]) * 0.001
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    if expected_target_indices is None:
        candidates = np.flatnonzero(particle_types == 0)
        if target_count > len(candidates):
            raise ValueError("target_count exceeds the number of A particles")
        target_indices = np.random.default_rng(target_seed).choice(
            candidates,
            size=target_count,
            replace=False,
        )
        target_indices.sort()
    else:
        target_indices = np.asarray(expected_target_indices, dtype=int)
        if np.any(particle_types[target_indices] != 0):
            raise ValueError("fixed target indices must remain A particles")
    all_positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    positions = all_positions[:, target_indices]
    velocities = np.asarray(trajectory["velocities"], dtype=float)[:, target_indices]
    forces = conservative_force_series(
        all_positions,
        particle_types=particle_types,
        box_lengths=np.asarray(trajectory["box_lengths"], dtype=float),
        target_indices=target_indices,
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        positions=positions,
        velocities=velocities,
        forces=forces,
        target_indices=target_indices,
        frame_time=np.asarray(frame_time),
        trajectory_sha256=np.asarray(trajectory_hash),
        target_count=np.asarray(target_count),
        target_seed=np.asarray(target_seed),
        thermodynamic_claim_allowed=np.asarray(0.0),
    )
    return {
        "positions": positions,
        "velocities": velocities,
        "forces": forces,
        "target_indices": target_indices,
        "frame_time": frame_time,
        "trajectory_sha256": trajectory_hash,
    }


def ngp(vectors: np.ndarray) -> float:
    squared = np.sum(np.asarray(vectors, dtype=float) ** 2, axis=-1)
    second = float(np.mean(squared))
    if second <= 0.0:
        return 0.0
    return 3.0 * float(np.mean(squared**2)) / (5.0 * second**2) - 1.0


def evaluate_displacements(
    observed_positions: np.ndarray,
    predicted_displacements: np.ndarray,
    *,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    fs_denominator_floor: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    curves: list[dict[str, float]] = []
    fs_errors: list[float] = []
    ngp_errors: list[float] = []
    terminal_msd_error = math.nan
    for lag in lags:
        lag = int(lag)
        observed = observed_positions[lag:] - observed_positions[:-lag]
        observed = observed.reshape(-1, 3)
        predicted = predicted_displacements[lag]
        observed_msd = float(np.mean(np.sum(observed**2, axis=1)))
        predicted_msd = float(np.mean(np.sum(predicted**2, axis=1)))
        observed_ngp = ngp(observed)
        predicted_ngp = ngp(predicted)
        ngp_error = abs(predicted_ngp - observed_ngp)
        ngp_errors.append(ngp_error)
        if lag == int(lags[-1]):
            terminal_msd_error = abs(predicted_msd / observed_msd - 1.0)
        for wave_number in wave_numbers:
            observed_fs = float(np.mean(np.cos(float(wave_number) * observed)))
            predicted_fs = float(np.mean(np.cos(float(wave_number) * predicted)))
            fs_error = abs(predicted_fs - observed_fs) / max(abs(observed_fs), fs_denominator_floor)
            fs_errors.append(fs_error)
            curves.append(
                {
                    "lag_frames": float(lag),
                    "observed_msd": observed_msd,
                    "predicted_msd": predicted_msd,
                    "observed_ngp": observed_ngp,
                    "predicted_ngp": predicted_ngp,
                    "wave_number": float(wave_number),
                    "observed_fs": observed_fs,
                    "predicted_fs": predicted_fs,
                    "fs_scaled_absolute_error": fs_error,
                }
            )
    return (
        {
            "terminal_diffusion_relative_error": float(terminal_msd_error),
            "multi_k_fs_max_relative_error": float(max(fs_errors)),
            "ngp_max_absolute_error": float(max(ngp_errors)),
        },
        curves,
    )


def event_rate(
    positions: np.ndarray,
    *,
    frame_time: float,
    threshold: float,
    half_window: int,
) -> tuple[float, int]:
    events = extract_nonrecrossing_phop_events(
        positions,
        threshold=threshold,
        half_window=half_window,
        recrossing_radius=math.sqrt(threshold),
    )
    exposure = (len(positions) - 2 * half_window + 1) * frame_time * positions.shape[1]
    return float(len(events["time"]) / exposure), int(len(events["time"]))


def standard_error(values: list[float]) -> float:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--target-seed", type=int, default=20260714)
    parser.add_argument("--history-length", type=int, default=64)
    parser.add_argument("--primary-mode-count", type=int, default=8)
    parser.add_argument("--mode-counts", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--simulation-count", type=int, default=4000)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 400])
    parser.add_argument("--wave-numbers", type=float, nargs="+", default=[1.0, 3.0, 7.25])
    parser.add_argument("--half-window", type=int, default=40)
    parser.add_argument("--phop-threshold", type=float, default=0.08)
    parser.add_argument("--maximum-singular-value", type=float, default=0.999)
    parser.add_argument("--fs-denominator-floor", type=float, default=0.05)
    args = parser.parse_args()

    if (
        args.expected_clone_count < 2
        or args.target_count < 1
        or args.history_length < 2
        or args.primary_mode_count not in args.mode_counts
        or any(mode < 1 or mode > args.history_length for mode in args.mode_counts)
        or args.simulation_count < 100
        or args.half_window < 1
        or args.phop_threshold <= 0.0
        or args.fs_denominator_floor <= 0.0
    ):
        raise ValueError("slow-bath dimensions, sampling, and event controls must be valid")
    lags = np.asarray(sorted(set(args.lags)), dtype=int)
    if np.any(lags < 1):
        raise ValueError("lags must be positive")
    wave_numbers = np.asarray(args.wave_numbers, dtype=float)

    manifest_path = args.clone_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        int(manifest["clone_count"]) != args.expected_clone_count
        or manifest["dynamics"] != "nve_plus_langevin"
        or not bool(manifest["dump_velocity_force"])
        or bool(manifest["thermodynamic_claim_allowed"])
        or not math.isclose(float(manifest["saved_frame_interval_tau"]), 0.01)
        or float(manifest["duration_tau"]) < 10.0
    ):
        raise ValueError("manifest does not match the preregistered long KA protocol")
    clone_paths = sorted(
        path / "trajectory.lammpstrj"
        for path in args.clone_directory.glob("clone_*")
        if (path / "trajectory.lammpstrj").is_file()
    )
    if len(clone_paths) != args.expected_clone_count:
        raise ValueError("completed clone count does not match the manifest")

    reduced: list[dict[str, np.ndarray | float | str]] = []
    target_indices: np.ndarray | None = None
    for clone_index, trajectory_path in enumerate(clone_paths, start=1):
        clone = load_or_reduce_clone(
            trajectory_path,
            clone_index=clone_index,
            cache_directory=args.cache_directory,
            target_count=args.target_count,
            target_seed=args.target_seed,
            expected_target_indices=target_indices,
        )
        if target_indices is None:
            target_indices = np.asarray(clone["target_indices"], dtype=int)
        reduced.append(clone)
    frame_times = {float(clone["frame_time"]) for clone in reduced}
    if len(frame_times) != 1 or not math.isclose(next(iter(frame_times)), 0.01):
        raise ValueError("reduced clones must share the preregistered 0.01 tau grid")
    frame_time = next(iter(frame_times))
    if int(lags[-1]) >= min(len(np.asarray(clone["positions"])) for clone in reduced) - args.history_length:
        raise ValueError("maximum lag leaves no held displacement origins")

    model_definitions = [("raw_force_delay_2", 0)] + [
        (f"hankel_slow_{mode_count}", mode_count) for mode_count in args.mode_counts
    ]
    details: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for held_index in range(len(reduced)):
        training = [clone for index, clone in enumerate(reduced) if index != held_index]
        held = reduced[held_index]
        for model_index, (model_name, mode_count) in enumerate(model_definitions):
            if mode_count == 0:
                history_length = 3
                basis_fit = None
                training_modes = [
                    np.transpose(
                        assemble_force_hankel(np.asarray(clone["forces"]), history_length=history_length),
                        (0, 1, 3, 2),
                    )
                    for clone in training
                ]
                held_modes = np.transpose(
                    assemble_force_hankel(np.asarray(held["forces"]), history_length=history_length),
                    (0, 1, 3, 2),
                )
                captured_variance = 1.0
            else:
                history_length = args.history_length
                basis_fit = fit_force_hankel_basis(
                    [np.asarray(clone["forces"]) for clone in training],
                    history_length=history_length,
                    mode_count=mode_count,
                )
                training_modes = [project_force_hankel(np.asarray(clone["forces"]), basis_fit) for clone in training]
                held_modes = project_force_hankel(np.asarray(held["forces"]), basis_fit)
                captured_variance = float(basis_fit["captured_variance_fraction"])
            training_states = [
                np.concatenate(
                    [np.asarray(clone["velocities"])[history_length - 1 :, :, None, :], modes],
                    axis=2,
                )
                for clone, modes in zip(training, training_modes, strict=True)
            ]
            held_state = np.concatenate(
                [np.asarray(held["velocities"])[history_length - 1 :, :, None, :], held_modes],
                axis=2,
            )
            held_positions = np.asarray(held["positions"])[history_length - 1 :]
            model = fit_covariance_contracted_model(
                training_states,
                maximum_singular_value=args.maximum_singular_value,
            )
            diagnostic = heldout_state_diagnostics(model, held_state)
            predicted_displacements = simulate_slow_bath_displacements(
                model,
                step_count=int(lags[-1]),
                simulation_count=args.simulation_count,
                frame_time=frame_time,
                seed=args.target_seed + 1000 * (held_index + 1) + model_index,
            )
            macro, curves = evaluate_displacements(
                held_positions,
                predicted_displacements,
                lags=lags,
                wave_numbers=wave_numbers,
                fs_denominator_floor=args.fs_denominator_floor,
            )
            observed_rate, observed_event_count = event_rate(
                held_positions,
                frame_time=frame_time,
                threshold=args.phop_threshold,
                half_window=args.half_window,
            )
            predicted_rate, predicted_event_count = event_rate(
                predicted_displacements,
                frame_time=frame_time,
                threshold=args.phop_threshold,
                half_window=args.half_window,
            )
            event_rate_error = (
                abs(predicted_rate / observed_rate - 1.0) if observed_rate > 0.0 else math.inf
            )
            common = {
                "model": model_name,
                "held_clone_index": float(held_index + 1),
                "history_length": float(history_length),
                "history_time_tau": float((history_length - 1) * frame_time),
                "slow_mode_count": float(mode_count),
                "state_mode_count": float(held_state.shape[2]),
                "captured_force_history_variance_fraction": captured_variance,
                "trajectory_sha256": str(held["trajectory_sha256"]),
            }
            details.append(
                {
                    "record": "held_clone",
                    **common,
                    **diagnostic,
                    **macro,
                    "observed_event_rate": observed_rate,
                    "predicted_event_rate": predicted_rate,
                    "event_rate_relative_error": event_rate_error,
                    "observed_event_count": float(observed_event_count),
                    "predicted_event_count": float(predicted_event_count),
                    "spectral_radius": float(model["spectral_radius"]),
                    "maximum_unclipped_singular_value": float(model["maximum_unclipped_singular_value"]),
                    "maximum_clipped_singular_value": float(model["maximum_clipped_singular_value"]),
                    "stationary_covariance_relative_error": float(model["stationary_covariance_relative_error"]),
                    "fit_parameters_from_macro_observables": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
            for curve in curves:
                curve_rows.append({"record": "held_curve", **common, **curve})

    summary_rows: list[dict[str, object]] = []
    metrics = (
        "heldout_state_r_squared",
        "heldout_velocity_r_squared",
        "maximum_held_residual_state_correlation",
        "maximum_held_residual_lag_correlation",
        "terminal_diffusion_relative_error",
        "multi_k_fs_max_relative_error",
        "ngp_max_absolute_error",
        "event_rate_relative_error",
        "spectral_radius",
        "stationary_covariance_relative_error",
        "captured_force_history_variance_fraction",
    )
    for model_name, mode_count in model_definitions:
        rows = [row for row in details if row["model"] == model_name]
        summary: dict[str, object] = {
            "record": "aggregate_model",
            "model": model_name,
            "held_clone_count": float(len(rows)),
            "slow_mode_count": float(mode_count),
            "fit_parameters_from_macro_observables": 0.0,
            "state_dependent_memory_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_standard_error"] = standard_error(values)
            summary[f"maximum_{metric}"] = float(np.max(values))
        summary_rows.append(summary)

    primary = next(row for row in summary_rows if int(row["slow_mode_count"]) == args.primary_mode_count)
    numerical_gate = (
        float(primary["maximum_spectral_radius"]) <= 1.0 + 1.0e-8
        and float(primary["maximum_stationary_covariance_relative_error"]) <= 1.0e-8
    )
    orthogonality_gate = (
        float(primary["maximum_maximum_held_residual_state_correlation"]) <= 0.20
        and float(primary["maximum_maximum_held_residual_lag_correlation"]) <= 0.20
    )
    macro_gate = (
        float(primary["terminal_diffusion_relative_error"]) <= 0.20
        and float(primary["multi_k_fs_max_relative_error"]) <= 0.20
        and float(primary["ngp_max_absolute_error"]) <= 0.10
        and float(primary["event_rate_relative_error"]) <= 0.20
    )
    slow_bath_allowed = numerical_gate and orthogonality_gate and macro_gate
    summary_rows.append(
        {
            "record": "verdict",
            "model": f"hankel_slow_{args.primary_mode_count}",
            "held_clone_count": float(len(reduced)),
            "slow_mode_count": float(args.primary_mode_count),
            "integrity_gate_pass": 1.0,
            "numerical_gate_pass": float(numerical_gate),
            "orthogonality_gate_pass": float(orthogonality_gate),
            "macro_event_gate_pass": float(macro_gate),
            "hankel_slow_force_bath_allowed": float(slow_bath_allowed),
            "state_dependent_memory_allowed": 0.0,
            "complete_event_clock_closure_allowed": 0.0,
            "kramers_escape_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), details)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)


if __name__ == "__main__":
    main()
