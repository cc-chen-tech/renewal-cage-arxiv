#!/usr/bin/env python3
"""Test separate microscopic cage-center and relative drift-history baths."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
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
from analyze_ka_smooth_cage_hankel_bath import (  # noqa: E402
    load_force_cache,
    standard_error,
)
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_slow_force_bath import (  # noqa: E402
    fit_covariance_contracted_model,
    fit_force_hankel_basis,
    heldout_state_diagnostics,
    project_force_hankel,
    simulate_slow_bath_displacements,
)
from ka_smooth_cage import smooth_cage_projected_observables_batch  # noqa: E402
from ka_smooth_cage_bath import (  # noqa: E402
    heldout_linear_velocity_diagnostic,
    heldout_residual_lag_profile,
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


def conservative_rerun_input(
    *,
    parent_restart: Path,
    source_trajectory: Path,
    output_trajectory: Path,
    dump_interval_steps: int,
) -> str:
    """Return a force-only LAMMPS rerun that preserves x, images, and v."""

    return f"""units lj
atom_style atomic
read_restart {parent_restart}
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
thermo 1000000
dump conservative all custom {dump_interval_steps} {output_trajectory} id type x y z ix iy iz vx vy vz fx fy fz
dump_modify conservative sort id
rerun {source_trajectory} dump x y z ix iy iz vx vy vz box yes
"""


def load_smooth_cache(
    path: Path,
    *,
    reduced: dict[str, np.ndarray | float | str],
) -> dict[str, np.ndarray]:
    with np.load(path) as cache:
        targets = np.asarray(cache["target_indices"], dtype=int)
        relative_position = np.asarray(cache["relative_position"], dtype=float)
        relative_velocity = np.asarray(cache["relative_velocity"], dtype=float)
        valid = (
            str(cache["trajectory_sha256"]) == str(reduced["trajectory_sha256"])
            and np.array_equal(targets, reduced["target_indices"])
            and relative_position.shape == np.asarray(reduced["positions"]).shape
            and relative_velocity.shape == relative_position.shape
            and float(cache["thermodynamic_claim_allowed"]) == 0.0
        )
        if not valid:
            raise ValueError(f"invalid smooth-cage cache: {path}")
        return {
            "relative_position": relative_position,
            "relative_velocity": relative_velocity,
        }


def _drift_cache_payload(path: Path) -> dict[str, np.ndarray | float | str]:
    with np.load(path) as cache:
        keys = (
            "relative_position",
            "relative_velocity",
            "center_velocity",
            "relative_drift",
            "center_drift",
            "projected_force",
            "geometric_drift",
            "target_indices",
            "trajectory_sha256",
            "lammps_binary_sha256",
            "parent_restart_sha256",
            "directional_step",
            "maximum_force_cache_absolute_error",
            "force_cache_relative_rms_error",
            "force_cache_correlation",
            "maximum_noise_reconstruction_relative_error",
            "minimum_joint_noise_covariance_eigenvalue",
            "maximum_jacobian_gram_condition_number",
            "maximum_directional_step_relative_error",
            "geometric_to_projected_force_rms_ratio",
            "thermodynamic_claim_allowed",
        )
        return {key: np.asarray(cache[key]) for key in keys}


def load_or_extract_drift(
    source_trajectory: Path,
    *,
    reduced: dict[str, np.ndarray | float | str],
    smooth_cache_path: Path,
    drift_cache_path: Path,
    rerun_directory: Path,
    lammps_binary: Path,
    parent_restart: Path,
    friction: float,
    temperature: float,
    directional_step: float,
    sensitivity_steps: np.ndarray,
    target_batch_size: int,
    retain_rerun_dump: bool,
) -> dict[str, np.ndarray | float | str]:
    rerun_directory = rerun_directory.resolve()
    source_hash = file_sha256(source_trajectory)
    binary_hash = file_sha256(lammps_binary)
    restart_hash = file_sha256(parent_restart)
    targets = np.asarray(reduced["target_indices"], dtype=int)
    expected_shape = np.asarray(reduced["positions"]).shape
    if source_hash != str(reduced["trajectory_sha256"]):
        raise ValueError("source trajectory hash does not match exact-force cache")
    if drift_cache_path.is_file():
        cached = _drift_cache_payload(drift_cache_path)
        valid = (
            str(cached["trajectory_sha256"]) == source_hash
            and str(cached["lammps_binary_sha256"]) == binary_hash
            and str(cached["parent_restart_sha256"]) == restart_hash
            and np.array_equal(np.asarray(cached["target_indices"], dtype=int), targets)
            and math.isclose(float(cached["directional_step"]), directional_step)
            and float(cached["thermodynamic_claim_allowed"]) == 0.0
        )
        arrays = (
            "relative_position",
            "relative_velocity",
            "center_velocity",
            "relative_drift",
            "center_drift",
            "projected_force",
            "geometric_drift",
        )
        if valid and all(np.asarray(cached[key]).shape == expected_shape for key in arrays):
            return cached

    smooth = load_smooth_cache(smooth_cache_path, reduced=reduced)
    rerun_directory.mkdir(parents=True, exist_ok=True)
    clone_label = drift_cache_path.stem.replace("_decomposed_drift", "")
    rerun_dump = rerun_directory / f"{clone_label}_conservative.lammpstrj"
    rerun_input = rerun_directory / f"{clone_label}_conservative.in"
    rerun_log = rerun_directory / f"{clone_label}_conservative.log"
    dump_interval_steps = int(round(float(reduced["frame_time"]) / 0.001))
    if not rerun_dump.is_file():
        rerun_input.write_text(
            conservative_rerun_input(
                parent_restart=parent_restart.resolve(),
                source_trajectory=source_trajectory.resolve(),
                output_trajectory=rerun_dump.resolve(),
                dump_interval_steps=dump_interval_steps,
            )
        )
        subprocess.run(
            [
                str(lammps_binary),
                "-log",
                str(rerun_log.resolve()),
                "-screen",
                "none",
                "-in",
                str(rerun_input.resolve()),
            ],
            check=True,
            cwd=rerun_directory,
        )
    trajectory = load_lammps_custom_trajectory(rerun_dump)
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    velocities = np.asarray(trajectory["velocities"], dtype=float)
    conservative_force = np.asarray(trajectory["forces"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    reference_force = np.asarray(reduced["forces"])
    force_difference = conservative_force[:, targets] - reference_force
    target_force_error = float(np.max(np.abs(force_difference)))
    force_relative_rms_error = float(
        np.sqrt(np.mean(force_difference**2))
        / max(np.sqrt(np.mean(reference_force**2)), np.finfo(float).tiny)
    )
    force_correlation = float(
        np.corrcoef(conservative_force[:, targets].ravel(), reference_force.ravel())[0, 1]
    )
    if (
        positions.shape != velocities.shape
        or conservative_force.shape != positions.shape
        or positions[:, targets].shape != expected_shape
        or np.max(np.abs(velocities[:, targets] - np.asarray(reduced["velocities"]))) > 1e-10
        or np.max(np.abs(positions[:, targets] - np.asarray(reduced["positions"]))) > 5e-6
        # Sparse cutoff crossings can amplify text-coordinate rounding.  The
        # RMS and correlation gates distinguish those from a wrong force law.
        or target_force_error > 1e-1
        or force_relative_rms_error > 5e-5
        or force_correlation < 0.99999999
        or np.any(particle_types[targets] != 0)
    ):
        raise ValueError("conservative rerun does not align with the authoritative reduced cache")

    relative_position = np.empty(expected_shape, dtype=float)
    relative_velocity = np.empty_like(relative_position)
    center_velocity = np.empty_like(relative_position)
    relative_drift = np.empty_like(relative_position)
    center_drift = np.empty_like(relative_position)
    projected_force = np.empty_like(relative_position)
    geometric_drift = np.empty_like(relative_position)
    maximum_noise_error = 0.0
    minimum_joint_eigenvalue = math.inf
    maximum_gram_condition = 0.0
    maximum_step_error = 0.0
    sensitivity_frames = set(np.linspace(0, len(positions) - 1, 9, dtype=int))
    for frame in range(len(positions)):
        common = {
            "velocities": velocities[frame],
            "forces": conservative_force[frame],
            "particle_types": particle_types,
            "box_lengths": box_lengths,
            "target_indices": targets,
            "friction": friction,
            "temperature": temperature,
            "target_batch_size": target_batch_size,
        }
        result = smooth_cage_projected_observables_batch(
            positions[frame],
            directional_step=directional_step,
            **common,
        )
        relative_position[frame] = result["relative_position"]
        relative_velocity[frame] = result["relative_velocity"]
        center_velocity[frame] = result["center_velocity"]
        relative_drift[frame] = result["relative_drift"]
        center_drift[frame] = result["center_drift"]
        projected_force[frame] = result["projected_force"]
        geometric_drift[frame] = result["geometric_drift"]
        center_covariance = np.asarray(result["center_noise_covariance_rate"])
        relative_covariance = np.asarray(result["relative_noise_covariance_rate"])
        cross_covariance = np.asarray(result["center_relative_noise_covariance_rate"])
        recovered = (
            center_covariance
            + relative_covariance
            + cross_covariance
            + np.transpose(cross_covariance, (0, 2, 1))
        )
        expected_noise = 2.0 * friction * temperature * np.broadcast_to(
            np.eye(3), recovered.shape
        )
        maximum_noise_error = max(
            maximum_noise_error,
            float(np.linalg.norm(recovered - expected_noise) / np.linalg.norm(expected_noise)),
        )
        joint = np.concatenate(
            [
                np.concatenate([center_covariance, cross_covariance], axis=2),
                np.concatenate(
                    [np.transpose(cross_covariance, (0, 2, 1)), relative_covariance],
                    axis=2,
                ),
            ],
            axis=1,
        )
        minimum_joint_eigenvalue = min(
            minimum_joint_eigenvalue,
            float(np.min(np.linalg.eigvalsh(joint))),
        )
        gram_eigenvalues = np.linalg.eigvalsh(np.asarray(result["jacobian_gram"]))
        maximum_gram_condition = max(
            maximum_gram_condition,
            float(np.max(gram_eigenvalues[:, -1] / gram_eigenvalues[:, 0])),
        )
        if frame in sensitivity_frames:
            primary = np.asarray(result["relative_drift"])
            for step in sensitivity_steps:
                alternative = smooth_cage_projected_observables_batch(
                    positions[frame],
                    directional_step=float(step),
                    **common,
                )
                maximum_step_error = max(
                    maximum_step_error,
                    float(
                        np.linalg.norm(np.asarray(alternative["relative_drift"]) - primary)
                        / max(np.linalg.norm(primary), np.finfo(float).tiny)
                    ),
                )
    if (
        np.max(np.abs(relative_position - smooth["relative_position"])) > 2e-5
        or np.max(np.abs(relative_velocity - smooth["relative_velocity"])) > 2e-5
    ):
        raise ValueError("projected drift rerun does not reproduce the smooth-cage cache")
    geometric_ratio = float(
        np.sqrt(np.mean(geometric_drift**2))
        / max(np.sqrt(np.mean(projected_force**2)), np.finfo(float).tiny)
    )
    drift_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        drift_cache_path,
        relative_position=relative_position,
        relative_velocity=relative_velocity,
        center_velocity=center_velocity,
        relative_drift=relative_drift,
        center_drift=center_drift,
        projected_force=projected_force,
        geometric_drift=geometric_drift,
        target_indices=targets,
        trajectory_sha256=np.asarray(source_hash),
        lammps_binary_sha256=np.asarray(binary_hash),
        parent_restart_sha256=np.asarray(restart_hash),
        directional_step=np.asarray(directional_step),
        maximum_force_cache_absolute_error=np.asarray(target_force_error),
        force_cache_relative_rms_error=np.asarray(force_relative_rms_error),
        force_cache_correlation=np.asarray(force_correlation),
        maximum_noise_reconstruction_relative_error=np.asarray(maximum_noise_error),
        minimum_joint_noise_covariance_eigenvalue=np.asarray(minimum_joint_eigenvalue),
        maximum_jacobian_gram_condition_number=np.asarray(maximum_gram_condition),
        maximum_directional_step_relative_error=np.asarray(maximum_step_error),
        geometric_to_projected_force_rms_ratio=np.asarray(geometric_ratio),
        thermodynamic_claim_allowed=np.asarray(0.0),
    )
    if not retain_rerun_dump:
        rerun_dump.unlink(missing_ok=True)
        rerun_input.unlink(missing_ok=True)
        rerun_log.unlink(missing_ok=True)
    return _drift_cache_payload(drift_cache_path)


def assemble_raw_state(velocity: np.ndarray, modes: np.ndarray) -> np.ndarray:
    offset = len(velocity) - len(modes)
    return np.concatenate([velocity[offset:, :, None, :], modes], axis=2)


def assemble_split_state(
    clone: dict[str, np.ndarray | float | str],
    center_modes: np.ndarray,
    relative_modes: np.ndarray,
) -> np.ndarray:
    offset = len(np.asarray(clone["center_velocity"])) - len(center_modes)
    return np.concatenate(
        [
            np.asarray(clone["center_velocity"])[offset:, :, None, :],
            np.asarray(clone["relative_velocity"])[offset:, :, None, :],
            np.asarray(clone["relative_position"])[offset:, :, None, :],
            center_modes,
            relative_modes,
        ],
        axis=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--cage-cache-directory", type=Path, required=True)
    parser.add_argument("--drift-cache-directory", type=Path, required=True)
    parser.add_argument("--rerun-directory", type=Path, required=True)
    parser.add_argument("--lammps-binary", type=Path, required=True)
    parser.add_argument("--parent-restart", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-clone-count", type=int, default=4)
    parser.add_argument("--target-count", type=int, default=64)
    parser.add_argument("--target-seed", type=int, default=20260714)
    parser.add_argument("--target-batch-size", type=int, default=16)
    parser.add_argument("--history-length", type=int, default=64)
    parser.add_argument("--raw-mode-count", type=int, default=16)
    parser.add_argument("--split-mode-counts", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--primary-split-mode-count", type=int, default=8)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--directional-step", type=float, default=1e-5)
    parser.add_argument("--sensitivity-steps", type=float, nargs="+", default=[5e-6, 2e-5])
    parser.add_argument("--simulation-count", type=int, default=4000)
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 400])
    parser.add_argument("--wave-numbers", type=float, nargs="+", default=[1.0, 3.0, 7.25])
    parser.add_argument("--half-window", type=int, default=40)
    parser.add_argument("--phop-threshold", type=float, default=0.08)
    parser.add_argument("--maximum-singular-value", type=float, default=0.999)
    parser.add_argument("--fs-denominator-floor", type=float, default=0.05)
    parser.add_argument("--maximum-lag-correlation", type=float, default=0.70)
    parser.add_argument("--lag-improvement-fraction", type=float, default=0.80)
    parser.add_argument("--retain-rerun-dumps", action="store_true")
    args = parser.parse_args()

    split_counts = tuple(sorted(set(args.split_mode_counts)))
    sensitivity_steps = np.asarray(args.sensitivity_steps, dtype=float)
    if (
        args.expected_clone_count < 2
        or args.target_count < 1
        or args.target_batch_size < 1
        or args.history_length < 2
        or args.raw_mode_count < 1
        or not split_counts
        or args.primary_split_mode_count not in split_counts
        or any(count < 1 or count > args.history_length for count in split_counts)
        or args.friction <= 0.0
        or args.temperature <= 0.0
        or args.directional_step <= 0.0
        or np.any(sensitivity_steps <= 0.0)
        or args.simulation_count < 100
        or not 0.0 < args.lag_improvement_fraction < 1.0
    ):
        raise ValueError("invalid decomposed-drift controls")
    if not args.lammps_binary.is_file() or not args.parent_restart.is_file():
        raise ValueError("LAMMPS binary and parent restart must exist")
    lags = np.asarray(sorted(set(args.lags)), dtype=int)
    wave_numbers = np.asarray(args.wave_numbers, dtype=float)
    manifest = json.loads((args.clone_directory / "manifest.json").read_text())
    if (
        int(manifest["clone_count"]) != args.expected_clone_count
        or manifest["dynamics"] != "nve_plus_langevin"
        or bool(manifest["thermodynamic_claim_allowed"])
        or not math.isclose(float(manifest["saved_frame_interval_tau"]), 0.01)
        or not math.isclose(float(manifest["langevin_damping"]), args.friction)
        or not math.isclose(float(manifest["temperature"]), args.temperature)
    ):
        raise ValueError("manifest does not match the decomposed-drift protocol")
    trajectory_paths = sorted(
        path / "trajectory.lammpstrj"
        for path in args.clone_directory.glob("clone_*")
        if (path / "trajectory.lammpstrj").is_file()
    )
    if len(trajectory_paths) != args.expected_clone_count:
        raise ValueError("completed clone count does not match the manifest")

    clones: list[dict[str, np.ndarray | float | str]] = []
    targets: np.ndarray | None = None
    for clone_index, source_path in enumerate(trajectory_paths, start=1):
        reduced = load_force_cache(
            args.cache_directory / f"clone_{clone_index:03d}_reduced.npz",
            target_count=args.target_count,
            target_seed=args.target_seed,
            expected_targets=targets,
        )
        if targets is None:
            targets = np.asarray(reduced["target_indices"], dtype=int)
        drift = load_or_extract_drift(
            source_path,
            reduced=reduced,
            smooth_cache_path=args.cage_cache_directory
            / f"clone_{clone_index:03d}_smooth_cage.npz",
            drift_cache_path=args.drift_cache_directory
            / f"clone_{clone_index:03d}_decomposed_drift.npz",
            rerun_directory=args.rerun_directory,
            lammps_binary=args.lammps_binary.resolve(),
            parent_restart=args.parent_restart.resolve(),
            friction=args.friction,
            temperature=args.temperature,
            directional_step=args.directional_step,
            sensitivity_steps=sensitivity_steps,
            target_batch_size=args.target_batch_size,
            retain_rerun_dump=args.retain_rerun_dumps,
        )
        clones.append({**reduced, **drift})
    frame_time = float(clones[0]["frame_time"])

    details: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    model_names = [f"raw_hankel_{args.raw_mode_count}"] + [
        f"split_cage_drift_{count}" for count in split_counts
    ]
    for held_index, held in enumerate(clones):
        training_indices = [index for index in range(len(clones)) if index != held_index]
        total_basis = fit_force_hankel_basis(
            [np.asarray(clones[index]["forces"]) for index in training_indices],
            history_length=args.history_length,
            mode_count=args.raw_mode_count,
        )
        total_modes = [
            project_force_hankel(np.asarray(clone["forces"]), total_basis) for clone in clones
        ]
        candidates: list[
            tuple[str, list[np.ndarray], np.ndarray, np.ndarray, dict[str, float]]
        ] = []
        raw_states = [
            assemble_raw_state(np.asarray(clone["velocities"]), modes)
            for clone, modes in zip(clones, total_modes, strict=True)
        ]
        raw_weights = np.zeros(raw_states[0].shape[2], dtype=float)
        raw_weights[0] = 1.0
        candidates.append(
            (
                model_names[0],
                raw_states,
                raw_states[held_index],
                raw_weights,
                {
                    "captured_total_drift_variance_fraction": float(
                        total_basis["captured_variance_fraction"]
                    ),
                    "captured_center_drift_variance_fraction": math.nan,
                    "captured_relative_drift_variance_fraction": math.nan,
                    "split_mode_count": 0.0,
                },
            )
        )
        for split_count in split_counts:
            center_basis = fit_force_hankel_basis(
                [np.asarray(clones[index]["center_drift"]) for index in training_indices],
                history_length=args.history_length,
                mode_count=split_count,
            )
            relative_basis = fit_force_hankel_basis(
                [np.asarray(clones[index]["relative_drift"]) for index in training_indices],
                history_length=args.history_length,
                mode_count=split_count,
            )
            center_modes = [
                project_force_hankel(np.asarray(clone["center_drift"]), center_basis)
                for clone in clones
            ]
            relative_modes = [
                project_force_hankel(np.asarray(clone["relative_drift"]), relative_basis)
                for clone in clones
            ]
            states = [
                assemble_split_state(clone, center_mode, relative_mode)
                for clone, center_mode, relative_mode in zip(
                    clones, center_modes, relative_modes, strict=True
                )
            ]
            weights = np.zeros(states[0].shape[2], dtype=float)
            weights[:2] = 1.0
            candidates.append(
                (
                    f"split_cage_drift_{split_count}",
                    states,
                    states[held_index],
                    weights,
                    {
                        "captured_total_drift_variance_fraction": math.nan,
                        "captured_center_drift_variance_fraction": float(
                            center_basis["captured_variance_fraction"]
                        ),
                        "captured_relative_drift_variance_fraction": float(
                            relative_basis["captured_variance_fraction"]
                        ),
                        "split_mode_count": float(split_count),
                    },
                )
            )

        for model_index, (model_name, states, held_state, velocity_weights, capture) in enumerate(
            candidates
        ):
            model = fit_covariance_contracted_model(
                [states[index] for index in training_indices],
                maximum_singular_value=args.maximum_singular_value,
            )
            diagnostic = heldout_state_diagnostics(model, held_state)
            tagged = heldout_linear_velocity_diagnostic(
                model,
                held_state,
                velocity_weights=velocity_weights,
            )
            lag_profile = heldout_residual_lag_profile(model, held_state)
            lag_by_mode = np.asarray(lag_profile["maximum_by_residual_mode"])
            group_lag: dict[str, float]
            if model_name.startswith("raw_"):
                group_lag = {
                    "maximum_center_velocity_residual_lag_correlation": math.nan,
                    "maximum_relative_velocity_residual_lag_correlation": math.nan,
                    "maximum_relative_position_residual_lag_correlation": math.nan,
                    "maximum_center_drift_residual_lag_correlation": math.nan,
                    "maximum_relative_drift_residual_lag_correlation": float(
                        np.max(lag_by_mode[1:])
                    ),
                }
            else:
                split_count = int(capture["split_mode_count"])
                group_lag = {
                    "maximum_center_velocity_residual_lag_correlation": float(lag_by_mode[0]),
                    "maximum_relative_velocity_residual_lag_correlation": float(lag_by_mode[1]),
                    "maximum_relative_position_residual_lag_correlation": float(lag_by_mode[2]),
                    "maximum_center_drift_residual_lag_correlation": float(
                        np.max(lag_by_mode[3 : 3 + split_count])
                    ),
                    "maximum_relative_drift_residual_lag_correlation": float(
                        np.max(lag_by_mode[3 + split_count :])
                    ),
                }
            predicted_displacements = simulate_slow_bath_displacements(
                model,
                step_count=int(lags[-1]),
                simulation_count=args.simulation_count,
                frame_time=frame_time,
                seed=args.target_seed + 1000 * (held_index + 1) + model_index,
                velocity_weights=velocity_weights,
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
                "state_mode_count": float(held_state.shape[2]),
                **capture,
                "trajectory_sha256": str(held["trajectory_sha256"]),
            }
            details.append(
                {
                    **common,
                    **diagnostic,
                    **tagged,
                    **group_lag,
                    **macro,
                    "observed_event_rate": observed_rate,
                    "predicted_event_rate": predicted_rate,
                    "event_rate_relative_error": event_error,
                    "observed_event_count": float(observed_count),
                    "predicted_event_count": float(predicted_count),
                    "maximum_force_cache_absolute_error": float(
                        held["maximum_force_cache_absolute_error"]
                    ),
                    "force_cache_relative_rms_error": float(
                        held["force_cache_relative_rms_error"]
                    ),
                    "force_cache_correlation": float(held["force_cache_correlation"]),
                    "maximum_noise_reconstruction_relative_error": float(
                        held["maximum_noise_reconstruction_relative_error"]
                    ),
                    "minimum_joint_noise_covariance_eigenvalue": float(
                        held["minimum_joint_noise_covariance_eigenvalue"]
                    ),
                    "maximum_directional_step_relative_error": float(
                        held["maximum_directional_step_relative_error"]
                    ),
                    "geometric_to_projected_force_rms_ratio": float(
                        held["geometric_to_projected_force_rms_ratio"]
                    ),
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
        "heldout_linear_velocity_r_squared",
        "maximum_held_residual_state_correlation",
        "maximum_held_residual_lag_correlation",
        "terminal_diffusion_relative_error",
        "multi_k_fs_max_relative_error",
        "ngp_max_absolute_error",
        "event_rate_relative_error",
        "spectral_radius",
        "stationary_covariance_relative_error",
        "maximum_force_cache_absolute_error",
        "force_cache_relative_rms_error",
        "force_cache_correlation",
        "maximum_noise_reconstruction_relative_error",
        "minimum_joint_noise_covariance_eigenvalue",
        "maximum_directional_step_relative_error",
        "geometric_to_projected_force_rms_ratio",
    )
    summaries: list[dict[str, object]] = []
    for model_name in model_names:
        rows = [row for row in details if row["model"] == model_name]
        summary: dict[str, object] = {
            "record": "aggregate_model",
            "model": model_name,
            "held_clone_count": float(len(rows)),
            "state_mode_count": float(rows[0]["state_mode_count"]),
            "split_mode_count": float(rows[0]["split_mode_count"]),
            "fit_parameters_from_macro_observables": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_standard_error"] = standard_error(values)
            summary[f"maximum_{metric}"] = float(np.max(values))
            summary[f"minimum_{metric}"] = float(np.min(values))
        for metric in (
            "maximum_center_velocity_residual_lag_correlation",
            "maximum_relative_velocity_residual_lag_correlation",
            "maximum_relative_position_residual_lag_correlation",
            "maximum_center_drift_residual_lag_correlation",
            "maximum_relative_drift_residual_lag_correlation",
        ):
            values = np.asarray([float(row[metric]) for row in rows])
            if np.any(np.isfinite(values)):
                summary[metric] = float(np.nanmean(values))
                summary[f"maximum_{metric}"] = float(np.nanmax(values))
        summaries.append(summary)

    baseline_name = model_names[0]
    primary_name = f"split_cage_drift_{args.primary_split_mode_count}"
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
    integrity_gate = (
        float(primary_summary["maximum_maximum_force_cache_absolute_error"]) <= 1e-1
        and float(primary_summary["maximum_force_cache_relative_rms_error"]) <= 5e-5
        and float(primary_summary["minimum_force_cache_correlation"]) >= 0.99999999
        and float(primary_summary["maximum_maximum_noise_reconstruction_relative_error"])
        <= 1e-10
        and float(primary_summary["minimum_minimum_joint_noise_covariance_eigenvalue"])
        >= -1e-10
        and float(primary_summary["maximum_maximum_directional_step_relative_error"])
        <= 1e-3
    )
    numerical_gate = (
        float(primary_summary["maximum_spectral_radius"]) <= 1.0 + 1e-8
        and float(primary_summary["maximum_stationary_covariance_relative_error"]) <= 1e-8
    )
    residual_gate = (
        float(primary_summary["heldout_linear_velocity_r_squared"]) >= 0.97
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
    allowed = integrity_gate and numerical_gate and residual_gate and macro_gate
    summaries.append(
        {
            "record": "verdict",
            "model": primary_name,
            "held_clone_count": float(len(clones)),
            "integrity_gate_pass": float(integrity_gate),
            "numerical_gate_pass": float(numerical_gate),
            "residual_memory_gate_pass": float(residual_gate),
            "macro_event_gate_pass": float(macro_gate),
            "every_fold_lag_improves": float(every_fold_improves),
            "baseline_maximum_residual_lag_correlation": baseline_lag,
            "primary_maximum_residual_lag_correlation": primary_lag,
            "primary_to_baseline_lag_ratio": primary_lag / baseline_lag,
            "decomposed_cage_drift_bath_allowed": float(allowed),
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
