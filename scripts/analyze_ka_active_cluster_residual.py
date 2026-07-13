#!/usr/bin/env python3
"""Audit a finite active KA cluster by reconstructing its raw Langevin residual."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import active_cluster_langevin_residual  # noqa: E402
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def initial_cluster_indices(positions: np.ndarray, *, target: int, box_lengths: np.ndarray, radius: float) -> np.ndarray:
    displacement = positions[0] - positions[0, target]
    displacement -= box_lengths * np.rint(displacement / box_lengths)
    selected = np.flatnonzero(np.linalg.norm(displacement, axis=1) < radius)
    if target not in selected:
        selected = np.concatenate([np.array([target], dtype=int), selected])
    return np.unique(selected)


def cutoff_safe_external_indices(
    positions: np.ndarray,
    *,
    active_indices: np.ndarray,
    box_lengths: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Select every non-active particle entering an active-particle radius."""

    selected = np.zeros(positions.shape[1], dtype=bool)
    for frame in positions:
        displacement = frame[:, None, :] - frame[active_indices][None, :, :]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        selected |= np.any(np.sum(displacement**2, axis=2) < radius**2, axis=1)
    selected[active_indices] = False
    return np.flatnonzero(selected)


def concatenate_residuals(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    active_indices: np.ndarray,
    external_indices: np.ndarray,
    frame_time: float,
    friction: float,
    chunk_frames: int,
) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {"full_force": [], "omitted_force": [], "full_residual": [], "retained_residual": []}
    for start in range(0, len(positions) - 1, chunk_frames):
        stop = min(start + chunk_frames + 1, len(positions))
        result = active_cluster_langevin_residual(
            positions[start:stop],
            velocities[start:stop],
            particle_types=particle_types,
            box_lengths=box_lengths,
            active_indices=active_indices,
            external_indices=external_indices,
            frame_time=frame_time,
            friction=friction,
        )
        for key in outputs:
            value = np.asarray(result[key])
            # Adjacent chunks share a force-evaluation boundary frame, while
            # their interval residuals are already disjoint.
            if start and key in {"full_force", "omitted_force"}:
                value = value[1:]
            outputs[key].append(value)
    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}


def residual_metrics(residual: np.ndarray, *, frame_time: float, friction: float, temperature: float) -> dict[str, float]:
    variance = float(np.mean(residual**2))
    lag1 = (
        float(np.mean(residual[:-1] * residual[1:]) / variance)
        if len(residual) > 1 and variance > 0.0
        else math.nan
    )
    cross_terms: list[float] = []
    for first in range(residual.shape[1]):
        for second in range(first):
            cross_terms.append(float(np.mean(residual[:, first] * residual[:, second]) / variance) if variance > 0.0 else math.nan)
    return {
        "residual_mean": float(np.mean(residual)),
        "residual_rms": float(math.sqrt(variance)),
        "residual_fdt_variance_ratio": variance * frame_time / (2.0 * friction * temperature),
        "residual_lag1_component_correlation": lag1,
        "residual_mean_absolute_cross_particle_correlation": float(np.mean(np.abs(cross_terms))) if cross_terms else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clone_directory", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--active-radii", type=float, nargs="+", default=[1.0, 1.25, 1.5, 1.75])
    parser.add_argument("--external-radius", type=float, default=2.55)
    parser.add_argument("--frame-stop", type=int, default=401)
    parser.add_argument("--chunk-frames", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.58)
    parser.add_argument("--friction", type=float, default=1.0)
    args = parser.parse_args()
    if args.target_count < 1 or args.external_radius < 2.5 or args.frame_stop < 3 or args.chunk_frames < 1:
        raise ValueError("target count, cutoff-safe external radius, frame stop, and chunk size must be valid")
    if any(radius <= 0.0 for radius in args.active_radii) or args.temperature <= 0.0 or args.friction <= 0.0:
        raise ValueError("active radii, temperature, and friction must be positive")
    clones = sorted(path for path in args.clone_directory.glob("clone_*") if (path / "trajectory.lammpstrj").is_file())
    if len(clones) < 2:
        raise ValueError("at least two completed x/v/f clones are required")
    rows: list[dict[str, object]] = []
    targets: np.ndarray | None = None
    for clone_index, clone in enumerate(clones, start=1):
        trajectory = load_lammps_custom_trajectory(clone / "trajectory.lammpstrj")
        if "velocities" not in trajectory:
            raise ValueError("trajectory must contain velocities")
        positions = np.asarray(trajectory["unwrapped_positions"])
        velocities = np.asarray(trajectory["velocities"])
        if args.frame_stop > len(positions):
            raise ValueError("frame stop exceeds the available trajectory")
        positions = positions[: args.frame_stop]
        velocities = velocities[: args.frame_stop]
        timesteps = np.asarray(trajectory["timesteps"][: args.frame_stop])
        interval = np.diff(timesteps)
        if not len(interval) or not np.all(interval == interval[0]):
            raise ValueError("trajectory must have a uniform output interval")
        frame_time = float(interval[0]) * 0.001
        types = np.asarray(trajectory["particle_types"])
        box_lengths = np.asarray(trajectory["box_lengths"])
        if targets is None:
            targets = np.random.default_rng(args.seed).choice(
                np.flatnonzero(types == 0), size=args.target_count, replace=False
            )
        for target in targets:
            for radius in args.active_radii:
                active = initial_cluster_indices(positions, target=int(target), box_lengths=box_lengths, radius=radius)
                external = cutoff_safe_external_indices(
                    positions, active_indices=active, box_lengths=box_lengths, radius=args.external_radius
                )
                cutoff_safe = concatenate_residuals(
                    positions,
                    velocities,
                    particle_types=types,
                    box_lengths=box_lengths,
                    active_indices=active,
                    external_indices=external,
                    frame_time=frame_time,
                    friction=args.friction,
                    chunk_frames=args.chunk_frames,
                )
                active_only = concatenate_residuals(
                    positions,
                    velocities,
                    particle_types=types,
                    box_lengths=box_lengths,
                    active_indices=active,
                    external_indices=np.empty(0, dtype=int),
                    frame_time=frame_time,
                    friction=args.friction,
                    chunk_frames=args.chunk_frames,
                )
                full_force_rms = float(np.sqrt(np.mean(cutoff_safe["full_force"] ** 2)))
                cutoff_omitted_rms = float(np.sqrt(np.mean(cutoff_safe["omitted_force"] ** 2)))
                active_omitted_rms = float(np.sqrt(np.mean(active_only["omitted_force"] ** 2)))
                residual_difference_rms = float(
                    np.sqrt(np.mean((cutoff_safe["retained_residual"] - cutoff_safe["full_residual"]) ** 2))
                )
                rows.append(
                    {
                        "clone_index": float(clone_index),
                        "target_particle_index": float(target),
                        "active_radius": radius,
                        "active_cluster_size": float(len(active)),
                        "cutoff_safe_external_size": float(len(external)),
                        "frame_time_tau": frame_time,
                        "time_window_tau": (len(positions) - 1) * frame_time,
                        "full_force_rms": full_force_rms,
                        "cutoff_safe_omitted_force_relative_rms": cutoff_omitted_rms / max(full_force_rms, 1e-30),
                        "active_only_omitted_force_relative_rms": active_omitted_rms / max(full_force_rms, 1e-30),
                        "cutoff_safe_residual_difference_relative_rms": residual_difference_rms / max(float(np.sqrt(np.mean(cutoff_safe["full_residual"] ** 2))), 1e-30),
                        **{f"full_{key}": value for key, value in residual_metrics(cutoff_safe["full_residual"], frame_time=frame_time, friction=args.friction, temperature=args.temperature).items()},
                        **{f"active_only_{key}": value for key, value in residual_metrics(active_only["retained_residual"], frame_time=frame_time, friction=args.friction, temperature=args.temperature).items()},
                        "coordinate_definition": "raw_KA_active_particle_phase_space",
                        "force_partition": "active_active_plus_cutoff_safe_recorded_external_bath",
                        "residual_definition": "Delta_v/dt-mean_retained_force+gamma_mean_v",
                        "model": "trajectory_reconstructed_finite_active_cluster_Langevin",
                        "fit_parameters_from_macro_observables": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
    summary: list[dict[str, object]] = []
    for radius in args.active_radii:
        matching = [row for row in rows if float(row["active_radius"]) == radius]
        output: dict[str, object] = {
            "active_radius": radius,
            "clone_count": float(len({float(row["clone_index"]) for row in matching})),
            "target_count": float(args.target_count),
            "time_window_tau": float(matching[0]["time_window_tau"]),
            "external_radius": args.external_radius,
            "model": "trajectory_reconstructed_finite_active_cluster_Langevin",
            "fit_parameters_from_macro_observables": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in (
            "active_cluster_size",
            "cutoff_safe_external_size",
            "cutoff_safe_omitted_force_relative_rms",
            "active_only_omitted_force_relative_rms",
            "cutoff_safe_residual_difference_relative_rms",
            "full_residual_fdt_variance_ratio",
            "full_residual_lag1_component_correlation",
            "full_residual_mean_absolute_cross_particle_correlation",
            "active_only_residual_lag1_component_correlation",
        ):
            values = np.asarray([float(row[metric]) for row in matching])
            output[metric] = float(np.mean(values))
            output[f"{metric}_standard_error"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        output["cutoff_safe_force_partition_allowed"] = float(
            output["cutoff_safe_omitted_force_relative_rms"] <= 1e-10
            and output["cutoff_safe_residual_difference_relative_rms"] <= 1e-10
        )
        summary.append(output)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_details.csv"), rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)


if __name__ == "__main__":
    main()
