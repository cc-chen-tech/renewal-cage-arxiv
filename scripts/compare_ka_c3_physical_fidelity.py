#!/usr/bin/env python3
"""Compare hard-cutoff and C3-switched KA short-time physical observables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_local_cage import ka_lj_force_and_isotropic_curvature  # noqa: E402
from ka_replicates import load_lammps_custom_trajectory  # noqa: E402


SIGMA = np.asarray([[1.0, 0.8], [0.8, 0.88]])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scaled_pair_histogram(
    positions: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    *,
    edges: np.ndarray,
    chunk_size: int = 128,
) -> np.ndarray:
    """Return an all-pair histogram of scaled separation `r/sigma_ij`."""

    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    particle_count = len(positions)
    all_indices = np.arange(particle_count)
    for start in range(0, particle_count, chunk_size):
        stop = min(start + chunk_size, particle_count)
        displacement = positions[start:stop, None, :] - positions[None, :, :]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        distance = np.linalg.norm(displacement, axis=2)
        pair_sigma = SIGMA[particle_types[start:stop, None], particle_types[None, :]]
        scaled = distance / pair_sigma
        upper_triangle = all_indices[None, :] > np.arange(start, stop)[:, None]
        values = scaled[upper_triangle & (scaled >= edges[0]) & (scaled < edges[-1])]
        counts += np.histogram(values, bins=edges)[0]
    return counts


def full_force_curvature(
    positions: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
    *,
    potential_protocol: str,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    force_blocks = []
    curvature_blocks = []
    for start in range(0, len(positions), chunk_size):
        target = np.arange(start, min(start + chunk_size, len(positions)))
        force, curvature = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            potential_protocol=potential_protocol,
        )
        force_blocks.append(force)
        curvature_blocks.append(curvature)
    return np.vstack(force_blocks), np.concatenate(curvature_blocks)


def tagged_msd(trajectory: dict[str, np.ndarray]) -> np.ndarray:
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    displacement = positions[:, particle_types == 0] - positions[0, particle_types == 0]
    return np.mean(np.sum(displacement**2, axis=2), axis=1)


def relative_difference(comparison: float, reference: float) -> float:
    return (comparison - reference) / abs(reference) if reference != 0.0 else float("nan")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-trajectory", type=Path, required=True)
    parser.add_argument("--c3-trajectory", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--curve-output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.hard_trajectory, args.c3_trajectory):
        if not path.is_file():
            raise ValueError(f"trajectory does not exist: {path}")
    hard = load_lammps_custom_trajectory(args.hard_trajectory)
    c3 = load_lammps_custom_trajectory(args.c3_trajectory)
    for key in ("timesteps", "particle_types", "box_lengths"):
        if not np.array_equal(hard[key], c3[key]):
            raise ValueError(f"hard and C3 trajectories disagree in {key}")
    if len(hard["timesteps"]) < 2:
        raise ValueError("physical fidelity requires a multi-frame trajectory")

    particle_types = np.asarray(hard["particle_types"], dtype=int)
    box_lengths = np.asarray(hard["box_lengths"], dtype=float)
    hard_initial = np.asarray(hard["wrapped_positions"][0], dtype=float)
    c3_initial = np.asarray(c3["wrapped_positions"][0], dtype=float)
    edges = np.linspace(0.8, 2.5, 86)
    hard_counts = scaled_pair_histogram(hard_initial, particle_types, box_lengths, edges=edges)
    c3_counts = scaled_pair_histogram(c3_initial, particle_types, box_lengths, edges=edges)
    hard_probability = hard_counts / np.sum(hard_counts)
    c3_probability = c3_counts / np.sum(c3_counts)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hard_force, hard_curvature = full_force_curvature(
        hard_initial,
        particle_types,
        box_lengths,
        potential_protocol="ka_lj_cut",
    )
    c3_force, c3_curvature = full_force_curvature(
        c3_initial,
        particle_types,
        box_lengths,
        potential_protocol="ka_lj_c3_switch",
    )
    hard_force_norm = np.linalg.norm(hard_force, axis=1)
    c3_force_norm = np.linalg.norm(c3_force, axis=1)
    hard_msd = tagged_msd(hard)
    c3_msd = tagged_msd(c3)
    times = (np.asarray(hard["timesteps"], dtype=float) - float(hard["timesteps"][0])) * 0.001
    positive_time = times >= 0.01
    msd_relative = np.abs(c3_msd[positive_time] - hard_msd[positive_time]) / np.maximum(
        hard_msd[positive_time],
        1e-15,
    )

    summary = {
        "particle_count": len(particle_types),
        "frame_count": len(times),
        "maximum_time_tau": float(times[-1]),
        "hard_trajectory_sha256": file_sha256(args.hard_trajectory),
        "c3_trajectory_sha256": file_sha256(args.c3_trajectory),
        "scaled_pair_histogram_total_variation": float(0.5 * np.sum(np.abs(c3_probability - hard_probability))),
        "scaled_pair_histogram_wasserstein_bin_approximation": float(
            np.sum(np.abs(np.cumsum(c3_probability) - np.cumsum(hard_probability))) * (edges[1] - edges[0])
        ),
        "hard_force_norm_mean": float(np.mean(hard_force_norm)),
        "c3_force_norm_mean": float(np.mean(c3_force_norm)),
        "force_norm_mean_relative_difference": relative_difference(
            float(np.mean(c3_force_norm)),
            float(np.mean(hard_force_norm)),
        ),
        "hard_force_norm_p95": float(np.quantile(hard_force_norm, 0.95)),
        "c3_force_norm_p95": float(np.quantile(c3_force_norm, 0.95)),
        "hard_curvature_mean": float(np.mean(hard_curvature)),
        "c3_curvature_mean": float(np.mean(c3_curvature)),
        "curvature_mean_relative_difference": relative_difference(
            float(np.mean(c3_curvature)),
            float(np.mean(hard_curvature)),
        ),
        "hard_curvature_p95": float(np.quantile(hard_curvature, 0.95)),
        "c3_curvature_p95": float(np.quantile(c3_curvature, 0.95)),
        "hard_msd_final": float(hard_msd[-1]),
        "c3_msd_final": float(c3_msd[-1]),
        "msd_final_relative_difference": relative_difference(float(c3_msd[-1]), float(hard_msd[-1])),
        "maximum_msd_relative_difference_after_0p01_tau": float(np.max(msd_relative)),
        "uncertainty_scope": "single_audit_path_particle_distribution_descriptive_only",
        "equilibrium_glass_claim_allowed": False,
        "thermodynamic_claim_allowed": False,
    }
    write_rows(args.summary_output, [summary])

    curve_rows: list[dict[str, object]] = []
    for center, hard_value, c3_value in zip(centers, hard_probability, c3_probability):
        curve_rows.append(
            {
                "record": "scaled_pair_distance_probability",
                "coordinate": center,
                "hard_value": hard_value,
                "c3_value": c3_value,
                "thermodynamic_claim_allowed": 0,
            }
        )
    for time, hard_value, c3_value in zip(times, hard_msd, c3_msd):
        curve_rows.append(
            {
                "record": "A_particle_msd",
                "coordinate": time,
                "hard_value": hard_value,
                "c3_value": c3_value,
                "thermodynamic_claim_allowed": 0,
            }
        )
    write_rows(args.curve_output, curve_rows)
    print(
        f"pair TV={summary['scaled_pair_histogram_total_variation']:.4g}; "
        f"force mean delta={summary['force_norm_mean_relative_difference']:.4g}; "
        f"MSD final delta={summary['msd_final_relative_difference']:.4g}"
    )


if __name__ == "__main__":
    main()
