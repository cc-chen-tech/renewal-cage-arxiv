#!/usr/bin/env python3
"""Reduce completed KA replicate dumps to independent-sample observables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import load_lammps_custom_trajectory, summarize_replicate_curves  # noqa: E402
from renewal_cage import (  # noqa: E402
    block_trajectory_observables,
    summarize_block_trajectory_observables,
)


def parse_numbers(value: str, dtype: type[float] | type[int]) -> np.ndarray:
    return np.array([dtype(item) for item in value.split(",")], dtype=dtype)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_ensemble(
    ensemble_directory: Path,
    *,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    diffusion_lag: int,
    overlap_radius: float,
    origin_stride: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    manifest = json.loads((ensemble_directory / "ensemble_manifest.json").read_text())
    expected_replicates = int(manifest["replicate_count"])
    curve_rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        timesteps = trajectory["timesteps"]
        if len(timesteps) < 2 or not np.all(np.diff(timesteps) == 1000):
            raise ValueError("replicate frames must be separated by exactly one tau")
        production_time = float(manifest["production_time_tau"])
        if timesteps[0] != 0 or timesteps[-1] != int(round(production_time * 1000.0)):
            raise ValueError("replicate does not cover the preregistered production window")
        positions = trajectory["unwrapped_positions"]
        particle_mask = trajectory["particle_types"] == 0
        rows = block_trajectory_observables(
            positions,
            lags=lags,
            block_size=len(positions) - 1,
            wave_numbers=wave_numbers,
            overlap_radius=overlap_radius,
            particle_mask=particle_mask,
            origin_stride=origin_stride,
        )
        for row in rows:
            row["block_index"] = float(replicate_index - 1)
            row["replicate"] = float(replicate_index)
            row["temperature"] = float(manifest["temperature"])
            row["independence_class"] = str(manifest["independence_class"])
            row["thermodynamic_claim_allowed"] = 0.0
        curve_rows.extend(rows)

    if len({int(row["replicate"]) for row in curve_rows}) != expected_replicates:
        raise ValueError("replicate count does not match the ensemble manifest")
    fs_keys = [f"fs_k{wave_number:g}".replace(".", "p") for wave_number in wave_numbers]
    curve_summary_rows = summarize_replicate_curves(
        curve_rows,
        metric_keys=["msd", "ngp_3d", "overlap_mean", "overlap_chi4", *fs_keys],
    )
    for row in curve_summary_rows:
        row["temperature"] = float(manifest["temperature"])
        row["independence_class"] = str(manifest["independence_class"])
        row["thermodynamic_claim_allowed"] = 0.0
    fs_key = f"fs_k{wave_numbers[np.argmin(abs(wave_numbers - 7.25))]:g}".replace(".", "p")
    replicate_rows, summary_rows = summarize_block_trajectory_observables(
        curve_rows,
        fs_key=fs_key,
        diffusion_lag=diffusion_lag,
    )
    for row in replicate_rows:
        row["replicate"] = float(row.pop("block_index") + 1.0)
        row["temperature"] = float(manifest["temperature"])
        row["independence_class"] = str(manifest["independence_class"])
        row["thermodynamic_claim_allowed"] = 0.0
    for row in summary_rows:
        row["temperature"] = float(manifest["temperature"])
        row["independent_replicate_count"] = float(expected_replicates)
        row["independence_class"] = str(manifest["independence_class"])
        row["maximum_absolute_initial_fs"] = float(manifest["maximum_absolute_fs_observed"])
        row["thermodynamic_claim_allowed"] = 0.0
    return curve_rows, curve_summary_rows, replicate_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--lags", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--wave-numbers", default="5,7.25,9")
    parser.add_argument("--diffusion-lag", type=int, default=512)
    parser.add_argument("--overlap-radius", type=float, default=0.3)
    parser.add_argument("--origin-stride", type=int, default=4)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    curves, curve_summary, replicates, summary = analyze_ensemble(
        args.ensemble_directory,
        lags=parse_numbers(args.lags, int),
        wave_numbers=parse_numbers(args.wave_numbers, float),
        diffusion_lag=args.diffusion_lag,
        overlap_radius=args.overlap_radius,
        origin_stride=args.origin_stride,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curves.csv"), curves)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_curve_summary.csv"),
        curve_summary,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicates)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)


if __name__ == "__main__":
    main()
