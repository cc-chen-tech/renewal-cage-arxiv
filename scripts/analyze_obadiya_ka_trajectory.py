#!/usr/bin/env python3
"""Derive block-uncertainty observables from an Obadiya-Sussman KA trajectory."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renewal_cage import (  # noqa: E402
    block_trajectory_observables,
    periodic_unwrap_trajectory,
    summarize_block_trajectory_observables,
)


def parse_numbers(value: str, dtype: type[float] | type[int]) -> np.ndarray:
    return np.array([dtype(item) for item in value.split(",")], dtype=dtype)


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    input_path: Path,
    *,
    temperature: float,
    block_size: int,
    diffusion_lag: int,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    overlap_radius: float,
    origin_stride: int,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float | str]]]:
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)
    box_lengths = np.asarray(payload["Box_size"][0], dtype=float)
    particle_types = np.asarray(payload["Particle_types"][0])
    position_frames = payload["Positions"]
    particle_mask = particle_types == 0
    block_count = (len(position_frames) - 1) // block_size
    if block_count < 3:
        raise ValueError("at least three complete nonoverlapping blocks are required")

    curve_rows: list[dict[str, float]] = []
    for block_index in range(block_count):
        start = block_index * block_size
        wrapped = np.asarray(position_frames[start : start + block_size + 1], dtype=np.float32)
        unwrapped = periodic_unwrap_trajectory(wrapped, box_lengths)
        rows = block_trajectory_observables(
            unwrapped,
            lags=lags,
            block_size=block_size,
            wave_numbers=wave_numbers,
            overlap_radius=overlap_radius,
            particle_mask=particle_mask,
            origin_stride=origin_stride,
        )
        for row in rows:
            row["block_index"] = float(block_index)
            row["temperature"] = temperature
            row["block_size"] = float(block_size)
            row["origin_stride"] = float(origin_stride)
            row["overlap_radius"] = overlap_radius
        curve_rows.extend(rows)

    fs_key = f"fs_k{wave_numbers[np.argmin(np.abs(wave_numbers - 7.25))]:g}".replace(".", "p")
    block_rows, summary_rows = summarize_block_trajectory_observables(
        curve_rows,
        fs_key=fs_key,
        diffusion_lag=diffusion_lag,
    )
    for row in block_rows:
        row["temperature"] = temperature
        row["block_size"] = float(block_size)
        row["diffusion_lag"] = float(diffusion_lag)
    for row in summary_rows:
        row["temperature"] = temperature
        row["block_size"] = float(block_size)
        row["diffusion_lag"] = float(diffusion_lag)
        row["source_doi"] = "10.5281/zenodo.7469766"
        row["uncertainty_method"] = "nonoverlapping_time_block_standard_error"
        row["independent_simulation_replicates"] = 1.0
        row["thermodynamic_claim_allowed"] = 0.0
    return curve_rows, block_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--diffusion-lag", type=int, required=True)
    parser.add_argument("--lags", default="1,2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument("--wave-numbers", default="5,7.25,9")
    parser.add_argument("--overlap-radius", type=float, default=0.3)
    parser.add_argument("--origin-stride", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    curves, blocks, summary = analyze(
        args.input,
        temperature=args.temperature,
        block_size=args.block_size,
        diffusion_lag=args.diffusion_lag,
        lags=parse_numbers(args.lags, int),
        wave_numbers=parse_numbers(args.wave_numbers, float),
        overlap_radius=args.overlap_radius,
        origin_stride=args.origin_stride,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curves.csv"), curves)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_blocks.csv"), blocks)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)


if __name__ == "__main__":
    main()
