#!/usr/bin/env python3
"""Extract preregistered p_hop event clocks and threshold robustness."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renewal_cage import (  # noqa: E402
    event_clock_statistics,
    extract_nonrecrossing_phop_events,
    periodic_unwrap_trajectory,
    phop_values,
)


def parse_thresholds(value: str) -> np.ndarray:
    thresholds = np.array([float(item) for item in value.split(",")], dtype=float)
    if np.any(thresholds <= 0.0) or len(np.unique(thresholds)) != len(thresholds):
        raise ValueError("thresholds must be unique and positive")
    return thresholds


def concatenate_events(parts: list[dict[str, np.ndarray]], dimension: int) -> dict[str, np.ndarray]:
    if not parts:
        return {
            "particle": np.empty(0, dtype=int),
            "time": np.empty(0, dtype=int),
            "phop": np.empty(0, dtype=float),
            "jump_vector": np.empty((0, dimension), dtype=float),
        }
    return {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in ("particle", "time", "phop", "jump_vector")
    }


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_threshold_blocks(
    input_path: Path,
    *,
    temperature: float,
    block_size: int,
    thresholds: np.ndarray,
    half_window: int,
    particle_chunk: int,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)
    box_lengths = np.asarray(payload["Box_size"][0], dtype=float)
    particle_types = np.asarray(payload["Particle_types"][0])
    a_indices = np.flatnonzero(particle_types == 0)
    position_frames = payload["Positions"]
    block_count = (len(position_frames) - 1) // block_size
    if block_count < 3:
        raise ValueError("at least three complete blocks are required")

    block_rows: list[dict[str, float | str]] = []
    nominal_event_rows: list[dict[str, float | str]] = []
    nominal_threshold = float(thresholds[np.argmin(abs(thresholds - 0.2))])
    for block_index in range(block_count):
        start = block_index * block_size
        wrapped = np.stack(
            [frame[a_indices] for frame in position_frames[start : start + block_size + 1]]
        ).astype(np.float32, copy=False)
        unwrapped = periodic_unwrap_trajectory(wrapped, box_lengths)
        threshold_parts: dict[float, list[dict[str, np.ndarray]]] = {
            float(threshold): [] for threshold in thresholds
        }
        for chunk_start in range(0, len(a_indices), particle_chunk):
            chunk_stop = min(chunk_start + particle_chunk, len(a_indices))
            chunk = unwrapped[:, chunk_start:chunk_stop]
            activity_times, activity_values = phop_values(chunk, half_window=half_window)
            for threshold in thresholds:
                threshold_value = float(threshold)
                events = extract_nonrecrossing_phop_events(
                    chunk,
                    threshold=threshold_value,
                    half_window=half_window,
                    recrossing_radius=math.sqrt(threshold_value),
                    activity_times=activity_times,
                    activity_values=activity_values,
                )
                events["particle"] += chunk_start
                threshold_parts[threshold_value].append(events)

        for threshold in thresholds:
            threshold_value = float(threshold)
            events = concatenate_events(threshold_parts[threshold_value], unwrapped.shape[2])
            statistics = event_clock_statistics(
                events,
                duration=float(block_size),
                particle_count=len(a_indices),
                dimension=unwrapped.shape[2],
            )
            row: dict[str, float | str] = {
                "temperature": temperature,
                "block_index": float(block_index),
                "block_size": float(block_size),
                "half_window": float(half_window),
                "threshold": threshold_value,
                "recrossing_radius": math.sqrt(threshold_value),
                "source_doi": "10.5281/zenodo.7469766",
                "event_definition": "candelier_phop_contiguous_peak_recursive_ABA_removal",
                "thermodynamic_claim_allowed": 0.0,
            }
            row.update(statistics)
            block_rows.append(row)
            if math.isclose(threshold_value, nominal_threshold):
                for index in range(len(events["particle"])):
                    jump = events["jump_vector"][index]
                    nominal_event_rows.append(
                        {
                            "temperature": temperature,
                            "block_index": float(block_index),
                            "particle": float(events["particle"][index]),
                            "local_time": float(events["time"][index]),
                            "global_time": float(start + events["time"][index]),
                            "phop": float(events["phop"][index]),
                            "jump_x": float(jump[0]),
                            "jump_y": float(jump[1]),
                            "jump_z": float(jump[2]),
                            "threshold": nominal_threshold,
                            "half_window": float(half_window),
                            "thermodynamic_claim_allowed": 0.0,
                        }
                    )
    return block_rows, nominal_event_rows


def summarize_thresholds(
    block_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    summary: list[dict[str, float | str]] = []
    metrics = [
        "event_count",
        "event_rate",
        "exchange_mean",
        "exchange_cv2",
        "stationary_persistence_mean",
        "persistence_exchange_ratio",
        "jump_squared_mean",
        "count_fano",
        "jump_correlation_lag1_over_q",
        "jump_correlation_lag2_over_q",
        "uncorrelated_diffusion",
        "correlated_diffusion",
    ]
    for threshold in sorted({float(row["threshold"]) for row in block_rows}):
        selected = [row for row in block_rows if float(row["threshold"]) == threshold]
        row: dict[str, float | str] = {
            "temperature": float(selected[0]["temperature"]),
            "threshold": threshold,
            "block_count": float(len(selected)),
            "half_window": float(selected[0]["half_window"]),
            "source_doi": str(selected[0]["source_doi"]),
            "thermodynamic_claim_allowed": 0.0,
        }
        for metric in metrics:
            values = np.array([float(item[metric]) for item in selected])
            standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_standard_error"] = standard_error
            row[f"{metric}_ci95_low"] = float(np.mean(values) - 1.96 * standard_error)
            row[f"{metric}_ci95_high"] = float(np.mean(values) + 1.96 * standard_error)
        summary.append(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--thresholds", default="0.16,0.18,0.20,0.22,0.24")
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--particle-chunk", type=int, default=256)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    blocks, events = extract_threshold_blocks(
        args.input,
        temperature=args.temperature,
        block_size=args.block_size,
        thresholds=parse_thresholds(args.thresholds),
        half_window=args.half_window,
        particle_chunk=args.particle_chunk,
    )
    summary = summarize_thresholds(blocks)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_blocks.csv"), blocks)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_events_pc020.csv"), events)


if __name__ == "__main__":
    main()
