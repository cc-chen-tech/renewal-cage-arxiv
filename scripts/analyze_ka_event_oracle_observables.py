#!/usr/bin/env python3
"""Test whether observed held-out events can represent trajectory observables."""

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

from ka_replicates import (  # noqa: E402
    event_cumulative_trajectory,
    extract_debye_waller_cage_jumps,
    load_lammps_custom_trajectory,
    position_fluctuation_values,
)
from renewal_cage import (  # noqa: E402
    block_trajectory_observables,
    extract_nonrecrossing_phop_events,
    phop_values,
)


def parse_numbers(value: str) -> np.ndarray:
    return np.array([float(item) for item in value.split(",")], dtype=float)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--heldout-replicates", type=Path, required=True)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--phop-threshold", type=float, default=0.15)
    parser.add_argument("--phop-half-window", type=int, default=5)
    parser.add_argument("--lags", type=parse_numbers, required=True)
    parser.add_argument("--wave-numbers", type=parse_numbers, default=parse_numbers("2,4,7.25"))
    parser.add_argument("--origin-stride", type=int, default=8)
    parser.add_argument("--overlap-radius", type=float, default=0.3)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    lags = args.lags.astype(int)
    if np.any(lags != args.lags):
        raise ValueError("lags must be integers")
    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    calibration_rows = {
        int(float(row["replicate"])): row for row in read_rows(args.heldout_replicates)
    }
    rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = args.ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        threshold = float(calibration_rows[replicate_index]["debye_waller_factor"])
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        heldout = np.asarray(positions[args.calibration_time :], dtype=float)
        fluctuation_times, fluctuation = position_fluctuation_values(
            heldout,
            half_window=args.fluctuation_half_window,
        )
        dw_events = extract_debye_waller_cage_jumps(
            heldout,
            debye_waller_factor=threshold,
            half_window=args.fluctuation_half_window,
            activity_times=fluctuation_times,
            activity_values=fluctuation,
        )
        phop_times, phop = phop_values(heldout, half_window=args.phop_half_window)
        phop_events = extract_nonrecrossing_phop_events(
            heldout,
            threshold=args.phop_threshold,
            half_window=args.phop_half_window,
            recrossing_radius=math.sqrt(args.phop_threshold),
            activity_times=phop_times,
            activity_values=phop,
        )
        dw_path = event_cumulative_trajectory(
            dw_events,
            frame_count=len(heldout),
            particle_count=heldout.shape[1],
            dimension=heldout.shape[2],
        )
        phop_path = event_cumulative_trajectory(
            phop_events,
            frame_count=len(heldout),
            particle_count=heldout.shape[1],
            dimension=heldout.shape[2],
        )
        residual = heldout - dw_path
        representations = {
            "observed_trajectory": heldout,
            "oracle_debye_waller_event_path": dw_path,
            "oracle_phop_event_path": phop_path,
            "debye_waller_residual_path": residual,
        }
        for representation, path in representations.items():
            local = block_trajectory_observables(
                path,
                lags=lags,
                block_size=len(path) - 1,
                wave_numbers=args.wave_numbers,
                overlap_radius=args.overlap_radius,
                origin_stride=args.origin_stride,
            )
            for row in local:
                row.update(
                    {
                        "replicate": float(replicate_index),
                        "temperature": float(manifest["temperature"]),
                        "representation": representation,
                        "calibration_debye_waller_factor": threshold,
                        "heldout_dw_event_count": float(len(dw_events["time"])),
                        "heldout_phop_event_count": float(len(phop_events["time"])),
                        "oracle_uses_heldout_events": float(
                            representation != "observed_trajectory"
                        ),
                        "prediction_claim_allowed": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
            rows.extend(local)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curves.csv"), rows)


if __name__ == "__main__":
    main()
