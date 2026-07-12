#!/usr/bin/env python3
"""Summarize the temperature selectivity of jump/cage channel transfer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_one(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    return rows[0]


def write_one(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-verdict", type=Path, required=True)
    parser.add_argument("--low-verdict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    high = read_one(args.high_verdict)
    low = read_one(args.low_verdict)
    high_pass = float(high["retrospective_ensemble_transfer_candidate_pass"])
    low_pass = float(low["retrospective_ensemble_transfer_candidate_pass"])
    row = {
        "high_temperature": float(high["temperature"]),
        "low_temperature": float(low["temperature"]),
        "high_temperature_channel_transfer_pass": high_pass,
        "low_temperature_channel_transfer_pass": low_pass,
        "jump_cage_scale_separation_emerges_on_cooling": float(
            high_pass == 0.0 and low_pass == 1.0
        ),
        "low_maximum_ensemble_msd_relative_error": float(
            low["maximum_ensemble_msd_relative_error"]
        ),
        "low_maximum_ensemble_ngp_absolute_error": float(
            low["maximum_ensemble_ngp_absolute_error"]
        ),
        "low_maximum_ensemble_fs_absolute_error": float(
            low["maximum_ensemble_fs_absolute_error"]
        ),
        "heldout_events_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "source_doi": "10.3390/ijms23073556",
        "source_alignment": "cage_jump_dynamics_becomes_progressively_more_marked_on_cooling",
        "retrospective_candidate_claim_allowed": low_pass,
        "preregistered_heldout_prediction_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    write_one(args.output, row)


if __name__ == "__main__":
    main()
