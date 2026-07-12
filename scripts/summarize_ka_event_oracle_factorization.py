#!/usr/bin/env python3
"""Test a no-fit jump/cage factorization using held-out oracle event paths."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import independent_isotropic_channel_moments  # noqa: E402


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
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--maximum-msd-relative-error", type=float, default=0.1)
    parser.add_argument("--maximum-ngp-absolute-error", type=float, default=0.3)
    parser.add_argument("--maximum-fs-absolute-error", type=float, default=0.03)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    source = read_rows(args.curves)
    fs_keys = [key for key in source[0] if key.startswith("fs_k")]
    representations = {
        representation: {
            (int(float(row["replicate"])), int(float(row["lag"]))): row
            for row in source
            if row["representation"] == representation
        }
        for representation in {
            "observed_trajectory",
            "oracle_debye_waller_event_path",
            "debye_waller_residual_path",
        }
    }
    keys = representations["observed_trajectory"].keys()
    if any(table.keys() != keys for table in representations.values()):
        raise ValueError("oracle representations must share replicate-lag keys")
    rows: list[dict[str, object]] = []
    for replicate, lag in sorted(keys):
        observed = representations["observed_trajectory"][(replicate, lag)]
        event = representations["oracle_debye_waller_event_path"][(replicate, lag)]
        residual = representations["debye_waller_residual_path"][(replicate, lag)]
        moments = independent_isotropic_channel_moments(
            first_msd=float(event["msd"]),
            first_ngp=float(event["ngp_3d"]),
            second_msd=float(residual["msd"]),
            second_ngp=float(residual["ngp_3d"]),
            dimension=3,
        )
        row: dict[str, object] = {
            "replicate": float(replicate),
            "temperature": float(observed["temperature"]),
            "lag": float(lag),
            "observed_msd": float(observed["msd"]),
            "event_msd": float(event["msd"]),
            "residual_msd": float(residual["msd"]),
            "factorized_msd": moments["combined_msd"],
            "msd_relative_error": abs(
                moments["combined_msd"] / float(observed["msd"]) - 1.0
            ),
            "observed_ngp": float(observed["ngp_3d"]),
            "event_ngp": float(event["ngp_3d"]),
            "residual_ngp": float(residual["ngp_3d"]),
            "factorized_ngp": moments["combined_ngp"],
            "ngp_absolute_error": abs(
                moments["combined_ngp"] - float(observed["ngp_3d"])
            ),
            "oracle_uses_heldout_events_and_residual": 1.0,
            "prediction_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for fs_key in fs_keys:
            factorized = float(event[fs_key]) * float(residual[fs_key])
            row[f"observed_{fs_key}"] = float(observed[fs_key])
            row[f"event_{fs_key}"] = float(event[fs_key])
            row[f"residual_{fs_key}"] = float(residual[fs_key])
            row[f"factorized_{fs_key}"] = factorized
            row[f"absolute_error_{fs_key}"] = abs(factorized - float(observed[fs_key]))
        rows.append(row)
    maximum_fs_error = max(
        float(row[f"absolute_error_{fs_key}"])
        for row in rows
        for fs_key in fs_keys
    )
    verdict = {
        "temperature": rows[0]["temperature"],
        "independent_replicate_count": float(len({float(row["replicate"]) for row in rows})),
        "lag_count": float(len({float(row["lag"]) for row in rows})),
        "wave_number_count": float(len(fs_keys)),
        "maximum_msd_relative_error": max(float(row["msd_relative_error"]) for row in rows),
        "maximum_ngp_absolute_error": max(float(row["ngp_absolute_error"]) for row in rows),
        "maximum_fs_absolute_error": maximum_fs_error,
        "msd_tolerance": args.maximum_msd_relative_error,
        "ngp_tolerance": args.maximum_ngp_absolute_error,
        "fs_tolerance": args.maximum_fs_absolute_error,
        "oracle_jump_cage_factorization_supported": float(
            max(float(row["msd_relative_error"]) for row in rows)
            <= args.maximum_msd_relative_error
            and max(float(row["ngp_absolute_error"]) for row in rows)
            <= args.maximum_ngp_absolute_error
            and maximum_fs_error <= args.maximum_fs_absolute_error
        ),
        "gate_scope": "posthoc_oracle_representation_diagnostic",
        "calibration_prediction_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_rows.csv"), rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
