#!/usr/bin/env python3
"""Separate reproducible halo measurements from rejected transport closure claims."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import summarize_paired_curve_stability  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-shells", type=Path, required=True)
    parser.add_argument("--heldout-shells", type=Path, required=True)
    parser.add_argument("--calibration-halo-verdict", type=Path, required=True)
    parser.add_argument("--heldout-halo-verdict", type=Path, required=True)
    parser.add_argument("--primary-closure-verdict", type=Path, required=True)
    parser.add_argument("--radius-sensitivity-verdict", type=Path, required=True)
    parser.add_argument("--relative-equivalence-margin", type=float, default=0.2)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    calibration = read_rows(args.calibration_shells)
    heldout = read_rows(args.heldout_shells)
    curve, verdict = summarize_paired_curve_stability(
        calibration,
        heldout,
        bin_key="distance_midpoint",
        metric_key="event_to_control_squared_ratio",
        relative_equivalence_margin=args.relative_equivalence_margin,
    )
    calibration_halo = read_one(args.calibration_halo_verdict)
    heldout_halo = read_one(args.heldout_halo_verdict)
    primary = read_one(args.primary_closure_verdict)
    sensitivity = read_one(args.radius_sensitivity_verdict)
    calibration_radius = float(calibration_halo["halo_radius_lower_bound"])
    heldout_radius = float(heldout_halo["halo_radius_lower_bound"])
    primary_pass = float(primary["heldout_transport_pass"])
    sensitivity_pass = float(sensitivity["heldout_transport_pass"])
    profile_equivalent = float(verdict["paired_curve_equivalent"])
    radius_stable = float(calibration_radius == heldout_radius)
    verdict.update(
        {
            "calibration_halo_radius_lower_bound": calibration_radius,
            "heldout_halo_radius_lower_bound": heldout_radius,
            "binary_radius_gate_stable": radius_stable,
            "radius_difference_interpretation": (
                "ci_significance_boundary_not_profile_shift"
                if profile_equivalent == 1.0 and radius_stable == 0.0
                else "no_radius_boundary_discrepancy"
            ),
            "primary_calibration_radius": float(primary["halo_radius"]),
            "primary_closure_pass": primary_pass,
            "posthoc_sensitivity_radius": float(sensitivity["halo_radius"]),
            "posthoc_sensitivity_closure_pass": sensitivity_pass,
            "closure_rejection_robust_to_radius_boundary": float(
                primary_pass == 0.0 and sensitivity_pass == 0.0
            ),
            "spatial_measurement_claim_allowed": profile_equivalent,
            "spatial_model_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    for row in curve:
        midpoint = float(row["distance_midpoint"])
        matching = [source for source in calibration if float(source["distance_midpoint"]) == midpoint]
        row["distance_low"] = float(matching[0]["distance_low"])
        row["distance_high"] = float(matching[0]["distance_high"])
        row["spatial_measurement_claim_allowed"] = profile_equivalent
        row["spatial_model_claim_allowed"] = 0.0
        row["thermodynamic_claim_allowed"] = 0.0
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), curve)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
