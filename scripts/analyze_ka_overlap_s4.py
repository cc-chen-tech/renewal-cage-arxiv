#!/usr/bin/env python3
"""Compute a claim-limited overlap S4 and Ornstein-Zernike identifiability gate."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    fit_ornstein_zernike_structure_factor,
    load_lammps_custom_trajectory,
    overlap_four_point_structure_factor,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--lag", type=int, required=True)
    parser.add_argument("--overlap-radius", type=float, default=0.3)
    parser.add_argument("--origin-stride", type=int, default=32)
    parser.add_argument("--maximum-integer-squared", type=int, default=6)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    trajectory = load_lammps_custom_trajectory(args.trajectory)
    rows = overlap_four_point_structure_factor(
        trajectory["unwrapped_positions"],
        box_lengths=trajectory["box_lengths"],
        lag=args.lag,
        overlap_radius=args.overlap_radius,
        origin_stride=args.origin_stride,
        maximum_integer_squared=args.maximum_integer_squared,
    )
    for row in rows:
        row["temperature"] = args.temperature
        row["lag"] = float(args.lag)
        row["overlap_radius"] = args.overlap_radius
        row["particle_selection"] = "all_particles"
        row["ensemble_correction_available"] = 0.0
        row["independent_replicate_count"] = 1.0
        row["xi4_claim_allowed"] = 0.0
        row["thermodynamic_claim_allowed"] = 0.0

    fit: dict[str, object] = fit_ornstein_zernike_structure_factor(rows[1:])
    fit.update(
        {
            "temperature": args.temperature,
            "lag": float(args.lag),
            "raw_q0_susceptibility": float(rows[0]["s4"]),
            "minimum_q_s4": float(rows[1]["s4"]),
            "minimum_q_wavevector_standard_deviation": float(
                rows[1]["s4_wavevector_standard_deviation"]
            ),
            "minimum_q_wavevector_min": float(rows[1]["s4_wavevector_min"]),
            "minimum_q_wavevector_max": float(rows[1]["s4_wavevector_max"]),
            "minimum_q_to_raw_q0_ratio": float(rows[1]["s4"]) / float(rows[0]["s4"]),
            "all_minimum_q_directions_exceed_raw_q0": float(
                float(rows[1]["s4_wavevector_min"]) > float(rows[0]["s4"])
            ),
            "ensemble_correction_available": 0.0,
            "xi4_identifiable": float(bool(fit["fit_valid"])),
            "verdict": "xi4_not_identifiable_negative_OZ_intercept"
            if not bool(fit["fit_valid"])
            else "exploratory_OZ_fit_requires_replicate_validation",
            "independent_replicate_count": 1.0,
            "xi4_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curve.csv"), rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_fit.csv"), [fit])


if __name__ == "__main__":
    main()
