#!/usr/bin/env python3
"""Measure the low-k validity range of the observed fourth-cumulant closure."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import fourth_cumulant_scattering  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty path-cumulant table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wave_number_from_observed_key(key: str) -> float:
    return float(key.removeprefix("observed_fs_k").replace("p", "."))


def cumulant_scattering_rows(
    rows: Sequence[dict[str, object]],
    *,
    error_tolerance: float = 0.03,
) -> list[dict[str, object]]:
    if (
        not rows
        or not math.isfinite(error_tolerance)
        or error_tolerance <= 0.0
    ):
        raise ValueError("nonempty rows and a positive finite tolerance are required")
    fs_keys = sorted(key for key in rows[0] if key.startswith("observed_fs_k"))
    if not fs_keys:
        raise ValueError("input rows contain no observed scattering columns")
    groups = sorted(
        {(float(row["temperature"]), float(row["lag"])) for row in rows}
    )
    output: list[dict[str, object]] = []
    for temperature, lag in groups:
        selected = [
            row
            for row in rows
            if float(row["temperature"]) == temperature
            and float(row["lag"]) == lag
        ]
        replicate_ids = {int(float(row["replicate"])) for row in selected}
        if len(replicate_ids) != len(selected):
            raise ValueError("each temperature-lag group must contain unique replicates")
        msd = float(np.mean([float(row["observed_msd"]) for row in selected]))
        ngp = float(np.mean([float(row["observed_ngp"]) for row in selected]))
        for fs_key in fs_keys:
            wave_number = wave_number_from_observed_key(fs_key)
            observed_fs = float(np.mean([float(row[fs_key]) for row in selected]))
            predicted_fs = float(
                fourth_cumulant_scattering(
                    np.asarray([msd]),
                    np.asarray([ngp]),
                    wave_number,
                )[0]
            )
            unit_interval = math.isfinite(predicted_fs) and 0.0 <= predicted_fs <= 1.0
            absolute_error = (
                abs(predicted_fs - observed_fs) if math.isfinite(predicted_fs) else math.inf
            )
            output.append(
                {
                    "temperature": temperature,
                    "lag": lag,
                    "wave_number": wave_number,
                    "independent_replicate_count": float(len(replicate_ids)),
                    "observed_msd": msd,
                    "observed_ngp": ngp,
                    "cumulant_fs": predicted_fs,
                    "observed_fs": observed_fs,
                    "absolute_error": absolute_error,
                    "error_tolerance": error_tolerance,
                    "approximation_in_unit_interval": float(unit_interval),
                    "within_validity_tolerance": float(
                        unit_interval and absolute_error <= error_tolerance
                    ),
                    "observed_msd_used": 1.0,
                    "observed_ngp_used": 1.0,
                    "heldout_prediction_claim_allowed": 0.0,
                    "microdynamic_closure_claim_allowed": 0.0,
                    "spatial_facilitation_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    return output


def cumulant_scattering_validity(
    rows: Sequence[dict[str, object]],
    *,
    error_tolerance: float = 0.03,
) -> list[dict[str, object]]:
    diagnostic_rows = cumulant_scattering_rows(
        rows,
        error_tolerance=error_tolerance,
    )
    result: list[dict[str, object]] = []
    groups = sorted(
        {(float(row["temperature"]), float(row["wave_number"])) for row in diagnostic_rows}
    )
    for temperature, wave_number in groups:
        selected = sorted(
            (
                row
                for row in diagnostic_rows
                if float(row["temperature"]) == temperature
                and float(row["wave_number"]) == wave_number
            ),
            key=lambda row: float(row["lag"]),
        )
        valid_count = 0
        longest_valid_lag = 0.0
        first_invalid_lag = -1.0
        for row in selected:
            if float(row["within_validity_tolerance"]) == 1.0:
                valid_count += 1
                longest_valid_lag = float(row["lag"])
            else:
                first_invalid_lag = float(row["lag"])
                break
        all_valid = valid_count == len(selected)
        result.append(
            {
                "temperature": temperature,
                "wave_number": wave_number,
                "tested_lag_count": float(len(selected)),
                "contiguous_valid_lag_count": float(valid_count),
                "longest_contiguous_valid_lag": longest_valid_lag,
                "first_invalid_lag": first_invalid_lag,
                "all_tested_lags_valid": float(all_valid),
                "maximum_absolute_error": max(float(row["absolute_error"]) for row in selected),
                "unit_interval_failure_count": float(
                    sum(
                        float(row["approximation_in_unit_interval"]) == 0.0
                        for row in selected
                    )
                ),
                "error_tolerance": error_tolerance,
                "observed_msd_used": 1.0,
                "observed_ngp_used": 1.0,
                "heldout_prediction_claim_allowed": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("factorization_rows", type=Path)
    parser.add_argument("--error-tolerance", type=float, default=0.03)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    input_rows = read_rows(args.factorization_rows)
    diagnostic_rows = cumulant_scattering_rows(
        input_rows,
        error_tolerance=args.error_tolerance,
    )
    validity_rows = cumulant_scattering_validity(
        input_rows,
        error_tolerance=args.error_tolerance,
    )
    prefix = args.output_prefix
    write_rows(prefix.with_name(prefix.name + "_rows.csv"), diagnostic_rows)
    write_rows(prefix.with_name(prefix.name + "_validity.csv"), validity_rows)


if __name__ == "__main__":
    main()
