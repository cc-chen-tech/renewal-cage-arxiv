#!/usr/bin/env python3
"""Score calibration-only reversible cage-center returns across temperatures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    consecutive_cage_anchor_returns,
    debye_waller_factor_from_msd,
    extract_debye_waller_cage_jumps,
    load_lammps_custom_trajectory,
    position_fluctuation_values,
)


RADIUS_SCALES = (0.5, 1.0, 1.5)


def calibration_msd_curve(
    positions: np.ndarray,
    *,
    maximum_lag: int,
    lag_count: int,
    origin_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the calibration-only MSD curve used for Debye-Waller selection."""

    if maximum_lag < 4 or lag_count < 5 or origin_stride < 1:
        raise ValueError("Debye-Waller lag controls are invalid")
    maximum_lag = min(maximum_lag, len(positions) // 3)
    lags = np.unique(np.geomspace(1, maximum_lag, lag_count).astype(int))
    values = []
    for lag in lags:
        origins = np.arange(0, len(positions) - lag, origin_stride)
        displacement = positions[origins + lag] - positions[origins]
        values.append(float(np.mean(np.sum(displacement**2, axis=2))))
    return lags, np.asarray(values)


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty cage-anchor table")
    fieldnames = list(rows[0])
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _claim_flags() -> dict[str, float]:
    return {
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def classify_cage_anchor_returns(
    low_rows: Sequence[dict[str, object]],
    high_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Apply the frozen all-scale separation and primary-null gate."""

    if not low_rows or not high_rows:
        raise ValueError("low- and high-temperature return rows must not be empty")
    low_scales = {float(row["radius_scale"]) for row in low_rows}
    high_scales = {float(row["radius_scale"]) for row in high_rows}
    expected = set(RADIUS_SCALES)
    if low_scales != expected or high_scales != expected:
        raise ValueError("return rows must contain exactly the frozen radius scales")
    if any(
        not np.isfinite(float(row[key]))
        for row in (*low_rows, *high_rows)
        for key in ("return_fraction", "isotropic_null_fraction")
    ):
        raise ValueError("return fractions and null fractions must be finite")

    scale_separation: dict[str, float] = {}
    for scale in RADIUS_SCALES:
        low = [float(row["return_fraction"]) for row in low_rows if float(row["radius_scale"]) == scale]
        high = [float(row["return_fraction"]) for row in high_rows if float(row["radius_scale"]) == scale]
        scale_separation[f"minimum_low_return_fraction_s{scale:g}".replace(".", "p")] = min(low)
        scale_separation[f"maximum_high_return_fraction_s{scale:g}".replace(".", "p")] = max(high)
        scale_separation[f"radius_scale_separated_s{scale:g}".replace(".", "p")] = float(min(low) > max(high))
    primary = [row for row in low_rows if float(row["radius_scale"]) == 1.0]
    primary_excess = [
        float(row["return_fraction"]) / float(row["isotropic_null_fraction"])
        if float(row["isotropic_null_fraction"]) > 0.0
        else 0.0
        for row in primary
    ]
    all_scales_separated = all(
        bool(scale_separation[f"radius_scale_separated_s{scale:g}".replace(".", "p")])
        for scale in RADIUS_SCALES
    )
    primary_null_excess = all(value >= 1.35 for value in primary_excess)
    return {
        **scale_separation,
        "low_temperature_replicate_count": float(len({int(float(row["replicate"])) for row in low_rows})),
        "high_temperature_replicate_count": float(len({int(float(row["replicate"])) for row in high_rows})),
        "primary_radius_scale": 1.0,
        "minimum_primary_low_return_excess_ratio": min(primary_excess),
        "primary_null_excess_ratio_tolerance": 1.35,
        "all_radius_scales_separated": float(all_scales_separated),
        "primary_radius_null_excess_pass": float(primary_null_excess),
        "cage_anchor_return_signal_ready": float(all_scales_separated and primary_null_excess),
        "calibration_events_only": 1.0,
        "heldout_events_used_in_calibration": 0.0,
        **_claim_flags(),
    }


def calibration_return_rows(
    ensemble_directory: Path,
    *,
    calibration_time: int,
    maximum_dw_lag: int,
    dw_lag_count: int,
    origin_stride: int,
    fluctuation_half_window: int,
) -> tuple[list[dict[str, object]], float]:
    """Measure all frozen return radii from each calibration trajectory only."""

    if min(calibration_time, maximum_dw_lag, dw_lag_count, origin_stride, fluctuation_half_window) < 1:
        raise ValueError("calibration controls must be positive")
    manifest = json.loads((ensemble_directory / "ensemble_manifest.json").read_text())
    temperature = float(manifest["temperature"])
    rows: list[dict[str, object]] = []
    for specification in manifest.get("replicates", []):
        replicate = int(specification["replicate"])
        directory = ensemble_directory / str(specification["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(
            directory / "trajectory.lammpstrj",
            maximum_frame_count=calibration_time + 1,
        )
        if len(trajectory["unwrapped_positions"]) < calibration_time + 1:
            raise ValueError(f"replicate {replicate} is shorter than the calibration window")
        calibration = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        lags, msd = calibration_msd_curve(
            calibration,
            maximum_lag=min(maximum_dw_lag, calibration_time - 1),
            lag_count=dw_lag_count,
            origin_stride=origin_stride,
        )
        dw = debye_waller_factor_from_msd(lags, msd)
        times, fluctuation = position_fluctuation_values(
            calibration,
            half_window=fluctuation_half_window,
        )
        events = extract_debye_waller_cage_jumps(
            calibration,
            debye_waller_factor=float(dw["debye_waller_factor"]),
            half_window=fluctuation_half_window,
            activity_times=times,
            activity_values=fluctuation,
        )
        for radius_scale in RADIUS_SCALES:
            measurement = consecutive_cage_anchor_returns(
                events,
                debye_waller_factor=float(dw["debye_waller_factor"]),
                radius_scale=radius_scale,
            )
            rows.append(
                {
                    "replicate": float(replicate),
                    "temperature": temperature,
                    "calibration_time": float(calibration_time),
                    "radius_scale": radius_scale,
                    "debye_waller_factor": float(dw["debye_waller_factor"]),
                    "debye_waller_lag": float(dw["debye_waller_lag"]),
                    "fluctuation_half_window": float(fluctuation_half_window),
                    "calibration_event_count": float(len(events["time"])),
                    **measurement,
                    "calibration_events_only": 1.0,
                    "heldout_events_used_in_calibration": 0.0,
                    "macro_fit_parameter_count": 0.0,
                    **_claim_flags(),
                }
            )
    if not rows:
        raise ValueError("ensemble manifest contains no replicates")
    return rows, temperature


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("low_temperature_ensemble", type=Path)
    parser.add_argument("high_temperature_ensemble", type=Path)
    parser.add_argument("--low-calibration-time", type=int, default=5000)
    parser.add_argument("--high-calibration-time", type=int, default=750)
    parser.add_argument("--maximum-dw-lag", type=int, default=100)
    parser.add_argument("--dw-lag-count", type=int, default=25)
    parser.add_argument("--origin-stride", type=int, default=10)
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    low_rows, low_temperature = calibration_return_rows(
        args.low_temperature_ensemble,
        calibration_time=args.low_calibration_time,
        maximum_dw_lag=args.maximum_dw_lag,
        dw_lag_count=args.dw_lag_count,
        origin_stride=args.origin_stride,
        fluctuation_half_window=args.fluctuation_half_window,
    )
    high_rows, high_temperature = calibration_return_rows(
        args.high_temperature_ensemble,
        calibration_time=args.high_calibration_time,
        maximum_dw_lag=args.maximum_dw_lag,
        dw_lag_count=args.dw_lag_count,
        origin_stride=args.origin_stride,
        fluctuation_half_window=args.fluctuation_half_window,
    )
    if not low_temperature < high_temperature:
        raise ValueError("low-temperature ensemble must have the lower temperature")
    verdict = classify_cage_anchor_returns(low_rows, high_rows)
    verdict.update(
        {
            "low_temperature": low_temperature,
            "high_temperature": high_temperature,
            "low_calibration_time": float(args.low_calibration_time),
            "high_calibration_time": float(args.high_calibration_time),
            "fluctuation_half_window": float(args.fluctuation_half_window),
        }
    )
    prefix = args.output_prefix
    write_rows(prefix.with_name(prefix.name + "_rows.csv"), [*low_rows, *high_rows])
    write_rows(prefix.with_name(prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
