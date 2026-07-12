#!/usr/bin/env python3
"""Compare p_hop and Debye-Waller cage-jump clocks on held-out KA diffusion."""

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
    debye_waller_factor_from_msd,
    extract_debye_waller_cage_jumps,
    jump_vector_correlation_curve,
    load_lammps_custom_trajectory,
    position_fluctuation_values,
    summarize_heldout_event_transport,
    trajectory_diffusion_estimate,
)
from renewal_cage import (  # noqa: E402
    event_clock_statistics,
    extract_nonrecrossing_phop_events,
    phop_values,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def calibration_msd_curve(
    positions: np.ndarray,
    *,
    maximum_lag: int,
    lag_count: int,
    origin_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if maximum_lag < 4 or lag_count < 5 or origin_stride < 1:
        raise ValueError("Debye-Waller lag controls are invalid")
    maximum_lag = min(maximum_lag, len(positions) // 3)
    lags = np.unique(np.geomspace(1, maximum_lag, lag_count).astype(int))
    values = []
    for lag in lags:
        origins = np.arange(0, len(positions) - lag, origin_stride)
        displacement = positions[origins + lag] - positions[origins]
        values.append(float(np.mean(np.sum(displacement**2, axis=2))))
    return lags, np.array(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--calibration-time", type=int, default=5000)
    parser.add_argument("--heldout-diffusion-lag", type=int, default=4096)
    parser.add_argument("--origin-stride", type=int, default=8)
    parser.add_argument("--maximum-dw-lag", type=int, default=512)
    parser.add_argument("--dw-lag-count", type=int, default=50)
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--phop-threshold", type=float, default=0.15)
    parser.add_argument("--phop-half-window", type=int, default=5)
    parser.add_argument("--maximum-correlation-lag", type=int, default=2)
    parser.add_argument("--diagnostic-correlation-lag", type=int, default=10)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--maximum-coverage", type=float, default=1.2)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    production_time = int(round(float(manifest["production_time_tau"])))
    if production_time != 2 * args.calibration_time:
        raise ValueError("closure requires equal calibration and held-out windows")
    if args.diagnostic_correlation_lag < args.maximum_correlation_lag:
        raise ValueError("diagnostic correlation lag must include the primary truncation")
    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = args.ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        calibration = positions[: args.calibration_time + 1]
        heldout = positions[args.calibration_time :]
        lags, msd = calibration_msd_curve(
            calibration,
            maximum_lag=args.maximum_dw_lag,
            lag_count=args.dw_lag_count,
            origin_stride=args.origin_stride,
        )
        dw = debye_waller_factor_from_msd(lags, msd)
        slope = np.gradient(np.log(msd), np.log(lags))
        for lag, value, local_slope in zip(lags, msd, slope):
            curve_rows.append(
                {
                    "replicate": float(replicate_index),
                    "temperature": float(manifest["temperature"]),
                    "lag": float(lag),
                    "calibration_msd": float(value),
                    "log_msd_slope": float(local_slope),
                    "selected_debye_waller_lag": dw["debye_waller_lag"],
                    "selected_debye_waller_factor": dw["debye_waller_factor"],
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        fluctuation_times, fluctuation = position_fluctuation_values(
            calibration,
            half_window=args.fluctuation_half_window,
        )
        dw_events = extract_debye_waller_cage_jumps(
            calibration,
            debye_waller_factor=float(dw["debye_waller_factor"]),
            half_window=args.fluctuation_half_window,
            activity_times=fluctuation_times,
            activity_values=fluctuation,
        )
        dw_stats = event_clock_statistics(
            dw_events,
            duration=float(args.calibration_time),
            particle_count=calibration.shape[1],
            dimension=calibration.shape[2],
            max_correlation_lag=args.maximum_correlation_lag,
        )
        local_correlations = jump_vector_correlation_curve(
            dw_events,
            maximum_lag=args.diagnostic_correlation_lag,
        )
        for row in local_correlations:
            row.update(
                {
                    "replicate": float(replicate_index),
                    "temperature": float(manifest["temperature"]),
                    "primary_correlation_truncation": float(args.maximum_correlation_lag),
                    "used_in_primary_prediction": float(
                        row["event_lag"] <= args.maximum_correlation_lag
                    ),
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        correlation_rows.extend(local_correlations)
        phop_times, phop = phop_values(calibration, half_window=args.phop_half_window)
        phop_events = extract_nonrecrossing_phop_events(
            calibration,
            threshold=args.phop_threshold,
            half_window=args.phop_half_window,
            recrossing_radius=math.sqrt(args.phop_threshold),
            activity_times=phop_times,
            activity_values=phop,
        )
        phop_stats = event_clock_statistics(
            phop_events,
            duration=float(args.calibration_time),
            particle_count=calibration.shape[1],
            dimension=calibration.shape[2],
            max_correlation_lag=args.maximum_correlation_lag,
        )
        observed = trajectory_diffusion_estimate(
            heldout,
            lag=args.heldout_diffusion_lag,
            origin_stride=args.origin_stride,
        )
        rows.append(
            {
                "replicate": float(replicate_index),
                "temperature": float(manifest["temperature"]),
                "calibration_time": float(args.calibration_time),
                "heldout_time": float(production_time - args.calibration_time),
                "heldout_diffusion_lag": float(args.heldout_diffusion_lag),
                "debye_waller_lag": dw["debye_waller_lag"],
                "debye_waller_factor": dw["debye_waller_factor"],
                "minimum_log_msd_slope": dw["minimum_log_msd_slope"],
                "fluctuation_half_window": float(args.fluctuation_half_window),
                "dw_active_fraction": float(np.mean(fluctuation > dw["debye_waller_factor"])),
                "dw_event_count": dw_stats["event_count"],
                "dw_event_rate": dw_stats["event_rate"],
                "dw_mean_jump_squared": dw_stats["jump_squared_mean"],
                "dw_mean_jump_duration": float(np.mean(dw_events["jump_duration"])),
                "dw_jump_correlation_lag1_over_q": dw_stats["jump_correlation_lag1_over_q"],
                "dw_jump_correlation_lag2_over_q": dw_stats["jump_correlation_lag2_over_q"],
                "phop_event_count": phop_stats["event_count"],
                "phop_correlated_diffusion": phop_stats["correlated_diffusion"],
                "debye_waller_uncorrelated_diffusion": dw_stats["uncorrelated_diffusion"],
                "debye_waller_correlated_diffusion": dw_stats["correlated_diffusion"],
                "observed_diffusion": observed,
                "phop_correlated_coverage": phop_stats["correlated_diffusion"] / observed,
                "debye_waller_uncorrelated_coverage": dw_stats["uncorrelated_diffusion"] / observed,
                "debye_waller_correlated_coverage": dw_stats["correlated_diffusion"] / observed,
                "macro_fit_parameter_count": 0.0,
                "event_definition": "rolling_position_variance_above_calibration_debye_waller_factor",
                "sota_algorithm_alignment": "basic_debye_waller_variance_without_high_temperature_noise_correction",
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    summary, verdict = summarize_heldout_event_transport(
        rows,
        minimum_coverage=args.minimum_coverage,
        maximum_coverage=args.maximum_coverage,
        model_columns={
            "fixed_phop_direction_correlated": "phop_correlated_diffusion",
            "debye_waller_uncorrelated": "debye_waller_uncorrelated_diffusion",
            "debye_waller_direction_correlated": "debye_waller_correlated_diffusion",
        },
        primary_model="debye_waller_direction_correlated",
    )
    verdict.update(
        {
            "temperature": float(manifest["temperature"]),
            "calibration_time": float(args.calibration_time),
            "heldout_time": float(production_time - args.calibration_time),
            "fluctuation_half_window": float(args.fluctuation_half_window),
            "phop_threshold": args.phop_threshold,
            "event_definition_claim_allowed": float(verdict["heldout_transport_pass"]),
            "sota_algorithm_alignment": "basic_debye_waller_variance_without_high_temperature_noise_correction",
            "thermodynamic_claim_allowed": 0.0,
        }
    )
    for row in summary:
        row["temperature"] = float(manifest["temperature"])
        row["calibration_time"] = float(args.calibration_time)
        row["heldout_time"] = float(production_time - args.calibration_time)
        row["thermodynamic_claim_allowed"] = 0.0
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_dw_curve_replicates.csv"), curve_rows)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_correlation_replicates.csv"),
        correlation_rows,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
