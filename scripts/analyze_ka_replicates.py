#!/usr/bin/env python3
"""Reduce completed KA replicate dumps to independent-sample observables."""

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
    independent_sample_ci95,
    load_lammps_custom_trajectory,
    summarize_replicate_curves,
)
from renewal_cage import (  # noqa: E402
    block_trajectory_observables,
    event_clock_statistics,
    extract_nonrecrossing_phop_events,
    phop_values,
    summarize_block_trajectory_observables,
    waiting_time_shuffle_diagnostics,
)


def parse_numbers(value: str, dtype: type[float] | type[int]) -> np.ndarray:
    return np.array([dtype(item) for item in value.split(",")], dtype=dtype)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_event_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = (
        "event_rate",
        "exchange_mean",
        "exchange_cv2",
        "stationary_persistence_mean",
        "persistence_exchange_ratio",
        "jump_squared_mean",
        "count_fano",
        "correlated_diffusion",
    )
    summary: list[dict[str, object]] = []
    for metric in metrics:
        values = np.array([float(row[metric]) for row in rows])
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(values))
        ci_low, ci_high, critical = independent_sample_ci95(
            mean=mean,
            standard_error=standard_error,
            sample_count=len(values),
        )
        summary.append(
            {
                "metric": metric,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "ci95_critical_value": critical,
                "ci95_method": "student_t_independent_replicates",
                "independent_replicate_count": float(len(values)),
                "temperature": float(rows[0]["temperature"]),
                "threshold": float(rows[0]["threshold"]),
                "half_window": float(rows[0]["half_window"]),
                "independence_class": str(rows[0]["independence_class"]),
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return summary


def classify_waiting_diagnostic(rows: list[dict[str, object]]) -> dict[str, object]:
    empirical_error = float(np.median([float(row["empirical_iid_relative_error"]) for row in rows]))
    gamma_error = float(np.median([float(row["gamma_iid_relative_error"]) for row in rows]))
    memory_excess = float(np.median([float(row["temporal_memory_excess_fraction"]) for row in rows]))
    environment_excess = float(
        np.median([float(row["persistent_environment_excess_fraction"]) for row in rows])
    )
    correlation_z = float(
        np.median(
            [
                abs(
                    (float(row["waiting_lag1_correlation"]) - float(row["shuffle_lag1_correlation_mean"]))
                    / float(row["shuffle_lag1_correlation_standard_deviation"])
                )
                for row in rows
                if float(row["shuffle_lag1_correlation_standard_deviation"]) > 0.0
            ]
        )
    )
    complete_fraction = float(
        np.median([float(row["complete_wait_particle_fraction"]) for row in rows])
    )
    waits_per_supported_particle = float(
        np.median(
            [
                float(row["complete_waiting_time_count"])
                / float(row["particles_with_complete_wait"])
                for row in rows
                if float(row["particles_with_complete_wait"]) > 0.0
            ]
        )
    )
    environment_identifiable = complete_fraction >= 0.8 and waits_per_supported_particle >= 10.0
    collective_rows = [row for row in rows if float(row["window_count"]) >= 4.0]
    minimum_collective_ratio = float(
        min(float(row["collective_covariance_ratio"]) for row in collective_rows)
    )
    collective_detected = minimum_collective_ratio > 1.0
    if empirical_error <= 0.15:
        verdict = "empirical_iid_waiting_law_sufficient"
    elif memory_excess > 0.20 and correlation_z >= 2.0:
        verdict = "temporal_waiting_memory_required"
    elif environment_excess > 0.20:
        verdict = "persistent_particle_environment_required"
    elif gamma_error - empirical_error > 0.15:
        verdict = "gamma_shape_failure_empirical_law_improves"
    else:
        verdict = "waiting_failure_unresolved"
    return {
        "median_empirical_iid_relative_error": empirical_error,
        "median_gamma_iid_relative_error": gamma_error,
        "median_temporal_memory_excess_fraction": memory_excess,
        "median_persistent_environment_excess_fraction": environment_excess,
        "median_waiting_correlation_z_vs_shuffle": correlation_z,
        "median_complete_wait_particle_fraction": complete_fraction,
        "median_complete_waits_per_supported_particle": waits_per_supported_particle,
        "persistent_environment_identifiable": float(environment_identifiable),
        "environment_support_gate": "posthoc_conservative_fraction_0p8_and_ten_waits",
        "minimum_collective_covariance_ratio_supported_windows": minimum_collective_ratio,
        "collective_covariance_detected": float(collective_detected),
        "waiting_failure_verdict": verdict,
    }


def summarize_waiting_diagnostic_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    replicate_rows: list[dict[str, object]] = []
    for replicate in sorted({float(row["replicate"]) for row in rows}):
        selected = [row for row in rows if float(row["replicate"]) == replicate]
        summary = {"replicate": replicate}
        summary.update(classify_waiting_diagnostic(selected))
        replicate_rows.append(summary)
    verdicts = [str(row["waiting_failure_verdict"]) for row in replicate_rows]
    consensus = verdicts[0] if len(set(verdicts)) == 1 else "replicate_verdicts_disagree"
    consensus_count = max(verdicts.count(verdict) for verdict in set(verdicts))
    verdict: dict[str, object] = {
        "consensus_verdict": consensus,
        "consensus_replicate_count": float(consensus_count),
        "independent_replicate_count": float(len(replicate_rows)),
        "all_replicates_agree": float(len(set(verdicts)) == 1),
        "finite_memory_model_required": float(
            consensus == "temporal_waiting_memory_required"
        ),
        "persistent_environment_identifiable": float(
            all(float(row["persistent_environment_identifiable"]) == 1.0 for row in replicate_rows)
        ),
        "collective_covariance_replicate_count": float(
            sum(float(row["collective_covariance_detected"]) == 1.0 for row in replicate_rows)
        ),
        "spatial_cooperation_test_required": float(
            all(float(row["collective_covariance_detected"]) == 1.0 for row in replicate_rows)
        ),
        "spatial_cooperation_proven": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for metric in (
        "median_empirical_iid_relative_error",
        "median_gamma_iid_relative_error",
        "median_temporal_memory_excess_fraction",
        "median_persistent_environment_excess_fraction",
        "median_waiting_correlation_z_vs_shuffle",
    ):
        values = np.array([float(row[metric]) for row in replicate_rows])
        verdict[f"mean_{metric}"] = float(np.mean(values))
        verdict[f"standard_error_{metric}"] = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    return replicate_rows, verdict


def analyze_ensemble(
    ensemble_directory: Path,
    *,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    diffusion_lag: int,
    overlap_radius: float,
    origin_stride: int,
    waiting_count_windows: np.ndarray,
    waiting_shuffle_replicates: int,
    waiting_random_seed: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    manifest = json.loads((ensemble_directory / "ensemble_manifest.json").read_text())
    expected_replicates = int(manifest["replicate_count"])
    curve_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    waiting_rows: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        timesteps = trajectory["timesteps"]
        if len(timesteps) < 2 or not np.all(np.diff(timesteps) == 1000):
            raise ValueError("replicate frames must be separated by exactly one tau")
        production_time = float(manifest["production_time_tau"])
        if timesteps[0] != 0 or timesteps[-1] != int(round(production_time * 1000.0)):
            raise ValueError("replicate does not cover the preregistered production window")
        positions = trajectory["unwrapped_positions"]
        particle_mask = trajectory["particle_types"] == 0
        rows = block_trajectory_observables(
            positions,
            lags=lags,
            block_size=len(positions) - 1,
            wave_numbers=wave_numbers,
            overlap_radius=overlap_radius,
            particle_mask=particle_mask,
            origin_stride=origin_stride,
        )
        for row in rows:
            row["block_index"] = float(replicate_index - 1)
            row["replicate"] = float(replicate_index)
            row["temperature"] = float(manifest["temperature"])
            row["independence_class"] = str(manifest["independence_class"])
            row["thermodynamic_claim_allowed"] = 0.0
        curve_rows.extend(rows)
        a_positions = positions[:, particle_mask]
        activity_times, activity_values = phop_values(a_positions, half_window=5)
        events = extract_nonrecrossing_phop_events(
            a_positions,
            threshold=0.2,
            half_window=5,
            recrossing_radius=math.sqrt(0.2),
            activity_times=activity_times,
            activity_values=activity_values,
        )
        event_row: dict[str, object] = {
            "replicate": float(replicate_index),
            "temperature": float(manifest["temperature"]),
            "threshold": 0.2,
            "half_window": 5.0,
            "event_definition": "candelier_phop_contiguous_peak_recursive_ABA_removal",
            "independence_class": str(manifest["independence_class"]),
            "thermodynamic_claim_allowed": 0.0,
        }
        event_row.update(
            event_clock_statistics(
                events,
                duration=production_time,
                particle_count=int(np.sum(particle_mask)),
                dimension=3,
            )
        )
        event_rows.append(event_row)
        for window_index, count_window in enumerate(waiting_count_windows):
            waiting_row: dict[str, object] = waiting_time_shuffle_diagnostics(
                events,
                duration=production_time,
                particle_count=int(np.sum(particle_mask)),
                count_window=float(count_window),
                shuffle_replicates=waiting_shuffle_replicates,
                random_seed=waiting_random_seed + 1000 * replicate_index + window_index,
            )
            waiting_row.update(
                {
                    "replicate": float(replicate_index),
                    "temperature": float(manifest["temperature"]),
                    "threshold": 0.2,
                    "half_window": 5.0,
                    "independence_class": str(manifest["independence_class"]),
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
            waiting_rows.append(waiting_row)

    if len({int(row["replicate"]) for row in curve_rows}) != expected_replicates:
        raise ValueError("replicate count does not match the ensemble manifest")
    fs_keys = [f"fs_k{wave_number:g}".replace(".", "p") for wave_number in wave_numbers]
    curve_summary_rows = summarize_replicate_curves(
        curve_rows,
        metric_keys=["msd", "ngp_3d", "overlap_mean", "overlap_chi4", *fs_keys],
    )
    for row in curve_summary_rows:
        row["temperature"] = float(manifest["temperature"])
        row["independence_class"] = str(manifest["independence_class"])
        row["thermodynamic_claim_allowed"] = 0.0
    fs_key = f"fs_k{wave_numbers[np.argmin(abs(wave_numbers - 7.25))]:g}".replace(".", "p")
    replicate_rows, summary_rows = summarize_block_trajectory_observables(
        curve_rows,
        fs_key=fs_key,
        diffusion_lag=diffusion_lag,
    )
    for row in replicate_rows:
        row["replicate"] = float(row.pop("block_index") + 1.0)
        row["temperature"] = float(manifest["temperature"])
        row["independence_class"] = str(manifest["independence_class"])
        row["thermodynamic_claim_allowed"] = 0.0
    for row in summary_rows:
        ci_low, ci_high, critical = independent_sample_ci95(
            mean=float(row["mean"]),
            standard_error=float(row["standard_error"]),
            sample_count=expected_replicates,
        )
        row["ci95_low"] = ci_low
        row["ci95_high"] = ci_high
        row["ci95_critical_value"] = critical
        row["ci95_method"] = "student_t_independent_replicates"
        row["temperature"] = float(manifest["temperature"])
        row["independent_replicate_count"] = float(expected_replicates)
        row["independence_class"] = str(manifest["independence_class"])
        row["maximum_absolute_initial_fs"] = float(manifest["maximum_absolute_fs_observed"])
        row["thermodynamic_claim_allowed"] = 0.0
    waiting_replicates, waiting_verdict = summarize_waiting_diagnostic_rows(waiting_rows)
    for row in waiting_replicates:
        row["temperature"] = float(manifest["temperature"])
        row["thermodynamic_claim_allowed"] = 0.0
    waiting_verdict["temperature"] = float(manifest["temperature"])
    return (
        curve_rows,
        curve_summary_rows,
        replicate_rows,
        summary_rows,
        event_rows,
        summarize_event_rows(event_rows),
        waiting_rows,
        waiting_replicates,
        [waiting_verdict],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--lags", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--wave-numbers", default="5,7.25,9")
    parser.add_argument("--diffusion-lag", type=int, default=512)
    parser.add_argument("--overlap-radius", type=float, default=0.3)
    parser.add_argument("--origin-stride", type=int, default=4)
    parser.add_argument("--waiting-count-windows", default="128,256,512")
    parser.add_argument("--waiting-shuffle-replicates", type=int, default=32)
    parser.add_argument("--waiting-random-seed", type=int, default=1729)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    (
        curves,
        curve_summary,
        replicates,
        summary,
        events,
        event_summary,
        waiting,
        waiting_replicates,
        waiting_verdict,
    ) = analyze_ensemble(
        args.ensemble_directory,
        lags=parse_numbers(args.lags, int),
        wave_numbers=parse_numbers(args.wave_numbers, float),
        diffusion_lag=args.diffusion_lag,
        overlap_radius=args.overlap_radius,
        origin_stride=args.origin_stride,
        waiting_count_windows=parse_numbers(args.waiting_count_windows, float),
        waiting_shuffle_replicates=args.waiting_shuffle_replicates,
        waiting_random_seed=args.waiting_random_seed,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curves.csv"), curves)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_curve_summary.csv"),
        curve_summary,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicates)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_event_replicates.csv"), events)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_event_summary.csv"),
        event_summary,
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_waiting.csv"), waiting)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_waiting_replicates.csv"),
        waiting_replicates,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_waiting_verdict.csv"),
        waiting_verdict,
    )


if __name__ == "__main__":
    main()
