#!/usr/bin/env python3
"""Test whether the low-temperature waiting diagnosis survives jump thresholds."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_replicates import classify_waiting_diagnostic  # noqa: E402
from ka_replicates import (  # noqa: E402
    extract_debye_waller_cage_jumps,
    load_lammps_custom_trajectory,
    position_fluctuation_values,
)
from renewal_cage import waiting_time_shuffle_diagnostics  # noqa: E402


def parse_values(value: str) -> np.ndarray:
    return np.array([float(item) for item in value.split(",")], dtype=float)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one row")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _threshold_mechanism(row: dict[str, object]) -> dict[str, bool | str]:
    empirical_error = float(row["median_empirical_iid_relative_error"])
    gamma_error = float(row["median_gamma_iid_relative_error"])
    shuffle_error = float(row["median_sequence_shuffle_relative_error"])
    maximum_shuffle_error = float(row["maximum_sequence_shuffle_relative_error"])
    long_window_shuffle_error = float(
        row["long_window_sequence_shuffle_relative_error"]
    )
    memory = float(row["median_temporal_ordering_contribution_fraction"])
    environment = float(row["median_particle_identity_contribution_fraction"])
    correlation_z = float(row["median_waiting_correlation_z_vs_shuffle"])
    identifiable = float(row["persistent_environment_identifiable"]) == 1.0
    gamma_supported = gamma_error > empirical_error
    empirical_sufficient = empirical_error <= 0.15
    shuffle_sufficient = shuffle_error <= 0.15
    all_windows_sufficient = maximum_shuffle_error <= 0.15
    long_window_failure = long_window_shuffle_error > 0.15
    memory_supported = memory >= 0.05 and correlation_z >= 2.0
    environment_supported = identifiable and environment >= 0.05
    if memory_supported and environment_supported and 0.5 <= memory / environment <= 2.0:
        dominant = "mixed_particle_environment_and_event_memory"
    elif environment_supported and environment > 2.0 * memory:
        dominant = "persistent_particle_environment"
    elif memory_supported and memory > 2.0 * environment:
        dominant = "temporal_waiting_memory"
    elif shuffle_sufficient:
        dominant = "particle_conditioned_empirical_shuffle"
    elif empirical_sufficient:
        dominant = "empirical_iid_waiting_law"
    elif gamma_supported:
        dominant = "gamma_shape_misspecification"
    else:
        dominant = "unresolved"
    return {
        "gamma_supported": gamma_supported,
        "empirical_sufficient": empirical_sufficient,
        "shuffle_sufficient": shuffle_sufficient,
        "all_windows_sufficient": all_windows_sufficient,
        "long_window_failure": long_window_failure,
        "memory_supported": memory_supported,
        "memory_dominant": dominant == "temporal_waiting_memory",
        "environment_supported": environment_supported,
        "dominant": dominant,
    }


def classify_waiting_threshold_sensitivity(
    rows: list[dict[str, object]],
    *,
    finite_exchange_supported: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Require mechanism selection to agree across thresholds and replicas."""

    if not rows:
        raise ValueError("waiting threshold rows must be nonempty")
    scales = sorted({float(row["threshold_scale"]) for row in rows})
    if len(scales) < 3:
        raise ValueError("waiting threshold audit requires at least three scales")
    replicate_rows: list[dict[str, object]] = []
    all_diagnostics: list[dict[str, bool | str]] = []
    for replicate in sorted({float(row["replicate"]) for row in rows}):
        selected = sorted(
            [row for row in rows if float(row["replicate"]) == replicate],
            key=lambda row: float(row["threshold_scale"]),
        )
        if [float(row["threshold_scale"]) for row in selected] != scales:
            raise ValueError("each replicate must contain every threshold scale")
        diagnostics = [_threshold_mechanism(row) for row in selected]
        all_diagnostics.extend(diagnostics)
        dominant_values = {str(item["dominant"]) for item in diagnostics}
        replicate_rows.append(
            {
                "replicate": replicate,
                "threshold_scales": ";".join(f"{value:g}" for value in scales),
                "dominant_mechanism": (
                    next(iter(dominant_values))
                    if len(dominant_values) == 1
                    else "threshold_dependent"
                ),
                "dominant_mechanism_stable": float(len(dominant_values) == 1),
                "gamma_shape_misspecification_all_thresholds": float(
                    all(bool(item["gamma_supported"]) for item in diagnostics)
                ),
                "temporal_waiting_memory_all_thresholds": float(
                    all(bool(item["memory_supported"]) for item in diagnostics)
                ),
                "temporal_waiting_memory_dominant_any_threshold": float(
                    any(bool(item["memory_dominant"]) for item in diagnostics)
                ),
                "persistent_particle_environment_all_thresholds": float(
                    all(bool(item["environment_supported"]) for item in diagnostics)
                ),
                "maximum_sequence_shuffle_relative_error": max(
                    float(row["maximum_sequence_shuffle_relative_error"])
                    for row in selected
                ),
                "maximum_median_sequence_shuffle_relative_error": max(
                    float(row["median_sequence_shuffle_relative_error"])
                    for row in selected
                ),
                "long_window_shuffle_failure_any_threshold": float(
                    any(bool(item["long_window_failure"]) for item in diagnostics)
                ),
                "long_window_shuffle_failure_all_thresholds": float(
                    all(bool(item["long_window_failure"]) for item in diagnostics)
                ),
                "minimum_temporal_ordering_contribution_fraction": min(
                    float(row["median_temporal_ordering_contribution_fraction"])
                    for row in selected
                ),
                "minimum_particle_identity_contribution_fraction": min(
                    float(row["median_particle_identity_contribution_fraction"])
                    for row in selected
                ),
            }
        )
    dominant_values = {str(row["dominant_mechanism"]) for row in replicate_rows}
    stable = len(dominant_values) == 1 and "threshold_dependent" not in dominant_values
    dominant = next(iter(dominant_values)) if stable else "unresolved"
    gamma_supported = all(bool(item["gamma_supported"]) for item in all_diagnostics)
    empirical_sufficient = all(bool(item["empirical_sufficient"]) for item in all_diagnostics)
    shuffle_sufficient = all(bool(item["shuffle_sufficient"]) for item in all_diagnostics)
    all_windows_sufficient = all(
        bool(item["all_windows_sufficient"]) for item in all_diagnostics
    )
    long_window_failure_any = any(
        bool(item["long_window_failure"]) for item in all_diagnostics
    )
    long_window_failure_all = all(
        bool(item["long_window_failure"]) for item in all_diagnostics
    )
    memory_supported = all(bool(item["memory_supported"]) for item in all_diagnostics)
    memory_dominant = any(bool(item["memory_dominant"]) for item in all_diagnostics)
    memory_parameter_claim = memory_supported and long_window_failure_all
    environment_supported = all(bool(item["environment_supported"]) for item in all_diagnostics)
    collective = all(float(row["collective_covariance_detected"]) == 1.0 for row in rows)
    if (
        dominant in {
            "mixed_particle_environment_and_event_memory",
            "persistent_particle_environment",
        }
        and finite_exchange_supported
        and shuffle_sufficient
    ):
        implication = "finite_exchange_particle_conditioned_renewal"
    elif dominant == "temporal_waiting_memory":
        implication = "finite_memory_waiting_sequence"
    elif dominant == "empirical_iid_waiting_law":
        implication = "empirical_iid_stationary_renewal"
    else:
        implication = "no_minimal_model_selected"
    verdict: dict[str, object] = {
        "threshold_scales": ";".join(f"{value:g}" for value in scales),
        "independent_replicate_count": float(len(replicate_rows)),
        "threshold_robust_dominant_mechanism": float(stable),
        "dominant_mechanism": dominant,
        "gamma_shape_misspecification_supported": float(gamma_supported),
        "empirical_waiting_law_sufficient": float(empirical_sufficient),
        "gamma_shape_misspecification_sufficient": float(
            gamma_supported and empirical_sufficient
        ),
        "median_window_particle_conditioned_shuffle_sufficient": float(
            shuffle_sufficient
        ),
        "all_window_particle_conditioned_shuffle_sufficient": float(
            all_windows_sufficient
        ),
        "temporal_waiting_memory_supported": float(memory_supported),
        "temporal_waiting_memory_dominant": float(memory_dominant),
        "temporal_waiting_memory_parameter_claim_allowed": float(
            memory_parameter_claim
        ),
        "long_window_shuffle_failure_any_threshold": float(
            long_window_failure_any
        ),
        "long_window_shuffle_failure_all_thresholds": float(
            long_window_failure_all
        ),
        "persistent_particle_environment_supported": float(environment_supported),
        "finite_exchange_supported_by_prior_identity_decay": float(finite_exchange_supported),
        "maximum_sequence_shuffle_relative_error": max(
            float(row["maximum_sequence_shuffle_relative_error"])
            for row in replicate_rows
        ),
        "maximum_median_sequence_shuffle_relative_error": max(
            float(row["maximum_median_sequence_shuffle_relative_error"])
            for row in replicate_rows
        ),
        "minimum_temporal_ordering_contribution_fraction": min(
            float(row["minimum_temporal_ordering_contribution_fraction"])
            for row in replicate_rows
        ),
        "minimum_particle_identity_contribution_fraction": min(
            float(row["minimum_particle_identity_contribution_fraction"])
            for row in replicate_rows
        ),
        "spatial_cooperation_test_required": float(collective),
        "spatial_cooperation_proven": 0.0,
        "minimal_model_implication": implication,
        "thermodynamic_claim_allowed": 0.0,
    }
    return replicate_rows, verdict


def summarize_threshold_diagnostic_rows(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Express shuffle and particle-identity effects on one common Fano scale."""

    if not rows:
        raise ValueError("threshold diagnostic rows must be nonempty")
    summary = {"replicate": float(rows[0]["replicate"])}
    summary.update(classify_waiting_diagnostic(rows))
    actual = np.array([float(row["actual_count_fano"]) for row in rows])
    shuffled = np.array([float(row["sequence_shuffle_count_fano"]) for row in rows])
    pooled = np.array([float(row["pooled_empirical_iid_count_fano"]) for row in rows])
    shuffle_errors = np.abs(shuffled / actual - 1.0)
    longest_index = int(
        np.argmax([float(row["count_window"]) for row in rows])
    )
    summary.update(
        {
            "median_sequence_shuffle_relative_error": float(
                np.median(shuffle_errors)
            ),
            "maximum_sequence_shuffle_relative_error": float(
                np.max(shuffle_errors)
            ),
            "long_window_sequence_shuffle_relative_error": float(
                shuffle_errors[longest_index]
            ),
            "median_temporal_ordering_contribution_fraction": float(
                np.median((actual - shuffled) / actual)
            ),
            "median_particle_identity_contribution_fraction": float(
                np.median((shuffled - pooled) / actual)
            ),
        }
    )
    return summary


def write_svg(path: Path, rows: list[dict[str, object]]) -> None:
    scales = sorted({float(row["threshold_scale"]) for row in rows})
    metrics = (
        ("median_gamma_iid_relative_error", "Gamma-law error", "#D55E00"),
        ("median_empirical_iid_relative_error", "pooled empirical error", "#0072B2"),
        (
            "median_temporal_ordering_contribution_fraction",
            "ordering contribution",
            "#CC79A7",
        ),
        (
            "median_particle_identity_contribution_fraction",
            "particle-identity contribution",
            "#009E73",
        ),
    )
    width, height = 860, 480
    left, top, plot_width, plot_height = 80, 55, 680, 300
    y_min, y_max = 0.05, 10.0

    def y_position(value: float) -> float:
        fraction = (math.log10(value) - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        )
        return top + plot_height * (1.0 - fraction)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124}.grid{stroke:#DADCE0}</style>',
        '<text x="430" y="28" text-anchor="middle" font-size="18" font-weight="700">Low-temperature waiting diagnosis is threshold robust</text>',
    ]
    for value in (0.05, 0.1, 0.3, 1.0, 3.0, 10.0):
        y = y_position(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:g}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            '<text x="430" y="395" text-anchor="middle" font-size="13">Debye-Waller threshold scale</text>',
            '<text x="20" y="205" text-anchor="middle" font-size="13" transform="rotate(-90 20 205)">relative Fano effect (log scale)</text>',
        ]
    )
    for metric_index, (key, label, color) in enumerate(metrics):
        points = []
        for scale in scales:
            values = [float(row[key]) for row in rows if float(row["threshold_scale"]) == scale]
            x = left + plot_width * (scale - scales[0]) / (scales[-1] - scales[0])
            y = y_position(float(np.mean(values)))
            points.append(f"{x:.2f},{y:.2f}")
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        x = 95 + (metric_index % 2) * 390
        y = 425 + (metric_index // 2) * 28
        lines.append(f'<line x1="{x}" y1="{y}" x2="{x + 25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{x + 32}" y="{y + 5}" font-size="11">{label}</text>')
    for scale in scales:
        x = left + plot_width * (scale - scales[0]) / (scales[-1] - scales[0])
        lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle" font-size="11">{scale:.1f}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--heldout-replicates", type=Path, required=True)
    parser.add_argument("--environment-crossover-verdict", type=Path, required=True)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--threshold-scales", type=parse_values, default=parse_values("0.9,1.0,1.1"))
    parser.add_argument("--count-windows", type=parse_values, default=parse_values("50,100,200,500,1000"))
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--shuffle-replicates", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=83117)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    if len(args.threshold_scales) < 3 or np.any(args.threshold_scales <= 0.0):
        raise ValueError("at least three positive threshold scales are required")
    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    thresholds = {
        int(float(row["replicate"])): float(row["debye_waller_factor"])
        for row in read_rows(args.heldout_replicates)
    }
    window_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for replicate_spec in manifest["replicates"]:
        replicate = int(replicate_spec["replicate"])
        directory = args.ensemble_directory / str(replicate_spec["directory"])
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        calibration = positions[: args.calibration_time + 1]
        times, fluctuations = position_fluctuation_values(
            calibration, half_window=args.fluctuation_half_window
        )
        for scale_index, threshold_scale in enumerate(args.threshold_scales):
            events = extract_debye_waller_cage_jumps(
                calibration,
                debye_waller_factor=thresholds[replicate] * threshold_scale,
                half_window=args.fluctuation_half_window,
                activity_times=times,
                activity_values=fluctuations,
            )
            local_rows = []
            for window_index, count_window in enumerate(args.count_windows):
                result = waiting_time_shuffle_diagnostics(
                    events,
                    duration=float(args.calibration_time),
                    particle_count=calibration.shape[1],
                    count_window=float(count_window),
                    shuffle_replicates=args.shuffle_replicates,
                    random_seed=args.random_seed + 10000 * replicate + 100 * scale_index + window_index,
                )
                result.update(
                    {
                        "replicate": float(replicate),
                        "temperature": float(manifest["temperature"]),
                        "threshold_scale": float(threshold_scale),
                        "debye_waller_factor": thresholds[replicate] * threshold_scale,
                    }
                )
                local_rows.append(result)
                window_rows.append(result)
            summary = summarize_threshold_diagnostic_rows(local_rows)
            summary.update(
                {
                    "temperature": float(manifest["temperature"]),
                    "threshold_scale": float(threshold_scale),
                    "debye_waller_factor": thresholds[replicate] * threshold_scale,
                }
            )
            threshold_rows.append(summary)
    crossover = read_one(args.environment_crossover_verdict)
    replicate_rows, verdict = classify_waiting_threshold_sensitivity(
        threshold_rows,
        finite_exchange_supported=float(crossover["finite_exchange_environment_claim_allowed"]) == 1.0,
    )
    verdict.update(
        {
            "temperature": float(manifest["temperature"]),
            "calibration_time": float(args.calibration_time),
            "count_windows": ";".join(f"{value:g}" for value in args.count_windows),
            "shuffle_replicates": float(args.shuffle_replicates),
            "event_definition": "debye_waller_finite_duration_cage_jump",
        }
    )
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_windows.csv"), window_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_thresholds.csv"), threshold_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicate_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])
    write_svg(args.output_svg, threshold_rows)


if __name__ == "__main__":
    main()
