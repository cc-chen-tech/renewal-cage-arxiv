#!/usr/bin/env python3
"""Certify restart-level uncertainty and cooling trends for three KA temperatures."""

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

from ka_replicates import independent_group_ratio  # noqa: E402


SCALAR_METRICS = (
    "diffusion",
    "alpha_relaxation_time",
    "diffusion_alpha_product",
    "ngp_peak",
    "persistence_exchange_ratio",
)
CURVE_METRICS = ("msd", "ngp_3d", "fs_k5", "fs_k7p25", "fs_k9")
DIRECTIONS = {
    "diffusion": "decrease",
    "alpha_relaxation_time": "increase",
    "diffusion_alpha_product": "increase",
    "ngp_peak": "increase",
    "persistence_exchange_ratio": "increase",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def first_two_trajectory_timesteps(path: Path) -> tuple[int, int]:
    timesteps: list[int] = []
    with path.open() as handle:
        for line in handle:
            if line.strip() == "ITEM: TIMESTEP":
                timesteps.append(int(next(handle).strip()))
                if len(timesteps) == 2:
                    return timesteps[0], timesteps[1]
    raise ValueError(f"{path} does not contain two trajectory frames")


def physical_time_gate(
    manifest_paths: list[Path],
    manifests: list[dict[str, object]],
) -> bool:
    """Check current files against the upstream one-tau frame convention."""

    checks = []
    for manifest_path, manifest in zip(manifest_paths, manifests):
        for replicate in manifest["replicates"]:
            trajectory = (
                manifest_path.parent
                / str(replicate["directory"])
                / "trajectory.lammpstrj"
            )
            checks.append(first_two_trajectory_timesteps(trajectory) == (0, 1000))
    return bool(checks and all(checks))


def scalar_coverage_rows(
    summary_rows: list[dict[str, str]],
    event_summary_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    combined = {row["metric"]: row for row in summary_rows + event_summary_rows}
    temperature = float(summary_rows[0]["temperature"])
    rows: list[dict[str, object]] = []
    for metric in SCALAR_METRICS:
        row = combined[metric]
        mean = float(row["mean"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        count = float(row["independent_replicate_count"])
        ready = (
            row["ci95_method"] == "student_t_independent_replicates"
            and count >= 3.0
            and all(math.isfinite(value) for value in (mean, low, high))
            and low <= mean <= high
        )
        relative_half_width = (high - low) / (2.0 * max(abs(mean), 1e-15))
        rows.append(
            {
                "temperature": temperature,
                "observable_class": "scalar",
                "metric": metric,
                "point_count": 1.0,
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                "median_relative_ci_half_width": relative_half_width,
                "maximum_relative_ci_half_width": relative_half_width,
                "maximum_absolute_ci_half_width": (high - low) / 2.0,
                "independent_replicate_count": count,
                "ci95_method": row["ci95_method"],
                "uncertainty_ready": float(ready),
                "precision_metric": "relative_ci_half_width",
                "precision_threshold": 0.5,
                "precision_ready": float(relative_half_width <= 0.5),
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return rows


def curve_coverage_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    temperature = float(rows[0]["temperature"])
    output: list[dict[str, object]] = []
    for metric in CURVE_METRICS:
        selected = [row for row in rows if row["metric"] == metric]
        means = np.array([float(row["mean"]) for row in selected])
        lows = np.array([float(row["ci95_low"]) for row in selected])
        highs = np.array([float(row["ci95_high"]) for row in selected])
        counts = {float(row["independent_replicate_count"]) for row in selected}
        methods = {row["ci95_method"] for row in selected}
        relative = (highs - lows) / (2.0 * np.maximum(np.abs(means), 1e-12))
        absolute = (highs - lows) / 2.0
        ready = (
            len(selected) >= 3
            and len(counts) == 1
            and min(counts) >= 3.0
            and methods == {"student_t_independent_replicates"}
            and np.all(np.isfinite(means))
            and np.all(np.isfinite(lows))
            and np.all(np.isfinite(highs))
            and np.all(lows <= means)
            and np.all(means <= highs)
        )
        if metric == "msd":
            precision_metric = "maximum_relative_ci_half_width"
            precision_value = float(np.max(relative))
            precision_threshold = 0.35
        elif metric == "ngp_3d":
            precision_metric = "maximum_absolute_ci_half_width"
            precision_value = float(np.max(absolute))
            precision_threshold = 0.1
        else:
            precision_metric = "maximum_absolute_ci_half_width"
            precision_value = float(np.max(absolute))
            precision_threshold = 0.1
        output.append(
            {
                "temperature": temperature,
                "observable_class": "curve",
                "metric": metric,
                "point_count": float(len(selected)),
                "mean": math.nan,
                "ci95_low": math.nan,
                "ci95_high": math.nan,
                "median_relative_ci_half_width": float(np.median(relative)),
                "maximum_relative_ci_half_width": float(np.max(relative)),
                "maximum_absolute_ci_half_width": float(np.max(absolute)),
                "independent_replicate_count": min(counts),
                "ci95_method": next(iter(methods)) if len(methods) == 1 else "mixed",
                "uncertainty_ready": float(ready),
                "precision_metric": precision_metric,
                "precision_threshold": precision_threshold,
                "precision_ready": float(precision_value <= precision_threshold),
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return output


def cooling_trend_rows(
    replicate_groups: list[list[dict[str, str]]],
    event_replicate_groups: list[list[dict[str, str]]],
) -> list[dict[str, object]]:
    groups: dict[float, dict[str, list[float]]] = {}
    for replicate_rows, event_rows in zip(replicate_groups, event_replicate_groups):
        temperature = float(replicate_rows[0]["temperature"])
        groups[temperature] = {
            metric: [float(row[metric]) for row in replicate_rows]
            for metric in SCALAR_METRICS
            if metric != "persistence_exchange_ratio"
        }
        groups[temperature]["persistence_exchange_ratio"] = [
            float(row["persistence_exchange_ratio"]) for row in event_rows
        ]
    temperatures = sorted(groups, reverse=True)
    rows: list[dict[str, object]] = []
    for high_temperature, low_temperature in zip(temperatures[:-1], temperatures[1:]):
        for metric in SCALAR_METRICS:
            high = np.asarray(groups[high_temperature][metric], dtype=float)
            low = np.asarray(groups[low_temperature][metric], dtype=float)
            if DIRECTIONS[metric] == "increase":
                comparison = independent_group_ratio(
                    low, high, relative_equivalence_margin=0.1
                )
            else:
                comparison = independent_group_ratio(
                    high, low, relative_equivalence_margin=0.1
                )
            rows.append(
                {
                    "high_temperature": high_temperature,
                    "low_temperature": low_temperature,
                    "metric": metric,
                    "cooling_direction": DIRECTIONS[metric],
                    "cooling_effect_ratio": comparison["mean_ratio"],
                    "ci95_low_ratio": comparison["ci95_low_ratio"],
                    "ci95_high_ratio": comparison["ci95_high_ratio"],
                    "ci95_method": comparison["ci95_method"],
                    "high_temperature_replicate_count": float(len(high)),
                    "low_temperature_replicate_count": float(len(low)),
                    "trend_pass": comparison["growth_detected"],
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    return rows


def build_uncertainty_verdict(
    manifests: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    trend_rows: list[dict[str, object]],
    *,
    physical_time_gate_pass: bool,
) -> dict[str, object]:
    ordered_manifests = sorted(manifests, key=lambda row: float(row["temperature"]))
    temperatures = [float(manifest["temperature"]) for manifest in ordered_manifests]
    counts = [float(manifest["replicate_count"]) for manifest in ordered_manifests]
    initial_independence = all(
        float(manifest["maximum_absolute_fs_observed"])
        <= float(manifest["maximum_absolute_fs_allowed"])
        for manifest in manifests
    )
    independence_classes = {
        str(manifest["independence_class"]) for manifest in manifests
    }
    independence_class_consistent = len(independence_classes) == 1
    restart_ready = (
        len(temperatures) >= 2
        and min(counts) >= 3.0
        and initial_independence
        and independence_class_consistent
        and physical_time_gate_pass
    )
    parent_ready = restart_ready and all(
        bool(manifest["independently_prepared_parent_samples"])
        for manifest in manifests
    )

    def coverage_ready(observable_class: str, metrics: tuple[str, ...]) -> bool:
        selected = [
            row
            for row in coverage_rows
            if row["observable_class"] == observable_class
        ]
        observed = {
            (float(row["temperature"]), str(row["metric"]))
            for row in selected
            if float(row["uncertainty_ready"]) == 1.0
        }
        required = {(temperature, metric) for temperature in temperatures for metric in metrics}
        return observed == required

    scalar_ready = coverage_ready("scalar", SCALAR_METRICS)
    curve_ready = coverage_ready("curve", CURVE_METRICS)

    def precision_ready(observable_class: str, metrics: tuple[str, ...]) -> bool:
        selected = [
            row
            for row in coverage_rows
            if row["observable_class"] == observable_class
        ]
        observed = {
            (float(row["temperature"]), str(row["metric"]))
            for row in selected
            if float(row["precision_ready"]) == 1.0
        }
        required = {(temperature, metric) for temperature in temperatures for metric in metrics}
        return observed == required

    scalar_precision = precision_ready("scalar", SCALAR_METRICS)
    curve_precision = precision_ready("curve", CURVE_METRICS)
    precision_blockers = sorted(
        f"T{float(row['temperature']):g}:{row['metric']}"
        for row in coverage_rows
        if float(row["precision_ready"]) == 0.0
    )
    expected_trends = len(SCALAR_METRICS) * (len(temperatures) - 1)
    trend_chain = (
        len(trend_rows) == expected_trends
        and all(float(row["trend_pass"]) == 1.0 for row in trend_rows)
    )
    return {
        "temperature_count": float(len(temperatures)),
        "temperatures": ";".join(f"{value:g}" for value in temperatures),
        "restart_replicate_counts_by_temperature": ";".join(
            f"{temperature:g}:{count:g}"
            for temperature, count in zip(temperatures, counts)
        ),
        "minimum_restart_replicate_count": min(counts),
        "initial_configuration_decorrelation_gate_pass": float(initial_independence),
        "independence_class": (
            next(iter(independence_classes))
            if independence_class_consistent
            else "mixed"
        ),
        "physical_time_definition_consistent": float(physical_time_gate_pass),
        "saved_frame_interval_tau": 1.0,
        "physical_time_gate_source": "current_first_two_frames_plus_upstream_all_frame_spacing_gate",
        "restart_ensemble_uncertainty_ready": float(restart_ready),
        "core_scalar_uncertainty_ready": float(scalar_ready),
        "curve_uncertainty_ready": float(curve_ready),
        "core_scalar_precision_ready": float(scalar_precision),
        "curve_precision_ready": float(curve_precision),
        "precision_blockers": ";".join(precision_blockers) or "none",
        "cooling_trend_pass_count": float(
            sum(float(row["trend_pass"]) == 1.0 for row in trend_rows)
        ),
        "cooling_trend_test_count": float(len(trend_rows)),
        "three_temperature_trend_chain_pass": float(trend_chain),
        "independently_prepared_parent_ensemble_ready": float(parent_ready),
        "allowed_claim_level": (
            "three_temperature_restart_ensemble_dynamical_trends"
            if restart_ready and scalar_ready and curve_ready
            else "insufficient_restart_ensemble_uncertainty"
        ),
        "next_required_action": (
            "independent_parent_samples_and_low_temperature_curve_precision"
            if not parent_ready or not curve_precision
            else "calibration_heldout_multivariate_prediction"
        ),
        "thermodynamic_claim_allowed": 0.0,
    }


def write_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 900, 500
    left, top, plot_width, plot_height = 85, 55, 740, 310
    transitions = sorted(
        {(float(row["high_temperature"]), float(row["low_temperature"])) for row in rows},
        reverse=True,
    )
    metrics = list(SCALAR_METRICS)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6B7280")
    maximum = max(float(row["ci95_high_ratio"]) for row in rows)
    y_min, y_max = 0.8, 1.25 * maximum

    def y_position(value: float) -> float:
        fraction = (math.log(value) - math.log(y_min)) / (math.log(y_max) - math.log(y_min))
        return top + plot_height * (1.0 - fraction)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124}.grid{stroke:#DADCE0}</style>',
        '<text x="450" y="28" text-anchor="middle" font-size="18" font-weight="700">Independent-restart cooling trends with 95% intervals</text>',
    ]
    for value in (1.0, 3.0, 10.0, 30.0, 100.0):
        if value > y_max:
            continue
        y = y_position(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:g}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            '<text x="22" y="210" text-anchor="middle" font-size="13" transform="rotate(-90 22 210)">cooling effect factor (log scale)</text>',
        ]
    )
    group_width = plot_width / len(transitions)
    for transition_index, transition in enumerate(transitions):
        center = left + group_width * (transition_index + 0.5)
        for metric_index, (metric, color) in enumerate(zip(metrics, colors)):
            row = next(
                row
                for row in rows
                if (float(row["high_temperature"]), float(row["low_temperature"])) == transition
                and row["metric"] == metric
            )
            x = center + (metric_index - 2) * 42
            low = max(float(row["ci95_low_ratio"]), y_min)
            high = float(row["ci95_high_ratio"])
            mean = float(row["cooling_effect_ratio"])
            lines.append(f'<line x1="{x:.2f}" y1="{y_position(low):.2f}" x2="{x:.2f}" y2="{y_position(high):.2f}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<circle cx="{x:.2f}" cy="{y_position(mean):.2f}" r="4" fill="{color}"/>')
        lines.append(
            f'<text x="{center:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="12">T={transition[0]:g} to {transition[1]:g}</text>'
        )
    labels = ("D slowdown", "tau_alpha", "D tau_alpha", "NGP peak", "tau_p/tau_x")
    for index, (label, color) in enumerate(zip(labels, colors)):
        x = 100 + (index % 3) * 275
        y = 425 + (index // 3) * 28
        lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{x + 12}" y="{y + 5}" font-size="11">{label}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs=3, required=True)
    parser.add_argument("--replicates", type=Path, nargs=3, required=True)
    parser.add_argument("--summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--event-replicates", type=Path, nargs=3, required=True)
    parser.add_argument("--event-summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--curve-summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()

    manifests = [json.loads(path.read_text()) for path in args.manifests]
    summaries = [read_rows(path) for path in args.summaries]
    event_summaries = [read_rows(path) for path in args.event_summaries]
    coverage: list[dict[str, object]] = []
    for summary, event_summary, curve_path in zip(
        summaries, event_summaries, args.curve_summaries
    ):
        coverage.extend(scalar_coverage_rows(summary, event_summary))
        coverage.extend(curve_coverage_rows(read_rows(curve_path)))
    trends = cooling_trend_rows(
        [read_rows(path) for path in args.replicates],
        [read_rows(path) for path in args.event_replicates],
    )
    verdict = build_uncertainty_verdict(
        manifests,
        coverage,
        trends,
        physical_time_gate_pass=physical_time_gate(args.manifests, manifests),
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_coverage.csv"),
        coverage,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_trends.csv"),
        trends,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"),
        [verdict],
    )
    write_svg(args.output_svg, trends)


if __name__ == "__main__":
    main()
