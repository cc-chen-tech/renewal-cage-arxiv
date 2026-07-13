#!/usr/bin/env python3
"""Test whether low-temperature event-rate drift is a jump-threshold artifact."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_rate_stability import exact_rate_trend  # noqa: E402
from ka_replicates import (  # noqa: E402
    extract_debye_waller_cage_jumps,
    load_lammps_custom_trajectory,
    particle_event_count_matrix,
    position_fluctuation_values,
)


def parse_scales(value: str) -> np.ndarray:
    return np.array([float(item) for item in value.split(",")], dtype=float)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_threshold_stability(
    rows: list[dict[str, object]],
) -> dict[str, float]:
    """Require a common trend sign and bounded amplitude under threshold perturbation."""

    if len(rows) < 3:
        raise ValueError("threshold stability requires at least three scales")
    ordered = sorted(rows, key=lambda row: float(row["threshold_scale"]))
    changes = np.array(
        [float(row["normalized_total_linear_change"]) for row in ordered]
    )
    nonzero = changes[np.abs(changes) > 1e-12]
    sign_stable = bool(len(nonzero) == len(changes) and np.all(np.sign(nonzero) == np.sign(nonzero[0])))
    amplitude_span = float(np.max(changes) - np.min(changes))
    amplitude_stable = amplitude_span <= 0.10
    return {
        "trend_sign_stable_across_thresholds": float(sign_stable),
        "trend_amplitude_span_across_thresholds": amplitude_span,
        "maximum_allowed_trend_amplitude_span": 0.10,
        "trend_amplitude_stable_across_thresholds": float(amplitude_stable),
        "threshold_robust_trend": float(sign_stable and amplitude_stable),
        "minimum_absolute_total_linear_change": float(np.min(np.abs(changes))),
        "maximum_absolute_total_linear_change": float(np.max(np.abs(changes))),
    }


def write_svg(path: Path, replicate_rows: list[dict[str, object]]) -> None:
    width, height = 800, 430
    left, top, plot_width, plot_height = 75, 55, 660, 285
    y_min, y_max = -0.18, 0.30
    colors = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124;stroke-width:1}.grid{stroke:#DADCE0;stroke-width:1}</style>',
        '<text x="400" y="27" text-anchor="middle" font-size="18" font-weight="700">Low-temperature rate trend is threshold robust</text>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4.0
        y = top + plot_height - plot_height * (value - y_min) / (y_max - y_min)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:.2f}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="375" text-anchor="middle" font-size="13">Debye-Waller threshold scale</text>',
            '<text x="18" y="200" text-anchor="middle" font-size="13" transform="rotate(-90 18 200)">normalized six-window linear change</text>',
        ]
    )
    scales = sorted({float(row["threshold_scale"]) for row in replicate_rows})
    for replicate in sorted({int(float(row["replicate"])) for row in replicate_rows}):
        local = sorted(
            [row for row in replicate_rows if int(float(row["replicate"])) == replicate],
            key=lambda row: float(row["threshold_scale"]),
        )
        points = " ".join(
            f'{left + plot_width * (float(row["threshold_scale"]) - scales[0]) / (scales[-1] - scales[0]):.2f},{top + plot_height - plot_height * (float(row["normalized_total_linear_change"]) - y_min) / (y_max - y_min):.2f}'
            for row in local
        )
        lines.append(f'<polyline fill="none" stroke="{colors[replicate]}" stroke-width="2.5" points="{points}"/>')
        for row in local:
            x = left + plot_width * (float(row["threshold_scale"]) - scales[0]) / (scales[-1] - scales[0])
            y = top + plot_height - plot_height * (float(row["normalized_total_linear_change"]) - y_min) / (y_max - y_min)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors[replicate]}"/>')
    for scale in scales:
        x = left + plot_width * (scale - scales[0]) / (scales[-1] - scales[0])
        lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle" font-size="11">{scale:.1f}</text>')
    for index, replicate in enumerate((1, 2, 3)):
        x = 220 + index * 155
        lines.append(f'<line x1="{x}" y1="405" x2="{x + 28}" y2="405" stroke="{colors[replicate]}" stroke-width="3"/>')
        lines.append(f'<text x="{x + 36}" y="410" font-size="11">replicate {replicate}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--heldout-replicates", type=Path, required=True)
    parser.add_argument("--threshold-scales", type=parse_scales, default=parse_scales("0.9,1.0,1.1"))
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--block-size", type=float, default=20.0)
    parser.add_argument("--rate-window-count", type=int, default=6)
    parser.add_argument("--failed-rate-replicate", type=int, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    if (
        len(args.threshold_scales) < 3
        or np.any(args.threshold_scales <= 0.0)
        or not np.all(np.diff(args.threshold_scales) > 0.0)
    ):
        raise ValueError("threshold scales must be increasing positive values")
    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    duration = float(manifest["production_time_tau"])
    thresholds = {
        int(float(row["replicate"])): float(row["debye_waller_factor"])
        for row in read_rows(args.heldout_replicates)
    }
    window_rows: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    for replicate_spec in manifest["replicates"]:
        replicate = int(replicate_spec["replicate"])
        directory = args.ensemble_directory / str(replicate_spec["directory"])
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        times, fluctuations = position_fluctuation_values(
            positions,
            half_window=args.fluctuation_half_window,
        )
        for threshold_scale in args.threshold_scales:
            events = extract_debye_waller_cage_jumps(
                positions,
                debye_waller_factor=thresholds[replicate] * threshold_scale,
                half_window=args.fluctuation_half_window,
                activity_times=times,
                activity_values=fluctuations,
            )
            counts = particle_event_count_matrix(
                events,
                duration=duration,
                particle_count=positions.shape[1],
                block_size=args.block_size,
            )
            windows = np.array_split(np.arange(counts.shape[1]), args.rate_window_count)
            values = np.array(
                [float(np.mean(counts[:, block_indices])) for block_indices in windows]
            )
            trend = exact_rate_trend(values)
            trend.update(
                {
                    "replicate": float(replicate),
                    "temperature": float(manifest["temperature"]),
                    "threshold_scale": float(threshold_scale),
                    "debye_waller_factor": thresholds[replicate] * threshold_scale,
                    "block_size": args.block_size,
                    "event_count": float(np.sum(counts)),
                }
            )
            replicate_rows.append(trend)
            for window_index, value in enumerate(values):
                window_rows.append(
                    {
                        "replicate": float(replicate),
                        "temperature": float(manifest["temperature"]),
                        "threshold_scale": float(threshold_scale),
                        "block_size": args.block_size,
                        "rate_window_index": float(window_index),
                        "mean_count_per_particle_block": value,
                        "normalized_rate": value / float(np.mean(values)),
                    }
                )
    stability_rows: list[dict[str, object]] = []
    for replicate in sorted({int(float(row["replicate"])) for row in replicate_rows}):
        local = [row for row in replicate_rows if int(float(row["replicate"])) == replicate]
        stability = classify_threshold_stability(local)
        stability.update(
            {
                "replicate": float(replicate),
                "minimum_exact_permutation_p_value": min(
                    float(row["exact_two_sided_permutation_p_value"]) for row in local
                ),
                "strict_trend_at_any_threshold": float(
                    any(float(row["strict_trend_detected"]) == 1.0 for row in local)
                ),
                "borderline_trend_at_all_thresholds": float(
                    all(float(row["borderline_trend_detected"]) == 1.0 for row in local)
                ),
            }
        )
        stability_rows.append(stability)
    failed = next(
        row
        for row in stability_rows
        if int(float(row["replicate"])) == args.failed_rate_replicate
    )
    verdict = {
        "temperature": float(manifest["temperature"]),
        "independent_replicate_count": float(len(stability_rows)),
        "threshold_scales": ";".join(f"{value:g}" for value in args.threshold_scales),
        "failed_rate_replicate": float(args.failed_rate_replicate),
        "failed_replicate_threshold_robust_trend": failed["threshold_robust_trend"],
        "failed_replicate_strict_trend_at_any_threshold": failed[
            "strict_trend_at_any_threshold"
        ],
        "failed_replicate_borderline_trend_at_all_thresholds": failed[
            "borderline_trend_at_all_thresholds"
        ],
        "failed_replicate_minimum_p_value": failed[
            "minimum_exact_permutation_p_value"
        ],
        "jump_threshold_artifact_supported": float(
            float(failed["threshold_robust_trend"]) == 0.0
        ),
        "threshold_robust_rate_anomaly_detected": float(
            float(failed["threshold_robust_trend"]) == 1.0
        ),
        "systematic_rate_nonstationarity_claim_allowed": 0.0,
        "new_rate_state_parameter_claim_allowed": 0.0,
        "macro_observable_prediction_claim_allowed": 0.0,
        "next_required_test": "new_independent_low_temperature_trajectory_or_longer_third_window",
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_windows.csv"), window_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_replicates.csv"), replicate_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_stability.csv"), stability_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])
    write_svg(args.output_svg, replicate_rows)


if __name__ == "__main__":
    main()
