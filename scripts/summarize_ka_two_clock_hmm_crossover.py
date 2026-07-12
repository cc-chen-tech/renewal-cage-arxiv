#!/usr/bin/env python3
"""Summarize the cooling crossover to HMM emissions with two exchange clocks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain one data row")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_crossover(
    *,
    high_second_clock_selection: float,
    low_second_clock_selection: float,
    low_shape_pass: float,
    low_full_pass: float,
) -> dict[str, float]:
    selected = bool(
        high_second_clock_selection == 0.0
        and low_second_clock_selection == 1.0
        and low_shape_pass == 1.0
    )
    return {
        "cooling_induced_hybrid_clock_selected": float(selected),
        "conditional_event_shape_crossover_closure": float(selected),
        "absolute_rate_drift_unresolved": float(selected and low_full_pass < 1.0),
        "macro_observable_prediction_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def aggregate_curve(rows: list[dict[str, str]], block_size: float) -> list[dict[str, float]]:
    selected = [row for row in rows if float(row["block_size"]) == block_size]
    result = []
    for lag in sorted({float(row["lag_time"]) for row in selected}):
        local = [row for row in selected if float(row["lag_time"]) == lag]
        result.append(
            {
                "lag_time": lag,
                "heldout": float(np.mean([float(row["heldout_identity_correlation"]) for row in local])),
                "hybrid": float(np.mean([float(row["hybrid_prediction"]) for row in local])),
                "markov": float(np.mean([float(row["markov_hmm_prediction"]) for row in local])),
            }
        )
    return result


def write_svg(
    path: Path,
    high_curve: list[dict[str, float]],
    low_curve: list[dict[str, float]],
    low_hybrid_rows: list[dict[str, str]],
    low_gamma_rows: list[dict[str, str]],
    *,
    block_size: float,
) -> None:
    width, height = 920, 660
    lefts = [70, 515]
    panel_width = 355
    top, curve_height = 55, 270
    colors = {"heldout": "#202124", "hybrid": "#009E73", "markov": "#0072B2"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124;stroke-width:1}.grid{stroke:#DADCE0;stroke-width:1}.obs{fill:none;stroke-width:3}.model{fill:none;stroke-width:2;stroke-dasharray:7 5}</style>',
        '<text x="460" y="27" text-anchor="middle" font-size="18" font-weight="700">Two exchange clocks reconcile local emissions and long memory</text>',
    ]
    for panel, (label, curve) in enumerate((("T = 0.58", high_curve), ("T = 0.45", low_curve))):
        left = lefts[panel]
        max_time = max(row["lag_time"] for row in curve)
        max_y = max(0.08, max(row[key] for row in curve for key in colors) * 1.08)
        for tick in range(5):
            value = max_y * tick / 4.0
            y = top + curve_height - curve_height * value / max_y
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" y2="{y:.2f}"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:.2f}</text>')
        lines.extend(
            [
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + curve_height}"/>',
                f'<line class="axis" x1="{left}" y1="{top + curve_height}" x2="{left + panel_width}" y2="{top + curve_height}"/>',
                f'<text x="{left + panel_width / 2:.2f}" y="{top - 13}" text-anchor="middle" font-size="15" font-weight="700">{label}: identity</text>',
                f'<text x="{left + panel_width / 2:.2f}" y="{top + curve_height + 35}" text-anchor="middle" font-size="12">lag time (block size = {block_size:g})</text>',
            ]
        )
        for tick in range(5):
            value = max_time * tick / 4.0
            x = left + panel_width * tick / 4.0
            lines.append(f'<text x="{x:.2f}" y="{top + curve_height + 18}" text-anchor="middle" font-size="10">{value:.0f}</text>')
        for key in colors:
            points = " ".join(
                f'{left + panel_width * row["lag_time"] / max_time:.2f},{top + curve_height - curve_height * row[key] / max_y:.2f}'
                for row in curve
            )
            css = "obs" if key == "heldout" else "model"
            lines.append(f'<polyline class="{css}" stroke="{colors[key]}" points="{points}"/>')

    legend = (("heldout", "held-out"), ("hybrid", "two-clock HMM"), ("markov", "single-clock HMM"))
    for index, (key, label) in enumerate(legend):
        x = 205 + index * 205
        dash = "" if key == "heldout" else ' stroke-dasharray="7 5"'
        lines.append(f'<line x1="{x}" y1="365" x2="{x + 28}" y2="365" stroke="{colors[key]}" stroke-width="3"{dash}/>')
        lines.append(f'<text x="{x + 36}" y="370" font-size="12">{label}</text>')

    hybrid_by_key = {
        (row["replicate"], row["block_size"]): row for row in low_hybrid_rows
    }
    gamma_by_key = {(row["replicate"], row["block_size"]): row for row in low_gamma_rows}
    keys = sorted(hybrid_by_key, key=lambda key: (float(key[1]), float(key[0])))
    bottom, metric_height = 430, 170
    for panel, (title, hybrid_metric, baseline_metric, threshold) in enumerate(
        (
            ("Low-T identity RMSE", "identity_rmse", "hmm_identity_rmse", 0.05),
            ("Low-T lag-1 pair TV", "pair_tv_distance", "two_clock_pair_tv_distance", 0.03),
        )
    ):
        left = lefts[panel]
        maximum = 0.08
        lines.extend(
            [
                f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom + metric_height}"/>',
                f'<line class="axis" x1="{left}" y1="{bottom + metric_height}" x2="{left + panel_width}" y2="{bottom + metric_height}"/>',
                f'<text x="{left + panel_width / 2:.2f}" y="{bottom - 14}" text-anchor="middle" font-size="14" font-weight="700">{title}</text>',
            ]
        )
        threshold_y = bottom + metric_height - metric_height * threshold / maximum
        lines.append(f'<line x1="{left}" y1="{threshold_y:.2f}" x2="{left + panel_width}" y2="{threshold_y:.2f}" stroke="#C5221F" stroke-width="1.5" stroke-dasharray="5 4"/>')
        for index, key in enumerate(keys):
            x = left + (index + 0.5) * panel_width / len(keys)
            hybrid_value = float(hybrid_by_key[key][hybrid_metric])
            if panel == 0:
                baseline_value = float(gamma_by_key[key][baseline_metric])
            else:
                baseline_value = float(gamma_by_key[key][baseline_metric])
            for offset, value, color in ((-7, baseline_value, "#7B61A8"), (7, hybrid_value, "#009E73")):
                y = bottom + metric_height - metric_height * min(value, maximum) / maximum
                lines.append(f'<circle cx="{x + offset:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            lines.append(f'<text x="{x:.2f}" y="{bottom + metric_height + 16}" text-anchor="middle" font-size="9">{int(float(key[0]))}/{int(float(key[1]))}</text>')
        lines.append(f'<text x="{left - 8}" y="{bottom + 4}" text-anchor="end" font-size="10">{maximum:.2f}</text>')
        lines.append(f'<text x="{left - 8}" y="{bottom + metric_height + 4}" text-anchor="end" font-size="10">0</text>')
    lines.extend(
        [
            '<circle cx="350" cy="635" r="4" fill="#7B61A8"/><text x="360" y="639" font-size="11">single mechanism</text>',
            '<circle cx="500" cy="635" r="4" fill="#009E73"/><text x="510" y="639" font-size="11">hybrid</text>',
        ]
    )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-prefix", type=Path, required=True)
    parser.add_argument("--low-prefix", type=Path, required=True)
    parser.add_argument("--low-gamma-replicates", type=Path, required=True)
    parser.add_argument("--plot-block-size", type=float, default=50.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    high = read_one(args.high_prefix.with_name(args.high_prefix.name + "_verdict.csv"))
    low = read_one(args.low_prefix.with_name(args.low_prefix.name + "_verdict.csv"))
    if (
        float(high["calibration_horizon"]) != float(low["calibration_horizon"])
        or float(high["scoring_horizon"]) != float(low["scoring_horizon"])
        or high["common_block_sizes"] != low["common_block_sizes"]
    ):
        raise ValueError("temperature comparison requires common horizons and block sizes")
    verdict = classify_crossover(
        high_second_clock_selection=float(
            high["second_clock_calibration_selection_fraction"]
        ),
        low_second_clock_selection=float(
            low["second_clock_calibration_selection_fraction"]
        ),
        low_shape_pass=float(low["conditional_shape_distribution_pass_fraction"]),
        low_full_pass=float(low["full_hybrid_transfer_pass_fraction"]),
    )
    low_rows = read_rows(args.low_prefix.with_name(args.low_prefix.name + "_replicates.csv"))
    drift_replicates = sorted(
        {
            int(float(row["replicate"]))
            for row in low_rows
            if row["primary_failure"] == "mean_count_drift"
        }
    )
    verdict.update(
        {
            "common_calibration_horizon": float(high["calibration_horizon"]),
            "common_scoring_horizon": float(high["scoring_horizon"]),
            "common_block_sizes": high["common_block_sizes"],
            "high_full_hybrid_pass_fraction": float(
                high["full_hybrid_transfer_pass_fraction"]
            ),
            "low_conditional_shape_pass_fraction": float(
                low["conditional_shape_distribution_pass_fraction"]
            ),
            "low_full_hybrid_pass_fraction": float(
                low["full_hybrid_transfer_pass_fraction"]
            ),
            "low_maximum_fano_relative_error": float(low["maximum_fano_relative_error"]),
            "low_maximum_count_tv_distance": float(low["maximum_count_tv_distance"]),
            "low_maximum_identity_rmse": float(low["maximum_identity_rmse"]),
            "low_maximum_pair_tv_distance": float(low["maximum_pair_tv_distance"]),
            "rate_drift_replicates": ";".join(map(str, drift_replicates))
            if drift_replicates
            else "none",
            "next_required_test": "third_window_or_new_trajectory_test_of_low_temperature_rate_drift",
        }
    )
    write_rows(args.output_csv, [verdict])
    high_curve = aggregate_curve(
        read_rows(args.high_prefix.with_name(args.high_prefix.name + "_curve.csv")),
        args.plot_block_size,
    )
    low_curve = aggregate_curve(
        read_rows(args.low_prefix.with_name(args.low_prefix.name + "_curve.csv")),
        args.plot_block_size,
    )
    write_svg(
        args.output_svg,
        high_curve,
        low_curve,
        low_rows,
        read_rows(args.low_gamma_replicates),
        block_size=args.plot_block_size,
    )


if __name__ == "__main__":
    main()
