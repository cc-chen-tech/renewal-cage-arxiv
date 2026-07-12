#!/usr/bin/env python3
"""Summarize the one-to-two-clock gamma-refresh Cox crossover."""

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
    high_outcome: str,
    low_outcome: str,
    high_single_transfer: float,
    low_single_transfer: float,
    low_two_transfer: float,
) -> dict[str, float]:
    selected = bool(
        high_outcome == "single_clock_gamma_refresh_count_moment_closure"
        and low_outcome == "two_clock_gamma_refresh_count_moment_closure"
        and high_single_transfer == 1.0
        and low_two_transfer == 1.0
        and low_single_transfer < low_two_transfer
    )
    return {
        "cooling_induced_second_refresh_clock_required": float(selected),
        "low_temperature_pass_gain": low_two_transfer - low_single_transfer,
        "count_moment_crossover_closure": float(selected),
        "full_count_distribution_claim_allowed": 0.0,
        "heldout_macro_prediction_claim_allowed": 0.0,
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
                "single": float(np.mean([float(row["single_clock_prediction"]) for row in local])),
                "two": float(np.mean([float(row["two_clock_prediction"]) for row in local])),
                "static": float(np.mean([float(row["static_prediction"]) for row in local])),
            }
        )
    return result


def write_svg(
    path: Path,
    high_curve: list[dict[str, float]],
    low_curve: list[dict[str, float]],
    high_replicates: list[dict[str, str]],
    low_replicates: list[dict[str, str]],
    *,
    block_size: float,
) -> None:
    width, height = 920, 680
    lefts = [70, 515]
    panel_width = 355
    top, curve_height = 55, 270
    colors = {"heldout": "#202124", "single": "#0072B2", "two": "#009E73", "static": "#D55E00"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124;stroke-width:1}.grid{stroke:#DADCE0;stroke-width:1}.obs{fill:none;stroke-width:3}.model{fill:none;stroke-width:2;stroke-dasharray:7 5}</style>',
        '<text x="460" y="27" text-anchor="middle" font-size="18" font-weight="700">A second finite refresh clock emerges on cooling</text>',
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
                f'<text x="{left + panel_width / 2:.2f}" y="{top + curve_height + 36}" text-anchor="middle" font-size="12">lag time (block size = {block_size:g})</text>',
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

    scatter_top, scatter_height = 405, 205
    for panel, (label, rows) in enumerate((("T = 0.58", high_replicates), ("T = 0.45", low_replicates))):
        left = lefts[panel]
        calibration = np.array([float(row["calibration_fano_factor"]) for row in rows])
        heldout = np.array([float(row["heldout_fano_factor"]) for row in rows])
        minimum = min(float(np.min(calibration)), float(np.min(heldout))) * 0.97
        maximum = max(float(np.max(calibration)), float(np.max(heldout))) * 1.03
        lines.extend(
            [
                f'<line class="axis" x1="{left}" y1="{scatter_top}" x2="{left}" y2="{scatter_top + scatter_height}"/>',
                f'<line class="axis" x1="{left}" y1="{scatter_top + scatter_height}" x2="{left + panel_width}" y2="{scatter_top + scatter_height}"/>',
                f'<text x="{left + panel_width / 2:.2f}" y="{scatter_top - 15}" text-anchor="middle" font-size="14" font-weight="700">{label}: count Fano</text>',
                f'<line x1="{left}" y1="{scatter_top + scatter_height}" x2="{left + panel_width}" y2="{scatter_top}" stroke="#8A8D91" stroke-width="1.5" stroke-dasharray="5 4"/>',
                f'<text x="{left + panel_width / 2:.2f}" y="{height - 22}" text-anchor="middle" font-size="12">calibration Fano</text>',
            ]
        )
        for value_x, value_y in zip(calibration, heldout):
            x = left + panel_width * (value_x - minimum) / (maximum - minimum)
            y = scatter_top + scatter_height - scatter_height * (value_y - minimum) / (maximum - minimum)
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#009E73" stroke="#ffffff" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{scatter_top + 4}" text-anchor="end" font-size="10">{maximum:.2f}</text>')
        lines.append(f'<text x="{left - 12}" y="{scatter_top + scatter_height + 4}" text-anchor="end" font-size="10">{minimum:.2f}</text>')
        lines.append(f'<text x="{left}" y="{scatter_top + scatter_height + 18}" text-anchor="middle" font-size="10">{minimum:.2f}</text>')
        lines.append(f'<text x="{left + panel_width}" y="{scatter_top + scatter_height + 18}" text-anchor="middle" font-size="10">{maximum:.2f}</text>')
    legend = (("heldout", "held-out"), ("single", "one clock"), ("two", "two clocks"), ("static", "static disorder"))
    for index, (key, label) in enumerate(legend):
        x = 155 + index * 185
        dash = "" if key == "heldout" else ' stroke-dasharray="7 5"'
        lines.append(f'<line x1="{x}" y1="365" x2="{x + 28}" y2="365" stroke="{colors[key]}" stroke-width="3"{dash}/>')
        lines.append(f'<text x="{x + 36}" y="370" font-size="12">{label}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-prefix", type=Path, required=True)
    parser.add_argument("--low-prefix", type=Path, required=True)
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
    high_single = float(high["single_clock_transfer_pass_fraction"])
    low_single = float(low["single_clock_transfer_pass_fraction"])
    low_two = float(low["two_clock_transfer_pass_fraction"])
    verdict = classify_crossover(
        high_outcome=high["event_level_outcome"],
        low_outcome=low["event_level_outcome"],
        high_single_transfer=high_single,
        low_single_transfer=low_single,
        low_two_transfer=low_two,
    )
    verdict.update(
        {
            "common_calibration_horizon": float(high["calibration_horizon"]),
            "common_scoring_horizon": float(high["scoring_horizon"]),
            "common_block_sizes": high["common_block_sizes"],
            "excluded_underidentified_block_sizes": low[
                "excluded_underidentified_block_sizes"
            ],
            "high_single_clock_pass_fraction": high_single,
            "low_single_clock_pass_fraction": low_single,
            "low_two_clock_pass_fraction": low_two,
            "high_maximum_fano_relative_error": float(
                high["maximum_heldout_fano_relative_error"]
            ),
            "low_maximum_fano_relative_error": float(
                low["maximum_heldout_fano_relative_error"]
            ),
            "low_maximum_identity_rmse": float(low["maximum_two_clock_identity_rmse"]),
            "high_maximum_count_tv_distance": float(
                high["maximum_heldout_count_tv_distance"]
            ),
            "low_maximum_count_tv_distance": float(
                low["maximum_heldout_count_tv_distance"]
            ),
            "marginal_count_distribution_crossover_closure": float(
                float(high["marginal_count_distribution_claim_allowed"]) == 1.0
                and float(low["marginal_count_distribution_claim_allowed"]) == 1.0
            ),
            "high_gamma_pair_pass_fraction": float(
                high["gamma_pair_distribution_pass_fraction"]
            ),
            "low_gamma_pair_pass_fraction": float(
                low["gamma_pair_distribution_pass_fraction"]
            ),
            "low_hmm_pair_pass_fraction": float(
                low["hmm_pair_distribution_pass_fraction"]
            ),
            "joint_count_pair_crossover_closure": float(
                float(high["joint_count_pair_distribution_claim_allowed"]) == 1.0
                and float(low["joint_count_pair_distribution_claim_allowed"]) == 1.0
            ),
            "hybrid_semimarkov_emission_model_required": float(
                low["hybrid_semimarkov_emission_model_required"]
            ),
            "full_count_sequence_likelihood_claim_allowed": 0.0,
            "positive_intensity_generator": 1.0,
            "finite_recovery_enforced": 1.0,
            "next_required_test": "fit_hmm_emissions_with_two_clock_semimarkov_residence_then_macro_propagation",
        }
    )
    write_rows(args.output_csv, [verdict])
    high_curve_rows = read_rows(args.high_prefix.with_name(args.high_prefix.name + "_curve.csv"))
    low_curve_rows = read_rows(args.low_prefix.with_name(args.low_prefix.name + "_curve.csv"))
    high_replicates = read_rows(
        args.high_prefix.with_name(args.high_prefix.name + "_replicates.csv")
    )
    low_replicates = read_rows(
        args.low_prefix.with_name(args.low_prefix.name + "_replicates.csv")
    )
    write_svg(
        args.output_svg,
        aggregate_curve(high_curve_rows, args.plot_block_size),
        aggregate_curve(low_curve_rows, args.plot_block_size),
        high_replicates,
        low_replicates,
        block_size=args.plot_block_size,
    )


if __name__ == "__main__":
    main()
