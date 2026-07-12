#!/usr/bin/env python3
"""Summarize cooling-induced broadening of the finite exchange spectrum."""

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
    low_selection_fraction: float,
    low_transfer_fraction: float,
    low_hmm_transfer_fraction: float,
) -> dict[str, float | str]:
    broadening = bool(
        high_outcome == "single_mode_exchange_sufficient"
        and low_outcome.startswith("two_mode_finite_exchange_spectrum")
        and low_selection_fraction == 1.0
        and low_transfer_fraction > low_hmm_transfer_fraction
    )
    return {
        "cooling_induced_exchange_spectrum_broadening": float(broadening),
        "heldout_pass_gain_over_markov_hmm": (
            low_transfer_fraction - low_hmm_transfer_fraction
        ),
        "semi_markov_generator_required": float(broadening),
        "full_event_clock_closure_claim_allowed": 0.0,
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
                "single": float(np.mean([float(row["single_mode_prediction"]) for row in local])),
                "two_mode": float(np.mean([float(row["two_mode_prediction"]) for row in local])),
                "static": float(np.mean([float(row["static_prediction"]) for row in local])),
            }
        )
    return result


def write_svg(
    path: Path,
    high_curve: list[dict[str, float]],
    low_curve: list[dict[str, float]],
    *,
    block_size: float,
) -> None:
    width, height = 920, 430
    margin_left, panel_width, gap = 70, 355, 90
    top, panel_height = 55, 290
    colors = {
        "heldout": "#202124",
        "single": "#0072B2",
        "two_mode": "#009E73",
        "static": "#D55E00",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124;stroke-width:1}.grid{stroke:#DADCE0;stroke-width:1}.obs{fill:none;stroke-width:3}.model{fill:none;stroke-width:2;stroke-dasharray:7 5}</style>',
        '<text x="460" y="27" text-anchor="middle" font-size="18" font-weight="700">Cooling broadens the finite exchange spectrum</text>',
    ]
    for panel, (label, curve) in enumerate((("T = 0.58", high_curve), ("T = 0.45", low_curve))):
        left = margin_left + panel * (panel_width + gap)
        max_time = max(row["lag_time"] for row in curve)
        max_y = max(0.08, max(row[key] for row in curve for key in colors) * 1.08)
        for tick in range(5):
            value = max_y * tick / 4.0
            y = top + panel_height - panel_height * value / max_y
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" y2="{y:.2f}"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:.2f}</text>')
        lines.extend(
            [
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}"/>',
                f'<line class="axis" x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}"/>',
                f'<text x="{left + panel_width / 2:.2f}" y="{top - 13}" text-anchor="middle" font-size="15" font-weight="700">{label}</text>',
                f'<text x="{left + panel_width / 2:.2f}" y="{height - 35}" text-anchor="middle" font-size="13">lag time (block size = {block_size:g})</text>',
            ]
        )
        for tick in range(5):
            value = max_time * tick / 4.0
            x = left + panel_width * tick / 4.0
            lines.append(f'<text x="{x:.2f}" y="{top + panel_height + 20}" text-anchor="middle" font-size="11">{value:.0f}</text>')
        for key in colors:
            points = " ".join(
                f'{left + panel_width * row["lag_time"] / max_time:.2f},{top + panel_height - panel_height * row[key] / max_y:.2f}'
                for row in curve
            )
            css = "obs" if key == "heldout" else "model"
            lines.append(f'<polyline class="{css}" stroke="{colors[key]}" points="{points}"/>')
    legend = (
        ("heldout", "held-out"),
        ("single", "single mode"),
        ("two_mode", "two finite modes"),
        ("static", "static disorder"),
    )
    for index, (key, label) in enumerate(legend):
        x = 170 + index * 180
        dash = "" if key == "heldout" else ' stroke-dasharray="7 5"'
        lines.append(f'<line x1="{x}" y1="402" x2="{x + 28}" y2="402" stroke="{colors[key]}" stroke-width="3"{dash}/>')
        lines.append(f'<text x="{x + 36}" y="407" font-size="12">{label}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-prefix", type=Path, required=True)
    parser.add_argument("--low-prefix", type=Path, required=True)
    parser.add_argument("--high-hmm-verdict", type=Path, required=True)
    parser.add_argument("--low-hmm-verdict", type=Path, required=True)
    parser.add_argument("--low-hmm-replicates", type=Path, required=True)
    parser.add_argument("--plot-block-size", type=float, default=50.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()

    high = read_one(args.high_prefix.with_name(args.high_prefix.name + "_verdict.csv"))
    low = read_one(args.low_prefix.with_name(args.low_prefix.name + "_verdict.csv"))
    high_hmm = read_one(args.high_hmm_verdict)
    low_hmm = read_one(args.low_hmm_verdict)
    if (
        float(high["calibration_horizon"]) != float(low["calibration_horizon"])
        or float(high["scoring_horizon"]) != float(low["scoring_horizon"])
    ):
        raise ValueError("temperature comparison requires common calibration and scoring horizons")
    low_selection = float(low["two_mode_calibration_selection_fraction"])
    low_transfer = float(low["two_mode_heldout_transfer_pass_fraction"])
    analyzed_blocks = {
        float(value) for value in low["common_block_sizes"].split(";")
    }
    low_hmm_rows = [
        row
        for row in read_rows(args.low_hmm_replicates)
        if float(row["block_size"]) in analyzed_blocks
    ]
    low_hmm_transfer = float(
        np.mean([float(row["finite_exchange_hmm_transfer_pass"]) for row in low_hmm_rows])
    )
    verdict = classify_crossover(
        high_outcome=high["event_level_outcome"],
        low_outcome=low["event_level_outcome"],
        low_selection_fraction=low_selection,
        low_transfer_fraction=low_transfer,
        low_hmm_transfer_fraction=low_hmm_transfer,
    )
    low_rows = read_rows(args.low_prefix.with_name(args.low_prefix.name + "_replicates.csv"))
    failures = [
        f"rep{int(float(row['replicate']))}_block{float(row['block_size']):g}_{row['primary_failure']}"
        for row in low_rows
        if float(row["two_mode_heldout_transfer_pass"]) == 0.0
    ]
    verdict.update(
        {
            "common_calibration_horizon": float(high["calibration_horizon"]),
            "common_scoring_horizon": float(high["scoring_horizon"]),
            "high_two_mode_selection_fraction": float(
                high["two_mode_calibration_selection_fraction"]
            ),
            "low_two_mode_selection_fraction": low_selection,
            "high_two_mode_transfer_fraction": float(
                high["two_mode_heldout_transfer_pass_fraction"]
            ),
            "low_two_mode_transfer_fraction": low_transfer,
            "low_markov_hmm_transfer_fraction": low_hmm_transfer,
            "low_two_mode_pass_count": low_transfer * float(low["replica_block_count"]),
            "low_markov_hmm_pass_count": low_hmm_transfer * float(low["replica_block_count"]),
            "remaining_failure_count": float(len(failures)),
            "remaining_failures": ";".join(failures) if failures else "none",
            "excluded_underidentified_block_sizes": low[
                "excluded_underidentified_block_sizes"
            ],
            "identified_resolution_closure": float(
                low_selection == 1.0 and low_transfer == 1.0
            ),
            "full_resolution_scope": float(low["full_resolution_scope"]),
            "high_markov_hmm_sufficient": float(high_hmm["two_state_poisson_hmm_sufficient"]),
            "mechanism_statement": "cooling_broadens_finite_exchange_spectrum_before_static_limit",
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
    write_svg(args.output_svg, high_curve, low_curve, block_size=args.plot_block_size)


if __name__ == "__main__":
    main()
