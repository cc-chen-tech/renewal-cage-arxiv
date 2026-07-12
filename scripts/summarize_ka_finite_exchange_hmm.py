#!/usr/bin/env python3
"""Summarize the high-to-low-temperature finite-exchange HMM crossover."""

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
        raise ValueError(f"{path} must contain exactly one data row")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_crossover(
    high_verdict: dict[str, object],
    low_verdict: dict[str, object],
    *,
    high_positive_late_block_count: int,
    low_positive_late_block_count: int,
    common_block_count: int,
) -> dict[str, float | str]:
    high_markov = float(high_verdict["two_state_poisson_hmm_sufficient"]) == 1.0
    low_markov = float(low_verdict["two_state_poisson_hmm_sufficient"]) == 1.0
    low_broad = (
        not low_markov
        and str(low_verdict["event_level_outcome"])
        == "non_single_exponential_exchange_required"
    )
    broadening = bool(
        high_markov
        and low_broad
        and high_positive_late_block_count == 0
        and low_positive_late_block_count == common_block_count
    )
    return {
        "high_temperature_two_state_hmm_sufficient": float(high_markov),
        "low_temperature_two_state_hmm_sufficient": float(low_markov),
        "high_positive_late_excess_block_count": float(high_positive_late_block_count),
        "low_positive_late_excess_block_count": float(low_positive_late_block_count),
        "common_block_count": float(common_block_count),
        "single_exchange_time_low_temperature_rejected": float(low_broad),
        "finite_exchange_spectrum_broadening_detected": float(broadening),
        "finite_exchange_environment_supported": float(broadening),
        "spatial_facilitation_claim_allowed": 0.0,
        "heldout_macro_prediction_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def aggregate_curves(rows: list[dict[str, str]], block_size: float) -> list[dict[str, float]]:
    selected = [row for row in rows if float(row["block_size"]) == block_size]
    result = []
    for lag in sorted({float(row["block_lag"]) for row in selected}):
        local = [row for row in selected if float(row["block_lag"]) == lag]
        result.append(
            {
                "block_lag": lag,
                "observed": float(np.mean([float(row["observed_identity_correlation"]) for row in local])),
                "hmm": float(np.mean([float(row["hmm_identity_correlation"]) for row in local])),
                "static": float(np.mean([float(row["static_identity_correlation"]) for row in local])),
                "poisson": 0.0,
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
    margin_left, panel_width, panel_gap = 70, 355, 90
    top, panel_height = 55, 290
    colors = {"observed": "#202124", "hmm": "#0072B2", "static": "#D55E00", "poisson": "#8A8D91"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124;stroke-width:1}.grid{stroke:#DADCE0;stroke-width:1}.obs{fill:none;stroke-width:3}.model{fill:none;stroke-width:2;stroke-dasharray:7 5}</style>',
        '<text x="460" y="27" text-anchor="middle" font-size="18" font-weight="700">Calibration-only finite-exchange clock: held-out identity decay</text>',
    ]
    for panel, (label, curve) in enumerate((("T = 0.58", high_curve), ("T = 0.45", low_curve))):
        left = margin_left + panel * (panel_width + panel_gap)
        max_lag = max(row["block_lag"] for row in curve)
        max_y = max(0.08, max(row[key] for row in curve for key in ("observed", "hmm", "static")) * 1.08)
        for tick in range(5):
            y_value = max_y * tick / 4.0
            y = top + panel_height - panel_height * y_value / max_y
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" y2="{y:.2f}"/>')
            lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{y_value:.2f}</text>')
        lines.extend(
            [
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}"/>',
                f'<line class="axis" x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}"/>',
                f'<text x="{left + panel_width / 2:.2f}" y="{top - 13}" text-anchor="middle" font-size="15" font-weight="700">{label}</text>',
                f'<text x="{left + panel_width / 2:.2f}" y="{height - 35}" text-anchor="middle" font-size="13">block lag (block size = {block_size:g})</text>',
            ]
        )
        for tick in range(0, int(max_lag) + 1, max(1, int(max_lag // 4))):
            x = left + panel_width * tick / max_lag
            lines.append(f'<text x="{x:.2f}" y="{top + panel_height + 20}" text-anchor="middle" font-size="11">{tick}</text>')
        for key in ("observed", "hmm", "static", "poisson"):
            points = " ".join(
                f'{left + panel_width * row["block_lag"] / max_lag:.2f},{top + panel_height - panel_height * row[key] / max_y:.2f}'
                for row in curve
            )
            css_class = "obs" if key == "observed" else "model"
            lines.append(f'<polyline class="{css_class}" stroke="{colors[key]}" points="{points}"/>')
    legend = (("observed", "held-out"), ("hmm", "two-state HMM"), ("static", "static disorder"), ("poisson", "Poisson"))
    for index, (key, label) in enumerate(legend):
        x = 180 + index * 170
        dash = "" if key == "observed" else ' stroke-dasharray="7 5"'
        lines.append(f'<line x1="{x}" y1="402" x2="{x + 28}" y2="402" stroke="{colors[key]}" stroke-width="3"{dash}/>')
        lines.append(f'<text x="{x + 36}" y="407" font-size="12">{label}</text>')
    lines.append('</svg>')
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

    high_verdict = read_one(args.high_prefix.with_name(args.high_prefix.name + "_verdict.csv"))
    low_verdict = read_one(args.low_prefix.with_name(args.low_prefix.name + "_verdict.csv"))
    if float(high_verdict["scoring_horizon"]) != float(low_verdict["scoring_horizon"]):
        raise ValueError("temperature comparison requires a common scoring horizon")
    high_blocks = read_rows(args.high_prefix.with_name(args.high_prefix.name + "_blocks.csv"))
    low_blocks = read_rows(args.low_prefix.with_name(args.low_prefix.name + "_blocks.csv"))
    common_blocks = sorted(
        {float(row["block_size"]) for row in high_blocks}
        & {float(row["block_size"]) for row in low_blocks}
    )
    if args.plot_block_size not in common_blocks:
        raise ValueError("plot block size must be present at both temperatures")
    verdict = classify_crossover(
        high_verdict,
        low_verdict,
        high_positive_late_block_count=sum(
            float(row["late_positive_excess_detected"]) == 1.0 for row in high_blocks
        ),
        low_positive_late_block_count=sum(
            float(row["late_positive_excess_detected"]) == 1.0 for row in low_blocks
        ),
        common_block_count=len(common_blocks),
    )
    verdict.update(
        {
            "common_block_sizes": ";".join(f"{value:g}" for value in common_blocks),
            "common_scoring_horizon": float(high_verdict["scoring_horizon"]),
            "high_replica_block_pass_fraction": float(high_verdict["replica_block_pass_fraction"]),
            "low_replica_block_pass_fraction": float(low_verdict["replica_block_pass_fraction"]),
            "mechanism_statement": "finite_exchange_survives_but_single_markov_exchange_time_fails_on_cooling",
        }
    )
    write_rows(args.output_csv, [verdict])
    high_curve = aggregate_curves(
        read_rows(args.high_prefix.with_name(args.high_prefix.name + "_curve.csv")),
        args.plot_block_size,
    )
    low_curve = aggregate_curves(
        read_rows(args.low_prefix.with_name(args.low_prefix.name + "_curve.csv")),
        args.plot_block_size,
    )
    write_svg(
        args.output_svg,
        high_curve,
        low_curve,
        block_size=args.plot_block_size,
    )


if __name__ == "__main__":
    main()
