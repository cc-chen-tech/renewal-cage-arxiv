#!/usr/bin/env python3
"""Audit whether held-out event-rate drift is systematic or a split-local anomaly."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_rate_trend(values: np.ndarray) -> dict[str, float]:
    """Compute an exact two-sided permutation test for a linear window trend."""

    rates = np.asarray(values, dtype=float)
    if (
        rates.ndim != 1
        or not 3 <= len(rates) <= 8
        or np.any(~np.isfinite(rates))
        or np.any(rates <= 0.0)
    ):
        raise ValueError("rate trend requires three to eight finite positive windows")
    normalized = rates / np.mean(rates)
    times = np.arange(len(rates), dtype=float)
    slope = float(np.polyfit(times, normalized, 1)[0])
    null = np.array(
        [
            np.polyfit(times, np.asarray(permutation) / np.mean(rates), 1)[0]
            for permutation in itertools.permutations(rates)
        ]
    )
    p_value = (float(np.sum(np.abs(null) >= abs(slope))) + 1.0) / (len(null) + 1.0)
    half = len(rates) // 2
    first_half = float(np.mean(rates[:half]))
    second_half = float(np.mean(rates[-half:]))
    return {
        "window_count": float(len(rates)),
        "normalized_slope_per_window": slope,
        "normalized_total_linear_change": slope * (len(rates) - 1),
        "exact_two_sided_permutation_p_value": p_value,
        "strict_trend_detected": float(p_value <= 0.05),
        "borderline_trend_detected": float(p_value <= 0.10),
        "first_half_mean_rate": first_half,
        "second_half_mean_rate": second_half,
        "second_to_first_half_rate_ratio": second_half / first_half,
        "maximum_window_relative_deviation": float(
            np.max(np.abs(normalized - 1.0))
        ),
    }


def write_svg(
    path: Path,
    high_rows: list[dict[str, str]],
    low_rows: list[dict[str, str]],
    *,
    block_size: float,
) -> None:
    width, height = 840, 430
    left, top, plot_width, plot_height = 75, 55, 700, 285
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124;stroke-width:1}.grid{stroke:#DADCE0;stroke-width:1}</style>',
        '<text x="420" y="27" text-anchor="middle" font-size="18" font-weight="700">Six-window event-rate stability audit</text>',
    ]
    y_min, y_max = 0.78, 1.22
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4.0
        y = top + plot_height - plot_height * (value - y_min) / (y_max - y_min)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:.2f}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="370" text-anchor="middle" font-size="13">rate-window index (block size = {block_size:g})</text>',
            '<text x="18" y="200" text-anchor="middle" font-size="13" transform="rotate(-90 18 200)">rate / trajectory mean</text>',
        ]
    )

    def series(rows: list[dict[str, str]], replicate: int | None) -> np.ndarray:
        selected = [row for row in rows if float(row["block_size"]) == block_size]
        if replicate is not None:
            selected = [row for row in selected if int(float(row["replicate"])) == replicate]
            selected = sorted(selected, key=lambda row: float(row["rate_window_index"]))
            values = np.array([float(row["mean_count_per_particle_block"]) for row in selected])
            return values / np.mean(values)
        window_indices = sorted({int(float(row["rate_window_index"])) for row in selected})
        values = np.array(
            [
                np.mean(
                    [
                        float(row["mean_count_per_particle_block"])
                        for row in selected
                        if int(float(row["rate_window_index"])) == index
                    ]
                )
                for index in window_indices
            ]
        )
        return values / np.mean(values)

    curves = [
        ("high-T mean", series(high_rows, None), "#7B61A8", "6 4"),
        ("low-T rep 1", series(low_rows, 1), "#0072B2", ""),
        ("low-T rep 2", series(low_rows, 2), "#D55E00", ""),
        ("low-T rep 3", series(low_rows, 3), "#009E73", ""),
    ]
    for label, values, color, dash in curves:
        points = " ".join(
            f'{left + plot_width * index / (len(values) - 1):.2f},{top + plot_height - plot_height * (value - y_min) / (y_max - y_min):.2f}'
            for index, value in enumerate(values)
        )
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{dash_attribute} points="{points}"/>')
    for index in range(6):
        x = left + plot_width * index / 5.0
        lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle" font-size="11">{index + 1}</text>')
    for index, (label, _, color, dash) in enumerate(curves):
        x = 105 + index * 175
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<line x1="{x}" y1="405" x2="{x + 28}" y2="405" stroke="{color}" stroke-width="3"{dash_attribute}/>')
        lines.append(f'<text x="{x + 35}" y="410" font-size="11">{label}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-rate-windows", type=Path, required=True)
    parser.add_argument("--low-rate-windows", type=Path, required=True)
    parser.add_argument("--low-hybrid-replicates", type=Path, required=True)
    parser.add_argument("--audit-block-size", type=float, default=20.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    high_rows = read_rows(args.high_rate_windows)
    low_rows = read_rows(args.low_rate_windows)
    hybrid_rows = read_rows(args.low_hybrid_replicates)
    audit_rows: list[dict[str, object]] = []
    for temperature_group, rows in (("high", high_rows), ("low", low_rows)):
        for replicate in sorted({int(float(row["replicate"])) for row in rows}):
            for block_size in sorted({float(row["block_size"]) for row in rows}):
                selected = sorted(
                    [
                        row
                        for row in rows
                        if int(float(row["replicate"])) == replicate
                        and float(row["block_size"]) == block_size
                    ],
                    key=lambda row: float(row["rate_window_index"]),
                )
                trend = exact_rate_trend(
                    np.array(
                        [float(row["mean_count_per_particle_block"]) for row in selected]
                    )
                )
                trend.update(
                    {
                        "temperature_group": temperature_group,
                        "temperature": float(selected[0]["temperature"]),
                        "replicate": float(replicate),
                        "block_size": block_size,
                    }
                )
                audit_rows.append(trend)
    low_audit = [
        row
        for row in audit_rows
        if row["temperature_group"] == "low"
        and float(row["block_size"]) == args.audit_block_size
    ]
    failed_rate_replicates = sorted(
        {
            int(float(row["replicate"]))
            for row in hybrid_rows
            if row["primary_failure"] == "mean_count_drift"
        }
    )
    strict = [row for row in low_audit if float(row["strict_trend_detected"]) == 1.0]
    borderline = [
        row for row in low_audit if float(row["borderline_trend_detected"]) == 1.0
    ]
    verdict = {
        "audit_block_size": args.audit_block_size,
        "rate_window_count": float(low_audit[0]["window_count"]),
        "low_independent_replicate_count": float(len(low_audit)),
        "failed_absolute_rate_gate_replicates": ";".join(map(str, failed_rate_replicates)),
        "strict_low_temperature_trend_replicate_count": float(len(strict)),
        "borderline_low_temperature_trend_replicate_count": float(len(borderline)),
        "minimum_low_temperature_trend_p_value": min(
            float(row["exact_two_sided_permutation_p_value"]) for row in low_audit
        ),
        "maximum_low_temperature_total_linear_change": max(
            abs(float(row["normalized_total_linear_change"])) for row in low_audit
        ),
        "systematic_rate_nonstationarity_claim_allowed": 0.0,
        "new_rate_state_parameter_claim_allowed": 0.0,
        "rate_stationarity_claim_allowed": float(
            not failed_rate_replicates and not borderline
        ),
        "macro_observable_prediction_claim_allowed": 0.0,
        "next_required_test": "new_independent_low_temperature_trajectory_or_longer_third_window",
        "thermodynamic_claim_allowed": 0.0,
    }
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_rows.csv"), audit_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])
    write_svg(
        args.output_svg,
        high_rows,
        low_rows,
        block_size=args.audit_block_size,
    )


if __name__ == "__main__":
    main()
