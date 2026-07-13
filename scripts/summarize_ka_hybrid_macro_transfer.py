#!/usr/bin/env python3
"""Select the next minimal kernel after held-out hybrid macro propagation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one row")
    return rows[0]


def write_one(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def classify_crossover(
    *,
    low: dict[str, str],
    high: dict[str, str],
) -> dict[str, float | str]:
    low_diffusion = float(low["diffusion_relative_error"]) <= 0.15
    low_alpha = (
        math.isfinite(float(low["alpha_relaxation_relative_error"]))
        and float(low["alpha_relaxation_relative_error"]) <= 0.20
    )
    low_tail = float(low["maximum_count_tail_probability"]) <= 0.01
    low_ngp = float(low["maximum_ensemble_ngp_absolute_error"]) <= 0.30
    low_fs = float(low["maximum_ensemble_fs_absolute_error"]) <= 0.03
    direction_included = float(low["jump_direction_correlation_included"]) == 1.0
    rejected = (
        low_diffusion
        and low_tail
        and direction_included
        and not low_alpha
        and not low_ngp
        and not low_fs
        and float(low["joint_macro_transfer_pass"]) == 0.0
    )
    return {
        "low_temperature_diffusion_transfer_pass": float(low_diffusion),
        "low_temperature_alpha_transfer_pass": float(low_alpha),
        "low_temperature_ngp_transfer_pass": float(low_ngp),
        "low_temperature_multik_fs_transfer_pass": float(low_fs),
        "low_temperature_count_kernel_identifiable": float(low_tail),
        "high_temperature_count_kernel_identifiable": float(
            float(high["maximum_count_tail_probability"]) <= 0.01
        ),
        "high_temperature_diffusion_transfer_pass": float(
            float(high["diffusion_relative_error"]) <= 0.15
        ),
        "high_temperature_joint_macro_transfer_pass": float(
            high["joint_macro_transfer_pass"]
        ),
        "independent_count_jump_kernel_rejected": float(rejected),
        "additional_exchange_clock_supported": 0.0,
        "next_minimal_extension": (
            "mobility_state_conditioned_jump_cage_kernel"
            if rejected
            else "no_unique_kernel_extension_selected"
        ),
        "heldout_macro_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def write_svg(path: Path, rows: list[dict[str, str]]) -> None:
    width, height = 900, 390
    margin, top, panel_width, panel_height, gap = 60, 55, 215, 245, 35
    panels = (
        ("MSD", "predicted_msd", "observed_msd"),
        ("NGP", "predicted_ngp", "observed_ngp"),
        ("F_s(k=7.25)", "predicted_fs_k7p25", "observed_fs_k7p25"),
    )
    lags = [float(row["lag"]) for row in rows]
    log_min, log_max = math.log10(min(lags)), math.log10(max(lags))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124}.grid{stroke:#DADCE0}</style>',
        '<text x="450" y="27" text-anchor="middle" font-size="18" font-weight="700">Two-clock count law needs a state-conditioned displacement kernel</text>',
    ]
    for panel_index, (title, predicted_key, observed_key) in enumerate(panels):
        left = margin + panel_index * (panel_width + gap)
        predicted = [float(row[predicted_key]) for row in rows]
        observed = [float(row[observed_key]) for row in rows]
        y_min = min(0.0, min(predicted + observed))
        y_max = max(predicted + observed) * 1.08
        if y_max <= y_min:
            y_max = y_min + 1.0

        def x_position(lag: float) -> float:
            return left + panel_width * (math.log10(lag) - log_min) / (log_max - log_min)

        def y_position(value: float) -> float:
            return top + panel_height * (1.0 - (value - y_min) / (y_max - y_min))

        lines.append(f'<text x="{left + panel_width / 2:.2f}" y="45" text-anchor="middle" font-size="13" font-weight="700">{title}</text>')
        for tick in range(4):
            value = y_min + (y_max - y_min) * tick / 3.0
            y = y_position(value)
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" y2="{y:.2f}"/>')
            lines.append(f'<text x="{left - 7}" y="{y + 4:.2f}" text-anchor="end" font-size="9">{value:.2g}</text>')
        lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}"/>')
        for values, color in ((observed, "#202124"), (predicted, "#D55E00")):
            points = " ".join(
                f"{x_position(lag):.2f},{y_position(value):.2f}"
                for lag, value in zip(lags, values)
            )
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        lines.append(f'<text x="{left + panel_width / 2:.2f}" y="325" text-anchor="middle" font-size="11">lag time (log scale)</text>')
    lines.extend(
        [
            '<line x1="320" y1="365" x2="350" y2="365" stroke="#202124" stroke-width="3"/>',
            '<text x="360" y="370" font-size="11">held-out observation</text>',
            '<line x1="515" y1="365" x2="545" y2="365" stroke="#D55E00" stroke-width="3"/>',
            '<text x="555" y="370" font-size="11">calibration-only prediction</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-verdict", type=Path, required=True)
    parser.add_argument("--high-verdict", type=Path, required=True)
    parser.add_argument("--low-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    result = classify_crossover(
        low=read_one(args.low_verdict),
        high=read_one(args.high_verdict),
    )
    write_one(args.output, result)
    write_svg(args.output_svg, read_rows(args.low_summary))


if __name__ == "__main__":
    main()
