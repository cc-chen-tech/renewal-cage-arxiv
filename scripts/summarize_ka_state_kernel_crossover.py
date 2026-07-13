#!/usr/bin/env python3
"""Select cooling-induced memory from state-kernel held-out transfer."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_one(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one row")
    return rows[0]


def write_one(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def classify_state_kernel_crossover(
    *,
    low: dict[str, str],
    high: dict[str, str],
) -> dict[str, float | str]:
    low_curve = float(low["curve_transfer_pass"]) == 1.0
    high_curve = float(high["curve_transfer_pass"]) == 1.0
    low_diffusion = float(low["diffusion_relative_error"]) <= 0.15
    high_diffusion = float(high["diffusion_relative_error"]) <= 0.15
    higher_order_memory = high_curve and not low_curve and low_diffusion and high_diffusion
    return {
        "high_temperature_curve_closure": float(high_curve),
        "low_temperature_curve_closure": float(low_curve),
        "diffusion_transfer_pass_both_temperatures": float(
            low_diffusion and high_diffusion
        ),
        "high_temperature_alpha_crossing_ready": float(
            high["alpha_crossing_ready"]
        ),
        "low_temperature_ngp_absolute_error": float(
            low["maximum_ensemble_ngp_absolute_error"]
        ),
        "low_temperature_fs_absolute_error": float(
            low["maximum_ensemble_fs_absolute_error"]
        ),
        "high_temperature_ngp_absolute_error": float(
            high["maximum_ensemble_ngp_absolute_error"]
        ),
        "high_temperature_fs_absolute_error": float(
            high["maximum_ensemble_fs_absolute_error"]
        ),
        "cooling_induced_higher_order_memory_required": float(higher_order_memory),
        "additional_mobility_clock_supported": 0.0,
        "next_minimal_extension": (
            "non_markov_multiblock_orientation_cage_persistence_kernel"
            if higher_order_memory
            else "no_unique_multiblock_extension_selected"
        ),
        "heldout_macro_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def write_svg(
    path: Path,
    *,
    low: dict[str, str],
    high: dict[str, str],
) -> None:
    width, height = 820, 430
    left, top, plot_width, plot_height = 80, 55, 660, 270
    metrics = (
        ("MSD curve", "maximum_ensemble_msd_relative_error", 0.10),
        ("NGP curve", "maximum_ensemble_ngp_absolute_error", 0.30),
        ("multi-k F_s", "maximum_ensemble_fs_absolute_error", 0.03),
        ("diffusion", "diffusion_relative_error", 0.15),
    )
    normalized = {
        "high": [float(high[key]) / tolerance for _, key, tolerance in metrics],
        "low": [float(low[key]) / tolerance for _, key, tolerance in metrics],
    }
    y_min, y_max = 0.1, 10.0

    def y_position(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        fraction = (math.log10(clipped) - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        )
        return top + plot_height * (1.0 - fraction)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124}.grid{stroke:#DADCE0}</style>',
        '<text x="410" y="27" text-anchor="middle" font-size="18" font-weight="700">Cooling breaks the two-state Markov displacement kernel</text>',
    ]
    for value in (0.1, 0.3, 1.0, 3.0, 10.0):
        y = y_position(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:g}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            '<text x="20" y="190" text-anchor="middle" font-size="13" transform="rotate(-90 20 190)">error / tolerance (log scale)</text>',
        ]
    )
    group_width = plot_width / len(metrics)
    colors = {"high": "#0072B2", "low": "#D55E00"}
    for index, (label, _, _) in enumerate(metrics):
        center = left + group_width * (index + 0.5)
        for group, offset in (("high", -12.0), ("low", 12.0)):
            value = normalized[group][index]
            lines.append(f'<circle cx="{center + offset:.2f}" cy="{y_position(value):.2f}" r="5" fill="{colors[group]}"/>')
        lines.append(f'<text x="{center:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="11">{label}</text>')
    lines.extend(
        [
            '<circle cx="285" cy="390" r="5" fill="#0072B2"/>',
            '<text x="298" y="395" font-size="11">T=0.58</text>',
            '<circle cx="430" cy="390" r="5" fill="#D55E00"/>',
            '<text x="443" y="395" font-size="11">T=0.45</text>',
            '<text x="690" y="65" text-anchor="end" font-size="10">pass below 1</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-verdict", type=Path, required=True)
    parser.add_argument("--high-verdict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    low = read_one(args.low_verdict)
    high = read_one(args.high_verdict)
    write_one(
        args.output,
        classify_state_kernel_crossover(low=low, high=high),
    )
    write_svg(args.output_svg, low=low, high=high)


if __name__ == "__main__":
    main()
