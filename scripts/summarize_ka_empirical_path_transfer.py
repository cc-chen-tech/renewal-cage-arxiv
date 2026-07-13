#!/usr/bin/env python3
"""Select multiblock cage-path memory from two-temperature transfer nulls."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_one(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def _by_model(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {row["model"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("each path model must have exactly one verdict row")
    return result


def classify_empirical_path_crossover(
    low_verdicts: list[dict[str, str]],
    high_verdicts: list[dict[str, str]],
    markov_low: dict[str, str],
) -> dict[str, float | str]:
    low = _by_model(low_verdicts)
    high = _by_model(high_verdicts)
    required_models = {
        "contiguous_empirical_path",
        "within_particle_time_shuffle",
        "direction_randomized_path",
    }
    if not required_models <= low.keys() or "contiguous_empirical_path" not in high:
        raise ValueError("path crossover requires all low-temperature nulls and high contiguous")
    contiguous_low = low["contiguous_empirical_path"]
    shuffled_low = low["within_particle_time_shuffle"]
    direction_low = low["direction_randomized_path"]
    contiguous_low_pass = float(contiguous_low["curve_transfer_pass"]) == 1.0
    contiguous_high_pass = (
        float(high["contiguous_empirical_path"]["curve_transfer_pass"]) == 1.0
    )
    markov_low_fail = float(markov_low["curve_transfer_pass"]) == 0.0
    shared_ngp_failure = (
        float(markov_low["maximum_ensemble_ngp_absolute_error"]) > 0.30
        and float(shuffled_low["maximum_ensemble_ngp_absolute_error"]) > 0.30
    )
    shared_fs_failure = (
        float(markov_low["maximum_ensemble_fs_absolute_error"]) > 0.03
        and float(shuffled_low["maximum_ensemble_fs_absolute_error"]) > 0.03
    )
    shared_failure = markov_low_fail and (shared_ngp_failure or shared_fs_failure)
    replicate_count = int(float(contiguous_low["paired_replicate_count"]))
    contiguous_better = int(
        float(contiguous_low["paired_contiguous_better_replicate_count"])
    )
    required_consensus = math.ceil(2.0 * replicate_count / 3.0)
    replicate_consensus = (
        replicate_count >= 3 and contiguous_better >= required_consensus
    )
    shuffle_precision = float(shuffled_low["shuffle_precision_pass"]) == 1.0
    path_required = (
        contiguous_low_pass
        and contiguous_high_pass
        and shared_failure
        and replicate_consensus
        and shuffle_precision
    )
    direction_pass = float(direction_low["curve_transfer_pass"]) == 1.0
    ordered_recoil_required = path_required and not direction_pass
    return {
        "low_temperature_contiguous_closure": float(contiguous_low_pass),
        "high_temperature_contiguous_closure": float(contiguous_high_pass),
        "low_temperature_markov_failure": float(markov_low_fail),
        "shared_low_temperature_ngp_failure": float(shared_ngp_failure),
        "shared_low_temperature_fs_failure": float(shared_fs_failure),
        "shared_low_temperature_higher_order_failure": float(shared_failure),
        "shuffle_precision_pass": float(shuffle_precision),
        "paired_contiguous_better_replicate_count": float(contiguous_better),
        "paired_replicate_count": float(replicate_count),
        "required_replicate_consensus_count": float(required_consensus),
        "replicate_consensus_pass": float(replicate_consensus),
        "single_particle_multiblock_path_memory_required": float(path_required),
        "amplitude_persistence_alone_sufficient": float(
            path_required and direction_pass
        ),
        "ordered_recoil_path_required": float(ordered_recoil_required),
        "additional_mobility_clock_supported": 0.0,
        "next_minimal_extension": (
            "conditional_reversible_cage_path_kernel"
            if ordered_recoil_required
            else "no_unique_recoil_kernel_selected"
        ),
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def write_svg(
    path: Path,
    *,
    low_verdicts: list[dict[str, str]],
    high_verdicts: list[dict[str, str]],
) -> None:
    low = _by_model(low_verdicts)
    high = _by_model(high_verdicts)
    models = (
        ("contiguous_empirical_path", "Contiguous"),
        ("within_particle_time_shuffle", "Time shuffle"),
        ("direction_randomized_path", "Direction null"),
    )
    metrics = (
        ("MSD", "maximum_ensemble_msd_relative_error", 0.10),
        ("NGP", "maximum_ensemble_ngp_absolute_error", 0.30),
        ("multi-k F_s", "maximum_ensemble_fs_absolute_error", 0.03),
    )
    width, height = 900, 470
    left, top, plot_width, plot_height = 85, 62, 740, 300
    y_min, y_max = 0.05, 100.0

    def y(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        fraction = (math.log10(clipped) - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        )
        return top + plot_height * (1.0 - fraction)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#202124}.axis{stroke:#202124}.grid{stroke:#DADCE0}</style>',
        '<text x="450" y="29" text-anchor="middle" font-size="18" font-weight="700">Ordered cage paths close held-out dynamics</text>',
    ]
    for value in (0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        yy = y(value)
        lines.append(
            f'<line class="grid" x1="{left}" y1="{yy:.2f}" x2="{left + plot_width}" y2="{yy:.2f}"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" font-size="10">{value:g}</text>'
        )
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
            '<text x="20" y="215" text-anchor="middle" font-size="12" transform="rotate(-90 20 215)">maximum error / tolerance</text>',
            f'<line x1="{left}" y1="{y(1.0):.2f}" x2="{left + plot_width}" y2="{y(1.0):.2f}" stroke="#202124" stroke-width="1.5"/>',
        ]
    )
    group_width = plot_width / len(models)
    metric_offsets = (-38.0, 0.0, 38.0)
    colors = {"low": "#D55E00", "high": "#0072B2"}
    for model_index, (model, label) in enumerate(models):
        center = left + group_width * (model_index + 0.5)
        for metric_index, (_, key, tolerance) in enumerate(metrics):
            x_center = center + metric_offsets[metric_index]
            for temperature_key, verdicts, offset in (
                ("high", high, -5.0),
                ("low", low, 5.0),
            ):
                if model not in verdicts:
                    continue
                normalized = float(verdicts[model][key]) / tolerance
                lines.append(
                    f'<circle cx="{x_center + offset:.2f}" cy="{y(normalized):.2f}" r="4.5" fill="{colors[temperature_key]}"/>'
                )
            lines.append(
                f'<text x="{x_center:.2f}" y="{top + plot_height + 18}" text-anchor="middle" font-size="9">{metrics[metric_index][0]}</text>'
            )
        lines.append(
            f'<text x="{center:.2f}" y="{top + plot_height + 39}" text-anchor="middle" font-size="11" font-weight="700">{label}</text>'
        )
    lines.extend(
        [
            '<circle cx="340" cy="438" r="5" fill="#0072B2"/>',
            '<text x="352" y="442" font-size="11">T=0.58</text>',
            '<circle cx="465" cy="438" r="5" fill="#D55E00"/>',
            '<text x="477" y="442" font-size="11">T=0.45</text>',
            '<text x="650" y="74" font-size="10">pass below 1</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-verdict", type=Path, required=True)
    parser.add_argument("--high-verdict", type=Path, required=True)
    parser.add_argument("--markov-low-verdict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    low = read_rows(args.low_verdict)
    high = read_rows(args.high_verdict)
    markov_rows = read_rows(args.markov_low_verdict)
    if len(markov_rows) != 1:
        raise ValueError("the low-temperature Markov verdict must contain one row")
    write_one(
        args.output,
        classify_empirical_path_crossover(low, high, markov_rows[0]),
    )
    write_svg(args.output_svg, low_verdicts=low, high_verdicts=high)


if __name__ == "__main__":
    main()
