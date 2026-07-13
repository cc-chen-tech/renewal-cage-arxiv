#!/usr/bin/env python3
"""Combine nonlinear path-memory and cumulant-range diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Sequence


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty nonlinear-path gate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _flag(row: dict[str, object], key: str) -> bool:
    return float(row[key]) == 1.0


def _high_resolution_state(row: dict[str, object]) -> str:
    if _flag(row, "nonlinear_single_particle_path_memory_required"):
        return "nonlinear_memory_required"
    valid_null = all(
        _flag(row, key)
        for key in (
            "surrogate_quality_pass",
            "surrogate_precision_pass",
            "stationarity_control_pass",
            "contiguous_ensemble_curve_pass",
            "radial_surrogate_msd_pass",
        )
    ) and not _flag(row, "radial_surrogate_higher_order_failure")
    return "radial_spectral_null_sufficient" if valid_null else "unresolved"


def _horizons(rows: Sequence[dict[str, object]]) -> dict[float, float]:
    result = {
        float(row["wave_number"]): float(row["longest_contiguous_valid_lag"])
        for row in rows
    }
    if not result:
        raise ValueError("cumulant validity rows must not be empty")
    return result


def classify_nonlinear_path_gate(
    low_verdict: dict[str, object],
    high_block20_verdict: dict[str, object],
    high_block10_verdict: dict[str, object],
    low_cumulant_validity: Sequence[dict[str, object]],
    high_cumulant_validity: Sequence[dict[str, object]],
    empirical_path_crossover: dict[str, object],
) -> dict[str, object]:
    low_required_count = int(float(low_verdict["required_replicate_count"]))
    low_gate_ready = all(
        _flag(low_verdict, key)
        for key in (
            "surrogate_quality_pass",
            "surrogate_precision_pass",
            "stationarity_control_pass",
            "contiguous_ensemble_curve_pass",
            "radial_surrogate_msd_pass",
            "radial_surrogate_higher_order_failure",
            "replicate_consensus_pass",
            "nonlinear_single_particle_path_memory_required",
            "linear_spectrum_null_rejected",
        )
    ) and all(
        int(float(low_verdict[key])) == low_required_count
        for key in (
            "surrogate_failure_replicate_count",
            "paired_contiguous_better_replicate_count",
        )
    )
    state20 = _high_resolution_state(high_block20_verdict)
    state10 = _high_resolution_state(high_block10_verdict)
    high_resolved = state20 == state10 and state20 != "unresolved"
    high_sensitivity = (
        state20 != state10
        or _flag(high_block20_verdict, "radial_surrogate_higher_order_failure")
        != _flag(high_block10_verdict, "radial_surrogate_higher_order_failure")
        or _flag(high_block20_verdict, "surrogate_quality_pass")
        != _flag(high_block10_verdict, "surrogate_quality_pass")
    )
    binary_crossover = (
        low_gate_ready
        and high_resolved
        and state20 == "radial_spectral_null_sufficient"
    )
    low_horizons = _horizons(low_cumulant_validity)
    high_horizons = _horizons(high_cumulant_validity)
    low_ordered = all(
        low_horizons[left] >= low_horizons[right]
        for left, right in zip(
            sorted(low_horizons)[:-1],
            sorted(low_horizons)[1:],
        )
    )
    empirical_support = _flag(
        empirical_path_crossover,
        "single_particle_multiblock_path_memory_required",
    ) and _flag(empirical_path_crossover, "ordered_recoil_path_required")
    result: dict[str, object] = {
        "low_temperature": float(low_verdict["temperature"]),
        "high_temperature": float(high_block20_verdict["temperature"]),
        "low_temperature_gate_ready": float(low_gate_ready),
        "low_temperature_nonlinear_path_memory_required": float(
            _flag(low_verdict, "nonlinear_single_particle_path_memory_required")
        ),
        "low_temperature_linear_spectrum_null_rejected": float(
            _flag(low_verdict, "linear_spectrum_null_rejected")
        ),
        "low_temperature_surrogate_failure_replicate_count": float(
            low_verdict["surrogate_failure_replicate_count"]
        ),
        "low_temperature_paired_contiguous_better_replicate_count": float(
            low_verdict["paired_contiguous_better_replicate_count"]
        ),
        "low_temperature_required_replicate_count": float(low_required_count),
        "high_block20_state": state20,
        "high_block10_state": state10,
        "high_block20_quality_pass": float(
            _flag(high_block20_verdict, "surrogate_quality_pass")
        ),
        "high_block10_quality_pass": float(
            _flag(high_block10_verdict, "surrogate_quality_pass")
        ),
        "high_block20_stationarity_pass": float(
            _flag(high_block20_verdict, "stationarity_control_pass")
        ),
        "high_block10_stationarity_pass": float(
            _flag(high_block10_verdict, "stationarity_control_pass")
        ),
        "high_block20_higher_order_failure": float(
            _flag(high_block20_verdict, "radial_surrogate_higher_order_failure")
        ),
        "high_block10_higher_order_failure": float(
            _flag(high_block10_verdict, "radial_surrogate_higher_order_failure")
        ),
        "high_temperature_resolution_sensitivity": float(high_sensitivity),
        "high_temperature_mechanism_resolved": float(high_resolved),
        "binary_temperature_crossover_claim_allowed": float(binary_crossover),
        "low_k_cumulant_horizon_order_pass": float(low_ordered),
        "observed_cumulant_diagnostic_only": 1.0,
        "empirical_ordered_path_result_reproduced": float(empirical_support),
        "publication_mechanism_result_ready": float(
            low_gate_ready and low_ordered and empirical_support
        ),
        "unique_microscopic_model_selected": 0.0,
        "next_minimal_model_candidate": "finite_lifetime_reversible_cage_state",
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for temperature_name, horizons in (
        ("low", low_horizons),
        ("high", high_horizons),
    ):
        for wave_number, horizon in sorted(horizons.items()):
            suffix = f"k{wave_number:g}".replace(".", "p")
            result[f"{temperature_name}_cumulant_horizon_{suffix}"] = horizon
    return result


def normalized_model_errors(
    summary_rows: Sequence[dict[str, object]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model in ("contiguous_empirical_path", "radial_multivariate_surrogate"):
        selected = [row for row in summary_rows if str(row["model"]) == model]
        if not selected:
            raise ValueError(f"low-temperature summary is missing model {model}")
        fs_names = sorted(
            {
                name
                for row in selected
                for name in row
                if name.startswith("ensemble_absolute_error_fs_k")
            }
        )
        result[model] = {
            "MSD": max(float(row["ensemble_msd_relative_error"]) for row in selected)
            / 0.10,
            "NGP": max(float(row["ensemble_ngp_absolute_error"]) for row in selected)
            / 0.30,
            "Fs": max(float(row[name]) for row in selected for name in fs_names) / 0.03,
        }
    return result


def _associated_path(path: Path, old_suffix: str, new_suffix: str) -> Path:
    if not path.name.endswith(old_suffix):
        raise ValueError(f"expected path ending in {old_suffix}: {path}")
    return path.with_name(path.name[: -len(old_suffix)] + new_suffix)


def write_svg(
    path: Path,
    *,
    normalized_errors: dict[str, dict[str, float]],
    cumulant_rows: Sequence[dict[str, object]],
) -> None:
    width, height = 1080, 520
    left = (70.0, 82.0, 430.0, 350.0)
    right = (600.0, 82.0, 430.0, 350.0)
    y_max = 6.0
    colors = {
        "contiguous_empirical_path": "#167a72",
        "radial_multivariate_surrogate": "#d65a31",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1080" height="520" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#20242a;letter-spacing:0}.title{font-size:20px;font-weight:700}.panel{font-size:15px;font-weight:700}.axis{font-size:12px}.legend{font-size:12px}</style>',
        '<text x="40" y="32" class="title">Nonlinear cage-path cumulant gate</text>',
        '<text x="70" y="62" class="panel">A  Held-out errors / frozen tolerance</text>',
        '<text x="600" y="62" class="panel">B  Observed-cumulant Fs error</text>',
    ]

    def y_coordinate(value: float, panel: tuple[float, float, float, float]) -> float:
        return panel[1] + panel[3] * (1.0 - min(max(value, 0.0), y_max) / y_max)

    for panel in (left, right):
        x, y, w, h = panel
        parts.extend(
            [
                f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" stroke="#343a40"/>',
                f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" stroke="#343a40"/>',
            ]
        )
        for tick in (0, 1, 2, 4, 6):
            yy = y_coordinate(float(tick), panel)
            parts.append(
                f'<line x1="{x}" y1="{yy:.2f}" x2="{x+w}" y2="{yy:.2f}" stroke="#e3e6e8"/>'
            )
            parts.append(f'<text x="{x-10}" y="{yy+4:.2f}" text-anchor="end" class="axis">{tick}</text>')
        threshold_y = y_coordinate(1.0, panel)
        parts.append(
            f'<line x1="{x}" y1="{threshold_y:.2f}" x2="{x+w}" y2="{threshold_y:.2f}" stroke="#9b2c2c" stroke-width="1.5" stroke-dasharray="6 5"/>'
        )

    categories = ("MSD", "NGP", "Fs")
    x_positions = [left[0] + 75 + index * 140 for index in range(3)]
    offsets = {
        "contiguous_empirical_path": -16.0,
        "radial_multivariate_surrogate": 16.0,
    }
    for model, values in normalized_errors.items():
        color = colors[model]
        points = []
        for category, x in zip(categories, x_positions):
            xx = x + offsets[model]
            yy = y_coordinate(values[category], left)
            points.append(f"{xx:.2f},{yy:.2f}")
            parts.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5" fill="{color}"/>')
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
    for category, x in zip(categories, x_positions):
        parts.append(f'<text x="{x}" y="452" text-anchor="middle" class="axis">{category}</text>')
    parts.append('<text x="48" y="250" transform="rotate(-90 48 250)" text-anchor="middle" class="axis">normalized error</text>')

    valid_cumulant = [
        row
        for row in cumulant_rows
        if math.isfinite(float(row["absolute_error"])) and float(row["lag"]) > 0.0
    ]
    lags = [float(row["lag"]) for row in valid_cumulant]
    log_min, log_max = math.log10(min(lags)), math.log10(max(lags))
    wave_colors = {2.0: "#167a72", 4.0: "#d65a31", 7.25: "#4b5563"}
    for wave_number in sorted({float(row["wave_number"]) for row in valid_cumulant}):
        selected = sorted(
            (row for row in valid_cumulant if float(row["wave_number"]) == wave_number),
            key=lambda row: float(row["lag"]),
        )
        points = []
        for row in selected:
            lag = float(row["lag"])
            ratio = float(row["absolute_error"]) / float(row["error_tolerance"])
            xx = right[0] + right[2] * (math.log10(lag) - log_min) / (log_max - log_min)
            yy = y_coordinate(ratio, right)
            points.append(f"{xx:.2f},{yy:.2f}")
        color = wave_colors.get(wave_number, "#6b7280")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
    for lag in sorted({min(lags), max(lags)}):
        xx = right[0] + right[2] * (math.log10(lag) - log_min) / (log_max - log_min)
        parts.append(f'<text x="{xx:.2f}" y="452" text-anchor="middle" class="axis">{lag:g}</text>')
    parts.append('<text x="815" y="480" text-anchor="middle" class="axis">lag (log scale)</text>')
    parts.append('<text x="578" y="250" transform="rotate(-90 578 250)" text-anchor="middle" class="axis">|Fs(4)-Fs| / 0.03</text>')

    legend_items = [
        (90, "contiguous path", colors["contiguous_empirical_path"]),
        (285, "radial + spectrum null", colors["radial_multivariate_surrogate"]),
        (675, "k=2", wave_colors[2.0]),
        (775, "k=4", wave_colors[4.0]),
        (875, "k=7.25", wave_colors[7.25]),
    ]
    for legend_x, label, color in legend_items:
        safe = html.escape(label)
        parts.append(f'<line x1="{legend_x}" y1="503" x2="{legend_x+22}" y2="503" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+28}" y="507" class="legend">{safe}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-verdict", type=Path, required=True)
    parser.add_argument("--high-block20-verdict", type=Path, required=True)
    parser.add_argument("--high-block10-verdict", type=Path, required=True)
    parser.add_argument("--low-cumulant-validity", type=Path, required=True)
    parser.add_argument("--high-cumulant-validity", type=Path, required=True)
    parser.add_argument("--empirical-path-crossover", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args(argv)
    low = read_rows(args.low_verdict)[0]
    high20 = read_rows(args.high_block20_verdict)[0]
    high10 = read_rows(args.high_block10_verdict)[0]
    low_validity = read_rows(args.low_cumulant_validity)
    high_validity = read_rows(args.high_cumulant_validity)
    empirical = read_rows(args.empirical_path_crossover)[0]
    result = classify_nonlinear_path_gate(
        low,
        high20,
        high10,
        low_validity,
        high_validity,
        empirical,
    )
    low_summary = read_rows(
        _associated_path(args.low_verdict, "_verdict.csv", "_summary.csv")
    )
    normalized_errors = normalized_model_errors(low_summary)
    for model, values in normalized_errors.items():
        model_key = "contiguous" if model == "contiguous_empirical_path" else "radial"
        for metric, value in values.items():
            result[f"low_{model_key}_{metric.lower()}_normalized_error"] = value
    low_cumulant_rows = read_rows(
        _associated_path(
            args.low_cumulant_validity,
            "_validity.csv",
            "_rows.csv",
        )
    )
    write_rows(args.output_csv, [result])
    write_svg(
        args.output_svg,
        normalized_errors=normalized_errors,
        cumulant_rows=low_cumulant_rows,
    )


if __name__ == "__main__":
    main()
