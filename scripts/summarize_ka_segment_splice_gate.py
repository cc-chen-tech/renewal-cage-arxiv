#!/usr/bin/env python3
"""Select finite path-memory horizons from the frozen segment-splice gate."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path


MODELS = (
    "within_particle_segment_shuffle",
    "cross_particle_segment_splice",
)
CLAIM_FIELDS = (
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)
SUPPORT_FIELDS = (
    "realization_completeness_pass",
    "exact_information_pass",
    "provenance_claim_boundary_pass",
    "stationarity_control_pass",
    "precision_pass",
    "global_source_segment_schedule_preserved",
)
FROZEN_GRIDS = {
    0.45: (1, 2, 5, 10, 25, 50, 125, 250),
    0.58: (1, 2, 4, 8, 16, 32, 37),
}
FROZEN_REPLICATES = {0.45: (1, 2, 3), 0.58: (1, 2, 3, 4, 5)}


def _finite(row: dict[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid field: {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"field must be finite: {key}")
    return value


def select_monotone_memory_length(
    cell_rows: Sequence[dict[str, object]],
    *,
    model: str,
    temperature: float,
    block_count: int,
    required_grid: Sequence[int],
) -> int | None:
    """Return the first passing non-full length whose entire later tail passes."""

    grid = tuple(int(length) for length in required_grid)
    if (
        model not in MODELS
        or temperature not in FROZEN_GRIDS
        or grid != FROZEN_GRIDS[temperature]
        or block_count != grid[-1]
        or any(length < 1 for length in grid)
    ):
        raise ValueError("memory selector requires the exact frozen grid")
    selected = [
        row
        for row in cell_rows
        if row.get("model") == model
        and math.isclose(_finite(row, "temperature"), temperature)
    ]
    lengths = [int(_finite(row, "segment_length")) for row in selected]
    if len(selected) != len(grid) or sorted(lengths) != sorted(grid) or len(set(lengths)) != len(lengths):
        raise ValueError("memory selector requires one cell per frozen length")
    by_length = {int(_finite(row, "segment_length")): row for row in selected}
    curve_pass: dict[int, bool] = {}
    for length in grid:
        row = by_length[length]
        if any(_finite(row, field) != 1.0 for field in SUPPORT_FIELDS):
            raise ValueError("memory selector cannot use unresolved support gates")
        if any(_finite(row, field) != 0.0 for field in CLAIM_FIELDS):
            raise ValueError("memory selector claim boundaries are not closed")
        is_full = length == block_count
        if (
            _finite(row, "full_path_control") != float(is_full)
            or _finite(row, "memory_length_selectable") != float(not is_full)
        ):
            raise ValueError("full-path control is incorrectly marked selectable")
        curve_value = _finite(row, "curve_transfer_pass")
        if curve_value not in {0.0, 1.0}:
            raise ValueError("curve pass must be binary")
        curve_pass[length] = bool(curve_value)
    selectable = grid[:-1]
    for index, length in enumerate(selectable):
        if all(curve_pass[later] for later in selectable[index:]):
            return length
    return None


def _score_lookup(
    rows: Sequence[dict[str, object]],
    *,
    temperature: float,
) -> dict[tuple[str, int, int], float]:
    grid = FROZEN_GRIDS[temperature]
    replicates = FROZEN_REPLICATES[temperature]
    selected = [
        row for row in rows if math.isclose(_finite(row, "temperature"), temperature)
    ]
    expected = {
        (model, length, replicate)
        for model in MODELS
        for length in grid
        for replicate in replicates
    }
    keys = [
        (
            str(row.get("model")),
            int(_finite(row, "segment_length")),
            int(_finite(row, "replicate")),
        )
        for row in selected
    ]
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("replicate-score rows do not contain the frozen grid")
    if any(_finite(row, field) != 0.0 for row in selected for field in CLAIM_FIELDS):
        raise ValueError("replicate-score claim boundaries are not closed")
    return {
        key: _finite(row, "higher_order_score")
        for key, row in zip(keys, selected, strict=True)
    }


def classify_segment_splice_gate(
    low_cells: Sequence[dict[str, object]],
    high_cells: Sequence[dict[str, object]],
    low_replicate_scores: Sequence[dict[str, object]],
    high_replicate_scores: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Recompute finite horizons and select one preregistered mechanism state."""

    result: dict[str, object] = {
        "mechanism_state": "mechanism_unresolved",
        "gate_input_completeness_pass": 0.0,
        "global_source_segment_schedule_preserved": 0.0,
        "substantive_interpretation_condition": (
            "conditional_on_preserved_global_source_segment_schedule"
        ),
        "low_within_memory_length_resolved": 0.0,
        "low_within_memory_length": 0.0,
        "low_cross_memory_length_resolved": 0.0,
        "low_cross_memory_length": 0.0,
        "high_within_memory_length_resolved": 0.0,
        "high_within_memory_length": 0.0,
        "high_cross_memory_length_resolved": 0.0,
        "high_cross_memory_length": 0.0,
        "selected_low_replicate_scores_pass": 0.0,
        "persistent_within_strictly_better_all_replicates": 0.0,
        "within_cooling_memory_growth": 0.0,
        "cross_cooling_memory_growth": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    try:
        cells = {0.45: low_cells, 0.58: high_cells}
        scores = {
            0.45: _score_lookup(low_replicate_scores, temperature=0.45),
            0.58: _score_lookup(high_replicate_scores, temperature=0.58),
        }
        lengths: dict[tuple[float, str], int | None] = {}
        for temperature in (0.45, 0.58):
            grid = FROZEN_GRIDS[temperature]
            for model in MODELS:
                lengths[(temperature, model)] = select_monotone_memory_length(
                    cells[temperature],
                    model=model,
                    temperature=temperature,
                    block_count=grid[-1],
                    required_grid=grid,
                )
    except (ValueError, KeyError, TypeError):
        return result

    result["gate_input_completeness_pass"] = 1.0
    result["global_source_segment_schedule_preserved"] = 1.0
    for temperature, label in ((0.45, "low"), (0.58, "high")):
        for model, short in (
            ("within_particle_segment_shuffle", "within"),
            ("cross_particle_segment_splice", "cross"),
        ):
            length = lengths[(temperature, model)]
            result[f"{label}_{short}_memory_length_resolved"] = float(length is not None)
            result[f"{label}_{short}_memory_length"] = float(length or 0)

    low_within = lengths[(0.45, "within_particle_segment_shuffle")]
    low_cross = lengths[(0.45, "cross_particle_segment_splice")]
    high_within = lengths[(0.58, "within_particle_segment_shuffle")]
    high_cross = lengths[(0.58, "cross_particle_segment_splice")]
    result["within_cooling_memory_growth"] = float(
        low_within is not None and high_within is not None and low_within > high_within
    )
    result["cross_cooling_memory_growth"] = float(
        low_cross is not None and high_cross is not None and low_cross > high_cross
    )

    if low_within is None and low_cross is None:
        result["mechanism_state"] = "longer_or_richer_path_state_required"
        return result
    if low_cross is not None and (low_within is None or low_cross < low_within):
        result["mechanism_state"] = "null_family_pathology_unresolved"
        return result

    selected = []
    if low_within is not None:
        selected.append(("within_particle_segment_shuffle", low_within))
    if low_cross is not None:
        selected.append(("cross_particle_segment_splice", low_cross))
    low_scores = scores[0.45]
    selected_scores_pass = all(
        low_scores[(model, length, replicate)] <= 1.0
        for model, length in selected
        for replicate in FROZEN_REPLICATES[0.45]
    )
    result["selected_low_replicate_scores_pass"] = float(selected_scores_pass)
    if not selected_scores_pass:
        return result

    if low_within == low_cross:
        result["mechanism_state"] = (
            "finite_single_particle_path_memory_sufficient_conditional_on_global_schedule"
        )
        return result

    if low_within is not None and (low_cross is None or low_cross > low_within):
        strict_order = all(
            low_scores[("within_particle_segment_shuffle", low_within, replicate)]
            < low_scores[("cross_particle_segment_splice", low_within, replicate)]
            for replicate in FROZEN_REPLICATES[0.45]
        )
        result["persistent_within_strictly_better_all_replicates"] = float(strict_order)
        if strict_order:
            result["mechanism_state"] = (
                "persistent_environment_identity_required_beyond_local_path"
            )
    return result


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty segment-splice verdict")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_gate_svg(
    path: Path,
    cell_rows: Sequence[dict[str, object]],
    verdict: dict[str, object],
) -> None:
    """Write a deterministic two-panel normalized-error scan."""

    width, height = 1180, 680
    panel_width, panel_height = 455.0, 410.0
    panel_top = 105.0
    panel_lefts = {0.45: 90.0, 0.58: 650.0}
    colors = {
        "within_particle_segment_shuffle": "#176b87",
        "cross_particle_segment_splice": "#c64b2c",
    }
    metrics = (
        ("MSD / 0.10", "maximum_ensemble_msd_relative_error", 0.10, ""),
        ("NGP / 0.30", "maximum_ensemble_ngp_absolute_error", 0.30, "6 4"),
        ("Fs / 0.03", "maximum_ensemble_fs_absolute_error", 0.03, "2 4"),
    )
    elements: list[str] = []
    for temperature in (0.45, 0.58):
        left = panel_lefts[temperature]
        local = [
            row
            for row in cell_rows
            if math.isclose(_finite(row, "temperature"), temperature)
        ]
        lengths = sorted({int(_finite(row, "segment_length")) for row in local})
        if lengths != list(FROZEN_GRIDS[temperature]):
            raise ValueError("SVG requires the complete frozen cell grid")
        log_min = math.log(float(lengths[0]))
        log_max = math.log(float(lengths[-1]))

        def x_position(length: int) -> float:
            fraction = (math.log(float(length)) - log_min) / (log_max - log_min)
            return left + fraction * panel_width

        def y_position(value: float) -> float:
            return panel_top + panel_height * (1.0 - min(max(value, 0.0), 2.5) / 2.5)

        gate_y = y_position(1.0)
        full_x = x_position(lengths[-1])
        elements.extend(
            (
                f'<rect x="{left:.1f}" y="{panel_top:.1f}" width="{panel_width:.1f}" height="{panel_height:.1f}" fill="#ffffff" stroke="#c9c9c3"/>',
                f'<line x1="{left:.1f}" y1="{gate_y:.1f}" x2="{left + panel_width:.1f}" y2="{gate_y:.1f}" stroke="#333333" stroke-width="1.5" stroke-dasharray="7 5"/>',
                f'<text x="{left + 8:.1f}" y="{gate_y - 7:.1f}" font-size="12">tolerance = 1</text>',
                f'<line x1="{full_x:.1f}" y1="{panel_top:.1f}" x2="{full_x:.1f}" y2="{panel_top + panel_height:.1f}" stroke="#777777" stroke-width="2"/>',
                f'<text x="{full_x - 6:.1f}" y="{panel_top + 18:.1f}" text-anchor="end" font-size="11">full-path control</text>',
                f'<text x="{left + panel_width / 2:.1f}" y="82" text-anchor="middle" font-size="20" font-weight="bold">T = {temperature:.2f}</text>',
            )
        )
        for model in MODELS:
            model_rows = {
                int(_finite(row, "segment_length")): row
                for row in local
                if row.get("model") == model
            }
            if set(model_rows) != set(lengths):
                raise ValueError("SVG model rows do not cover the frozen grid")
            for _, key, tolerance, dash in metrics:
                points = []
                for length in lengths:
                    normalized = _finite(model_rows[length], key) / tolerance
                    points.append((x_position(length), y_position(normalized)))
                dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
                point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
                elements.append(
                    f'<polyline points="{point_text}" fill="none" stroke="{colors[model]}" stroke-width="2.2"{dash_attribute}/>'
                )
                elements.extend(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{colors[model]}"/>'
                    for x, y in points
                )
        for length in lengths:
            x = x_position(length)
            elements.append(
                f'<text x="{x:.2f}" y="{panel_top + panel_height + 23:.1f}" text-anchor="middle" font-size="11">{20 * length:g}</text>'
            )
        elements.append(
            f'<text x="{left + panel_width / 2:.1f}" y="{panel_top + panel_height + 52:.1f}" text-anchor="middle">retained horizon tau_L</text>'
        )

    legend = []
    for index, (model, label) in enumerate(
        (
            ("within_particle_segment_shuffle", "within-particle segments"),
            ("cross_particle_segment_splice", "cross-particle splice"),
        )
    ):
        y = 574 + 24 * index
        legend.append(
            f'<line x1="90" y1="{y}" x2="126" y2="{y}" stroke="{colors[model]}" stroke-width="3"/><text x="136" y="{y + 5}">{label}</text>'
        )
    for index, (label, _, _, dash) in enumerate(metrics):
        x = 390 + 190 * index
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        legend.append(
            f'<line x1="{x}" y1="585" x2="{x + 36}" y2="585" stroke="#333333" stroke-width="2"{dash_attribute}/><text x="{x + 44}" y="590">{label}</text>'
        )
    mechanism_state = str(verdict.get("mechanism_state", "mechanism_unresolved"))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f5f5f1"/>
<style>text{{font-family:Arial,sans-serif;fill:#202020;font-size:14px;letter-spacing:0}}</style>
<text x="590" y="36" text-anchor="middle" font-size="24" font-weight="bold">Segment-splice path-memory gate</text>
{''.join(elements)}
{''.join(legend)}
<text x="590" y="640" text-anchor="middle" font-size="15" font-weight="bold">{mechanism_state}</text>
<text x="590" y="662" text-anchor="middle" font-size="12">Path-hierarchy diagnostic only; no microscopic, spatial-facilitation, or thermodynamic claim.</text>
</svg>'''
    if "nan" in svg.lower() or "inf" in svg.lower():
        raise ValueError("gate SVG contains nonfinite coordinates")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the frozen segment-splice path-memory mechanism state."
    )
    parser.add_argument("--low-cells", type=Path, required=True)
    parser.add_argument("--high-cells", type=Path, required=True)
    parser.add_argument("--low-replicate-scores", type=Path, required=True)
    parser.add_argument("--high-replicate-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    low_cells = _read_rows(args.low_cells)
    high_cells = _read_rows(args.high_cells)
    verdict = classify_segment_splice_gate(
        low_cells,
        high_cells,
        _read_rows(args.low_replicate_scores),
        _read_rows(args.high_replicate_scores),
    )
    _write_rows(args.output, [verdict])
    write_gate_svg(args.output_svg, low_cells + high_cells, verdict)


if __name__ == "__main__":
    main()
