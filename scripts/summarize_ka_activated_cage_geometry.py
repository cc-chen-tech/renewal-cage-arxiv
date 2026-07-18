#!/usr/bin/env python3
"""Evaluate calibration cage-jump geometry against held-out KA shape diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from summarize_ka_transport_clock_shape_quotient import (
    FROZEN_PROTOCOLS,
    _validated_stationarity_pass,
    manifests_from_provenance,
)


STRONG_ZERO_FLAGS = (
    "blind_prediction_claim_allowed",
    "finite_exchange_resolved",
    "static_environment_resolved",
    "spatial_facilitation_resolved",
    "activated_cage_geometry_resolved",
    "microdynamic_closure_claim_allowed",
    "thermodynamic_claim_allowed",
)

FS_ABSOLUTE_ERROR_TOLERANCE = 0.03
MINIMUM_SUPPORT_FRACTION = 0.8
EXPECTED_REPLICATE_COUNT = {0.45: 3, 0.58: 5}
CSV_FLOAT_SIGNIFICANT_DIGITS = 10
WAVE_NUMBERS = {"k2": 2.0, "k4": 4.0, "k7p25": 7.25}


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def empirical_geometry_quotient(
    *,
    msd: float,
    ngp: float,
    jump_msd: float,
    jump_component_fourth_moment: float,
    jump_characteristic: Mapping[float, float],
) -> dict[str, object]:
    msd = _positive_finite(msd, "msd")
    ngp = _nonnegative_finite(ngp, "ngp")
    jump_msd = _positive_finite(jump_msd, "jump_msd")
    fourth = _positive_finite(
        jump_component_fourth_moment,
        "jump_component_fourth_moment",
    )
    if not jump_characteristic:
        raise ValueError("jump_characteristic must be nonempty")
    characteristic = {float(k): float(value) for k, value in jump_characteristic.items()}
    if any(
        not math.isfinite(k)
        or k <= 0.0
        or not math.isfinite(value)
        or value < -1.0
        or value > 1.0
        for k, value in characteristic.items()
    ):
        raise ValueError("jump characteristic entries must have positive k and values in [-1, 1]")

    count = ngp * msd**2 / (3.0 * fourth)
    cage = (msd - count * jump_msd) / 6.0
    supported = count >= 0.0 and cage >= 0.0
    predicted = (
        {
            wave_number: math.exp(
                -wave_number**2 * cage + count * (value - 1.0)
            )
            for wave_number, value in characteristic.items()
        }
        if supported
        else {}
    )
    return {
        "supported": float(supported),
        "mean_event_count": count,
        "cage_variance": cage,
        "predicted_fs": predicted,
    }


def fixed_length_geometry_quotient(
    *,
    msd: float,
    ngp: float,
    jump_msd: float,
    wave_numbers: Iterable[float],
) -> dict[str, object]:
    jump_msd = _positive_finite(jump_msd, "jump_msd")
    length = math.sqrt(jump_msd)
    characteristic = {}
    for value in wave_numbers:
        wave_number = _positive_finite(value, "wave_number")
        argument = wave_number * length
        characteristic[wave_number] = math.sin(argument) / argument
    return empirical_geometry_quotient(
        msd=msd,
        ngp=ngp,
        jump_msd=jump_msd,
        jump_component_fourth_moment=jump_msd**2 / 5.0,
        jump_characteristic=characteristic,
    )


def classify_geometry_gate(
    rows: list[dict[str, object]],
    *,
    stationarity_pass: Mapping[float, bool],
    provenance_pass: Mapping[float, bool],
) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("geometry gate requires nonempty rows")
    gates = []
    temperatures = sorted({float(row["temperature"]) for row in rows})
    for temperature in temperatures:
        local = [row for row in rows if float(row["temperature"]) == temperature]
        supported = [
            row for row in local if float(row["empirical_geometry_supported"]) == 1.0
        ]
        replicate_ids = sorted({int(row["replicate"]) for row in local})
        supported_replicates = {
            int(row["replicate"])
            for row in supported
        }
        expected_replicates = EXPECTED_REPLICATE_COUNT.get(
            temperature,
            len(replicate_ids),
        )
        all_replicates_supported = (
            len(replicate_ids) == expected_replicates
            and supported_replicates == set(replicate_ids)
        )
        support_fraction = len(supported) / len(local)
        support_pass = (
            support_fraction >= MINIMUM_SUPPORT_FRACTION
            and all_replicates_supported
        )
        maxima: dict[str, float | None] = {}
        for key in ("k2", "k4", "k7p25"):
            field = f"empirical_fs_{key}_absolute_error"
            values = [float(row[field]) for row in supported]
            maxima[key] = max(values) if values else None
        curve_pass = bool(supported) and all(
            value is not None and value <= FS_ABSOLUTE_ERROR_TOLERANCE
            for value in maxima.values()
        )
        source_stationarity = bool(stationarity_pass.get(temperature, False))
        source_provenance = bool(provenance_pass.get(temperature, False))
        primary = temperature == 0.45
        exploratory_support = (
            primary
            and source_stationarity
            and source_provenance
            and support_pass
            and curve_pass
        )
        if exploratory_support:
            status = "empirical_activated_jump_geometry_supported_exploratory"
            next_action = "derive_distributed_basin_langevin_kramers_closure"
        elif not source_provenance:
            status = "unresolved_provenance"
            next_action = "repair_source_provenance"
        elif not source_stationarity:
            status = "canary_only_nonstationary_source"
            next_action = "do_not_use_for_cooling_claim"
        elif not support_pass:
            status = "compound_poisson_cage_decomposition_unsupported"
            next_action = "test_count_or_cage_jump_dependence"
        else:
            status = "empirical_jump_geometry_shape_failure"
            next_action = "test_correlated_or_non_poisson_event_geometry"
        gate = {
            "temperature": temperature,
            "analysis_status": status,
            "replicate_count": float(len(replicate_ids)),
            "row_count": float(len(local)),
            "source_stationarity_pass": float(source_stationarity),
            "replicate_provenance_validation_pass": float(source_provenance),
            "minimum_support_fraction": MINIMUM_SUPPORT_FRACTION,
            "empirical_supported_row_count": float(len(supported)),
            "empirical_support_fraction": support_fraction,
            "empirical_support_coverage_pass": float(support_pass),
            "all_primary_replicates_supported": float(all_replicates_supported),
            "fixed_length_supported_row_count": float(
                sum(float(row["fixed_length_geometry_supported"]) for row in local)
            ),
            "fixed_length_support_fraction": sum(
                float(row["fixed_length_geometry_supported"]) for row in local
            )
            / len(local),
            "fs_k2_empirical_max_absolute_error": maxima["k2"] if maxima["k2"] is not None else "",
            "fs_k4_empirical_max_absolute_error": maxima["k4"] if maxima["k4"] is not None else "",
            "fs_k7p25_empirical_max_absolute_error": maxima["k7p25"] if maxima["k7p25"] is not None else "",
            "fs_absolute_error_tolerance": FS_ABSOLUTE_ERROR_TOLERANCE,
            "curve_transfer_pass": float(curve_pass),
            "empirical_activated_jump_geometry_supported_exploratory": float(
                exploratory_support
            ),
            "high_temperature_canary_only": float(not primary),
            "next_required_action": next_action,
        }
        gate.update({flag: 0.0 for flag in STRONG_ZERO_FLAGS})
        gates.append(gate)
    return gates


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_csv_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize a nonfinite geometry value")
        return format(value, f".{CSV_FLOAT_SIGNIFICANT_DIGITS}g")
    return value


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty geometry table")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("geometry output rows must share one schema")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: canonical_csv_value(value) for field, value in row.items()}
            for row in rows
        )


def _validated_geometry_by_replicate(
    rows: Sequence[dict[str, object]],
    *,
    temperature: float,
) -> dict[int, dict[str, object]]:
    expected = EXPECTED_REPLICATE_COUNT[temperature]
    selected: dict[int, dict[str, object]] = {}
    for row in rows:
        if float(row["temperature"]) != temperature:
            raise ValueError("geometry table temperature mismatch")
        replicate_value = float(row["replicate"])
        if not replicate_value.is_integer():
            raise ValueError("geometry replicate labels must be exact integers")
        replicate = int(replicate_value)
        if replicate in selected:
            raise ValueError("geometry table contains duplicate replicates")
        if float(row["calibration_events_only"]) != 1.0 or float(row["heldout_events_used"]) != 0.0:
            raise ValueError("geometry table is not calibration-only")
        for field in STRONG_ZERO_FLAGS:
            if float(row[field]) != 0.0:
                raise ValueError("geometry claim boundaries must remain closed")
        for field in (
            "event_count",
            "jump_msd",
            "jump_radial_fourth_moment",
            "jump_component_fourth_moment",
        ):
            _positive_finite(float(row[field]), field)
        for key in WAVE_NUMBERS:
            value = float(row[f"jump_characteristic_{key}"])
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("geometry characteristic must lie in [-1, 1]")
        selected[replicate] = row
    if tuple(sorted(selected)) != tuple(range(1, expected + 1)):
        raise ValueError("geometry table does not cover the frozen replicate grid")
    return selected


def compute_geometry_rows(
    gamma_rows: Sequence[dict[str, object]],
    geometry_by_temperature: Mapping[float, Sequence[dict[str, object]]],
) -> list[dict[str, object]]:
    geometry = {
        temperature: _validated_geometry_by_replicate(
            geometry_by_temperature[temperature],
            temperature=temperature,
        )
        for temperature in (0.45, 0.58)
    }
    expected_keys = {
        (temperature, replicate, lag)
        for temperature in (0.45, 0.58)
        for replicate in range(1, EXPECTED_REPLICATE_COUNT[temperature] + 1)
        for lag in FROZEN_PROTOCOLS[temperature]["lags"]
    }
    source: dict[tuple[float, int, int], dict[str, object]] = {}
    for row in gamma_rows:
        temperature = float(row["temperature"])
        replicate = int(row["replicate"])
        lag = int(row["lag"])
        key = (temperature, replicate, lag)
        if key in source:
            raise ValueError("gamma diagnostic contains duplicate rows")
        source[key] = row
    if set(source) != expected_keys:
        raise ValueError("gamma diagnostic does not cover the frozen grid")

    result = []
    for temperature, replicate, lag in sorted(expected_keys):
        diagnostic = source[(temperature, replicate, lag)]
        micro = geometry[temperature][replicate]
        characteristic = {
            wave_number: float(micro[f"jump_characteristic_{key}"])
            for key, wave_number in WAVE_NUMBERS.items()
        }
        empirical = empirical_geometry_quotient(
            msd=float(diagnostic["heldout_msd"]),
            ngp=float(diagnostic["heldout_ngp"]),
            jump_msd=float(micro["jump_msd"]),
            jump_component_fourth_moment=float(
                micro["jump_component_fourth_moment"]
            ),
            jump_characteristic=characteristic,
        )
        fixed = fixed_length_geometry_quotient(
            msd=float(diagnostic["heldout_msd"]),
            ngp=float(diagnostic["heldout_ngp"]),
            jump_msd=float(micro["jump_msd"]),
            wave_numbers=WAVE_NUMBERS.values(),
        )
        row: dict[str, object] = {
            "temperature": temperature,
            "replicate": replicate,
            "lag": lag,
            "heldout_msd": float(diagnostic["heldout_msd"]),
            "heldout_ngp": float(diagnostic["heldout_ngp"]),
            "jump_event_count": float(micro["event_count"]),
            "jump_msd": float(micro["jump_msd"]),
            "jump_radial_fourth_moment": float(micro["jump_radial_fourth_moment"]),
            "jump_component_fourth_moment": float(micro["jump_component_fourth_moment"]),
            "jump_characteristic_k2": characteristic[2.0],
            "jump_characteristic_k4": characteristic[4.0],
            "jump_characteristic_k7p25": characteristic[7.25],
            "empirical_geometry_supported": empirical["supported"],
            "empirical_mean_event_count": empirical["mean_event_count"],
            "empirical_cage_variance": empirical["cage_variance"],
            "fixed_length_geometry_supported": fixed["supported"],
            "fixed_length_mean_event_count": fixed["mean_event_count"],
            "fixed_length_cage_variance": fixed["cage_variance"],
            "heldout_msd_used_as_diagnostic_input": 1.0,
            "heldout_ngp_used_as_diagnostic_input": 1.0,
            "calibration_jump_geometry_only": 1.0,
        }
        for key, wave_number in WAVE_NUMBERS.items():
            observed = float(diagnostic[f"observed_fs_{key}"])
            gamma = float(diagnostic[f"gamma_fs_{key}"])
            empirical_prediction = empirical["predicted_fs"].get(wave_number, "")
            fixed_prediction = fixed["predicted_fs"].get(wave_number, "")
            row[f"observed_fs_{key}"] = observed
            row[f"gamma_fs_{key}"] = gamma
            row[f"gamma_fs_{key}_absolute_error"] = abs(gamma - observed)
            row[f"empirical_fs_{key}"] = empirical_prediction
            row[f"empirical_fs_{key}_absolute_error"] = (
                abs(float(empirical_prediction) - observed)
                if empirical_prediction != ""
                else ""
            )
            row[f"empirical_fs_{key}_signed_residual"] = (
                float(empirical_prediction) - observed
                if empirical_prediction != ""
                else ""
            )
            row[f"fixed_length_fs_{key}"] = fixed_prediction
            row[f"fixed_length_fs_{key}_absolute_error"] = (
                abs(float(fixed_prediction) - observed)
                if fixed_prediction != ""
                else ""
            )
        row.update({field: 0.0 for field in STRONG_ZERO_FLAGS})
        result.append(row)
    return result


def _svg_text(
    parts: list[str],
    x: float,
    y: float,
    value: object,
    *,
    size: int = 12,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#263238",
) -> None:
    parts.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{html.escape(str(value))}</text>'
    )


def write_geometry_svg(
    path: Path,
    gates: Sequence[dict[str, object]],
) -> None:
    if {float(row["temperature"]) for row in gates} != {0.45, 0.58}:
        raise ValueError("geometry SVG requires exact T045/T058 gates")
    width, height = 980, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    _svg_text(parts, 60, 42, "Activated cage-jump geometry quotient", size=24, weight=700)
    _svg_text(parts, 60, 68, "absolute Fs error (tolerance = 0.03)", size=13, fill="#46515c")
    for panel_index, temperature in enumerate((0.45, 0.58)):
        gate = next(row for row in gates if float(row["temperature"]) == temperature)
        left = 60 + 450 * panel_index
        top, panel_width, panel_height = 100, 410, 300
        parts.append(
            f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="#fbfcfd" stroke="#c7ced5"/>'
        )
        title = "T=0.45 primary" if temperature == 0.45 else "T=0.58 canary only"
        _svg_text(parts, left + 18, top + 30, title, size=16, weight=700)
        values = [
            float(gate[f"fs_{key}_empirical_max_absolute_error"])
            for key in WAVE_NUMBERS
        ]
        y_max = max(0.04, max(values) * 1.15)
        plot_left, plot_right = left + 55, left + 385
        plot_top, plot_bottom = top + 58, top + 238
        for tick in (0.0, FS_ABSOLUTE_ERROR_TOLERANCE, y_max):
            y = plot_bottom - tick / y_max * (plot_bottom - plot_top)
            color = "#a43b32" if tick == FS_ABSOLUTE_ERROR_TOLERANCE else "#e2e6ea"
            dash = ' stroke-dasharray="5 4"' if tick == FS_ABSOLUTE_ERROR_TOLERANCE else ""
            parts.append(
                f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" stroke="{color}"{dash}/>'
            )
            _svg_text(parts, plot_left - 7, y + 4, f"{tick:.3g}", size=10, anchor="end", fill="#5b6570")
        spacing = (plot_right - plot_left) / 3
        for index, (key, value) in enumerate(zip(WAVE_NUMBERS, values)):
            center = plot_left + spacing * (index + 0.5)
            bar_height = value / y_max * (plot_bottom - plot_top)
            color = "#2f7d68" if value <= FS_ABSOLUTE_ERROR_TOLERANCE else "#b34a42"
            parts.append(
                f'<rect x="{center - 16:.2f}" y="{plot_bottom - bar_height:.2f}" width="32" height="{bar_height:.2f}" fill="{color}"/>'
            )
            label = {"k2": "k=2", "k4": "k=4", "k7p25": "k=7.25"}[key]
            _svg_text(parts, center, plot_bottom + 20, label, size=11, anchor="middle")
        _svg_text(
            parts,
            left + 18,
            top + 275,
            (
                f'empirical support: {int(float(gate["empirical_supported_row_count"]))}/'
                f'{int(float(gate["row_count"]))}; '
                f'{("support gate failed" if temperature == 0.45 else "nonstationary canary")}'
            ),
            size=11,
            fill="#46515c",
        )
    low = next(row for row in gates if float(row["temperature"]) == 0.45)
    _svg_text(
        parts,
        60,
        435,
        f'fixed-length null: {int(float(low["fixed_length_supported_row_count"]))}/{int(float(low["row_count"]))} supported at T=0.45',
        size=13,
        weight=700,
    )
    _svg_text(
        parts,
        60,
        462,
        "Boundary: held-out MSD and NGP are diagnostic inputs; no blind, unique-potential, spatial, microscopic, or thermodynamic claim.",
        size=12,
        fill="#46515c",
    )
    parts.append("</svg>\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low-geometry", type=Path, required=True)
    parser.add_argument("--high-geometry", type=Path, required=True)
    parser.add_argument("--gamma-rows", type=Path, required=True)
    parser.add_argument("--low-stationarity", type=Path, required=True)
    parser.add_argument("--high-stationarity", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args(argv)

    provenance = manifests_from_provenance(read_rows(args.provenance))
    rows = compute_geometry_rows(
        read_rows(args.gamma_rows),
        {
            0.45: read_rows(args.low_geometry),
            0.58: read_rows(args.high_geometry),
        },
    )
    stationarity = {
        0.45: _validated_stationarity_pass(
            read_rows(args.low_stationarity),
            temperature=0.45,
        ),
        0.58: _validated_stationarity_pass(
            read_rows(args.high_stationarity),
            temperature=0.58,
        ),
    }
    gates = classify_geometry_gate(
        rows,
        stationarity_pass=stationarity,
        provenance_pass={
            temperature: bool(
                float(provenance[temperature]["replicate_provenance_validation_pass"])
            )
            for temperature in provenance
        },
    )
    write_rows(args.output_rows, rows)
    write_rows(args.output_gate, gates)
    write_geometry_svg(args.output_svg, gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
