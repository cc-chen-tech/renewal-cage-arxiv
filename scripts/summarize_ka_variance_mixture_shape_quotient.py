#!/usr/bin/env python3
"""Test fixed-MSD variance-mixture closure of multi-k shape residuals."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections.abc import Sequence
from pathlib import Path


OBSERVABLE_WAVE_NUMBERS = {
    "fs_k2": 2.0,
    "fs_k4": 4.0,
    "fs_k7p25": 7.25,
}
EXPECTED_OBSERVABLES = frozenset({"ngp", *OBSERVABLE_WAVE_NUMBERS})
SOURCE_CLOSED_CLAIMS = (
    "blind_prediction_claim_allowed",
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)
OUTPUT_CLOSED_CLAIMS = (
    "blind_prediction_claim_allowed",
    "static_environment_resolved",
    "finite_exchange_resolved",
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)
FS_TOLERANCE = 0.03
NGP_TOLERANCE = 0.30
CSV_FLOAT_SIGNIFICANT_DIGITS = 8
SOURCE_GATE_CLOSED_CLAIMS = (
    "blind_prediction_claim_allowed",
    "clock_only_closure_allowed",
    "cooling_enhanced_shape_memory_claim_allowed",
    "finite_exchange_resolved",
    "static_environment_resolved",
    "spatial_facilitation_resolved",
    "microdynamic_closure_claim_allowed",
    "thermodynamic_claim_allowed",
)


def _finite(row: dict[str, object], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid field: {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"field must be finite: {field}")
    return value


def _exact_int(row: dict[str, object], field: str) -> int:
    if isinstance(row.get(field), bool):
        raise ValueError(f"field must be an exact integer: {field}")
    value = _finite(row, field)
    if not value.is_integer():
        raise ValueError(f"field must be an exact integer: {field}")
    return int(value)


def mixture_log_scattering(alpha: float, x: float, family: str) -> float:
    """Return log E[exp(-x S)] for a unit-mean variance multiplier."""

    try:
        local_alpha = float(alpha)
        local_x = float(x)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha and x must be finite nonnegative scalars") from error
    if (
        not math.isfinite(local_alpha)
        or not math.isfinite(local_x)
        or local_alpha < 0.0
        or local_x < 0.0
        or family not in {"gamma", "inverse_gaussian"}
    ):
        raise ValueError("family and finite nonnegative alpha/x are required")
    if local_alpha == 0.0:
        return -local_x
    if family == "gamma":
        return -math.log1p(local_alpha * local_x) / local_alpha
    return (1.0 - math.sqrt(1.0 + 2.0 * local_alpha * local_x)) / local_alpha


def _validate_source_row(row: dict[str, object]) -> tuple[float, int, int, str, int]:
    temperature = _finite(row, "temperature")
    replicate = _exact_int(row, "replicate")
    lag = _exact_int(row, "lag")
    observable = str(row.get("observable", ""))
    support = _exact_int(row, "in_calibration_msd_support")
    if (
        temperature not in {0.45, 0.58}
        or replicate < 1
        or lag < 1
        or observable not in EXPECTED_OBSERVABLES
        or support not in {0, 1}
        or _finite(row, "heldout_msd_used_as_diagnostic_input") != 1.0
        or any(_finite(row, field) != 0.0 for field in SOURCE_CLOSED_CLAIMS)
    ):
        raise ValueError("source quotient row violates the frozen claim boundary")
    expected_tolerance = NGP_TOLERANCE if observable == "ngp" else FS_TOLERANCE
    if _finite(row, "tolerance") != expected_tolerance:
        raise ValueError("source quotient tolerance is not frozen")
    return temperature, replicate, lag, observable, support


def compute_shape_quotient_rows(
    quotient_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Score supported fixed-MSD cells with truncated and resummed quotients."""

    if not quotient_rows:
        raise ValueError("transport-clock quotient rows must be nonempty")
    cells: dict[tuple[float, int, int], dict[str, dict[str, object]]] = {}
    supports: dict[tuple[float, int, int], set[int]] = {}
    for row in quotient_rows:
        temperature, replicate, lag, observable, support = _validate_source_row(row)
        key = (temperature, replicate, lag)
        local = cells.setdefault(key, {})
        if observable in local:
            raise ValueError("source quotient contains a duplicate observable cell")
        local[observable] = row
        supports.setdefault(key, set()).add(support)
    if any(set(cell) != EXPECTED_OBSERVABLES for cell in cells.values()):
        raise ValueError("every source quotient cell must contain the exact observable set")
    if any(len(values) != 1 for values in supports.values()):
        raise ValueError("source quotient support must be cell-consistent")

    result: list[dict[str, object]] = []
    for (temperature, replicate, lag), cell in sorted(cells.items()):
        if next(iter(supports[(temperature, replicate, lag)])) == 0:
            continue
        ngp = cell["ngp"]
        heldout_msd = _finite(ngp, "heldout_msd")
        calibration_msd = _finite(ngp, "calibration_msd")
        heldout_alpha = _finite(ngp, "heldout_value")
        calibration_alpha = _finite(ngp, "msd_matched_value")
        if (
            heldout_msd <= 0.0
            or calibration_msd <= 0.0
            or heldout_alpha < 0.0
            or calibration_alpha < 0.0
        ):
            raise ValueError("supported MSD and NGP inputs must be physically admissible")
        for observable, wave_number in OBSERVABLE_WAVE_NUMBERS.items():
            row = cell[observable]
            if (
                _finite(row, "heldout_msd") != heldout_msd
                or _finite(row, "calibration_msd") != calibration_msd
            ):
                raise ValueError("supported observables must share the cell MSD inputs")
            heldout_fs = _finite(row, "heldout_value")
            baseline_fs = _finite(row, "msd_matched_value")
            x = wave_number**2 * heldout_msd / 6.0
            fourth_log_correction = (heldout_alpha - calibration_alpha) * x**2 / 2.0
            gamma_log_correction = mixture_log_scattering(
                heldout_alpha,
                x,
                "gamma",
            ) - mixture_log_scattering(calibration_alpha, x, "gamma")
            inverse_gaussian_log_correction = mixture_log_scattering(
                heldout_alpha,
                x,
                "inverse_gaussian",
            ) - mixture_log_scattering(
                calibration_alpha,
                x,
                "inverse_gaussian",
            )
            fourth_fs = baseline_fs * math.exp(fourth_log_correction)
            gamma_fs = baseline_fs * math.exp(gamma_log_correction)
            inverse_gaussian_fs = baseline_fs * math.exp(
                inverse_gaussian_log_correction
            )
            values = (
                x,
                fourth_log_correction,
                gamma_log_correction,
                inverse_gaussian_log_correction,
                fourth_fs,
                gamma_fs,
                inverse_gaussian_fs,
            )
            if any(not math.isfinite(value) for value in values):
                raise ValueError("shape quotient produced a nonfinite value")
            baseline_error = abs(baseline_fs - heldout_fs)
            fourth_error = abs(fourth_fs - heldout_fs)
            gamma_error = abs(gamma_fs - heldout_fs)
            inverse_gaussian_error = abs(inverse_gaussian_fs - heldout_fs)
            output: dict[str, object] = {
                "temperature": temperature,
                "replicate": float(replicate),
                "lag": float(lag),
                "observable": observable,
                "wave_number": wave_number,
                "heldout_msd": heldout_msd,
                "expansion_coordinate": x,
                "calibration_ngp": calibration_alpha,
                "heldout_ngp": heldout_alpha,
                "ngp_residual": heldout_alpha - calibration_alpha,
                "baseline_fs": baseline_fs,
                "heldout_fs": heldout_fs,
                "fourth_log_correction": fourth_log_correction,
                "gamma_log_correction": gamma_log_correction,
                "inverse_gaussian_log_correction": inverse_gaussian_log_correction,
                "fourth_fs": fourth_fs,
                "gamma_fs": gamma_fs,
                "inverse_gaussian_fs": inverse_gaussian_fs,
                "baseline_absolute_error": baseline_error,
                "fourth_absolute_error": fourth_error,
                "gamma_absolute_error": gamma_error,
                "inverse_gaussian_absolute_error": inverse_gaussian_error,
                "baseline_normalized_error": baseline_error / FS_TOLERANCE,
                "fourth_normalized_error": fourth_error / FS_TOLERANCE,
                "gamma_normalized_error": gamma_error / FS_TOLERANCE,
                "inverse_gaussian_normalized_error": (
                    inverse_gaussian_error / FS_TOLERANCE
                ),
                "resummation_family_spread": abs(
                    gamma_fs - inverse_gaussian_fs
                ),
                "resummation_family_spread_normalized": abs(
                    gamma_fs - inverse_gaussian_fs
                )
                / FS_TOLERANCE,
                "fs_error_tolerance": FS_TOLERANCE,
                "in_calibration_msd_support": 1.0,
                "heldout_msd_used_as_diagnostic_input": 1.0,
                "heldout_ngp_used_as_diagnostic_input": 1.0,
                "macro_fit_parameter_count": 0.0,
                **{field: 0.0 for field in OUTPUT_CLOSED_CLAIMS},
            }
            result.append(output)
    if not result:
        raise ValueError("source quotient has no supported shape rows")
    return result


def _validated_source_gate(
    source_gate_rows: Sequence[dict[str, object]],
) -> dict[float, dict[str, object]]:
    if len(source_gate_rows) != 2:
        raise ValueError("source transport-clock gate must contain two temperatures")
    result: dict[float, dict[str, object]] = {}
    for row in source_gate_rows:
        temperature = _finite(row, "temperature")
        if temperature not in {0.45, 0.58} or temperature in result:
            raise ValueError("source transport-clock gate temperature grid is invalid")
        if (
            any(_finite(row, field) != 0.0 for field in SOURCE_GATE_CLOSED_CLAIMS)
            or _finite(row, "replicate_provenance_validation_pass") != 1.0
            or _finite(row, "parent_sample_count") != 1.0
            or _finite(row, "independent_replicate_count") != 0.0
            or _finite(row, "independently_prepared_parent_samples") != 0.0
            or str(row.get("independence_class"))
            != "decorrelated_parent_frames_plus_velocity_seeds"
            or _exact_int(row, "replicate_count") < 1
        ):
            raise ValueError("source transport-clock provenance or claims are invalid")
        stationarity = _finite(
            row,
            "source_ensemble_stationarity_all_comparisons_pass",
        )
        separation = _finite(row, "clock_shape_separation_supported_exploratory")
        canary = _finite(row, "high_temperature_canary_only")
        high_control = _finite(row, "high_temperature_control_resolved")
        if (
            stationarity not in {0.0, 1.0}
            or separation not in {0.0, 1.0}
            or canary not in {0.0, 1.0}
            or high_control != 0.0
            or (
                temperature == 0.45
                and (stationarity, separation, canary) != (1.0, 1.0, 0.0)
            )
            or (
                temperature == 0.58
                and (stationarity, separation, canary) != (0.0, 0.0, 1.0)
            )
        ):
            raise ValueError("source transport-clock gate state is not frozen")
        result[temperature] = row
    if set(result) != {0.45, 0.58}:
        raise ValueError("source transport-clock gate is incomplete")
    return result


def _validate_scored_rows(
    rows: Sequence[dict[str, object]],
    source_gate: dict[float, dict[str, object]],
) -> None:
    if not rows:
        raise ValueError("scored shape quotient rows must be nonempty")
    expected_keys: set[tuple[float, int, int, float]] = set()
    for row in rows:
        temperature = _finite(row, "temperature")
        replicate = _exact_int(row, "replicate")
        lag = _exact_int(row, "lag")
        wave_number = _finite(row, "wave_number")
        key = (temperature, replicate, lag, wave_number)
        if (
            temperature not in source_gate
            or wave_number not in set(OBSERVABLE_WAVE_NUMBERS.values())
            or key in expected_keys
            or _finite(row, "in_calibration_msd_support") != 1.0
            or _finite(row, "heldout_msd_used_as_diagnostic_input") != 1.0
            or _finite(row, "heldout_ngp_used_as_diagnostic_input") != 1.0
            or _finite(row, "macro_fit_parameter_count") != 0.0
            or any(_finite(row, field) != 0.0 for field in OUTPUT_CLOSED_CLAIMS)
        ):
            raise ValueError("scored shape row violates support or claim boundaries")
        expected_keys.add(key)
        tolerance = _finite(row, "fs_error_tolerance")
        heldout = _finite(row, "heldout_fs")
        predictions = {
            "baseline": _finite(row, "baseline_fs"),
            "fourth": _finite(row, "fourth_fs"),
            "gamma": _finite(row, "gamma_fs"),
            "inverse_gaussian": _finite(row, "inverse_gaussian_fs"),
        }
        if tolerance != FS_TOLERANCE:
            raise ValueError("scored shape row tolerance is not frozen")
        for method, prediction in predictions.items():
            absolute = _finite(row, f"{method}_absolute_error")
            normalized = _finite(row, f"{method}_normalized_error")
            if not math.isclose(absolute, abs(prediction - heldout), abs_tol=1e-14):
                raise ValueError("scored shape absolute error is inconsistent")
            if not math.isclose(normalized, absolute / tolerance, abs_tol=1e-13):
                raise ValueError("scored shape normalized error is inconsistent")
    for temperature, source in source_gate.items():
        local = [key for key in expected_keys if key[0] == temperature]
        replicates = {key[1] for key in local}
        if replicates != set(range(1, _exact_int(source, "replicate_count") + 1)):
            raise ValueError("scored rows do not cover the frozen replicate identities")
        cell_waves: dict[tuple[int, int], set[float]] = {}
        for _, replicate, lag, wave_number in local:
            cell_waves.setdefault((replicate, lag), set()).add(wave_number)
        if not cell_waves or any(
            waves != set(OBSERVABLE_WAVE_NUMBERS.values())
            for waves in cell_waves.values()
        ):
            raise ValueError("scored rows do not contain complete multi-k cells")


def classify_shape_quotient_gate(
    rows: Sequence[dict[str, object]],
    source_gate_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate the frozen resummation gate without selecting a family."""

    source_gate = _validated_source_gate(source_gate_rows)
    _validate_scored_rows(rows, source_gate)
    result: list[dict[str, object]] = []
    wave_labels = {2.0: "k2", 4.0: "k4", 7.25: "k7p25"}
    methods = ("baseline", "fourth", "gamma", "inverse_gaussian")
    for temperature in (0.45, 0.58):
        local = [row for row in rows if _finite(row, "temperature") == temperature]
        maxima = {
            method: max(_finite(row, f"{method}_normalized_error") for row in local)
            for method in methods
        }
        by_wave: dict[str, float] = {}
        for wave_number, label in wave_labels.items():
            selected = [row for row in local if _finite(row, "wave_number") == wave_number]
            for method in methods:
                by_wave[f"{label}_{method}_max_normalized_error"] = max(
                    _finite(row, f"{method}_normalized_error") for row in selected
                )
        gamma_pass = maxima["gamma"] <= 1.0
        inverse_gaussian_pass = maxima["inverse_gaussian"] <= 1.0
        family_robust = gamma_pass and inverse_gaussian_pass
        source = source_gate[temperature]
        stationarity = _finite(
            source,
            "source_ensemble_stationarity_all_comparisons_pass",
        )
        source_separation = _finite(
            source,
            "clock_shape_separation_supported_exploratory",
        )
        low_supported = (
            temperature == 0.45
            and stationarity == 1.0
            and source_separation == 1.0
            and (maxima["baseline"] > 1.0 or maxima["fourth"] > 1.0)
            and family_robust
        )
        gate: dict[str, object] = {
            "temperature": temperature,
            "analysis_status": (
                "variance_mixture_shape_closure_exploratory"
                if low_supported
                else "high_temperature_canary_only"
                if temperature == 0.58
                else "shape_closure_unresolved"
            ),
            "supported_shape_row_count": float(len(local)),
            "supported_cell_count": float(len(local) // 3),
            "source_replicate_count": float(_exact_int(source, "replicate_count")),
            "source_stationarity_pass": stationarity,
            "source_clock_shape_separation_supported_exploratory": source_separation,
            "maximum_baseline_normalized_error": maxima["baseline"],
            "maximum_fourth_normalized_error": maxima["fourth"],
            "maximum_gamma_normalized_error": maxima["gamma"],
            "maximum_inverse_gaussian_normalized_error": maxima["inverse_gaussian"],
            **by_wave,
            "baseline_all_k_pass": float(maxima["baseline"] <= 1.0),
            "fourth_all_k_pass": float(maxima["fourth"] <= 1.0),
            "gamma_all_k_pass": float(gamma_pass),
            "inverse_gaussian_all_k_pass": float(inverse_gaussian_pass),
            "family_robust_resummation_pass": float(family_robust),
            "maximum_resummation_family_spread_normalized": max(
                _finite(row, "resummation_family_spread_normalized") for row in local
            ),
            "marginal_variance_mixture_shape_closure_supported_exploratory": float(
                low_supported
            ),
            "variance_mixture_family_selected": 0.0,
            "transient_marginal_shape_only": 1.0,
            "high_temperature_canary_only": float(temperature == 0.58),
            "high_temperature_control_resolved": 0.0,
            "replicate_provenance_validation_pass": _finite(
                source,
                "replicate_provenance_validation_pass",
            ),
            "parent_sample_count": _finite(source, "parent_sample_count"),
            "independent_replicate_count": _finite(
                source,
                "independent_replicate_count",
            ),
            "independently_prepared_parent_samples": _finite(
                source,
                "independently_prepared_parent_samples",
            ),
            "independence_class": str(source["independence_class"]),
            "heldout_msd_used_as_diagnostic_input": 1.0,
            "heldout_ngp_used_as_diagnostic_input": 1.0,
            "macro_fit_parameter_count": 0.0,
            "confirmatory_independent_parent_replication_required": 1.0,
            "next_required_action": (
                "independent_parents_then_blind_ngp_prediction"
            ),
            **{field: 0.0 for field in OUTPUT_CLOSED_CLAIMS},
        }
        result.append(gate)
    return result


def canonical_csv_value(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize a nonfinite shape quotient value")
        return format(value, f".{CSV_FLOAT_SIGNIFICANT_DIGITS}g")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty shape quotient table")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("shape quotient output rows must have an identical schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: canonical_csv_value(value) for field, value in row.items()}
            for row in rows
        )


def _svg_text(
    parts: list[str],
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#20272b",
) -> None:
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{html.escape(text)}</text>'
    )


def _svg_rotated_text(
    parts: list[str],
    x: float,
    y: float,
    text: str,
    *,
    size: int = 11,
    fill: str = "#596268",
) -> None:
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="middle" fill="{fill}" '
        f'transform="rotate(-90 {x:.1f} {y:.1f})">{html.escape(text)}</text>'
    )


def render_svg(gates: Sequence[dict[str, object]]) -> str:
    if len(gates) != 2 or {_finite(row, "temperature") for row in gates} != {0.45, 0.58}:
        raise ValueError("SVG requires the exact two-temperature gate")
    width, height = 1280, 720
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    _svg_text(parts, 54, 48, "Variance-mixture shape quotient", size=27, weight=700)
    _svg_text(
        parts,
        54,
        75,
        "Fixed-MSD multi-k residual: fourth-order truncation versus nonlinear resummation",
        size=14,
        fill="#596268",
    )
    methods = (
        ("baseline", "clock only", "#4C78A8"),
        ("fourth", "fourth order", "#E45756"),
        ("gamma", "gamma", "#59A14F"),
        ("inverse_gaussian", "inverse Gaussian", "#B279A2"),
    )
    legend_x = 54
    for index, (_, label, color) in enumerate(methods):
        x = legend_x + index * 174
        parts.append(f'<rect x="{x}" y="93" width="14" height="14" fill="{color}"/>')
        _svg_text(parts, x + 21, 105, label, size=12)
    cap = 6.0
    panel_top = 128.0
    panel_height = 432.0
    plot_top = panel_top + 52.0
    plot_height = 310.0
    plot_bottom = plot_top + plot_height
    for panel_index, temperature in enumerate((0.45, 0.58)):
        gate = next(row for row in gates if _finite(row, "temperature") == temperature)
        left = 54.0 + panel_index * 610.0
        panel_width = 572.0
        parts.append(
            f'<rect x="{left}" y="{panel_top}" width="{panel_width}" height="{panel_height}" '
            'rx="4" fill="#fafbfb" stroke="#ccd2d5"/>'
        )
        _svg_text(parts, left + 20, panel_top + 30, f"T={temperature:.2f}", size=18, weight=700)
        status = (
            "family-robust resummation pass"
            if temperature == 0.45
            else "canary only"
        )
        _svg_text(parts, left + 102, panel_top + 29, status, size=12, fill="#596268")
        for tick in range(7):
            y = plot_bottom - tick / cap * plot_height
            parts.append(
                f'<line x1="{left + 70}" y1="{y:.1f}" x2="{left + panel_width - 18}" '
                f'y2="{y:.1f}" stroke="#e1e5e7" stroke-width="1"/>'
            )
            _svg_text(parts, left + 62, y + 4, str(tick), size=10, anchor="end", fill="#687176")
        tolerance_y = plot_bottom - plot_height / cap
        parts.append(
            f'<line x1="{left + 70}" y1="{tolerance_y:.1f}" x2="{left + panel_width - 18}" '
            f'y2="{tolerance_y:.1f}" stroke="#20272b" stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        _svg_text(parts, left + panel_width - 22, tolerance_y - 6, "tolerance", size=10, anchor="end")
        group_centers = (left + 150, left + 298, left + 446)
        waves = (("k2", "k=2"), ("k4", "k=4"), ("k7p25", "k=7.25"))
        for center, (wave_key, wave_label) in zip(group_centers, waves, strict=True):
            for method_index, (method, _, color) in enumerate(methods):
                value = _finite(gate, f"{wave_key}_{method}_max_normalized_error")
                shown = min(value, cap)
                bar_height = shown / cap * plot_height
                x = center - 42 + method_index * 22
                y = plot_bottom - bar_height
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="17" height="{bar_height:.1f}" '
                    f'fill="{color}" opacity="0.92"/>'
                )
                label = f"{value:.2f}" if value < 10.0 else f"{value:.1f}"
                _svg_text(
                    parts,
                    x + 8.5,
                    max(plot_top + 12, y - 5),
                    label,
                    size=9,
                    anchor="middle",
                    fill="#343b3f",
                )
            _svg_text(parts, center, plot_bottom + 25, wave_label, size=12, weight=700, anchor="middle")
        _svg_rotated_text(
            parts,
            left + 22,
            plot_top + plot_height / 2,
            "maximum normalized error (tolerance units)",
        )
    _svg_text(
        parts,
        54,
        600,
        "No macro fit: held-out MSD and NGP are diagnostic inputs.",
        size=13,
        weight=700,
    )
    _svg_text(
        parts,
        54,
        626,
        "Both resummations pass on the same supported rows; marginal variance-mixture closure is exploratory, family unresolved.",
        size=12,
        fill="#596268",
    )
    _svg_text(
        parts,
        54,
        650,
        "T=0.58 canary only; stationarity unresolved.",
        size=12,
        fill="#596268",
    )
    _svg_text(
        parts,
        54,
        674,
        "Blind and mechanism claims remain 0; microscopic, spatial, and thermodynamic claims remain 0.",
        size=12,
        fill="#596268",
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotient-rows", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = compute_shape_quotient_rows(read_rows(args.quotient_rows))
    gates = classify_shape_quotient_gate(rows, read_rows(args.source_gate))
    write_rows(args.output_rows, rows)
    write_rows(args.output_gate, gates)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(render_svg(gates))


if __name__ == "__main__":
    main()
