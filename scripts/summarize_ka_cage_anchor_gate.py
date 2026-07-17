#!/usr/bin/env python3
"""Combine cage-return and recoil-null evidence into the frozen mechanism gate."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence


RADIUS_SCALES = (0.5, 1.0, 1.5)
CLAIM_KEYS = (
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)
RECOIL_VERDICT_KEYS = (
    "temperature",
    "calibration_time",
    "block_size",
    "radial_bin_count",
    "independent_replicate_count",
    "required_replicate_count",
    "required_realizations_per_replicate",
    "surrogate_realizations_per_replicate",
    "surrogate_base_seed",
    "maximum_ensemble_msd_relative_error",
    "maximum_ensemble_ngp_absolute_error",
    "maximum_ensemble_fs_absolute_error",
    "maximum_ensemble_msd_mc_relative_se",
    "maximum_ensemble_ngp_mc_se",
    "maximum_ensemble_fs_mc_se",
    "heldout_events_used_in_calibration",
)
QUALITY_LIMITS = (
    ("radial_mean_relative_error", 0.02),
    ("radial_standard_deviation_relative_error", 0.02),
    ("lag_one_cosine_mean_absolute_error", 0.02),
    ("lag_one_cosine_quantile_maximum_absolute_error", 0.03),
    ("normalized_lag_one_dot_correlation_absolute_error", 0.02),
)


def _claim_flags() -> dict[str, float]:
    return {key: 0.0 for key in CLAIM_KEYS}


def _finite(row: dict[str, object], key: str) -> float:
    if key not in row:
        raise ValueError(f"missing required field: {key}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"field {key} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"field {key} must be finite")
    return value


def _integer(row: dict[str, object], key: str) -> int:
    value = _finite(row, key)
    if value != int(value):
        raise ValueError(f"field {key} must be an integer")
    return int(value)


def _nonnegative(row: dict[str, object], key: str) -> float:
    value = _finite(row, key)
    if value < 0.0:
        raise ValueError(f"field {key} must be nonnegative")
    return value


def _recoil_seed(base_seed: int, *, replicate: int, realization: int) -> int:
    return int((base_seed + 1_000_003 * replicate + 97_409 * realization) % (2**63 - 1))


def _validate_return_rows(
    rows: Sequence[dict[str, object]],
    *,
    temperature: float,
    calibration_time: int,
    replicate_count: int,
) -> None:
    if not rows:
        raise ValueError("return rows must not be empty")
    keys = set()
    replicates = set()
    for row in rows:
        replicate = _integer(row, "replicate")
        scale = _finite(row, "radius_scale")
        if _finite(row, "temperature") != temperature:
            raise ValueError("return rows have the wrong temperature")
        if _finite(row, "calibration_time") != float(calibration_time):
            raise ValueError("return rows have the wrong calibration time")
        if _finite(row, "fluctuation_half_window") != 5.0:
            raise ValueError("return rows must use fluctuation half-window 5")
        if _finite(row, "calibration_events_only") != 1.0:
            raise ValueError("return rows must contain calibration events only")
        if _finite(row, "heldout_events_used_in_calibration") != 0.0:
            raise ValueError("return rows must not use held-out events in calibration")
        if scale not in RADIUS_SCALES:
            raise ValueError("return rows must use exactly the frozen radius scales")
        return_fraction = _finite(row, "return_fraction")
        null_fraction = _finite(row, "isotropic_null_fraction")
        if not 0.0 <= return_fraction <= 1.0 or not 0.0 <= null_fraction <= 1.0:
            raise ValueError("return fractions must lie in [0, 1]")
        for key in CLAIM_KEYS:
            if _finite(row, key) != 0.0:
                raise ValueError(f"claim boundary must remain zero: {key}")
        keys.add((replicate, scale))
        replicates.add(replicate)
    expected_replicates = set(range(1, replicate_count + 1))
    expected_keys = {
        (replicate, scale)
        for replicate in expected_replicates
        for scale in RADIUS_SCALES
    }
    if replicates != expected_replicates or keys != expected_keys or len(rows) != len(keys):
        raise ValueError("return table must contain one row per replicate and radius scale")


def _validate_recoil_evidence(
    quality_rows: Sequence[dict[str, object]],
    verdict: dict[str, object],
    *,
    temperature: float,
    calibration_time: int,
    replicate_count: int,
) -> dict[str, float]:
    if not quality_rows:
        raise ValueError("recoil quality rows must not be empty")
    for key in RECOIL_VERDICT_KEYS:
        _finite(verdict, key)
    if _finite(verdict, "temperature") != temperature:
        raise ValueError("recoil verdict has the wrong temperature")
    if _finite(verdict, "calibration_time") != float(calibration_time):
        raise ValueError("recoil verdict has the wrong calibration time")
    if _finite(verdict, "block_size") != 20.0:
        raise ValueError("the frozen recoil gate requires block size 20")
    if _finite(verdict, "radial_bin_count") != 8.0:
        raise ValueError("the frozen recoil gate requires 8 radial bins")
    if (
        _finite(verdict, "required_realizations_per_replicate") != 16.0
        or _finite(verdict, "surrogate_realizations_per_replicate") != 16.0
    ):
        raise ValueError("the frozen recoil gate requires 16 realizations")
    if (
        _finite(verdict, "independent_replicate_count") != float(replicate_count)
        or _finite(verdict, "required_replicate_count") != float(replicate_count)
    ):
        raise ValueError("recoil verdict has the wrong replicate count")
    if _finite(verdict, "heldout_events_used_in_calibration") != 0.0:
        raise ValueError("recoil verdict must not use held-out events in calibration")
    for key in CLAIM_KEYS:
        if _finite(verdict, key) != 0.0:
            raise ValueError(f"claim boundary must remain zero: {key}")

    base_seed = _integer(verdict, "surrogate_base_seed")
    if base_seed < 0:
        raise ValueError("surrogate base seed must be nonnegative")
    pairs = set()
    maxima = {key: 0.0 for key, _ in QUALITY_LIMITS}
    for row in quality_rows:
        replicate = _integer(row, "replicate")
        realization = _integer(row, "realization")
        pair = (replicate, realization)
        pairs.add(pair)
        if _finite(row, "temperature") != temperature:
            raise ValueError("quality rows have the wrong temperature")
        if _finite(row, "calibration_time") != float(calibration_time):
            raise ValueError("quality rows have the wrong calibration time")
        if _finite(row, "block_size") != 20.0:
            raise ValueError("quality rows must use block size 20")
        if _finite(row, "radial_bin_count") != 8.0:
            raise ValueError("quality rows must use 8 radial bins")
        if _finite(row, "surrogate_realizations_per_replicate") != 16.0:
            raise ValueError("quality rows must declare 16 realizations")
        if _integer(row, "surrogate_base_seed") != base_seed:
            raise ValueError("quality and verdict base seeds must agree")
        expected_seed = _recoil_seed(
            base_seed,
            replicate=replicate,
            realization=realization,
        )
        if _integer(row, "surrogate_seed") != expected_seed:
            raise ValueError("quality row surrogate seed is inconsistent")
        if _finite(row, "calibration_path_used_in_kernel") != 1.0:
            raise ValueError("quality rows must use the calibration path in the kernel")
        if _finite(row, "heldout_events_used_in_calibration") != 0.0:
            raise ValueError("quality rows must not use held-out events in calibration")
        for key in CLAIM_KEYS:
            if _finite(row, key) != 0.0:
                raise ValueError(f"claim boundary must remain zero: {key}")
        for key, _ in QUALITY_LIMITS:
            maxima[key] = max(maxima[key], _nonnegative(row, key))
    expected_pairs = {
        (replicate, realization)
        for replicate in range(1, replicate_count + 1)
        for realization in range(16)
    }
    if pairs != expected_pairs or len(quality_rows) != len(expected_pairs):
        raise ValueError("quality rows must form a complete replicate-realization grid")
    return maxima


def _ordered_path_upper_bound(row: dict[str, object]) -> bool:
    if row.get("model") != "contiguous_empirical_path":
        raise ValueError("ordered-path row must be the contiguous empirical path")
    required = {
        "temperature": 0.45,
        "independent_replicate_count": 3.0,
        "heldout_path_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "calibration_path_distribution_used": 1.0,
    }
    for key, expected in required.items():
        if _finite(row, key) != expected:
            raise ValueError(f"ordered-path provenance mismatch: {key}")
    for key in CLAIM_KEYS:
        if _finite(row, key) != 0.0:
            raise ValueError(f"claim boundary must remain zero: {key}")
    return (
        _nonnegative(row, "maximum_ensemble_msd_relative_error") <= 0.10
        and _nonnegative(row, "maximum_ensemble_ngp_absolute_error") <= 0.30
        and _nonnegative(row, "maximum_ensemble_fs_absolute_error") <= 0.03
    )


def classify_cage_anchor_gate(
    low_returns: Sequence[dict[str, object]],
    high_returns: Sequence[dict[str, object]],
    low_quality: Sequence[dict[str, object]],
    high_quality: Sequence[dict[str, object]],
    low_recoil: dict[str, object],
    high_recoil: dict[str, object],
    ordered_path: dict[str, object],
) -> dict[str, object]:
    """Select cage-anchor memory only when every frozen condition passes."""

    _validate_return_rows(
        low_returns,
        temperature=0.45,
        calibration_time=5000,
        replicate_count=3,
    )
    _validate_return_rows(
        high_returns,
        temperature=0.58,
        calibration_time=750,
        replicate_count=5,
    )
    low_quality_maxima = _validate_recoil_evidence(
        low_quality,
        low_recoil,
        temperature=0.45,
        calibration_time=5000,
        replicate_count=3,
    )
    high_quality_maxima = _validate_recoil_evidence(
        high_quality,
        high_recoil,
        temperature=0.58,
        calibration_time=750,
        replicate_count=5,
    )
    ordered_path_pass = _ordered_path_upper_bound(ordered_path)

    return_values: dict[str, float] = {}
    separated = True
    for scale in RADIUS_SCALES:
        suffix = f"s{scale:g}".replace(".", "p")
        low_minimum = min(
            _finite(row, "return_fraction")
            for row in low_returns
            if _finite(row, "radius_scale") == scale
        )
        high_maximum = max(
            _finite(row, "return_fraction")
            for row in high_returns
            if _finite(row, "radius_scale") == scale
        )
        scale_pass = low_minimum > high_maximum
        return_values[f"minimum_low_return_fraction_{suffix}"] = low_minimum
        return_values[f"maximum_high_return_fraction_{suffix}"] = high_maximum
        return_values[f"radius_scale_separated_{suffix}"] = float(scale_pass)
        separated = separated and scale_pass
    primary_excess = min(
        _finite(row, "return_fraction") / _finite(row, "isotropic_null_fraction")
        if _finite(row, "isotropic_null_fraction") > 0.0
        else 0.0
        for row in low_returns
        if _finite(row, "radius_scale") == 1.0
    )
    primary_pass = primary_excess >= 1.35
    return_signal = separated and primary_pass

    low_quality_pass = all(
        low_quality_maxima[key] <= limit for key, limit in QUALITY_LIMITS
    )
    high_quality_pass = all(
        high_quality_maxima[key] <= limit for key, limit in QUALITY_LIMITS
    )
    low_precision = (
        _nonnegative(low_recoil, "maximum_ensemble_msd_mc_relative_se") <= 0.01
        and _nonnegative(low_recoil, "maximum_ensemble_ngp_mc_se") <= 0.03
        and _nonnegative(low_recoil, "maximum_ensemble_fs_mc_se") <= 0.003
    )
    high_precision = (
        _nonnegative(high_recoil, "maximum_ensemble_msd_mc_relative_se") <= 0.01
        and _nonnegative(high_recoil, "maximum_ensemble_ngp_mc_se") <= 0.03
        and _nonnegative(high_recoil, "maximum_ensemble_fs_mc_se") <= 0.003
    )
    both_recoil_valid = (
        low_quality_pass and high_quality_pass and low_precision and high_precision
    )
    high_curve_closed = (
        _nonnegative(high_recoil, "maximum_ensemble_msd_relative_error") <= 0.10
        and _nonnegative(high_recoil, "maximum_ensemble_ngp_absolute_error") <= 0.30
        and _nonnegative(high_recoil, "maximum_ensemble_fs_absolute_error") <= 0.03
    )
    low_curve_closed = (
        _nonnegative(low_recoil, "maximum_ensemble_msd_relative_error") <= 0.10
        and _nonnegative(low_recoil, "maximum_ensemble_ngp_absolute_error") <= 0.30
        and _nonnegative(low_recoil, "maximum_ensemble_fs_absolute_error") <= 0.03
    )
    low_ngp_failure = _nonnegative(low_recoil, "maximum_ensemble_ngp_absolute_error") > 0.30
    low_fs_failure = _nonnegative(low_recoil, "maximum_ensemble_fs_absolute_error") > 0.03
    selected = (
        return_signal
        and both_recoil_valid
        and high_curve_closed
        and not low_curve_closed
        and low_ngp_failure
        and low_fs_failure
        and ordered_path_pass
    )

    if selected:
        mechanism_state = "cage_anchor_memory_required"
    elif not return_signal:
        mechanism_state = "return_signal_rejected"
    elif not both_recoil_valid:
        mechanism_state = "unresolved_recoil_quality_or_precision"
    elif not high_curve_closed:
        mechanism_state = "high_temperature_curve_open"
    elif low_curve_closed:
        mechanism_state = "low_temperature_curve_closed"
    elif not (low_ngp_failure and low_fs_failure):
        mechanism_state = "low_temperature_dual_failure_absent"
    else:
        mechanism_state = "ordered_path_upper_bound_absent"

    return {
        "low_temperature": 0.45,
        "high_temperature": 0.58,
        "low_calibration_time": 5000.0,
        "high_calibration_time": 750.0,
        "block_size": 20.0,
        "radial_bin_count": 8.0,
        "recoil_realizations_per_replicate": 16.0,
        "primary_radius_scale": 1.0,
        "primary_null_excess_ratio_tolerance": 1.35,
        "msd_relative_error_tolerance": 0.10,
        "ngp_absolute_error_tolerance": 0.30,
        "fs_absolute_error_tolerance": 0.03,
        "msd_mc_relative_se_tolerance": 0.01,
        "ngp_mc_se_tolerance": 0.03,
        "fs_mc_se_tolerance": 0.003,
        **return_values,
        "all_radius_scales_separated": float(separated),
        "minimum_primary_low_return_excess_ratio": primary_excess,
        "primary_radius_null_excess_pass": float(primary_pass),
        "cage_anchor_return_signal_ready": float(return_signal),
        "low_temperature_recoil_quality_precision_pass": float(low_quality_pass and low_precision),
        "high_temperature_recoil_quality_precision_pass": float(high_quality_pass and high_precision),
        "high_temperature_curve_transfer_pass": float(high_curve_closed),
        "low_temperature_curve_transfer_pass": float(low_curve_closed),
        "low_temperature_ngp_failure": float(low_ngp_failure),
        "low_temperature_fs_failure": float(low_fs_failure),
        "ordered_calibration_path_upper_bound_pass": float(ordered_path_pass),
        "low_maximum_ensemble_msd_relative_error": _finite(low_recoil, "maximum_ensemble_msd_relative_error"),
        "low_maximum_ensemble_ngp_absolute_error": _finite(low_recoil, "maximum_ensemble_ngp_absolute_error"),
        "low_maximum_ensemble_fs_absolute_error": _finite(low_recoil, "maximum_ensemble_fs_absolute_error"),
        "high_maximum_ensemble_msd_relative_error": _finite(high_recoil, "maximum_ensemble_msd_relative_error"),
        "high_maximum_ensemble_ngp_absolute_error": _finite(high_recoil, "maximum_ensemble_ngp_absolute_error"),
        "high_maximum_ensemble_fs_absolute_error": _finite(high_recoil, "maximum_ensemble_fs_absolute_error"),
        **{
            f"low_maximum_{key}": value
            for key, value in low_quality_maxima.items()
        },
        **{
            f"high_maximum_{key}": value
            for key, value in high_quality_maxima.items()
        },
        "cage_anchor_memory_required": float(selected),
        "mechanism_state": mechanism_state,
        "next_minimal_model_candidate": "anchor_aware_reversible_cage_semi_markov" if selected else "none_selected",
        **_claim_flags(),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def _read_one(path: Path) -> dict[str, str]:
    rows = _read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"input table must contain exactly one row: {path}")
    return rows[0]


def _read_ordered_path(path: Path) -> dict[str, str]:
    rows = _read_rows(path)
    selected = [row for row in rows if row.get("model") == "contiguous_empirical_path"]
    if len(selected) != 1:
        raise ValueError("ordered-path verdict must contain one contiguous empirical path row")
    return selected[0]


def write_gate_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def write_gate_svg(
    path: Path,
    low_returns: Sequence[dict[str, object]],
    high_returns: Sequence[dict[str, object]],
    gate: dict[str, object],
) -> None:
    width, height = 960, 560
    left, top, plot_width, plot_height = 90, 115, 460, 330
    low_color, high_color = "#b33a3a", "#176b87"
    x_for = {scale: left + index * plot_width / 2 for index, scale in enumerate(RADIUS_SCALES)}
    y_for = lambda value: top + plot_height * (1.0 - float(value))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="960" height="560" fill="#f7f7f4"/>',
        '<text x="48" y="48" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#172126">Cooling-induced cage-anchor memory gate</text>',
        '<text x="48" y="76" font-family="Arial, sans-serif" font-size="14" fill="#4d585d">Calibration returns and held-out one-step recoil transfer</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#ffffff" stroke="#a9b0b3"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e5e5"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#596268">{tick:g}</text>')
    for scale in RADIUS_SCALES:
        x = x_for[scale]
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#30383c">{scale:g}</text>')
    for rows, color, offset in ((low_returns, low_color, -6), (high_returns, high_color, 6)):
        for row in rows:
            x = x_for[_finite(row, "radius_scale")] + offset
            y = y_for(_finite(row, "return_fraction"))
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}" fill-opacity="0.78"/>')
    parts.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{top + plot_height + 55}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#30383c">Return radius / a_DW</text>',
            f'<text x="25" y="{top + plot_height / 2}" transform="rotate(-90 25 {top + plot_height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#30383c">Return fraction</text>',
            f'<circle cx="112" cy="510" r="5" fill="{low_color}"/><text x="125" y="515" font-family="Arial, sans-serif" font-size="13" fill="#30383c">T=0.45</text>',
            f'<circle cx="202" cy="510" r="5" fill="{high_color}"/><text x="215" y="515" font-family="Arial, sans-serif" font-size="13" fill="#30383c">T=0.58</text>',
            '<text x="610" y="126" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#172126">Frozen decision conditions</text>',
        ]
    )
    conditions = (
        ("All return radii separate", gate["all_radius_scales_separated"]),
        ("Primary return/null ratio >= 1.35", gate["primary_radius_null_excess_pass"]),
        ("Recoil quality and precision: T=0.45", gate["low_temperature_recoil_quality_precision_pass"]),
        ("Recoil quality and precision: T=0.58", gate["high_temperature_recoil_quality_precision_pass"]),
        ("T=0.58 closes MSD, NGP, and Fs", gate["high_temperature_curve_transfer_pass"]),
        ("T=0.45 fails NGP and multi-k Fs", float(gate["low_temperature_ngp_failure"]) * float(gate["low_temperature_fs_failure"])),
        ("Ordered path remains upper bound", gate["ordered_calibration_path_upper_bound_pass"]),
    )
    for index, (label, passed) in enumerate(conditions):
        y = 165 + 38 * index
        color = "#2f7d4a" if float(passed) == 1.0 else "#b33a3a"
        symbol = "PASS" if float(passed) == 1.0 else "FAIL"
        parts.append(f'<rect x="610" y="{y - 15}" width="52" height="22" rx="3" fill="{color}"/>')
        parts.append(f'<text x="636" y="{y + 1}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="700" fill="#ffffff">{symbol}</text>')
        parts.append(f'<text x="676" y="{y + 1}" font-family="Arial, sans-serif" font-size="13" fill="#30383c">{label}</text>')
    selected = float(gate["cage_anchor_memory_required"]) == 1.0
    result_color = "#2f7d4a" if selected else "#b33a3a"
    result = "CAGE-ANCHOR MEMORY REQUIRED" if selected else "CAGE-ANCHOR MEMORY NOT SELECTED"
    parts.extend(
        [
            f'<rect x="610" y="445" width="302" height="54" rx="4" fill="{result_color}"/>',
            f'<text x="761" y="478" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#ffffff">{result}</text>',
            '<text x="610" y="528" font-family="Arial, sans-serif" font-size="11" fill="#596268">Dynamical mechanism only; closure, facilitation,</text>',
            '<text x="610" y="543" font-family="Arial, sans-serif" font-size="11" fill="#596268">and thermodynamic claims remain 0.</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("return_rows", type=Path)
    parser.add_argument("low_recoil_quality", type=Path)
    parser.add_argument("high_recoil_quality", type=Path)
    parser.add_argument("low_recoil_verdict", type=Path)
    parser.add_argument("high_recoil_verdict", type=Path)
    parser.add_argument("low_ordered_path_verdict", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = _read_rows(args.return_rows)
    low_returns = [row for row in rows if _finite(row, "temperature") == 0.45]
    high_returns = [row for row in rows if _finite(row, "temperature") == 0.58]
    if len(low_returns) + len(high_returns) != len(rows):
        raise ValueError("return table contains an unsupported temperature")
    low_quality = _read_rows(args.low_recoil_quality)
    high_quality = _read_rows(args.high_recoil_quality)
    low_recoil = _read_one(args.low_recoil_verdict)
    high_recoil = _read_one(args.high_recoil_verdict)
    ordered_path = _read_ordered_path(args.low_ordered_path_verdict)
    gate = classify_cage_anchor_gate(
        low_returns,
        high_returns,
        low_quality,
        high_quality,
        low_recoil,
        high_recoil,
        ordered_path,
    )
    write_gate_csv(args.output_csv, gate)
    write_gate_svg(args.output_svg, low_returns, high_returns, gate)


if __name__ == "__main__":
    main()
