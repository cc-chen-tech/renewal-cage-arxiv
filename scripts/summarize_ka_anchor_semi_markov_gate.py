#!/usr/bin/env python3
"""Select the frozen two-temperature anchor semi-Markov mechanism state."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_anchor_semi_markov_transfer import (  # noqa: E402
    MODELS,
    classify_anchor_transfer,
    write_rows,
)


CLAIM_FIELDS = (
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)


def _finite(row: dict[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid field: {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"field must be finite: {key}")
    return value


def _claim_limited(rows: Sequence[dict[str, object]]) -> bool:
    return bool(rows) and all(_finite(row, key) == 0.0 for row in rows for key in CLAIM_FIELDS)


def _temperature_row(
    rows: Sequence[dict[str, object]],
    temperature: float,
) -> dict[str, object]:
    selected = [row for row in rows if math.isclose(_finite(row, "temperature"), temperature)]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one verdict row at T={temperature:g}")
    return selected[0]


def _higher_order_scores(
    rows: Sequence[dict[str, object]],
    *,
    expected_replicates: int,
    model: str | None,
) -> dict[int, float]:
    selected = [row for row in rows if model is None or row.get("model") == model]
    replicate_ids = {int(_finite(row, "replicate")) for row in selected}
    if replicate_ids != set(range(1, expected_replicates + 1)):
        raise ValueError("higher-order rows do not contain the exact replicate set")
    result: dict[int, float] = {}
    for replicate in sorted(replicate_ids):
        local = [row for row in selected if int(_finite(row, "replicate")) == replicate]
        fs_names = sorted(
            {key for row in local for key in row if key.startswith("absolute_error_fs_k")}
        )
        if not fs_names:
            raise ValueError("higher-order rows contain no scattering errors")
        result[replicate] = max(
            max(_finite(row, "ngp_absolute_error") / 0.30 for row in local),
            max(_finite(row, key) / 0.03 for row in local for key in fs_names),
        )
    return result


def _validate_competitors(
    recoil_verdict_rows: Sequence[dict[str, object]],
    empirical_verdict_rows: Sequence[dict[str, object]],
) -> None:
    if not _claim_limited(recoil_verdict_rows) or not _claim_limited(empirical_verdict_rows):
        raise ValueError("competitor verdicts must retain all claim boundaries")
    for temperature, calibration, replicates in ((0.45, 5000.0, 3.0), (0.58, 750.0, 5.0)):
        recoil = _temperature_row(recoil_verdict_rows, temperature)
        if (
            _finite(recoil, "calibration_time") != calibration
            or _finite(recoil, "block_size") != 20.0
            or _finite(recoil, "required_realizations_per_replicate") != 16.0
            or _finite(recoil, "quality_pass") != 1.0
            or _finite(recoil, "heldout_events_used_in_calibration") != 0.0
        ):
            raise ValueError("one-step recoil provenance is incomplete")
        if "required_replicate_count" in recoil and _finite(recoil, "required_replicate_count") != replicates:
            raise ValueError("one-step recoil replicate count is not frozen")
        empirical = _temperature_row(empirical_verdict_rows, temperature)
        if (
            empirical.get("model") != "contiguous_empirical_path"
            or _finite(empirical, "curve_transfer_pass") != 1.0
            or _finite(empirical, "heldout_path_used_in_prediction") != 0.0
            or _finite(empirical, "macro_fit_parameter_count") != 0.0
        ):
            raise ValueError("contiguous empirical upper bound is incomplete")


def classify_anchor_semi_markov_gate(
    *,
    low_quality_rows: Sequence[dict[str, object]],
    low_summary_rows: Sequence[dict[str, object]],
    low_replicate_rows: Sequence[dict[str, object]],
    high_quality_rows: Sequence[dict[str, object]],
    high_summary_rows: Sequence[dict[str, object]],
    high_replicate_rows: Sequence[dict[str, object]],
    low_recoil_rows: Sequence[dict[str, object]],
    recoil_verdict_rows: Sequence[dict[str, object]],
    empirical_verdict_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Recompute all model gates before selecting one preregistered state."""

    result: dict[str, object] = {
        "mechanism_state": "mechanism_unresolved",
        "provenance_and_competitor_completeness_pass": 0.0,
        "all_low_anchor_replicates_improve_over_recoil": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    try:
        _validate_competitors(recoil_verdict_rows, empirical_verdict_rows)
        verdicts: dict[tuple[str, str], dict[str, object]] = {}
        for label, quality, summary, replicates, count in (
            ("low", low_quality_rows, low_summary_rows, low_replicate_rows, 3),
            ("high", high_quality_rows, high_summary_rows, high_replicate_rows, 5),
        ):
            for model in MODELS:
                verdicts[(label, model)] = classify_anchor_transfer(
                    [row for row in quality if row.get("model") == model],
                    [row for row in summary if row.get("model") == model],
                    [row for row in replicates if row.get("model") == model],
                    model=model,
                    expected_replicates=count,
                )
        anchor_scores = _higher_order_scores(
            low_replicate_rows,
            expected_replicates=3,
            model="anchor_aware_semi_markov",
        )
        recoil_scores = _higher_order_scores(
            low_recoil_rows,
            expected_replicates=3,
            model=None,
        )
        all_improve = all(anchor_scores[index] < recoil_scores[index] for index in anchor_scores)
    except (ValueError, KeyError, TypeError):
        return result

    result["provenance_and_competitor_completeness_pass"] = 1.0
    result["all_low_anchor_replicates_improve_over_recoil"] = float(all_improve)
    for label in ("low", "high"):
        for model in MODELS:
            prefix = f"{label}_{model}"
            verdict = verdicts[(label, model)]
            for key in (
                "quality_pass",
                "precision_pass",
                "raw_curve_transfer_pass",
                "curve_transfer_pass",
                "maximum_ensemble_msd_relative_error",
                "maximum_ensemble_ngp_absolute_error",
                "maximum_ensemble_fs_absolute_error",
            ):
                result[f"{prefix}_{key}"] = verdict[key]

    anchor_low = verdicts[("low", "anchor_aware_semi_markov")]
    anchor_high = verdicts[("high", "anchor_aware_semi_markov")]
    control_low = verdicts[("low", "state_schedule_without_anchor_geometry")]
    control_high = verdicts[("high", "state_schedule_without_anchor_geometry")]
    anchor_closes = bool(anchor_low["curve_transfer_pass"] and anchor_high["curve_transfer_pass"])
    if not anchor_closes:
        result["mechanism_state"] = "anchor_aware_model_rejected"
        return result
    if control_low["curve_transfer_pass"] and control_high["curve_transfer_pass"]:
        result["mechanism_state"] = "semi_markov_state_clock_sufficient_anchor_not_identified"
        return result
    low_control_higher_order_failure = (
        float(control_low["maximum_ensemble_ngp_absolute_error"]) > 0.30
        or float(control_low["maximum_ensemble_fs_absolute_error"]) > 0.03
    )
    if control_high["curve_transfer_pass"] and low_control_higher_order_failure and all_improve:
        result["mechanism_state"] = "anchor_geometry_required_within_tested_models"
    return result


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def _prefixed(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(prefix.name + suffix)


def write_gate_svg(path: Path, verdict: dict[str, object]) -> None:
    """Write a compact deterministic gate overview without external plotting state."""

    models = (
        ("Low anchor", "low_anchor_aware_semi_markov"),
        ("Low clock", "low_state_schedule_without_anchor_geometry"),
        ("High anchor", "high_anchor_aware_semi_markov"),
        ("High clock", "high_state_schedule_without_anchor_geometry"),
    )
    colors = ("#15616d", "#ff7d00", "#5f0f40")
    metrics = (
        ("MSD / 0.10", "maximum_ensemble_msd_relative_error", 0.10),
        ("NGP / 0.30", "maximum_ensemble_ngp_absolute_error", 0.30),
        ("Fs / 0.03", "maximum_ensemble_fs_absolute_error", 0.03),
    )
    width, height = 980, 520
    bars: list[str] = []
    for group, (label, prefix) in enumerate(models):
        x0 = 95 + group * 210
        bars.append(f'<text x="{x0 + 54}" y="420" text-anchor="middle">{label}</text>')
        for index, (_, key, tolerance) in enumerate(metrics):
            value = float(verdict.get(f"{prefix}_{key}", 0.0)) / tolerance
            bar_height = min(value, 2.0) * 140.0
            x = x0 + index * 40
            y = 380 - bar_height
            bars.append(
                f'<rect x="{x}" y="{y:.2f}" width="28" height="{bar_height:.2f}" fill="{colors[index]}"/>'
            )
    legend = "".join(
        f'<rect x="{120 + index * 210}" y="452" width="16" height="16" fill="{colors[index]}"/>'
        f'<text x="{142 + index * 210}" y="465">{label}</text>'
        for index, (label, _, _) in enumerate(metrics)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f4"/>
<style>text{{font-family:Arial,sans-serif;fill:#1e1e1e;font-size:14px;letter-spacing:0}}</style>
<text x="490" y="34" text-anchor="middle" font-size="22">Anchor-aware semi-Markov held-out gate</text>
<line x1="75" y1="240" x2="905" y2="240" stroke="#b42318" stroke-width="2" stroke-dasharray="6 5"/>
<text x="910" y="244" fill="#b42318">gate = 1</text>
{''.join(bars)}
{legend}
<text x="490" y="492" text-anchor="middle" font-weight="bold">{verdict['mechanism_state']}</text>
<text x="490" y="510" text-anchor="middle" font-size="12">Dynamical diagnostic only; no spatial-facilitation or thermodynamic-glass claim.</text>
</svg>'''
    if "nan" in svg.lower() or "inf" in svg.lower():
        raise ValueError("gate SVG contains a nonfinite value")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("low_prefix", type=Path)
    parser.add_argument("high_prefix", type=Path)
    parser.add_argument("--low-recoil-rows", type=Path, required=True)
    parser.add_argument("--low-recoil-verdict", type=Path, required=True)
    parser.add_argument("--high-recoil-verdict", type=Path, required=True)
    parser.add_argument("--low-empirical-verdict", type=Path, required=True)
    parser.add_argument("--high-empirical-verdict", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args(argv)

    empirical = [
        row
        for path in (args.low_empirical_verdict, args.high_empirical_verdict)
        for row in _read_rows(path)
        if row.get("model") == "contiguous_empirical_path"
    ]
    verdict = classify_anchor_semi_markov_gate(
        low_quality_rows=_read_rows(_prefixed(args.low_prefix, "_quality.csv")),
        low_summary_rows=_read_rows(_prefixed(args.low_prefix, "_summary.csv")),
        low_replicate_rows=_read_rows(_prefixed(args.low_prefix, "_rows.csv")),
        high_quality_rows=_read_rows(_prefixed(args.high_prefix, "_quality.csv")),
        high_summary_rows=_read_rows(_prefixed(args.high_prefix, "_summary.csv")),
        high_replicate_rows=_read_rows(_prefixed(args.high_prefix, "_rows.csv")),
        low_recoil_rows=_read_rows(args.low_recoil_rows),
        recoil_verdict_rows=_read_rows(args.low_recoil_verdict)
        + _read_rows(args.high_recoil_verdict),
        empirical_verdict_rows=empirical,
    )
    write_rows(args.output_csv, [verdict])
    write_gate_svg(args.output_svg, verdict)


if __name__ == "__main__":
    main()
