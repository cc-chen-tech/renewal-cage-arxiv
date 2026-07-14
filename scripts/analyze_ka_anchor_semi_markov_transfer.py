#!/usr/bin/env python3
"""Classify anchor-aware semi-Markov transfer gates."""

from __future__ import annotations

import math
from collections.abc import Sequence


REQUIRED_REALIZATIONS = 16
RADIAL_BIN_COUNT = 8

QUALITY_LIMITS = {
    "scheduled_return_fraction_absolute_error": 0.02,
    "scheduled_return_given_return_absolute_error": 0.03,
    "scheduled_return_given_escape_absolute_error": 0.03,
    "return_holding_time_mean_relative_error": 0.05,
    "escape_holding_time_mean_relative_error": 0.05,
    "return_holding_time_quantile_maximum_relative_error": 0.10,
    "escape_holding_time_quantile_maximum_relative_error": 0.10,
    "radial_mean_relative_error": 0.02,
    "radial_standard_deviation_relative_error": 0.02,
    "lag_one_cosine_mean_absolute_error": 0.02,
    "lag_one_cosine_quantile_maximum_absolute_error": 0.03,
}

ANCHOR_GEOMETRY_LIMITS = {
    "geometric_return_fraction_absolute_error": 0.02,
    "return_closure_quantile_maximum_error_over_dw": 0.05,
}

CLAIM_FLAGS = (
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)


def _finite_float(row: dict[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing or nonnumeric field: {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"field must be finite: {key}")
    return value


def validate_anchor_protocol(
    *,
    temperature: float,
    calibration_time: int,
    block_size: int,
    radial_bin_count: int,
    surrogate_realizations: int,
    replicate_count: int,
) -> None:
    """Validate the frozen public anchor-transfer protocols."""

    frozen = {
        0.45: {"calibration_time": 5000, "replicate_count": 3},
        0.58: {"calibration_time": 750, "replicate_count": 5},
    }
    protocol = frozen.get(float(temperature))
    if protocol is None:
        raise ValueError("unsupported anchor-transfer temperature")
    if int(calibration_time) != protocol["calibration_time"]:
        raise ValueError("unexpected anchor-transfer calibration time")
    if int(replicate_count) != protocol["replicate_count"]:
        raise ValueError("unexpected anchor-transfer replicate count")
    if int(block_size) != 20:
        raise ValueError("anchor-transfer block size must be 20")
    if int(radial_bin_count) != RADIAL_BIN_COUNT:
        raise ValueError("anchor-transfer radial bin count must be 8")
    if int(surrogate_realizations) != REQUIRED_REALIZATIONS:
        raise ValueError("anchor-transfer realization count must be 16")


def _validate_quality_rows(
    quality_rows: Sequence[dict[str, object]],
    *,
    model: str,
    expected_replicates: int,
) -> tuple[set[int], bool]:
    if not quality_rows:
        raise ValueError("quality rows must not be empty")
    if expected_replicates < 1:
        raise ValueError("expected replicate count must be positive")

    pairs = {
        (int(_finite_float(row, "replicate")), int(_finite_float(row, "realization")))
        for row in quality_rows
    }
    replicate_ids = {replicate for replicate, _ in pairs}
    required_realizations = set(range(REQUIRED_REALIZATIONS))
    complete = (
        len(replicate_ids) == expected_replicates
        and len(pairs) == len(quality_rows) == expected_replicates * REQUIRED_REALIZATIONS
        and all(
            {realization for candidate, realization in pairs if candidate == replicate}
            == required_realizations
            for replicate in replicate_ids
        )
    )
    if not complete:
        raise ValueError("anchor-transfer quality rows must contain the exact realization grid")

    for row in quality_rows:
        if row.get("model", model) != model:
            raise ValueError("quality rows must match the requested model")
        if _finite_float(row, "calibration_time") != 5000.0:
            raise ValueError("anchor-transfer classifier expects the frozen T=0.45 calibration window")
        if _finite_float(row, "block_size") != 20.0:
            raise ValueError("anchor-transfer block size must be frozen at 20")
        if _finite_float(row, "radial_bin_count") != float(RADIAL_BIN_COUNT):
            raise ValueError("anchor-transfer radial bin count must be frozen at 8")
        if _finite_float(row, "required_realizations_per_replicate") != float(REQUIRED_REALIZATIONS):
            raise ValueError("anchor-transfer realization count must be frozen at 16")
        if _finite_float(row, "calibration_events_only") != 1.0:
            raise ValueError("anchor-transfer quality rows must be calibration-only")
        for key in (
            "heldout_events_used_in_calibration",
            "heldout_cage_residual_used_in_prediction",
            "macro_fit_parameter_count",
            *CLAIM_FLAGS,
        ):
            if _finite_float(row, key) != 0.0:
                raise ValueError(f"anchor-transfer provenance or claim flag is nonzero: {key}")

    quality_pass = all(
        all(_finite_float(row, key) <= limit for key, limit in QUALITY_LIMITS.items())
        and _finite_float(row, "unsupported_tuple_count") == 0.0
        for row in quality_rows
    )
    if model == "anchor_aware_semi_markov":
        quality_pass = quality_pass and all(
            _finite_float(row, "geometric_return_quality_required") == 1.0
            and all(
                _finite_float(row, key) <= limit
                for key, limit in ANCHOR_GEOMETRY_LIMITS.items()
            )
            for row in quality_rows
        )
    return replicate_ids, bool(quality_pass)


def classify_anchor_transfer(
    quality_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
    replicate_rows: Sequence[dict[str, object]],
    *,
    model: str,
    expected_replicates: int,
) -> dict[str, object]:
    """Classify quality, Monte Carlo precision, and held-out curve transfer."""

    if model not in {"anchor_aware_semi_markov", "state_schedule_without_anchor_geometry"}:
        raise ValueError("unsupported anchor-transfer model")
    if not summary_rows or not replicate_rows:
        raise ValueError("summary and replicate rows must not be empty")

    replicate_ids, quality_pass = _validate_quality_rows(
        quality_rows,
        model=model,
        expected_replicates=expected_replicates,
    )
    summary_replicates = {_finite_float(row, "independent_replicate_count") for row in summary_rows}
    if summary_replicates != {float(expected_replicates)}:
        raise ValueError("summary rows must record the requested replicate count")

    replicate_curve_ids = {int(_finite_float(row, "replicate")) for row in replicate_rows}
    if replicate_curve_ids != replicate_ids:
        raise ValueError("replicate curve rows must cover the quality replicates")

    maximum_msd_mc = max(_finite_float(row, "ensemble_msd_mc_relative_se") for row in summary_rows)
    maximum_ngp_mc = max(_finite_float(row, "ensemble_ngp_mc_se") for row in summary_rows)
    fs_mc_names = sorted(
        key
        for row in summary_rows
        for key in row
        if key.startswith("ensemble_fs_k") and key.endswith("_mc_se")
    )
    if not fs_mc_names:
        raise ValueError("summary rows must include multi-k Monte Carlo errors")
    maximum_fs_mc = max(_finite_float(row, key) for row in summary_rows for key in fs_mc_names)
    precision_pass = maximum_msd_mc <= 0.01 and maximum_ngp_mc <= 0.03 and maximum_fs_mc <= 0.003

    maximum_msd = max(_finite_float(row, "ensemble_msd_relative_error") for row in summary_rows)
    maximum_ngp = max(_finite_float(row, "ensemble_ngp_absolute_error") for row in summary_rows)
    fs_error_names = sorted(
        key
        for row in summary_rows
        for key in row
        if key.startswith("ensemble_absolute_error_fs_k")
    )
    if not fs_error_names:
        raise ValueError("summary rows must include multi-k scattering errors")
    maximum_fs = max(_finite_float(row, key) for row in summary_rows for key in fs_error_names)
    raw_curve_pass = maximum_msd <= 0.10 and maximum_ngp <= 0.30 and maximum_fs <= 0.03

    curve_pass = quality_pass and precision_pass and raw_curve_pass
    if not quality_pass:
        mechanism_state = "unresolved_quality"
    elif not precision_pass:
        mechanism_state = "unresolved_precision"
    elif curve_pass:
        mechanism_state = "curve_closed"
    else:
        mechanism_state = "curve_open"

    return {
        "model": model,
        "independent_replicate_count": float(len(replicate_ids)),
        "required_replicate_count": float(expected_replicates),
        "required_realizations_per_replicate": float(REQUIRED_REALIZATIONS),
        "radial_bin_count": float(RADIAL_BIN_COUNT),
        "quality_realization_completeness_pass": 1.0,
        "quality_pass": float(quality_pass),
        "precision_pass": float(precision_pass),
        "raw_curve_transfer_pass": float(raw_curve_pass),
        "curve_transfer_pass": float(curve_pass),
        "mechanism_state": mechanism_state,
        "maximum_ensemble_msd_relative_error": maximum_msd,
        "maximum_ensemble_ngp_absolute_error": maximum_ngp,
        "maximum_ensemble_fs_absolute_error": maximum_fs,
        "maximum_ensemble_msd_mc_relative_se": maximum_msd_mc,
        "maximum_ensemble_ngp_mc_se": maximum_ngp_mc,
        "maximum_ensemble_fs_mc_se": maximum_fs_mc,
        "heldout_events_used_in_calibration": 0.0,
        "heldout_cage_residual_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
