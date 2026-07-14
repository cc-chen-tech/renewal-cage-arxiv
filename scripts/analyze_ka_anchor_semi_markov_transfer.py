#!/usr/bin/env python3
"""Evaluate frozen anchor-aware semi-Markov models on held-out KA curves."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


RADIAL_BIN_COUNT = 8
REQUIRED_REALIZATIONS = 16
FROZEN_PROTOCOLS = {
    0.45: {"calibration_time": 5000, "replicate_count": 3},
    0.58: {"calibration_time": 750, "replicate_count": 5},
}

SHARED_QUALITY_LIMITS = {
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
ANCHOR_QUALITY_LIMITS = {
    "geometric_return_fraction_absolute_error": 0.02,
    "return_closure_quantile_maximum_error_over_dw": 0.05,
}
ZERO_PROVENANCE_FIELDS = (
    "heldout_events_used_in_calibration",
    "heldout_cage_residual_used_in_prediction",
    "macro_fit_parameter_count",
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)


def _as_exact_integer(value: object, name: str) -> int:
    number = float(value)
    integer = int(number)
    if not math.isfinite(number) or number != integer:
        raise ValueError(f"{name} must be an exact integer")
    return integer


def _finite(row: dict[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid field: {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"field must be finite: {key}")
    return value


def _maximum(rows: Sequence[dict[str, object]], key: str) -> float:
    if not rows:
        raise ValueError("rows must not be empty")
    return max(_finite(row, key) for row in rows)


def validate_anchor_protocol(
    *,
    temperature: float,
    calibration_time: int,
    block_size: int,
    radial_bin_count: int,
    surrogate_realizations: int,
    replicate_count: int,
) -> None:
    """Reject any drift from the preregistered transfer protocol."""

    matched_temperature = next(
        (candidate for candidate in FROZEN_PROTOCOLS if math.isclose(float(temperature), candidate)),
        None,
    )
    if matched_temperature is None:
        raise ValueError("the anchor protocol is frozen only at T=0.45 and T=0.58")
    expected = FROZEN_PROTOCOLS[matched_temperature]
    controls = {
        "calibration_time": calibration_time,
        "block_size": block_size,
        "radial_bin_count": radial_bin_count,
        "surrogate_realizations": surrogate_realizations,
        "replicate_count": replicate_count,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in controls.values()):
        raise ValueError("frozen protocol controls must be integers")
    if calibration_time != expected["calibration_time"]:
        raise ValueError("calibration time does not match the frozen temperature protocol")
    if replicate_count != expected["replicate_count"]:
        raise ValueError("replicate count does not match the frozen temperature protocol")
    if block_size != 20 or radial_bin_count != RADIAL_BIN_COUNT:
        raise ValueError("the frozen anchor protocol requires block 20 and eight radial bins")
    if surrogate_realizations != REQUIRED_REALIZATIONS:
        raise ValueError("the frozen anchor protocol requires exactly 16 realizations")


def _validate_quality_provenance(
    quality_rows: Sequence[dict[str, object]],
    *,
    model: str,
    expected_replicates: int,
) -> set[int]:
    if model not in {"anchor_aware_semi_markov", "state_schedule_without_anchor_geometry"}:
        raise ValueError("unknown anchor transfer model")
    if expected_replicates < 1:
        raise ValueError("expected replicate count must be positive")
    expected_pairs = {
        (replicate, realization)
        for replicate in range(1, expected_replicates + 1)
        for realization in range(REQUIRED_REALIZATIONS)
    }
    pairs: list[tuple[int, int]] = []
    temperatures: set[float] = set()
    seeds: set[int] = set()
    for row in quality_rows:
        if row.get("model") != model:
            raise ValueError("quality rows must contain exactly the classified model")
        replicate = _as_exact_integer(row.get("replicate"), "replicate")
        realization = _as_exact_integer(row.get("realization"), "realization")
        pairs.append((replicate, realization))
        temperature = _finite(row, "temperature")
        temperatures.add(temperature)
        seeds.add(_as_exact_integer(row.get("surrogate_base_seed"), "surrogate_base_seed"))
        if _finite(row, "calibration_events_only") != 1.0:
            raise ValueError("the kernel must use calibration events only")
        for key in ZERO_PROVENANCE_FIELDS:
            if _finite(row, key) != 0.0:
                raise ValueError(f"forbidden provenance or claim flag: {key}")
        required_geometry = float(model == "anchor_aware_semi_markov")
        if _finite(row, "geometric_return_quality_required") != required_geometry:
            raise ValueError("geometry quality provenance does not match the model")
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise ValueError("quality rows must contain the exact replicate-realization grid")
    if len(temperatures) != 1 or len(seeds) != 1:
        raise ValueError("temperature and base seed must be constant across quality rows")
    temperature = temperatures.pop()
    expected = FROZEN_PROTOCOLS.get(temperature)
    if expected is None:
        raise ValueError("quality rows use an unfrozen temperature")
    for row in quality_rows:
        validate_anchor_protocol(
            temperature=temperature,
            calibration_time=_as_exact_integer(row.get("calibration_time"), "calibration_time"),
            block_size=_as_exact_integer(row.get("block_size"), "block_size"),
            radial_bin_count=_as_exact_integer(row.get("radial_bin_count"), "radial_bin_count"),
            surrogate_realizations=_as_exact_integer(
                row.get("required_realizations_per_replicate"),
                "required_realizations_per_replicate",
            ),
            replicate_count=expected["replicate_count"],
        )
    return {replicate for replicate, _ in pairs}


def classify_anchor_transfer(
    quality_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
    replicate_rows: Sequence[dict[str, object]],
    *,
    model: str,
    expected_replicates: int,
) -> dict[str, object]:
    """Apply preregistered quality, precision, and held-out curve gates."""

    if not quality_rows or not summary_rows or not replicate_rows:
        raise ValueError("quality, summary, and replicate rows must not be empty")
    replicate_ids = _validate_quality_provenance(
        quality_rows,
        model=model,
        expected_replicates=expected_replicates,
    )
    if any(row.get("model") != model for row in (*summary_rows, *replicate_rows)):
        raise ValueError("curve rows must contain exactly the classified model")
    observed_replicates = {
        _as_exact_integer(row.get("replicate"), "replicate") for row in replicate_rows
    }
    if observed_replicates != replicate_ids:
        raise ValueError("replicate curve rows do not match the quality grid")

    quality_maxima = {key: _maximum(quality_rows, key) for key in SHARED_QUALITY_LIMITS}
    quality_pass = all(
        quality_maxima[key] <= limit for key, limit in SHARED_QUALITY_LIMITS.items()
    )
    unsupported_maximum = _maximum(quality_rows, "unsupported_tuple_count")
    quality_pass = quality_pass and unsupported_maximum == 0.0
    geometry_maxima = {key: _maximum(quality_rows, key) for key in ANCHOR_QUALITY_LIMITS}
    if model == "anchor_aware_semi_markov":
        quality_pass = quality_pass and all(
            geometry_maxima[key] <= limit for key, limit in ANCHOR_QUALITY_LIMITS.items()
        )

    maximum_msd_mc = _maximum(summary_rows, "ensemble_msd_mc_relative_se")
    maximum_ngp_mc = _maximum(summary_rows, "ensemble_ngp_mc_se")
    fs_mc_names = sorted(
        {
            key
            for row in summary_rows
            for key in row
            if key.startswith("ensemble_fs_k") and key.endswith("_mc_se")
        }
    )
    if not fs_mc_names:
        raise ValueError("summary rows must include multi-k scattering precision")
    maximum_fs_mc = max(_finite(row, key) for row in summary_rows for key in fs_mc_names)
    precision_pass = maximum_msd_mc <= 0.01 and maximum_ngp_mc <= 0.03 and maximum_fs_mc <= 0.003

    maximum_msd = _maximum(summary_rows, "ensemble_msd_relative_error")
    maximum_ngp = _maximum(summary_rows, "ensemble_ngp_absolute_error")
    fs_error_names = sorted(
        {
            key
            for row in summary_rows
            for key in row
            if key.startswith("ensemble_absolute_error_fs_k")
        }
    )
    if not fs_error_names:
        raise ValueError("summary rows must include multi-k scattering errors")
    maximum_fs = max(_finite(row, key) for row in summary_rows for key in fs_error_names)
    raw_curve_pass = maximum_msd <= 0.10 and maximum_ngp <= 0.30 and maximum_fs <= 0.03
    curve_pass = quality_pass and precision_pass and raw_curve_pass
    if not quality_pass:
        mechanism_state = "unresolved_quality"
    elif not precision_pass:
        mechanism_state = "unresolved_precision"
    elif raw_curve_pass:
        mechanism_state = "curve_closed"
    else:
        mechanism_state = "curve_open"

    result: dict[str, object] = {
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
        "maximum_unsupported_tuple_count": unsupported_maximum,
        "maximum_ensemble_msd_relative_error": maximum_msd,
        "maximum_ensemble_ngp_absolute_error": maximum_ngp,
        "maximum_ensemble_fs_absolute_error": maximum_fs,
        "maximum_ensemble_msd_mc_relative_se": maximum_msd_mc,
        "maximum_ensemble_ngp_mc_se": maximum_ngp_mc,
        "maximum_ensemble_fs_mc_se": maximum_fs_mc,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    result.update({f"maximum_{key}": value for key, value in quality_maxima.items()})
    result.update({f"maximum_{key}": value for key, value in geometry_maxima.items()})
    return result

