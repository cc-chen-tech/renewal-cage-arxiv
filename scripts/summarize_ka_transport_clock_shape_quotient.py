#!/usr/bin/env python3
"""Separate transport-clock drift from fixed-MSD shape residuals."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections.abc import Sequence
from pathlib import Path


MODELS = (
    "within_particle_segment_shuffle",
    "cross_particle_segment_splice",
)
OBSERVABLES = {
    "ngp": ("predicted_ngp", "observed_ngp", 0.30),
    "fs_k2": ("predicted_fs_k2", "observed_fs_k2", 0.03),
    "fs_k4": ("predicted_fs_k4", "observed_fs_k4", 0.03),
    "fs_k7p25": ("predicted_fs_k7p25", "observed_fs_k7p25", 0.03),
}
SOURCE_CLAIM_FIELDS = (
    "microdynamic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)
CLOSED_GATE_FIELDS = (
    "blind_prediction_claim_allowed",
    "clock_only_closure_allowed",
    "cooling_enhanced_shape_memory_claim_allowed",
    "finite_exchange_resolved",
    "static_environment_resolved",
    "spatial_facilitation_resolved",
    "microdynamic_closure_claim_allowed",
    "thermodynamic_claim_allowed",
)
STATIONARITY_COMPARISONS = {
    "early_late",
    "early_heldout",
    "late_heldout",
}
FULL_PATH_MODEL_EQUIVALENCE_ATOL = 1e-12
CSV_FLOAT_SIGNIFICANT_DIGITS = 15
INDEPENDENCE_CLASS = "decorrelated_parent_frames_plus_velocity_seeds"
FROZEN_PROTOCOLS = {
    0.45: {
        "full_length": 250,
        "lags": (20, 100, 200, 500, 1000, 2000, 3000),
        "replicate_count": 3,
    },
    0.58: {
        "full_length": 37,
        "lags": (20, 100, 200, 400, 600),
        "replicate_count": 5,
    },
}


def canonical_csv_value(value: object) -> object:
    """Serialize floats independently of platform-level final-bit drift."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize a nonfinite quotient value")
        return format(value, f".{CSV_FLOAT_SIGNIFICANT_DIGITS}g")
    return value


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


def interpolate_no_extrapolation(
    xs: Sequence[float],
    ys: Sequence[float],
    target: float,
) -> float | None:
    """Piecewise-linearly interpolate a strictly increasing finite support."""

    try:
        x_values = tuple(float(value) for value in xs)
        y_values = tuple(float(value) for value in ys)
        local_target = float(target)
    except (TypeError, ValueError) as error:
        raise ValueError("interpolation inputs must be finite numeric sequences") from error
    if (
        len(x_values) < 2
        or len(x_values) != len(y_values)
        or any(not math.isfinite(value) for value in x_values + y_values)
        or not math.isfinite(local_target)
        or any(
            x_values[index] >= x_values[index + 1]
            for index in range(len(x_values) - 1)
        )
    ):
        raise ValueError("interpolation support must be finite and strictly increasing")
    if local_target < x_values[0] or local_target > x_values[-1]:
        return None
    for index in range(len(x_values) - 1):
        low = x_values[index]
        high = x_values[index + 1]
        if low <= local_target <= high:
            fraction = (local_target - low) / (high - low)
            return y_values[index] + fraction * (
                y_values[index + 1] - y_values[index]
            )
    return y_values[-1]


def _manifest_replicates(
    manifest: dict[str, object],
    *,
    temperature: float,
) -> tuple[int, ...]:
    try:
        manifest_temperature = _finite(manifest, "temperature")
        replicate_count = _exact_int(manifest, "replicate_count")
        replicate_rows = manifest["replicates"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("manifest is missing the frozen replicate contract") from error
    if (
        manifest_temperature != temperature
        or not isinstance(replicate_rows, list)
        or manifest.get("independently_prepared_parent_samples") is not False
        or manifest.get("independence_class") != INDEPENDENCE_CLASS
        or _finite(manifest, "replicate_provenance_validation_pass") != 1.0
        or _finite(manifest, "parent_sample_count") != 1.0
        or _finite(manifest, "independent_replicate_count") != 0.0
        or manifest.get("thermodynamic_claim_allowed") is not False
    ):
        raise ValueError("manifest provenance or claim boundary is invalid")
    try:
        replicates = tuple(_exact_int(row, "replicate") for row in replicate_rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("manifest replicate rows are invalid") from error
    if (
        len(replicates) != replicate_count
        or len(set(replicates)) != len(replicates)
        or tuple(sorted(replicates)) != replicates
        or any(replicate < 1 for replicate in replicates)
    ):
        raise ValueError("manifest replicate support is not exact")
    return replicates


def _manifest_expected_lags(manifest: dict[str, object]) -> tuple[int, ...]:
    try:
        lag_rows = manifest["expected_lags"]
    except KeyError as error:
        raise ValueError("manifest is missing the frozen lag contract") from error
    if not isinstance(lag_rows, list):
        raise ValueError("manifest expected_lags must be a list")
    lags = tuple(_exact_int({"lag": value}, "lag") for value in lag_rows)
    if (
        not lags
        or len(set(lags)) != len(lags)
        or tuple(sorted(lags)) != lags
        or any(lag < 1 for lag in lags)
    ):
        raise ValueError("manifest lag support is not exact")
    return lags


def _false_string(value: object, field: str) -> None:
    if str(value).strip().lower() not in {"false", "0", "0.0"}:
        raise ValueError(f"provenance field must remain false: {field}")


def manifests_from_provenance(
    provenance_rows: Sequence[dict[str, object]],
) -> dict[float, dict[str, object]]:
    """Reconstruct the minimal manifest contract from committed provenance."""

    if not provenance_rows:
        raise ValueError("provenance rows must be nonempty")
    grouped: dict[float, list[dict[str, object]]] = {0.45: [], 0.58: []}
    keys: set[tuple[float, int]] = set()
    for row in provenance_rows:
        temperature = _finite(row, "temperature")
        if temperature not in grouped:
            raise ValueError("provenance contains an unexpected temperature")
        replicate = _exact_int(row, "replicate")
        key = (temperature, replicate)
        if key in keys:
            raise ValueError("provenance contains duplicate replicate rows")
        keys.add(key)
        if str(row.get("independence_class")) != INDEPENDENCE_CLASS:
            raise ValueError("provenance independence class is not the frozen restart class")
        _false_string(
            row.get("independently_prepared_parent_samples"),
            "independently_prepared_parent_samples",
        )
        _false_string(row.get("thermodynamic_claim_allowed"), "thermodynamic_claim_allowed")
        grouped[temperature].append(row)
    manifests: dict[float, dict[str, object]] = {}
    for temperature, rows in grouped.items():
        expected_count = int(FROZEN_PROTOCOLS[temperature]["replicate_count"])
        if len(rows) != expected_count:
            raise ValueError("provenance does not cover the exact frozen restart count")
        ordered = sorted(rows, key=lambda row: _exact_int(row, "replicate"))
        if tuple(_exact_int(row, "replicate") for row in ordered) != tuple(
            range(1, expected_count + 1)
        ):
            raise ValueError("provenance replicate labels do not match the frozen grid")
        source_hashes = {str(row.get("source_sha256", "")) for row in ordered}
        if len(source_hashes) != 1 or not next(iter(source_hashes)):
            raise ValueError("each temperature must have one identified parent sample")
        frames = [_exact_int(row, "source_frame_index") for row in ordered]
        seeds = [_exact_int(row, "velocity_seed") for row in ordered]
        if len(set(frames)) != expected_count or len(set(seeds)) != expected_count:
            raise ValueError("restart frames and velocity seeds must be distinct")
        manifests[temperature] = {
            "temperature": temperature,
            "replicate_count": len(ordered),
            "expected_lags": list(FROZEN_PROTOCOLS[temperature]["lags"]),
            "independently_prepared_parent_samples": False,
            "independence_class": INDEPENDENCE_CLASS,
            "replicate_provenance_validation_pass": 1.0,
            "parent_sample_count": 1.0,
            "independent_replicate_count": 0.0,
            "replicates": [
                {
                    "replicate": _exact_int(row, "replicate"),
                    "directory": f"replicate_{_exact_int(row, 'replicate'):02d}",
                }
                for row in ordered
            ],
            "thermodynamic_claim_allowed": False,
        }
    return manifests


def _validated_full_path_rows(
    source_rows: Sequence[dict[str, object]],
    manifest: dict[str, object],
    *,
    temperature: float,
    full_length: int,
    expected_lags: Sequence[int],
) -> tuple[list[dict[str, object]], tuple[int, ...], tuple[int, ...]]:
    if (
        isinstance(full_length, bool)
        or not isinstance(full_length, int)
        or full_length < 1
    ):
        raise ValueError("full_length must be a positive integer")
    try:
        lags = tuple(expected_lags)
    except TypeError as error:
        raise ValueError("expected_lags must be an integer sequence") from error
    if (
        not lags
        or len(set(lags)) != len(lags)
        or tuple(sorted(lags)) != lags
        or any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 1 for lag in lags)
    ):
        raise ValueError("expected_lags must be unique increasing positive integers")
    replicates = _manifest_replicates(manifest, temperature=temperature)
    if lags != _manifest_expected_lags(manifest):
        raise ValueError("requested lags differ from the frozen manifest grid")
    numeric_fields = {
        "temperature",
        "segment_length",
        "replicate",
        "lag",
        "predicted_msd",
        "observed_msd",
        "heldout_path_used_in_prediction",
        "macro_fit_parameter_count",
        *SOURCE_CLAIM_FIELDS,
    }
    numeric_fields.update(
        field
        for predicted, observed, _ in OBSERVABLES.values()
        for field in (predicted, observed)
    )
    if not source_rows:
        raise ValueError("source rows must be nonempty")
    selected: list[dict[str, object]] = []
    source_keys: set[tuple[str, float, int, int, int]] = set()
    for row in source_rows:
        for field in numeric_fields:
            _finite(row, field)
        for field in SOURCE_CLAIM_FIELDS:
            if _finite(row, field) != 0.0:
                raise ValueError("source claim boundaries must remain closed")
        if (
            _finite(row, "heldout_path_used_in_prediction") != 0.0
            or _finite(row, "macro_fit_parameter_count") != 0.0
        ):
            raise ValueError("source rows contain heldout leakage or macro fitting")
        model = str(row.get("model"))
        key = (
            model,
            _finite(row, "temperature"),
            _exact_int(row, "segment_length"),
            _exact_int(row, "replicate"),
            _exact_int(row, "lag"),
        )
        if key in source_keys:
            raise ValueError("source table contains duplicate cells")
        source_keys.add(key)
        if key[1] == temperature and key[2] == full_length and model in MODELS:
            selected.append(row)
    expected = {
        (model, temperature, full_length, replicate, lag)
        for model in MODELS
        for replicate in replicates
        for lag in lags
    }
    actual = {
        (
            str(row["model"]),
            _finite(row, "temperature"),
            _exact_int(row, "segment_length"),
            _exact_int(row, "replicate"),
            _exact_int(row, "lag"),
        )
        for row in selected
    }
    if actual != expected:
        raise ValueError("full-path source rows do not match the frozen grid")
    indexed = {
        (str(row["model"]), _exact_int(row, "replicate"), _exact_int(row, "lag")): row
        for row in selected
    }
    equivalent_fields = (
        "predicted_msd",
        "observed_msd",
        *(field for pair in OBSERVABLES.values() for field in pair[:2]),
    )
    for replicate in replicates:
        for lag in lags:
            within = indexed[(MODELS[0], replicate, lag)]
            cross = indexed[(MODELS[1], replicate, lag)]
            for field in equivalent_fields:
                if abs(_finite(within, field) - _finite(cross, field)) > FULL_PATH_MODEL_EQUIVALENCE_ATOL:
                    raise ValueError(f"full-path models disagree for {field}")
    within_rows = [row for row in selected if str(row["model"]) == MODELS[0]]
    return within_rows, replicates, lags


def compute_transport_clock_shape_rows(
    source_rows: Sequence[dict[str, object]],
    manifest: dict[str, object],
    *,
    temperature: float,
    full_length: int,
    expected_lags: Sequence[int],
    minimum_support_fraction: float = 0.80,
) -> list[dict[str, object]]:
    """Compute replicate-resolved same-time and fixed-MSD residual rows."""

    try:
        support_threshold = float(minimum_support_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError("minimum_support_fraction must lie in (0,1]") from error
    if not math.isfinite(support_threshold) or not 0.0 < support_threshold <= 1.0:
        raise ValueError("minimum_support_fraction must lie in (0,1]")
    rows, replicates, lags = _validated_full_path_rows(
        source_rows,
        manifest,
        temperature=temperature,
        full_length=full_length,
        expected_lags=expected_lags,
    )
    result: list[dict[str, object]] = []
    for replicate in replicates:
        local = sorted(
            [row for row in rows if _exact_int(row, "replicate") == replicate],
            key=lambda row: _exact_int(row, "lag"),
        )
        calibration_msd = [_finite(row, "predicted_msd") for row in local]
        if any(
            calibration_msd[index] >= calibration_msd[index + 1]
            for index in range(len(calibration_msd) - 1)
        ):
            raise ValueError("calibration MSD must increase strictly with lag")
        support = [
            calibration_msd[0] <= _finite(row, "observed_msd") <= calibration_msd[-1]
            for row in local
        ]
        support_fraction = sum(support) / len(local)
        alpha_window_points = sum(
            in_support and 0.1 <= _finite(row, "observed_fs_k4") <= 0.9
            for row, in_support in zip(local, support)
        )
        if support_fraction < support_threshold or alpha_window_points < 1:
            raise ValueError("replicate lacks frozen MSD support or anchor-alpha coverage")
        drift = [
            _finite(row, "predicted_msd") - _finite(row, "observed_msd")
            for row in local
        ]
        drift_sign = (
            1.0
            if all(value > 0.0 for value in drift)
            else -1.0
            if all(value < 0.0 for value in drift)
            else 0.0
        )
        for row, in_support in zip(local, support):
            heldout_msd = _finite(row, "observed_msd")
            for observable, (predicted_field, observed_field, tolerance) in OBSERVABLES.items():
                same_time = _finite(row, predicted_field)
                heldout = _finite(row, observed_field)
                matched = interpolate_no_extrapolation(
                    calibration_msd,
                    [_finite(item, predicted_field) for item in local],
                    heldout_msd,
                )
                if bool(matched is not None) != in_support:
                    raise ValueError("interpolation support classification is inconsistent")
                same_error = abs(same_time - heldout)
                matched_error = "" if matched is None else abs(matched - heldout)
                result.append(
                    {
                        "temperature": temperature,
                        "replicate": float(replicate),
                        "lag": float(_exact_int(row, "lag")),
                        "observable": observable,
                        "tolerance": tolerance,
                        "calibration_msd": _finite(row, "predicted_msd"),
                        "heldout_msd": heldout_msd,
                        "same_time_value": same_time,
                        "heldout_value": heldout,
                        "msd_matched_value": "" if matched is None else matched,
                        "same_time_absolute_error": same_error,
                        "msd_matched_absolute_error": matched_error,
                        "same_time_normalized_error": same_error / tolerance,
                        "msd_matched_normalized_error": (
                            "" if matched is None else float(matched_error) / tolerance
                        ),
                        "in_calibration_msd_support": float(in_support),
                        "replicate_support_fraction": support_fraction,
                        "replicate_anchor_alpha_window_point_count": float(alpha_window_points),
                        "replicate_clock_drift_sign": drift_sign,
                        "heldout_msd_used_as_diagnostic_input": 1.0,
                        "blind_prediction_claim_allowed": 0.0,
                        "microdynamic_closure_claim_allowed": 0.0,
                        "spatial_facilitation_claim_allowed": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
    return result


def _validated_stationarity_pass(
    rows: Sequence[dict[str, object]],
    *,
    temperature: float,
) -> bool:
    if not rows:
        raise ValueError("stationarity rows must be nonempty")
    comparisons: set[str] = set()
    passes: list[bool] = []
    for row in rows:
        if _finite(row, "temperature") != temperature:
            raise ValueError("stationarity temperature does not match its table")
        comparison = str(row.get("comparison"))
        if comparison in comparisons:
            raise ValueError("stationarity table contains duplicate comparisons")
        comparisons.add(comparison)
        for field in SOURCE_CLAIM_FIELDS:
            if _finite(row, field) != 0.0:
                raise ValueError("stationarity claim boundaries must remain closed")
        value = _finite(row, "curve_transfer_pass")
        if value not in (0.0, 1.0):
            raise ValueError("curve_transfer_pass must be binary")
        passes.append(value == 1.0)
    if comparisons != STATIONARITY_COMPARISONS:
        raise ValueError("stationarity table does not cover the exact comparison set")
    return all(passes)


def _validated_quotient_temperature_rows(
    rows: Sequence[dict[str, object]],
    manifest: dict[str, object],
    *,
    temperature: float,
    minimum_support_fraction: float,
) -> tuple[list[dict[str, object]], tuple[int, ...], tuple[int, ...]]:
    replicates = _manifest_replicates(manifest, temperature=temperature)
    selected = [row for row in rows if _finite(row, "temperature") == temperature]
    if not selected:
        raise ValueError("quotient rows do not cover the required temperature")
    keys: set[tuple[int, int, str]] = set()
    lags: set[int] = set()
    for row in selected:
        replicate = _exact_int(row, "replicate")
        lag = _exact_int(row, "lag")
        observable = str(row.get("observable"))
        if observable not in OBSERVABLES:
            raise ValueError("quotient row contains an unknown observable")
        key = (replicate, lag, observable)
        if key in keys:
            raise ValueError("quotient rows contain duplicate cells")
        keys.add(key)
        lags.add(lag)
        expected_tolerance = OBSERVABLES[observable][2]
        if _finite(row, "tolerance") != expected_tolerance:
            raise ValueError("quotient tolerance differs from the frozen source gate")
        for field in (
            "heldout_msd_used_as_diagnostic_input",
            "in_calibration_msd_support",
            "blind_prediction_claim_allowed",
            *SOURCE_CLAIM_FIELDS,
        ):
            value = _finite(row, field)
            if field == "heldout_msd_used_as_diagnostic_input":
                if value != 1.0:
                    raise ValueError("heldout MSD diagnostic input must remain explicit")
            elif value != 0.0 and field != "in_calibration_msd_support":
                raise ValueError("quotient claim boundaries must remain closed")
            elif field == "in_calibration_msd_support" and value not in (0.0, 1.0):
                raise ValueError("support flag must be binary")
        support_fraction = _finite(row, "replicate_support_fraction")
        alpha_count = _finite(row, "replicate_anchor_alpha_window_point_count")
        drift_sign = _finite(row, "replicate_clock_drift_sign")
        if (
            support_fraction < minimum_support_fraction
            or support_fraction > 1.0
            or alpha_count < 1.0
            or not alpha_count.is_integer()
            or drift_sign not in (-1.0, 0.0, 1.0)
        ):
            raise ValueError("quotient replicate support contract is invalid")
        same_error = _finite(row, "same_time_absolute_error")
        same_normalized = _finite(row, "same_time_normalized_error")
        if abs(same_normalized - same_error / expected_tolerance) > 1e-12:
            raise ValueError("same-time normalized error is inconsistent")
        in_support = _finite(row, "in_calibration_msd_support") == 1.0
        matched_error = row.get("msd_matched_absolute_error")
        matched_normalized = row.get("msd_matched_normalized_error")
        matched_value = row.get("msd_matched_value")
        if in_support:
            local_error = _finite(row, "msd_matched_absolute_error")
            local_normalized = _finite(row, "msd_matched_normalized_error")
            _finite(row, "msd_matched_value")
            if abs(local_normalized - local_error / expected_tolerance) > 1e-12:
                raise ValueError("MSD-matched normalized error is inconsistent")
        elif any(value not in ("", None) for value in (matched_error, matched_normalized, matched_value)):
            raise ValueError("unsupported rows must not contain extrapolated values")
    ordered_lags = tuple(sorted(lags))
    if ordered_lags != _manifest_expected_lags(manifest):
        raise ValueError("quotient rows do not cover the frozen manifest lag grid")
    expected = {
        (replicate, lag, observable)
        for replicate in replicates
        for lag in ordered_lags
        for observable in OBSERVABLES
    }
    if keys != expected:
        raise ValueError("quotient rows do not form a complete replicate-lag grid")
    for replicate in replicates:
        local = [row for row in selected if _exact_int(row, "replicate") == replicate]
        if len({_finite(row, "replicate_support_fraction") for row in local}) != 1:
            raise ValueError("replicate support fraction changes across quotient rows")
        if len({_finite(row, "replicate_anchor_alpha_window_point_count") for row in local}) != 1:
            raise ValueError("replicate alpha-window count changes across quotient rows")
        if len({_finite(row, "replicate_clock_drift_sign") for row in local}) != 1:
            raise ValueError("replicate clock-drift sign changes across quotient rows")
    return selected, replicates, ordered_lags


def classify_transport_clock_shape_gate(
    quotient_rows: Sequence[dict[str, object]],
    stationarity_rows: dict[float, Sequence[dict[str, object]]],
    manifests: dict[float, dict[str, object]],
    *,
    minimum_support_fraction: float = 0.80,
) -> list[dict[str, object]]:
    """Classify T045 separation and retain T058 as an unresolved canary."""

    try:
        support_threshold = float(minimum_support_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError("minimum_support_fraction must lie in (0,1]") from error
    if not math.isfinite(support_threshold) or not 0.0 < support_threshold <= 1.0:
        raise ValueError("minimum_support_fraction must lie in (0,1]")
    temperatures = (0.45, 0.58)
    if set(stationarity_rows) != set(temperatures) or set(manifests) != set(temperatures):
        raise ValueError("classifier requires the exact T045/T058 control tables")
    result: list[dict[str, object]] = []
    for temperature in temperatures:
        local, replicates, lags = _validated_quotient_temperature_rows(
            quotient_rows,
            manifests[temperature],
            temperature=temperature,
            minimum_support_fraction=support_threshold,
        )
        stationarity_pass = _validated_stationarity_pass(
            stationarity_rows[temperature],
            temperature=temperature,
        )
        full_grid_same: dict[str, float] = {}
        support_same: dict[str, float] = {}
        matched: dict[str, float] = {}
        for observable in OBSERVABLES:
            observable_rows = [row for row in local if row["observable"] == observable]
            supported = [
                row
                for row in observable_rows
                if _finite(row, "in_calibration_msd_support") == 1.0
            ]
            if not supported:
                raise ValueError("observable has no interpolation support")
            full_grid_same[observable] = max(
                _finite(row, "same_time_normalized_error") for row in observable_rows
            )
            support_same[observable] = max(
                _finite(row, "same_time_normalized_error") for row in supported
            )
            matched[observable] = max(
                _finite(row, "msd_matched_normalized_error") for row in supported
            )
        support_fractions = {
            replicate: _finite(
                next(row for row in local if _exact_int(row, "replicate") == replicate),
                "replicate_support_fraction",
            )
            for replicate in replicates
        }
        alpha_counts = {
            replicate: _finite(
                next(row for row in local if _exact_int(row, "replicate") == replicate),
                "replicate_anchor_alpha_window_point_count",
            )
            for replicate in replicates
        }
        drift_signs = {
            replicate: _finite(
                next(row for row in local if _exact_int(row, "replicate") == replicate),
                "replicate_clock_drift_sign",
            )
            for replicate in replicates
        }
        opposite_signs = -1.0 in drift_signs.values() and 1.0 in drift_signs.values()
        matched_pass = {observable: value <= 1.0 for observable, value in matched.items()}
        clock_shape_separation = (
            temperature == 0.45
            and stationarity_pass
            and min(support_fractions.values()) >= support_threshold
            and min(alpha_counts.values()) >= 1.0
            and support_same["fs_k2"] > 1.0
            and support_same["fs_k4"] > 1.0
            and matched_pass["fs_k2"]
            and matched_pass["fs_k4"]
            and (not matched_pass["ngp"] or not matched_pass["fs_k7p25"])
        )
        gate: dict[str, object] = {
            "temperature": temperature,
            "analysis_status": (
                "clock_shape_separation_exploratory"
                if temperature == 0.45 and clock_shape_separation
                else "high_temperature_canary_only"
                if temperature == 0.58
                else "decomposition_unresolved"
            ),
            "replicate_count": float(len(replicates)),
            "lag_count_per_replicate": float(len(lags)),
            "minimum_replicate_support_fraction": min(support_fractions.values()),
            "minimum_anchor_alpha_window_point_count": min(alpha_counts.values()),
            "support_coverage_pass": float(
                min(support_fractions.values()) >= support_threshold
            ),
            "anchor_alpha_window_inclusion_pass": float(
                min(alpha_counts.values()) >= 1.0
            ),
            "replicate_clock_drift_opposite_signs": float(opposite_signs),
            "all_replicates_calibration_slower_than_heldout": float(
                set(drift_signs.values()) == {-1.0}
            ),
            "source_ensemble_stationarity_all_comparisons_pass": float(stationarity_pass),
            "replicate_provenance_validation_pass": 1.0,
            "parent_sample_count": 1.0,
            "independent_replicate_count": 0.0,
            "independently_prepared_parent_samples": 0.0,
            "independence_class": INDEPENDENCE_CLASS,
        }
        for observable in OBSERVABLES:
            gate[f"{observable}_max_normalized_full_grid_same_time_error"] = full_grid_same[observable]
            gate[f"{observable}_max_normalized_support_same_time_error"] = support_same[observable]
            gate[f"{observable}_max_normalized_msd_matched_error"] = matched[observable]
            gate[f"{observable}_msd_matched_residual_pass"] = float(matched_pass[observable])
        gate.update(
            {
                "clock_shape_separation_supported_exploratory": float(
                    clock_shape_separation
                ),
                "high_temperature_canary_only": float(temperature == 0.58),
                "high_temperature_control_resolved": 0.0,
                "confirmatory_independent_parent_replication_required": 1.0,
                "next_required_action": (
                    "independent_parents_replicate_stationarity_then_blind_event_clock"
                ),
                **{field: 0.0 for field in CLOSED_GATE_FIELDS},
            }
        )
        result.append(gate)
    if {
        _finite(row, "temperature") for row in quotient_rows
    } != set(temperatures):
        raise ValueError("quotient table contains an unexpected temperature")
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a nonempty CSV artifact."""

    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write a rectangular CSV table with canonical float formatting."""

    if not rows:
        raise ValueError("cannot write an empty quotient table")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("quotient output rows must have an identical schema")
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
    fill: str = "#17212b",
) -> None:
    parts.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{html.escape(text)}</text>'
    )


def write_transport_clock_shape_svg(
    path: Path,
    quotient_rows: Sequence[dict[str, object]],
    gates: Sequence[dict[str, object]],
) -> None:
    """Write a deterministic two-panel normalized-residual figure."""

    if {float(gate["temperature"]) for gate in gates} != {0.45, 0.58}:
        raise ValueError("SVG requires exact T045/T058 gates")
    width = 1120
    height = 650
    panel_top = 105.0
    panel_height = 350.0
    panel_width = 480.0
    panel_lefts = (70.0, 570.0)
    observables = ("ngp", "fs_k2", "fs_k4", "fs_k7p25")
    labels = ("NGP", "Fs(k=2)", "Fs(k=4)", "Fs(k=7.25)")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1120" height="650" fill="#ffffff"/>',
    ]
    _svg_text(parts, 70, 43, "Transport clock / shape quotient", size=25, weight=700)
    _svg_text(
        parts,
        70,
        70,
        "same heldout rows before and after calibration-MSD matching; no extrapolation",
        size=14,
        fill="#46515c",
    )
    for panel_index, temperature in enumerate((0.45, 0.58)):
        gate = next(row for row in gates if float(row["temperature"]) == temperature)
        left = panel_lefts[panel_index]
        values = [
            float(gate[f"{observable}_max_normalized_support_same_time_error"])
            for observable in observables
        ] + [
            float(gate[f"{observable}_max_normalized_msd_matched_error"])
            for observable in observables
        ]
        y_max = max(1.4, math.ceil(max(values) * 5.0) / 5.0 + 0.2)
        parts.append(
            f'<rect x="{left:.2f}" y="{panel_top:.2f}" width="{panel_width:.2f}" '
            f'height="{panel_height:.2f}" fill="#fbfcfd" stroke="#c7ced5"/>'
        )
        title = "T=0.45 exploratory decomposition" if temperature == 0.45 else "T=0.58 canary only"
        _svg_text(parts, left + 16, panel_top + 28, title, size=16, weight=700)
        plot_left = left + 55.0
        plot_right = left + panel_width - 20.0
        plot_top = panel_top + 50.0
        plot_bottom = panel_top + panel_height - 55.0
        plot_height = plot_bottom - plot_top
        for tick in range(int(math.floor(y_max)) + 1):
            y = plot_bottom - tick / y_max * plot_height
            parts.append(
                f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" y2="{y:.2f}" '
                f'stroke="#e2e6ea" stroke-width="1"/>'
            )
            _svg_text(parts, plot_left - 8, y + 5, str(tick), size=12, anchor="end", fill="#5b6570")
        tolerance_y = plot_bottom - 1.0 / y_max * plot_height
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{tolerance_y:.2f}" x2="{plot_right:.2f}" '
            f'y2="{tolerance_y:.2f}" stroke="#9b3a32" stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        _svg_text(
            parts,
            plot_right - 2,
            plot_top + 12,
            "frozen tolerance = 1",
            size=11,
            anchor="end",
            fill="#9b3a32",
        )
        group_width = (plot_right - plot_left) / len(observables)
        for index, (observable, label) in enumerate(zip(observables, labels)):
            center = plot_left + (index + 0.5) * group_width
            same = float(gate[f"{observable}_max_normalized_support_same_time_error"])
            matched = float(gate[f"{observable}_max_normalized_msd_matched_error"])
            for offset, value, color in (
                (-13.0, same, "#7d8791"),
                (13.0, matched, "#2f7d68" if matched <= 1.0 else "#b34a42"),
            ):
                bar_height = min(value, y_max) / y_max * plot_height
                parts.append(
                    f'<rect x="{center + offset - 10:.2f}" y="{plot_bottom - bar_height:.2f}" '
                    f'width="20" height="{bar_height:.2f}" fill="{color}"/>'
                )
            _svg_text(parts, center, plot_bottom + 24, label, size=11, anchor="middle")
        local_rows = [row for row in quotient_rows if float(row["temperature"]) == temperature]
        signs = []
        for replicate in sorted({int(float(row["replicate"])) for row in local_rows}):
            sign = int(
                float(next(row for row in local_rows if int(float(row["replicate"])) == replicate)["replicate_clock_drift_sign"])
            )
            signs.append("+" if sign > 0 else "-" if sign < 0 else "mixed")
        _svg_text(
            parts,
            left + 16,
            panel_top + panel_height - 18,
            "replicate clock-drift signs: " + ", ".join(signs),
            size=12,
            fill="#46515c",
        )
    parts.extend(
        [
            '<rect x="70" y="490" width="980" height="125" fill="#f4f6f7" stroke="#c7ced5"/>',
        ]
    )
    _svg_text(parts, 90, 522, "T=0.45: low/intermediate-k scattering transfers; NGP and cage-scale shape do not.", size=15, weight=700)
    _svg_text(parts, 90, 552, "clock-only closure rejected; residual mechanism remains unresolved", size=14, fill="#9b3a32")
    _svg_text(parts, 90, 580, "heldout MSD is a diagnostic input, not a blind prediction", size=14)
    _svg_text(parts, 90, 604, "T=0.58 canary only; no cooling, spatial, microscopic, or thermodynamic claim", size=13, fill="#46515c")
    _svg_text(parts, 850, 92, "gray: same time", size=12, fill="#5b6570")
    _svg_text(parts, 960, 92, "color: MSD matched", size=12, fill="#2f7d68")
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the frozen transport-clock / shape quotient."
    )
    parser.add_argument("--low-rows", type=Path, required=True)
    parser.add_argument("--high-rows", type=Path, required=True)
    parser.add_argument("--low-stationarity", type=Path, required=True)
    parser.add_argument("--high-stationarity", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = manifests_from_provenance(read_rows(args.provenance))
    source_tables = {
        0.45: read_rows(args.low_rows),
        0.58: read_rows(args.high_rows),
    }
    stationarity = {
        0.45: read_rows(args.low_stationarity),
        0.58: read_rows(args.high_stationarity),
    }
    quotient_rows: list[dict[str, object]] = []
    for temperature in (0.45, 0.58):
        protocol = FROZEN_PROTOCOLS[temperature]
        quotient_rows.extend(
            compute_transport_clock_shape_rows(
                source_tables[temperature],
                manifests[temperature],
                temperature=temperature,
                full_length=int(protocol["full_length"]),
                expected_lags=protocol["lags"],
            )
        )
    gates = classify_transport_clock_shape_gate(
        quotient_rows,
        stationarity,
        manifests,
    )
    write_rows(args.output_rows, quotient_rows)
    write_rows(args.output_gate, gates)
    write_transport_clock_shape_svg(args.output_svg, quotient_rows, gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
