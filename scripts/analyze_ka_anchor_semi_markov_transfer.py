#!/usr/bin/env python3
"""Evaluate frozen anchor-aware semi-Markov models on held-out KA curves."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_anchor_semi_markov import (  # noqa: E402
    anchor_path_quality,
    extract_anchor_transition_kernel,
    simulate_anchor_semi_markov,
)
from ka_replicates import (  # noqa: E402
    compound_jump_cage_observables,
    cumulative_block_observables,
    extract_debye_waller_cage_jumps,
    load_lammps_custom_trajectory,
    position_fluctuation_values,
)


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
MODELS = (
    "anchor_aware_semi_markov",
    "state_schedule_without_anchor_geometry",
)
WAVE_NUMBERS = np.asarray([2.0, 4.0, 7.25])


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


def anchor_seed(base_seed: int, *, replicate: int, realization: int) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (base_seed, replicate, realization)
    ):
        raise ValueError("anchor seed indices must be nonnegative integers")
    return int((base_seed + 1_000_003 * replicate + 97_409 * realization) % (2**63 - 1))


def _standard_error(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or np.any(~np.isfinite(array)):
        raise ValueError("standard-error values must be a nonempty finite vector")
    return float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0


def _claim_flags() -> dict[str, float]:
    return {
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty anchor transfer table")
    fieldnames = list(rows[0])
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _index_factorization_rows(
    rows: Sequence[dict[str, str]],
    *,
    calibration_time: int,
    block_size: int,
) -> dict[tuple[int, int], dict[str, str]]:
    indexed: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        replicate = _as_exact_integer(row.get("replicate"), "replicate")
        lag = _as_exact_integer(row.get("lag"), "lag")
        if lag > calibration_time or lag % block_size:
            continue
        key = (replicate, lag)
        if key in indexed:
            raise ValueError(f"duplicate factorization row: replicate {replicate}, lag {lag}")
        indexed[key] = row
    if not indexed:
        raise ValueError("factorization table has no frozen block-compatible rows")
    return indexed


def _events_to_blocks(
    synthetic: dict[str, np.ndarray],
    *,
    particle_count: int,
    duration: int,
    block_size: int,
) -> np.ndarray:
    block_count = duration // block_size
    complete_duration = block_count * block_size
    blocks = np.zeros((particle_count, block_count, 3), dtype=float)
    particle = np.asarray(synthetic["particle"], dtype=int)
    time = np.asarray(synthetic["time"], dtype=int)
    vectors = np.asarray(synthetic["jump_vector"], dtype=float)
    selected = (time >= 1) & (time <= complete_duration)
    block = (time[selected] - 1) // block_size
    if np.any(particle[selected] < 0) or np.any(particle[selected] >= particle_count):
        raise ValueError("synthetic particle ids exceed the trajectory population")
    np.add.at(blocks, (particle[selected], block), vectors[selected])
    return blocks


def _factorized_prediction(
    event_observables: dict[str, float],
    residual: dict[str, str],
) -> dict[str, float]:
    prediction: dict[str, float] = {}
    first_fs_key = "characteristic_k2"
    base = compound_jump_cage_observables(
        mean_count=1.0,
        factorial_second_count=0.0,
        jump_msd=float(event_observables["msd"]),
        jump_fourth_moment=float(event_observables["fourth_moment"]),
        count_pgf=float(event_observables[first_fs_key]),
        cage_msd=float(residual["residual_msd"]),
        cage_ngp=float(residual["residual_ngp"]),
        cage_fs=float(residual["residual_fs_k2"]),
        dimension=3,
    )
    prediction["msd"] = float(base["factorized_msd"])
    prediction["ngp"] = float(base["factorized_ngp"])
    for wave_number in WAVE_NUMBERS:
        suffix = f"k{wave_number:g}".replace(".", "p")
        event_key = f"characteristic_{suffix}"
        prediction[f"fs_{suffix}"] = (
            float(event_observables[event_key]) * float(residual[f"residual_fs_{suffix}"])
        )
    return prediction


def _prediction_row(
    *,
    model: str,
    replicate: int,
    realization: int,
    temperature: float,
    lag: int,
    block_size: int,
    prediction: dict[str, float],
    observed: dict[str, str],
    base_seed: int,
) -> dict[str, object]:
    observed_msd = float(observed["observed_msd"])
    if observed_msd <= 0.0:
        raise ValueError("held-out observed MSD must be positive")
    row: dict[str, object] = {
        "model": model,
        "replicate": float(replicate),
        "realization": float(realization),
        "temperature": temperature,
        "lag": float(lag),
        "block_size": float(block_size),
        "predicted_msd": prediction["msd"],
        "observed_msd": observed_msd,
        "msd_relative_error": abs(prediction["msd"] / observed_msd - 1.0),
        "predicted_ngp": prediction["ngp"],
        "observed_ngp": float(observed["observed_ngp"]),
        "ngp_absolute_error": abs(prediction["ngp"] - float(observed["observed_ngp"])),
        "surrogate_base_seed": float(base_seed),
        "calibration_events_only": 1.0,
        "heldout_events_used_in_calibration": 0.0,
        "heldout_cage_residual_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        **_claim_flags(),
    }
    for wave_number in WAVE_NUMBERS:
        suffix = f"k{wave_number:g}".replace(".", "p")
        observed_value = float(observed[f"observed_fs_{suffix}"])
        row[f"predicted_fs_{suffix}"] = prediction[f"fs_{suffix}"]
        row[f"observed_fs_{suffix}"] = observed_value
        row[f"absolute_error_fs_{suffix}"] = abs(prediction[f"fs_{suffix}"] - observed_value)
    return row


def analyze_replicate_anchor_paths(
    kernel: dict[str, object],
    *,
    particle_count: int,
    residual_by_lag: dict[int, dict[str, str]],
    heldout_by_lag: dict[int, dict[str, str]],
    replicate: int,
    temperature: float,
    calibration_time: int,
    block_size: int,
    surrogate_realizations: int,
    base_seed: int,
) -> dict[str, list[dict[str, object]]]:
    lags = sorted(set(residual_by_lag) & set(heldout_by_lag))
    if not lags:
        raise ValueError("calibration residual and held-out target have no common lags")
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for realization in range(surrogate_realizations):
        seed = anchor_seed(base_seed, replicate=replicate, realization=realization)
        for model in MODELS:
            synthetic = simulate_anchor_semi_markov(
                kernel,
                np.random.default_rng(seed),
                duration=calibration_time,
                maximum_lag=0,
                model=model,
            )
            quality_rows.append(
                {
                    "model": model,
                    "replicate": float(replicate),
                    "realization": float(realization),
                    "temperature": temperature,
                    "calibration_time": float(calibration_time),
                    "block_size": float(block_size),
                    "radial_bin_count": float(RADIAL_BIN_COUNT),
                    "required_realizations_per_replicate": float(surrogate_realizations),
                    "surrogate_base_seed": float(base_seed),
                    "surrogate_seed": float(seed),
                    **anchor_path_quality(kernel, synthetic, model=model),
                    "calibration_events_only": 1.0,
                    "heldout_events_used_in_calibration": 0.0,
                    "heldout_cage_residual_used_in_prediction": 0.0,
                    "macro_fit_parameter_count": 0.0,
                    **_claim_flags(),
                }
            )
            blocks = _events_to_blocks(
                synthetic,
                particle_count=particle_count,
                duration=calibration_time,
                block_size=block_size,
            )
            for lag in lags:
                event_observables = cumulative_block_observables(
                    blocks,
                    block_count=lag // block_size,
                    wave_numbers=WAVE_NUMBERS,
                )
                prediction = _factorized_prediction(event_observables, residual_by_lag[lag])
                rows.append(
                    _prediction_row(
                        model=model,
                        replicate=replicate,
                        realization=realization,
                        temperature=temperature,
                        lag=lag,
                        block_size=block_size,
                        prediction=prediction,
                        observed=heldout_by_lag[lag],
                        base_seed=base_seed,
                    )
                )
    return {"rows": rows, "quality_rows": quality_rows}


def summarize_anchor_realizations(
    rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows:
        raise ValueError("anchor realization rows must not be empty")
    replicate_rows: list[dict[str, object]] = []
    groups = sorted(
        {
            (str(row["model"]), int(float(row["replicate"])), float(row["lag"]))
            for row in rows
        }
    )
    for model, replicate, lag in groups:
        selected = [
            row
            for row in rows
            if row["model"] == model
            and int(float(row["replicate"])) == replicate
            and float(row["lag"]) == lag
        ]
        observed_msd_values = {float(row["observed_msd"]) for row in selected}
        observed_ngp_values = {float(row["observed_ngp"]) for row in selected}
        if len(observed_msd_values) != 1 or len(observed_ngp_values) != 1:
            raise ValueError("held-out values must agree across realizations")
        observed_msd = observed_msd_values.pop()
        result: dict[str, object] = {
            "model": model,
            "replicate": float(replicate),
            "lag": lag,
            "realization_count": float(len(selected)),
            "predicted_msd": float(np.mean([float(row["predicted_msd"]) for row in selected])),
            "observed_msd": observed_msd,
            "predicted_msd_mc_se": _standard_error([float(row["predicted_msd"]) for row in selected]),
            "predicted_ngp": float(np.mean([float(row["predicted_ngp"]) for row in selected])),
            "observed_ngp": observed_ngp_values.pop(),
            "predicted_ngp_mc_se": _standard_error([float(row["predicted_ngp"]) for row in selected]),
            **_claim_flags(),
        }
        result["msd_relative_error"] = abs(float(result["predicted_msd"]) / observed_msd - 1.0)
        result["ngp_absolute_error"] = abs(float(result["predicted_ngp"]) - float(result["observed_ngp"]))
        for wave_number in WAVE_NUMBERS:
            suffix = f"k{wave_number:g}".replace(".", "p")
            predicted_key = f"predicted_fs_{suffix}"
            observed_key = f"observed_fs_{suffix}"
            observed_values = {float(row[observed_key]) for row in selected}
            if len(observed_values) != 1:
                raise ValueError("held-out scattering values must agree across realizations")
            values = [float(row[predicted_key]) for row in selected]
            result[predicted_key] = float(np.mean(values))
            result[observed_key] = observed_values.pop()
            result[f"absolute_error_fs_{suffix}"] = abs(
                float(result[predicted_key]) - float(result[observed_key])
            )
            result[f"predicted_fs_{suffix}_mc_se"] = _standard_error(values)
        replicate_rows.append(result)

    summary_rows: list[dict[str, object]] = []
    for model, lag in sorted({(str(row["model"]), float(row["lag"])) for row in replicate_rows}):
        selected = [row for row in replicate_rows if row["model"] == model and float(row["lag"]) == lag]
        replicate_count = len(selected)
        predicted_msd = float(np.mean([float(row["predicted_msd"]) for row in selected]))
        observed_msd = float(np.mean([float(row["observed_msd"]) for row in selected]))
        predicted_ngp = float(np.mean([float(row["predicted_ngp"]) for row in selected]))
        observed_ngp = float(np.mean([float(row["observed_ngp"]) for row in selected]))
        summary: dict[str, object] = {
            "model": model,
            "lag": lag,
            "independent_replicate_count": float(replicate_count),
            "predicted_msd": predicted_msd,
            "observed_msd": observed_msd,
            "ensemble_msd_relative_error": abs(predicted_msd / observed_msd - 1.0),
            "ensemble_msd_mc_relative_se": math.sqrt(
                sum(float(row["predicted_msd_mc_se"]) ** 2 for row in selected)
            ) / replicate_count / observed_msd,
            "predicted_ngp": predicted_ngp,
            "observed_ngp": observed_ngp,
            "ensemble_ngp_absolute_error": abs(predicted_ngp - observed_ngp),
            "ensemble_ngp_mc_se": math.sqrt(
                sum(float(row["predicted_ngp_mc_se"]) ** 2 for row in selected)
            ) / replicate_count,
            "replicate_first_aggregation": 1.0,
            **_claim_flags(),
        }
        for wave_number in WAVE_NUMBERS:
            suffix = f"k{wave_number:g}".replace(".", "p")
            predicted_key = f"predicted_fs_{suffix}"
            observed_key = f"observed_fs_{suffix}"
            predicted = float(np.mean([float(row[predicted_key]) for row in selected]))
            observed = float(np.mean([float(row[observed_key]) for row in selected]))
            summary[predicted_key] = predicted
            summary[observed_key] = observed
            summary[f"ensemble_absolute_error_fs_{suffix}"] = abs(predicted - observed)
            summary[f"ensemble_fs_{suffix}_mc_se"] = math.sqrt(
                sum(float(row[f"predicted_fs_{suffix}_mc_se"]) ** 2 for row in selected)
            ) / replicate_count
        summary_rows.append(summary)
    return replicate_rows, summary_rows


def _transition_summary(
    kernel: dict[str, object],
    *,
    replicate: int,
    temperature: float,
    calibration_time: int,
    debye_waller_factor: float,
) -> dict[str, object]:
    current = np.asarray(kernel["current_state"], dtype=int)
    following = np.asarray(kernel["next_state"], dtype=int)
    wait = np.asarray(kernel["holding_time"], dtype=float)
    return {
        "replicate": float(replicate),
        "temperature": temperature,
        "calibration_time": float(calibration_time),
        "debye_waller_factor": debye_waller_factor,
        "transition_count": float(len(current)),
        "active_particle_count": float(len(kernel["active_particle_ids"])),
        "non_propagating_particle_count": float(len(kernel["non_propagating_particle_ids"])),
        "return_fraction": float(np.mean(following)),
        "return_given_return": float(np.mean(following[current == 1])),
        "return_given_escape": float(np.mean(following[current == 0])),
        "return_mean_holding_time": float(np.mean(wait[following == 1])),
        "escape_mean_holding_time": float(np.mean(wait[following == 0])),
        "calibration_events_only": 1.0,
        "heldout_events_used_in_calibration": 0.0,
        **_claim_flags(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--calibration-residual", type=Path, required=True)
    parser.add_argument("--heldout-factorization", type=Path, required=True)
    parser.add_argument("--debye-waller-table", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--radial-bin-count", type=int, default=RADIAL_BIN_COUNT)
    parser.add_argument("--surrogate-realizations", type=int, default=REQUIRED_REALIZATIONS)
    parser.add_argument("--fluctuation-half-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=84031)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    specifications = manifest.get("replicates", [])
    temperature = float(manifest["temperature"])
    validate_anchor_protocol(
        temperature=temperature,
        calibration_time=args.calibration_time,
        block_size=args.block_size,
        radial_bin_count=args.radial_bin_count,
        surrogate_realizations=args.surrogate_realizations,
        replicate_count=len(specifications),
    )
    if args.seed < 0 or args.fluctuation_half_window < 1:
        raise ValueError("seed and fluctuation window controls are invalid")

    residual_index = _index_factorization_rows(
        _read_rows(args.calibration_residual),
        calibration_time=args.calibration_time,
        block_size=args.block_size,
    )
    heldout_index = _index_factorization_rows(
        _read_rows(args.heldout_factorization),
        calibration_time=args.calibration_time,
        block_size=args.block_size,
    )
    dw_rows = _read_rows(args.debye_waller_table)
    dw_by_replicate = {
        _as_exact_integer(row.get("replicate"), "replicate"): float(row["debye_waller_factor"])
        for row in dw_rows
    }

    transition_rows: list[dict[str, object]] = []
    realization_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for specification in specifications:
        replicate = int(specification["replicate"])
        directory = args.ensemble_directory / str(specification["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate} is not marked COMPLETE")
        if replicate not in dw_by_replicate:
            raise ValueError(f"replicate {replicate} lacks a calibration Debye-Waller factor")
        trajectory = load_lammps_custom_trajectory(
            directory / "trajectory.lammpstrj",
            maximum_frame_count=args.calibration_time + 1,
        )
        if len(trajectory["unwrapped_positions"]) < args.calibration_time + 1:
            raise ValueError(f"replicate {replicate} is shorter than the calibration window")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        times, fluctuation = position_fluctuation_values(
            positions,
            half_window=args.fluctuation_half_window,
        )
        events = extract_debye_waller_cage_jumps(
            positions,
            debye_waller_factor=dw_by_replicate[replicate],
            half_window=args.fluctuation_half_window,
            activity_times=times,
            activity_values=fluctuation,
        )
        kernel = extract_anchor_transition_kernel(
            events,
            debye_waller_factor=dw_by_replicate[replicate],
            radial_bin_count=args.radial_bin_count,
        )
        transition_rows.append(
            _transition_summary(
                kernel,
                replicate=replicate,
                temperature=temperature,
                calibration_time=args.calibration_time,
                debye_waller_factor=dw_by_replicate[replicate],
            )
        )
        residual = {lag: row for (candidate, lag), row in residual_index.items() if candidate == replicate}
        heldout = {lag: row for (candidate, lag), row in heldout_index.items() if candidate == replicate}
        result = analyze_replicate_anchor_paths(
            kernel,
            particle_count=positions.shape[1],
            residual_by_lag=residual,
            heldout_by_lag=heldout,
            replicate=replicate,
            temperature=temperature,
            calibration_time=args.calibration_time,
            block_size=args.block_size,
            surrogate_realizations=args.surrogate_realizations,
            base_seed=args.seed,
        )
        realization_rows.extend(result["rows"])
        quality_rows.extend(result["quality_rows"])
        del trajectory, positions, times, fluctuation, events, kernel, result
        gc.collect()

    replicate_rows, summary_rows = summarize_anchor_realizations(realization_rows)
    verdict_rows = [
        classify_anchor_transfer(
            [row for row in quality_rows if row["model"] == model],
            [row for row in summary_rows if row["model"] == model],
            [row for row in replicate_rows if row["model"] == model],
            model=model,
            expected_replicates=len(specifications),
        )
        for model in MODELS
    ]
    metadata = {
        "temperature": temperature,
        "calibration_time": float(args.calibration_time),
        "block_size": float(args.block_size),
        "radial_bin_count": float(args.radial_bin_count),
        "required_realizations_per_replicate": float(args.surrogate_realizations),
        "surrogate_base_seed": float(args.seed),
        "calibration_residual_source": str(args.calibration_residual),
        "heldout_target_source": str(args.heldout_factorization),
        "debye_waller_source": str(args.debye_waller_table),
    }
    for table in (replicate_rows, summary_rows, verdict_rows):
        for row in table:
            row.update(metadata)

    prefix = args.output_prefix
    write_rows(prefix.with_name(prefix.name + "_transitions.csv"), transition_rows)
    write_rows(prefix.with_name(prefix.name + "_quality.csv"), quality_rows)
    write_rows(prefix.with_name(prefix.name + "_realizations.csv"), realization_rows)
    write_rows(prefix.with_name(prefix.name + "_rows.csv"), replicate_rows)
    write_rows(prefix.with_name(prefix.name + "_summary.csv"), summary_rows)
    write_rows(prefix.with_name(prefix.name + "_verdict.csv"), verdict_rows)


if __name__ == "__main__":
    main()
