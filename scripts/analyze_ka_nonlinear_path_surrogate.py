#!/usr/bin/env python3
"""Test constrained spectral surrogates against held-out glassy dynamics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import math
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    cumulative_block_observables,
    load_lammps_custom_trajectory,
    phase_randomized_cross_spectrum,
    radial_multivariate_surrogate,
)


def surrogate_seed(base_seed: int, *, replicate: int, realization: int) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (base_seed, replicate, realization)
    ):
        raise ValueError("surrogate seed indices must be nonnegative integers")
    return int(
        (base_seed + 1_000_003 * replicate + 97_409 * realization) % (2**63 - 1)
    )


def _standard_error(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 1 or np.any(~np.isfinite(array)):
        raise ValueError("standard-error values must be a nonempty finite vector")
    return float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0


def _prediction_row(
    *,
    model: str,
    replicate: int,
    realization: int,
    temperature: float,
    lag: int,
    block_size: int,
    prediction: dict[str, float],
    observed: dict[str, object],
    fs_keys: Sequence[str],
    surrogate_realizations: int,
    iteration_count: int,
    base_seed: int,
) -> dict[str, object]:
    observed_msd = float(observed["observed_msd"])
    observed_ngp = float(observed["observed_ngp"])
    if observed_msd <= 0.0:
        raise ValueError("observed MSD must be positive")
    row: dict[str, object] = {
        "model": model,
        "replicate": float(replicate),
        "realization": float(realization),
        "temperature": float(temperature),
        "lag": float(lag),
        "block_size": float(block_size),
        "block_count": float(lag // block_size),
        "particle_window_count": float(prediction["particle_window_count"]),
        "predicted_msd": float(prediction["msd"]),
        "observed_msd": observed_msd,
        "msd_relative_error": abs(float(prediction["msd"]) / observed_msd - 1.0),
        "predicted_ngp": float(prediction["ngp"]),
        "observed_ngp": observed_ngp,
        "ngp_absolute_error": abs(float(prediction["ngp"]) - observed_ngp),
        "surrogate_realizations": float(surrogate_realizations),
        "surrogate_iteration_count": float(
            iteration_count if model == "radial_multivariate_surrogate" else 0
        ),
        "surrogate_base_seed": float(base_seed),
        "heldout_path_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "calibration_path_distribution_used": 1.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for fs_key in fs_keys:
        suffix = fs_key.removeprefix("observed_")
        kernel_key = suffix.replace("fs_", "characteristic_")
        predicted = float(prediction[kernel_key])
        observed_value = float(observed[fs_key])
        row[f"predicted_{suffix}"] = predicted
        row[fs_key] = observed_value
        row[f"absolute_error_{suffix}"] = abs(predicted - observed_value)
    return row


def _one_block_quality(
    reference: np.ndarray,
    surrogate: np.ndarray,
    *,
    wave_numbers: np.ndarray,
) -> dict[str, float]:
    reference_observables = cumulative_block_observables(
        reference,
        block_count=1,
        wave_numbers=wave_numbers,
    )
    surrogate_observables = cumulative_block_observables(
        surrogate,
        block_count=1,
        wave_numbers=wave_numbers,
    )
    fs_keys = [key for key in reference_observables if key.startswith("characteristic_k")]
    return {
        "one_block_msd_relative_error": abs(
            surrogate_observables["msd"] / reference_observables["msd"] - 1.0
        ),
        "one_block_ngp_absolute_error": abs(
            surrogate_observables["ngp"] - reference_observables["ngp"]
        ),
        "one_block_fs_maximum_absolute_error": max(
            abs(surrogate_observables[key] - reference_observables[key])
            for key in fs_keys
        ),
    }


def analyze_replicate_block_paths(
    block_displacements: np.ndarray,
    heldout: dict[int, dict[str, object]],
    *,
    replicate: int,
    temperature: float,
    block_size: int,
    wave_numbers: np.ndarray,
    fs_keys: Sequence[str],
    surrogate_realizations: int,
    iteration_count: int,
    base_seed: int,
) -> dict[str, list[dict[str, object]]]:
    blocks = np.asarray(block_displacements, dtype=float)
    if (
        blocks.ndim != 3
        or blocks.shape[1] < 2
        or np.any(~np.isfinite(blocks))
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
        or isinstance(surrogate_realizations, bool)
        or not isinstance(surrogate_realizations, int)
        or surrogate_realizations < 1
        or isinstance(iteration_count, bool)
        or not isinstance(iteration_count, int)
        or iteration_count < 1
        or not heldout
    ):
        raise ValueError("valid blocks, held-out rows, and positive controls are required")
    local_lags = sorted(heldout)
    if any(lag % block_size or lag // block_size > blocks.shape[1] for lag in local_lags):
        raise ValueError("held-out lags must fit the calibration block path")

    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for lag in local_lags:
        prediction = cumulative_block_observables(
            blocks,
            block_count=lag // block_size,
            wave_numbers=wave_numbers,
        )
        rows.append(
            _prediction_row(
                model="contiguous_empirical_path",
                replicate=replicate,
                realization=-1,
                temperature=temperature,
                lag=lag,
                block_size=block_size,
                prediction=prediction,
                observed=heldout[lag],
                fs_keys=fs_keys,
                surrogate_realizations=surrogate_realizations,
                iteration_count=iteration_count,
                base_seed=base_seed,
            )
        )

    for realization in range(surrogate_realizations):
        seed = surrogate_seed(
            base_seed,
            replicate=replicate,
            realization=realization,
        )
        phase = phase_randomized_cross_spectrum(blocks, np.random.default_rng(seed))
        radial_result = radial_multivariate_surrogate(
            blocks,
            np.random.default_rng(seed),
            iteration_count=iteration_count,
        )
        radial = np.asarray(radial_result["displacements"], dtype=float)
        quality = _one_block_quality(
            blocks,
            radial,
            wave_numbers=wave_numbers,
        )
        quality_rows.append(
            {
                "model": "radial_multivariate_surrogate",
                "replicate": float(replicate),
                "realization": float(realization),
                "temperature": float(temperature),
                "block_size": float(block_size),
                "surrogate_seed": float(seed),
                "surrogate_iteration_count": float(iteration_count),
                "radial_distribution_maximum_absolute_error": float(
                    radial_result["radial_distribution_maximum_absolute_error"]
                ),
                "cross_spectral_matrix_nrmse": float(
                    radial_result["cross_spectral_matrix_nrmse"]
                ),
                **quality,
            }
        )
        for model, surrogate in (
            ("phase_randomized_cross_spectrum", phase),
            ("radial_multivariate_surrogate", radial),
        ):
            for lag in local_lags:
                prediction = cumulative_block_observables(
                    surrogate,
                    block_count=lag // block_size,
                    wave_numbers=wave_numbers,
                )
                rows.append(
                    _prediction_row(
                        model=model,
                        replicate=replicate,
                        realization=realization,
                        temperature=temperature,
                        lag=lag,
                        block_size=block_size,
                        prediction=prediction,
                        observed=heldout[lag],
                        fs_keys=fs_keys,
                        surrogate_realizations=surrogate_realizations,
                        iteration_count=iteration_count,
                        base_seed=base_seed,
                    )
                )

    half_block_count = blocks.shape[1] // 2
    early = blocks[:, :half_block_count]
    late = blocks[:, -half_block_count:]
    stationarity_detail_rows: list[dict[str, object]] = []
    for lag in local_lags:
        block_count = lag // block_size
        if block_count > half_block_count:
            continue
        early_observables = cumulative_block_observables(
            early,
            block_count=block_count,
            wave_numbers=wave_numbers,
        )
        late_observables = cumulative_block_observables(
            late,
            block_count=block_count,
            wave_numbers=wave_numbers,
        )
        row: dict[str, object] = {
            "replicate": float(replicate),
            "temperature": float(temperature),
            "lag": float(lag),
            "block_size": float(block_size),
            "early_msd": early_observables["msd"],
            "late_msd": late_observables["msd"],
            "observed_msd": float(heldout[lag]["observed_msd"]),
            "early_ngp": early_observables["ngp"],
            "late_ngp": late_observables["ngp"],
            "observed_ngp": float(heldout[lag]["observed_ngp"]),
        }
        for fs_key in fs_keys:
            suffix = fs_key.removeprefix("observed_")
            kernel_key = suffix.replace("fs_", "characteristic_")
            row[f"early_{suffix}"] = early_observables[kernel_key]
            row[f"late_{suffix}"] = late_observables[kernel_key]
            row[fs_key] = float(heldout[lag][fs_key])
        stationarity_detail_rows.append(row)
    if not stationarity_detail_rows:
        raise ValueError("no held-out lag fits the stationarity half paths")
    return {
        "rows": rows,
        "quality_rows": quality_rows,
        "stationarity_detail_rows": stationarity_detail_rows,
    }


def summarize_surrogate_realizations(
    rows: Sequence[dict[str, object]],
    *,
    fs_keys: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows or not fs_keys:
        raise ValueError("surrogate rows and scattering keys must not be empty")
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
            if str(row["model"]) == model
            and int(float(row["replicate"])) == replicate
            and float(row["lag"]) == lag
        ]
        observed_msd_values = {float(row["observed_msd"]) for row in selected}
        observed_ngp_values = {float(row["observed_ngp"]) for row in selected}
        if len(observed_msd_values) != 1 or len(observed_ngp_values) != 1:
            raise ValueError("observed replicate values must agree across realizations")
        predicted_msd_values = [float(row["predicted_msd"]) for row in selected]
        predicted_ngp_values = [float(row["predicted_ngp"]) for row in selected]
        observed_msd = observed_msd_values.pop()
        observed_ngp = observed_ngp_values.pop()
        if observed_msd <= 0.0:
            raise ValueError("observed MSD must be positive")
        result: dict[str, object] = {
            "model": model,
            "replicate": float(replicate),
            "lag": lag,
            "realization_count": float(len(selected)),
            "predicted_msd": float(np.mean(predicted_msd_values)),
            "observed_msd": observed_msd,
            "predicted_msd_mc_se": _standard_error(predicted_msd_values),
            "predicted_ngp": float(np.mean(predicted_ngp_values)),
            "observed_ngp": observed_ngp,
            "predicted_ngp_mc_se": _standard_error(predicted_ngp_values),
        }
        result["msd_relative_error"] = abs(
            float(result["predicted_msd"]) / observed_msd - 1.0
        )
        result["ngp_absolute_error"] = abs(
            float(result["predicted_ngp"]) - observed_ngp
        )
        for fs_key in fs_keys:
            suffix = fs_key.removeprefix("observed_")
            predicted_key = f"predicted_{suffix}"
            observed_values = {float(row[fs_key]) for row in selected}
            if len(observed_values) != 1:
                raise ValueError("observed scattering values must agree across realizations")
            predicted_values = [float(row[predicted_key]) for row in selected]
            observed_value = observed_values.pop()
            result[predicted_key] = float(np.mean(predicted_values))
            result[fs_key] = observed_value
            result[f"absolute_error_{suffix}"] = abs(
                float(result[predicted_key]) - observed_value
            )
            result[f"predicted_{suffix}_mc_se"] = _standard_error(predicted_values)
        replicate_rows.append(result)

    summary_rows: list[dict[str, object]] = []
    summary_groups = sorted(
        {(str(row["model"]), float(row["lag"])) for row in replicate_rows}
    )
    for model, lag in summary_groups:
        selected = [
            row
            for row in replicate_rows
            if str(row["model"]) == model and float(row["lag"]) == lag
        ]
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
            )
            / replicate_count
            / observed_msd,
            "predicted_ngp": predicted_ngp,
            "observed_ngp": observed_ngp,
            "ensemble_ngp_absolute_error": abs(predicted_ngp - observed_ngp),
            "ensemble_ngp_mc_se": math.sqrt(
                sum(float(row["predicted_ngp_mc_se"]) ** 2 for row in selected)
            )
            / replicate_count,
        }
        for fs_key in fs_keys:
            suffix = fs_key.removeprefix("observed_")
            predicted_key = f"predicted_{suffix}"
            predicted = float(np.mean([float(row[predicted_key]) for row in selected]))
            observed_value = float(np.mean([float(row[fs_key]) for row in selected]))
            summary[predicted_key] = predicted
            summary[fs_key] = observed_value
            summary[f"ensemble_absolute_error_{suffix}"] = abs(
                predicted - observed_value
            )
            summary[f"ensemble_{suffix}_mc_se"] = math.sqrt(
                sum(float(row[f"predicted_{suffix}_mc_se"]) ** 2 for row in selected)
            ) / replicate_count
        summary_rows.append(summary)
    return replicate_rows, summary_rows


def stationarity_comparisons(
    rows: Sequence[dict[str, object]],
    *,
    fs_keys: Sequence[str],
) -> list[dict[str, object]]:
    if not rows or not fs_keys:
        raise ValueError("stationarity rows and scattering keys must not be empty")
    comparisons = (
        ("early_late", "early", "late"),
        ("early_heldout", "early", "observed"),
        ("late_heldout", "late", "observed"),
    )
    result: list[dict[str, object]] = []
    lags = sorted({float(row["lag"]) for row in rows})
    for comparison, predicted_prefix, reference_prefix in comparisons:
        msd_errors: list[float] = []
        ngp_errors: list[float] = []
        fs_errors: list[float] = []
        for lag in lags:
            selected = [row for row in rows if float(row["lag"]) == lag]
            predicted_msd = float(
                np.mean([float(row[f"{predicted_prefix}_msd"]) for row in selected])
            )
            reference_msd = float(
                np.mean([float(row[f"{reference_prefix}_msd"]) for row in selected])
            )
            if reference_msd <= 0.0:
                raise ValueError("stationarity reference MSD must be positive")
            msd_errors.append(abs(predicted_msd / reference_msd - 1.0))
            predicted_ngp = float(
                np.mean([float(row[f"{predicted_prefix}_ngp"]) for row in selected])
            )
            reference_ngp = float(
                np.mean([float(row[f"{reference_prefix}_ngp"]) for row in selected])
            )
            ngp_errors.append(abs(predicted_ngp - reference_ngp))
            for fs_key in fs_keys:
                suffix = fs_key.removeprefix("observed_")
                predicted_fs = float(
                    np.mean(
                        [float(row[f"{predicted_prefix}_{suffix}"]) for row in selected]
                    )
                )
                reference_fs = float(
                    np.mean(
                        [float(row[f"{reference_prefix}_{suffix}"]) for row in selected]
                    )
                )
                fs_errors.append(abs(predicted_fs - reference_fs))
        maximum_msd = max(msd_errors)
        maximum_ngp = max(ngp_errors)
        maximum_fs = max(fs_errors)
        result.append(
            {
                "comparison": comparison,
                "lag_count": float(len(lags)),
                "maximum_msd_relative_error": maximum_msd,
                "maximum_ngp_absolute_error": maximum_ngp,
                "maximum_fs_absolute_error": maximum_fs,
                "msd_relative_error_tolerance": 0.10,
                "ngp_absolute_error_tolerance": 0.30,
                "fs_absolute_error_tolerance": 0.03,
                "curve_transfer_pass": float(
                    maximum_msd <= 0.10
                    and maximum_ngp <= 0.30
                    and maximum_fs <= 0.03
                ),
            }
        )
    return result


def replicate_higher_order_scores(
    replicate_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    if not replicate_rows:
        raise ValueError("replicate rows must not be empty")
    models = {str(row["model"]) for row in replicate_rows}
    required_models = {
        "contiguous_empirical_path",
        "radial_multivariate_surrogate",
    }
    if not required_models.issubset(models):
        raise ValueError("paired score rows require contiguous and radial models")
    replicates = sorted({int(float(row["replicate"])) for row in replicate_rows})
    result: list[dict[str, object]] = []
    for replicate in replicates:
        scores: dict[str, float] = {}
        for model in sorted(required_models):
            selected = [
                row
                for row in replicate_rows
                if str(row["model"]) == model
                and int(float(row["replicate"])) == replicate
            ]
            if not selected:
                raise ValueError(f"replicate {replicate} is missing model {model}")
            fs_error_names = sorted(
                {
                    name
                    for row in selected
                    for name in row
                    if name.startswith("absolute_error_fs_k")
                }
            )
            if not fs_error_names:
                raise ValueError("paired score rows contain no scattering errors")
            scores[model] = max(
                max(float(row["ngp_absolute_error"]) / 0.30 for row in selected),
                max(
                    float(row[name]) / 0.03
                    for row in selected
                    for name in fs_error_names
                ),
            )
        contiguous = scores["contiguous_empirical_path"]
        radial = scores["radial_multivariate_surrogate"]
        result.append(
            {
                "replicate": float(replicate),
                "contiguous_higher_order_score": contiguous,
                "radial_higher_order_score": radial,
                "radial_higher_order_failure": float(radial > 1.0),
                "paired_contiguous_better": float(contiguous < radial),
                "ngp_absolute_error_tolerance": 0.30,
                "fs_absolute_error_tolerance": 0.03,
                "heldout_path_used_in_prediction": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty nonlinear-path table")
    fieldnames = list(rows[0])
    extra = sorted({name for row in rows for name in row if name not in fieldnames})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames + extra,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def wave_number_from_observed_key(key: str) -> float:
    return float(key.removeprefix("observed_fs_k").replace("p", "."))


def _model_error_limits(
    summary_rows: Sequence[dict[str, object]],
    model: str,
) -> dict[str, float]:
    selected = [row for row in summary_rows if str(row["model"]) == model]
    if not selected:
        raise ValueError(f"summary rows are missing model {model}")
    fs_error_names = sorted(
        {
            name
            for row in selected
            for name in row
            if name.startswith("ensemble_absolute_error_fs_k")
        }
    )
    if not fs_error_names:
        raise ValueError(f"summary rows for {model} contain no scattering errors")
    fs_mc_names = sorted(
        {
            name
            for row in selected
            for name in row
            if name.startswith("ensemble_fs_k") and name.endswith("_mc_se")
        }
    )
    return {
        "msd": max(float(row["ensemble_msd_relative_error"]) for row in selected),
        "ngp": max(float(row["ensemble_ngp_absolute_error"]) for row in selected),
        "fs": max(float(row[name]) for row in selected for name in fs_error_names),
        "msd_mc": max(
            float(row.get("ensemble_msd_mc_relative_se", 0.0)) for row in selected
        ),
        "ngp_mc": max(float(row.get("ensemble_ngp_mc_se", 0.0)) for row in selected),
        "fs_mc": max(
            (float(row[name]) for row in selected for name in fs_mc_names),
            default=0.0,
        ),
    }


def classify_nonlinear_path_surrogate(
    summary_rows: Sequence[dict[str, object]],
    *,
    quality_rows: Sequence[dict[str, object]],
    replicate_scores: Sequence[dict[str, object]],
    stationarity_rows: Sequence[dict[str, object]],
    required_replicate_count: int,
) -> dict[str, object]:
    if (
        isinstance(required_replicate_count, bool)
        or not isinstance(required_replicate_count, int)
        or required_replicate_count < 1
    ):
        raise ValueError("required_replicate_count must be a positive integer")
    if not quality_rows or not replicate_scores or not stationarity_rows:
        raise ValueError("quality, replicate, and stationarity evidence must not be empty")
    contiguous = _model_error_limits(summary_rows, "contiguous_empirical_path")
    radial = _model_error_limits(summary_rows, "radial_multivariate_surrogate")
    numeric_quality = [
        float(row[name])
        for row in quality_rows
        for name in (
            "radial_distribution_maximum_absolute_error",
            "cross_spectral_matrix_nrmse",
            "one_block_msd_relative_error",
            "one_block_ngp_absolute_error",
            "one_block_fs_maximum_absolute_error",
        )
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in numeric_quality):
        raise ValueError("surrogate quality values must be finite and nonnegative")
    quality_pass = all(
        float(row["radial_distribution_maximum_absolute_error"]) <= 1e-12
        and float(row["cross_spectral_matrix_nrmse"]) <= 0.015
        and float(row["one_block_msd_relative_error"]) <= 0.01
        and float(row["one_block_ngp_absolute_error"]) <= 0.03
        and float(row["one_block_fs_maximum_absolute_error"]) <= 0.003
        for row in quality_rows
    )
    precision_pass = (
        radial["msd_mc"] <= 0.01
        and radial["ngp_mc"] <= 0.03
        and radial["fs_mc"] <= 0.003
    )
    stationarity_by_name = {
        str(row["comparison"]): bool(float(row["curve_transfer_pass"]))
        for row in stationarity_rows
    }
    required_stationarity = {"early_late", "early_heldout", "late_heldout"}
    stationarity_pass = required_stationarity.issubset(stationarity_by_name) and all(
        stationarity_by_name[name] for name in required_stationarity
    )
    contiguous_curve_pass = (
        contiguous["msd"] <= 0.10
        and contiguous["ngp"] <= 0.30
        and contiguous["fs"] <= 0.03
    )
    radial_msd_pass = radial["msd"] <= 0.10
    radial_higher_order_failure = radial["ngp"] > 0.30 or radial["fs"] > 0.03
    replicate_ids = {int(float(row["replicate"])) for row in replicate_scores}
    radial_failure_count = sum(
        float(row["radial_higher_order_score"]) > 1.0 for row in replicate_scores
    )
    contiguous_better_count = sum(
        float(row["contiguous_higher_order_score"])
        < float(row["radial_higher_order_score"])
        for row in replicate_scores
    )
    replicate_consensus = (
        len(replicate_ids) == required_replicate_count
        and len(replicate_scores) == required_replicate_count
        and radial_failure_count == required_replicate_count
        and contiguous_better_count == required_replicate_count
    )
    selected = (
        quality_pass
        and precision_pass
        and stationarity_pass
        and contiguous_curve_pass
        and radial_msd_pass
        and radial_higher_order_failure
        and replicate_consensus
    )
    return {
        "maximum_radial_distribution_absolute_error": max(
            float(row["radial_distribution_maximum_absolute_error"])
            for row in quality_rows
        ),
        "maximum_cross_spectral_matrix_nrmse": max(
            float(row["cross_spectral_matrix_nrmse"]) for row in quality_rows
        ),
        "surrogate_quality_pass": float(quality_pass),
        "maximum_ensemble_msd_mc_relative_se": radial["msd_mc"],
        "maximum_ensemble_ngp_mc_se": radial["ngp_mc"],
        "maximum_ensemble_fs_mc_se": radial["fs_mc"],
        "surrogate_precision_pass": float(precision_pass),
        "stationarity_control_pass": float(stationarity_pass),
        "contiguous_ensemble_curve_pass": float(contiguous_curve_pass),
        "radial_surrogate_msd_pass": float(radial_msd_pass),
        "radial_surrogate_higher_order_failure": float(radial_higher_order_failure),
        "surrogate_failure_replicate_count": float(radial_failure_count),
        "paired_contiguous_better_replicate_count": float(contiguous_better_count),
        "paired_replicate_count": float(len(replicate_ids)),
        "required_replicate_count": float(required_replicate_count),
        "replicate_consensus_pass": float(replicate_consensus),
        "nonlinear_single_particle_path_memory_required": float(selected),
        "one_block_radial_plus_two_point_spectrum_sufficient": 0.0,
        "linear_spectrum_null_rejected": float(selected),
        "calibration_nonstationarity_supported": 0.0,
        "next_minimal_model": (
            "finite_lifetime_reversible_cage_state" if selected else "unresolved"
        ),
        "heldout_path_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--heldout-factorization", type=Path, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--surrogate-realizations", type=int, default=8)
    parser.add_argument("--iteration-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=211003)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    controls = (
        args.calibration_time,
        args.block_size,
        args.surrogate_realizations,
        args.iteration_count,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in controls
    ) or args.seed < 0:
        raise ValueError("calibration, block, realization, iteration, and seed controls are invalid")
    if args.calibration_time // args.block_size < 2:
        raise ValueError("calibration window must contain at least two complete blocks")

    manifest = json.loads(
        (args.ensemble_directory / "ensemble_manifest.json").read_text()
    )
    replicate_specs = manifest.get("replicates", [])
    if not replicate_specs:
        raise ValueError("ensemble manifest contains no replicates")
    temperature = float(manifest["temperature"])
    heldout_rows = read_rows(args.heldout_factorization)
    fs_keys = sorted(key for key in heldout_rows[0] if key.startswith("observed_fs_k"))
    if not fs_keys:
        raise ValueError("held-out factorization contains no observed scattering columns")
    wave_numbers = np.asarray(
        [wave_number_from_observed_key(key) for key in fs_keys],
        dtype=float,
    )
    heldout_by_replicate: dict[int, dict[int, dict[str, object]]] = {}
    for row in heldout_rows:
        replicate = int(float(row["replicate"]))
        lag = int(float(row["lag"]))
        if lag % args.block_size != 0 or lag > args.calibration_time:
            continue
        local = heldout_by_replicate.setdefault(replicate, {})
        if lag in local:
            raise ValueError(f"duplicate held-out row for replicate {replicate}, lag {lag}")
        local[lag] = row

    output_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    stationarity_detail_rows: list[dict[str, object]] = []
    for replicate_spec in replicate_specs:
        replicate = int(replicate_spec["replicate"])
        local_heldout = heldout_by_replicate.get(replicate)
        if not local_heldout:
            raise ValueError(f"replicate {replicate} has no block-compatible held-out lag")
        directory = args.ensemble_directory / str(replicate_spec["directory"])
        trajectory = load_lammps_custom_trajectory(
            directory / "trajectory.lammpstrj",
            maximum_frame_count=args.calibration_time + 1,
        )
        if len(trajectory["unwrapped_positions"]) < args.calibration_time + 1:
            raise ValueError(f"replicate {replicate} trajectory is shorter than calibration time")
        positions = trajectory["unwrapped_positions"][
            :,
            trajectory["particle_types"] == 0,
        ]
        block_total = args.calibration_time // args.block_size
        starts = np.arange(block_total) * args.block_size
        blocks = np.transpose(
            positions[starts + args.block_size] - positions[starts],
            (1, 0, 2),
        ).astype(float)
        del positions, trajectory
        gc.collect()
        result = analyze_replicate_block_paths(
            blocks,
            local_heldout,
            replicate=replicate,
            temperature=temperature,
            block_size=args.block_size,
            wave_numbers=wave_numbers,
            fs_keys=fs_keys,
            surrogate_realizations=args.surrogate_realizations,
            iteration_count=args.iteration_count,
            base_seed=args.seed,
        )
        output_rows.extend(result["rows"])
        quality_rows.extend(result["quality_rows"])
        stationarity_detail_rows.extend(result["stationarity_detail_rows"])
        del blocks, result
        gc.collect()

    replicate_rows, summary_rows = summarize_surrogate_realizations(
        output_rows,
        fs_keys=fs_keys,
    )
    replicate_scores = replicate_higher_order_scores(replicate_rows)
    stationarity_rows = stationarity_comparisons(
        stationarity_detail_rows,
        fs_keys=fs_keys,
    )
    verdict = classify_nonlinear_path_surrogate(
        summary_rows,
        quality_rows=quality_rows,
        replicate_scores=replicate_scores,
        stationarity_rows=stationarity_rows,
        required_replicate_count=len(replicate_specs),
    )
    metadata = {
        "temperature": temperature,
        "calibration_time": float(args.calibration_time),
        "block_size": float(args.block_size),
        "independent_replicate_count": float(len(replicate_specs)),
        "surrogate_realizations_per_replicate": float(args.surrogate_realizations),
        "surrogate_iteration_count": float(args.iteration_count),
        "surrogate_base_seed": float(args.seed),
    }
    for row in summary_rows:
        row.update(
            {
                **metadata,
                "heldout_path_used_in_prediction": 0.0,
                "macro_fit_parameter_count": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    for rows in (replicate_scores, stationarity_rows):
        for row in rows:
            row.update(metadata)
            row["microdynamic_closure_claim_allowed"] = 0.0
            row["spatial_facilitation_claim_allowed"] = 0.0
            row["thermodynamic_claim_allowed"] = 0.0
    verdict.update(metadata)
    verdict["quality_realization_count"] = float(len(quality_rows))

    prefix = args.output_prefix
    write_rows(prefix.with_name(prefix.name + "_rows.csv"), output_rows)
    write_rows(prefix.with_name(prefix.name + "_quality.csv"), quality_rows)
    write_rows(prefix.with_name(prefix.name + "_summary.csv"), summary_rows)
    write_rows(
        prefix.with_name(prefix.name + "_replicate_scores.csv"),
        replicate_scores,
    )
    write_rows(
        prefix.with_name(prefix.name + "_stationarity.csv"),
        stationarity_rows,
    )
    write_rows(prefix.with_name(prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
