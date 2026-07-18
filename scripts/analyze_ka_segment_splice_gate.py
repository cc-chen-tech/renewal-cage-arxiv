#!/usr/bin/env python3
"""Run and score paired segment-splice calibration-path nulls."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


MODELS = (
    "within_particle_segment_shuffle",
    "cross_particle_segment_splice",
)
STATIONARITY_COMPARISONS = {"early_late", "early_heldout", "late_heldout"}


def summarize_segment_rows(
    rows: Sequence[dict[str, object]],
    *,
    fs_keys: Sequence[str],
) -> list[dict[str, object]]:
    """Aggregate replicate predictions equally after realization averaging."""

    if not rows or not fs_keys:
        raise ValueError("prediction rows and scattering keys must not be empty")
    groups = sorted(
        {
            (
                float(row["temperature"]),
                str(row["model"]),
                int(float(row["segment_length"])),
                int(float(row["lag"])),
            )
            for row in rows
        }
    )
    result: list[dict[str, object]] = []
    for temperature, model, segment_length, lag in groups:
        selected = [
            row
            for row in rows
            if float(row["temperature"]) == temperature
            and str(row["model"]) == model
            and int(float(row["segment_length"])) == segment_length
            and int(float(row["lag"])) == lag
        ]
        replicates = {int(float(row["replicate"])) for row in selected}
        if len(replicates) != len(selected):
            raise ValueError("prediction rows must contain one row per replicate and cell")
        predicted_msd = float(np.mean([float(row["predicted_msd"]) for row in selected]))
        observed_msd = float(np.mean([float(row["observed_msd"]) for row in selected]))
        predicted_ngp = float(np.mean([float(row["predicted_ngp"]) for row in selected]))
        observed_ngp = float(np.mean([float(row["observed_ngp"]) for row in selected]))
        replicate_count = len(selected)
        summary: dict[str, object] = {
            "temperature": temperature,
            "model": model,
            "segment_length": float(segment_length),
            "lag": float(lag),
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
            "heldout_path_used_in_prediction": 0.0,
            "macro_fit_parameter_count": 0.0,
            "microdynamic_closure_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for observed_key in fs_keys:
            suffix = observed_key.removeprefix("observed_")
            predicted = float(
                np.mean([float(row[f"predicted_{suffix}"]) for row in selected])
            )
            observed = float(np.mean([float(row[observed_key]) for row in selected]))
            summary[f"predicted_{suffix}"] = predicted
            summary[observed_key] = observed
            summary[f"ensemble_absolute_error_{suffix}"] = abs(predicted - observed)
            summary[f"ensemble_{suffix}_mc_se"] = math.sqrt(
                sum(
                    float(row[f"predicted_{suffix}_mc_se"]) ** 2
                    for row in selected
                )
            ) / replicate_count
        result.append(summary)
    return result


def _zero_claim_boundaries(rows: Sequence[dict[str, object]]) -> bool:
    return all(
        float(row["heldout_path_used_in_prediction"]) == 0.0
        and float(row["macro_fit_parameter_count"]) == 0.0
        and float(row["microdynamic_closure_claim_allowed"]) == 0.0
        and float(row["spatial_facilitation_claim_allowed"]) == 0.0
        and float(row["thermodynamic_claim_allowed"]) == 0.0
        for row in rows
    )


def classify_segment_cells(
    summary_rows: Sequence[dict[str, object]],
    quality_rows: Sequence[dict[str, object]],
    stationarity_rows: Sequence[dict[str, object]],
    *,
    expected_grids: dict[float, tuple[int, ...]],
    expected_replicates: dict[float, tuple[int, ...]],
    expected_realizations: int,
) -> list[dict[str, object]]:
    """Recompute each segment cell from raw quality, precision, and curve rows."""

    if (
        not summary_rows
        or not quality_rows
        or not stationarity_rows
        or not expected_grids
        or set(expected_grids) != set(expected_replicates)
        or isinstance(expected_realizations, bool)
        or not isinstance(expected_realizations, int)
        or expected_realizations < 1
    ):
        raise ValueError("complete segment classifier inputs are required")
    temperatures = set(expected_grids)
    expected_cells = {
        (temperature, model, length)
        for temperature, grid in expected_grids.items()
        for model in MODELS
        for length in grid
    }
    present_cells = {
        (
            float(row["temperature"]),
            str(row["model"]),
            int(float(row["segment_length"])),
        )
        for row in summary_rows
    }
    if present_cells != expected_cells:
        raise ValueError("summary rows do not contain the exact frozen model-length grid")
    for temperature in temperatures:
        actual_replicates = {
            int(float(row["replicate"]))
            for row in quality_rows
            if float(row["temperature"]) == temperature
        }
        if actual_replicates != set(expected_replicates[temperature]):
            raise ValueError("quality rows do not contain the exact replicate set")
    stationarity_by_temperature: dict[float, bool] = {}
    for temperature in temperatures:
        local = {
            str(row["comparison"]): float(row["curve_transfer_pass"]) == 1.0
            for row in stationarity_rows
            if float(row["temperature"]) == temperature
        }
        stationarity_by_temperature[temperature] = (
            STATIONARITY_COMPARISONS.issubset(local)
            and all(local[name] for name in STATIONARITY_COMPARISONS)
        )
    results: list[dict[str, object]] = []
    for temperature, model, segment_length in sorted(expected_cells):
        summaries = [
            row
            for row in summary_rows
            if float(row["temperature"]) == temperature
            and str(row["model"]) == model
            and int(float(row["segment_length"])) == segment_length
        ]
        quality = [
            row
            for row in quality_rows
            if float(row["temperature"]) == temperature
            and str(row["model"]) == model
            and int(float(row["segment_length"])) == segment_length
        ]
        expected_pairs = {
            (replicate, realization)
            for replicate in expected_replicates[temperature]
            for realization in range(expected_realizations)
        }
        actual_pairs = [
            (int(float(row["replicate"])), int(float(row["realization"])))
            for row in quality
        ]
        realization_complete = (
            len(actual_pairs) == len(expected_pairs)
            and len(set(actual_pairs)) == len(actual_pairs)
            and set(actual_pairs) == expected_pairs
        )
        exact_information = realization_complete and all(
            float(row["source_token_reuse_minimum"]) == 1.0
            and float(row["source_token_reuse_maximum"]) == 1.0
            and float(row["ordered_token_multiset_preserved"]) == 1.0
            and float(row["global_block_provenance_multiset_preserved"]) == 1.0
            and float(row["global_block_vector_multiset_preserved"]) == 1.0
            and float(row["segment_length_histogram_preserved"]) == 1.0
            and float(row["internal_adjacent_pair_multiset_preserved"]) == 1.0
            and float(row["complete_particle_paths"]) == 1.0
            and (
                float(row["within_particle_vector_multiset_preserved"]) == 1.0
                if model == "within_particle_segment_shuffle"
                else float(row["same_source_assignment_fraction"]) == 0.0
                and float(row["adjacent_same_source_segment_fraction"]) == 0.0
            )
            for row in quality
        )
        provenance_pass = _zero_claim_boundaries(quality) and _zero_claim_boundaries(
            summaries
        )
        fs_error_names = sorted(
            {
                name
                for row in summaries
                for name in row
                if name.startswith("ensemble_absolute_error_fs_k")
            }
        )
        fs_mc_names = sorted(
            {
                name
                for row in summaries
                for name in row
                if name.startswith("ensemble_fs_k") and name.endswith("_mc_se")
            }
        )
        if not fs_error_names or not fs_mc_names:
            raise ValueError("summary rows must contain scattering errors and precision")
        maximum_msd = max(float(row["ensemble_msd_relative_error"]) for row in summaries)
        maximum_ngp = max(float(row["ensemble_ngp_absolute_error"]) for row in summaries)
        maximum_fs = max(float(row[name]) for row in summaries for name in fs_error_names)
        maximum_msd_mc = max(
            float(row["ensemble_msd_mc_relative_se"]) for row in summaries
        )
        maximum_ngp_mc = max(float(row["ensemble_ngp_mc_se"]) for row in summaries)
        maximum_fs_mc = max(float(row[name]) for row in summaries for name in fs_mc_names)
        precision_pass = (
            maximum_msd_mc <= 0.01
            and maximum_ngp_mc <= 0.03
            and maximum_fs_mc <= 0.003
        )
        curve_pass = maximum_msd <= 0.10 and maximum_ngp <= 0.30 and maximum_fs <= 0.03
        stationarity_pass = stationarity_by_temperature[temperature]
        cell_pass = (
            exact_information
            and provenance_pass
            and stationarity_pass
            and precision_pass
            and curve_pass
        )
        results.append(
            {
                "temperature": temperature,
                "model": model,
                "segment_length": float(segment_length),
                "lag_count": float(len(summaries)),
                "required_replicate_count": float(len(expected_replicates[temperature])),
                "required_realization_count": float(expected_realizations),
                "realization_completeness_pass": float(realization_complete),
                "exact_information_pass": float(exact_information),
                "provenance_claim_boundary_pass": float(provenance_pass),
                "stationarity_control_pass": float(stationarity_pass),
                "maximum_ensemble_msd_relative_error": maximum_msd,
                "maximum_ensemble_ngp_absolute_error": maximum_ngp,
                "maximum_ensemble_fs_absolute_error": maximum_fs,
                "maximum_ensemble_msd_mc_relative_se": maximum_msd_mc,
                "maximum_ensemble_ngp_mc_se": maximum_ngp_mc,
                "maximum_ensemble_fs_mc_se": maximum_fs_mc,
                "precision_pass": float(precision_pass),
                "curve_transfer_pass": float(curve_pass),
                "cell_pass": float(cell_pass),
                "heldout_path_used_in_prediction": 0.0,
                "macro_fit_parameter_count": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return results
