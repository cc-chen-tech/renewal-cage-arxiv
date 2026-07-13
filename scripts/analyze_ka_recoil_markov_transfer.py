#!/usr/bin/env python3
"""Test a calibration-only radial recoil Markov null on held-out curves."""

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

from ka_replicates import (  # noqa: E402
    cumulative_block_observables,
    load_lammps_custom_trajectory,
    radial_recoil_markov_quality,
    radial_recoil_markov_surrogate,
)


def recoil_seed(base_seed: int, *, replicate: int, realization: int) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (base_seed, replicate, realization)
    ):
        raise ValueError("recoil seed indices must be nonnegative integers")
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


def _wave_number_from_observed_key(key: str) -> float:
    return float(key.removeprefix("observed_fs_k").replace("p", "."))


def _validate_frozen_recoil_protocol(
    *,
    block_size: int,
    surrogate_realizations: int,
) -> None:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size != 20:
        raise ValueError("the frozen recoil gate requires block_size exactly 20")
    if (
        isinstance(surrogate_realizations, bool)
        or not isinstance(surrogate_realizations, int)
        or surrogate_realizations != 16
    ):
        raise ValueError("the frozen recoil gate requires exactly 16 surrogate realizations")


def _prediction_row(
    *,
    replicate: int,
    realization: int,
    temperature: float,
    lag: int,
    block_size: int,
    prediction: dict[str, float],
    observed: dict[str, object],
    fs_keys: Sequence[str],
    surrogate_realizations: int,
    base_seed: int,
) -> dict[str, object]:
    observed_msd = float(observed["observed_msd"])
    if observed_msd <= 0.0:
        raise ValueError("held-out observed MSD must be positive")
    row: dict[str, object] = {
        "model": "radial_recoil_markov_surrogate",
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
        "observed_ngp": float(observed["observed_ngp"]),
        "ngp_absolute_error": abs(float(prediction["ngp"]) - float(observed["observed_ngp"])),
        "surrogate_realizations": float(surrogate_realizations),
        "surrogate_base_seed": float(base_seed),
        "calibration_path_used_in_kernel": 1.0,
        "heldout_path_used_in_prediction": 0.0,
        "heldout_events_used_in_calibration": 0.0,
        "macro_fit_parameter_count": 0.0,
        **_claim_flags(),
    }
    for fs_key in fs_keys:
        suffix = fs_key.removeprefix("observed_")
        predicted_key = suffix.replace("fs_", "characteristic_")
        predicted = float(prediction[predicted_key])
        observed_value = float(observed[fs_key])
        row[f"predicted_{suffix}"] = predicted
        row[fs_key] = observed_value
        row[f"absolute_error_{suffix}"] = abs(predicted - observed_value)
    return row


def analyze_replicate_recoil_paths(
    block_displacements: np.ndarray,
    heldout: dict[int, dict[str, object]],
    *,
    replicate: int,
    temperature: float,
    block_size: int,
    wave_numbers: np.ndarray,
    fs_keys: Sequence[str],
    surrogate_realizations: int,
    base_seed: int,
) -> dict[str, list[dict[str, object]]]:
    _validate_frozen_recoil_protocol(
        block_size=block_size,
        surrogate_realizations=surrogate_realizations,
    )
    blocks = np.asarray(block_displacements, dtype=float)
    if (
        blocks.ndim != 3
        or blocks.shape[1] < 8
        or blocks.shape[2] != 3
        or np.any(~np.isfinite(blocks))
        or not heldout
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
        or isinstance(surrogate_realizations, bool)
        or not isinstance(surrogate_realizations, int)
        or surrogate_realizations < 1
    ):
        raise ValueError("valid calibration block paths and positive controls are required")
    local_lags = sorted(heldout)
    if any(lag % block_size or lag // block_size > blocks.shape[1] for lag in local_lags):
        raise ValueError("held-out lags must fit the calibration block path")
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for realization in range(surrogate_realizations):
        seed = recoil_seed(base_seed, replicate=replicate, realization=realization)
        surrogate = radial_recoil_markov_surrogate(
            blocks,
            np.random.default_rng(seed),
            radial_bin_count=8,
        )
        quality_rows.append(
            {
                "model": "radial_recoil_markov_surrogate",
                "replicate": float(replicate),
                "realization": float(realization),
                "temperature": float(temperature),
                "block_size": float(block_size),
                "surrogate_seed": float(seed),
                **radial_recoil_markov_quality(blocks, surrogate),
                "calibration_path_used_in_kernel": 1.0,
                "heldout_events_used_in_calibration": 0.0,
                **_claim_flags(),
            }
        )
        for lag in local_lags:
            prediction = cumulative_block_observables(
                surrogate,
                block_count=lag // block_size,
                wave_numbers=wave_numbers,
            )
            rows.append(
                _prediction_row(
                    replicate=replicate,
                    realization=realization,
                    temperature=temperature,
                    lag=lag,
                    block_size=block_size,
                    prediction=prediction,
                    observed=heldout[lag],
                    fs_keys=fs_keys,
                    surrogate_realizations=surrogate_realizations,
                    base_seed=base_seed,
                )
            )
    return {"rows": rows, "quality_rows": quality_rows}


def summarize_recoil_realizations(
    rows: Sequence[dict[str, object]],
    *,
    fs_keys: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not rows or not fs_keys:
        raise ValueError("recoil rows and scattering keys must not be empty")
    replicate_rows: list[dict[str, object]] = []
    for replicate, lag in sorted({(int(float(row["replicate"])), float(row["lag"])) for row in rows}):
        selected = [
            row for row in rows
            if int(float(row["replicate"])) == replicate and float(row["lag"]) == lag
        ]
        observed_msd_values = {float(row["observed_msd"]) for row in selected}
        observed_ngp_values = {float(row["observed_ngp"]) for row in selected}
        if len(observed_msd_values) != 1 or len(observed_ngp_values) != 1:
            raise ValueError("held-out values must agree across surrogate realizations")
        observed_msd = observed_msd_values.pop()
        if observed_msd <= 0.0:
            raise ValueError("held-out observed MSD must be positive")
        result: dict[str, object] = {
            "model": "radial_recoil_markov_surrogate",
            "replicate": float(replicate),
            "lag": lag,
            "realization_count": float(len(selected)),
            "predicted_msd": float(np.mean([float(row["predicted_msd"]) for row in selected])),
            "observed_msd": observed_msd,
            "predicted_msd_mc_se": _standard_error([float(row["predicted_msd"]) for row in selected]),
            "predicted_ngp": float(np.mean([float(row["predicted_ngp"]) for row in selected])),
            "observed_ngp": observed_ngp_values.pop(),
            "predicted_ngp_mc_se": _standard_error([float(row["predicted_ngp"]) for row in selected]),
            "calibration_path_used_in_kernel": 1.0,
            "heldout_path_used_in_prediction": 0.0,
            "heldout_events_used_in_calibration": 0.0,
            **_claim_flags(),
        }
        result["msd_relative_error"] = abs(float(result["predicted_msd"]) / observed_msd - 1.0)
        result["ngp_absolute_error"] = abs(float(result["predicted_ngp"]) - float(result["observed_ngp"]))
        for fs_key in fs_keys:
            suffix = fs_key.removeprefix("observed_")
            predicted_key = f"predicted_{suffix}"
            observed_values = {float(row[fs_key]) for row in selected}
            if len(observed_values) != 1:
                raise ValueError("held-out scattering values must agree across realizations")
            values = [float(row[predicted_key]) for row in selected]
            result[predicted_key] = float(np.mean(values))
            result[fs_key] = observed_values.pop()
            result[f"absolute_error_{suffix}"] = abs(float(result[predicted_key]) - float(result[fs_key]))
            result[f"predicted_{suffix}_mc_se"] = _standard_error(values)
        replicate_rows.append(result)

    summary_rows: list[dict[str, object]] = []
    for lag in sorted({float(row["lag"]) for row in replicate_rows}):
        selected = [row for row in replicate_rows if float(row["lag"]) == lag]
        replicate_count = len(selected)
        predicted_msd = float(np.mean([float(row["predicted_msd"]) for row in selected]))
        observed_msd = float(np.mean([float(row["observed_msd"]) for row in selected]))
        predicted_ngp = float(np.mean([float(row["predicted_ngp"]) for row in selected]))
        observed_ngp = float(np.mean([float(row["observed_ngp"]) for row in selected]))
        summary: dict[str, object] = {
            "model": "radial_recoil_markov_surrogate",
            "lag": lag,
            "independent_replicate_count": float(replicate_count),
            "predicted_msd": predicted_msd,
            "observed_msd": observed_msd,
            "ensemble_msd_relative_error": abs(predicted_msd / observed_msd - 1.0),
            "ensemble_msd_mc_relative_se": math.sqrt(sum(float(row["predicted_msd_mc_se"]) ** 2 for row in selected)) / replicate_count / observed_msd,
            "predicted_ngp": predicted_ngp,
            "observed_ngp": observed_ngp,
            "ensemble_ngp_absolute_error": abs(predicted_ngp - observed_ngp),
            "ensemble_ngp_mc_se": math.sqrt(sum(float(row["predicted_ngp_mc_se"]) ** 2 for row in selected)) / replicate_count,
            "replicate_first_aggregation": 1.0,
            "heldout_path_used_in_prediction": 0.0,
            **_claim_flags(),
        }
        for fs_key in fs_keys:
            suffix = fs_key.removeprefix("observed_")
            predicted_key = f"predicted_{suffix}"
            observed = float(np.mean([float(row[fs_key]) for row in selected]))
            predicted = float(np.mean([float(row[predicted_key]) for row in selected]))
            summary[predicted_key] = predicted
            summary[fs_key] = observed
            summary[f"ensemble_absolute_error_{suffix}"] = abs(predicted - observed)
            summary[f"ensemble_{suffix}_mc_se"] = math.sqrt(sum(float(row[f"predicted_{suffix}_mc_se"]) ** 2 for row in selected)) / replicate_count
        summary_rows.append(summary)
    return replicate_rows, summary_rows


def classify_recoil_transfer(
    quality_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
    *,
    required_replicate_count: int,
    required_realization_count: int,
) -> dict[str, object]:
    """Use quality and Monte Carlo precision gates before curve classification."""

    if not quality_rows or not summary_rows:
        raise ValueError("quality and summary rows must not be empty")
    if required_replicate_count < 1 or required_realization_count != 16:
        raise ValueError("the frozen recoil gate requires positive replicas and 16 realizations")
    pairs = {(int(float(row["replicate"])), int(float(row["realization"]))) for row in quality_rows}
    replicate_ids = {replicate for replicate, _ in pairs}
    complete = (
        len(replicate_ids) == required_replicate_count
        and len(pairs) == len(quality_rows) == required_replicate_count * required_realization_count
        and all(
            {realization for candidate, realization in pairs if candidate == replicate}
            == set(range(required_realization_count))
            for replicate in replicate_ids
        )
    )
    quality_pass = complete and all(
        float(row["radial_mean_relative_error"]) <= 0.02
        and float(row["radial_standard_deviation_relative_error"]) <= 0.02
        and float(row["lag_one_cosine_mean_absolute_error"]) <= 0.02
        and float(row["lag_one_cosine_quantile_maximum_absolute_error"]) <= 0.03
        and float(row["normalized_lag_one_dot_correlation_absolute_error"]) <= 0.02
        for row in quality_rows
    )
    maximum_msd_mc = max(float(row["ensemble_msd_mc_relative_se"]) for row in summary_rows)
    maximum_ngp_mc = max(float(row["ensemble_ngp_mc_se"]) for row in summary_rows)
    fs_mc_names = sorted({key for row in summary_rows for key in row if key.startswith("ensemble_fs_k") and key.endswith("_mc_se")})
    maximum_fs_mc = max((float(row[key]) for row in summary_rows for key in fs_mc_names), default=0.0)
    precision_pass = maximum_msd_mc <= 0.01 and maximum_ngp_mc <= 0.03 and maximum_fs_mc <= 0.003
    maximum_msd = max(float(row["ensemble_msd_relative_error"]) for row in summary_rows)
    maximum_ngp = max(float(row["ensemble_ngp_absolute_error"]) for row in summary_rows)
    fs_error_names = sorted({key for row in summary_rows for key in row if key.startswith("ensemble_absolute_error_fs_k")})
    if not fs_error_names:
        raise ValueError("summary rows must include multi-k scattering errors")
    maximum_fs = max(float(row[key]) for row in summary_rows for key in fs_error_names)
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
        "independent_replicate_count": float(len(replicate_ids)),
        "required_replicate_count": float(required_replicate_count),
        "required_realizations_per_replicate": float(required_realization_count),
        "quality_realization_completeness_pass": float(complete),
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
        "cage_anchor_memory_required": 0.0,
        "heldout_events_used_in_calibration": 0.0,
        "next_required_test": "cross_temperature_cage_anchor_gate",
        **_claim_flags(),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty recoil table")
    fieldnames = list(rows[0])
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extras, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--heldout-factorization", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--surrogate-realizations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=781031)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    _validate_frozen_recoil_protocol(
        block_size=args.block_size,
        surrogate_realizations=args.surrogate_realizations,
    )
    if min(args.calibration_time, args.block_size, args.surrogate_realizations) < 1 or args.seed < 0:
        raise ValueError("calibration, block, realization, and seed controls are invalid")
    if args.calibration_time // args.block_size < 8:
        raise ValueError("calibration window must contain at least eight complete blocks")
    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    specifications = manifest.get("replicates", [])
    if not specifications:
        raise ValueError("ensemble manifest contains no replicates")
    heldout_rows = _read_rows(args.heldout_factorization)
    fs_keys = sorted(key for key in heldout_rows[0] if key.startswith("observed_fs_k"))
    if not fs_keys:
        raise ValueError("held-out factorization contains no observed scattering columns")
    wave_numbers = np.asarray([_wave_number_from_observed_key(key) for key in fs_keys])
    heldout_by_replicate: dict[int, dict[int, dict[str, object]]] = {}
    for row in heldout_rows:
        replicate = int(float(row["replicate"]))
        lag = int(float(row["lag"]))
        if lag % args.block_size == 0 and lag <= args.calibration_time:
            local = heldout_by_replicate.setdefault(replicate, {})
            if lag in local:
                raise ValueError(f"duplicate held-out row for replicate {replicate}, lag {lag}")
            local[lag] = row
    rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for specification in specifications:
        replicate = int(specification["replicate"])
        heldout = heldout_by_replicate.get(replicate)
        if not heldout:
            raise ValueError(f"replicate {replicate} has no block-compatible held-out lag")
        directory = args.ensemble_directory / str(specification["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(
            directory / "trajectory.lammpstrj",
            maximum_frame_count=args.calibration_time + 1,
        )
        if len(trajectory["unwrapped_positions"]) < args.calibration_time + 1:
            raise ValueError(f"replicate {replicate} is shorter than the calibration window")
        positions = trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0]
        starts = np.arange(args.calibration_time // args.block_size) * args.block_size
        blocks = np.transpose(positions[starts + args.block_size] - positions[starts], (1, 0, 2)).astype(float)
        result = analyze_replicate_recoil_paths(
            blocks,
            heldout,
            replicate=replicate,
            temperature=float(manifest["temperature"]),
            block_size=args.block_size,
            wave_numbers=wave_numbers,
            fs_keys=fs_keys,
            surrogate_realizations=args.surrogate_realizations,
            base_seed=args.seed,
        )
        rows.extend(result["rows"])
        quality_rows.extend(result["quality_rows"])
        del trajectory, positions, blocks, result
        gc.collect()
    replicate_rows, summary_rows = summarize_recoil_realizations(rows, fs_keys=fs_keys)
    verdict = classify_recoil_transfer(
        quality_rows,
        summary_rows,
        required_replicate_count=len(specifications),
        required_realization_count=args.surrogate_realizations,
    )
    metadata = {
        "temperature": float(manifest["temperature"]),
        "calibration_time": float(args.calibration_time),
        "block_size": float(args.block_size),
        "surrogate_realizations_per_replicate": float(args.surrogate_realizations),
        "surrogate_base_seed": float(args.seed),
    }
    for table in (quality_rows, summary_rows):
        for row in table:
            row.update(metadata)
    verdict.update(metadata)
    prefix = args.output_prefix
    write_rows(prefix.with_name(prefix.name + "_rows.csv"), rows)
    write_rows(prefix.with_name(prefix.name + "_quality.csv"), quality_rows)
    write_rows(prefix.with_name(prefix.name + "_summary.csv"), summary_rows)
    write_rows(prefix.with_name(prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
