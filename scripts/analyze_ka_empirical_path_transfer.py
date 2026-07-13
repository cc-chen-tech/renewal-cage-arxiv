#!/usr/bin/env python3
"""Compare calibration-only contiguous block paths with memory-destroying nulls."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    cumulative_block_observables,
    direction_randomized_block_observables,
    load_lammps_custom_trajectory,
    within_particle_time_shuffle,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty path-transfer table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wave_number_from_observed_key(key: str) -> float:
    return float(key.removeprefix("observed_fs_k").replace("p", "."))


def path_shuffle_seed(base_seed: int, *, replicate: int, realization: int) -> int:
    if min(base_seed, replicate, realization) < 0:
        raise ValueError("shuffle seed indices must be nonnegative")
    return int((base_seed + 1_000_003 * replicate + 97_409 * realization) % (2**63 - 1))


def _higher_order_replicate_scores(
    replicate_rows: list[dict[str, object]],
) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    for row in replicate_rows:
        key = (str(row["model"]), int(float(row["replicate"])))
        fs_errors = [
            float(value)
            for name, value in row.items()
            if name.startswith("absolute_error_fs_k")
        ]
        score = max(
            float(row["ngp_absolute_error"]) / 0.30,
            max(fs_errors, default=0.0) / 0.03,
        )
        scores[key] = max(scores.get(key, 0.0), score)
    return scores


def classify_path_model_transfer(
    summary_rows: list[dict[str, object]],
    *,
    replicate_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not summary_rows:
        raise ValueError("summary_rows must not be empty")
    models = sorted({str(row["model"]) for row in summary_rows})
    replicate_scores = _higher_order_replicate_scores(replicate_rows)
    common_replicates = sorted(
        replicate
        for model, replicate in replicate_scores
        if model == "contiguous_empirical_path"
        and ("within_particle_time_shuffle", replicate) in replicate_scores
    )
    contiguous_better = sum(
        replicate_scores[("contiguous_empirical_path", replicate)]
        < replicate_scores[("within_particle_time_shuffle", replicate)]
        for replicate in common_replicates
    )
    verdicts: list[dict[str, object]] = []
    for model in models:
        selected = [row for row in summary_rows if str(row["model"]) == model]
        maximum_msd = max(float(row["ensemble_msd_relative_error"]) for row in selected)
        maximum_ngp = max(float(row["ensemble_ngp_absolute_error"]) for row in selected)
        fs_error_names = sorted(
            {
                name
                for row in selected
                for name in row
                if name.startswith("ensemble_absolute_error_fs_k")
            }
        )
        maximum_fs = max(
            float(row[name]) for row in selected for name in fs_error_names
        )
        maximum_msd_mc = max(
            float(row.get("ensemble_msd_mc_relative_se", 0.0)) for row in selected
        )
        maximum_ngp_mc = max(
            float(row.get("ensemble_ngp_mc_se", 0.0)) for row in selected
        )
        fs_mc_names = sorted(
            {
                name
                for row in selected
                for name in row
                if name.startswith("ensemble_fs_k") and name.endswith("_mc_se")
            }
        )
        maximum_fs_mc = max(
            (float(row[name]) for row in selected for name in fs_mc_names),
            default=0.0,
        )
        precision_pass = (
            model != "within_particle_time_shuffle"
            or (
                maximum_msd_mc <= 0.01
                and maximum_ngp_mc <= 0.03
                and maximum_fs_mc <= 0.003
            )
        )
        raw_curve_pass = (
            maximum_msd <= 0.10
            and maximum_ngp <= 0.30
            and maximum_fs <= 0.03
        )
        verdicts.append(
            {
                "model": model,
                "lag_count": float(len(selected)),
                "maximum_ensemble_msd_relative_error": maximum_msd,
                "maximum_ensemble_ngp_absolute_error": maximum_ngp,
                "maximum_ensemble_fs_absolute_error": maximum_fs,
                "maximum_ensemble_msd_mc_relative_se": maximum_msd_mc,
                "maximum_ensemble_ngp_mc_se": maximum_ngp_mc,
                "maximum_ensemble_fs_mc_se": maximum_fs_mc,
                "msd_relative_error_tolerance": 0.10,
                "ngp_absolute_error_tolerance": 0.30,
                "fs_absolute_error_tolerance": 0.03,
                "shuffle_msd_mc_relative_se_tolerance": 0.01,
                "shuffle_ngp_mc_se_tolerance": 0.03,
                "shuffle_fs_mc_se_tolerance": 0.003,
                "raw_curve_tolerance_pass": float(raw_curve_pass),
                "shuffle_precision_pass": float(precision_pass),
                "curve_transfer_pass": float(raw_curve_pass and precision_pass),
                "paired_contiguous_better_replicate_count": float(contiguous_better),
                "paired_replicate_count": float(len(common_replicates)),
                "heldout_path_used_in_prediction": 0.0,
                "macro_fit_parameter_count": 0.0,
                "calibration_path_distribution_used": 1.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return verdicts


def _standard_error(values: list[float]) -> float:
    return (
        float(np.std(values, ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )


def _prediction_row(
    *,
    model: str,
    replicate: int,
    temperature: float,
    lag: int,
    block_size: int,
    prediction: dict[str, float],
    prediction_se: dict[str, float],
    observed: dict[str, str],
    fs_keys: list[str],
    shuffle_realizations: int,
    base_seed: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model,
        "replicate": float(replicate),
        "temperature": temperature,
        "lag": float(lag),
        "block_size": float(block_size),
        "block_count": float(lag // block_size),
        "particle_window_count": prediction["particle_window_count"],
        "predicted_msd": prediction["msd"],
        "observed_msd": float(observed["observed_msd"]),
        "msd_relative_error": abs(
            prediction["msd"] / float(observed["observed_msd"]) - 1.0
        ),
        "predicted_msd_mc_se": prediction_se.get("msd", 0.0),
        "predicted_ngp": prediction["ngp"],
        "observed_ngp": float(observed["observed_ngp"]),
        "ngp_absolute_error": abs(
            prediction["ngp"] - float(observed["observed_ngp"])
        ),
        "predicted_ngp_mc_se": prediction_se.get("ngp", 0.0),
        "shuffle_realizations": float(
            shuffle_realizations if model == "within_particle_time_shuffle" else 0
        ),
        "shuffle_base_seed": float(base_seed),
        "heldout_path_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "calibration_path_distribution_used": 1.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for fs_key in fs_keys:
        suffix = fs_key.removeprefix("observed_")
        kernel_key = fs_key.removeprefix("observed_").replace("fs_", "characteristic_")
        predicted = prediction[kernel_key]
        row[f"predicted_{suffix}"] = predicted
        row[fs_key] = float(observed[fs_key])
        row[f"absolute_error_{suffix}"] = abs(predicted - float(observed[fs_key]))
        row[f"predicted_{suffix}_mc_se"] = prediction_se.get(kernel_key, 0.0)
    return row


def summarize_path_rows(
    rows: list[dict[str, object]],
    *,
    fs_keys: list[str],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        for lag in sorted(
            {float(row["lag"]) for row in rows if str(row["model"]) == model}
        ):
            selected = [
                row
                for row in rows
                if str(row["model"]) == model and float(row["lag"]) == lag
            ]
            predicted_msd = float(np.mean([float(row["predicted_msd"]) for row in selected]))
            observed_msd = float(np.mean([float(row["observed_msd"]) for row in selected]))
            predicted_ngp = float(np.mean([float(row["predicted_ngp"]) for row in selected]))
            observed_ngp = float(np.mean([float(row["observed_ngp"]) for row in selected]))
            replicate_count = len(selected)
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
                predicted = float(
                    np.mean([float(row[f"predicted_{suffix}"]) for row in selected])
                )
                observed_value = float(
                    np.mean([float(row[fs_key]) for row in selected])
                )
                summary[f"predicted_{suffix}"] = predicted
                summary[fs_key] = observed_value
                summary[f"ensemble_absolute_error_{suffix}"] = abs(
                    predicted - observed_value
                )
                summary[f"ensemble_{suffix}_mc_se"] = math.sqrt(
                    sum(
                        float(row[f"predicted_{suffix}_mc_se"]) ** 2
                        for row in selected
                    )
                ) / replicate_count
            summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--calibration-time", type=int, required=True)
    parser.add_argument("--heldout-factorization", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--shuffle-realizations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=45101)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if min(args.calibration_time, args.block_size, args.shuffle_realizations) < 1:
        raise ValueError("calibration time, block size, and realizations must be positive")

    manifest = json.loads(
        (args.ensemble_directory / "ensemble_manifest.json").read_text()
    )
    heldout_rows = read_rows(args.heldout_factorization)
    heldout = {
        (int(float(row["replicate"])), int(float(row["lag"]))): row
        for row in heldout_rows
        if int(float(row["lag"])) % args.block_size == 0
    }
    fs_keys = sorted(key for key in heldout_rows[0] if key.startswith("observed_fs_k"))
    wave_numbers = np.array([wave_number_from_observed_key(key) for key in fs_keys])
    output_rows: list[dict[str, object]] = []

    for replicate_spec in manifest["replicates"]:
        replicate = int(replicate_spec["replicate"])
        local_lags = sorted(lag for local_replicate, lag in heldout if local_replicate == replicate)
        if not local_lags:
            raise ValueError(f"replicate {replicate} has no held-out block-compatible lag")
        directory = args.ensemble_directory / str(replicate_spec["directory"])
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"][
            : args.calibration_time + 1,
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

        deterministic: dict[str, dict[int, dict[str, float]]] = {
            "contiguous_empirical_path": {},
            "direction_randomized_path": {},
        }
        for lag in local_lags:
            block_count = lag // args.block_size
            deterministic["contiguous_empirical_path"][lag] = cumulative_block_observables(
                blocks,
                block_count=block_count,
                wave_numbers=wave_numbers,
            )
            deterministic["direction_randomized_path"][lag] = (
                direction_randomized_block_observables(
                    blocks,
                    block_count=block_count,
                    wave_numbers=wave_numbers,
                )
            )

        shuffle_samples: dict[int, list[dict[str, float]]] = {
            lag: [] for lag in local_lags
        }
        for realization in range(args.shuffle_realizations):
            rng = np.random.default_rng(
                path_shuffle_seed(
                    args.seed,
                    replicate=replicate,
                    realization=realization,
                )
            )
            shuffled = within_particle_time_shuffle(blocks, rng)
            for lag in local_lags:
                shuffle_samples[lag].append(
                    cumulative_block_observables(
                        shuffled,
                        block_count=lag // args.block_size,
                        wave_numbers=wave_numbers,
                    )
                )
            del shuffled

        for model, predictions in deterministic.items():
            for lag, prediction in predictions.items():
                output_rows.append(
                    _prediction_row(
                        model=model,
                        replicate=replicate,
                        temperature=float(manifest["temperature"]),
                        lag=lag,
                        block_size=args.block_size,
                        prediction=prediction,
                        prediction_se={},
                        observed=heldout[(replicate, lag)],
                        fs_keys=fs_keys,
                        shuffle_realizations=args.shuffle_realizations,
                        base_seed=args.seed,
                    )
                )
        for lag, samples in shuffle_samples.items():
            metric_keys = ["msd", "ngp"] + [
                key for key in samples[0] if key.startswith("characteristic_k")
            ]
            prediction = {
                key: float(np.mean([sample[key] for sample in samples]))
                for key in metric_keys
            }
            prediction["particle_window_count"] = samples[0]["particle_window_count"]
            prediction_se = {
                key: _standard_error([sample[key] for sample in samples])
                for key in metric_keys
            }
            output_rows.append(
                _prediction_row(
                    model="within_particle_time_shuffle",
                    replicate=replicate,
                    temperature=float(manifest["temperature"]),
                    lag=lag,
                    block_size=args.block_size,
                    prediction=prediction,
                    prediction_se=prediction_se,
                    observed=heldout[(replicate, lag)],
                    fs_keys=fs_keys,
                    shuffle_realizations=args.shuffle_realizations,
                    base_seed=args.seed,
                )
            )
        del blocks
        gc.collect()

    summary_rows = summarize_path_rows(output_rows, fs_keys=fs_keys)
    verdict_rows = classify_path_model_transfer(
        summary_rows,
        replicate_rows=output_rows,
    )
    for row in verdict_rows:
        row["temperature"] = float(manifest["temperature"])
        row["independent_replicate_count"] = float(len(manifest["replicates"]))
        row["shuffle_realizations"] = float(args.shuffle_realizations)
        row["shuffle_base_seed"] = float(args.seed)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_rows.csv"), output_rows)
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
        summary_rows,
    )
    write_rows(
        args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"),
        verdict_rows,
    )


if __name__ == "__main__":
    main()
