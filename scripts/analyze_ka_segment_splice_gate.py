#!/usr/bin/env python3
"""Run and score paired segment-splice calibration-path nulls."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import load_lammps_custom_trajectory  # noqa: E402
from ka_segment_splice import (  # noqa: E402
    audit_segment_surrogate,
    cross_particle_segment_splice,
    cumulative_observables_many_lags,
    within_particle_segment_shuffle,
)


MODELS = (
    "within_particle_segment_shuffle",
    "cross_particle_segment_splice",
)
STATIONARITY_COMPARISONS = {"early_late", "early_heldout", "late_heldout"}


def validate_ensemble_manifest(
    manifest: dict[str, object],
    *,
    temperature: float,
) -> tuple[tuple[int, str], ...]:
    """Validate the frozen ensemble identity and return replicate directories."""

    protocol = frozen_protocol(temperature)
    expected_replicates = tuple(protocol["replicates"])
    try:
        manifest_temperature = float(manifest["temperature"])
        replicate_count = int(manifest["replicate_count"])
        thermodynamic_claim_allowed = manifest["thermodynamic_claim_allowed"]
        replicate_specs = manifest["replicates"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("ensemble manifest is incomplete") from error
    if (
        manifest_temperature != temperature
        or replicate_count != len(expected_replicates)
        or thermodynamic_claim_allowed is not False
        or not isinstance(replicate_specs, list)
        or len(replicate_specs) != len(expected_replicates)
    ):
        raise ValueError("ensemble manifest violates the frozen protocol")
    validated: list[tuple[int, str]] = []
    for expected_replicate, spec in zip(
        expected_replicates,
        replicate_specs,
        strict=True,
    ):
        if not isinstance(spec, dict):
            raise ValueError("ensemble replicate specifications must be mappings")
        try:
            replicate = int(spec["replicate"])
            directory = str(spec["directory"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("ensemble replicate specification is incomplete") from error
        if replicate != expected_replicate or directory != f"replicate_{replicate:02d}":
            raise ValueError("ensemble replicate identity violates the frozen protocol")
        validated.append((replicate, directory))
    return tuple(validated)


def calibration_blocks_from_trajectory(
    trajectory: dict[str, object],
    *,
    calibration_time: int,
    block_size: int,
) -> np.ndarray:
    """Extract contiguous Type-A displacement blocks from a loaded trajectory."""

    if (
        isinstance(calibration_time, bool)
        or not isinstance(calibration_time, int)
        or calibration_time < 1
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 1
        or calibration_time < block_size
    ):
        raise ValueError("calibration time must contain at least one complete block")
    try:
        positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
        particle_types = np.asarray(trajectory["particle_types"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("trajectory is missing positions or particle types") from error
    if (
        positions.ndim != 3
        or positions.shape[0] < calibration_time + 1
        or positions.shape[2] != 3
        or particle_types.ndim != 1
        or particle_types.shape[0] != positions.shape[1]
        or np.any(~np.isfinite(positions[: calibration_time + 1]))
    ):
        raise ValueError("trajectory does not cover the requested calibration path")
    type_a = particle_types == 0
    if not np.any(type_a):
        raise ValueError("trajectory contains no Type-A particles")
    block_count = calibration_time // block_size
    starts = np.arange(block_count) * block_size
    selected = positions[: calibration_time + 1, type_a]
    return np.transpose(
        selected[starts + block_size] - selected[starts],
        (1, 0, 2),
    ).astype(float)


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write one nonempty, rectangular CSV table deterministically."""

    if not rows:
        raise ValueError("cannot write an empty segment-splice table")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("segment-splice table rows must have an identical schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_protocol_outputs(
    output_directory: Path,
    result: dict[str, object],
) -> tuple[Path, ...]:
    """Write the exact five-table artifact bundle for both temperatures."""

    tables = (
        ("quality", "quality_rows"),
        ("rows", "prediction_rows"),
        ("summary", "summary_rows"),
        ("cells", "cell_rows"),
        ("replicate_scores", "replicate_scores"),
    )
    paths: list[Path] = []
    for temperature, code in ((0.45, "045"), (0.58, "058")):
        for suffix, result_key in tables:
            source = result.get(result_key)
            if not isinstance(source, list):
                raise ValueError(f"protocol result is missing {result_key}")
            rows = [row for row in source if float(row["temperature"]) == temperature]
            path = output_directory / (
                f"renewal_cage_ka_replicates_T{code}_segment_splice_{suffix}.csv"
            )
            write_rows(path, rows)
            paths.append(path)
    return tuple(paths)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input table is empty: {path}")
    return rows


def load_frozen_blocks(
    ensemble_directory: Path,
    *,
    temperature: float,
    block_size: int,
) -> dict[int, np.ndarray]:
    """Load only the frozen calibration prefix for every ensemble replicate."""

    protocol = frozen_protocol(temperature)
    calibration_time = int(protocol["calibration_time"])
    manifest_path = ensemble_directory / "ensemble_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read ensemble manifest: {manifest_path}") from error
    replicate_directories = validate_ensemble_manifest(
        manifest,
        temperature=temperature,
    )
    result: dict[int, np.ndarray] = {}
    for replicate, directory_name in replicate_directories:
        trajectory_path = ensemble_directory / directory_name / "trajectory.lammpstrj"
        trajectory = load_lammps_custom_trajectory(
            trajectory_path,
            maximum_frame_count=calibration_time + 1,
        )
        result[replicate] = calibration_blocks_from_trajectory(
            trajectory,
            calibration_time=calibration_time,
            block_size=block_size,
        )
        del trajectory
        gc.collect()
    return result


def _validate_target_rows(
    heldout_rows: Sequence[dict[str, object]],
    stationarity_rows: Sequence[dict[str, object]],
    *,
    temperature: float,
    block_size: int,
) -> None:
    protocol = frozen_protocol(temperature)
    expected_replicates = set(protocol["replicates"])
    heldout_replicates = {
        int(float(row["replicate"]))
        for row in heldout_rows
        if float(row["temperature"]) == temperature
        and int(float(row["lag"])) % block_size == 0
        and 1 <= int(float(row["lag"])) // block_size <= int(protocol["block_count"])
    }
    stationarity_comparisons = {
        str(row["comparison"])
        for row in stationarity_rows
        if float(row["temperature"]) == temperature
    }
    if heldout_replicates != expected_replicates:
        raise ValueError("held-out table does not cover the frozen replicate set")
    if stationarity_comparisons != STATIONARITY_COMPARISONS:
        raise ValueError("stationarity table does not contain the exact comparison set")


def _annotate_provenance(
    result: dict[str, object],
    *,
    base_seed: int,
    source_paths: dict[float, tuple[Path, Path, Path]],
) -> None:
    for result_key in (
        "quality_rows",
        "prediction_rows",
        "summary_rows",
        "cell_rows",
        "replicate_scores",
    ):
        rows = result[result_key]
        if not isinstance(rows, list):
            raise ValueError(f"protocol result is missing {result_key}")
        for row in rows:
            temperature = float(row["temperature"])
            ensemble, heldout, stationarity = source_paths[temperature]
            row["segment_splice_base_seed"] = base_seed
            row["source_ensemble_directory"] = str(ensemble.resolve())
            row["source_heldout_factorization"] = str(heldout.resolve())
            row["source_stationarity_control"] = str(stationarity.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen two-temperature segment-splice memory gate."
    )
    parser.add_argument("--low-ensemble-directory", type=Path, required=True)
    parser.add_argument("--high-ensemble-directory", type=Path, required=True)
    parser.add_argument("--low-heldout-factorization", type=Path, required=True)
    parser.add_argument("--high-heldout-factorization", type=Path, required=True)
    parser.add_argument("--low-stationarity", type=Path, required=True)
    parser.add_argument("--high-stationarity", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--initial-realizations", type=int, default=16)
    parser.add_argument("--extended-realizations", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=20260718)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def frozen_protocol(temperature: float) -> dict[str, object]:
    """Return the approved trajectory and length grid for one temperature."""

    if temperature == 0.45:
        return {
            "temperature": 0.45,
            "calibration_time": 5000,
            "block_count": 250,
            "segment_lengths": (1, 2, 5, 10, 25, 50, 125, 250),
            "replicates": (1, 2, 3),
        }
    if temperature == 0.58:
        return {
            "temperature": 0.58,
            "calibration_time": 750,
            "block_count": 37,
            "segment_lengths": (1, 2, 4, 8, 16, 32, 37),
            "replicates": (1, 2, 3, 4, 5),
        }
    raise ValueError("temperature must be one of the frozen values 0.45 or 0.58")


def segment_realization_seed(
    base_seed: int,
    *,
    temperature: float,
    replicate: int,
    segment_length: int,
    realization: int,
) -> int:
    """Generate a stable common seed for one paired surrogate realization."""

    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or base_seed < 0
        or temperature not in {0.45, 0.58}
        or isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or replicate < 1
        or isinstance(segment_length, bool)
        or not isinstance(segment_length, int)
        or segment_length < 1
        or isinstance(realization, bool)
        or not isinstance(realization, int)
        or realization < 0
    ):
        raise ValueError("seed coordinates must be valid frozen nonnegative indices")
    payload = (
        f"{base_seed}|{int(round(temperature * 100))}|{replicate}|"
        f"{segment_length}|{realization}"
    ).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**63 - 1
    )


def validate_real_protocol_controls(
    *,
    block_size: int,
    initial_realizations: int,
    extended_realizations: int,
) -> None:
    if (
        block_size != 20
        or initial_realizations != 16
        or extended_realizations != 64
    ):
        raise ValueError("real protocol requires block size 20 and realization grid 16/64")


def _standard_error(values: Sequence[float]) -> float:
    return (
        float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )


def _assignment_sha256(source_particle: np.ndarray, source_segment: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(source_particle, dtype=np.int64).tobytes())
    digest.update(np.asarray(source_segment, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _prediction_row(
    *,
    model: str,
    temperature: float,
    segment_length: int,
    replicate: int,
    lag: int,
    block_size: int,
    realization_count: int,
    prediction: dict[str, float],
    prediction_se: dict[str, float],
    observed: dict[str, object],
    fs_keys: Sequence[str],
) -> dict[str, object]:
    observed_msd = float(observed["observed_msd"])
    observed_ngp = float(observed["observed_ngp"])
    if not math.isfinite(observed_msd) or observed_msd <= 0.0:
        raise ValueError("held-out observed MSD must be positive and finite")
    row: dict[str, object] = {
        "model": model,
        "temperature": temperature,
        "segment_length": float(segment_length),
        "tau_L": float(block_size * segment_length),
        "replicate": float(replicate),
        "lag": float(lag),
        "block_size": float(block_size),
        "block_count": float(lag // block_size),
        "realization_count": float(realization_count),
        "particle_window_count": prediction["particle_window_count"],
        "predicted_msd": prediction["msd"],
        "observed_msd": observed_msd,
        "msd_relative_error": abs(prediction["msd"] / observed_msd - 1.0),
        "predicted_msd_mc_se": prediction_se["msd"],
        "predicted_ngp": prediction["ngp"],
        "observed_ngp": observed_ngp,
        "ngp_absolute_error": abs(prediction["ngp"] - observed_ngp),
        "predicted_ngp_mc_se": prediction_se["ngp"],
        "heldout_path_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    for fs_key in fs_keys:
        suffix = fs_key.removeprefix("observed_")
        characteristic = suffix.replace("fs_", "characteristic_")
        predicted = prediction[characteristic]
        observed_fs = float(observed[fs_key])
        row[f"predicted_{suffix}"] = predicted
        row[fs_key] = observed_fs
        row[f"absolute_error_{suffix}"] = abs(predicted - observed_fs)
        row[f"predicted_{suffix}_mc_se"] = prediction_se[characteristic]
    return row


def analyze_segment_replicate(
    blocks: np.ndarray,
    heldout_rows: Sequence[dict[str, object]],
    *,
    temperature: float,
    replicate: int,
    segment_lengths: Sequence[int],
    realization_count: int,
    base_seed: int,
    block_size: int,
) -> dict[str, list[dict[str, object]]]:
    """Generate paired segment nulls and realization-average one replicate."""

    protocol = frozen_protocol(temperature)
    values = np.asarray(blocks, dtype=float)
    if (
        values.ndim != 3
        or values.shape[1] != int(protocol["block_count"])
        or values.shape[2] != 3
        or np.any(~np.isfinite(values))
        or tuple(segment_lengths) != tuple(protocol["segment_lengths"])
        and tuple(segment_lengths) not in {(1, 2),}
        or isinstance(realization_count, bool)
        or not isinstance(realization_count, int)
        or realization_count < 1
        or block_size != 20
        or replicate not in protocol["replicates"]
    ):
        raise ValueError("replicate analysis inputs violate the frozen block protocol")
    local_rows = [
        dict(row)
        for row in heldout_rows
        if int(float(row["replicate"])) == replicate
        and float(row["temperature"]) == temperature
        and int(float(row["lag"])) % block_size == 0
        and 1 <= int(float(row["lag"])) // block_size <= values.shape[1]
    ]
    if not local_rows:
        raise ValueError("replicate has no compatible held-out target rows")
    heldout_by_lag = {int(float(row["lag"])): row for row in local_rows}
    if len(heldout_by_lag) != len(local_rows):
        raise ValueError("held-out target rows must be unique by lag")
    fs_keys = sorted(key for key in local_rows[0] if key.startswith("observed_fs_k"))
    if not fs_keys or any(set(row) != set(local_rows[0]) for row in local_rows):
        raise ValueError("held-out target schema must be complete and consistent")
    waves = np.asarray(
        [float(key.removeprefix("observed_fs_k").replace("p", ".")) for key in fs_keys]
    )
    block_counts = tuple(lag // block_size for lag in sorted(heldout_by_lag))
    samples: dict[tuple[str, int, int], list[dict[str, float]]] = {}
    quality_rows: list[dict[str, object]] = []
    for segment_length in segment_lengths:
        for realization in range(realization_count):
            seed = segment_realization_seed(
                base_seed,
                temperature=temperature,
                replicate=replicate,
                segment_length=int(segment_length),
                realization=realization,
            )
            within = within_particle_segment_shuffle(
                values,
                segment_length=int(segment_length),
                rng=np.random.default_rng(seed),
            )
            cross = cross_particle_segment_splice(
                values,
                segment_length=int(segment_length),
                rng=np.random.default_rng(seed),
            )
            paired_order = float(
                np.array_equal(within.source_segment, cross.source_segment)
                and np.array_equal(
                    within.target_segment_lengths,
                    cross.target_segment_lengths,
                )
            )
            for model, surrogate in (
                ("within_particle_segment_shuffle", within),
                ("cross_particle_segment_splice", cross),
            ):
                audit = audit_segment_surrogate(
                    values,
                    surrogate,
                    segment_length=int(segment_length),
                    model=model,
                )
                quality_rows.append(
                    {
                        "temperature": temperature,
                        "model": model,
                        "segment_length": float(segment_length),
                        "tau_L": float(block_size * int(segment_length)),
                        "replicate": float(replicate),
                        "realization": float(realization),
                        "realization_seed": seed,
                        "assignment_sha256": _assignment_sha256(
                            surrogate.source_particle,
                            surrogate.source_segment,
                        ),
                        "paired_segment_order_match": paired_order,
                        **audit,
                    }
                )
                observables = cumulative_observables_many_lags(
                    surrogate.blocks,
                    block_counts=block_counts,
                    wave_numbers=waves,
                )
                for lag, block_count in zip(sorted(heldout_by_lag), block_counts, strict=True):
                    samples.setdefault((model, int(segment_length), lag), []).append(
                        observables[block_count]
                    )
    prediction_rows: list[dict[str, object]] = []
    metric_keys = ("msd", "ngp") + tuple(
        f"characteristic_k{wave:g}".replace(".", "p") for wave in waves
    )
    for (model, segment_length, lag), realization_samples in sorted(samples.items()):
        prediction = {
            key: float(np.mean([sample[key] for sample in realization_samples]))
            for key in metric_keys
        }
        prediction["particle_window_count"] = realization_samples[0][
            "particle_window_count"
        ]
        prediction_se = {
            key: _standard_error([sample[key] for sample in realization_samples])
            for key in metric_keys
        }
        prediction_rows.append(
            _prediction_row(
                model=model,
                temperature=temperature,
                segment_length=segment_length,
                replicate=replicate,
                lag=lag,
                block_size=block_size,
                realization_count=realization_count,
                prediction=prediction,
                prediction_se=prediction_se,
                observed=heldout_by_lag[lag],
                fs_keys=fs_keys,
            )
        )
    replicate_scores: list[dict[str, object]] = []
    for model in MODELS:
        for segment_length in segment_lengths:
            selected = [
                row
                for row in prediction_rows
                if row["model"] == model
                and int(float(row["segment_length"])) == int(segment_length)
            ]
            fs_error_names = sorted(
                name for name in selected[0] if name.startswith("absolute_error_fs_k")
            )
            higher_order_score = max(
                max(float(row["ngp_absolute_error"]) / 0.30 for row in selected),
                max(
                    float(row[name]) / 0.03
                    for row in selected
                    for name in fs_error_names
                ),
            )
            replicate_scores.append(
                {
                    "temperature": temperature,
                    "model": model,
                    "segment_length": float(segment_length),
                    "replicate": float(replicate),
                    "higher_order_score": higher_order_score,
                    "replicate_curve_pass": float(
                        higher_order_score <= 1.0
                        and max(float(row["msd_relative_error"]) for row in selected)
                        <= 0.10
                    ),
                    "heldout_path_used_in_prediction": 0.0,
                    "macro_fit_parameter_count": 0.0,
                    "microdynamic_closure_claim_allowed": 0.0,
                    "spatial_facilitation_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    return {
        "quality_rows": quality_rows,
        "prediction_rows": prediction_rows,
        "replicate_scores": replicate_scores,
    }


def _run_from_blocks_once(
    blocks_by_temperature: dict[float, dict[int, np.ndarray]],
    heldout_by_temperature: dict[float, Sequence[dict[str, object]]],
    stationarity_rows: Sequence[dict[str, object]],
    *,
    realization_count: int,
    base_seed: int,
    block_size: int,
) -> dict[str, object]:
    quality_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    replicate_scores: list[dict[str, object]] = []
    for temperature in (0.45, 0.58):
        protocol = frozen_protocol(temperature)
        if set(blocks_by_temperature[temperature]) != set(protocol["replicates"]):
            raise ValueError("block paths do not contain the exact frozen replicate set")
        for replicate in protocol["replicates"]:
            local = analyze_segment_replicate(
                blocks_by_temperature[temperature][replicate],
                heldout_by_temperature[temperature],
                temperature=temperature,
                replicate=replicate,
                segment_lengths=protocol["segment_lengths"],
                realization_count=realization_count,
                base_seed=base_seed,
                block_size=block_size,
            )
            quality_rows.extend(local["quality_rows"])
            prediction_rows.extend(local["prediction_rows"])
            replicate_scores.extend(local["replicate_scores"])
    fs_keys = sorted(
        key
        for key in heldout_by_temperature[0.45][0]
        if key.startswith("observed_fs_k")
    )
    summary_rows = summarize_segment_rows(prediction_rows, fs_keys=fs_keys)
    cell_rows = classify_segment_cells(
        summary_rows,
        quality_rows,
        stationarity_rows,
        expected_grids={
            temperature: tuple(frozen_protocol(temperature)["segment_lengths"])
            for temperature in (0.45, 0.58)
        },
        expected_replicates={
            temperature: tuple(frozen_protocol(temperature)["replicates"])
            for temperature in (0.45, 0.58)
        },
        expected_realizations=realization_count,
    )
    return {
        "realization_count": realization_count,
        "quality_rows": quality_rows,
        "prediction_rows": prediction_rows,
        "summary_rows": summary_rows,
        "cell_rows": cell_rows,
        "replicate_scores": replicate_scores,
    }


def run_two_temperature_from_blocks(
    blocks_by_temperature: dict[float, dict[int, np.ndarray]],
    heldout_by_temperature: dict[float, Sequence[dict[str, object]]],
    stationarity_rows: Sequence[dict[str, object]],
    *,
    initial_realizations: int,
    extended_realizations: int,
    base_seed: int,
    block_size: int,
) -> dict[str, object]:
    """Run the paired grid and extend every cell together when precision requires."""

    if (
        set(blocks_by_temperature) != {0.45, 0.58}
        or set(heldout_by_temperature) != {0.45, 0.58}
        or block_size != 20
        or isinstance(initial_realizations, bool)
        or not isinstance(initial_realizations, int)
        or initial_realizations < 1
        or isinstance(extended_realizations, bool)
        or not isinstance(extended_realizations, int)
        or extended_realizations < initial_realizations
    ):
        raise ValueError("two-temperature inputs and nested realization counts are required")
    initial = _run_from_blocks_once(
        blocks_by_temperature,
        heldout_by_temperature,
        stationarity_rows,
        realization_count=initial_realizations,
        base_seed=base_seed,
        block_size=block_size,
    )
    if extended_realizations > initial_realizations and any(
        float(row["precision_pass"]) == 0.0 for row in initial["cell_rows"]
    ):
        return _run_from_blocks_once(
            blocks_by_temperature,
            heldout_by_temperature,
            stationarity_rows,
            realization_count=extended_realizations,
            base_seed=base_seed,
            block_size=block_size,
        )
    return initial


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
            and float(row["global_source_segment_schedule_preserved"]) == 1.0
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
                "global_source_segment_schedule_preserved": float(
                    all(
                        float(row["global_source_segment_schedule_preserved"]) == 1.0
                        for row in quality
                    )
                ),
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
                "full_path_control": float(
                    segment_length == max(expected_grids[temperature])
                ),
                "memory_length_selectable": float(
                    segment_length < max(expected_grids[temperature])
                ),
                "heldout_path_used_in_prediction": 0.0,
                "macro_fit_parameter_count": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    args = build_parser().parse_args(argv)
    validate_real_protocol_controls(
        block_size=args.block_size,
        initial_realizations=args.initial_realizations,
        extended_realizations=args.extended_realizations,
    )
    if isinstance(args.base_seed, bool) or args.base_seed < 0:
        raise ValueError("base seed must be a nonnegative integer")

    heldout_by_temperature = {
        0.45: read_rows(args.low_heldout_factorization),
        0.58: read_rows(args.high_heldout_factorization),
    }
    stationarity_by_temperature = {
        0.45: read_rows(args.low_stationarity),
        0.58: read_rows(args.high_stationarity),
    }
    for temperature in (0.45, 0.58):
        _validate_target_rows(
            heldout_by_temperature[temperature],
            stationarity_by_temperature[temperature],
            temperature=temperature,
            block_size=args.block_size,
        )
    blocks_by_temperature = {
        0.45: load_frozen_blocks(
            args.low_ensemble_directory,
            temperature=0.45,
            block_size=args.block_size,
        ),
        0.58: load_frozen_blocks(
            args.high_ensemble_directory,
            temperature=0.58,
            block_size=args.block_size,
        ),
    }
    stationarity_rows = (
        stationarity_by_temperature[0.45] + stationarity_by_temperature[0.58]
    )
    result = run_two_temperature_from_blocks(
        blocks_by_temperature,
        heldout_by_temperature,
        stationarity_rows,
        initial_realizations=args.initial_realizations,
        extended_realizations=args.extended_realizations,
        base_seed=args.base_seed,
        block_size=args.block_size,
    )
    _annotate_provenance(
        result,
        base_seed=args.base_seed,
        source_paths={
            0.45: (
                args.low_ensemble_directory,
                args.low_heldout_factorization,
                args.low_stationarity,
            ),
            0.58: (
                args.high_ensemble_directory,
                args.high_heldout_factorization,
                args.high_stationarity,
            ),
        },
    )
    return write_protocol_outputs(args.output_directory, result)


if __name__ == "__main__":
    main()
