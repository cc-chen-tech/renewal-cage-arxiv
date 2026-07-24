#!/usr/bin/env python3
"""Recompute the preregistered, parent-first PRL memory-closure gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_segment_splice_gate import load_frozen_blocks  # noqa: E402
from ka_prl_memory_closure import (  # noqa: E402
    ABLATION_MODELS,
    FROZEN_PROTOCOLS,
    audit_parent_provenance,
    build_claim_ledger,
    classify_correlated_parent_diagnostic,
    classify_memory_closure_gate,
    generate_ablation_path,
    generate_exchange_schedule,
    summarize_model_verdicts,
    summarize_parents,
    summarize_restarts,
    validate_realization_grid,
)
from ka_segment_splice import cumulative_observables_many_lags  # noqa: E402


BLOCK_SIZE = 20
WAVE_NUMBERS = np.asarray((2.0, 4.0, 7.25))
BASE_SEED = 20260718
SPECTRAL_MODEL = "two_point_path_spectrum"
UPPER_CONTROL = "contiguous_empirical_upper_control"


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read CSV input: {path}") from error
    if not rows:
        raise ValueError(f"CSV input is empty: {path}")
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash runtime input: {path}") from error
    return digest.hexdigest()


def _required_runtime_hash(
    rows: Sequence[Mapping[str, object]], field: str
) -> str:
    values = {str(row.get(field, "")) for row in rows}
    if len(values) != 1:
        raise ValueError(f"lineage rows disagree on {field}")
    value = values.pop()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"lineage field is not a SHA256: {field}")
    return value


def validate_runtime_lineage_hashes(
    *,
    ensemble_directory: Path,
    heldout_targets: Path,
    environment_crossings: Path,
    spectral_rows: Path,
    lineage_rows: Sequence[Mapping[str, object]],
    temperature: float,
) -> None:
    """Bind the actual full-mode files to the previously audited lineage rows."""

    selected = [
        row
        for row in lineage_rows
        if math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12)
    ]
    expected_restarts = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    by_restart = {int(float(row["replicate"])): row for row in selected}
    if len(by_restart) != len(selected) or set(by_restart) != expected_restarts:
        raise ValueError("runtime lineage misses or duplicates a frozen restart")
    shared_inputs = (
        (
            ensemble_directory / "ensemble_manifest.json",
            "ensemble_manifest_sha256",
            "ensemble manifest",
        ),
        (heldout_targets, "heldout_table_sha256", "heldout target"),
        (
            environment_crossings,
            "environment_table_sha256",
            "environment crossing",
        ),
        (spectral_rows, "spectral_table_sha256", "spectral"),
    )
    for path, field, label in shared_inputs:
        if _file_sha256(path) != _required_runtime_hash(selected, field):
            raise ValueError(f"runtime {label} hash disagrees with audited lineage")
    manifest_path = ensemble_directory / "ensemble_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        specs = {
            int(spec["replicate"]): str(spec["directory"])
            for spec in manifest["replicates"]
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("runtime ensemble manifest cannot resolve restart paths") from error
    if set(specs) != expected_restarts:
        raise ValueError("runtime ensemble manifest restart set is incomplete")
    for restart, row in by_restart.items():
        child_manifest = ensemble_directory / specs[restart] / "manifest.json"
        recorded = str(row.get("replicate_manifest_sha256", ""))
        if _file_sha256(child_manifest) != recorded:
            raise ValueError(
                f"runtime replicate manifest hash disagrees with lineage: restart={restart}"
            )
        trajectory = ensemble_directory / specs[restart] / "trajectory.lammpstrj"
        recorded_trajectory = str(row.get("trajectory_sha256", ""))
        recorded_size = int(float(row.get("trajectory_size_bytes", -1)))
        if str(row.get("trajectory_hash_scope", "")) != "complete_file":
            raise ValueError("runtime trajectory lineage must hash the complete file")
        try:
            runtime_size = trajectory.stat().st_size
        except OSError as error:
            raise ValueError(f"cannot stat runtime trajectory: restart={restart}") from error
        if runtime_size != recorded_size or _file_sha256(trajectory) != recorded_trajectory:
            raise ValueError(
                f"runtime trajectory hash or size disagrees with lineage: restart={restart}"
            )


def _canonical_value(value: object) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("artifact floats must be finite")
        return str(int(number)) if number.is_integer() else repr(number)
    return str(value)


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty artifact: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("artifact table rows must have one rectangular schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical_value(row[key]) for key in fields})
    temporary.replace(path)


def _audit(
    *, provenance: Path, stationarity: Path, input_lineage: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return audit_parent_provenance(
        provenance_rows=read_rows(provenance),
        stationarity_rows=read_rows(stationarity),
        lineage_rows=read_rows(input_lineage),
    )


def _model_seed(
    *, temperature: float, restart: int, model: str, realization: int
) -> int:
    payload = (
        f"{BASE_SEED}|{temperature:g}|{restart}|{model}|{realization}"
    ).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**63 - 1
    )


def _frozen_lags(temperature: float) -> tuple[int, ...]:
    return tuple(
        int(value) for value in str(FROZEN_PROTOCOLS[temperature]["lag_grid"]).split(";")
    )


def _heldout_targets(
    rows: Sequence[Mapping[str, object]], *, temperature: float
) -> dict[tuple[int, int], dict[str, float]]:
    frozen_lags = set(_frozen_lags(temperature))
    targets: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        if not math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12):
            continue
        restart = int(float(row["replicate"]))
        lag = int(float(row["lag"]))
        if lag not in frozen_lags:
            continue
        target = {
            "msd": float(row["observed_msd"]),
            "ngp": float(row["observed_ngp"]),
            "fs_k2": float(row["observed_fs_k2"]),
            "fs_k4": float(row["observed_fs_k4"]),
            "fs_k7p25": float(row["observed_fs_k7p25"]),
        }
        if any(not math.isfinite(value) for value in target.values()) or target["msd"] <= 0:
            raise ValueError("held-out target observables must be finite with positive MSD")
        key = (restart, lag)
        if key in targets:
            raise ValueError("held-out targets must be unique by restart and lag")
        targets[key] = target
    expected_restarts = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    expected = {
        (restart, lag) for restart in expected_restarts for lag in frozen_lags
    }
    if set(targets) != expected:
        raise ValueError("held-out targets do not cover the frozen restart-lag grid")
    return targets


def _environment_times(
    rows: Sequence[Mapping[str, object]], *, temperature: float
) -> dict[int, float]:
    group = "low" if temperature == 0.45 else "high"
    result: dict[int, float] = {}
    for row in rows:
        if str(row["temperature_group"]) != group or not math.isclose(
            float(row["block_size"]), BLOCK_SIZE, abs_tol=1e-12
        ):
            continue
        restart = int(float(row["replicate"]))
        value = float(row["efold_crossing_time"])
        if not math.isfinite(value) or value <= 0.0 or restart in result:
            raise ValueError("environment crossing rows must be unique and positive")
        result[restart] = value
    expected = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    if set(result) != expected:
        raise ValueError("environment crossing table misses a frozen restart")
    return result


def _prediction_rows_for_path(
    blocks: np.ndarray,
    *,
    temperature: float,
    restart: int,
    model: str,
    realization: int,
    targets: Mapping[tuple[int, int], Mapping[str, float]],
    information: Mapping[str, object],
    lineage: Mapping[str, object],
) -> list[dict[str, object]]:
    lags = _frozen_lags(temperature)
    block_counts = tuple(lag // BLOCK_SIZE for lag in lags)
    observables = cumulative_observables_many_lags(
        blocks,
        block_counts=block_counts,
        wave_numbers=WAVE_NUMBERS,
    )
    rows: list[dict[str, object]] = []
    for lag, block_count in zip(lags, block_counts, strict=True):
        prediction = observables[block_count]
        target = targets[(restart, lag)]
        predicted = {
            "msd": float(prediction["msd"]),
            "ngp": float(prediction["ngp"]),
            "fs_k2": float(prediction["characteristic_k2"]),
            "fs_k4": float(prediction["characteristic_k4"]),
            "fs_k7p25": float(prediction["characteristic_k7p25"]),
        }
        rows.append(
            {
                "temperature": temperature,
                "restart": restart,
                "parent_id": str(lineage["parent_id"]),
                "source_doi": str(lineage["source_doi"]),
                "source_sha256": str(lineage["source_sha256"]),
                "source_frame_index": int(float(lineage["source_frame_index"])),
                "velocity_seed": int(float(lineage["velocity_seed"])),
                "model": model,
                "realization": realization,
                "lag": lag,
                "block_size": BLOCK_SIZE,
                "predicted_msd": predicted["msd"],
                "predicted_ngp": predicted["ngp"],
                "predicted_fs_k2": predicted["fs_k2"],
                "predicted_fs_k4": predicted["fs_k4"],
                "predicted_fs_k7p25": predicted["fs_k7p25"],
                "target_msd": target["msd"],
                "target_ngp": target["ngp"],
                "target_fs_k2": target["fs_k2"],
                "target_fs_k4": target["fs_k4"],
                "target_fs_k7p25": target["fs_k7p25"],
                "realization_msd_relative_error": abs(
                    predicted["msd"] / target["msd"] - 1.0
                ),
                "realization_ngp_absolute_error": abs(
                    predicted["ngp"] - target["ngp"]
                ),
                "realization_fs_k2_absolute_error": abs(
                    predicted["fs_k2"] - target["fs_k2"]
                ),
                "realization_fs_k4_absolute_error": abs(
                    predicted["fs_k4"] - target["fs_k4"]
                ),
                "realization_fs_k7p25_absolute_error": abs(
                    predicted["fs_k7p25"] - target["fs_k7p25"]
                ),
                "mc_contribution_msd": predicted["msd"],
                "mc_contribution_ngp": predicted["ngp"],
                "mc_contribution_fs_k2": predicted["fs_k2"],
                "mc_contribution_fs_k4": predicted["fs_k4"],
                "mc_contribution_fs_k7p25": predicted["fs_k7p25"],
                "support_pass": 1.0,
                "environment_time_tau": float(
                    information.get("environment_time_tau", 0.0)
                ),
                "exchange_probability": float(
                    information.get("exchange_probability", 0.0)
                ),
                "environment_exchange_count": float(
                    information.get("environment_exchange_count", 0.0)
                ),
                "forced_terminal_exchange_count": float(
                    information.get("forced_terminal_exchange_count", 0.0)
                ),
                "model_seed": int(information.get("model_seed", -1)),
                "exchange_schedule_seed": int(
                    information.get("exchange_schedule_seed", -1)
                ),
                "exchange_schedule_sha256": str(
                    information.get("exchange_schedule_sha256", "not_applicable")
                ),
                "ensemble_manifest_sha256": str(
                    lineage.get("ensemble_manifest_sha256", "not_recorded")
                ),
                "replicate_manifest_sha256": str(
                    lineage.get("replicate_manifest_sha256", "not_recorded")
                ),
                "trajectory_sha256": str(lineage["trajectory_sha256"]),
                "trajectory_size_bytes": int(float(lineage["trajectory_size_bytes"])),
                "trajectory_hash_scope": str(lineage["trajectory_hash_scope"]),
                "heldout_table_sha256": str(
                    lineage.get("heldout_table_sha256", "not_recorded")
                ),
                "environment_table_sha256": str(
                    lineage.get("environment_table_sha256", "not_recorded")
                ),
                "spectral_table_sha256": str(
                    lineage.get("spectral_table_sha256", "not_recorded")
                ),
                "input_lineage_join_pass": float(lineage["input_lineage_join_pass"]),
                "heldout_path_used_in_prediction": 0.0,
                "heldout_observables_used_as_model_inputs": 0.0,
                "calibration_budget_equal_to_nulls": 1.0,
                "one_step_jump_law_retained": float(
                    information.get("one_step_jump_law_retained", 1.0)
                ),
                "two_point_path_spectrum_retained": float(
                    information.get("two_point_path_spectrum_retained", 0.0)
                ),
                "particle_identity_retained": float(
                    information.get("particle_identity_retained", 0.0)
                ),
                "static_particle_environment_retained": float(
                    information.get("static_particle_environment_retained", 0.0)
                ),
                "finite_exchange_environment_retained": float(
                    information.get("finite_exchange_environment_retained", 0.0)
                ),
                "ordered_path_memory_retained": float(
                    information.get("ordered_path_memory_retained", 0.0)
                ),
                "microdynamic_closure_claim_allowed": 0.0,
                "complete_microscopic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "thermodynamic_glass_transition_claim_allowed": 0.0,
            }
        )
    return rows


def _spectral_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    temperature: float,
    targets: Mapping[tuple[int, int], Mapping[str, float]],
    lineage_by_restart: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        if str(row["model"]) != "radial_multivariate_surrogate" or not math.isclose(
            float(row["temperature"]), temperature, abs_tol=1e-12
        ):
            continue
        restart = int(float(row["replicate"]))
        lineage = lineage_by_restart[restart]
        lag = int(float(row["lag"]))
        if lag not in _frozen_lags(temperature):
            continue
        target = targets[(restart, lag)]
        for observable, source in (
            ("msd", "observed_msd"),
            ("ngp", "observed_ngp"),
            ("fs_k2", "observed_fs_k2"),
            ("fs_k4", "observed_fs_k4"),
            ("fs_k7p25", "observed_fs_k7p25"),
        ):
            if not math.isclose(
                float(row[source]), target[observable], rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("spectral rows and held-out targets disagree")
        output.append(
            {
                "temperature": temperature,
                "restart": restart,
                "parent_id": str(lineage["parent_id"]),
                "source_doi": str(lineage["source_doi"]),
                "source_sha256": str(lineage["source_sha256"]),
                "source_frame_index": int(float(lineage["source_frame_index"])),
                "velocity_seed": int(float(lineage["velocity_seed"])),
                "model": SPECTRAL_MODEL,
                "realization": int(float(row["realization"])),
                "lag": lag,
                "block_size": BLOCK_SIZE,
                "predicted_msd": float(row["predicted_msd"]),
                "predicted_ngp": float(row["predicted_ngp"]),
                "predicted_fs_k2": float(row["predicted_fs_k2"]),
                "predicted_fs_k4": float(row["predicted_fs_k4"]),
                "predicted_fs_k7p25": float(row["predicted_fs_k7p25"]),
                "target_msd": target["msd"],
                "target_ngp": target["ngp"],
                "target_fs_k2": target["fs_k2"],
                "target_fs_k4": target["fs_k4"],
                "target_fs_k7p25": target["fs_k7p25"],
                "realization_msd_relative_error": abs(
                    float(row["predicted_msd"]) / target["msd"] - 1.0
                ),
                "realization_ngp_absolute_error": abs(
                    float(row["predicted_ngp"]) - target["ngp"]
                ),
                "realization_fs_k2_absolute_error": abs(
                    float(row["predicted_fs_k2"]) - target["fs_k2"]
                ),
                "realization_fs_k4_absolute_error": abs(
                    float(row["predicted_fs_k4"]) - target["fs_k4"]
                ),
                "realization_fs_k7p25_absolute_error": abs(
                    float(row["predicted_fs_k7p25"]) - target["fs_k7p25"]
                ),
                "mc_contribution_msd": float(row["predicted_msd"]),
                "mc_contribution_ngp": float(row["predicted_ngp"]),
                "mc_contribution_fs_k2": float(row["predicted_fs_k2"]),
                "mc_contribution_fs_k4": float(row["predicted_fs_k4"]),
                "mc_contribution_fs_k7p25": float(row["predicted_fs_k7p25"]),
                "support_pass": 1.0,
                "environment_time_tau": 0.0,
                "exchange_probability": 0.0,
                "environment_exchange_count": 0.0,
                "forced_terminal_exchange_count": 0.0,
                "model_seed": int(float(row.get("surrogate_base_seed", 211003))),
                "exchange_schedule_seed": -1,
                "exchange_schedule_sha256": "not_applicable",
                "ensemble_manifest_sha256": str(lineage["ensemble_manifest_sha256"]),
                "replicate_manifest_sha256": str(lineage["replicate_manifest_sha256"]),
                "trajectory_sha256": str(lineage["trajectory_sha256"]),
                "trajectory_size_bytes": int(float(lineage["trajectory_size_bytes"])),
                "trajectory_hash_scope": str(lineage["trajectory_hash_scope"]),
                "heldout_table_sha256": str(lineage["heldout_table_sha256"]),
                "environment_table_sha256": str(lineage["environment_table_sha256"]),
                "spectral_table_sha256": str(lineage["spectral_table_sha256"]),
                "input_lineage_join_pass": float(lineage["input_lineage_join_pass"]),
                "heldout_path_used_in_prediction": 0.0,
                "heldout_observables_used_as_model_inputs": 0.0,
                "calibration_budget_equal_to_nulls": 1.0,
                "one_step_jump_law_retained": 1.0,
                "two_point_path_spectrum_retained": 1.0,
                "particle_identity_retained": 0.0,
                "static_particle_environment_retained": 0.0,
                "finite_exchange_environment_retained": 0.0,
                "ordered_path_memory_retained": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "complete_microscopic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "thermodynamic_glass_transition_claim_allowed": 0.0,
            }
        )
    expected = (
        (3 if temperature == 0.45 else 5) * len(_frozen_lags(temperature)) * 8
    )
    if len(output) != expected:
        raise ValueError("spectral source does not contain the frozen eight-realization grid")
    return sorted(
        output,
        key=lambda row: (int(row["restart"]), int(row["realization"]), int(row["lag"])),
    )


def _predict_restart_rows(
    *,
    restart: int,
    blocks: np.ndarray,
    targets: Mapping[tuple[int, int], Mapping[str, float]],
    environment_time: float,
    temperature: float,
    realizations: int,
    lineage: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = _prediction_rows_for_path(
        blocks,
        temperature=temperature,
        restart=restart,
        model=UPPER_CONTROL,
        realization=0,
        targets=targets,
        information={
            "one_step_jump_law_retained": 1.0,
            "particle_identity_retained": 1.0,
            "static_particle_environment_retained": 1.0,
            "ordered_path_memory_retained": 1.0,
            "environment_time_tau": environment_time,
            "model_seed": -1,
        },
        lineage=lineage,
    )
    unpaired_models = sorted(
        ABLATION_MODELS.difference(
            {"finite_exchange_environment", "full_candidate"}
        )
    )
    for model in unpaired_models:
        for realization in range(realizations):
            model_seed = _model_seed(
                temperature=temperature,
                restart=restart,
                model=model,
                realization=realization,
            )
            generated, audit = generate_ablation_path(
                blocks,
                model=model,
                environment_time=environment_time,
                block_size=BLOCK_SIZE,
                rng=np.random.default_rng(model_seed),
            )
            audit = {
                **audit,
                "environment_time_tau": environment_time,
                "model_seed": model_seed,
            }
            rows.extend(
                _prediction_rows_for_path(
                    generated,
                    temperature=temperature,
                    restart=restart,
                    model=model,
                    realization=realization,
                    targets=targets,
                    information=audit,
                    lineage=lineage,
                )
            )
    for realization in range(realizations):
        schedule_seed = _model_seed(
            temperature=temperature,
            restart=restart,
            model="paired_exchange_schedule",
            realization=realization,
        )
        schedule = generate_exchange_schedule(
            blocks,
            environment_time=environment_time,
            block_size=BLOCK_SIZE,
            rng=np.random.default_rng(schedule_seed),
        )
        for model in ("finite_exchange_environment", "full_candidate"):
            model_seed = _model_seed(
                temperature=temperature,
                restart=restart,
                model=model,
                realization=realization,
            )
            generated, audit = generate_ablation_path(
                blocks,
                model=model,
                environment_time=environment_time,
                block_size=BLOCK_SIZE,
                rng=np.random.default_rng(model_seed),
                exchange_schedule=schedule,
            )
            audit = {
                **audit,
                "environment_time_tau": environment_time,
                "model_seed": model_seed,
                "exchange_schedule_seed": schedule_seed,
            }
            rows.extend(
                _prediction_rows_for_path(
                    generated,
                    temperature=temperature,
                    restart=restart,
                    model=model,
                    realization=realization,
                    targets=targets,
                    information=audit,
                    lineage=lineage,
                )
            )
    return rows


def predict_correlated_parent_diagnostic(
    *,
    blocks_by_restart: Mapping[int, np.ndarray],
    target_rows: Sequence[Mapping[str, object]],
    crossing_rows: Sequence[Mapping[str, object]],
    spectral_source_rows: Sequence[Mapping[str, object]],
    temperature: float,
    realizations: int,
    lineage_rows: Sequence[Mapping[str, object]],
    workers: int = 1,
) -> list[dict[str, object]]:
    """Run the frozen model family as a correlated-parent diagnostic only."""

    if realizations not in {16, 64}:
        raise ValueError("realizations must be one of the frozen values 16 or 64")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 3:
        raise ValueError("workers must be an integer from one to three")
    targets = _heldout_targets(target_rows, temperature=temperature)
    environment_times = _environment_times(crossing_rows, temperature=temperature)
    expected_restarts = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    if set(blocks_by_restart) != expected_restarts:
        raise ValueError("calibration blocks miss a frozen restart")
    lineage_by_restart: dict[int, Mapping[str, object]] = {}
    for row in lineage_rows:
        if not math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12):
            continue
        restart = int(float(row["replicate"]))
        if restart in lineage_by_restart:
            raise ValueError("input lineage must be unique by temperature and restart")
        lineage_by_restart[restart] = row
    if set(lineage_by_restart) != expected_restarts:
        raise ValueError("input lineage misses a frozen restart")
    rows: list[dict[str, object]] = []
    restart_arguments = [
        {
            "restart": restart,
            "blocks": np.asarray(blocks_by_restart[restart], dtype=float),
            "targets": targets,
            "environment_time": environment_times[restart],
            "temperature": temperature,
            "realizations": realizations,
            "lineage": lineage_by_restart[restart],
        }
        for restart in sorted(blocks_by_restart)
    ]
    if workers == 1:
        for arguments in restart_arguments:
            rows.extend(_predict_restart_rows(**arguments))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_predict_restart_rows, **arguments)
                for arguments in restart_arguments
            ]
            for future in futures:
                rows.extend(future.result())
    rows.extend(
        _spectral_rows(
            spectral_source_rows,
            temperature=temperature,
            targets=targets,
            lineage_by_restart=lineage_by_restart,
        )
    )
    return sorted(
        rows,
        key=lambda row: (
            float(row["temperature"]),
            str(row["model"]),
            int(row["restart"]),
            int(row["realization"]),
            int(row["lag"]),
        ),
    )


def _annotate_verdicts(
    verdicts: Sequence[Mapping[str, object]], *, gate_state: str
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "independent_parent_gate_state": gate_state,
            "correlated_restart_diagnostic_only": 1.0,
            "positive_memory_closure_claim_allowed": 0.0,
            "microdynamic_closure_claim_allowed": 0.0,
            "complete_microscopic_closure_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
            "thermodynamic_glass_transition_claim_allowed": 0.0,
        }
        for row in verdicts
    ]


def _annotate_evidence_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            **row,
            "positive_memory_closure_claim_allowed": 0.0,
            "microdynamic_closure_claim_allowed": 0.0,
            "complete_microscopic_closure_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
            "thermodynamic_glass_transition_claim_allowed": 0.0,
        }
        for row in rows
    ]


def precision_escalation_required(
    restart_summaries: Sequence[Mapping[str, object]],
) -> bool:
    """Return the frozen common-grid escalation decision from 16-cell summaries."""

    generated = [
        row for row in restart_summaries if str(row["model"]) in ABLATION_MODELS
    ]
    if not generated:
        raise ValueError("precision escalation requires generated-model summaries")
    return any(float(row["restart_precision_pass"]) == 0.0 for row in generated)


def warm_control_is_eligible(
    blocker: Mapping[str, object], verdicts: Sequence[Mapping[str, object]]
) -> bool:
    full = [
        row
        for row in verdicts
        if math.isclose(float(row["temperature"]), 0.58, abs_tol=1e-12)
        and str(row["model"]) == "full_candidate"
    ]
    parents = {str(row["parent_id"]) for row in full}
    return (
        int(float(blocker["missing_parent_count"])) == 0
        and float(blocker["stationarity_pass"]) == 1.0
        and float(blocker.get("input_lineage_join_pass", 0.0)) == 1.0
        and len(full) == 5
        and len(parents) == 5
        and all(float(row["curve_gate_pass"]) == 1.0 for row in full)
    )


def initial_grid_precision_trigger(
    realization_rows: Sequence[Mapping[str, object]],
    *,
    parent_ledger: Sequence[Mapping[str, object]],
    temperature: float,
    final_realizations: int,
) -> bool:
    """Reconstruct the frozen first-16 decision from either a 16 or 64 grid."""

    if final_realizations not in {16, 64}:
        raise ValueError("final realization count must be 16 or 64")
    if final_realizations == 64:
        initial_rows = [
            row
            for row in realization_rows
            if str(row["model"]) not in ABLATION_MODELS
            or int(float(row["realization"])) < 16
        ]
    else:
        initial_rows = list(realization_rows)
    validate_realization_grid(
        initial_rows,
        parent_ledger=parent_ledger,
        temperature=temperature,
        generated_realizations=16,
    )
    return precision_escalation_required(summarize_restarts(initial_rows))


def build_diagnostic_tables(
    *,
    realization_rows: Sequence[Mapping[str, object]],
    parent_ledger: Sequence[Mapping[str, object]],
    blockers: Sequence[Mapping[str, object]],
    temperature: float,
    realizations: int,
    allow_pending_precision_escalation: bool = False,
) -> dict[str, object]:
    validate_realization_grid(
        realization_rows,
        parent_ledger=parent_ledger,
        temperature=temperature,
        generated_realizations=realizations,
    )
    restart_summaries = summarize_restarts(realization_rows)
    precision_triggered = initial_grid_precision_trigger(
        realization_rows,
        parent_ledger=parent_ledger,
        temperature=temperature,
        final_realizations=realizations,
    )
    if realizations == 16 and precision_triggered and not allow_pending_precision_escalation:
        raise ValueError(
            "stored 16-realization grid violates the automatic precision escalation"
        )
    if realizations == 64 and not precision_triggered:
        raise ValueError(
            "stored 64-realization grid is not justified by the first-16 precision decision"
        )
    parent_summaries = summarize_parents(restart_summaries, parent_ledger)
    upper_controls = [
        row for row in parent_summaries if str(row["model"]) == UPPER_CONTROL
    ]
    model_parent_rows = [
        row for row in parent_summaries if str(row["model"]) != UPPER_CONTROL
    ]
    gate = classify_memory_closure_gate(
        parent_summaries=model_parent_rows,
        blockers=blockers,
        upper_control_parents=upper_controls,
    )
    diagnostic = classify_correlated_parent_diagnostic(
        parent_summaries=model_parent_rows,
        upper_control_parents=upper_controls,
        temperature=temperature,
    )
    verdicts = summarize_model_verdicts(parent_summaries)
    low_full = [
        row
        for row in verdicts
        if float(row["temperature"]) == 0.45 and str(row["model"]) == "full_candidate"
    ]
    run_blocker = next(
        row
        for row in blockers
        if math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12)
    )
    warm_control_eligible = float(
        temperature == 0.58 and warm_control_is_eligible(run_blocker, verdicts)
    )
    gate.update(
        {
            "run_temperature": temperature,
            "diagnostic_realizations": realizations,
            "initial_diagnostic_realizations": 16,
            "precision_escalation_triggered": float(precision_triggered),
            "correlated_parent_diagnostic_full_candidate_pass": float(
                bool(low_full)
                and all(float(row["curve_gate_pass"]) == 1.0 for row in low_full)
            ),
            "correlated_parent_diagnostic_only": 1.0,
            "heldout_observables_used_as_model_inputs": 0.0,
            "gate_or_claim_tuned_after_results": 0.0,
            "correlated_parent_diagnostic_state": diagnostic["diagnostic_state"],
            "correlated_parent_diagnostic_failure_localization": diagnostic[
                "diagnostic_failure_localization"
            ],
            "run_evidence_role": str(run_blocker["evidence_role"]),
            "run_stationarity_pass": float(run_blocker["stationarity_pass"]),
            "run_input_lineage_join_pass": float(
                run_blocker.get("input_lineage_join_pass", 0.0)
            ),
            "warm_control_eligible": warm_control_eligible,
            "warm_canary_only": float(temperature == 0.58 and not warm_control_eligible),
        }
    )
    return {
        "restart_summaries": _annotate_evidence_rows(restart_summaries),
        "parent_summaries": _annotate_evidence_rows(parent_summaries),
        "model_verdicts": _annotate_verdicts(
            verdicts, gate_state=str(gate["mechanism_state"])
        ),
        "gate": gate,
        "claim_ledger": build_claim_ledger(gate),
    }


def recompute_committed_memory_closure_tables(data_directory: Path) -> dict[str, object]:
    """Re-audit source tables and rebuild every committed table downstream of paths."""

    data = Path(data_directory)
    parent_ledger, blockers = audit_parent_provenance(
        provenance_rows=read_rows(
            data / "renewal_cage_ka_replicates_T058_T045_provenance.csv"
        ),
        stationarity_rows=read_rows(
            data / "renewal_cage_ka_prl_parent_stationarity.csv"
        ),
        lineage_rows=read_rows(data / "renewal_cage_ka_prl_input_lineage.csv"),
    )
    realization_rows = read_rows(
        data / "renewal_cage_ka_prl_memory_closure_restart_rows.csv"
    )
    temperatures = {float(row["temperature"]) for row in realization_rows}
    if len(temperatures) != 1:
        raise ValueError("committed diagnostic must contain exactly one temperature")
    temperature = temperatures.pop()
    full_realizations_by_restart: dict[int, set[int]] = {}
    for row in realization_rows:
        if str(row["model"]) != "full_candidate":
            continue
        full_realizations_by_restart.setdefault(
            int(float(row["restart"])), set()
        ).add(int(float(row["realization"])))
    realization_counts = {len(values) for values in full_realizations_by_restart.values()}
    if len(realization_counts) != 1 or not realization_counts:
        raise ValueError("full candidate realization grid is incomplete by restart")
    realizations = realization_counts.pop()
    if realizations not in {16, 64} or any(
        labels != set(range(realizations))
        for labels in full_realizations_by_restart.values()
    ):
        raise ValueError("full candidate realization labels violate the frozen grid")
    tables = build_diagnostic_tables(
        realization_rows=realization_rows,
        parent_ledger=parent_ledger,
        blockers=blockers,
        temperature=temperature,
        realizations=realizations,
    )
    return {
        "parent_ledger": parent_ledger,
        "blockers": blockers,
        "realization_rows": realization_rows,
        **tables,
    }


def recompute_committed_memory_closure_gate(data_directory: Path) -> dict[str, object]:
    """Compatibility wrapper returning the fully re-audited committed gate."""

    return recompute_committed_memory_closure_tables(data_directory)["gate"]


def write_svg(path: Path, verdicts: Sequence[Mapping[str, object]], gate: Mapping[str, object]) -> None:
    ordered = sorted(
        verdicts,
        key=lambda row: (str(row["model"]), str(row["parent_id"])),
    )
    width = 1120
    height = 150 + 24 * len(ordered)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#18212b}.title{font:700 22px ui-sans-serif,system-ui}.sub{font:14px ui-sans-serif,system-ui;fill:#4d5966}.pass{fill:#16794b}.fail{fill:#b42318}</style>',
        '<text x="28" y="38" class="title">PRL memory closure — parent-first gate</text>',
        f'<text x="28" y="66" class="sub">gate: {html.escape(str(gate["mechanism_state"]))}</text>',
        '<text x="28" y="90" class="sub">Rows below are correlated-parent diagnostics; restart averages cannot open the claim.</text>',
        '<text x="28" y="122">model</text><text x="520" y="122">parent</text><text x="800" y="122">worst child score</text><text x="1010" y="122">curve</text>',
    ]
    for index, row in enumerate(ordered):
        y = 148 + 24 * index
        passed = float(row["curve_gate_pass"]) == 1.0
        score = float(row["maximum_child_restart_higher_order_score"])
        lines.extend(
            [
                f'<text x="28" y="{y}">{html.escape(str(row["model"]))}</text>',
                f'<text x="520" y="{y}">{html.escape(str(row["parent_id"]))}</text>',
                f'<text x="800" y="{y}">{score:.6g}</text>',
                f'<text x="1010" y="{y}" class="{"pass" if passed else "fail"}">{"PASS" if passed else "FAIL"}</text>',
            ]
        )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen independent-parent PRL memory-closure gate."
    )
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--parent-stationarity", type=Path, required=True)
    parser.add_argument("--input-lineage", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--rebuild-derived", action="store_true")
    parser.add_argument("--restart-rows-input", type=Path)
    parser.add_argument("--run-temperature", type=float, choices=(0.45, 0.58))
    parser.add_argument("--ensemble-directory", type=Path)
    parser.add_argument("--heldout-targets", type=Path)
    parser.add_argument("--environment-crossings", type=Path)
    parser.add_argument("--spectral-rows", type=Path)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--output-parent-ledger", type=Path, required=True)
    parser.add_argument("--output-blockers", type=Path, required=True)
    parser.add_argument("--output-restart-rows", type=Path)
    parser.add_argument("--output-restart-summary", type=Path)
    parser.add_argument("--output-parent-summary", type=Path)
    parser.add_argument("--output-model-verdicts", type=Path)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-claim-ledger", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.base_seed != BASE_SEED:
        raise ValueError("the preregistered base seed is frozen at 20260718")
    if args.audit_only and args.rebuild_derived:
        raise ValueError("audit-only and rebuild-derived modes are mutually exclusive")

    if args.audit_only:
        if any(
            value is not None
            for value in (
                args.run_temperature,
                args.ensemble_directory,
                args.heldout_targets,
                args.environment_crossings,
                args.spectral_rows,
                args.output_restart_rows,
                args.output_restart_summary,
                args.output_parent_summary,
                args.output_model_verdicts,
                args.output_svg,
                args.restart_rows_input,
            )
        ):
            raise ValueError("audit-only mode cannot accept trajectory diagnostic paths")
        parent_ledger, blockers = _audit(
            provenance=args.provenance,
            stationarity=args.parent_stationarity,
            input_lineage=args.input_lineage,
        )
        gate = classify_memory_closure_gate(
            parent_summaries=[], blockers=blockers, upper_control_parents=[]
        )
        write_rows(args.output_parent_ledger, parent_ledger)
        write_rows(args.output_blockers, blockers)
        write_rows(args.output_gate, [gate])
        write_rows(args.output_claim_ledger, build_claim_ledger(gate))
        return

    if args.rebuild_derived:
        forbidden = (
            args.ensemble_directory,
            args.heldout_targets,
            args.environment_crossings,
            args.spectral_rows,
            args.output_restart_rows,
        )
        if any(value is not None for value in forbidden):
            raise ValueError("rebuild-derived mode cannot accept raw diagnostic paths")
        required_rebuild = {
            "restart_rows_input": args.restart_rows_input,
            "run_temperature": args.run_temperature,
            "output_restart_summary": args.output_restart_summary,
            "output_parent_summary": args.output_parent_summary,
            "output_model_verdicts": args.output_model_verdicts,
            "output_svg": args.output_svg,
        }
        missing_rebuild = [
            name for name, value in required_rebuild.items() if value is None
        ]
        if missing_rebuild:
            raise ValueError(
                f"rebuild-derived mode is missing: {';'.join(missing_rebuild)}"
            )
        parent_ledger, blockers = _audit(
            provenance=args.provenance,
            stationarity=args.parent_stationarity,
            input_lineage=args.input_lineage,
        )
        realization_rows = read_rows(args.restart_rows_input)
        full_labels = {
            int(float(row["realization"]))
            for row in realization_rows
            if str(row["model"]) == "full_candidate"
        }
        if full_labels != set(range(16)) and full_labels != set(range(64)):
            raise ValueError("stored full-candidate realization labels are incomplete")
        realizations = len(full_labels)
        tables = build_diagnostic_tables(
            realization_rows=realization_rows,
            parent_ledger=parent_ledger,
            blockers=blockers,
            temperature=float(args.run_temperature),
            realizations=realizations,
        )
        write_rows(args.output_parent_ledger, parent_ledger)
        write_rows(args.output_blockers, blockers)
        write_rows(args.output_restart_summary, tables["restart_summaries"])
        write_rows(args.output_parent_summary, tables["parent_summaries"])
        write_rows(args.output_model_verdicts, tables["model_verdicts"])
        write_rows(args.output_gate, [tables["gate"]])
        write_rows(args.output_claim_ledger, tables["claim_ledger"])
        write_svg(args.output_svg, tables["model_verdicts"], tables["gate"])
        return

    if args.restart_rows_input is not None:
        raise ValueError("full diagnostic mode cannot accept restart-rows-input")

    required = {
        "run_temperature": args.run_temperature,
        "ensemble_directory": args.ensemble_directory,
        "heldout_targets": args.heldout_targets,
        "environment_crossings": args.environment_crossings,
        "spectral_rows": args.spectral_rows,
        "output_restart_rows": args.output_restart_rows,
        "output_restart_summary": args.output_restart_summary,
        "output_parent_summary": args.output_parent_summary,
        "output_model_verdicts": args.output_model_verdicts,
        "output_svg": args.output_svg,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"full diagnostic mode is missing: {';'.join(missing)}")
    parent_ledger, blockers = _audit(
        provenance=args.provenance,
        stationarity=args.parent_stationarity,
        input_lineage=args.input_lineage,
    )
    temperature = float(args.run_temperature)
    lineage_rows = read_rows(args.input_lineage)
    validate_runtime_lineage_hashes(
        ensemble_directory=args.ensemble_directory,
        heldout_targets=args.heldout_targets,
        environment_crossings=args.environment_crossings,
        spectral_rows=args.spectral_rows,
        lineage_rows=lineage_rows,
        temperature=temperature,
    )
    blocks_by_restart = load_frozen_blocks(
        args.ensemble_directory,
        temperature=temperature,
        block_size=BLOCK_SIZE,
    )
    realization_rows = predict_correlated_parent_diagnostic(
        blocks_by_restart=blocks_by_restart,
        target_rows=read_rows(args.heldout_targets),
        crossing_rows=read_rows(args.environment_crossings),
        spectral_source_rows=read_rows(args.spectral_rows),
        temperature=temperature,
        realizations=16,
        lineage_rows=lineage_rows,
        workers=args.workers,
    )
    tables = build_diagnostic_tables(
        realization_rows=realization_rows,
        parent_ledger=parent_ledger,
        blockers=blockers,
        temperature=temperature,
        realizations=16,
        allow_pending_precision_escalation=True,
    )
    precision_triggered = precision_escalation_required(tables["restart_summaries"])
    if precision_triggered:
        realization_rows = predict_correlated_parent_diagnostic(
            blocks_by_restart=blocks_by_restart,
            target_rows=read_rows(args.heldout_targets),
            crossing_rows=read_rows(args.environment_crossings),
            spectral_source_rows=read_rows(args.spectral_rows),
            temperature=temperature,
            realizations=64,
            lineage_rows=lineage_rows,
            workers=args.workers,
        )
        tables = build_diagnostic_tables(
            realization_rows=realization_rows,
            parent_ledger=parent_ledger,
            blockers=blockers,
            temperature=temperature,
            realizations=64,
        )
    write_rows(args.output_parent_ledger, parent_ledger)
    write_rows(args.output_blockers, blockers)
    write_rows(args.output_restart_rows, realization_rows)
    write_rows(args.output_restart_summary, tables["restart_summaries"])
    write_rows(args.output_parent_summary, tables["parent_summaries"])
    write_rows(args.output_model_verdicts, tables["model_verdicts"])
    write_rows(args.output_gate, [tables["gate"]])
    write_rows(args.output_claim_ledger, tables["claim_ledger"])
    write_svg(args.output_svg, tables["model_verdicts"], tables["gate"])


if __name__ == "__main__":
    main()
