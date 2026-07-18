#!/usr/bin/env python3
"""Build restart-specific stationarity and end-to-end PRL input-lineage audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_segment_splice_gate import load_frozen_blocks  # noqa: E402
from ka_prl_memory_closure import (  # noqa: E402
    CURVE_LIMITS,
    FROZEN_PROTOCOLS,
    parent_identifier,
)
from ka_segment_splice import cumulative_observables_many_lags  # noqa: E402


BLOCK_SIZE = 20
WAVE_NUMBERS = np.asarray((2.0, 4.0, 7.25))
SPECTRAL_BASE_SEED = 211003
SPECTRAL_REALIZATIONS = 8
SPECTRAL_ITERATIONS = {0.45: 110, 0.58: 240}


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"cannot read CSV input: {path}") from error
    if not rows:
        raise ValueError(f"CSV input is empty: {path}")
    return rows


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty audit: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("audit rows must use one rectangular schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash input: {path}") from error
    return digest.hexdigest()


def _provenance_by_child(
    rows: Sequence[Mapping[str, object]], temperature: float
) -> dict[int, Mapping[str, object]]:
    output: dict[int, Mapping[str, object]] = {}
    for row in rows:
        if not math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12):
            continue
        restart = int(float(row["replicate"]))
        if restart in output:
            raise ValueError("provenance rows must be unique by temperature and restart")
        parent_identifier(row)
        output[restart] = row
    expected = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    if set(output) != expected:
        raise ValueError("provenance misses the frozen restart set")
    return output


def _targets_by_child(
    rows: Sequence[Mapping[str, object]], temperature: float
) -> dict[int, dict[int, Mapping[str, object]]]:
    frozen_lags = {
        int(value)
        for value in str(FROZEN_PROTOCOLS[temperature]["lag_grid"]).split(";")
    }
    output: dict[int, dict[int, Mapping[str, object]]] = {}
    for row in rows:
        if not math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12):
            continue
        restart = int(float(row["replicate"]))
        lag = int(float(row["lag"]))
        if lag not in frozen_lags:
            continue
        local = output.setdefault(restart, {})
        if lag in local:
            raise ValueError("held-out targets must be unique by restart and lag")
        local[lag] = row
    expected_restarts = {1, 2, 3} if temperature == 0.45 else {1, 2, 3, 4, 5}
    if set(output) != expected_restarts or any(set(local) != frozen_lags for local in output.values()):
        raise ValueError("held-out targets do not cover the frozen restart-lag grid")
    return output


def restart_stationarity_rows(
    *,
    blocks_by_restart: Mapping[int, np.ndarray],
    target_rows: Sequence[Mapping[str, object]],
    provenance_rows: Sequence[Mapping[str, object]],
    temperature: float,
) -> list[dict[str, object]]:
    """Score all three comparisons separately for each correlated child restart."""

    provenance = _provenance_by_child(provenance_rows, temperature)
    targets = _targets_by_child(target_rows, temperature)
    if set(blocks_by_restart) != set(provenance):
        raise ValueError("calibration blocks do not match the provenance restart set")
    output: list[dict[str, object]] = []
    comparisons = (
        ("early_late", "early", "late"),
        ("early_heldout", "early", "heldout"),
        ("late_heldout", "late", "heldout"),
    )
    for restart in sorted(provenance):
        blocks = np.asarray(blocks_by_restart[restart], dtype=float)
        if blocks.ndim != 3 or blocks.shape[2] != 3 or blocks.shape[1] < 2:
            raise ValueError("stationarity requires particle-by-block-by-3 calibration paths")
        half = blocks.shape[1] // 2
        eligible_lags = tuple(
            lag for lag in sorted(targets[restart]) if lag // BLOCK_SIZE <= half
        )
        if not eligible_lags:
            raise ValueError("no frozen lag fits a calibration half")
        block_counts = tuple(lag // BLOCK_SIZE for lag in eligible_lags)
        early = cumulative_observables_many_lags(
            blocks[:, :half], block_counts=block_counts, wave_numbers=WAVE_NUMBERS
        )
        late = cumulative_observables_many_lags(
            blocks[:, -half:], block_counts=block_counts, wave_numbers=WAVE_NUMBERS
        )
        for comparison, prediction_name, reference_name in comparisons:
            msd_errors: list[float] = []
            ngp_errors: list[float] = []
            fs_errors: list[float] = []
            for lag, block_count in zip(eligible_lags, block_counts, strict=True):
                prediction = early[block_count] if prediction_name == "early" else late[block_count]
                if reference_name == "early":
                    reference = early[block_count]
                elif reference_name == "late":
                    reference = late[block_count]
                else:
                    target = targets[restart][lag]
                    reference = {
                        "msd": float(target["observed_msd"]),
                        "ngp": float(target["observed_ngp"]),
                        "characteristic_k2": float(target["observed_fs_k2"]),
                        "characteristic_k4": float(target["observed_fs_k4"]),
                        "characteristic_k7p25": float(target["observed_fs_k7p25"]),
                    }
                reference_msd = float(reference["msd"])
                if reference_msd <= 0.0:
                    raise ValueError("stationarity reference MSD must be positive")
                msd_errors.append(abs(float(prediction["msd"]) / reference_msd - 1.0))
                ngp_errors.append(abs(float(prediction["ngp"]) - float(reference["ngp"])))
                for suffix in ("k2", "k4", "k7p25"):
                    fs_errors.append(
                        abs(
                            float(prediction[f"characteristic_{suffix}"])
                            - float(reference[f"characteristic_{suffix}"])
                        )
                    )
            maximum_msd = max(msd_errors)
            maximum_ngp = max(ngp_errors)
            maximum_fs = max(fs_errors)
            source = provenance[restart]
            output.append(
                {
                    "temperature": temperature,
                    "replicate": restart,
                    "parent_id": parent_identifier(source),
                    "source_doi": str(source["source_doi"]),
                    "source_sha256": str(source["source_sha256"]),
                    "source_frame_index": int(float(source["source_frame_index"])),
                    "velocity_seed": int(float(source["velocity_seed"])),
                    "comparison": comparison,
                    "lag_count": len(eligible_lags),
                    "lag_grid": ";".join(map(str, eligible_lags)),
                    "maximum_msd_relative_error": maximum_msd,
                    "maximum_ngp_absolute_error": maximum_ngp,
                    "maximum_fs_absolute_error": maximum_fs,
                    "msd_relative_error_tolerance": CURVE_LIMITS["msd"],
                    "ngp_absolute_error_tolerance": CURVE_LIMITS["ngp"],
                    "fs_absolute_error_tolerance": CURVE_LIMITS["fs"],
                    "curve_transfer_pass": int(
                        maximum_msd <= CURVE_LIMITS["msd"]
                        and maximum_ngp <= CURVE_LIMITS["ngp"]
                        and maximum_fs <= CURVE_LIMITS["fs"]
                    ),
                    "restart_is_independent_sample": 0,
                    "positive_memory_closure_claim_allowed": 0,
                    "microdynamic_closure_claim_allowed": 0,
                    "complete_microscopic_closure_claim_allowed": 0,
                    "spatial_facilitation_claim_allowed": 0,
                    "thermodynamic_claim_allowed": 0,
                    "thermodynamic_glass_transition_claim_allowed": 0,
                }
            )
    return output


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


def _manifest_matches(row: Mapping[str, object], manifest: Mapping[str, object]) -> bool:
    return (
        str(manifest.get("source_doi", "")) == str(row["source_doi"])
        and str(manifest.get("source_sha256", "")) == str(row["source_sha256"])
        and int(float(manifest.get("source_frame_index", -1)))
        == int(float(row["source_frame_index"]))
        and int(float(manifest.get("velocity_seed", -1)))
        == int(float(row["velocity_seed"]))
    )


def _rows_embed_parent(
    rows: Sequence[Mapping[str, object]], provenance: Mapping[str, object]
) -> bool:
    required = {
        "parent_id",
        "source_doi",
        "source_sha256",
        "source_frame_index",
        "velocity_seed",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        return False
    expected_parent = parent_identifier(provenance)
    for row in rows:
        if (
            str(row["parent_id"]) != expected_parent
            or str(row["source_doi"]) != str(provenance["source_doi"])
            or str(row["source_sha256"]) != str(provenance["source_sha256"])
            or int(float(row["source_frame_index"])) != int(float(provenance["source_frame_index"]))
            or int(float(row["velocity_seed"])) != int(float(provenance["velocity_seed"]))
        ):
            raise ValueError("embedded table lineage disagrees with parent provenance")
    return True


def input_lineage_rows(
    *,
    ensemble_directory: Path,
    heldout_path: Path,
    environment_path: Path,
    spectral_path: Path,
    provenance_rows: Sequence[Mapping[str, object]],
    temperature: float,
) -> list[dict[str, object]]:
    provenance = _provenance_by_child(provenance_rows, temperature)
    ensemble_manifest_path = ensemble_directory / "ensemble_manifest.json"
    ensemble_manifest = _json(ensemble_manifest_path)
    ensemble_sha = file_sha256(ensemble_manifest_path)
    if (
        str(ensemble_manifest.get("source_doi", ""))
        != str(next(iter(provenance.values()))["source_doi"])
        or str(ensemble_manifest.get("source_sha256", ""))
        != str(next(iter(provenance.values()))["source_sha256"])
    ):
        raise ValueError("ensemble manifest parent disagrees with provenance")
    manifest_specs = {
        int(spec["replicate"]): spec
        for spec in ensemble_manifest.get("replicates", [])
        if isinstance(spec, dict) and "replicate" in spec
    }
    if set(manifest_specs) != set(provenance):
        raise ValueError("ensemble manifest restart set disagrees with provenance")
    heldout_rows = read_rows(heldout_path)
    environment_rows = read_rows(environment_path)
    spectral_rows = read_rows(spectral_path)
    heldout_sha = file_sha256(heldout_path)
    environment_sha = file_sha256(environment_path)
    spectral_sha = file_sha256(spectral_path)
    output: list[dict[str, object]] = []
    group = "low" if temperature == 0.45 else "high"
    for restart, source in sorted(provenance.items()):
        spec = manifest_specs[restart]
        ensemble_join = (
            int(float(spec.get("source_frame_index", -1))) == int(float(source["source_frame_index"]))
            and int(float(spec.get("velocity_seed", -1))) == int(float(source["velocity_seed"]))
        )
        replicate_manifest_path = ensemble_directory / str(spec["directory"]) / "manifest.json"
        replicate_manifest = _json(replicate_manifest_path)
        replicate_join = _manifest_matches(source, replicate_manifest)
        trajectory_path = (
            ensemble_directory / str(spec["directory"]) / "trajectory.lammpstrj"
        )
        try:
            trajectory_size = trajectory_path.stat().st_size
        except OSError as error:
            raise ValueError(f"cannot stat trajectory input: {trajectory_path}") from error
        if trajectory_size <= 0:
            raise ValueError("trajectory input must be nonempty")
        local_heldout = [
            row
            for row in heldout_rows
            if math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12)
            and int(float(row["replicate"])) == restart
        ]
        local_environment = [
            row
            for row in environment_rows
            if str(row["temperature_group"]) == group
            and int(float(row["replicate"])) == restart
            and math.isclose(float(row["block_size"]), BLOCK_SIZE, abs_tol=1e-12)
        ]
        local_spectral = [
            row
            for row in spectral_rows
            if math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12)
            and int(float(row["replicate"])) == restart
            and str(row["model"]) == "radial_multivariate_surrogate"
        ]
        if not local_heldout or len(local_environment) != 1 or not local_spectral:
            raise ValueError("auxiliary input misses a frozen restart")
        spectral_metadata_pass = all(
            int(float(row.get("surrogate_base_seed", -1))) == SPECTRAL_BASE_SEED
            and int(float(row.get("surrogate_realizations", -1)))
            == SPECTRAL_REALIZATIONS
            and int(float(row.get("surrogate_iteration_count", -1)))
            == SPECTRAL_ITERATIONS[temperature]
            and math.isclose(float(row.get("block_size", -1)), BLOCK_SIZE, abs_tol=1e-12)
            and float(row.get("heldout_path_used_in_prediction", 1)) == 0.0
            and float(row.get("calibration_path_distribution_used", 0)) == 1.0
            and float(row.get("macro_fit_parameter_count", 1)) == 0.0
            for row in local_spectral
        )
        heldout_join = _rows_embed_parent(local_heldout, source)
        environment_join = _rows_embed_parent(local_environment, source)
        spectral_join = _rows_embed_parent(local_spectral, source)
        aggregate = all(
            (
                ensemble_join,
                replicate_join,
                heldout_join,
                environment_join,
                spectral_join,
                spectral_metadata_pass,
            )
        )
        output.append(
            {
                "temperature": temperature,
                "replicate": restart,
                "parent_id": parent_identifier(source),
                "source_doi": str(source["source_doi"]),
                "source_sha256": str(source["source_sha256"]),
                "source_frame_index": int(float(source["source_frame_index"])),
                "velocity_seed": int(float(source["velocity_seed"])),
                "ensemble_manifest_sha256": ensemble_sha,
                "ensemble_manifest_parent_join_pass": int(ensemble_join),
                "replicate_manifest_sha256": file_sha256(replicate_manifest_path),
                "replicate_manifest_parent_join_pass": int(replicate_join),
                "trajectory_sha256": file_sha256(trajectory_path),
                "trajectory_size_bytes": trajectory_size,
                "trajectory_hash_scope": "complete_file",
                "heldout_table_sha256": heldout_sha,
                "heldout_parent_join_pass": int(heldout_join),
                "environment_table_sha256": environment_sha,
                "environment_parent_join_pass": int(environment_join),
                "spectral_table_sha256": spectral_sha,
                "spectral_base_seed": SPECTRAL_BASE_SEED,
                "spectral_realization_count": SPECTRAL_REALIZATIONS,
                "spectral_iteration_count": SPECTRAL_ITERATIONS[temperature],
                "spectral_frozen_metadata_pass": int(spectral_metadata_pass),
                "spectral_parent_join_pass": int(spectral_join),
                "input_lineage_join_pass": int(aggregate),
                "lineage_blocker": (
                    "none"
                    if aggregate
                    else "derived_auxiliary_tables_lack_embedded_parent_identity"
                ),
                "positive_memory_closure_claim_allowed": 0,
                "microdynamic_closure_claim_allowed": 0,
                "complete_microscopic_closure_claim_allowed": 0,
                "spatial_facilitation_claim_allowed": 0,
                "thermodynamic_claim_allowed": 0,
                "thermodynamic_glass_transition_claim_allowed": 0,
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit restart-specific stationarity and PRL input lineage."
    )
    parser.add_argument("--provenance", type=Path, required=True)
    for name in ("low", "high"):
        parser.add_argument(f"--{name}-ensemble-directory", type=Path, required=True)
        parser.add_argument(f"--{name}-heldout-targets", type=Path, required=True)
        parser.add_argument(f"--{name}-spectral-rows", type=Path, required=True)
    parser.add_argument("--environment-crossings", type=Path, required=True)
    parser.add_argument("--output-stationarity", type=Path, required=True)
    parser.add_argument("--output-lineage", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    provenance = read_rows(args.provenance)
    stationarity: list[dict[str, object]] = []
    lineage: list[dict[str, object]] = []
    for temperature, prefix in ((0.45, "low"), (0.58, "high")):
        ensemble = getattr(args, f"{prefix}_ensemble_directory")
        targets = getattr(args, f"{prefix}_heldout_targets")
        spectral = getattr(args, f"{prefix}_spectral_rows")
        blocks = load_frozen_blocks(
            ensemble, temperature=temperature, block_size=BLOCK_SIZE
        )
        stationarity.extend(
            restart_stationarity_rows(
                blocks_by_restart=blocks,
                target_rows=read_rows(targets),
                provenance_rows=provenance,
                temperature=temperature,
            )
        )
        lineage.extend(
            input_lineage_rows(
                ensemble_directory=ensemble,
                heldout_path=targets,
                environment_path=args.environment_crossings,
                spectral_path=spectral,
                provenance_rows=provenance,
                temperature=temperature,
            )
        )
    write_rows(args.output_stationarity, stationarity)
    write_rows(args.output_lineage, lineage)


if __name__ == "__main__":
    main()
