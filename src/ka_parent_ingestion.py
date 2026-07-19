"""Parent-first import and frozen gate inputs for independent KA acquisitions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .ka_parent_completion import CLAIM_FLAGS, validate_completion_record
from .ka_prl_memory_closure import (
    CLOSED_CLAIM_FIELDS,
    CURVE_LIMITS,
    FROZEN_PROTOCOLS,
    build_claim_ledger,
)
from .ka_replicates import radial_multivariate_surrogate
from .ka_segment_splice import cumulative_observables_many_lags


BLOCK_SIZE = 20
WAVE_NUMBERS = np.asarray((2.0, 4.0, 7.25))
FROZEN_LAGS = tuple(
    int(value) for value in str(FROZEN_PROTOCOLS[0.45]["lag_grid"]).split(";")
)
SPECTRAL_BASE_SEED = 211003
SPECTRAL_REALIZATIONS = 8
SPECTRAL_ITERATIONS = 110
SIX_ABLATION_MODELS = frozenset(
    {
        "mean_rate_null",
        "one_step_jump_law",
        "two_point_path_spectrum",
        "static_particle_environment",
        "finite_exchange_environment",
        "full_candidate",
    }
)


def _closed_claims() -> dict[str, int]:
    return {
        **CLAIM_FLAGS,
        "microdynamic_closure_claim_allowed": 0,
        "thermodynamic_claim_allowed": 0,
    }


def _digest_rows(rows: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        [dict(row) for row in rows], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_parent_trajectory(
    trajectory: Mapping[str, object],
    *,
    timestep_tau: float = 0.001,
    frame_interval_tau: int = 1,
    calibration_time_tau: int = 5000,
    heldout_time_tau: int = 5000,
    block_size_tau: int = BLOCK_SIZE,
) -> dict[str, object]:
    """Validate and split one production trajectory at the frozen 5000-tau boundary."""

    timesteps = np.asarray(trajectory.get("timesteps"), dtype=np.int64)
    positions = np.asarray(trajectory.get("unwrapped_positions"), dtype=float)
    particle_types = np.asarray(trajectory.get("particle_types"), dtype=int)
    total_time = calibration_time_tau + heldout_time_tau
    expected_frames = total_time // frame_interval_tau + 1
    expected_step_interval = int(round(frame_interval_tau / timestep_tau))
    expected_timesteps = np.arange(expected_frames, dtype=np.int64) * expected_step_interval
    if (
        timesteps.shape != (expected_frames,)
        or not np.array_equal(timesteps, expected_timesteps)
        or positions.ndim != 3
        or positions.shape[0] != expected_frames
        or positions.shape[2] != 3
        or particle_types.shape != (positions.shape[1],)
        or np.any(~np.isfinite(positions))
    ):
        raise ValueError("trajectory does not match the frozen 0..10000 tau frame grid")
    if block_size_tau % frame_interval_tau != 0:
        raise ValueError("block size must be an integer number of saved frames")
    type_a = particle_types == 0
    if np.count_nonzero(type_a) < 2:
        raise ValueError("trajectory requires at least two Type-A particles")
    block_stride = block_size_tau // frame_interval_tau
    calibration_boundary = calibration_time_tau // frame_interval_tau
    heldout_boundary = total_time // frame_interval_tau

    def blocks(start: int, stop: int) -> np.ndarray:
        starts = np.arange(start, stop, block_stride, dtype=int)
        if len(starts) == 0 or starts[-1] + block_stride > stop:
            raise ValueError("split window does not contain complete frozen blocks")
        selected = positions[:, type_a]
        return np.transpose(
            selected[starts + block_stride] - selected[starts], (1, 0, 2)
        ).astype(float)

    return {
        "calibration_blocks": blocks(0, calibration_boundary),
        "heldout_blocks": blocks(calibration_boundary, heldout_boundary),
        "calibration_start_tau": 0.0,
        "calibration_stop_tau": float(calibration_time_tau),
        "heldout_start_tau": float(calibration_time_tau),
        "heldout_stop_tau": float(total_time),
        "shared_boundary_frame_count": 1,
        "heldout_observables_used_as_model_inputs": 0,
    }


def heldout_target_rows(
    heldout_blocks: np.ndarray,
    *,
    parent_id: str,
    replicate: int,
    trajectory_sha256: str,
    parent_manifest_sha256: str,
) -> list[dict[str, object]]:
    blocks = np.asarray(heldout_blocks, dtype=float)
    block_counts = tuple(lag // BLOCK_SIZE for lag in FROZEN_LAGS)
    if blocks.ndim != 3 or blocks.shape[1] < max(block_counts) or blocks.shape[2] != 3:
        raise ValueError("heldout path does not cover the frozen lag grid")
    observed = cumulative_observables_many_lags(
        blocks, block_counts=block_counts, wave_numbers=WAVE_NUMBERS
    )
    rows = []
    for lag, block_count in zip(FROZEN_LAGS, block_counts, strict=True):
        row = observed[block_count]
        rows.append(
            {
                "temperature": 0.45,
                "replicate": int(replicate),
                "parent_id": str(parent_id),
                "trajectory_sha256": str(trajectory_sha256),
                "parent_manifest_sha256": str(parent_manifest_sha256),
                "lag": lag,
                "block_size": BLOCK_SIZE,
                "observed_msd": float(row["msd"]),
                "observed_ngp": float(row["ngp"]),
                "observed_fs_k2": float(row["characteristic_k2"]),
                "observed_fs_k4": float(row["characteristic_k4"]),
                "observed_fs_k7p25": float(row["characteristic_k7p25"]),
                "heldout_path_used_in_prediction": 0,
                "heldout_observables_used_as_model_inputs": 0,
                **_closed_claims(),
            }
        )
    return rows


def stationarity_rows(
    calibration_blocks: np.ndarray,
    target_rows: Sequence[Mapping[str, object]],
    *,
    parent_id: str,
    replicate: int,
    trajectory_sha256: str,
    parent_manifest_sha256: str,
) -> list[dict[str, object]]:
    blocks = np.asarray(calibration_blocks, dtype=float)
    if blocks.ndim != 3 or blocks.shape[1] < 2 or blocks.shape[2] != 3:
        raise ValueError("stationarity requires particle-by-block-by-3 calibration paths")
    targets = {int(row["lag"]): row for row in target_rows}
    if set(targets) != set(FROZEN_LAGS):
        raise ValueError("stationarity targets do not cover the frozen lag grid")
    half = blocks.shape[1] // 2
    eligible_lags = tuple(lag for lag in FROZEN_LAGS if lag // BLOCK_SIZE <= half)
    block_counts = tuple(lag // BLOCK_SIZE for lag in eligible_lags)
    early = cumulative_observables_many_lags(
        blocks[:, :half], block_counts=block_counts, wave_numbers=WAVE_NUMBERS
    )
    late = cumulative_observables_many_lags(
        blocks[:, -half:], block_counts=block_counts, wave_numbers=WAVE_NUMBERS
    )
    output: list[dict[str, object]] = []
    for comparison, prediction_name, reference_name in (
        ("early_late", "early", "late"),
        ("early_heldout", "early", "heldout"),
        ("late_heldout", "late", "heldout"),
    ):
        msd_errors: list[float] = []
        ngp_errors: list[float] = []
        fs_errors: list[float] = []
        for lag, count in zip(eligible_lags, block_counts, strict=True):
            prediction = early[count] if prediction_name == "early" else late[count]
            if reference_name == "late":
                reference = late[count]
            else:
                target = targets[lag]
                reference = {
                    "msd": float(target["observed_msd"]),
                    "ngp": float(target["observed_ngp"]),
                    "characteristic_k2": float(target["observed_fs_k2"]),
                    "characteristic_k4": float(target["observed_fs_k4"]),
                    "characteristic_k7p25": float(target["observed_fs_k7p25"]),
                }
            reference_msd = float(reference["msd"])
            if reference_msd <= 0:
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
        output.append(
            {
                "temperature": 0.45,
                "replicate": int(replicate),
                "parent_id": str(parent_id),
                "trajectory_sha256": str(trajectory_sha256),
                "parent_manifest_sha256": str(parent_manifest_sha256),
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
                **_closed_claims(),
            }
        )
    return output


def _environment_crossing(blocks: np.ndarray) -> dict[str, float]:
    mobility = np.linalg.norm(np.asarray(blocks, dtype=float), axis=2)
    if mobility.ndim != 2 or min(mobility.shape) < 2:
        raise ValueError("environment lifetime requires particles and blocks")
    correlations = [1.0]
    lags = [float(BLOCK_SIZE)]
    maximum_lag = min(100, mobility.shape[1] - 1)
    for lag in range(1, maximum_lag + 1):
        first = mobility[:, :-lag].ravel()
        second = mobility[:, lag:].ravel()
        first = first - np.mean(first)
        second = second - np.mean(second)
        denominator = math.sqrt(float(np.dot(first, first) * np.dot(second, second)))
        correlation = float(np.dot(first, second) / denominator) if denominator > 0 else 0.0
        correlations.append(correlation)
        lags.append(float((lag + 1) * BLOCK_SIZE))
    target = 1.0 / math.e
    candidates = [index for index, value in enumerate(correlations) if value <= target]
    if not candidates:
        raise ValueError("calibration environment correlation has no frozen e-fold crossing")
    upper = candidates[0]
    if upper == 0 or correlations[upper] <= 0.0:
        crossing = lags[upper]
    else:
        lower = upper - 1
        fraction = (math.log(target) - math.log(correlations[lower])) / (
            math.log(correlations[upper]) - math.log(correlations[lower])
        )
        crossing = math.exp(
            math.log(lags[lower])
            + fraction * (math.log(lags[upper]) - math.log(lags[lower]))
        )
    return {
        "efold_crossing_time": float(crossing),
        "initial_correlation": 1.0,
        "target_correlation": target,
        "crossing_upper_lag": lags[upper],
    }


def _spectral_source_rows(
    blocks: np.ndarray,
    targets: Sequence[Mapping[str, object]],
    *,
    parent_id: str,
    replicate: int,
    trajectory_sha256: str,
    parent_manifest_sha256: str,
) -> list[dict[str, object]]:
    target_by_lag = {int(row["lag"]): row for row in targets}
    block_counts = tuple(lag // BLOCK_SIZE for lag in FROZEN_LAGS)
    output: list[dict[str, object]] = []
    for realization in range(SPECTRAL_REALIZATIONS):
        seed = int(
            (SPECTRAL_BASE_SEED + 1_000_003 * replicate + 97_409 * realization)
            % (2**63 - 1)
        )
        surrogate_result = radial_multivariate_surrogate(
            np.asarray(blocks, dtype=float),
            np.random.default_rng(seed),
            iteration_count=SPECTRAL_ITERATIONS,
        )
        observed = cumulative_observables_many_lags(
            np.asarray(surrogate_result["displacements"], dtype=float),
            block_counts=block_counts,
            wave_numbers=WAVE_NUMBERS,
        )
        for lag, block_count in zip(FROZEN_LAGS, block_counts, strict=True):
            prediction = observed[block_count]
            target = target_by_lag[lag]
            output.append(
                {
                    "model": "radial_multivariate_surrogate",
                    "temperature": 0.45,
                    "replicate": int(replicate),
                    "realization": realization,
                    "parent_id": str(parent_id),
                    "trajectory_sha256": str(trajectory_sha256),
                    "parent_manifest_sha256": str(parent_manifest_sha256),
                    "lag": lag,
                    "block_size": BLOCK_SIZE,
                    "predicted_msd": float(prediction["msd"]),
                    "predicted_ngp": float(prediction["ngp"]),
                    "predicted_fs_k2": float(prediction["characteristic_k2"]),
                    "predicted_fs_k4": float(prediction["characteristic_k4"]),
                    "predicted_fs_k7p25": float(prediction["characteristic_k7p25"]),
                    "observed_msd": float(target["observed_msd"]),
                    "observed_ngp": float(target["observed_ngp"]),
                    "observed_fs_k2": float(target["observed_fs_k2"]),
                    "observed_fs_k4": float(target["observed_fs_k4"]),
                    "observed_fs_k7p25": float(target["observed_fs_k7p25"]),
                    "surrogate_base_seed": SPECTRAL_BASE_SEED,
                    "surrogate_realizations": SPECTRAL_REALIZATIONS,
                    "surrogate_iteration_count": SPECTRAL_ITERATIONS,
                    "surrogate_seed": seed,
                    "heldout_path_used_in_prediction": 0,
                    "heldout_observables_used_as_model_inputs": 0,
                    "calibration_path_distribution_used": 1,
                    "macro_fit_parameter_count": 0,
                    **_closed_claims(),
                }
            )
    return output


def completion_import_blockers(
    acquisition_manifest: Mapping[str, object],
    completion_record: Mapping[str, object],
    *,
    acquisition_manifest_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    completion = validate_completion_record(completion_record)
    if completion["acquisition_manifest_sha256"] != acquisition_manifest_sha256:
        raise ValueError("completion record does not join the acquisition manifest")
    parents = {
        str(parent["parent_id"]): parent
        for parent in acquisition_manifest.get("parents", [])
        if isinstance(parent, Mapping)
    }
    jobs = {str(job["parent_id"]): job for job in completion["jobs"]}
    if not parents or set(jobs) != set(parents):
        raise ValueError("completion parents do not match the acquisition manifest")
    ledger = []
    for replicate, parent_id in enumerate(sorted(parents), start=1):
        parent = parents[parent_id]
        job = jobs[parent_id]
        ledger.append(
            {
                "temperature": float(acquisition_manifest["temperature"]),
                "replicate": replicate,
                "parent_id": parent_id,
                "velocity_seed": int(parent["velocity_seed"]),
                "independence_class": "independent_type_assignment_melt_cool_hold_history",
                "completion_state": str(job["completion_state"]),
                "exit_code": job.get("exit_code"),
                "exit_code_source": str(job["exit_code_source"]),
                "production_frame_count": int(job["production_frame_count"]),
                "trajectory_sha256": str(job["trajectory_sha256"]),
                "trajectory_size_bytes": int(job["trajectory_size_bytes"]),
                "parent_manifest_sha256": str(job["parent_manifest_sha256"]),
                "final_restart_sha256": str(job["final_restart_sha256"]),
                "error_match_count": int(job["error_match_count"]),
                "scientific_ingestion_allowed": int(job["scientific_ingestion_allowed"]),
                **_closed_claims(),
            }
        )
    available = sum(int(row["scientific_ingestion_allowed"]) for row in ledger)
    states = {str(row["completion_state"]) for row in ledger}
    if "blocked_missing_observed_exit_code" in states:
        blocker_state = "blocked_missing_observed_exit_code"
    elif any(value.startswith("failed_") for value in states):
        blocker_state = "failed_acquisition_completion"
    elif available < 3:
        blocker_state = "missing_independent_parents"
    else:
        blocker_state = "eligible_for_trajectory_import"
    blocker = {
        "temperature": 0.45,
        "evidence_role": "primary",
        "required_parent_count": 3,
        "available_parent_count": available,
        "missing_parent_count": max(3 - available, 0),
        "stationarity_pass": 0,
        "input_lineage_join_pass": 0,
        "blocker_state": blocker_state,
        "trajectory_transfer_state": "not_requested_completion_gate_closed",
        "authentication_blocker": "none_current_authenticated_session_not_persisted",
        "authentication_requirement_if_session_expires": (
            "requires_ssh_key_or_interactive_reauthentication"
        ),
        **_closed_claims(),
    }
    return ledger, blocker


def run_frozen_six_ablation_gate(
    prepared_parents: Sequence[Mapping[str, object]],
    *,
    realizations: int = 64,
    workers: int = 1,
    fixture_only: bool = False,
) -> dict[str, object]:
    """Run stationarity and the frozen gate with one row-unit per source parent."""

    if realizations != 64:
        raise ValueError("independent-parent acquisition gate is frozen at 64 realizations")
    if len(prepared_parents) != 3:
        raise ValueError("the primary gate requires exactly three prepared T=0.45 parents")
    targets: list[dict[str, object]] = []
    stationarity: list[dict[str, object]] = []
    environments: list[dict[str, object]] = []
    spectral: list[dict[str, object]] = []
    parent_ledger: list[dict[str, object]] = []
    for parent in sorted(prepared_parents, key=lambda row: int(row["replicate"])):
        replicate = int(parent["replicate"])
        parent_id = str(parent["parent_id"])
        trajectory_sha256 = str(parent["trajectory_sha256"])
        parent_manifest_sha256 = str(parent["parent_manifest_sha256"])
        local_targets = heldout_target_rows(
            np.asarray(parent["heldout_blocks"], dtype=float),
            parent_id=parent_id,
            replicate=replicate,
            trajectory_sha256=trajectory_sha256,
            parent_manifest_sha256=parent_manifest_sha256,
        )
        local_stationarity = stationarity_rows(
            np.asarray(parent["calibration_blocks"], dtype=float),
            local_targets,
            parent_id=parent_id,
            replicate=replicate,
            trajectory_sha256=trajectory_sha256,
            parent_manifest_sha256=parent_manifest_sha256,
        )
        crossing = _environment_crossing(
            np.asarray(parent["calibration_blocks"], dtype=float)
        )
        local_environment = {
            "temperature": 0.45,
            "temperature_group": "low",
            "replicate": replicate,
            "parent_id": parent_id,
            "trajectory_sha256": trajectory_sha256,
            "parent_manifest_sha256": parent_manifest_sha256,
            "block_size": BLOCK_SIZE,
            **crossing,
            "calibration_only": 1,
            "heldout_observables_used_as_model_inputs": 0,
            **_closed_claims(),
        }
        local_spectral = _spectral_source_rows(
            np.asarray(parent["calibration_blocks"], dtype=float),
            local_targets,
            parent_id=parent_id,
            replicate=replicate,
            trajectory_sha256=trajectory_sha256,
            parent_manifest_sha256=parent_manifest_sha256,
        )
        targets.extend(local_targets)
        stationarity.extend(local_stationarity)
        environments.append(local_environment)
        spectral.extend(local_spectral)
        stationarity_pass = int(
            all(int(row["curve_transfer_pass"]) == 1 for row in local_stationarity)
        )
        parent_ledger.append(
            {
                "temperature": 0.45,
                "replicate": replicate,
                "parent_id": parent_id,
                "calibration_parent_id": parent_id,
                "heldout_parent_id": parent_id,
                "calibration_heldout_same_parent": 1,
                "velocity_seed": int(parent["velocity_seed"]),
                "trajectory_sha256": trajectory_sha256,
                "trajectory_size_bytes": int(parent["trajectory_size_bytes"]),
                "trajectory_hash_scope": "complete_file",
                "parent_manifest_sha256": parent_manifest_sha256,
                "trajectory_length_tau": 10000.0,
                "calibration_start_tau": 0.0,
                "calibration_stop_tau": 5000.0,
                "heldout_start_tau": 5000.0,
                "heldout_stop_tau": 10000.0,
                "available_observables": "MSD;NGP;Fs_k2;Fs_k4;Fs_k7p25",
                "lag_grid": ";".join(map(str, FROZEN_LAGS)),
                "evidence_role": "fixture_only" if fixture_only else "primary",
                "recorded_independence_class": (
                    "synthetic_fixture" if fixture_only else "independent_type_assignment_melt_cool_hold_history"
                ),
                "stationarity_pass": stationarity_pass,
                "parent_stationarity_pass": stationarity_pass,
                "input_lineage_join_pass": 1,
                "parent_input_lineage_join_pass": 1,
                "parent_unit_contribution": 1,
                **_closed_claims(),
            }
        )
    shared_hashes = {
        "heldout_table_sha256": _digest_rows(targets),
        "environment_table_sha256": _digest_rows(environments),
        "spectral_table_sha256": _digest_rows(spectral),
    }
    blocks_by_restart: dict[int, np.ndarray] = {}
    lineage = []
    for parent in prepared_parents:
        replicate = int(parent["replicate"])
        parent_id = str(parent["parent_id"])
        trajectory_sha256 = str(parent["trajectory_sha256"])
        parent_manifest_sha256 = str(parent["parent_manifest_sha256"])
        blocks_by_restart[replicate] = np.asarray(parent["calibration_blocks"], dtype=float)
        lineage.append(
            {
                "temperature": 0.45,
                "replicate": replicate,
                "parent_id": parent_id,
                "source_doi": f"acquisition:{parent_id}",
                "source_sha256": trajectory_sha256,
                "source_frame_index": 0,
                "velocity_seed": int(parent["velocity_seed"]),
                "ensemble_manifest_sha256": parent_manifest_sha256,
                "replicate_manifest_sha256": parent_manifest_sha256,
                "trajectory_sha256": trajectory_sha256,
                "trajectory_size_bytes": int(parent["trajectory_size_bytes"]),
                "trajectory_hash_scope": "complete_file",
                **shared_hashes,
                "input_lineage_join_pass": 1,
            }
        )
    stationarity_by_parent = {
        str(row["parent_id"]): int(row["parent_stationarity_pass"])
        for row in parent_ledger
    }
    blockers = [
        {
            "temperature": 0.45,
            "evidence_role": "fixture_only" if fixture_only else "primary",
            "required_parent_count": 3,
            "available_parent_count": 3,
            "missing_parent_count": 0,
            "stationarity_eligible_parent_count": sum(stationarity_by_parent.values()),
            "fully_eligible_parent_count": sum(stationarity_by_parent.values()),
            "stationarity_pass": int(all(stationarity_by_parent.values())),
            "input_lineage_join_pass": 1,
            "blocker_state": (
                "fixture_only_claim_closed"
                if fixture_only
                else (
                    "eligible" if all(stationarity_by_parent.values()) else "stationarity_failed"
                )
            ),
            **_closed_claims(),
        }
    ]
    from scripts.analyze_ka_prl_memory_closure import (  # local import avoids a script dependency at module load
        build_diagnostic_tables,
        predict_correlated_parent_diagnostic,
    )

    realization_rows = predict_correlated_parent_diagnostic(
        blocks_by_restart=blocks_by_restart,
        target_rows=targets,
        crossing_rows=environments,
        spectral_source_rows=spectral,
        temperature=0.45,
        realizations=realizations,
        lineage_rows=lineage,
        workers=workers,
    )
    tables = build_diagnostic_tables(
        realization_rows=realization_rows,
        parent_ledger=parent_ledger,
        blockers=blockers,
        temperature=0.45,
        realizations=realizations,
    )
    if fixture_only:
        gate = dict(tables["gate"])
        gate["mechanism_state"] = "fixture_only_claim_closed"
        gate["fixture_only"] = 1
        for field in ("positive_memory_closure_claim_allowed", *CLOSED_CLAIM_FIELDS):
            gate[field] = 0.0
        tables["gate"] = gate
        tables["claim_ledger"] = build_claim_ledger(gate)
    else:
        tables["gate"]["fixture_only"] = 0
    return {
        "parent_ledger": parent_ledger,
        "blockers": blockers,
        "heldout_target_rows": targets,
        "stationarity_rows": stationarity,
        "environment_rows": environments,
        "spectral_rows": spectral,
        "input_lineage_rows": lineage,
        "realization_rows": realization_rows,
        **tables,
    }
