"""Parent-first gates for the preregistered PRL memory-closure candidate."""

from __future__ import annotations

import math
import hashlib
from collections.abc import Mapping, Sequence

import numpy as np


FROZEN_PROTOCOLS: dict[float, dict[str, object]] = {
    0.45: {
        "required_parent_count": 3,
        "calibration_time": 5000.0,
        "heldout_time": 5000.0,
        "production_time": 10000.0,
        "evidence_role": "primary",
        "lag_grid": "20;100;200;500;1000;2000;3000",
    },
    0.58: {
        "required_parent_count": 5,
        "calibration_time": 750.0,
        "heldout_time": 750.0,
        "production_time": 1500.0,
        "evidence_role": "canary_only",
        "lag_grid": "20;100;200;400;600",
    },
}

REQUIRED_STATIONARITY_COMPARISONS = frozenset(
    {"early_late", "early_heldout", "late_heldout"}
)
FROZEN_OBSERVABLES = "MSD;NGP;Fs_k2;Fs_k4;Fs_k7p25"

ABLATION_MODELS = frozenset(
    {
        "mean_rate_null",
        "one_step_jump_law",
        "static_particle_environment",
        "finite_exchange_environment",
        "full_candidate",
    }
)

PREDICTION_OBSERVABLES = (
    "msd",
    "ngp",
    "fs_k2",
    "fs_k4",
    "fs_k7p25",
)
CURVE_LIMITS = {"msd": 0.10, "ngp": 0.30, "fs": 0.03}
MC_LIMITS = {"msd": 0.01, "ngp": 0.03, "fs": 0.003}
REQUIRED_MECHANISM_ABLATIONS = (
    "mean_rate_null",
    "one_step_jump_law",
    "two_point_path_spectrum",
    "static_particle_environment",
    "finite_exchange_environment",
)
CLOSED_CLAIM_FIELDS = (
    "microdynamic_closure_claim_allowed",
    "complete_microscopic_closure_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
    "thermodynamic_glass_transition_claim_allowed",
)


def _as_float(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid provenance field: {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"provenance field must be finite: {key}")
    return value


def _as_int(row: Mapping[str, object], key: str) -> int:
    value = _as_float(row, key)
    integer = int(value)
    if value != integer:
        raise ValueError(f"provenance field must be an integer: {key}")
    return integer


def _as_bool(row: Mapping[str, object], key: str) -> bool:
    try:
        value = row[key]
    except KeyError as error:
        raise ValueError(f"missing provenance field: {key}") from error
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"missing or invalid provenance field: {key}")


def parent_identifier(row: Mapping[str, object]) -> str:
    """Return the parent-trajectory identifier, never a restart identifier."""

    try:
        doi = str(row["source_doi"]).strip()
        digest = str(row["source_sha256"]).strip().lower()
    except KeyError as error:
        raise ValueError("parent provenance requires DOI and SHA256") from error
    if not doi or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("parent provenance requires DOI and a hexadecimal SHA256")
    return f"{doi}:{digest}"


def _stationarity_pass(rows: Sequence[Mapping[str, object]]) -> bool:
    comparisons = [str(row.get("comparison", "")) for row in rows]
    if set(comparisons) != REQUIRED_STATIONARITY_COMPARISONS or len(comparisons) != len(
        REQUIRED_STATIONARITY_COMPARISONS
    ):
        raise ValueError("stationarity comparisons must be complete and unique")
    return all(_as_float(row, "curve_transfer_pass") == 1.0 for row in rows)


def _parent_blocker_state(
    *, missing_parent_count: int, stationarity_pass: bool, lineage_pass: bool
) -> str:
    if missing_parent_count and not stationarity_pass and not lineage_pass:
        return "stationarity_lineage_and_independent_parents"
    if missing_parent_count and not stationarity_pass:
        return "stationarity_and_independent_parents"
    if missing_parent_count and not lineage_pass:
        return "lineage_and_independent_parents"
    if missing_parent_count:
        return "missing_independent_parents"
    if not stationarity_pass and not lineage_pass:
        return "stationarity_and_lineage_failed"
    if not stationarity_pass:
        return "stationarity_failed"
    if not lineage_pass:
        return "input_lineage_failed"
    return "eligible"


def _child_key(row: Mapping[str, object]) -> tuple[float, int]:
    try:
        return float(row["temperature"]), int(float(row["replicate"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("parent audit rows must be restart-specific") from error


def _lineage_pass(
    row: Mapping[str, object], provenance: Mapping[str, object], parent_id: str
) -> bool:
    if str(row.get("parent_id", "")) != parent_id:
        raise ValueError("input lineage parent ID disagrees with provenance")
    for key in ("source_doi", "source_sha256"):
        if str(row.get(key, "")) != str(provenance[key]):
            raise ValueError(f"input lineage {key} disagrees with provenance")
    for key in ("source_frame_index", "velocity_seed"):
        if _as_int(row, key) != _as_int(provenance, key):
            raise ValueError(f"input lineage {key} disagrees with provenance")
    trajectory_digest = str(row.get("trajectory_sha256", ""))
    if len(trajectory_digest) != 64 or any(
        char not in "0123456789abcdef" for char in trajectory_digest
    ):
        raise ValueError("input lineage trajectory SHA256 is invalid")
    if _as_int(row, "trajectory_size_bytes") <= 0:
        raise ValueError("input lineage trajectory size must be positive")
    if str(row.get("trajectory_hash_scope", "")) != "complete_file":
        raise ValueError("input lineage must hash the complete trajectory file")
    component_fields = (
        "ensemble_manifest_parent_join_pass",
        "replicate_manifest_parent_join_pass",
        "heldout_parent_join_pass",
        "environment_parent_join_pass",
        "spectral_frozen_metadata_pass",
        "spectral_parent_join_pass",
    )
    components_pass = all(_as_bool(row, key) for key in component_fields)
    recorded = _as_bool(row, "input_lineage_join_pass")
    if recorded != components_pass:
        raise ValueError("input lineage aggregate flag disagrees with component joins")
    return recorded


def audit_parent_provenance(
    *,
    provenance_rows: Sequence[Mapping[str, object]],
    stationarity_rows: Sequence[Mapping[str, object]],
    lineage_rows: Sequence[Mapping[str, object]],
    protocol_by_temperature: Mapping[float, Mapping[str, object]] = FROZEN_PROTOCOLS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Audit restart provenance while counting unique source parents only once."""

    if not provenance_rows:
        raise ValueError("parent provenance rows must not be empty")
    provenance_keys = [_child_key(row) for row in provenance_rows]
    if len(set(provenance_keys)) != len(provenance_keys):
        raise ValueError("parent provenance rows must be unique by temperature and restart")
    stationarity_by_child: dict[tuple[float, int], list[Mapping[str, object]]] = {}
    for row in stationarity_rows:
        stationarity_by_child.setdefault(_child_key(row), []).append(row)
    lineage_by_child: dict[tuple[float, int], Mapping[str, object]] = {}
    for row in lineage_rows:
        key = _child_key(row)
        if key in lineage_by_child:
            raise ValueError("input lineage rows must be unique by temperature and restart")
        lineage_by_child[key] = row
    ledger: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    for temperature, protocol in sorted(protocol_by_temperature.items()):
        selected = [
            row
            for row in provenance_rows
            if math.isclose(_as_float(row, "temperature"), temperature, abs_tol=1e-12)
        ]
        if not selected:
            raise ValueError(f"missing provenance rows for T={temperature}")
        parent_ids = [parent_identifier(row) for row in selected]
        unique_parent_ids = sorted(set(parent_ids))
        child_stationarity: dict[int, bool] = {}
        child_lineage: dict[int, bool] = {}
        for row, parent_id in zip(selected, parent_ids, strict=True):
            replicate = _as_int(row, "replicate")
            key = (temperature, replicate)
            try:
                local_stationarity = stationarity_by_child[key]
            except KeyError as error:
                raise ValueError(
                    f"missing restart-specific stationarity rows for T={temperature}, restart={replicate}"
                ) from error
            if any(str(item.get("parent_id", "")) != parent_id for item in local_stationarity):
                raise ValueError("stationarity parent ID disagrees with provenance")
            child_stationarity[replicate] = _stationarity_pass(local_stationarity)
            try:
                local_lineage = lineage_by_child[key]
            except KeyError as error:
                raise ValueError(
                    f"missing input lineage row for T={temperature}, restart={replicate}"
                ) from error
            child_lineage[replicate] = _lineage_pass(local_lineage, row, parent_id)
        expected_child_keys = set(provenance_keys)
        unexpected_stationarity = set(stationarity_by_child).difference(expected_child_keys)
        unexpected_lineage = set(lineage_by_child).difference(expected_child_keys)
        if unexpected_stationarity or unexpected_lineage:
            raise ValueError("parent audit inputs contain an unknown restart")
        parent_stationarity = {
            parent_id: all(
                child_stationarity[_as_int(row, "replicate")]
                for row, local_parent in zip(selected, parent_ids, strict=True)
                if local_parent == parent_id
            )
            for parent_id in unique_parent_ids
        }
        parent_lineage = {
            parent_id: all(
                child_lineage[_as_int(row, "replicate")]
                for row, local_parent in zip(selected, parent_ids, strict=True)
                if local_parent == parent_id
            )
            for parent_id in unique_parent_ids
        }
        stationarity_pass = all(parent_stationarity.values())
        lineage_pass = all(parent_lineage.values())
        first_restart_by_parent = {
            parent_id: min(
                _as_int(row, "replicate")
                for row, local_parent in zip(selected, parent_ids, strict=True)
                if local_parent == parent_id
            )
            for parent_id in unique_parent_ids
        }
        required_parent_count = int(protocol["required_parent_count"])
        available_parent_count = len(unique_parent_ids)
        missing_parent_count = max(required_parent_count - available_parent_count, 0)
        production_time = float(protocol["production_time"])
        for row, parent_id in sorted(
            zip(selected, parent_ids, strict=True),
            key=lambda item: _as_int(item[0], "replicate"),
        ):
            replicate = _as_int(row, "replicate")
            recorded_production = _as_float(row, "production_time_tau")
            if not math.isclose(recorded_production, production_time, abs_tol=1e-12):
                raise ValueError("production length does not match the frozen protocol")
            ledger.append(
                {
                    "temperature": temperature,
                    "replicate": replicate,
                    "parent_id": parent_id,
                    "calibration_parent_id": parent_id,
                    "heldout_parent_id": parent_id,
                    "calibration_heldout_same_parent": 1.0,
                    "source_doi": str(row["source_doi"]),
                    "source_sha256": str(row["source_sha256"]),
                    "source_frame_index": _as_int(row, "source_frame_index"),
                    "velocity_seed": _as_int(row, "velocity_seed"),
                    "trajectory_length_tau": recorded_production,
                    "calibration_start_tau": 0.0,
                    "calibration_stop_tau": float(protocol["calibration_time"]),
                    "heldout_start_tau": float(protocol["calibration_time"]),
                    "heldout_stop_tau": recorded_production,
                    "stationarity_pass": float(child_stationarity[replicate]),
                    "parent_stationarity_pass": float(parent_stationarity[parent_id]),
                    "input_lineage_join_pass": float(child_lineage[replicate]),
                    "parent_input_lineage_join_pass": float(parent_lineage[parent_id]),
                    "trajectory_sha256": str(
                        lineage_by_child[(temperature, replicate)][
                            "trajectory_sha256"
                        ]
                    ),
                    "trajectory_size_bytes": _as_int(
                        lineage_by_child[(temperature, replicate)],
                        "trajectory_size_bytes",
                    ),
                    "trajectory_hash_scope": str(
                        lineage_by_child[(temperature, replicate)][
                            "trajectory_hash_scope"
                        ]
                    ),
                    "available_observables": FROZEN_OBSERVABLES,
                    "lag_grid": str(protocol["lag_grid"]),
                    "evidence_role": str(protocol["evidence_role"]),
                    "recorded_independence_class": str(
                        row.get("independence_class", "")
                    ),
                    "recorded_independently_prepared_parent_samples": float(
                        _as_bool(row, "independently_prepared_parent_samples")
                    ),
                    "audited_independence_class": "correlated_restarts_of_one_parent_trajectory",
                    "restart_is_independent_sample": 0.0,
                    "parent_unit_contribution": int(
                        replicate == first_restart_by_parent[parent_id]
                    ),
                    "positive_memory_closure_claim_allowed": 0.0,
                    "microdynamic_closure_claim_allowed": 0.0,
                    "complete_microscopic_closure_claim_allowed": 0.0,
                    "spatial_facilitation_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                    "thermodynamic_glass_transition_claim_allowed": 0.0,
                }
            )
        blocker_state = _parent_blocker_state(
            missing_parent_count=missing_parent_count,
            stationarity_pass=stationarity_pass,
            lineage_pass=lineage_pass,
        )
        stationarity_eligible_parent_count = sum(parent_stationarity.values())
        fully_eligible_parent_count = sum(
            parent_stationarity[parent_id] and parent_lineage[parent_id]
            for parent_id in unique_parent_ids
        )
        blockers.append(
            {
                "temperature": temperature,
                "evidence_role": str(protocol["evidence_role"]),
                "required_parent_count": required_parent_count,
                "available_parent_count": available_parent_count,
                "missing_parent_count": missing_parent_count,
                "restart_label_count": len(selected),
                "stationarity_pass": float(stationarity_pass),
                "stationarity_eligible_parent_count": stationarity_eligible_parent_count,
                "input_lineage_join_pass": float(lineage_pass),
                "fully_eligible_parent_count": fully_eligible_parent_count,
                "eligible_parent_deficit": max(
                    required_parent_count - fully_eligible_parent_count, 0
                ),
                "blocker_state": blocker_state,
                "required_trajectory_length_tau": production_time,
                "required_calibration_time_tau": float(protocol["calibration_time"]),
                "required_heldout_time_tau": float(protocol["heldout_time"]),
                "required_observables": FROZEN_OBSERVABLES,
                "shared_parent_resampling_can_satisfy": 0.0,
                "next_required_action": (
                    f"acquire_{missing_parent_count}_independent_parent_trajectories_requalify_stationarity_and_repair_lineage"
                    if missing_parent_count and not stationarity_pass and not lineage_pass
                    else f"acquire_{missing_parent_count}_independent_parent_trajectories_and_requalify_stationarity"
                    if missing_parent_count and not stationarity_pass
                    else f"acquire_{missing_parent_count}_independent_parent_trajectories_and_repair_lineage"
                    if missing_parent_count and not lineage_pass
                    else f"acquire_{missing_parent_count}_independent_parent_trajectories"
                    if missing_parent_count
                    else (
                        "replace_or_requalify_stationarity_and_repair_lineage"
                        if not stationarity_pass and not lineage_pass
                        else "replace_or_requalify_stationarity_control"
                        if not stationarity_pass
                        else "repair_input_lineage"
                        if not lineage_pass
                        else "run_preregistered_parent_first_gate"
                    )
                ),
                "positive_memory_closure_claim_allowed": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "complete_microscopic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
                "thermodynamic_glass_transition_claim_allowed": 0.0,
            }
        )
    return ledger, blockers


def _validated_block_paths(block_paths: np.ndarray) -> np.ndarray:
    blocks = np.asarray(block_paths, dtype=float)
    if blocks.ndim != 3 or blocks.shape[2] != 3:
        raise ValueError("block paths must have shape (particle, block, 3)")
    if blocks.shape[0] < 2 or blocks.shape[1] < 2:
        raise ValueError("block paths require at least two particles and blocks")
    if not np.all(np.isfinite(blocks)):
        raise ValueError("block paths must be finite")
    return blocks


def _different_particle(
    current: int, particle_count: int, rng: np.random.Generator
) -> int:
    draw = int(rng.integers(particle_count - 1))
    return draw + int(draw >= current)


def _exchange_schedule_digest(
    source_particle: np.ndarray, exchange_before: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(source_particle, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(exchange_before, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def generate_exchange_schedule(
    block_paths: np.ndarray,
    *,
    environment_time: float,
    block_size: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Precompute the finite-exchange process shared by the paired order ablation."""

    blocks = _validated_block_paths(block_paths)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not math.isfinite(environment_time) or environment_time <= 0.0:
        raise ValueError("environment_time must be positive and finite")
    if not math.isfinite(block_size) or block_size <= 0.0:
        raise ValueError("block_size must be positive and finite")
    particle_count, block_count, _ = blocks.shape
    exchange_probability = -math.expm1(-block_size / environment_time)
    source_particle = np.empty((particle_count, block_count), dtype=int)
    ordered_source_block = np.empty((particle_count, block_count), dtype=int)
    exchange_before = np.zeros((particle_count, block_count), dtype=bool)
    forced_exchange_before = np.zeros((particle_count, block_count), dtype=bool)
    exchange_count = 0
    forced_exchange_count = 0
    for target_particle in range(particle_count):
        current_particle = target_particle
        current_block = int(rng.integers(block_count))
        for output_block in range(block_count):
            source_particle[target_particle, output_block] = current_particle
            ordered_source_block[target_particle, output_block] = current_block
            if output_block == block_count - 1:
                continue
            forced_exchange = current_block == block_count - 1
            stochastic_exchange = rng.random() < exchange_probability
            if forced_exchange or stochastic_exchange:
                current_particle = _different_particle(
                    current_particle, particle_count, rng
                )
                current_block = int(rng.integers(block_count))
                exchange_before[target_particle, output_block + 1] = True
                forced_exchange_before[target_particle, output_block + 1] = (
                    forced_exchange
                )
                exchange_count += 1
                forced_exchange_count += int(forced_exchange)
            else:
                current_block += 1
    return {
        "source_particle": source_particle,
        "ordered_source_block": ordered_source_block,
        "exchange_before": exchange_before,
        "forced_exchange_before": forced_exchange_before,
        "environment_exchange_count": exchange_count,
        "forced_terminal_exchange_count": forced_exchange_count,
        "exchange_probability": exchange_probability,
        "exchange_schedule_sha256": _exchange_schedule_digest(
            source_particle, exchange_before
        ),
    }


def _validated_exchange_schedule(
    schedule: Mapping[str, object], *, particle_count: int, block_count: int
) -> dict[str, object]:
    try:
        source_particle = np.asarray(schedule["source_particle"], dtype=int)
        ordered_source_block = np.asarray(schedule["ordered_source_block"], dtype=int)
        exchange_before = np.asarray(schedule["exchange_before"], dtype=bool)
        forced_exchange_before = np.asarray(
            schedule["forced_exchange_before"], dtype=bool
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("exchange schedule is incomplete") from error
    shape = (particle_count, block_count)
    if any(
        array.shape != shape
        for array in (
            source_particle,
            ordered_source_block,
            exchange_before,
            forced_exchange_before,
        )
    ):
        raise ValueError("exchange schedule shape disagrees with block paths")
    if (
        np.any(source_particle < 0)
        or np.any(source_particle >= particle_count)
        or np.any(ordered_source_block < 0)
        or np.any(ordered_source_block >= block_count)
        or np.any(exchange_before[:, 0])
        or np.any(forced_exchange_before & ~exchange_before)
    ):
        raise ValueError("exchange schedule contains invalid indices or flags")
    if any(
        source_particle[target, 0] != target
        for target in range(particle_count)
    ):
        raise ValueError("exchange schedule must start from each target identity")
    for target in range(particle_count):
        for index in range(1, block_count):
            changed = source_particle[target, index] != source_particle[target, index - 1]
            if changed != bool(exchange_before[target, index]):
                raise ValueError("exchange flags disagree with source-particle changes")
            if not changed and ordered_source_block[target, index] != ordered_source_block[target, index - 1] + 1:
                raise ValueError("ordered schedule is not contiguous between exchanges")
    recorded_digest = str(schedule.get("exchange_schedule_sha256", ""))
    digest = _exchange_schedule_digest(source_particle, exchange_before)
    if recorded_digest != digest:
        raise ValueError("exchange schedule digest disagrees with its arrays")
    return {
        **schedule,
        "source_particle": source_particle,
        "ordered_source_block": ordered_source_block,
        "exchange_before": exchange_before,
        "forced_exchange_before": forced_exchange_before,
        "exchange_schedule_sha256": digest,
    }


def _information_audit(
    *,
    model: str,
    source_particle: np.ndarray,
    source_block: np.ndarray,
    exchange_count: int = 0,
    forced_exchange_count: int = 0,
    exchange_before: np.ndarray | None = None,
    exchange_probability: float = 0.0,
    exchange_schedule_sha256: str = "not_applicable",
) -> dict[str, object]:
    empirical_jump = float(model != "mean_rate_null")
    particle_identity = float(
        model
        in {
            "static_particle_environment",
            "finite_exchange_environment",
            "full_candidate",
        }
    )
    finite_exchange = float(
        model in {"finite_exchange_environment", "full_candidate"}
    )
    ordered = float(model == "full_candidate")
    if source_particle.shape[0] < 2 or source_particle.shape[1] == 0:
        shared_schedule = 0.0
    else:
        shared_schedule = float(
            all(
                np.array_equal(source_particle[0], source_particle[index])
                and np.array_equal(source_block[0], source_block[index])
                for index in range(1, source_particle.shape[0])
            )
        )
    return {
        "model": model,
        "one_step_jump_law_retained": empirical_jump,
        "particle_identity_retained": particle_identity,
        "static_particle_environment_retained": float(
            model == "static_particle_environment"
        ),
        "finite_exchange_environment_retained": finite_exchange,
        "ordered_path_memory_retained": ordered,
        "global_source_segment_schedule_preserved": shared_schedule,
        "environment_exchange_count": float(exchange_count),
        "forced_terminal_exchange_count": float(forced_exchange_count),
        "exchange_probability": float(exchange_probability),
        "exchange_schedule_sha256": exchange_schedule_sha256,
        "exchange_before": (
            np.zeros_like(source_particle, dtype=bool)
            if exchange_before is None
            else exchange_before
        ),
        "source_wrap_count": 0.0,
        "source_particle": source_particle,
        "source_block": source_block,
    }


def generate_ablation_path(
    block_paths: np.ndarray,
    *,
    model: str,
    environment_time: float,
    block_size: float,
    rng: np.random.Generator,
    exchange_schedule: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate one frozen-budget ablation path from calibration blocks only.

    The particle-conditioned models use source-particle identity as the persistent
    environment state.  Ordered paths advance contiguously and force an exchange
    at a source endpoint instead of wrapping that source trajectory.
    """

    blocks = _validated_block_paths(block_paths)
    if model not in ABLATION_MODELS:
        raise ValueError(f"unknown preregistered ablation model: {model}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not math.isfinite(environment_time) or environment_time <= 0.0:
        raise ValueError("environment_time must be positive and finite")
    if not math.isfinite(block_size) or block_size <= 0.0:
        raise ValueError("block_size must be positive and finite")

    particle_count, block_count, _ = blocks.shape
    empty_sources = np.full((particle_count, block_count), -1, dtype=int)
    if model == "mean_rate_null":
        mean_squared_step = float(np.mean(np.sum(blocks * blocks, axis=2)))
        generated = rng.normal(
            scale=math.sqrt(mean_squared_step / 3.0), size=blocks.shape
        )
        return generated, _information_audit(
            model=model,
            source_particle=empty_sources,
            source_block=empty_sources.copy(),
        )

    if model == "one_step_jump_law":
        flat_indices = rng.integers(
            particle_count * block_count,
            size=(particle_count, block_count),
        )
        source_particle = flat_indices // block_count
        source_block = flat_indices % block_count
        generated = blocks[source_particle, source_block]
        return generated, _information_audit(
            model=model,
            source_particle=source_particle,
            source_block=source_block,
        )

    if model == "static_particle_environment":
        source_particle = np.broadcast_to(
            np.arange(particle_count, dtype=int)[:, None],
            (particle_count, block_count),
        ).copy()
        source_block = rng.integers(
            block_count, size=(particle_count, block_count)
        )
        generated = blocks[source_particle, source_block]
        return generated, _information_audit(
            model=model,
            source_particle=source_particle,
            source_block=source_block,
        )

    if exchange_schedule is None:
        exchange_schedule = generate_exchange_schedule(
            blocks,
            environment_time=environment_time,
            block_size=block_size,
            rng=rng,
        )
    schedule = _validated_exchange_schedule(
        exchange_schedule, particle_count=particle_count, block_count=block_count
    )
    source_particle = np.asarray(schedule["source_particle"], dtype=int)
    exchange_before = np.asarray(schedule["exchange_before"], dtype=bool)
    if model == "full_candidate":
        source_block = np.asarray(schedule["ordered_source_block"], dtype=int)
    else:
        source_block = rng.integers(block_count, size=(particle_count, block_count))
        ordered_start = np.asarray(schedule["ordered_source_block"], dtype=int)
        source_block[:, 0] = ordered_start[:, 0]
        source_block[exchange_before] = ordered_start[exchange_before]
    generated = blocks[source_particle, source_block]

    return generated, _information_audit(
        model=model,
        source_particle=source_particle,
        source_block=source_block,
        exchange_count=int(schedule["environment_exchange_count"]),
        forced_exchange_count=int(schedule["forced_terminal_exchange_count"]),
        exchange_before=exchange_before,
        exchange_probability=float(schedule["exchange_probability"]),
        exchange_schedule_sha256=str(schedule["exchange_schedule_sha256"]),
    )


def _group_rows(
    rows: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> list[tuple[tuple[object, ...], list[Mapping[str, object]]]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        try:
            key = tuple(row[name] for name in keys)
        except KeyError as error:
            raise ValueError(f"missing grouping field: {error.args[0]}") from error
        groups.setdefault(key, []).append(row)
    return sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))


def _finite_values(rows: Sequence[Mapping[str, object]], key: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid prediction field: {key}") from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"prediction field must be finite: {key}")
    return values


def _constant_target(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = _finite_values(rows, key)
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError(f"target changed within aggregation unit: {key}")
    return float(values[0])


def _add_curve_errors(row: dict[str, object]) -> None:
    target_msd = float(row["target_msd"])
    msd_delta = abs(float(row["predicted_msd"]) - target_msd)
    row["msd_relative_error"] = (
        msd_delta / abs(target_msd)
        if target_msd != 0.0
        else (0.0 if msd_delta == 0.0 else math.inf)
    )
    row["ngp_absolute_error"] = abs(
        float(row["predicted_ngp"]) - float(row["target_ngp"])
    )
    for observable in ("fs_k2", "fs_k4", "fs_k7p25"):
        row[f"absolute_error_{observable}"] = abs(
            float(row[f"predicted_{observable}"])
            - float(row[f"target_{observable}"])
        )


def _msd_mc_relative_se(row: Mapping[str, object]) -> float:
    target_msd = abs(float(row["target_msd"]))
    if not math.isfinite(target_msd) or target_msd <= 0.0:
        raise ValueError("target MSD must be positive for relative Monte Carlo SE")
    value = float(row.get("mc_se_msd", 0.0)) / target_msd
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("MSD Monte Carlo SE must be finite and nonnegative")
    return value


def summarize_restarts(
    realization_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Average stochastic realizations inside each child restart first."""

    if not realization_rows:
        return []
    output: list[dict[str, object]] = []
    keys = ("temperature", "restart", "model", "lag")
    for group_key, rows in _group_rows(realization_rows, keys):
        summary = dict(zip(keys, group_key, strict=True))
        realization_labels = {int(float(row["realization"])) for row in rows}
        if len(realization_labels) != len(rows):
            raise ValueError("realization labels must be unique within a restart")
        summary["realization_count"] = len(rows)
        for observable in PREDICTION_OBSERVABLES:
            predictions = _finite_values(rows, f"predicted_{observable}")
            summary[f"predicted_{observable}"] = float(np.mean(predictions))
            summary[f"mc_se_{observable}"] = (
                float(np.std(predictions, ddof=1) / math.sqrt(len(predictions)))
                if len(predictions) > 1
                else 0.0
            )
            summary[f"target_{observable}"] = _constant_target(
                rows, f"target_{observable}"
            )
        summary["support_pass"] = min(
            float(row.get("support_pass", 1.0)) for row in rows
        )
        _add_curve_errors(summary)
        summary["mc_relative_se_msd"] = _msd_mc_relative_se(summary)
        output.append(summary)
    for _, rows in _group_rows(output, ("temperature", "restart", "model")):
        restart_curve_pass = float(curve_pass(rows))
        restart_maximum_higher_order_score = max(
            higher_order_score(row) for row in rows
        )
        restart_higher_order_pass = float(
            all(higher_order_score(row) <= 1.0 for row in rows)
        )
        restart_precision_pass = float(
            all(
                _msd_mc_relative_se(row) <= MC_LIMITS["msd"]
                and float(row["mc_se_ngp"]) <= MC_LIMITS["ngp"]
                and max(
                    float(value)
                    for key, value in row.items()
                    if str(key).startswith("mc_se_fs_k")
                )
                <= MC_LIMITS["fs"]
                for row in rows
            )
        )
        for row in rows:
            row["restart_curve_gate_pass"] = restart_curve_pass
            row["restart_higher_order_gate_pass"] = restart_higher_order_pass
            row["restart_precision_pass"] = restart_precision_pass
            row["restart_maximum_higher_order_score"] = (
                restart_maximum_higher_order_score
            )
    return output


def validate_realization_grid(
    realization_rows: Sequence[Mapping[str, object]],
    *,
    parent_ledger: Sequence[Mapping[str, object]],
    temperature: float,
    generated_realizations: int,
) -> None:
    """Reject any missing, duplicate, truncated, or provenance-mismatched cell."""

    if generated_realizations not in {16, 64}:
        raise ValueError("generated realization grid must be 16 or 64")
    frozen_lags = {
        int(value)
        for value in str(FROZEN_PROTOCOLS[temperature]["lag_grid"]).split(";")
    }
    ledger_by_restart: dict[int, Mapping[str, object]] = {}
    for row in parent_ledger:
        if not math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12):
            continue
        restart = int(float(row["replicate"]))
        parent_id = str(row["parent_id"])
        if restart in ledger_by_restart and str(ledger_by_restart[restart]["parent_id"]) != parent_id:
            raise ValueError("parent ledger maps a restart to multiple parents")
        ledger_by_restart[restart] = row
    if not ledger_by_restart:
        raise ValueError("realization grid has no parent ledger rows")
    expected_models = {
        **{model: set(range(generated_realizations)) for model in ABLATION_MODELS},
        "two_point_path_spectrum": set(range(8)),
        "contiguous_empirical_upper_control": {0},
    }
    expected = {
        (restart, model, realization, lag)
        for restart in ledger_by_restart
        for model, realizations in expected_models.items()
        for realization in realizations
        for lag in frozen_lags
    }
    observed: dict[tuple[int, str, int, int], Mapping[str, object]] = {}
    finite_schedule: dict[tuple[int, int], tuple[str, float, float]] = {}
    full_schedule: dict[tuple[int, int], tuple[str, float, float]] = {}
    required_finite_fields = (
        "predicted_msd",
        "predicted_ngp",
        "predicted_fs_k2",
        "predicted_fs_k4",
        "predicted_fs_k7p25",
        "target_msd",
        "target_ngp",
        "target_fs_k2",
        "target_fs_k4",
        "target_fs_k7p25",
        "realization_msd_relative_error",
        "realization_ngp_absolute_error",
        "realization_fs_k2_absolute_error",
        "realization_fs_k4_absolute_error",
        "realization_fs_k7p25_absolute_error",
        "mc_contribution_msd",
        "mc_contribution_ngp",
        "mc_contribution_fs_k2",
        "mc_contribution_fs_k4",
        "mc_contribution_fs_k7p25",
    )
    for row in realization_rows:
        if not math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12):
            raise ValueError("realization artifact mixes temperatures")
        restart = int(float(row["restart"]))
        model = str(row["model"])
        realization = int(float(row["realization"]))
        lag = int(float(row["lag"]))
        key = (restart, model, realization, lag)
        if key in observed:
            raise ValueError("realization grid contains a duplicate cell")
        observed[key] = row
        if restart not in ledger_by_restart or str(row.get("parent_id", "")) != str(ledger_by_restart[restart]["parent_id"]):
            raise ValueError("realization row parent provenance disagrees with ledger")
        ledger_row = ledger_by_restart[restart]
        if (
            str(row.get("trajectory_sha256", ""))
            != str(ledger_row.get("trajectory_sha256", ""))
            or int(float(row.get("trajectory_size_bytes", -1)))
            != int(float(ledger_row.get("trajectory_size_bytes", -2)))
            or str(row.get("trajectory_hash_scope", "")) != "complete_file"
        ):
            raise ValueError("realization row trajectory lineage disagrees with ledger")
        for field in required_finite_fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"realization field must be finite: {field}")
        for field in CLOSED_CLAIM_FIELDS:
            if float(row.get(field, math.nan)) != 0.0:
                raise ValueError(f"realization claim boundary must remain closed: {field}")
        if model in {"finite_exchange_environment", "full_candidate"}:
            local = (
                str(row["exchange_schedule_sha256"]),
                float(row["environment_exchange_count"]),
                float(row["forced_terminal_exchange_count"]),
            )
            schedule_key = (restart, realization)
            destination = finite_schedule if model == "finite_exchange_environment" else full_schedule
            if schedule_key in destination and destination[schedule_key] != local:
                raise ValueError("paired exchange schedule changes across lags")
            destination[schedule_key] = local
    if set(observed) != expected:
        missing = len(expected.difference(observed))
        extra = len(set(observed).difference(expected))
        raise ValueError(
            f"realization grid is incomplete or contains extra cells: missing={missing}, extra={extra}"
        )
    if finite_schedule != full_schedule:
        raise ValueError("finite and full candidates do not share the exchange schedule")


def summarize_parents(
    restart_summaries: Sequence[Mapping[str, object]],
    parent_ledger: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Average child restarts inside their source parent before scoring errors."""

    parent_by_child: dict[tuple[float, int], str] = {}
    for row in parent_ledger:
        key = (float(row["temperature"]), int(float(row["replicate"])))
        parent = str(row["parent_id"])
        if key in parent_by_child and parent_by_child[key] != parent:
            raise ValueError("child restart maps to multiple parents")
        parent_by_child[key] = parent

    parent_rows: list[dict[str, object]] = []
    for row in restart_summaries:
        child_key = (float(row["temperature"]), int(float(row["restart"])))
        try:
            parent_id = parent_by_child[child_key]
        except KeyError as error:
            raise ValueError("restart summary is missing parent provenance") from error
        enriched = dict(row)
        enriched["parent_id"] = parent_id
        parent_rows.append(enriched)

    output: list[dict[str, object]] = []
    keys = ("temperature", "parent_id", "model", "lag")
    for group_key, rows in _group_rows(parent_rows, keys):
        summary = dict(zip(keys, group_key, strict=True))
        summary["child_restart_count"] = len(rows)
        summary["realization_count"] = sum(
            int(float(row["realization_count"])) for row in rows
        )
        for observable in PREDICTION_OBSERVABLES:
            predictions = _finite_values(rows, f"predicted_{observable}")
            targets = _finite_values(rows, f"target_{observable}")
            child_se = _finite_values(rows, f"mc_se_{observable}")
            summary[f"predicted_{observable}"] = float(np.mean(predictions))
            summary[f"target_{observable}"] = float(np.mean(targets))
            summary[f"mc_se_{observable}"] = float(
                math.sqrt(float(np.sum(child_se * child_se))) / len(rows)
            )
        summary["support_pass"] = min(float(row["support_pass"]) for row in rows)
        summary["all_child_restart_curve_gate_pass"] = min(
            float(row.get("restart_curve_gate_pass", 1.0)) for row in rows
        )
        summary["all_child_restart_higher_order_gate_pass"] = min(
            float(row.get("restart_higher_order_gate_pass", 1.0)) for row in rows
        )
        summary["all_child_restart_precision_pass"] = min(
            float(row.get("restart_precision_pass", 1.0)) for row in rows
        )
        summary["failed_child_restart_count"] = sum(
            float(row.get("restart_curve_gate_pass", 1.0)) == 0.0 for row in rows
        )
        summary["maximum_child_restart_higher_order_score"] = max(
            float(row.get("restart_maximum_higher_order_score", 0.0))
            for row in rows
        )
        _add_curve_errors(summary)
        summary["mc_relative_se_msd"] = _msd_mc_relative_se(summary)
        output.append(summary)
    return output


def _fs_error_values(row: Mapping[str, object]) -> list[float]:
    values = [
        float(value)
        for key, value in row.items()
        if str(key).startswith("absolute_error_fs_k")
    ]
    if not values:
        raise ValueError("curve row is missing multi-k Fs errors")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("curve errors must be finite")
    return values


def higher_order_score(row: Mapping[str, object]) -> float:
    return max(
        float(row["ngp_absolute_error"]) / CURVE_LIMITS["ngp"],
        max(_fs_error_values(row)) / CURVE_LIMITS["fs"],
    )


def curve_pass(rows: Sequence[Mapping[str, object]]) -> bool:
    if not rows:
        return False
    return all(
        float(row.get("support_pass", 1.0)) == 1.0
        and float(row.get("all_child_restart_curve_gate_pass", 1.0)) == 1.0
        and float(row["msd_relative_error"]) <= CURVE_LIMITS["msd"]
        and float(row["ngp_absolute_error"]) <= CURVE_LIMITS["ngp"]
        and max(_fs_error_values(row)) <= CURVE_LIMITS["fs"]
        and _msd_mc_relative_se(row) <= MC_LIMITS["msd"]
        and float(row.get("mc_se_ngp", 0.0)) <= MC_LIMITS["ngp"]
        and max(
            float(value)
            for key, value in row.items()
            if str(key).startswith("mc_se_fs_k")
        )
        <= MC_LIMITS["fs"]
        for row in rows
    )


def summarize_model_verdicts(
    parent_summaries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    verdicts: list[dict[str, object]] = []
    keys = ("temperature", "parent_id", "model")
    for group_key, rows in _group_rows(parent_summaries, keys):
        verdict = dict(zip(keys, group_key, strict=True))
        verdict["lag_count"] = len(rows)
        verdict["curve_gate_pass"] = float(curve_pass(rows))
        verdict["higher_order_gate_pass"] = float(
            all(
                higher_order_score(row) <= 1.0
                and float(
                    row.get("all_child_restart_higher_order_gate_pass", 1.0)
                )
                == 1.0
                for row in rows
            )
        )
        verdict["maximum_higher_order_score"] = max(
            higher_order_score(row) for row in rows
        )
        verdict["maximum_child_restart_higher_order_score"] = max(
            float(row.get("maximum_child_restart_higher_order_score", 0.0))
            for row in rows
        )
        verdict["failed_child_restart_count"] = max(
            int(float(row.get("failed_child_restart_count", 0.0))) for row in rows
        )
        verdict["support_pass"] = min(
            float(row.get("support_pass", 1.0)) for row in rows
        )
        verdict["precision_pass"] = float(
            all(
                _msd_mc_relative_se(row) <= MC_LIMITS["msd"]
                and float(row.get("all_child_restart_precision_pass", 1.0)) == 1.0
                and float(row.get("mc_se_ngp", 0.0)) <= MC_LIMITS["ngp"]
                and max(
                    float(value)
                    for key, value in row.items()
                    if str(key).startswith("mc_se_fs_k")
                )
                <= MC_LIMITS["fs"]
                for row in rows
            )
        )
        verdicts.append(verdict)
    return verdicts


def _closed_gate_defaults() -> dict[str, object]:
    return {
        "mechanism_state": "unclassified",
        "failure_localization": "not_applicable",
        "positive_memory_closure_claim_allowed": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "complete_microscopic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
        "thermodynamic_glass_transition_claim_allowed": 0.0,
    }


def _verdict_lookup(
    verdicts: Sequence[Mapping[str, object]], model: str, *, temperature: float = 0.45
) -> dict[str, Mapping[str, object]]:
    return {
        str(row["parent_id"]): row
        for row in verdicts
        if math.isclose(float(row["temperature"]), temperature, abs_tol=1e-12)
        and str(row["model"]) == model
    }


def _required_ablation_pattern_holds(
    verdicts: Sequence[Mapping[str, object]], full: Mapping[str, Mapping[str, object]]
) -> bool:
    if len(full) < 3:
        return False
    for model in REQUIRED_MECHANISM_ABLATIONS:
        ablation = _verdict_lookup(verdicts, model)
        if set(ablation) != set(full):
            return False
        higher_order_failures = sum(
            float(row["higher_order_gate_pass"]) == 0.0 for row in ablation.values()
        )
        strict_improvements = sum(
            float(full[parent]["maximum_higher_order_score"])
            < float(ablation[parent]["maximum_higher_order_score"])
            for parent in full
        )
        if higher_order_failures < 2 or strict_improvements < 2:
            return False
    return True


def _localize_candidate_failure(
    *,
    verdicts: Sequence[Mapping[str, object]],
    full: Mapping[str, Mapping[str, object]],
    upper: Mapping[str, Mapping[str, object]],
    temperature: float = 0.45,
) -> str:
    if any(
        float(row["support_pass"]) == 0.0 or float(row["precision_pass"]) == 0.0
        for row in full.values()
    ):
        return "data_volume_or_support"
    if set(upper) != set(full) or any(
        float(row["curve_gate_pass"]) == 0.0 for row in upper.values()
    ):
        return "cross_particle_or_unmodeled_coupling"
    static = _verdict_lookup(
        verdicts, "static_particle_environment", temperature=temperature
    )
    finite = _verdict_lookup(
        verdicts, "finite_exchange_environment", temperature=temperature
    )
    if static and all(float(row["curve_gate_pass"]) == 1.0 for row in static.values()):
        return "environment_lifetime_or_state"
    if finite and any(
        parent in finite
        and float(finite[parent]["maximum_higher_order_score"])
        < float(full[parent]["maximum_higher_order_score"])
        for parent in full
    ):
        return "ordered_path_kernel"
    return "multiple_unresolved"


def classify_correlated_parent_diagnostic(
    *,
    parent_summaries: Sequence[Mapping[str, object]],
    upper_control_parents: Sequence[Mapping[str, object]],
    temperature: float = 0.45,
) -> dict[str, object]:
    """Describe model behavior without treating correlated restarts as evidence."""

    verdicts = summarize_model_verdicts(parent_summaries)
    upper_verdicts = summarize_model_verdicts(upper_control_parents)
    full = _verdict_lookup(verdicts, "full_candidate", temperature=temperature)
    upper = _verdict_lookup(
        upper_verdicts,
        "contiguous_empirical_upper_control",
        temperature=temperature,
    )
    result = {
        "diagnostic_state": "unavailable",
        "diagnostic_failure_localization": "not_applicable",
        "correlated_parent_diagnostic_only": 1.0,
        "positive_memory_closure_claim_allowed": 0.0,
    }
    if not full:
        return result
    if any(float(row["curve_gate_pass"]) == 0.0 for row in full.values()) or any(
        float(row["curve_gate_pass"]) == 0.0 for row in upper.values()
    ) or set(upper) != set(full):
        result["diagnostic_state"] = "candidate_rejected"
        result["diagnostic_failure_localization"] = _localize_candidate_failure(
            verdicts=verdicts, full=full, upper=upper, temperature=temperature
        )
        return result
    result["diagnostic_state"] = "candidate_passed_correlated_parent_diagnostic"
    return result


def classify_memory_closure_gate(
    *,
    parent_summaries: Sequence[Mapping[str, object]],
    blockers: Sequence[Mapping[str, object]],
    upper_control_parents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the frozen fail-closed gate at the independent-parent level."""

    result = _closed_gate_defaults()
    primary_blockers = [
        row for row in blockers if str(row.get("evidence_role", "primary")) != "canary_only"
    ]
    if any(int(float(row["missing_parent_count"])) > 0 for row in primary_blockers):
        result["mechanism_state"] = "blocked_independent_parent_validation"
        return result
    if any(float(row["stationarity_pass"]) == 0.0 for row in primary_blockers):
        result["mechanism_state"] = "blocked_stationarity_control"
        return result
    if any(float(row.get("input_lineage_join_pass", 0.0)) == 0.0 for row in primary_blockers):
        result["mechanism_state"] = "blocked_input_lineage"
        return result

    verdicts = summarize_model_verdicts(parent_summaries)
    upper_verdicts = summarize_model_verdicts(upper_control_parents)
    full = _verdict_lookup(verdicts, "full_candidate")
    upper = _verdict_lookup(upper_verdicts, "contiguous_empirical_upper_control")
    if len(full) < 3:
        result["mechanism_state"] = "blocked_independent_parent_validation"
        return result
    frozen_lags = {
        int(value) for value in str(FROZEN_PROTOCOLS[0.45]["lag_grid"]).split(";")
    }
    required_models = set(REQUIRED_MECHANISM_ABLATIONS) | {"full_candidate"}
    for parent_id in full:
        for model in required_models:
            rows = [
                row
                for row in parent_summaries
                if math.isclose(float(row["temperature"]), 0.45, abs_tol=1e-12)
                and str(row["parent_id"]) == parent_id
                and str(row["model"]) == model
            ]
            if {int(float(row["lag"])) for row in rows} != frozen_lags or len(rows) != len(frozen_lags):
                result["mechanism_state"] = "blocked_incomplete_frozen_grid"
                return result
        controls = [
            row
            for row in upper_control_parents
            if math.isclose(float(row["temperature"]), 0.45, abs_tol=1e-12)
            and str(row["parent_id"]) == parent_id
        ]
        if {int(float(row["lag"])) for row in controls} != frozen_lags or len(controls) != len(frozen_lags):
            result["mechanism_state"] = "blocked_incomplete_frozen_grid"
            return result
    if any(float(row["curve_gate_pass"]) == 0.0 for row in full.values()) or any(
        float(row["curve_gate_pass"]) == 0.0 for row in upper.values()
    ) or set(upper) != set(full):
        result["mechanism_state"] = "candidate_rejected"
        result["failure_localization"] = _localize_candidate_failure(
            verdicts=verdicts, full=full, upper=upper
        )
        return result
    if not _required_ablation_pattern_holds(verdicts, full):
        result["mechanism_state"] = "ablation_pattern_unresolved"
        return result
    result["mechanism_state"] = (
        "positive_memory_closure_supported_within_tested_family"
    )
    result["positive_memory_closure_claim_allowed"] = 1.0
    return result


def build_claim_ledger(gate: Mapping[str, object]) -> list[dict[str, object]]:
    """Emit the frozen candidate claim and permanently closed scope boundaries."""

    mechanism_state = str(gate["mechanism_state"])
    positive_allowed = float(gate["positive_memory_closure_claim_allowed"])
    statements = (
        (
            "positive_memory_closure",
            "In the low-temperature Kob-Andersen glass former, relaxation dynamics "
            "cannot be reconstructed from the mean event rate, one-step jump law, or "
            "two-point path spectrum. Accurate reconstruction requires ordered "
            "cage-path memory and a persistent particle-level environment.",
            positive_allowed,
            mechanism_state,
        ),
        (
            "complete_microscopic_closure",
            "The tested single-particle memory model is a complete microscopic closure.",
            0.0,
            "outside_preregistered_claim_boundary",
        ),
        (
            "spatial_facilitation",
            "The tested evidence establishes spatial facilitation.",
            0.0,
            "outside_preregistered_claim_boundary",
        ),
        (
            "thermodynamic_glass_transition",
            "The tested evidence establishes a thermodynamic glass transition.",
            0.0,
            "outside_preregistered_claim_boundary",
        ),
    )
    return [
        {
            "claim_id": claim_id,
            "claim_statement": statement,
            "claim_allowed": allowed,
            "claim_state": state,
            "question": "What microscopic information is minimally required to reconstruct low-temperature glassy relaxation?",
            "wording_frozen_before_positive_model_run": 1.0,
            "thresholds_frozen_before_positive_model_run": 1.0,
        }
        for claim_id, statement, allowed, state in statements
    ]
