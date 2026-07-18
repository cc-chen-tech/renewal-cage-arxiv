"""Parent-first gates for the preregistered PRL memory-closure candidate."""

from __future__ import annotations

import math
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
    *, missing_parent_count: int, stationarity_pass: bool
) -> str:
    if missing_parent_count and not stationarity_pass:
        return "stationarity_and_independent_parents"
    if missing_parent_count:
        return "missing_independent_parents"
    if not stationarity_pass:
        return "stationarity_failed"
    return "eligible"


def audit_parent_provenance(
    *,
    provenance_rows: Sequence[Mapping[str, object]],
    stationarity_by_temperature: Mapping[
        float, Sequence[Mapping[str, object]]
    ],
    protocol_by_temperature: Mapping[float, Mapping[str, object]] = FROZEN_PROTOCOLS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Audit restart provenance while counting unique source parents only once."""

    if not provenance_rows:
        raise ValueError("parent provenance rows must not be empty")
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
        try:
            stationarity_rows = stationarity_by_temperature[temperature]
        except KeyError as error:
            raise ValueError(f"missing stationarity rows for T={temperature}") from error
        stationarity_pass = _stationarity_pass(stationarity_rows)
        parent_ids = [parent_identifier(row) for row in selected]
        unique_parent_ids = sorted(set(parent_ids))
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
                    "source_doi": str(row["source_doi"]),
                    "source_sha256": str(row["source_sha256"]),
                    "source_frame_index": _as_int(row, "source_frame_index"),
                    "velocity_seed": _as_int(row, "velocity_seed"),
                    "trajectory_length_tau": recorded_production,
                    "calibration_start_tau": 0.0,
                    "calibration_stop_tau": float(protocol["calibration_time"]),
                    "heldout_start_tau": float(protocol["calibration_time"]),
                    "heldout_stop_tau": recorded_production,
                    "stationarity_pass": float(stationarity_pass),
                    "available_observables": FROZEN_OBSERVABLES,
                    "lag_grid": str(protocol["lag_grid"]),
                    "evidence_role": str(protocol["evidence_role"]),
                    "recorded_independence_class": str(
                        row.get("independence_class", "")
                    ),
                    "audited_independence_class": "correlated_restarts_of_one_parent_trajectory",
                    "restart_is_independent_sample": 0.0,
                    "parent_unit_contribution": int(
                        replicate == first_restart_by_parent[parent_id]
                    ),
                    "microdynamic_closure_claim_allowed": 0.0,
                    "spatial_facilitation_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
        blocker_state = _parent_blocker_state(
            missing_parent_count=missing_parent_count,
            stationarity_pass=stationarity_pass,
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
                "blocker_state": blocker_state,
                "required_trajectory_length_tau": production_time,
                "required_calibration_time_tau": float(protocol["calibration_time"]),
                "required_heldout_time_tau": float(protocol["heldout_time"]),
                "required_observables": FROZEN_OBSERVABLES,
                "shared_parent_resampling_can_satisfy": 0.0,
                "next_required_action": (
                    f"acquire_{missing_parent_count}_independent_parent_trajectories"
                    if missing_parent_count
                    else (
                        "replace_or_requalify_stationarity_control"
                        if not stationarity_pass
                        else "run_preregistered_parent_first_gate"
                    )
                ),
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
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


def _information_audit(
    *,
    model: str,
    source_particle: np.ndarray,
    source_block: np.ndarray,
    exchange_count: int = 0,
    forced_exchange_count: int = 0,
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

    ordered = model == "full_candidate"
    exchange_probability = -math.expm1(-block_size / environment_time)
    source_particle = np.empty((particle_count, block_count), dtype=int)
    source_block = np.empty((particle_count, block_count), dtype=int)
    generated = np.empty_like(blocks)
    exchange_count = 0
    forced_exchange_count = 0
    for target_particle in range(particle_count):
        current_particle = target_particle
        current_block = int(rng.integers(block_count))
        for output_block in range(block_count):
            source_particle[target_particle, output_block] = current_particle
            source_block[target_particle, output_block] = current_block
            generated[target_particle, output_block] = blocks[
                current_particle, current_block
            ]
            if output_block == block_count - 1:
                continue

            forced_exchange = ordered and current_block == block_count - 1
            stochastic_exchange = rng.random() < exchange_probability
            if forced_exchange or stochastic_exchange:
                current_particle = _different_particle(
                    current_particle, particle_count, rng
                )
                current_block = int(rng.integers(block_count))
                exchange_count += 1
                forced_exchange_count += int(forced_exchange)
            elif ordered:
                current_block += 1
            else:
                current_block = int(rng.integers(block_count))

    return generated, _information_audit(
        model=model,
        source_particle=source_particle,
        source_block=source_block,
        exchange_count=exchange_count,
        forced_exchange_count=forced_exchange_count,
    )
