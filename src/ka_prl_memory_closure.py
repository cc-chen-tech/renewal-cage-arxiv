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
                    "stationarity_pass": float(stationarity_pass),
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
                    f"acquire_{missing_parent_count}_independent_parent_trajectories_and_requalify_stationarity"
                    if missing_parent_count and not stationarity_pass
                    else f"acquire_{missing_parent_count}_independent_parent_trajectories"
                    if missing_parent_count
                    else (
                        "replace_or_requalify_stationarity_control"
                        if not stationarity_pass
                        else "run_preregistered_parent_first_gate"
                    )
                ),
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
    verdicts: Sequence[Mapping[str, object]], model: str
) -> dict[str, Mapping[str, object]]:
    return {
        str(row["parent_id"]): row
        for row in verdicts
        if float(row["temperature"]) == 0.45 and str(row["model"]) == model
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
    static = _verdict_lookup(verdicts, "static_particle_environment")
    finite = _verdict_lookup(verdicts, "finite_exchange_environment")
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
) -> dict[str, object]:
    """Describe model behavior without treating correlated restarts as evidence."""

    verdicts = summarize_model_verdicts(parent_summaries)
    upper_verdicts = summarize_model_verdicts(upper_control_parents)
    full = _verdict_lookup(verdicts, "full_candidate")
    upper = _verdict_lookup(upper_verdicts, "contiguous_empirical_upper_control")
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
            verdicts=verdicts, full=full, upper=upper
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

    verdicts = summarize_model_verdicts(parent_summaries)
    upper_verdicts = summarize_model_verdicts(upper_control_parents)
    full = _verdict_lookup(verdicts, "full_candidate")
    upper = _verdict_lookup(upper_verdicts, "contiguous_empirical_upper_control")
    if len(full) < 3:
        result["mechanism_state"] = "blocked_independent_parent_validation"
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
