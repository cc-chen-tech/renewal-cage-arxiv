"""Parent-first gates for the preregistered PRL memory-closure candidate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


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
