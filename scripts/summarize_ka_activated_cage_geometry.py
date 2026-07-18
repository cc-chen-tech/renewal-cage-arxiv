#!/usr/bin/env python3
"""Evaluate calibration cage-jump geometry against held-out KA shape diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


STRONG_ZERO_FLAGS = (
    "blind_prediction_claim_allowed",
    "finite_exchange_resolved",
    "static_environment_resolved",
    "spatial_facilitation_resolved",
    "activated_cage_geometry_resolved",
    "microdynamic_closure_claim_allowed",
    "thermodynamic_claim_allowed",
)

FS_ABSOLUTE_ERROR_TOLERANCE = 0.03
MINIMUM_SUPPORT_FRACTION = 0.8
EXPECTED_REPLICATE_COUNT = {0.45: 3, 0.58: 5}


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def empirical_geometry_quotient(
    *,
    msd: float,
    ngp: float,
    jump_msd: float,
    jump_component_fourth_moment: float,
    jump_characteristic: Mapping[float, float],
) -> dict[str, object]:
    msd = _positive_finite(msd, "msd")
    ngp = _nonnegative_finite(ngp, "ngp")
    jump_msd = _positive_finite(jump_msd, "jump_msd")
    fourth = _positive_finite(
        jump_component_fourth_moment,
        "jump_component_fourth_moment",
    )
    if not jump_characteristic:
        raise ValueError("jump_characteristic must be nonempty")
    characteristic = {float(k): float(value) for k, value in jump_characteristic.items()}
    if any(
        not math.isfinite(k)
        or k <= 0.0
        or not math.isfinite(value)
        or value < -1.0
        or value > 1.0
        for k, value in characteristic.items()
    ):
        raise ValueError("jump characteristic entries must have positive k and values in [-1, 1]")

    count = ngp * msd**2 / (3.0 * fourth)
    cage = (msd - count * jump_msd) / 6.0
    supported = count >= 0.0 and cage >= 0.0
    predicted = (
        {
            wave_number: math.exp(
                -wave_number**2 * cage + count * (value - 1.0)
            )
            for wave_number, value in characteristic.items()
        }
        if supported
        else {}
    )
    return {
        "supported": float(supported),
        "mean_event_count": count,
        "cage_variance": cage,
        "predicted_fs": predicted,
    }


def fixed_length_geometry_quotient(
    *,
    msd: float,
    ngp: float,
    jump_msd: float,
    wave_numbers: Iterable[float],
) -> dict[str, object]:
    jump_msd = _positive_finite(jump_msd, "jump_msd")
    length = math.sqrt(jump_msd)
    characteristic = {}
    for value in wave_numbers:
        wave_number = _positive_finite(value, "wave_number")
        argument = wave_number * length
        characteristic[wave_number] = math.sin(argument) / argument
    return empirical_geometry_quotient(
        msd=msd,
        ngp=ngp,
        jump_msd=jump_msd,
        jump_component_fourth_moment=jump_msd**2 / 5.0,
        jump_characteristic=characteristic,
    )


def classify_geometry_gate(
    rows: list[dict[str, object]],
    *,
    stationarity_pass: Mapping[float, bool],
    provenance_pass: Mapping[float, bool],
) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("geometry gate requires nonempty rows")
    gates = []
    temperatures = sorted({float(row["temperature"]) for row in rows})
    for temperature in temperatures:
        local = [row for row in rows if float(row["temperature"]) == temperature]
        supported = [
            row for row in local if float(row["empirical_geometry_supported"]) == 1.0
        ]
        replicate_ids = sorted({int(row["replicate"]) for row in local})
        supported_replicates = {
            int(row["replicate"])
            for row in supported
        }
        expected_replicates = EXPECTED_REPLICATE_COUNT.get(
            temperature,
            len(replicate_ids),
        )
        all_replicates_supported = (
            len(replicate_ids) == expected_replicates
            and supported_replicates == set(replicate_ids)
        )
        support_fraction = len(supported) / len(local)
        support_pass = (
            support_fraction >= MINIMUM_SUPPORT_FRACTION
            and all_replicates_supported
        )
        maxima = {}
        for key in ("k2", "k4", "k7p25"):
            field = f"empirical_fs_{key}_absolute_error"
            maxima[key] = max(
                (float(row[field]) for row in supported),
                default=math.nan,
            )
        curve_pass = bool(supported) and all(
            value <= FS_ABSOLUTE_ERROR_TOLERANCE
            for value in maxima.values()
        )
        source_stationarity = bool(stationarity_pass.get(temperature, False))
        source_provenance = bool(provenance_pass.get(temperature, False))
        primary = temperature == 0.45
        exploratory_support = (
            primary
            and source_stationarity
            and source_provenance
            and support_pass
            and curve_pass
        )
        if exploratory_support:
            status = "empirical_activated_jump_geometry_supported_exploratory"
            next_action = "derive_distributed_basin_langevin_kramers_closure"
        elif not source_provenance:
            status = "unresolved_provenance"
            next_action = "repair_source_provenance"
        elif not source_stationarity:
            status = "canary_only_nonstationary_source"
            next_action = "do_not_use_for_cooling_claim"
        elif not support_pass:
            status = "compound_poisson_cage_decomposition_unsupported"
            next_action = "test_count_or_cage_jump_dependence"
        else:
            status = "empirical_jump_geometry_shape_failure"
            next_action = "test_correlated_or_non_poisson_event_geometry"
        gate = {
            "temperature": temperature,
            "analysis_status": status,
            "replicate_count": float(len(replicate_ids)),
            "row_count": float(len(local)),
            "source_stationarity_pass": float(source_stationarity),
            "replicate_provenance_validation_pass": float(source_provenance),
            "minimum_support_fraction": MINIMUM_SUPPORT_FRACTION,
            "empirical_supported_row_count": float(len(supported)),
            "empirical_support_fraction": support_fraction,
            "empirical_support_coverage_pass": float(support_pass),
            "all_primary_replicates_supported": float(all_replicates_supported),
            "fixed_length_supported_row_count": float(
                sum(float(row["fixed_length_geometry_supported"]) for row in local)
            ),
            "fixed_length_support_fraction": sum(
                float(row["fixed_length_geometry_supported"]) for row in local
            )
            / len(local),
            "fs_k2_empirical_max_absolute_error": maxima["k2"],
            "fs_k4_empirical_max_absolute_error": maxima["k4"],
            "fs_k7p25_empirical_max_absolute_error": maxima["k7p25"],
            "fs_absolute_error_tolerance": FS_ABSOLUTE_ERROR_TOLERANCE,
            "curve_transfer_pass": float(curve_pass),
            "empirical_activated_jump_geometry_supported_exploratory": float(
                exploratory_support
            ),
            "high_temperature_canary_only": float(not primary),
            "next_required_action": next_action,
        }
        gate.update({flag: 0.0 for flag in STRONG_ZERO_FLAGS})
        gates.append(gate)
    return gates
