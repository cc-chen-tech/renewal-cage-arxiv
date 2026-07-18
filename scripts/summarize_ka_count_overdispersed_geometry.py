#!/usr/bin/env python3
"""Test whether count overdispersion closes the activated-jump quotient."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


WAVE_NUMBERS = (2.0, 4.0, 7.25)
FS_TOLERANCE = 0.03
MINIMUM_SUPPORT_FRACTION = 0.8
CLAIM_FLAGS = (
    "blind_prediction_claim_allowed",
    "finite_exchange_resolved",
    "static_environment_resolved",
    "spatial_facilitation_resolved",
    "activated_cage_geometry_resolved",
    "transient_potential_identified_in_ka",
    "microdynamic_closure_claim_allowed",
    "thermodynamic_claim_allowed",
)


def wave_key(wave_number: float) -> str:
    return f"k{wave_number:g}".replace(".", "p")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gamma_poisson_count_pgf(
    *,
    mean_count: float,
    fano_factor: float,
    argument: float,
) -> float:
    values = (mean_count, fano_factor, argument)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("count PGF inputs must be finite")
    if mean_count < 0.0 or fano_factor < 1.0 or not -1.0 <= argument <= 1.0:
        raise ValueError("count PGF inputs are outside their physical domain")
    if math.isclose(fano_factor, 1.0, abs_tol=1e-12):
        return math.exp(mean_count * (argument - 1.0))
    excess = fano_factor - 1.0
    base = 1.0 + excess * (1.0 - argument)
    return base ** (-mean_count / excess)


def count_overdispersed_geometry_row(
    geometry: dict[str, str],
    macro: dict[str, str],
) -> dict[str, float]:
    geometry_key = (
        int(float(geometry["replicate"])),
        int(float(geometry["lag"])),
    )
    macro_key = (int(float(macro["replicate"])), int(float(macro["lag"])))
    if geometry_key != macro_key:
        raise ValueError("geometry and count rows do not have the same key")
    msd = float(geometry["heldout_msd"])
    ngp = float(geometry["heldout_ngp"])
    jump_msd = float(geometry["jump_msd"])
    jump_fourth_x = float(geometry["jump_component_fourth_moment"])
    predicted_count = float(macro["predicted_mean_count"])
    predicted_variance = float(macro["predicted_count_variance"])
    values = (msd, ngp, jump_msd, jump_fourth_x, predicted_count, predicted_variance)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("geometry and count moments must be finite")
    if (
        msd < 0.0
        or ngp < 0.0
        or jump_msd <= 0.0
        or jump_fourth_x <= 0.0
        or predicted_count <= 0.0
        or predicted_variance < predicted_count
    ):
        raise ValueError("geometry and count moments are outside their physical domain")
    fano = predicted_variance / predicted_count
    component_second = jump_msd / 3.0
    denominator = jump_fourth_x + 3.0 * (fano - 1.0) * component_second**2
    inferred_count = ngp * msd**2 / (3.0 * denominator)
    cage_variance = (msd - inferred_count * jump_msd) / 6.0
    supported = inferred_count >= 0.0 and cage_variance >= 0.0
    row: dict[str, float] = {
        "temperature": float(geometry["temperature"]),
        "replicate": float(geometry_key[0]),
        "lag": float(geometry_key[1]),
        "heldout_msd": msd,
        "heldout_ngp": ngp,
        "jump_msd": jump_msd,
        "jump_component_fourth_moment": jump_fourth_x,
        "predicted_count_mean": predicted_count,
        "predicted_count_variance": predicted_variance,
        "count_fano_factor": fano,
        "inferred_mean_event_count": inferred_count,
        "inferred_cage_variance": cage_variance,
        "supported": float(supported),
        "heldout_msd_used_as_diagnostic_input": 1.0,
        "heldout_ngp_used_as_diagnostic_input": 1.0,
        "calibration_count_overdispersion_only": 1.0,
    }
    for wave_number in WAVE_NUMBERS:
        key = wave_key(wave_number)
        observed = float(geometry[f"observed_fs_{key}"])
        characteristic = float(geometry[f"jump_characteristic_{key}"])
        predicted = math.nan
        if supported:
            count_pgf = gamma_poisson_count_pgf(
                mean_count=inferred_count,
                fano_factor=fano,
                argument=characteristic,
            )
            predicted = math.exp(-wave_number**2 * cage_variance) * count_pgf
        row[f"observed_fs_{key}"] = observed
        row[f"predicted_fs_{key}"] = predicted
        row[f"fs_{key}_absolute_error"] = (
            abs(predicted - observed) if supported else math.nan
        )
    for flag in CLAIM_FLAGS:
        row[flag] = 0.0
    return row


def summarize_temperature(
    rows: list[dict[str, float]],
    *,
    source_stationarity_pass: bool,
) -> dict[str, float | str]:
    if not rows:
        raise ValueError("temperature summary requires rows")
    temperatures = {float(row["temperature"]) for row in rows}
    if len(temperatures) != 1:
        raise ValueError("temperature summary cannot mix temperatures")
    supported = [row for row in rows if float(row["supported"]) == 1.0]
    support_fraction = len(supported) / len(rows)
    errors = {
        wave_key(wave_number): max(
            (float(row[f"fs_{wave_key(wave_number)}_absolute_error"]) for row in supported),
            default=math.nan,
        )
        for wave_number in WAVE_NUMBERS
    }
    curve_pass = (
        source_stationarity_pass
        and support_fraction >= MINIMUM_SUPPORT_FRACTION
        and all(value <= FS_TOLERANCE for value in errors.values())
    )
    temperature = temperatures.pop()
    result: dict[str, float | str] = {
        "temperature": temperature,
        "analysis_status": (
            "count_overdispersion_geometry_closure"
            if curve_pass
            else (
                "canary_only_nonstationary_source"
                if not source_stationarity_pass
                else "count_overdispersion_restores_moment_support_high_k_unresolved"
            )
        ),
        "row_count": float(len(rows)),
        "replicate_count": float(len({int(row["replicate"]) for row in rows})),
        "source_stationarity_pass": float(source_stationarity_pass),
        "minimum_support_fraction": MINIMUM_SUPPORT_FRACTION,
        "supported_row_count": float(len(supported)),
        "support_fraction": support_fraction,
        "support_coverage_pass": float(support_fraction >= MINIMUM_SUPPORT_FRACTION),
        "fs_k2_max_absolute_error": errors["k2"],
        "fs_k4_max_absolute_error": errors["k4"],
        "fs_k7p25_max_absolute_error": errors["k7p25"],
        "fs_absolute_error_tolerance": FS_TOLERANCE,
        "curve_transfer_pass": float(curve_pass),
        "count_overdispersion_moment_support_supported_exploratory": float(
            support_fraction >= MINIMUM_SUPPORT_FRACTION
        ),
        "high_temperature_canary_only": float(temperature > 0.5),
        "next_required_action": "test_joint_rate_recoil_cage_dynamics",
    }
    for flag in CLAIM_FLAGS:
        result[flag] = 0.0
    return result


def analyze_committed_tables(
    root: Path,
) -> tuple[list[dict[str, float]], list[dict[str, float | str]]]:
    data = root / "data"
    geometry = read_rows(data / "renewal_cage_ka_activated_cage_geometry_rows.csv")
    source_gates = {
        float(row["temperature"]): bool(float(row["source_stationarity_pass"]))
        for row in read_rows(data / "renewal_cage_ka_activated_cage_geometry_gate.csv")
    }
    macro: dict[tuple[float, int, int], dict[str, str]] = {}
    for label, temperature in (("045", 0.45), ("058", 0.58)):
        path = data / f"renewal_cage_ka_replicates_T{label}_state_joint_finite_gk_macro_rows.csv"
        for row in read_rows(path):
            key = (temperature, int(float(row["replicate"])), int(float(row["lag"])))
            macro[key] = row
    rows: list[dict[str, float]] = []
    for geometry_row in geometry:
        key = (
            float(geometry_row["temperature"]),
            int(float(geometry_row["replicate"])),
            int(float(geometry_row["lag"])),
        )
        if key not in macro:
            raise ValueError(f"missing count row for {key}")
        rows.append(count_overdispersed_geometry_row(geometry_row, macro[key]))
    rows.sort(key=lambda row: (row["temperature"], row["replicate"], row["lag"]))
    gates = [
        summarize_temperature(
            [row for row in rows if float(row["temperature"]) == temperature],
            source_stationarity_pass=source_gates[temperature],
        )
        for temperature in sorted(source_gates)
    ]
    return rows, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    args = parser.parse_args()
    rows, gates = analyze_committed_tables(args.root)
    write_rows(args.output_rows, rows)
    write_rows(args.output_gate, gates)


if __name__ == "__main__":
    main()
