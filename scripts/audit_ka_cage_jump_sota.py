#!/usr/bin/env python3
"""Audit whether p_hop events reproduce published 3D KALJ cage-jump relations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import independent_group_ratio  # noqa: E402


SOURCE_DOI = "10.3390/ijms23073556"
SOURCE_URL = "https://www.mdpi.com/1422-0067/23/7/3556"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def read_one(path: Path) -> dict[str, str]:
    rows = read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    return rows[0]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_alignment(
    *,
    microscopic_growth: dict[str, float | str],
    macroscopic_growth: dict[str, float | str],
    microscopic_invariant: dict[str, float | str],
    jump_length: dict[str, float | str],
    low_temperature_ctwr_pass: float,
) -> dict[str, float | str]:
    ctwr_consistent = float(low_temperature_ctwr_pass == 0.0)
    invariant_consistent = float(microscopic_invariant["equivalent_to_unity"])
    decoupling_order_consistent = float(
        float(microscopic_growth["ci95_low_ratio"])
        > float(macroscopic_growth["ci95_high_ratio"])
    )
    jump_direction_consistent = float(jump_length["decrease_detected"])
    validated = float(
        ctwr_consistent == 1.0
        and invariant_consistent == 1.0
        and decoupling_order_consistent == 1.0
        and jump_direction_consistent == 1.0
    )
    return {
        "low_temperature_ctwr_failure_consistent": ctwr_consistent,
        "microscopic_invariant_consistent": invariant_consistent,
        "relative_decoupling_order_consistent": decoupling_order_consistent,
        "jump_length_cooling_direction_consistent": jump_direction_consistent,
        "current_phop_elementary_cage_jump_claim_allowed": validated,
        "primary_mismatch": (
            "microscopic_decoupling_weaker_than_macroscopic"
            if decoupling_order_consistent == 0.0
            else "none"
        ),
        "next_required_test": (
            "event_definition_completeness_against_cage_jump_segmentation"
            if validated == 0.0
            else "heldout_multobservables_with_validated_events"
        ),
        "thermodynamic_claim_allowed": 0.0,
    }


def metric_row(
    metric_id: str,
    source_claim: str,
    comparison: dict[str, float | str],
    outcome: str,
) -> dict[str, object]:
    return {
        "metric_id": metric_id,
        "source_doi": SOURCE_DOI,
        "source_url": SOURCE_URL,
        "source_system": "3d_Kob_Andersen_80_20_Lennard_Jones",
        "source_temperature_range": "0.445_to_0.6",
        "our_temperature_pair": "0.58_to_0.45",
        "source_claim": source_claim,
        **comparison,
        "alignment_outcome": outcome,
        "event_definition_numerically_aligned": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-event-replicates", type=Path, required=True)
    parser.add_argument("--low-event-replicates", type=Path, required=True)
    parser.add_argument("--high-macro-replicates", type=Path, required=True)
    parser.add_argument("--low-macro-replicates", type=Path, required=True)
    parser.add_argument("--low-ctwr-verdict", type=Path, required=True)
    parser.add_argument("--relative-equivalence-margin", type=float, default=0.15)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    high_event = read_rows(args.high_event_replicates)
    low_event = read_rows(args.low_event_replicates)
    high_macro = read_rows(args.high_macro_replicates)
    low_macro = read_rows(args.low_macro_replicates)

    def event_values(rows: list[dict[str, str]], key: str) -> np.ndarray:
        return np.array([float(row[key]) for row in rows])

    high_micro = event_values(high_event, "persistence_exchange_ratio")
    low_micro = event_values(low_event, "persistence_exchange_ratio")
    high_jump = event_values(high_event, "jump_squared_mean")
    low_jump = event_values(low_event, "jump_squared_mean")
    high_invariant = high_micro * high_jump
    low_invariant = low_micro * low_jump
    high_se = np.array([float(row["diffusion_alpha_product"]) for row in high_macro])
    low_se = np.array([float(row["diffusion_alpha_product"]) for row in low_macro])

    micro_growth = independent_group_ratio(
        low_micro,
        high_micro,
        relative_equivalence_margin=args.relative_equivalence_margin,
    )
    macro_growth = independent_group_ratio(
        low_se,
        high_se,
        relative_equivalence_margin=args.relative_equivalence_margin,
    )
    invariant = independent_group_ratio(
        low_invariant,
        high_invariant,
        relative_equivalence_margin=args.relative_equivalence_margin,
    )
    jump_length = independent_group_ratio(
        low_jump,
        high_jump,
        relative_equivalence_margin=args.relative_equivalence_margin,
    )
    low_ctwr = read_one(args.low_ctwr_verdict)
    verdict = classify_alignment(
        microscopic_growth=micro_growth,
        macroscopic_growth=macro_growth,
        microscopic_invariant=invariant,
        jump_length=jump_length,
        low_temperature_ctwr_pass=float(low_ctwr["heldout_transport_pass"]),
    )
    verdict.update(
        {
            "source_doi": SOURCE_DOI,
            "source_url": SOURCE_URL,
            "high_temperature": 0.58,
            "low_temperature": 0.45,
            "high_temperature_replicate_count": float(len(high_event)),
            "low_temperature_replicate_count": float(len(low_event)),
            "microscopic_decoupling_growth": micro_growth["mean_ratio"],
            "microscopic_decoupling_ci95_low": micro_growth["ci95_low_ratio"],
            "microscopic_decoupling_ci95_high": micro_growth["ci95_high_ratio"],
            "macroscopic_se_growth": macro_growth["mean_ratio"],
            "macroscopic_se_ci95_low": macro_growth["ci95_low_ratio"],
            "macroscopic_se_ci95_high": macro_growth["ci95_high_ratio"],
            "microscopic_invariant_cooling_ratio": invariant["mean_ratio"],
            "jump_squared_cooling_ratio": jump_length["mean_ratio"],
            "relative_equivalence_margin": args.relative_equivalence_margin,
        }
    )
    rows = [
        metric_row(
            "microscopic_persistence_exchange_growth",
            "microscopic persistence-caging decoupling grows strongly on cooling",
            micro_growth,
            "growth_detected_but_weaker_than_macroscopic",
        ),
        metric_row(
            "macroscopic_stokes_einstein_growth",
            "macroscopic diffusion-relaxation decoupling grows on cooling",
            macro_growth,
            "growth_detected",
        ),
        metric_row(
            "microscopic_time_length_invariant",
            "persistence/caging ratio is proportional to inverse mean squared jump length",
            invariant,
            (
                "equivalent_with_small_detectable_drift"
                if float(invariant["equivalent_to_unity"]) == 1.0
                else "not_equivalent"
            ),
        ),
        metric_row(
            "mean_squared_jump_length_cooling",
            "mean squared jump length decreases on cooling",
            jump_length,
            "decrease_detected_but_magnitude_underresolved",
        ),
    ]
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_metrics.csv"), rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
