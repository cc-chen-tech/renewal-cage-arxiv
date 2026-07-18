#!/usr/bin/env python3
"""Run paired continuous-Langevin ablations for barrier and recoil memory."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transient_periodic_langevin import (  # noqa: E402
    TransientPeriodicParams,
    displacement_observables,
    event_clock_statistics,
    simulate_transient_periodic_langevin,
    stable_cage_events,
)


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


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def run_ablation(
    *,
    seed: int,
    quick: bool,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    """Run four models with a frozen parameter family and paired seeds."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    simulation = (
        {
            "trajectory_count": 96,
            "burnin_steps": 2000,
            "production_steps": 8000,
        }
        if quick
        else {
            "trajectory_count": 384,
            "burnin_steps": 10000,
            "production_steps": 40000,
        }
    )
    model_parameters = (
        ("static_periodic", 0.0, 0.0),
        ("rate_only", 0.0, 1.5),
        ("elastic_only", 1.0, 0.0),
        ("full_transient", 1.0, 1.5),
    )
    dt = 0.002
    record_stride = 10
    record_dt = dt * record_stride
    duration = simulation["production_steps"] * dt
    rows: list[dict[str, float | str]] = []
    for model, elastic_stiffness, barrier_coupling in model_parameters:
        params = TransientPeriodicParams(
            temperature=1.0,
            period=1.0,
            base_barrier=3.0,
            elastic_stiffness=elastic_stiffness,
            barrier_stiffness=1.0,
            barrier_coupling=barrier_coupling,
            gamma_x=1.0,
            gamma_q=20.0,
            gamma_z=20.0,
        )
        result = simulate_transient_periodic_langevin(
            params,
            trajectory_count=simulation["trajectory_count"],
            dimension=3,
            dt=dt,
            burnin_steps=simulation["burnin_steps"],
            production_steps=simulation["production_steps"],
            record_stride=record_stride,
            seed=seed,
        )
        events = stable_cage_events(
            result["positions"],
            period=params.period,
            dwell_frames=5,
            frame_dt=record_dt,
        )
        clock = event_clock_statistics(
            events,
            trajectory_count=simulation["trajectory_count"],
            dimension=3,
            duration=duration,
            count_window=2.0,
        )
        record_count = result["positions"].shape[0]
        lag_frames = [1, 5, 25, 100, min(400, record_count - 1)]
        observables = displacement_observables(
            result["positions"],
            lag_frames=lag_frames,
            wave_numbers=[2.0, 4.0, 7.25],
        )
        maximum_ngp = max(float(row["ngp"]) for row in observables)
        final = observables[-1]
        row: dict[str, float | str] = {
            "model": model,
            "seed": float(seed),
            "quick_mode": float(quick),
            "trajectory_count": float(simulation["trajectory_count"]),
            "dimension": 3.0,
            "temperature": params.temperature,
            "period": params.period,
            "base_barrier": params.base_barrier,
            "elastic_stiffness": elastic_stiffness,
            "barrier_coupling": barrier_coupling,
            "barrier_stiffness": params.barrier_stiffness,
            "gamma_x": params.gamma_x,
            "gamma_q": params.gamma_q,
            "gamma_z": params.gamma_z,
            "dt": dt,
            "record_dt": record_dt,
            "burnin_steps": float(simulation["burnin_steps"]),
            "production_steps": float(simulation["production_steps"]),
            "dwell_frames": 5.0,
            "count_window": 2.0,
            "event_count": clock["event_count"],
            "mean_window_count": clock["mean_window_count"],
            "count_fano_factor": clock["count_fano_factor"],
            "mean_persistence_time": _finite_or_zero(clock["mean_persistence_time"]),
            "mean_exchange_time": _finite_or_zero(clock["mean_exchange_time"]),
            "persistence_exchange_ratio": _finite_or_zero(
                clock["persistence_exchange_ratio"]
            ),
            "successive_vector_pair_count": clock["successive_vector_pair_count"],
            "successive_vector_correlation": _finite_or_zero(
                clock["successive_vector_correlation"]
            ),
            "maximum_ngp": maximum_ngp,
            "final_lag_time": float(final["lag_frames"]) * record_dt,
            "final_msd": float(final["msd"]),
            "final_fs_k2": float(final["fs_k2"]),
            "final_fs_k4": float(final["fs_k4"]),
            "final_fs_k7p25": float(final["fs_k7p25"]),
            "maximum_euler_displacement": float(result["maximum_euler_displacement"]),
            "all_finite": float(result["all_finite"]),
            "trajectory_continuity_pass": float(
                float(result["maximum_euler_displacement"]) < 0.5 * params.period
            ),
            "parameter_adjustment_from_design": 0.0,
            "synthetic_capability_only": 1.0,
        }
        for flag in CLAIM_FLAGS:
            row[flag] = 0.0
        rows.append(row)

    by_model = {str(row["model"]): row for row in rows}
    static = by_model["static_periodic"]
    rate = by_model["rate_only"]
    elastic = by_model["elastic_only"]
    full = by_model["full_transient"]
    fano_margin = 0.02
    correlation_margin = 0.02
    rate_pass = float(rate["count_fano_factor"]) > float(
        static["count_fano_factor"]
    ) + fano_margin
    elastic_pass = float(elastic["successive_vector_correlation"]) < float(
        static["successive_vector_correlation"]
    ) - correlation_margin
    full_pass = (
        float(full["count_fano_factor"])
        > float(static["count_fano_factor"]) + fano_margin
        and float(full["successive_vector_correlation"])
        < float(static["successive_vector_correlation"]) - correlation_margin
    )
    gate: dict[str, float | str] = {
        "analysis_status": (
            "synthetic_joint_rate_recoil_capability"
            if rate_pass and elastic_pass and full_pass
            else "synthetic_ablation_unresolved"
        ),
        "seed": float(seed),
        "quick_mode": float(quick),
        "model_count": 4.0,
        "all_models_finite": float(all(float(row["all_finite"]) == 1.0 for row in rows)),
        "all_models_continuous": float(
            all(float(row["trajectory_continuity_pass"]) == 1.0 for row in rows)
        ),
        "minimum_event_count": min(float(row["event_count"]) for row in rows),
        "fano_comparison_margin": fano_margin,
        "correlation_comparison_margin": correlation_margin,
        "rate_only_minus_static_fano": float(rate["count_fano_factor"])
        - float(static["count_fano_factor"]),
        "elastic_minus_static_step_correlation": float(
            elastic["successive_vector_correlation"]
        )
        - float(static["successive_vector_correlation"]),
        "full_minus_static_fano": float(full["count_fano_factor"])
        - float(static["count_fano_factor"]),
        "full_minus_static_step_correlation": float(
            full["successive_vector_correlation"]
        )
        - float(static["successive_vector_correlation"]),
        "rate_disorder_count_fano_increase": float(rate_pass),
        "elastic_memory_more_negative_step_correlation": float(elastic_pass),
        "full_model_joint_signature_pass": float(full_pass),
        "synthetic_capability_only": 1.0,
        "next_required_action": "infer_q_z_proxies_from_ka_calibration",
    }
    for flag in CLAIM_FLAGS:
        gate[flag] = 0.0
    return rows, gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    args = parser.parse_args()
    rows, gate = run_ablation(seed=args.seed, quick=args.quick)
    write_rows(args.output_rows, rows)
    write_rows(args.output_gate, [gate])


if __name__ == "__main__":
    main()
