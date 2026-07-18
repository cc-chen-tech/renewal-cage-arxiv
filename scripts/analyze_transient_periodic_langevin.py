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


def render_ablation_svg(
    rows: list[dict[str, float | str]],
    gate: dict[str, float | str],
) -> str:
    """Render a deterministic compact comparison of rate and recoil metrics."""

    if [str(row["model"]) for row in rows] != [
        "static_periodic",
        "rate_only",
        "elastic_only",
        "full_transient",
    ]:
        raise ValueError("ablation rows must use the frozen model order")
    colors = ("#687078", "#2f6fb0", "#2f8a62", "#c84d43")
    labels = ("static", "rate only", "elastic only", "full")
    fano_values = [float(row["count_fano_factor"]) for row in rows]
    correlation_values = [float(row["successive_vector_correlation"]) for row in rows]
    if any(not math.isfinite(value) for value in fano_values + correlation_values):
        raise ValueError("ablation metrics must be finite")
    fano_top = max(1.5, math.ceil(max(fano_values) * 10.0) / 10.0 + 0.1)
    plot_top = 108.0
    plot_bottom = 360.0
    plot_height = plot_bottom - plot_top
    fano_y = lambda value: plot_bottom - plot_height * value / fano_top
    correlation_low = -0.12
    correlation_high = 0.04
    correlation_y = lambda value: plot_top + plot_height * (
        correlation_high - value
    ) / (correlation_high - correlation_low)
    correlation_zero = correlation_y(0.0)
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="520" viewBox="0 0 980 520">',
        '<rect x="0" y="0" width="980" height="520" fill="#ffffff"/>',
        '<text x="490" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="19" font-weight="700" fill="#1f2529">Continuous transient-periodic Langevin ablation</text>',
        '<text x="490" y="54" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#4a5359">same thermal SDE and seed; barrier disorder and elastic recoil are switched separately</text>',
        '<text x="260" y="84" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#1f2529">count Fano factor</text>',
        '<text x="740" y="84" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#1f2529">successive cage-step correlation</text>',
        '<line x1="70" y1="360" x2="450" y2="360" stroke="#3f474c" stroke-width="1"/>',
        f'<line x1="550" y1="{correlation_zero:.3f}" x2="930" y2="{correlation_zero:.3f}" stroke="#3f474c" stroke-width="1"/>',
        f'<text x="558" y="{correlation_zero - 5.0:.3f}" font-family="Arial, sans-serif" font-size="10" fill="#5b646a">zero</text>',
    ]
    for tick in (0.0, 0.5, 1.0, fano_top):
        y = fano_y(tick)
        lines.append(
            f'<line x1="66" y1="{y:.3f}" x2="450" y2="{y:.3f}" stroke="#e1e5e7" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="60" y="{y + 4.0:.3f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#5b646a">{tick:g}</text>'
        )
    for tick in (-0.12, -0.08, -0.04, 0.0, 0.04):
        y = correlation_y(tick)
        lines.append(
            f'<line x1="546" y1="{y:.3f}" x2="930" y2="{y:.3f}" stroke="#e1e5e7" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="540" y="{y + 4.0:.3f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10" fill="#5b646a">{tick:.2f}</text>'
        )
    for index, (label, color, fano, correlation) in enumerate(
        zip(labels, colors, fano_values, correlation_values)
    ):
        left_x = 92.0 + 88.0 * index
        fano_top_y = fano_y(fano)
        lines.extend(
            [
                f'<rect x="{left_x:.1f}" y="{fano_top_y:.3f}" width="54" height="{plot_bottom - fano_top_y:.3f}" fill="{color}"/>',
                f'<text x="{left_x + 27.0:.1f}" y="{fano_top_y - 7.0:.3f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#1f2529">{fano:.3f}</text>',
                f'<text x="{left_x + 27.0:.1f}" y="382" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333b40">{label}</text>',
            ]
        )
        right_x = 572.0 + 88.0 * index
        value_y = correlation_y(correlation)
        rectangle_y = min(correlation_zero, value_y)
        rectangle_height = abs(value_y - correlation_zero)
        lines.extend(
            [
                f'<rect x="{right_x:.1f}" y="{rectangle_y:.3f}" width="54" height="{rectangle_height:.3f}" fill="{color}"/>',
                f'<text x="{right_x + 27.0:.1f}" y="{value_y + 15.0:.3f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#1f2529">{correlation:.3f}</text>',
                f'<text x="{right_x + 27.0:.1f}" y="382" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#333b40">{label}</text>',
            ]
        )
    lines.extend(
        [
            '<rect x="70" y="414" width="860" height="67" fill="#f5f7f8" stroke="#d4dade"/>',
            f'<text x="90" y="438" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#1f2529">rate Fano increase: {int(float(gate["rate_disorder_count_fano_increase"]))}   elastic recoil ordering: {int(float(gate["elastic_memory_more_negative_step_correlation"]))}   full joint signature: {int(float(gate["full_model_joint_signature_pass"]))}</text>',
            '<text x="90" y="461" font-family="Arial, sans-serif" font-size="12" fill="#4a5359">Synthetic capability only. No KA hidden coordinate, spatial facilitation, thermodynamic transition, or blind macro closure is identified.</text>',
            '<text x="490" y="505" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#687078">continuous trajectories; non-recrossing dwell = 5 recorded frames</text>',
            '</svg>',
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    args = parser.parse_args()
    rows, gate = run_ablation(seed=args.seed, quick=args.quick)
    write_rows(args.output_rows, rows)
    write_rows(args.output_gate, [gate])
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    args.output_figure.write_text(render_ablation_svg(rows, gate))


if __name__ == "__main__":
    main()
