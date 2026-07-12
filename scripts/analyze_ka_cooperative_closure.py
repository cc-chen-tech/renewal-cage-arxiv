#!/usr/bin/env python3
"""Test a calibration-only cooperative-cluster diffusion closure on held-out KA data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    cooperative_cluster_diffusion_coefficient,
    event_activity_duration_statistics,
    event_conditioned_neighbor_displacement,
    isolated_event_response_amplitude,
    load_lammps_custom_trajectory,
    spatiotemporal_event_cluster_statistics,
    summarize_heldout_event_transport,
    summarize_neighbor_halo_replicates,
    trajectory_diffusion_estimate,
)
from renewal_cage import (  # noqa: E402
    event_clock_statistics,
    extract_nonrecrossing_phop_events,
    phop_values,
)


def parse_edges(value: str) -> np.ndarray:
    return np.array([float(item) for item in value.split(",")], dtype=float)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def integrated_excess(
    rows: list[dict[str, object]],
    *,
    sampled_event_count: float,
    maximum_distance: float,
) -> float:
    return float(
        sum(
            float(row["event_pair_count"])
            / sampled_event_count
            * (
                float(row["event_mean_squared_displacement"])
                - float(row["control_mean_squared_displacement"])
            )
            for row in rows
            if float(row["distance_high"]) <= maximum_distance
        )
    )


def resolve_halo_radius(
    calibration_radius: float,
    fixed_radius: float | None,
) -> tuple[float, str]:
    if calibration_radius <= 0.0:
        raise ValueError("calibration halo radius must be positive")
    if fixed_radius is None:
        return calibration_radius, "calibration_ci"
    if fixed_radius <= 0.0:
        raise ValueError("fixed halo radius must be positive")
    return fixed_radius, "posthoc_sensitivity"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ensemble_directory", type=Path)
    parser.add_argument("--calibration-time", type=int, default=5000)
    parser.add_argument("--heldout-diffusion-lag", type=int, default=4096)
    parser.add_argument("--origin-stride", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--sample-events", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=84517)
    parser.add_argument(
        "--distance-edges",
        type=parse_edges,
        default=parse_edges("0,1.4,2,3,4,5,7,10,13.1"),
    )
    parser.add_argument("--response-lag-exchange-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--maximum-coverage", type=float, default=1.2)
    parser.add_argument("--fixed-halo-radius", type=float)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    if args.sample_events < 0:
        raise ValueError("sample-events must be zero for all events or a positive count")
    if not 0.0 < args.response_lag_exchange_fraction < 0.5:
        raise ValueError("response-lag-exchange-fraction must lie between zero and one half")
    manifest = json.loads((args.ensemble_directory / "ensemble_manifest.json").read_text())
    production_time = int(round(float(manifest["production_time_tau"])))
    if production_time != 2 * args.calibration_time:
        raise ValueError("closure requires equal calibration and held-out windows")

    shell_rows: list[dict[str, object]] = []
    provisional_halo_rows: list[dict[str, object]] = []
    stored: list[dict[str, object]] = []
    for replicate in manifest["replicates"]:
        replicate_index = int(replicate["replicate"])
        directory = args.ensemble_directory / str(replicate["directory"])
        if not (directory / "COMPLETE").is_file():
            raise ValueError(f"replicate {replicate_index} is not marked COMPLETE")
        trajectory = load_lammps_custom_trajectory(directory / "trajectory.lammpstrj")
        positions = trajectory["unwrapped_positions"]
        a_positions = positions[:, trajectory["particle_types"] == 0]
        calibration = a_positions[: args.calibration_time + 1]
        heldout = a_positions[args.calibration_time :]
        activity_times, activity = phop_values(calibration, half_window=args.half_window)
        events = extract_nonrecrossing_phop_events(
            calibration,
            threshold=args.threshold,
            half_window=args.half_window,
            recrossing_radius=math.sqrt(args.threshold),
            activity_times=activity_times,
            activity_values=activity,
        )
        valid = np.flatnonzero(
            (events["time"] >= args.half_window)
            & (events["time"] + args.half_window <= len(calibration))
        )
        if args.sample_events == 0:
            selected = valid
        else:
            if len(valid) < args.sample_events:
                raise ValueError(f"replicate {replicate_index} has too few complete events")
            rng = np.random.default_rng(args.random_seed + replicate_index)
            selected = np.sort(rng.choice(valid, size=args.sample_events, replace=False))
        rng = np.random.default_rng(args.random_seed + 1000 + replicate_index)
        event_particles = events["particle"][selected]
        controls = rng.integers(0, calibration.shape[1] - 1, size=len(selected))
        controls += controls >= event_particles
        local_shells, halo = event_conditioned_neighbor_displacement(
            calibration,
            events,
            box_lengths=trajectory["box_lengths"],
            distance_edges=args.distance_edges,
            half_window=args.half_window,
            event_indices=selected,
            control_particles=controls,
            integration_max_distance=float(args.distance_edges[-1]),
        )
        for row in local_shells:
            row["replicate"] = float(replicate_index)
            row["temperature"] = float(manifest["temperature"])
            row["threshold"] = args.threshold
            row["control_definition"] = "same_time_uniform_random_non_focal_particle"
        shell_rows.extend(local_shells)
        halo["replicate"] = float(replicate_index)
        provisional_halo_rows.append(halo)

        event_stats = event_clock_statistics(
            events,
            duration=float(args.calibration_time),
            particle_count=calibration.shape[1],
            dimension=calibration.shape[2],
        )
        duration = event_activity_duration_statistics(
            activity_times,
            activity,
            events,
            threshold=args.threshold,
        )
        response_lag = max(
            args.half_window,
            int(round(args.response_lag_exchange_fraction * event_stats["exchange_mean"])),
        )
        response = isolated_event_response_amplitude(
            calibration,
            events,
            response_lag=response_lag,
            half_window=args.half_window,
        )
        observed = trajectory_diffusion_estimate(
            heldout,
            lag=args.heldout_diffusion_lag,
            origin_stride=args.origin_stride,
        )
        stored.append(
            {
                "replicate": float(replicate_index),
                "events": events,
                "box_lengths": trajectory["box_lengths"],
                "particle_count": float(calibration.shape[1]),
                "event_stats": event_stats,
                "duration": duration,
                "response": response,
                "observed_diffusion": observed,
                "sampled_event_count": float(len(selected)),
                "mean_self_jump_squared": halo["mean_self_jump_squared"],
            }
        )

    _, provisional_verdict = summarize_neighbor_halo_replicates(
        shell_rows,
        provisional_halo_rows,
    )
    halo_radius, halo_radius_source = resolve_halo_radius(
        float(provisional_verdict["halo_radius_lower_bound"]),
        args.fixed_halo_radius,
    )

    micro_rows: list[dict[str, object]] = []
    final_halo_rows: list[dict[str, object]] = []
    for item in stored:
        replicate_index = int(float(item["replicate"]))
        local_shells = [
            row for row in shell_rows if int(float(row["replicate"])) == replicate_index
        ]
        halo_excess = integrated_excess(
            local_shells,
            sampled_event_count=float(item["sampled_event_count"]),
            maximum_distance=halo_radius,
        )
        cluster_time = int(float(item["duration"]["cluster_time_window"]))
        cluster = spatiotemporal_event_cluster_statistics(
            item["events"],
            box_lengths=item["box_lengths"],
            maximum_time_separation=cluster_time,
            maximum_distance=halo_radius,
        )
        event_stats = item["event_stats"]
        response = item["response"]
        event_rate = float(event_stats["event_count"]) / (
            float(item["particle_count"]) * args.calibration_time
        )
        cooperative = cooperative_cluster_diffusion_coefficient(
            event_rate=event_rate,
            self_jump_squared=float(item["mean_self_jump_squared"]),
            integrated_neighbor_excess=halo_excess,
            response_amplitude=float(response["mean_response_amplitude"]),
            mean_cluster_size=float(cluster["mean_cluster_size"]),
            dimension=3,
        )
        observed = float(item["observed_diffusion"])
        micro_rows.append(
            {
                "replicate": float(replicate_index),
                "temperature": float(manifest["temperature"]),
                "threshold": args.threshold,
                "calibration_time": float(args.calibration_time),
                "heldout_time": float(production_time - args.calibration_time),
                "halo_radius": halo_radius,
                "halo_radius_source": halo_radius_source,
                "event_rate": event_rate,
                "mean_self_jump_squared": float(item["mean_self_jump_squared"]),
                "integrated_neighbor_excess": halo_excess,
                "integrated_neighbor_excess_over_self_jump_squared": halo_excess
                / float(item["mean_self_jump_squared"]),
                "median_event_duration": float(item["duration"]["median_duration"]),
                "cluster_time_window": float(cluster_time),
                "cluster_count": float(cluster["cluster_count"]),
                "mean_cluster_size": float(cluster["mean_cluster_size"]),
                "nontrivial_event_fraction": float(cluster["nontrivial_event_fraction"]),
                "response_lag": float(response["response_lag"]),
                "isolated_event_count": float(response["isolated_event_count"]),
                "mean_response_amplitude": float(response["mean_response_amplitude"]),
                "observed_diffusion": observed,
                "uncorrelated_event_diffusion": float(event_stats["uncorrelated_diffusion"]),
                "correlated_event_diffusion": float(event_stats["correlated_diffusion"]),
                "cooperative_cluster_diffusion": cooperative,
                "cooperative_cluster_coverage": cooperative / observed,
                "macro_fit_parameter_count": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
        final_halo_rows.append(
            {
                "replicate": float(replicate_index),
                "integrated_neighbor_excess_over_self_jump_squared": halo_excess
                / float(item["mean_self_jump_squared"]),
            }
        )

    halo_curve, halo_verdict = summarize_neighbor_halo_replicates(
        shell_rows,
        final_halo_rows,
    )
    summary, verdict = summarize_heldout_event_transport(
        micro_rows,
        minimum_coverage=args.minimum_coverage,
        maximum_coverage=args.maximum_coverage,
        model_columns={
            "uncorrelated_event_clock": "uncorrelated_event_diffusion",
            "correlated_event_clock": "correlated_event_diffusion",
            "cooperative_cluster_response": "cooperative_cluster_diffusion",
        },
        primary_model="cooperative_cluster_response",
    )
    baseline = next(row for row in summary if row["model"] == "correlated_event_clock")
    cooperative = next(row for row in summary if row["model"] == "cooperative_cluster_response")
    verdict.update(
        {
            "temperature": float(manifest["temperature"]),
            "threshold": args.threshold,
            "calibration_time": float(args.calibration_time),
            "heldout_time": float(production_time - args.calibration_time),
            "halo_radius": halo_radius,
            "halo_radius_source": halo_radius_source,
            "mean_absolute_coverage_error_correlated_event_clock": abs(
                float(baseline["mean_coverage"]) - 1.0
            ),
            "mean_absolute_coverage_error_cooperative_cluster": abs(
                float(cooperative["mean_coverage"]) - 1.0
            ),
            "cooperative_model_improves_mean_coverage": float(
                abs(float(cooperative["mean_coverage"]) - 1.0)
                < abs(float(baseline["mean_coverage"]) - 1.0)
            ),
            "posthoc_sensitivity_only": float(halo_radius_source == "posthoc_sensitivity"),
            "spatial_model_claim_allowed": float(
                verdict["heldout_transport_pass"]
                and halo_radius_source == "calibration_ci"
            ),
        }
    )
    for row in summary:
        row["temperature"] = float(manifest["temperature"])
        row["threshold"] = args.threshold
        row["calibration_time"] = float(args.calibration_time)
        row["heldout_time"] = float(production_time - args.calibration_time)
        row["thermodynamic_claim_allowed"] = 0.0
    for row in halo_curve:
        row["temperature"] = float(manifest["temperature"])
        row["threshold"] = args.threshold
    halo_verdict["temperature"] = float(manifest["temperature"])
    halo_verdict["threshold"] = args.threshold
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_halo_shell_replicates.csv"), shell_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_halo_curve_summary.csv"), halo_curve)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_halo_verdict.csv"), [halo_verdict])
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_micro_inputs.csv"), micro_rows)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_verdict.csv"), [verdict])


if __name__ == "__main__":
    main()
