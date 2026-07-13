#!/usr/bin/env python3
"""Run the fixed, time-split event-clock closure on one KA pickle trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prl_event_clock_closure import time_split_event_clock_closure  # noqa: E402
from renewal_cage import periodic_unwrap_trajectory  # noqa: E402


def parse_numbers(value: str, dtype: type[float] | type[int]) -> np.ndarray:
    return np.array([dtype(item) for item in value.split(",")], dtype=dtype)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_a_positions(input_path: Path) -> np.ndarray:
    """Load A-particle coordinates from either the public pickle or local KA dump."""

    if input_path.suffix == ".lammpstrj":
        from ka_replicates import load_lammps_custom_trajectory  # noqa: E402

        trajectory = load_lammps_custom_trajectory(input_path)
        return np.asarray(trajectory["unwrapped_positions"][:, trajectory["particle_types"] == 0], dtype=float)
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)
    box_lengths = np.asarray(payload["Box_size"][0], dtype=float)
    particle_types = np.asarray(payload["Particle_types"][0])
    a_indices = np.flatnonzero(particle_types == 0)
    wrapped = np.stack([frame[a_indices] for frame in payload["Positions"]]).astype(np.float32, copy=False)
    return periodic_unwrap_trajectory(wrapped, box_lengths)


def run(input_path: Path, args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    unwrapped = load_a_positions(input_path)
    train_stop = args.train_stop if args.train_stop is not None else (len(unwrapped) - 1) // 2
    result = time_split_event_clock_closure(
        unwrapped,
        train_stop=train_stop,
        threshold=args.threshold,
        half_window=args.half_window,
        lags=parse_numbers(args.lags, int),
        wave_numbers=parse_numbers(args.wave_numbers, float),
        overlap_radius=args.overlap_radius,
        origin_stride=args.origin_stride,
    )
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for variant, score in result["variant_scores"].items():
        summary = {
            key: value
            for key, value in result.items()
            if key not in {"variant_scores", "observed_lags", "observed_ngp", "variants"}
        }
        summary.update(
            {
                "variant": variant,
                "source_doi": "10.5281/zenodo.7469766",
                "source_class": "real_public_continuous_trajectory"
                if input_path.suffix != ".lammpstrj"
                else "real_reproduced_KA_trajectory",
                "source_path": str(input_path),
                "prediction_scope": "time_split_train_microstatistics_to_heldout_macro_observables",
                "observed_diffusion": score["observed_diffusion"],
                "predicted_diffusion": score["predicted_diffusion"],
                "diffusion_relative_error": score["diffusion_relative_error"],
                "ngp_relative_rmse": score["ngp"]["relative_rmse"],
                "ngp_max_relative_error": score["ngp"]["max_relative_error"],
                "predicted_count_fano_proxy": score["predicted_count_fano_proxy"],
                "observed_overlap_chi4_peak": score["observed_overlap_chi4_peak"],
                "finite_flight_duration": score["finite_flight_duration"],
                "multi_k_scores_json": json.dumps(score["fs"], sort_keys=True),
                "macro_fit_parameter_count": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
        )
        summary_rows.append(summary)
        for index, (lag, observed_ngp) in enumerate(zip(result["observed_lags"], result["observed_ngp"])):
            curve_rows.append(
                {
                    "variant": variant,
                    "lag": float(lag),
                    "observed_ngp_3d": float(observed_ngp),
                    "predicted_ngp_3d": float(score["predicted_ngp"][index]),
                    "predicted_fs_by_k_json": json.dumps(
                        {str(k): float(values[index]) for k, values in score["predicted_fs"].items()},
                        sort_keys=True,
                    ),
                    "source_doi": "10.5281/zenodo.7469766",
                    "event_definition": result["event_definition"],
                    "threshold": result["threshold"],
                    "half_window": result["half_window"],
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    return summary_rows, curve_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--train-stop", type=int)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--lags", default="8,16,32,64,128,256,512,1024,2048,4096")
    parser.add_argument("--wave-numbers", default="5,7.25,9")
    parser.add_argument("--overlap-radius", type=float, default=0.3)
    parser.add_argument("--origin-stride", type=int, default=8)
    args = parser.parse_args()
    summary, curves = run(args.input, args)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary)
    write_rows(args.output_prefix.with_name(args.output_prefix.name + "_curves.csv"), curves)


if __name__ == "__main__":
    main()
