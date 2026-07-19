#!/usr/bin/env python3
"""Generate lag-conditioned KA event-transfer counterfactuals from trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_replicates import (  # noqa: E402
    extract_debye_waller_cage_jumps,
    independent_isotropic_channel_moments,
    interval_conditioned_event_statistics,
    mix_interval_count_and_path_kernel,
    pool_interval_path_kernels,
    position_fluctuation_values,
)


LAGS_BY_TEMPERATURE = {
    "045": (20, 50, 100, 200, 500, 1000, 2000, 3000, 4096),
    "058": (10, 20, 50, 100, 200, 400, 600),
}
CALIBRATION_TIME = {"045": 5000, "058": 750}
WAVE_NUMBERS = np.array([2.0, 4.0, 7.25])
ORIGIN_STRIDE = 8


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty event-transfer table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wave_key(wave_number: float) -> str:
    return f"k{wave_number:g}".replace(".", "p")


def canonicalize_interval_statistics(
    statistics: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map legacy scratch-cache names to the public sufficient-statistic schema."""

    aliases = {
        "count_pmf": ("count_pmf", "pmf"),
        "sample_count": ("sample_count", "samples"),
        "conditional_msd": ("conditional_msd", "msd"),
        "conditional_fourth_moment": ("conditional_fourth_moment", "fourth"),
    }
    result: dict[str, np.ndarray] = {}
    for output, candidates in aliases.items():
        source = next((candidate for candidate in candidates if candidate in statistics), None)
        if source is None:
            raise ValueError(f"interval cache lacks {output}")
        result[output] = np.asarray(statistics[source])
    for wave_number in WAVE_NUMBERS:
        suffix = wave_key(float(wave_number))
        output = f"conditional_characteristic_{suffix}"
        candidates = (
            output,
            f"fs_k{wave_number:g}",
            f"fs_{suffix}",
        )
        source = next((candidate for candidate in candidates if candidate in statistics), None)
        if source is None:
            raise ValueError(f"interval cache lacks {output}")
        result[output] = np.asarray(statistics[source])
    lengths = {len(values) for values in result.values()}
    if len(lengths) != 1:
        raise ValueError("interval cache arrays are not aligned")
    return result


def save_interval_statistics(
    path: Path, statistics: dict[int, dict[str, np.ndarray]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        f"lag_{lag}_{key}": np.asarray(value)
        for lag, local in statistics.items()
        for key, value in canonicalize_interval_statistics(local).items()
    }
    np.savez_compressed(path, **arrays)


def load_interval_statistics(
    path: Path, lags: tuple[int, ...]
) -> dict[int, dict[str, np.ndarray]]:
    with np.load(path) as archive:
        result = {}
        for lag in lags:
            prefix = f"lag_{lag}_"
            local = {
                key[len(prefix) :]: np.asarray(archive[key])
                for key in archive.files
                if key.startswith(prefix)
            }
            if not local:
                raise ValueError(f"interval cache lacks lag {lag}")
            result[lag] = canonicalize_interval_statistics(local)
    return result


def load_a_positions(
    path: Path,
    *,
    expected_frame_count: int,
    first_frame: int,
    stop_frame: int,
) -> np.ndarray:
    """Stream only type-A unwrapped positions into a bounded float32 cube."""

    if not 0 <= first_frame < stop_frame <= expected_frame_count:
        raise ValueError("requested frame window lies outside the trajectory")
    positions: np.ndarray | None = None
    type_a: np.ndarray | None = None
    frame = 0
    with path.open() as handle:
        while True:
            marker = handle.readline()
            if marker == "":
                break
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError("unexpected timestep header")
            handle.readline()
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("unexpected atom-count header")
            particle_count = int(handle.readline())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("unexpected box header")
            bounds = np.array(
                [[float(value) for value in handle.readline().split()[:2]] for _ in range(3)]
            )
            box_lengths = bounds[:, 1] - bounds[:, 0]
            if handle.readline().strip() != "ITEM: ATOMS id type x y z ix iy iz":
                raise ValueError("unexpected atom schema")
            values = np.fromstring(
                "".join(handle.readline() for _ in range(particle_count)), sep=" "
            ).reshape(particle_count, 8)
            if frame == 0:
                type_a = values[:, 1].astype(int) == 1
                positions = np.empty(
                    (stop_frame - first_frame, int(np.sum(type_a)), 3), dtype=np.float32
                )
            if frame >= expected_frame_count:
                raise ValueError("trajectory contains more frames than its manifest")
            assert positions is not None and type_a is not None
            if first_frame <= frame < stop_frame:
                positions[frame - first_frame] = (
                    values[type_a, 2:5] + values[type_a, 5:8] * box_lengths
                ).astype(np.float32)
            frame += 1
    if positions is None or frame != expected_frame_count:
        raise ValueError(
            f"trajectory frame count {frame} does not match expected {expected_frame_count}"
        )
    return positions


def chunked_position_fluctuation_values(
    positions: np.ndarray,
    *,
    half_window: int,
    particle_chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the established activity rule with bounded prefix-array memory."""

    times = np.arange(half_window, len(positions) - half_window, dtype=int)
    activity = np.empty((len(times), positions.shape[1]), dtype=np.float32)
    for first in range(0, positions.shape[1], particle_chunk):
        stop = min(first + particle_chunk, positions.shape[1])
        local_times, local_activity = position_fluctuation_values(
            positions[:, first:stop], half_window=half_window
        )
        if not np.array_equal(local_times, times):
            raise ValueError("chunked activity times are inconsistent")
        activity[:, first:stop] = local_activity.astype(np.float32)
    return times, activity


def _rows_by_key(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, str]]:
    return {
        (int(float(row["replicate"])), int(float(row["lag"]))): row for row in rows
    }


def combine_event_and_residual(
    event: dict[str, float], residual: dict[str, str]
) -> dict[str, float]:
    combined = independent_isotropic_channel_moments(
        first_msd=event["event_msd"],
        first_ngp=event["event_ngp"],
        second_msd=float(residual["residual_msd"]),
        second_ngp=float(residual["residual_ngp"]),
        dimension=3,
    )
    result = {
        "msd": combined["combined_msd"],
        "ngp": combined["combined_ngp"],
    }
    for wave_number in WAVE_NUMBERS:
        suffix = wave_key(float(wave_number))
        result[f"fs_{suffix}"] = (
            event[f"event_characteristic_{suffix}"]
            * float(residual[f"residual_fs_{suffix}"])
        )
    return result


def _default_trajectory_root() -> Path:
    configured = os.environ.get("KA_REPLICATE_ROOT")
    if configured:
        return Path(configured)
    local = ROOT / "tmp" / "ka_replicates"
    if local.is_dir():
        return local
    if ROOT.parent.name == ".worktrees":
        return ROOT.parent.parent / "tmp" / "ka_replicates"
    return local


def _prediction_metrics(
    event: dict[str, float],
    predicted: dict[str, float],
    target: dict[str, str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "count_tail_probability": event["omitted_count_probability"],
        "predicted_msd": predicted["msd"],
        "observed_msd": float(target["observed_msd"]),
        "msd_relative_error": abs(
            predicted["msd"] / float(target["observed_msd"]) - 1.0
        ),
        "predicted_ngp": predicted["ngp"],
        "observed_ngp": float(target["observed_ngp"]),
        "ngp_absolute_error": abs(
            predicted["ngp"] - float(target["observed_ngp"])
        ),
    }
    for wave_number in WAVE_NUMBERS:
        suffix = wave_key(float(wave_number))
        row[f"predicted_fs_{suffix}"] = predicted[f"fs_{suffix}"]
        row[f"observed_fs_{suffix}"] = float(target[f"observed_fs_{suffix}"])
        row[f"absolute_error_fs_{suffix}"] = abs(
            predicted[f"fs_{suffix}"] - float(target[f"observed_fs_{suffix}"])
        )
    return row


def generate_tables(
    *,
    trajectory_root: Path,
    cache_directory: Path,
    labels: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detailed: list[dict[str, object]] = []
    ablation: list[dict[str, object]] = []
    for label in labels:
        ensemble_directory = trajectory_root / f"T{label}"
        manifest = json.loads((ensemble_directory / "ensemble_manifest.json").read_text())
        expected_frame_count = int(round(float(manifest["production_time_tau"]))) + 1
        calibration_time = CALIBRATION_TIME[label]
        lags = LAGS_BY_TEMPERATURE[label]
        prefix = ROOT / "data" / f"renewal_cage_ka_replicates_T{label}"
        calibration_residual = _rows_by_key(
            read_rows(
                prefix.with_name(
                    prefix.name + "_calibration_event_channels_factorization_rows.csv"
                )
            )
        )
        heldout_residual = _rows_by_key(
            read_rows(prefix.with_name(prefix.name + "_event_oracle_factorization_rows.csv"))
        )
        thresholds = {
            int(float(row["replicate"])): float(row["debye_waller_factor"])
            for row in read_rows(
                prefix.with_name(prefix.name + "_debye_waller_heldout_replicates.csv")
            )
        }
        for replicate_spec in manifest["replicates"]:
            replicate = int(replicate_spec["replicate"])
            directory = ensemble_directory / str(replicate_spec["directory"])
            statistics: dict[str, dict[int, dict[str, np.ndarray]]] = {}
            for window, first_frame, stop_frame in (
                ("calibration", 0, calibration_time + 1),
                ("heldout", calibration_time, expected_frame_count),
            ):
                cache_path = cache_directory / f"T{label}_replicate_{replicate:02d}_{window}.npz"
                if cache_path.is_file():
                    statistics[window] = load_interval_statistics(cache_path, lags)
                    continue
                positions = load_a_positions(
                    directory / "trajectory.lammpstrj",
                    expected_frame_count=expected_frame_count,
                    first_frame=first_frame,
                    stop_frame=stop_frame,
                )
                activity_times, activity = chunked_position_fluctuation_values(
                    positions, half_window=5
                )
                events = extract_debye_waller_cage_jumps(
                    positions,
                    debye_waller_factor=thresholds[replicate],
                    half_window=5,
                    activity_times=activity_times,
                    activity_values=activity,
                )
                statistics[window] = interval_conditioned_event_statistics(
                    events,
                    frame_count=len(positions),
                    particle_count=positions.shape[1],
                    lags=lags,
                    wave_numbers=WAVE_NUMBERS,
                    origin_stride=ORIGIN_STRIDE,
                )
                save_interval_statistics(cache_path, statistics[window])
            pooled_calibration_kernel = pool_interval_path_kernels(
                statistics["calibration"], wave_numbers=WAVE_NUMBERS
            )
            for lag in lags:
                target = heldout_residual[(replicate, lag)]
                for count_source in ("calibration", "heldout"):
                    for kernel_source in ("calibration", "heldout"):
                        event = mix_interval_count_and_path_kernel(
                            statistics[count_source][lag],
                            statistics[kernel_source][lag],
                            wave_numbers=WAVE_NUMBERS,
                            dimension=3,
                        )
                        for residual_source, residual in (
                            ("calibration", calibration_residual[(replicate, lag)]),
                            ("heldout", heldout_residual[(replicate, lag)]),
                        ):
                            predicted = combine_event_and_residual(event, residual)
                            row: dict[str, object] = {
                                "temperature": float(label) / 100.0,
                                "replicate": float(replicate),
                                "lag": float(lag),
                                "count_source": count_source,
                                "kernel_source": kernel_source,
                                "residual_source": residual_source,
                                "model_code": "".join(
                                    "h" if source == "heldout" else "c"
                                    for source in (count_source, kernel_source, residual_source)
                                ),
                                **_prediction_metrics(event, predicted, target),
                            }
                            detailed.append(row)
                for kernel_model, kernel in (
                    ("lag_conditioned", statistics["calibration"][lag]),
                    ("lag_pooled", pooled_calibration_kernel),
                ):
                    event = mix_interval_count_and_path_kernel(
                        statistics["calibration"][lag],
                        kernel,
                        wave_numbers=WAVE_NUMBERS,
                        dimension=3,
                    )
                    predicted = combine_event_and_residual(
                        event, calibration_residual[(replicate, lag)]
                    )
                    ablation.append(
                        {
                            "temperature": float(label) / 100.0,
                            "replicate": float(replicate),
                            "lag": float(lag),
                            "kernel_model": kernel_model,
                            **_prediction_metrics(event, predicted, target),
                        }
                    )
    return detailed, ablation


def generate_rows(
    *,
    trajectory_root: Path,
    cache_directory: Path,
    labels: tuple[str, ...],
) -> list[dict[str, object]]:
    """Compatibility wrapper returning the complete three-channel cube."""

    return generate_tables(
        trajectory_root=trajectory_root,
        cache_directory=cache_directory,
        labels=labels,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-root", type=Path, default=_default_trajectory_root())
    parser.add_argument("--cache-directory", type=Path, default=Path("/tmp/ka_interval_stats"))
    parser.add_argument("--temperature", choices=("045", "058", "both"), default="both")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "renewal_cage_ka_interval_conditioned_event_transfer_rows.csv",
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=ROOT
        / "data"
        / "renewal_cage_ka_interval_conditioned_event_transfer_kernel_ablation_rows.csv",
    )
    arguments = parser.parse_args()
    labels = ("045", "058") if arguments.temperature == "both" else (arguments.temperature,)
    rows, ablation = generate_tables(
        trajectory_root=arguments.trajectory_root,
        cache_directory=arguments.cache_directory,
        labels=labels,
    )
    write_rows(arguments.output, rows)
    write_rows(arguments.ablation_output, ablation)
    print(f"wrote {len(rows)} three-channel rows to {arguments.output}")
    print(f"wrote {len(ablation)} kernel-ablation rows to {arguments.ablation_output}")


if __name__ == "__main__":
    main()
