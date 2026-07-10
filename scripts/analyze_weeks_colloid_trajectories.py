#!/usr/bin/env python3
"""Recompute true-time diagnostics from the public Vivek et al. 2D data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))

from renewal_cage import (  # noqa: E402
    censored_event_clock_identifiability_verdict,
    weeks_colloid_true_time_verdict,
)


BASE_URL = "https://faculty.college.emory.edu/sites/weeks/data/2dskanda"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    area_fraction: float
    time_step_seconds: float
    published_tau_alpha_seconds: float

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.sample_id}.zip"


SAMPLES = (
    Sample("t2_10_29b", 0.76, 16.5, 3000.0),
    Sample("t2_10_30b", 0.78, 16.5, 14000.0),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_sample(sample: Sample, cache_dir: Path, *, fetch: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sample.sample_id}.zip"
    if path.exists():
        return path
    if not fetch:
        raise FileNotFoundError(
            f"missing {path}; rerun with --fetch to retrieve the public source"
        )
    urllib.request.urlretrieve(sample.url, path)  # noqa: S310 - fixed public source above
    return path


def load_tracks(path: Path) -> tuple[list[tuple[int, np.ndarray, np.ndarray, dict[int, int]]], list[dict[int, np.ndarray]]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".txt"):
            raise ValueError(f"unexpected public trajectory archive layout: {path}")
        with archive.open(names[0]) as handle:
            table = np.loadtxt(handle)

    if table.ndim != 2 or table.shape[1] != 5:
        raise ValueError("Weeks trajectory table must have x,y,d,t,id columns")
    time = table[:, 3].astype(int)
    particle_id = table[:, 4].astype(int)
    if time.min() != 0 or not np.array_equal(np.unique(time), np.arange(time.max() + 1)):
        raise ValueError("Weeks trajectory time indices must be consecutive and start at zero")

    tracks: list[tuple[int, np.ndarray, np.ndarray, dict[int, int]]] = []
    for identifier in np.unique(particle_id):
        rows = table[particle_id == identifier]
        times = rows[:, 3].astype(int)
        positions = rows[:, :2]
        tracks.append((int(identifier), times, positions, {int(value): index for index, value in enumerate(times)}))

    frames: list[dict[int, np.ndarray]] = []
    for frame_time in range(time.max() + 1):
        rows = table[time == frame_time]
        frames.append({int(row[4]): row[:2] for row in rows})
    return tracks, frames


def lag_grid(frame_count: int, point_count: int = 24) -> np.ndarray:
    if frame_count < 4:
        raise ValueError("at least four frames are required")
    return np.unique(
        np.round(np.geomspace(1.0, max(2.0, 0.667 * (frame_count - 1)), point_count)).astype(int)
    )


def raw_lag_observables(
    tracks: list[tuple[int, np.ndarray, np.ndarray, dict[int, int]]],
    lag: int,
    wave_number: float,
) -> dict[str, float]:
    displacements: list[np.ndarray] = []
    for _, times, positions, index in tracks:
        pairs = [
            (current, index[int(time) + lag])
            for current, time in enumerate(times)
            if int(time) + lag in index
        ]
        if pairs:
            left, right = zip(*pairs)
            displacements.append(positions[np.asarray(right)] - positions[np.asarray(left)])
    if not displacements:
        raise ValueError("lag has no particle pairs")
    displacement = np.concatenate(displacements)
    squared = np.sum(displacement**2, axis=1)
    msd = float(np.mean(squared))
    ngp = float(np.mean(squared**2) / (2.0 * msd**2) - 1.0)
    self_scattering = float(
        0.5
        * (
            np.mean(np.cos(wave_number * displacement[:, 0]))
            + np.mean(np.cos(wave_number * displacement[:, 1]))
        )
    )
    return {
        "pair_count": float(len(displacement)),
        "msd": msd,
        "ngp_2d": ngp,
        "self_intermediate_scattering": self_scattering,
    }


def cage_relative_ngp(
    frames: list[dict[int, np.ndarray]],
    lag: int,
    neighbor_count: int = 6,
) -> dict[str, float]:
    """Measure a same-origin cage-relative displacement at one physical lag."""

    if neighbor_count < 1:
        raise ValueError("neighbor_count must be positive")
    displacements: list[np.ndarray] = []
    origin_count = 0
    for origin in range(len(frames) - lag):
        common_ids = sorted(set(frames[origin]).intersection(frames[origin + lag]))
        if len(common_ids) <= neighbor_count:
            continue
        initial = np.stack([frames[origin][identifier] for identifier in common_ids])
        final = np.stack([frames[origin + lag][identifier] for identifier in common_ids])
        displacement = final - initial
        separation = initial[:, None, :] - initial[None, :, :]
        distance_squared = np.sum(separation**2, axis=2)
        np.fill_diagonal(distance_squared, np.inf)
        neighbors = np.argpartition(distance_squared, neighbor_count - 1, axis=1)[:, :neighbor_count]
        displacements.append(displacement - np.mean(displacement[neighbors], axis=1))
        origin_count += 1
    if not displacements:
        raise ValueError("lag has no cage-relative particle pairs")
    displacement = np.concatenate(displacements)
    squared = np.sum(displacement**2, axis=1)
    msd = float(np.mean(squared))
    return {
        "pair_count": float(len(displacement)),
        "origin_count": float(origin_count),
        "msd": msd,
        "ngp_2d": float(np.mean(squared**2) / (2.0 * msd**2) - 1.0),
    }


def cage_relative_coordinates(
    frames: list[dict[int, np.ndarray]],
    neighbor_count: int = 6,
) -> list[dict[int, np.ndarray]]:
    """Return instantaneous six-neighbor cage-relative coordinates by particle ID."""

    if neighbor_count < 1:
        raise ValueError("neighbor_count must be positive")
    coordinates: list[dict[int, np.ndarray]] = []
    for frame in frames:
        identifiers = np.asarray(sorted(frame), dtype=int)
        if len(identifiers) <= neighbor_count:
            coordinates.append({})
            continue
        positions = np.stack([frame[int(identifier)] for identifier in identifiers])
        separation = positions[:, None, :] - positions[None, :, :]
        distance_squared = np.sum(separation**2, axis=2)
        np.fill_diagonal(distance_squared, np.inf)
        neighbors = np.argpartition(distance_squared, neighbor_count - 1, axis=1)[:, :neighbor_count]
        relative = positions - np.mean(positions[neighbors], axis=1)
        coordinates.append(
            {int(identifier): relative[index] for index, identifier in enumerate(identifiers)}
        )
    return coordinates


def contiguous_segments(
    times: np.ndarray,
    coordinates: np.ndarray,
    min_track_frames: int,
) -> list[np.ndarray]:
    """Split field-of-view tracks at gaps before applying an event protocol."""

    if min_track_frames < 2:
        raise ValueError("min_track_frames must be at least two")
    boundaries = np.flatnonzero(np.diff(times) != 1) + 1
    chunks = np.split(coordinates, boundaries)
    return [chunk for chunk in chunks if len(chunk) >= min_track_frames]


def candelier_jump_indices(
    coordinates: np.ndarray,
    cage_length_threshold: float,
    min_subsegment_frames: int = 50,
) -> list[int]:
    """Recursively locate Candelier maxima above a cage-length threshold.

    The separation score is a squared length, so the supplied cage-length
    threshold is squared before comparison.
    """

    if cage_length_threshold <= 0.0:
        raise ValueError("cage_length_threshold must be positive")
    if min_subsegment_frames < 2:
        raise ValueError("min_subsegment_frames must be at least two")

    def best_split(segment: np.ndarray) -> tuple[int | None, float]:
        size = len(segment)
        if size < 2 * min_subsegment_frames:
            return None, 0.0
        cumulative = np.concatenate(
            [np.zeros((1, segment.shape[1])), np.cumsum(segment, axis=0)]
        )
        cumulative_norm_squared = np.concatenate(
            [[0.0], np.cumsum(np.sum(segment**2, axis=1))]
        )
        candidates = np.arange(min_subsegment_frames, size - min_subsegment_frames + 1)
        left_count = candidates.astype(float)
        right_count = float(size) - left_count
        left_mean = cumulative[candidates] / left_count[:, None]
        right_mean = (cumulative[size] - cumulative[candidates]) / right_count[:, None]
        left_mean_squared = cumulative_norm_squared[candidates] / left_count
        right_mean_squared = (cumulative_norm_squared[size] - cumulative_norm_squared[candidates]) / right_count
        left_about_right = left_mean_squared - 2.0 * np.sum(left_mean * right_mean, axis=1) + np.sum(right_mean**2, axis=1)
        right_about_left = right_mean_squared - 2.0 * np.sum(right_mean * left_mean, axis=1) + np.sum(left_mean**2, axis=1)
        score = np.sqrt((left_count / size) * (right_count / size)) * np.sqrt(
            np.maximum(left_about_right, 0.0) * np.maximum(right_about_left, 0.0)
        )
        index = int(np.argmax(score))
        return int(candidates[index]), float(score[index])

    def split(offset: int, segment: np.ndarray) -> list[int]:
        split_index, score = best_split(segment)
        if split_index is None or score < cage_length_threshold**2:
            return []
        left = split(offset, segment[:split_index])
        right = split(offset + split_index, segment[split_index:])
        return left + [offset + split_index] + right

    return split(0, coordinates)


def event_clock_row(
    sample: Sample,
    tracks: list[tuple[int, np.ndarray, np.ndarray, dict[int, int]]],
    relative_frames: list[dict[int, np.ndarray]],
    *,
    threshold: float,
    min_track_frames: int,
    scan_kind: str,
) -> dict[str, float | str]:
    """Extract first-event and exchange statistics with explicit right censoring."""

    first_event_times: list[float] = []
    first_exposures: list[float] = []
    exchange_intervals: list[float] = []
    event_track_count = 0
    right_censored_track_count = 0
    for particle_id, times, _, _ in tracks:
        relative_coordinates = np.asarray(
            [relative_frames[int(time)][particle_id] for time in times], dtype=float
        )
        for segment in contiguous_segments(times, relative_coordinates, min_track_frames):
            event_track_count += 1
            jump_indices = candelier_jump_indices(segment, threshold)
            exposure = float(len(segment) - 1)
            if not jump_indices:
                first_exposures.append(exposure)
                right_censored_track_count += 1
                continue
            first_time = float(jump_indices[0])
            first_event_times.append(first_time)
            first_exposures.append(first_time)
            exchange_intervals.extend(
                float(right - left) for left, right in zip(jump_indices, jump_indices[1:])
            )

    if event_track_count == 0:
        raise ValueError("no contiguous tracks meet the selected observation horizon")
    if not first_event_times:
        raise ValueError("no observed first cage escapes at the selected threshold")
    if not exchange_intervals:
        raise ValueError("no exchange intervals at the selected threshold")

    naive_persistence_mean = float(np.mean(first_event_times))
    censored_persistence_mean = float(np.sum(first_exposures) / len(first_event_times))
    exchange_mean = float(np.mean(exchange_intervals))
    return {
        "source_id": "vivek2017_2d_hard_colloid",
        "sample_id": sample.sample_id,
        "area_fraction": sample.area_fraction,
        "time_step_seconds": sample.time_step_seconds,
        "threshold": threshold,
        "min_track_frames": float(min_track_frames),
        "scan_kind": scan_kind,
        "event_track_count": float(event_track_count),
        "observed_first_escape_count": float(len(first_event_times)),
        "right_censored_first_track_count": float(right_censored_track_count),
        "exchange_interval_count": float(len(exchange_intervals)),
        "naive_persistence_mean": naive_persistence_mean,
        "censored_persistence_mean": censored_persistence_mean,
        "exchange_mean": exchange_mean,
        "naive_persistence_exchange_ratio": naive_persistence_mean / exchange_mean,
        "censored_persistence_exchange_ratio": censored_persistence_mean / exchange_mean,
    }


def analyze_event_clock_sample(
    sample: Sample,
    path: Path,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    """Audit whether this finite-FOV data set identifies a PE clock ratio."""

    tracks, frames = load_tracks(path)
    relative_frames = cage_relative_coordinates(frames)
    scan_specification = (
        (0.4, 100, "threshold_scan"),
        (0.6, 100, "threshold_scan"),
        (0.8, 100, "threshold_scan"),
        (0.6, 250, "horizon_scan"),
        (0.6, 500, "horizon_scan"),
    )
    rows = [
        event_clock_row(
            sample,
            tracks,
            relative_frames,
            threshold=threshold,
            min_track_frames=min_track_frames,
            scan_kind=scan_kind,
        )
        for threshold, min_track_frames, scan_kind in scan_specification
    ]
    for row in rows:
        row["source_url"] = sample.url
        row["source_sha256"] = sha256(path)

    verdict = censored_event_clock_identifiability_verdict(
        verdict_id="weeks_hard_colloid_event_clock_censoring_verdict",
        sample_id=sample.sample_id,
        event_clock_rows=rows,
        min_exchange_interval_count=20,
        max_threshold_log_ratio_spread=0.15,
        max_horizon_log_ratio_spread=0.30,
    )
    verdict.update(
        {
            "source_id": "vivek2017_2d_hard_colloid",
            "area_fraction": sample.area_fraction,
            "time_step_seconds": sample.time_step_seconds,
            "source_url": sample.url,
            "source_sha256": sha256(path),
        }
    )
    return rows, verdict


def analyze_sample(sample: Sample, path: Path, wave_number: float = 3.0) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    tracks, frames = load_tracks(path)
    rows: list[dict[str, float | str]] = []
    for lag in lag_grid(len(frames)):
        observable = raw_lag_observables(tracks, int(lag), wave_number)
        rows.append(
            {
                "source_id": "vivek2017_2d_hard_colloid",
                "sample_id": sample.sample_id,
                "area_fraction": sample.area_fraction,
                "time_step_seconds": sample.time_step_seconds,
                "published_tau_alpha_seconds": sample.published_tau_alpha_seconds,
                "source_sha256": sha256(path),
                "wave_number_inverse_microns": wave_number,
                "lag_frames": float(lag),
                "lag_seconds": float(lag * sample.time_step_seconds),
                "lag_over_published_tau_alpha": float(lag * sample.time_step_seconds / sample.published_tau_alpha_seconds),
                **observable,
            }
        )

    peak = max(rows, key=lambda row: float(row["ngp_2d"]))
    late = rows[-1]
    peak_cage = cage_relative_ngp(frames, int(float(peak["lag_frames"])))
    late_cage = cage_relative_ngp(frames, int(float(late["lag_frames"])))
    verdict = weeks_colloid_true_time_verdict(
        verdict_id="weeks_hard_colloid_true_time_verdict",
        source_id="vivek2017_2d_hard_colloid",
        sample_id=sample.sample_id,
        published_tau_alpha=sample.published_tau_alpha_seconds,
        raw_observables={
            "peak_time": float(peak["lag_seconds"]),
            "peak_ngp": float(peak["ngp_2d"]),
            "late_time": float(late["lag_seconds"]),
            "late_ngp": float(late["ngp_2d"]),
            "late_pair_count": float(late["pair_count"]),
        },
        cage_relative_observables={
            "peak_time": float(peak["lag_seconds"]),
            "peak_ngp": float(peak_cage["ngp_2d"]),
            "late_time": float(late["lag_seconds"]),
            "late_ngp": float(late_cage["ngp_2d"]),
            "late_pair_count": float(late_cage["pair_count"]),
        },
        minimum_late_pair_count=1000.0,
        minimum_representation_log_difference=0.3,
    )
    verdict.update(
        {
            "area_fraction": sample.area_fraction,
            "time_step_seconds": sample.time_step_seconds,
            "source_url": sample.url,
            "source_sha256": sha256(path),
            "raw_peak_lag_seconds": float(peak["lag_seconds"]),
            "raw_late_lag_seconds": float(late["lag_seconds"]),
            "cage_relative_peak_pair_count": float(peak_cage["pair_count"]),
            "cage_relative_late_pair_count": float(late_cage["pair_count"]),
            "cage_relative_peak_origin_count": float(peak_cage["origin_count"]),
            "cage_relative_late_origin_count": float(late_cage["origin_count"]),
        }
    )
    return rows, verdict


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        raise ValueError("rows must be nonempty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, rows: list[dict[str, float | str]], verdicts: list[dict[str, float | str]]) -> None:
    width, height = 920, 510
    left, right, top, bottom = 84, 42, 54, 74
    plot_width, plot_height = width - left - right, height - top - bottom
    sample_ids = [str(row["sample_id"]) for row in verdicts]
    colors = {sample_ids[0]: "#1f77b4", sample_ids[-1]: "#d62728"}
    x_values = np.asarray([float(row["lag_over_published_tau_alpha"]) for row in rows])
    y_values = np.asarray([float(row["ngp_2d"]) for row in rows])
    log_x = np.log10(x_values)
    x_min, x_max = float(log_x.min()), float(log_x.max())
    y_min, y_max = 0.0, max(1.0, float(y_values.max()) * 1.08)

    def x_coord(value: float) -> float:
        return left + (math.log10(value) - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124} .axis{stroke:#202124;stroke-width:1.3} .grid{stroke:#d9d9d9;stroke-width:1} .series{fill:none;stroke-width:2.5} .marker{stroke:white;stroke-width:1}</style>',
        f'<text x="{left}" y="29" font-size="18">Public 2D hard-colloid trajectories: true-time NGP evidence</text>',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
    ]
    for tick in (0.1, 1.0, 10.0):
        if x_values.min() <= tick <= x_values.max():
            x = x_coord(tick)
            lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
            lines.append(f'<text x="{x:.2f}" y="{top + plot_height + 23}" text-anchor="middle" font-size="12">{tick:g}</text>')
    for tick in np.linspace(0.0, y_max, 5):
        y = y_coord(float(tick))
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{tick:.1f}</text>')
    for sample_id in sample_ids:
        series = [row for row in rows if str(row["sample_id"]) == sample_id]
        points = " ".join(
            f'{x_coord(float(row["lag_over_published_tau_alpha"])):.2f},{y_coord(float(row["ngp_2d"])):.2f}'
            for row in series
        )
        color = colors[sample_id]
        lines.append(f'<polyline class="series" stroke="{color}" points="{points}"/>')
        for row in series:
            lines.append(
                f'<circle class="marker" fill="{color}" cx="{x_coord(float(row["lag_over_published_tau_alpha"])):.2f}" cy="{y_coord(float(row["ngp_2d"])):.2f}" r="3.2"/>'
            )
    for index, verdict in enumerate(verdicts):
        color = colors[str(verdict["sample_id"])]
        label = f"phi={float(verdict['area_fraction']):.2f}, {str(verdict['sample_id'])}"
        lines.append(f'<line x1="{left + 22}" y1="{top + 20 + 22 * index}" x2="{left + 44}" y2="{top + 20 + 22 * index}" stroke="{color}" stroke-width="2.5"/>')
        lines.append(f'<text x="{left + 52}" y="{top + 24 + 22 * index}" font-size="12">{label}</text>')
    lines.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 18}" text-anchor="middle" font-size="14">lag time / published tau_alpha</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" transform="rotate(-90 20 {top + plot_height / 2:.2f})" text-anchor="middle" font-size="14">two-dimensional non-Gaussian parameter</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_event_clock_svg(
    path: Path,
    rows: list[dict[str, float | str]],
    verdicts: list[dict[str, float | str]],
) -> None:
    """Plot track-horizon sensitivity at the common threshold used by the audit."""

    width, height = 920, 490
    top, bottom = 60, 78
    panel_width, panel_height = 360, height - top - bottom
    panel_lefts = (92, 510)
    y_min, y_max = math.log10(0.7), math.log10(200.0)
    x_min, x_max = 90.0, 510.0
    colors = {"naive": "#1f77b4", "right-censored": "#d62728"}

    def x_coord(value: float, left: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * panel_width

    def y_coord(value: float) -> float:
        return top + (y_max - math.log10(value)) / (y_max - y_min) * panel_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124} .axis{stroke:#202124;stroke-width:1.3} .grid{stroke:#d9d9d9;stroke-width:1} .series{fill:none;stroke-width:2.4} .marker{stroke:white;stroke-width:1}</style>',
        '<text x="92" y="30" font-size="18">Cage-jump event clocks: observation-window identifiability audit</text>',
        '<text x="92" y="48" font-size="12">Candelier-type cage-relative segmentation is threshold-stable, but PE ratios drift strongly with field-of-view track horizon.</text>',
    ]
    for index, verdict in enumerate(verdicts):
        left = panel_lefts[index]
        sample_id = str(verdict["sample_id"])
        lines.extend(
            [
                f'<line class="axis" x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}"/>',
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}"/>',
                f'<text x="{left + panel_width / 2:.2f}" y="{top - 14}" text-anchor="middle" font-size="14">phi={float(verdict["area_fraction"]):.2f}, {sample_id}</text>',
            ]
        )
        for tick in (100.0, 250.0, 500.0):
            x = x_coord(tick, left)
            lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + panel_height}"/>')
            lines.append(f'<text x="{x:.2f}" y="{top + panel_height + 22}" text-anchor="middle" font-size="11">{int(tick)}</text>')
        for tick in (1.0, 3.0, 10.0, 30.0, 100.0):
            y = y_coord(tick)
            lines.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" y2="{y:.2f}"/>')
            lines.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{tick:g}</text>')
        horizon_rows = sorted(
            [
                row for row in rows
                if str(row["sample_id"]) == sample_id and float(row["threshold"]) == float(verdict["selected_horizon_threshold"])
            ],
            key=lambda row: float(row["min_track_frames"]),
        )
        for label, key in (
            ("naive", "naive_persistence_exchange_ratio"),
            ("right-censored", "censored_persistence_exchange_ratio"),
        ):
            points = " ".join(
                f'{x_coord(float(row["min_track_frames"]), left):.2f},{y_coord(float(row[key])):.2f}'
                for row in horizon_rows
            )
            lines.append(f'<polyline class="series" stroke="{colors[label]}" points="{points}"/>')
            for row in horizon_rows:
                lines.append(f'<circle class="marker" fill="{colors[label]}" cx="{x_coord(float(row["min_track_frames"]), left):.2f}" cy="{y_coord(float(row[key])):.2f}" r="3.2"/>')
        lines.append(f'<text x="{left + panel_width / 2:.2f}" y="{height - 22}" text-anchor="middle" font-size="13">minimum contiguous track length (frames)</text>')
        lines.append(f'<text x="{left + 8}" y="{top + panel_height - 12}" font-size="11">threshold stable=1; PE inversion=0</text>')
    lines.extend(
        [
            '<line x1="92" y1="438" x2="112" y2="438" stroke="#1f77b4" stroke-width="2.4"/>',
            '<text x="118" y="442" font-size="12">naive first-event mean / exchange mean</text>',
            '<line x1="350" y1="438" x2="370" y2="438" stroke="#d62728" stroke-width="2.4"/>',
            '<text x="376" y="442" font-size="12">right-censored first-event estimate / exchange mean</text>',
            '<text x="22" y="{:.2f}" transform="rotate(-90 22 {:.2f})" text-anchor="middle" font-size="13">persistence / exchange ratio (log scale)</text>'.format(top + panel_height / 2, top + panel_height / 2),
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="download the fixed public source files when absent")
    parser.add_argument("--cache-dir", type=Path, default=DATA_DIR / "third_party" / "weeks")
    parser.add_argument("--curve-csv", type=Path, default=DATA_DIR / "renewal_cage_weeks_hard_colloid_true_time_curve.csv")
    parser.add_argument("--verdict-csv", type=Path, default=DATA_DIR / "renewal_cage_weeks_hard_colloid_true_time_verdict.csv")
    parser.add_argument("--svg", type=Path, default=FIGURE_DIR / "renewal_cage_weeks_hard_colloid_true_time.svg")
    parser.add_argument("--event-clock-csv", type=Path, default=DATA_DIR / "renewal_cage_weeks_hard_colloid_event_clock_censoring.csv")
    parser.add_argument("--event-clock-verdict-csv", type=Path, default=DATA_DIR / "renewal_cage_weeks_hard_colloid_event_clock_censoring_verdict.csv")
    parser.add_argument("--event-clock-svg", type=Path, default=FIGURE_DIR / "renewal_cage_weeks_hard_colloid_event_clock_censoring.svg")
    args = parser.parse_args()

    curve_rows: list[dict[str, float | str]] = []
    verdict_rows: list[dict[str, float | str]] = []
    event_clock_rows: list[dict[str, float | str]] = []
    event_clock_verdict_rows: list[dict[str, float | str]] = []
    for sample in SAMPLES:
        source = fetch_sample(sample, args.cache_dir, fetch=args.fetch)
        rows, verdict = analyze_sample(sample, source)
        curve_rows.extend(rows)
        verdict_rows.append(verdict)
        event_rows, event_verdict = analyze_event_clock_sample(sample, source)
        event_clock_rows.extend(event_rows)
        event_clock_verdict_rows.append(event_verdict)
    write_csv(args.curve_csv, curve_rows)
    write_csv(args.verdict_csv, verdict_rows)
    write_svg(args.svg, curve_rows, verdict_rows)
    write_csv(args.event_clock_csv, event_clock_rows)
    write_csv(args.event_clock_verdict_csv, event_clock_verdict_rows)
    write_event_clock_svg(args.event_clock_svg, event_clock_rows, event_clock_verdict_rows)


if __name__ == "__main__":
    main()
