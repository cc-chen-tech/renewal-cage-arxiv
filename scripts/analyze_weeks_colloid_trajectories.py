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

from renewal_cage import weeks_colloid_true_time_verdict  # noqa: E402


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


def load_tracks(path: Path) -> tuple[list[tuple[np.ndarray, np.ndarray, dict[int, int]]], list[dict[int, np.ndarray]]]:
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

    tracks: list[tuple[np.ndarray, np.ndarray, dict[int, int]]] = []
    for identifier in np.unique(particle_id):
        rows = table[particle_id == identifier]
        times = rows[:, 3].astype(int)
        positions = rows[:, :2]
        tracks.append((times, positions, {int(value): index for index, value in enumerate(times)}))

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
    tracks: list[tuple[np.ndarray, np.ndarray, dict[int, int]]],
    lag: int,
    wave_number: float,
) -> dict[str, float]:
    displacements: list[np.ndarray] = []
    for times, positions, index in tracks:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="download the fixed public source files when absent")
    parser.add_argument("--cache-dir", type=Path, default=DATA_DIR / "third_party" / "weeks")
    parser.add_argument("--curve-csv", type=Path, default=DATA_DIR / "renewal_cage_weeks_hard_colloid_true_time_curve.csv")
    parser.add_argument("--verdict-csv", type=Path, default=DATA_DIR / "renewal_cage_weeks_hard_colloid_true_time_verdict.csv")
    parser.add_argument("--svg", type=Path, default=FIGURE_DIR / "renewal_cage_weeks_hard_colloid_true_time.svg")
    args = parser.parse_args()

    curve_rows: list[dict[str, float | str]] = []
    verdict_rows: list[dict[str, float | str]] = []
    for sample in SAMPLES:
        source = fetch_sample(sample, args.cache_dir, fetch=args.fetch)
        rows, verdict = analyze_sample(sample, source)
        curve_rows.extend(rows)
        verdict_rows.append(verdict)
    write_csv(args.curve_csv, curve_rows)
    write_csv(args.verdict_csv, verdict_rows)
    write_svg(args.svg, curve_rows, verdict_rows)


if __name__ == "__main__":
    main()
