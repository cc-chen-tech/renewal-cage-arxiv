#!/usr/bin/env python3
"""Generate reproducible figures for the delayed renewal cage model."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renewal_cage import (  # noqa: E402
    DelayedRenewalCageParams,
    delayed_poisson_mean,
    dimensionless_peak_prediction,
    gaussian_radial_3d,
    moments_1d,
    ngp_1d,
    radial_van_hove_3d,
)


DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"


def scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        return np.full_like(values, (low + high) / 2.0)
    return low + (values - vmin) * (high - low) / (vmax - vmin)


def polyline(x: np.ndarray, y: np.ndarray, color: str, width: float = 2.2) -> str:
    coords = " ".join(f"{xi:.2f},{yi:.2f}" for xi, yi in zip(x, y))
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" />'
    )


def write_main_csv(path: Path, time: np.ndarray, params: DelayedRenewalCageParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    moments = moments_1d(time, params)
    alpha = ngp_1d(time, params)
    with path.open("w", newline="") as f:
        fieldnames = ["time", "local_variance", "renewal_mean", "msd", "m4", "ngp_1d"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, t in enumerate(time):
            writer.writerow(
                {
                    "time": float(t),
                    "local_variance": float(moments["local_variance"][idx]),
                    "renewal_mean": float(moments["renewal_mean"][idx]),
                    "msd": float(moments["m2"][idx]),
                    "m4": float(moments["m4"][idx]),
                    "ngp_1d": float(alpha[idx]),
                }
            )


def peak_summary(time: np.ndarray, alpha: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmax(alpha))
    return float(time[idx]), float(alpha[idx])


def write_sweep_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_dimensionless_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    write_sweep_csv(path, rows)


def write_van_hove_csv(
    path: Path,
    radius: np.ndarray,
    curves: list[tuple[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = ["radius"] + [label for label, _ in curves]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, r in enumerate(radius):
            row: dict[str, float] = {"radius": float(r)}
            for label, density in curves:
                row[label] = float(density[idx])
            writer.writerow(row)


def write_tail_ratio_csv(
    path: Path,
    radius: np.ndarray,
    renewal_curves: list[tuple[str, np.ndarray]],
    gaussian_curves: list[tuple[str, np.ndarray]],
    *,
    tail_radius: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = radius >= tail_radius
    rows = []
    for (label, renewal), (_, gaussian) in zip(renewal_curves, gaussian_curves):
        renewal_tail = float(np.trapezoid(renewal[mask], radius[mask]))
        gaussian_tail = float(np.trapezoid(gaussian[mask], radius[mask]))
        rows.append(
            {
                "time_label": label,
                "tail_radius": tail_radius,
                "renewal_tail_probability": renewal_tail,
                "gaussian_tail_probability": gaussian_tail,
                "tail_ratio": renewal_tail / gaussian_tail,
            }
        )
    write_sweep_csv(path, rows)


def write_svg(
    path: Path,
    time: np.ndarray,
    params: DelayedRenewalCageParams,
    delay_curves: list[tuple[str, np.ndarray]],
    jump_curves: list[tuple[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    moments = moments_1d(time, params)
    alpha = ngp_1d(time, params)
    renewal = delayed_poisson_mean(time, params)

    width, height = 1180, 820
    panels = {
        "a": (75, 90, 525, 350),
        "b": (660, 90, 1110, 350),
        "c": (75, 490, 525, 750),
        "d": (660, 490, 1110, 750),
    }
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#d69e2e"]

    def axes(panel: tuple[int, int, int, int], title: str) -> str:
        left, top, right, bottom = panel
        return f"""
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 32}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">time</text>
"""

    def plot(panel: tuple[int, int, int, int], series: list[tuple[np.ndarray, str]]) -> str:
        left, top, right, bottom = panel
        x = scale(time, left, right)
        y_values = np.concatenate([s[0] for s in series])
        y_scaled = scale(y_values, bottom, top)
        out = []
        start = 0
        for values, color in series:
            segment = y_scaled[start : start + len(values)]
            out.append(polyline(x, segment, color))
            start += len(values)
        return "\n".join(out)

    delay_series = [(curve, colors[idx]) for idx, (_, curve) in enumerate(delay_curves)]
    jump_series = [(curve, colors[idx]) for idx, (_, curve) in enumerate(jump_curves)]

    legend_delay = "\n".join(
        f'<text x="88" y="{520 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{colors[idx]}">{label}</text>'
        for idx, (label, _) in enumerate(delay_curves)
    )
    legend_jump = "\n".join(
        f'<text x="673" y="{520 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{colors[idx]}">{label}</text>'
        for idx, (label, _) in enumerate(jump_curves)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Delayed renewal cage model</text>

  {axes(panels["a"], "A. MSD: cage plateau followed by renewal diffusion")}
  {plot(panels["a"], [(moments["local_variance"], "#718096"), (moments["m2"], "#2b6cb0")])}
  <text x="95" y="125" font-family="Arial, sans-serif" font-size="12" fill="#718096">local cage variance</text>
  <text x="95" y="143" font-family="Arial, sans-serif" font-size="12" fill="#2b6cb0">total MSD</text>

  {axes(panels["b"], "B. NGP peak and long-time decay")}
  {plot(panels["b"], [(alpha, "#c05621"), (1.0 / np.maximum(renewal, 1e-12), "#a0aec0")])}
  <text x="680" y="125" font-family="Arial, sans-serif" font-size="12" fill="#c05621">NGP</text>
  <text x="680" y="143" font-family="Arial, sans-serif" font-size="12" fill="#a0aec0">1 / renewal count asymptote</text>

  {axes(panels["c"], "C. Delay time controls peak position")}
  {plot(panels["c"], delay_series)}
  {legend_delay}

  {axes(panels["d"], "D. Jump variance controls peak height")}
  {plot(panels["d"], jump_series)}
  {legend_jump}
</svg>
"""
    path.write_text(svg)


def write_dimensionless_svg(
    path: Path,
    scaled_time: np.ndarray,
    collapse_curves: list[tuple[str, np.ndarray]],
    radius: np.ndarray,
    van_hove_curves: list[tuple[str, np.ndarray]],
    gaussian_curves: list[tuple[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#d69e2e"]

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 44}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    x_a = scale(scaled_time, left_a, right_a)
    all_ngp = np.concatenate([curve for _, curve in collapse_curves])
    y_all = scale(all_ngp, bottom, top)
    collapse_lines = []
    start = 0
    for idx, (label, curve) in enumerate(collapse_curves):
        segment = y_all[start : start + len(curve)]
        collapse_lines.append(polyline(x_a, segment, colors[idx]))
        collapse_lines.append(
            f'<text x="{left_a + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{colors[idx]}">{label}</text>'
        )
        start += len(curve)

    x_b = scale(radius, left_b, right_b)
    all_density = np.concatenate([curve for _, curve in van_hove_curves] + [curve for _, curve in gaussian_curves])
    y_density = scale(all_density, bottom, top)
    van_hove_lines = []
    start = 0
    for idx, (label, curve) in enumerate(van_hove_curves):
        segment = y_density[start : start + len(curve)]
        van_hove_lines.append(polyline(x_b, segment, colors[idx]))
        van_hove_lines.append(
            f'<text x="{left_b + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{colors[idx]}">{label}</text>'
        )
        start += len(curve)
    for idx, (label, curve) in enumerate(gaussian_curves):
        segment = y_density[start : start + len(curve)]
        van_hove_lines.append(polyline(x_b, segment, "#a0aec0", width=1.4))
        if idx == 0:
            van_hove_lines.append(
                f'<text x="{left_b + 18}" y="{top + 25 + (len(van_hove_curves) + 1) * 18}" font-family="Arial, sans-serif" font-size="12" fill="#718096">matched Gaussian baselines</text>'
            )
        start += len(curve)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Dimensionless renewal-cage predictions</text>
  {axes(left_a, right_a, "E. Peak collapse in plateau regime", "t / t*")}
  {"".join(collapse_lines)}
  {axes(left_b, right_b, "F. 3D radial van Hove distribution", "radius")}
  {"".join(van_hove_lines)}
</svg>
"""
    path.write_text(svg)


def main() -> None:
    params = DelayedRenewalCageParams(
        cage_variance=1.0,
        cage_tau=0.7,
        jump_variance=0.8,
        renewal_rate=0.18,
        renewal_delay=3.0,
    )
    time = np.linspace(1e-4, 180.0, 1800)
    write_main_csv(DATA_DIR / "renewal_cage_main.csv", time, params)

    delay_values = [1.2, 2.0, 3.0, 5.0]
    jump_values = [0.3, 0.6, 0.9, 1.3]

    sweep_rows: list[dict[str, float | str]] = []
    delay_curves: list[tuple[str, np.ndarray]] = []
    for delay in delay_values:
        p = DelayedRenewalCageParams(
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            jump_variance=params.jump_variance,
            renewal_rate=params.renewal_rate,
            renewal_delay=delay,
        )
        alpha = ngp_1d(time, p)
        peak_time, peak_value = peak_summary(time, alpha)
        delay_curves.append((f"delay={delay:g}", alpha))
        sweep_rows.append({"sweep": "delay", "value": delay, "peak_time": peak_time, "peak_ngp": peak_value})

    jump_curves: list[tuple[str, np.ndarray]] = []
    for jump in jump_values:
        p = DelayedRenewalCageParams(
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            jump_variance=jump,
            renewal_rate=params.renewal_rate,
            renewal_delay=params.renewal_delay,
        )
        alpha = ngp_1d(time, p)
        peak_time, peak_value = peak_summary(time, alpha)
        jump_curves.append((f"jump={jump:g}", alpha))
        sweep_rows.append({"sweep": "jump_variance", "value": jump, "peak_time": peak_time, "peak_ngp": peak_value})

    write_sweep_csv(DATA_DIR / "renewal_cage_sweeps.csv", sweep_rows)
    write_svg(FIGURE_DIR / "renewal_cage_results.svg", time, params, delay_curves, jump_curves)

    scaled_time = np.linspace(0.01, 4.0, 700)
    collapse_rows: list[dict[str, float | str]] = []
    collapse_curves: list[tuple[str, np.ndarray]] = []
    for ratio in [0.3, 0.6, 0.9, 1.2]:
        p = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=ratio,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        prediction = dimensionless_peak_prediction(p)
        physical_time = scaled_time * prediction["peak_time"]
        alpha = ngp_1d(physical_time, p)
        scaled_alpha = alpha / prediction["peak_ngp"]
        collapse_curves.append((f"q/A={ratio:g}", scaled_alpha))
        for idx, value in enumerate(scaled_time):
            collapse_rows.append(
                {
                    "q_over_A": ratio,
                    "t_over_t_peak": float(value),
                    "alpha_over_alpha_peak": float(scaled_alpha[idx]),
                    "predicted_t_peak": float(prediction["peak_time"]),
                    "predicted_alpha_peak": float(prediction["peak_ngp"]),
                }
            )
    write_dimensionless_csv(DATA_DIR / "renewal_cage_dimensionless.csv", collapse_rows)

    radius = np.linspace(0.0, 24.0, 1400)
    van_hove_times = [2.0, 11.30637498610339, 80.0]
    van_hove_curves = []
    gaussian_curves = []
    for value in van_hove_times:
        density = radial_van_hove_3d(radius, time=value, params=params, max_count=140)
        van_hove_curves.append((f"t={value:.1f}", density))
        coordinate_variance = moments_1d(np.array([value]), params)["m2"][0]
        gaussian_curves.append((f"gaussian_t={value:.1f}", gaussian_radial_3d(radius, coordinate_variance=coordinate_variance)))
    write_van_hove_csv(DATA_DIR / "renewal_cage_van_hove.csv", radius, van_hove_curves + gaussian_curves)
    write_tail_ratio_csv(
        DATA_DIR / "renewal_cage_tail_ratios.csv",
        radius,
        van_hove_curves,
        gaussian_curves,
        tail_radius=5.0,
    )
    write_dimensionless_svg(
        FIGURE_DIR / "renewal_cage_dimensionless.svg",
        scaled_time,
        collapse_curves,
        radius,
        van_hove_curves,
        gaussian_curves,
    )


if __name__ == "__main__":
    main()
