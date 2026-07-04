#!/usr/bin/env python3
"""Build an arXiv-safe source package with PDF figures.

The repository stores SVG figures for easy browser viewing. arXiv source uploads are
more robust when figures are PDF files referenced from LaTeX, so this script
recreates the two manuscript figures directly from CSV outputs using reportlab.
"""

from __future__ import annotations

import csv
import math
import zipfile
from pathlib import Path

import numpy as np
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas

rl_config.invariant = 1

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PAPER_DIR = ROOT / "paper"
PAPER_FIGURE_DIR = PAPER_DIR / "figures"
DIST_DIR = ROOT / "dist"


def read_csv_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    columns: dict[str, list[float]] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, value in row.items():
            columns[key].append(float(value))
    return {key: np.array(value, dtype=float) for key, value in columns.items()}


def scale(values: np.ndarray, low: float, high: float, data_range: tuple[float, float] | None = None) -> np.ndarray:
    if data_range is None:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    else:
        vmin, vmax = data_range
    if math.isclose(vmin, vmax):
        return np.full_like(values, (low + high) / 2.0)
    return low + (values - vmin) * (high - low) / (vmax - vmin)


def format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def tick_values(data_min: float, data_max: float) -> list[float]:
    mid = 0.5 * (data_min + data_max)
    return [data_min, mid, data_max]


def draw_axes(
    c: canvas.Canvas,
    left: float,
    bottom: float,
    width: float,
    height: float,
    title: str,
    xlabel: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.line(left, bottom, left + width, bottom)
    c.line(left, bottom, left, bottom + height)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, bottom + height + 12, title)
    c.setFont("Helvetica", 8)
    c.drawCentredString(left + width / 2.0, bottom - 18, xlabel)
    c.setFont("Helvetica", 6)
    c.setStrokeColor(colors.grey)
    for value in tick_values(*x_range):
        x = float(scale(np.array([value]), left, left + width, x_range)[0])
        c.line(x, bottom, x, bottom - 3)
        c.drawCentredString(x, bottom - 10, format_tick(value))
    for value in tick_values(*y_range):
        y = float(scale(np.array([value]), bottom, bottom + height, y_range)[0])
        c.line(left - 3, y, left, y)
        c.drawRightString(left - 5, y - 2, format_tick(value))
    c.setStrokeColor(colors.black)


def draw_polyline(c: canvas.Canvas, x: np.ndarray, y: np.ndarray, color: colors.Color, line_width: float = 1.2) -> None:
    path = c.beginPath()
    started = False
    for xi, yi in zip(x, y):
        if not np.isfinite(yi):
            started = False
            continue
        if not started:
            path.moveTo(float(xi), float(yi))
            started = True
        else:
            path.lineTo(float(xi), float(yi))
    c.setStrokeColor(color)
    c.setLineWidth(line_width)
    c.drawPath(path)


def draw_panel(
    c: canvas.Canvas,
    left: float,
    bottom: float,
    width: float,
    height: float,
    x_values: np.ndarray,
    series: list[tuple[str, np.ndarray, colors.Color]],
    title: str,
    xlabel: str = "time",
    y_range: tuple[float, float] | None = None,
) -> None:
    x_range = (float(np.nanmin(x_values)), float(np.nanmax(x_values)))
    all_y = np.concatenate([values for _, values, _ in series])
    if y_range is None:
        y_range = (float(np.nanmin(all_y)), float(np.nanmax(all_y)))
    draw_axes(c, left, bottom, width, height, title, xlabel, x_range, y_range)
    x = scale(x_values, left, left + width, x_range)
    y_scaled = scale(all_y, bottom, bottom + height, y_range)
    start = 0
    for label, values, color in series:
        y = y_scaled[start : start + len(values)]
        draw_polyline(c, x, y, color)
        start += len(values)
    c.setFont("Helvetica", 7)
    for idx, (label, _, color) in enumerate(series[:5]):
        c.setFillColor(color)
        c.drawString(left + 8, bottom + height - 13 - idx * 10, label)
    c.setFillColor(colors.black)


def write_results_pdf(path: Path) -> None:
    main = read_csv_columns(DATA_DIR / "renewal_cage_main.csv")
    with (DATA_DIR / "renewal_cage_sweeps.csv").open() as f:
        sweeps = list(csv.DictReader(f))
    time = main["time"]
    delay_curves = []
    jump_curves = []
    for row in sweeps:
        value = float(row["value"])
        if row["sweep"] == "delay":
            # Regenerate from stored main time using the CSV created by the main script
            # is not possible for all sweep curves, so show peak summaries as flat markers.
            delay_curves.append((value, float(row["peak_time"]), float(row["peak_ngp"])))
        else:
            jump_curves.append((value, float(row["peak_time"]), float(row["peak_ngp"])))

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Delayed renewal cage model")

    draw_panel(
        c,
        45,
        315,
        320,
        190,
        time,
        [
            ("local cage variance", main["local_variance"], colors.grey),
            ("total MSD", main["msd"], colors.HexColor("#2b6cb0")),
        ],
        "A. MSD: cage plateau followed by renewal diffusion",
    )
    renewal = np.maximum(main["renewal_mean"], 1e-12)
    ngp_ymax = float(np.nanmax(main["ngp_1d"]) * 1.15)
    inverse_renewal = 1.0 / renewal
    inverse_renewal[inverse_renewal > ngp_ymax] = np.nan
    draw_panel(
        c,
        430,
        315,
        320,
        190,
        time,
        [
            ("NGP", main["ngp_1d"], colors.HexColor("#c05621")),
            ("1 / renewal count", inverse_renewal, colors.lightgrey),
        ],
        "B. NGP peak and long-time decay",
        y_range=(0.0, ngp_ymax),
    )
    # Peak summary panels use point-connected summaries from the sweep CSV.
    delay_arr = np.array(delay_curves)
    jump_arr = np.array(jump_curves)
    draw_panel(
        c,
        45,
        70,
        320,
        175,
        delay_arr[:, 0],
        [("peak time vs delay", delay_arr[:, 1], colors.HexColor("#2f855a"))],
        "C. Delay time shifts peak position",
        xlabel="delay",
    )
    draw_panel(
        c,
        430,
        70,
        320,
        175,
        jump_arr[:, 0],
        [("peak height vs jump variance", jump_arr[:, 2], colors.HexColor("#805ad5"))],
        "D. Jump variance controls peak height",
        xlabel="jump variance",
    )
    c.showPage()
    c.save()


def write_dimensionless_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_dimensionless.csv").open() as f:
        collapse = list(csv.DictReader(f))
    van_hove = read_csv_columns(DATA_DIR / "renewal_cage_van_hove.csv")
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in collapse:
        grouped.setdefault(row["q_over_A"], []).append((float(row["t_over_t_peak"]), float(row["alpha_over_alpha_peak"])))

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Dimensionless renewal-cage predictions")

    colors_list = [colors.HexColor("#2b6cb0"), colors.HexColor("#c05621"), colors.HexColor("#2f855a"), colors.HexColor("#805ad5")]
    collapse_series = []
    for idx, (label, points) in enumerate(sorted(grouped.items())):
        arr = np.array(points)
        collapse_series.append((f"q/A={label}", arr[:, 1], colors_list[idx % len(colors_list)]))
    x_values = np.array(grouped[sorted(grouped.keys())[0]])[:, 0]
    draw_panel(c, 45, 160, 320, 280, x_values, collapse_series, "E. Peak collapse", xlabel="t / t*", y_range=(0.0, 1.05))

    radius = van_hove["radius"]
    van_series = []
    for key, color in [
        ("t=2.0", colors.HexColor("#2b6cb0")),
        ("t=11.3", colors.HexColor("#c05621")),
        ("t=80.0", colors.HexColor("#2f855a")),
        ("gaussian_t=11.3", colors.lightgrey),
    ]:
        van_series.append((key, van_hove[key], color))
    draw_panel(c, 430, 160, 320, 280, radius, van_series, "F. Radial van Hove vs Gaussian", xlabel="radius")

    c.showPage()
    c.save()


def write_scattering_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_scattering.csv").open() as f:
        rows = list(csv.DictReader(f))
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(row["wave_number"], []).append(
            {
                "time": float(row["time"]),
                "self_intermediate_scattering": float(row["self_intermediate_scattering"]),
                "normalized_alpha_decay": float(row["normalized_alpha_decay"]),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Self-intermediate scattering predictions")

    colors_list = [colors.HexColor("#2b6cb0"), colors.HexColor("#c05621"), colors.HexColor("#2f855a")]
    labels = sorted(grouped.keys(), key=float)
    time = np.array([point["time"] for point in grouped[labels[0]]])
    scattering_series = []
    alpha_series = []
    for idx, label in enumerate(labels):
        points = grouped[label]
        color = colors_list[idx % len(colors_list)]
        scattering_series.append(
            (f"k={float(label):g}", np.array([point["self_intermediate_scattering"] for point in points]), color)
        )
        alpha_series.append(
            (f"k={float(label):g}", np.array([point["normalized_alpha_decay"] for point in points]), color)
        )

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        time,
        scattering_series,
        "G. F_s(k,t): cage plateau and alpha relaxation",
        y_range=(0.0, 1.0),
    )
    draw_panel(
        c,
        430,
        160,
        320,
        280,
        time,
        alpha_series,
        "H. Cage-normalized alpha decay",
        y_range=(0.0, 1.0),
    )

    c.showPage()
    c.save()


def write_temperature_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_temperature.csv")
    inverse_shift = 1.0 / data["temperature"] - 1.0 / data["temperature"][0]
    diffusion = data["diffusion_coefficient"]
    tau_alpha = data["tau_alpha"]
    peak_time = data["predicted_ngp_peak_time"]
    se_product = data["normalized_stokes_einstein_product"]
    lambda_tau = data["lambda_tau_delay"]
    peak_height = data["predicted_ngp_peak"]

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Temperature-dependent renewal diagnostics")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        inverse_shift,
        [
            ("D / D_hot", diffusion / diffusion[0], colors.HexColor("#2b6cb0")),
            ("tau_alpha / tau_hot", tau_alpha / tau_alpha[0], colors.HexColor("#c05621")),
            ("t_NGP / t_NGP,hot", peak_time / peak_time[0], colors.HexColor("#2f855a")),
        ],
        "I. Transport and relaxation decouple on cooling",
        xlabel="inverse-temperature shift",
    )
    draw_panel(
        c,
        430,
        160,
        320,
        280,
        inverse_shift,
        [
            ("D tau_alpha / hot", se_product, colors.HexColor("#805ad5")),
            ("lambda tau_d / hot", lambda_tau / lambda_tau[0], colors.HexColor("#2f855a")),
            ("alpha_peak / hot", peak_height / peak_height[0], colors.HexColor("#c05621")),
        ],
        "J. Stokes-Einstein product and delayed-renewal control",
        xlabel="inverse-temperature shift",
    )

    c.showPage()
    c.save()


def write_barrier_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_susceptibility.csv").open() as f:
        susceptibility_rows = list(csv.DictReader(f))
    with (DATA_DIR / "renewal_cage_barrier.csv").open() as f:
        rows = list(csv.DictReader(f))

    grouped_susceptibility: dict[str, list[tuple[float, float]]] = {}
    for row in susceptibility_rows:
        grouped_susceptibility.setdefault(row["wave_number"], []).append(
            (float(row["time"]), float(row["renewal_scattering_susceptibility"]))
        )
    labels = sorted(grouped_susceptibility.keys(), key=float)
    time = np.array([point[0] for point in grouped_susceptibility[labels[0]]])
    colors_list = [colors.HexColor("#2b6cb0"), colors.HexColor("#c05621"), colors.HexColor("#2f855a")]
    susceptibility_series = []
    for idx, label in enumerate(labels):
        susceptibility_series.append(
            (
                f"k={float(label):g}",
                np.array([point[1] for point in grouped_susceptibility[label]]),
                colors_list[idx % len(colors_list)],
            )
        )

    final_rows = []
    for gap in sorted({float(row["barrier_gap"]) for row in rows}):
        gap_rows = [row for row in rows if float(row["barrier_gap"]) == gap]
        cold = min(gap_rows, key=lambda row: float(row["temperature"]))
        final_rows.append(cold)
    gaps = np.array([float(row["barrier_gap"]) for row in final_rows])
    cold_product = np.array([float(row["normalized_stokes_einstein_product"]) for row in final_rows])
    cold_lambda_tau = np.array([float(row["lambda_tau_delay"]) for row in final_rows])

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Activated barrier and renewal susceptibility diagnostics")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        time,
        susceptibility_series,
        "K. Renewal-count scattering susceptibility",
        y_range=(0.0, float(max(np.nanmax(values) for _, values, _ in susceptibility_series) * 1.15)),
    )
    draw_panel(
        c,
        430,
        160,
        320,
        280,
        gaps,
        [
            ("cold D tau_alpha / hot", cold_product, colors.HexColor("#805ad5")),
            ("cold lambda tau_d / first gap", cold_lambda_tau / cold_lambda_tau[0], colors.HexColor("#2f855a")),
        ],
        "L. Barrier gap amplifies SE violation",
        xlabel="E_d - E_lambda",
    )

    c.showPage()
    c.save()


def build_arxiv_package(output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = DIST_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results_pdf = PAPER_FIGURE_DIR / "renewal_cage_results.pdf"
    dimensionless_pdf = PAPER_FIGURE_DIR / "renewal_cage_dimensionless.pdf"
    scattering_pdf = PAPER_FIGURE_DIR / "renewal_cage_scattering.pdf"
    temperature_pdf = PAPER_FIGURE_DIR / "renewal_cage_temperature.pdf"
    barrier_pdf = PAPER_FIGURE_DIR / "renewal_cage_barrier.pdf"
    write_results_pdf(results_pdf)
    write_dimensionless_pdf(dimensionless_pdf)
    write_scattering_pdf(scattering_pdf)
    write_temperature_pdf(temperature_pdf)
    write_barrier_pdf(barrier_pdf)

    zip_path = output_dir / "renewal-cage-arxiv-source.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(PAPER_DIR / "main.tex", "main.tex")
        archive.write(PAPER_DIR / "references.bib", "references.bib")
        archive.write(results_pdf, "figures/renewal_cage_results.pdf")
        archive.write(dimensionless_pdf, "figures/renewal_cage_dimensionless.pdf")
        archive.write(scattering_pdf, "figures/renewal_cage_scattering.pdf")
        archive.write(temperature_pdf, "figures/renewal_cage_temperature.pdf")
        archive.write(barrier_pdf, "figures/renewal_cage_barrier.pdf")
    return zip_path


if __name__ == "__main__":
    path = build_arxiv_package()
    print(path)
