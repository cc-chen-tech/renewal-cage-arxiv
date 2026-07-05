#!/usr/bin/env python3
"""Build an arXiv-safe source package with PDF figures.

The repository stores SVG figures for easy browser viewing. arXiv source uploads are
more robust when figures are PDF files referenced from LaTeX, so this script
recreates manuscript figures directly from CSV outputs using reportlab.
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
    fractional_exponent = data["fractional_stokes_einstein_exponent"]
    activation_energy = data["apparent_alpha_activation_energy"]

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
            ("xi_SE", fractional_exponent, colors.HexColor("#2b6cb0")),
            ("E_app / hot", activation_energy / activation_energy[0], colors.HexColor("#d69e2e")),
        ],
        "J. Stokes-Einstein product and delayed-renewal control",
        xlabel="inverse-temperature shift",
    )

    c.showPage()
    c.save()


def write_alpha_shape_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_alpha_shape.csv").open() as f:
        rows = list(csv.DictReader(f))
    labels = list(dict.fromkeys(row["temperature_label"] for row in rows))
    grouped = {label: [row for row in rows if row["temperature_label"] == label] for label in labels}
    scaled_time = np.array([float(row["scaled_time"]) for row in grouped[labels[0]]])
    shape_series = []
    colors_list = [colors.HexColor("#2b6cb0"), colors.HexColor("#c05621"), colors.HexColor("#2f855a")]
    for idx, label in enumerate(labels):
        shape_series.append(
            (
                label,
                np.array([float(row["alpha_shape"]) for row in grouped[label]]),
                colors_list[idx % len(colors_list)],
            )
        )
    summary = [grouped[label][0] for label in labels]
    inverse_shift = np.array([float(row["inverse_temperature_shift"]) for row in summary])
    rms = np.array([float(row["rms_log_shape_residual"]) for row in summary])
    control = np.array([float(row["shape_control"]) for row in summary])
    tau_over_delay = np.array([float(row["tau_alpha_over_delay"]) for row in summary])

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Alpha-shape time-temperature superposition diagnostic")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        scaled_time,
        shape_series,
        "U. Alpha shape after tau_alpha scaling",
        xlabel="t / tau_alpha",
    )
    draw_panel(
        c,
        430,
        160,
        320,
        280,
        inverse_shift,
        [
            ("RMS log-shape residual", rms, colors.HexColor("#805ad5")),
            ("Gamma lambda tau_d / hot", control / control[0], colors.HexColor("#2f855a")),
            ("tau_alpha/tau_d / hot", tau_over_delay / tau_over_delay[0], colors.HexColor("#c05621")),
        ],
        "V. Collapse residual and control variable",
        xlabel="inverse-temperature shift",
    )

    c.showPage()
    c.save()


def write_facilitated_exchange_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_facilitated_exchange.csv")
    inverse_shift = data["inverse_temperature_shift"]
    ratio = data["heterogeneity_ratio"]
    amplitude = data["late_ngp_renewal_amplitude"]
    shape = data["shape"]
    exchange_count = data["exchange_renewal_count"]
    renormalization = data["alpha_rate_renormalization"]
    alpha_slope = data["late_alpha_decay_per_renewal"]

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Facilitated finite-exchange heterogeneity law")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        inverse_shift,
        [
            ("c=R_x/kappa0", ratio, colors.HexColor("#2b6cb0")),
            ("R alpha_2 late", amplitude, colors.HexColor("#c05621")),
            ("R_x / hot", exchange_count / exchange_count[0], colors.HexColor("#2f855a")),
            ("kappa0 / hot", shape / shape[0], colors.HexColor("#805ad5")),
        ],
        "W. Cooling grows exchange heterogeneity",
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
            ("alpha-rate renormalization", renormalization, colors.HexColor("#805ad5")),
            ("late alpha slope / hot", alpha_slope / alpha_slope[0], colors.HexColor("#2b6cb0")),
        ],
        "X. Alpha decay per renewal slows",
        xlabel="inverse-temperature shift",
    )

    c.showPage()
    c.save()


def write_persistence_exchange_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_persistence_exchange.csv").open() as f:
        rows = list(csv.DictReader(f))
    summary = [row for row in rows if row["record_type"] == "summary"]
    curves = [row for row in rows if row["record_type"] == "curve"]

    ratios = np.array([float(row["persistence_exchange_ratio"]) for row in summary])
    se_product = np.array([float(row["stokes_einstein_product"]) for row in summary])
    late_ngp = np.array([float(row["late_ngp"]) for row in summary])

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Persistence/exchange renewal diagnostic")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "First cage persistence slows alpha relaxation while exchange fixes long-time diffusion.")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        ratios,
        [
            ("D tau_alpha / Poisson", se_product / se_product[0], colors.HexColor("#805ad5")),
            ("late NGP / Poisson", late_ngp / late_ngp[0], colors.grey),
        ],
        "Y. SE decoupling at fixed diffusion",
        xlabel="persistence / exchange mean",
    )

    labels = sorted({float(row["persistence_exchange_ratio"]) for row in curves})
    series = []
    colors_list = [colors.HexColor("#2b6cb0"), colors.HexColor("#c05621")]
    first_time = None
    for idx, label in enumerate(labels):
        label_rows = [row for row in curves if float(row["persistence_exchange_ratio"]) == label]
        time = np.array([float(row["time"]) for row in label_rows])
        if first_time is None:
            first_time = np.log10(time)
        series.append(
            (
                f"tau_p/tau_x={label:g}",
                np.array([float(row["ngp"]) for row in label_rows]),
                colors_list[idx % len(colors_list)],
            )
        )
    draw_panel(
        c,
        430,
        160,
        320,
        280,
        first_time,
        series,
        "Z. NGP recovery after persistence delay",
        xlabel="log10 time",
    )
    c.showPage()
    c.save()


def write_persistence_exchange_protocol_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_persistence_exchange_protocol.csv").open() as f:
        rows = list(csv.DictReader(f))

    labels = [row["scenario"].replace("_", " ") for row in rows]
    valid = np.array([float(row["valid_alpha_transport"]) for row in rows])
    passes = np.array([float(row["passes_late_ngp"]) for row in rows])
    true_ratio = float(rows[0]["true_persistence_exchange_ratio"])
    inferred_ratio = np.array(
        [
            float(row["inferred_persistence_exchange_ratio"]) if float(row["valid_alpha_transport"]) > 0.5 else np.nan
            for row in rows
        ]
    )
    residual = np.array(
        [
            abs(float(row["late_ngp_log_residual"])) if float(row["valid_alpha_transport"]) > 0.5 else np.nan
            for row in rows
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Persistence/exchange inversion protocol")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "D and alpha time infer hidden clocks; late NGP is a held-out falsification observable.")

    def draw_protocol_panel(left: float, bottom: float, width: float, height: float, title: str) -> None:
        c.setStrokeColor(colors.black)
        c.line(left, bottom, left + width, bottom)
        c.line(left, bottom, left, bottom + height)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(left, bottom + height + 13, title)

    left_a, left_b, bottom, width, height = 45.0, 430.0, 150.0, 320.0, 280.0
    draw_protocol_panel(left_a, bottom, width, height, "AA. Hidden-clock inversion")
    draw_protocol_panel(left_b, bottom, width, height, "AB. Held-out late NGP residual")

    finite_ratio = inferred_ratio[np.isfinite(inferred_ratio)]
    ratio_min = min(float(np.min(finite_ratio)), true_ratio)
    ratio_max = max(float(np.max(finite_ratio)), true_ratio)
    if math.isclose(ratio_min, ratio_max):
        ratio_min -= 1.0
        ratio_max += 1.0
    finite_residual = residual[np.isfinite(residual)]
    residual_max = max(float(np.max(finite_residual)), 0.1)

    def y_ratio(value: float) -> float:
        return bottom + (value - ratio_min) / (ratio_max - ratio_min) * height

    def y_residual(value: float) -> float:
        return bottom + value / residual_max * height

    c.setStrokeColor(colors.grey)
    c.setDash(4, 3)
    c.line(left_a, y_ratio(true_ratio), left_a + width, y_ratio(true_ratio))
    c.line(left_b, y_residual(0.1), left_b + width, y_residual(0.1))
    c.setDash()
    c.setFont("Helvetica", 7)
    c.drawString(left_a + 8, y_ratio(true_ratio) + 4, "true ratio")
    c.drawString(left_b + 8, y_residual(0.1) + 4, "|log residual|=0.1")

    x_positions_a = np.linspace(left_a + 55, left_a + width - 55, len(rows))
    x_positions_b = np.linspace(left_b + 55, left_b + width - 55, len(rows))
    for idx, label in enumerate(labels):
        color = colors.HexColor("#2f855a") if passes[idx] > 0.5 else colors.HexColor("#c05621")
        c.setFillColor(color)
        if valid[idx] > 0.5:
            c.circle(float(x_positions_a[idx]), y_ratio(float(inferred_ratio[idx])), 4.5, fill=1, stroke=0)
            c.circle(float(x_positions_b[idx]), y_residual(float(residual[idx])), 4.5, fill=1, stroke=0)
        else:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(float(x_positions_a[idx] - 4), bottom - 42, "x")
            c.drawString(float(x_positions_b[idx] - 4), bottom - 42, "x")
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(float(x_positions_a[idx]), bottom - 24, label)
        c.drawCentredString(float(x_positions_b[idx]), bottom - 24, label)

    c.showPage()
    c.save()


def write_persistence_exchange_joint_protocol_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_persistence_exchange_joint_protocol.csv").open() as f:
        rows = list(csv.DictReader(f))
    summary = [row for row in rows if row["record_type"] == "summary"]
    multik = [row for row in rows if row["record_type"] == "multik_alpha"]
    scenarios = [row["scenario"].replace("_", " ") for row in summary]

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Joint persistence/exchange inversion protocol")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "Anchor alpha time plus D infer clocks; multi-k alpha, late NGP, and chi4 proxy are held out.")

    left_a, left_b, bottom, width, height = 45.0, 430.0, 150.0, 320.0, 280.0
    for left, title in [(left_a, "AC. Joint pass/fail checks"), (left_b, "AD. Held-out multi-k alpha residual")]:
        c.setStrokeColor(colors.black)
        c.line(left, bottom, left + width, bottom)
        c.line(left, bottom, left, bottom + height)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(left, bottom + height + 13, title)

    checks = [
        ("multi-k", "multik_tau_alpha_consistent", colors.HexColor("#2b6cb0")),
        ("late NGP", "late_ngp_consistent", colors.HexColor("#2f855a")),
        ("chi4", "chi4_proxy_growth_consistent", colors.HexColor("#805ad5")),
    ]
    x_positions_a = np.linspace(left_a + 70, left_a + width - 70, len(summary))
    for idx, row in enumerate(summary):
        for check_idx, (label, key, good_color) in enumerate(checks):
            y = bottom + height - 62 - check_idx * 62
            color = good_color if float(row[key]) > 0.5 else colors.HexColor("#c05621")
            c.setFillColor(color)
            c.circle(float(x_positions_a[idx]), y, 6.5, fill=1, stroke=0)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 6.5)
            c.drawCentredString(float(x_positions_a[idx]), y - 17, label)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(float(x_positions_a[idx]), bottom - 24, scenarios[idx])

    finite_residual = np.array([abs(float(row["tau_alpha_log_residual"])) for row in multik])
    residual_max = max(float(np.max(finite_residual)), 0.03)

    def y_residual(value: float) -> float:
        return bottom + value / residual_max * height

    c.setStrokeColor(colors.grey)
    c.setDash(4, 3)
    c.line(left_b, y_residual(0.02), left_b + width, y_residual(0.02))
    c.setDash()
    c.setFont("Helvetica", 7)
    c.drawString(left_b + 8, y_residual(0.02) + 4, "|log residual|=0.02")

    k_values = sorted({float(row["wave_number"]) for row in multik})
    offsets = np.linspace(-20, 20, len(k_values))
    x_positions_b = np.linspace(left_b + 80, left_b + width - 80, len(summary))
    for idx, scenario in enumerate([row["scenario"] for row in summary]):
        for k_idx, wave_number in enumerate(k_values):
            row = next(
                row
                for row in multik
                if row["scenario"] == scenario and math.isclose(float(row["wave_number"]), wave_number)
            )
            residual = abs(float(row["tau_alpha_log_residual"]))
            c.setFillColor(colors.HexColor("#2b6cb0") if residual <= 0.02 else colors.HexColor("#c05621"))
            c.circle(float(x_positions_b[idx] + offsets[k_idx]), y_residual(residual), 4.5, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(float(x_positions_b[idx]), bottom - 24, scenario.replace("_", " "))

    consistent = summary[0]
    c.setFont("Helvetica", 8)
    c.drawString(
        45,
        92,
        "consistent: "
        f"tau_p/tau_x={float(consistent['inferred_persistence_exchange_ratio']):.1f}; "
        f"SE growth={float(consistent['stokes_einstein_growth_over_poisson']):.2f}; "
        f"chi4 growth={float(consistent['chi4_peak_growth_over_poisson']):.2f}",
    )
    c.showPage()
    c.save()


def write_persistence_exchange_uncertainty_protocol_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_persistence_exchange_uncertainty_protocol.csv").open() as f:
        rows = list(csv.DictReader(f))
    scenarios = [row["scenario"].replace("_", " ") for row in rows]
    z_threshold = float(rows[0]["z_threshold"])
    z_values = np.array(
        [
            [float(row["max_multik_tau_alpha_z"]), float(row["late_ngp_z"]), float(row["chi4_peak_z"])]
            for row in rows
        ]
    )
    z_max = max(float(np.max(z_values)), z_threshold) * 1.12

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Uncertainty-weighted data protocol")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "Observed alpha times, late NGP, and chi4 peak are scored by log-residual z scores.")

    left, bottom, width, height = 65.0, 140.0, 680.0, 300.0
    c.setStrokeColor(colors.black)
    c.line(left, bottom, left + width, bottom)
    c.line(left, bottom, left, bottom + height)

    def y(value: float) -> float:
        return bottom + value / z_max * height

    c.setStrokeColor(colors.grey)
    c.setDash(4, 3)
    c.line(left, y(z_threshold), left + width, y(z_threshold))
    c.setDash()
    c.setFont("Helvetica", 7)
    c.drawString(left + width - 46, y(z_threshold) + 4, f"z={z_threshold:g}")

    x_groups = np.linspace(left + 170, left + width - 170, len(rows))
    offsets = [-20, 0, 20]
    labels = ["multi-k", "late NGP", "chi4"]
    color_list = [colors.HexColor("#2b6cb0"), colors.HexColor("#2f855a"), colors.HexColor("#c05621")]
    for idx, scenario in enumerate(scenarios):
        for jdx, label in enumerate(labels):
            value = z_values[idx, jdx]
            x = float(x_groups[idx] + offsets[jdx])
            c.setFillColor(color_list[jdx])
            c.rect(x - 7, y(value), 14, bottom - y(value), fill=1, stroke=0)
            c.setFont("Helvetica", 6)
            c.drawCentredString(x, bottom - 42 - 11 * jdx, label)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7)
        c.drawCentredString(float(x_groups[idx]), bottom - 24, scenario)

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 8)
    consistent = rows[0]
    mismatch = rows[1]
    c.drawString(
        65,
        82,
        "consistent: "
        f"max z={max(float(consistent['max_multik_tau_alpha_z']), float(consistent['late_ngp_z']), float(consistent['chi4_peak_z'])):.2g}; "
        "chi4 mismatch: "
        f"z={float(mismatch['chi4_peak_z']):.2f}",
    )
    c.showPage()
    c.save()


def write_glass_audit_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_glass_audit.csv").open() as f:
        rows = list(csv.DictReader(f))
    temperature_rows = [row for row in rows if row["record_type"] == "temperature_row"]
    summary_rows = [row for row in rows if row["record_type"] == "summary_flag"]

    inverse_shift = np.array([float(row["inverse_temperature_shift"]) for row in temperature_rows])
    tau_alpha = np.array([float(row["tau_alpha_exchange"]) for row in temperature_rows])
    diffusion = np.array([float(row["diffusion_coefficient"]) for row in temperature_rows])
    se_product = np.array([float(row["normalized_stokes_einstein_product"]) for row in temperature_rows])
    heterogeneity = np.array([float(row["heterogeneity_ratio"]) for row in temperature_rows])
    chi4 = np.array([float(row["chi4_peak"]) for row in temperature_rows])
    beta = np.array([float(row["median_alpha_window_beta"]) for row in temperature_rows])

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Glass-dynamics phenomenon audit")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "Dynamical signatures are checked from one delayed-renewal law; no thermodynamic transition is claimed.")

    draw_panel(
        c,
        45,
        300,
        320,
        175,
        inverse_shift,
        [
            ("tau_alpha exchange / hot", tau_alpha / tau_alpha[0], colors.HexColor("#2b6cb0")),
            ("1 / diffusion / hot", diffusion[0] / diffusion, colors.HexColor("#c05621")),
            ("D tau_alpha / hot", se_product, colors.HexColor("#805ad5")),
        ],
        "A. Transport and alpha slowdown",
        xlabel="inverse-temperature shift",
    )
    draw_panel(
        c,
        430,
        300,
        320,
        175,
        inverse_shift,
        [
            ("exchange ratio / hot", heterogeneity / heterogeneity[0], colors.HexColor("#2f855a")),
            ("chi4 peak / hot", chi4 / chi4[0], colors.HexColor("#d69e2e")),
            ("alpha-window beta", beta, colors.HexColor("#805ad5")),
        ],
        "B. Dynamic heterogeneity signatures",
        xlabel="inverse-temperature shift",
    )

    flags = [(row["signature"], float(row["flag_value"])) for row in summary_rows]
    chart_left, chart_bottom, chart_width, chart_height = 45, 72, 705, 145
    c.setFont("Helvetica-Bold", 10)
    c.drawString(chart_left, chart_bottom + chart_height + 16, "C. Supported signatures")
    c.line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom)
    c.line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height)
    bar_gap = 4
    bar_width = chart_width / len(flags) - bar_gap
    for idx, (label, value) in enumerate(flags):
        x = chart_left + idx * (bar_width + bar_gap)
        h = value * (chart_height - 20)
        color = colors.HexColor("#2f855a") if value >= 1.0 else colors.lightgrey
        c.setFillColor(color)
        c.rect(x, chart_bottom, bar_width, h, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 5.5)
        c.saveState()
        c.translate(x + 3, chart_bottom - 4)
        c.rotate(-45)
        c.drawRightString(0, 0, label.replace("_", " "))
        c.restoreState()
    c.showPage()
    c.save()


def write_glass_phase_diagram_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_glass_phase_diagram.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Barrier-facilitation glass signature phase diagram")

    gaps = sorted(set(data["delay_barrier_gap"]))
    exchange_sums = sorted(set(data["exchange_barrier_sum"]))
    left, bottom, width, height = 65, 165, 310, 230
    cell_w = width / len(exchange_sums)
    cell_h = height / len(gaps)
    scores = data["supported_dynamic_signatures"]
    score_min = float(np.nanmin(scores))
    score_max = float(np.nanmax(scores))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, bottom + height + 18, "A. Supported dynamic signatures")
    for row_idx, gap in enumerate(gaps):
        y = bottom + row_idx * cell_h
        c.setFont("Helvetica", 7)
        c.drawRightString(left - 8, y + cell_h / 2.0, f"{gap:g}")
        for col_idx, exchange_sum in enumerate(exchange_sums):
            mask = (data["delay_barrier_gap"] == gap) & (data["exchange_barrier_sum"] == exchange_sum)
            value = float(scores[mask][0])
            shade = 0.25 + 0.65 * (value - score_min) / (score_max - score_min)
            c.setFillColor(colors.Color(0.95 - 0.25 * shade, 0.86 - 0.10 * shade, 0.58))
            x = left + col_idx * cell_w
            c.rect(x, y, cell_w - 3, cell_h - 3, fill=1, stroke=0)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 8)
            c.drawString(x + 8, y + cell_h / 2.0, f"{value:.0f}/9")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 8)
    for col_idx, exchange_sum in enumerate(exchange_sums):
        c.drawCentredString(left + col_idx * cell_w + cell_w / 2.0, bottom - 14, f"{exchange_sum:g}")
    c.drawCentredString(left + width / 2.0, bottom - 30, "exchange barrier sum")
    c.saveState()
    c.translate(left - 40, bottom + height / 2.0)
    c.rotate(90)
    c.drawCentredString(0, 0, "delay barrier gap")
    c.restoreState()

    x = np.array([gap + 0.08 * exchange_sum for gap, exchange_sum in zip(data["delay_barrier_gap"], data["exchange_barrier_sum"])])
    draw_panel(
        c,
        450,
        165,
        285,
        230,
        x,
        [
            ("cold D tau_alpha / hot", data["cold_se_product_ratio"], colors.HexColor("#805ad5")),
            ("cold heterogeneity / hot", data["cold_heterogeneity_growth_ratio"], colors.HexColor("#2f855a")),
            ("closure flag", data["complete_dynamic_closure"], colors.HexColor("#c05621")),
        ],
        "B. Cold-end diagnostic growth",
        xlabel="delay gap plus exchange offset",
    )
    c.showPage()
    c.save()


def write_spatial_chi4_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_spatial_chi4.csv")
    inverse_shift = 1.0 / data["temperature"] - 1.0 / data["temperature"][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Spatial facilitation chi4 closure")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "A diffusive facilitation front turns the persistence clock into xi4 and Ncorr.")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        inverse_shift,
        [
            ("xi4 / hot", data["length_growth"], colors.HexColor("#2b6cb0")),
            ("Ncorr / hot", data["correlation_size_growth"], colors.HexColor("#2f855a")),
            ("chi4 peak / hot", data["chi4_peak_growth"], colors.HexColor("#c05621")),
        ],
        "AC. Clock-derived dynamic length",
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
            ("chi4 peak time / hot", data["chi4_peak_time"] / data["chi4_peak_time"][0], colors.HexColor("#805ad5")),
            ("tau alpha / hot", data["tau_alpha"] / data["tau_alpha"][0], colors.HexColor("#d69e2e")),
        ],
        "AD. Timing of spatial susceptibility",
        xlabel="inverse-temperature shift",
    )
    c.showPage()
    c.save()


def write_thermodynamic_closure_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_thermodynamic_closure.csv")
    inverse_shift = 1.0 / data["temperature"] - 1.0 / data["temperature"][0]
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Thermodynamic entropy closure")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "Kauzmann entropy extrapolation drives Adam-Gibbs renewal slowdown.")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        inverse_shift,
        [
            ("s_c / hot", data["configurational_entropy"] / data["configurational_entropy"][0], colors.HexColor("#2b6cb0")),
            ("Delta c_p / hot", data["excess_heat_capacity"] / data["excess_heat_capacity"][0], colors.HexColor("#2f855a")),
        ],
        "AE. Configurational entropy sector",
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
            ("tau_AG / hot", data["thermodynamic_slowdown"], colors.HexColor("#c05621")),
            ("tau_alpha / hot", data["tau_alpha_growth"], colors.HexColor("#805ad5")),
        ],
        "AF. Adam-Gibbs kinetic coupling",
        xlabel="inverse-temperature shift",
    )
    c.showPage()
    c.save()


def write_mct_beta_closure_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_mct_beta_closure.csv")
    inverse_shift = data["inverse_temperature_shift"]
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "MCT beta-window closure")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "Effective critical decay and von-Schweidler departure around the cage plateau.")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        inverse_shift,
        [
            ("plateau f_c", data["plateau"] / data["plateau"][0], colors.HexColor("#2b6cb0")),
            ("lambda_a", data["lambda_from_a"] / data["lambda_from_a"][0], colors.HexColor("#2f855a")),
            ("lambda_b", data["lambda_from_b"] / data["lambda_from_b"][0], colors.HexColor("#c05621")),
        ],
        "AG. Beta exponent diagnostics",
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
            ("t_beta / hot", data["beta_time"] / data["beta_time"][0], colors.HexColor("#2b6cb0")),
            ("von exit / hot", data["von_schweidler_exit_time"] / data["von_schweidler_exit_time"][0], colors.HexColor("#c05621")),
            ("tau alpha / hot", data["alpha_time"] / data["alpha_time"][0], colors.HexColor("#805ad5")),
        ],
        "AH. Beta-to-alpha clock separation",
        xlabel="inverse-temperature shift",
    )
    c.showPage()
    c.save()


def write_sota_benchmark_consistency_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_benchmark_consistency.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_id = {row["benchmark_id"]: row for row in rows}
    cage_row = by_id["debye_waller_cage_localization"]
    mct_row = by_id["kob_andersen_1995_beta_window"]
    mct_exponent_row = by_id["kob_andersen_1995_mct_exponent_parameter"]
    recovery_row = by_id["gaussian_recovery_finite_exchange_vs_static_disorder"]
    ngp_peak_row = by_id["ngp_peak_shift_on_cooling"]
    se_row = by_id["stokes_einstein_fractional_decoupling"]
    heterogeneity_row = by_id["dynamic_heterogeneity_chi4_growth"]
    spatial_front_row = by_id["spatial_facilitation_constant_front_law"]
    tts_row = by_id["alpha_tts_breakdown_shape_residual"]
    stretched_row = by_id["kww_alpha_stretching_on_cooling"]
    persistence_exchange_row = by_id["persistence_exchange_transport_inversion"]
    joint_row = by_id["joint_persistence_exchange_multik_chi4_protocol"]
    van_hove_row = by_id["kob_andersen_van_hove_tail_recovery"]
    fragility_row = by_id["angell_adam_gibbs_fragility_growth"]
    thermodynamic_scope_row = by_id["thermodynamic_transition_scope_boundary"]
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA benchmark consistency")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "Literature-level conclusions encoded as explicit model consistency diagnostics.")

    observed_model = np.array(
        [
            float(mct_row["observed_critical_decay"]),
            float(mct_row["model_predicts_visible_critical_decay"]),
            float(mct_row["observed_von_schweidler"]),
            float(mct_row["model_predicts_visible_von_schweidler"]),
        ]
    )
    draw_panel(
        c,
        45,
        160,
        320,
        280,
        np.arange(len(observed_model), dtype=float),
        [
            ("observed/model flags", observed_model, colors.HexColor("#2b6cb0")),
        ],
        "AI. Cage and MCT checks",
        xlabel="critical obs/model, von obs/model",
        y_range=(0.0, 1.05),
    )
    recovery_values = np.array(
        [
            float(recovery_row["finite_exchange_late_ngp"]),
            float(recovery_row["static_gamma_late_ngp"]),
            float(recovery_row["recovery_threshold"]),
        ]
    )
    draw_panel(
        c,
        430,
        160,
        320,
        280,
        np.arange(len(recovery_values), dtype=float),
        [
            ("late NGP / threshold", recovery_values, colors.HexColor("#c05621")),
        ],
        "AJ. Gaussian recovery mechanism",
        xlabel="finite exchange, static, threshold",
    )
    c.setFont("Helvetica", 9)
    c.drawString(
        45,
        118,
        "cage row consistent = "
        f"{int(float(cage_row['overall_consistent']))}; "
        f"f_DW = {float(cage_row['debye_waller_plateau']):.3f}; "
        f"renewal frac = {float(cage_row['renewal_msd_fraction']):.3f}",
    )
    c.drawString(45, 104, f"MCT row consistent = {int(float(mct_row['overall_consistent']))}")
    c.drawString(
        45,
        90,
        "exponent row consistent = "
        f"{int(float(mct_exponent_row['overall_consistent']))}; "
        f"lambda_a = {float(mct_exponent_row['lambda_from_a']):.3f}; "
        f"lambda_b = {float(mct_exponent_row['lambda_from_b']):.3f}",
    )
    c.drawString(
        45,
        76,
        "NGP peak row consistent = "
        f"{int(float(ngp_peak_row['overall_consistent']))}; "
        f"t_peak growth = {float(ngp_peak_row['peak_time_growth']):.2f}; "
        f"peak growth = {float(ngp_peak_row['peak_height_growth']):.2f}",
    )
    c.drawString(
        45,
        62,
        "joint row consistent = "
        f"{int(float(joint_row['overall_consistent']))}; "
        f"SE growth = {float(joint_row['joint_stokes_einstein_growth_over_poisson']):.2f}; "
        f"mismatch residual = {float(joint_row['rejected_mismatch_abs_log_residual']):.2f}",
    )
    c.drawString(
        45,
        48,
        "thermo scope row consistent = "
        f"{int(float(thermodynamic_scope_row['overall_consistent']))}; "
        f"dynamic entropy = {int(float(thermodynamic_scope_row['dynamic_model_derives_entropy']))}; "
        f"AG = {float(thermodynamic_scope_row['thermodynamic_adam_gibbs_slowdown']):.2g}",
    )
    c.drawString(430, 118, f"recovery row consistent = {int(float(recovery_row['overall_consistent']))}")
    c.drawString(
        430,
        104,
        "SE row consistent = "
        f"{int(float(se_row['overall_consistent']))}; "
        f"D tau growth = {float(se_row['se_product_growth']):.2f}; "
        f"xi_SE = {float(se_row['cold_fractional_exponent']):.3f}",
    )
    c.drawString(
        430,
        90,
        "chi4 row consistent = "
        f"{int(float(heterogeneity_row['overall_consistent']))}; "
        f"xi4 growth = {float(heterogeneity_row['length_growth']):.2f}; "
        f"chi4 growth = {float(heterogeneity_row['chi4_peak_growth_benchmark']):.1f}",
    )
    c.drawString(
        430,
        78,
        "front-law row consistent = "
        f"{int(float(spatial_front_row['overall_consistent']))}; "
        f"Df cv = {float(spatial_front_row['facilitation_diffusivity_relative_std']):.2g}; "
        f"xi4 growth = {float(spatial_front_row['length_growth']):.2f}",
    )
    c.drawString(
        430,
        68,
        "TTS row consistent = "
        f"{int(float(tts_row['overall_consistent']))}; "
        f"residual = {float(tts_row['cold_shape_residual']):.3f}; "
        f"C growth = {float(tts_row['alpha_shape_control_growth']):.2f}",
    )
    c.drawString(
        430,
        56,
        "KWW row consistent = "
        f"{int(float(stretched_row['overall_consistent']))}; "
        f"beta hot/cold = {float(stretched_row['hot_kww_beta']):.2f}/"
        f"{float(stretched_row['cold_kww_beta']):.2f}; "
        f"resid = {float(stretched_row['cold_fit_residual']):.3f}",
    )
    c.drawString(
        430,
        44,
        "persistence/exchange row consistent = "
        f"{int(float(persistence_exchange_row['overall_consistent']))}; "
        f"tau_p/tau_x = {float(persistence_exchange_row['inferred_persistence_exchange_ratio']):.1f}; "
        f"late residual = {float(persistence_exchange_row['late_ngp_log_residual_benchmark']):.2g}",
    )
    c.drawString(
        430,
        32,
        "van Hove row consistent = "
        f"{int(float(van_hove_row['overall_consistent']))}; "
        f"peak tail = {float(van_hove_row['peak_tail_ratio']):.2f}; "
        f"late tail = {float(van_hove_row['late_tail_ratio']):.2f}",
    )
    c.drawString(
        430,
        20,
        "fragility row consistent = "
        f"{int(float(fragility_row['overall_consistent']))}; "
        f"m growth = {float(fragility_row['fragility_index_growth']):.2f}; "
        f"AG = {float(fragility_row['adam_gibbs_slowdown']):.2g}",
    )
    c.showPage()
    c.save()


def write_literature_inversion_readiness_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_literature_inversion_readiness.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Literature inversion readiness")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Coverage checks separate qualitative benchmark support from quantitative, uncertainty-weighted inversion readiness.",
    )
    left, top = 70, page_h - 95
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top, "benchmark")
    c.drawString(left + 250, top, "coverage")
    c.drawString(left + 335, top, "qual")
    c.drawString(left + 380, top, "quant")
    c.drawString(left + 430, top, "unc")
    c.drawString(left + 475, top, "next action")
    c.setFont("Helvetica", 7)
    for idx, row in enumerate(rows):
        y = top - 18 - idx * 24
        coverage = float(row["observable_coverage_fraction"])
        quantitative = int(float(row["quantitative_inversion_ready"]))
        color = colors.HexColor("#2b6cb0") if quantitative else colors.HexColor("#d69e2e")
        c.setFillColor(colors.black)
        c.drawString(left, y, row["benchmark_id"][:42])
        c.setStrokeColor(colors.HexColor("#cbd5e0"))
        c.rect(left + 250, y - 3, 65, 8, stroke=1, fill=0)
        c.setFillColor(color)
        c.rect(left + 250, y - 3, 65 * coverage, 8, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawString(left + 320, y, f"{coverage:.2f}")
        c.drawString(left + 342, y, str(int(float(row["qualitative_comparison_ready"]))))
        c.drawString(left + 392, y, str(quantitative))
        c.drawString(left + 440, y, str(int(float(row["uncertainty_weighted_ready"]))))
        c.drawString(left + 475, y, row["next_action"][:70])
    c.showPage()
    c.save()


def write_observable_falsification_matrix_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_observable_falsification_matrix.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Observable falsification matrix")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Benchmark observables are mapped to the diagnostic protocols they can structurally or quantitatively falsify.",
    )
    benchmarks = list(dict.fromkeys(row["benchmark_id"] for row in rows))
    diagnostics = list(dict.fromkeys(row["diagnostic_id"] for row in rows))
    by_key = {(row["benchmark_id"], row["diagnostic_id"]): row for row in rows}
    left, top = 185, page_h - 92
    cell_w, cell_h = 108, 42
    c.setFont("Helvetica", 6.5)
    for idx, diagnostic_id in enumerate(diagnostics):
        c.drawString(left + idx * cell_w, top + 16, diagnostic_id[:18])
    for y_idx, benchmark_id in enumerate(benchmarks):
        y = top - 8 - y_idx * cell_h
        c.setFillColor(colors.black)
        c.drawString(42, y + 18, benchmark_id[:34])
        for x_idx, diagnostic_id in enumerate(diagnostics):
            row = by_key[(benchmark_id, diagnostic_id)]
            x = left + x_idx * cell_w
            structural = int(float(row["structural_falsification_ready"]))
            quantitative = int(float(row["quantitative_falsification_ready"]))
            coverage = float(row["observable_coverage_fraction"])
            if quantitative:
                fill = colors.HexColor("#2f855a")
            elif structural:
                fill = colors.HexColor("#2b6cb0")
            elif coverage >= 0.5:
                fill = colors.HexColor("#d69e2e")
            else:
                fill = colors.HexColor("#c05621")
            c.setFillColor(fill)
            c.rect(x, y, cell_w - 5, cell_h - 6, stroke=0, fill=1)
            c.setFillColor(colors.white)
            c.drawString(x + 5, y + 21, f"cov {coverage:.2f}")
            c.drawString(x + 5, y + 9, f"block {row['primary_blocker'][:12]}")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 7)
    c.drawString(42, 32, "green: quantitative ready; blue: all observables but no machine-readable uncertainties; yellow/red: blocked by missing observables")
    c.showPage()
    c.save()


def write_barrier_requirements_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_barrier_requirements.csv").open() as f:
        rows = list(csv.DictReader(f))
    amplification_rows = [row for row in rows if row["record_type"] == "amplification"]
    requirement = next(row for row in rows if row["record_type"] == "requirements")

    gaps = sorted({float(row["delay_barrier_gap"]) for row in amplification_rows})
    exchange_sums = sorted({float(row["exchange_barrier_sum"]) for row in amplification_rows})
    max_growth = max(float(row["combined_slowing_growth"]) for row in amplification_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Closed barrier-threshold requirements")

    left, bottom, width, height = 70, 165, 310, 230
    cell_w = width / len(exchange_sums)
    cell_h = height / len(gaps)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, bottom + height + 18, "A. Combined slowing amplification")
    for row_idx, gap in enumerate(gaps):
        y = bottom + row_idx * cell_h
        c.setFont("Helvetica", 7)
        c.drawRightString(left - 8, y + cell_h / 2.0, f"{gap:g}")
        for col_idx, exchange_sum in enumerate(exchange_sums):
            row = next(
                item
                for item in amplification_rows
                if float(item["delay_barrier_gap"]) == gap and float(item["exchange_barrier_sum"]) == exchange_sum
            )
            growth = float(row["combined_slowing_growth"])
            shade = math.log(growth) / math.log(max_growth) if max_growth > 1.0 else 0.0
            c.setFillColor(colors.Color(0.92 - 0.28 * shade, 0.82 - 0.12 * shade, 0.52))
            x0 = left + col_idx * cell_w
            c.rect(x0, y, cell_w - 3, cell_h - 3, fill=1, stroke=0)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 8)
            c.drawString(x0 + 8, y + cell_h / 2.0, f"{growth:.1f}x")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 8)
    for col_idx, exchange_sum in enumerate(exchange_sums):
        c.drawCentredString(left + col_idx * cell_w + cell_w / 2.0, bottom - 14, f"{exchange_sum:g}")
    c.drawCentredString(left + width / 2.0, bottom - 30, "exchange barrier sum")
    c.saveState()
    c.translate(left - 42, bottom + height / 2.0)
    c.rotate(90)
    c.drawCentredString(0, 0, "delay barrier gap")
    c.restoreState()

    labels = ["lambda tau_d", "heterogeneity c", "combined"]
    targets = [
        float(requirement["target_lambda_tau_delay_growth"]),
        float(requirement["target_heterogeneity_ratio_growth"]),
        float(requirement["target_combined_growth"]),
    ]
    barriers = [
        float(requirement["required_delay_barrier_gap"]),
        float(requirement["required_exchange_barrier_sum"]),
        float(requirement["required_combined_barrier"]),
    ]
    chart_left, chart_bottom, chart_width, chart_height = 455, 165, 270, 230
    c.setFont("Helvetica-Bold", 10)
    c.drawString(chart_left, chart_bottom + chart_height + 18, "B. Minimum barriers for target growth")
    c.line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom)
    c.line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height)
    colors_list = [colors.HexColor("#2b6cb0"), colors.HexColor("#c05621"), colors.HexColor("#805ad5")]
    for idx, (label, target, barrier) in enumerate(zip(labels, targets, barriers)):
        x = chart_left + 32 + idx * 78
        h = barrier / max(barriers) * (chart_height - 30)
        c.setFillColor(colors_list[idx])
        c.rect(x, chart_bottom, 52, h, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7)
        c.drawCentredString(x + 26, chart_bottom - 13, label)
        c.drawCentredString(x + 26, chart_bottom + h + 8, f"E={barrier:.1f}")
        c.drawCentredString(x + 26, chart_bottom - 25, f"{target:.1f}x")
    c.showPage()
    c.save()


def write_mechanism_selection_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_mechanism_selection.csv").open() as f:
        rows = list(csv.DictReader(f))

    cases = list(dict.fromkeys(row["case"] for row in rows))
    candidates = list(dict.fromkeys(row["candidate_model"] for row in rows))
    capped_scores = []
    for row in rows:
        score = float(row["score"])
        capped_scores.append(4.0 if not math.isfinite(score) else min(math.log10(1.0 + score), 4.0))
    max_score = max(capped_scores)

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Late mechanism-selection diagnostic")
    c.setFont("Helvetica", 8)
    c.drawString(42, page_h - 48, "Two late NGP points and one alpha slope select among Poisson, static-gamma, and finite-exchange mechanisms.")

    left, bottom, width, height = 58, 125, 675, 285
    c.line(left, bottom, left + width, bottom)
    c.line(left, bottom, left, bottom + height)
    group_w = width / len(cases)
    bar_w = group_w / (len(candidates) + 1)
    colors_list = [colors.HexColor("#2b6cb0"), colors.HexColor("#c05621"), colors.HexColor("#2f855a")]
    for case_idx, case in enumerate(cases):
        x_group = left + case_idx * group_w
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(x_group + group_w / 2.0, bottom - 18, case.replace("_", " "))
        for cand_idx, candidate in enumerate(candidates):
            row = next(item for item in rows if item["case"] == case and item["candidate_model"] == candidate)
            score = float(row["score"])
            plotted = 4.0 if not math.isfinite(score) else min(math.log10(1.0 + score), 4.0)
            x = x_group + (cand_idx + 0.5) * bar_w
            h = plotted / max_score * (height - 22)
            c.setFillColor(colors_list[cand_idx % len(colors_list)])
            c.rect(x, bottom, bar_w * 0.75, h, fill=1, stroke=0)
            if float(row["passes"]) >= 1.0:
                c.setFillColor(colors.black)
                c.setFont("Helvetica-Bold", 7)
                c.drawCentredString(x + bar_w * 0.38, bottom + h + 7, "best")
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, bottom + height + 16, "A. log10(1 + candidate score), capped")
    c.setFont("Helvetica", 7)
    for idx, candidate in enumerate(candidates):
        c.setFillColor(colors_list[idx % len(colors_list)])
        c.drawString(left + 8 + idx * 115, bottom + height - 12, candidate)
    c.setFillColor(colors.black)
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


def write_heterogeneity_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_heterogeneity.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Finite-exchange heterogeneity extension")

    log_time = data["log10_time"]
    draw_panel(
        c,
        52,
        78,
        320,
        280,
        log_time,
        [
            ("Poisson alpha decay", data["poisson_alpha_decay"], colors.HexColor("#2b6cb0")),
            ("gamma-exchange alpha decay", data["gamma_exchange_alpha_decay"], colors.HexColor("#c05621")),
        ],
        "O. Alpha relaxation from renewal heterogeneity",
        xlabel="log10 time",
        y_range=(0.0, 1.0),
    )
    draw_panel(
        c,
        430,
        78,
        320,
        280,
        log_time,
        [
            ("Poisson NGP", data["poisson_ngp"], colors.HexColor("#2b6cb0")),
            ("gamma-exchange NGP", data["gamma_exchange_ngp"], colors.HexColor("#c05621")),
            ("local beta", data["gamma_exchange_local_beta"], colors.HexColor("#2f855a")),
        ],
        "P. Enhanced NGP with recovery",
        xlabel="log10 time",
    )

    c.showPage()
    c.save()


def write_heterogeneity_map_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_heterogeneity_map.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Finite-exchange diagnostic map")

    log_ratio = data["log10_one_plus_ratio"]
    inferred_error = np.abs(data["inferred_ratio_from_alpha_rate"] - data["heterogeneity_ratio"])
    draw_panel(
        c,
        52,
        78,
        320,
        280,
        log_ratio,
        [
            ("R alpha_2 -> 1+c", data["late_ngp_renewal_amplitude"], colors.HexColor("#2b6cb0")),
            ("alpha-rate inferred c error", inferred_error, colors.HexColor("#c05621")),
        ],
        "Q. Late NGP amplitude versus exchange ratio",
        xlabel="log10(1+c)",
    )
    draw_panel(
        c,
        430,
        78,
        320,
        280,
        log_ratio,
        [
            ("alpha-rate renormalization", data["alpha_rate_renormalization"], colors.HexColor("#805ad5")),
            ("passes joint criterion", data["passes_joint_criterion"], colors.HexColor("#2f855a")),
        ],
        "R. Alpha slowing and observable window",
        xlabel="log10(1+c)",
        y_range=(0.0, 1.05),
    )

    c.showPage()
    c.save()


def write_static_null_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_static_null.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Static gamma null versus finite exchange")

    log_time = data["log10_time"]
    draw_panel(
        c,
        52,
        78,
        320,
        280,
        log_time,
        [
            ("Poisson alpha decay", data["poisson_alpha_decay"], colors.HexColor("#2b6cb0")),
            ("finite-exchange alpha decay", data["gamma_exchange_alpha_decay"], colors.HexColor("#c05621")),
            ("static-gamma alpha decay", data["static_gamma_alpha_decay"], colors.HexColor("#805ad5")),
        ],
        "S. Static disorder broadens alpha decay",
        xlabel="log10 time",
        y_range=(0.0, 1.0),
    )
    draw_panel(
        c,
        430,
        78,
        320,
        280,
        log_time,
        [
            ("finite-exchange NGP", data["gamma_exchange_ngp"], colors.HexColor("#c05621")),
            ("static-gamma NGP", data["static_gamma_ngp"], colors.HexColor("#805ad5")),
            ("static plateau 1/kappa0", data["static_gamma_late_ngp_plateau"], colors.grey),
        ],
        "T. Static disorder lacks Gaussian recovery",
        xlabel="log10 time",
    )

    c.showPage()
    c.save()


def write_inversion_pdf(path: Path) -> None:
    data = read_csv_columns(DATA_DIR / "renewal_cage_inversion.csv")
    diffusion_scale = data["diffusion_scale"]
    margin = data["existence_margin"]
    jump_ratio = data["inferred_jump_to_cage_variance"]
    time_residual = data["log_peak_time_residual"]
    height_residual = data["log_peak_height_residual"]

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Observable inversion and falsifiability diagnostics")

    draw_panel(
        c,
        45,
        160,
        320,
        280,
        diffusion_scale,
        [
            ("existence margin", margin, colors.HexColor("#2b6cb0")),
            ("margin=1 threshold", np.ones_like(margin), colors.grey),
            ("inferred q/A", jump_ratio, colors.HexColor("#c05621")),
        ],
        "M. Scattering-transport existence margin",
        xlabel="D / D_observed",
    )
    draw_panel(
        c,
        430,
        160,
        320,
        280,
        diffusion_scale,
        [
            ("log t*_pred / t*_obs", time_residual, colors.HexColor("#2f855a")),
            ("log alpha*_pred / alpha*_obs", height_residual, colors.HexColor("#805ad5")),
        ],
        "N. NGP peak residuals after inversion",
        xlabel="D / D_observed",
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
    alpha_shape_pdf = PAPER_FIGURE_DIR / "renewal_cage_alpha_shape.pdf"
    facilitated_exchange_pdf = PAPER_FIGURE_DIR / "renewal_cage_facilitated_exchange.pdf"
    persistence_exchange_pdf = PAPER_FIGURE_DIR / "renewal_cage_persistence_exchange.pdf"
    persistence_exchange_protocol_pdf = PAPER_FIGURE_DIR / "renewal_cage_persistence_exchange_protocol.pdf"
    persistence_exchange_joint_protocol_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_persistence_exchange_joint_protocol.pdf"
    )
    persistence_exchange_uncertainty_protocol_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_persistence_exchange_uncertainty_protocol.pdf"
    )
    glass_audit_pdf = PAPER_FIGURE_DIR / "renewal_cage_glass_audit.pdf"
    glass_phase_diagram_pdf = PAPER_FIGURE_DIR / "renewal_cage_glass_phase_diagram.pdf"
    spatial_chi4_pdf = PAPER_FIGURE_DIR / "renewal_cage_spatial_chi4.pdf"
    thermodynamic_closure_pdf = PAPER_FIGURE_DIR / "renewal_cage_thermodynamic_closure.pdf"
    mct_beta_closure_pdf = PAPER_FIGURE_DIR / "renewal_cage_mct_beta_closure.pdf"
    sota_benchmark_consistency_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_benchmark_consistency.pdf"
    literature_inversion_readiness_pdf = PAPER_FIGURE_DIR / "renewal_cage_literature_inversion_readiness.pdf"
    observable_falsification_matrix_pdf = PAPER_FIGURE_DIR / "renewal_cage_observable_falsification_matrix.pdf"
    barrier_requirements_pdf = PAPER_FIGURE_DIR / "renewal_cage_barrier_requirements.pdf"
    mechanism_selection_pdf = PAPER_FIGURE_DIR / "renewal_cage_mechanism_selection.pdf"
    barrier_pdf = PAPER_FIGURE_DIR / "renewal_cage_barrier.pdf"
    heterogeneity_pdf = PAPER_FIGURE_DIR / "renewal_cage_heterogeneity.pdf"
    heterogeneity_map_pdf = PAPER_FIGURE_DIR / "renewal_cage_heterogeneity_map.pdf"
    static_null_pdf = PAPER_FIGURE_DIR / "renewal_cage_static_null.pdf"
    inversion_pdf = PAPER_FIGURE_DIR / "renewal_cage_inversion.pdf"
    write_results_pdf(results_pdf)
    write_dimensionless_pdf(dimensionless_pdf)
    write_scattering_pdf(scattering_pdf)
    write_temperature_pdf(temperature_pdf)
    write_alpha_shape_pdf(alpha_shape_pdf)
    write_facilitated_exchange_pdf(facilitated_exchange_pdf)
    write_persistence_exchange_pdf(persistence_exchange_pdf)
    write_persistence_exchange_protocol_pdf(persistence_exchange_protocol_pdf)
    write_persistence_exchange_joint_protocol_pdf(persistence_exchange_joint_protocol_pdf)
    write_persistence_exchange_uncertainty_protocol_pdf(persistence_exchange_uncertainty_protocol_pdf)
    write_glass_audit_pdf(glass_audit_pdf)
    write_glass_phase_diagram_pdf(glass_phase_diagram_pdf)
    write_spatial_chi4_pdf(spatial_chi4_pdf)
    write_thermodynamic_closure_pdf(thermodynamic_closure_pdf)
    write_mct_beta_closure_pdf(mct_beta_closure_pdf)
    write_sota_benchmark_consistency_pdf(sota_benchmark_consistency_pdf)
    write_literature_inversion_readiness_pdf(literature_inversion_readiness_pdf)
    write_observable_falsification_matrix_pdf(observable_falsification_matrix_pdf)
    write_barrier_requirements_pdf(barrier_requirements_pdf)
    write_mechanism_selection_pdf(mechanism_selection_pdf)
    write_barrier_pdf(barrier_pdf)
    write_heterogeneity_pdf(heterogeneity_pdf)
    write_heterogeneity_map_pdf(heterogeneity_map_pdf)
    write_static_null_pdf(static_null_pdf)
    write_inversion_pdf(inversion_pdf)

    zip_path = output_dir / "renewal-cage-arxiv-source.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(PAPER_DIR / "main.tex", "main.tex")
        archive.write(PAPER_DIR / "references.bib", "references.bib")
        archive.write(results_pdf, "figures/renewal_cage_results.pdf")
        archive.write(dimensionless_pdf, "figures/renewal_cage_dimensionless.pdf")
        archive.write(scattering_pdf, "figures/renewal_cage_scattering.pdf")
        archive.write(temperature_pdf, "figures/renewal_cage_temperature.pdf")
        archive.write(alpha_shape_pdf, "figures/renewal_cage_alpha_shape.pdf")
        archive.write(facilitated_exchange_pdf, "figures/renewal_cage_facilitated_exchange.pdf")
        archive.write(persistence_exchange_pdf, "figures/renewal_cage_persistence_exchange.pdf")
        archive.write(persistence_exchange_protocol_pdf, "figures/renewal_cage_persistence_exchange_protocol.pdf")
        archive.write(
            persistence_exchange_joint_protocol_pdf,
            "figures/renewal_cage_persistence_exchange_joint_protocol.pdf",
        )
        archive.write(
            persistence_exchange_uncertainty_protocol_pdf,
            "figures/renewal_cage_persistence_exchange_uncertainty_protocol.pdf",
        )
        archive.write(glass_audit_pdf, "figures/renewal_cage_glass_audit.pdf")
        archive.write(glass_phase_diagram_pdf, "figures/renewal_cage_glass_phase_diagram.pdf")
        archive.write(spatial_chi4_pdf, "figures/renewal_cage_spatial_chi4.pdf")
        archive.write(thermodynamic_closure_pdf, "figures/renewal_cage_thermodynamic_closure.pdf")
        archive.write(mct_beta_closure_pdf, "figures/renewal_cage_mct_beta_closure.pdf")
        archive.write(sota_benchmark_consistency_pdf, "figures/renewal_cage_sota_benchmark_consistency.pdf")
        archive.write(literature_inversion_readiness_pdf, "figures/renewal_cage_literature_inversion_readiness.pdf")
        archive.write(
            observable_falsification_matrix_pdf,
            "figures/renewal_cage_observable_falsification_matrix.pdf",
        )
        archive.write(barrier_requirements_pdf, "figures/renewal_cage_barrier_requirements.pdf")
        archive.write(mechanism_selection_pdf, "figures/renewal_cage_mechanism_selection.pdf")
        archive.write(barrier_pdf, "figures/renewal_cage_barrier.pdf")
        archive.write(heterogeneity_pdf, "figures/renewal_cage_heterogeneity.pdf")
        archive.write(heterogeneity_map_pdf, "figures/renewal_cage_heterogeneity_map.pdf")
        archive.write(static_null_pdf, "figures/renewal_cage_static_null.pdf")
        archive.write(inversion_pdf, "figures/renewal_cage_inversion.pdf")
    return zip_path


if __name__ == "__main__":
    path = build_arxiv_package()
    print(path)
