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


def write_translation_rotation_protocol_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_translation_rotation_protocol.csv").open() as f:
        rows = list(csv.DictReader(f))

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Translation-rotation renewal diagnostic")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "A rotational renewal clock extends persistence/exchange tests to Debye-Stokes-Einstein decoupling.",
    )

    labels = [row["scenario"].replace("_", " ") for row in rows]
    ratio = np.array([float(row["translation_rotation_ratio"]) for row in rows])
    dse = np.array([float(row["rotational_dse_product"]) for row in rows])
    residual = np.array([abs(float(row["rotational_tau_log_residual"])) for row in rows])
    detected = np.array([float(row["translation_rotation_decoupling_detected"]) for row in rows])

    left_a, left_b, bottom, width, height = 55.0, 430.0, 150.0, 300.0, 260.0
    for left, title in [(left_a, "A. Clock decoupling"), (left_b, "B. Inversion residual")]:
        c.setStrokeColor(colors.black)
        c.line(left, bottom, left + width, bottom)
        c.line(left, bottom, left, bottom + height)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(left, bottom + height + 13, title)

    x_positions = np.linspace(left_a + 80, left_a + width - 80, len(rows))
    y_max = max(float(np.max(ratio)), float(np.max(dse / dse[0])), 1.0) * 1.15
    for idx, label in enumerate(labels):
        color = colors.HexColor("#c05621") if detected[idx] > 0.5 else colors.HexColor("#2f855a")
        ratio_h = ratio[idx] / y_max * height
        dse_h = (dse[idx] / dse[0]) / y_max * height
        c.setFillColor(color)
        c.rect(float(x_positions[idx] - 18), bottom, 18, ratio_h, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#805ad5"))
        c.rect(float(x_positions[idx] + 6), bottom, 18, dse_h, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(float(x_positions[idx]), bottom - 24, label)
        c.drawCentredString(float(x_positions[idx]), bottom - 35, f"tau_r/tau_a={ratio[idx]:.2f}")

    residual_max = max(float(np.max(residual)), 0.01)
    x_res = np.linspace(left_b + 85, left_b + width - 85, len(rows))
    for idx, label in enumerate(labels):
        y = bottom + residual[idx] / residual_max * height if residual_max > 0.0 else bottom
        c.setFillColor(colors.HexColor("#2b6cb0"))
        c.circle(float(x_res[idx]), y, 5, fill=1, stroke=0)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(float(x_res[idx]), bottom - 24, label)
        c.drawCentredString(float(x_res[idx]), bottom - 35, f"|res|={residual[idx]:.1e}")

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 8)
    c.drawString(55, 95, "Orange bars mark detected translation-rotation decoupling; purple bars show D tau_rot normalized by the coupled clock.")
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


def write_benchmark_fusion_readiness_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_benchmark_fusion_readiness.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Cross-benchmark fusion readiness")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Multi-paper validation is allowed only when observables, system, temperature grid, and ensemble identity align.",
    )
    left, top = 55, page_h - 95
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top, "fusion")
    c.drawString(left + 250, top, "coverage")
    c.drawString(left + 345, top, "sys/grid/ens")
    c.drawString(left + 455, top, "struct")
    c.drawString(left + 505, top, "quant")
    c.drawString(left + 555, top, "primary blocker")
    c.setFont("Helvetica", 7.5)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 35
        coverage = float(row["observable_coverage_fraction"])
        structural = int(float(row["structural_fusion_ready"]))
        quantitative = int(float(row["quantitative_fusion_ready"]))
        color = colors.HexColor("#2f855a") if quantitative else colors.HexColor("#2b6cb0") if structural else colors.HexColor("#c05621")
        c.setFillColor(colors.black)
        c.drawString(left, y, row["fusion_id"][:38])
        c.setStrokeColor(colors.HexColor("#cbd5e0"))
        c.rect(left + 250, y - 3, 70, 9, stroke=1, fill=0)
        c.setFillColor(color)
        c.rect(left + 250, y - 3, 70 * coverage, 9, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawString(left + 325, y, f"{coverage:.2f}")
        c.drawString(
            left + 345,
            y,
            f"{int(float(row['shared_system_consistent']))}/"
            f"{int(float(row['shared_temperature_grid_consistent']))}/"
            f"{int(float(row['shared_ensemble_consistent']))}",
        )
        c.drawString(left + 468, y, str(structural))
        c.drawString(left + 518, y, str(quantitative))
        c.drawString(left + 555, y, row["primary_blocker"][:28])
        c.drawString(left + 55, y - 12, f"sources: {row['benchmark_sources'][:85]}")
    c.showPage()
    c.save()


def write_raw_curve_ingestion_contract_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_raw_curve_ingestion_contract.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Raw-curve ingestion contract")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "KA I/II fused validation requires machine-readable observable columns and uncertainty columns before quantitative inversion.",
    )
    left, top = 52, page_h - 95
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top, "observable")
    c.drawString(left + 190, top, "diagnostic")
    c.drawString(left + 390, top, "struct")
    c.drawString(left + 440, top, "unc")
    c.drawString(left + 485, top, "primary blocker")
    c.drawString(left + 620, top, "missing uncertainty columns")
    c.setFont("Helvetica", 7.5)
    for idx, row in enumerate(rows):
        y = top - 24 - idx * 38
        structural = int(float(row["structural_ingestion_ready"]))
        uncertainty = int(float(row["uncertainty_ingestion_ready"]))
        color = colors.HexColor("#2f855a") if uncertainty else colors.HexColor("#2b6cb0") if structural else colors.HexColor("#c05621")
        c.setFillColor(colors.black)
        c.drawString(left, y, row["observable_id"][:28])
        c.drawString(left + 190, y, row["target_diagnostic"][:30])
        c.setFillColor(color)
        c.rect(left + 390, y - 4, 16, 10, stroke=0, fill=1)
        c.rect(left + 440, y - 4, 16, 10, stroke=0, fill=1 if uncertainty else 0)
        c.setFillColor(colors.black)
        c.drawString(left + 410, y, str(structural))
        c.drawString(left + 460, y, str(uncertainty))
        c.drawString(left + 485, y, row["primary_blocker"][:24])
        c.drawString(left + 620, y, row["missing_uncertainty_columns"][:60])
    c.showPage()
    c.save()


def write_sota_claim_alignment_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_claim_alignment.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA claim alignment audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Source-level claims are mapped to diagnostics, data readiness, and scope boundaries.",
    )
    left, top = 45, page_h - 88
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(left, top, "source")
    c.drawString(left + 120, top, "phenomenon")
    c.drawString(left + 330, top, "alignment")
    c.drawString(left + 420, top, "support")
    c.drawString(left + 535, top, "closure")
    c.drawString(left + 590, top, "fit")
    c.drawString(left + 630, top, "blocker")
    palette = {
        "supported": colors.HexColor("#2f855a"),
        "partial": colors.HexColor("#2b6cb0"),
        "scope_boundary": colors.HexColor("#805ad5"),
        "not_supported": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 20 - idx * 38
        alignment = row["claim_alignment"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["source_key"][:20])
        c.drawString(left + 120, y, row["phenomenon"].replace("_", " ")[:35])
        c.setFillColor(palette[alignment])
        c.rect(left + 330, y - 4, 72, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 335, y, alignment.replace("_", " ")[:14])
        c.setFillColor(colors.black)
        c.drawString(left + 420, y, row["model_support_level"].replace("_", " ")[:17])
        c.drawString(left + 535, y, str(int(float(row["requires_external_closure"]))))
        c.drawString(left + 590, y, str(int(float(row["quantitative_fit_ready"]))))
        c.drawString(left + 630, y, row["primary_blocker"].replace("_", " ")[:24])
        c.drawString(left + 120, y - 10, row["observed_claim"][:96])
    c.showPage()
    c.save()


def write_sota_signed_constraints_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_signed_constraints.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA signed-constraint audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Representative literature conclusions are encoded as required signatures and forbidden overclaims.",
    )
    left, top = 45, page_h - 88
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(left, top, "source")
    c.drawString(left + 120, top, "scope")
    c.drawString(left + 245, top, "constraint class")
    c.drawString(left + 382, top, "closure")
    c.drawString(left + 435, top, "publish")
    c.drawString(left + 492, top, "missing")
    c.drawString(left + 612, top, "forbidden made")
    palette = {
        "sota_consistent": colors.HexColor("#2f855a"),
        "closure_assisted_consistent": colors.HexColor("#2b6cb0"),
        "scope_boundary_consistent": colors.HexColor("#805ad5"),
        "missing_signature": colors.HexColor("#c05621"),
        "overclaimed_boundary": colors.HexColor("#b83280"),
        "not_supported": colors.HexColor("#718096"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 20 - idx * 38
        constraint_class = row["signed_constraint_class"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["source_key"][:22])
        c.drawString(left + 120, y, row["model_scope"].replace("_", " ")[:20])
        c.setFillColor(palette[constraint_class])
        c.rect(left + 245, y - 4, 118, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 250, y, constraint_class.replace("_", " ")[:24])
        c.setFillColor(colors.black)
        c.drawString(left + 382, y, str(int(float(row["requires_external_closure"]))))
        c.drawString(left + 435, y, str(int(float(row["publishable_alignment"]))))
        c.drawString(left + 492, y, row["missing_expected_signatures"].replace("_", " ")[:22])
        c.drawString(left + 612, y, row["forbidden_claims_made"].replace("_", " ")[:25])
        c.drawString(left + 120, y - 10, row["source_observation"][:100])
    c.showPage()
    c.save()


def write_sota_evidence_verdict_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_evidence_verdict.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA evidence verdict ledger")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Claim alignment, signed constraints, and reanalysis state are collapsed into manuscript-safe evidence grades.",
    )
    left, top = 42, page_h - 88
    c.setFont("Helvetica-Bold", 7.4)
    c.drawString(left, top, "source")
    c.drawString(left + 135, top, "phenomenon")
    c.drawString(left + 310, top, "evidence grade")
    c.drawString(left + 475, top, "allowed claim")
    c.drawString(left + 630, top, "publish")
    c.drawString(left + 680, top, "reanalyze")
    palette = {
        "direct_dynamical_support": colors.HexColor("#2f855a"),
        "closure_assisted_support": colors.HexColor("#2b6cb0"),
        "thermodynamic_scope_boundary": colors.HexColor("#805ad5"),
        "pending_trajectory_reanalysis": colors.HexColor("#d69e2e"),
        "overclaimed_or_forbidden": colors.HexColor("#b83280"),
        "not_supported": colors.HexColor("#718096"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 18 - idx * 32
        grade = row["evidence_grade"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["source_key"][:24])
        c.drawString(left + 135, y, row["phenomenon"].replace("_", " ")[:30])
        c.setFillColor(palette[grade])
        c.rect(left + 310, y - 4, 145, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 315, y, grade.replace("_", " ")[:28])
        c.setFillColor(colors.black)
        c.drawString(left + 475, y, row["allowed_manuscript_claim"].replace("_", " ")[:27])
        c.drawString(left + 635, y, str(int(float(row["publishable_without_overclaim"]))))
        c.drawString(left + 690, y, str(int(float(row["trajectory_reanalysis_required"]))))
        c.drawString(left + 135, y - 10, f"stage: {row['reanalysis_stage'].replace('_', ' ')[:64]}")
    c.showPage()
    c.save()


def write_sota_evidence_class_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_evidence_class.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA evidence-class gate")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Simulations, experiments, repositories, canaries, and thermodynamic claims are separated before quantitative inversion.",
    )
    left, top = 42, page_h - 88
    c.setFont("Helvetica-Bold", 7.4)
    c.drawString(left, top, "source")
    c.drawString(left + 145, top, "modality")
    c.drawString(left + 245, top, "evidence class")
    c.drawString(left + 485, top, "trend")
    c.drawString(left + 535, top, "invert")
    c.drawString(left + 592, top, "closure")
    c.drawString(left + 648, top, "blocker")
    palette = {
        "uncertainty_weighted_quantitative_test": colors.HexColor("#2f855a"),
        "structural_simulation_support": colors.HexColor("#2b6cb0"),
        "qualitative_experimental_trend": colors.HexColor("#d69e2e"),
        "closure_assisted_experimental_constraint": colors.HexColor("#805ad5"),
        "metadata_reanalysis_candidate": colors.HexColor("#718096"),
        "thermodynamic_scope_boundary": colors.HexColor("#b83280"),
        "closure_assisted_simulation_constraint": colors.HexColor("#805ad5"),
        "not_supported": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 18 - idx * 36
        evidence_class = row["evidence_class"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["source_key"][:26])
        c.drawString(left + 145, y, row["source_modality"].replace("_", " ")[:17])
        c.setFillColor(palette[evidence_class])
        c.rect(left + 245, y - 4, 220, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 250, y, evidence_class.replace("_", " ")[:38])
        c.setFillColor(colors.black)
        c.drawString(left + 495, y, str(int(float(row["trend_comparison_allowed"]))))
        c.drawString(left + 548, y, str(int(float(row["quantitative_inversion_allowed"]))))
        c.drawString(left + 606, y, str(int(float(row["requires_external_closure"]))))
        c.drawString(left + 648, y, row["primary_blocker"].replace("_", " ")[:23])
        c.drawString(left + 245, y - 10, f"missing inputs: {row['missing_quantitative_inputs'].replace('_', ' ')[:78]}")
    c.showPage()
    c.save()


def write_simultaneous_closure_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_simultaneous_closure.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Simultaneous dynamical-signature closure gate")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "A minimal D plus anchor-alpha inversion must pass held-out alpha, late-NGP, SE-growth, and chi4-proxy checks.",
    )
    left, top = 42, page_h - 88
    c.setFont("Helvetica-Bold", 7.4)
    c.drawString(left, top, "protocol")
    c.drawString(left + 220, top, "closure stage")
    c.drawString(left + 520, top, "held")
    c.drawString(left + 560, top, "pass")
    c.drawString(left + 610, top, "SE growth")
    c.drawString(left + 675, top, "blocker")
    palette = {
        "simultaneous_dynamical_signature_closure_passed": colors.HexColor("#2f855a"),
        "dynamical_heldout_prediction_failed": colors.HexColor("#c05621"),
        "scored_protocol_incomplete": colors.HexColor("#718096"),
    }
    c.setFont("Helvetica", 7)
    for idx, row in enumerate(rows):
        y = top - 20 - idx * 42
        stage = row["closure_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["protocol_id"].replace("_", " ")[:34])
        c.setFillColor(palette[stage])
        c.rect(left + 220, y - 4, 270, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 225, y, stage.replace("_", " ")[:44])
        c.setFillColor(colors.black)
        c.drawString(left + 527, y, str(int(float(row["heldout_count"]))))
        c.drawString(left + 570, y, str(int(float(row["all_required_dynamical_predictions_pass"]))))
        c.drawString(left + 616, y, f"{float(row['stokes_einstein_growth_over_poisson']):.2f}")
        c.drawString(left + 675, y, row["primary_blocker"].replace("_", " ")[:22])
        c.drawString(left + 220, y - 12, f"held-out: {row['heldout_observables'].replace('_', ' ')[:82]}")
    c.showPage()
    c.save()


def write_real_benchmark_assimilation_gate_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_real_benchmark_assimilation_gate.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Real benchmark assimilation gate")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Published benchmarks must pass coverage, shared-system, machine-readable, and uncertainty gates before quantitative inversion.",
    )
    left, top = 44, page_h - 88
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "benchmark")
    c.drawString(left + 182, top, "protocol")
    c.drawString(left + 320, top, "coverage")
    c.drawString(left + 390, top, "stage")
    c.drawString(left + 545, top, "struct")
    c.drawString(left + 588, top, "unc")
    c.drawString(left + 625, top, "blocker")
    palette = {
        "uncertainty_weighted_inversion": colors.HexColor("#2f855a"),
        "structural_digitization_ready": colors.HexColor("#2b6cb0"),
        "qualitative_alignment_only": colors.HexColor("#d69e2e"),
        "scope_boundary_only": colors.HexColor("#805ad5"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 39
        stage = row["assimilation_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["benchmark_id"].replace("_", " ")[:32])
        c.drawString(left + 182, y, row["target_protocol"].replace("_", " ")[:24])
        c.drawString(left + 320, y, f"{float(row['required_observable_coverage']):.2f}")
        c.setFillColor(palette[stage])
        c.rect(left + 390, y - 4, 135, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 395, y, stage.replace("_", " ")[:24])
        c.setFillColor(colors.black)
        c.drawString(left + 548, y, str(int(float(row["structural_inversion_ready"]))))
        c.drawString(left + 592, y, str(int(float(row["uncertainty_weighted_ready"]))))
        c.drawString(left + 625, y, row["primary_blocker"].replace("_", " ")[:27])
        c.drawString(left + 182, y - 10, f"source: {row['source_key'][:54]}")
    c.showPage()
    c.save()


def write_cross_observable_prediction_ledger_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_cross_observable_prediction_ledger.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Cross-observable prediction ledger")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Calibration inputs are separated from held-out predictions and closure variables.",
    )
    left, top = 42, page_h - 86
    c.setFont("Helvetica-Bold", 7.2)
    c.drawString(left, top, "protocol")
    c.drawString(left + 215, top, "class")
    c.drawString(left + 375, top, "fit")
    c.drawString(left + 410, top, "held")
    c.drawString(left + 455, top, "closure")
    c.drawString(left + 505, top, "risk")
    c.drawString(left + 555, top, "held-out predictions")
    palette = {
        "predictive_diagnostic": colors.HexColor("#2f855a"),
        "closure_assisted_prediction": colors.HexColor("#2b6cb0"),
        "underconstrained_fit": colors.HexColor("#c05621"),
        "failed_prediction": colors.HexColor("#c53030"),
        "scope_boundary": colors.HexColor("#805ad5"),
        "not_supported": colors.HexColor("#718096"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 39
        klass = row["prediction_class"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["protocol_id"].replace("_", " ")[:34])
        c.setFillColor(palette[klass])
        c.rect(left + 215, y - 4, 138, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 220, y, klass.replace("_", " ")[:24])
        c.setFillColor(colors.black)
        c.drawString(left + 380, y, str(int(float(row["calibration_count"]))))
        c.drawString(left + 420, y, str(int(float(row["heldout_prediction_count"]))))
        c.drawString(left + 468, y, str(int(float(row["closure_observable_count"]))))
        c.drawString(left + 515, y, str(int(float(row["fit_only_overclaim_risk"]))))
        c.drawString(left + 555, y, row["heldout_predictions"].replace("_", " ")[:42])
        c.drawString(left + 80, y - 10, f"fit: {row['calibration_observables'].replace('_', ' ')[:80]}")
    c.showPage()
    c.save()


def write_inversion_identifiability_audit_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_inversion_identifiability_audit.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Inversion identifiability audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Pre-fit rank, held-out prediction, closure, and degeneracy checks prevent fit-only overclaims.",
    )
    left, top = 42, page_h - 86
    c.setFont("Helvetica-Bold", 7.2)
    c.drawString(left, top, "protocol")
    c.drawString(left + 215, top, "class")
    c.drawString(left + 394, top, "fit")
    c.drawString(left + 428, top, "par")
    c.drawString(left + 465, top, "rank")
    c.drawString(left + 505, top, "held")
    c.drawString(left + 545, top, "closure")
    c.drawString(left + 595, top, "risk")
    c.drawString(left + 635, top, "held-out predictions")
    palette = {
        "identifiable_prediction": colors.HexColor("#2f855a"),
        "conditionally_identifiable": colors.HexColor("#2b6cb0"),
        "underidentified_fit": colors.HexColor("#c05621"),
        "degenerate_fit": colors.HexColor("#c53030"),
        "scope_boundary": colors.HexColor("#805ad5"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 35
        klass = row["identifiability_class"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["protocol_id"].replace("_", " ")[:34])
        c.setFillColor(palette[klass])
        c.rect(left + 215, y - 4, 154, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 220, y, klass.replace("_", " ")[:26])
        c.setFillColor(colors.black)
        c.drawString(left + 400, y, str(int(float(row["fit_observable_count"]))))
        c.drawString(left + 435, y, str(int(float(row["inferred_parameter_count"]))))
        c.drawString(left + 476, y, str(int(float(row["rank_margin"]))))
        c.drawString(left + 515, y, str(int(float(row["heldout_prediction_count"]))))
        c.drawString(left + 560, y, str(int(float(row["closure_count"]))))
        c.drawString(left + 605, y, str(int(float(row["overclaim_risk"]))))
        c.drawString(left + 635, y, row["heldout_predictions"].replace("_", " ")[:36])
        c.drawString(left + 80, y - 10, f"params: {row['inferred_parameters'].replace('_', ' ')[:76]}")
    c.showPage()
    c.save()


def write_frontier_benchmark_horizon_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_frontier_benchmark_horizon.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Frontier benchmark horizon")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Recent SOTA sources are classified as reanalysis targets, closure horizons, model-extension gaps, or scope boundaries.",
    )
    left, top = 42, page_h - 86
    c.setFont("Helvetica-Bold", 7.2)
    c.drawString(left, top, "benchmark")
    c.drawString(left + 220, top, "class")
    c.drawString(left + 420, top, "cov")
    c.drawString(left + 456, top, "score")
    c.drawString(left + 500, top, "blocker")
    c.drawString(left + 630, top, "missing")
    palette = {
        "trajectory_reanalysis_candidate": colors.HexColor("#2f855a"),
        "transport_heterogeneity_candidate": colors.HexColor("#2b6cb0"),
        "model_extension_required": colors.HexColor("#c05621"),
        "closure_horizon": colors.HexColor("#805ad5"),
        "scope_boundary": colors.HexColor("#718096"),
        "quantitative_inversion_candidate": colors.HexColor("#276749"),
        "structural_inversion_candidate": colors.HexColor("#319795"),
        "qualitative_horizon": colors.HexColor("#d69e2e"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 45
        klass = row["horizon_class"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["benchmark_id"].replace("_", " ")[:35])
        c.setFillColor(palette[klass])
        c.rect(left + 220, y - 4, 172, 12, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 225, y, klass.replace("_", " ")[:28])
        c.setFillColor(colors.black)
        c.drawString(left + 425, y, f"{float(row['effective_observable_coverage']):.2f}")
        c.drawString(left + 463, y, f"{float(row['frontier_priority_score']):.2f}")
        c.drawString(left + 500, y, row["primary_blocker"].replace("_", " ")[:24])
        c.drawString(left + 630, y, row["missing_observables"].replace("_", " ")[:38])
        c.drawString(left + 80, y - 10, f"source: {row['source_key'].replace('_', ' ')[:82]}")
    c.showPage()
    c.save()


def write_sota_source_provenance_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_source_provenance.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA source provenance gate")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "A citation is not promoted to an inversion input unless repository, raw-data, metadata, and reuse conditions are explicit.",
    )
    left, top = 48, page_h - 92
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(left, top, "source")
    c.drawString(left + 240, top, "stage")
    c.drawString(left + 455, top, "traj")
    c.drawString(left + 500, top, "raw")
    c.drawString(left + 540, top, "digitize")
    c.drawString(left + 600, top, "blocker")
    palette = {
        "trajectory_reanalysis_source": colors.HexColor("#2f855a"),
        "raw_curve_reanalysis_source": colors.HexColor("#2b6cb0"),
        "machine_readable_but_not_reanalysis_permitted": colors.HexColor("#d69e2e"),
        "machine_readable_source_incomplete_metadata": colors.HexColor("#c05621"),
        "citation_only_source": colors.HexColor("#718096"),
        "scope_boundary_source": colors.HexColor("#805ad5"),
    }
    c.setFont("Helvetica", 7.1)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 46
        stage = row["provenance_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["source_id"].replace("_", " ")[:36])
        c.setFillColor(palette[stage])
        c.rect(left + 240, y - 4, 185, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 246, y, stage.replace("_", " ")[:29])
        c.setFillColor(colors.black)
        c.drawString(left + 462, y, str(int(float(row["can_enter_trajectory_protocol"]))))
        c.drawString(left + 507, y, str(int(float(row["can_enter_raw_curve_protocol"]))))
        c.drawString(left + 550, y, str(int(float(row["requires_digitization"]))))
        c.drawString(left + 600, y, row["primary_blocker"].replace("_", " ")[:26])
        c.drawString(left + 80, y - 12, f"provenance: {row['provenance_items'].replace('_', ' ')[:94]}")
    c.showPage()
    c.save()


def write_sota_data_accession_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_data_accession.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA data accession manifest")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Remote archives are recorded with DOI, checksum, size, license, and local-cache status before reanalysis is claimed.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "accession")
    c.drawString(left + 220, top, "stage")
    c.drawString(left + 450, top, "GB")
    c.drawString(left + 490, top, "access")
    c.drawString(left + 540, top, "local")
    c.drawString(left + 585, top, "blocker")
    palette = {
        "remote_trajectory_accession_ready": colors.HexColor("#2f855a"),
        "remote_raw_curve_accession_ready": colors.HexColor("#2b6cb0"),
        "local_trajectory_cache_ready": colors.HexColor("#276749"),
        "local_raw_curve_cache_ready": colors.HexColor("#2c5282"),
        "metadata_incomplete_accession": colors.HexColor("#c05621"),
        "citation_only_no_accession": colors.HexColor("#718096"),
        "scope_boundary_accession": colors.HexColor("#805ad5"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 48
        stage = row["accession_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["accession_id"].replace("_", " ")[:34])
        c.setFillColor(palette[stage])
        c.rect(left + 220, y - 4, 190, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 226, y, stage.replace("_", " ")[:30])
        c.setFillColor(colors.black)
        c.drawString(left + 452, y, f"{float(row['archive_size_gb']):.2f}")
        c.drawString(left + 500, y, str(int(float(row["accession_ready"]))))
        c.drawString(left + 548, y, str(int(float(row["ready_for_local_reanalysis"]))))
        c.drawString(left + 585, y, row["primary_blocker"].replace("_", " ")[:28])
        c.drawString(left + 72, y - 12, f"doi: {row['doi']}; archive: {row['archive_name']}; md5: {row['archive_md5'][:32]}")
    c.showPage()
    c.save()


def write_sota_zenodo_record_fingerprint_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_zenodo_record_fingerprint.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA Zenodo record fingerprint")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "The cached GlassBench Zenodo API record verifies DOI, license, sizes, and md5 checksums before real reanalysis.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "fingerprint")
    c.drawString(left + 230, top, "stage")
    c.drawString(left + 438, top, "record")
    c.drawString(left + 485, top, "real")
    c.drawString(left + 525, top, "GB")
    c.drawString(left + 570, top, "blocker")
    palette = {
        "zenodo_record_verified": colors.HexColor("#2f855a"),
        "zenodo_record_mismatch": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 49
        stage = row["fingerprint_stage"]
        archive_gb = float(row["archive_size_bytes"]) / 1_000_000_000.0
        c.setFillColor(colors.black)
        c.drawString(left, y, row["fingerprint_id"].replace("_", " ")[:36])
        c.setFillColor(palette[stage])
        c.rect(left + 230, y - 4, 175, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 236, y, stage.replace("_", " ")[:28])
        c.setFillColor(colors.black)
        c.drawString(left + 450, y, str(int(float(row["zenodo_record_fingerprint_ready"]))))
        c.drawString(left + 495, y, str(int(float(row["real_reanalysis_ready"]))))
        c.drawString(left + 525, y, f"{archive_gb:.2f}")
        c.drawString(left + 570, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(
            left + 72,
            y - 12,
            f"doi: {row['doi']}; license: {row['license_id']}; archive md5: {row['archive_md5'][:32]}; README md5: {row['readme_md5'][:32]}",
        )
    c.showPage()
    c.save()


def write_sota_remote_zip_central_directory_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_remote_zip_central_directory.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA remote ZIP central directory")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "HTTP Range reads verify GlassBench ZIP64 roots and entry count without downloading the full archive payload.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "remote structure")
    c.drawString(left + 230, top, "stage")
    c.drawString(left + 462, top, "entries")
    c.drawString(left + 512, top, "roots")
    c.drawString(left + 558, top, "real")
    c.drawString(left + 598, top, "blocker")
    palette = {
        "remote_zip_structure_verified": colors.HexColor("#2f855a"),
        "remote_zip_structure_and_cache_ready": colors.HexColor("#276749"),
        "remote_zip_structure_incomplete": colors.HexColor("#c05621"),
        "remote_central_directory_missing": colors.HexColor("#c05621"),
        "remote_central_directory_empty": colors.HexColor("#c05621"),
        "remote_archive_size_mismatch": colors.HexColor("#c05621"),
        "remote_range_unavailable": colors.HexColor("#718096"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 53
        stage = row["remote_zip_structure_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["remote_structure_id"].replace("_", " ")[:36])
        c.setFillColor(palette[stage])
        c.rect(left + 230, y - 4, 198, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 236, y, stage.replace("_", " ")[:31])
        c.setFillColor(colors.black)
        c.drawString(left + 470, y, str(int(float(row["entry_count"]))))
        c.drawString(left + 520, y, f"{float(row['root_coverage']):.2f}")
        c.drawString(left + 568, y, str(int(float(row["real_reanalysis_ready"]))))
        c.drawString(left + 598, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(
            left + 72,
            y - 12,
            f"zip64={int(float(row['zip64']))}; range={int(float(row['range_supported']))}; cd={int(float(row['central_directory_size_bytes']))} bytes at {int(float(row['central_directory_offset']))}",
        )
        c.drawString(left + 72, y - 24, f"roots: {row['present_roots'].replace('_', ' ')[:96]}")
    c.showPage()
    c.save()


def write_sota_glassbench_payload_index_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_payload_index.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA GlassBench payload index")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Remote ZIP entries are mapped to system-level trajectory, model, and result payloads before real fitting is claimed.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "system")
    c.drawString(left + 60, top, "stage")
    c.drawString(left + 285, top, "traj")
    c.drawString(left + 328, top, "model")
    c.drawString(left + 376, top, "result")
    c.drawString(left + 430, top, "index")
    c.drawString(left + 472, top, "real")
    c.drawString(left + 515, top, "blocker")
    palette = {
        "remote_payload_index_verified": colors.HexColor("#2f855a"),
        "local_payload_index_ready": colors.HexColor("#276749"),
        "remote_payload_missing_trajectory": colors.HexColor("#c05621"),
        "remote_payload_missing_model": colors.HexColor("#c05621"),
        "remote_payload_missing_results": colors.HexColor("#c05621"),
        "remote_payload_temperature_mismatch": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 54
        stage = row["payload_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["system_id"])
        c.setFillColor(palette[stage])
        c.rect(left + 60, y - 4, 188, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 66, y, stage.replace("_", " ")[:30])
        c.setFillColor(colors.black)
        c.drawString(left + 295, y, str(int(float(row["trajectory_payload_count"]))))
        c.drawString(left + 342, y, str(int(float(row["model_payload_count"]))))
        c.drawString(left + 392, y, str(int(float(row["result_curve_count"]))))
        c.drawString(left + 440, y, str(int(float(row["payload_index_ready"]))))
        c.drawString(left + 482, y, str(int(float(row["real_reanalysis_ready"]))))
        c.drawString(left + 515, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(
            left + 60,
            y - 12,
            f"common all: {row['common_temperatures']}; model/result: {row['common_model_result_temperatures']}",
        )
        c.drawString(
            left + 60,
            y - 24,
            f"trajectory T: {row['trajectory_temperatures']}; model T: {row['model_temperatures']}; result T: {row['result_temperatures']}",
        )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_payload_locator_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_payload_locator.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA GlassBench trajectory payload locator")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Remote central-directory entries are resolved to concrete trajectory payload files before byte-range extraction or PE inversion is claimed.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "target")
    c.drawString(left + 75, top, "stage")
    c.drawString(left + 305, top, "located")
    c.drawString(left + 355, top, "range")
    c.drawString(left + 400, top, "real")
    c.drawString(left + 445, top, "blocker")
    palette = {
        "remote_trajectory_payload_located": colors.HexColor("#2f855a"),
        "remote_trajectory_payload_range_fetch_ready": colors.HexColor("#276749"),
        "local_trajectory_payload_available": colors.HexColor("#276749"),
        "remote_trajectory_payload_missing": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 52
        stage = row["locator_stage"]
        target = f"{row['system_id']} T={row['temperature']}"
        c.setFillColor(colors.black)
        c.drawString(left, y, target)
        c.setFillColor(palette[stage])
        c.rect(left + 75, y - 4, 205, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 81, y, stage.replace("_", " ")[:31])
        c.setFillColor(colors.black)
        c.drawString(left + 320, y, str(int(float(row["remote_payload_located"]))))
        c.drawString(left + 370, y, str(int(float(row["range_fetch_ready"]))))
        c.drawString(left + 412, y, str(int(float(row["real_reanalysis_ready"]))))
        c.drawString(left + 445, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(left + 75, y - 12, f"path: {row['source_path'][:95]}")
        c.drawString(
            left + 75,
            y - 24,
            f"format={row['payload_format']}; range-supported={int(float(row['range_supported']))}; entry-metadata={int(float(row['entry_metadata_ready']))}; full-cache={int(float(row['full_archive_cached']))}",
        )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_entry_metadata_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_entry_metadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA GlassBench trajectory entry metadata")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "ZIP central-directory and local-header range reads verify KA2D trajectory member ranges without downloading large members.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "target")
    c.drawString(left + 75, top, "stage")
    c.drawString(left + 325, top, "MB")
    c.drawString(left + 370, top, "meta")
    c.drawString(left + 415, top, "policy")
    c.drawString(left + 462, top, "extract")
    c.drawString(left + 515, top, "blocker")
    palette = {
        "trajectory_entry_metadata_ready_payload_size_blocked": colors.HexColor("#2b6cb0"),
        "trajectory_entry_metadata_ready_fetch_policy_ready": colors.HexColor("#2f855a"),
        "trajectory_entry_metadata_incomplete": colors.HexColor("#c05621"),
        "trajectory_entry_metadata_missing": colors.HexColor("#c05621"),
        "trajectory_payload_missing": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 52
        stage = row["metadata_stage"]
        target = f"{row['system_id']} T={row['temperature']}"
        size_mb = float(row["compressed_size_bytes"]) / 1_000_000.0
        c.setFillColor(colors.black)
        c.drawString(left, y, target)
        c.setFillColor(palette[stage])
        c.rect(left + 75, y - 4, 225, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 81, y, stage.replace("_", " ")[:35])
        c.setFillColor(colors.black)
        c.drawString(left + 325, y, f"{size_mb:.1f}")
        c.drawString(left + 382, y, str(int(float(row["entry_metadata_ready"]))))
        c.drawString(left + 430, y, str(int(float(row["full_member_fetch_within_policy"]))))
        c.drawString(left + 480, y, str(int(float(row["trajectory_extraction_ready"]))))
        c.drawString(left + 515, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(left + 75, y - 12, f"path: {row['source_path'][:95]}")
        c.drawString(
            left + 75,
            y - 24,
            f"method={row['compression_method']}; local-header={int(float(row['local_header_offset']))}; data-range={int(float(row['compressed_data_range_start']))}-{int(float(row['compressed_data_range_end']))}",
        )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_member_stream_probe_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_member_stream_probe.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA GlassBench trajectory member stream probe")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Small compressed-member range reads are raw-deflate inflated to verify tar.xz prefixes without downloading full members.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "target")
    c.drawString(left + 75, top, "stage")
    c.drawString(left + 350, top, "probe")
    c.drawString(left + 405, top, "stream")
    c.drawString(left + 455, top, "xz")
    c.drawString(left + 495, top, "extract")
    c.drawString(left + 545, top, "blocker")
    palette = {
        "trajectory_member_prefix_verified_streaming_extraction_blocked": colors.HexColor("#2b6cb0"),
        "trajectory_member_stream_probe_missing": colors.HexColor("#c05621"),
        "trajectory_member_prefix_probe_failed": colors.HexColor("#c05621"),
        "trajectory_entry_metadata_incomplete": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 52
        stage = row["probe_stage"]
        target = f"{row['system_id']} T={row['temperature']}"
        c.setFillColor(colors.black)
        c.drawString(left, y, target)
        c.setFillColor(palette[stage])
        c.rect(left + 75, y - 4, 250, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 81, y, stage.replace("_", " ")[:39])
        c.setFillColor(colors.black)
        c.drawString(left + 360, y, str(int(float(row["compressed_probe_bytes"]))))
        c.drawString(left + 420, y, str(int(float(row["stream_inflate_ready"]))))
        c.drawString(left + 465, y, str(int(float(row["xz_magic_verified"]))))
        c.drawString(left + 512, y, str(int(float(row["trajectory_extraction_ready"]))))
        c.drawString(left + 545, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(left + 75, y - 12, f"path: {row['source_path'][:95]}")
        c.drawString(
            left + 75,
            y - 24,
            f"md5={row['compressed_probe_md5']}; inflated-prefix={int(float(row['inflated_prefix_bytes']))}; hex={row['inflated_prefix_hex'][:24]}",
        )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_inner_tar_header_probe_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA GlassBench inner tar header probe")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "ZIP-member prefixes are streamed through raw deflate and XZ to verify tar directories and NPZ trajectory headers.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "target")
    c.drawString(left + 75, top, "stage")
    c.drawString(left + 335, top, "tar")
    c.drawString(left + 380, top, "npz")
    c.drawString(left + 425, top, "layout")
    c.drawString(left + 475, top, "extract")
    c.drawString(left + 525, top, "blocker")
    palette = {
        "trajectory_inner_tar_layout_verified_extraction_blocked": colors.HexColor("#2b6cb0"),
        "trajectory_inner_tar_header_probe_missing": colors.HexColor("#c05621"),
        "trajectory_inner_tar_header_probe_failed": colors.HexColor("#c05621"),
        "trajectory_member_prefix_incomplete": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 52
        stage = row["tar_probe_stage"]
        target = f"{row['system_id']} T={row['temperature']}"
        c.setFillColor(colors.black)
        c.drawString(left, y, target)
        c.setFillColor(palette[stage])
        c.rect(left + 75, y - 4, 238, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 81, y, stage.replace("_", " ")[:37])
        c.setFillColor(colors.black)
        c.drawString(left + 345, y, str(int(float(row["tar_magic_verified"]))))
        c.drawString(left + 392, y, str(int(float(row["npz_member_header_verified"]))))
        c.drawString(left + 445, y, str(int(float(row["trajectory_layout_ready"]))))
        c.drawString(left + 492, y, str(int(float(row["trajectory_extraction_ready"]))))
        c.drawString(left + 525, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(left + 75, y - 12, f"root={row['root_directory']}; first NPZ={row['first_npz_member'][:88]}")
        c.drawString(
            left + 75,
            y - 24,
            f"probe={int(float(row['compressed_probe_bytes']))}; tar={int(float(row['tar_probe_bytes']))}; npz-count={int(float(row['npz_member_count_in_probe']))}; splits={row['split_labels_in_probe']}",
        )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_npz_schema_probe_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_npz_schema_probe.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA GlassBench NPZ trajectory schema probe")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "First NPZ members are opened from streamed trajectory prefixes to verify coordinate-array schemas.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "target")
    c.drawString(left + 75, top, "stage")
    c.drawString(left + 330, top, "schema")
    c.drawString(left + 380, top, "coords")
    c.drawString(left + 430, top, "shape")
    c.drawString(left + 505, top, "extract")
    c.drawString(left + 555, top, "blocker")
    palette = {
        "trajectory_npz_coordinate_schema_verified": colors.HexColor("#2b6cb0"),
        "trajectory_npz_schema_probe_missing": colors.HexColor("#c05621"),
        "trajectory_npz_schema_probe_failed": colors.HexColor("#c05621"),
        "trajectory_inner_tar_layout_incomplete": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 52
        stage = row["schema_probe_stage"]
        target = f"{row['system_id']} T={row['temperature']}"
        shape = (
            f"{int(float(row['frame_count']))}x"
            f"{int(float(row['particle_count']))}x"
            f"{int(float(row['spatial_dimension']))}"
        )
        c.setFillColor(colors.black)
        c.drawString(left, y, target)
        c.setFillColor(palette[stage])
        c.rect(left + 75, y - 4, 232, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 81, y, stage.replace("_", " ")[:36])
        c.setFillColor(colors.black)
        c.drawString(left + 345, y, str(int(float(row["npz_schema_ready"]))))
        c.drawString(left + 397, y, str(int(float(row["coordinate_array_ready"]))))
        c.drawString(left + 430, y, shape)
        c.drawString(left + 522, y, str(int(float(row["trajectory_extraction_ready"]))))
        c.drawString(left + 555, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(left + 75, y - 12, f"member={row['first_npz_member'][:95]}")
        c.drawString(
            left + 75,
            y - 24,
            f"bytes={int(float(row['npz_member_bytes']))}; md5={row['npz_member_md5']}; arrays={row['array_names'][:70]}",
        )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_first_npz_observable_smoke_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA GlassBench first-NPZ observable smoke")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Single streamed NPZ members are reduced to minimal-image frame-index MSD and 2D NGP summaries.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "target")
    c.drawString(left + 75, top, "stage")
    c.drawString(left + 342, top, "MSD")
    c.drawString(left + 392, top, "NGP")
    c.drawString(left + 442, top, "peak")
    c.drawString(left + 492, top, "extract")
    c.drawString(left + 540, top, "blocker")
    palette = {
        "first_npz_msd_ngp_smoke_ready_reanalysis_blocked": colors.HexColor("#2b6cb0"),
        "first_npz_observable_smoke_missing": colors.HexColor("#c05621"),
        "first_npz_observable_smoke_failed": colors.HexColor("#c05621"),
        "trajectory_npz_schema_incomplete": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 52
        stage = row["smoke_stage"]
        target = f"{row['system_id']} T={row['temperature']}"
        c.setFillColor(colors.black)
        c.drawString(left, y, target)
        c.setFillColor(palette[stage])
        c.rect(left + 75, y - 4, 246, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 81, y, stage.replace("_", " ")[:38])
        c.setFillColor(colors.black)
        c.drawString(left + 342, y, f"{float(row['final_msd']):.3g}")
        c.drawString(left + 392, y, f"{float(row['final_ngp_2d']):.3g}")
        c.drawString(left + 442, y, f"{float(row['peak_ngp_2d']):.3g}")
        c.drawString(left + 510, y, str(int(float(row["trajectory_extraction_ready"]))))
        c.drawString(left + 540, y, row["primary_blocker"].replace("_", " ")[:32])
        c.drawString(left + 75, y - 12, f"member={row['first_npz_member'][:96]}")
        c.drawString(
            left + 75,
            y - 24,
            f"method={row['observable_method']}; positions={row['positions_md5']}; final frame={int(float(row['final_frame_index']))}; peak frame={int(float(row['peak_ngp_frame_index']))}",
        )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_first_npz_observable_curve_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.csv").open() as f:
        rows = list(csv.DictReader(f))
    ready_rows = [row for row in rows if float(row["observable_curve_ready"]) > 0.5]
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA GlassBench first-NPZ observable curves")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Frame-index minimal-image MSD and 2D NGP curves from one streamed KA2D NPZ member per temperature.",
    )
    left, bottom = 54, 88
    width, height = 310, 250
    gap = 84
    max_frame = max([float(row["frame_index"]) for row in ready_rows] + [1.0])
    max_msd = max([float(row["msd"]) for row in ready_rows] + [1.0])
    max_ngp = max([float(row["ngp_2d"]) for row in ready_rows] + [1.0])
    palette = {"0.23": colors.HexColor("#2b6cb0"), "0.30": colors.HexColor("#c05621")}

    def draw_curve_panel(x0: float, title: str, value_key: str, ymax: float) -> None:
        c.setStrokeColor(colors.HexColor("#cbd5e0"))
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.rect(x0, bottom, width, height, stroke=1, fill=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x0, bottom + height + 15, title)
        c.setFont("Helvetica", 7)
        c.drawString(x0, bottom - 18, "frame index")
        c.drawString(x0, bottom + height + 3, f"max={ymax:.3g}")
        for temp, color in palette.items():
            subset = [row for row in ready_rows if row["temperature"] == temp]
            subset.sort(key=lambda row: float(row["frame_index"]))
            if len(subset) < 2:
                continue
            c.setStrokeColor(color)
            c.setLineWidth(1.8)
            points = [
                (
                    x0 + float(row["frame_index"]) / max_frame * width,
                    bottom + float(row[value_key]) / ymax * height,
                )
                for row in subset
            ]
            path_obj = c.beginPath()
            path_obj.moveTo(points[0][0], points[0][1])
            for x, y in points[1:]:
                path_obj.lineTo(x, y)
            c.drawPath(path_obj)

    draw_curve_panel(left, "MSD curve", "msd", max_msd)
    draw_curve_panel(left + width + gap, "2D NGP curve", "ngp_2d", max_ngp)
    c.setFont("Helvetica", 8)
    c.setFillColor(palette["0.23"])
    c.rect(54, 38, 9, 9, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(68, 38, "T=0.23 first NPZ")
    c.setFillColor(palette["0.30"])
    c.rect(175, 38, 9, 9, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(189, 38, "T=0.30 first NPZ")
    c.drawString(322, 38, "single-member frame-index curves; no physical-time or uncertainty-weighted inversion")
    c.showPage()
    c.save()


def write_sota_glassbench_short_window_trend_canary_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_short_window_trend_canary.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench short-window real-data canary")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Real first-NPZ frame-index curves are allowed to support a short-window MSD/NGP sanity check only.",
    )
    left, top = 48, page_h - 96
    colors_by_stage = {
        "short_window_real_data_canary_ready_inversion_blocked": colors.HexColor("#2b6cb0"),
        "short_window_trend_canary_failed": colors.HexColor("#c05621"),
        "short_window_trend_canary_incomplete": colors.HexColor("#b7791f"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "comparison")
    c.drawString(left + 120, top + 18, "stage")
    c.drawString(left + 390, top + 18, "canary values")
    for index, row in enumerate(rows):
        y = top - index * 58
        stage = row["canary_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} {row["cold_temperature"]}->{row["hot_temperature"]}')
        c.setFillColor(color)
        c.rect(left + 120, y - 12, 250, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 128, y - 3, stage.replace("_", " ")[:43])
        c.setFillColor(colors.black)
        c.drawString(
            left + 390,
            y,
            "ready={}; inversion={}; thermodynamic={}; hot/cold MSD={:.3g}".format(
                int(float(row["short_window_real_data_canary_ready"])),
                int(float(row["sota_inversion_ready"])),
                int(float(row["thermodynamic_claim_allowed"])),
                float(row["hot_to_cold_final_msd_ratio"]),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 390,
            y - 14,
            "peak NGP cold/hot=({:.3g}, {:.3g}); frames={}; blocker={}".format(
                float(row["cold_peak_ngp_2d"]),
                float(row["hot_peak_ngp_2d"]),
                int(float(row["common_frame_count"])),
                row["primary_blocker"].replace("_", " ")[:40],
            ),
        )
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "This canary does not supply lag-time units, multi-member uncertainties, Fs(k,t), chi4, alpha relaxation, or thermodynamic evidence.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_timebase_bridge_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_timebase_bridge.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench trajectory-result timebase bridge")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Result time grids are gated before being attached to first-NPZ trajectory frame-index curves.",
    )
    left, top = 48, page_h - 92
    row_h = 58
    colors_by_stage = {
        "trajectory_timebase_ready_observable_inversion_blocked": colors.HexColor("#2f855a"),
        "trajectory_result_timebase_length_mismatch": colors.HexColor("#c05621"),
        "trajectory_result_timebase_mapping_required": colors.HexColor("#b7791f"),
        "trajectory_result_timebase_missing": colors.HexColor("#805ad5"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 100, top + 18, "stage")
    c.drawString(left + 370, top + 18, "timebase checks")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["timebase_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 12, 245, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:42])
        c.setFillColor(colors.black)
        c.drawString(
            left + 370,
            y,
            "ready={}; frames={}; time points={}; match={}; explicit mapping={}".format(
                int(float(row["trajectory_timebase_ready"])),
                int(float(row["frame_count"])),
                int(float(row["time_point_count"])),
                int(float(row["frame_time_point_count_match"])),
                int(float(row["explicit_frame_time_mapping"])),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 370,
            y - 14,
            f'time grid: {row["time_grid_path"][:70]}; next: {row["next_required_action"].replace("_", " ")[:45]}',
        )
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "Physical-time promotion requires same-temperature time grids, matching counts, and an explicit frame-time mapping; full inversion remains a later gate.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_frame_time_mapping_audit_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_frame_time_mapping_audit.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench frame-time mapping audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Candidate mappings are classified before attaching result time grids to trajectory frame-index curves.",
    )
    left, top = 48, page_h - 92
    row_h = 64
    colors_by_stage = {
        "frame_time_mapping_ready": colors.HexColor("#2f855a"),
        "subsample_mapping_ready": colors.HexColor("#2f855a"),
        "count_matched_mapping_metadata_required": colors.HexColor("#2b6cb0"),
        "integer_stride_mapping_metadata_required": colors.HexColor("#b7791f"),
        "ambiguous_frame_time_mapping": colors.HexColor("#c05621"),
        "frame_time_mapping_missing": colors.HexColor("#4a5568"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 100, top + 18, "mapping stage")
    c.drawString(left + 370, top + 18, "candidate checks")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["mapping_audit_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 12, 245, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:42])
        c.setFillColor(colors.black)
        c.drawString(
            left + 370,
            y,
            "ready={}; exact={}; stride={}; interpolation={}; ratio={:.3g}".format(
                int(float(row["publishable_frame_time_mapping_ready"])),
                int(float(row["exact_count_match"])),
                int(float(row["integer_stride_subsample_candidate"])),
                int(float(row["endpoint_interpolation_candidate"])),
                float(row["frame_to_result_stride_ratio"]),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 370,
            y - 14,
            f'accepted: {row["accepted_mapping_class"].replace("_", " ")}; provisional: {row["provisional_mapping_class"].replace("_", " ")[:55]}',
        )
        c.drawString(
            left + 370,
            y - 27,
            f'metadata: {row["minimum_required_metadata"].replace("_", " ")[:80]}',
        )
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "Endpoint interpolation alone is kept as provisional and cannot support a publishable trajectory-time inversion.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_real_inversion_gap_ledger_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_real_inversion_gap_ledger.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench real-data inversion gap ledger")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "The allowed manuscript claim is the minimum over canary, timebase, ensemble, observable, semantics, and uncertainty gates.",
    )
    left, top = 48, page_h - 92
    row_h = 62
    colors_by_stage = {
        "real_data_quantitative_inversion_ready": colors.HexColor("#2f855a"),
        "real_data_canary_timebase_blocked": colors.HexColor("#c05621"),
        "real_data_canary_ensemble_blocked": colors.HexColor("#b7791f"),
        "real_data_canary_observable_set_blocked": colors.HexColor("#805ad5"),
        "real_data_canary_uncertainty_blocked": colors.HexColor("#2b6cb0"),
        "real_data_coordinate_canary_missing": colors.HexColor("#4a5568"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 100, top + 18, "stage")
    c.drawString(left + 370, top + 18, "claim and blockers")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["ledger_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 12, 245, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:42])
        c.setFillColor(colors.black)
        c.drawString(
            left + 370,
            y,
            "claim={}; ready={}; blocker={}".format(
                row["allowed_claim_level"].replace("_", " ")[:42],
                int(float(row["quantitative_real_inversion_ready"])),
                row["primary_blocker"].replace("_", " "),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(left + 370, y - 14, f'next: {row["next_required_actions"].replace("_", " ")[:90]}')
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "Current GlassBench rows allow a short-window coordinate trend only; they do not support full persistence/exchange inversion or thermodynamic claims.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_real_inversion_unlock_protocol_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_real_inversion_unlock_protocol.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench real-inversion unlock protocol")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Minimum additional payload required before promoting real-data canaries to uncertainty-weighted trajectory inversion.",
    )
    left, top = 48, page_h - 92
    row_h = 64
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 100, top + 18, "unlock stage")
    c.drawString(left + 360, top + 18, "minimum missing payload")
    for index, row in enumerate(rows):
        y = top - index * row_h
        ready = bool(float(row["minimum_unlock_ready"]))
        color = colors.HexColor("#2f855a") if ready else colors.HexColor("#c05621")
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 12, 235, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, row["unlock_stage"].replace("_", " ")[:40])
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7)
        c.drawString(
            left + 360,
            y,
            "frames={}; result times={}; mapping={}; extra members={}".format(
                int(float(row["frame_count"])),
                int(float(row["time_point_count"])),
                int(float(row["frame_time_mapping_present"])),
                int(math.ceil(float(row["additional_member_count_needed"]))),
            ),
        )
        c.setFont("Helvetica", 6.6)
        c.drawString(
            left + 360,
            y - 14,
            f'payload: {row["minimum_required_payload"].replace("_", " ")[:95]}',
        )
        c.drawString(
            left + 360,
            y - 27,
            f'after unlock: {row["post_unlock_claim_level"].replace("_", " ")}',
        )
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "This is a data-acquisition and observable-computation protocol, not a thermodynamic glass-transition claim.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_first_npz_inversion_readiness_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench first-NPZ SOTA inversion readiness")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Frame-index curves are gated before physical-time, ensemble, and uncertainty-weighted comparisons.",
    )
    left, top = 48, page_h - 94
    row_h = 64
    colors_by_stage = {
        "uncertainty_weighted_sota_inversion_ready": colors.HexColor("#2f855a"),
        "frame_index_curve_only": colors.HexColor("#2b6cb0"),
        "single_member_curve_only": colors.HexColor("#805ad5"),
        "structural_curve_without_uncertainty": colors.HexColor("#b7791f"),
        "upstream_curve_incomplete": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 20, "target")
    c.drawString(left + 92, top + 20, "stage")
    c.drawString(left + 335, top + 20, "requirements and next action")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["readiness_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.black)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 92, y - 12, 225, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 100, y - 3, stage.replace("_", " ")[:42])
        c.setFillColor(colors.black)
        c.drawString(
            left + 335,
            y,
            "ready={}; physical={}; ensemble={}; uncertainty={}; frames={}; members={}".format(
                int(float(row["sota_inversion_ready"])),
                int(float(row["physical_time_ready"])),
                int(float(row["ensemble_ready"])),
                int(float(row["uncertainty_ready"])),
                int(float(row["frame_count"])),
                int(float(row["member_count"])),
            ),
        )
        c.setFont("Helvetica", 6.7)
        c.drawString(
            left + 335,
            y - 14,
            f'missing obs: {row["missing_observables"].replace("_", " ")[:85]}',
        )
        c.drawString(
            left + 335,
            y - 27,
            f'next: {row["next_required_action"].replace("_", " ")[:92]}',
        )
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "No row is promoted to a real SOTA inversion until lag-time, ensemble members, Fs/chi4 observables, and positive uncertainties are present.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_npz_ensemble_horizon_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench NPZ ensemble-member horizon")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Inner-tar prefix probes count visible NPZ members before multi-member extraction or real reanalysis.",
    )
    left, top = 48, page_h - 92
    row_h = 58
    colors_by_stage = {
        "member_index_horizon_ready_extraction_blocked": colors.HexColor("#2b6cb0"),
        "prefix_member_horizon_ready_extraction_blocked": colors.HexColor("#2b6cb0"),
        "prefix_member_horizon_short": colors.HexColor("#b7791f"),
        "trajectory_layout_incomplete": colors.HexColor("#c05621"),
        "sota_inversion_already_ready": colors.HexColor("#2f855a"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 92, top + 18, "stage")
    c.drawString(left + 335, top + 18, "member-count evidence")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["horizon_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(colors.black)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 92, y - 12, 225, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 100, y - 3, stage.replace("_", " ")[:42])
        c.setFillColor(colors.black)
        c.drawString(
            left + 335,
            y,
            "prefix members={}; extracted curves={}; threshold={}; gap={}; extraction={}".format(
                int(float(row["prefix_npz_member_count"])),
                int(float(row["extracted_curve_member_count"])),
                int(float(row["min_member_count"])),
                int(float(row["member_count_gap_to_threshold"])),
                int(float(row["multi_npz_extraction_ready"])),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 335,
            y - 14,
            f'split={row["split_labels_in_probe"]}; tar probe={int(float(row["tar_probe_bytes"]))} bytes; first={row["first_npz_member"][:84]}',
        )
        c.drawString(left + 335, y - 27, f'next={row["next_required_action"].replace("_", " ")[:96]}')
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "The horizon counts headers visible in the prefix only; it does not claim multi-NPZ extraction, uncertainties, or real inversion.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_npz_member_index_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_npz_member_index.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench extended NPZ member index")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Extended range probes index enough trajectory members for the ensemble threshold; multi-member extraction remains pending.",
    )
    left, top = 48, page_h - 92
    row_h = 62
    colors_by_stage = {
        "member_index_threshold_ready_extraction_pending": colors.HexColor("#2b6cb0"),
        "member_index_threshold_short": colors.HexColor("#b7791f"),
        "member_index_missing": colors.HexColor("#c05621"),
        "trajectory_layout_incomplete": colors.HexColor("#4a5568"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 92, top + 18, "index stage")
    c.drawString(left + 365, top + 18, "member-list evidence")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["member_index_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 92, y - 12, 250, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 100, y - 3, stage.replace("_", " ")[:44])
        c.setFillColor(colors.black)
        c.drawString(
            left + 365,
            y,
            "indexed members={}/{}; threshold pass={}; split={}".format(
                int(float(row["indexed_npz_member_count"])),
                int(float(row["required_member_count"])),
                int(float(row["member_count_threshold_pass"])),
                row["split_labels_in_index"],
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 365,
            y - 14,
            f'first four={row["first_four_member_ids"][:100]}',
        )
        c.drawString(left + 365, y - 27, f'next={row["next_required_action"].replace("_", " ")[:96]}')
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "This is member-list evidence only; physical lag times and uncertainty-weighted observable extraction are still required.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_trajectory_member_ensemble_observable_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench four-member frame-index observables")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "The first four indexed KA2D trajectory members are aggregated into frame-index observable means and standard errors.",
    )
    left, top = 48, page_h - 92
    row_h = 62
    colors_by_stage = {
        "frame_index_member_ensemble_uncertainty_ready": colors.HexColor("#2b6cb0"),
        "member_ensemble_below_threshold": colors.HexColor("#b7791f"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 92, top + 18, "uncertainty stage")
    c.drawString(left + 370, top + 18, "frame-index observables")
    frame_one_rows = [row for row in rows if int(float(row["frame_index"])) == 1]
    for index, row in enumerate(frame_one_rows):
        y = top - index * row_h
        stage = row["ensemble_observable_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 92, y - 12, 250, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 100, y - 3, stage.replace("_", " ")[:44])
        c.setFillColor(colors.black)
        c.drawString(
            left + 370,
            y,
            "members={}; MSD={:.5g} +/- {:.2g}; chi4={:.2f} +/- {:.2f}".format(
                int(float(row["member_count"])),
                float(row["msd"]),
                float(row["sigma_msd"]),
                float(row["chi4_overlap"]),
                float(row["sigma_chi4_overlap"]),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 370,
            y - 14,
            f'frame={int(float(row["frame_index"]))}; Fs(k)={row["self_intermediate_scattering_by_k"][:56]}',
        )
        c.drawString(left + 370, y - 27, f'blocker={row["primary_blocker"]}; next={row["next_required_action"]}')
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "This is a frame-index uncertainty gate only; physical lag times are still required before real inversion.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_ka2d_timecode_semantics_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_ka2d_timecode_semantics.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench KA2D time-code semantics")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Official trajectory README maps tc file codes to lag times and identifies positions[20] as isoconfigurational replicas.",
    )
    left, top = 48, page_h - 92
    row_h = 42
    colors_by_stage = {
        "physical_timecode_semantics_ready_sparse_coverage": colors.HexColor("#2b6cb0"),
        "physical_timecode_semantics_ready_member_uncertainty_short": colors.HexColor("#b7791f"),
        "physical_timecode_curve_ready": colors.HexColor("#2f855a"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 104, top + 18, "semantic stage")
    c.drawString(left + 370, top + 18, "corrected fixed-time observables")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["timecode_semantics_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]} {row["time_code"]}')
        c.setFillColor(color)
        c.rect(left + 104, y - 12, 246, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 112, y - 3, stage.replace("_", " ")[:43])
        c.setFillColor(colors.black)
        c.drawString(
            left + 370,
            y,
            "t={:g}; tau_alpha={:g}; members={}; time codes={}/{}".format(
                float(row["lag_time"]),
                float(row["tau_alpha"]),
                int(float(row["member_count"])),
                int(float(row["available_time_code_count"])),
                int(float(row["required_time_code_count"])),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 370,
            y - 14,
            "MSD={:.5g}; chi4(replica)={:.3g}; axis0 replica={}; frame-axis time={}".format(
                float(row["msd"]),
                float(row["chi4_overlap_replica"]),
                int(float(row["axis0_is_isoconfigurational_replica"])),
                int(float(row["frame_axis_is_physical_time"])),
            ),
        )
        c.drawString(left + 370, y - 27, f'blocker={row["primary_blocker"]}; next={row["next_required_action"][:54]}')
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "This corrects the trajectory-axis semantics; full time-code coverage is still required before real inversion.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_timecode_curve_bridge_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_timecode_curve_bridge.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench time-code curve bridge")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Corrected KA2D physical-time rows are translated into the trajectory PE pre-inversion schema without promoting blockers.",
    )
    left, top = 48, page_h - 98
    row_h = 54
    colors_by_stage = {
        "glassbench_timecode_curve_bridge_ready": colors.HexColor("#2f855a"),
        "glassbench_timecode_curve_bridge_incomplete": colors.HexColor("#c05621"),
        "glassbench_timecode_curve_upstream_incomplete": colors.HexColor("#2b6cb0"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 22, "target")
    c.drawString(left + 100, top + 22, "bridge stage")
    c.drawString(left + 345, top + 22, "real-data inversion status")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["bridge_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 13, 228, 25, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:39])
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7.5)
        c.drawString(
            left + 345,
            y,
            "lags={}; tc curve={}; real curve={}; PE inversion={}; blocker={}".format(
                int(float(row["lag_count"])),
                int(float(row["timecode_curve_ready"])),
                int(float(row["real_time_observable_curve_ready"])),
                int(float(row["real_pe_inversion_ready"])),
                row["primary_blocker"],
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 345,
            y - 14,
            "latest t/tau_alpha={:.3g}; anchor Fs={:.3g}; D-window={}; thermodynamic claim={}".format(
                float(row["latest_lag_time_over_tau_alpha"]),
                float(row["latest_self_intermediate_scattering_anchor"]),
                int(float(row["diffusion_asymptote_window_ready"])),
                int(float(row["thermodynamic_claim_allowed"])),
            ),
        )
        c.drawString(left + 345, y - 27, f'next={row["next_required_action"][:70]}')
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "This promotes one real physical-time GlassBench curve to a quantitative pre-inversion blocker, not to a completed real-data fit.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_timecode_signature_support_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_timecode_signature_support.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench time-code signature support")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Real KA2D time-code curves are scored for dynamical signatures before persistence/exchange inversion is claimed.",
    )
    left, top = 48, page_h - 100
    row_h = 62
    colors_by_stage = {
        "real_curve_dynamic_signature_support_and_inversion_ready": colors.HexColor("#2f855a"),
        "real_curve_dynamic_signature_support_preinversion": colors.HexColor("#b7791f"),
        "timecode_curve_upstream_incomplete": colors.HexColor("#2b6cb0"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 24, "target")
    c.drawString(left + 100, top + 24, "signature stage")
    c.drawString(left + 360, top + 24, "real-curve dynamical support")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["signature_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 13, 244, 25, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:43])
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7.5)
        c.drawString(
            left + 360,
            y,
            "supported={}/4; real curve={}; PE inversion={}; blocker={}".format(
                int(float(row["supported_dynamical_signature_count"])),
                int(float(row["real_time_observable_curve_ready"])),
                int(float(row["real_pe_inversion_ready"])),
                row["primary_blocker"],
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 360,
            y - 14,
            "MSD growth={:.3g}; Fs decay={:.3g}; alpha crossed={}".format(
                float(row["msd_growth_factor"]),
                float(row["self_intermediate_decay"]),
                int(float(row["alpha_threshold_crossed"])),
            ),
        )
        c.drawString(
            left + 360,
            y - 27,
            "NGP peak t={:.4g}, recovery={:.2f}; chi4 peak t={:.4g}, recovery={:.2f}".format(
                float(row["ngp_peak_time"]),
                float(row["ngp_late_recovery_fraction"]),
                float(row["chi4_peak_time"]),
                float(row["chi4_late_recovery_fraction"]),
            ),
        )
        c.drawString(left + 360, y - 40, f'next={row["next_required_action"][:66]}')
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "The score supports dynamical glass signatures only; thermodynamic glass-transition claims remain disallowed.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_alpha_threshold_horizon_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_alpha_threshold_horizon.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench alpha-threshold horizon audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Tau-alpha metadata is checked against the anchor Fs=e^-1 crossing used by the persistence/exchange inversion.",
    )
    left, top = 48, page_h - 100
    row_h = 62
    colors_by_stage = {
        "alpha_threshold_horizon_inversion_ready": colors.HexColor("#2f855a"),
        "alpha_threshold_crossed_preinversion": colors.HexColor("#b7791f"),
        "alpha_threshold_not_yet_reached": colors.HexColor("#c05621"),
        "metadata_tau_alpha_anchor_fs_mismatch": colors.HexColor("#9f1239"),
        "timecode_curve_upstream_incomplete": colors.HexColor("#2b6cb0"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 24, "target")
    c.drawString(left + 100, top + 24, "audit stage")
    c.drawString(left + 365, top + 24, "threshold and inversion status")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["audit_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 13, 250, 25, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:44])
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7.5)
        c.drawString(
            left + 365,
            y,
            "tau reached={}; alpha crossed={}; consistent={}; blocker={}".format(
                int(float(row["metadata_tau_alpha_reached"])),
                int(float(row["alpha_threshold_crossed"])),
                int(float(row["metadata_tau_alpha_consistent_with_anchor_fs"])),
                row["primary_blocker"],
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 365,
            y - 14,
            "latest t/tau_alpha(meta)={:.3g}; anchor Fs={:.3g}; extension={:.3g}x".format(
                float(row["latest_lag_time_over_tau_alpha_metadata"]),
                float(row["latest_self_intermediate_scattering_anchor"]),
                float(row["estimated_lag_extension_factor"]),
            ),
        )
        c.drawString(
            left + 365,
            y - 27,
            "PE inversion={}; thermodynamic claim={}; next={}".format(
                int(float(row["real_pe_inversion_ready"])),
                int(float(row["thermodynamic_claim_allowed"])),
                row["next_required_action"][:58],
            ),
        )
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "A metadata/anchor-Fs mismatch blocks real-data inversion until the alpha definition or longer horizon is resolved.",
    )
    c.showPage()
    c.save()


def write_sota_dynamic_signature_alignment_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_dynamic_signature_alignment.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA dynamic-signature alignment")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Model diagnostics, literature claims, and current GlassBench evidence are aligned without promoting thermodynamic or fit claims.",
    )
    left, top = 48, page_h - 92
    row_h = 45
    colors_by_stage = {
        "real_curve_supported": colors.HexColor("#2f855a"),
        "real_curve_supported_pre_alpha_threshold": colors.HexColor("#b7791f"),
        "real_proxy_supported_spatial_boundary": colors.HexColor("#805ad5"),
        "model_literature_supported_real_inversion_blocked": colors.HexColor("#c05621"),
        "scope_boundary_not_explained": colors.HexColor("#4a5568"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "signature")
    c.drawString(left + 170, top + 18, "alignment stage")
    c.drawString(left + 420, top + 18, "support and blocker")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["alignment_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#2b6cb0"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 7.8)
        c.drawString(left, y, row["signature"].replace("_", " ")[:30])
        c.setFillColor(color)
        c.rect(left + 170, y - 12, 232, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6.8)
        c.drawString(left + 178, y - 3, stage.replace("_", " ")[:40])
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7.2)
        c.drawString(
            left + 420,
            y,
            "model={:.1g}; literature={}; real={}; inversion={}; thermo={}".format(
                float(row["model_support"]),
                int(float(row["literature_qualitative_support"])),
                int(float(row["real_glassbench_support"])),
                int(float(row["real_quantitative_inversion_ready"])),
                int(float(row["thermodynamic_claim_allowed"])),
            ),
        )
        c.setFont("Helvetica", 6.7)
        c.drawString(left + 420, y - 14, f'phenomenon={row["phenomenon"].replace("_", " ")[:64]}')
        c.drawString(left + 420, y - 27, f'blocker={row["primary_blocker"].replace("_", " ")[:64]}')
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "Rows with real support are dynamical signatures only; persistence/exchange inversion and thermodynamic claims remain separately gated.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_microdynamic_closed_loop_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_microdynamic_closed_loop.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench microdynamic closed-loop audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Real frame-index microstatistics, real macro dynamical signatures, and missing cage-jump clock inputs are separated.",
    )
    left, top = 48, page_h - 100
    row_h = 62
    colors_by_stage = {
        "real_microdynamic_closed_loop_ready": colors.HexColor("#2f855a"),
        "real_microstats_macro_signatures_closed_loop_blocked": colors.HexColor("#9f1239"),
        "real_microstats_macro_signature_incomplete": colors.HexColor("#c05621"),
        "macro_timecode_upstream_incomplete": colors.HexColor("#2b6cb0"),
        "trajectory_microstatistics_upstream_incomplete": colors.HexColor("#4a5568"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 24, "target")
    c.drawString(left + 100, top + 24, "closed-loop stage")
    c.drawString(left + 370, top + 24, "micro-to-macro evidence status")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["closed_loop_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 13, 252, 25, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:44])
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7.5)
        c.drawString(
            left + 370,
            y,
            "frame microstats={}; macro signatures={}; prediction={}; blocker={}".format(
                int(float(row["frame_index_microstats_ready"])),
                int(float(row["macro_signature_ready"])),
                int(float(row["micro_to_macro_prediction_ready"])),
                row["primary_blocker"],
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 370,
            y - 14,
            "cage proxy={:.3g}; short NGP peak={:.3g}; short Fs decay={:.3g}; signatures={:.0f}".format(
                float(row["cage_length_proxy"]),
                float(row["short_frame_ngp_peak"]),
                float(row["short_frame_fs_decay"]),
                float(row["macro_signature_count"]),
            ),
        )
        c.drawString(left + 370, y - 27, f'missing={row["missing_closed_loop_inputs"][:82]}')
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "This convergence audit blocks held-out micro-to-macro prediction claims until cage jumps and clocks are extracted.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_cage_jump_proxy_canary_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_cage_jump_proxy_canary.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench cage-jump proxy canary")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Aggregate frame-index MSD, NGP, and Fs decay mark jump-like candidates; particle-resolved event clocks remain blocked.",
    )
    left, top = 48, page_h - 100
    row_h = 62
    colors_by_stage = {
        "aggregate_cage_jump_proxy_ready_particle_events_blocked": colors.HexColor("#9f1239"),
        "aggregate_cage_jump_proxy_incomplete": colors.HexColor("#4a5568"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 24, "target")
    c.drawString(left + 100, top + 24, "canary stage")
    c.drawString(left + 370, top + 24, "aggregate proxy status")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["canary_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 13, 252, 25, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:44])
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 7.5)
        c.drawString(
            left + 370,
            y,
            "proxy={}; particle events={}; physical clock={}; blocker={}".format(
                int(float(row["aggregate_jump_proxy_ready"])),
                int(float(row["particle_resolved_jump_events_ready"])),
                int(float(row["physical_time_jump_clock_ready"])),
                row["primary_blocker"],
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 370,
            y - 14,
            "peak frame={:.0f}; proxy jump length={:.3g}; score={:.3g}; short Fs decay={:.3g}".format(
                float(row["peak_proxy_event_frame"]),
                float(row["proxy_jump_length"]),
                float(row["proxy_event_score"]),
                float(row["max_short_frame_fs_decay"]),
            ),
        )
        c.drawString(left + 370, y - 27, f'missing={row["missing_event_clock_inputs"][:82]}')
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        34,
        "This canary is an aggregate trajectory proxy only; it does not replace particle-resolved cage-jump segmentation.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_visible_member_ensemble_audit_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_visible_member_ensemble_audit.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench visible-member ensemble audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Prefix-visible NPZ headers are gated before they are used as an uncertainty ensemble.",
    )
    left, top = 48, page_h - 92
    row_h = 64
    colors_by_stage = {
        "visible_member_ensemble_ready_for_uncertainty": colors.HexColor("#2f855a"),
        "visible_prefix_not_publishable_ensemble": colors.HexColor("#c05621"),
        "visible_member_ensemble_layout_blocked": colors.HexColor("#4a5568"),
        "visible_member_split_policy_missing": colors.HexColor("#b7791f"),
        "visible_member_ensemble_incomplete": colors.HexColor("#805ad5"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 100, top + 18, "ensemble stage")
    c.drawString(left + 370, top + 18, "member evidence")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["ensemble_audit_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 12, 245, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:42])
        c.setFillColor(colors.black)
        c.drawString(
            left + 370,
            y,
            "ready={}; prefix={}/{}; full list={}; split={}".format(
                int(float(row["publishable_ensemble_uncertainty_ready"])),
                int(float(row["prefix_npz_member_count"])),
                int(float(row["required_member_count"])),
                int(float(row["full_member_id_list_visible"])),
                row["split_labels_in_probe"],
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(left + 370, y - 14, f'first member id: {row["first_member_id"][:60]}')
        c.drawString(
            left + 370,
            y - 27,
            f'next: {row["next_required_actions"].replace("_", " ")[:85]}',
        )
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "Current prefix evidence does not yet justify member-ensemble uncertainty or thermodynamic claims.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_observable_coverage_audit_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_observable_coverage_audit.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench observable coverage audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Frame-index MSD/NGP rows are gated before being used as real persistence/exchange inversion inputs.",
    )
    left, top = 48, page_h - 92
    row_h = 64
    colors_by_stage = {
        "real_inversion_observable_set_ready": colors.HexColor("#2f855a"),
        "frame_index_msd_ngp_only": colors.HexColor("#c05621"),
        "required_observable_set_incomplete": colors.HexColor("#b7791f"),
        "observable_semantics_incomplete": colors.HexColor("#805ad5"),
        "trajectory_observable_curve_missing": colors.HexColor("#4a5568"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 100, top + 18, "observable stage")
    c.drawString(left + 370, top + 18, "coverage evidence")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["observable_audit_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 12, 245, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:42])
        c.setFillColor(colors.black)
        c.drawString(
            left + 370,
            y,
            "ready={}; coverage={}; semantics={}; proxy substitution={}; frames={}".format(
                int(float(row["publishable_real_inversion_observable_set_ready"])),
                int(float(row["observable_coverage_ready"])),
                int(float(row["remote_result_semantics_ready"])),
                int(float(row["proxy_observable_substitution_allowed"])),
                int(float(row["frame_count"])),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 370,
            y - 14,
            f'available: {row["available_trajectory_observables"].replace("_", " ")[:72]}',
        )
        c.drawString(
            left + 370,
            y - 27,
            f'missing: {row["missing_observables"].replace("_", " ")[:76]}; next: {row["next_required_actions"].replace("_", " ")[:52]}',
        )
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "Proxy rhomax or ML-feature result curves are not substitutes for lag time, multi-k F_s, overlap chi4, and direct semantics.",
    )
    c.showPage()
    c.save()


def write_sota_glassbench_first_npz_structural_observable_plan_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_glassbench_first_npz_structural_observable_plan.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "GlassBench first-NPZ structural-observable plan")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Coordinate schema and implemented trajectory observables are separated from the missing raw first-NPZ bytes.",
    )
    left, top = 48, page_h - 92
    row_h = 64
    colors_by_stage = {
        "structural_observable_compute_ready": colors.HexColor("#2f855a"),
        "coordinate_schema_ready_positions_bytes_missing": colors.HexColor("#c05621"),
        "coordinate_schema_incomplete": colors.HexColor("#4a5568"),
    }
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top + 18, "target")
    c.drawString(left + 100, top + 18, "compute stage")
    c.drawString(left + 390, top + 18, "coordinate-to-observable status")
    for index, row in enumerate(rows):
        y = top - index * row_h
        stage = row["compute_plan_stage"]
        color = colors_by_stage.get(stage, colors.HexColor("#4a5568"))
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y, f'{row["system_id"]} T={row["temperature"]}')
        c.setFillColor(color)
        c.rect(left + 100, y - 12, 265, 24, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 7)
        c.drawString(left + 108, y - 3, stage.replace("_", " ")[:45])
        c.setFillColor(colors.black)
        c.drawString(
            left + 390,
            y,
            "schema={}; raw bytes={}; after extraction={}; immediate={}; frames={}; npz={:.0f} bytes".format(
                int(float(row["coordinate_schema_ready"])),
                int(float(row["raw_coordinate_bytes_cached"])),
                int(float(row["computable_after_npz_extraction"])),
                int(float(row["immediately_computable_from_current_cache"])),
                int(float(row["frame_count"])),
                float(row["npz_member_bytes"]),
            ),
        )
        c.setFont("Helvetica", 6.8)
        c.drawString(
            left + 390,
            y - 14,
            f'protocol: {row["implemented_observable_protocol"].replace("_", " ")[:82]}',
        )
        c.drawString(
            left + 390,
            y - 27,
            f'remaining: {row["remaining_missing_after_structural_compute"].replace("_", " ")}; next: {row["next_required_actions"].replace("_", " ")[:60]}',
        )
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.black)
    c.drawString(
        42,
        34,
        "The current cache is sufficient to prove computability after extraction, not to claim that F_s or chi4 have already been measured.",
    )
    c.showPage()
    c.save()


def write_sota_remote_result_curve_cache_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_remote_result_curve_cache.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA remote result-curve cache")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Small GlassBench result curves are range-fetched, CRC/md5 verified, and numerically parsed before adapters are claimed.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "system")
    c.drawString(left + 60, top, "stage")
    c.drawString(left + 280, top, "files")
    c.drawString(left + 325, top, "temps")
    c.drawString(left + 375, top, "cache")
    c.drawString(left + 425, top, "invert")
    c.drawString(left + 478, top, "blocker")
    palette = {
        "range_result_curves_verified": colors.HexColor("#2f855a"),
        "range_result_curves_missing": colors.HexColor("#c05621"),
        "range_result_curve_roles_incomplete": colors.HexColor("#c05621"),
        "range_result_curve_crc_mismatch": colors.HexColor("#c05621"),
        "range_result_curve_digest_missing": colors.HexColor("#c05621"),
        "range_result_curve_size_blocked": colors.HexColor("#c05621"),
        "range_result_curve_parse_blocked": colors.HexColor("#c05621"),
        "range_result_curve_range_missing": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 50
        stage = row["curve_cache_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["system_id"])
        c.setFillColor(palette[stage])
        c.rect(left + 60, y - 4, 190, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 66, y, stage.replace("_", " ")[:30])
        c.setFillColor(colors.black)
        c.drawString(left + 290, y, str(int(float(row["curve_file_count"]))))
        c.drawString(left + 338, y, str(int(float(row["temperature_count"]))))
        c.drawString(left + 390, y, str(int(float(row["curve_cache_ready"]))))
        c.drawString(left + 440, y, str(int(float(row["real_inversion_ready"]))))
        c.drawString(left + 478, y, row["primary_blocker"].replace("_", " ")[:30])
        c.drawString(left + 60, y - 12, f"roles: {row['available_roles']}; temperatures: {row['temperature_grid']}")
        c.drawString(
            left + 60,
            y - 24,
            f"CRC={int(float(row['crc32_verified']))}; md5={int(float(row['md5_available']))}; numeric={int(float(row['numeric_parse_ready']))}; range={int(float(row['range_fetch_ready']))}",
        )
    c.showPage()
    c.save()


def write_sota_remote_result_curve_fetch_gap_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_remote_result_curve_fetch_gap.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA remote result-curve fetch gap")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Dynamic-heterogeneity targets visible in GlassBench are tracked before range-cached comparison is claimed.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "target")
    c.drawString(left + 90, top, "stage")
    c.drawString(left + 385, top, "central")
    c.drawString(left + 435, top, "cache")
    c.drawString(left + 485, top, "fetch")
    c.drawString(left + 535, top, "compare")
    c.drawString(left + 595, top, "invert")
    c.drawString(left + 645, top, "blocker")
    palette = {
        "remote_target_missing": colors.HexColor("#c05621"),
        "remote_target_present_range_cache_missing": colors.HexColor("#2b6cb0"),
        "range_cache_target_parse_blocked": colors.HexColor("#c05621"),
        "range_cache_target_ready_for_observable_comparison": colors.HexColor("#2f855a"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 42
        stage = row["fetch_gap_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, f"{row['system_id']} T={row['temperature']} {row['curve_role']}")
        c.setFillColor(palette[stage])
        c.rect(left + 90, y - 4, 250, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 96, y, stage.replace("_", " ")[:38])
        c.setFillColor(colors.black)
        c.drawString(left + 400, y, str(int(float(row["central_directory_present"]))))
        c.drawString(left + 450, y, str(int(float(row["range_cache_present"]))))
        c.drawString(left + 500, y, str(int(float(row["targeted_fetch_ready"]))))
        c.drawString(left + 555, y, str(int(float(row["observable_comparison_ready"]))))
        c.drawString(left + 610, y, str(int(float(row["real_inversion_ready"]))))
        c.drawString(left + 645, y, row["primary_blocker"].replace("_", " ")[:26])
        c.drawString(left + 90, y - 13, row["target_path"][:100])
    c.showPage()
    c.save()


def write_sota_remote_result_curve_target_fetch_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_remote_result_curve_target_fetch.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA remote result-curve target fetch")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Targeted GlassBench chi4 bytes are fetched, but header-only payloads cannot support numeric comparison.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.3)
    c.drawString(left, top, "target")
    c.drawString(left + 90, top, "stage")
    c.drawString(left + 385, top, "fetch")
    c.drawString(left + 430, top, "crc")
    c.drawString(left + 475, top, "header")
    c.drawString(left + 525, top, "numeric")
    c.drawString(left + 580, top, "compare")
    c.drawString(left + 640, top, "blocker")
    palette = {
        "remote_target_missing": colors.HexColor("#c05621"),
        "target_fetch_missing": colors.HexColor("#c05621"),
        "target_fetch_checksum_blocked": colors.HexColor("#c05621"),
        "target_fetch_header_only_parse_blocked": colors.HexColor("#2b6cb0"),
        "target_fetch_numeric_parse_blocked": colors.HexColor("#c05621"),
        "target_fetch_numeric_ready_for_observable_comparison": colors.HexColor("#2f855a"),
    }
    c.setFont("Helvetica", 6.9)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 42
        stage = row["target_fetch_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, f"{row['system_id']} T={row['temperature']} {row['curve_role']}")
        c.setFillColor(palette[stage])
        c.rect(left + 90, y - 4, 250, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 96, y, stage.replace("_", " ")[:38])
        c.setFillColor(colors.black)
        c.drawString(left + 400, y, str(int(float(row["target_fetch_present"]))))
        c.drawString(left + 445, y, str(int(float(row["target_fetch_checksum_ready"]))))
        c.drawString(left + 492, y, str(int(float(row["header_only_payload"]))))
        c.drawString(left + 545, y, str(int(float(row["numeric_payload_ready"]))))
        c.drawString(left + 602, y, str(int(float(row["observable_comparison_ready"]))))
        c.drawString(left + 640, y, row["primary_blocker"].replace("_", " ")[:26])
        c.drawString(left + 90, y - 13, row["target_path"][:100])
    c.showPage()
    c.save()


def write_sota_remote_result_curve_published_semantics_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_remote_result_curve_published_semantics.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA published curve semantic audit")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Cached GlassBench FIG payloads are audited before treating ML benchmark curves as physical observables.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.1)
    c.drawString(left, top, "payload")
    c.drawString(left + 235, top, "stage")
    c.drawString(left + 525, top, "rows")
    c.drawString(left + 565, top, "cols")
    c.drawString(left + 605, top, "phys")
    c.drawString(left + 645, top, "compare")
    c.drawString(left + 700, top, "blocker")
    palette = {
        "published_curve_numeric_payload_blocked": colors.HexColor("#c05621"),
        "published_curve_physical_observable_label_uncertainty_missing": colors.HexColor("#2b6cb0"),
        "published_curve_ml_benchmark_not_physical_observable": colors.HexColor("#805ad5"),
    }
    c.setFont("Helvetica", 6.5)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 43
        stage = row["semantic_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["source_path"].replace("GlassBench/", "")[:36])
        c.setFillColor(palette[stage])
        c.rect(left + 235, y - 4, 250, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 241, y, stage.replace("_", " ")[:38])
        c.setFillColor(colors.black)
        c.drawString(left + 535, y, str(int(float(row["numeric_row_count"]))))
        c.drawString(left + 575, y, str(int(float(row["numeric_column_count"]))))
        c.drawString(left + 618, y, str(int(float(row["physical_observable_label_match"]))))
        c.drawString(left + 665, y, str(int(float(row["observable_comparison_ready"]))))
        c.drawString(left + 700, y, row["primary_blocker"].replace("_", " ")[:24])
        c.drawString(left + 235, y - 12, f"headers: {row['header_tokens']}"[:102])
    c.showPage()
    c.save()


def write_sota_remote_result_curve_payload_adapter_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_remote_result_curve_payload_adapter.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA remote result-curve payload adapter")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Cached GlassBench numeric payloads are paired into time/rhomax structural rows before any model inversion is claimed.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.2)
    c.drawString(left, top, "system")
    c.drawString(left + 55, top, "T")
    c.drawString(left + 95, top, "role")
    c.drawString(left + 160, top, "stage")
    c.drawString(left + 405, top, "pts")
    c.drawString(left + 450, top, "align")
    c.drawString(left + 495, top, "struct")
    c.drawString(left + 540, top, "real")
    c.drawString(left + 585, top, "blocker")
    palette = {
        "range_curve_payload_adapter_ready": colors.HexColor("#2f855a"),
        "range_curve_time_grid_missing": colors.HexColor("#c05621"),
        "range_curve_value_missing": colors.HexColor("#c05621"),
        "range_curve_payload_checksum_mismatch": colors.HexColor("#c05621"),
        "range_curve_payload_shape_mismatch": colors.HexColor("#c05621"),
        "range_curve_payload_parse_blocked": colors.HexColor("#c05621"),
        "range_curve_time_alignment_mismatch": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 39
        stage = row["adapter_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["system_id"])
        c.drawString(left + 55, y, row["temperature"])
        c.drawString(left + 95, y, row["curve_role"])
        c.setFillColor(palette[stage])
        c.rect(left + 160, y - 4, 210, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 166, y, stage.replace("_", " ")[:32])
        c.setFillColor(colors.black)
        c.drawString(left + 408, y, str(int(float(row["value_point_count"]))))
        c.drawString(left + 458, y, str(int(float(row["time_grid_matches_value_time"]))))
        c.drawString(left + 505, y, str(int(float(row["structural_adapter_ready"]))))
        c.drawString(left + 548, y, str(int(float(row["real_inversion_ready"]))))
        c.drawString(left + 585, y, row["primary_blocker"].replace("_", " ")[:26])
        c.drawString(left + 160, y - 12, row["value_curve_path"][:96])
    c.showPage()
    c.save()


def write_sota_remote_result_curve_observable_semantics_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_remote_result_curve_observable_semantics.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA remote result-curve observable semantics")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Structural rhomax rows are proxy observables until model-diagnostic semantics and uncertainties are supplied.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.2)
    c.drawString(left, top, "system")
    c.drawString(left + 55, top, "T")
    c.drawString(left + 95, top, "role")
    c.drawString(left + 160, top, "stage")
    c.drawString(left + 450, top, "proxy")
    c.drawString(left + 495, top, "diag")
    c.drawString(left + 540, top, "real")
    c.drawString(left + 585, top, "blocker")
    palette = {
        "structural_adapter_blocked": colors.HexColor("#c05621"),
        "observable_semantics_unmapped": colors.HexColor("#c05621"),
        "proxy_observable_ready_model_semantics_incomplete": colors.HexColor("#2b6cb0"),
        "observable_uncertainty_missing": colors.HexColor("#c05621"),
        "model_observable_semantics_ready": colors.HexColor("#2f855a"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 39
        stage = row["semantics_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["system_id"])
        c.drawString(left + 55, y, row["temperature"])
        c.drawString(left + 95, y, row["curve_role"])
        c.setFillColor(palette[stage])
        c.rect(left + 160, y - 4, 250, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 166, y, stage.replace("_", " ")[:38])
        c.setFillColor(colors.black)
        c.drawString(left + 460, y, str(int(float(row["proxy_observable_ready"]))))
        c.drawString(left + 505, y, str(int(float(row["diagnostic_semantics_ready"]))))
        c.drawString(left + 548, y, str(int(float(row["real_inversion_ready"]))))
        c.drawString(left + 585, y, row["primary_blocker"].replace("_", " ")[:26])
        c.drawString(left + 160, y - 12, f"missing: {row['missing_model_semantics']}"[:98])
    c.showPage()
    c.save()


def write_sota_readme_schema_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_sota_readme_schema.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "SOTA README schema gate")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "README-level evidence checks systems, folder tokens, license, and citation guidance before local adapters are claimed.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.4)
    c.drawString(left, top, "schema")
    c.drawString(left + 230, top, "stage")
    c.drawString(left + 445, top, "KA")
    c.drawString(left + 480, top, "KA2D")
    c.drawString(left + 525, top, "traj")
    c.drawString(left + 565, top, "schema")
    c.drawString(left + 620, top, "blocker")
    palette = {
        "remote_readme_schema_ready": colors.HexColor("#2f855a"),
        "local_archive_schema_ready": colors.HexColor("#276749"),
        "metadata_incomplete_schema": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 7.0)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 45
        stage = row["schema_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["schema_id"].replace("_", " ")[:36])
        c.setFillColor(palette[stage])
        c.rect(left + 230, y - 4, 180, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 236, y, stage.replace("_", " ")[:28])
        c.setFillColor(colors.black)
        c.drawString(left + 450, y, str(int(float(row["has_ka_system"]))))
        c.drawString(left + 490, y, str(int(float(row["has_ka2d_system"]))))
        c.drawString(left + 532, y, str(int(float(row["has_trajectory_folder"]))))
        c.drawString(left + 577, y, str(int(float(row["schema_ready"]))))
        c.drawString(left + 620, y, row["primary_blocker"].replace("_", " ")[:26])
        c.drawString(left + 72, y - 12, f"systems: {row['systems']}; folders: {row['folder_tokens']}; citations={row['citation_count']}")
    c.showPage()
    c.save()


def write_trajectory_adapter_contract_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_trajectory_adapter_contract.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Trajectory adapter contract")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Local trajectory adapters must expose coordinate, time-grid, identity, box, state-point, species, and units fields before trajectory diagnostics are claimed.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.4)
    c.drawString(left, top, "contract")
    c.drawString(left + 230, top, "stage")
    c.drawString(left + 430, top, "system")
    c.drawString(left + 485, top, "fields")
    c.drawString(left + 545, top, "ready")
    c.drawString(left + 590, top, "blocker")
    palette = {
        "remote_adapter_contract_only": colors.HexColor("#805ad5"),
        "metadata_incomplete_adapter": colors.HexColor("#c05621"),
        "local_trajectory_adapter_ready": colors.HexColor("#2f855a"),
    }
    c.setFont("Helvetica", 7.0)
    for idx, row in enumerate(rows):
        y = top - 25 - idx * 43
        stage = row["adapter_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["contract_id"].replace("_", " ")[:34])
        c.setFillColor(palette[stage])
        c.rect(left + 230, y - 4, 170, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 236, y, stage.replace("_", " ")[:27])
        c.setFillColor(colors.black)
        c.drawString(left + 430, y, row["system_id"][:9])
        c.drawString(
            left + 486,
            y,
            f"{int(float(row['available_required_field_count']))}/{int(float(row['required_field_count']))}",
        )
        c.drawString(left + 550, y, str(int(float(row["adapter_ready"]))))
        c.drawString(left + 590, y, row["primary_blocker"].replace("_", " ")[:28])
        c.drawString(left + 70, y - 12, f"missing: {row['missing_local_fields'][:94]}")
    c.showPage()
    c.save()


def write_raw_curve_diagnostic_readiness_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_raw_curve_diagnostic_readiness.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Raw-curve diagnostic readiness")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Observable column contracts are aggregated into protocol-level readiness before real-data inversion.",
    )
    left, top = 55, page_h - 95
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top, "diagnostic")
    c.drawString(left + 250, top, "struct")
    c.drawString(left + 305, top, "unc")
    c.drawString(left + 355, top, "primary blocker")
    c.drawString(left + 505, top, "required observables")
    c.setFont("Helvetica", 7.5)
    for idx, row in enumerate(rows):
        y = top - 24 - idx * 38
        structural = int(float(row["structural_diagnostic_ready"]))
        uncertainty = int(float(row["uncertainty_diagnostic_ready"]))
        color = colors.HexColor("#2f855a") if uncertainty else colors.HexColor("#2b6cb0") if structural else colors.HexColor("#c05621")
        c.setFillColor(colors.black)
        c.drawString(left, y, row["diagnostic_id"][:36])
        c.setFillColor(color)
        c.rect(left + 250, y - 4, 16, 10, stroke=0, fill=1)
        c.rect(left + 305, y - 4, 16, 10, stroke=0, fill=1 if uncertainty else 0)
        c.setFillColor(colors.black)
        c.drawString(left + 270, y, str(structural))
        c.drawString(left + 325, y, str(uncertainty))
        c.drawString(left + 355, y, row["primary_blocker"][:24])
        c.drawString(left + 505, y, row["required_observables"][:70])
        c.drawString(left + 90, y - 12, f"blockers: {row['blocking_observables_or_columns'][:90]}")
    c.showPage()
    c.save()


def write_raw_curve_persistence_exchange_protocol_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_raw_curve_persistence_exchange_protocol.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Raw-curve persistence/exchange inversion")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Machine-readable alpha, late NGP, diffusion, and chi4 curves are reduced to held-out z-score diagnostics.",
    )
    left, bottom = 80, 105
    plot_w, plot_h = 640, 300
    z_threshold = float(rows[0]["z_threshold"])
    z_values = np.array(
        [
            [
                float(row["max_multik_tau_alpha_z"]),
                float(row["late_ngp_z"]),
                float(row["chi4_peak_z"]),
            ]
            for row in rows
        ]
    )
    z_max = max(float(np.max(z_values)), z_threshold) * 1.15

    c.setStrokeColor(colors.black)
    c.line(left, bottom, left + plot_w, bottom)
    c.line(left, bottom, left, bottom + plot_h)
    threshold_y = bottom + z_threshold * plot_h / z_max
    c.setStrokeColor(colors.grey)
    c.setDash(4, 3)
    c.line(left, threshold_y, left + plot_w, threshold_y)
    c.setDash()
    c.setFont("Helvetica", 8)
    c.drawString(left + plot_w + 8, threshold_y - 3, f"z={z_threshold:g}")
    labels = ["multi-k alpha", "late NGP", "chi4 peak"]
    keys = [
        "multik_tau_alpha_z_consistent",
        "late_ngp_z_consistent",
        "chi4_peak_z_consistent",
    ]
    palette = [colors.HexColor("#2b6cb0"), colors.HexColor("#2f855a"), colors.HexColor("#805ad5")]
    group_x = np.linspace(left + 190, left + plot_w - 190, len(rows))
    offsets = [-34, 0, 34]
    c.setFont("Helvetica", 7)
    for idx, row in enumerate(rows):
        for jdx, label in enumerate(labels):
            value = z_values[idx, jdx]
            passed = float(row[keys[jdx]]) > 0.5
            color = palette[jdx] if passed else colors.HexColor("#c05621")
            bar_h = value * plot_h / z_max
            x = group_x[idx] + offsets[jdx]
            c.setFillColor(color)
            c.rect(x - 10, bottom, 20, bar_h, stroke=0, fill=1)
            c.drawString(x - 22, bottom - 36 - jdx * 9, label[:16])
        c.setFillColor(colors.black)
        c.drawCentredString(group_x[idx], bottom - 20, row["scenario"].replace("_", " "))
        c.drawString(
            group_x[idx] - 72,
            bottom + plot_h - 24,
            f"tau_p/tau_x={float(row['persistence_exchange_ratio']):.2f}",
        )
        c.drawString(
            group_x[idx] - 72,
            bottom + plot_h - 38,
            f"SE growth={float(row['stokes_einstein_growth_over_poisson']):.2f}",
        )
        c.drawString(
            group_x[idx] - 72,
            bottom + plot_h - 52,
            f"pass={int(float(row['raw_curve_protocol_passes']))}",
        )
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 8)
    c.drawString(34, bottom + 145, "absolute z score")
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


def write_trajectory_observable_protocol_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_trajectory_observable_protocol.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Trajectory-to-observable bridge")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Particle trajectories are reduced to MSD, NGP, self-intermediate scattering, and overlap chi4 rows.",
    )
    left, bottom = 80, 105
    plot_w, plot_h = 640, 300
    lag = np.array([float(row["lag_time"]) for row in rows])
    series = [
        ("MSD", np.array([float(row["msd"]) for row in rows]), colors.HexColor("#2b6cb0")),
        ("NGP", np.array([float(row["ngp"]) for row in rows]), colors.HexColor("#c05621")),
        ("chi4", np.array([float(row["chi4_overlap"]) for row in rows]), colors.HexColor("#805ad5")),
        (
            "F_s",
            np.array([float(row["self_intermediate_scattering"]) for row in rows]),
            colors.HexColor("#2f855a"),
        ),
    ]

    def scaled_x(values: np.ndarray) -> np.ndarray:
        return left + (values - np.min(values)) * plot_w / max(float(np.max(values) - np.min(values)), 1e-12)

    def scaled_y(values: np.ndarray) -> np.ndarray:
        span = max(float(np.max(values) - np.min(values)), 1e-12)
        return bottom + (values - np.min(values)) * plot_h / span

    c.setStrokeColor(colors.black)
    c.line(left, bottom, left + plot_w, bottom)
    c.line(left, bottom, left, bottom + plot_h)
    c.setFont("Helvetica", 7.5)
    for idx, (label, values, color) in enumerate(series):
        xs = scaled_x(lag)
        ys = scaled_y(values)
        c.setStrokeColor(color)
        c.setFillColor(color)
        for j in range(len(xs) - 1):
            c.line(float(xs[j]), float(ys[j]), float(xs[j + 1]), float(ys[j + 1]))
        for x, y in zip(xs, ys):
            c.circle(float(x), float(y), 2.6, stroke=0, fill=1)
        c.drawString(left + idx * 105, bottom - 35, label)
    peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
    c.setFillColor(colors.black)
    c.drawString(
        left,
        bottom - 58,
        f"peak chi4 lag={float(peak['lag_time']):.1f}; chi4={float(peak['chi4_overlap']):.3f}; NGP={float(peak['ngp']):.3f}",
    )
    c.drawString(left, bottom - 72, f"observable set: {peak['structural_observable_set']}")
    c.showPage()
    c.save()


def write_trajectory_cage_jump_events_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_trajectory_cage_jump_events.csv").open() as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Particle-resolved cage-jump event clock")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Synthetic trajectory canary: threshold jumps define persistence and exchange intervals before real-benchmark inversion.",
    )
    left, top = 80, page_h - 105
    labels = [
        ("jump events", float(row["total_jump_event_count"]), colors.HexColor("#2b6cb0")),
        ("particles with jumps", float(row["particles_with_jump_count"]), colors.HexColor("#2f855a")),
        ("exchange intervals", float(row["exchange_interval_count"]), colors.HexColor("#805ad5")),
        ("mean persistence", float(row["persistence_mean"]), colors.HexColor("#c05621")),
        ("mean exchange", float(row["exchange_mean"]), colors.HexColor("#718096")),
    ]
    max_value = max(value for _, value, _ in labels)
    c.setFont("Helvetica", 8)
    for idx, (label, value, color) in enumerate(labels):
        y0 = top - idx * 42
        bar_w = 470 * value / max(max_value, 1e-12)
        c.setFillColor(colors.black)
        c.drawString(left, y0 + 7, label)
        c.setFillColor(color)
        c.rect(left + 135, y0, bar_w, 18, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawString(left + 145 + bar_w, y0 + 5, f"{value:.3g}")
    c.setFont("Helvetica", 7.5)
    c.drawString(
        left,
        94,
        f"stage={row['event_protocol_stage']}; blocker={row['primary_blocker']}; thermodynamic_claim_allowed={int(float(row['thermodynamic_claim_allowed']))}",
    )
    c.drawString(left, 78, f"scope={row['scope_note']}")
    c.showPage()
    c.save()


def write_trajectory_event_clock_macro_predictions_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_trajectory_event_clock_macro_predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Event-clock micro-to-macro prediction")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Particle jump clocks predict D, multi-k alpha, late NGP, and chi4 without fitting macro observables.",
    )
    left, top = 70, page_h - 105
    metrics = [
        ("D", "diffusion_z", colors.HexColor("#2b6cb0")),
        ("tau alpha", "max_tau_alpha_z", colors.HexColor("#2f855a")),
        ("late NGP", "late_ngp_z", colors.HexColor("#c05621")),
        ("chi4", "chi4_peak_z", colors.HexColor("#805ad5")),
    ]
    max_z = max(max(float(row[key]) for _, key, _ in metrics) for row in rows)
    max_z = max(max_z, 2.0)
    c.setFont("Helvetica", 7.5)
    threshold_y = top - 65 * 2.0 / max_z
    c.setStrokeColor(colors.HexColor("#718096"))
    c.setDash(3, 3)
    c.line(left, threshold_y, left + 580, threshold_y)
    c.setDash()
    c.setFillColor(colors.HexColor("#718096"))
    c.drawString(left + 590, threshold_y - 3, "z=2")
    for row_idx, row in enumerate(rows):
        y0 = top - row_idx * 135
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(left, y0 + 12, row["protocol_id"])
        c.setFont("Helvetica", 7.5)
        for metric_idx, (label, key, color) in enumerate(metrics):
            value = float(row[key])
            x0 = left + metric_idx * 132
            bar_h = 65 * min(value / max_z, 1.0)
            c.setFillColor(color)
            c.rect(x0, y0 - 68, 34, bar_h, stroke=0, fill=1)
            c.setFillColor(colors.black)
            c.drawString(x0, y0 - 82, label)
            c.drawString(x0, y0 - 70 + bar_h, f"{value:.2g}")
        c.setFillColor(colors.black)
        c.drawString(left, y0 - 106, f"stage={row['prediction_stage']}; blocker={row['primary_blocker']}")
    c.showPage()
    c.save()


def write_trajectory_event_clock_threshold_robustness_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_trajectory_event_clock_threshold_robustness.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Event-clock threshold robustness")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Micro-to-macro predictions must pass across a stable cage-jump threshold window and fail outside it.",
    )
    left, bottom = 78, 108
    plot_w, plot_h = 640, 290
    thresholds = np.array([float(row["jump_displacement_threshold"]) for row in rows])
    z_values = []
    for row in rows:
        if float(row["event_clock_ready"]) == 1.0:
            z_values.append(
                max(
                    float(row["diffusion_z"]),
                    float(row["max_tau_alpha_z"]),
                    float(row["late_ngp_z"]),
                    float(row["chi4_peak_z"]),
                )
            )
        else:
            z_values.append(2.5)
    z = np.array(z_values)
    zmax = max(3.0, float(np.max(z)))

    def sx(value: float) -> float:
        return left + (value - float(np.min(thresholds))) * plot_w / max(float(np.max(thresholds) - np.min(thresholds)), 1e-12)

    def sy(value: float) -> float:
        return bottom + plot_h - value * plot_h / zmax

    c.setStrokeColor(colors.black)
    c.line(left, bottom, left + plot_w, bottom)
    c.line(left, bottom, left, bottom + plot_h)
    c.setStrokeColor(colors.HexColor("#718096"))
    c.setDash(3, 3)
    c.line(left, sy(2.0), left + plot_w, sy(2.0))
    c.setDash()
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#718096"))
    c.drawString(left + plot_w - 26, sy(2.0) + 4, "z=2")
    for row, value in zip(rows, z):
        passed = float(row["threshold_prediction_pass"]) == 1.0
        ready = float(row["event_clock_ready"]) == 1.0
        color = colors.HexColor("#2f855a" if passed else ("#c05621" if ready else "#718096"))
        x0 = sx(float(row["jump_displacement_threshold"]))
        y0 = sy(float(value))
        c.setFillColor(color)
        c.circle(x0, y0, 4, stroke=0, fill=1)
        c.setFillColor(colors.black)
        c.drawString(x0 - 8, bottom - 18, f"{float(row['jump_displacement_threshold']):.2g}")
    c.setFillColor(colors.black)
    c.drawString(left, 70, f"stable threshold window count={int(max(float(row['stable_threshold_window_count']) for row in rows))}")
    c.drawString(left, 56, "thermodynamic_claim_allowed=0; real_benchmark_closed_loop_ready=0")
    c.showPage()
    c.save()


def write_trajectory_uncertainty_protocol_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_trajectory_uncertainty_protocol.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Trajectory uncertainty bridge")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Time-origin block jackknife supplies uncertainty columns for trajectory-derived observables.",
    )
    left, bottom = 80, 105
    plot_w, plot_h = 640, 300
    lag = np.array([float(row["lag_time"]) for row in rows])
    series = [
        ("sigma MSD", np.array([float(row["sigma_msd"]) for row in rows]), colors.HexColor("#2b6cb0")),
        ("sigma NGP", np.array([float(row["sigma_ngp"]) for row in rows]), colors.HexColor("#c05621")),
        (
            "sigma F_s",
            np.array([float(row["sigma_self_intermediate_scattering"]) for row in rows]),
            colors.HexColor("#2f855a"),
        ),
        ("sigma chi4", np.array([float(row["sigma_chi4_overlap"]) for row in rows]), colors.HexColor("#805ad5")),
    ]

    def scaled_x(values: np.ndarray) -> np.ndarray:
        return left + (values - np.min(values)) * plot_w / max(float(np.max(values) - np.min(values)), 1e-12)

    def scaled_y(values: np.ndarray) -> np.ndarray:
        span = max(float(np.max(values) - np.min(values)), 1e-12)
        return bottom + (values - np.min(values)) * plot_h / span

    c.setStrokeColor(colors.black)
    c.line(left, bottom, left + plot_w, bottom)
    c.line(left, bottom, left, bottom + plot_h)
    c.setFont("Helvetica", 7.5)
    for idx, (label, values, color) in enumerate(series):
        xs = scaled_x(lag)
        ys = scaled_y(values)
        c.setStrokeColor(color)
        c.setFillColor(color)
        for j in range(len(xs) - 1):
            c.line(float(xs[j]), float(ys[j]), float(xs[j + 1]), float(ys[j + 1]))
        for x, y in zip(xs, ys):
            c.circle(float(x), float(y), 2.6, stroke=0, fill=1)
        c.drawString(left + idx * 105, bottom - 35, label)
    peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
    c.setFillColor(colors.black)
    c.drawString(
        left,
        bottom - 58,
        f"method={peak['uncertainty_method']}; blocks={int(float(peak['jackknife_block_count']))}; blocker={peak['primary_blocker']}",
    )
    c.drawString(
        left,
        bottom - 72,
        f"peak chi4 row sigmas: MSD={float(peak['sigma_msd']):.3f}, Fs={float(peak['sigma_self_intermediate_scattering']):.3f}, chi4={float(peak['sigma_chi4_overlap']):.3f}",
    )
    c.showPage()
    c.save()


def write_trajectory_member_ensemble_uncertainty_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_trajectory_member_ensemble_uncertainty.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Trajectory member-ensemble uncertainty")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Independent trajectory members supply standard-error columns before uncertainty-weighted inversion.",
    )
    left, bottom = 80, 105
    plot_w, plot_h = 640, 300
    lag = np.array([float(row["lag_time"]) for row in rows])
    series = [
        ("member sigma MSD", np.array([float(row["sigma_msd"]) for row in rows]), colors.HexColor("#2b6cb0")),
        ("member sigma NGP", np.array([float(row["sigma_ngp"]) for row in rows]), colors.HexColor("#c05621")),
        (
            "member sigma F_s",
            np.array([float(row["sigma_self_intermediate_scattering"]) for row in rows]),
            colors.HexColor("#2f855a"),
        ),
        (
            "member sigma chi4",
            np.array([float(row["sigma_chi4_overlap"]) for row in rows]),
            colors.HexColor("#805ad5"),
        ),
    ]

    def scaled_x(values: np.ndarray) -> np.ndarray:
        return left + (values - np.min(values)) * plot_w / max(float(np.max(values) - np.min(values)), 1e-12)

    def scaled_y(values: np.ndarray) -> np.ndarray:
        span = max(float(np.max(values) - np.min(values)), 1e-12)
        return bottom + (values - np.min(values)) * plot_h / span

    c.setStrokeColor(colors.black)
    c.line(left, bottom, left + plot_w, bottom)
    c.line(left, bottom, left, bottom + plot_h)
    c.setFont("Helvetica", 7.5)
    for idx, (label, values, color) in enumerate(series):
        xs = scaled_x(lag)
        ys = scaled_y(values)
        c.setStrokeColor(color)
        c.setFillColor(color)
        for j in range(len(xs) - 1):
            c.line(float(xs[j]), float(ys[j]), float(xs[j + 1]), float(ys[j + 1]))
        for x, y in zip(xs, ys):
            c.circle(float(x), float(y), 2.6, stroke=0, fill=1)
        c.drawString(left + idx * 120, bottom - 35, label)
    peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
    c.setFillColor(colors.black)
    c.drawString(
        left,
        bottom - 58,
        f"method={peak['uncertainty_method']}; members={int(float(peak['member_count']))}; threshold={int(float(peak['min_member_count']))}; blocker={peak['primary_blocker']}",
    )
    c.drawString(
        left,
        bottom - 72,
        f"stage={peak['ensemble_stage']}; target={peak['target_protocol']}",
    )
    c.showPage()
    c.save()


def write_trajectory_inversion_readiness_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_trajectory_inversion_readiness.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Trajectory inversion readiness gate")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Trajectory-derived observables are promoted only when structural observables and uncertainty columns are both present.",
    )
    left, top = 55, page_h - 95
    c.setFont("Helvetica-Bold", 8)
    c.drawString(left, top, "benchmark")
    c.drawString(left + 245, top, "stage")
    c.drawString(left + 470, top, "struct")
    c.drawString(left + 520, top, "unc")
    c.drawString(left + 565, top, "blocker")
    palette = {
        "uncertainty_weighted_trajectory_inversion": colors.HexColor("#2f855a"),
        "structural_trajectory_only": colors.HexColor("#2b6cb0"),
        "trajectory_blocked": colors.HexColor("#c05621"),
    }
    c.setFont("Helvetica", 7.5)
    for idx, row in enumerate(rows):
        y = top - 28 - idx * 45
        stage = row["readiness_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["benchmark_id"].replace("_", " ")[:34])
        c.setFillColor(palette[stage])
        c.rect(left + 245, y - 4, 195, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 250, y, stage.replace("_", " ")[:32])
        c.setFillColor(colors.black)
        c.drawString(left + 470, y, str(int(float(row["structural_trajectory_ready"]))))
        c.drawString(left + 520, y, str(int(float(row["uncertainty_weighted_ready"]))))
        c.drawString(left + 565, y, row["primary_blocker"].replace("_", " ")[:28])
        c.drawString(left + 245, y - 13, f"missing sigma: {row['missing_uncertainty_columns'][:70]}")
    c.showPage()
    c.save()


def write_benchmark_publication_ladder_pdf(path: Path) -> None:
    with (DATA_DIR / "renewal_cage_benchmark_publication_ladder.csv").open() as f:
        rows = list(csv.DictReader(f))
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(42, page_h - 34, "Benchmark publication claim ladder")
    c.setFont("Helvetica", 8)
    c.drawString(
        42,
        page_h - 48,
        "Readiness, reanalysis, and held-out prediction gates are collapsed into manuscript-safe claim levels.",
    )
    left, top = 42, page_h - 92
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(left, top, "ladder row")
    c.drawString(left + 215, top, "publication stage")
    c.drawString(left + 405, top, "allowed claim")
    c.drawString(left + 565, top, "real")
    c.drawString(left + 608, top, "overreach")
    c.drawString(left + 680, top, "next action")
    palette = {
        "metadata_verified_not_reanalysis": colors.HexColor("#d69e2e"),
        "synthetic_prediction_canary_passed": colors.HexColor("#2f855a"),
        "fit_only_overclaim_blocked": colors.HexColor("#b83280"),
        "thermodynamic_scope_boundary": colors.HexColor("#805ad5"),
        "uncertainty_weighted_real_reanalysis": colors.HexColor("#276749"),
        "structural_or_uncertainty_gate_incomplete": colors.HexColor("#2b6cb0"),
        "heldout_prediction_not_passed": colors.HexColor("#c05621"),
        "forbidden_claim_blocked": colors.HexColor("#b83280"),
        "not_supported": colors.HexColor("#718096"),
    }
    c.setFont("Helvetica", 6.8)
    for idx, row in enumerate(rows):
        y = top - 22 - idx * 38
        stage = row["publication_stage"]
        c.setFillColor(colors.black)
        c.drawString(left, y, row["ladder_id"].replace("_", " ")[:31])
        c.setFillColor(palette[stage])
        c.rect(left + 215, y - 4, 172, 13, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.drawString(left + 220, y, stage.replace("_", " ")[:28])
        c.setFillColor(colors.black)
        c.drawString(left + 405, y, row["allowed_manuscript_claim"].replace("_", " ")[:27])
        c.drawString(left + 570, y, str(int(float(row["real_data_quantitative_comparison"]))))
        c.drawString(left + 625, y, str(int(float(row["claim_overreach_if_called_fit"]))))
        c.drawString(left + 680, y, row["next_required_action"].replace("_", " ")[:34])
        c.drawString(left + 215, y - 11, f"source: {row['source_key'].replace('_', ' ')[:72]}")
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
    translation_rotation_protocol_pdf = PAPER_FIGURE_DIR / "renewal_cage_translation_rotation_protocol.pdf"
    glass_audit_pdf = PAPER_FIGURE_DIR / "renewal_cage_glass_audit.pdf"
    glass_phase_diagram_pdf = PAPER_FIGURE_DIR / "renewal_cage_glass_phase_diagram.pdf"
    spatial_chi4_pdf = PAPER_FIGURE_DIR / "renewal_cage_spatial_chi4.pdf"
    thermodynamic_closure_pdf = PAPER_FIGURE_DIR / "renewal_cage_thermodynamic_closure.pdf"
    mct_beta_closure_pdf = PAPER_FIGURE_DIR / "renewal_cage_mct_beta_closure.pdf"
    sota_benchmark_consistency_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_benchmark_consistency.pdf"
    sota_claim_alignment_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_claim_alignment.pdf"
    sota_signed_constraints_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_signed_constraints.pdf"
    sota_evidence_verdict_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_evidence_verdict.pdf"
    sota_evidence_class_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_evidence_class.pdf"
    simultaneous_closure_pdf = PAPER_FIGURE_DIR / "renewal_cage_simultaneous_closure.pdf"
    real_benchmark_assimilation_gate_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_real_benchmark_assimilation_gate.pdf"
    )
    cross_observable_prediction_ledger_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_cross_observable_prediction_ledger.pdf"
    )
    inversion_identifiability_audit_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_inversion_identifiability_audit.pdf"
    )
    frontier_benchmark_horizon_pdf = PAPER_FIGURE_DIR / "renewal_cage_frontier_benchmark_horizon.pdf"
    sota_source_provenance_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_source_provenance.pdf"
    sota_data_accession_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_data_accession.pdf"
    sota_zenodo_record_fingerprint_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_zenodo_record_fingerprint.pdf"
    )
    sota_remote_zip_central_directory_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_remote_zip_central_directory.pdf"
    )
    sota_glassbench_payload_index_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_payload_index.pdf"
    sota_glassbench_trajectory_payload_locator_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_payload_locator.pdf"
    )
    sota_glassbench_trajectory_entry_metadata_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_entry_metadata.pdf"
    )
    sota_glassbench_trajectory_member_stream_probe_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_member_stream_probe.pdf"
    )
    sota_glassbench_trajectory_inner_tar_header_probe_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.pdf"
    )
    sota_glassbench_trajectory_npz_schema_probe_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_npz_schema_probe.pdf"
    )
    sota_glassbench_trajectory_first_npz_observable_smoke_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.pdf"
    )
    sota_glassbench_trajectory_first_npz_observable_curve_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.pdf"
    )
    sota_glassbench_short_window_trend_canary_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_short_window_trend_canary.pdf"
    )
    sota_glassbench_trajectory_timebase_bridge_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_timebase_bridge.pdf"
    )
    sota_glassbench_frame_time_mapping_audit_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_frame_time_mapping_audit.pdf"
    )
    sota_glassbench_real_inversion_gap_ledger_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_real_inversion_gap_ledger.pdf"
    )
    sota_glassbench_real_inversion_unlock_protocol_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_real_inversion_unlock_protocol.pdf"
    )
    sota_glassbench_trajectory_first_npz_inversion_readiness_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.pdf"
    )
    sota_glassbench_trajectory_npz_ensemble_horizon_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.pdf"
    )
    sota_glassbench_trajectory_npz_member_index_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_npz_member_index.pdf"
    )
    sota_glassbench_trajectory_member_ensemble_observable_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.pdf"
    )
    sota_glassbench_ka2d_timecode_semantics_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_ka2d_timecode_semantics.pdf"
    )
    sota_glassbench_timecode_curve_bridge_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_timecode_curve_bridge.pdf"
    )
    sota_glassbench_timecode_signature_support_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_timecode_signature_support.pdf"
    )
    sota_glassbench_alpha_threshold_horizon_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_alpha_threshold_horizon.pdf"
    )
    sota_dynamic_signature_alignment_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_dynamic_signature_alignment.pdf"
    )
    sota_glassbench_microdynamic_closed_loop_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_microdynamic_closed_loop.pdf"
    )
    sota_glassbench_cage_jump_proxy_canary_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_cage_jump_proxy_canary.pdf"
    )
    sota_glassbench_visible_member_ensemble_audit_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_visible_member_ensemble_audit.pdf"
    )
    sota_glassbench_observable_coverage_audit_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_observable_coverage_audit.pdf"
    )
    sota_glassbench_first_npz_structural_observable_plan_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_glassbench_first_npz_structural_observable_plan.pdf"
    )
    sota_remote_result_curve_cache_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_remote_result_curve_cache.pdf"
    )
    sota_remote_result_curve_fetch_gap_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_remote_result_curve_fetch_gap.pdf"
    )
    sota_remote_result_curve_target_fetch_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_remote_result_curve_target_fetch.pdf"
    )
    sota_remote_result_curve_published_semantics_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_remote_result_curve_published_semantics.pdf"
    )
    sota_remote_result_curve_payload_adapter_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_remote_result_curve_payload_adapter.pdf"
    )
    sota_remote_result_curve_observable_semantics_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_sota_remote_result_curve_observable_semantics.pdf"
    )
    sota_readme_schema_pdf = PAPER_FIGURE_DIR / "renewal_cage_sota_readme_schema.pdf"
    trajectory_adapter_contract_pdf = PAPER_FIGURE_DIR / "renewal_cage_trajectory_adapter_contract.pdf"
    literature_inversion_readiness_pdf = PAPER_FIGURE_DIR / "renewal_cage_literature_inversion_readiness.pdf"
    observable_falsification_matrix_pdf = PAPER_FIGURE_DIR / "renewal_cage_observable_falsification_matrix.pdf"
    benchmark_fusion_readiness_pdf = PAPER_FIGURE_DIR / "renewal_cage_benchmark_fusion_readiness.pdf"
    raw_curve_ingestion_contract_pdf = PAPER_FIGURE_DIR / "renewal_cage_raw_curve_ingestion_contract.pdf"
    raw_curve_diagnostic_readiness_pdf = PAPER_FIGURE_DIR / "renewal_cage_raw_curve_diagnostic_readiness.pdf"
    raw_curve_persistence_exchange_protocol_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_raw_curve_persistence_exchange_protocol.pdf"
    )
    trajectory_observable_protocol_pdf = PAPER_FIGURE_DIR / "renewal_cage_trajectory_observable_protocol.pdf"
    trajectory_cage_jump_events_pdf = PAPER_FIGURE_DIR / "renewal_cage_trajectory_cage_jump_events.pdf"
    trajectory_event_clock_macro_predictions_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_trajectory_event_clock_macro_predictions.pdf"
    )
    trajectory_event_clock_threshold_robustness_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_trajectory_event_clock_threshold_robustness.pdf"
    )
    trajectory_uncertainty_protocol_pdf = PAPER_FIGURE_DIR / "renewal_cage_trajectory_uncertainty_protocol.pdf"
    trajectory_member_ensemble_uncertainty_pdf = (
        PAPER_FIGURE_DIR / "renewal_cage_trajectory_member_ensemble_uncertainty.pdf"
    )
    trajectory_inversion_readiness_pdf = PAPER_FIGURE_DIR / "renewal_cage_trajectory_inversion_readiness.pdf"
    benchmark_publication_ladder_pdf = PAPER_FIGURE_DIR / "renewal_cage_benchmark_publication_ladder.pdf"
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
    write_translation_rotation_protocol_pdf(translation_rotation_protocol_pdf)
    write_glass_audit_pdf(glass_audit_pdf)
    write_glass_phase_diagram_pdf(glass_phase_diagram_pdf)
    write_spatial_chi4_pdf(spatial_chi4_pdf)
    write_thermodynamic_closure_pdf(thermodynamic_closure_pdf)
    write_mct_beta_closure_pdf(mct_beta_closure_pdf)
    write_sota_benchmark_consistency_pdf(sota_benchmark_consistency_pdf)
    write_sota_claim_alignment_pdf(sota_claim_alignment_pdf)
    write_sota_signed_constraints_pdf(sota_signed_constraints_pdf)
    write_sota_evidence_verdict_pdf(sota_evidence_verdict_pdf)
    write_sota_evidence_class_pdf(sota_evidence_class_pdf)
    write_simultaneous_closure_pdf(simultaneous_closure_pdf)
    write_real_benchmark_assimilation_gate_pdf(real_benchmark_assimilation_gate_pdf)
    write_cross_observable_prediction_ledger_pdf(cross_observable_prediction_ledger_pdf)
    write_inversion_identifiability_audit_pdf(inversion_identifiability_audit_pdf)
    write_frontier_benchmark_horizon_pdf(frontier_benchmark_horizon_pdf)
    write_sota_source_provenance_pdf(sota_source_provenance_pdf)
    write_sota_data_accession_pdf(sota_data_accession_pdf)
    write_sota_zenodo_record_fingerprint_pdf(sota_zenodo_record_fingerprint_pdf)
    write_sota_remote_zip_central_directory_pdf(sota_remote_zip_central_directory_pdf)
    write_sota_glassbench_payload_index_pdf(sota_glassbench_payload_index_pdf)
    write_sota_glassbench_trajectory_payload_locator_pdf(sota_glassbench_trajectory_payload_locator_pdf)
    write_sota_glassbench_trajectory_entry_metadata_pdf(sota_glassbench_trajectory_entry_metadata_pdf)
    write_sota_glassbench_trajectory_member_stream_probe_pdf(sota_glassbench_trajectory_member_stream_probe_pdf)
    write_sota_glassbench_trajectory_inner_tar_header_probe_pdf(
        sota_glassbench_trajectory_inner_tar_header_probe_pdf
    )
    write_sota_glassbench_trajectory_npz_schema_probe_pdf(sota_glassbench_trajectory_npz_schema_probe_pdf)
    write_sota_glassbench_trajectory_first_npz_observable_smoke_pdf(
        sota_glassbench_trajectory_first_npz_observable_smoke_pdf
    )
    write_sota_glassbench_trajectory_first_npz_observable_curve_pdf(
        sota_glassbench_trajectory_first_npz_observable_curve_pdf
    )
    write_sota_glassbench_short_window_trend_canary_pdf(
        sota_glassbench_short_window_trend_canary_pdf
    )
    write_sota_glassbench_trajectory_timebase_bridge_pdf(
        sota_glassbench_trajectory_timebase_bridge_pdf
    )
    write_sota_glassbench_frame_time_mapping_audit_pdf(
        sota_glassbench_frame_time_mapping_audit_pdf
    )
    write_sota_glassbench_real_inversion_gap_ledger_pdf(
        sota_glassbench_real_inversion_gap_ledger_pdf
    )
    write_sota_glassbench_real_inversion_unlock_protocol_pdf(
        sota_glassbench_real_inversion_unlock_protocol_pdf
    )
    write_sota_glassbench_trajectory_first_npz_inversion_readiness_pdf(
        sota_glassbench_trajectory_first_npz_inversion_readiness_pdf
    )
    write_sota_glassbench_trajectory_npz_member_index_pdf(
        sota_glassbench_trajectory_npz_member_index_pdf
    )
    write_sota_glassbench_trajectory_member_ensemble_observable_pdf(
        sota_glassbench_trajectory_member_ensemble_observable_pdf
    )
    write_sota_glassbench_ka2d_timecode_semantics_pdf(
        sota_glassbench_ka2d_timecode_semantics_pdf
    )
    write_sota_glassbench_timecode_curve_bridge_pdf(
        sota_glassbench_timecode_curve_bridge_pdf
    )
    write_sota_glassbench_timecode_signature_support_pdf(
        sota_glassbench_timecode_signature_support_pdf
    )
    write_sota_glassbench_alpha_threshold_horizon_pdf(
        sota_glassbench_alpha_threshold_horizon_pdf
    )
    write_sota_dynamic_signature_alignment_pdf(
        sota_dynamic_signature_alignment_pdf
    )
    write_sota_glassbench_microdynamic_closed_loop_pdf(
        sota_glassbench_microdynamic_closed_loop_pdf
    )
    write_sota_glassbench_cage_jump_proxy_canary_pdf(
        sota_glassbench_cage_jump_proxy_canary_pdf
    )
    write_sota_glassbench_trajectory_npz_ensemble_horizon_pdf(
        sota_glassbench_trajectory_npz_ensemble_horizon_pdf
    )
    write_sota_glassbench_visible_member_ensemble_audit_pdf(
        sota_glassbench_visible_member_ensemble_audit_pdf
    )
    write_sota_glassbench_observable_coverage_audit_pdf(
        sota_glassbench_observable_coverage_audit_pdf
    )
    write_sota_glassbench_first_npz_structural_observable_plan_pdf(
        sota_glassbench_first_npz_structural_observable_plan_pdf
    )
    write_sota_remote_result_curve_cache_pdf(sota_remote_result_curve_cache_pdf)
    write_sota_remote_result_curve_fetch_gap_pdf(sota_remote_result_curve_fetch_gap_pdf)
    write_sota_remote_result_curve_target_fetch_pdf(sota_remote_result_curve_target_fetch_pdf)
    write_sota_remote_result_curve_published_semantics_pdf(sota_remote_result_curve_published_semantics_pdf)
    write_sota_remote_result_curve_payload_adapter_pdf(sota_remote_result_curve_payload_adapter_pdf)
    write_sota_remote_result_curve_observable_semantics_pdf(sota_remote_result_curve_observable_semantics_pdf)
    write_sota_readme_schema_pdf(sota_readme_schema_pdf)
    write_trajectory_adapter_contract_pdf(trajectory_adapter_contract_pdf)
    write_literature_inversion_readiness_pdf(literature_inversion_readiness_pdf)
    write_observable_falsification_matrix_pdf(observable_falsification_matrix_pdf)
    write_benchmark_fusion_readiness_pdf(benchmark_fusion_readiness_pdf)
    write_raw_curve_ingestion_contract_pdf(raw_curve_ingestion_contract_pdf)
    write_raw_curve_diagnostic_readiness_pdf(raw_curve_diagnostic_readiness_pdf)
    write_raw_curve_persistence_exchange_protocol_pdf(raw_curve_persistence_exchange_protocol_pdf)
    write_trajectory_observable_protocol_pdf(trajectory_observable_protocol_pdf)
    write_trajectory_cage_jump_events_pdf(trajectory_cage_jump_events_pdf)
    write_trajectory_event_clock_macro_predictions_pdf(trajectory_event_clock_macro_predictions_pdf)
    write_trajectory_event_clock_threshold_robustness_pdf(trajectory_event_clock_threshold_robustness_pdf)
    write_trajectory_uncertainty_protocol_pdf(trajectory_uncertainty_protocol_pdf)
    write_trajectory_member_ensemble_uncertainty_pdf(trajectory_member_ensemble_uncertainty_pdf)
    write_trajectory_inversion_readiness_pdf(trajectory_inversion_readiness_pdf)
    write_benchmark_publication_ladder_pdf(benchmark_publication_ladder_pdf)
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
        archive.write(translation_rotation_protocol_pdf, "figures/renewal_cage_translation_rotation_protocol.pdf")
        archive.write(glass_audit_pdf, "figures/renewal_cage_glass_audit.pdf")
        archive.write(glass_phase_diagram_pdf, "figures/renewal_cage_glass_phase_diagram.pdf")
        archive.write(spatial_chi4_pdf, "figures/renewal_cage_spatial_chi4.pdf")
        archive.write(thermodynamic_closure_pdf, "figures/renewal_cage_thermodynamic_closure.pdf")
        archive.write(mct_beta_closure_pdf, "figures/renewal_cage_mct_beta_closure.pdf")
        archive.write(sota_benchmark_consistency_pdf, "figures/renewal_cage_sota_benchmark_consistency.pdf")
        archive.write(sota_claim_alignment_pdf, "figures/renewal_cage_sota_claim_alignment.pdf")
        archive.write(sota_signed_constraints_pdf, "figures/renewal_cage_sota_signed_constraints.pdf")
        archive.write(sota_evidence_verdict_pdf, "figures/renewal_cage_sota_evidence_verdict.pdf")
        archive.write(sota_evidence_class_pdf, "figures/renewal_cage_sota_evidence_class.pdf")
        archive.write(simultaneous_closure_pdf, "figures/renewal_cage_simultaneous_closure.pdf")
        archive.write(
            real_benchmark_assimilation_gate_pdf,
            "figures/renewal_cage_real_benchmark_assimilation_gate.pdf",
        )
        archive.write(
            cross_observable_prediction_ledger_pdf,
            "figures/renewal_cage_cross_observable_prediction_ledger.pdf",
        )
        archive.write(
            inversion_identifiability_audit_pdf,
            "figures/renewal_cage_inversion_identifiability_audit.pdf",
        )
        archive.write(frontier_benchmark_horizon_pdf, "figures/renewal_cage_frontier_benchmark_horizon.pdf")
        archive.write(sota_source_provenance_pdf, "figures/renewal_cage_sota_source_provenance.pdf")
        archive.write(sota_data_accession_pdf, "figures/renewal_cage_sota_data_accession.pdf")
        archive.write(
            sota_zenodo_record_fingerprint_pdf,
            "figures/renewal_cage_sota_zenodo_record_fingerprint.pdf",
        )
        archive.write(
            sota_remote_zip_central_directory_pdf,
            "figures/renewal_cage_sota_remote_zip_central_directory.pdf",
        )
        archive.write(sota_glassbench_payload_index_pdf, "figures/renewal_cage_sota_glassbench_payload_index.pdf")
        archive.write(
            sota_glassbench_trajectory_payload_locator_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_payload_locator.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_entry_metadata_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_entry_metadata.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_member_stream_probe_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_member_stream_probe.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_inner_tar_header_probe_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_npz_schema_probe_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_npz_schema_probe.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_first_npz_observable_smoke_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_first_npz_observable_curve_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.pdf",
        )
        archive.write(
            sota_glassbench_short_window_trend_canary_pdf,
            "figures/renewal_cage_sota_glassbench_short_window_trend_canary.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_timebase_bridge_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_timebase_bridge.pdf",
        )
        archive.write(
            sota_glassbench_frame_time_mapping_audit_pdf,
            "figures/renewal_cage_sota_glassbench_frame_time_mapping_audit.pdf",
        )
        archive.write(
            sota_glassbench_real_inversion_gap_ledger_pdf,
            "figures/renewal_cage_sota_glassbench_real_inversion_gap_ledger.pdf",
        )
        archive.write(
            sota_glassbench_real_inversion_unlock_protocol_pdf,
            "figures/renewal_cage_sota_glassbench_real_inversion_unlock_protocol.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_first_npz_inversion_readiness_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_npz_member_index_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_npz_member_index.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_member_ensemble_observable_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.pdf",
        )
        archive.write(
            sota_glassbench_ka2d_timecode_semantics_pdf,
            "figures/renewal_cage_sota_glassbench_ka2d_timecode_semantics.pdf",
        )
        archive.write(
            sota_glassbench_timecode_curve_bridge_pdf,
            "figures/renewal_cage_sota_glassbench_timecode_curve_bridge.pdf",
        )
        archive.write(
            sota_glassbench_timecode_signature_support_pdf,
            "figures/renewal_cage_sota_glassbench_timecode_signature_support.pdf",
        )
        archive.write(
            sota_glassbench_alpha_threshold_horizon_pdf,
            "figures/renewal_cage_sota_glassbench_alpha_threshold_horizon.pdf",
        )
        archive.write(
            sota_dynamic_signature_alignment_pdf,
            "figures/renewal_cage_sota_dynamic_signature_alignment.pdf",
        )
        archive.write(
            sota_glassbench_microdynamic_closed_loop_pdf,
            "figures/renewal_cage_sota_glassbench_microdynamic_closed_loop.pdf",
        )
        archive.write(
            sota_glassbench_cage_jump_proxy_canary_pdf,
            "figures/renewal_cage_sota_glassbench_cage_jump_proxy_canary.pdf",
        )
        archive.write(
            sota_glassbench_trajectory_npz_ensemble_horizon_pdf,
            "figures/renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.pdf",
        )
        archive.write(
            sota_glassbench_visible_member_ensemble_audit_pdf,
            "figures/renewal_cage_sota_glassbench_visible_member_ensemble_audit.pdf",
        )
        archive.write(
            sota_glassbench_observable_coverage_audit_pdf,
            "figures/renewal_cage_sota_glassbench_observable_coverage_audit.pdf",
        )
        archive.write(
            sota_glassbench_first_npz_structural_observable_plan_pdf,
            "figures/renewal_cage_sota_glassbench_first_npz_structural_observable_plan.pdf",
        )
        archive.write(
            sota_remote_result_curve_cache_pdf,
            "figures/renewal_cage_sota_remote_result_curve_cache.pdf",
        )
        archive.write(
            sota_remote_result_curve_fetch_gap_pdf,
            "figures/renewal_cage_sota_remote_result_curve_fetch_gap.pdf",
        )
        archive.write(
            sota_remote_result_curve_target_fetch_pdf,
            "figures/renewal_cage_sota_remote_result_curve_target_fetch.pdf",
        )
        archive.write(
            sota_remote_result_curve_published_semantics_pdf,
            "figures/renewal_cage_sota_remote_result_curve_published_semantics.pdf",
        )
        archive.write(
            sota_remote_result_curve_payload_adapter_pdf,
            "figures/renewal_cage_sota_remote_result_curve_payload_adapter.pdf",
        )
        archive.write(
            sota_remote_result_curve_observable_semantics_pdf,
            "figures/renewal_cage_sota_remote_result_curve_observable_semantics.pdf",
        )
        archive.write(sota_readme_schema_pdf, "figures/renewal_cage_sota_readme_schema.pdf")
        archive.write(trajectory_adapter_contract_pdf, "figures/renewal_cage_trajectory_adapter_contract.pdf")
        archive.write(literature_inversion_readiness_pdf, "figures/renewal_cage_literature_inversion_readiness.pdf")
        archive.write(
            observable_falsification_matrix_pdf,
            "figures/renewal_cage_observable_falsification_matrix.pdf",
        )
        archive.write(benchmark_fusion_readiness_pdf, "figures/renewal_cage_benchmark_fusion_readiness.pdf")
        archive.write(raw_curve_ingestion_contract_pdf, "figures/renewal_cage_raw_curve_ingestion_contract.pdf")
        archive.write(raw_curve_diagnostic_readiness_pdf, "figures/renewal_cage_raw_curve_diagnostic_readiness.pdf")
        archive.write(
            raw_curve_persistence_exchange_protocol_pdf,
            "figures/renewal_cage_raw_curve_persistence_exchange_protocol.pdf",
        )
        archive.write(trajectory_observable_protocol_pdf, "figures/renewal_cage_trajectory_observable_protocol.pdf")
        archive.write(trajectory_cage_jump_events_pdf, "figures/renewal_cage_trajectory_cage_jump_events.pdf")
        archive.write(
            trajectory_event_clock_macro_predictions_pdf,
            "figures/renewal_cage_trajectory_event_clock_macro_predictions.pdf",
        )
        archive.write(
            trajectory_event_clock_threshold_robustness_pdf,
            "figures/renewal_cage_trajectory_event_clock_threshold_robustness.pdf",
        )
        archive.write(trajectory_uncertainty_protocol_pdf, "figures/renewal_cage_trajectory_uncertainty_protocol.pdf")
        archive.write(
            trajectory_member_ensemble_uncertainty_pdf,
            "figures/renewal_cage_trajectory_member_ensemble_uncertainty.pdf",
        )
        archive.write(trajectory_inversion_readiness_pdf, "figures/renewal_cage_trajectory_inversion_readiness.pdf")
        archive.write(benchmark_publication_ladder_pdf, "figures/renewal_cage_benchmark_publication_ladder.pdf")
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
