#!/usr/bin/env python3
"""Test a positive scalar-mobility Langevin closure against KA shape data."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from summarize_ka_transport_clock_shape_quotient import (
    FROZEN_PROTOCOLS,
    INDEPENDENCE_CLASS,
    SOURCE_CLAIM_FIELDS,
    _exact_int,
    _finite,
    _validated_full_path_rows,
    _validated_stationarity_pass,
    _svg_text,
    manifests_from_provenance,
    read_rows,
    write_rows,
)


WAVE_NUMBERS = {
    "fs_k2": (2.0, "observed_fs_k2"),
    "fs_k4": (4.0, "observed_fs_k4"),
    "fs_k7p25": (7.25, "observed_fs_k7p25"),
}
FS_TOLERANCE = 0.03
SIMULATION_MSD_RELATIVE_TOLERANCE = 0.015
SIMULATION_NGP_ABSOLUTE_TOLERANCE = 0.035
SIMULATION_FS_ABSOLUTE_TOLERANCE = 0.012
CLOSED_GATE_FIELDS = (
    "blind_prediction_claim_allowed",
    "finite_exchange_resolved",
    "static_environment_resolved",
    "spatial_facilitation_resolved",
    "activated_cage_geometry_resolved",
    "microdynamic_closure_claim_allowed",
    "thermodynamic_claim_allowed",
)


def _positive_finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def gaussian_fs(*, msd: float, wave_number: float) -> float:
    """Return the isotropic Gaussian characteristic function in three dimensions."""

    local_msd = _positive_finite(msd, "msd")
    local_k = _positive_finite(wave_number, "wave_number")
    return math.exp(-(local_k * local_k * local_msd) / 6.0)


def gamma_variance_fs(*, msd: float, alpha2: float, wave_number: float) -> float:
    """Return Fs for a gamma-distributed total variance coordinate."""

    local_msd = _positive_finite(msd, "msd")
    local_alpha = _nonnegative_finite(alpha2, "alpha2")
    local_k = _positive_finite(wave_number, "wave_number")
    exponent = local_k * local_k * local_msd / 6.0
    if local_alpha <= 1e-14:
        return math.exp(-exponent)
    return math.exp(-math.log1p(local_alpha * exponent) / local_alpha)


def shifted_gamma_variance_fs(
    *,
    msd: float,
    alpha2: float,
    cage_variance: float,
    wave_number: float,
) -> float:
    """Return Fs for a deterministic cage variance plus a gamma mobility variance."""

    local_msd = _positive_finite(msd, "msd")
    local_alpha = _nonnegative_finite(alpha2, "alpha2")
    local_k = _positive_finite(wave_number, "wave_number")
    local_cage = _nonnegative_finite(cage_variance, "cage_variance")
    mean_variance = local_msd / 6.0
    if local_cage > mean_variance:
        raise ValueError("cage_variance cannot exceed total mean variance")
    if local_alpha <= 1e-14 or mean_variance - local_cage <= 1e-14:
        return math.exp(-(local_k * local_k * mean_variance))
    mobile_mean = mean_variance - local_cage
    variance = local_alpha * mean_variance * mean_variance
    shape = mobile_mean * mobile_mean / variance
    scale = variance / mobile_mean
    return math.exp(
        -(local_k * local_k * local_cage)
        - shape * math.log1p(local_k * local_k * scale)
    )


def infer_cage_variance(
    *,
    msd: float,
    alpha2: float,
    fs_k2: float,
) -> float | None:
    """Invert the shifted-gamma k=2 relation only inside its exact bracket."""

    local_msd = _positive_finite(msd, "msd")
    local_alpha = _positive_finite(alpha2, "alpha2")
    try:
        target = float(fs_k2)
    except (TypeError, ValueError) as error:
        raise ValueError("fs_k2 must lie strictly between zero and one") from error
    if not math.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("fs_k2 must lie strictly between zero and one")
    low = 0.0
    high = local_msd / 6.0
    low_value = shifted_gamma_variance_fs(
        msd=local_msd,
        alpha2=local_alpha,
        cage_variance=low,
        wave_number=2.0,
    )
    high_value = shifted_gamma_variance_fs(
        msd=local_msd,
        alpha2=local_alpha,
        cage_variance=high,
        wave_number=2.0,
    )
    bracket_low = min(low_value, high_value)
    bracket_high = max(low_value, high_value)
    if target < bracket_low or target > bracket_high:
        return None
    if target == low_value:
        return low
    if target == high_value:
        return high
    increasing = high_value > low_value
    for _ in range(80):
        middle = 0.5 * (low + high)
        middle_value = shifted_gamma_variance_fs(
            msd=local_msd,
            alpha2=local_alpha,
            cage_variance=middle,
            wave_number=2.0,
        )
        if (middle_value < target) == increasing:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def compute_gamma_variance_mixture_rows(
    source_rows: Sequence[dict[str, object]],
    manifest: dict[str, object],
    *,
    temperature: float,
    full_length: int,
    expected_lags: Sequence[int],
) -> list[dict[str, object]]:
    """Compute one diagnostic row per full-path replicate and lag."""

    selected, replicates, lags = _validated_full_path_rows(
        source_rows,
        manifest,
        temperature=temperature,
        full_length=full_length,
        expected_lags=expected_lags,
    )
    indexed = {
        (_exact_int(row, "replicate"), _exact_int(row, "lag")): row
        for row in selected
    }
    result: list[dict[str, object]] = []
    for replicate in replicates:
        for lag in lags:
            source = indexed[(replicate, lag)]
            msd = _finite(source, "observed_msd")
            alpha2 = _finite(source, "observed_ngp")
            if msd <= 0.0 or alpha2 < 0.0:
                raise ValueError("variance-mixture inputs require positive MSD and nonnegative NGP")
            observed = {
                observable: _finite(source, field)
                for observable, (_, field) in WAVE_NUMBERS.items()
            }
            if not 0.0 < observed["fs_k2"] < 1.0 or any(
                abs(value) > 1.0 for value in observed.values()
            ):
                raise ValueError("observed scattering values violate characteristic bounds")
            cage_variance = infer_cage_variance(
                msd=msd,
                alpha2=alpha2,
                fs_k2=observed["fs_k2"],
            )
            row: dict[str, object] = {
                "temperature": temperature,
                "replicate": float(replicate),
                "lag": float(lag),
                "heldout_msd": msd,
                "heldout_ngp": alpha2,
                "cage_variance_root_supported": float(cage_variance is not None),
                "inferred_cage_variance": "" if cage_variance is None else cage_variance,
                "inferred_cage_fraction": (
                    "" if cage_variance is None else cage_variance / (msd / 6.0)
                ),
                "heldout_msd_used_as_diagnostic_input": 1.0,
                "heldout_ngp_used_as_diagnostic_input": 1.0,
                "heldout_fs_k2_used_for_cage_diagnostic": 1.0,
                "blind_prediction_claim_allowed": 0.0,
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
            for observable, (wave_number, _) in WAVE_NUMBERS.items():
                gaussian = gaussian_fs(msd=msd, wave_number=wave_number)
                gamma = gamma_variance_fs(
                    msd=msd,
                    alpha2=alpha2,
                    wave_number=wave_number,
                )
                row[f"observed_{observable}"] = observed[observable]
                row[f"gaussian_{observable}"] = gaussian
                row[f"gamma_{observable}"] = gamma
                row[f"gaussian_{observable}_normalized_error"] = (
                    abs(gaussian - observed[observable]) / FS_TOLERANCE
                )
                row[f"gamma_{observable}_normalized_error"] = (
                    abs(gamma - observed[observable]) / FS_TOLERANCE
                )
                if observable == "fs_k2" or cage_variance is None:
                    shifted: float | str = ""
                    shifted_error: float | str = ""
                else:
                    shifted = shifted_gamma_variance_fs(
                        msd=msd,
                        alpha2=alpha2,
                        cage_variance=cage_variance,
                        wave_number=wave_number,
                    )
                    shifted_error = abs(shifted - observed[observable]) / FS_TOLERANCE
                row[f"cage_gamma_{observable}"] = shifted
                row[f"cage_gamma_{observable}_normalized_error"] = shifted_error
            result.append(row)
    return result


def classify_gamma_variance_mixture_gate(
    rows: Sequence[dict[str, object]],
    stationarity_rows: dict[float, Sequence[dict[str, object]]],
    manifests: dict[float, dict[str, object]],
    *,
    minimum_cage_support_fraction: float = 0.80,
) -> list[dict[str, object]]:
    """Classify scalar-mobility closure while keeping all mechanism claims closed."""

    support_threshold = _positive_finite(
        minimum_cage_support_fraction,
        "minimum_cage_support_fraction",
    )
    if support_threshold > 1.0:
        raise ValueError("minimum_cage_support_fraction cannot exceed one")
    temperatures = (0.45, 0.58)
    if set(stationarity_rows) != set(temperatures) or set(manifests) != set(temperatures):
        raise ValueError("gate requires exact T045/T058 controls")
    result: list[dict[str, object]] = []
    all_temperatures: set[float] = set()
    for temperature in temperatures:
        manifest = manifests[temperature]
        replicates = tuple(
            _exact_int(row, "replicate") for row in manifest["replicates"]
        )
        lags = tuple(int(value) for value in manifest["expected_lags"])
        local = [row for row in rows if _finite(row, "temperature") == temperature]
        keys = [(_exact_int(row, "replicate"), _exact_int(row, "lag")) for row in local]
        expected = {(replicate, lag) for replicate in replicates for lag in lags}
        if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError("diagnostic rows do not form the frozen replicate-lag grid")
        all_temperatures.add(temperature)
        for row in local:
            for field in (
                "heldout_msd_used_as_diagnostic_input",
                "heldout_ngp_used_as_diagnostic_input",
                "heldout_fs_k2_used_for_cage_diagnostic",
            ):
                if _finite(row, field) != 1.0:
                    raise ValueError("diagnostic-input disclosure is incomplete")
            for field in SOURCE_CLAIM_FIELDS:
                if _finite(row, field) != 0.0:
                    raise ValueError("source claim boundaries must remain closed")
        stationarity_pass = _validated_stationarity_pass(
            stationarity_rows[temperature],
            temperature=temperature,
        )
        gamma_max = {
            observable: max(
                _finite(row, f"gamma_{observable}_normalized_error") for row in local
            )
            for observable in WAVE_NUMBERS
        }
        gaussian_max = {
            observable: max(
                _finite(row, f"gaussian_{observable}_normalized_error") for row in local
            )
            for observable in WAVE_NUMBERS
        }
        support_by_replicate = {
            replicate: sum(
                _finite(row, "cage_variance_root_supported") == 1.0
                for row in local
                if _exact_int(row, "replicate") == replicate
            )
            / len(lags)
            for replicate in replicates
        }
        supported = [
            row for row in local if _finite(row, "cage_variance_root_supported") == 1.0
        ]
        cage_max = {
            observable: (
                max(
                    _finite(row, f"cage_gamma_{observable}_normalized_error")
                    for row in supported
                )
                if supported
                else ""
            )
            for observable in ("fs_k4", "fs_k7p25")
        }
        support_pass = min(support_by_replicate.values()) >= support_threshold
        scalar_pass = (
            temperature == 0.45
            and stationarity_pass
            and all(value <= 1.0 for value in gamma_max.values())
        )
        low_intermediate_only = (
            temperature == 0.45
            and stationarity_pass
            and gamma_max["fs_k2"] <= 1.0
            and gamma_max["fs_k4"] <= 1.0
            and gamma_max["fs_k7p25"] > 1.0
        )
        cage_pass = (
            temperature == 0.45
            and stationarity_pass
            and support_pass
            and all(float(value) <= 1.0 for value in cage_max.values())
        )
        gate: dict[str, object] = {
            "temperature": temperature,
            "analysis_status": (
                "scalar_mobility_cage_scale_residual"
                if low_intermediate_only
                else "high_temperature_canary_only"
                if temperature == 0.58
                else "variance_mixture_unresolved"
            ),
            "replicate_count": float(len(replicates)),
            "lag_count_per_replicate": float(len(lags)),
            "source_ensemble_stationarity_all_comparisons_pass": float(stationarity_pass),
            "replicate_provenance_validation_pass": 1.0,
            "parent_sample_count": 1.0,
            "independent_replicate_count": 0.0,
            "independently_prepared_parent_samples": 0.0,
            "independence_class": INDEPENDENCE_CLASS,
            "minimum_cage_root_support_fraction": min(support_by_replicate.values()),
            "total_cage_root_supported_row_count": float(len(supported)),
            "total_row_count": float(len(local)),
            "cage_plus_mobility_support_coverage_pass": float(support_pass),
        }
        for observable in WAVE_NUMBERS:
            gate[f"{observable}_gaussian_max_normalized_error"] = gaussian_max[observable]
            gate[f"{observable}_gamma_max_normalized_error"] = gamma_max[observable]
        for observable in ("fs_k4", "fs_k7p25"):
            gate[f"{observable}_cage_gamma_supported_max_normalized_error"] = cage_max[observable]
        gate.update(
            {
                "scalar_mobility_shape_closure_supported_exploratory": float(scalar_pass),
                "scalar_mobility_low_intermediate_k_supported_exploratory": float(
                    low_intermediate_only
                ),
                "cage_plus_mobility_shape_closure_supported_exploratory": float(cage_pass),
                "high_temperature_canary_only": float(temperature == 0.58),
                "high_temperature_control_resolved": 0.0,
                "next_required_action": (
                    "test_non_gamma_tail_or_activated_transient_potential_with_independent_parents"
                ),
                **{field: 0.0 for field in CLOSED_GATE_FIELDS},
            }
        )
        result.append(gate)
    if all_temperatures != set(temperatures) or {
        _finite(row, "temperature") for row in rows
    } != set(temperatures):
        raise ValueError("diagnostic table contains an unexpected temperature")
    return result


def simulate_squared_ou_mobility(
    *,
    tau_ratios: Sequence[float],
    particle_count: int,
    step_count: int,
    seed: int,
    mobility_dimension: int = 4,
    mean_diffusivity: float = 1.0,
    observation_time: float = 1.0,
) -> list[dict[str, float]]:
    """Simulate a stationary squared-OU mobility and conditional Brownian endpoint."""

    if (
        isinstance(particle_count, bool)
        or not isinstance(particle_count, int)
        or particle_count < 100
        or isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 2
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(mobility_dimension, bool)
        or not isinstance(mobility_dimension, int)
        or mobility_dimension < 1
    ):
        raise ValueError("simulation counts, seed, and mobility dimension are invalid")
    local_mean = _positive_finite(mean_diffusivity, "mean_diffusivity")
    local_time = _positive_finite(observation_time, "observation_time")
    ratios = tuple(_positive_finite(value, "tau_D_over_t") for value in tau_ratios)
    if not ratios or len(set(ratios)) != len(ratios):
        raise ValueError("tau_ratios must be unique and nonempty")
    generator = np.random.default_rng(seed)
    component_variance = local_mean / mobility_dimension
    dt = local_time / step_count
    analytic_msd = 6.0 * local_mean * local_time
    analytic_ngp = 2.0 / mobility_dimension
    result: list[dict[str, float]] = []
    for ratio in ratios:
        tau_d = ratio * local_time
        rho = math.exp(-dt / tau_d)
        innovation_scale = math.sqrt(component_variance * (1.0 - rho * rho))
        mobility = generator.normal(
            0.0,
            math.sqrt(component_variance),
            size=(particle_count, mobility_dimension),
        )
        diffusivity = np.sum(mobility * mobility, axis=1)
        integrated = np.zeros(particle_count, dtype=float)
        for _ in range(step_count):
            mobility = rho * mobility + innovation_scale * generator.normal(
                size=mobility.shape
            )
            next_diffusivity = np.sum(mobility * mobility, axis=1)
            integrated += 0.5 * (diffusivity + next_diffusivity) * dt
            diffusivity = next_diffusivity
        displacement = generator.normal(size=(particle_count, 3)) * np.sqrt(
            2.0 * integrated
        )[:, None]
        radius_squared = np.sum(displacement * displacement, axis=1)
        empirical_msd = float(np.mean(radius_squared))
        empirical_ngp = float(
            3.0 * np.mean(radius_squared * radius_squared)
            / (5.0 * empirical_msd * empirical_msd)
            - 1.0
        )
        row: dict[str, float] = {
            "tau_D_over_t": ratio,
            "particle_count": float(particle_count),
            "step_count": float(step_count),
            "mobility_dimension": float(mobility_dimension),
            "analytic_msd": analytic_msd,
            "empirical_msd": empirical_msd,
            "analytic_ngp": analytic_ngp,
            "empirical_ngp": empirical_ngp,
            "msd_relative_error": abs(empirical_msd / analytic_msd - 1.0),
            "ngp_absolute_error": abs(empirical_ngp - analytic_ngp),
        }
        fs_errors: list[float] = []
        for wave_number in (0.5, 1.0, 2.0):
            suffix = str(wave_number).replace(".", "p")
            row[f"analytic_fs_k{suffix}"] = gamma_variance_fs(
                msd=analytic_msd,
                alpha2=analytic_ngp,
                wave_number=wave_number,
            )
            row[f"empirical_fs_k{suffix}"] = float(
                np.mean(np.cos(wave_number * displacement[:, 0]))
            )
            error = abs(
                row[f"empirical_fs_k{suffix}"] - row[f"analytic_fs_k{suffix}"]
            )
            row[f"fs_k{suffix}_absolute_error"] = error
            fs_errors.append(error)
        row["maximum_fs_absolute_error"] = max(fs_errors)
        row["slow_environment_limit_validation_pass"] = float(
            ratio >= 100.0
            and row["msd_relative_error"] <= SIMULATION_MSD_RELATIVE_TOLERANCE
            and row["ngp_absolute_error"] <= SIMULATION_NGP_ABSOLUTE_TOLERANCE
            and row["maximum_fs_absolute_error"] <= SIMULATION_FS_ABSOLUTE_TOLERANCE
        )
        result.append(row)
    return result


def write_gamma_variance_mixture_svg(
    path: Path,
    gates: Sequence[dict[str, object]],
    simulation_rows: Sequence[dict[str, object]],
) -> None:
    """Write a deterministic three-panel diagnostic figure."""

    if {float(gate["temperature"]) for gate in gates} != {0.45, 0.58}:
        raise ValueError("SVG requires exact T045/T058 gates")
    if [float(row["tau_D_over_t"]) for row in simulation_rows] != [1.0, 10.0, 100.0]:
        raise ValueError("SVG requires the frozen squared-OU tau grid")
    width = 1120
    height = 650
    panel_top = 125.0
    panel_height = 300.0
    panel_width = 300.0
    panel_lefts = (70.0, 395.0, 720.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    _svg_text(parts, 70, 42, "Gamma variance-mixture Langevin diagnostic", size=25, weight=700)
    _svg_text(
        parts,
        70,
        69,
        "MSD + NGP predict multi-k scattering; squared-OU mobility validates the slow-field limit",
        size=14,
        fill="#46515c",
    )
    _svg_text(
        parts,
        70,
        96,
        "maximum normalized Fs error (tolerance units)",
        size=12,
        fill="#46515c",
    )
    for panel_index, temperature in enumerate((0.45, 0.58)):
        gate = next(row for row in gates if float(row["temperature"]) == temperature)
        left = panel_lefts[panel_index]
        parts.append(
            f'<rect x="{left:.2f}" y="{panel_top:.2f}" width="{panel_width:.2f}" '
            f'height="{panel_height:.2f}" fill="#fbfcfd" stroke="#c7ced5"/>'
        )
        title = "T=0.45 scalar mobility" if temperature == 0.45 else "T=0.58 canary only"
        _svg_text(parts, left + 16, panel_top + 28, title, size=16, weight=700)
        values = [
            float(gate[f"{observable}_gamma_max_normalized_error"])
            for observable in WAVE_NUMBERS
        ]
        y_max = max(1.2, 1.1 * max(values))
        plot_left = left + 42.0
        plot_right = left + panel_width - 18.0
        plot_top = panel_top + 55.0
        plot_bottom = panel_top + 245.0
        for tick in (0.0, 1.0, y_max):
            y = plot_bottom - (tick / y_max) * (plot_bottom - plot_top)
            parts.append(
                f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" '
                f'y2="{y:.2f}" stroke="{("#a43b32" if tick == 1.0 else "#e2e6ea")}" '
                f'stroke-width="{("1.5" if tick == 1.0 else "1")}" '
                f'{("stroke-dasharray=\"5 4\"" if tick == 1.0 else "")}/>'
            )
            _svg_text(
                parts,
                plot_left - 7,
                y + 4,
                f"{tick:.2g}",
                size=11,
                anchor="end",
                fill="#5b6570",
            )
        spacing = (plot_right - plot_left) / 3.0
        for index, (label, value) in enumerate(zip(("k=2", "k=4", "k=7.25"), values)):
            center = plot_left + spacing * (index + 0.5)
            bar_height = (value / y_max) * (plot_bottom - plot_top)
            color = "#2f7d68" if value <= 1.0 else "#b34a42"
            parts.append(
                f'<rect x="{center - 13:.2f}" y="{plot_bottom - bar_height:.2f}" '
                f'width="26" height="{bar_height:.2f}" fill="{color}"/>'
            )
            _svg_text(parts, center, plot_bottom + 20, label, size=11, anchor="middle")
        _svg_text(
            parts,
            left + 16,
            panel_top + 282,
            (
                f'cage-root support: {int(float(gate["total_cage_root_supported_row_count"]))}/'
                f'{int(float(gate["total_row_count"]))}'
            ),
            size=11,
            fill="#46515c",
        )

    left = panel_lefts[2]
    parts.append(
        f'<rect x="{left:.2f}" y="{panel_top:.2f}" width="{panel_width:.2f}" '
        f'height="{panel_height:.2f}" fill="#fbfcfd" stroke="#c7ced5"/>'
    )
    _svg_text(parts, left + 16, panel_top + 28, "squared-OU Langevin", size=16, weight=700)
    _svg_text(parts, left + 16, panel_top + 48, "NGP convergence to slow-field gamma limit", size=11, fill="#46515c")
    plot_left = left + 45.0
    plot_right = left + panel_width - 20.0
    plot_top = panel_top + 68.0
    plot_bottom = panel_top + 245.0
    empirical_ngps = [float(row["empirical_ngp"]) for row in simulation_rows]
    analytic = float(simulation_rows[-1]["analytic_ngp"])
    y_min = min(0.0, 1.1 * min(empirical_ngps))
    y_max = max(0.55, 1.1 * max(empirical_ngps + [analytic]))

    def simulation_y(value: float) -> float:
        return plot_bottom - ((value - y_min) / (y_max - y_min)) * (
            plot_bottom - plot_top
        )

    ticks = sorted({y_min, 0.0, analytic, y_max})
    for tick in ticks:
        y = simulation_y(tick)
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{y:.2f}" x2="{plot_right:.2f}" '
            f'y2="{y:.2f}" stroke="#e2e6ea" stroke-width="1"/>'
        )
        _svg_text(parts, plot_left - 7, y + 4, f"{tick:g}", size=11, anchor="end", fill="#5b6570")
    analytic_y = simulation_y(analytic)
    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{analytic_y:.2f}" x2="{plot_right:.2f}" '
        f'y2="{analytic_y:.2f}" stroke="#a43b32" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    points = []
    for index, row in enumerate(simulation_rows):
        x = plot_left + index * (plot_right - plot_left) / 2.0
        empirical = float(row["empirical_ngp"])
        y = simulation_y(empirical)
        if not math.isfinite(y) or y < plot_top or y > plot_bottom:
            raise ValueError("simulation NGP lies outside the frozen SVG axis")
        points.append(f"{x:.2f},{y:.2f}")
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2f7d68"/>')
        _svg_text(
            parts,
            x,
            plot_bottom + 20,
            f'{float(row["tau_D_over_t"]):g}',
            size=11,
            anchor="middle",
        )
    parts.append(
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#2f7d68" stroke-width="2"/>'
    )
    _svg_text(parts, left + 16, panel_top + 282, "x: tau_D / observation time", size=11, fill="#46515c")
    parts.append('<rect x="70" y="465" width="950" height="145" fill="#f4f6f7" stroke="#c7ced5"/>')
    _svg_text(parts, 92, 500, "At T=0.45, scalar mobility fails at cage scale (k=7.25).", size=15, weight=700)
    _svg_text(parts, 92, 530, "heldout MSD and NGP are diagnostic inputs; this is not a blind prediction", size=14)
    _svg_text(parts, 92, 558, "cage + mobility inversion has insufficient root support; residual mechanism unresolved", size=14, fill="#9b3a32")
    _svg_text(parts, 92, 586, "T=0.58 canary only; no cooling, spatial, microscopic-closure, or thermodynamic claim", size=13, fill="#46515c")
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the frozen gamma variance-mixture Langevin diagnostic."
    )
    parser.add_argument("--low-rows", type=Path, required=True)
    parser.add_argument("--high-rows", type=Path, required=True)
    parser.add_argument("--low-stationarity", type=Path, required=True)
    parser.add_argument("--high-stationarity", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-gate", type=Path, required=True)
    parser.add_argument("--output-simulation", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--simulation-particles", type=int, default=80_000)
    parser.add_argument("--simulation-steps", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = manifests_from_provenance(read_rows(args.provenance))
    source_tables = {
        0.45: read_rows(args.low_rows),
        0.58: read_rows(args.high_rows),
    }
    stationarity = {
        0.45: read_rows(args.low_stationarity),
        0.58: read_rows(args.high_stationarity),
    }
    rows: list[dict[str, object]] = []
    for temperature in (0.45, 0.58):
        protocol = FROZEN_PROTOCOLS[temperature]
        rows.extend(
            compute_gamma_variance_mixture_rows(
                source_tables[temperature],
                manifests[temperature],
                temperature=temperature,
                full_length=int(protocol["full_length"]),
                expected_lags=protocol["lags"],
            )
        )
    gates = classify_gamma_variance_mixture_gate(rows, stationarity, manifests)
    simulation_rows = simulate_squared_ou_mobility(
        tau_ratios=(1.0, 10.0, 100.0),
        particle_count=args.simulation_particles,
        step_count=args.simulation_steps,
        seed=20260718,
    )
    write_rows(args.output_rows, rows)
    write_rows(args.output_gate, gates)
    write_rows(args.output_simulation, simulation_rows)
    write_gamma_variance_mixture_svg(args.output_svg, gates, simulation_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
