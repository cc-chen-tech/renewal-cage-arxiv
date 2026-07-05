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
    ActivatedBarrierParams,
    DelayedRenewalCageParams,
    FacilitatedExchangeLawParams,
    GammaExchangeParams,
    TemperatureLawParams,
    alpha_relaxation_shape_curve,
    alpha_relaxation_time,
    alpha_shape_superposition_residual,
    activated_barrier_temperature_law,
    correlated_domain_susceptibility,
    delayed_poisson_mean,
    delayed_renewal_shape,
    dimensionless_peak_prediction,
    gaussian_radial_3d,
    gamma_exchange_asymptotic_diagnostics,
    gamma_exchange_count_moments,
    gamma_exchange_diagnostic_map,
    gamma_exchange_temperature_scan,
    infer_gamma_exchange_multik_collapse,
    infer_gamma_exchange_ratio_from_alpha_rate,
    infer_gamma_exchange_uncertainty_from_late_observables,
    gamma_exchange_ngp_1d,
    gamma_exchange_normalized_alpha_decay,
    gamma_exchange_scattering_susceptibility,
    glass_phenomenon_audit,
    glass_signature_phase_diagram,
    infer_parameters_from_full_observables,
    infer_parameters_from_scattering_transport,
    infer_renewal_correlation_size,
    late_mechanism_selection,
    local_alpha_stretching_exponent,
    moments_1d,
    ngp_1d,
    normalized_alpha_decay,
    observable_consistency_diagnostics,
    plateau_peak_diagnostics,
    peak_relaxation_coupling,
    radial_van_hove_3d,
    renewal_scattering_susceptibility,
    self_intermediate_scattering,
    static_gamma_asymptotic_diagnostics,
    static_gamma_ngp_1d,
    static_gamma_normalized_alpha_decay,
    temperature_dependent_params,
    temperature_scan,
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
    segments: list[str] = []
    current: list[str] = []
    for xi, yi in zip(x, y):
        if np.isfinite(xi) and np.isfinite(yi):
            current.append(f"{xi:.2f},{yi:.2f}")
        elif len(current) > 1:
            segments.append(" ".join(current))
            current = []
        else:
            current = []
    if len(current) > 1:
        segments.append(" ".join(current))
    return "\n".join(
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" />'
        for coords in segments
    )


def write_main_csv(path: Path, time: np.ndarray, params: DelayedRenewalCageParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    moments = moments_1d(time, params)
    alpha = ngp_1d(time, params)
    with path.open("w", newline="") as f:
        fieldnames = ["time", "local_variance", "renewal_mean", "msd", "m4", "ngp_1d"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for idx, r in enumerate(radius):
            row: dict[str, float] = {"radius": float(r)}
            for label, density in curves:
                row[label] = float(density[idx])
            writer.writerow(row)


def write_scattering_csv(
    path: Path,
    time: np.ndarray,
    params: DelayedRenewalCageParams,
    wave_numbers: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for wave_number in wave_numbers:
        scattering = self_intermediate_scattering(wave_number, time, params)
        alpha_decay = normalized_alpha_decay(wave_number, time, params)
        plateau = float(np.exp(-0.5 * wave_number**2 * params.cage_variance))
        alpha_rate = params.renewal_rate * (1.0 - float(np.exp(-0.5 * wave_number**2 * params.jump_variance)))
        for idx, value in enumerate(time):
            rows.append(
                {
                    "time": float(value),
                    "wave_number": wave_number,
                    "self_intermediate_scattering": float(scattering[idx]),
                    "normalized_alpha_decay": float(alpha_decay[idx]),
                    "debye_waller_plateau": plateau,
                    "long_time_alpha_rate": alpha_rate,
                }
            )
    write_sweep_csv(path, rows)


def write_peak_relaxation_csv(
    path: Path,
    params: DelayedRenewalCageParams,
    wave_numbers: list[float],
) -> list[dict[str, float]]:
    rows = []
    for wave_number in wave_numbers:
        rows.append(peak_relaxation_coupling(wave_number, params))
    write_sweep_csv(path, rows)
    return rows


def write_temperature_csv(
    path: Path,
    temperatures: np.ndarray,
    law: TemperatureLawParams,
    *,
    wave_number: float,
) -> list[dict[str, float]]:
    rows = temperature_scan(temperatures, law, wave_number=wave_number)
    write_sweep_csv(path, rows)
    return rows


def write_alpha_shape_csv(
    path: Path,
    scaled_time: np.ndarray,
    temperatures: list[float],
    law: TemperatureLawParams,
    *,
    wave_number: float,
) -> list[dict[str, float | str]]:
    reference_temperature = temperatures[0]
    reference_params = temperature_dependent_params(reference_temperature, law)
    reference_curve = alpha_relaxation_shape_curve(wave_number, reference_params, scaled_time)
    rows: list[dict[str, float | str]] = []
    for temperature in temperatures:
        params = temperature_dependent_params(temperature, law)
        curve = alpha_relaxation_shape_curve(wave_number, params, scaled_time)
        residual = alpha_shape_superposition_residual(
            wave_number,
            reference_params,
            params,
            scaled_time,
        )
        inverse_shift = 1.0 / temperature - 1.0 / reference_temperature
        for idx, value in enumerate(scaled_time):
            rows.append(
                {
                    "temperature_label": f"T={temperature:.2f}",
                    "temperature": temperature,
                    "inverse_temperature_shift": inverse_shift,
                    "scaled_time": float(value),
                    "alpha_shape": float(curve[idx]),
                    "reference_alpha_shape": float(reference_curve[idx]),
                    "log_shape_residual": float(np.log(curve[idx]) - np.log(reference_curve[idx])),
                    "shape_control": residual["candidate_control"],
                    "reference_control": residual["reference_control"],
                    "tau_alpha_over_delay": residual["candidate_tau_alpha_over_delay"],
                    "rms_log_shape_residual": residual["rms_log_shape_residual"],
                    "max_abs_log_shape_residual": residual["max_abs_log_shape_residual"],
                }
            )
    write_sweep_csv(path, rows)
    return rows


def write_facilitated_exchange_csv(
    path: Path,
    temperatures: np.ndarray,
    cage_law: TemperatureLawParams,
    exchange_law: FacilitatedExchangeLawParams,
    *,
    wave_number: float,
) -> list[dict[str, float]]:
    rows = gamma_exchange_temperature_scan(
        temperatures,
        cage_law,
        exchange_law,
        wave_number=wave_number,
    )
    reference_ratio = rows[0]["heterogeneity_ratio"]
    reference_late_alpha = rows[0]["late_alpha_decay_per_renewal"]
    for row in rows:
        row["inverse_temperature_shift"] = 1.0 / row["temperature"] - 1.0 / cage_law.reference_temperature
        row["heterogeneity_ratio_over_hot"] = row["heterogeneity_ratio"] / reference_ratio
        row["late_alpha_decay_per_renewal_over_hot"] = (
            row["late_alpha_decay_per_renewal"] / reference_late_alpha
        )
    write_sweep_csv(path, rows)
    return rows


def write_glass_audit_csv(
    path: Path,
    temperatures: np.ndarray,
    cage_law: TemperatureLawParams,
    exchange_law: FacilitatedExchangeLawParams,
    *,
    wave_number: float,
) -> list[dict[str, float | str]]:
    audit = glass_phenomenon_audit(
        temperatures,
        cage_law,
        exchange_law,
        wave_number=wave_number,
    )
    rows: list[dict[str, float | str]] = []
    for row in audit["rows"]:
        out: dict[str, float | str] = {
            "record_type": "temperature_row",
            "signature": "",
            "flag_value": np.nan,
        }
        out.update(row)
        rows.append(out)
    for key, value in audit.items():
        if key == "rows":
            continue
        rows.append(
            {
                "record_type": "summary_flag",
                "signature": key,
                "temperature": np.nan,
                "inverse_temperature_shift": np.nan,
                "diffusion_coefficient": np.nan,
                "tau_alpha_poisson": np.nan,
                "tau_alpha_exchange": np.nan,
                "normalized_stokes_einstein_product": np.nan,
                "ngp_peak_time": np.nan,
                "ngp_peak_height": np.nan,
                "heterogeneity_ratio": np.nan,
                "late_ngp_renewal_amplitude": np.nan,
                "alpha_rate_renormalization": np.nan,
                "median_alpha_window_beta": np.nan,
                "chi4_peak": np.nan,
                "late_ngp": np.nan,
                "later_ngp": np.nan,
                "gaussian_recovery_ratio": np.nan,
                "apparent_alpha_activation_energy": np.nan,
                "local_fragility_index": np.nan,
                "flag_value": value,
            }
        )
    write_sweep_csv(path, rows)
    return rows


def write_glass_phase_diagram_csv(
    path: Path,
    temperatures: np.ndarray,
    base_cage_law: TemperatureLawParams,
    base_exchange_law: FacilitatedExchangeLawParams,
    *,
    wave_number: float,
    delay_barrier_gaps: list[float],
    exchange_barrier_sums: list[float],
) -> list[dict[str, float]]:
    rows = glass_signature_phase_diagram(
        temperatures,
        base_cage_law,
        base_exchange_law,
        wave_number=wave_number,
        delay_barrier_gaps=delay_barrier_gaps,
        exchange_barrier_sums=exchange_barrier_sums,
    )
    write_sweep_csv(path, rows)
    return rows


def write_barrier_csv(
    path: Path,
    temperatures: np.ndarray,
    *,
    base_barrier: ActivatedBarrierParams,
    wave_number: float,
    gap_values: list[float],
) -> list[dict[str, float]]:
    rows = []
    reference_gap = gap_values[0]
    reference_final_product = None
    for gap in gap_values:
        barrier = ActivatedBarrierParams(
            reference_temperature=base_barrier.reference_temperature,
            cage_variance_ref=base_barrier.cage_variance_ref,
            cage_tau_ref=base_barrier.cage_tau_ref,
            jump_to_cage_ref=base_barrier.jump_to_cage_ref,
            renewal_rate_ref=base_barrier.renewal_rate_ref,
            renewal_delay_ref=base_barrier.renewal_delay_ref,
            renewal_rate_barrier=base_barrier.renewal_rate_barrier,
            delay_onset_barrier=base_barrier.renewal_rate_barrier + gap,
            cage_stiffening_barrier=base_barrier.cage_stiffening_barrier,
            jump_to_cage_barrier=base_barrier.jump_to_cage_barrier,
            cage_tau_barrier=base_barrier.cage_tau_barrier,
        )
        law = activated_barrier_temperature_law(barrier)
        scan = temperature_scan(temperatures, law, wave_number=wave_number)
        final_product = scan[-1]["normalized_stokes_einstein_product"]
        if reference_final_product is None:
            reference_final_product = final_product
        for row in scan:
            out = dict(row)
            out["barrier_gap"] = gap
            out["relative_gap"] = gap - reference_gap
            out["cold_product_ratio_vs_reference_gap"] = final_product / reference_final_product
            rows.append(out)
    write_sweep_csv(path, rows)
    return rows


def write_susceptibility_csv(
    path: Path,
    time: np.ndarray,
    params: DelayedRenewalCageParams,
    wave_numbers: list[float],
) -> list[tuple[str, np.ndarray]]:
    curves = []
    rows = []
    for wave_number in wave_numbers:
        susceptibility = renewal_scattering_susceptibility(wave_number, time, params)
        curves.append((f"k={wave_number:g}", susceptibility))
        for idx, value in enumerate(time):
            rows.append(
                {
                    "time": float(value),
                    "wave_number": wave_number,
                    "renewal_scattering_susceptibility": float(susceptibility[idx]),
                }
            )
    write_sweep_csv(path, rows)
    return curves


def write_chi4_bridge_csv(
    path: Path,
    time: np.ndarray,
    params: DelayedRenewalCageParams,
    *,
    wave_number: float,
    correlation_sizes: list[float],
    synthetic_correlation_size: float,
) -> dict[str, float]:
    rows = []
    single_particle = renewal_scattering_susceptibility(wave_number, time, params)
    observed_peak = synthetic_correlation_size * float(np.max(single_particle))
    inferred = infer_renewal_correlation_size(
        observed_chi4_peak=observed_peak,
        wave_number=wave_number,
        t=time,
        params=params,
    )
    for correlation_size in correlation_sizes:
        chi4 = correlated_domain_susceptibility(
            wave_number,
            time,
            params,
            correlation_size=correlation_size,
        )
        for idx, value in enumerate(time):
            rows.append(
                {
                    "time": float(value),
                    "wave_number": wave_number,
                    "correlation_size": correlation_size,
                    "renewal_chi4": float(chi4[idx]),
                    "single_particle_chi_R": float(single_particle[idx]),
                    "synthetic_observed_chi4_peak": observed_peak,
                    "inferred_correlation_size": inferred["correlation_size"],
                    "model_peak_time": inferred["peak_time"],
                }
            )
    write_sweep_csv(path, rows)
    return inferred


def write_heterogeneity_csv(
    path: Path,
    time: np.ndarray,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
    *,
    wave_number: float,
) -> list[dict[str, float]]:
    renewal = delayed_poisson_mean(time, params)
    count = gamma_exchange_count_moments(time, params, heterogeneity)
    poisson_decay = normalized_alpha_decay(wave_number, time, params)
    gamma_decay = gamma_exchange_normalized_alpha_decay(wave_number, time, params, heterogeneity)
    poisson_alpha = ngp_1d(time, params)
    gamma_alpha = gamma_exchange_ngp_1d(time, params, heterogeneity)
    poisson_chi = renewal_scattering_susceptibility(wave_number, time, params)
    gamma_chi = gamma_exchange_scattering_susceptibility(wave_number, time, params, heterogeneity)
    stretching = local_alpha_stretching_exponent(time, gamma_decay)
    rows = []
    for idx, value in enumerate(time):
        rows.append(
            {
                "time": float(value),
                "log10_time": float(np.log10(value)),
                "renewal_mean": float(renewal[idx]),
                "effective_shape": float(count["effective_shape"][idx]),
                "poisson_alpha_decay": float(poisson_decay[idx]),
                "gamma_exchange_alpha_decay": float(gamma_decay[idx]),
                "poisson_ngp": float(poisson_alpha[idx]),
                "gamma_exchange_ngp": float(gamma_alpha[idx]),
                "gamma_exchange_local_beta": float(stretching[idx]),
                "poisson_scattering_susceptibility": float(poisson_chi[idx]),
                "gamma_exchange_scattering_susceptibility": float(gamma_chi[idx]),
            }
        )
    write_sweep_csv(path, rows)
    return rows


def write_heterogeneity_diagnostics_csv(
    path: Path,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
    *,
    wave_number: float,
) -> dict[str, float]:
    diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)
    inferred_ratio_from_alpha_rate = infer_gamma_exchange_ratio_from_alpha_rate(
        gamma_k=diagnostics["gamma_k"],
        observed_decay_per_renewal=diagnostics["late_alpha_decay_per_renewal"],
    )
    row = {
        **diagnostics,
        "inferred_ratio_from_late_ngp_amplitude": diagnostics["late_ngp_renewal_amplitude"] - 1.0,
        "inferred_ratio_from_alpha_rate": inferred_ratio_from_alpha_rate,
        "log_ratio_residual": float(np.log(inferred_ratio_from_alpha_rate / diagnostics["heterogeneity_ratio"])),
    }
    write_sweep_csv(path, [row])
    return row


def write_heterogeneity_map_csv(
    path: Path,
    params: DelayedRenewalCageParams,
    *,
    shape: float,
    heterogeneity_ratios: list[float],
    wave_number: float,
) -> list[dict[str, float]]:
    rows = gamma_exchange_diagnostic_map(
        wave_number=wave_number,
        params=params,
        shape=shape,
        heterogeneity_ratios=heterogeneity_ratios,
    )
    write_sweep_csv(path, rows)
    return rows


def write_heterogeneity_protocol_csv(
    path: Path,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
    *,
    wave_number: float,
    late_time: float,
) -> list[dict[str, float | str]]:
    late_array = np.array([late_time])
    renewal = float(delayed_poisson_mean(late_array, params)[0])
    late_ngp = float(gamma_exchange_ngp_1d(late_array, params, heterogeneity)[0])
    diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)
    renewal_std = 0.01 * renewal
    late_ngp_std = 0.01 * late_ngp
    alpha_slope_std = 0.002
    consistent = infer_gamma_exchange_uncertainty_from_late_observables(
        wave_number=wave_number,
        params=params,
        late_renewal_count=renewal,
        late_ngp=late_ngp,
        observed_alpha_decay_per_renewal=diagnostics["late_alpha_decay_per_renewal"],
        late_renewal_count_std=renewal_std,
        late_ngp_std=late_ngp_std,
        alpha_decay_per_renewal_std=alpha_slope_std,
    )
    gamma_k = consistent["gamma_k"]
    mismatched_alpha_rate = float(np.log1p(gamma_k * 2.0) / 2.0)
    inconsistent = infer_gamma_exchange_uncertainty_from_late_observables(
        wave_number=wave_number,
        params=params,
        late_renewal_count=renewal,
        late_ngp=late_ngp,
        observed_alpha_decay_per_renewal=mismatched_alpha_rate,
        late_renewal_count_std=renewal_std,
        late_ngp_std=late_ngp_std,
        alpha_decay_per_renewal_std=alpha_slope_std,
    )
    rows: list[dict[str, float | str]] = []
    for label, row in [("consistent", consistent), ("inconsistent_alpha_slope", inconsistent)]:
        rows.append({"case": label, **row})
    write_sweep_csv(path, rows)
    return rows


def write_heterogeneity_multik_csv(
    path: Path,
    params: DelayedRenewalCageParams,
    *,
    wave_numbers: list[float],
    exchange_ratio: float,
    late_renewal_count: float,
    late_ngp: float,
) -> list[dict[str, float | str]]:
    def rate_for(wave_number: float, ratio: float) -> float:
        gamma = 1.0 - float(np.exp(-0.5 * wave_number**2 * params.jump_variance))
        return float(np.log1p(gamma * ratio) / ratio)

    consistent_rates = [rate_for(wave_number, exchange_ratio) for wave_number in wave_numbers]
    mismatched_rates = [
        rate_for(wave_number, 2.0 if np.isclose(wave_number, 1.1) else exchange_ratio)
        for wave_number in wave_numbers
    ]
    rows: list[dict[str, float | str]] = []
    for label, rates in [("consistent_multik", consistent_rates), ("mismatch_k_1.1", mismatched_rates)]:
        collapse = infer_gamma_exchange_multik_collapse(
            wave_numbers=wave_numbers,
            params=params,
            late_renewal_count=late_renewal_count,
            late_ngp=late_ngp,
            observed_alpha_decay_per_renewal=rates,
            alpha_decay_per_renewal_std=[0.002 for _ in wave_numbers],
            late_renewal_count_std=0.01 * late_renewal_count,
            late_ngp_std=0.01 * late_ngp,
        )
        for row in collapse["per_wave_number"]:
            rows.append(
                {
                    "case": label,
                    "wave_number": row["wave_number"],
                    "observed_alpha_decay_per_renewal": row["observed_alpha_decay_per_renewal"],
                    "ratio_from_late_ngp": collapse["ratio_from_late_ngp"],
                    "ratio_from_alpha_rate": row["ratio_from_alpha_rate"],
                    "ratio_from_alpha_rate_std": row["ratio_from_alpha_rate_std"],
                    "weighted_mean_ratio_from_alpha": collapse["weighted_mean_ratio_from_alpha"],
                    "weighted_mean_ratio_from_alpha_std": collapse["weighted_mean_ratio_from_alpha_std"],
                    "collapse_z_score": collapse["collapse_z_score"],
                    "alpha_ratio_reduced_chi_square": collapse["alpha_ratio_reduced_chi_square"],
                    "passes_multik_collapse": collapse["passes_multik_collapse"],
                }
            )
    write_sweep_csv(path, rows)
    return rows


def write_static_null_csv(
    path: Path,
    time: np.ndarray,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
    *,
    wave_number: float,
) -> list[dict[str, float]]:
    renewal = delayed_poisson_mean(time, params)
    poisson_decay = normalized_alpha_decay(wave_number, time, params)
    gamma_decay = gamma_exchange_normalized_alpha_decay(wave_number, time, params, heterogeneity)
    static_decay = static_gamma_normalized_alpha_decay(wave_number, time, params, heterogeneity.shape)
    poisson_alpha = ngp_1d(time, params)
    gamma_alpha = gamma_exchange_ngp_1d(time, params, heterogeneity)
    static_alpha = static_gamma_ngp_1d(time, params, heterogeneity.shape)
    gamma_slope = np.divide(
        -np.log(gamma_decay),
        renewal,
        out=np.full_like(renewal, np.nan),
        where=renewal > 0.0,
    )
    static_slope = np.divide(
        -np.log(static_decay),
        renewal,
        out=np.full_like(renewal, np.nan),
        where=renewal > 0.0,
    )
    static_diagnostics = static_gamma_asymptotic_diagnostics(wave_number, params, heterogeneity.shape)
    gamma_diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)
    rows = []
    for idx, value in enumerate(time):
        rows.append(
            {
                "time": float(value),
                "log10_time": float(np.log10(value)),
                "renewal_mean": float(renewal[idx]),
                "poisson_alpha_decay": float(poisson_decay[idx]),
                "gamma_exchange_alpha_decay": float(gamma_decay[idx]),
                "static_gamma_alpha_decay": float(static_decay[idx]),
                "poisson_ngp": float(poisson_alpha[idx]),
                "gamma_exchange_ngp": float(gamma_alpha[idx]),
                "static_gamma_ngp": float(static_alpha[idx]),
                "gamma_exchange_alpha_slope_per_renewal": float(gamma_slope[idx]),
                "static_gamma_alpha_slope_per_renewal": float(static_slope[idx]),
                "static_gamma_late_ngp_plateau": static_diagnostics["late_ngp_plateau"],
                "static_gamma_late_alpha_decay_per_renewal": static_diagnostics["late_alpha_decay_per_renewal"],
                "gamma_exchange_late_ngp_renewal_amplitude": gamma_diagnostics["late_ngp_renewal_amplitude"],
                "gamma_exchange_late_alpha_decay_per_renewal": gamma_diagnostics[
                    "late_alpha_decay_per_renewal"
                ],
            }
        )
    write_sweep_csv(path, rows)
    return rows


def write_mechanism_selection_csv(
    path: Path,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
    *,
    wave_number: float,
    earlier_time: float,
    later_time: float,
) -> list[dict[str, float | str]]:
    times = np.array([earlier_time, later_time], dtype=float)
    renewal = delayed_poisson_mean(times, params)
    gamma = 1.0 - float(np.exp(-0.5 * wave_number**2 * params.jump_variance))
    static_decay = static_gamma_normalized_alpha_decay(wave_number, np.array([later_time]), params, heterogeneity.shape)[0]
    cases = [
        ("poisson", ngp_1d(times, params), gamma),
        (
            "static_gamma",
            static_gamma_ngp_1d(times, params, heterogeneity.shape),
            -float(np.log(static_decay)) / float(renewal[1]),
        ),
        (
            "finite_exchange",
            gamma_exchange_ngp_1d(times, params, heterogeneity),
            gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)[
                "late_alpha_decay_per_renewal"
            ],
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for case_name, alpha, alpha_slope in cases:
        selection = late_mechanism_selection(
            wave_number=wave_number,
            params=params,
            earlier_renewal_count=float(renewal[0]),
            earlier_ngp=float(alpha[0]),
            later_renewal_count=float(renewal[1]),
            later_ngp=float(alpha[1]),
            observed_alpha_decay_per_renewal=float(alpha_slope),
        )
        for model_name in ["poisson", "static_gamma", "finite_exchange"]:
            model = selection[model_name]
            rows.append(
                {
                    "case": case_name,
                    "candidate_model": model_name,
                    "best_model": selection["best_model"],
                    "passes": model["passes"],
                    "score": model["score"],
                    "earlier_ngp_log_residual": model["earlier_ngp_log_residual"],
                    "later_ngp_log_residual": model["later_ngp_log_residual"],
                    "alpha_slope_log_residual": model["alpha_slope_log_residual"],
                    "observed_earlier_ngp": float(alpha[0]),
                    "observed_later_ngp": float(alpha[1]),
                    "observed_alpha_decay_per_renewal": float(alpha_slope),
                    "predicted_earlier_ngp": model["predicted_earlier_ngp"],
                    "predicted_later_ngp": model["predicted_later_ngp"],
                    "predicted_alpha_decay_per_renewal": model["predicted_alpha_decay_per_renewal"],
                    "earlier_renewal_count": float(renewal[0]),
                    "later_renewal_count": float(renewal[1]),
                }
            )
    write_sweep_csv(path, rows)
    return rows


def write_inversion_csv(
    path: Path,
    params: DelayedRenewalCageParams,
    *,
    wave_number: float,
    diffusion_scales: list[float],
) -> list[dict[str, float]]:
    plateau = float(np.exp(-0.5 * wave_number**2 * params.cage_variance))
    diffusion = 0.5 * params.renewal_rate * params.jump_variance
    observed_tau_alpha = alpha_relaxation_time(wave_number, params)
    observed_peak = dimensionless_peak_prediction(params)
    rows: list[dict[str, float]] = []
    for scale_value in diffusion_scales:
        scaled_diffusion = diffusion * scale_value
        try:
            inferred = infer_parameters_from_scattering_transport(
                wave_number=wave_number,
                debye_waller_plateau=plateau,
                diffusion_coefficient=scaled_diffusion,
                tau_alpha=observed_tau_alpha,
                renewal_delay=params.renewal_delay,
            )
            rows.append(
                {
                    "diffusion_scale": scale_value,
                    "valid": 1.0,
                    "existence_margin": inferred["existence_margin"],
                    "inferred_jump_to_cage_variance": inferred["jump_to_cage_variance"],
                    "inferred_renewal_rate": inferred["renewal_rate"],
                    "predicted_ngp_peak_time": inferred["predicted_ngp_peak_time"],
                    "predicted_ngp_peak": inferred["predicted_ngp_peak"],
                    "observed_ngp_peak_time": observed_peak["peak_time"],
                    "observed_ngp_peak": observed_peak["peak_ngp"],
                    "log_peak_time_residual": np.log(inferred["predicted_ngp_peak_time"] / observed_peak["peak_time"]),
                    "log_peak_height_residual": np.log(inferred["predicted_ngp_peak"] / observed_peak["peak_ngp"]),
                }
            )
        except ValueError:
            shape = delayed_renewal_shape(observed_tau_alpha / params.renewal_delay)
            margin = scaled_diffusion * params.renewal_delay * shape * wave_number**2
            rows.append(
                {
                    "diffusion_scale": scale_value,
                    "valid": 0.0,
                    "existence_margin": margin,
                    "inferred_jump_to_cage_variance": np.nan,
                    "inferred_renewal_rate": np.nan,
                    "predicted_ngp_peak_time": np.nan,
                    "predicted_ngp_peak": np.nan,
                    "observed_ngp_peak_time": observed_peak["peak_time"],
                    "observed_ngp_peak": observed_peak["peak_ngp"],
                    "log_peak_time_residual": np.nan,
                    "log_peak_height_residual": np.nan,
                }
            )
    write_sweep_csv(path, rows)
    return rows


def write_full_inference_csv(
    path: Path,
    params: DelayedRenewalCageParams,
    *,
    wave_number: float,
) -> dict[str, float]:
    plateau = float(np.exp(-0.5 * wave_number**2 * params.cage_variance))
    diffusion = 0.5 * params.renewal_rate * params.jump_variance
    observed_tau_alpha = alpha_relaxation_time(wave_number, params)
    observed_peak = dimensionless_peak_prediction(params)
    inferred = infer_parameters_from_full_observables(
        wave_number=wave_number,
        debye_waller_plateau=plateau,
        diffusion_coefficient=diffusion,
        tau_alpha=observed_tau_alpha,
        peak_time=observed_peak["peak_time"],
        peak_ngp=observed_peak["peak_ngp"],
    )
    row = {
        "wave_number": wave_number,
        "observed_debye_waller_plateau": plateau,
        "observed_diffusion_coefficient": diffusion,
        "observed_tau_alpha": observed_tau_alpha,
        "observed_peak_time": observed_peak["peak_time"],
        "observed_peak_ngp": observed_peak["peak_ngp"],
        "true_cage_variance": params.cage_variance,
        "true_jump_variance": params.jump_variance,
        "true_renewal_rate": params.renewal_rate,
        "true_renewal_delay": params.renewal_delay,
        **inferred,
    }
    write_sweep_csv(path, [row])
    return row


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


def write_diagnostics_csv(
    path: Path,
    *,
    peak_time: float,
    peak_ngp: float,
    params: DelayedRenewalCageParams,
) -> None:
    diagnostics = plateau_peak_diagnostics(
        peak_ngp=peak_ngp,
        peak_time=peak_time,
        renewal_delay=params.renewal_delay,
    )
    row = {
        "peak_time": peak_time,
        "peak_ngp": peak_ngp,
        "true_jump_to_cage_variance": params.jump_variance / params.cage_variance,
        "diagnostic_jump_to_cage_variance": diagnostics["jump_to_cage_variance"],
        "true_renewal_rate": params.renewal_rate,
        "diagnostic_renewal_rate": diagnostics["renewal_rate"],
        "target_renewal_count": diagnostics["target_renewal_count"],
        "renewal_rate_times_peak_time": diagnostics["renewal_rate_times_peak_time"],
    }
    write_sweep_csv(path, [row])


def write_consistency_csv(
    path: Path,
    *,
    peak_time: float,
    peak_ngp: float,
    late_time: float,
    late_ngp: float,
    params: DelayedRenewalCageParams,
) -> None:
    diagnostics = observable_consistency_diagnostics(
        peak_ngp=peak_ngp,
        peak_time=peak_time,
        renewal_delay=params.renewal_delay,
        late_time=late_time,
        late_ngp=late_ngp,
    )
    row = {
        "peak_time": peak_time,
        "peak_ngp": peak_ngp,
        "late_time": late_time,
        "late_ngp": late_ngp,
        "true_renewal_rate": params.renewal_rate,
        "peak_renewal_rate": diagnostics["peak_renewal_rate"],
        "late_renewal_rate_exact": diagnostics["late_renewal_rate_exact"],
        "late_renewal_rate_asymptotic": diagnostics["late_renewal_rate_asymptotic"],
        "exact_rate_ratio": diagnostics["exact_rate_ratio"],
        "asymptotic_rate_ratio": diagnostics["asymptotic_rate_ratio"],
        "log_exact_rate_residual": diagnostics["log_exact_rate_residual"],
        "log_asymptotic_rate_residual": diagnostics["log_asymptotic_rate_residual"],
    }
    write_sweep_csv(path, [row])


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


def write_scattering_svg(
    path: Path,
    time: np.ndarray,
    scattering_curves: list[tuple[str, np.ndarray]],
    alpha_curves: list[tuple[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]

    def axes(left: int, right: int, title: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 32}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">time</text>
"""

    def plot(left: int, right: int, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(time, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            color = colors[idx % len(colors)]
            out.append(polyline(x, segment, color))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{color}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Self-intermediate scattering predictions</text>
  {axes(left_a, right_a, "G. F_s(k,t): cage plateau and alpha relaxation")}
  {plot(left_a, right_a, scattering_curves)}
  {axes(left_b, right_b, "H. Cage-normalized alpha decay")}
  {plot(left_b, right_b, alpha_curves)}
</svg>
"""
    path.write_text(svg)


def write_temperature_svg(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]
    inverse_shift = np.array([1.0 / row["temperature"] - 1.0 for row in rows])

    def axes(left: int, right: int, title: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 78}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">inverse-temperature shift</text>
"""

    def plot(left: int, right: int, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(inverse_shift, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            color = colors[idx % len(colors)]
            out.append(polyline(x, segment, color))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{color}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    diffusion = np.array([row["diffusion_coefficient"] for row in rows])
    tau_alpha = np.array([row["tau_alpha"] for row in rows])
    se_product = np.array([row["normalized_stokes_einstein_product"] for row in rows])
    lambda_tau = np.array([row["lambda_tau_delay"] for row in rows])
    peak_time = np.array([row["predicted_ngp_peak_time"] for row in rows])
    peak_height = np.array([row["predicted_ngp_peak"] for row in rows])
    fractional_exponent = np.array([row["fractional_stokes_einstein_exponent"] for row in rows])
    activation_energy = np.array([row["apparent_alpha_activation_energy"] for row in rows])
    left_curves = [
        ("D / D_hot", diffusion / diffusion[0]),
        ("tau_alpha / tau_hot", tau_alpha / tau_alpha[0]),
        ("t_NGP / t_NGP,hot", peak_time / peak_time[0]),
    ]
    right_curves = [
        ("D tau_alpha / hot", se_product),
        ("lambda tau_d / hot", lambda_tau / lambda_tau[0]),
        ("alpha_peak / hot", peak_height / peak_height[0]),
        ("xi_SE", fractional_exponent),
        ("E_app / hot", activation_energy / activation_energy[0]),
    ]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Temperature-dependent renewal diagnostics</text>
  {axes(left_a, right_a, "I. Transport and relaxation decouple on cooling")}
  {plot(left_a, right_a, left_curves)}
  {axes(left_b, right_b, "J. Stokes-Einstein product and delayed-renewal control")}
  {plot(left_b, right_b, right_curves)}
</svg>
"""
    path.write_text(svg)


def write_alpha_shape_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 55}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, x_values: np.ndarray, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(x_values, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, palette[idx % len(palette)]))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{palette[idx % len(palette)]}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    labels = list(dict.fromkeys(str(row["temperature_label"]) for row in rows))
    shape_curves = []
    scaled_time = None
    summary_rows = []
    for label in labels:
        label_rows = [row for row in rows if row["temperature_label"] == label]
        if scaled_time is None:
            scaled_time = np.array([float(row["scaled_time"]) for row in label_rows])
        shape_curves.append((label, np.array([float(row["alpha_shape"]) for row in label_rows])))
        summary_rows.append(label_rows[0])

    inverse_shift = np.array([float(row["inverse_temperature_shift"]) for row in summary_rows])
    rms = np.array([float(row["rms_log_shape_residual"]) for row in summary_rows])
    control = np.array([float(row["shape_control"]) for row in summary_rows])
    tau_over_delay = np.array([float(row["tau_alpha_over_delay"]) for row in summary_rows])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Alpha-shape time-temperature superposition diagnostic</text>
  {axes(left_a, right_a, "U. Alpha shape after tau_alpha scaling", "t / tau_alpha")}
  {plot(left_a, right_a, scaled_time, shape_curves)}
  {axes(left_b, right_b, "V. Collapse residual and control variable", "inverse-temperature shift")}
  {plot(left_b, right_b, inverse_shift, [("RMS log-shape residual", rms), ("Gamma lambda tau_d / hot", control / control[0]), ("tau_alpha/tau_d", tau_over_delay / tau_over_delay[0])])}
</svg>
"""
    path.write_text(svg)


def write_facilitated_exchange_svg(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 78}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, x_values: np.ndarray, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(x_values, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, palette[idx % len(palette)]))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{palette[idx % len(palette)]}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    inverse_shift = np.array([row["inverse_temperature_shift"] for row in rows])
    ratio = np.array([row["heterogeneity_ratio"] for row in rows])
    amplitude = np.array([row["late_ngp_renewal_amplitude"] for row in rows])
    shape = np.array([row["shape"] for row in rows])
    exchange_count = np.array([row["exchange_renewal_count"] for row in rows])
    renormalization = np.array([row["alpha_rate_renormalization"] for row in rows])
    alpha_slope = np.array([row["late_alpha_decay_per_renewal"] for row in rows])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Facilitated finite-exchange heterogeneity law</text>
  {axes(left_a, right_a, "W. Cooling grows exchange heterogeneity", "inverse-temperature shift")}
  {plot(left_a, right_a, inverse_shift, [("c=R_x/kappa0", ratio), ("R alpha2 late", amplitude), ("R_x/R_x hot", exchange_count / exchange_count[0]), ("kappa0/kappa hot", shape / shape[0])])}
  {axes(left_b, right_b, "X. Alpha decay per renewal slows", "inverse-temperature shift")}
  {plot(left_b, right_b, inverse_shift, [("alpha-rate renormalization", renormalization), ("late alpha slope / hot", alpha_slope / alpha_slope[0])])}
</svg>
"""
    path.write_text(svg)


def write_glass_audit_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 650
    left_a, top_a, right_a, bottom_a = 80, 90, 540, 330
    left_b, top_b, right_b, bottom_b = 690, 90, 1130, 330
    left_c, top_c, right_c, bottom_c = 80, 410, 1130, 590
    palette = ["#2b6cb0", "#c05621", "#805ad5", "#2f855a"]
    temperature_rows = [row for row in rows if row["record_type"] == "temperature_row"]
    summary_rows = [row for row in rows if row["record_type"] == "summary_flag"]
    flag_map = {str(row["signature"]): float(row["flag_value"]) for row in summary_rows}

    def axes(left: int, top: int, right: int, bottom: int, title: str, xlabel: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 70}" y="{bottom + 36}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(
        left: int,
        top: int,
        right: int,
        bottom: int,
        x_values: np.ndarray,
        curves: list[tuple[str, np.ndarray]],
    ) -> str:
        x = scale(x_values, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, palette[idx % len(palette)]))
            out.append(
                f'<text x="{left + 16}" y="{top + 22 + idx * 17}" font-family="Arial, sans-serif" font-size="12" fill="{palette[idx % len(palette)]}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    x = np.array([float(row["inverse_temperature_shift"]) for row in temperature_rows])
    tau = np.array([float(row["tau_alpha_exchange"]) for row in temperature_rows])
    diffusion = np.array([float(row["diffusion_coefficient"]) for row in temperature_rows])
    se_product = np.array([float(row["normalized_stokes_einstein_product"]) for row in temperature_rows])
    heterogeneity = np.array([float(row["heterogeneity_ratio"]) for row in temperature_rows])
    chi4 = np.array([float(row["chi4_peak"]) for row in temperature_rows])
    beta = np.array([float(row["median_alpha_window_beta"]) for row in temperature_rows])

    signature_order = [
        "diffusion_slowdown",
        "alpha_slowdown",
        "ngp_peak_shift",
        "stokes_einstein_violation",
        "fragility_growth",
        "heterogeneity_growth",
        "stretched_alpha_window",
        "chi4_peak_growth",
        "gaussian_recovery",
        "thermodynamic_transition",
    ]
    bars = []
    group_width = (right_c - left_c) / len(signature_order)
    bar_width = group_width - 10
    for idx, signature in enumerate(signature_order):
        value = flag_map[signature]
        x0 = left_c + idx * group_width + 5
        bar_height = value * (bottom_c - top_c - 40)
        color = "#2f855a" if value >= 1.0 else "#a0aec0"
        bars.append(
            f'<rect x="{x0:.2f}" y="{bottom_c - bar_height:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" />'
        )
        label = signature.replace("_", " ")
        bars.append(
            f'<text x="{x0 + 3:.2f}" y="{bottom_c + 16}" font-family="Arial, sans-serif" font-size="10" transform="rotate(35 {x0 + 3:.2f} {bottom_c + 16})">{label}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="80" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Glass-dynamics phenomenon audit</text>
  <text x="80" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">One delayed-renewal cage law is tested against dynamical signatures; no thermodynamic transition is claimed.</text>
  {axes(left_a, top_a, right_a, bottom_a, "A. Transport and relaxation under cooling", "inverse-temperature shift")}
  {plot(left_a, top_a, right_a, bottom_a, x, [("tau_alpha exchange", tau / tau[0]), ("1 / diffusion", diffusion[0] / diffusion), ("D tau_alpha", se_product)])}
  {axes(left_b, top_b, right_b, bottom_b, "B. Heterogeneity signatures", "inverse-temperature shift")}
  {plot(left_b, top_b, right_b, bottom_b, x, [("exchange ratio", heterogeneity / heterogeneity[0]), ("chi4 peak", chi4 / chi4[0]), ("alpha-window beta", beta)])}
  <line x1="{left_c}" y1="{bottom_c}" x2="{right_c}" y2="{bottom_c}" stroke="#222" />
  <line x1="{left_c}" y1="{bottom_c}" x2="{left_c}" y2="{top_c}" stroke="#222" />
  <text x="{left_c}" y="{top_c - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">C. Supported signatures from the effective dynamics</text>
  {"".join(bars)}
</svg>
"""
    path.write_text(svg)


def write_glass_phase_diagram_svg(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 620
    left, top, right, bottom = 95, 95, 690, 500
    panel_b_left, panel_b_right = 780, 1110
    gaps = sorted({row["delay_barrier_gap"] for row in rows})
    exchange_sums = sorted({row["exchange_barrier_sum"] for row in rows})
    row_map = {(row["delay_barrier_gap"], row["exchange_barrier_sum"]): row for row in rows}
    cell_w = (right - left) / len(exchange_sums)
    cell_h = (bottom - top) / len(gaps)
    max_signatures = max(row["tested_dynamic_signatures"] for row in rows)
    max_se = max(row["cold_se_product_ratio"] for row in rows)
    max_het = max(row["cold_heterogeneity_growth_ratio"] for row in rows)

    cells = []
    for i, gap in enumerate(gaps):
        y = bottom - (i + 1) * cell_h
        cells.append(
            f'<text x="35" y="{y + cell_h / 2 + 4:.2f}" font-family="Arial, sans-serif" font-size="12">gap={gap:g}</text>'
        )
        for j, exchange_sum in enumerate(exchange_sums):
            row = row_map[(gap, exchange_sum)]
            x = left + j * cell_w
            fraction = row["supported_dynamic_signatures"] / max_signatures
            green = int(180 - 95 * fraction)
            fill = "#2f855a" if row["complete_dynamic_closure"] == 1.0 else f"rgb(220,{green},130)"
            cells.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w - 3:.2f}" height="{cell_h - 3:.2f}" fill="{fill}" stroke="#ffffff" />'
            )
            cells.append(
                f'<text x="{x + 14:.2f}" y="{y + cell_h / 2 + 4:.2f}" font-family="Arial, sans-serif" font-size="12" fill="#111">{row["supported_dynamic_signatures"]:.0f}/{row["tested_dynamic_signatures"]:.0f}</text>'
            )
    x_labels = []
    for j, exchange_sum in enumerate(exchange_sums):
        x = left + j * cell_w + 12
        x_labels.append(
            f'<text x="{x:.2f}" y="{bottom + 28}" font-family="Arial, sans-serif" font-size="12">{exchange_sum:g}</text>'
        )

    def mini_plot(metric: str, y0: int, title: str, scale_max: float, color: str) -> str:
        bars = [
            row
            for row in rows
            if row["delay_barrier_gap"] == max(gaps)
        ]
        bar_w = (panel_b_right - panel_b_left) / len(bars) - 10
        out = [
            f'<text x="{panel_b_left}" y="{y0 - 18}" font-family="Arial, sans-serif" font-size="15" font-weight="700">{title}</text>',
            f'<line x1="{panel_b_left}" y1="{y0 + 125}" x2="{panel_b_right}" y2="{y0 + 125}" stroke="#222" />',
            f'<line x1="{panel_b_left}" y1="{y0 + 125}" x2="{panel_b_left}" y2="{y0}" stroke="#222" />',
        ]
        for idx, row in enumerate(bars):
            height_value = 120.0 * row[metric] / scale_max if scale_max > 0.0 else 0.0
            x = panel_b_left + idx * (bar_w + 10) + 6
            out.append(
                f'<rect x="{x:.2f}" y="{y0 + 125 - height_value:.2f}" width="{bar_w:.2f}" height="{height_value:.2f}" fill="{color}" opacity="0.85" />'
            )
            out.append(
                f'<text x="{x:.2f}" y="{y0 + 145}" font-family="Arial, sans-serif" font-size="10">{row["exchange_barrier_sum"]:g}</text>'
            )
        return "\n".join(out)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="80" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Barrier-facilitation signature phase diagram</text>
  <text x="80" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Cells show supported dynamic signatures. Full closure needs delay-barrier growth and finite-exchange growth.</text>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">A. Dynamic-signature closure grid</text>
  {"".join(cells)}
  {"".join(x_labels)}
  <text x="{left + 185}" y="{bottom + 55}" font-family="Arial, sans-serif" font-size="13">exchange barrier sum</text>
  <text x="18" y="315" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 18 315)">delay barrier gap</text>
  {mini_plot("cold_se_product_ratio", 140, "B. Cold SE product at largest delay gap", max_se, "#2b6cb0")}
  {mini_plot("cold_heterogeneity_growth_ratio", 360, "C. Heterogeneity growth at largest delay gap", max_het, "#c05621")}
</svg>
"""
    path.write_text(svg)


def write_barrier_svg(
    path: Path,
    time: np.ndarray,
    susceptibility_curves: list[tuple[str, np.ndarray]],
    barrier_rows: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 62}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, x_values: np.ndarray, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(x_values, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, colors[idx % len(colors)]))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{colors[idx % len(colors)]}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    final_by_gap = {}
    lambda_tau_by_gap = {}
    for row in barrier_rows:
        gap = row["barrier_gap"]
        if row["temperature"] == min(r["temperature"] for r in barrier_rows if r["barrier_gap"] == gap):
            final_by_gap[gap] = row["normalized_stokes_einstein_product"]
            lambda_tau_by_gap[gap] = row["lambda_tau_delay"]
    gaps = np.array(sorted(final_by_gap.keys()))
    cold_product = np.array([final_by_gap[gap] for gap in gaps])
    cold_lambda_tau = np.array([lambda_tau_by_gap[gap] for gap in gaps])
    right_curves = [
        ("cold D tau_alpha / hot", cold_product),
        ("cold lambda tau_d", cold_lambda_tau / cold_lambda_tau[0]),
    ]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Activated barrier and renewal susceptibility diagnostics</text>
  {axes(left_a, right_a, "K. Renewal-count scattering susceptibility", "time")}
  {plot(left_a, right_a, time, susceptibility_curves)}
  {axes(left_b, right_b, "L. Barrier gap amplifies SE violation", "E_d - E_lambda")}
  {plot(left_b, right_b, gaps, right_curves)}
</svg>
"""
    path.write_text(svg)


def write_inversion_svg(path: Path, inversion_rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    colors = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 68}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, x_values: np.ndarray, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(x_values, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        finite = all_values[np.isfinite(all_values)]
        y_all = scale(finite, bottom, top)
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        out = []
        for idx, (label, curve) in enumerate(curves):
            y = np.full_like(curve, np.nan, dtype=float)
            mask = np.isfinite(curve)
            y[mask] = bottom + (curve[mask] - y_min) * (top - bottom) / (y_max - y_min)
            out.append(polyline(x, y, colors[idx % len(colors)]))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{colors[idx % len(colors)]}">{label}</text>'
            )
        return "\n".join(out)

    diffusion_scale = np.array([row["diffusion_scale"] for row in inversion_rows])
    margin = np.array([row["existence_margin"] for row in inversion_rows])
    jump_ratio = np.array([row["inferred_jump_to_cage_variance"] for row in inversion_rows])
    time_residual = np.array([row["log_peak_time_residual"] for row in inversion_rows])
    height_residual = np.array([row["log_peak_height_residual"] for row in inversion_rows])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Observable inversion and falsifiability diagnostics</text>
  {axes(left_a, right_a, "M. Scattering-transport existence margin", "D / D_observed")}
  {plot(left_a, right_a, diffusion_scale, [("existence margin", margin), ("margin=1 threshold", np.ones_like(margin)), ("inferred q/A", jump_ratio)])}
  <text x="{left_a + 18}" y="{bottom - 52}" font-family="Arial, sans-serif" font-size="12" fill="#718096">margin below 1 has no positive-q solution</text>
  {axes(left_b, right_b, "N. NGP peak residuals after inversion", "D / D_observed")}
  {plot(left_b, right_b, diffusion_scale, [("log t*_pred / t*_obs", time_residual), ("log alpha*_pred / alpha*_obs", height_residual)])}
</svg>
"""
    path.write_text(svg)


def write_heterogeneity_svg(path: Path, heterogeneity_rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 45}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, x_values: np.ndarray, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(x_values, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, palette[idx % len(palette)]))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{palette[idx % len(palette)]}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    log_time = np.array([row["log10_time"] for row in heterogeneity_rows])
    poisson_decay = np.array([row["poisson_alpha_decay"] for row in heterogeneity_rows])
    gamma_decay = np.array([row["gamma_exchange_alpha_decay"] for row in heterogeneity_rows])
    poisson_ngp = np.array([row["poisson_ngp"] for row in heterogeneity_rows])
    gamma_ngp = np.array([row["gamma_exchange_ngp"] for row in heterogeneity_rows])
    local_beta = np.array([row["gamma_exchange_local_beta"] for row in heterogeneity_rows])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Finite-exchange heterogeneity extension</text>
  {axes(left_a, right_a, "O. Alpha relaxation from renewal heterogeneity", "log10 time")}
  {plot(left_a, right_a, log_time, [("Poisson alpha decay", poisson_decay), ("gamma-exchange alpha decay", gamma_decay)])}
  {axes(left_b, right_b, "P. Enhanced NGP with long-time recovery", "log10 time")}
  {plot(left_b, right_b, log_time, [("Poisson NGP", poisson_ngp), ("gamma-exchange NGP", gamma_ngp), ("local beta", local_beta)])}
</svg>
"""
    path.write_text(svg)


def write_heterogeneity_map_svg(path: Path, map_rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    palette = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5"]

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 65}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, x_values: np.ndarray, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(x_values, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, palette[idx % len(palette)]))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{palette[idx % len(palette)]}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    log_ratio = np.array([row["log10_one_plus_ratio"] for row in map_rows])
    amplitude = np.array([row["late_ngp_renewal_amplitude"] for row in map_rows])
    alpha_slope = np.array([row["alpha_rate_renormalization"] for row in map_rows])
    criterion = np.array([row["passes_joint_criterion"] for row in map_rows])
    inferred = np.array([row["inferred_ratio_from_alpha_rate"] for row in map_rows])
    ratio = np.array([row["heterogeneity_ratio"] for row in map_rows])
    inferred_error = np.abs(inferred - ratio)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Finite-exchange diagnostic map</text>
  {axes(left_a, right_a, "Q. Late NGP amplitude versus exchange ratio", "log10(1+c)")}
  {plot(left_a, right_a, log_ratio, [("R alpha2 -> 1+c", amplitude), ("alpha-rate inferred c error", inferred_error)])}
  {axes(left_b, right_b, "R. Alpha slowing and joint observable window", "log10(1+c)")}
  {plot(left_b, right_b, log_ratio, [("alpha-rate renormalization", alpha_slope), ("passes joint criterion", criterion)])}
</svg>
"""
    path.write_text(svg)


def write_static_null_svg(path: Path, static_rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 560
    left_a, top, right_a, bottom = 75, 85, 525, 460
    left_b, right_b = 660, 1110
    palette = ["#2b6cb0", "#c05621", "#805ad5", "#2f855a"]

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 45}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, x_values: np.ndarray, curves: list[tuple[str, np.ndarray]]) -> str:
        x = scale(x_values, left, right)
        all_values = np.concatenate([curve for _, curve in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, palette[idx % len(palette)]))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{palette[idx % len(palette)]}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    log_time = np.array([row["log10_time"] for row in static_rows])
    poisson_decay = np.array([row["poisson_alpha_decay"] for row in static_rows])
    exchange_decay = np.array([row["gamma_exchange_alpha_decay"] for row in static_rows])
    static_decay = np.array([row["static_gamma_alpha_decay"] for row in static_rows])
    exchange_ngp = np.array([row["gamma_exchange_ngp"] for row in static_rows])
    static_ngp = np.array([row["static_gamma_ngp"] for row in static_rows])
    static_plateau = np.array([row["static_gamma_late_ngp_plateau"] for row in static_rows])
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Static gamma null versus finite exchange</text>
  {axes(left_a, right_a, "S. Static disorder broadens alpha decay", "log10 time")}
  {plot(left_a, right_a, log_time, [("Poisson alpha decay", poisson_decay), ("finite-exchange alpha decay", exchange_decay), ("static-gamma alpha decay", static_decay)])}
  {axes(left_b, right_b, "T. Static disorder lacks Gaussian recovery", "log10 time")}
  {plot(left_b, right_b, log_time, [("finite-exchange NGP", exchange_ngp), ("static-gamma NGP", static_ngp), ("static plateau 1/kappa0", static_plateau)])}
</svg>
"""
    path.write_text(svg)


def write_mechanism_selection_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1080, 560
    top, bottom = 95, 455
    left, right = 90, 1010
    cases = ["poisson", "static_gamma", "finite_exchange"]
    models = ["poisson", "static_gamma", "finite_exchange"]
    palette = {
        "poisson": "#2b6cb0",
        "static_gamma": "#805ad5",
        "finite_exchange": "#c05621",
    }
    row_map = {(row["case"], row["candidate_model"]): row for row in rows}
    max_score = max(float(row["score"]) for row in rows if np.isfinite(float(row["score"])))
    y_max = max(1.0, max_score * 1.05)

    def y(value: float) -> float:
        return bottom - value * (bottom - top) / y_max

    group_width = (right - left) / len(cases)
    bar_width = 42
    bars = []
    labels = []
    for case_idx, case in enumerate(cases):
        group_left = left + case_idx * group_width
        center = group_left + group_width / 2.0
        labels.append(
            f'<text x="{center - 40}" y="{bottom + 34}" font-family="Arial, sans-serif" font-size="13">{case}</text>'
        )
        for model_idx, model in enumerate(models):
            row = row_map[(case, model)]
            raw_score = float(row["score"])
            score = raw_score if np.isfinite(raw_score) else y_max
            x = center - 1.5 * bar_width + model_idx * bar_width
            y_top = y(score)
            opacity = "1.0" if row["passes"] == 1.0 else "0.45"
            label = f"{raw_score:.2f}" if np.isfinite(raw_score) else "invalid"
            bars.append(
                f'<rect x="{x:.2f}" y="{y_top:.2f}" width="{bar_width - 6}" height="{bottom - y_top:.2f}" fill="{palette[model]}" opacity="{opacity}" />'
            )
            bars.append(
                f'<text x="{x - 2:.2f}" y="{y_top - 7:.2f}" font-family="Arial, sans-serif" font-size="10" fill="{palette[model]}">{label}</text>'
            )
    legend = []
    for idx, model in enumerate(models):
        legend.append(
            f'<rect x="{left + idx * 220}" y="505" width="14" height="14" fill="{palette[model]}" />'
            f'<text x="{left + 20 + idx * 220}" y="517" font-family="Arial, sans-serif" font-size="13">{model}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="90" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700">Late-time mechanism selection</text>
  <text x="90" y="68" font-family="Arial, sans-serif" font-size="13" fill="#444">Lower score means the candidate matches two late NGP points and the alpha slope.</text>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="34" y="275" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 34 275)">joint log-residual score</text>
  <line x1="{left}" y1="{y(0.2):.2f}" x2="{right}" y2="{y(0.2):.2f}" stroke="#999" stroke-dasharray="5 4" />
  <text x="{right - 78}" y="{y(0.2) - 7:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#555">pass threshold</text>
  {"".join(bars)}
  {"".join(labels)}
  {"".join(legend)}
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
    default_alpha = ngp_1d(time, params)
    default_peak_time, default_peak_value = peak_summary(time, default_alpha)
    write_diagnostics_csv(
        DATA_DIR / "renewal_cage_diagnostics.csv",
        peak_time=default_peak_time,
        peak_ngp=default_peak_value,
        params=params,
    )
    late_time = 180.0
    late_ngp = float(ngp_1d(np.array([late_time]), params)[0])
    write_consistency_csv(
        DATA_DIR / "renewal_cage_consistency.csv",
        peak_time=default_peak_time,
        peak_ngp=default_peak_value,
        late_time=late_time,
        late_ngp=late_ngp,
        params=params,
    )

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

    scattering_time = np.linspace(0.0, 180.0, 1200)
    wave_numbers = [0.6, 1.1, 1.8]
    write_scattering_csv(DATA_DIR / "renewal_cage_scattering.csv", scattering_time, params, wave_numbers)
    write_peak_relaxation_csv(DATA_DIR / "renewal_cage_peak_relaxation.csv", params, wave_numbers)
    scattering_curves = []
    alpha_curves = []
    for wave_number in wave_numbers:
        scattering_curves.append(
            (f"k={wave_number:g}", self_intermediate_scattering(wave_number, scattering_time, params))
        )
        alpha_curves.append(
            (f"k={wave_number:g}", normalized_alpha_decay(wave_number, scattering_time, params))
        )

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
    write_scattering_svg(
        FIGURE_DIR / "renewal_cage_scattering.svg",
        scattering_time,
        scattering_curves,
        alpha_curves,
    )

    temperature_law = TemperatureLawParams(
        reference_temperature=1.0,
        cage_variance_ref=params.cage_variance,
        cage_tau_ref=params.cage_tau,
        jump_to_cage_ref=params.jump_variance / params.cage_variance,
        renewal_rate_ref=params.renewal_rate,
        renewal_delay_ref=params.renewal_delay,
        rate_activation=2.0,
        delay_activation=5.0,
        cage_stiffening=0.2,
        jump_to_cage_growth=0.25,
    )
    temperatures = np.linspace(1.0, 0.62, 18)
    temperature_rows = write_temperature_csv(
        DATA_DIR / "renewal_cage_temperature.csv",
        temperatures,
        temperature_law,
        wave_number=1.1,
    )
    write_temperature_svg(FIGURE_DIR / "renewal_cage_temperature.svg", temperature_rows)
    alpha_shape_rows = write_alpha_shape_csv(
        DATA_DIR / "renewal_cage_alpha_shape.csv",
        np.geomspace(0.15, 4.0, 180),
        [1.0, 0.78, 0.62],
        temperature_law,
        wave_number=1.1,
    )
    write_alpha_shape_svg(FIGURE_DIR / "renewal_cage_alpha_shape.svg", alpha_shape_rows)
    exchange_law = FacilitatedExchangeLawParams(
        reference_temperature=1.0,
        shape_ref=0.4,
        exchange_renewal_count_ref=10.0,
        shape_broadening_barrier=1.5,
        exchange_slowing_barrier=2.5,
    )
    facilitated_exchange_rows = write_facilitated_exchange_csv(
        DATA_DIR / "renewal_cage_facilitated_exchange.csv",
        temperatures,
        temperature_law,
        exchange_law,
        wave_number=1.1,
    )
    write_facilitated_exchange_svg(
        FIGURE_DIR / "renewal_cage_facilitated_exchange.svg",
        facilitated_exchange_rows,
    )
    glass_audit_rows = write_glass_audit_csv(
        DATA_DIR / "renewal_cage_glass_audit.csv",
        temperatures,
        temperature_law,
        exchange_law,
        wave_number=1.1,
    )
    write_glass_audit_svg(FIGURE_DIR / "renewal_cage_glass_audit.svg", glass_audit_rows)
    glass_phase_rows = write_glass_phase_diagram_csv(
        DATA_DIR / "renewal_cage_glass_phase_diagram.csv",
        temperatures,
        TemperatureLawParams(
            reference_temperature=temperature_law.reference_temperature,
            cage_variance_ref=temperature_law.cage_variance_ref,
            cage_tau_ref=temperature_law.cage_tau_ref,
            jump_to_cage_ref=temperature_law.jump_to_cage_ref,
            renewal_rate_ref=temperature_law.renewal_rate_ref,
            renewal_delay_ref=temperature_law.renewal_delay_ref,
            rate_activation=temperature_law.rate_activation,
            delay_activation=temperature_law.rate_activation,
            cage_stiffening=temperature_law.cage_stiffening,
            jump_to_cage_growth=temperature_law.jump_to_cage_growth,
            cage_tau_activation=temperature_law.cage_tau_activation,
        ),
        exchange_law,
        wave_number=1.1,
        delay_barrier_gaps=[0.0, 1.5, 3.0],
        exchange_barrier_sums=[0.0, 2.0, 4.0],
    )
    write_glass_phase_diagram_svg(
        FIGURE_DIR / "renewal_cage_glass_phase_diagram.svg",
        glass_phase_rows,
    )

    barrier = ActivatedBarrierParams(
        reference_temperature=1.0,
        cage_variance_ref=params.cage_variance,
        cage_tau_ref=params.cage_tau,
        jump_to_cage_ref=params.jump_variance / params.cage_variance,
        renewal_rate_ref=params.renewal_rate,
        renewal_delay_ref=params.renewal_delay,
        renewal_rate_barrier=2.0,
        delay_onset_barrier=5.0,
        cage_stiffening_barrier=0.2,
        jump_to_cage_barrier=0.25,
    )
    barrier_rows = write_barrier_csv(
        DATA_DIR / "renewal_cage_barrier.csv",
        temperatures,
        base_barrier=barrier,
        wave_number=1.1,
        gap_values=[0.0, 1.5, 3.0, 4.5],
    )
    susceptibility_curves = write_susceptibility_csv(
        DATA_DIR / "renewal_cage_susceptibility.csv",
        scattering_time,
        params,
        wave_numbers,
    )
    chi4_inference = write_chi4_bridge_csv(
        DATA_DIR / "renewal_cage_chi4.csv",
        scattering_time,
        params,
        wave_number=1.1,
        correlation_sizes=[1.0, 4.0, 12.0],
        synthetic_correlation_size=12.0,
    )
    heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
    heterogeneity_time = np.geomspace(0.02, 30000.0, 1400)
    heterogeneity_rows = write_heterogeneity_csv(
        DATA_DIR / "renewal_cage_heterogeneity.csv",
        heterogeneity_time,
        params,
        heterogeneity,
        wave_number=1.1,
    )
    write_heterogeneity_diagnostics_csv(
        DATA_DIR / "renewal_cage_heterogeneity_diagnostics.csv",
        params,
        heterogeneity,
        wave_number=1.1,
    )
    heterogeneity_map_rows = write_heterogeneity_map_csv(
        DATA_DIR / "renewal_cage_heterogeneity_map.csv",
        params,
        shape=heterogeneity.shape,
        heterogeneity_ratios=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 25.0, 40.0],
        wave_number=1.1,
    )
    write_heterogeneity_protocol_csv(
        DATA_DIR / "renewal_cage_heterogeneity_protocol.csv",
        params,
        heterogeneity,
        wave_number=1.1,
        late_time=30000.0,
    )
    late_array = np.array([30000.0])
    write_heterogeneity_multik_csv(
        DATA_DIR / "renewal_cage_heterogeneity_multik.csv",
        params,
        wave_numbers=[0.6, 1.1, 1.8],
        exchange_ratio=heterogeneity.exchange_renewal_count / heterogeneity.shape,
        late_renewal_count=float(delayed_poisson_mean(late_array, params)[0]),
        late_ngp=float(gamma_exchange_ngp_1d(late_array, params, heterogeneity)[0]),
    )
    static_null_rows = write_static_null_csv(
        DATA_DIR / "renewal_cage_static_null.csv",
        heterogeneity_time,
        params,
        heterogeneity,
        wave_number=1.1,
    )
    mechanism_selection_rows = write_mechanism_selection_csv(
        DATA_DIR / "renewal_cage_mechanism_selection.csv",
        params,
        heterogeneity,
        wave_number=1.1,
        earlier_time=10000.0,
        later_time=30000.0,
    )
    write_barrier_svg(
        FIGURE_DIR / "renewal_cage_barrier.svg",
        scattering_time,
        susceptibility_curves,
        barrier_rows,
    )
    write_heterogeneity_svg(FIGURE_DIR / "renewal_cage_heterogeneity.svg", heterogeneity_rows)
    write_heterogeneity_map_svg(FIGURE_DIR / "renewal_cage_heterogeneity_map.svg", heterogeneity_map_rows)
    write_static_null_svg(FIGURE_DIR / "renewal_cage_static_null.svg", static_null_rows)
    write_mechanism_selection_svg(FIGURE_DIR / "renewal_cage_mechanism_selection.svg", mechanism_selection_rows)
    inversion_rows = write_inversion_csv(
        DATA_DIR / "renewal_cage_inversion.csv",
        params,
        wave_number=1.1,
        diffusion_scales=[0.6, 0.75, 0.8, 1.0, 1.5, 2.0, 3.0],
    )
    write_full_inference_csv(DATA_DIR / "renewal_cage_full_inference.csv", params, wave_number=1.1)
    write_inversion_svg(FIGURE_DIR / "renewal_cage_inversion.svg", inversion_rows)


if __name__ == "__main__":
    main()
