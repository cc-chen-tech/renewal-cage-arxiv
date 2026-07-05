#!/usr/bin/env python3
"""Generate reproducible figures for the delayed renewal cage model."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renewal_cage import (  # noqa: E402
    ActivatedBarrierParams,
    ConfigurationalEntropyParams,
    DelayedRenewalCageParams,
    FacilitatedExchangeLawParams,
    GammaExchangeParams,
    MCTBetaParams,
    TemperatureLawParams,
    alpha_relaxation_shape_curve,
    alpha_relaxation_time,
    alpha_shape_superposition_residual,
    alpha_tts_benchmark_consistency,
    activated_barrier_temperature_law,
    adam_gibbs_thermodynamic_scan,
    barrier_amplification_laws,
    benchmark_fusion_readiness,
    cage_localization_benchmark_consistency,
    cage_localization_diagnostics,
    correlated_domain_susceptibility,
    delayed_poisson_mean,
    delayed_renewal_shape,
    dimensionless_peak_prediction,
    cross_observable_prediction_ledger,
    dynamic_heterogeneity_benchmark_consistency,
    fragility_benchmark_consistency,
    frontier_benchmark_horizon,
    gaussian_radial_3d,
    gaussian_recovery_benchmark_consistency,
    gamma_exchange_alpha_relaxation_time,
    gamma_exchange_asymptotic_diagnostics,
    gamma_exchange_count_moments,
    gamma_exchange_diagnostic_map,
    gamma_exchange_temperature_scan,
    infer_gamma_exchange_multik_collapse,
    infer_gamma_exchange_ratio_from_alpha_rate,
    infer_gamma_exchange_uncertainty_from_late_observables,
    inversion_identifiability_audit,
    gamma_exchange_ngp_1d,
    gamma_exchange_normalized_alpha_decay,
    gamma_exchange_scattering_susceptibility,
    glass_phenomenon_audit,
    glass_signature_phase_diagram,
    infer_parameters_from_full_observables,
    infer_parameters_from_scattering_transport,
    infer_persistence_exchange_from_alpha_transport,
    infer_renewal_correlation_size,
    infer_spatial_facilitation_diffusivity,
    joint_inversion_benchmark_consistency,
    kww_alpha_fit,
    late_mechanism_selection,
    literature_inversion_readiness,
    minimal_barrier_requirements,
    local_alpha_stretching_exponent,
    mct_beta_benchmark_consistency,
    mct_beta_correlator,
    mct_exponent_benchmark_consistency,
    mct_beta_temperature_scan,
    moments_1d,
    ngp_peak_benchmark_consistency,
    ngp_1d,
    normalized_alpha_decay,
    observable_consistency_diagnostics,
    observable_falsification_matrix,
    plateau_peak_diagnostics,
    peak_relaxation_coupling,
    persistence_exchange_alpha_relaxation_time,
    persistence_exchange_benchmark_consistency,
    persistence_exchange_data_protocol,
    persistence_exchange_diffusion_coefficient,
    persistence_exchange_joint_diagnostic,
    persistence_exchange_ngp_1d,
    persistence_exchange_normalized_alpha_decay,
    persistence_exchange_scan,
    persistence_exchange_scattering_susceptibility,
    PersistenceExchangeParams,
    radial_van_hove_3d,
    raw_curve_ingestion_contract,
    raw_curve_diagnostic_readiness,
    raw_curve_persistence_exchange_protocol,
    real_benchmark_assimilation_gate,
    renewal_scattering_susceptibility,
    self_intermediate_scattering,
    spatial_facilitation_chi4_scan,
    spatial_facilitation_growth_law_consistency,
    sota_claim_alignment,
    sota_data_accession_gate,
    sota_readme_schema_gate,
    sota_source_provenance_gate,
    sota_signed_constraint_audit,
    static_gamma_asymptotic_diagnostics,
    static_gamma_ngp_1d,
    static_gamma_normalized_alpha_decay,
    stokes_einstein_benchmark_consistency,
    stretched_alpha_benchmark_consistency,
    thermodynamic_scope_benchmark_consistency,
    temperature_dependent_params,
    temperature_dependent_gamma_exchange,
    temperature_scan,
    trajectory_adapter_contract,
    trajectory_observable_protocol,
    trajectory_observable_uncertainty_protocol,
    trajectory_table_csv_adapter,
    trajectory_table_adapter,
    trajectory_inversion_readiness_gate,
    TranslationRotationExchangeParams,
    translation_rotation_decoupling_diagnostic,
    translation_rotation_inversion_protocol,
    translation_rotation_rotational_relaxation_time,
    van_hove_tail_benchmark_consistency,
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


def write_kww_alpha_csv(
    path: Path,
    temperatures: np.ndarray,
    law: TemperatureLawParams,
    exchange_law: FacilitatedExchangeLawParams,
    *,
    wave_number: float,
    min_decay: float,
    max_decay: float,
    time_points: int = 700,
) -> list[dict[str, float]]:
    if time_points < 50:
        raise ValueError("time_points must be at least 50")
    rows: list[dict[str, float]] = []
    reference_temperature = float(temperatures[0])
    for temperature in temperatures:
        params = temperature_dependent_params(float(temperature), law)
        heterogeneity = temperature_dependent_gamma_exchange(float(temperature), exchange_law)
        tau_alpha = gamma_exchange_alpha_relaxation_time(wave_number, params, heterogeneity)
        upper = max(50.0 * tau_alpha, 80.0 * params.renewal_delay, 10.0 * params.cage_tau)
        time = np.logspace(-3.0, math.log10(upper), time_points)
        decay = gamma_exchange_normalized_alpha_decay(wave_number, time, params, heterogeneity)
        fit = kww_alpha_fit(time, decay, min_decay=min_decay, max_decay=max_decay)
        rows.append(
            {
                "temperature": float(temperature),
                "inverse_temperature_shift": 1.0 / float(temperature) - 1.0 / reference_temperature,
                "tau_alpha": tau_alpha,
                "renewal_delay": params.renewal_delay,
                "renewal_rate": params.renewal_rate,
                "heterogeneity_shape": heterogeneity.shape,
                "exchange_renewal_count": heterogeneity.exchange_renewal_count,
                **fit,
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


def write_barrier_requirements_csv(
    path: Path,
    *,
    hot_temperature: float,
    cold_temperature: float,
    delay_barrier_gaps: list[float],
    exchange_barrier_sums: list[float],
    target_lambda_tau_delay_growth: float,
    target_heterogeneity_ratio_growth: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for gap in delay_barrier_gaps:
        for exchange_sum in exchange_barrier_sums:
            laws = barrier_amplification_laws(
                hot_temperature=hot_temperature,
                cold_temperature=cold_temperature,
                delay_barrier_gap=gap,
                exchange_barrier_sum=exchange_sum,
            )
            rows.append(
                {
                    "record_type": "amplification",
                    **laws,
                    "target_lambda_tau_delay_growth": np.nan,
                    "target_heterogeneity_ratio_growth": np.nan,
                    "target_combined_growth": np.nan,
                    "required_delay_barrier_gap": np.nan,
                    "required_exchange_barrier_sum": np.nan,
                    "required_combined_barrier": np.nan,
                }
            )
    requirements = minimal_barrier_requirements(
        hot_temperature=hot_temperature,
        cold_temperature=cold_temperature,
        target_lambda_tau_delay_growth=target_lambda_tau_delay_growth,
        target_heterogeneity_ratio_growth=target_heterogeneity_ratio_growth,
    )
    rows.append(
        {
            "record_type": "requirements",
            "delay_barrier_gap": np.nan,
            "exchange_barrier_sum": np.nan,
            "lambda_tau_delay_growth": np.nan,
            "heterogeneity_ratio_growth": np.nan,
            "combined_slowing_growth": np.nan,
            **requirements,
        }
    )
    write_sweep_csv(path, rows)
    return rows


def write_persistence_exchange_csv(
    path: Path,
    *,
    ratios: list[float],
    exchange_mean: float,
    wave_number: float,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    summary = persistence_exchange_scan(
        ratios=ratios,
        exchange_mean=exchange_mean,
        wave_number=wave_number,
    )
    for row in summary:
        rows.append(
            {
                "record_type": "summary",
                **row,
                "time": np.nan,
                "ngp": np.nan,
                "alpha_decay": np.nan,
            }
        )

    time = np.geomspace(0.03, 650.0, 260)
    for ratio in [ratios[0], ratios[-1]]:
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=ratio * exchange_mean,
            exchange_mean=exchange_mean,
        )
        alpha = persistence_exchange_ngp_1d(time, params, max_count=700)
        decay = persistence_exchange_normalized_alpha_decay(wave_number, time, params, max_count=700)
        tau_alpha = persistence_exchange_alpha_relaxation_time(wave_number, params)
        diffusion = persistence_exchange_diffusion_coefficient(params)
        for idx, value in enumerate(time):
            rows.append(
                {
                    "record_type": "curve",
                    "persistence_exchange_ratio": float(ratio),
                    "persistence_mean": params.persistence_mean,
                    "exchange_mean": params.exchange_mean,
                    "diffusion_coefficient": diffusion,
                    "tau_alpha": tau_alpha,
                    "stokes_einstein_product": diffusion * tau_alpha,
                    "late_time": np.nan,
                    "late_ngp": np.nan,
                    "time": float(value),
                    "ngp": float(alpha[idx]),
                    "alpha_decay": float(decay[idx]),
                }
            )
    write_sweep_csv(path, rows)
    return rows


def write_translation_rotation_protocol_csv(path: Path, *, wave_number: float) -> list[dict[str, float | str]]:
    """Generate a two-clock translation/rotation decoupling diagnostic."""

    scenarios = [
        (
            "coupled_clock",
            TranslationRotationExchangeParams(
                cage_variance=1.0,
                cage_tau=0.2,
                jump_variance=0.7,
                translational_persistence_mean=4.0,
                translational_exchange_mean=1.0,
                rotational_persistence_mean=4.0,
                rotational_exchange_mean=1.0,
                rotational_step_correlation=0.62,
            ),
        ),
        (
            "rotationally_slow_clock",
            TranslationRotationExchangeParams(
                cage_variance=1.0,
                cage_tau=0.2,
                jump_variance=0.7,
                translational_persistence_mean=4.0,
                translational_exchange_mean=1.0,
                rotational_persistence_mean=14.0,
                rotational_exchange_mean=1.0,
                rotational_step_correlation=0.62,
            ),
        ),
    ]
    rows: list[dict[str, float | str]] = []
    for scenario, params in scenarios:
        diagnostic = translation_rotation_decoupling_diagnostic(
            scenario,
            params,
            wave_number=wave_number,
        )
        inversion = translation_rotation_inversion_protocol(
            benchmark_id=scenario,
            wave_number=wave_number,
            jump_variance=params.jump_variance,
            diffusion_coefficient=diagnostic["diffusion_coefficient"],
            observed_tau_alpha=diagnostic["tau_alpha"],
            observed_rotational_relaxation_time=diagnostic["rotational_relaxation_time"],
            rotational_step_correlation=params.rotational_step_correlation,
            rotational_exchange_mean=params.rotational_exchange_mean,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
        )
        rows.append(
            {
                **diagnostic,
                "inferred_translational_persistence_mean": inversion["translational_persistence_mean"],
                "inferred_rotational_persistence_mean": inversion["rotational_persistence_mean"],
                "tau_alpha_log_residual": inversion["tau_alpha_log_residual"],
                "rotational_tau_log_residual": inversion["rotational_tau_log_residual"],
                "poisson_rotational_relaxation_time": inversion["poisson_rotational_relaxation_time"],
            }
        )
    write_sweep_csv(path, rows)
    return rows


def write_persistence_exchange_protocol_csv(
    path: Path,
    *,
    wave_number: float,
    jump_variance: float,
    exchange_mean: float,
    true_ratio: float,
) -> list[dict[str, float | str]]:
    params = PersistenceExchangeParams(
        cage_variance=1.0,
        cage_tau=0.2,
        jump_variance=jump_variance,
        persistence_mean=true_ratio * exchange_mean,
        exchange_mean=exchange_mean,
    )
    diffusion = persistence_exchange_diffusion_coefficient(params)
    tau_alpha = persistence_exchange_alpha_relaxation_time(wave_number, params)
    late_time = 80.0 * params.persistence_mean
    predicted_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])
    scenarios = [
        ("consistent", tau_alpha, predicted_late_ngp),
        ("late_ngp_mismatch", tau_alpha, 4.0 * predicted_late_ngp),
    ]
    poisson_params = PersistenceExchangeParams(
        cage_variance=params.cage_variance,
        cage_tau=params.cage_tau,
        jump_variance=jump_variance,
        persistence_mean=exchange_mean,
        exchange_mean=exchange_mean,
    )
    scenarios.append(
        (
            "alpha_too_fast",
            0.8 * persistence_exchange_alpha_relaxation_time(wave_number, poisson_params),
            predicted_late_ngp,
        )
    )

    rows: list[dict[str, float | str]] = []
    for scenario, observed_tau_alpha, observed_late_ngp in scenarios:
        try:
            inferred = infer_persistence_exchange_from_alpha_transport(
                wave_number=wave_number,
                jump_variance=jump_variance,
                diffusion_coefficient=diffusion,
                observed_tau_alpha=observed_tau_alpha,
                late_time=late_time,
                observed_late_ngp=observed_late_ngp,
            )
            late_residual = inferred["late_ngp_log_residual"]
            rows.append(
                {
                    "scenario": scenario,
                    "valid_alpha_transport": 1.0,
                    "passes_late_ngp": float(abs(late_residual) < 0.1),
                    "true_persistence_exchange_ratio": true_ratio,
                    "inferred_persistence_exchange_ratio": inferred["persistence_exchange_ratio"],
                    "diffusion_coefficient": diffusion,
                    "observed_tau_alpha": observed_tau_alpha,
                    "poisson_tau_alpha": inferred["poisson_tau_alpha"],
                    "late_time": late_time,
                    "observed_late_ngp": observed_late_ngp,
                    "predicted_late_ngp": inferred["predicted_late_ngp"],
                    "late_ngp_log_residual": late_residual,
                    "tau_alpha_log_residual": inferred["tau_alpha_log_residual"],
                }
            )
        except ValueError:
            rows.append(
                {
                    "scenario": scenario,
                    "valid_alpha_transport": 0.0,
                    "passes_late_ngp": 0.0,
                    "true_persistence_exchange_ratio": true_ratio,
                    "inferred_persistence_exchange_ratio": np.nan,
                    "diffusion_coefficient": diffusion,
                    "observed_tau_alpha": observed_tau_alpha,
                    "poisson_tau_alpha": persistence_exchange_alpha_relaxation_time(wave_number, poisson_params),
                    "late_time": late_time,
                    "observed_late_ngp": observed_late_ngp,
                    "predicted_late_ngp": np.nan,
                    "late_ngp_log_residual": np.nan,
                    "tau_alpha_log_residual": np.nan,
                }
            )
    write_sweep_csv(path, rows)
    return rows


def write_persistence_exchange_joint_protocol_csv(
    path: Path,
    *,
    anchor_wave_number: float,
    wave_numbers: list[float],
    jump_variance: float,
    exchange_mean: float,
    true_ratio: float,
) -> list[dict[str, float | str]]:
    params = PersistenceExchangeParams(
        cage_variance=1.0,
        cage_tau=0.2,
        jump_variance=jump_variance,
        persistence_mean=true_ratio * exchange_mean,
        exchange_mean=exchange_mean,
    )
    diffusion = persistence_exchange_diffusion_coefficient(params)
    late_time = 80.0 * params.persistence_mean
    observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])
    observed_tau_alpha = {
        wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
        for wave_number in wave_numbers
    }
    scenarios = [
        ("consistent", observed_tau_alpha),
        (
            "multik_alpha_mismatch",
            {
                wave_number: (1.25 * value if math.isclose(wave_number, max(wave_numbers)) else value)
                for wave_number, value in observed_tau_alpha.items()
            },
        ),
    ]
    time_grid = np.geomspace(0.05, 300.0, 260)
    rows: list[dict[str, float | str]] = []
    for scenario, tau_by_k in scenarios:
        diagnostic = persistence_exchange_joint_diagnostic(
            anchor_wave_number=anchor_wave_number,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=tau_by_k,
            jump_variance=jump_variance,
            diffusion_coefficient=diffusion,
            late_time=late_time,
            observed_late_ngp=observed_late_ngp,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            max_multik_abs_log_residual=0.02,
            max_late_ngp_abs_log_residual=0.02,
            min_chi4_peak_growth=1.5,
        )
        inferred_params = PersistenceExchangeParams(
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            jump_variance=jump_variance,
            persistence_mean=diagnostic["persistence_mean"],
            exchange_mean=diagnostic["exchange_mean"],
        )
        common = {
            "scenario": scenario,
            "true_persistence_exchange_ratio": true_ratio,
            "inferred_persistence_exchange_ratio": diagnostic["persistence_exchange_ratio"],
            "exchange_mean": diagnostic["exchange_mean"],
            "persistence_mean": diagnostic["persistence_mean"],
            "diffusion_coefficient": diffusion,
            "anchor_wave_number": anchor_wave_number,
            "late_time": late_time,
            "observed_late_ngp": observed_late_ngp,
            "predicted_late_ngp": diagnostic["predicted_late_ngp"],
            "late_ngp_log_residual": diagnostic["late_ngp_log_residual"],
            "max_multik_tau_alpha_abs_log_residual": diagnostic["max_multik_tau_alpha_abs_log_residual"],
            "stokes_einstein_growth_over_poisson": diagnostic["stokes_einstein_growth_over_poisson"],
            "chi4_peak": diagnostic["chi4_peak"],
            "poisson_chi4_peak": diagnostic["poisson_chi4_peak"],
            "chi4_peak_growth_over_poisson": diagnostic["chi4_peak_growth_over_poisson"],
            "multik_tau_alpha_consistent": diagnostic["multik_tau_alpha_consistent"],
            "late_ngp_consistent": diagnostic["late_ngp_consistent"],
            "chi4_proxy_growth_consistent": diagnostic["chi4_proxy_growth_consistent"],
            "passes_joint_protocol": diagnostic["passes_joint_protocol"],
        }
        rows.append(
            {
                "record_type": "summary",
                **common,
                "wave_number": np.nan,
                "observed_tau_alpha": np.nan,
                "predicted_tau_alpha": np.nan,
                "tau_alpha_log_residual": np.nan,
            }
        )
        for wave_number in wave_numbers:
            predicted_tau = persistence_exchange_alpha_relaxation_time(wave_number, inferred_params)
            rows.append(
                {
                    "record_type": "multik_alpha",
                    **common,
                    "wave_number": wave_number,
                    "observed_tau_alpha": tau_by_k[wave_number],
                    "predicted_tau_alpha": predicted_tau,
                    "tau_alpha_log_residual": math.log(tau_by_k[wave_number] / predicted_tau),
                }
            )
    write_sweep_csv(path, rows)
    return rows


def write_persistence_exchange_uncertainty_protocol_csv(
    path: Path,
    *,
    anchor_wave_number: float,
    wave_numbers: list[float],
    jump_variance: float,
    exchange_mean: float,
    true_ratio: float,
) -> list[dict[str, float | str]]:
    params = PersistenceExchangeParams(
        cage_variance=1.0,
        cage_tau=0.2,
        jump_variance=jump_variance,
        persistence_mean=true_ratio * exchange_mean,
        exchange_mean=exchange_mean,
    )
    diffusion = persistence_exchange_diffusion_coefficient(params)
    late_time = 80.0 * params.persistence_mean
    observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])
    observed_tau_alpha = {
        wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
        for wave_number in wave_numbers
    }
    time_grid = np.geomspace(0.05, 300.0, 260)
    predicted_chi4_peak = float(
        np.max(persistence_exchange_scattering_susceptibility(anchor_wave_number, time_grid, params))
    )
    scenarios = [
        ("consistent", predicted_chi4_peak),
        ("chi4_peak_mismatch", 2.0 * predicted_chi4_peak),
    ]
    rows: list[dict[str, float | str]] = []
    for scenario, observed_chi4_peak in scenarios:
        scored = persistence_exchange_data_protocol(
            anchor_wave_number=anchor_wave_number,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            tau_alpha_relative_error_by_k={wave_number: 0.05 for wave_number in wave_numbers},
            jump_variance=jump_variance,
            diffusion_coefficient=diffusion,
            late_time=late_time,
            observed_late_ngp=observed_late_ngp,
            late_ngp_relative_error=0.08,
            observed_chi4_peak=observed_chi4_peak,
            chi4_peak_relative_error=0.1,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=2.0,
        )
        rows.append(
            {
                "scenario": scenario,
                "true_persistence_exchange_ratio": true_ratio,
                "inferred_persistence_exchange_ratio": scored["persistence_exchange_ratio"],
                "observed_chi4_peak": observed_chi4_peak,
                "predicted_chi4_peak": scored["predicted_chi4_peak"],
                "chi4_peak_log_residual": scored["chi4_peak_log_residual"],
                "max_multik_tau_alpha_z": scored["max_multik_tau_alpha_z"],
                "late_ngp_z": scored["late_ngp_z"],
                "chi4_peak_z": scored["chi4_peak_z"],
                "z_threshold": scored["z_threshold"],
                "multik_tau_alpha_z_consistent": scored["multik_tau_alpha_z_consistent"],
                "late_ngp_z_consistent": scored["late_ngp_z_consistent"],
                "chi4_peak_z_consistent": scored["chi4_peak_z_consistent"],
                "passes_uncertainty_protocol": scored["passes_uncertainty_protocol"],
            }
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


def write_spatial_chi4_csv(
    path: Path,
    temperatures: np.ndarray,
    law: TemperatureLawParams,
    *,
    wave_number: float,
    facilitation_diffusivity: float,
    particle_density: float,
) -> list[dict[str, float]]:
    rows = spatial_facilitation_chi4_scan(
        temperatures=temperatures,
        law=law,
        wave_number=wave_number,
        facilitation_diffusivity=facilitation_diffusivity,
        particle_density=particle_density,
    )
    write_sweep_csv(path, rows)
    return rows


def write_thermodynamic_closure_csv(
    path: Path,
    temperatures: np.ndarray,
    entropy_law: ConfigurationalEntropyParams,
    *,
    activation_free_energy: float,
    tau_ref: float,
    renewal_rate_ref: float,
    wave_number: float,
    cage_variance: float,
    cage_tau: float,
    jump_variance: float,
) -> list[dict[str, float]]:
    rows = adam_gibbs_thermodynamic_scan(
        temperatures=temperatures,
        entropy_law=entropy_law,
        activation_free_energy=activation_free_energy,
        tau_ref=tau_ref,
        renewal_rate_ref=renewal_rate_ref,
        wave_number=wave_number,
        cage_variance=cage_variance,
        cage_tau=cage_tau,
        jump_variance=jump_variance,
    )
    write_sweep_csv(path, rows)
    return rows


def write_mct_beta_closure_csv(
    path: Path,
    temperatures: np.ndarray,
    base: MCTBetaParams,
    *,
    beta_time_activation: float,
    plateau_growth: float,
    alpha_time_ref: float,
    alpha_activation: float,
) -> list[dict[str, float]]:
    rows = mct_beta_temperature_scan(
        temperatures=temperatures,
        base=base,
        beta_time_activation=beta_time_activation,
        plateau_growth=plateau_growth,
        alpha_time_ref=alpha_time_ref,
        alpha_activation=alpha_activation,
    )
    write_sweep_csv(path, rows)
    return rows


def write_spatial_facilitation_inversion_csv(
    path: Path,
    spatial_chi4_rows: list[dict[str, float]],
    *,
    dimension: int,
    particle_density: float,
    microscopic_length: float,
    max_diffusivity_relative_std: float,
    min_length_growth: float,
) -> list[dict[str, float]]:
    persistence_times = np.array([row["renewal_delay"] for row in spatial_chi4_rows], dtype=float)
    observed_lengths = np.array([row["dynamic_correlation_length"] for row in spatial_chi4_rows], dtype=float)
    inferred_rows = infer_spatial_facilitation_diffusivity(
        persistence_times=persistence_times,
        observed_dynamic_lengths=observed_lengths,
        dimension=dimension,
        particle_density=particle_density,
        microscopic_length=microscopic_length,
    )
    summary = spatial_facilitation_growth_law_consistency(
        persistence_times=persistence_times,
        observed_dynamic_lengths=observed_lengths,
        observed_diffusive_front_growth=True,
        dimension=dimension,
        particle_density=particle_density,
        microscopic_length=microscopic_length,
        max_diffusivity_relative_std=max_diffusivity_relative_std,
        min_length_growth=min_length_growth,
    )
    rows = []
    for source, inferred in zip(spatial_chi4_rows, inferred_rows):
        rows.append(
            {
                "temperature": source["temperature"],
                "inverse_temperature_shift": 1.0 / source["temperature"] - 1.0 / spatial_chi4_rows[0]["temperature"],
                **inferred,
                "length_growth": source["length_growth"],
                "correlation_size_growth": source["correlation_size_growth"],
                "chi4_peak_growth": source["chi4_peak_growth"],
                "facilitation_diffusivity_mean": summary["facilitation_diffusivity_mean"],
                "facilitation_diffusivity_relative_std": summary["facilitation_diffusivity_relative_std"],
                "max_diffusivity_relative_std": summary["max_diffusivity_relative_std"],
                "growth_law_overall_consistent": summary["overall_consistent"],
            }
        )
    write_sweep_csv(path, rows)
    return rows


def write_sota_benchmark_consistency_csv(
    path: Path,
    beta: MCTBetaParams,
    params: DelayedRenewalCageParams,
    heterogeneity: GammaExchangeParams,
    temperature_law: TemperatureLawParams,
    temperature_rows: list[dict[str, float]],
    spatial_chi4_rows: list[dict[str, float]],
    alpha_shape_rows: list[dict[str, float | str]],
    kww_alpha_rows: list[dict[str, float]],
    persistence_exchange_joint_protocol_rows: list[dict[str, float | str]],
    tail_ratio_rows: list[dict[str, float | str]],
    thermodynamic_rows: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    fieldnames = [
        "benchmark_id",
        "benchmark_family",
        "observed_cage_localization",
        "debye_waller_plateau",
        "min_debye_waller_plateau",
        "max_debye_waller_plateau",
        "renewal_msd_fraction",
        "max_renewal_msd_fraction",
        "alpha_to_cage_time_ratio",
        "min_alpha_to_cage_time_ratio",
        "model_predicts_cage_localization",
        "debye_waller_consistent",
        "renewal_fraction_consistent",
        "alpha_separation_consistent",
        "observed_critical_decay",
        "observed_von_schweidler",
        "required_decades",
        "critical_window_decades",
        "von_schweidler_window_decades",
        "model_predicts_visible_critical_decay",
        "model_predicts_visible_von_schweidler",
        "critical_decay_consistent",
        "von_schweidler_consistent",
        "observed_common_exponent_parameter",
        "critical_exponent_benchmark",
        "von_schweidler_exponent_benchmark",
        "lambda_from_a",
        "lambda_from_b",
        "lambda_mismatch",
        "lambda_relative_mismatch",
        "max_lambda_relative_mismatch",
        "model_predicts_common_exponent_parameter",
        "mct_exponent_parameter_consistent",
        "observed_gaussian_recovery",
        "finite_exchange_late_ngp",
        "static_gamma_late_ngp",
        "recovery_threshold",
        "model_predicts_gaussian_recovery",
        "static_null_predicts_gaussian_recovery",
        "finite_exchange_recovery_consistent",
        "static_null_recovery_consistent",
        "mechanism_selection_consistent",
        "observed_transient_ngp_peak",
        "hot_peak_time",
        "cold_peak_time",
        "peak_time_growth",
        "hot_peak_ngp",
        "cold_peak_ngp",
        "peak_height_growth",
        "late_ngp",
        "min_peak_time_growth",
        "min_peak_height",
        "min_peak_height_growth",
        "max_late_ngp",
        "model_predicts_transient_ngp_peak",
        "peak_time_growth_consistent",
        "peak_height_consistent",
        "peak_height_growth_consistent",
        "late_recovery_consistent",
        "observed_stokes_einstein_violation",
        "hot_se_product",
        "cold_se_product",
        "se_product_growth",
        "cold_fractional_exponent",
        "min_product_growth",
        "max_fractional_exponent",
        "model_predicts_stokes_einstein_violation",
        "se_product_growth_consistent",
        "fractional_exponent_consistent",
        "observed_dynamic_heterogeneity_growth",
        "length_growth",
        "correlation_size_growth",
        "chi4_peak_growth_benchmark",
        "min_length_growth",
        "min_correlation_size_growth",
        "min_chi4_peak_growth",
        "model_predicts_dynamic_heterogeneity_growth",
        "length_growth_consistent",
        "correlation_size_growth_consistent",
        "chi4_peak_growth_consistent",
        "observed_diffusive_front_growth",
        "number_of_points",
        "facilitation_diffusivity_mean",
        "facilitation_diffusivity_std",
        "facilitation_diffusivity_relative_std",
        "max_diffusivity_relative_std",
        "persistence_time_growth",
        "model_predicts_diffusive_front_growth",
        "constant_diffusivity_consistent",
        "facilitation_growth_law_consistent",
        "observed_tts_breakdown",
        "cold_shape_residual",
        "alpha_shape_control_growth",
        "residual_threshold",
        "min_control_growth",
        "model_predicts_tts_breakdown",
        "tts_residual_consistent",
        "tts_control_consistent",
        "observed_stretched_alpha",
        "hot_kww_beta",
        "cold_kww_beta",
        "kww_beta_drop",
        "min_beta_drop",
        "max_cold_beta",
        "cold_fit_residual",
        "max_fit_residual",
        "model_predicts_stretched_alpha",
        "beta_drop_consistent",
        "cold_beta_consistent",
        "fit_quality_consistent",
        "observed_persistence_exchange_decoupling",
        "inferred_persistence_exchange_ratio",
        "min_persistence_exchange_ratio",
        "late_ngp_log_residual_benchmark",
        "max_late_ngp_abs_log_residual",
        "invalid_poisson_alpha_rejected",
        "model_predicts_persistence_exchange_decoupling",
        "persistence_exchange_ratio_consistent",
        "persistence_exchange_late_ngp_consistent",
        "persistence_exchange_rejection_consistent",
        "observed_joint_inversion_closure",
        "joint_inferred_persistence_exchange_ratio",
        "min_joint_persistence_exchange_ratio",
        "joint_stokes_einstein_growth_over_poisson",
        "min_joint_stokes_einstein_growth",
        "joint_multik_tau_alpha_abs_log_residual",
        "max_joint_multik_abs_log_residual",
        "joint_late_ngp_abs_log_residual",
        "max_joint_late_ngp_abs_log_residual",
        "joint_chi4_peak_growth_over_poisson",
        "min_joint_chi4_peak_growth",
        "rejected_mismatch_abs_log_residual",
        "min_rejected_mismatch_abs_log_residual",
        "model_predicts_joint_inversion_closure",
        "joint_ratio_consistent",
        "joint_se_consistent",
        "joint_multik_consistent",
        "joint_late_ngp_consistent",
        "joint_chi4_consistent",
        "joint_mismatch_rejected",
        "observed_transient_van_hove_tail",
        "observed_late_tail_gaussian_recovery",
        "peak_tail_ratio",
        "late_tail_ratio",
        "late_tail_abs_deviation",
        "peak_ngp_benchmark",
        "min_peak_tail_ratio",
        "max_late_tail_deviation",
        "min_peak_ngp",
        "model_predicts_transient_van_hove_tail",
        "model_predicts_tail_gaussian_recovery",
        "van_hove_tail_consistent",
        "tail_recovery_consistent",
        "peak_ngp_consistent",
        "observed_fragility_growth",
        "observed_adam_gibbs_slowdown",
        "hot_activation_energy",
        "cold_activation_energy",
        "activation_energy_growth",
        "hot_fragility_index",
        "cold_fragility_index",
        "fragility_index_growth",
        "adam_gibbs_slowdown",
        "min_activation_growth",
        "min_fragility_growth",
        "min_adam_gibbs_slowdown",
        "material_specific_origin_claimed",
        "model_predicts_fragility_growth",
        "model_predicts_adam_gibbs_slowdown",
        "activation_growth_consistent",
        "fragility_index_consistent",
        "adam_gibbs_slowdown_consistent",
        "fragility_scope_boundary_consistent",
        "observed_heat_capacity_anomaly",
        "observed_kauzmann_extrapolation",
        "dynamic_model_derives_entropy",
        "entropy_closure_supplied",
        "thermodynamic_adam_gibbs_slowdown",
        "min_thermodynamic_adam_gibbs_slowdown",
        "material_specific_entropy_origin_claimed",
        "model_predicts_heat_capacity_anomaly_from_dynamics",
        "model_predicts_kauzmann_transition_from_dynamics",
        "entropy_closure_required",
        "heat_capacity_scope_consistent",
        "kauzmann_scope_consistent",
        "thermodynamic_scope_boundary_consistent",
        "overall_consistent",
    ]

    def normalize(row: dict[str, float | str], family: str) -> dict[str, float | str]:
        normalized = {key: np.nan for key in fieldnames}
        normalized.update(row)
        normalized["benchmark_family"] = family
        return normalized

    cage_diagnostic = cage_localization_diagnostics(
        wave_number=1.1,
        plateau_time=1.0,
        params=params,
    )
    cage_row = cage_localization_benchmark_consistency(
        benchmark_id="debye_waller_cage_localization",
        observed_cage_localization=True,
        debye_waller_plateau=cage_diagnostic["debye_waller_plateau"],
        renewal_msd_fraction=cage_diagnostic["renewal_msd_fraction"],
        alpha_to_cage_time_ratio=cage_diagnostic["alpha_to_cage_time_ratio"],
        min_debye_waller_plateau=0.2,
        max_debye_waller_plateau=0.95,
        max_renewal_msd_fraction=0.05,
        min_alpha_to_cage_time_ratio=20.0,
    )
    mct_row = mct_beta_benchmark_consistency(
        beta,
        benchmark_id="kob_andersen_1995_beta_window",
        observed_critical_decay=False,
        observed_von_schweidler=True,
        observation_min_time=0.85 * beta.beta_time,
        observation_max_time=500.0 * beta.beta_time,
        alpha_time=80.0 * beta.beta_time,
        required_decades=0.5,
    )
    mct_exponent_row = mct_exponent_benchmark_consistency(
        benchmark_id="kob_andersen_1995_mct_exponent_parameter",
        observed_common_exponent_parameter=True,
        critical_exponent=beta.critical_exponent,
        von_schweidler_exponent=beta.von_schweidler_exponent,
        max_lambda_relative_mismatch=0.05,
    )
    late_time = 30000.0
    finite_exchange_late_ngp = float(gamma_exchange_ngp_1d(np.array([late_time]), params, heterogeneity)[0])
    static_gamma_late_ngp = float(static_gamma_ngp_1d(np.array([late_time]), params, heterogeneity.shape)[0])
    recovery_row = gaussian_recovery_benchmark_consistency(
        benchmark_id="gaussian_recovery_finite_exchange_vs_static_disorder",
        observed_gaussian_recovery=True,
        finite_exchange_late_ngp=finite_exchange_late_ngp,
        static_gamma_late_ngp=static_gamma_late_ngp,
        recovery_threshold=0.05,
    )
    ngp_peak_row = ngp_peak_benchmark_consistency(
        benchmark_id="ngp_peak_shift_on_cooling",
        observed_transient_ngp_peak=True,
        hot_peak_time=temperature_rows[0]["predicted_ngp_peak_time"],
        cold_peak_time=temperature_rows[-1]["predicted_ngp_peak_time"],
        hot_peak_ngp=temperature_rows[0]["predicted_ngp_peak"],
        cold_peak_ngp=temperature_rows[-1]["predicted_ngp_peak"],
        late_ngp=finite_exchange_late_ngp,
        min_peak_time_growth=2.0,
        min_peak_height=0.05,
        min_peak_height_growth=1.1,
        max_late_ngp=0.05,
    )
    se_scan = temperature_scan(np.array([1.0, 0.82, 0.72, 0.62]), temperature_law, wave_number=1.1)
    se_row = stokes_einstein_benchmark_consistency(
        benchmark_id="stokes_einstein_fractional_decoupling",
        observed_stokes_einstein_violation=True,
        hot_se_product=se_scan[0]["stokes_einstein_product"],
        cold_se_product=se_scan[-1]["stokes_einstein_product"],
        cold_fractional_exponent=se_scan[-1]["fractional_stokes_einstein_exponent"],
        min_product_growth=1.5,
        max_fractional_exponent=0.9,
    )
    cold_spatial = spatial_chi4_rows[-1]
    heterogeneity_row = dynamic_heterogeneity_benchmark_consistency(
        benchmark_id="dynamic_heterogeneity_chi4_growth",
        observed_dynamic_heterogeneity_growth=True,
        length_growth=cold_spatial["length_growth"],
        correlation_size_growth=cold_spatial["correlation_size_growth"],
        chi4_peak_growth=cold_spatial["chi4_peak_growth"],
        min_length_growth=1.5,
        min_correlation_size_growth=2.0,
        min_chi4_peak_growth=2.0,
    )
    spatial_front_row = {
        "benchmark_id": "spatial_facilitation_constant_front_law",
        **spatial_facilitation_growth_law_consistency(
            persistence_times=np.array([row["renewal_delay"] for row in spatial_chi4_rows], dtype=float),
            observed_dynamic_lengths=np.array(
                [row["dynamic_correlation_length"] for row in spatial_chi4_rows],
                dtype=float,
            ),
            observed_diffusive_front_growth=True,
            dimension=3,
            particle_density=0.85,
            microscopic_length=1.0,
            max_diffusivity_relative_std=0.05,
            min_length_growth=1.5,
        ),
    }
    alpha_summary_by_temperature: dict[float, dict[str, float | str]] = {}
    for row in alpha_shape_rows:
        alpha_summary_by_temperature.setdefault(float(row["temperature"]), row)
    if not alpha_summary_by_temperature:
        raise ValueError("alpha_shape_rows must be nonempty")
    if len(temperature_rows) < 2:
        raise ValueError("temperature_rows must contain at least hot and cold endpoints")
    hot_alpha = alpha_summary_by_temperature[max(alpha_summary_by_temperature)]
    cold_alpha = alpha_summary_by_temperature[min(alpha_summary_by_temperature)]
    tts_row = alpha_tts_benchmark_consistency(
        benchmark_id="alpha_tts_breakdown_shape_residual",
        observed_tts_breakdown=True,
        cold_shape_residual=float(cold_alpha["rms_log_shape_residual"]),
        alpha_shape_control_growth=float(cold_alpha["shape_control"]) / float(hot_alpha["shape_control"]),
        residual_threshold=0.25,
        min_control_growth=2.0,
    )
    if not kww_alpha_rows:
        raise ValueError("kww_alpha_rows must be nonempty")
    hot_kww = kww_alpha_rows[0]
    cold_kww = kww_alpha_rows[-1]
    stretched_row = stretched_alpha_benchmark_consistency(
        benchmark_id="kww_alpha_stretching_on_cooling",
        observed_stretched_alpha=True,
        hot_kww_beta=hot_kww["kww_beta"],
        cold_kww_beta=cold_kww["kww_beta"],
        min_beta_drop=0.05,
        max_cold_beta=0.9,
        max_fit_residual=0.08,
        cold_fit_residual=cold_kww["rms_log_residual"],
    )
    px_params = PersistenceExchangeParams(
        cage_variance=1.0,
        cage_tau=0.2,
        jump_variance=0.7,
        persistence_mean=9.0,
        exchange_mean=1.0,
    )
    wave_number = 1.1
    px_late_time = 80.0 * px_params.persistence_mean
    diffusion = persistence_exchange_diffusion_coefficient(px_params)
    tau_alpha = persistence_exchange_alpha_relaxation_time(wave_number, px_params)
    observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([px_late_time]), px_params)[0])
    inferred = infer_persistence_exchange_from_alpha_transport(
        wave_number=wave_number,
        jump_variance=px_params.jump_variance,
        diffusion_coefficient=diffusion,
        observed_tau_alpha=tau_alpha,
        cage_variance=px_params.cage_variance,
        cage_tau=px_params.cage_tau,
        late_time=px_late_time,
        observed_late_ngp=observed_late_ngp,
    )
    poisson_params = PersistenceExchangeParams(
        cage_variance=px_params.cage_variance,
        cage_tau=px_params.cage_tau,
        jump_variance=px_params.jump_variance,
        persistence_mean=px_params.exchange_mean,
        exchange_mean=px_params.exchange_mean,
    )
    invalid_poisson_alpha_rejected = False
    try:
        infer_persistence_exchange_from_alpha_transport(
            wave_number=wave_number,
            jump_variance=poisson_params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(poisson_params),
            observed_tau_alpha=0.8 * persistence_exchange_alpha_relaxation_time(wave_number, poisson_params),
            cage_variance=poisson_params.cage_variance,
            cage_tau=poisson_params.cage_tau,
        )
    except ValueError:
        invalid_poisson_alpha_rejected = True
    persistence_exchange_row = persistence_exchange_benchmark_consistency(
        benchmark_id="persistence_exchange_transport_inversion",
        observed_persistence_exchange_decoupling=True,
        inferred_persistence_exchange_ratio=inferred["persistence_exchange_ratio"],
        late_ngp_log_residual=inferred["late_ngp_log_residual"],
        invalid_poisson_alpha_rejected=invalid_poisson_alpha_rejected,
        min_persistence_exchange_ratio=2.0,
        max_late_ngp_abs_log_residual=0.1,
    )
    joint_summaries = {
        str(row["scenario"]): row
        for row in persistence_exchange_joint_protocol_rows
        if str(row["record_type"]) == "summary"
    }
    if "consistent" not in joint_summaries or "multik_alpha_mismatch" not in joint_summaries:
        raise ValueError("persistence_exchange_joint_protocol_rows must include consistent and mismatch summaries")
    joint_consistent = joint_summaries["consistent"]
    joint_mismatch = joint_summaries["multik_alpha_mismatch"]
    joint_row = joint_inversion_benchmark_consistency(
        benchmark_id="joint_persistence_exchange_multik_chi4_protocol",
        observed_joint_inversion_closure=True,
        inferred_persistence_exchange_ratio=float(joint_consistent["inferred_persistence_exchange_ratio"]),
        stokes_einstein_growth_over_poisson=float(joint_consistent["stokes_einstein_growth_over_poisson"]),
        max_multik_tau_alpha_abs_log_residual=float(joint_consistent["max_multik_tau_alpha_abs_log_residual"]),
        late_ngp_log_residual=float(joint_consistent["late_ngp_log_residual"]),
        chi4_peak_growth_over_poisson=float(joint_consistent["chi4_peak_growth_over_poisson"]),
        rejected_mismatch_abs_log_residual=float(joint_mismatch["max_multik_tau_alpha_abs_log_residual"]),
        min_persistence_exchange_ratio=2.0,
        min_stokes_einstein_growth=2.0,
        max_multik_abs_log_residual=0.02,
        max_late_ngp_abs_log_residual=0.02,
        min_chi4_peak_growth=1.5,
        min_rejected_mismatch_abs_log_residual=0.1,
    )
    tail_by_label = {str(row["time_label"]): row for row in tail_ratio_rows}
    if "t=11.3" not in tail_by_label or "t=80.0" not in tail_by_label:
        raise ValueError("tail_ratio_rows must include peak and late rows")
    peak_ngp = float(np.max(ngp_1d(np.linspace(0.0, 80.0, 800), params)))
    van_hove_row = van_hove_tail_benchmark_consistency(
        benchmark_id="kob_andersen_van_hove_tail_recovery",
        observed_transient_van_hove_tail=True,
        observed_late_gaussian_recovery=True,
        peak_tail_ratio=float(tail_by_label["t=11.3"]["tail_ratio"]),
        late_tail_ratio=float(tail_by_label["t=80.0"]["tail_ratio"]),
        peak_ngp=peak_ngp,
        min_peak_tail_ratio=1.5,
        max_late_tail_deviation=0.15,
        min_peak_ngp=0.05,
    )
    fragility_row = fragility_benchmark_consistency(
        benchmark_id="angell_adam_gibbs_fragility_growth",
        observed_fragility_growth=True,
        observed_adam_gibbs_slowdown=True,
        hot_activation_energy=temperature_rows[0]["apparent_alpha_activation_energy"],
        cold_activation_energy=temperature_rows[-1]["apparent_alpha_activation_energy"],
        hot_fragility_index=temperature_rows[0]["local_fragility_index"],
        cold_fragility_index=temperature_rows[-1]["local_fragility_index"],
        adam_gibbs_slowdown=thermodynamic_rows[-1]["thermodynamic_slowdown"],
        material_specific_origin_claimed=False,
        min_activation_growth=1.2,
        min_fragility_growth=1.5,
        min_adam_gibbs_slowdown=10.0,
    )
    thermodynamic_scope_row = thermodynamic_scope_benchmark_consistency(
        benchmark_id="thermodynamic_transition_scope_boundary",
        observed_heat_capacity_anomaly=True,
        observed_kauzmann_extrapolation=True,
        dynamic_model_derives_entropy=False,
        entropy_closure_supplied=True,
        adam_gibbs_slowdown=thermodynamic_rows[-1]["thermodynamic_slowdown"],
        min_adam_gibbs_slowdown=10.0,
        material_specific_entropy_origin_claimed=False,
    )
    rows = [
        normalize(cage_row, "cage_localization"),
        normalize(mct_row, "mct_beta_window"),
        normalize(mct_exponent_row, "mct_exponent_parameter"),
        normalize(recovery_row, "gaussian_recovery_mechanism_selection"),
        normalize(ngp_peak_row, "ngp_peak_shift"),
        normalize(se_row, "stokes_einstein_fractional_decoupling"),
        normalize(heterogeneity_row, "dynamic_heterogeneity_chi4_growth"),
        normalize(spatial_front_row, "spatial_facilitation_growth_law"),
        normalize(tts_row, "alpha_tts_breakdown"),
        normalize(stretched_row, "stretched_alpha_kww"),
        normalize(persistence_exchange_row, "persistence_exchange_inversion"),
        normalize(joint_row, "joint_inversion_falsification"),
        normalize(van_hove_row, "van_hove_tail_recovery"),
        normalize(fragility_row, "fragility_adam_gibbs"),
        normalize(thermodynamic_scope_row, "thermodynamic_scope_boundary"),
    ]
    write_sweep_csv(path, rows)
    return rows


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
) -> list[dict[str, float | str]]:
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
    return rows


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


def write_sota_comparison_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = [
        {
            "phenomenon": "msd_plateau_cage_localization",
            "benchmark_observable": "MSD plateau and cage rearrangements in colloids and simulations",
            "benchmark_source": "weeks2002cage;pastore2017cage",
            "model_prediction": "local cage variance L(t) plus delayed cage-center renewal",
            "model_status": "supported",
            "next_gap": "calibrate A and tau_c against trajectory-level cage metrics",
        },
        {
            "phenomenon": "transient_ngp_and_van_hove_tail",
            "benchmark_observable": "transient NGP peak and broad self van Hove tails",
            "benchmark_source": "kob1995vanhove;guan2014fickian;rusciano2022fickian",
            "model_prediction": "variance mixture from delayed renewal count gives NGP peak and transient radial tail",
            "model_status": "supported",
            "next_gap": "fit public trajectory data rather than synthetic curves",
        },
        {
            "phenomenon": "long_time_gaussian_recovery",
            "benchmark_observable": "non-Gaussianity decays after many rearrangements",
            "benchmark_source": "chubynsky2014diffusing;chechkin2017brownian;berthier2023comment",
            "model_prediction": "alpha_2 decays as inverse renewal count for finite exchange",
            "model_status": "supported",
            "next_gap": "quantify finite observation-window corrections in published datasets",
        },
        {
            "phenomenon": "alpha_relaxation_fs",
            "benchmark_observable": "self-intermediate scattering alpha relaxation and KWW-like decay",
            "benchmark_source": "kob1995intermediate",
            "model_prediction": "F_s(k,t) factorizes into cage plateau and renewal PGF alpha decay",
            "model_status": "supported",
            "next_gap": "compare full F_s(k,t) curves across wave numbers",
        },
        {
            "phenomenon": "stretched_alpha_relaxation",
            "benchmark_observable": "stretched relaxation and broad distribution of local relaxation times",
            "benchmark_source": "ediger2000spatial;kob1995intermediate",
            "model_prediction": "finite-exchange gamma renewal heterogeneity gives beta_loc<1 with Gaussian recovery",
            "model_status": "supported",
            "next_gap": "infer exchange ratio from real multi-k alpha slopes",
        },
        {
            "phenomenon": "stokes_einstein_violation",
            "benchmark_observable": "diffusion and structural relaxation decouple on cooling",
            "benchmark_source": "ediger2000spatial;berthier2011theoretical",
            "model_prediction": "persistence/exchange decoupling increases D tau_alpha at fixed diffusion",
            "model_status": "supported",
            "next_gap": "calibrate tau_p/tau_x against simulation or experiment",
        },
        {
            "phenomenon": "persistence_exchange_decoupling",
            "benchmark_observable": "first persistence time separates from subsequent exchange time in glass formers",
            "benchmark_source": "hedges2007persistence",
            "model_prediction": "D fixes tau_x, alpha time fixes tau_p, late NGP is held out",
            "model_status": "supported",
            "next_gap": "apply inversion to published or newly simulated trajectory clocks",
        },
        {
            "phenomenon": "tts_breakdown",
            "benchmark_observable": "alpha-shape superposition can fail as relaxation slows",
            "benchmark_source": "kob1995intermediate;berthier2011theoretical",
            "model_prediction": "scaled alpha shape controlled by C_k=Gamma_k lambda tau_d",
            "model_status": "partial",
            "next_gap": "benchmark residual against temperature series of F_s(k,t)",
        },
        {
            "phenomenon": "fragility_growth",
            "benchmark_observable": "apparent activation energy grows in fragile liquids and Adam-Gibbs slowdown can amplify relaxation",
            "benchmark_source": "angell1995formation;adam1965temperature;berthier2011theoretical",
            "model_prediction": "barrier law produces apparent activation and local fragility proxies while Adam-Gibbs enters as an entropy-driven closure",
            "model_status": "partial",
            "next_gap": "derive material-specific barriers from microscopic structure",
        },
        {
            "phenomenon": "spatial_chi4_length",
            "benchmark_observable": "four-point susceptibility and dynamic correlation length grow",
            "benchmark_source": "lacevic2003fourpoint;berthier2011theoretical;berthier2024experimental",
            "model_prediction": "diffusive facilitation front maps persistence time to xi4 and N_corr",
            "model_status": "partial",
            "next_gap": "replace the compact-domain closure by a full spatial four-point field theory",
        },
        {
            "phenomenon": "mct_beta_relaxation",
            "benchmark_observable": "beta-relaxation scaling, von Schweidler law, and MCT exponents",
            "benchmark_source": "gotze1992relaxation;kob1995intermediate;berthier2011theoretical",
            "model_prediction": "effective beta-window envelope gives critical decay and von Schweidler departure around the cage plateau",
            "model_status": "partial",
            "next_gap": "derive the memory kernel and exponent parameter from microscopic structure",
        },
        {
            "phenomenon": "thermodynamic_glass_transition",
            "benchmark_observable": "configurational entropy, heat-capacity anomaly, ideal-glass/Kauzmann questions",
            "benchmark_source": "kauzmann1948nature;adam1965temperature;lubchenko2007theory;berthier2011theoretical",
            "model_prediction": "Kauzmann entropy extrapolation plus Adam-Gibbs renewal slowdown",
            "model_status": "partial",
            "next_gap": "derive configurational entropy and barriers from microscopic structure",
        },
    ]
    write_sweep_csv(path, rows)
    return rows


def write_literature_inversion_readiness_csv(path: Path) -> list[dict[str, float | str]]:
    """Record whether benchmark papers are ready for quantitative inversion."""

    rows = [
        literature_inversion_readiness(
            benchmark_id="kob_andersen_van_hove_1995",
            benchmark_source="kob1995vanhove",
            required_observables=["time_grid", "van_hove_tail", "ngp", "diffusion"],
            available_observables=["time_grid", "van_hove_tail", "ngp", "diffusion"],
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
            next_action="digitize van-Hove/NGP curves or rerun a KA trajectory simulation",
        ),
        literature_inversion_readiness(
            benchmark_id="kob_andersen_intermediate_scattering_1995",
            benchmark_source="kob1995intermediate",
            required_observables=["time_grid", "self_intermediate_scattering", "tau_alpha", "wave_numbers"],
            available_observables=["time_grid", "self_intermediate_scattering", "tau_alpha", "wave_numbers"],
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
            next_action="digitize multi-k F_s curves and attach temperature-dependent uncertainty estimates",
        ),
        literature_inversion_readiness(
            benchmark_id="hedges_persistence_exchange_2007",
            benchmark_source="hedges2007persistence",
            required_observables=["diffusion", "tau_alpha", "persistence_time", "exchange_time", "late_ngp"],
            available_observables=["persistence_time", "exchange_time", "tau_alpha"],
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
            next_action="combine trajectory-clock extraction with diffusion and late-NGP measurements",
        ),
        literature_inversion_readiness(
            benchmark_id="weeks_weitz_cage_2002",
            benchmark_source="weeks2002cage",
            required_observables=["particle_trajectories", "msd_plateau", "cage_jump_events", "cage_time"],
            available_observables=["particle_trajectories", "msd_plateau", "cage_jump_events"],
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
            next_action="recover trajectory-level cage metrics in the renewal-cage observable schema",
        ),
        literature_inversion_readiness(
            benchmark_id="lacevic_four_point_2003",
            benchmark_source="lacevic2003fourpoint",
            required_observables=["tau_alpha", "chi4_peak", "dynamic_length", "diffusion"],
            available_observables=["tau_alpha", "chi4_peak", "dynamic_length"],
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
            next_action="combine four-point data with transport to test the facilitation-front closure",
        ),
        literature_inversion_readiness(
            benchmark_id="guan_granick_fickian_ngp_2014",
            benchmark_source="guan2014fickian",
            required_observables=["time_grid", "msd", "ngp", "van_hove_tail"],
            available_observables=["time_grid", "msd", "ngp", "van_hove_tail"],
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
            next_action="digitize colloid MSD/NGP/van-Hove curves for Brownian-non-Gaussian comparison",
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_sota_claim_alignment_csv(path: Path) -> list[dict[str, float | str]]:
    """Map source-level SOTA claims to model diagnostics and scope boundaries."""

    rows = [
        sota_claim_alignment(
            claim_id="kob_andersen_van_hove_caging_ngp",
            source_key="kob1995vanhove",
            phenomenon="cage_plateau_transient_ngp_van_hove_tail",
            claim_type="dynamical_signature",
            observed_claim="binary Lennard-Jones cooling shows caging, non-Gaussian displacement structure, and van-Hove tails",
            model_diagnostic="msd_ngp_van_hove_gaussian_recovery",
            model_support_level="derived",
            data_readiness="structural_raw",
            primary_blocker="uncertainty_columns",
        ),
        sota_claim_alignment(
            claim_id="kob_andersen_intermediate_scattering_alpha",
            source_key="kob1995intermediate",
            phenomenon="self_intermediate_scattering_alpha_relaxation",
            claim_type="dynamical_signature",
            observed_claim="self-intermediate scattering has plateau and alpha relaxation with temperature-dependent shape",
            model_diagnostic="multi_k_alpha_shape_raw_curve_protocol",
            model_support_level="derived",
            data_readiness="structural_raw",
            primary_blocker="uncertainty_columns",
        ),
        sota_claim_alignment(
            claim_id="hedges_persistence_exchange_decoupling",
            source_key="hedges2007persistence",
            phenomenon="persistence_exchange_decoupling",
            claim_type="transport_decoupling",
            observed_claim="persistence and exchange clocks decouple in atomistic glass formers",
            model_diagnostic="raw_curve_persistence_exchange_protocol",
            model_support_level="derived",
            data_readiness="qualitative",
            primary_blocker="machine_readable_joint_curves",
        ),
        sota_claim_alignment(
            claim_id="lacevic_four_point_dynamic_length",
            source_key="lacevic2003fourpoint",
            phenomenon="chi4_peak_and_dynamic_length_growth",
            claim_type="spatial_heterogeneity",
            observed_claim="four-point correlations reveal growing dynamic heterogeneity and a dynamic length scale",
            model_diagnostic="spatial_facilitation_chi4_proxy",
            model_support_level="effective_closure",
            data_readiness="qualitative",
            primary_blocker="shared_transport_and_four_point_grid",
        ),
        sota_claim_alignment(
            claim_id="jung_berthier_experimental_dynamic_heterogeneity",
            source_key="berthier2024experimental",
            phenomenon="experimental_dynamic_heterogeneity",
            claim_type="spatial_heterogeneity",
            observed_claim="machine-learning transfer predicts dynamic heterogeneity near experimental glass transition",
            model_diagnostic="chi4_proxy_and_front_diffusivity_closure",
            model_support_level="effective_closure",
            data_readiness="qualitative",
            primary_blocker="trajectory_level_chi4_input",
        ),
        sota_claim_alignment(
            claim_id="guan_granick_fickian_non_gaussian",
            source_key="guan2014fickian",
            phenomenon="fickian_non_gaussian_diffusion",
            claim_type="dynamical_signature",
            observed_claim="hard-sphere colloids can be Fickian while retaining non-Gaussian displacement distributions",
            model_diagnostic="transient_ngp_with_long_time_gaussian_recovery",
            model_support_level="derived",
            data_readiness="qualitative",
            primary_blocker="digitized_uncertainty_weighted_curves",
        ),
        sota_claim_alignment(
            claim_id="angell_fragility_growth",
            source_key="angell1995formation",
            phenomenon="fragility_growth",
            claim_type="dynamical_signature",
            observed_claim="fragile glass formers show rapidly growing apparent activation on cooling",
            model_diagnostic="activated_barrier_fragility_proxy",
            model_support_level="effective_closure",
            data_readiness="qualitative",
            primary_blocker="material_specific_barrier_origin",
        ),
        sota_claim_alignment(
            claim_id="kauzmann_adam_gibbs_entropy_boundary",
            source_key="kauzmann1948nature;adam1965temperature",
            phenomenon="configurational_entropy_and_ideal_glass_scope",
            claim_type="thermodynamic_transition",
            observed_claim="entropy extrapolation and Adam-Gibbs relaxation connect thermodynamics to slowdown",
            model_diagnostic="adam_gibbs_entropy_closure",
            model_support_level="closure_only",
            data_readiness="qualitative",
            primary_blocker="thermodynamic_input_law",
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_sota_signed_constraints_csv(path: Path) -> list[dict[str, float | str]]:
    """Encode representative SOTA conclusions as signed model constraints."""

    rows = [
        sota_signed_constraint_audit(
            constraint_id="kob_andersen_van_hove_signed_constraints",
            source_key="kob1995vanhove",
            model_scope="dynamical_signature",
            source_observation="KA cooling shows cage plateau, transient NGP, broad van-Hove tails, and recovery",
            expected_signatures=[
                "msd_plateau",
                "transient_ngp_peak",
                "van_hove_tail",
                "late_gaussian_recovery",
            ],
            passed_signatures=[
                "msd_plateau",
                "transient_ngp_peak",
                "van_hove_tail",
                "late_gaussian_recovery",
            ],
            forbidden_claims=["thermodynamic_transition_derived"],
            made_claims=["finite_exchange_dynamic_diagnostic"],
            support_level="derived",
            quantitative_fit_ready=False,
        ),
        sota_signed_constraint_audit(
            constraint_id="kob_andersen_alpha_signed_constraints",
            source_key="kob1995intermediate",
            model_scope="dynamical_signature",
            source_observation="KA self-intermediate scattering shows plateau, alpha slowing, and temperature-dependent shape",
            expected_signatures=[
                "self_intermediate_plateau",
                "alpha_slowing",
                "multi_k_alpha_shape",
                "tts_breakdown_diagnostic",
            ],
            passed_signatures=[
                "self_intermediate_plateau",
                "alpha_slowing",
                "multi_k_alpha_shape",
                "tts_breakdown_diagnostic",
            ],
            forbidden_claims=["unique_mct_critical_law_derived"],
            made_claims=["renewal_alpha_shape_diagnostic"],
            support_level="derived",
            quantitative_fit_ready=False,
        ),
        sota_signed_constraint_audit(
            constraint_id="hedges_persistence_exchange_signed_constraints",
            source_key="hedges2007persistence",
            model_scope="transport_decoupling",
            source_observation="persistence and exchange clocks decouple in atomistic glass formers",
            expected_signatures=[
                "persistence_exchange_ratio_growth",
                "stokes_einstein_product_growth",
                "late_ngp_recovery_constraint",
            ],
            passed_signatures=[
                "persistence_exchange_ratio_growth",
                "stokes_einstein_product_growth",
                "late_ngp_recovery_constraint",
            ],
            forbidden_claims=["static_disorder_as_only_mechanism"],
            made_claims=["finite_exchange_mechanism_selection"],
            support_level="derived",
            quantitative_fit_ready=False,
        ),
        sota_signed_constraint_audit(
            constraint_id="lacevic_four_point_signed_constraints",
            source_key="lacevic2003fourpoint",
            model_scope="spatial_heterogeneity",
            source_observation="four-point susceptibility and dynamic length grow on cooling",
            expected_signatures=["chi4_peak_growth", "dynamic_length_growth"],
            passed_signatures=["chi4_peak_growth", "dynamic_length_growth"],
            forbidden_claims=["microscopic_dynamic_length_derived"],
            made_claims=["chi4_proxy_closure"],
            support_level="effective_closure",
            quantitative_fit_ready=False,
        ),
        sota_signed_constraint_audit(
            constraint_id="jung_berthier_experimental_signed_constraints",
            source_key="berthier2024experimental",
            model_scope="spatial_heterogeneity",
            source_observation="experimental dynamic heterogeneity grows close to the laboratory glass transition",
            expected_signatures=["dynamic_heterogeneity_growth", "near_tg_trajectory_reanalysis_target"],
            passed_signatures=["dynamic_heterogeneity_growth", "near_tg_trajectory_reanalysis_target"],
            forbidden_claims=["trajectory_level_chi4_derived_without_data"],
            made_claims=["frontier_reanalysis_candidate"],
            support_level="effective_closure",
            quantitative_fit_ready=False,
        ),
        sota_signed_constraint_audit(
            constraint_id="guan_granick_fickian_ngp_signed_constraints",
            source_key="guan2014fickian",
            model_scope="dynamical_signature",
            source_observation="colloidal particles can be Fickian while displacement distributions remain non-Gaussian",
            expected_signatures=["fickian_diffusion_with_transient_ngp", "late_gaussian_recovery"],
            passed_signatures=["fickian_diffusion_with_transient_ngp", "late_gaussian_recovery"],
            forbidden_claims=["fickian_diffusion_implies_gaussian_displacements"],
            made_claims=["finite_exchange_fickian_non_gaussian_window"],
            support_level="derived",
            quantitative_fit_ready=False,
        ),
        sota_signed_constraint_audit(
            constraint_id="fragility_signed_constraints",
            source_key="angell1995formation;adam1965temperature",
            model_scope="dynamical_signature",
            source_observation="fragile liquids show growing apparent activation while Adam-Gibbs uses entropy input",
            expected_signatures=["apparent_activation_growth", "material_origin_not_derived"],
            passed_signatures=["apparent_activation_growth", "material_origin_not_derived"],
            forbidden_claims=["material_specific_fragility_origin_derived"],
            made_claims=["barrier_law_condition"],
            support_level="effective_closure",
            quantitative_fit_ready=False,
        ),
        sota_signed_constraint_audit(
            constraint_id="kauzmann_thermodynamic_signed_boundary",
            source_key="kauzmann1948nature;adam1965temperature",
            model_scope="thermodynamic_transition",
            source_observation="entropy extrapolation and heat-capacity anomalies require thermodynamic input",
            expected_signatures=["entropy_closure_required", "heat_capacity_not_derived"],
            passed_signatures=["entropy_closure_required", "heat_capacity_not_derived"],
            forbidden_claims=["ideal_glass_transition_derived", "heat_capacity_anomaly_derived"],
            made_claims=["thermodynamic_scope_boundary"],
            support_level="closure_only",
            quantitative_fit_ready=False,
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_real_benchmark_assimilation_gate_csv(path: Path) -> list[dict[str, float | str]]:
    """Gate real benchmark sources before any quantitative inversion claim."""

    rows = [
        real_benchmark_assimilation_gate(
            benchmark_id="kob_andersen_published_figures",
            source_key="kob1995vanhove;kob1995intermediate",
            target_protocol="alpha_vanhove_transport",
            available_observables=[
                "time_grid",
                "temperature_grid",
                "wave_numbers",
                "self_intermediate_scattering",
                "van_hove_tail",
                "ngp",
                "diffusion",
            ],
            has_shared_system=True,
            has_machine_readable_curves=False,
            has_uncertainty_estimates=False,
            model_scope="dynamical_signature",
        ),
        real_benchmark_assimilation_gate(
            benchmark_id="kob_andersen_structural_digitization_candidate",
            source_key="kob1995vanhove;kob1995intermediate",
            target_protocol="alpha_vanhove_transport",
            available_observables=[
                "time_grid",
                "temperature_grid",
                "wave_numbers",
                "self_intermediate_scattering",
                "van_hove_tail",
                "ngp",
                "diffusion",
            ],
            has_shared_system=True,
            has_machine_readable_curves=True,
            has_uncertainty_estimates=False,
            model_scope="dynamical_signature",
        ),
        real_benchmark_assimilation_gate(
            benchmark_id="hedges_persistence_exchange_published_curves",
            source_key="hedges2007persistence",
            target_protocol="persistence_exchange_chi4",
            available_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "persistence_time",
                "exchange_time",
            ],
            has_shared_system=True,
            has_machine_readable_curves=False,
            has_uncertainty_estimates=False,
            model_scope="transport_decoupling",
        ),
        real_benchmark_assimilation_gate(
            benchmark_id="lacevic_four_point_shared_transport_gap",
            source_key="lacevic2003fourpoint",
            target_protocol="spatial_chi4_front",
            available_observables=[
                "temperature_grid",
                "tau_alpha",
                "chi4_peak",
                "dynamic_length",
            ],
            has_shared_system=True,
            has_machine_readable_curves=False,
            has_uncertainty_estimates=False,
            model_scope="spatial_heterogeneity",
        ),
        real_benchmark_assimilation_gate(
            benchmark_id="berthier_ml_experimental_dynamic_heterogeneity",
            source_key="berthier2024experimental",
            target_protocol="spatial_chi4_front",
            available_observables=[
                "temperature_grid",
                "tau_alpha",
                "chi4_peak",
                "dynamic_length",
            ],
            has_shared_system=False,
            has_machine_readable_curves=False,
            has_uncertainty_estimates=False,
            model_scope="spatial_heterogeneity",
        ),
        real_benchmark_assimilation_gate(
            benchmark_id="guan_granick_fickian_non_gaussian_published_curves",
            source_key="guan2014fickian",
            target_protocol="alpha_vanhove_transport",
            available_observables=[
                "time_grid",
                "temperature_grid",
                "van_hove_tail",
                "ngp",
                "diffusion",
            ],
            has_shared_system=True,
            has_machine_readable_curves=False,
            has_uncertainty_estimates=False,
            model_scope="dynamical_signature",
        ),
        real_benchmark_assimilation_gate(
            benchmark_id="kauzmann_adam_gibbs_entropy_boundary",
            source_key="kauzmann1948nature;adam1965temperature",
            target_protocol="thermodynamic_entropy_closure",
            available_observables=["temperature_grid", "configurational_entropy", "tau_alpha"],
            has_shared_system=True,
            has_machine_readable_curves=True,
            has_uncertainty_estimates=True,
            model_scope="thermodynamic_transition",
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_cross_observable_prediction_ledger_csv(path: Path) -> list[dict[str, float | str]]:
    """Separate fitted inputs from held-out predictions for each diagnostic protocol."""

    rows = [
        cross_observable_prediction_ledger(
            protocol_id="alpha_vanhove_transport_raw_curves",
            source_key="kob1995vanhove;kob1995intermediate",
            model_scope="dynamical_signature",
            support_level="derived",
            calibration_observables=["diffusion", "debye_waller_plateau", "anchor_tau_alpha"],
            heldout_predictions=["multi_k_tau_alpha", "late_ngp", "van_hove_tail_recovery"],
            closure_observables=[],
            failed_predictions=[],
        ),
        cross_observable_prediction_ledger(
            protocol_id="joint_persistence_exchange_multik_chi4",
            source_key="raw_curve_persistence_exchange_protocol",
            model_scope="transport_decoupling",
            support_level="derived",
            calibration_observables=["diffusion", "anchor_tau_alpha"],
            heldout_predictions=[
                "multi_k_tau_alpha",
                "late_ngp",
                "stokes_einstein_product",
                "chi4_peak_proxy",
            ],
            closure_observables=[],
            failed_predictions=[],
        ),
        cross_observable_prediction_ledger(
            protocol_id="late_mechanism_selection",
            source_key="static_gamma_null;finite_exchange_renewal",
            model_scope="dynamical_signature",
            support_level="derived",
            calibration_observables=["late_ngp_point_1", "late_ngp_point_2"],
            heldout_predictions=["alpha_slope", "gaussian_recovery_class"],
            closure_observables=[],
            failed_predictions=[],
        ),
        cross_observable_prediction_ledger(
            protocol_id="spatial_chi4_front_closure",
            source_key="lacevic2003fourpoint;berthier2024experimental",
            model_scope="spatial_heterogeneity",
            support_level="effective_closure",
            calibration_observables=["tau_alpha", "diffusion"],
            heldout_predictions=["chi4_peak", "dynamic_length"],
            closure_observables=["front_diffusivity"],
            failed_predictions=[],
        ),
        cross_observable_prediction_ledger(
            protocol_id="single_alpha_fit_only_null",
            source_key="hypothetical_alpha_only_fit",
            model_scope="dynamical_signature",
            support_level="derived",
            calibration_observables=["tau_alpha"],
            heldout_predictions=[],
            closure_observables=[],
            failed_predictions=[],
        ),
        cross_observable_prediction_ledger(
            protocol_id="thermodynamic_entropy_boundary",
            source_key="kauzmann1948nature;adam1965temperature",
            model_scope="thermodynamic_transition",
            support_level="closure_only",
            calibration_observables=["configurational_entropy", "temperature_grid"],
            heldout_predictions=["heat_capacity_anomaly", "ideal_glass_transition"],
            closure_observables=["entropy_law"],
            failed_predictions=[],
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_inversion_identifiability_audit_csv(path: Path) -> list[dict[str, float | str]]:
    """Audit whether inversion protocols are identifiable before quantitative fitting."""

    rows = [
        inversion_identifiability_audit(
            protocol_id="scattering_transport_inversion",
            source_key="kob1995intermediate;delayed_renewal_closed_form",
            model_scope="dynamical_signature",
            fit_observables=[
                "debye_waller_plateau",
                "diffusion",
                "anchor_tau_alpha",
            ],
            inferred_parameters=["cage_variance", "jump_variance", "renewal_rate"],
            heldout_predictions=["ngp_peak_height", "ngp_peak_time", "late_gaussian_recovery"],
            external_closures=[],
            degenerate_parameters=[],
        ),
        inversion_identifiability_audit(
            protocol_id="full_observable_inversion",
            source_key="synthetic_full_ngp_scattering_protocol",
            model_scope="dynamical_signature",
            fit_observables=[
                "debye_waller_plateau",
                "diffusion",
                "ngp_peak_time",
                "ngp_peak_height",
            ],
            inferred_parameters=["cage_variance", "jump_variance", "renewal_rate", "renewal_delay"],
            heldout_predictions=["tau_alpha", "multi_k_alpha_shape"],
            external_closures=[],
            degenerate_parameters=[],
        ),
        inversion_identifiability_audit(
            protocol_id="joint_persistence_exchange_multik_chi4",
            source_key="raw_curve_persistence_exchange_protocol",
            model_scope="transport_decoupling",
            fit_observables=["diffusion", "anchor_tau_alpha"],
            inferred_parameters=["exchange_time", "persistence_time"],
            heldout_predictions=[
                "multi_k_tau_alpha",
                "late_ngp",
                "stokes_einstein_product",
                "chi4_peak_proxy",
            ],
            external_closures=[],
            degenerate_parameters=[],
        ),
        inversion_identifiability_audit(
            protocol_id="single_alpha_fit_only_null",
            source_key="hypothetical_alpha_only_fit",
            model_scope="dynamical_signature",
            fit_observables=["tau_alpha"],
            inferred_parameters=["exchange_time", "persistence_time"],
            heldout_predictions=[],
            external_closures=[],
            degenerate_parameters=[],
        ),
        inversion_identifiability_audit(
            protocol_id="static_vs_finite_exchange_late_window",
            source_key="late_mechanism_selection",
            model_scope="dynamical_signature",
            fit_observables=["late_ngp_point_1", "late_ngp_point_2", "alpha_slope"],
            inferred_parameters=["static_shape", "exchange_ratio"],
            heldout_predictions=["gaussian_recovery_class"],
            external_closures=[],
            degenerate_parameters=["static_shape_vs_exchange_ratio"],
        ),
        inversion_identifiability_audit(
            protocol_id="spatial_chi4_front_closure",
            source_key="lacevic2003fourpoint;berthier2024experimental",
            model_scope="spatial_heterogeneity",
            fit_observables=["tau_alpha", "diffusion", "chi4_peak"],
            inferred_parameters=["correlation_length", "front_diffusivity"],
            heldout_predictions=["dynamic_length"],
            external_closures=["front_diffusivity_law"],
            degenerate_parameters=[],
        ),
        inversion_identifiability_audit(
            protocol_id="thermodynamic_entropy_boundary",
            source_key="kauzmann1948nature;adam1965temperature",
            model_scope="thermodynamic_transition",
            fit_observables=["configurational_entropy", "temperature_grid"],
            inferred_parameters=["kauzmann_temperature", "entropy_slope"],
            heldout_predictions=["heat_capacity_anomaly", "ideal_glass_transition"],
            external_closures=["entropy_law"],
            degenerate_parameters=[],
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_frontier_benchmark_horizon_csv(path: Path) -> list[dict[str, float | str]]:
    """Track recent SOTA benchmarks as direct reanalysis, closure, or extension targets."""

    rows = [
        frontier_benchmark_horizon(
            benchmark_id="glassbench_trajectory_horizon",
            source_key="jung2025roadmap_glassbench",
            source_year=2025,
            model_scope="dynamical_signature",
            target_protocol="alpha_vanhove_transport",
            available_observables=[
                "particle_trajectories",
                "time_grid",
                "temperature_grid",
                "structure",
                "local_mobility_labels",
            ],
            required_observables=[
                "particle_trajectories",
                "time_grid",
                "temperature_grid",
                "self_intermediate_scattering",
                "ngp",
                "diffusion",
            ],
            has_machine_readable_repository=True,
            has_uncertainty_estimates=False,
            has_shared_transport_grid=True,
            requires_external_closure=False,
            model_extension_required=False,
        ),
        frontier_benchmark_horizon(
            benchmark_id="gst_nn_potential_transport_horizon",
            source_key="marcorini2025gst_dynamic_heterogeneity",
            source_year=2025,
            model_scope="transport_decoupling",
            target_protocol="joint_persistence_exchange_multik_chi4",
            available_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "stokes_einstein_product",
                "chi4_peak",
                "fragility_proxy",
            ],
            required_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "late_ngp",
                "multi_k_tau_alpha",
                "chi4_peak",
            ],
            has_machine_readable_repository=True,
            has_uncertainty_estimates=False,
            has_shared_transport_grid=True,
            requires_external_closure=False,
            model_extension_required=False,
        ),
        frontier_benchmark_horizon(
            benchmark_id="near_tg_molecular_motion_rotational_gap",
            source_key="simon2026molecular_motion",
            source_year=2026,
            model_scope="transport_decoupling",
            target_protocol="translation_rotation_persistence_exchange",
            available_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "rotational_relaxation",
                "stokes_einstein_product",
            ],
            required_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "rotational_relaxation",
            ],
            has_machine_readable_repository=True,
            has_uncertainty_estimates=False,
            has_shared_transport_grid=True,
            requires_external_closure=False,
            model_extension_required=False,
        ),
        frontier_benchmark_horizon(
            benchmark_id="experimental_dynamic_heterogeneity_closure_horizon",
            source_key="berthier2024experimental",
            source_year=2024,
            model_scope="spatial_heterogeneity",
            target_protocol="spatial_chi4_front_closure",
            available_observables=[
                "temperature_grid",
                "tau_alpha",
                "dynamic_length",
                "local_dynamic_heterogeneity",
            ],
            required_observables=[
                "temperature_grid",
                "tau_alpha",
                "diffusion",
                "chi4_peak",
                "dynamic_length",
            ],
            has_machine_readable_repository=False,
            has_uncertainty_estimates=False,
            has_shared_transport_grid=False,
            requires_external_closure=True,
            model_extension_required=False,
        ),
        frontier_benchmark_horizon(
            benchmark_id="heat_capacity_entropy_frontier",
            source_key="thermodynamic_calorimetry_candidate",
            source_year=2025,
            model_scope="thermodynamic_transition",
            target_protocol="thermodynamic_entropy_closure",
            available_observables=["temperature_grid", "configurational_entropy", "heat_capacity"],
            required_observables=["temperature_grid", "configurational_entropy", "heat_capacity"],
            has_machine_readable_repository=True,
            has_uncertainty_estimates=True,
            has_shared_transport_grid=True,
            requires_external_closure=True,
            model_extension_required=False,
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_sota_source_provenance_csv(path: Path) -> list[dict[str, float | str]]:
    """Gate SOTA sources by provenance before treating them as inversion inputs."""

    rows = [
        sota_source_provenance_gate(
            source_id="glassbench_zenodo_trajectory_release",
            citation_key="jung2025roadmap",
            source_type="dataset_repository",
            model_scope="dynamical_signature",
            provenance_items=[
                "doi",
                "repository_url",
                "machine_readable_files",
                "raw_particle_trajectories",
                "simulation_protocol_metadata",
                "license_or_terms",
            ],
            supported_observables=[
                "particle_trajectories",
                "time_grid",
                "temperature_grid",
                "structure",
            ],
            required_downstream_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_inversion_readiness_gate",
            ],
            has_reanalysis_permission=True,
        ),
        sota_source_provenance_gate(
            source_id="hedges_persistence_exchange_jcp_article",
            citation_key="hedges2007persistence",
            source_type="article",
            model_scope="transport_decoupling",
            provenance_items=["doi", "published_figures"],
            supported_observables=["persistence_time", "exchange_time", "tau_alpha"],
            required_downstream_protocols=[
                "persistence_exchange_protocol",
                "persistence_exchange_uncertainty_protocol",
            ],
            has_reanalysis_permission=False,
        ),
        sota_source_provenance_gate(
            source_id="kauzmann_entropy_thermodynamic_boundary",
            citation_key="kauzmann1948nature",
            source_type="article",
            model_scope="thermodynamic_transition",
            provenance_items=["doi", "published_figures", "thermodynamic_observables"],
            supported_observables=["configurational_entropy", "heat_capacity"],
            required_downstream_protocols=["thermodynamic_entropy_closure"],
            has_reanalysis_permission=True,
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_sota_data_accession_csv(path: Path) -> list[dict[str, float | str]]:
    """Record public data accessions before attempting local reanalysis."""

    rows = [
        sota_data_accession_gate(
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            citation_key="jung2025roadmap",
            model_scope="dynamical_signature",
            landing_url="https://zenodo.org/records/10118191",
            doi="10.5281/zenodo.10118191",
            archive_name="GlassBench.zip",
            archive_md5="82c83a7146eb749e13417e4350022417",
            archive_size_bytes=6042260027,
            license_id="cc-by-4.0",
            has_public_landing_page=True,
            has_downloadable_archive=True,
            has_schema_or_readme=True,
            has_trajectory_files=True,
            has_precomputed_descriptors=True,
            local_cache_present=False,
            intended_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_inversion_readiness_gate",
            ],
        ),
        sota_data_accession_gate(
            accession_id="hedges_jcp_article_no_archive",
            source_id="hedges_persistence_exchange_jcp_article",
            citation_key="hedges2007persistence",
            model_scope="transport_decoupling",
            landing_url="https://doi.org/10.1063/1.2817607",
            doi="10.1063/1.2817607",
            archive_name="none",
            archive_md5="none",
            archive_size_bytes=0,
            license_id="article",
            has_public_landing_page=True,
            has_downloadable_archive=False,
            has_schema_or_readme=False,
            has_trajectory_files=False,
            has_precomputed_descriptors=False,
            local_cache_present=False,
            intended_protocols=["persistence_exchange_uncertainty_protocol"],
        ),
        sota_data_accession_gate(
            accession_id="kauzmann_entropy_scope_boundary",
            source_id="kauzmann_entropy_thermodynamic_boundary",
            citation_key="kauzmann1948nature",
            model_scope="thermodynamic_transition",
            landing_url="https://doi.org/10.1021/cr60135a002",
            doi="10.1021/cr60135a002",
            archive_name="none",
            archive_md5="none",
            archive_size_bytes=0,
            license_id="article",
            has_public_landing_page=True,
            has_downloadable_archive=False,
            has_schema_or_readme=False,
            has_trajectory_files=False,
            has_precomputed_descriptors=False,
            local_cache_present=False,
            intended_protocols=["thermodynamic_entropy_closure"],
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_sota_readme_schema_csv(path: Path) -> list[dict[str, float | str]]:
    """Record README-level schema evidence for remote SOTA trajectory archives."""

    rows = [
        sota_readme_schema_gate(
            schema_id="glassbench_readme_schema",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            systems=["KA", "KA2D"],
            folder_tokens=["_trajectories", "_models", "_results"],
            license_statement="Creative Commons Attribution 4.0 International",
            required_citations=["10.1063/5.0129791", "10.1103/PhysRevLett.130.238202"],
            intended_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_inversion_readiness_gate",
            ],
            local_archive_inspected=False,
        ),
        sota_readme_schema_gate(
            schema_id="hedges_schema_missing_trajectories",
            accession_id="hedges_jcp_article_no_archive",
            source_id="hedges_persistence_exchange_jcp_article",
            systems=["KA"],
            folder_tokens=["_models", "_results"],
            license_statement="article",
            required_citations=["10.1063/1.2817607"],
            intended_protocols=["trajectory_observable_protocol"],
            local_archive_inspected=False,
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_trajectory_adapter_contract_csv(path: Path) -> list[dict[str, float | str]]:
    """Record local trajectory-adapter requirements after remote schema accession."""

    required_fields = [
        "archive_root",
        "trajectory_folder",
        "coordinate_file",
        "time_grid",
        "particle_identity",
        "box_geometry",
        "temperature_or_state_point",
        "species_labels",
        "units_metadata",
    ]
    protocols = [
        "trajectory_observable_protocol",
        "trajectory_uncertainty_protocol",
        "trajectory_inversion_readiness_gate",
    ]
    rows = [
        trajectory_adapter_contract(
            contract_id="glassbench_ka_remote_contract",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            system_id="KA",
            expected_archive_roots=["KA/_trajectories", "KA/_models", "KA/_results"],
            required_local_fields=required_fields,
            available_local_fields=["archive_root", "trajectory_folder"],
            intended_protocols=protocols,
            local_archive_inspected=False,
        ),
        trajectory_adapter_contract(
            contract_id="glassbench_ka2d_remote_contract",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            system_id="KA2D",
            expected_archive_roots=["KA2D/_trajectories", "KA2D/_models", "KA2D/_results"],
            required_local_fields=required_fields,
            available_local_fields=["archive_root", "trajectory_folder"],
            intended_protocols=protocols,
            local_archive_inspected=False,
        ),
        trajectory_adapter_contract(
            contract_id="synthetic_local_trajectory_adapter",
            accession_id="synthetic_local_cache",
            source_id="synthetic_intermediate_scattering_fixture",
            system_id="synthetic",
            expected_archive_roots=["synthetic/_trajectories"],
            required_local_fields=required_fields,
            available_local_fields=required_fields,
            intended_protocols=protocols,
            local_archive_inspected=True,
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_observable_falsification_matrix_csv(
    path: Path,
    literature_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Map benchmark observables to the diagnostics they can structurally falsify."""

    diagnostic_requirements = {
        "van_hove_gaussian_recovery": ["time_grid", "van_hove_tail", "ngp", "diffusion"],
        "multi_k_alpha_shape": [
            "time_grid",
            "self_intermediate_scattering",
            "tau_alpha",
            "wave_numbers",
        ],
        "joint_persistence_exchange_chi4": [
            "diffusion",
            "tau_alpha",
            "persistence_time",
            "exchange_time",
            "late_ngp",
            "chi4_peak",
        ],
        "stokes_einstein_decoupling": ["temperature_grid", "diffusion", "tau_alpha"],
        "four_point_facilitation_proxy": ["tau_alpha", "chi4_peak", "dynamic_length", "diffusion"],
    }
    rows: list[dict[str, float | str]] = []
    for literature_row in literature_rows:
        available = [
            observable
            for observable in str(literature_row["available_observables"]).split(";")
            if observable and observable != "none"
        ]
        rows.extend(
            observable_falsification_matrix(
                benchmark_id=str(literature_row["benchmark_id"]),
                benchmark_source=str(literature_row["benchmark_source"]),
                available_observables=available,
                diagnostic_requirements=diagnostic_requirements,
                has_machine_readable_data=bool(float(literature_row["has_machine_readable_data"])),
                has_uncertainty_estimates=bool(float(literature_row["has_uncertainty_estimates"])),
            )
        )
    write_sweep_csv(path, rows)
    return rows


def write_benchmark_fusion_readiness_csv(path: Path) -> list[dict[str, float | str]]:
    """Record whether multi-paper benchmark fusions preserve joint-diagnostic identity."""

    rows = [
        benchmark_fusion_readiness(
            fusion_id="kob_andersen_i_ii_dynamic_closure",
            benchmark_sources=["kob1995vanhove", "kob1995intermediate"],
            required_observables=[
                "time_grid",
                "van_hove_tail",
                "ngp",
                "diffusion",
                "self_intermediate_scattering",
                "tau_alpha",
                "wave_numbers",
            ],
            available_observables_by_benchmark={
                "kob1995vanhove": ["time_grid", "van_hove_tail", "ngp", "diffusion"],
                "kob1995intermediate": [
                    "time_grid",
                    "self_intermediate_scattering",
                    "tau_alpha",
                    "wave_numbers",
                ],
            },
            system_tags={
                "kob1995vanhove": "kob_andersen_binary_lj",
                "kob1995intermediate": "kob_andersen_binary_lj",
            },
            temperature_grid_tags={
                "kob1995vanhove": "ka_1995_grid",
                "kob1995intermediate": "ka_1995_grid",
            },
            ensemble_tags={
                "kob1995vanhove": "ka_1995_simulation",
                "kob1995intermediate": "ka_1995_simulation",
            },
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
        ),
        benchmark_fusion_readiness(
            fusion_id="ka_lacevic_four_point_splice",
            benchmark_sources=["kob1995intermediate", "lacevic2003fourpoint"],
            required_observables=[
                "self_intermediate_scattering",
                "tau_alpha",
                "chi4_peak",
                "dynamic_length",
            ],
            available_observables_by_benchmark={
                "kob1995intermediate": ["self_intermediate_scattering", "tau_alpha"],
                "lacevic2003fourpoint": ["tau_alpha", "chi4_peak", "dynamic_length"],
            },
            system_tags={
                "kob1995intermediate": "kob_andersen_binary_lj",
                "lacevic2003fourpoint": "kob_andersen_binary_lj",
            },
            temperature_grid_tags={
                "kob1995intermediate": "ka_1995_grid",
                "lacevic2003fourpoint": "lacevic_2003_grid",
            },
            ensemble_tags={
                "kob1995intermediate": "ka_1995_simulation",
                "lacevic2003fourpoint": "lacevic_2003_simulation",
            },
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
        ),
        benchmark_fusion_readiness(
            fusion_id="hedges_lacevic_exchange_chi4_splice",
            benchmark_sources=["hedges2007persistence", "lacevic2003fourpoint", "kob1995vanhove"],
            required_observables=[
                "diffusion",
                "tau_alpha",
                "persistence_time",
                "exchange_time",
                "late_ngp",
                "chi4_peak",
            ],
            available_observables_by_benchmark={
                "hedges2007persistence": ["persistence_time", "exchange_time", "tau_alpha"],
                "lacevic2003fourpoint": ["tau_alpha", "chi4_peak", "dynamic_length"],
                "kob1995vanhove": ["time_grid", "ngp", "diffusion"],
            },
            system_tags={
                "hedges2007persistence": "atomistic_glass_former",
                "lacevic2003fourpoint": "kob_andersen_binary_lj",
                "kob1995vanhove": "kob_andersen_binary_lj",
            },
            temperature_grid_tags={
                "hedges2007persistence": "hedges_2007_grid",
                "lacevic2003fourpoint": "lacevic_2003_grid",
                "kob1995vanhove": "ka_1995_grid",
            },
            ensemble_tags={
                "hedges2007persistence": "hedges_2007_simulation",
                "lacevic2003fourpoint": "lacevic_2003_simulation",
                "kob1995vanhove": "ka_1995_simulation",
            },
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


def write_raw_curve_ingestion_contract_csv(path: Path) -> list[dict[str, float | str]]:
    """Define the machine-readable curve contract for the KA I/II fused benchmark."""

    rows = raw_curve_ingestion_contract(
        benchmark_id="kob_andersen_i_ii_dynamic_closure",
        observable_requirements={
            "ka_self_intermediate_scattering": {
                "required_columns": ["temperature", "wave_number", "time", "F_s"],
                "uncertainty_columns": ["sigma_F_s"],
                "target_diagnostic": "multi_k_alpha_shape",
            },
            "ka_van_hove_ngp": {
                "required_columns": ["temperature", "time", "radius", "G_s", "alpha2", "diffusion"],
                "uncertainty_columns": ["sigma_G_s", "sigma_alpha2", "sigma_diffusion"],
                "target_diagnostic": "van_hove_gaussian_recovery",
            },
            "ka_joint_transport_alpha": {
                "required_columns": [
                    "temperature",
                    "wave_number",
                    "tau_alpha",
                    "diffusion",
                    "alpha_shape_residual",
                ],
                "uncertainty_columns": ["sigma_tau_alpha", "sigma_diffusion", "sigma_alpha_shape_residual"],
                "target_diagnostic": "stokes_einstein_and_tts",
            },
        },
        available_columns_by_observable={
            "ka_self_intermediate_scattering": ["temperature", "wave_number", "time", "F_s"],
            "ka_van_hove_ngp": ["temperature", "time", "radius", "G_s", "alpha2", "diffusion"],
            "ka_joint_transport_alpha": [
                "temperature",
                "wave_number",
                "tau_alpha",
                "diffusion",
                "alpha_shape_residual",
            ],
        },
        machine_readable=True,
        shared_temperature_grid=True,
        shared_time_units=True,
    )
    write_sweep_csv(path, rows)
    return rows


def write_raw_curve_diagnostic_readiness_csv(
    path: Path,
    contract_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Aggregate KA raw-curve ingestion contracts into diagnostic readiness."""

    rows = raw_curve_diagnostic_readiness(
        benchmark_id="kob_andersen_i_ii_dynamic_closure",
        contract_rows=contract_rows,
        diagnostic_observables={
            "multi_k_alpha_shape": ["ka_self_intermediate_scattering"],
            "van_hove_gaussian_recovery": ["ka_van_hove_ngp"],
            "combined_alpha_vanhove_transport_closure": [
                "ka_self_intermediate_scattering",
                "ka_van_hove_ngp",
                "ka_joint_transport_alpha",
            ],
        },
    )
    write_sweep_csv(path, rows)
    return rows


def write_raw_curve_persistence_exchange_protocol_csv(path: Path) -> list[dict[str, float | str]]:
    """Run the persistence/exchange protocol from synthetic machine-readable curves."""

    params = PersistenceExchangeParams(
        cage_variance=1.0,
        cage_tau=0.2,
        jump_variance=0.7,
        persistence_mean=7.0,
        exchange_mean=1.0,
    )
    wave_numbers = [0.7, 1.1, 1.6]
    alpha_time = np.geomspace(0.02, 800.0, 1600)
    alpha_curves = {
        wave_number: (
            alpha_time,
            persistence_exchange_normalized_alpha_decay(wave_number, alpha_time, params),
        )
        for wave_number in wave_numbers
    }
    late_time = 80.0 * params.persistence_mean
    ngp_time = np.geomspace(0.1, 1200.0, 1400)
    ngp_values = persistence_exchange_ngp_1d(ngp_time, params)
    chi4_time = np.geomspace(0.02, 400.0, 900)
    chi4_curve = (
        chi4_time,
        persistence_exchange_scattering_susceptibility(1.1, chi4_time, params),
    )
    common_kwargs = dict(
        anchor_wave_number=1.1,
        alpha_curves_by_k=alpha_curves,
        jump_variance=params.jump_variance,
        diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
        late_time=late_time,
        chi4_curve=chi4_curve,
        tau_alpha_relative_error_by_k={wave_number: 0.03 for wave_number in wave_numbers},
        late_ngp_relative_error=0.05,
        chi4_peak_relative_error=0.05,
        cage_variance=params.cage_variance,
        cage_tau=params.cage_tau,
        z_threshold=3.0,
    )
    rows = []
    for scenario, multiplier in [
        ("consistent_raw_curves", 1.0),
        ("late_ngp_mismatch", 1.8),
    ]:
        row = raw_curve_persistence_exchange_protocol(
            benchmark_id=f"synthetic_{scenario}",
            ngp_curve=(ngp_time, multiplier * ngp_values),
            **common_kwargs,
        )
        row["scenario"] = scenario
        row["late_ngp_multiplier"] = multiplier
        rows.append(row)
    write_sweep_csv(path, rows)
    return rows


def synthetic_intermittent_trajectory() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic trajectory with intermittent mobile domains."""
    frame_count = 9
    particle_count = 12
    times = np.arange(frame_count, dtype=float)
    increments = np.zeros((frame_count - 1, particle_count, 1), dtype=float)
    for step in range(frame_count - 1):
        if step % 2 == 0:
            increments[step, particle_count // 2 :, 0] = 2.0
        if step in {3, 4}:
            increments[step, : particle_count // 3, 0] = -1.5
    positions = np.concatenate(
        [
            np.zeros((1, particle_count, 1), dtype=float),
            np.cumsum(increments, axis=0),
        ],
        axis=0,
    )
    return positions, times


def synthetic_intermittent_trajectory_table() -> list[dict[str, float | int | str]]:
    """Serialize the deterministic intermittent trajectory as a local particle table."""

    positions, times = synthetic_intermittent_trajectory()
    records: list[dict[str, float | int | str]] = []
    for frame_idx, time_value in enumerate(times):
        for particle_idx in range(positions.shape[1]):
            record: dict[str, float | int | str] = {
                "frame": int(frame_idx),
                "time": float(time_value),
                "particle_id": f"p{particle_idx:02d}",
                "x": float(positions[frame_idx, particle_idx, 0]),
            }
            records.append(record)
    # Reverse the table to prove the adapter, not insertion order, defines arrays.
    return list(reversed(records))


def write_trajectory_observable_protocol_csv(path: Path) -> list[dict[str, float | str]]:
    """Extract raw observables from a deterministic intermittent trajectory."""

    positions, times = synthetic_intermittent_trajectory()
    rows = trajectory_observable_protocol(
        positions=positions,
        times=times,
        lag_indices=[1, 2, 3, 4],
        wave_numbers=[0.7, 1.1, 1.6],
        overlap_radius=0.5,
    )
    for row in rows:
        row["benchmark_id"] = "synthetic_intermittent_trajectory"
        row["target_protocol"] = "trajectory_to_raw_curve_bridge"
        row["machine_readable_trajectory"] = 1.0
        row["uncertainty_estimates"] = 0.0
        row["primary_blocker"] = "uncertainty_estimates"
    write_sweep_csv(path, rows)
    return rows


def write_trajectory_adapter_demo_csv(path: Path) -> list[dict[str, float | str]]:
    """Extract trajectory observables through the local particle-table adapter."""

    adapted = trajectory_table_adapter(
        records=synthetic_intermittent_trajectory_table(),
        frame_column="frame",
        time_column="time",
        particle_column="particle_id",
        coordinate_columns=["x"],
    )
    rows = trajectory_observable_protocol(
        positions=adapted["positions"],
        times=adapted["times"],
        lag_indices=[1, 2, 3, 4],
        wave_numbers=[0.7, 1.1, 1.6],
        overlap_radius=0.5,
    )
    for row in rows:
        row["benchmark_id"] = "synthetic_intermittent_trajectory_table"
        row["adapter_source"] = "synthetic_local_particle_table"
        row["adapter_ready"] = adapted["adapter_ready"]
        row["frame_count"] = adapted["frame_count"]
        row["particle_count"] = adapted["particle_count"]
        row["dimension"] = adapted["dimension"]
        row["coordinate_columns"] = adapted["coordinate_columns"]
        row["target_protocol"] = "local_table_adapter_to_observable_bridge"
        row["primary_blocker"] = "none"
    write_sweep_csv(path, rows)
    return rows


def write_trajectory_csv_adapter_demo_csv(path: Path) -> list[dict[str, float | str]]:
    """Extract trajectory observables through a local CSV file adapter."""

    source_path = DATA_DIR / "renewal_cage_trajectory_csv_adapter_source.csv"
    source_records = synthetic_intermittent_trajectory_table()
    write_sweep_csv(source_path, source_records)
    adapted = trajectory_table_csv_adapter(
        csv_path=source_path,
        frame_column="frame",
        time_column="time",
        particle_column="particle_id",
        coordinate_columns=["x"],
        metadata={
            "box_geometry": "one_dimensional_periodic_fixture",
            "temperature_or_state_point": "synthetic_intermittent_state",
            "species_labels": "A",
            "units_metadata": "reduced_LJ_units",
        },
    )
    rows = trajectory_observable_protocol(
        positions=adapted["positions"],
        times=adapted["times"],
        lag_indices=[1, 2, 3, 4],
        wave_numbers=[0.7, 1.1, 1.6],
        overlap_radius=0.5,
    )
    for row in rows:
        row["benchmark_id"] = "synthetic_intermittent_trajectory_csv"
        row["adapter_source"] = "synthetic_local_csv_file"
        row["adapter_stage"] = adapted["adapter_stage"]
        row["adapter_ready"] = adapted["adapter_ready"]
        row["missing_metadata_fields"] = adapted["missing_metadata_fields"]
        row["primary_blocker"] = adapted["primary_blocker"]
        row["row_count"] = adapted["row_count"]
        row["csv_columns"] = adapted["csv_columns"]
        row["box_geometry"] = adapted["box_geometry"]
        row["temperature_or_state_point"] = adapted["temperature_or_state_point"]
        row["species_labels"] = adapted["species_labels"]
        row["units_metadata"] = adapted["units_metadata"]
        row["target_protocol"] = "csv_file_adapter_to_observable_bridge"
    write_sweep_csv(path, rows)
    return rows


def write_trajectory_uncertainty_protocol_csv(path: Path) -> list[dict[str, float | str]]:
    """Estimate trajectory-observable uncertainties from time-origin jackknife blocks."""

    positions, times = synthetic_intermittent_trajectory()
    rows = trajectory_observable_uncertainty_protocol(
        positions=positions,
        times=times,
        lag_indices=[1, 2, 3, 4],
        wave_numbers=[0.7, 1.1, 1.6],
        overlap_radius=0.5,
        block_count=4,
    )
    for row in rows:
        row["benchmark_id"] = "synthetic_intermittent_trajectory"
        row["target_protocol"] = "trajectory_uncertainty_to_raw_curve_bridge"
        row["machine_readable_trajectory"] = 1.0
    write_sweep_csv(path, rows)
    return rows


def write_trajectory_inversion_readiness_csv(
    path: Path,
    *,
    observable_rows: list[dict[str, float | str]],
    uncertainty_rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Gate trajectory-derived observables before promoting them to inversion."""

    required_observables = ["msd", "ngp", "self_intermediate_scattering", "overlap_chi4"]
    required_uncertainty_columns = [
        "sigma_msd",
        "sigma_ngp",
        "sigma_self_intermediate_scattering",
        "sigma_chi4_overlap",
    ]
    rows = [
        trajectory_inversion_readiness_gate(
            benchmark_id="synthetic_intermittent_trajectory_uncertainty",
            source_key="synthetic_trajectory_reanalysis",
            target_protocol="trajectory_alpha_vanhove_chi4_transport",
            trajectory_rows=uncertainty_rows,
            required_observables=required_observables,
            required_uncertainty_columns=required_uncertainty_columns,
            has_shared_time_grid=True,
            has_shared_particle_identity=True,
        ),
        trajectory_inversion_readiness_gate(
            benchmark_id="synthetic_intermittent_trajectory_structural_only",
            source_key="synthetic_trajectory_reanalysis",
            target_protocol="trajectory_alpha_vanhove_chi4_transport",
            trajectory_rows=observable_rows,
            required_observables=required_observables,
            required_uncertainty_columns=required_uncertainty_columns,
            has_shared_time_grid=True,
            has_shared_particle_identity=True,
        ),
    ]
    write_sweep_csv(path, rows)
    return rows


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


def write_spatial_chi4_svg(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left_a, top, right_a, bottom = 75, 90, 520, 430
    left_b, right_b = 660, 1040
    inverse_shift = np.array([1.0 / row["temperature"] - 1.0 / rows[0]["temperature"] for row in rows])
    length_growth = np.array([row["length_growth"] for row in rows])
    size_growth = np.array([row["correlation_size_growth"] for row in rows])
    chi4_growth = np.array([row["chi4_peak_growth"] for row in rows])
    peak_times = np.array([row["chi4_peak_time"] for row in rows])
    tau_alpha = np.array([row["tau_alpha"] for row in rows])

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 76}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, curves: list[tuple[str, np.ndarray, str]]) -> str:
        x = scale(inverse_shift, left, right)
        all_values = np.concatenate([curve for _, curve, _ in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve, color) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, color))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{color}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Spatial facilitation chi4 closure</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">A diffusive facilitation front turns the persistence clock into xi4 and Ncorr.</text>
  {axes(left_a, right_a, "A. Clock-derived dynamic length", "inverse-temperature shift")}
  {plot(left_a, right_a, [("xi4 / hot", length_growth, "#2b6cb0"), ("Ncorr / hot", size_growth, "#2f855a"), ("chi4 peak / hot", chi4_growth, "#c05621")])}
  {axes(left_b, right_b, "B. Timing of the spatial susceptibility", "inverse-temperature shift")}
  {plot(left_b, right_b, [("chi4 peak time", peak_times / peak_times[0], "#805ad5"), ("tau alpha", tau_alpha / tau_alpha[0], "#d69e2e")])}
</svg>
"""
    path.write_text(svg)


def write_thermodynamic_closure_svg(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left_a, top, right_a, bottom = 75, 90, 520, 430
    left_b, right_b = 660, 1040
    inverse_shift = np.array([1.0 / row["temperature"] - 1.0 / rows[0]["temperature"] for row in rows])
    entropy = np.array([row["configurational_entropy"] for row in rows])
    heat_capacity = np.array([row["excess_heat_capacity"] for row in rows])
    tau_ag = np.array([row["thermodynamic_slowdown"] for row in rows])
    tau_alpha = np.array([row["tau_alpha_growth"] for row in rows])

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 76}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot(left: int, right: int, curves: list[tuple[str, np.ndarray, str]]) -> str:
        x = scale(inverse_shift, left, right)
        all_values = np.concatenate([curve for _, curve, _ in curves])
        y_all = scale(all_values, bottom, top)
        out = []
        start = 0
        for idx, (label, curve, color) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, color))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{color}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Thermodynamic entropy closure</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Kauzmann entropy extrapolation drives Adam-Gibbs renewal slowdown.</text>
  {axes(left_a, right_a, "A. Configurational entropy sector", "inverse-temperature shift")}
  {plot(left_a, right_a, [("s_c", entropy / entropy[0], "#2b6cb0"), ("Delta c_p", heat_capacity / heat_capacity[0], "#2f855a")])}
  {axes(left_b, right_b, "B. Adam-Gibbs kinetic coupling", "inverse-temperature shift")}
  {plot(left_b, right_b, [("tau_AG / hot", tau_ag, "#c05621"), ("tau_alpha / hot", tau_alpha, "#805ad5")])}
</svg>
"""
    path.write_text(svg)


def write_mct_beta_closure_svg(path: Path, rows: list[dict[str, float]], base: MCTBetaParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left_a, top, right_a, bottom = 75, 90, 520, 430
    left_b, right_b = 660, 1040
    scaled_time = np.geomspace(0.1, 20.0, 260)
    time = scaled_time * base.beta_time
    correlator = mct_beta_correlator(time, base)
    plateau = np.full_like(correlator, base.plateau)
    inverse_shift = np.array([row["inverse_temperature_shift"] for row in rows])
    beta_time = np.array([row["beta_time"] for row in rows])
    exit_time = np.array([row["von_schweidler_exit_time"] for row in rows])
    alpha_time = np.array([row["alpha_time"] for row in rows])

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 70}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    def plot_scaled_time(left: int, right: int) -> str:
        log_scaled_time = np.log10(scaled_time)
        x = scale(log_scaled_time, left, right)
        y_all = scale(np.concatenate([correlator, plateau]), bottom, top)
        y_corr = y_all[: len(correlator)]
        y_plateau = y_all[len(correlator) :]
        marker_x = left + (0.0 - np.min(log_scaled_time)) * (right - left) / (
            np.max(log_scaled_time) - np.min(log_scaled_time)
        )
        return "\n".join(
            [
                polyline(x, y_corr, "#2b6cb0"),
                polyline(x, y_plateau, "#555555", width=1.2),
                f'<line x1="{marker_x:.2f}" y1="{top}" x2="{marker_x:.2f}" y2="{bottom}" stroke="#888" stroke-dasharray="4 4" />',
                f'<text x="{left + 18}" y="{top + 25}" font-family="Arial, sans-serif" font-size="12" fill="#2b6cb0">phi_beta(t)</text>',
                f'<text x="{left + 18}" y="{top + 43}" font-family="Arial, sans-serif" font-size="12" fill="#555555">plateau f_c</text>',
            ]
        )

    def plot_cooling(left: int, right: int) -> str:
        x = scale(inverse_shift, left, right)
        curves = [
            ("t_beta / hot", beta_time / beta_time[0], "#2b6cb0"),
            ("von exit / hot", exit_time / exit_time[0], "#c05621"),
            ("tau alpha / hot", alpha_time / alpha_time[0], "#805ad5"),
        ]
        all_values = np.concatenate([curve for _, curve, _ in curves])
        y_all = scale(np.log10(all_values), bottom, top)
        out = []
        start = 0
        for idx, (label, curve, color) in enumerate(curves):
            segment = y_all[start : start + len(curve)]
            out.append(polyline(x, segment, color))
            out.append(
                f'<text x="{left + 18}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{color}">{label}</text>'
            )
            start += len(curve)
        return "\n".join(out)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">MCT beta-window closure</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">An effective beta envelope adds critical decay and von-Schweidler departure around the cage plateau.</text>
  {axes(left_a, right_a, "A. Two-sided beta power laws", "log10(t / t_beta)")}
  {plot_scaled_time(left_a, right_a)}
  {axes(left_b, right_b, "B. Cooling separates beta and alpha clocks", "inverse-temperature shift")}
  {plot_cooling(left_b, right_b)}
</svg>
"""
    path.write_text(svg)


def write_sota_benchmark_consistency_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 630
    by_id = {str(row["benchmark_id"]): row for row in rows}
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
    left_a, top, right_a, bottom = 90, 105, 520, 430
    left_b, right_b = 660, 1040
    metrics = [
        ("obs critical", float(mct_row["observed_critical_decay"]), "#718096"),
        ("model critical", float(mct_row["model_predicts_visible_critical_decay"]), "#2b6cb0"),
        ("obs von", float(mct_row["observed_von_schweidler"]), "#718096"),
        ("model von", float(mct_row["model_predicts_visible_von_schweidler"]), "#c05621"),
    ]
    bars = []
    bar_w = 54
    gap = 38
    for idx, (label, value, color) in enumerate(metrics):
        x = left_a + idx * (bar_w + gap)
        h = value * (bottom - top)
        bars.append(f'<rect x="{x}" y="{bottom - h:.2f}" width="{bar_w}" height="{h:.2f}" fill="{color}" />')
        bars.append(
            f'<text x="{x - 8}" y="{bottom + 24}" font-family="Arial, sans-serif" font-size="11">{label}</text>'
        )
    recovery_values = np.array(
        [
            float(recovery_row["finite_exchange_late_ngp"]),
            float(recovery_row["static_gamma_late_ngp"]),
            float(recovery_row["recovery_threshold"]),
        ]
    )
    x_recovery = np.array([0.0, 1.0, 2.0])
    x = scale(x_recovery, left_b, right_b)
    y = scale(np.log10(recovery_values), bottom, top)
    recovery_labels = [
        ("finite exchange", "#2f855a"),
        ("static disorder", "#805ad5"),
        ("recovery threshold", "#555555"),
    ]
    points = []
    for idx, ((label, color), xi, yi, value) in enumerate(zip(recovery_labels, x, y, recovery_values)):
        points.append(f'<circle cx="{xi:.2f}" cy="{yi:.2f}" r="5" fill="{color}" />')
        points.append(
            f'<text x="{left_b + 12}" y="{top + 25 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{color}">{label}: {value:.3g}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">SOTA benchmark consistency</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Benchmark rows convert literature-level conclusions into explicit consistency checks.</text>
  <line x1="{left_a}" y1="{bottom}" x2="{right_a}" y2="{bottom}" stroke="#222" />
  <line x1="{left_a}" y1="{bottom}" x2="{left_a}" y2="{top}" stroke="#222" />
  <text x="{left_a}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">A. Cage and MCT checks</text>
  {"".join(bars)}
  <text x="{left_a}" y="{bottom + 52}" font-family="Arial, sans-serif" font-size="11">cage row consistent = {int(float(cage_row['overall_consistent']))}; f_DW = {float(cage_row['debye_waller_plateau']):.3f}, renewal MSD frac = {float(cage_row['renewal_msd_fraction']):.3f}</text>
  <text x="{left_a}" y="{bottom + 68}" font-family="Arial, sans-serif" font-size="11">MCT row consistent = {int(float(mct_row['overall_consistent']))}</text>
  <text x="{left_a}" y="{bottom + 84}" font-family="Arial, sans-serif" font-size="11">exponent row consistent = {int(float(mct_exponent_row['overall_consistent']))}; lambda_a = {float(mct_exponent_row['lambda_from_a']):.3f}, lambda_b = {float(mct_exponent_row['lambda_from_b']):.3f}</text>
  <text x="{left_a}" y="{bottom + 100}" font-family="Arial, sans-serif" font-size="11">NGP peak row consistent = {int(float(ngp_peak_row['overall_consistent']))}; t_peak growth = {float(ngp_peak_row['peak_time_growth']):.2f}, peak growth = {float(ngp_peak_row['peak_height_growth']):.2f}</text>
  <text x="{left_a}" y="{bottom + 116}" font-family="Arial, sans-serif" font-size="11">joint row consistent = {int(float(joint_row['overall_consistent']))}; SE growth = {float(joint_row['joint_stokes_einstein_growth_over_poisson']):.2f}, mismatch residual = {float(joint_row['rejected_mismatch_abs_log_residual']):.2f}</text>
  <line x1="{left_b}" y1="{bottom}" x2="{right_b}" y2="{bottom}" stroke="#222" />
  <line x1="{left_b}" y1="{bottom}" x2="{left_b}" y2="{top}" stroke="#222" />
  <text x="{left_b}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">B. Gaussian recovery mechanism</text>
  {polyline(x, y, "#222222", width=1.2)}
  {"".join(points)}
  <text x="{left_b}" y="{bottom + 52}" font-family="Arial, sans-serif" font-size="11">recovery row consistent = {int(float(recovery_row['overall_consistent']))}</text>
  <text x="{left_b}" y="{bottom + 68}" font-family="Arial, sans-serif" font-size="11">SE row consistent = {int(float(se_row['overall_consistent']))}; D tau growth = {float(se_row['se_product_growth']):.2f}, xi_SE = {float(se_row['cold_fractional_exponent']):.3f}</text>
  <text x="{left_b}" y="{bottom + 84}" font-family="Arial, sans-serif" font-size="11">chi4 row consistent = {int(float(heterogeneity_row['overall_consistent']))}; xi4 growth = {float(heterogeneity_row['length_growth']):.2f}, chi4 growth = {float(heterogeneity_row['chi4_peak_growth_benchmark']):.1f}</text>
  <text x="{left_b}" y="{bottom + 100}" font-family="Arial, sans-serif" font-size="11">front-law row consistent = {int(float(spatial_front_row['overall_consistent']))}; Df cv = {float(spatial_front_row['facilitation_diffusivity_relative_std']):.2g}, xi4 growth = {float(spatial_front_row['length_growth']):.2f}</text>
  <text x="{left_b}" y="{bottom + 116}" font-family="Arial, sans-serif" font-size="11">TTS row consistent = {int(float(tts_row['overall_consistent']))}; residual = {float(tts_row['cold_shape_residual']):.3f}, C growth = {float(tts_row['alpha_shape_control_growth']):.2f}</text>
  <text x="{left_b}" y="{bottom + 132}" font-family="Arial, sans-serif" font-size="11">KWW row consistent = {int(float(stretched_row['overall_consistent']))}; beta hot/cold = {float(stretched_row['hot_kww_beta']):.2f}/{float(stretched_row['cold_kww_beta']):.2f}, residual = {float(stretched_row['cold_fit_residual']):.3f}</text>
  <text x="{left_b}" y="{bottom + 148}" font-family="Arial, sans-serif" font-size="11">persistence/exchange row consistent = {int(float(persistence_exchange_row['overall_consistent']))}; tau_p/tau_x = {float(persistence_exchange_row['inferred_persistence_exchange_ratio']):.1f}, late residual = {float(persistence_exchange_row['late_ngp_log_residual_benchmark']):.2g}</text>
  <text x="{left_b}" y="{bottom + 164}" font-family="Arial, sans-serif" font-size="11">van Hove row consistent = {int(float(van_hove_row['overall_consistent']))}; peak tail = {float(van_hove_row['peak_tail_ratio']):.2f}, late tail = {float(van_hove_row['late_tail_ratio']):.2f}</text>
  <text x="{left_b}" y="{bottom + 180}" font-family="Arial, sans-serif" font-size="11">fragility row consistent = {int(float(fragility_row['overall_consistent']))}; m growth = {float(fragility_row['fragility_index_growth']):.2f}, AG slowdown = {float(fragility_row['adam_gibbs_slowdown']):.2g}</text>
  <text x="{left_b}" y="{bottom + 196}" font-family="Arial, sans-serif" font-size="11">thermo scope row consistent = {int(float(thermodynamic_scope_row['overall_consistent']))}; dynamic entropy = {int(float(thermodynamic_scope_row['dynamic_model_derives_entropy']))}, AG = {float(thermodynamic_scope_row['thermodynamic_adam_gibbs_slowdown']):.2g}</text>
</svg>
"""
    path.write_text(svg)


def write_sota_claim_alignment_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 650
    left, top = 75, 100
    row_h = 54
    colors_by_alignment = {
        "supported": "#2f855a",
        "partial": "#2b6cb0",
        "scope_boundary": "#805ad5",
        "not_supported": "#c05621",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        alignment = str(row["claim_alignment"])
        color = colors_by_alignment[alignment]
        readiness = str(row["data_readiness"]).replace("_", " ")
        closure = int(float(row["requires_external_closure"]))
        fit_ready = int(float(row["quantitative_fit_ready"]))
        marks.append(
            f'<text x="{left}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">{str(row["source_key"])[:28]}</text>'
        )
        marks.append(
            f'<text x="{left + 185}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">{str(row["phenomenon"]).replace("_", " ")[:42]}</text>'
        )
        marks.append(
            f'<rect x="{left + 500}" y="{y + 1}" width="110" height="22" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 512}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11" fill="#fff">{alignment.replace("_", " ")}</text>'
        )
        marks.append(
            f'<text x="{left + 632}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">support: {str(row["model_support_level"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<text x="{left + 790}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">closure={closure}; fit={fit_ready}</text>'
        )
        marks.append(
            f'<text x="{left + 925}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">block: {str(row["primary_blocker"]).replace("_", " ")[:27]}</text>'
        )
        marks.append(
            f'<text x="{left + 185}" y="{y + 36}" font-family="Arial, sans-serif" font-size="9" fill="#555">{str(row["observed_claim"])[:115]}</text>'
        )
        marks.append(
            f'<text x="{left + 632}" y="{y + 36}" font-family="Arial, sans-serif" font-size="9" fill="#555">data: {readiness}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">SOTA claim alignment audit</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Source-level claims are mapped to diagnostics, data readiness, and scope boundaries to prevent overclaiming.</text>
  <text x="{left}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700">source</text>
  <text x="{left + 185}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700">phenomenon and source claim</text>
  <text x="{left + 500}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700">alignment</text>
  <text x="{left + 632}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700">model/data status</text>
  {"".join(marks)}
  <rect x="75" y="580" width="14" height="14" fill="#2f855a" /><text x="96" y="592" font-family="Arial, sans-serif" font-size="12">derived dynamical support</text>
  <rect x="260" y="580" width="14" height="14" fill="#2b6cb0" /><text x="281" y="592" font-family="Arial, sans-serif" font-size="12">effective closure or proxy</text>
  <rect x="455" y="580" width="14" height="14" fill="#805ad5" /><text x="476" y="592" font-family="Arial, sans-serif" font-size="12">explicit scope boundary</text>
</svg>
"""
    path.write_text(svg)


def write_sota_signed_constraints_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 650
    left, top = 75, 100
    row_h = 54
    colors_by_class = {
        "sota_consistent": "#2f855a",
        "closure_assisted_consistent": "#2b6cb0",
        "scope_boundary_consistent": "#805ad5",
        "missing_signature": "#c05621",
        "overclaimed_boundary": "#b83280",
        "not_supported": "#718096",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        constraint_class = str(row["signed_constraint_class"])
        color = colors_by_class[constraint_class]
        missing = str(row["missing_expected_signatures"]).replace("_", " ")
        forbidden = str(row["forbidden_claims_made"]).replace("_", " ")
        closure = int(float(row["requires_external_closure"]))
        publishable = int(float(row["publishable_alignment"]))
        marks.append(
            f'<text x="{left}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">{str(row["source_key"])[:30]}</text>'
        )
        marks.append(
            f'<text x="{left + 195}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">{str(row["model_scope"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<rect x="{left + 380}" y="{y + 1}" width="150" height="22" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 390}" y="{y + 16}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{constraint_class.replace("_", " ")}</text>'
        )
        marks.append(
            f'<text x="{left + 550}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">closure={closure}; publishable={publishable}</text>'
        )
        marks.append(
            f'<text x="{left + 735}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">missing: {missing[:30]}</text>'
        )
        marks.append(
            f'<text x="{left + 930}" y="{y + 17}" font-family="Arial, sans-serif" font-size="11">forbidden made: {forbidden[:22]}</text>'
        )
        marks.append(
            f'<text x="{left + 195}" y="{y + 36}" font-family="Arial, sans-serif" font-size="9" fill="#555">{str(row["source_observation"])[:118]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">SOTA signed-constraint audit</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Representative literature conclusions are converted into required signatures and forbidden overclaims.</text>
  <text x="{left}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700">source</text>
  <text x="{left + 195}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700">scope and observation</text>
  <text x="{left + 380}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700">constraint class</text>
  <text x="{left + 550}" y="{top - 20}" font-family="Arial, sans-serif" font-size="12" font-weight="700">status</text>
  {"".join(marks)}
  <rect x="75" y="580" width="14" height="14" fill="#2f855a" /><text x="96" y="592" font-family="Arial, sans-serif" font-size="12">direct dynamical constraint satisfied</text>
  <rect x="300" y="580" width="14" height="14" fill="#2b6cb0" /><text x="321" y="592" font-family="Arial, sans-serif" font-size="12">closure-assisted constraint</text>
  <rect x="505" y="580" width="14" height="14" fill="#805ad5" /><text x="526" y="592" font-family="Arial, sans-serif" font-size="12">thermodynamic scope boundary</text>
</svg>
"""
    path.write_text(svg)


def write_real_benchmark_assimilation_gate_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 590
    left, top = 70, 104
    row_h = 56
    colors = {
        "uncertainty_weighted_inversion": "#2f855a",
        "structural_digitization_ready": "#2b6cb0",
        "qualitative_alignment_only": "#d69e2e",
        "scope_boundary_only": "#805ad5",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        stage = str(row["assimilation_stage"])
        color = colors[stage]
        coverage = float(row["required_observable_coverage"])
        structural = int(float(row["structural_inversion_ready"]))
        weighted = int(float(row["uncertainty_weighted_ready"]))
        marks.append(
            f'<text x="{left}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">{str(row["benchmark_id"]).replace("_", " ")[:40]}</text>'
        )
        marks.append(
            f'<text x="{left + 275}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">{str(row["target_protocol"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<rect x="{left + 455}" y="{y}" width="{210 * coverage:.1f}" height="11" fill="#2f855a" opacity="0.82" />'
        )
        marks.append(
            f'<rect x="{left + 455}" y="{y}" width="210" height="11" fill="none" stroke="#333" stroke-width="0.6" />'
        )
        marks.append(
            f'<rect x="{left + 690}" y="{y - 3}" width="168" height="22" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 698}" y="{y + 12}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{stage.replace("_", " ")[:28]}</text>'
        )
        marks.append(
            f'<text x="{left + 880}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">struct={structural}; weighted={weighted}</text>'
        )
        marks.append(
            f'<text x="{left + 275}" y="{y + 34}" font-family="Arial, sans-serif" font-size="9" fill="#555">source: {str(row["source_key"])[:50]}</text>'
        )
        marks.append(
            f'<text x="{left + 690}" y="{y + 34}" font-family="Arial, sans-serif" font-size="9" fill="#555">blocker: {str(row["primary_blocker"]).replace("_", " ")[:44]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Real benchmark assimilation gate</text>
  <text x="70" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Published benchmark claims must pass observable coverage, shared-system, machine-readable, and uncertainty gates before quantitative inversion.</text>
  <text x="{left}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">benchmark</text>
  <text x="{left + 275}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">protocol</text>
  <text x="{left + 455}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">required-observable coverage</text>
  <text x="{left + 690}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">assimilation stage</text>
  {"".join(marks)}
  <rect x="70" y="535" width="14" height="14" fill="#2f855a" /><text x="92" y="547" font-family="Arial, sans-serif" font-size="12">uncertainty-weighted real inversion</text>
  <rect x="310" y="535" width="14" height="14" fill="#2b6cb0" /><text x="332" y="547" font-family="Arial, sans-serif" font-size="12">structural digitization only</text>
  <rect x="520" y="535" width="14" height="14" fill="#d69e2e" /><text x="542" y="547" font-family="Arial, sans-serif" font-size="12">qualitative alignment only</text>
  <rect x="720" y="535" width="14" height="14" fill="#805ad5" /><text x="742" y="547" font-family="Arial, sans-serif" font-size="12">scope boundary</text>
</svg>
"""
    path.write_text(svg)


def write_cross_observable_prediction_ledger_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 560
    left, top = 70, 100
    row_h = 58
    colors = {
        "predictive_diagnostic": "#2f855a",
        "closure_assisted_prediction": "#2b6cb0",
        "underconstrained_fit": "#c05621",
        "failed_prediction": "#c53030",
        "scope_boundary": "#805ad5",
        "not_supported": "#718096",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        klass = str(row["prediction_class"])
        color = colors[klass]
        calibration = int(float(row["calibration_count"]))
        heldout = int(float(row["heldout_prediction_count"]))
        closure = int(float(row["closure_observable_count"]))
        risk = int(float(row["fit_only_overclaim_risk"]))
        marks.append(
            f'<text x="{left}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">{str(row["protocol_id"]).replace("_", " ")[:42]}</text>'
        )
        marks.append(
            f'<rect x="{left + 310}" y="{y - 4}" width="175" height="22" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 318}" y="{y + 11}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{klass.replace("_", " ")[:27]}</text>'
        )
        marks.append(
            f'<text x="{left + 510}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">fit={calibration}; held-out={heldout}; closure={closure}; risk={risk}</text>'
        )
        marks.append(
            f'<text x="{left + 70}" y="{y + 34}" font-family="Arial, sans-serif" font-size="9" fill="#555">fit: {str(row["calibration_observables"]).replace("_", " ")[:64]}</text>'
        )
        marks.append(
            f'<text x="{left + 510}" y="{y + 34}" font-family="Arial, sans-serif" font-size="9" fill="#555">predict: {str(row["heldout_predictions"]).replace("_", " ")[:76]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Cross-observable prediction ledger</text>
  <text x="70" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Each protocol separates calibration inputs from held-out predictions and external closure variables.</text>
  <text x="{left}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">protocol</text>
  <text x="{left + 310}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">prediction class</text>
  <text x="{left + 510}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">ledger counts</text>
  {"".join(marks)}
  <rect x="70" y="510" width="14" height="14" fill="#2f855a" /><text x="92" y="522" font-family="Arial, sans-serif" font-size="12">held-out predictive diagnostic</text>
  <rect x="285" y="510" width="14" height="14" fill="#2b6cb0" /><text x="307" y="522" font-family="Arial, sans-serif" font-size="12">closure-assisted prediction</text>
  <rect x="500" y="510" width="14" height="14" fill="#c05621" /><text x="522" y="522" font-family="Arial, sans-serif" font-size="12">fit-only overclaim risk</text>
  <rect x="690" y="510" width="14" height="14" fill="#805ad5" /><text x="712" y="522" font-family="Arial, sans-serif" font-size="12">scope boundary</text>
</svg>
"""
    path.write_text(svg)


def write_inversion_identifiability_audit_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 600
    left, top = 70, 105
    row_h = 58
    colors = {
        "identifiable_prediction": "#2f855a",
        "conditionally_identifiable": "#2b6cb0",
        "underidentified_fit": "#c05621",
        "degenerate_fit": "#c53030",
        "scope_boundary": "#805ad5",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        klass = str(row["identifiability_class"])
        color = colors[klass]
        rank_margin = int(float(row["rank_margin"]))
        fit_count = int(float(row["fit_observable_count"]))
        parameter_count = int(float(row["inferred_parameter_count"]))
        heldout_count = int(float(row["heldout_prediction_count"]))
        closure_count = int(float(row["closure_count"]))
        risk = int(float(row["overclaim_risk"]))
        marks.append(
            f'<text x="{left}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">{str(row["protocol_id"]).replace("_", " ")[:42]}</text>'
        )
        marks.append(
            f'<rect x="{left + 315}" y="{y - 4}" width="190" height="22" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 323}" y="{y + 11}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{klass.replace("_", " ")[:29]}</text>'
        )
        marks.append(
            f'<text x="{left + 525}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">fit={fit_count}; params={parameter_count}; margin={rank_margin}; held={heldout_count}; closure={closure_count}; risk={risk}</text>'
        )
        marks.append(
            f'<text x="{left + 70}" y="{y + 34}" font-family="Arial, sans-serif" font-size="9" fill="#555">parameters: {str(row["inferred_parameters"]).replace("_", " ")[:62]}</text>'
        )
        marks.append(
            f'<text x="{left + 525}" y="{y + 34}" font-family="Arial, sans-serif" font-size="9" fill="#555">held-out: {str(row["heldout_predictions"]).replace("_", " ")[:72]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Inversion identifiability audit</text>
  <text x="70" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Each inversion protocol is checked for fit-rank margin, held-out predictions, closure dependence, and explicit parameter degeneracy.</text>
  <text x="{left}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">protocol</text>
  <text x="{left + 315}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">identifiability class</text>
  <text x="{left + 525}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">pre-fit audit counts</text>
  {"".join(marks)}
  <rect x="70" y="550" width="14" height="14" fill="#2f855a" /><text x="92" y="562" font-family="Arial, sans-serif" font-size="12">identifiable with held-out predictions</text>
  <rect x="325" y="550" width="14" height="14" fill="#2b6cb0" /><text x="347" y="562" font-family="Arial, sans-serif" font-size="12">closure-assisted</text>
  <rect x="500" y="550" width="14" height="14" fill="#c05621" /><text x="522" y="562" font-family="Arial, sans-serif" font-size="12">underidentified</text>
  <rect x="665" y="550" width="14" height="14" fill="#c53030" /><text x="687" y="562" font-family="Arial, sans-serif" font-size="12">degenerate fit</text>
  <rect x="820" y="550" width="14" height="14" fill="#805ad5" /><text x="842" y="562" font-family="Arial, sans-serif" font-size="12">scope boundary</text>
</svg>
"""
    path.write_text(svg)


def write_frontier_benchmark_horizon_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 540
    left, top = 70, 100
    row_h = 65
    colors = {
        "trajectory_reanalysis_candidate": "#2f855a",
        "transport_heterogeneity_candidate": "#2b6cb0",
        "model_extension_required": "#c05621",
        "closure_horizon": "#805ad5",
        "scope_boundary": "#718096",
        "quantitative_inversion_candidate": "#276749",
        "structural_inversion_candidate": "#319795",
        "qualitative_horizon": "#d69e2e",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        klass = str(row["horizon_class"])
        color = colors[klass]
        coverage = float(row["effective_observable_coverage"])
        score = float(row["frontier_priority_score"])
        marks.append(
            f'<text x="{left}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">{str(row["benchmark_id"]).replace("_", " ")[:42]}</text>'
        )
        marks.append(
            f'<rect x="{left + 315}" y="{y - 4}" width="210" height="22" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 323}" y="{y + 11}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{klass.replace("_", " ")[:32]}</text>'
        )
        marks.append(
            f'<rect x="{left + 555}" y="{y - 2}" width="150" height="12" fill="#e2e8f0" />'
        )
        marks.append(
            f'<rect x="{left + 555}" y="{y - 2}" width="{150 * coverage:.1f}" height="12" fill="#2b6cb0" />'
        )
        marks.append(
            f'<rect x="{left + 730}" y="{y - 2}" width="150" height="12" fill="#e2e8f0" />'
        )
        marks.append(
            f'<rect x="{left + 730}" y="{y - 2}" width="{150 * score:.1f}" height="12" fill="#2f855a" />'
        )
        marks.append(
            f'<text x="{left + 905}" y="{y + 8}" font-family="Arial, sans-serif" font-size="10">blocker: {str(row["primary_blocker"]).replace("_", " ")[:28]}</text>'
        )
        marks.append(
            f'<text x="{left + 70}" y="{y + 34}" font-family="Arial, sans-serif" font-size="9" fill="#555">source: {str(row["source_key"]).replace("_", " ")[:72]}</text>'
        )
        marks.append(
            f'<text x="{left + 555}" y="{y + 34}" font-family="Arial, sans-serif" font-size="9" fill="#555">missing: {str(row["missing_observables"]).replace("_", " ")[:76]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Frontier benchmark horizon</text>
  <text x="70" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Recent SOTA sources are classified as reanalysis targets, transport/heterogeneity candidates, model-extension gaps, closures, or scope boundaries.</text>
  <text x="{left}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">benchmark</text>
  <text x="{left + 315}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">horizon class</text>
  <text x="{left + 555}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">coverage</text>
  <text x="{left + 730}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">priority</text>
  {"".join(marks)}
  <rect x="70" y="490" width="14" height="14" fill="#2f855a" /><text x="92" y="502" font-family="Arial, sans-serif" font-size="12">trajectory reanalysis</text>
  <rect x="245" y="490" width="14" height="14" fill="#2b6cb0" /><text x="267" y="502" font-family="Arial, sans-serif" font-size="12">transport/heterogeneity</text>
  <rect x="455" y="490" width="14" height="14" fill="#c05621" /><text x="477" y="502" font-family="Arial, sans-serif" font-size="12">model extension</text>
  <rect x="620" y="490" width="14" height="14" fill="#805ad5" /><text x="642" y="502" font-family="Arial, sans-serif" font-size="12">closure horizon</text>
  <rect x="780" y="490" width="14" height="14" fill="#718096" /><text x="802" y="502" font-family="Arial, sans-serif" font-size="12">scope boundary</text>
</svg>
"""
    path.write_text(svg)


def write_sota_source_provenance_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 430
    left, top = 75, 105
    row_h = 68
    colors = {
        "trajectory_reanalysis_source": "#2f855a",
        "raw_curve_reanalysis_source": "#2b6cb0",
        "machine_readable_but_not_reanalysis_permitted": "#d69e2e",
        "machine_readable_source_incomplete_metadata": "#c05621",
        "citation_only_source": "#718096",
        "scope_boundary_source": "#805ad5",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        stage = str(row["provenance_stage"])
        color = colors[stage]
        trajectory = int(float(row["can_enter_trajectory_protocol"]))
        raw_curve = int(float(row["can_enter_raw_curve_protocol"]))
        digitize = int(float(row["requires_digitization"]))
        marks.append(
            f'<text x="{left}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">{str(row["source_id"]).replace("_", " ")[:44]}</text>'
        )
        marks.append(
            f'<rect x="{left + 330}" y="{y - 4}" width="235" height="24" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 340}" y="{y + 12}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{stage.replace("_", " ")[:36]}</text>'
        )
        marks.append(
            f'<text x="{left + 590}" y="{y + 15}" font-family="Arial, sans-serif" font-size="11">traj={trajectory}; raw={raw_curve}; digitize={digitize}; blocker={str(row["primary_blocker"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<text x="{left + 70}" y="{y + 38}" font-family="Arial, sans-serif" font-size="9" fill="#555">provenance: {str(row["provenance_items"]).replace("_", " ")[:96]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">SOTA source provenance gate</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">A citation is not promoted to an inversion input unless repository, raw-data, metadata, and reuse conditions are all explicit.</text>
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">source</text>
  <text x="{left + 330}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">provenance stage</text>
  <text x="{left + 590}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">protocol entry</text>
  {"".join(marks)}
  <rect x="75" y="365" width="14" height="14" fill="#2f855a" /><text x="96" y="377" font-family="Arial, sans-serif" font-size="12">trajectory reanalysis source</text>
  <rect x="285" y="365" width="14" height="14" fill="#718096" /><text x="306" y="377" font-family="Arial, sans-serif" font-size="12">citation only</text>
  <rect x="430" y="365" width="14" height="14" fill="#805ad5" /><text x="451" y="377" font-family="Arial, sans-serif" font-size="12">scope boundary</text>
</svg>
"""
    path.write_text(svg)


def write_sota_data_accession_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 450
    left, top = 75, 105
    row_h = 72
    colors = {
        "remote_trajectory_accession_ready": "#2f855a",
        "remote_raw_curve_accession_ready": "#2b6cb0",
        "local_trajectory_cache_ready": "#276749",
        "local_raw_curve_cache_ready": "#2c5282",
        "metadata_incomplete_accession": "#c05621",
        "citation_only_no_accession": "#718096",
        "scope_boundary_accession": "#805ad5",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        stage = str(row["accession_stage"])
        color = colors[stage]
        accession_ready = int(float(row["accession_ready"]))
        local_ready = int(float(row["ready_for_local_reanalysis"]))
        size_gb = float(row["archive_size_gb"])
        marks.append(
            f'<text x="{left}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">{str(row["accession_id"]).replace("_", " ")[:42]}</text>'
        )
        marks.append(
            f'<rect x="{left + 305}" y="{y - 4}" width="230" height="24" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 314}" y="{y + 12}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{stage.replace("_", " ")[:35]}</text>'
        )
        marks.append(
            f'<text x="{left + 560}" y="{y + 15}" font-family="Arial, sans-serif" font-size="11">accession={accession_ready}; local={local_ready}; size={size_gb:.2f} GB; blocker={str(row["primary_blocker"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<text x="{left + 72}" y="{y + 38}" font-family="Arial, sans-serif" font-size="9" fill="#555">doi: {row["doi"]}; archive: {row["archive_name"]}; md5: {str(row["archive_md5"])[:32]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">SOTA data accession manifest</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Remote archives are recorded with DOI, checksum, size, license, and local-cache status before any real-data reanalysis is claimed.</text>
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">accession</text>
  <text x="{left + 305}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">stage</text>
  <text x="{left + 560}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">readiness</text>
  {"".join(marks)}
  <rect x="75" y="385" width="14" height="14" fill="#2f855a" /><text x="96" y="397" font-family="Arial, sans-serif" font-size="12">remote trajectory archive ready</text>
  <rect x="310" y="385" width="14" height="14" fill="#718096" /><text x="331" y="397" font-family="Arial, sans-serif" font-size="12">citation only</text>
  <rect x="455" y="385" width="14" height="14" fill="#805ad5" /><text x="476" y="397" font-family="Arial, sans-serif" font-size="12">scope boundary</text>
</svg>
"""
    path.write_text(svg)


def write_sota_readme_schema_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1160, 380
    left, top = 75, 105
    row_h = 72
    colors = {
        "remote_readme_schema_ready": "#2f855a",
        "local_archive_schema_ready": "#276749",
        "metadata_incomplete_schema": "#c05621",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        stage = str(row["schema_stage"])
        color = colors[stage]
        schema_ready = int(float(row["schema_ready"]))
        local_ready = int(float(row["ready_for_local_adapter"]))
        marks.append(
            f'<text x="{left}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11">{str(row["schema_id"]).replace("_", " ")[:42]}</text>'
        )
        marks.append(
            f'<rect x="{left + 310}" y="{y - 4}" width="220" height="24" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 320}" y="{y + 12}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{stage.replace("_", " ")[:33]}</text>'
        )
        marks.append(
            f'<text x="{left + 555}" y="{y + 15}" font-family="Arial, sans-serif" font-size="11">schema={schema_ready}; local adapter={local_ready}; blocker={str(row["primary_blocker"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<text x="{left + 72}" y="{y + 38}" font-family="Arial, sans-serif" font-size="9" fill="#555">systems: {row["systems"]}; folders: {row["folder_tokens"]}; citations={int(float(row["citation_count"]))}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">SOTA README schema gate</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">README-level evidence checks systems, folder tokens, license, and citation guidance before local archive adapters are claimed.</text>
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">schema</text>
  <text x="{left + 310}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">stage</text>
  <text x="{left + 555}" y="{top - 24}" font-family="Arial, sans-serif" font-size="12" font-weight="700">readiness</text>
  {"".join(marks)}
  <rect x="75" y="315" width="14" height="14" fill="#2f855a" /><text x="96" y="327" font-family="Arial, sans-serif" font-size="12">remote README schema ready</text>
  <rect x="315" y="315" width="14" height="14" fill="#c05621" /><text x="336" y="327" font-family="Arial, sans-serif" font-size="12">metadata incomplete</text>
</svg>
"""
    path.write_text(svg)


def write_trajectory_adapter_contract_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 430
    left, top = 70, 112
    row_h = 76
    colors = {
        "remote_adapter_contract_only": "#805ad5",
        "metadata_incomplete_adapter": "#c05621",
        "local_trajectory_adapter_ready": "#2f855a",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        stage = str(row["adapter_stage"])
        color = colors[stage]
        missing = str(row["missing_local_fields"])
        missing_label = missing.replace("_", " ")[:50]
        marks.append(
            f'<text x="{left}" y="{y + 14}" font-family="Arial, sans-serif" font-size="11">{str(row["contract_id"]).replace("_", " ")[:42]}</text>'
        )
        marks.append(
            f'<rect x="{left + 300}" y="{y - 5}" width="232" height="24" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 310}" y="{y + 11}" font-family="Arial, sans-serif" font-size="10" fill="#fff">{stage.replace("_", " ")[:35]}</text>'
        )
        marks.append(
            f'<text x="{left + 555}" y="{y + 13}" font-family="Arial, sans-serif" font-size="11">system={row["system_id"]}; ready={int(float(row["adapter_ready"]))}; inspected={int(float(row["local_archive_inspected"]))}; blocker={str(row["primary_blocker"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<text x="{left + 70}" y="{y + 38}" font-family="Arial, sans-serif" font-size="9" fill="#555">fields {int(float(row["available_required_field_count"]))}/{int(float(row["required_field_count"]))}; missing: {missing_label}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="70" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Trajectory adapter contract</text>
  <text x="70" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Remote archive metadata is separated from local coordinate, time-grid, identity, box, state-point, species, and units fields needed by trajectory diagnostics.</text>
  <text x="{left}" y="{top - 25}" font-family="Arial, sans-serif" font-size="12" font-weight="700">contract</text>
  <text x="{left + 300}" y="{top - 25}" font-family="Arial, sans-serif" font-size="12" font-weight="700">adapter stage</text>
  <text x="{left + 555}" y="{top - 25}" font-family="Arial, sans-serif" font-size="12" font-weight="700">gate result</text>
  {"".join(marks)}
  <rect x="70" y="365" width="14" height="14" fill="#805ad5" /><text x="91" y="377" font-family="Arial, sans-serif" font-size="12">remote contract only</text>
  <rect x="255" y="365" width="14" height="14" fill="#2f855a" /><text x="276" y="377" font-family="Arial, sans-serif" font-size="12">local adapter ready</text>
  <rect x="430" y="365" width="14" height="14" fill="#c05621" /><text x="451" y="377" font-family="Arial, sans-serif" font-size="12">local metadata incomplete</text>
</svg>
"""
    path.write_text(svg)


def write_literature_inversion_readiness_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 520
    left, top, right, bottom = 150, 95, 1030, 365
    row_h = (bottom - top) / len(rows)
    bars = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        coverage = float(row["observable_coverage_fraction"])
        qualitative = int(float(row["qualitative_comparison_ready"]))
        quantitative = int(float(row["quantitative_inversion_ready"]))
        uncertainty = int(float(row["uncertainty_weighted_ready"]))
        bar_w = coverage * 360.0
        color = "#2b6cb0" if quantitative else "#d69e2e"
        label = str(row["benchmark_id"]).replace("_", " ")
        bars.append(
            f'<text x="35" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="11">{label}</text>'
        )
        bars.append(
            f'<rect x="{left}" y="{y + 5:.2f}" width="360" height="14" fill="#edf2f7" stroke="#cbd5e0" />'
        )
        bars.append(f'<rect x="{left}" y="{y + 5:.2f}" width="{bar_w:.2f}" height="14" fill="{color}" />')
        bars.append(
            f'<text x="{left + 375}" y="{y + 17:.2f}" font-family="Arial, sans-serif" font-size="11">coverage={coverage:.2f}; qualitative={qualitative}; quantitative={quantitative}; uncertainty={uncertainty}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Literature inversion readiness</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Coverage checks distinguish qualitative benchmark support from quantitative, uncertainty-weighted inversion readiness.</text>
  {"".join(bars)}
  <text x="150" y="430" font-family="Arial, sans-serif" font-size="12" fill="#444">Orange bars: literature figures support comparison but still need digitization, machine-readable data, or uncertainty estimates.</text>
  <text x="150" y="450" font-family="Arial, sans-serif" font-size="12" fill="#444">Blue bars would indicate a benchmark row ready for direct quantitative inversion.</text>
</svg>
"""
    path.write_text(svg)


def write_barrier_requirements_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left_a, top, right_a, bottom = 80, 90, 520, 430
    left_b, right_b = 670, 1040
    amplification_rows = [row for row in rows if row["record_type"] == "amplification"]
    requirement = next(row for row in rows if row["record_type"] == "requirements")
    gaps = sorted({float(row["delay_barrier_gap"]) for row in amplification_rows})
    exchange_sums = sorted({float(row["exchange_barrier_sum"]) for row in amplification_rows})
    max_growth = max(float(row["combined_slowing_growth"]) for row in amplification_rows)
    cell_w = (right_a - left_a) / len(exchange_sums)
    cell_h = (bottom - top) / len(gaps)
    cells = []
    for i, gap in enumerate(gaps):
        y = bottom - (i + 1) * cell_h
        cells.append(
            f'<text x="24" y="{y + cell_h / 2 + 4:.2f}" font-family="Arial, sans-serif" font-size="12">gap={gap:g}</text>'
        )
        for j, exchange_sum in enumerate(exchange_sums):
            row = next(
                item
                for item in amplification_rows
                if float(item["delay_barrier_gap"]) == gap and float(item["exchange_barrier_sum"]) == exchange_sum
            )
            growth = float(row["combined_slowing_growth"])
            shade = int(235 - 120 * math.log(growth) / math.log(max_growth)) if max_growth > 1.0 else 235
            x = left_a + j * cell_w
            cells.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w - 3:.2f}" height="{cell_h - 3:.2f}" fill="rgb({shade},210,150)" stroke="#fff" />'
            )
            cells.append(
                f'<text x="{x + 10:.2f}" y="{y + cell_h / 2 + 4:.2f}" font-family="Arial, sans-serif" font-size="11">{growth:.1f}x</text>'
            )
    labels = []
    for j, exchange_sum in enumerate(exchange_sums):
        labels.append(
            f'<text x="{left_a + j * cell_w + 12:.2f}" y="{bottom + 28}" font-family="Arial, sans-serif" font-size="12">{exchange_sum:g}</text>'
        )

    target_values = [
        ("lambda tau_d", float(requirement["target_lambda_tau_delay_growth"]), float(requirement["required_delay_barrier_gap"]), "#2b6cb0"),
        ("heterogeneity c", float(requirement["target_heterogeneity_ratio_growth"]), float(requirement["required_exchange_barrier_sum"]), "#c05621"),
        ("combined", float(requirement["target_combined_growth"]), float(requirement["required_combined_barrier"]), "#805ad5"),
    ]
    max_required = max(value[2] for value in target_values)
    bars = []
    bar_w = 70
    for idx, (label, target, required, color) in enumerate(target_values):
        x = left_b + idx * 105
        h = 260.0 * required / max_required if max_required > 0.0 else 0.0
        bars.append(
            f'<rect x="{x}" y="{bottom - h:.2f}" width="{bar_w}" height="{h:.2f}" fill="{color}" opacity="0.88" />'
        )
        bars.append(
            f'<text x="{x}" y="{bottom + 22}" font-family="Arial, sans-serif" font-size="11">{label}</text>'
        )
        bars.append(
            f'<text x="{x}" y="{bottom - h - 8:.2f}" font-family="Arial, sans-serif" font-size="11" fill="{color}">E={required:.2f}</text>'
        )
        bars.append(
            f'<text x="{x}" y="{bottom + 39}" font-family="Arial, sans-serif" font-size="10" fill="#555">target {target:.1f}x</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="80" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Closed barrier requirements</text>
  <text x="80" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Amplification over the cooling interval is exp(E Delta(1/T)); target growth inverts to E=log(target)/Delta(1/T).</text>
  <line x1="{left_a}" y1="{bottom}" x2="{right_a}" y2="{bottom}" stroke="#222" />
  <line x1="{left_a}" y1="{bottom}" x2="{left_a}" y2="{top}" stroke="#222" />
  <text x="{left_a}" y="{top - 22}" font-family="Arial, sans-serif" font-size="17" font-weight="700">A. Combined slowing amplification</text>
  {"".join(cells)}
  {"".join(labels)}
  <text x="{left_a + 120}" y="{bottom + 55}" font-family="Arial, sans-serif" font-size="13">exchange barrier sum</text>
  <text x="18" y="280" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 18 280)">delay barrier gap</text>
  <line x1="{left_b}" y1="{bottom}" x2="{right_b}" y2="{bottom}" stroke="#222" />
  <line x1="{left_b}" y1="{bottom}" x2="{left_b}" y2="{top}" stroke="#222" />
  <text x="{left_b}" y="{top - 22}" font-family="Arial, sans-serif" font-size="17" font-weight="700">B. Minimum barriers for target growth</text>
  {"".join(bars)}
</svg>
"""
    path.write_text(svg)


def write_persistence_exchange_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left_a, top, right_a, bottom = 75, 90, 520, 430
    left_b, right_b = 660, 1040
    summary_rows = [row for row in rows if row["record_type"] == "summary"]
    curve_rows = [row for row in rows if row["record_type"] == "curve"]

    ratios = np.array([float(row["persistence_exchange_ratio"]) for row in summary_rows])
    se_product = np.array([float(row["stokes_einstein_product"]) for row in summary_rows])
    late_ngp = np.array([float(row["late_ngp"]) for row in summary_rows])

    def axes(left: int, right: int, title: str, xlabel: str) -> str:
        return f"""
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
  <text x="{(left + right) / 2 - 80}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">{xlabel}</text>
"""

    x_a = scale(ratios, left_a, right_a)
    y_a = scale(se_product / se_product[0], bottom, top)
    y_late = scale(late_ngp / late_ngp[0], bottom, top)

    labels = sorted({float(row["persistence_exchange_ratio"]) for row in curve_rows})
    curve_lines = []
    colors = ["#2b6cb0", "#c05621"]
    all_curve_values = np.concatenate(
        [
            np.array([float(row["ngp"]) for row in curve_rows if float(row["persistence_exchange_ratio"]) == label])
            for label in labels
        ]
    )
    for idx, label in enumerate(labels):
        label_rows = [row for row in curve_rows if float(row["persistence_exchange_ratio"]) == label]
        time = np.array([float(row["time"]) for row in label_rows])
        alpha = np.array([float(row["ngp"]) for row in label_rows])
        x = scale(np.log10(time), left_b, right_b)
        y = scale(alpha, bottom, top)
        curve_lines.append(polyline(x, y, colors[idx % len(colors)]))
        curve_lines.append(
            f'<text x="{left_b + 20}" y="{top + 28 + idx * 18}" font-family="Arial, sans-serif" font-size="12" fill="{colors[idx % len(colors)]}">tau_p/tau_x={label:g}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Persistence/exchange renewal diagnostic</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">First cage persistence slows alpha relaxation while the exchange clock fixes long-time diffusion.</text>
  {axes(left_a, right_a, "A. SE decoupling at fixed diffusion", "persistence / exchange mean")}
  {polyline(x_a, y_a, "#805ad5")}
  {polyline(x_a, y_late, "#a0aec0")}
  <text x="{left_a + 20}" y="{top + 28}" font-family="Arial, sans-serif" font-size="12" fill="#805ad5">D tau_alpha / Poisson limit</text>
  <text x="{left_a + 20}" y="{top + 46}" font-family="Arial, sans-serif" font-size="12" fill="#a0aec0">late NGP / Poisson limit</text>
  {axes(left_b, right_b, "B. NGP recovery despite delayed persistence", "log10 time")}
  {"".join(curve_lines)}
</svg>
"""
    path.write_text(svg)


def write_translation_rotation_protocol_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 500
    left, top, bottom = 90, 105, 380
    panel_w = 330
    scenarios = [str(row["scenario"]).replace("_", " ") for row in rows]
    x_positions = np.linspace(left + 95, left + panel_w - 70, len(rows))
    ratio_values = np.array([float(row["translation_rotation_ratio"]) for row in rows])
    dse_values = np.array([float(row["rotational_dse_product"]) for row in rows])
    residual_values = np.array([abs(float(row["rotational_tau_log_residual"])) for row in rows])

    def scale_with_range(values: np.ndarray, data_max: float) -> np.ndarray:
        if data_max <= 0.0:
            return np.full_like(values, bottom)
        return bottom + values / data_max * (top - bottom)

    ratio_y = scale_with_range(ratio_values, max(1.0, float(np.max(ratio_values))) * 1.15)
    dse_y = scale_with_range(dse_values / dse_values[0], max(1.0, float(np.max(dse_values / dse_values[0]))) * 1.15)
    residual_y = scale_with_range(residual_values, max(0.01, float(np.max(residual_values))) * 1.2)

    bars = []
    for idx, row in enumerate(rows):
        detected = float(row["translation_rotation_decoupling_detected"]) > 0.5
        color = "#c05621" if detected else "#2f855a"
        x0 = x_positions[idx]
        bars.append(f'<rect x="{x0 - 22:.1f}" y="{ratio_y[idx]:.1f}" width="28" height="{bottom - ratio_y[idx]:.1f}" fill="{color}" />')
        bars.append(f'<rect x="{x0 + 14:.1f}" y="{dse_y[idx]:.1f}" width="28" height="{bottom - dse_y[idx]:.1f}" fill="#805ad5" />')
        bars.append(f'<text x="{x0 - 55:.1f}" y="{bottom + 24}" font-family="Arial, sans-serif" font-size="11">{scenarios[idx]}</text>')
        bars.append(
            f'<text x="{x0 - 55:.1f}" y="{bottom + 42}" font-family="Arial, sans-serif" font-size="9" fill="#555">tau_r/tau_a={ratio_values[idx]:.2f}; pass={int(detected)}</text>'
        )
        bars.append(
            f'<circle cx="{left + panel_w + 145 + idx * 95:.1f}" cy="{residual_y[idx]:.1f}" r="5" fill="#2b6cb0" />'
        )
        bars.append(
            f'<text x="{left + panel_w + 112 + idx * 95:.1f}" y="{bottom + 24}" font-family="Arial, sans-serif" font-size="11">{scenarios[idx][:18]}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Translation-rotation renewal diagnostic</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">A rotational renewal clock extends persistence/exchange diagnostics to Debye-Stokes-Einstein and translation-rotation decoupling tests.</text>
  <line x1="{left}" y1="{bottom}" x2="{left + panel_w}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 22}" font-family="Arial, sans-serif" font-size="17" font-weight="700">A. Clock decoupling</text>
  <line x1="{left + panel_w + 80}" y1="{bottom}" x2="{left + panel_w + 345}" y2="{bottom}" stroke="#222" />
  <line x1="{left + panel_w + 80}" y1="{bottom}" x2="{left + panel_w + 80}" y2="{top}" stroke="#222" />
  <text x="{left + panel_w + 80}" y="{top - 22}" font-family="Arial, sans-serif" font-size="17" font-weight="700">B. Inversion residual</text>
  {"".join(bars)}
  <rect x="{left + 18}" y="{top + 20}" width="14" height="14" fill="#c05621" /><text x="{left + 40}" y="{top + 32}" font-family="Arial, sans-serif" font-size="12">tau_rot / tau_alpha</text>
  <rect x="{left + 165}" y="{top + 20}" width="14" height="14" fill="#805ad5" /><text x="{left + 187}" y="{top + 32}" font-family="Arial, sans-serif" font-size="12">D tau_rot / coupled</text>
  <text x="{left + panel_w + 100}" y="{top + 32}" font-family="Arial, sans-serif" font-size="12" fill="#2b6cb0">|log tau_rot residual| after inversion</text>
</svg>
"""
    path.write_text(svg)


def write_persistence_exchange_protocol_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left_a, top, right_a, bottom = 75, 95, 520, 430
    left_b, right_b = 660, 1040
    labels = [str(row["scenario"]) for row in rows]
    x_positions_a = np.linspace(left_a + 70, right_a - 70, len(rows))
    x_positions_b = np.linspace(left_b + 70, right_b - 70, len(rows))
    valid_rows = [row for row in rows if float(row["valid_alpha_transport"]) > 0.5]
    ratio_values = np.array([float(row["inferred_persistence_exchange_ratio"]) for row in valid_rows])
    true_ratio = float(rows[0]["true_persistence_exchange_ratio"])
    ratio_scale_values = np.concatenate([ratio_values, np.array([true_ratio])])
    y_ratio = scale(ratio_scale_values, bottom, top)
    ratio_y_by_value = dict(zip(ratio_scale_values, y_ratio))

    residual_values = np.array(
        [
            abs(float(row["late_ngp_log_residual"])) if float(row["valid_alpha_transport"]) > 0.5 else np.nan
            for row in rows
        ]
    )
    finite_residual = residual_values[np.isfinite(residual_values)]
    residual_scale_values = np.concatenate([finite_residual, np.array([0.1])])
    y_residual = scale(residual_scale_values, bottom, top)
    residual_y_by_value = dict(zip(residual_scale_values, y_residual))

    def axes(left: int, right: int, title: str) -> str:
        return f"""
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{left}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">{title}</text>
"""

    ratio_points = []
    residual_points = []
    for idx, row in enumerate(rows):
        label = labels[idx].replace("_", " ")
        valid = float(row["valid_alpha_transport"]) > 0.5
        color = "#2f855a" if float(row["passes_late_ngp"]) > 0.5 else "#c05621"
        if valid:
            ratio_y = ratio_y_by_value[float(row["inferred_persistence_exchange_ratio"])]
            ratio_points.append(f'<circle cx="{x_positions_a[idx]:.1f}" cy="{ratio_y:.1f}" r="7" fill="{color}" />')
            residual_y = residual_y_by_value[abs(float(row["late_ngp_log_residual"]))]
            residual_points.append(f'<circle cx="{x_positions_b[idx]:.1f}" cy="{residual_y:.1f}" r="7" fill="{color}" />')
        else:
            ratio_points.append(f'<text x="{x_positions_a[idx] - 10:.1f}" y="{bottom - 70}" font-family="Arial, sans-serif" font-size="20" fill="#c05621">x</text>')
            residual_points.append(f'<text x="{x_positions_b[idx] - 10:.1f}" y="{bottom - 70}" font-family="Arial, sans-serif" font-size="20" fill="#c05621">x</text>')
        ratio_points.append(f'<text x="{x_positions_a[idx] - 42:.1f}" y="{bottom + 34}" font-family="Arial, sans-serif" font-size="11">{label}</text>')
        residual_points.append(f'<text x="{x_positions_b[idx] - 42:.1f}" y="{bottom + 34}" font-family="Arial, sans-serif" font-size="11">{label}</text>')

    true_y = ratio_y_by_value[true_ratio]
    threshold_y = residual_y_by_value[0.1]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Persistence/exchange inversion protocol</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">D and alpha time infer hidden clocks; late NGP is a held-out falsification observable.</text>
  {axes(left_a, right_a, "A. Inferred persistence/exchange ratio")}
  <line x1="{left_a}" y1="{true_y:.1f}" x2="{right_a}" y2="{true_y:.1f}" stroke="#718096" stroke-dasharray="5 4" />
  <text x="{left_a + 14}" y="{true_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#718096">true ratio</text>
  {"".join(ratio_points)}
  {axes(left_b, right_b, "B. Held-out late NGP residual")}
  <line x1="{left_b}" y1="{threshold_y:.1f}" x2="{right_b}" y2="{threshold_y:.1f}" stroke="#718096" stroke-dasharray="5 4" />
  <text x="{left_b + 14}" y="{threshold_y - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#718096">|log residual|=0.1</text>
  {"".join(residual_points)}
</svg>
"""
    path.write_text(svg)


def write_persistence_exchange_joint_protocol_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 580
    left_a, top, right_a, bottom = 75, 98, 520, 435
    left_b, right_b = 660, 1040
    summary_rows = [row for row in rows if row["record_type"] == "summary"]
    multik_rows = [row for row in rows if row["record_type"] == "multik_alpha"]
    scenarios = [str(row["scenario"]) for row in summary_rows]

    x_a = np.linspace(left_a + 90, right_a - 90, len(summary_rows))
    checks = [
        ("multi-k alpha", "multik_tau_alpha_consistent", "#2b6cb0"),
        ("late NGP", "late_ngp_consistent", "#2f855a"),
        ("chi4 proxy", "chi4_proxy_growth_consistent", "#805ad5"),
    ]
    marker_rows = []
    for idx, row in enumerate(summary_rows):
        for check_idx, (label, key, color) in enumerate(checks):
            cy = top + 70 + check_idx * 78
            value = float(row[key])
            fill = color if value > 0.5 else "#c05621"
            marker_rows.append(f'<circle cx="{x_a[idx]:.1f}" cy="{cy:.1f}" r="12" fill="{fill}" />')
            marker_rows.append(
                f'<text x="{x_a[idx] - 34:.1f}" y="{cy + 30:.1f}" font-family="Arial, sans-serif" font-size="11">{label}</text>'
            )
        marker_rows.append(
            f'<text x="{x_a[idx] - 55:.1f}" y="{bottom + 35}" font-family="Arial, sans-serif" font-size="12">{scenarios[idx].replace("_", " ")}</text>'
        )

    finite_residuals = np.array([abs(float(row["tau_alpha_log_residual"])) for row in multik_rows])
    residual_values = np.concatenate([finite_residuals, np.array([0.02])])
    residual_min = 0.0
    residual_max = max(float(np.max(residual_values)), 0.03)

    def y_residual(value: float) -> float:
        return bottom + (value - residual_min) * (top - bottom) / (residual_max - residual_min)

    x_positions_b = np.linspace(left_b + 95, right_b - 95, len(summary_rows))
    k_values = sorted({float(row["wave_number"]) for row in multik_rows})
    offsets = np.linspace(-28, 28, len(k_values))
    residual_points = []
    for idx, scenario in enumerate(scenarios):
        for k_idx, wave_number in enumerate(k_values):
            row = next(
                row
                for row in multik_rows
                if str(row["scenario"]) == scenario and math.isclose(float(row["wave_number"]), wave_number)
            )
            residual = abs(float(row["tau_alpha_log_residual"]))
            color = "#2b6cb0" if residual <= 0.02 else "#c05621"
            residual_points.append(
                f'<circle cx="{x_positions_b[idx] + offsets[k_idx]:.1f}" cy="{y_residual(residual):.1f}" r="7" fill="{color}" />'
            )
            residual_points.append(
                f'<text x="{x_positions_b[idx] + offsets[k_idx] - 12:.1f}" y="{bottom + 50 + k_idx * 16}" font-family="Arial, sans-serif" font-size="10">k={wave_number:g}</text>'
            )
        residual_points.append(
            f'<text x="{x_positions_b[idx] - 55:.1f}" y="{bottom + 35}" font-family="Arial, sans-serif" font-size="12">{scenario.replace("_", " ")}</text>'
        )

    consistent = summary_rows[0]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Joint persistence/exchange inversion protocol</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">One anchor alpha time plus D infer clocks; multi-k alpha, late NGP, and chi4 proxy are held out.</text>
  <line x1="{left_a}" y1="{bottom}" x2="{right_a}" y2="{bottom}" stroke="#222" />
  <line x1="{left_a}" y1="{bottom}" x2="{left_a}" y2="{top}" stroke="#222" />
  <text x="{left_a}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">A. Joint protocol pass/fail</text>
  {"".join(marker_rows)}
  <text x="{left_a + 12}" y="{bottom - 34}" font-family="Arial, sans-serif" font-size="12">consistent case: tau_p/tau_x={float(consistent['inferred_persistence_exchange_ratio']):.1f}, SE growth={float(consistent['stokes_einstein_growth_over_poisson']):.2f}, chi4 growth={float(consistent['chi4_peak_growth_over_poisson']):.2f}</text>
  <line x1="{left_b}" y1="{bottom}" x2="{right_b}" y2="{bottom}" stroke="#222" />
  <line x1="{left_b}" y1="{bottom}" x2="{left_b}" y2="{top}" stroke="#222" />
  <text x="{left_b}" y="{top - 24}" font-family="Arial, sans-serif" font-size="17" font-weight="700">B. Held-out multi-k alpha residual</text>
  <line x1="{left_b}" y1="{y_residual(0.02):.1f}" x2="{right_b}" y2="{y_residual(0.02):.1f}" stroke="#718096" stroke-dasharray="5 4" />
  <text x="{left_b + 12}" y="{y_residual(0.02) - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#718096">|log residual|=0.02</text>
  {"".join(residual_points)}
</svg>
"""
    path.write_text(svg)


def write_persistence_exchange_uncertainty_protocol_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left, top, right, bottom = 90, 95, 1035, 425
    scenarios = [str(row["scenario"]) for row in rows]
    z_threshold = float(rows[0]["z_threshold"])
    z_values = np.array(
        [
            [float(row["max_multik_tau_alpha_z"]), float(row["late_ngp_z"]), float(row["chi4_peak_z"])]
            for row in rows
        ]
    )
    z_max = max(float(np.max(z_values)), z_threshold) * 1.12

    def y(value: float) -> float:
        return bottom + value * (top - bottom) / z_max

    x_groups = np.linspace(left + 170, right - 170, len(rows))
    offsets = [-34, 0, 34]
    colors = ["#2b6cb0", "#2f855a", "#c05621"]
    labels = ["multi-k alpha", "late NGP", "chi4 peak"]
    marks = []
    for idx, row in enumerate(rows):
        for jdx, label in enumerate(labels):
            value = z_values[idx, jdx]
            color = colors[jdx]
            marks.append(
                f'<rect x="{x_groups[idx] + offsets[jdx] - 12:.1f}" y="{y(value):.1f}" width="24" height="{bottom - y(value):.1f}" fill="{color}" />'
            )
            marks.append(
                f'<text x="{x_groups[idx] + offsets[jdx] - 14:.1f}" y="{bottom + 50 + jdx * 16}" font-family="Arial, sans-serif" font-size="10" fill="{color}">{label}</text>'
            )
        marks.append(
            f'<text x="{x_groups[idx] - 60:.1f}" y="{bottom + 34}" font-family="Arial, sans-serif" font-size="12">{scenarios[idx].replace("_", " ")}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="90" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Uncertainty-weighted data protocol</text>
  <text x="90" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Observed alpha times, late NGP, and chi4 peak are scored by log-residual z scores.</text>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="38" y="285" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 38 285)">absolute z score</text>
  <line x1="{left}" y1="{y(z_threshold):.1f}" x2="{right}" y2="{y(z_threshold):.1f}" stroke="#718096" stroke-dasharray="5 4" />
  <text x="{right - 82}" y="{y(z_threshold) - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#718096">z={z_threshold:g}</text>
  {"".join(marks)}
</svg>
"""
    path.write_text(svg)


def write_observable_falsification_matrix_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1180, 660
    left, top = 270, 105
    cell_w, cell_h = 135, 56
    benchmarks = list(dict.fromkeys(str(row["benchmark_id"]) for row in rows))
    diagnostics = list(dict.fromkeys(str(row["diagnostic_id"]) for row in rows))
    row_by_key = {
        (str(row["benchmark_id"]), str(row["diagnostic_id"])): row
        for row in rows
    }
    cells = []
    for y_idx, benchmark_id in enumerate(benchmarks):
        y = top + y_idx * cell_h
        cells.append(
            f'<text x="34" y="{y + 33}" font-family="Arial, sans-serif" font-size="11">{benchmark_id.replace("_", " ")[:36]}</text>'
        )
        for x_idx, diagnostic_id in enumerate(diagnostics):
            x = left + x_idx * cell_w
            row = row_by_key[(benchmark_id, diagnostic_id)]
            structural = int(float(row["structural_falsification_ready"]))
            quantitative = int(float(row["quantitative_falsification_ready"]))
            coverage = float(row["observable_coverage_fraction"])
            if quantitative:
                fill = "#2f855a"
            elif structural:
                fill = "#2b6cb0"
            elif coverage >= 0.5:
                fill = "#d69e2e"
            else:
                fill = "#c05621"
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" fill="{fill}" opacity="0.88" />'
            )
            cells.append(
                f'<text x="{x + 11}" y="{y + 22}" font-family="Arial, sans-serif" font-size="12" fill="#fff">cov {coverage:.2f}</text>'
            )
            cells.append(
                f'<text x="{x + 11}" y="{y + 39}" font-family="Arial, sans-serif" font-size="10" fill="#fff">block {str(row["primary_blocker"])[:13]}</text>'
            )
    headers = []
    for idx, diagnostic_id in enumerate(diagnostics):
        x = left + idx * cell_w + 4
        headers.append(
            f'<text x="{x}" y="88" font-family="Arial, sans-serif" font-size="11">{diagnostic_id.replace("_", " ")[:18]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Observable falsification matrix</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Rows show which literature benchmarks contain the observables needed to falsify each diagnostic protocol.</text>
  {"".join(headers)}
  {"".join(cells)}
  <rect x="75" y="585" width="14" height="14" fill="#2f855a" /><text x="96" y="597" font-family="Arial, sans-serif" font-size="12">quantitative ready</text>
  <rect x="235" y="585" width="14" height="14" fill="#2b6cb0" /><text x="256" y="597" font-family="Arial, sans-serif" font-size="12">all observables, no machine-readable uncertainty</text>
  <rect x="548" y="585" width="14" height="14" fill="#d69e2e" /><text x="569" y="597" font-family="Arial, sans-serif" font-size="12">partial observable coverage</text>
  <rect x="760" y="585" width="14" height="14" fill="#c05621" /><text x="781" y="597" font-family="Arial, sans-serif" font-size="12">underconstrained</text>
</svg>
"""
    path.write_text(svg)


def write_benchmark_fusion_readiness_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 480
    left, top, right, bottom = 95, 100, 1040, 330
    row_h = (bottom - top) / len(rows)
    bars = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        coverage = float(row["observable_coverage_fraction"])
        structural = int(float(row["structural_fusion_ready"]))
        quantitative = int(float(row["quantitative_fusion_ready"]))
        color = "#2f855a" if quantitative else "#2b6cb0" if structural else "#c05621"
        label = str(row["fusion_id"]).replace("_", " ")
        bars.append(
            f'<text x="{left}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">{label[:54]}</text>'
        )
        bars.append(
            f'<rect x="{left + 360}" y="{y + 5:.2f}" width="270" height="15" fill="#edf2f7" stroke="#cbd5e0" />'
        )
        bars.append(
            f'<rect x="{left + 360}" y="{y + 5:.2f}" width="{270 * coverage:.2f}" height="15" fill="{color}" />'
        )
        bars.append(
            f'<text x="{left + 645}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">cov={coverage:.2f}</text>'
        )
        bars.append(
            f'<text x="{left + 720}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">sys/grid/ens={int(float(row["shared_system_consistent"]))}/{int(float(row["shared_temperature_grid_consistent"]))}/{int(float(row["shared_ensemble_consistent"]))}</text>'
        )
        bars.append(
            f'<text x="{left + 870}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">block: {str(row["primary_blocker"])[:24]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Cross-benchmark fusion readiness</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Multi-paper validation must preserve system, temperature-grid, and ensemble identity before shared diagnostics are claimed.</text>
  {"".join(bars)}
  <rect x="95" y="390" width="14" height="14" fill="#2f855a" /><text x="116" y="402" font-family="Arial, sans-serif" font-size="12">quantitative fusion ready</text>
  <rect x="280" y="390" width="14" height="14" fill="#2b6cb0" /><text x="301" y="402" font-family="Arial, sans-serif" font-size="12">structural fusion only</text>
  <rect x="455" y="390" width="14" height="14" fill="#c05621" /><text x="476" y="402" font-family="Arial, sans-serif" font-size="12">invalid splice or missing observables</text>
</svg>
"""
    path.write_text(svg)


def write_raw_curve_ingestion_contract_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 500
    left, top, right, bottom = 95, 105, 1030, 330
    row_h = (bottom - top) / len(rows)
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        structural = int(float(row["structural_ingestion_ready"]))
        uncertainty = int(float(row["uncertainty_ingestion_ready"]))
        color = "#2f855a" if uncertainty else "#2b6cb0" if structural else "#c05621"
        marks.append(
            f'<text x="{left}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">{str(row["observable_id"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<rect x="{left + 295}" y="{y + 2:.2f}" width="130" height="22" fill="{color}" opacity="0.9" />'
        )
        marks.append(
            f'<text x="{left + 307}" y="{y + 17:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#fff">struct/unc={structural}/{uncertainty}</text>'
        )
        marks.append(
            f'<text x="{left + 455}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">diagnostic: {str(row["target_diagnostic"]).replace("_", " ")[:30]}</text>'
        )
        marks.append(
            f'<text x="{left + 720}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">block: {str(row["primary_blocker"])[:28]}</text>'
        )
        marks.append(
            f'<text x="{left + 120}" y="{y + 38:.2f}" font-family="Arial, sans-serif" font-size="10" fill="#555">missing uncertainty: {str(row["missing_uncertainty_columns"])[:95]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Raw-curve ingestion contract</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">KA I/II fused validation needs machine-readable observable columns plus uncertainty columns before quantitative inversion.</text>
  {"".join(marks)}
  <rect x="95" y="405" width="14" height="14" fill="#2f855a" /><text x="116" y="417" font-family="Arial, sans-serif" font-size="12">uncertainty-weighted ready</text>
  <rect x="300" y="405" width="14" height="14" fill="#2b6cb0" /><text x="321" y="417" font-family="Arial, sans-serif" font-size="12">structural columns present, uncertainty missing</text>
  <rect x="610" y="405" width="14" height="14" fill="#c05621" /><text x="631" y="417" font-family="Arial, sans-serif" font-size="12">blocked before ingestion</text>
</svg>
"""
    path.write_text(svg)


def write_raw_curve_diagnostic_readiness_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 450
    left, top, bottom = 95, 105, 300
    row_h = (bottom - top) / len(rows)
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        structural = int(float(row["structural_diagnostic_ready"]))
        uncertainty = int(float(row["uncertainty_diagnostic_ready"]))
        color = "#2f855a" if uncertainty else "#2b6cb0" if structural else "#c05621"
        marks.append(
            f'<text x="{left}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">{str(row["diagnostic_id"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<rect x="{left + 330}" y="{y + 2:.2f}" width="140" height="22" fill="{color}" opacity="0.9" />'
        )
        marks.append(
            f'<text x="{left + 342}" y="{y + 17:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#fff">struct/unc={structural}/{uncertainty}</text>'
        )
        marks.append(
            f'<text x="{left + 500}" y="{y + 18:.2f}" font-family="Arial, sans-serif" font-size="12">requires: {str(row["required_observables"])[:54]}</text>'
        )
        marks.append(
            f'<text x="{left + 500}" y="{y + 38:.2f}" font-family="Arial, sans-serif" font-size="10" fill="#555">block: {str(row["blocking_observables_or_columns"])[:72]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="75" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Raw-curve diagnostic readiness</text>
  <text x="75" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Column contracts are aggregated into protocol-level readiness before any real-data inversion claim.</text>
  {"".join(marks)}
  <rect x="95" y="365" width="14" height="14" fill="#2f855a" /><text x="116" y="377" font-family="Arial, sans-serif" font-size="12">uncertainty-weighted diagnostic ready</text>
  <rect x="350" y="365" width="14" height="14" fill="#2b6cb0" /><text x="371" y="377" font-family="Arial, sans-serif" font-size="12">structural diagnostic only</text>
  <rect x="560" y="365" width="14" height="14" fill="#c05621" /><text x="581" y="377" font-family="Arial, sans-serif" font-size="12">blocked diagnostic</text>
</svg>
"""
    path.write_text(svg)


def write_raw_curve_persistence_exchange_protocol_svg(
    path: Path,
    rows: list[dict[str, float | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left, top, right, bottom = 90, 98, 1035, 415
    z_threshold = float(rows[0]["z_threshold"])
    labels = ["multi-k alpha", "late NGP", "chi4 peak"]
    keys = ["max_multik_tau_alpha_z", "late_ngp_z", "chi4_peak_z"]
    colors = ["#2b6cb0", "#2f855a", "#805ad5"]
    z_values = np.array([[float(row[key]) for key in keys] for row in rows])
    z_max = max(float(np.max(z_values)), z_threshold) * 1.15

    def y(value: float) -> float:
        return bottom + value * (top - bottom) / z_max

    group_x = np.linspace(left + 190, right - 190, len(rows))
    offsets = [-42, 0, 42]
    marks = []
    for idx, row in enumerate(rows):
        scenario = str(row["scenario"]).replace("_", " ")
        for jdx, label in enumerate(labels):
            value = z_values[idx, jdx]
            consistent_key = [
                "multik_tau_alpha_z_consistent",
                "late_ngp_z_consistent",
                "chi4_peak_z_consistent",
            ][jdx]
            passed = float(row[consistent_key]) > 0.5
            color = colors[jdx] if passed else "#c05621"
            x = group_x[idx] + offsets[jdx]
            marks.append(
                f'<rect x="{x - 13:.1f}" y="{y(value):.1f}" width="26" height="{bottom - y(value):.1f}" fill="{color}" />'
            )
            marks.append(
                f'<text x="{x - 30:.1f}" y="{bottom + 52 + jdx * 16}" font-family="Arial, sans-serif" font-size="10" fill="{color}">{label}</text>'
            )
        marks.append(
            f'<text x="{group_x[idx] - 72:.1f}" y="{bottom + 35}" font-family="Arial, sans-serif" font-size="12">{scenario}</text>'
        )
        marks.append(
            f'<text x="{group_x[idx] - 88:.1f}" y="{top + 32}" font-family="Arial, sans-serif" font-size="12">tau_p/tau_x={float(row["persistence_exchange_ratio"]):.2f}</text>'
        )
        marks.append(
            f'<text x="{group_x[idx] - 88:.1f}" y="{top + 50}" font-family="Arial, sans-serif" font-size="12">SE growth={float(row["stokes_einstein_growth_over_poisson"]):.2f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="90" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Raw-curve persistence/exchange inversion</text>
  <text x="90" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Machine-readable F_s(k,t), late NGP, D, and chi4 curves are reduced to observables, then scored as held-out z tests.</text>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="38" y="285" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 38 285)">absolute z score</text>
  <line x1="{left}" y1="{y(z_threshold):.1f}" x2="{right}" y2="{y(z_threshold):.1f}" stroke="#718096" stroke-dasharray="5 4" />
  <text x="{right - 78}" y="{y(z_threshold) - 8:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#718096">z={z_threshold:g}</text>
  {"".join(marks)}
</svg>
"""
    path.write_text(svg)


def write_trajectory_observable_protocol_svg(
    path: Path,
    rows: list[dict[str, float | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left, top, right, bottom = 90, 95, 1035, 420
    lag = np.array([float(row["lag_time"]) for row in rows])
    msd = np.array([float(row["msd"]) for row in rows])
    ngp = np.array([float(row["ngp"]) for row in rows])
    chi4 = np.array([float(row["chi4_overlap"]) for row in rows])
    fs = np.array([float(row["self_intermediate_scattering"]) for row in rows])

    def x(values: np.ndarray) -> np.ndarray:
        return scale(values, left, right)

    def y(values: np.ndarray) -> np.ndarray:
        return scale(values, bottom, top)

    curves = [
        ("MSD", msd, "#2b6cb0"),
        ("NGP", ngp, "#c05621"),
        ("overlap chi4", chi4, "#805ad5"),
        ("F_s(k0,t)", fs, "#2f855a"),
    ]
    marks = []
    for idx, (label, values, color) in enumerate(curves):
        marks.append(polyline(x(lag), y(values), color, width=2.5))
        marks.append(
            f'<text x="{left + 18 + idx * 190}" y="{bottom + 56}" font-family="Arial, sans-serif" font-size="12" fill="{color}">{label}</text>'
        )
        for xx, yy in zip(x(lag), y(values)):
            marks.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}" />')
    peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="90" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Trajectory-to-observable bridge</text>
  <text x="90" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Particle trajectories are reduced to MSD, NGP, self-intermediate scattering, and overlap chi4 rows before inversion.</text>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{(left + right) / 2 - 42}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">lag time</text>
  <text x="38" y="300" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 38 300)">scaled observable</text>
  {"".join(marks)}
  <text x="{left}" y="{bottom + 90}" font-family="Arial, sans-serif" font-size="11">peak chi4 lag = {float(peak["lag_time"]):.1f}; chi4 = {float(peak["chi4_overlap"]):.3f}; NGP = {float(peak["ngp"]):.3f}</text>
  <text x="{left}" y="{bottom + 108}" font-family="Arial, sans-serif" font-size="11">observable set: {peak["structural_observable_set"]}</text>
</svg>
"""
    path.write_text(svg)


def write_trajectory_uncertainty_protocol_svg(
    path: Path,
    rows: list[dict[str, float | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1120, 560
    left, top, right, bottom = 90, 95, 1035, 420
    lag = np.array([float(row["lag_time"]) for row in rows])
    sigma_msd = np.array([float(row["sigma_msd"]) for row in rows])
    sigma_ngp = np.array([float(row["sigma_ngp"]) for row in rows])
    sigma_fs = np.array([float(row["sigma_self_intermediate_scattering"]) for row in rows])
    sigma_chi4 = np.array([float(row["sigma_chi4_overlap"]) for row in rows])

    def x(values: np.ndarray) -> np.ndarray:
        return scale(values, left, right)

    def y(values: np.ndarray) -> np.ndarray:
        return scale(values, bottom, top)

    curves = [
        ("sigma MSD", sigma_msd, "#2b6cb0"),
        ("sigma NGP", sigma_ngp, "#c05621"),
        ("sigma F_s", sigma_fs, "#2f855a"),
        ("sigma chi4", sigma_chi4, "#805ad5"),
    ]
    marks = []
    for idx, (label, values, color) in enumerate(curves):
        marks.append(polyline(x(lag), y(values), color, width=2.5))
        marks.append(
            f'<text x="{left + 18 + idx * 190}" y="{bottom + 56}" font-family="Arial, sans-serif" font-size="12" fill="{color}">{label}</text>'
        )
        for xx, yy in zip(x(lag), y(values)):
            marks.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}" />')
    peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="90" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Trajectory uncertainty bridge</text>
  <text x="90" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Time-origin block jackknife supplies uncertainty columns for trajectory-derived observables.</text>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
  <line x1="{left}" y1="{bottom}" x2="{left}" y2="{top}" stroke="#222" />
  <text x="{(left + right) / 2 - 42}" y="{bottom + 38}" font-family="Arial, sans-serif" font-size="13">lag time</text>
  <text x="38" y="300" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 38 300)">scaled uncertainty</text>
  {"".join(marks)}
  <text x="{left}" y="{bottom + 90}" font-family="Arial, sans-serif" font-size="11">method: {peak["uncertainty_method"]}; blocks = {int(float(peak["jackknife_block_count"]))}; primary blocker = {peak["primary_blocker"]}</text>
  <text x="{left}" y="{bottom + 108}" font-family="Arial, sans-serif" font-size="11">peak chi4 row: sigma_MSD = {float(peak["sigma_msd"]):.3f}, sigma_Fs = {float(peak["sigma_self_intermediate_scattering"]):.3f}, sigma_chi4 = {float(peak["sigma_chi4_overlap"]):.3f}</text>
</svg>
"""
    path.write_text(svg)


def write_trajectory_inversion_readiness_svg(
    path: Path,
    rows: list[dict[str, float | str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1080, 430
    left, top = 80, 110
    row_h = 82
    colors_by_stage = {
        "uncertainty_weighted_trajectory_inversion": "#2f855a",
        "structural_trajectory_only": "#2b6cb0",
        "trajectory_blocked": "#c05621",
    }
    marks = []
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        stage = str(row["readiness_stage"])
        color = colors_by_stage[stage]
        marks.append(
            f'<text x="{left}" y="{y + 17}" font-family="Arial, sans-serif" font-size="12">{str(row["benchmark_id"]).replace("_", " ")[:44]}</text>'
        )
        marks.append(
            f'<rect x="{left + 330}" y="{y}" width="230" height="24" fill="{color}" opacity="0.92" />'
        )
        marks.append(
            f'<text x="{left + 342}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11" fill="#fff">{stage.replace("_", " ")[:36]}</text>'
        )
        marks.append(
            f'<text x="{left + 590}" y="{y + 17}" font-family="Arial, sans-serif" font-size="12">lags={int(float(row["lag_count"]))}; structural={int(float(row["structural_trajectory_ready"]))}; uncertainty={int(float(row["uncertainty_weighted_ready"]))}</text>'
        )
        marks.append(
            f'<text x="{left + 330}" y="{y + 44}" font-family="Arial, sans-serif" font-size="10" fill="#555">blocker: {str(row["primary_blocker"]).replace("_", " ")}</text>'
        )
        marks.append(
            f'<text x="{left + 590}" y="{y + 44}" font-family="Arial, sans-serif" font-size="10" fill="#555">missing sigma: {str(row["missing_uncertainty_columns"]).replace("_", " ")[:58]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="80" y="42" font-family="Arial, sans-serif" font-size="24" font-weight="700">Trajectory inversion readiness gate</text>
  <text x="80" y="66" font-family="Arial, sans-serif" font-size="13" fill="#444">Trajectory-derived observables are promoted only when structural observables and uncertainty columns are both present.</text>
  <text x="{left}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">benchmark</text>
  <text x="{left + 330}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">stage</text>
  <text x="{left + 590}" y="{top - 22}" font-family="Arial, sans-serif" font-size="12" font-weight="700">readiness details</text>
  {"".join(marks)}
  <rect x="80" y="335" width="14" height="14" fill="#2f855a" /><text x="102" y="347" font-family="Arial, sans-serif" font-size="12">uncertainty-weighted trajectory inversion</text>
  <rect x="335" y="335" width="14" height="14" fill="#2b6cb0" /><text x="357" y="347" font-family="Arial, sans-serif" font-size="12">structural trajectory only</text>
  <rect x="535" y="335" width="14" height="14" fill="#c05621" /><text x="557" y="347" font-family="Arial, sans-serif" font-size="12">blocked trajectory</text>
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
    write_sota_comparison_csv(DATA_DIR / "renewal_cage_sota_comparison.csv")
    sota_claim_alignment_rows = write_sota_claim_alignment_csv(
        DATA_DIR / "renewal_cage_sota_claim_alignment.csv"
    )
    write_sota_claim_alignment_svg(
        FIGURE_DIR / "renewal_cage_sota_claim_alignment.svg",
        sota_claim_alignment_rows,
    )
    sota_signed_constraint_rows = write_sota_signed_constraints_csv(
        DATA_DIR / "renewal_cage_sota_signed_constraints.csv"
    )
    write_sota_signed_constraints_svg(
        FIGURE_DIR / "renewal_cage_sota_signed_constraints.svg",
        sota_signed_constraint_rows,
    )
    real_assimilation_rows = write_real_benchmark_assimilation_gate_csv(
        DATA_DIR / "renewal_cage_real_benchmark_assimilation_gate.csv"
    )
    write_real_benchmark_assimilation_gate_svg(
        FIGURE_DIR / "renewal_cage_real_benchmark_assimilation_gate.svg",
        real_assimilation_rows,
    )
    prediction_ledger_rows = write_cross_observable_prediction_ledger_csv(
        DATA_DIR / "renewal_cage_cross_observable_prediction_ledger.csv"
    )
    write_cross_observable_prediction_ledger_svg(
        FIGURE_DIR / "renewal_cage_cross_observable_prediction_ledger.svg",
        prediction_ledger_rows,
    )
    identifiability_rows = write_inversion_identifiability_audit_csv(
        DATA_DIR / "renewal_cage_inversion_identifiability_audit.csv"
    )
    write_inversion_identifiability_audit_svg(
        FIGURE_DIR / "renewal_cage_inversion_identifiability_audit.svg",
        identifiability_rows,
    )
    frontier_horizon_rows = write_frontier_benchmark_horizon_csv(
        DATA_DIR / "renewal_cage_frontier_benchmark_horizon.csv"
    )
    write_frontier_benchmark_horizon_svg(
        FIGURE_DIR / "renewal_cage_frontier_benchmark_horizon.svg",
        frontier_horizon_rows,
    )
    source_provenance_rows = write_sota_source_provenance_csv(
        DATA_DIR / "renewal_cage_sota_source_provenance.csv"
    )
    write_sota_source_provenance_svg(
        FIGURE_DIR / "renewal_cage_sota_source_provenance.svg",
        source_provenance_rows,
    )
    data_accession_rows = write_sota_data_accession_csv(
        DATA_DIR / "renewal_cage_sota_data_accession.csv"
    )
    write_sota_data_accession_svg(
        FIGURE_DIR / "renewal_cage_sota_data_accession.svg",
        data_accession_rows,
    )
    readme_schema_rows = write_sota_readme_schema_csv(
        DATA_DIR / "renewal_cage_sota_readme_schema.csv"
    )
    write_sota_readme_schema_svg(
        FIGURE_DIR / "renewal_cage_sota_readme_schema.svg",
        readme_schema_rows,
    )
    trajectory_adapter_contract_rows = write_trajectory_adapter_contract_csv(
        DATA_DIR / "renewal_cage_trajectory_adapter_contract.csv"
    )
    write_trajectory_adapter_contract_svg(
        FIGURE_DIR / "renewal_cage_trajectory_adapter_contract.svg",
        trajectory_adapter_contract_rows,
    )
    literature_readiness_rows = write_literature_inversion_readiness_csv(
        DATA_DIR / "renewal_cage_literature_inversion_readiness.csv"
    )
    write_literature_inversion_readiness_svg(
        FIGURE_DIR / "renewal_cage_literature_inversion_readiness.svg",
        literature_readiness_rows,
    )
    observable_falsification_rows = write_observable_falsification_matrix_csv(
        DATA_DIR / "renewal_cage_observable_falsification_matrix.csv",
        literature_readiness_rows,
    )
    write_observable_falsification_matrix_svg(
        FIGURE_DIR / "renewal_cage_observable_falsification_matrix.svg",
        observable_falsification_rows,
    )
    benchmark_fusion_rows = write_benchmark_fusion_readiness_csv(
        DATA_DIR / "renewal_cage_benchmark_fusion_readiness.csv"
    )
    write_benchmark_fusion_readiness_svg(
        FIGURE_DIR / "renewal_cage_benchmark_fusion_readiness.svg",
        benchmark_fusion_rows,
    )
    raw_curve_contract_rows = write_raw_curve_ingestion_contract_csv(
        DATA_DIR / "renewal_cage_raw_curve_ingestion_contract.csv"
    )
    write_raw_curve_ingestion_contract_svg(
        FIGURE_DIR / "renewal_cage_raw_curve_ingestion_contract.svg",
        raw_curve_contract_rows,
    )
    raw_diagnostic_rows = write_raw_curve_diagnostic_readiness_csv(
        DATA_DIR / "renewal_cage_raw_curve_diagnostic_readiness.csv",
        raw_curve_contract_rows,
    )
    write_raw_curve_diagnostic_readiness_svg(
        FIGURE_DIR / "renewal_cage_raw_curve_diagnostic_readiness.svg",
        raw_diagnostic_rows,
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
    tail_ratio_rows = write_tail_ratio_csv(
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
    kww_alpha_rows = write_kww_alpha_csv(
        DATA_DIR / "renewal_cage_kww_alpha.csv",
        np.array([1.0, 0.78, 0.62]),
        temperature_law,
        exchange_law,
        wave_number=1.1,
        min_decay=0.15,
        max_decay=0.85,
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
    spatial_chi4_rows = write_spatial_chi4_csv(
        DATA_DIR / "renewal_cage_spatial_chi4.csv",
        temperatures,
        temperature_law,
        wave_number=1.1,
        facilitation_diffusivity=0.04,
        particle_density=0.85,
    )
    write_spatial_chi4_svg(FIGURE_DIR / "renewal_cage_spatial_chi4.svg", spatial_chi4_rows)
    write_spatial_facilitation_inversion_csv(
        DATA_DIR / "renewal_cage_spatial_facilitation_inversion.csv",
        spatial_chi4_rows,
        dimension=3,
        particle_density=0.85,
        microscopic_length=1.0,
        max_diffusivity_relative_std=0.05,
        min_length_growth=1.5,
    )
    thermodynamic_rows = write_thermodynamic_closure_csv(
        DATA_DIR / "renewal_cage_thermodynamic_closure.csv",
        np.array([1.0, 0.82, 0.68, 0.58, 0.52]),
        ConfigurationalEntropyParams(
            reference_temperature=1.0,
            entropy_ref=1.2,
            kauzmann_temperature=0.45,
        ),
        activation_free_energy=1.6,
        tau_ref=3.0,
        renewal_rate_ref=params.renewal_rate,
        wave_number=1.1,
        cage_variance=params.cage_variance,
        cage_tau=params.cage_tau,
        jump_variance=params.jump_variance,
    )
    write_thermodynamic_closure_svg(
        FIGURE_DIR / "renewal_cage_thermodynamic_closure.svg",
        thermodynamic_rows,
    )
    mct_beta = MCTBetaParams(
        plateau=0.68,
        critical_amplitude=0.08,
        von_schweidler_amplitude=0.05,
        critical_exponent=0.32,
        von_schweidler_exponent=0.6,
        beta_time=4.0,
    )
    mct_beta_rows = write_mct_beta_closure_csv(
        DATA_DIR / "renewal_cage_mct_beta_closure.csv",
        np.array([1.0, 0.82, 0.68, 0.58, 0.52]),
        mct_beta,
        beta_time_activation=2.4,
        plateau_growth=0.14,
        alpha_time_ref=30.0,
        alpha_activation=5.0,
    )
    write_mct_beta_closure_svg(
        FIGURE_DIR / "renewal_cage_mct_beta_closure.svg",
        mct_beta_rows,
        mct_beta,
    )
    persistence_exchange_joint_protocol_rows = write_persistence_exchange_joint_protocol_csv(
        DATA_DIR / "renewal_cage_persistence_exchange_joint_protocol.csv",
        anchor_wave_number=1.1,
        wave_numbers=[0.7, 1.1, 1.6],
        jump_variance=0.7,
        exchange_mean=1.0,
        true_ratio=8.0,
    )
    write_persistence_exchange_joint_protocol_svg(
        FIGURE_DIR / "renewal_cage_persistence_exchange_joint_protocol.svg",
        persistence_exchange_joint_protocol_rows,
    )
    sota_benchmark_rows = write_sota_benchmark_consistency_csv(
        DATA_DIR / "renewal_cage_sota_benchmark_consistency.csv",
        mct_beta,
        params,
        GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0),
        temperature_law,
        temperature_rows,
        spatial_chi4_rows,
        alpha_shape_rows,
        kww_alpha_rows,
        persistence_exchange_joint_protocol_rows,
        tail_ratio_rows,
        thermodynamic_rows,
    )
    write_sota_benchmark_consistency_svg(
        FIGURE_DIR / "renewal_cage_sota_benchmark_consistency.svg",
        sota_benchmark_rows,
    )
    cooling_interval = 1.0 / temperatures[-1] - 1.0 / temperatures[0]
    barrier_requirement_rows = write_barrier_requirements_csv(
        DATA_DIR / "renewal_cage_barrier_requirements.csv",
        hot_temperature=float(temperatures[0]),
        cold_temperature=float(temperatures[-1]),
        delay_barrier_gaps=[0.0, 1.5, 3.0],
        exchange_barrier_sums=[0.0, 2.0, 4.0],
        target_lambda_tau_delay_growth=math.exp(3.0 * cooling_interval),
        target_heterogeneity_ratio_growth=math.exp(4.0 * cooling_interval),
    )
    write_barrier_requirements_svg(
        FIGURE_DIR / "renewal_cage_barrier_requirements.svg",
        barrier_requirement_rows,
    )
    persistence_exchange_rows = write_persistence_exchange_csv(
        DATA_DIR / "renewal_cage_persistence_exchange.csv",
        ratios=[1.0, 2.0, 4.0, 8.0, 12.0],
        exchange_mean=1.0,
        wave_number=1.1,
    )
    write_persistence_exchange_svg(
        FIGURE_DIR / "renewal_cage_persistence_exchange.svg",
        persistence_exchange_rows,
    )
    persistence_exchange_protocol_rows = write_persistence_exchange_protocol_csv(
        DATA_DIR / "renewal_cage_persistence_exchange_protocol.csv",
        wave_number=1.1,
        jump_variance=0.7,
        exchange_mean=1.0,
        true_ratio=9.0,
    )
    write_persistence_exchange_protocol_svg(
        FIGURE_DIR / "renewal_cage_persistence_exchange_protocol.svg",
        persistence_exchange_protocol_rows,
    )
    persistence_exchange_uncertainty_protocol_rows = write_persistence_exchange_uncertainty_protocol_csv(
        DATA_DIR / "renewal_cage_persistence_exchange_uncertainty_protocol.csv",
        anchor_wave_number=1.1,
        wave_numbers=[0.7, 1.1, 1.6],
        jump_variance=0.7,
        exchange_mean=1.0,
        true_ratio=8.0,
    )
    write_persistence_exchange_uncertainty_protocol_svg(
        FIGURE_DIR / "renewal_cage_persistence_exchange_uncertainty_protocol.svg",
        persistence_exchange_uncertainty_protocol_rows,
    )
    translation_rotation_rows = write_translation_rotation_protocol_csv(
        DATA_DIR / "renewal_cage_translation_rotation_protocol.csv",
        wave_number=1.1,
    )
    write_translation_rotation_protocol_svg(
        FIGURE_DIR / "renewal_cage_translation_rotation_protocol.svg",
        translation_rotation_rows,
    )
    raw_curve_persistence_exchange_rows = write_raw_curve_persistence_exchange_protocol_csv(
        DATA_DIR / "renewal_cage_raw_curve_persistence_exchange_protocol.csv"
    )
    write_raw_curve_persistence_exchange_protocol_svg(
        FIGURE_DIR / "renewal_cage_raw_curve_persistence_exchange_protocol.svg",
        raw_curve_persistence_exchange_rows,
    )
    trajectory_observable_rows = write_trajectory_observable_protocol_csv(
        DATA_DIR / "renewal_cage_trajectory_observable_protocol.csv"
    )
    write_trajectory_observable_protocol_svg(
        FIGURE_DIR / "renewal_cage_trajectory_observable_protocol.svg",
        trajectory_observable_rows,
    )
    write_trajectory_adapter_demo_csv(
        DATA_DIR / "renewal_cage_trajectory_adapter_demo.csv"
    )
    write_trajectory_csv_adapter_demo_csv(
        DATA_DIR / "renewal_cage_trajectory_csv_adapter_demo.csv"
    )
    trajectory_uncertainty_rows = write_trajectory_uncertainty_protocol_csv(
        DATA_DIR / "renewal_cage_trajectory_uncertainty_protocol.csv"
    )
    write_trajectory_uncertainty_protocol_svg(
        FIGURE_DIR / "renewal_cage_trajectory_uncertainty_protocol.svg",
        trajectory_uncertainty_rows,
    )
    trajectory_inversion_readiness_rows = write_trajectory_inversion_readiness_csv(
        DATA_DIR / "renewal_cage_trajectory_inversion_readiness.csv",
        observable_rows=trajectory_observable_rows,
        uncertainty_rows=trajectory_uncertainty_rows,
    )
    write_trajectory_inversion_readiness_svg(
        FIGURE_DIR / "renewal_cage_trajectory_inversion_readiness.svg",
        trajectory_inversion_readiness_rows,
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
