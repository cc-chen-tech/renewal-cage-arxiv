import math
import sys
import unittest
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
    alpha_relaxation_time,
    apparent_alpha_activation_energies,
    activated_barrier_temperature_law,
    alpha_tts_benchmark_consistency,
    alpha_relaxation_shape_curve,
    alpha_shape_superposition_residual,
    barrier_amplification_laws,
    benchmark_fusion_readiness,
    cage_localization_benchmark_consistency,
    cage_localization_diagnostics,
    correlated_domain_susceptibility,
    classify_delay_exponent,
    delayed_poisson_mean,
    delayed_renewal_shape,
    dimensionless_peak_prediction,
    fractional_stokes_einstein_exponents,
    gamma_exchange_temperature_scan,
    glass_phenomenon_audit,
    glass_signature_phase_diagram,
    dynamic_heterogeneity_benchmark_consistency,
    infer_parameters_from_full_observables,
    infer_renewal_correlation_size,
    infer_parameters_from_scattering_transport,
    inversion_identifiability_audit,
    joint_inversion_benchmark_consistency,
    literature_inversion_readiness,
    generalized_delay_ngp_short_time,
    gaussian_radial_3d,
    gamma_exchange_alpha_relaxation_time,
    gamma_exchange_count_moments,
    gamma_exchange_asymptotic_diagnostics,
    gamma_exchange_diagnostic_map,
    infer_gamma_exchange_from_late_observables,
    infer_gamma_exchange_multik_collapse,
    infer_gamma_exchange_uncertainty_from_late_observables,
    infer_gamma_exchange_ratio_from_alpha_rate,
    gamma_exchange_ngp_1d,
    gamma_exchange_normalized_alpha_decay,
    gamma_exchange_scattering_susceptibility,
    gamma_exchange_self_intermediate_scattering,
    fragility_benchmark_consistency,
    frontier_benchmark_horizon,
    gaussian_recovery_benchmark_consistency,
    infer_spatial_facilitation_diffusivity,
    kww_alpha_fit,
    long_time_diffusion_coefficient,
    local_alpha_stretching_exponent,
    late_mechanism_selection,
    minimal_barrier_requirements,
    MCTBetaParams,
    mct_beta_correlator,
    mct_beta_benchmark_consistency,
    mct_exponent_benchmark_consistency,
    mct_beta_temperature_scan,
    ngp_peak_benchmark_consistency,
    observable_consistency_diagnostics,
    observable_falsification_matrix,
    raw_curve_ingestion_contract,
    raw_curve_diagnostic_readiness,
    raw_curve_persistence_exchange_protocol,
    real_benchmark_assimilation_gate,
    cross_observable_prediction_ledger,
    persistence_exchange_benchmark_consistency,
    radial_van_hove_3d,
    van_hove_tail_benchmark_consistency,
    local_cage_variance,
    moments_1d,
    moments_3d,
    ngp_1d,
    ngp_3d,
    normalized_alpha_decay,
    plateau_ngp_branches,
    plateau_peak_diagnostics,
    peak_relaxation_coupling,
    PersistenceExchangeParams,
    persistence_exchange_alpha_relaxation_time,
    persistence_exchange_count_distribution,
    persistence_exchange_count_pgf,
    persistence_exchange_count_moments,
    persistence_exchange_diffusion_coefficient,
    persistence_exchange_data_protocol,
    persistence_exchange_joint_diagnostic,
    persistence_exchange_ngp_1d,
    persistence_exchange_normalized_alpha_decay,
    persistence_exchange_scan,
    persistence_exchange_scattering_susceptibility,
    renewal_scattering_susceptibility,
    self_intermediate_scattering,
    static_gamma_asymptotic_diagnostics,
    static_gamma_count_moments,
    static_gamma_ngp_1d,
    static_gamma_normalized_alpha_decay,
    spatial_facilitation_growth_law_consistency,
    stokes_einstein_benchmark_consistency,
    stretched_alpha_benchmark_consistency,
    stokes_einstein_product,
    sota_claim_alignment,
    sota_signed_constraint_audit,
    thermodynamic_scope_benchmark_consistency,
    temperature_dependent_params,
    temperature_dependent_gamma_exchange,
    temperature_scan,
    trajectory_observable_protocol,
    trajectory_observable_uncertainty_protocol,
    TranslationRotationExchangeParams,
    translation_rotation_decoupling_diagnostic,
    translation_rotation_inversion_protocol,
    translation_rotation_rotational_relaxation_time,
)


class DelayedRenewalCageTests(unittest.TestCase):
    def test_delayed_poisson_mean_has_cubic_short_time_onset(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=1.0,
            jump_variance=0.5,
            renewal_rate=2.0,
            renewal_delay=4.0,
        )
        t = np.array([1e-4, 2e-4, 4e-4])
        mean = delayed_poisson_mean(t, params)
        expected = params.renewal_rate * t**3 / (3.0 * params.renewal_delay**2)

        np.testing.assert_allclose(mean, expected, rtol=2e-4, atol=1e-16)

    def test_local_cage_variance_has_plateau(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.7,
            cage_tau=0.8,
            jump_variance=0.5,
            renewal_rate=0.2,
            renewal_delay=2.0,
        )
        t = np.array([0.0, 10.0])
        variance = local_cage_variance(t, params)

        self.assertAlmostEqual(variance[0], 0.0)
        self.assertAlmostEqual(variance[-1], params.cage_variance, delta=1e-5)

    def test_moments_generate_plateau_then_long_time_growth(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.8,
            jump_variance=0.4,
            renewal_rate=0.25,
            renewal_delay=2.5,
        )
        early = moments_1d(np.array([1.0]), params)["m2"][0]
        middle = moments_1d(np.array([8.0]), params)["m2"][0]
        late = moments_1d(np.array([80.0]), params)["m2"][0]

        self.assertGreater(middle, early)
        self.assertLess(abs(middle - params.cage_variance), 0.5)
        self.assertGreater(late, middle + 5.0)

    def test_ngp_starts_near_zero_has_peak_and_decays(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        t = np.linspace(1e-4, 220.0, 2600)
        alpha = ngp_1d(t, params)
        peak_idx = int(np.argmax(alpha))

        self.assertLess(alpha[0], 1e-3)
        self.assertGreater(alpha[peak_idx], 0.1)
        self.assertGreater(peak_idx, 10)
        self.assertLess(alpha[-1], alpha[peak_idx] / 5.0)

    def test_long_time_ngp_matches_inverse_renewal_count(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        t = np.array([500.0])
        renewal_count = delayed_poisson_mean(t, params)[0]
        alpha = ngp_1d(t, params)[0]

        self.assertAlmostEqual(alpha * renewal_count, 1.0, delta=0.03)

    def test_three_dimensional_ngp_matches_variance_mixture_ngp(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        t = np.linspace(0.1, 40.0, 200)

        np.testing.assert_allclose(ngp_3d(t, params), ngp_1d(t, params), rtol=1e-12, atol=1e-12)

        moments = moments_3d(t, params)
        self.assertTrue(np.all(moments["r2"] > 0.0))
        self.assertTrue(np.all(moments["r4"] > 0.0))

    def test_self_intermediate_scattering_has_cage_plateau_and_alpha_decay(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.array([0.0, 1.0, 600.0])
        scattering = self_intermediate_scattering(wave_number, times, params)
        plateau = math.exp(-0.5 * wave_number**2 * params.cage_variance)
        alpha_rate = params.renewal_rate * (1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance))

        self.assertAlmostEqual(scattering[0], 1.0)
        self.assertAlmostEqual(scattering[1], plateau, delta=0.02)
        self.assertAlmostEqual(-math.log(scattering[-1] / plateau) / times[-1], alpha_rate, delta=0.003)

    def test_normalized_alpha_decay_removes_cage_debye_waller_factor(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.array([0.0, 8.0, 600.0])
        decay = normalized_alpha_decay(wave_number, times, params)
        alpha_rate = params.renewal_rate * (1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance))

        self.assertAlmostEqual(decay[0], 1.0)
        self.assertLess(decay[1], decay[0])
        self.assertAlmostEqual(-math.log(decay[-1]) / times[-1], alpha_rate, delta=0.003)

    def test_alpha_relaxation_time_solves_cage_normalized_decay_threshold(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        tau_alpha = alpha_relaxation_time(wave_number, params)
        decay = normalized_alpha_decay(wave_number, np.array([tau_alpha]), params)[0]

        self.assertAlmostEqual(decay, math.exp(-1.0), delta=1e-12)
        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        self.assertGreater(tau_alpha, 1.0 / (params.renewal_rate * gamma))

    def test_peak_relaxation_coupling_links_ngp_peak_to_alpha_time(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        threshold = math.exp(-1.0)

        coupling = peak_relaxation_coupling(wave_number, params, threshold=threshold)
        peak = dimensionless_peak_prediction(params)
        tau_alpha = alpha_relaxation_time(wave_number, params, threshold=threshold)
        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        alpha_count = -math.log(threshold) / gamma
        peak_count = params.cage_variance / params.jump_variance

        self.assertAlmostEqual(coupling["peak_time"], peak["peak_time"], delta=1e-12)
        self.assertAlmostEqual(coupling["tau_alpha"], tau_alpha, delta=1e-12)
        self.assertAlmostEqual(coupling["peak_ngp"], peak["peak_ngp"], delta=1e-12)
        self.assertAlmostEqual(coupling["gamma_k"], gamma, delta=1e-12)
        self.assertAlmostEqual(coupling["peak_renewal_count"], peak_count, delta=1e-12)
        self.assertAlmostEqual(coupling["alpha_renewal_count"], alpha_count, delta=1e-12)
        self.assertAlmostEqual(
            coupling["alpha_to_peak_renewal_count_ratio"],
            alpha_count / peak_count,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            coupling["tau_alpha_over_peak_time"],
            tau_alpha / peak["peak_time"],
            delta=1e-12,
        )
        self.assertGreater(coupling["tau_alpha_over_peak_time"], 1.0)

    def test_alpha_shape_curve_depends_only_on_delay_control(self):
        params_a = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        params_b = DelayedRenewalCageParams(
            cage_variance=1.8,
            cage_tau=0.9,
            jump_variance=0.8,
            renewal_rate=0.09,
            renewal_delay=6.0,
        )
        scaled_times = np.geomspace(0.15, 4.0, 120)

        curve_a = alpha_relaxation_shape_curve(1.1, params_a, scaled_times)
        curve_b = alpha_relaxation_shape_curve(1.1, params_b, scaled_times)

        np.testing.assert_allclose(curve_a, curve_b, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(alpha_relaxation_shape_curve(1.1, params_a, np.array([1.0]))[0], 1.0)

    def test_alpha_shape_superposition_residual_detects_control_change(self):
        reference = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        same_shape = DelayedRenewalCageParams(
            cage_variance=0.7,
            cage_tau=0.4,
            jump_variance=0.8,
            renewal_rate=0.09,
            renewal_delay=6.0,
        )
        different_shape = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=12.0,
        )
        scaled_times = np.geomspace(0.15, 4.0, 120)

        collapsed = alpha_shape_superposition_residual(
            1.1,
            reference,
            same_shape,
            scaled_times,
        )
        broken = alpha_shape_superposition_residual(
            1.1,
            reference,
            different_shape,
            scaled_times,
        )

        self.assertLess(collapsed["rms_log_shape_residual"], 1e-12)
        self.assertGreater(broken["rms_log_shape_residual"], 0.2)
        self.assertAlmostEqual(collapsed["reference_control"], collapsed["candidate_control"], delta=1e-12)
        self.assertGreater(broken["candidate_control"], broken["reference_control"])

    def test_renewal_scattering_susceptibility_is_closed_form_variance(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.linspace(0.0, 220.0, 900)
        susceptibility = renewal_scattering_susceptibility(wave_number, times, params)
        scattering = self_intermediate_scattering(wave_number, times, params)
        renewal = delayed_poisson_mean(times, params)
        jump_characteristic = math.exp(-0.5 * wave_number**2 * params.jump_variance)
        relative = np.exp(renewal * (jump_characteristic - 1.0) ** 2) - 1.0

        self.assertAlmostEqual(susceptibility[0], 0.0)
        self.assertTrue(np.all(susceptibility >= -1e-14))
        self.assertGreater(float(np.max(susceptibility)), 0.02)
        self.assertLess(susceptibility[-1], float(np.max(susceptibility)) / 4.0)
        np.testing.assert_allclose(susceptibility, scattering**2 * relative, rtol=1e-12, atol=1e-14)

    def test_gamma_exchange_count_moments_add_recovering_overdispersion(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        times = np.array([20.0, 120.0, 30000.0])

        count = gamma_exchange_count_moments(times, params, heterogeneity)
        renewal = delayed_poisson_mean(times, params)
        kappa_eff = heterogeneity.shape * (1.0 + renewal / heterogeneity.exchange_renewal_count)

        np.testing.assert_allclose(count["mean"], renewal, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(count["variance"], renewal + renewal**2 / kappa_eff, rtol=1e-12, atol=1e-14)
        self.assertGreater(count["variance"][1] / count["mean"][1], 2.0)

        alpha = gamma_exchange_ngp_1d(times, params, heterogeneity)
        self.assertGreater(alpha[1], ngp_1d(np.array([times[1]]), params)[0])
        self.assertLess(alpha[-1], alpha[1] / 20.0)

    def test_gamma_exchange_alpha_decay_has_stretched_like_window_and_recovery(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1
        times = np.linspace(0.02, 400.0, 2400)

        decay = gamma_exchange_normalized_alpha_decay(wave_number, times, params, heterogeneity)
        renewal = delayed_poisson_mean(times, params)
        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        kappa_eff = heterogeneity.shape * (1.0 + renewal / heterogeneity.exchange_renewal_count)
        expected = (1.0 + gamma * renewal / kappa_eff) ** (-kappa_eff)
        exponent = local_alpha_stretching_exponent(times, decay)
        alpha_window = (-np.log(decay) > 0.5) & (-np.log(decay) < 2.0)

        np.testing.assert_allclose(decay, expected, rtol=1e-12, atol=1e-14)
        self.assertLess(float(np.nanmedian(exponent[alpha_window])), 0.9)
        self.assertLess(decay[-1], 0.01)

    def test_gamma_exchange_scattering_susceptibility_matches_negative_binomial_variance(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.6, exchange_renewal_count=8.0)
        wave_number = 1.1
        times = np.linspace(0.0, 220.0, 900)

        susceptibility = gamma_exchange_scattering_susceptibility(wave_number, times, params, heterogeneity)
        scattering = gamma_exchange_self_intermediate_scattering(wave_number, times, params, heterogeneity)
        renewal = delayed_poisson_mean(times, params)
        local = local_cage_variance(times, params)
        kappa_eff = heterogeneity.shape * (1.0 + renewal / heterogeneity.exchange_renewal_count)
        jump_characteristic = math.exp(-0.5 * wave_number**2 * params.jump_variance)
        second_moment = np.exp(-wave_number**2 * local) * (
            1.0 + renewal * (1.0 - jump_characteristic**2) / kappa_eff
        ) ** (-kappa_eff)

        self.assertAlmostEqual(susceptibility[0], 0.0)
        self.assertTrue(np.all(susceptibility >= -1e-14))
        np.testing.assert_allclose(susceptibility, second_moment - scattering**2, rtol=1e-12, atol=1e-14)
        self.assertGreater(float(np.max(susceptibility)), float(np.max(renewal_scattering_susceptibility(wave_number, times, params))))

    def test_gamma_exchange_asymptotics_link_late_ngp_and_alpha_rate(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1
        late_time = np.array([30000.0])

        diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)
        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        heterogeneity_ratio = heterogeneity.exchange_renewal_count / heterogeneity.shape
        expected_rate_per_renewal = math.log1p(gamma * heterogeneity_ratio) / heterogeneity_ratio
        renewal = delayed_poisson_mean(late_time, params)[0]
        ngp = gamma_exchange_ngp_1d(late_time, params, heterogeneity)[0]
        decay = gamma_exchange_normalized_alpha_decay(wave_number, late_time, params, heterogeneity)[0]

        self.assertAlmostEqual(diagnostics["heterogeneity_ratio"], heterogeneity_ratio, delta=1e-12)
        self.assertAlmostEqual(diagnostics["late_ngp_renewal_amplitude"], 1.0 + heterogeneity_ratio, delta=1e-12)
        self.assertAlmostEqual(diagnostics["late_alpha_decay_per_renewal"], expected_rate_per_renewal, delta=1e-12)
        self.assertAlmostEqual(
            diagnostics["late_alpha_rate"],
            params.renewal_rate * expected_rate_per_renewal,
            delta=1e-12,
        )
        self.assertAlmostEqual(renewal * ngp, diagnostics["late_ngp_renewal_amplitude"], delta=0.08)
        self.assertAlmostEqual(-math.log(decay) / renewal, expected_rate_per_renewal, delta=2e-4)
        self.assertLess(diagnostics["alpha_rate_renormalization"], 1.0)

    def test_gamma_exchange_alpha_rate_inverts_heterogeneity_ratio(self):
        gamma = 0.38368679808771045
        heterogeneity_ratio = 25.0
        observed_rate_per_renewal = math.log1p(gamma * heterogeneity_ratio) / heterogeneity_ratio

        inferred = infer_gamma_exchange_ratio_from_alpha_rate(
            gamma_k=gamma,
            observed_decay_per_renewal=observed_rate_per_renewal,
        )

        self.assertAlmostEqual(inferred, heterogeneity_ratio, delta=1e-10)
        self.assertAlmostEqual(
            infer_gamma_exchange_ratio_from_alpha_rate(
                gamma_k=gamma,
                observed_decay_per_renewal=gamma,
            ),
            0.0,
            delta=1e-12,
        )

    def test_gamma_exchange_diagnostic_map_classifies_observable_window(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )

        rows = gamma_exchange_diagnostic_map(
            wave_number=1.1,
            params=params,
            shape=0.4,
            heterogeneity_ratios=[0.0, 2.0, 25.0],
        )

        self.assertEqual([row["heterogeneity_ratio"] for row in rows], [0.0, 2.0, 25.0])
        self.assertAlmostEqual(rows[0]["late_ngp_renewal_amplitude"], 1.0, delta=1e-12)
        self.assertAlmostEqual(rows[0]["alpha_rate_renormalization"], 1.0, delta=1e-12)
        self.assertEqual(rows[0]["passes_joint_criterion"], 0.0)
        self.assertEqual(rows[1]["passes_joint_criterion"], 1.0)
        self.assertEqual(rows[2]["passes_joint_criterion"], 1.0)
        self.assertGreater(rows[2]["late_ngp_renewal_amplitude"], rows[1]["late_ngp_renewal_amplitude"])
        self.assertLess(rows[2]["alpha_rate_renormalization"], rows[1]["alpha_rate_renormalization"])
        self.assertAlmostEqual(rows[2]["inferred_ratio_from_alpha_rate"], 25.0, delta=1e-10)
        self.assertAlmostEqual(rows[2]["log_ratio_residual"], 0.0, delta=1e-12)

    def test_gamma_exchange_late_observable_protocol_accepts_consistent_data(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1
        late_time = np.array([30000.0])
        renewal = delayed_poisson_mean(late_time, params)[0]
        late_ngp = gamma_exchange_ngp_1d(late_time, params, heterogeneity)[0]
        diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)

        inferred = infer_gamma_exchange_from_late_observables(
            wave_number=wave_number,
            params=params,
            late_renewal_count=renewal,
            late_ngp=late_ngp,
            observed_alpha_decay_per_renewal=diagnostics["late_alpha_decay_per_renewal"],
        )

        self.assertAlmostEqual(inferred["ratio_from_late_ngp"], 25.0, delta=0.08)
        self.assertAlmostEqual(inferred["ratio_from_alpha_rate"], 25.0, delta=1e-10)
        self.assertLess(abs(inferred["log_ratio_residual"]), 0.004)
        self.assertEqual(inferred["passes_consistency"], 1.0)

    def test_gamma_exchange_late_observable_protocol_rejects_inconsistent_data(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        gamma = 1.0 - math.exp(-0.5 * 1.1**2 * params.jump_variance)
        alpha_rate_for_c2 = math.log1p(gamma * 2.0) / 2.0

        inferred = infer_gamma_exchange_from_late_observables(
            wave_number=1.1,
            params=params,
            late_renewal_count=5400.0,
            late_ngp=26.0 / 5400.0,
            observed_alpha_decay_per_renewal=alpha_rate_for_c2,
        )

        self.assertAlmostEqual(inferred["ratio_from_late_ngp"], 25.0, delta=1e-12)
        self.assertAlmostEqual(inferred["ratio_from_alpha_rate"], 2.0, delta=1e-10)
        self.assertGreater(abs(inferred["log_ratio_residual"]), 2.0)
        self.assertEqual(inferred["passes_consistency"], 0.0)

    def test_gamma_exchange_late_observable_uncertainty_scores_residuals(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1
        late_time = np.array([30000.0])
        renewal = delayed_poisson_mean(late_time, params)[0]
        late_ngp = gamma_exchange_ngp_1d(late_time, params, heterogeneity)[0]
        diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)

        scored = infer_gamma_exchange_uncertainty_from_late_observables(
            wave_number=wave_number,
            params=params,
            late_renewal_count=renewal,
            late_ngp=late_ngp,
            observed_alpha_decay_per_renewal=diagnostics["late_alpha_decay_per_renewal"],
            late_renewal_count_std=0.01 * renewal,
            late_ngp_std=0.01 * late_ngp,
            alpha_decay_per_renewal_std=0.002,
        )

        self.assertGreater(scored["ratio_from_late_ngp_std"], 0.0)
        self.assertGreater(scored["ratio_from_alpha_rate_std"], 0.0)
        self.assertGreater(scored["log_ratio_residual_std"], 0.0)
        self.assertLess(scored["log_ratio_z_score"], 1.0)
        self.assertEqual(scored["passes_statistical_consistency"], 1.0)

        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        alpha_rate_for_c2 = math.log1p(gamma * 2.0) / 2.0
        mismatched = infer_gamma_exchange_uncertainty_from_late_observables(
            wave_number=wave_number,
            params=params,
            late_renewal_count=renewal,
            late_ngp=late_ngp,
            observed_alpha_decay_per_renewal=alpha_rate_for_c2,
            late_renewal_count_std=0.01 * renewal,
            late_ngp_std=0.01 * late_ngp,
            alpha_decay_per_renewal_std=0.002,
        )

        self.assertGreater(mismatched["log_ratio_z_score"], 20.0)
        self.assertEqual(mismatched["passes_statistical_consistency"], 0.0)

    def test_gamma_exchange_multik_collapse_accepts_shared_exchange_ratio(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_numbers = [0.6, 1.1, 1.8]
        shared_ratio = 25.0
        rates = []
        for wave_number in wave_numbers:
            gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
            rates.append(math.log1p(gamma * shared_ratio) / shared_ratio)

        collapse = infer_gamma_exchange_multik_collapse(
            wave_numbers=wave_numbers,
            params=params,
            late_renewal_count=5400.0,
            late_ngp=26.0 / 5400.0,
            observed_alpha_decay_per_renewal=rates,
            alpha_decay_per_renewal_std=[0.002, 0.002, 0.002],
            late_renewal_count_std=54.0,
            late_ngp_std=0.01 * 26.0 / 5400.0,
        )

        self.assertEqual(len(collapse["per_wave_number"]), 3)
        self.assertAlmostEqual(collapse["ratio_from_late_ngp"], 25.0, delta=1e-12)
        self.assertAlmostEqual(collapse["weighted_mean_ratio_from_alpha"], 25.0, delta=1e-8)
        self.assertLess(collapse["collapse_z_score"], 1.0)
        self.assertEqual(collapse["passes_multik_collapse"], 1.0)

    def test_gamma_exchange_multik_collapse_rejects_wave_number_mismatch(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_numbers = [0.6, 1.1, 1.8]
        rates = []
        for idx, wave_number in enumerate(wave_numbers):
            target_ratio = 2.0 if idx == 1 else 25.0
            gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
            rates.append(math.log1p(gamma * target_ratio) / target_ratio)

        collapse = infer_gamma_exchange_multik_collapse(
            wave_numbers=wave_numbers,
            params=params,
            late_renewal_count=5400.0,
            late_ngp=26.0 / 5400.0,
            observed_alpha_decay_per_renewal=rates,
            alpha_decay_per_renewal_std=[0.002, 0.002, 0.002],
            late_renewal_count_std=54.0,
            late_ngp_std=0.01 * 26.0 / 5400.0,
        )

        self.assertGreater(collapse["collapse_z_score"], 20.0)
        self.assertEqual(collapse["passes_multik_collapse"], 0.0)

    def test_late_mechanism_selection_identifies_finite_exchange_recovery(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        times = np.array([10000.0, 30000.0])
        renewal = delayed_poisson_mean(times, params)
        alpha = gamma_exchange_ngp_1d(times, params, heterogeneity)
        observed_slope = gamma_exchange_asymptotic_diagnostics(1.1, params, heterogeneity)[
            "late_alpha_decay_per_renewal"
        ]

        selection = late_mechanism_selection(
            wave_number=1.1,
            params=params,
            earlier_renewal_count=float(renewal[0]),
            earlier_ngp=float(alpha[0]),
            later_renewal_count=float(renewal[1]),
            later_ngp=float(alpha[1]),
            observed_alpha_decay_per_renewal=observed_slope,
        )

        self.assertEqual(selection["best_model"], "finite_exchange")
        self.assertEqual(selection["finite_exchange"]["passes"], 1.0)
        self.assertEqual(selection["poisson"]["passes"], 0.0)
        self.assertEqual(selection["static_gamma"]["passes"], 0.0)
        self.assertAlmostEqual(selection["finite_exchange"]["inferred_exchange_ratio"], 25.0, delta=0.15)

    def test_late_mechanism_selection_identifies_static_gamma_plateau(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        shape = 0.4
        times = np.array([10000.0, 30000.0])
        renewal = delayed_poisson_mean(times, params)
        alpha = static_gamma_ngp_1d(times, params, shape)
        decay = static_gamma_normalized_alpha_decay(1.1, np.array([times[1]]), params, shape)[0]
        observed_slope = -math.log(decay) / renewal[1]

        selection = late_mechanism_selection(
            wave_number=1.1,
            params=params,
            earlier_renewal_count=float(renewal[0]),
            earlier_ngp=float(alpha[0]),
            later_renewal_count=float(renewal[1]),
            later_ngp=float(alpha[1]),
            observed_alpha_decay_per_renewal=float(observed_slope),
        )

        self.assertEqual(selection["best_model"], "static_gamma")
        self.assertEqual(selection["static_gamma"]["passes"], 1.0)
        self.assertEqual(selection["poisson"]["passes"], 0.0)
        self.assertEqual(selection["finite_exchange"]["passes"], 0.0)
        self.assertAlmostEqual(selection["static_gamma"]["inferred_static_shape"], shape, delta=1e-3)

    def test_late_mechanism_selection_identifies_minimal_poisson_renewal(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        times = np.array([10000.0, 30000.0])
        renewal = delayed_poisson_mean(times, params)
        alpha = ngp_1d(times, params)
        gamma = 1.0 - math.exp(-0.5 * 1.1**2 * params.jump_variance)

        selection = late_mechanism_selection(
            wave_number=1.1,
            params=params,
            earlier_renewal_count=float(renewal[0]),
            earlier_ngp=float(alpha[0]),
            later_renewal_count=float(renewal[1]),
            later_ngp=float(alpha[1]),
            observed_alpha_decay_per_renewal=gamma,
        )

        self.assertEqual(selection["best_model"], "poisson")
        self.assertEqual(selection["poisson"]["passes"], 1.0)
        self.assertEqual(selection["static_gamma"]["passes"], 0.0)
        self.assertEqual(selection["finite_exchange"]["passes"], 0.0)

    def test_static_gamma_null_has_nonzero_long_time_ngp_plateau(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        shape = 0.4
        times = np.array([30.0, 30000.0])

        count = static_gamma_count_moments(times, params, shape)
        renewal = delayed_poisson_mean(times, params)
        alpha = static_gamma_ngp_1d(times, params, shape)

        np.testing.assert_allclose(count["mean"], renewal, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(count["variance"], renewal + renewal**2 / shape, rtol=1e-12, atol=1e-14)
        self.assertGreater(alpha[-1], 2.45)
        self.assertAlmostEqual(alpha[-1], 1.0 / shape, delta=0.01)

    def test_static_gamma_alpha_decay_per_renewal_vanishes_at_long_times(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        shape = 0.4
        wave_number = 1.1
        times = np.array([300.0, 30000.0, 60000.0])

        decay = static_gamma_normalized_alpha_decay(wave_number, times, params, shape)
        renewal = delayed_poisson_mean(times, params)
        slopes = -np.log(decay) / renewal
        diagnostics = static_gamma_asymptotic_diagnostics(wave_number, params, shape)

        self.assertLess(slopes[-1], slopes[0] / 20.0)
        self.assertLess(slopes[-1], 0.002)
        self.assertEqual(diagnostics["late_alpha_decay_per_renewal"], 0.0)
        self.assertAlmostEqual(diagnostics["late_ngp_plateau"], 1.0 / shape, delta=1e-12)

    def test_static_gamma_null_contrasts_finite_exchange_gaussian_recovery(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        late_time = np.array([30000.0])

        static_alpha = static_gamma_ngp_1d(late_time, params, heterogeneity.shape)[0]
        exchange_alpha = gamma_exchange_ngp_1d(late_time, params, heterogeneity)[0]

        self.assertGreater(static_alpha, 2.45)
        self.assertLess(exchange_alpha, 0.01)
        self.assertGreater(static_alpha / exchange_alpha, 500.0)

    def test_correlated_domain_susceptibility_scales_renewal_component(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.linspace(0.0, 220.0, 900)
        single_particle = renewal_scattering_susceptibility(wave_number, times, params)
        correlated = correlated_domain_susceptibility(
            wave_number,
            times,
            params,
            correlation_size=7.5,
        )

        np.testing.assert_allclose(correlated, 7.5 * single_particle, rtol=1e-12, atol=1e-14)
        self.assertGreater(float(np.max(correlated)), float(np.max(single_particle)))

    def test_renewal_correlation_size_inverts_observed_chi4_peak(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.linspace(0.0, 220.0, 900)
        single_particle = renewal_scattering_susceptibility(wave_number, times, params)
        observed_peak = 12.0 * float(np.max(single_particle))

        inferred = infer_renewal_correlation_size(
            observed_chi4_peak=observed_peak,
            wave_number=wave_number,
            t=times,
            params=params,
        )

        self.assertAlmostEqual(inferred["correlation_size"], 12.0)
        self.assertAlmostEqual(inferred["model_single_particle_peak"], float(np.max(single_particle)))
        self.assertGreater(inferred["peak_time"], 0.0)

    def test_renewal_correlation_size_validates_observed_peak(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        with self.assertRaises(ValueError):
            infer_renewal_correlation_size(
                observed_chi4_peak=0.0,
                wave_number=1.1,
                t=np.linspace(0.0, 10.0, 20),
                params=params,
            )

    def test_spatial_facilitation_domain_maps_persistence_time_to_correlation_volume(self):
        domain_fn = getattr(sys.modules["renewal_cage"], "spatial_facilitation_domain", None)
        if domain_fn is None:
            self.fail("spatial_facilitation_domain is missing")

        domain = domain_fn(
            persistence_time=9.0,
            dimension=3,
            particle_density=0.85,
            facilitation_diffusivity=0.05,
            microscopic_length=1.0,
        )
        expected_length = math.sqrt(1.0**2 + 2.0 * 3.0 * 0.05 * 9.0)
        expected_volume = 4.0 * math.pi * expected_length**3 / 3.0

        self.assertAlmostEqual(domain["dynamic_correlation_length"], expected_length)
        self.assertAlmostEqual(domain["correlation_size"], 0.85 * expected_volume)
        self.assertEqual(domain["front_dynamic_exponent"], 2.0)

    def test_spatial_facilitation_chi4_scan_grows_on_cooling(self):
        scan_fn = getattr(sys.modules["renewal_cage"], "spatial_facilitation_chi4_scan", None)
        if scan_fn is None:
            self.fail("spatial_facilitation_chi4_scan is missing")
        law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.25,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=1.0,
            delay_activation=2.5,
        )

        rows = scan_fn(
            temperatures=np.array([1.0, 0.8, 0.65]),
            law=law,
            wave_number=1.1,
            facilitation_diffusivity=0.04,
            particle_density=0.85,
            time_points=250,
        )

        lengths = [row["dynamic_correlation_length"] for row in rows]
        sizes = [row["correlation_size"] for row in rows]
        peaks = [row["chi4_peak"] for row in rows]
        self.assertTrue(all(later > earlier for earlier, later in zip(lengths, lengths[1:])))
        self.assertTrue(all(later > earlier for earlier, later in zip(sizes, sizes[1:])))
        self.assertTrue(all(later > earlier for earlier, later in zip(peaks, peaks[1:])))
        self.assertGreater(rows[-1]["chi4_peak_growth"], 1.0)

    def test_spatial_facilitation_diffusivity_inversion_recovers_constant_front_law(self):
        persistence_times = np.array([3.0, 7.0, 15.0])
        true_diffusivity = 0.04
        observed_lengths = np.sqrt(1.0 + 2.0 * 3.0 * true_diffusivity * persistence_times)

        inferred = infer_spatial_facilitation_diffusivity(
            persistence_times=persistence_times,
            observed_dynamic_lengths=observed_lengths,
            dimension=3,
            microscopic_length=1.0,
        )
        summary = spatial_facilitation_growth_law_consistency(
            persistence_times=persistence_times,
            observed_dynamic_lengths=observed_lengths,
            observed_diffusive_front_growth=True,
            dimension=3,
            microscopic_length=1.0,
            max_diffusivity_relative_std=1.0e-12,
            min_length_growth=1.5,
        )

        self.assertTrue(all(abs(row["inferred_facilitation_diffusivity"] - true_diffusivity) < 1.0e-12 for row in inferred))
        self.assertLess(summary["facilitation_diffusivity_relative_std"], 1.0e-12)
        self.assertGreater(summary["length_growth"], 1.5)
        self.assertEqual(summary["model_predicts_diffusive_front_growth"], 1.0)
        self.assertEqual(summary["facilitation_growth_law_consistent"], 1.0)
        self.assertEqual(summary["overall_consistent"], 1.0)

    def test_spatial_facilitation_growth_law_rejects_nonconstant_front_diffusivity(self):
        persistence_times = np.array([3.0, 7.0, 15.0])
        true_diffusivity = 0.04
        observed_lengths = np.sqrt(1.0 + 2.0 * 3.0 * true_diffusivity * persistence_times)
        observed_lengths[-1] *= 1.25

        summary = spatial_facilitation_growth_law_consistency(
            persistence_times=persistence_times,
            observed_dynamic_lengths=observed_lengths,
            observed_diffusive_front_growth=True,
            dimension=3,
            microscopic_length=1.0,
            max_diffusivity_relative_std=0.05,
            min_length_growth=1.5,
        )

        self.assertGreater(summary["facilitation_diffusivity_relative_std"], 0.05)
        self.assertEqual(summary["model_predicts_diffusive_front_growth"], 0.0)
        self.assertEqual(summary["facilitation_growth_law_consistent"], 0.0)
        self.assertEqual(summary["overall_consistent"], 0.0)

    def test_activated_barrier_gap_controls_delayed_renewal_product(self):
        barrier = ActivatedBarrierParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            renewal_rate_barrier=2.0,
            delay_onset_barrier=5.0,
            cage_stiffening_barrier=0.2,
            jump_to_cage_barrier=0.25,
        )
        law = activated_barrier_temperature_law(barrier)
        hot = temperature_dependent_params(1.0, law)
        cold_temperature = 0.62
        cold = temperature_dependent_params(cold_temperature, law)
        delta = 1.0 / cold_temperature - 1.0 / barrier.reference_temperature
        expected_ratio = math.exp((barrier.delay_onset_barrier - barrier.renewal_rate_barrier) * delta)

        self.assertAlmostEqual(law.delay_activation - law.rate_activation, 3.0)
        self.assertAlmostEqual(
            cold.renewal_rate * cold.renewal_delay / (hot.renewal_rate * hot.renewal_delay),
            expected_ratio,
        )
        self.assertGreater(cold.renewal_rate * cold.renewal_delay, hot.renewal_rate * hot.renewal_delay)

    def test_temperature_law_encodes_cooling_trends(self):
        law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=3.0,
            cage_stiffening=0.25,
            jump_to_cage_growth=0.35,
        )
        hot = temperature_dependent_params(1.0, law)
        cold = temperature_dependent_params(0.62, law)

        self.assertLess(cold.renewal_rate, hot.renewal_rate)
        self.assertGreater(cold.renewal_delay, hot.renewal_delay)
        self.assertLess(cold.cage_variance, hot.cage_variance)
        self.assertGreater(cold.jump_variance / cold.cage_variance, hot.jump_variance / hot.cage_variance)

    def test_temperature_scan_produces_stokes_einstein_decoupling(self):
        law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=5.0,
            cage_stiffening=0.2,
            jump_to_cage_growth=0.25,
        )
        temperatures = np.array([1.0, 0.85, 0.72, 0.62])
        rows = temperature_scan(temperatures, law, wave_number=1.1)

        diffusion = np.array([row["diffusion_coefficient"] for row in rows])
        tau_alpha = np.array([row["tau_alpha"] for row in rows])
        se_ratio = np.array([row["normalized_stokes_einstein_product"] for row in rows])

        self.assertTrue(np.all(np.diff(diffusion) < 0.0))
        self.assertTrue(np.all(np.diff(tau_alpha) > 0.0))
        self.assertGreater(se_ratio[-1], 2.0)
        self.assertAlmostEqual(se_ratio[0], 1.0)
        first = temperature_dependent_params(float(temperatures[0]), law)
        self.assertAlmostEqual(rows[0]["stokes_einstein_product"], stokes_einstein_product(1.1, first))
        self.assertAlmostEqual(rows[0]["diffusion_coefficient"], long_time_diffusion_coefficient(first))
        self.assertIn("fractional_stokes_einstein_exponent", rows[-1])
        self.assertGreater(rows[-1]["fractional_stokes_einstein_exponent"], 0.0)
        self.assertLess(rows[-1]["fractional_stokes_einstein_exponent"], 1.0)
        self.assertIn("apparent_alpha_activation_energy", rows[-1])
        self.assertIn("local_fragility_index", rows[-1])
        self.assertGreater(rows[-1]["apparent_alpha_activation_energy"], rows[0]["apparent_alpha_activation_energy"])
        self.assertGreater(rows[-1]["local_fragility_index"], rows[0]["local_fragility_index"])

    def test_configurational_entropy_law_extrapolates_to_kauzmann_temperature(self):
        entropy_cls = getattr(sys.modules["renewal_cage"], "ConfigurationalEntropyParams", None)
        entropy_fn = getattr(sys.modules["renewal_cage"], "configurational_entropy", None)
        heat_fn = getattr(sys.modules["renewal_cage"], "excess_heat_capacity", None)
        if entropy_cls is None or entropy_fn is None or heat_fn is None:
            self.fail("thermodynamic entropy closure is missing")

        law = entropy_cls(reference_temperature=1.0, entropy_ref=1.2, kauzmann_temperature=0.45)
        temperatures = np.array([1.0, 0.7, 0.5])
        entropy = entropy_fn(temperatures, law)
        heat_capacity = heat_fn(temperatures, law)

        self.assertTrue(np.all(np.diff(entropy) < 0.0))
        self.assertAlmostEqual(float(entropy_fn(np.array([0.45]), law)[0]), 0.0)
        self.assertTrue(np.all(heat_capacity > 0.0))
        self.assertGreater(1.0 / (temperatures[-1] * entropy[-1]), 1.0 / (temperatures[0] * entropy[0]))

    def test_adam_gibbs_thermodynamic_scan_links_entropy_to_renewal_slowdown(self):
        entropy_cls = getattr(sys.modules["renewal_cage"], "ConfigurationalEntropyParams", None)
        scan_fn = getattr(sys.modules["renewal_cage"], "adam_gibbs_thermodynamic_scan", None)
        if entropy_cls is None or scan_fn is None:
            self.fail("Adam-Gibbs thermodynamic scan is missing")
        law = entropy_cls(reference_temperature=1.0, entropy_ref=1.2, kauzmann_temperature=0.45)
        temperatures = np.array([1.0, 0.8, 0.62, 0.5])

        rows = scan_fn(
            temperatures=temperatures,
            entropy_law=law,
            activation_free_energy=1.6,
            tau_ref=3.0,
            renewal_rate_ref=0.18,
            wave_number=1.1,
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
        )

        entropy = np.array([row["configurational_entropy"] for row in rows])
        tau_ag = np.array([row["adam_gibbs_tau"] for row in rows])
        tau_alpha = np.array([row["tau_alpha"] for row in rows])
        self.assertTrue(np.all(np.diff(entropy) < 0.0))
        self.assertTrue(np.all(np.diff(tau_ag) > 0.0))
        self.assertTrue(np.all(np.diff(tau_alpha) > 0.0))
        self.assertGreater(rows[-1]["thermodynamic_slowdown"], 10.0)
        self.assertGreater(rows[-1]["inverse_entropy_control"], rows[0]["inverse_entropy_control"])
        self.assertGreater(rows[-1]["excess_heat_capacity"], 0.0)

    def test_mct_beta_correlator_has_critical_and_von_schweidler_slopes(self):
        beta = MCTBetaParams(
            plateau=0.72,
            critical_amplitude=0.09,
            von_schweidler_amplitude=0.08,
            critical_exponent=0.31,
            von_schweidler_exponent=0.58,
            beta_time=12.0,
        )
        time = np.geomspace(1.2, 120.0, 160)
        correlator = mct_beta_correlator(time, beta)

        early = time < beta.beta_time
        late = time > beta.beta_time
        early_slope = np.polyfit(
            np.log(time[early] / beta.beta_time),
            np.log(correlator[early] - beta.plateau),
            1,
        )[0]
        late_slope = np.polyfit(
            np.log(time[late] / beta.beta_time),
            np.log(beta.plateau - correlator[late]),
            1,
        )[0]

        self.assertAlmostEqual(early_slope, -beta.critical_exponent, delta=0.01)
        self.assertAlmostEqual(late_slope, beta.von_schweidler_exponent, delta=0.01)
        self.assertLess(np.max(correlator), 1.0)
        self.assertGreater(np.min(correlator), 0.0)

    def test_mct_beta_temperature_scan_links_plateau_window_to_alpha_crossover(self):
        base = MCTBetaParams(
            plateau=0.68,
            critical_amplitude=0.08,
            von_schweidler_amplitude=0.05,
            critical_exponent=0.32,
            von_schweidler_exponent=0.6,
            beta_time=4.0,
        )
        temperatures = np.array([1.0, 0.82, 0.68, 0.58])

        rows = mct_beta_temperature_scan(
            temperatures=temperatures,
            base=base,
            beta_time_activation=2.4,
            plateau_growth=0.14,
            alpha_time_ref=30.0,
            alpha_activation=5.0,
        )

        beta_time = np.array([row["beta_time"] for row in rows])
        plateau = np.array([row["plateau"] for row in rows])
        separation = np.array([row["alpha_beta_separation"] for row in rows])
        self.assertTrue(np.all(np.diff(beta_time) > 0.0))
        self.assertTrue(np.all(np.diff(plateau) > 0.0))
        self.assertTrue(np.all(np.diff(separation) > 0.0))
        self.assertGreater(rows[-1]["von_schweidler_exit_time"], rows[-1]["beta_time"])
        self.assertLess(rows[-1]["von_schweidler_exit_time"], rows[-1]["alpha_time"])

    def test_mct_beta_benchmark_consistency_matches_kob_andersen_window(self):
        beta = MCTBetaParams(
            plateau=0.68,
            critical_amplitude=0.08,
            von_schweidler_amplitude=0.05,
            critical_exponent=0.32,
            von_schweidler_exponent=0.6,
            beta_time=4.0,
        )

        row = mct_beta_benchmark_consistency(
            beta,
            benchmark_id="kob_andersen_1995_beta_window",
            observed_critical_decay=False,
            observed_von_schweidler=True,
            observation_min_time=0.85 * beta.beta_time,
            observation_max_time=500.0 * beta.beta_time,
            alpha_time=80.0 * beta.beta_time,
            required_decades=0.5,
        )

        self.assertLess(row["critical_window_decades"], 0.5)
        self.assertGreater(row["von_schweidler_window_decades"], 0.5)
        self.assertEqual(row["model_predicts_visible_critical_decay"], 0.0)
        self.assertEqual(row["model_predicts_visible_von_schweidler"], 1.0)
        self.assertEqual(row["critical_decay_consistent"], 1.0)
        self.assertEqual(row["von_schweidler_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_mct_exponent_benchmark_consistency_checks_common_lambda_relation(self):
        row = mct_exponent_benchmark_consistency(
            benchmark_id="kob_andersen_1995_mct_exponent_parameter",
            observed_common_exponent_parameter=True,
            critical_exponent=0.32,
            von_schweidler_exponent=0.60,
            max_lambda_relative_mismatch=0.05,
        )

        self.assertEqual(row["model_predicts_common_exponent_parameter"], 1.0)
        self.assertLess(row["lambda_relative_mismatch"], 0.05)
        self.assertAlmostEqual(row["lambda_from_a"], 0.716312910468668, places=12)
        self.assertAlmostEqual(row["lambda_from_b"], 0.7246032624007417, places=12)
        self.assertEqual(row["mct_exponent_parameter_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_cage_localization_diagnostics_quantifies_debye_waller_plateau(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )

        row = cage_localization_diagnostics(
            wave_number=1.1,
            plateau_time=1.0,
            params=params,
        )

        self.assertAlmostEqual(row["debye_waller_plateau"], math.exp(-0.5 * 1.1**2), places=12)
        self.assertLess(row["renewal_msd_fraction"], 0.02)
        self.assertGreater(row["alpha_to_cage_time_ratio"], 50.0)
        self.assertGreater(row["cage_plateau_msd"], 0.95)
        self.assertLess(row["cage_plateau_msd"], 1.05)

    def test_cage_localization_benchmark_consistency_detects_cage_plateau(self):
        row = cage_localization_benchmark_consistency(
            benchmark_id="debye_waller_cage_localization",
            observed_cage_localization=True,
            debye_waller_plateau=0.5460744266397094,
            renewal_msd_fraction=0.0048,
            alpha_to_cage_time_ratio=147.0,
            min_debye_waller_plateau=0.2,
            max_debye_waller_plateau=0.95,
            max_renewal_msd_fraction=0.05,
            min_alpha_to_cage_time_ratio=20.0,
        )

        self.assertEqual(row["model_predicts_cage_localization"], 1.0)
        self.assertEqual(row["debye_waller_consistent"], 1.0)
        self.assertEqual(row["renewal_fraction_consistent"], 1.0)
        self.assertEqual(row["alpha_separation_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_gaussian_recovery_benchmark_consistency_rejects_static_disorder(self):
        row = gaussian_recovery_benchmark_consistency(
            benchmark_id="gaussian_recovery_finite_exchange_vs_static_disorder",
            observed_gaussian_recovery=True,
            finite_exchange_late_ngp=0.0048,
            static_gamma_late_ngp=2.5,
            recovery_threshold=0.05,
        )

        self.assertEqual(row["model_predicts_gaussian_recovery"], 1.0)
        self.assertEqual(row["static_null_predicts_gaussian_recovery"], 0.0)
        self.assertEqual(row["finite_exchange_recovery_consistent"], 1.0)
        self.assertEqual(row["static_null_recovery_consistent"], 0.0)
        self.assertEqual(row["mechanism_selection_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_ngp_peak_benchmark_consistency_detects_cooling_peak_shift(self):
        row = ngp_peak_benchmark_consistency(
            benchmark_id="ngp_peak_shift_on_cooling",
            observed_transient_ngp_peak=True,
            hot_peak_time=11.0,
            cold_peak_time=70.0,
            hot_peak_ngp=0.12,
            cold_peak_ngp=0.28,
            late_ngp=0.0048,
            min_peak_time_growth=2.0,
            min_peak_height=0.05,
            min_peak_height_growth=1.2,
            max_late_ngp=0.05,
        )

        self.assertEqual(row["model_predicts_transient_ngp_peak"], 1.0)
        self.assertEqual(row["peak_time_growth_consistent"], 1.0)
        self.assertEqual(row["peak_height_consistent"], 1.0)
        self.assertEqual(row["peak_height_growth_consistent"], 1.0)
        self.assertEqual(row["late_recovery_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["peak_time_growth"], row["min_peak_time_growth"])
        self.assertGreater(row["peak_height_growth"], row["min_peak_height_growth"])

    def test_stokes_einstein_benchmark_consistency_detects_fractional_decoupling(self):
        row = stokes_einstein_benchmark_consistency(
            benchmark_id="stokes_einstein_fractional_decoupling",
            observed_stokes_einstein_violation=True,
            hot_se_product=1.0,
            cold_se_product=2.03,
            cold_fractional_exponent=0.568,
            min_product_growth=1.5,
            max_fractional_exponent=0.9,
        )

        self.assertEqual(row["model_predicts_stokes_einstein_violation"], 1.0)
        self.assertEqual(row["se_product_growth_consistent"], 1.0)
        self.assertEqual(row["fractional_exponent_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["se_product_growth"], 2.0)
        self.assertLess(row["cold_fractional_exponent"], 1.0)

    def test_dynamic_heterogeneity_benchmark_consistency_detects_chi4_growth(self):
        row = dynamic_heterogeneity_benchmark_consistency(
            benchmark_id="dynamic_heterogeneity_chi4_growth",
            observed_dynamic_heterogeneity_growth=True,
            length_growth=3.09,
            correlation_size_growth=29.5,
            chi4_peak_growth=34.7,
            min_length_growth=1.5,
            min_correlation_size_growth=2.0,
            min_chi4_peak_growth=2.0,
        )

        self.assertEqual(row["model_predicts_dynamic_heterogeneity_growth"], 1.0)
        self.assertEqual(row["length_growth_consistent"], 1.0)
        self.assertEqual(row["correlation_size_growth_consistent"], 1.0)
        self.assertEqual(row["chi4_peak_growth_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_alpha_tts_benchmark_consistency_detects_shape_breakdown(self):
        row = alpha_tts_benchmark_consistency(
            benchmark_id="alpha_tts_breakdown_shape_residual",
            observed_tts_breakdown=True,
            cold_shape_residual=0.611,
            alpha_shape_control_growth=6.44,
            residual_threshold=0.25,
            min_control_growth=2.0,
        )

        self.assertEqual(row["model_predicts_tts_breakdown"], 1.0)
        self.assertEqual(row["tts_residual_consistent"], 1.0)
        self.assertEqual(row["tts_control_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["cold_shape_residual"], row["residual_threshold"])
        self.assertGreater(row["alpha_shape_control_growth"], row["min_control_growth"])

    def test_kww_alpha_fit_recovers_stretched_exponent_from_window(self):
        time = np.logspace(-1.0, 2.0, 300)
        tau = 7.5
        beta = 0.62
        decay = np.exp(-((time / tau) ** beta))

        fit = kww_alpha_fit(
            time,
            decay,
            min_decay=0.12,
            max_decay=0.88,
        )

        self.assertAlmostEqual(fit["kww_beta"], beta, places=12)
        self.assertAlmostEqual(fit["kww_tau"], tau, places=12)
        self.assertLess(fit["rms_log_residual"], 1.0e-12)
        self.assertGreaterEqual(fit["points_used"], 50)

    def test_stretched_alpha_benchmark_consistency_detects_cooling_stretching(self):
        row = stretched_alpha_benchmark_consistency(
            benchmark_id="kww_alpha_stretching_on_cooling",
            observed_stretched_alpha=True,
            hot_kww_beta=0.92,
            cold_kww_beta=0.62,
            min_beta_drop=0.15,
            max_cold_beta=0.8,
            max_fit_residual=0.04,
            cold_fit_residual=0.012,
        )

        self.assertEqual(row["model_predicts_stretched_alpha"], 1.0)
        self.assertEqual(row["beta_drop_consistent"], 1.0)
        self.assertEqual(row["cold_beta_consistent"], 1.0)
        self.assertEqual(row["fit_quality_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertAlmostEqual(row["kww_beta_drop"], 0.30)

    def test_persistence_exchange_benchmark_consistency_checks_inversion_protocol(self):
        row = persistence_exchange_benchmark_consistency(
            benchmark_id="persistence_exchange_transport_inversion",
            observed_persistence_exchange_decoupling=True,
            inferred_persistence_exchange_ratio=9.0,
            late_ngp_log_residual=0.0,
            invalid_poisson_alpha_rejected=True,
            min_persistence_exchange_ratio=2.0,
            max_late_ngp_abs_log_residual=0.1,
        )

        self.assertEqual(row["model_predicts_persistence_exchange_decoupling"], 1.0)
        self.assertEqual(row["persistence_exchange_ratio_consistent"], 1.0)
        self.assertEqual(row["persistence_exchange_late_ngp_consistent"], 1.0)
        self.assertEqual(row["persistence_exchange_rejection_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["inferred_persistence_exchange_ratio"], row["min_persistence_exchange_ratio"])
        self.assertLess(abs(row["late_ngp_log_residual_benchmark"]), row["max_late_ngp_abs_log_residual"])

    def test_joint_inversion_benchmark_consistency_checks_multik_chi4_and_rejection(self):
        row = joint_inversion_benchmark_consistency(
            benchmark_id="joint_persistence_exchange_multik_chi4_protocol",
            observed_joint_inversion_closure=True,
            inferred_persistence_exchange_ratio=8.0,
            stokes_einstein_growth_over_poisson=3.56,
            max_multik_tau_alpha_abs_log_residual=0.0,
            late_ngp_log_residual=0.0,
            chi4_peak_growth_over_poisson=2.58,
            rejected_mismatch_abs_log_residual=0.223,
            min_persistence_exchange_ratio=2.0,
            min_stokes_einstein_growth=2.0,
            max_multik_abs_log_residual=0.02,
            max_late_ngp_abs_log_residual=0.02,
            min_chi4_peak_growth=1.5,
            min_rejected_mismatch_abs_log_residual=0.1,
        )

        self.assertEqual(row["model_predicts_joint_inversion_closure"], 1.0)
        self.assertEqual(row["joint_ratio_consistent"], 1.0)
        self.assertEqual(row["joint_se_consistent"], 1.0)
        self.assertEqual(row["joint_multik_consistent"], 1.0)
        self.assertEqual(row["joint_late_ngp_consistent"], 1.0)
        self.assertEqual(row["joint_chi4_consistent"], 1.0)
        self.assertEqual(row["joint_mismatch_rejected"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_literature_inversion_readiness_separates_qualitative_from_quantitative_data(self):
        row = literature_inversion_readiness(
            benchmark_id="kob_andersen_van_hove_1995",
            benchmark_source="kob1995vanhove",
            required_observables=["time_grid", "van_hove_tail", "ngp", "diffusion"],
            available_observables=["time_grid", "van_hove_tail", "ngp"],
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
            next_action="digitize curves or rerun public simulation",
        )

        self.assertAlmostEqual(row["observable_coverage_fraction"], 0.75)
        self.assertEqual(row["missing_observables"], "diffusion")
        self.assertEqual(row["qualitative_comparison_ready"], 1.0)
        self.assertEqual(row["quantitative_inversion_ready"], 0.0)
        self.assertEqual(row["uncertainty_weighted_ready"], 0.0)

    def test_real_benchmark_assimilation_gate_marks_uncertainty_weighted_inversion_ready(self):
        row = real_benchmark_assimilation_gate(
            benchmark_id="public_ka_alpha_vanhove_release",
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
            has_uncertainty_estimates=True,
            model_scope="dynamical_signature",
        )

        self.assertEqual(row["assimilation_stage"], "uncertainty_weighted_inversion")
        self.assertEqual(row["structural_inversion_ready"], 1.0)
        self.assertEqual(row["uncertainty_weighted_ready"], 1.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_real_benchmark_assimilation_gate_blocks_structural_data_without_uncertainties(self):
        row = real_benchmark_assimilation_gate(
            benchmark_id="digitized_ka_alpha_vanhove_candidate",
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
        )

        self.assertEqual(row["assimilation_stage"], "structural_digitization_ready")
        self.assertEqual(row["structural_inversion_ready"], 1.0)
        self.assertEqual(row["uncertainty_weighted_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "uncertainty_columns")

    def test_real_benchmark_assimilation_gate_keeps_thermodynamics_as_scope_boundary(self):
        row = real_benchmark_assimilation_gate(
            benchmark_id="kauzmann_entropy_boundary",
            source_key="kauzmann1948nature;adam1965temperature",
            target_protocol="thermodynamic_entropy_closure",
            available_observables=["temperature_grid", "configurational_entropy", "tau_alpha"],
            has_shared_system=True,
            has_machine_readable_curves=True,
            has_uncertainty_estimates=True,
            model_scope="thermodynamic_transition",
        )

        self.assertEqual(row["assimilation_stage"], "scope_boundary_only")
        self.assertEqual(row["structural_inversion_ready"], 0.0)
        self.assertEqual(row["uncertainty_weighted_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "renewal_dynamics_not_thermodynamic_theory")

    def test_cross_observable_prediction_ledger_marks_heldout_predictive_diagnostic(self):
        row = cross_observable_prediction_ledger(
            protocol_id="joint_persistence_exchange_multik_chi4",
            source_key="synthetic_joint_protocol",
            model_scope="dynamical_signature",
            support_level="derived",
            calibration_observables=["diffusion", "anchor_tau_alpha"],
            heldout_predictions=["multi_k_tau_alpha", "late_ngp", "stokes_einstein_product"],
            closure_observables=[],
            failed_predictions=[],
        )

        self.assertEqual(row["prediction_class"], "predictive_diagnostic")
        self.assertEqual(row["calibration_count"], 2.0)
        self.assertEqual(row["heldout_prediction_count"], 3.0)
        self.assertEqual(row["fit_only_overclaim_risk"], 0.0)
        self.assertEqual(row["all_heldout_predictions_pass"], 1.0)

    def test_cross_observable_prediction_ledger_flags_fit_only_overclaim_risk(self):
        row = cross_observable_prediction_ledger(
            protocol_id="single_alpha_fit_only",
            source_key="hypothetical_alpha_fit",
            model_scope="dynamical_signature",
            support_level="derived",
            calibration_observables=["tau_alpha"],
            heldout_predictions=[],
            closure_observables=[],
            failed_predictions=[],
        )

        self.assertEqual(row["prediction_class"], "underconstrained_fit")
        self.assertEqual(row["heldout_prediction_count"], 0.0)
        self.assertEqual(row["fit_only_overclaim_risk"], 1.0)

    def test_cross_observable_prediction_ledger_keeps_closure_boundary_separate(self):
        row = cross_observable_prediction_ledger(
            protocol_id="spatial_chi4_front_closure",
            source_key="lacevic2003fourpoint",
            model_scope="spatial_heterogeneity",
            support_level="effective_closure",
            calibration_observables=["tau_alpha", "diffusion"],
            heldout_predictions=["chi4_peak", "dynamic_length"],
            closure_observables=["front_diffusivity"],
            failed_predictions=[],
        )

        self.assertEqual(row["prediction_class"], "closure_assisted_prediction")
        self.assertEqual(row["closure_observable_count"], 1.0)
        self.assertEqual(row["requires_external_closure"], 1.0)
        self.assertEqual(row["fit_only_overclaim_risk"], 0.0)

    def test_inversion_identifiability_audit_marks_heldout_protocol_identifiable(self):
        row = inversion_identifiability_audit(
            protocol_id="joint_persistence_exchange_multik_chi4",
            source_key="raw_curve_persistence_exchange_protocol",
            model_scope="transport_decoupling",
            fit_observables=["diffusion", "anchor_tau_alpha"],
            inferred_parameters=["exchange_time", "persistence_time"],
            heldout_predictions=["multi_k_tau_alpha", "late_ngp", "stokes_einstein_product"],
            external_closures=[],
            degenerate_parameters=[],
        )

        self.assertEqual(row["identifiability_class"], "identifiable_prediction")
        self.assertEqual(row["rank_margin"], 0.0)
        self.assertEqual(row["heldout_prediction_count"], 3.0)
        self.assertEqual(row["overclaim_risk"], 0.0)

    def test_inversion_identifiability_audit_flags_alpha_only_underidentified_fit(self):
        row = inversion_identifiability_audit(
            protocol_id="single_alpha_fit_only_null",
            source_key="hypothetical_alpha_only_fit",
            model_scope="dynamical_signature",
            fit_observables=["tau_alpha"],
            inferred_parameters=["exchange_time", "persistence_time"],
            heldout_predictions=[],
            external_closures=[],
            degenerate_parameters=[],
        )

        self.assertEqual(row["identifiability_class"], "underidentified_fit")
        self.assertEqual(row["rank_margin"], -1.0)
        self.assertEqual(row["overclaim_risk"], 1.0)

    def test_inversion_identifiability_audit_separates_closure_and_thermodynamic_boundaries(self):
        spatial = inversion_identifiability_audit(
            protocol_id="spatial_chi4_front_closure",
            source_key="lacevic2003fourpoint",
            model_scope="spatial_heterogeneity",
            fit_observables=["tau_alpha", "diffusion", "chi4_peak"],
            inferred_parameters=["correlation_length", "front_diffusivity"],
            heldout_predictions=["dynamic_length"],
            external_closures=["front_diffusivity_law"],
            degenerate_parameters=[],
        )
        thermo = inversion_identifiability_audit(
            protocol_id="thermodynamic_entropy_boundary",
            source_key="kauzmann1948nature",
            model_scope="thermodynamic_transition",
            fit_observables=["configurational_entropy", "temperature_grid"],
            inferred_parameters=["kauzmann_temperature", "entropy_slope"],
            heldout_predictions=["heat_capacity_anomaly"],
            external_closures=["entropy_law"],
            degenerate_parameters=[],
        )

        self.assertEqual(spatial["identifiability_class"], "conditionally_identifiable")
        self.assertEqual(spatial["requires_external_closure"], 1.0)
        self.assertEqual(spatial["overclaim_risk"], 0.0)
        self.assertEqual(thermo["identifiability_class"], "scope_boundary")
        self.assertEqual(thermo["overclaim_risk"], 0.0)

    def test_frontier_benchmark_horizon_marks_trajectory_reanalysis_candidate(self):
        row = frontier_benchmark_horizon(
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
        )

        self.assertEqual(row["horizon_class"], "trajectory_reanalysis_candidate")
        self.assertEqual(row["can_compute_missing_from_trajectories"], 1.0)
        self.assertIn("self_intermediate_scattering", row["computable_missing_observables"])
        self.assertEqual(row["overclaim_risk"], 0.0)

    def test_frontier_benchmark_horizon_marks_transport_heterogeneity_candidate(self):
        row = frontier_benchmark_horizon(
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
        )

        self.assertEqual(row["horizon_class"], "transport_heterogeneity_candidate")
        self.assertEqual(row["primary_blocker"], "late_ngp")
        self.assertGreater(row["frontier_priority_score"], 0.5)

    def test_frontier_benchmark_horizon_marks_translation_rotation_candidate_and_scope_boundary(self):
        rotational = frontier_benchmark_horizon(
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
        )
        thermo = frontier_benchmark_horizon(
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
        )

        self.assertEqual(rotational["horizon_class"], "structural_inversion_candidate")
        self.assertEqual(rotational["primary_blocker"], "uncertainty_estimates")
        self.assertEqual(rotational["model_extension_required"], 0.0)
        self.assertEqual(thermo["horizon_class"], "scope_boundary")
        self.assertEqual(thermo["overclaim_risk"], 0.0)

    def test_sota_claim_alignment_scores_supported_dynamic_claim(self):
        row = sota_claim_alignment(
            claim_id="hedges_persistence_exchange_decoupling",
            source_key="hedges2007persistence",
            phenomenon="persistence_exchange_decoupling",
            claim_type="dynamical_signature",
            observed_claim="persistence and exchange times decouple on cooling",
            model_diagnostic="raw_curve_persistence_exchange_protocol",
            model_support_level="derived",
            data_readiness="qualitative",
            primary_blocker="machine_readable_curves",
        )

        self.assertEqual(row["claim_alignment"], "supported")
        self.assertEqual(row["model_overclaims_source"], 0.0)
        self.assertEqual(row["requires_external_closure"], 0.0)
        self.assertEqual(row["quantitative_fit_ready"], 0.0)

    def test_sota_claim_alignment_marks_thermodynamic_boundary_as_not_derived(self):
        row = sota_claim_alignment(
            claim_id="kauzmann_entropy_transition",
            source_key="kauzmann1948nature",
            phenomenon="thermodynamic_glass_transition",
            claim_type="thermodynamic_transition",
            observed_claim="configurational entropy extrapolates toward an ideal-glass limit",
            model_diagnostic="adam_gibbs_entropy_closure",
            model_support_level="closure_only",
            data_readiness="qualitative",
            primary_blocker="thermodynamic_input_law",
        )

        self.assertEqual(row["claim_alignment"], "scope_boundary")
        self.assertEqual(row["model_overclaims_source"], 0.0)
        self.assertEqual(row["requires_external_closure"], 1.0)
        self.assertEqual(row["quantitative_fit_ready"], 0.0)

    def test_sota_claim_alignment_rejects_overclaimed_thermodynamic_derivation(self):
        with self.assertRaises(ValueError):
            sota_claim_alignment(
                claim_id="bad_thermodynamic_claim",
                source_key="kauzmann1948nature",
                phenomenon="thermodynamic_glass_transition",
                claim_type="thermodynamic_transition",
                observed_claim="ideal glass thermodynamics",
                model_diagnostic="renewal_dynamics",
                model_support_level="derived",
                data_readiness="qualitative",
                primary_blocker="none",
            )

    def test_sota_signed_constraint_audit_accepts_dynamic_signatures_without_forbidden_claims(self):
        row = sota_signed_constraint_audit(
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
        )

        self.assertEqual(row["signed_constraint_class"], "sota_consistent")
        self.assertEqual(row["missing_expected_signatures"], "none")
        self.assertEqual(row["forbidden_claims_made"], "none")
        self.assertEqual(row["all_required_signatures_pass"], 1.0)
        self.assertEqual(row["publishable_alignment"], 1.0)

    def test_sota_signed_constraint_audit_keeps_spatial_and_thermodynamic_boundaries(self):
        spatial = sota_signed_constraint_audit(
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
        )
        thermodynamic = sota_signed_constraint_audit(
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
        )
        overclaim = sota_signed_constraint_audit(
            constraint_id="bad_thermodynamic_overclaim",
            source_key="kauzmann1948nature",
            model_scope="thermodynamic_transition",
            source_observation="entropy anomaly",
            expected_signatures=["entropy_closure_required"],
            passed_signatures=["entropy_closure_required"],
            forbidden_claims=["ideal_glass_transition_derived"],
            made_claims=["ideal_glass_transition_derived"],
            support_level="closure_only",
            quantitative_fit_ready=False,
        )

        self.assertEqual(spatial["signed_constraint_class"], "closure_assisted_consistent")
        self.assertEqual(spatial["requires_external_closure"], 1.0)
        self.assertEqual(spatial["publishable_alignment"], 1.0)
        self.assertEqual(thermodynamic["signed_constraint_class"], "scope_boundary_consistent")
        self.assertEqual(thermodynamic["requires_external_closure"], 1.0)
        self.assertEqual(thermodynamic["publishable_alignment"], 1.0)
        self.assertEqual(overclaim["signed_constraint_class"], "overclaimed_boundary")
        self.assertEqual(overclaim["publishable_alignment"], 0.0)

    def test_observable_falsification_matrix_marks_diagnostic_blockers(self):
        rows = observable_falsification_matrix(
            benchmark_id="kob_andersen_combined_1995",
            benchmark_source="kob1995vanhove;kob1995intermediate",
            available_observables=[
                "time_grid",
                "self_intermediate_scattering",
                "tau_alpha",
                "wave_numbers",
                "van_hove_tail",
                "ngp",
            ],
            diagnostic_requirements={
                "multi_k_alpha_shape": [
                    "time_grid",
                    "self_intermediate_scattering",
                    "tau_alpha",
                    "wave_numbers",
                ],
                "van_hove_gaussian_recovery": ["time_grid", "van_hove_tail", "ngp", "diffusion"],
                "joint_persistence_exchange_chi4": [
                    "diffusion",
                    "tau_alpha",
                    "persistence_time",
                    "exchange_time",
                    "late_ngp",
                    "chi4_peak",
                ],
            },
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
        )

        by_diagnostic = {row["diagnostic_id"]: row for row in rows}
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["structural_falsification_ready"], 1.0)
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["quantitative_falsification_ready"], 0.0)
        self.assertEqual(by_diagnostic["van_hove_gaussian_recovery"]["missing_observables"], "diffusion")
        self.assertEqual(by_diagnostic["van_hove_gaussian_recovery"]["primary_blocker"], "diffusion")
        self.assertEqual(
            by_diagnostic["joint_persistence_exchange_chi4"]["missing_observables"],
            "diffusion;persistence_time;exchange_time;late_ngp;chi4_peak",
        )
        self.assertEqual(by_diagnostic["joint_persistence_exchange_chi4"]["structural_falsification_ready"], 0.0)

    def test_benchmark_fusion_readiness_requires_shared_system_and_grid(self):
        row = benchmark_fusion_readiness(
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
        )

        self.assertEqual(row["missing_observables"], "none")
        self.assertEqual(row["shared_system_consistent"], 1.0)
        self.assertEqual(row["shared_temperature_grid_consistent"], 1.0)
        self.assertEqual(row["shared_ensemble_consistent"], 1.0)
        self.assertEqual(row["structural_fusion_ready"], 1.0)
        self.assertEqual(row["quantitative_fusion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "machine_readable_data")

    def test_benchmark_fusion_readiness_rejects_cross_ensemble_splicing(self):
        row = benchmark_fusion_readiness(
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
        )

        self.assertEqual(row["missing_observables"], "none")
        self.assertEqual(row["shared_system_consistent"], 1.0)
        self.assertEqual(row["shared_temperature_grid_consistent"], 0.0)
        self.assertEqual(row["shared_ensemble_consistent"], 0.0)
        self.assertEqual(row["structural_fusion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "temperature_grid_mismatch")

    def test_raw_curve_ingestion_contract_marks_missing_uncertainty_columns(self):
        rows = raw_curve_ingestion_contract(
            benchmark_id="kob_andersen_i_ii_dynamic_closure",
            observable_requirements={
                "self_intermediate_scattering": {
                    "required_columns": ["temperature", "wave_number", "time", "F_s"],
                    "uncertainty_columns": ["sigma_F_s"],
                    "target_diagnostic": "multi_k_alpha_shape",
                },
                "van_hove_ngp": {
                    "required_columns": ["temperature", "time", "radius", "G_s", "alpha2", "diffusion"],
                    "uncertainty_columns": ["sigma_G_s", "sigma_alpha2", "sigma_diffusion"],
                    "target_diagnostic": "van_hove_gaussian_recovery",
                },
            },
            available_columns_by_observable={
                "self_intermediate_scattering": ["temperature", "wave_number", "time", "F_s"],
                "van_hove_ngp": [
                    "temperature",
                    "time",
                    "radius",
                    "G_s",
                    "alpha2",
                    "diffusion",
                    "sigma_G_s",
                    "sigma_alpha2",
                    "sigma_diffusion",
                ],
            },
            machine_readable=True,
            shared_temperature_grid=True,
            shared_time_units=True,
        )

        by_observable = {row["observable_id"]: row for row in rows}
        self.assertEqual(by_observable["self_intermediate_scattering"]["structural_ingestion_ready"], 1.0)
        self.assertEqual(by_observable["self_intermediate_scattering"]["uncertainty_ingestion_ready"], 0.0)
        self.assertEqual(by_observable["self_intermediate_scattering"]["missing_columns"], "none")
        self.assertEqual(by_observable["self_intermediate_scattering"]["missing_uncertainty_columns"], "sigma_F_s")
        self.assertEqual(by_observable["self_intermediate_scattering"]["primary_blocker"], "sigma_F_s")
        self.assertEqual(by_observable["van_hove_ngp"]["uncertainty_ingestion_ready"], 1.0)
        self.assertEqual(by_observable["van_hove_ngp"]["primary_blocker"], "none")

    def test_raw_curve_ingestion_contract_rejects_unshared_units_for_fused_input(self):
        rows = raw_curve_ingestion_contract(
            benchmark_id="kob_andersen_i_ii_dynamic_closure",
            observable_requirements={
                "self_intermediate_scattering": {
                    "required_columns": ["temperature", "wave_number", "time", "F_s"],
                    "uncertainty_columns": [],
                    "target_diagnostic": "multi_k_alpha_shape",
                }
            },
            available_columns_by_observable={
                "self_intermediate_scattering": ["temperature", "wave_number", "time", "F_s"],
            },
            machine_readable=True,
            shared_temperature_grid=False,
            shared_time_units=True,
        )

        row = rows[0]
        self.assertEqual(row["structural_ingestion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "temperature_grid_mismatch")

    def test_raw_curve_diagnostic_readiness_aggregates_contract_blockers(self):
        contract_rows = raw_curve_ingestion_contract(
            benchmark_id="kob_andersen_i_ii_dynamic_closure",
            observable_requirements={
                "self_intermediate_scattering": {
                    "required_columns": ["temperature", "wave_number", "time", "F_s"],
                    "uncertainty_columns": ["sigma_F_s"],
                    "target_diagnostic": "multi_k_alpha_shape",
                },
                "van_hove_ngp": {
                    "required_columns": ["temperature", "time", "radius", "G_s", "alpha2", "diffusion"],
                    "uncertainty_columns": ["sigma_G_s"],
                    "target_diagnostic": "van_hove_gaussian_recovery",
                },
            },
            available_columns_by_observable={
                "self_intermediate_scattering": ["temperature", "wave_number", "time", "F_s"],
                "van_hove_ngp": ["temperature", "time", "radius", "G_s", "alpha2", "diffusion", "sigma_G_s"],
            },
            machine_readable=True,
            shared_temperature_grid=True,
            shared_time_units=True,
        )
        rows = raw_curve_diagnostic_readiness(
            benchmark_id="kob_andersen_i_ii_dynamic_closure",
            contract_rows=contract_rows,
            diagnostic_observables={
                "multi_k_alpha_shape": ["self_intermediate_scattering"],
                "van_hove_gaussian_recovery": ["van_hove_ngp"],
                "combined_alpha_vanhove_closure": ["self_intermediate_scattering", "van_hove_ngp"],
            },
        )

        by_diagnostic = {row["diagnostic_id"]: row for row in rows}
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["structural_diagnostic_ready"], 1.0)
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["uncertainty_diagnostic_ready"], 0.0)
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["primary_blocker"], "sigma_F_s")
        self.assertEqual(by_diagnostic["van_hove_gaussian_recovery"]["uncertainty_diagnostic_ready"], 1.0)
        self.assertEqual(by_diagnostic["van_hove_gaussian_recovery"]["primary_blocker"], "none")
        self.assertEqual(by_diagnostic["combined_alpha_vanhove_closure"]["structural_diagnostic_ready"], 1.0)
        self.assertEqual(by_diagnostic["combined_alpha_vanhove_closure"]["uncertainty_diagnostic_ready"], 0.0)
        self.assertEqual(by_diagnostic["combined_alpha_vanhove_closure"]["primary_blocker"], "sigma_F_s")

    def test_raw_curve_persistence_exchange_protocol_extracts_observables_and_passes(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=7.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        time_grid = np.geomspace(0.02, 800.0, 1600)
        alpha_curves = {
            wave_number: (
                time_grid,
                persistence_exchange_normalized_alpha_decay(wave_number, time_grid, params),
            )
            for wave_number in wave_numbers
        }
        late_time = 80.0 * params.persistence_mean
        ngp_time = np.geomspace(0.1, 1200.0, 1400)
        ngp_curve = (ngp_time, persistence_exchange_ngp_1d(ngp_time, params))
        chi4_time = np.geomspace(0.02, 400.0, 900)
        chi4_curve = (
            chi4_time,
            persistence_exchange_scattering_susceptibility(1.1, chi4_time, params),
        )

        row = raw_curve_persistence_exchange_protocol(
            benchmark_id="synthetic_raw_curve_closure",
            anchor_wave_number=1.1,
            alpha_curves_by_k=alpha_curves,
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            ngp_curve=ngp_curve,
            chi4_curve=chi4_curve,
            tau_alpha_relative_error_by_k={wave_number: 0.03 for wave_number in wave_numbers},
            late_ngp_relative_error=0.05,
            chi4_peak_relative_error=0.05,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=3.0,
        )

        self.assertEqual(row["benchmark_id"], "synthetic_raw_curve_closure")
        self.assertEqual(row["raw_curve_protocol_passes"], 1.0)
        self.assertAlmostEqual(row["persistence_exchange_ratio"], 7.0, delta=0.1)
        self.assertGreater(row["stokes_einstein_growth_over_poisson"], 2.0)
        self.assertLess(row["max_multik_tau_alpha_z"], 1.0)
        self.assertLess(row["late_ngp_z"], 1.0)
        self.assertLess(row["chi4_peak_z"], 1.0)

    def test_raw_curve_persistence_exchange_protocol_rejects_late_ngp_mismatch(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=7.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        time_grid = np.geomspace(0.02, 800.0, 1600)
        alpha_curves = {
            wave_number: (
                time_grid,
                persistence_exchange_normalized_alpha_decay(wave_number, time_grid, params),
            )
            for wave_number in wave_numbers
        }
        late_time = 80.0 * params.persistence_mean
        ngp_time = np.geomspace(0.1, 1200.0, 1400)
        corrupted_ngp = 1.8 * persistence_exchange_ngp_1d(ngp_time, params)
        chi4_time = np.geomspace(0.02, 400.0, 900)
        chi4_curve = (
            chi4_time,
            persistence_exchange_scattering_susceptibility(1.1, chi4_time, params),
        )

        row = raw_curve_persistence_exchange_protocol(
            benchmark_id="synthetic_raw_curve_late_ngp_mismatch",
            anchor_wave_number=1.1,
            alpha_curves_by_k=alpha_curves,
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            ngp_curve=(ngp_time, corrupted_ngp),
            chi4_curve=chi4_curve,
            tau_alpha_relative_error_by_k={wave_number: 0.03 for wave_number in wave_numbers},
            late_ngp_relative_error=0.05,
            chi4_peak_relative_error=0.05,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=3.0,
        )

        self.assertEqual(row["late_ngp_z_consistent"], 0.0)
        self.assertEqual(row["raw_curve_protocol_passes"], 0.0)
        self.assertGreater(row["late_ngp_z"], 3.0)

    def test_trajectory_observable_protocol_extracts_msd_ngp_fs_and_chi4(self):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array(
            [
                [[0.0], [0.0], [0.0], [0.0]],
                [[0.0], [0.0], [2.0], [2.0]],
                [[0.0], [0.0], [2.0], [2.0]],
            ]
        )

        rows = trajectory_observable_protocol(
            positions=positions,
            times=times,
            lag_indices=[1, 2],
            wave_numbers=[math.pi],
            overlap_radius=0.5,
        )

        by_lag = {int(row["lag_index"]): row for row in rows}
        lag1 = by_lag[1]
        self.assertAlmostEqual(lag1["lag_time"], 1.0)
        self.assertAlmostEqual(lag1["msd"], 1.0)
        self.assertAlmostEqual(lag1["ngp"], 1.0 / 3.0)
        self.assertAlmostEqual(lag1["self_intermediate_scattering"], 1.0)
        self.assertAlmostEqual(lag1["overlap_mean"], 0.75)
        self.assertAlmostEqual(lag1["chi4_overlap"], 0.25)
        self.assertEqual(lag1["structural_observable_set"], "msd;ngp;self_intermediate_scattering;overlap_chi4")

        lag2 = by_lag[2]
        self.assertAlmostEqual(lag2["lag_time"], 2.0)
        self.assertAlmostEqual(lag2["msd"], 2.0)
        self.assertAlmostEqual(lag2["ngp"], -1.0 / 3.0)
        self.assertAlmostEqual(lag2["overlap_mean"], 0.5)

    def test_trajectory_observable_protocol_validates_inputs(self):
        with self.assertRaises(ValueError):
            trajectory_observable_protocol(
                positions=np.zeros((2, 3)),
                times=np.array([0.0, 1.0]),
                lag_indices=[1],
                wave_numbers=[1.0],
                overlap_radius=0.5,
            )

    def test_trajectory_observable_uncertainty_protocol_adds_jackknife_sigmas(self):
        times = np.arange(6.0)
        increments = np.array(
            [
                [[0.0], [0.0], [2.0], [2.0]],
                [[0.0], [2.0], [0.0], [2.0]],
                [[0.0], [0.0], [0.0], [2.0]],
                [[2.0], [0.0], [2.0], [0.0]],
                [[0.0], [2.0], [0.0], [0.0]],
            ]
        )
        positions = np.concatenate([np.zeros((1, 4, 1)), np.cumsum(increments, axis=0)])

        full = trajectory_observable_protocol(
            positions=positions,
            times=times,
            lag_indices=[1],
            wave_numbers=[1.1],
            overlap_radius=0.5,
        )[0]
        uncertain = trajectory_observable_uncertainty_protocol(
            positions=positions,
            times=times,
            lag_indices=[1],
            wave_numbers=[1.1],
            overlap_radius=0.5,
            block_count=3,
        )[0]

        self.assertAlmostEqual(uncertain["msd"], full["msd"])
        self.assertAlmostEqual(uncertain["ngp"], full["ngp"])
        self.assertAlmostEqual(
            uncertain["self_intermediate_scattering"],
            full["self_intermediate_scattering"],
        )
        self.assertEqual(uncertain["uncertainty_method"], "time_origin_block_jackknife")
        self.assertEqual(uncertain["uncertainty_estimates"], 1.0)
        self.assertEqual(uncertain["primary_blocker"], "none")
        self.assertGreater(uncertain["sigma_msd"], 0.0)
        self.assertGreater(uncertain["sigma_ngp"], 0.0)
        self.assertGreater(uncertain["sigma_self_intermediate_scattering"], 0.0)
        self.assertGreaterEqual(uncertain["sigma_chi4_overlap"], 0.0)

    def test_trajectory_observable_uncertainty_protocol_requires_multiple_blocks(self):
        with self.assertRaises(ValueError):
            trajectory_observable_uncertainty_protocol(
                positions=np.zeros((3, 2, 1)),
                times=np.array([0.0, 1.0, 2.0]),
                lag_indices=[1],
                wave_numbers=[1.0],
                overlap_radius=0.5,
                block_count=1,
            )

    def test_van_hove_tail_benchmark_consistency_detects_transient_tail_and_recovery(self):
        row = van_hove_tail_benchmark_consistency(
            benchmark_id="kob_andersen_van_hove_tail_recovery",
            observed_transient_van_hove_tail=True,
            observed_late_gaussian_recovery=True,
            peak_tail_ratio=2.895,
            late_tail_ratio=0.966,
            peak_ngp=0.126,
            min_peak_tail_ratio=1.5,
            max_late_tail_deviation=0.15,
            min_peak_ngp=0.05,
        )

        self.assertEqual(row["model_predicts_transient_van_hove_tail"], 1.0)
        self.assertEqual(row["model_predicts_tail_gaussian_recovery"], 1.0)
        self.assertEqual(row["van_hove_tail_consistent"], 1.0)
        self.assertEqual(row["tail_recovery_consistent"], 1.0)
        self.assertEqual(row["peak_ngp_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["peak_tail_ratio"], row["min_peak_tail_ratio"])
        self.assertLess(row["late_tail_abs_deviation"], row["max_late_tail_deviation"])

    def test_fragility_benchmark_consistency_checks_super_arrhenius_growth_without_origin_claim(self):
        row = fragility_benchmark_consistency(
            benchmark_id="angell_adam_gibbs_fragility_growth",
            observed_fragility_growth=True,
            observed_adam_gibbs_slowdown=True,
            hot_activation_energy=2.69,
            cold_activation_energy=3.43,
            hot_fragility_index=1.17,
            cold_fragility_index=2.41,
            adam_gibbs_slowdown=1.48e8,
            material_specific_origin_claimed=False,
            min_activation_growth=1.2,
            min_fragility_growth=1.5,
            min_adam_gibbs_slowdown=10.0,
        )

        self.assertEqual(row["model_predicts_fragility_growth"], 1.0)
        self.assertEqual(row["activation_growth_consistent"], 1.0)
        self.assertEqual(row["fragility_index_consistent"], 1.0)
        self.assertEqual(row["adam_gibbs_slowdown_consistent"], 1.0)
        self.assertEqual(row["fragility_scope_boundary_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["activation_energy_growth"], row["min_activation_growth"])
        self.assertGreater(row["fragility_index_growth"], row["min_fragility_growth"])

    def test_thermodynamic_scope_benchmark_consistency_keeps_entropy_closure_boundary(self):
        row = thermodynamic_scope_benchmark_consistency(
            benchmark_id="thermodynamic_transition_scope_boundary",
            observed_heat_capacity_anomaly=True,
            observed_kauzmann_extrapolation=True,
            dynamic_model_derives_entropy=False,
            entropy_closure_supplied=True,
            adam_gibbs_slowdown=1.48e8,
            min_adam_gibbs_slowdown=10.0,
            material_specific_entropy_origin_claimed=False,
        )

        self.assertEqual(row["model_predicts_heat_capacity_anomaly_from_dynamics"], 0.0)
        self.assertEqual(row["model_predicts_kauzmann_transition_from_dynamics"], 0.0)
        self.assertEqual(row["entropy_closure_required"], 1.0)
        self.assertEqual(row["adam_gibbs_slowdown_consistent"], 1.0)
        self.assertEqual(row["thermodynamic_scope_boundary_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_facilitated_exchange_law_grows_exchange_ratio_on_cooling(self):
        law = FacilitatedExchangeLawParams(
            reference_temperature=1.0,
            shape_ref=0.4,
            exchange_renewal_count_ref=10.0,
            shape_broadening_barrier=1.5,
            exchange_slowing_barrier=2.5,
        )

        hot = temperature_dependent_gamma_exchange(1.0, law)
        cold_temperature = 0.62
        cold = temperature_dependent_gamma_exchange(cold_temperature, law)
        delta = 1.0 / cold_temperature - 1.0 / law.reference_temperature
        expected_ratio_growth = math.exp(
            (law.shape_broadening_barrier + law.exchange_slowing_barrier) * delta
        )

        self.assertAlmostEqual(hot.shape, 0.4)
        self.assertAlmostEqual(hot.exchange_renewal_count, 10.0)
        self.assertLess(cold.shape, hot.shape)
        self.assertGreater(cold.exchange_renewal_count, hot.exchange_renewal_count)
        self.assertAlmostEqual(
            (cold.exchange_renewal_count / cold.shape) / (hot.exchange_renewal_count / hot.shape),
            expected_ratio_growth,
            delta=1e-12,
        )

    def test_gamma_exchange_temperature_scan_links_cooling_to_alpha_slowing(self):
        cage_law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=5.0,
            cage_stiffening=0.2,
            jump_to_cage_growth=0.25,
        )
        exchange_law = FacilitatedExchangeLawParams(
            reference_temperature=1.0,
            shape_ref=0.4,
            exchange_renewal_count_ref=10.0,
            shape_broadening_barrier=1.5,
            exchange_slowing_barrier=2.5,
        )
        temperatures = np.array([1.0, 0.78, 0.62])

        rows = gamma_exchange_temperature_scan(
            temperatures,
            cage_law,
            exchange_law,
            wave_number=1.1,
        )

        ratios = np.array([row["heterogeneity_ratio"] for row in rows])
        amplitudes = np.array([row["late_ngp_renewal_amplitude"] for row in rows])
        alpha_renormalization = np.array([row["alpha_rate_renormalization"] for row in rows])

        self.assertTrue(np.all(np.diff(ratios) > 0.0))
        self.assertTrue(np.all(np.diff(amplitudes) > 0.0))
        self.assertTrue(np.all(np.diff(alpha_renormalization) < 0.0))
        self.assertGreater(rows[-1]["late_ngp_renewal_amplitude"], 50.0)
        self.assertLess(rows[-1]["alpha_rate_renormalization"], rows[0]["alpha_rate_renormalization"] / 2.0)

    def test_gamma_exchange_alpha_relaxation_time_solves_finite_exchange_decay(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1

        tau_exchange = gamma_exchange_alpha_relaxation_time(wave_number, params, heterogeneity)
        tau_poisson = alpha_relaxation_time(wave_number, params)
        decay = gamma_exchange_normalized_alpha_decay(
            wave_number,
            np.array([tau_exchange]),
            params,
            heterogeneity,
        )[0]

        self.assertGreater(tau_exchange, tau_poisson)
        self.assertAlmostEqual(decay, math.exp(-1.0), delta=1e-10)

    def test_glass_phenomenon_audit_separates_supported_dynamics_from_thermodynamic_transition(self):
        cage_law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=5.0,
            cage_stiffening=0.2,
            jump_to_cage_growth=0.25,
        )
        exchange_law = FacilitatedExchangeLawParams(
            reference_temperature=1.0,
            shape_ref=0.4,
            exchange_renewal_count_ref=10.0,
            shape_broadening_barrier=1.5,
            exchange_slowing_barrier=2.5,
        )
        audit = glass_phenomenon_audit(
            np.array([1.0, 0.85, 0.72, 0.62]),
            cage_law,
            exchange_law,
            wave_number=1.1,
        )
        rows = audit["rows"]

        self.assertEqual(audit["diffusion_slowdown"], 1.0)
        self.assertEqual(audit["alpha_slowdown"], 1.0)
        self.assertEqual(audit["ngp_peak_shift"], 1.0)
        self.assertEqual(audit["stokes_einstein_violation"], 1.0)
        self.assertEqual(audit["fragility_growth"], 1.0)
        self.assertEqual(audit["heterogeneity_growth"], 1.0)
        self.assertEqual(audit["stretched_alpha_window"], 1.0)
        self.assertEqual(audit["chi4_peak_growth"], 1.0)
        self.assertEqual(audit["gaussian_recovery"], 1.0)
        self.assertEqual(audit["thermodynamic_transition"], 0.0)
        self.assertGreater(audit["supported_dynamic_signatures"], 8.0)
        self.assertGreater(rows[-1]["tau_alpha_exchange"] / rows[0]["tau_alpha_exchange"], 10.0)
        self.assertGreater(rows[-1]["chi4_peak"] / rows[0]["chi4_peak"], 1.0)

    def test_glass_signature_phase_diagram_identifies_barrier_facilitation_window(self):
        base_cage_law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=2.0,
            cage_stiffening=0.2,
            jump_to_cage_growth=0.25,
        )
        base_exchange_law = FacilitatedExchangeLawParams(
            reference_temperature=1.0,
            shape_ref=0.4,
            exchange_renewal_count_ref=10.0,
            shape_broadening_barrier=1.5,
            exchange_slowing_barrier=2.5,
        )

        rows = glass_signature_phase_diagram(
            np.array([1.0, 0.85, 0.72, 0.62]),
            base_cage_law,
            base_exchange_law,
            wave_number=1.1,
            delay_barrier_gaps=[0.0, 1.5, 3.0],
            exchange_barrier_sums=[0.0, 2.0, 4.0],
        )

        self.assertEqual(len(rows), 9)
        weak = next(row for row in rows if row["delay_barrier_gap"] == 0.0 and row["exchange_barrier_sum"] == 0.0)
        strong = next(row for row in rows if row["delay_barrier_gap"] == 3.0 and row["exchange_barrier_sum"] == 4.0)
        same_exchange_stronger_delay = next(
            row for row in rows if row["delay_barrier_gap"] == 3.0 and row["exchange_barrier_sum"] == 0.0
        )
        same_delay_stronger_exchange = next(
            row for row in rows if row["delay_barrier_gap"] == 0.0 and row["exchange_barrier_sum"] == 4.0
        )

        self.assertEqual(weak["complete_dynamic_closure"], 0.0)
        self.assertEqual(strong["complete_dynamic_closure"], 1.0)
        self.assertEqual(strong["thermodynamic_transition"], 0.0)
        self.assertGreater(strong["supported_dynamic_signatures"], weak["supported_dynamic_signatures"])
        self.assertGreater(same_exchange_stronger_delay["cold_se_product_ratio"], weak["cold_se_product_ratio"])
        self.assertGreater(
            same_delay_stronger_exchange["cold_heterogeneity_growth_ratio"],
            weak["cold_heterogeneity_growth_ratio"],
        )

    def test_barrier_amplification_laws_give_closed_cooling_factors(self):
        laws = barrier_amplification_laws(
            hot_temperature=1.0,
            cold_temperature=0.62,
            delay_barrier_gap=3.0,
            exchange_barrier_sum=4.0,
        )
        delta = 1.0 / 0.62 - 1.0

        self.assertAlmostEqual(laws["inverse_temperature_interval"], delta)
        self.assertAlmostEqual(laws["lambda_tau_delay_growth"], math.exp(3.0 * delta))
        self.assertAlmostEqual(laws["heterogeneity_ratio_growth"], math.exp(4.0 * delta))
        self.assertAlmostEqual(laws["combined_slowing_growth"], math.exp(7.0 * delta))

    def test_minimal_barrier_requirements_invert_target_growth_factors(self):
        requirements = minimal_barrier_requirements(
            hot_temperature=1.0,
            cold_temperature=0.62,
            target_lambda_tau_delay_growth=math.exp(3.0 * (1.0 / 0.62 - 1.0)),
            target_heterogeneity_ratio_growth=math.exp(4.0 * (1.0 / 0.62 - 1.0)),
        )

        self.assertAlmostEqual(requirements["required_delay_barrier_gap"], 3.0)
        self.assertAlmostEqual(requirements["required_exchange_barrier_sum"], 4.0)
        self.assertAlmostEqual(requirements["target_combined_growth"], requirements["target_lambda_tau_delay_growth"] * requirements["target_heterogeneity_ratio_growth"])

    def test_minimal_barrier_requirements_validate_temperature_order(self):
        with self.assertRaises(ValueError):
            minimal_barrier_requirements(
                hot_temperature=0.62,
                cold_temperature=1.0,
                target_lambda_tau_delay_growth=2.0,
                target_heterogeneity_ratio_growth=3.0,
            )

    def test_fractional_stokes_einstein_exponents_recover_power_law_slope(self):
        tau_alpha = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
        diffusion = 3.0 * tau_alpha ** (-0.72)
        exponents = fractional_stokes_einstein_exponents(diffusion, tau_alpha)

        np.testing.assert_allclose(exponents, 0.72, rtol=1e-12, atol=1e-12)

    def test_fractional_stokes_einstein_exponents_validate_inputs(self):
        with self.assertRaises(ValueError):
            fractional_stokes_einstein_exponents(np.array([1.0]), np.array([2.0]))
        with self.assertRaises(ValueError):
            fractional_stokes_einstein_exponents(np.array([1.0, -1.0]), np.array([2.0, 3.0]))

    def test_apparent_alpha_activation_energies_recover_arrhenius_barrier(self):
        temperatures = np.array([1.0, 0.9, 0.8, 0.7, 0.62])
        barrier = 4.2
        tau_alpha = 0.3 * np.exp(barrier / temperatures)
        energies = apparent_alpha_activation_energies(temperatures, tau_alpha)

        np.testing.assert_allclose(energies, barrier, rtol=1e-12, atol=1e-12)

    def test_apparent_alpha_activation_energies_validate_inputs(self):
        with self.assertRaises(ValueError):
            apparent_alpha_activation_energies(np.array([1.0]), np.array([2.0]))
        with self.assertRaises(ValueError):
            apparent_alpha_activation_energies(np.array([1.0, 0.0]), np.array([2.0, 3.0]))

    def test_radial_van_hove_distribution_normalizes(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        radius = np.linspace(0.0, 12.0, 4000)
        density = radial_van_hove_3d(radius, time=12.0, params=params, max_count=80)
        integral = np.trapezoid(density, radius)

        self.assertAlmostEqual(integral, 1.0, delta=2e-4)
        self.assertGreater(density[1], 0.0)

    def test_gaussian_radial_baseline_normalizes(self):
        radius = np.linspace(0.0, 12.0, 4000)
        density = gaussian_radial_3d(radius, coordinate_variance=2.5)
        integral = np.trapezoid(density, radius)

        self.assertAlmostEqual(integral, 1.0, delta=2e-4)

    def test_renewal_van_hove_has_heavier_peak_time_tail_than_gaussian_baseline(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        radius = np.linspace(0.0, 18.0, 5000)
        peak_time = 11.30637498610339
        renewal_density = radial_van_hove_3d(radius, time=peak_time, params=params, max_count=120)
        coordinate_variance = moments_1d(np.array([peak_time]), params)["m2"][0]
        gaussian_density = gaussian_radial_3d(radius, coordinate_variance=coordinate_variance)
        tail_mask = radius > 5.0
        renewal_tail = np.trapezoid(renewal_density[tail_mask], radius[tail_mask])
        gaussian_tail = np.trapezoid(gaussian_density[tail_mask], radius[tail_mask])

        self.assertGreater(renewal_tail / gaussian_tail, 1.5)

    def test_dimensionless_peak_prediction_matches_numeric_peak_in_plateau_regime(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        t = np.linspace(0.01, 180.0, 5000)
        alpha = ngp_1d(t, params)
        peak_time = float(t[int(np.argmax(alpha))])
        predicted = dimensionless_peak_prediction(params)

        self.assertAlmostEqual(peak_time, predicted["peak_time"], delta=0.2)
        self.assertAlmostEqual(float(np.max(alpha)), predicted["peak_ngp"], delta=0.005)

    def test_plateau_peak_diagnostics_recover_peak_scale_parameters(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        predicted = dimensionless_peak_prediction(params)
        diagnostics = plateau_peak_diagnostics(
            peak_ngp=predicted["peak_ngp"],
            peak_time=predicted["peak_time"],
            renewal_delay=params.renewal_delay,
        )

        self.assertAlmostEqual(diagnostics["jump_to_cage_variance"], params.jump_variance / params.cage_variance)
        self.assertAlmostEqual(diagnostics["target_renewal_count"], params.cage_variance / params.jump_variance)
        self.assertAlmostEqual(diagnostics["renewal_rate"], params.renewal_rate, delta=1e-12)

    def test_plateau_ngp_branches_invert_same_observed_value(self):
        beta = 0.8
        observed_ngp = 0.05
        branches = plateau_ngp_branches(jump_to_cage_variance=beta, observed_ngp=observed_ngp)

        self.assertLess(branches["early_y"], 1.0)
        self.assertGreater(branches["late_y"], 1.0)
        self.assertAlmostEqual(branches["early_y"] * branches["late_y"], 1.0)
        for branch in ("early_y", "late_y"):
            y = branches[branch]
            reconstructed = beta * y / (1.0 + y) ** 2
            self.assertAlmostEqual(reconstructed, observed_ngp, delta=1e-14)

    def test_plateau_ngp_branches_reject_values_above_peak_bound(self):
        with self.assertRaises(ValueError):
            plateau_ngp_branches(jump_to_cage_variance=0.8, observed_ngp=0.21)

    def test_observable_consistency_diagnostics_compare_peak_and_late_ngp(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        predicted = dimensionless_peak_prediction(params)
        late_time = 1200.0
        late_ngp = float(ngp_1d(np.array([late_time]), params)[0])
        diagnostics = observable_consistency_diagnostics(
            peak_ngp=predicted["peak_ngp"],
            peak_time=predicted["peak_time"],
            renewal_delay=params.renewal_delay,
            late_time=late_time,
            late_ngp=late_ngp,
        )

        self.assertAlmostEqual(diagnostics["jump_to_cage_variance"], params.jump_variance / params.cage_variance)
        self.assertAlmostEqual(diagnostics["peak_renewal_rate"], params.renewal_rate, delta=1e-12)
        self.assertAlmostEqual(diagnostics["late_renewal_rate_exact"], params.renewal_rate, delta=1e-6)
        self.assertAlmostEqual(diagnostics["late_renewal_rate_asymptotic"], params.renewal_rate, delta=2e-3)
        self.assertLess(abs(diagnostics["log_exact_rate_residual"]), 1e-5)
        self.assertLess(abs(diagnostics["log_asymptotic_rate_residual"]), 0.01)

    def test_observable_consistency_diagnostics_validate_inputs(self):
        with self.assertRaises(ValueError):
            observable_consistency_diagnostics(
                peak_ngp=0.2,
                peak_time=10.0,
                renewal_delay=3.0,
                late_time=0.0,
                late_ngp=0.01,
            )

    def test_scattering_transport_inversion_recovers_minimal_parameters(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        plateau = math.exp(-0.5 * wave_number**2 * params.cage_variance)
        diffusion = long_time_diffusion_coefficient(params)
        tau_alpha = alpha_relaxation_time(wave_number, params)
        peak = dimensionless_peak_prediction(params)

        inferred = infer_parameters_from_scattering_transport(
            wave_number=wave_number,
            debye_waller_plateau=plateau,
            diffusion_coefficient=diffusion,
            tau_alpha=tau_alpha,
            renewal_delay=params.renewal_delay,
        )

        self.assertAlmostEqual(inferred["cage_variance"], params.cage_variance)
        self.assertAlmostEqual(inferred["jump_variance"], params.jump_variance, delta=1e-11)
        self.assertAlmostEqual(inferred["renewal_rate"], params.renewal_rate, delta=1e-12)
        self.assertGreater(inferred["existence_margin"], 1.0)
        self.assertAlmostEqual(inferred["reconstructed_debye_waller_plateau"], plateau)
        self.assertAlmostEqual(inferred["reconstructed_diffusion_coefficient"], diffusion)
        self.assertAlmostEqual(inferred["reconstructed_tau_alpha"], tau_alpha, delta=1e-10)
        self.assertAlmostEqual(inferred["predicted_ngp_peak"], peak["peak_ngp"], delta=1e-12)
        self.assertAlmostEqual(inferred["predicted_ngp_peak_time"], peak["peak_time"], delta=1e-10)

    def test_scattering_transport_inversion_rejects_impossible_observables(self):
        with self.assertRaisesRegex(ValueError, "existence"):
            infer_parameters_from_scattering_transport(
                wave_number=1.1,
                debye_waller_plateau=0.55,
                diffusion_coefficient=1e-4,
                tau_alpha=10.0,
                renewal_delay=3.0,
            )

    def test_full_observable_inversion_recovers_renewal_delay_without_external_tau_d(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        plateau = math.exp(-0.5 * wave_number**2 * params.cage_variance)
        diffusion = long_time_diffusion_coefficient(params)
        tau_alpha = alpha_relaxation_time(wave_number, params)
        peak = dimensionless_peak_prediction(params)

        inferred = infer_parameters_from_full_observables(
            wave_number=wave_number,
            debye_waller_plateau=plateau,
            diffusion_coefficient=diffusion,
            tau_alpha=tau_alpha,
            peak_time=peak["peak_time"],
            peak_ngp=peak["peak_ngp"],
        )

        self.assertAlmostEqual(inferred["cage_variance"], params.cage_variance)
        self.assertAlmostEqual(inferred["jump_variance"], params.jump_variance)
        self.assertAlmostEqual(inferred["renewal_rate"], params.renewal_rate)
        self.assertAlmostEqual(inferred["renewal_delay"], params.renewal_delay, delta=1e-12)
        self.assertAlmostEqual(inferred["reconstructed_tau_alpha"], tau_alpha, delta=1e-10)
        self.assertLess(abs(inferred["log_tau_alpha_residual"]), 1e-12)

    def test_full_observable_inversion_rejects_impossible_peak_timing(self):
        with self.assertRaisesRegex(ValueError, "peak timing"):
            infer_parameters_from_full_observables(
                wave_number=1.1,
                debye_waller_plateau=0.55,
                diffusion_coefficient=0.02,
                tau_alpha=10.0,
                peak_time=1.0,
                peak_ngp=0.2,
            )

    def test_delayed_renewal_shape_is_positive_and_matches_integral(self):
        scaled_time = 2.5
        shape = delayed_renewal_shape(scaled_time)
        numeric = np.trapezoid((1.0 - np.exp(-np.linspace(0.0, scaled_time, 2000))) ** 2, np.linspace(0.0, scaled_time, 2000))

        self.assertGreater(shape, 0.0)
        self.assertAlmostEqual(shape, numeric, delta=1e-6)

    def test_generalized_delay_short_time_law_matches_square_delay_model(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        asymptotic = generalized_delay_ngp_short_time(params, delay_exponent=2.0)
        t = np.array([1e-5, 2e-5, 4e-5])
        alpha = ngp_1d(t, params)

        self.assertAlmostEqual(asymptotic["power"], 1.0)
        np.testing.assert_allclose(alpha / t, asymptotic["prefactor"], rtol=1e-3)

    def test_delay_exponent_classification_explains_square_delay_choice(self):
        self.assertEqual(classify_delay_exponent(0.5), "singular_origin")
        self.assertEqual(classify_delay_exponent(1.0), "finite_origin")
        self.assertEqual(classify_delay_exponent(2.0), "regular_zero_origin")

    def test_persistence_exchange_distribution_normalizes_and_starts_unrenewed(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )

        probability = persistence_exchange_count_distribution(np.array([0.0, 4.0, 20.0]), params, max_count=80)

        self.assertAlmostEqual(probability[0, 0], 1.0)
        self.assertTrue(np.allclose(np.sum(probability, axis=1), 1.0, atol=1e-8))
        self.assertGreater(probability[1, 0], 0.55)
        self.assertLess(probability[2, 0], 0.1)

    def test_persistence_exchange_poisson_limit_recovers_count_moments(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=2.0,
            exchange_mean=2.0,
        )
        times = np.array([1.0, 3.0, 7.0])

        moments = persistence_exchange_count_moments(times, params, max_count=80)

        np.testing.assert_allclose(moments["mean"], times / 2.0, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(moments["variance"], times / 2.0, rtol=2e-5, atol=2e-5)

    def test_persistence_exchange_pgf_matches_alpha_decay(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_number = 1.1
        times = np.array([0.5, 5.0, 20.0])
        jump_factor = math.exp(-0.5 * wave_number**2 * params.jump_variance)

        pgf = persistence_exchange_count_pgf(jump_factor, times, params)
        decay = persistence_exchange_normalized_alpha_decay(wave_number, times, params, max_count=400)

        np.testing.assert_allclose(pgf, decay, rtol=2e-5, atol=2e-8)
        np.testing.assert_allclose(persistence_exchange_count_pgf(1.0, times, params), np.ones_like(times))

    def test_persistence_exchange_closed_moments_match_distribution_moments(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        times = np.array([2.0, 8.0, 30.0])

        probability = persistence_exchange_count_distribution(times, params, max_count=200)
        counts = np.arange(probability.shape[1], dtype=float)
        distribution_mean = probability @ counts
        distribution_variance = probability @ (counts**2) - distribution_mean**2
        closed = persistence_exchange_count_moments(times, params)

        np.testing.assert_allclose(closed["mean"], distribution_mean, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(closed["variance"], distribution_variance, rtol=5e-5, atol=5e-6)

    def test_persistence_exchange_decoupling_increases_stokes_einstein_product(self):
        base = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=1.0,
            exchange_mean=1.0,
        )
        decoupled = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=12.0,
            exchange_mean=1.0,
        )
        wave_number = 1.1

        base_tau = persistence_exchange_alpha_relaxation_time(wave_number, base)
        decoupled_tau = persistence_exchange_alpha_relaxation_time(wave_number, decoupled)

        self.assertAlmostEqual(persistence_exchange_diffusion_coefficient(base), persistence_exchange_diffusion_coefficient(decoupled))
        self.assertGreater(decoupled_tau / base_tau, 3.0)
        self.assertGreater(
            persistence_exchange_diffusion_coefficient(decoupled) * decoupled_tau,
            3.0 * persistence_exchange_diffusion_coefficient(base) * base_tau,
        )

    def test_translation_rotation_exchange_detects_rotational_decoupling(self):
        coupled = TranslationRotationExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            translational_persistence_mean=4.0,
            translational_exchange_mean=1.0,
            rotational_persistence_mean=4.0,
            rotational_exchange_mean=1.0,
            rotational_step_correlation=0.62,
        )
        decoupled = TranslationRotationExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            translational_persistence_mean=4.0,
            translational_exchange_mean=1.0,
            rotational_persistence_mean=14.0,
            rotational_exchange_mean=1.0,
            rotational_step_correlation=0.62,
        )

        coupled_row = translation_rotation_decoupling_diagnostic(
            "coupled",
            coupled,
            wave_number=1.1,
        )
        decoupled_row = translation_rotation_decoupling_diagnostic(
            "rotationally_slow",
            decoupled,
            wave_number=1.1,
        )

        self.assertAlmostEqual(coupled_row["translation_rotation_ratio"], 1.0, delta=0.05)
        self.assertGreater(decoupled_row["translation_rotation_ratio"], 2.5)
        self.assertGreater(decoupled_row["rotational_dse_product"], coupled_row["rotational_dse_product"])

    def test_translation_rotation_inversion_recovers_hidden_rotational_clock(self):
        params = TranslationRotationExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            translational_persistence_mean=6.0,
            translational_exchange_mean=1.0,
            rotational_persistence_mean=15.0,
            rotational_exchange_mean=1.0,
            rotational_step_correlation=0.62,
        )
        wave_number = 1.1
        tau_alpha = persistence_exchange_alpha_relaxation_time(
            wave_number,
            PersistenceExchangeParams(
                cage_variance=params.cage_variance,
                cage_tau=params.cage_tau,
                jump_variance=params.jump_variance,
                persistence_mean=params.translational_persistence_mean,
                exchange_mean=params.translational_exchange_mean,
            ),
        )
        tau_rot = translation_rotation_rotational_relaxation_time(params)

        row = translation_rotation_inversion_protocol(
            benchmark_id="near_tg_molecular_motion_synthetic",
            wave_number=wave_number,
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(
                PersistenceExchangeParams(
                    cage_variance=params.cage_variance,
                    cage_tau=params.cage_tau,
                    jump_variance=params.jump_variance,
                    persistence_mean=params.translational_persistence_mean,
                    exchange_mean=params.translational_exchange_mean,
                )
            ),
            observed_tau_alpha=tau_alpha,
            observed_rotational_relaxation_time=tau_rot,
            rotational_step_correlation=params.rotational_step_correlation,
            rotational_exchange_mean=params.rotational_exchange_mean,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
        )

        self.assertAlmostEqual(row["translational_persistence_mean"], 6.0, places=7)
        self.assertAlmostEqual(row["rotational_persistence_mean"], 15.0, places=7)
        self.assertGreater(row["rotational_to_translational_persistence_ratio"], 2.0)
        self.assertEqual(row["translation_rotation_decoupling_detected"], 1.0)

    def test_persistence_exchange_transport_alpha_inversion_predicts_late_ngp(self):
        inference_fn = getattr(sys.modules["renewal_cage"], "infer_persistence_exchange_from_alpha_transport", None)
        if inference_fn is None:
            self.fail("infer_persistence_exchange_from_alpha_transport is missing")
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=9.0,
            exchange_mean=1.0,
        )
        wave_number = 1.1
        tau_alpha = persistence_exchange_alpha_relaxation_time(wave_number, params)
        diffusion = persistence_exchange_diffusion_coefficient(params)
        late_time = 80.0 * params.persistence_mean
        observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])

        inferred = inference_fn(
            wave_number=wave_number,
            jump_variance=params.jump_variance,
            diffusion_coefficient=diffusion,
            observed_tau_alpha=tau_alpha,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            late_time=late_time,
            observed_late_ngp=observed_late_ngp,
        )

        self.assertAlmostEqual(inferred["exchange_mean"], params.exchange_mean, places=10)
        self.assertAlmostEqual(inferred["persistence_mean"], params.persistence_mean, places=8)
        self.assertAlmostEqual(inferred["persistence_exchange_ratio"], 9.0, places=8)
        self.assertAlmostEqual(inferred["predicted_late_ngp"], observed_late_ngp, places=10)
        self.assertLess(abs(inferred["late_ngp_log_residual"]), 1e-8)

    def test_persistence_exchange_transport_alpha_inversion_rejects_too_fast_alpha(self):
        inference_fn = getattr(sys.modules["renewal_cage"], "infer_persistence_exchange_from_alpha_transport", None)
        if inference_fn is None:
            self.fail("infer_persistence_exchange_from_alpha_transport is missing")
        poisson_params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=1.0,
            exchange_mean=1.0,
        )
        wave_number = 1.1
        poisson_tau = persistence_exchange_alpha_relaxation_time(wave_number, poisson_params)

        with self.assertRaises(ValueError):
            inference_fn(
                wave_number=wave_number,
                jump_variance=poisson_params.jump_variance,
                diffusion_coefficient=persistence_exchange_diffusion_coefficient(poisson_params),
                observed_tau_alpha=0.8 * poisson_tau,
                cage_variance=poisson_params.cage_variance,
                cage_tau=poisson_params.cage_tau,
            )

    def test_persistence_exchange_joint_diagnostic_predicts_multik_late_ngp_and_chi4_proxy(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        diffusion = persistence_exchange_diffusion_coefficient(params)
        late_time = 80.0 * params.persistence_mean
        observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])
        time_grid = np.geomspace(0.05, 300.0, 260)

        diagnostic = persistence_exchange_joint_diagnostic(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            jump_variance=params.jump_variance,
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
        poisson = PersistenceExchangeParams(
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            jump_variance=params.jump_variance,
            persistence_mean=params.exchange_mean,
            exchange_mean=params.exchange_mean,
        )
        inferred = PersistenceExchangeParams(
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            jump_variance=params.jump_variance,
            persistence_mean=diagnostic["persistence_mean"],
            exchange_mean=diagnostic["exchange_mean"],
        )
        predicted_chi4 = np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, inferred))
        poisson_chi4 = np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, poisson))

        self.assertAlmostEqual(diagnostic["persistence_exchange_ratio"], 8.0, places=7)
        self.assertLess(diagnostic["max_multik_tau_alpha_abs_log_residual"], 1e-8)
        self.assertLess(abs(diagnostic["late_ngp_log_residual"]), 1e-8)
        self.assertGreater(diagnostic["stokes_einstein_growth_over_poisson"], 2.0)
        self.assertGreater(diagnostic["chi4_peak_growth_over_poisson"], 1.5)
        self.assertAlmostEqual(diagnostic["chi4_peak"], predicted_chi4, places=10)
        self.assertAlmostEqual(diagnostic["poisson_chi4_peak"], poisson_chi4, places=10)
        self.assertEqual(diagnostic["passes_joint_protocol"], 1.0)

    def test_persistence_exchange_joint_diagnostic_rejects_multik_alpha_mismatch(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        observed_tau_alpha[1.6] *= 1.25
        diagnostic = persistence_exchange_joint_diagnostic(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=80.0 * params.persistence_mean,
            observed_late_ngp=float(persistence_exchange_ngp_1d(np.array([80.0 * params.persistence_mean]), params)[0]),
            time_grid=np.geomspace(0.05, 300.0, 260),
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            max_multik_abs_log_residual=0.02,
            max_late_ngp_abs_log_residual=0.02,
            min_chi4_peak_growth=1.5,
        )

        self.assertGreater(diagnostic["max_multik_tau_alpha_abs_log_residual"], 0.02)
        self.assertEqual(diagnostic["multik_tau_alpha_consistent"], 0.0)
        self.assertEqual(diagnostic["passes_joint_protocol"], 0.0)

    def test_persistence_exchange_data_protocol_scores_uncertainty_weighted_observables(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        time_grid = np.geomspace(0.05, 300.0, 260)
        observed_chi4_peak = float(np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, params)))
        late_time = 80.0 * params.persistence_mean

        scored = persistence_exchange_data_protocol(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            tau_alpha_relative_error_by_k={wave_number: 0.05 for wave_number in wave_numbers},
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            observed_late_ngp=float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0]),
            late_ngp_relative_error=0.08,
            observed_chi4_peak=observed_chi4_peak,
            chi4_peak_relative_error=0.1,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=2.0,
        )

        self.assertLess(scored["max_multik_tau_alpha_z"], 1e-8)
        self.assertLess(scored["late_ngp_z"], 1e-8)
        self.assertLess(scored["chi4_peak_z"], 1e-8)
        self.assertEqual(scored["multik_tau_alpha_z_consistent"], 1.0)
        self.assertEqual(scored["late_ngp_z_consistent"], 1.0)
        self.assertEqual(scored["chi4_peak_z_consistent"], 1.0)
        self.assertEqual(scored["passes_uncertainty_protocol"], 1.0)

    def test_persistence_exchange_data_protocol_rejects_chi4_mismatch_beyond_uncertainty(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        time_grid = np.geomspace(0.05, 300.0, 260)
        predicted_chi4_peak = float(np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, params)))
        late_time = 80.0 * params.persistence_mean

        scored = persistence_exchange_data_protocol(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            tau_alpha_relative_error_by_k={wave_number: 0.05 for wave_number in wave_numbers},
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            observed_late_ngp=float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0]),
            late_ngp_relative_error=0.08,
            observed_chi4_peak=2.0 * predicted_chi4_peak,
            chi4_peak_relative_error=0.1,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=2.0,
        )

        self.assertGreater(scored["chi4_peak_z"], 2.0)
        self.assertEqual(scored["chi4_peak_z_consistent"], 0.0)
        self.assertEqual(scored["passes_uncertainty_protocol"], 0.0)

    def test_persistence_exchange_long_time_ngp_recovers_to_zero(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=12.0,
            exchange_mean=1.0,
        )
        times = np.array([8.0, 30.0, 650.0])

        alpha = persistence_exchange_ngp_1d(times, params, max_count=500)
        decay = persistence_exchange_normalized_alpha_decay(1.1, times, params, max_count=500)

        self.assertGreater(alpha[0], 0.05)
        self.assertLess(alpha[-1], alpha[0] / 20.0)
        self.assertLess(decay[-1], 1e-20)

    def test_persistence_exchange_scan_identifies_ratio_as_se_control(self):
        rows = persistence_exchange_scan(
            ratios=[1.0, 2.0, 4.0, 8.0, 12.0],
            exchange_mean=1.0,
            wave_number=1.1,
        )

        ratios = np.array([row["persistence_exchange_ratio"] for row in rows])
        se_products = np.array([row["stokes_einstein_product"] for row in rows])
        late_ngp = np.array([row["late_ngp"] for row in rows])

        np.testing.assert_allclose(ratios, [1.0, 2.0, 4.0, 8.0, 12.0])
        self.assertTrue(np.all(np.diff(se_products) > 0.0))
        self.assertLess(late_ngp[-1], 0.02)


if __name__ == "__main__":
    unittest.main()
