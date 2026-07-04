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
    alpha_relaxation_shape_curve,
    alpha_shape_superposition_residual,
    correlated_domain_susceptibility,
    classify_delay_exponent,
    delayed_poisson_mean,
    delayed_renewal_shape,
    dimensionless_peak_prediction,
    fractional_stokes_einstein_exponents,
    gamma_exchange_temperature_scan,
    infer_parameters_from_full_observables,
    infer_renewal_correlation_size,
    infer_parameters_from_scattering_transport,
    generalized_delay_ngp_short_time,
    gaussian_radial_3d,
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
    long_time_diffusion_coefficient,
    local_alpha_stretching_exponent,
    late_mechanism_selection,
    observable_consistency_diagnostics,
    radial_van_hove_3d,
    local_cage_variance,
    moments_1d,
    moments_3d,
    ngp_1d,
    ngp_3d,
    normalized_alpha_decay,
    plateau_ngp_branches,
    plateau_peak_diagnostics,
    peak_relaxation_coupling,
    renewal_scattering_susceptibility,
    self_intermediate_scattering,
    static_gamma_asymptotic_diagnostics,
    static_gamma_count_moments,
    static_gamma_ngp_1d,
    static_gamma_normalized_alpha_decay,
    stokes_einstein_product,
    temperature_dependent_params,
    temperature_dependent_gamma_exchange,
    temperature_scan,
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


if __name__ == "__main__":
    unittest.main()
