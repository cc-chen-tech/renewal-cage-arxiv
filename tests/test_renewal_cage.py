import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renewal_cage import (  # noqa: E402
    DelayedRenewalCageParams,
    TemperatureLawParams,
    alpha_relaxation_time,
    classify_delay_exponent,
    delayed_poisson_mean,
    delayed_renewal_shape,
    dimensionless_peak_prediction,
    generalized_delay_ngp_short_time,
    gaussian_radial_3d,
    long_time_diffusion_coefficient,
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
    self_intermediate_scattering,
    stokes_einstein_product,
    temperature_dependent_params,
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
