import json
import importlib.util
import pickle
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ka_replicates

from ka_replicates import (  # noqa: E402
    distance_resolved_event_count_covariance,
    fit_spatial_covariance_length,
    fit_ornstein_zernike_structure_factor,
    initial_configuration_fs,
    load_lammps_custom_trajectory,
    overlap_four_point_structure_factor,
    prepare_replicate,
    prepare_replicate_ensemble,
    summarize_replicate_curves,
    temperature_scan_verdict,
    validate_initial_frame_independence,
)


class KAReplicatePreparationTests(unittest.TestCase):
    def test_independent_group_ratio_reports_growth_and_equivalence_separately(self):
        compare = getattr(ka_replicates, "independent_group_ratio", None)
        self.assertIsNotNone(compare)

        growth = compare(
            np.array([2.0, 2.1, 1.9]),
            np.array([1.0, 1.05, 0.95]),
            relative_equivalence_margin=0.2,
        )
        equivalent = compare(
            np.array([1.02, 1.00, 0.98]),
            np.array([1.00, 1.01, 0.99]),
            relative_equivalence_margin=0.2,
        )

        self.assertAlmostEqual(growth["mean_ratio"], 2.0)
        self.assertGreater(growth["ci95_low_ratio"], 1.0)
        self.assertEqual(growth["growth_detected"], 1.0)
        self.assertEqual(growth["equivalent_to_unity"], 0.0)
        self.assertEqual(equivalent["growth_detected"], 0.0)
        self.assertEqual(equivalent["equivalent_to_unity"], 1.0)

    def test_cage_jump_sota_alignment_rejects_reversed_decoupling_order(self):
        script_path = ROOT / "scripts" / "audit_ka_cage_jump_sota.py"
        spec = importlib.util.spec_from_file_location("audit_ka_cage_jump_sota", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        verdict = module.classify_alignment(
            microscopic_growth={"ci95_low_ratio": 1.05, "ci95_high_ratio": 1.20},
            macroscopic_growth={"ci95_low_ratio": 1.90, "ci95_high_ratio": 2.50},
            microscopic_invariant={"equivalent_to_unity": 1.0},
            jump_length={"decrease_detected": 1.0, "mean_ratio": 0.96},
            low_temperature_ctwr_pass=0.0,
        )

        self.assertEqual(verdict["low_temperature_ctwr_failure_consistent"], 1.0)
        self.assertEqual(verdict["microscopic_invariant_consistent"], 1.0)
        self.assertEqual(verdict["relative_decoupling_order_consistent"], 0.0)
        self.assertEqual(verdict["current_phop_elementary_cage_jump_claim_allowed"], 0.0)

    def test_debye_waller_factor_selects_minimum_log_msd_slope(self):
        select = getattr(ka_replicates, "debye_waller_factor_from_msd", None)
        self.assertIsNotNone(select)
        result = select(
            np.array([1.0, 2.0, 4.0, 8.0, 16.0]),
            np.array([1.0, 1.8, 2.0, 2.1, 4.0]),
        )

        self.assertEqual(result["debye_waller_lag"], 4.0)
        self.assertEqual(result["debye_waller_factor"], 2.0)
        self.assertGreater(result["minimum_log_msd_slope"], 0.0)

    def test_signed_temperature_separation_detects_positive_to_negative_reversal(self):
        compare = getattr(ka_replicates, "signed_temperature_separation", None)
        self.assertIsNotNone(compare)
        result = compare(
            np.array([0.02, 0.03, 0.04, 0.03, 0.025]),
            np.array([-0.22, -0.24, -0.26]),
        )

        self.assertGreater(result["high_ci95_low"], 0.0)
        self.assertLess(result["low_ci95_high"], 0.0)
        self.assertEqual(result["positive_high_negative_low_reversal"], 1.0)
        self.assertEqual(result["confidence_intervals_separated"], 1.0)

    def test_jump_vector_correlation_curve_tracks_green_kubo_convergence(self):
        curve = getattr(ka_replicates, "jump_vector_correlation_curve", None)
        self.assertIsNotNone(curve)
        events = {
            "particle": np.array([0, 0, 0, 0]),
            "time": np.array([1, 2, 3, 4]),
            "jump_vector": np.array(
                [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
            ),
        }

        rows = curve(events, maximum_lag=2)

        self.assertEqual(rows[0]["pair_count"], 3.0)
        self.assertAlmostEqual(rows[0]["correlation_over_q"], -1.0)
        self.assertAlmostEqual(rows[1]["correlation_over_q"], 1.0)
        self.assertAlmostEqual(rows[0]["cumulative_green_kubo_factor"], -1.0)
        self.assertAlmostEqual(rows[1]["cumulative_green_kubo_factor"], 1.0)

    def test_particle_event_count_correlation_tracks_mobility_identity(self):
        correlate = getattr(ka_replicates, "particle_event_count_correlation_curve", None)
        cross_correlate = getattr(
            ka_replicates,
            "particle_event_count_cross_window_correlation",
            None,
        )
        self.assertIsNotNone(correlate)
        self.assertIsNotNone(cross_correlate)
        particles = []
        times = []
        for block in range(4):
            particles.extend([0, 0, 2])
            times.extend([10 * block + 1, 10 * block + 2, 10 * block + 3])
        persistent = correlate(
            {"particle": np.array(particles), "time": np.array(times)},
            duration=40.0,
            particle_count=3,
            block_size=10.0,
            maximum_lag=2,
        )
        alternating = correlate(
            {
                "particle": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
                "time": np.array([1, 2, 11, 12, 21, 22, 31, 32]),
            },
            duration=40.0,
            particle_count=2,
            block_size=10.0,
            maximum_lag=1,
        )

        self.assertAlmostEqual(persistent[0]["particle_identity_correlation"], 1.0)
        self.assertAlmostEqual(persistent[1]["particle_identity_correlation"], 1.0)
        self.assertAlmostEqual(alternating[0]["particle_identity_correlation"], -1.0)
        cross = cross_correlate(
            {"particle": np.array([0, 0, 2])},
            {"particle": np.array([0, 0, 2])},
            particle_count=3,
        )
        self.assertAlmostEqual(cross["particle_identity_correlation"], 1.0)

    def test_particle_event_count_matrix_keeps_complete_blocks(self):
        build = getattr(ka_replicates, "particle_event_count_matrix", None)
        self.assertIsNotNone(build)

        counts = build(
            {
                "particle": np.array([0, 0, 1, 1]),
                "time": np.array([0.0, 9.9, 10.0, 20.0]),
            },
            duration=20.0,
            particle_count=2,
            block_size=10.0,
        )

        np.testing.assert_array_equal(counts, [[2, 0], [0, 1]])

    def test_correlation_efold_crossing_interpolates_first_decay(self):
        crossing = getattr(ka_replicates, "correlation_efold_crossing", None)
        self.assertIsNotNone(crossing)
        result = crossing(
            [
                {"lag_time": 1.0, "particle_identity_correlation": 1.0},
                {"lag_time": 2.0, "particle_identity_correlation": 0.5},
                {"lag_time": 3.0, "particle_identity_correlation": 0.2},
            ]
        )

        self.assertGreater(result["efold_crossing_time"], 2.0)
        self.assertLess(result["efold_crossing_time"], 3.0)
        self.assertAlmostEqual(result["target_correlation"], 1.0 / np.e)

    def test_two_state_poisson_hmm_recovers_finite_exchange_clock(self):
        fit = getattr(ka_replicates, "fit_two_state_poisson_hmm", None)
        self.assertIsNotNone(fit)
        rng = np.random.default_rng(49271)
        transition = np.array([[0.96, 0.04], [0.08, 0.92]])
        means = np.array([0.20, 1.80])
        states = np.zeros((320, 180), dtype=int)
        counts = np.zeros_like(states)
        states[:, 0] = rng.choice(2, size=len(states), p=[2.0 / 3.0, 1.0 / 3.0])
        counts[:, 0] = rng.poisson(means[states[:, 0]])
        for block in range(1, states.shape[1]):
            uniforms = rng.random(len(states))
            states[:, block] = np.where(
                states[:, block - 1] == 0,
                uniforms >= transition[0, 0],
                uniforms >= transition[1, 0],
            )
            counts[:, block] = rng.poisson(means[states[:, block]])

        result = fit(counts, block_size=5.0, max_iterations=200, tolerance=1e-8)

        self.assertTrue(result["converged"])
        self.assertAlmostEqual(result["slow_mean_count"], means[0], delta=0.08)
        self.assertAlmostEqual(result["fast_mean_count"], means[1], delta=0.12)
        self.assertAlmostEqual(result["slow_to_fast_probability"], 0.04, delta=0.02)
        self.assertAlmostEqual(result["fast_to_slow_probability"], 0.08, delta=0.025)
        self.assertGreater(result["exchange_time"], 25.0)
        self.assertLess(result["exchange_time"], 60.0)
        self.assertGreater(result["log_likelihood"], result["single_poisson_log_likelihood"])

    def test_two_state_poisson_hmm_rejects_noninteger_counts(self):
        fit = getattr(ka_replicates, "fit_two_state_poisson_hmm", None)
        self.assertIsNotNone(fit)

        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            fit(np.array([[0.0, 0.5], [1.0, 2.0]]), block_size=1.0)

    def test_two_state_poisson_hmm_scores_heldout_count_statistics(self):
        predict = getattr(ka_replicates, "two_state_poisson_hmm_count_predictions", None)
        score = getattr(ka_replicates, "score_two_state_poisson_hmm", None)
        self.assertIsNotNone(predict)
        self.assertIsNotNone(score)
        fitted = {
            "slow_mean_count": 0.2,
            "fast_mean_count": 1.8,
            "slow_to_fast_probability": 0.04,
            "fast_to_slow_probability": 0.08,
            "stationary_slow_probability": 2.0 / 3.0,
            "stationary_fast_probability": 1.0 / 3.0,
            "exchange_eigenvalue": 0.88,
        }

        rows = predict(fitted, maximum_lag=2)
        expected_mean = (2.0 * 0.2 + 1.8) / 3.0
        expected_environment_variance = (2.0 / 3.0) * (1.0 / 3.0) * 1.6**2
        expected_variance = expected_mean + expected_environment_variance
        self.assertAlmostEqual(rows[0]["predicted_mean_count"], expected_mean)
        self.assertAlmostEqual(rows[0]["predicted_fano_factor"], expected_variance / expected_mean)
        self.assertAlmostEqual(
            rows[1]["predicted_particle_identity_correlation"],
            expected_environment_variance * 0.88**2 / expected_variance,
        )
        scored = score(np.array([[0, 1, 0], [2, 1, 3]]), fitted)
        self.assertTrue(np.isfinite(scored["log_likelihood"]))
        self.assertEqual(scored["observation_count"], 6.0)

    def test_two_mode_exchange_spectrum_recovers_broad_finite_decay(self):
        fit = getattr(ka_replicates, "fit_exponential_correlation_spectrum", None)
        predict = getattr(ka_replicates, "exponential_correlation_spectrum", None)
        self.assertIsNotNone(fit)
        self.assertIsNotNone(predict)
        times = np.arange(1.0, 61.0)
        correlations = 0.24 * np.exp(-times / 3.0) + 0.11 * np.exp(-times / 28.0)

        single = fit(times, correlations, component_count=1, grid_size=80)
        broad = fit(times, correlations, component_count=2, grid_size=80)
        reconstructed = predict(times, broad)

        self.assertLess(broad["rmse"], 0.002)
        self.assertLess(broad["bic"], single["bic"])
        self.assertGreater(broad["slow_time"] / broad["fast_time"], 4.0)
        self.assertGreater(broad["slow_amplitude"], 0.05)
        np.testing.assert_allclose(reconstructed, correlations, atol=0.006)

    def test_fano_anchored_exchange_spectrum_recovers_two_refresh_clocks(self):
        fit = getattr(ka_replicates, "fit_anchored_exponential_correlation_spectrum", None)
        predict = getattr(ka_replicates, "exponential_correlation_spectrum", None)
        self.assertIsNotNone(fit)
        self.assertIsNotNone(predict)
        times = np.arange(1.0, 61.0)
        correlations = 0.35 * (
            0.65 * np.exp(-times / 4.0) + 0.35 * np.exp(-times / 32.0)
        )

        fitted = fit(
            times,
            correlations,
            total_amplitude=0.35,
            component_count=2,
            grid_size=80,
        )
        reconstructed = predict(times, fitted)

        self.assertAlmostEqual(fitted["total_amplitude"], 0.35)
        self.assertAlmostEqual(fitted["fast_amplitude_fraction"], 0.65, delta=0.08)
        self.assertGreater(fitted["slow_time"] / fitted["fast_time"], 4.0)
        np.testing.assert_allclose(reconstructed, correlations, atol=0.006)

    def test_gamma_refresh_cox_generator_matches_analytic_count_moments(self):
        parameterize = getattr(ka_replicates, "gamma_refresh_cox_parameters", None)
        simulate = getattr(ka_replicates, "simulate_gamma_refresh_cox_counts", None)
        predict = getattr(ka_replicates, "gamma_refresh_cox_count_predictions", None)
        count_pmf = getattr(ka_replicates, "gamma_refresh_cox_count_pmf", None)
        pair_pmf = getattr(ka_replicates, "gamma_refresh_cox_pair_pmf", None)
        self.assertIsNotNone(parameterize)
        self.assertIsNotNone(simulate)
        self.assertIsNotNone(predict)
        self.assertIsNotNone(count_pmf)
        self.assertIsNotNone(pair_pmf)
        spectrum = {
            "fast_amplitude": 0.20,
            "fast_time": 4.0,
            "slow_amplitude": 0.10,
            "slow_time": 30.0,
            "total_amplitude": 0.30,
        }
        params = parameterize(
            mean_count=0.8,
            fano_factor=1.0 / 0.7,
            block_size=1.0,
            fitted_spectrum=spectrum,
        )
        counts = simulate(params, sequence_count=2400, block_count=180, random_seed=8291)
        rows = predict(params, maximum_lag=3)
        left = counts[:, :-1].ravel()
        right = counts[:, 1:].ravel()
        empirical_correlation = np.corrcoef(left, right)[0, 1]
        pmf = count_pmf(params, maximum_count=int(np.max(counts)))
        empirical_pmf = np.bincount(counts.ravel(), minlength=len(pmf)) / counts.size
        maximum_pair_count = int(np.max(counts))
        predicted_pair = pair_pmf(params, maximum_count=maximum_pair_count, block_lag=1)
        empirical_pair = np.zeros_like(predicted_pair)
        np.add.at(
            empirical_pair,
            (counts[:, :-1].ravel(), counts[:, 1:].ravel()),
            1.0,
        )
        empirical_pair /= np.sum(empirical_pair)

        self.assertTrue(np.all(counts >= 0))
        self.assertAlmostEqual(float(np.mean(counts)), 0.8, delta=0.02)
        self.assertAlmostEqual(
            float(np.var(counts) / np.mean(counts)),
            1.0 / 0.7,
            delta=0.06,
        )
        self.assertAlmostEqual(
            empirical_correlation,
            rows[0]["predicted_identity_correlation"],
            delta=0.02,
        )
        self.assertLess(float(np.sum(np.abs(empirical_pmf - pmf))), 0.025)
        self.assertLess(float(np.sum(np.abs(empirical_pair - predicted_pair))), 0.04)

    def test_gamma_refresh_cox_gate_requires_count_and_identity_transfer(self):
        script_path = ROOT / "scripts" / "analyze_ka_gamma_refresh_cox.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_gamma_refresh_cox", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        passing = {
            "calibration_bic_gain_two_over_one": 15.0,
            "heldout_mean_relative_error": 0.03,
            "heldout_fano_relative_error": 0.03,
            "heldout_count_tv_distance": 0.02,
            "two_clock_identity_rmse": 0.02,
            "single_clock_identity_rmse": 0.06,
            "two_clock_late_absolute_error": 0.02,
            "slow_time": 500.0,
            "maximum_candidate_time": 6000.0,
        }

        accepted = module.classify_cox_transfer(passing)
        rejected = module.classify_cox_transfer(
            {**passing, "heldout_fano_relative_error": 0.25}
        )
        distribution_rejected = module.classify_cox_transfer(
            {**passing, "heldout_count_tv_distance": 0.04}
        )
        mean_rejected = module.classify_cox_transfer(
            {**passing, "heldout_mean_relative_error": 0.12}
        )

        self.assertEqual(accepted["two_clock_cox_transfer_pass"], 1.0)
        self.assertEqual(rejected["two_clock_cox_transfer_pass"], 0.0)
        self.assertEqual(rejected["primary_failure"], "count_fano")
        self.assertEqual(distribution_rejected["two_clock_cox_transfer_pass"], 0.0)
        self.assertEqual(distribution_rejected["primary_failure"], "count_distribution")
        self.assertEqual(mean_rejected["two_clock_cox_transfer_pass"], 0.0)
        self.assertEqual(mean_rejected["primary_failure"], "mean_count")

    def test_gamma_refresh_crossover_selects_second_clock_only_after_cooling(self):
        script_path = ROOT / "scripts" / "summarize_ka_gamma_refresh_crossover.py"
        spec = importlib.util.spec_from_file_location(
            "summarize_ka_gamma_refresh_crossover", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        result = module.classify_crossover(
            high_outcome="single_clock_gamma_refresh_count_moment_closure",
            low_outcome="two_clock_gamma_refresh_count_moment_closure",
            high_single_transfer=1.0,
            low_single_transfer=4.0 / 6.0,
            low_two_transfer=1.0,
        )

        self.assertEqual(result["cooling_induced_second_refresh_clock_required"], 1.0)
        self.assertAlmostEqual(result["low_temperature_pass_gain"], 2.0 / 6.0)
        self.assertEqual(result["conditional_shape_crossover_closure"], 1.0)
        self.assertEqual(result["count_moment_crossover_closure"], 1.0)
        self.assertEqual(result["full_count_distribution_claim_allowed"], 0.0)

    def test_joint_count_pair_gate_uses_empirical_split_half_resolution(self):
        script_path = ROOT / "scripts" / "analyze_ka_gamma_refresh_cox.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_gamma_refresh_cox", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        result = module.classify_pair_transfer(
            gamma_pair_tv=0.05,
            hmm_pair_tv=0.015,
            empirical_split_half_tv=0.01,
        )

        self.assertEqual(result["pair_tv_tolerance"], 0.03)
        self.assertEqual(result["gamma_pair_distribution_pass"], 0.0)
        self.assertEqual(result["hmm_pair_distribution_pass"], 1.0)

    def test_two_clock_hmm_mixture_matches_analytic_memory_and_pair_distribution(self):
        parameterize = getattr(ka_replicates, "two_clock_hmm_mixture_parameters", None)
        predict = getattr(ka_replicates, "two_clock_hmm_mixture_count_predictions", None)
        pair_pmf = getattr(ka_replicates, "two_clock_hmm_mixture_pair_pmf", None)
        simulate = getattr(ka_replicates, "simulate_two_clock_hmm_mixture_counts", None)
        self.assertIsNotNone(parameterize)
        self.assertIsNotNone(predict)
        self.assertIsNotNone(pair_pmf)
        self.assertIsNotNone(simulate)
        hmm = {
            "slow_mean_count": 0.1,
            "fast_mean_count": 1.6,
            "stationary_slow_probability": 0.7,
            "stationary_fast_probability": 0.3,
        }
        spectrum = {
            "fast_amplitude_fraction": 0.6,
            "fast_time": 3.0,
            "slow_time": 25.0,
        }
        params = parameterize(hmm, spectrum, block_size=1.0)
        counts = simulate(params, sequence_count=3000, block_count=160, random_seed=9921)
        rows = predict(params, maximum_lag=2)
        maximum_count = int(np.max(counts))
        predicted_pair = pair_pmf(params, maximum_count=maximum_count, block_lag=1)
        empirical_pair = np.zeros_like(predicted_pair)
        np.add.at(
            empirical_pair,
            (counts[:, :-1].ravel(), counts[:, 1:].ravel()),
            1.0,
        )
        empirical_pair /= np.sum(empirical_pair)
        empirical_correlation = np.corrcoef(
            counts[:, :-1].ravel(),
            counts[:, 1:].ravel(),
        )[0, 1]

        self.assertAlmostEqual(float(np.mean(counts)), rows[0]["predicted_mean_count"], delta=0.02)
        self.assertAlmostEqual(
            float(np.var(counts) / np.mean(counts)),
            rows[0]["predicted_fano_factor"],
            delta=0.05,
        )
        self.assertAlmostEqual(
            empirical_correlation,
            rows[0]["predicted_identity_correlation"],
            delta=0.02,
        )
        self.assertLess(float(np.sum(np.abs(empirical_pair - predicted_pair))), 0.04)

    def test_two_clock_hmm_total_count_pgf_matches_cumulative_monte_carlo(self):
        predict = getattr(
            ka_replicates,
            "two_clock_hmm_mixture_total_count_statistics",
            None,
        )
        simulate = getattr(ka_replicates, "simulate_two_clock_hmm_mixture_counts", None)
        self.assertIsNotNone(predict)
        self.assertIsNotNone(simulate)
        parameters = {
            "slow_mean_count": 0.08,
            "fast_mean_count": 0.42,
            "stationary_slow_probability": 0.7,
            "stationary_fast_probability": 0.3,
            "fast_clock_weight": 0.6,
            "slow_clock_weight": 0.4,
            "fast_clock_retention": 0.35,
            "slow_clock_retention": 0.92,
            "block_size": 20.0,
        }
        block_count = 8
        argument = 0.73

        predicted = predict(
            parameters,
            block_count=block_count,
            pgf_argument=argument,
        )
        simulated = simulate(
            parameters,
            sequence_count=180000,
            block_count=block_count,
            random_seed=7319,
        ).sum(axis=1)

        self.assertAlmostEqual(predicted["mean_count"], float(np.mean(simulated)), delta=0.01)
        self.assertAlmostEqual(predicted["count_variance"], float(np.var(simulated)), delta=0.02)
        self.assertAlmostEqual(
            predicted["count_pgf"],
            float(np.mean(argument**simulated)),
            delta=0.002,
        )

    def test_two_clock_hmm_total_count_pmf_matches_cumulative_monte_carlo(self):
        predict = getattr(ka_replicates, "two_clock_hmm_mixture_total_count_pmf", None)
        simulate = getattr(ka_replicates, "simulate_two_clock_hmm_mixture_counts", None)
        self.assertIsNotNone(predict)
        self.assertIsNotNone(simulate)
        parameters = {
            "slow_mean_count": 0.08,
            "fast_mean_count": 0.42,
            "stationary_slow_probability": 0.7,
            "stationary_fast_probability": 0.3,
            "fast_clock_weight": 0.6,
            "slow_clock_weight": 0.4,
            "fast_clock_retention": 0.35,
            "slow_clock_retention": 0.92,
            "block_size": 20.0,
        }
        block_count = 8
        predicted = predict(parameters, block_count=block_count, maximum_count=15)
        simulated = simulate(
            parameters,
            sequence_count=180000,
            block_count=block_count,
            random_seed=8127,
        ).sum(axis=1)
        observed = np.bincount(np.minimum(simulated, 16), minlength=17) / len(simulated)
        expected = np.concatenate([predicted["count_pmf"], [predicted["tail_probability"]]])

        self.assertLess(0.5 * float(np.sum(np.abs(observed - expected))), 0.006)

    def test_compound_jump_cage_observables_use_factorial_count_fluctuations(self):
        propagate = getattr(ka_replicates, "compound_jump_cage_observables", None)
        self.assertIsNotNone(propagate)
        result = propagate(
            mean_count=3.0,
            factorial_second_count=12.0,
            jump_msd=0.4,
            jump_fourth_moment=0.32,
            count_pgf=0.55,
            cage_msd=0.2,
            cage_ngp=0.1,
            cage_fs=0.8,
            dimension=3,
        )

        self.assertAlmostEqual(result["jump_channel_msd"], 1.2)
        expected_jump_fourth = 3.0 * 0.32 + (5.0 / 3.0) * 12.0 * 0.4**2
        self.assertAlmostEqual(result["jump_channel_fourth_moment"], expected_jump_fourth)
        self.assertAlmostEqual(result["factorized_msd"], 1.4)
        self.assertAlmostEqual(result["factorized_fs"], 0.44)

    def test_green_kubo_jump_renormalization_scales_all_single_jump_inputs(self):
        renormalize = getattr(ka_replicates, "green_kubo_renormalized_jump_statistics", None)
        self.assertIsNotNone(renormalize)
        jumps = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

        result = renormalize(jumps, green_kubo_factor=0.25, wave_numbers=np.array([2.0]))

        self.assertAlmostEqual(result["raw_jump_msd"], 1.0)
        self.assertAlmostEqual(result["effective_jump_msd"], 0.25)
        self.assertAlmostEqual(result["effective_jump_fourth_moment"], 0.0625)
        self.assertAlmostEqual(result["jump_characteristic_k2"], np.mean(np.cos(jumps)))

    def test_correlated_jump_propagator_preserves_consecutive_direction_memory(self):
        propagate = getattr(ka_replicates, "correlated_jump_propagator", None)
        self.assertIsNotNone(propagate)
        events = {
            "particle": np.array([0, 0, 0, 1, 1, 1]),
            "time": np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            "jump_vector": np.array(
                [
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                ]
            ),
        }

        rows = propagate(
            events,
            maximum_count=3,
            wave_numbers=np.array([1.0]),
            minimum_sample_count=2,
        )

        self.assertEqual([row["jump_count"] for row in rows], [0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(rows[1]["conditional_msd"], 1.0)
        self.assertAlmostEqual(rows[2]["conditional_msd"], 0.0)
        self.assertAlmostEqual(rows[3]["conditional_msd"], 1.0)
        self.assertAlmostEqual(rows[2]["conditional_characteristic_k1"], 1.0)

    def test_two_clock_hmm_hybrid_gate_separates_rate_drift_from_shape_closure(self):
        script_path = ROOT / "scripts" / "analyze_ka_two_clock_hmm_mixture.py"
        spec = importlib.util.spec_from_file_location(
            "analyze_ka_two_clock_hmm_mixture", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        row = {
            "heldout_mean_relative_error": 0.12,
            "heldout_fano_relative_error": 0.08,
            "heldout_count_tv_distance": 0.02,
            "identity_rmse": 0.03,
            "late_identity_absolute_error": 0.02,
            "pair_distribution_pass": 1.0,
        }

        result = module.classify_hybrid_transfer(row)

        self.assertEqual(result["conditional_shape_distribution_pass"], 1.0)
        self.assertEqual(result["full_hybrid_transfer_pass"], 0.0)
        self.assertEqual(result["primary_failure"], "mean_count_drift")

    def test_two_clock_hmm_crossover_selects_hybrid_with_rate_drift_boundary(self):
        script_path = ROOT / "scripts" / "summarize_ka_two_clock_hmm_crossover.py"
        spec = importlib.util.spec_from_file_location(
            "summarize_ka_two_clock_hmm_crossover", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        result = module.classify_crossover(
            high_second_clock_selection=0.0,
            low_second_clock_selection=1.0,
            low_shape_pass=1.0,
            low_full_pass=4.0 / 6.0,
        )

        self.assertEqual(result["cooling_induced_hybrid_clock_selected"], 1.0)
        self.assertEqual(result["conditional_event_shape_crossover_closure"], 1.0)
        self.assertEqual(result["absolute_rate_drift_unresolved"], 1.0)
        self.assertEqual(result["macro_observable_prediction_claim_allowed"], 0.0)

    def test_rate_stability_exact_permutation_detects_monotonic_drift(self):
        script_path = ROOT / "scripts" / "analyze_ka_rate_stability.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_rate_stability", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        result = module.exact_rate_trend(np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]))

        self.assertGreater(result["normalized_total_linear_change"], 0.4)
        self.assertLess(result["exact_two_sided_permutation_p_value"], 0.01)
        self.assertEqual(result["strict_trend_detected"], 1.0)

    def test_rate_threshold_stability_requires_same_sign_and_bounded_change(self):
        script_path = ROOT / "scripts" / "analyze_ka_rate_threshold_sensitivity.py"
        spec = importlib.util.spec_from_file_location(
            "analyze_ka_rate_threshold_sensitivity", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        rows = [
            {"threshold_scale": 0.9, "normalized_total_linear_change": 0.19},
            {"threshold_scale": 1.0, "normalized_total_linear_change": 0.23},
            {"threshold_scale": 1.1, "normalized_total_linear_change": 0.25},
        ]

        result = module.classify_threshold_stability(rows)

        self.assertEqual(result["trend_sign_stable_across_thresholds"], 1.0)
        self.assertEqual(result["trend_amplitude_stable_across_thresholds"], 1.0)
        self.assertEqual(result["threshold_robust_trend"], 1.0)

    def test_exchange_spectrum_rejects_nonincreasing_times(self):
        fit = getattr(ka_replicates, "fit_exponential_correlation_spectrum", None)
        self.assertIsNotNone(fit)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            fit(np.array([1.0, 2.0, 2.0]), np.array([0.3, 0.2, 0.1]), component_count=2)

        with self.assertRaisesRegex(ValueError, "at least five"):
            fit(
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0.4, 0.3, 0.2, 0.1]),
                component_count=2,
            )

    def test_exchange_spectrum_gate_requires_selection_and_heldout_transfer(self):
        script_path = ROOT / "scripts" / "analyze_ka_exchange_spectrum.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_exchange_spectrum", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        passing = {
            "calibration_bic_gain_two_over_one": 12.0,
            "two_mode_heldout_rmse": 0.03,
            "single_mode_heldout_rmse": 0.07,
            "two_mode_late_absolute_error": 0.02,
            "slow_time": 400.0,
            "maximum_candidate_time": 6000.0,
        }

        accepted = module.classify_spectrum_transfer(passing)
        rejected = module.classify_spectrum_transfer(
            {**passing, "two_mode_late_absolute_error": 0.06}
        )

        self.assertEqual(accepted["two_mode_calibration_selected"], 1.0)
        self.assertEqual(accepted["two_mode_heldout_transfer_pass"], 1.0)
        self.assertEqual(rejected["two_mode_heldout_transfer_pass"], 0.0)
        self.assertEqual(rejected["primary_failure"], "late_identity_decay")

    def test_exchange_spectrum_crossover_requires_high_simple_low_broad_pattern(self):
        script_path = ROOT / "scripts" / "summarize_ka_exchange_spectrum_crossover.py"
        spec = importlib.util.spec_from_file_location(
            "summarize_ka_exchange_spectrum_crossover", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        result = module.classify_crossover(
            high_outcome="single_mode_exchange_sufficient",
            low_outcome="two_mode_finite_exchange_spectrum_closure",
            low_selection_fraction=1.0,
            low_transfer_fraction=1.0,
            low_hmm_transfer_fraction=1.0 / 6.0,
        )

        self.assertEqual(result["cooling_induced_exchange_spectrum_broadening"], 1.0)
        self.assertAlmostEqual(result["heldout_pass_gain_over_markov_hmm"], 5.0 / 6.0)
        self.assertEqual(result["semi_markov_generator_required"], 1.0)
        self.assertEqual(result["heldout_macro_prediction_claim_allowed"], 0.0)

    def test_finite_exchange_hmm_gate_requires_late_identity_prediction(self):
        script_path = ROOT / "scripts" / "analyze_ka_finite_exchange_hmm.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_finite_exchange_hmm", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        common = {
            "heldout_mean_relative_error": 0.04,
            "heldout_fano_relative_error": 0.12,
            "heldout_identity_correlation_rmse": 0.03,
            "heldout_hmm_log_likelihood_gain_per_observation": 0.02,
        }

        passed = module.classify_hmm_transfer({**common, "heldout_late_identity_absolute_error": 0.04})
        failed = module.classify_hmm_transfer({**common, "heldout_late_identity_absolute_error": 0.08})

        self.assertEqual(passed["finite_exchange_hmm_transfer_pass"], 1.0)
        self.assertEqual(failed["finite_exchange_hmm_transfer_pass"], 0.0)
        self.assertEqual(failed["primary_failure"], "late_identity_decay")

    def test_hmm_crossover_selects_broad_low_temperature_exchange(self):
        script_path = ROOT / "scripts" / "summarize_ka_finite_exchange_hmm.py"
        spec = importlib.util.spec_from_file_location("summarize_ka_finite_exchange_hmm", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        verdict = module.classify_crossover(
            {"two_state_poisson_hmm_sufficient": "1", "event_level_outcome": "two_state_poisson_hmm_sufficient"},
            {"two_state_poisson_hmm_sufficient": "0", "event_level_outcome": "non_single_exponential_exchange_required"},
            high_positive_late_block_count=0,
            low_positive_late_block_count=3,
            common_block_count=3,
        )

        self.assertEqual(verdict["finite_exchange_spectrum_broadening_detected"], 1.0)
        self.assertEqual(verdict["single_exchange_time_low_temperature_rejected"], 1.0)
        self.assertEqual(verdict["spatial_facilitation_claim_allowed"], 0.0)

    def test_hmm_crossover_does_not_call_unresolved_rejection_sufficient(self):
        script_path = ROOT / "scripts" / "summarize_ka_finite_exchange_hmm.py"
        spec = importlib.util.spec_from_file_location("summarize_ka_finite_exchange_hmm", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        verdict = module.classify_crossover(
            {"two_state_poisson_hmm_sufficient": "1", "event_level_outcome": "two_state_poisson_hmm_sufficient"},
            {"two_state_poisson_hmm_sufficient": "0", "event_level_outcome": "two_state_poisson_hmm_rejected_without_unique_replacement"},
            high_positive_late_block_count=0,
            low_positive_late_block_count=0,
            common_block_count=3,
        )

        self.assertEqual(verdict["low_temperature_two_state_hmm_sufficient"], 0.0)
        self.assertEqual(verdict["finite_exchange_spectrum_broadening_detected"], 0.0)

    def test_event_cumulative_trajectory_applies_jump_vectors_at_event_times(self):
        reconstruct = getattr(ka_replicates, "event_cumulative_trajectory", None)
        self.assertIsNotNone(reconstruct)
        events = {
            "particle": np.array([0, 0, 1]),
            "time": np.array([2, 4, 3]),
            "jump_vector": np.array(
                [[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]]
            ),
        }

        trajectory = reconstruct(
            events,
            frame_count=6,
            particle_count=2,
            dimension=2,
        )

        np.testing.assert_allclose(trajectory[1], [[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(trajectory[2], [[1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(trajectory[3], [[1.0, 0.0], [-1.0, 0.0]])
        np.testing.assert_allclose(trajectory[5], [[1.0, 2.0], [-1.0, 0.0]])

    def test_independent_isotropic_channel_convolution_preserves_gaussianity(self):
        combine = getattr(ka_replicates, "independent_isotropic_channel_moments", None)
        self.assertIsNotNone(combine)
        result = combine(
            first_msd=2.0,
            first_ngp=0.0,
            second_msd=3.0,
            second_ngp=0.0,
            dimension=3,
        )

        self.assertAlmostEqual(result["combined_msd"], 5.0)
        self.assertAlmostEqual(result["combined_ngp"], 0.0)
        self.assertAlmostEqual(result["combined_fourth_moment"], 125.0 / 3.0)

    def test_position_fluctuation_and_cage_jump_segmentation_are_translation_invariant(self):
        fluctuation = getattr(ka_replicates, "position_fluctuation_values", None)
        segment = getattr(ka_replicates, "extract_debye_waller_cage_jumps", None)
        self.assertIsNotNone(fluctuation)
        self.assertIsNotNone(segment)
        trajectory = np.zeros((13, 1, 3), dtype=float)
        trajectory[6:, 0, 0] = 1.0
        times = np.arange(1, 12)
        activity = np.zeros((11, 1), dtype=float)
        activity[4:7, 0] = 2.0

        measured_times, measured = fluctuation(trajectory, half_window=1)
        shifted_times, shifted = fluctuation(
            trajectory + np.array([10.0, -4.0, 3.0]),
            half_window=1,
        )
        events = segment(
            trajectory,
            debye_waller_factor=1.0,
            half_window=1,
            activity_times=times,
            activity_values=activity,
        )

        np.testing.assert_array_equal(measured_times, shifted_times)
        np.testing.assert_allclose(measured, shifted, atol=1e-12)
        self.assertEqual(len(events["time"]), 1)
        np.testing.assert_allclose(events["jump_vector"], [[1.0, 0.0, 0.0]])
        self.assertEqual(events["jump_duration"][0], 3.0)
        self.assertEqual(events["pre_cage_duration"][0], 4.0)
        self.assertEqual(events["post_cage_duration"][0], 4.0)

    def test_neighbor_halo_event_selection_uses_zero_for_all_events(self):
        script_path = ROOT / "scripts" / "analyze_ka_neighbor_halo.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_neighbor_halo", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        valid = np.array([2, 4, 6, 8])

        np.testing.assert_array_equal(
            module.select_event_indices(valid, 0, np.random.default_rng(1)),
            valid,
        )
        sampled = module.select_event_indices(valid, 2, np.random.default_rng(1))
        self.assertEqual(len(sampled), 2)
        self.assertTrue(set(sampled).issubset(set(valid)))
        with self.assertRaisesRegex(ValueError, "zero for all events"):
            module.select_event_indices(valid, -1, np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "too few complete events"):
            module.select_event_indices(valid, 5, np.random.default_rng(1))

    def test_cooperative_closure_fixed_halo_radius_is_labeled_posthoc(self):
        script_path = ROOT / "scripts" / "analyze_ka_cooperative_closure.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_cooperative_closure", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertEqual(module.resolve_halo_radius(5.0, None), (5.0, "calibration_ci"))
        self.assertEqual(module.resolve_halo_radius(5.0, 4.0), (4.0, "posthoc_sensitivity"))
        with self.assertRaisesRegex(ValueError, "positive"):
            module.resolve_halo_radius(5.0, 0.0)

    def test_small_sample_confidence_interval_uses_student_t(self):
        interval = getattr(ka_replicates, "independent_sample_ci95", None)
        self.assertIsNotNone(interval)

        low, high, critical = interval(mean=5.0, standard_error=2.0, sample_count=3)

        self.assertAlmostEqual(critical, 4.302652729911275)
        self.assertAlmostEqual(low, 5.0 - 2.0 * critical)
        self.assertAlmostEqual(high, 5.0 + 2.0 * critical)

    def test_spatial_script_aggregates_independent_replicates_without_enabling_model_claim(self):
        script_path = ROOT / "scripts" / "analyze_ka_spatial_covariance.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_spatial_covariance", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        aggregate = getattr(module, "aggregate_block_rows", None)
        self.assertIsNotNone(aggregate)
        rows = []
        for replicate, length in ((1.0, 1.0), (2.0, 2.0)):
            for block in (0.0, 1.0):
                for distance in (1.0, 2.0, 3.0, 4.0):
                    rows.append(
                        {
                            "replicate": replicate,
                            "block_index": block,
                            "distance_midpoint": distance,
                            "covariance_excess_over_all_pairs": 2.0 * np.exp(-distance / length),
                        }
                    )

        _, summary, _, fit_summary = aggregate(rows, minimum_distance=1.0)

        self.assertEqual(summary[0]["uncertainty_scope"], "independent_replicates")
        self.assertEqual(fit_summary["spatial_measurement_claim_allowed"], 1.0)
        self.assertEqual(fit_summary["spatial_model_claim_allowed"], 0.0)
        self.assertEqual(fit_summary["independent_replicate_count"], 2.0)

    def test_spatial_cli_aggregate_mode_does_not_require_temperature(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inputs = []
            for replicate, length in ((1, 1.0), (2, 2.0)):
                path = root / f"replicate_{replicate}.csv"
                rows = [
                    {
                        "block_index": block,
                        "distance_midpoint": distance,
                        "covariance_excess_over_all_pairs": 2.0 * np.exp(-distance / length),
                    }
                    for block in (0, 1)
                    for distance in (1.0, 2.0, 3.0, 4.0)
                ]
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                inputs.append(path)
            output = root / "ensemble"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_ka_spatial_covariance.py"),
                    "--aggregate-block-files",
                    *(str(path) for path in inputs),
                    "--fit-minimum-distance",
                    "1.0",
                    "--output-prefix",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "ensemble_fit_summary.csv").is_file())

    def test_binned_metric_summary_uses_replicate_means_for_uncertainty(self):
        summarize = getattr(ka_replicates, "summarize_replicate_binned_metric", None)
        self.assertIsNotNone(summarize)
        rows = [
            {"replicate": 1.0, "distance": 1.0, "block": 0.0, "covariance": 2.0},
            {"replicate": 1.0, "distance": 1.0, "block": 1.0, "covariance": 4.0},
            {"replicate": 2.0, "distance": 1.0, "block": 0.0, "covariance": 6.0},
            {"replicate": 2.0, "distance": 1.0, "block": 1.0, "covariance": 8.0},
            {"replicate": 1.0, "distance": 2.0, "block": 0.0, "covariance": 1.0},
            {"replicate": 2.0, "distance": 2.0, "block": 0.0, "covariance": 3.0},
        ]

        replicate_rows, summary_rows = summarize(
            rows,
            bin_key="distance",
            metric_key="covariance",
        )

        first_bin_replicates = [row for row in replicate_rows if row["distance"] == 1.0]
        first_bin_summary = next(row for row in summary_rows if row["distance"] == 1.0)
        self.assertEqual([row["covariance"] for row in first_bin_replicates], [3.0, 7.0])
        self.assertAlmostEqual(first_bin_summary["mean"], 5.0)
        self.assertAlmostEqual(first_bin_summary["standard_error"], 2.0)
        self.assertEqual(first_bin_summary["independent_replicate_count"], 2.0)
        self.assertEqual(first_bin_summary["within_replicate_block_count_min"], 2.0)
        self.assertEqual(first_bin_summary["within_replicate_block_count_max"], 2.0)

    def test_spatial_covariance_ensemble_fits_lengths_per_replicate(self):
        summarize = getattr(ka_replicates, "summarize_spatial_covariance_replicates", None)
        self.assertIsNotNone(summarize)
        rows = []
        for replicate, length in ((1.0, 1.0), (2.0, 2.0)):
            for block in (0.0, 1.0):
                for distance in (1.0, 2.0, 3.0, 4.0):
                    rows.append(
                        {
                            "replicate": replicate,
                            "block_index": block,
                            "distance_midpoint": distance,
                            "covariance_excess_over_all_pairs": 2.0 * np.exp(-distance / length),
                        }
                    )

        _, curve, fits, fit_summary = summarize(rows, minimum_distance=1.0)

        self.assertEqual(len(curve), 4)
        self.assertEqual([row["replicate"] for row in fits], [1.0, 2.0])
        self.assertAlmostEqual(fits[0]["correlation_length"], 1.0)
        self.assertAlmostEqual(fits[1]["correlation_length"], 2.0)
        self.assertAlmostEqual(fit_summary["mean_correlation_length"], 1.5)
        self.assertAlmostEqual(fit_summary["standard_error_correlation_length"], 0.5)
        self.assertEqual(fit_summary["independent_replicate_count"], 2.0)

    def test_overlap_s4_replicate_gate_rejects_one_invalid_oz_fit(self):
        summarize = getattr(ka_replicates, "summarize_overlap_s4_replicates", None)
        self.assertIsNotNone(summarize)
        rows = []
        for replicate in (1.0, 2.0):
            rows.append({"replicate": replicate, "integer_squared": 0.0, "wave_number": 0.0, "s4": 2.0})
        for q, valid_s4, invalid_s4 in (
            (0.2, 10.0 / 1.16, 25.0),
            (0.3, 10.0 / 1.36, 4.5),
            (0.4, 10.0 / 1.64, 2.0),
            (0.5, 10.0 / 2.0, 1.1),
        ):
            rows.append({"replicate": 1.0, "integer_squared": q, "wave_number": q, "s4": valid_s4})
            rows.append({"replicate": 2.0, "integer_squared": q, "wave_number": q, "s4": invalid_s4})

        _, fits, verdict = summarize(rows, ensemble_correction_available=False)

        self.assertEqual(sum(not row["fit_valid"] for row in fits), 1)
        self.assertEqual(verdict["invalid_replicate_fit_count"], 1.0)
        self.assertEqual(verdict["raw_oz_fit_reproducible"], 0.0)
        self.assertEqual(verdict["xi4_identifiable"], 0.0)
        self.assertEqual(verdict["xi4_claim_allowed"], 0.0)

    def test_overlap_s4_script_preserves_missing_ensemble_correction_gate(self):
        script_path = ROOT / "scripts" / "analyze_ka_overlap_s4.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_overlap_s4", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        aggregate = getattr(module, "aggregate_curve_rows", None)
        self.assertIsNotNone(aggregate)
        rows = []
        for replicate in (1.0, 2.0):
            rows.append({"replicate": replicate, "integer_squared": 0.0, "wave_number": 0.0, "s4": 2.0})
            for q in (0.2, 0.3, 0.4, 0.5):
                rows.append(
                    {
                        "replicate": replicate,
                        "integer_squared": q,
                        "wave_number": q,
                        "s4": 10.0 / (1.0 + (2.0 * q) ** 2),
                    }
                )

        _, _, verdict = aggregate(rows)

        self.assertEqual(verdict["raw_oz_fit_reproducible"], 1.0)
        self.assertEqual(verdict["ensemble_correction_available"], 0.0)
        self.assertEqual(verdict["xi4_identifiable"], 0.0)
        self.assertEqual(verdict["xi4_claim_allowed"], 0.0)

    def test_overlap_s4_cli_aggregate_mode_does_not_require_temperature_or_lag(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inputs = []
            for replicate in (1, 2):
                path = root / f"replicate_{replicate}.csv"
                rows = [{"integer_squared": 0.0, "wave_number": 0.0, "s4": 2.0}]
                rows.extend(
                    {
                        "integer_squared": q,
                        "wave_number": q,
                        "s4": 10.0 / (1.0 + (2.0 * q) ** 2),
                    }
                    for q in (0.2, 0.3, 0.4, 0.5)
                )
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                inputs.append(path)
            output = root / "ensemble"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_ka_overlap_s4.py"),
                    "--aggregate-curve-files",
                    *(str(path) for path in inputs),
                    "--output-prefix",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "ensemble_verdict.csv").is_file())

    def test_temperature_event_trend_reports_unequal_replicate_counts(self):
        script_path = ROOT / "scripts" / "summarize_ka_replicate_scan.py"
        spec = importlib.util.spec_from_file_location("summarize_ka_replicate_scan", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        metrics = (
            "event_rate",
            "exchange_mean",
            "stationary_persistence_mean",
            "persistence_exchange_ratio",
            "count_fano",
            "correlated_diffusion",
        )
        high = [
            {"metric": metric, "mean": 2.0, "ci95_low": 1.8, "ci95_high": 2.2, "independent_replicate_count": 5.0}
            for metric in metrics
        ]
        low = [
            {"metric": metric, "mean": 1.0, "ci95_low": 0.8, "ci95_high": 1.2, "independent_replicate_count": 3.0}
            for metric in metrics
        ]

        rows = module.event_temperature_trends(high, low)

        self.assertEqual(rows[0]["high_temperature_replicate_count"], 5.0)
        self.assertEqual(rows[0]["low_temperature_replicate_count"], 3.0)
        self.assertNotIn("independent_replicates_per_temperature", rows[0])

    def test_waiting_diagnostic_summary_requires_replica_consensus(self):
        script_path = ROOT / "scripts" / "analyze_ka_replicates.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_replicates", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        summarize = getattr(module, "summarize_waiting_diagnostic_rows", None)
        self.assertIsNotNone(summarize)
        rows = []
        for replicate in (1.0, 2.0, 3.0):
            for window in (512.0, 1024.0):
                rows.append(
                    {
                        "replicate": replicate,
                        "count_window": window,
                        "empirical_iid_relative_error": 0.08,
                        "gamma_iid_relative_error": 0.30,
                        "temporal_memory_excess_fraction": 0.03,
                        "persistent_environment_excess_fraction": 0.25,
                        "waiting_lag1_correlation": 0.01,
                        "shuffle_lag1_correlation_mean": 0.0,
                        "shuffle_lag1_correlation_standard_deviation": 0.02,
                        "complete_wait_particle_fraction": 0.5,
                        "complete_waiting_time_count": 200.0,
                        "particles_with_complete_wait": 100.0,
                        "window_count": 8.0,
                        "collective_covariance_ratio": 4.0,
                    }
                )

        replicate_rows, verdict = summarize(rows)

        self.assertEqual(len(replicate_rows), 3)
        self.assertTrue(all(row["waiting_failure_verdict"] == "empirical_iid_waiting_law_sufficient" for row in replicate_rows))
        self.assertEqual(verdict["consensus_verdict"], "empirical_iid_waiting_law_sufficient")
        self.assertEqual(verdict["consensus_replicate_count"], 3.0)
        self.assertEqual(verdict["independent_replicate_count"], 3.0)
        self.assertTrue(all(row["persistent_environment_identifiable"] == 0.0 for row in replicate_rows))
        self.assertEqual(verdict["persistent_environment_identifiable"], 0.0)
        self.assertEqual(verdict["collective_covariance_replicate_count"], 3.0)
        self.assertEqual(verdict["spatial_cooperation_test_required"], 1.0)
        self.assertEqual(verdict["spatial_cooperation_proven"], 0.0)

    def test_waiting_threshold_audit_separates_dominant_and_secondary_mechanisms(self):
        script_path = ROOT / "scripts" / "analyze_ka_waiting_threshold_sensitivity.py"
        spec = importlib.util.spec_from_file_location(
            "analyze_ka_waiting_threshold_sensitivity", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        rows = []
        for replicate in (1.0, 2.0, 3.0):
            for threshold_scale in (0.9, 1.0, 1.1):
                rows.append(
                    {
                        "replicate": replicate,
                        "threshold_scale": threshold_scale,
                        "median_empirical_iid_relative_error": 0.19,
                        "median_gamma_iid_relative_error": 0.34,
                        "median_sequence_shuffle_relative_error": 0.09,
                        "maximum_sequence_shuffle_relative_error": (
                            0.19 if threshold_scale == 0.9 else 0.14
                        ),
                        "long_window_sequence_shuffle_relative_error": (
                            0.19 if threshold_scale == 0.9 else 0.14
                        ),
                        "median_temporal_ordering_contribution_fraction": 0.09,
                        "median_particle_identity_contribution_fraction": 0.10,
                        "median_waiting_correlation_z_vs_shuffle": 15.0,
                        "persistent_environment_identifiable": 1.0,
                        "collective_covariance_detected": 1.0,
                    }
                )

        replicate_rows, verdict = module.classify_waiting_threshold_sensitivity(
            rows, finite_exchange_supported=True
        )

        self.assertEqual(len(replicate_rows), 3)
        self.assertTrue(
            all(
                row["dominant_mechanism"]
                == "mixed_particle_environment_and_event_memory"
                for row in replicate_rows
            )
        )
        self.assertEqual(verdict["threshold_robust_dominant_mechanism"], 1.0)
        self.assertEqual(
            verdict["dominant_mechanism"],
            "mixed_particle_environment_and_event_memory",
        )
        self.assertEqual(verdict["gamma_shape_misspecification_supported"], 1.0)
        self.assertEqual(verdict["empirical_waiting_law_sufficient"], 0.0)
        self.assertEqual(verdict["gamma_shape_misspecification_sufficient"], 0.0)
        self.assertEqual(verdict["temporal_waiting_memory_supported"], 1.0)
        self.assertEqual(verdict["temporal_waiting_memory_dominant"], 0.0)
        self.assertEqual(verdict["temporal_waiting_memory_parameter_claim_allowed"], 0.0)
        self.assertEqual(verdict["median_window_particle_conditioned_shuffle_sufficient"], 1.0)
        self.assertEqual(verdict["all_window_particle_conditioned_shuffle_sufficient"], 0.0)
        self.assertEqual(verdict["long_window_shuffle_failure_any_threshold"], 1.0)
        self.assertEqual(verdict["long_window_shuffle_failure_all_thresholds"], 0.0)
        self.assertEqual(verdict["persistent_particle_environment_supported"], 1.0)
        self.assertEqual(verdict["spatial_cooperation_test_required"], 1.0)
        self.assertEqual(verdict["spatial_cooperation_proven"], 0.0)
        self.assertEqual(
            verdict["minimal_model_implication"],
            "finite_exchange_particle_conditioned_renewal",
        )

    def test_three_temperature_uncertainty_certificate_keeps_parent_scope_boundary(self):
        script_path = ROOT / "scripts" / "audit_ka_three_temperature_uncertainty.py"
        spec = importlib.util.spec_from_file_location(
            "audit_ka_three_temperature_uncertainty", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        coverage = []
        for temperature, count in ((0.70, 5.0), (0.58, 5.0), (0.45, 3.0)):
            for metric in (
                "diffusion",
                "alpha_relaxation_time",
                "diffusion_alpha_product",
                "ngp_peak",
                "persistence_exchange_ratio",
            ):
                coverage.append(
                    {
                        "temperature": temperature,
                        "metric": metric,
                        "observable_class": "scalar",
                        "independent_replicate_count": count,
                        "uncertainty_ready": 1.0,
                        "precision_ready": 1.0,
                    }
                )
            for metric in ("msd", "ngp_3d", "fs_k5", "fs_k7p25", "fs_k9"):
                coverage.append(
                    {
                        "temperature": temperature,
                        "metric": metric,
                        "observable_class": "curve",
                        "independent_replicate_count": count,
                        "uncertainty_ready": 1.0,
                        "precision_ready": 1.0,
                    }
                )
        trends = [
            {"metric": metric, "trend_pass": 1.0}
            for metric in (
                "diffusion",
                "alpha_relaxation_time",
                "diffusion_alpha_product",
                "ngp_peak",
                "persistence_exchange_ratio",
            )
            for _ in range(2)
        ]
        manifests = [
            {
                "temperature": temperature,
                "replicate_count": count,
                "independently_prepared_parent_samples": False,
                "maximum_absolute_fs_observed": 0.03,
                "maximum_absolute_fs_allowed": 0.1,
                "independence_class": "decorrelated_parent_frames_plus_velocity_seeds",
            }
            for temperature, count in ((0.70, 5), (0.58, 5), (0.45, 3))
        ]

        verdict = module.build_uncertainty_verdict(
            manifests,
            coverage,
            trends,
            physical_time_gate_pass=True,
        )

        self.assertEqual(verdict["temperature_count"], 3.0)
        self.assertEqual(verdict["minimum_restart_replicate_count"], 3.0)
        self.assertEqual(
            verdict["restart_replicate_counts_by_temperature"],
            "0.45:3;0.58:5;0.7:5",
        )
        self.assertEqual(verdict["restart_ensemble_uncertainty_ready"], 1.0)
        self.assertEqual(verdict["physical_time_definition_consistent"], 1.0)
        self.assertEqual(verdict["saved_frame_interval_tau"], 1.0)
        self.assertEqual(verdict["core_scalar_uncertainty_ready"], 1.0)
        self.assertEqual(verdict["curve_uncertainty_ready"], 1.0)
        self.assertEqual(verdict["core_scalar_precision_ready"], 1.0)
        self.assertEqual(verdict["curve_precision_ready"], 1.0)
        self.assertEqual(verdict["three_temperature_trend_chain_pass"], 1.0)
        self.assertEqual(verdict["independently_prepared_parent_ensemble_ready"], 0.0)
        self.assertEqual(verdict["thermodynamic_claim_allowed"], 0.0)

    def test_hybrid_macro_transfer_requires_curves_and_derived_scalars(self):
        script_path = ROOT / "scripts" / "predict_ka_hybrid_macro_observables.py"
        spec = importlib.util.spec_from_file_location(
            "predict_ka_hybrid_macro_observables", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        curve_errors = {
            "maximum_ensemble_msd_relative_error": 0.08,
            "maximum_ensemble_ngp_absolute_error": 0.20,
            "maximum_ensemble_fs_absolute_error": 0.02,
        }
        scalar_errors = {
            "diffusion_relative_error": 0.07,
            "alpha_relaxation_relative_error": 0.12,
            "diffusion_alpha_product_relative_error": 0.15,
        }

        verdict = module.classify_macro_transfer(
            curve_errors,
            scalar_errors,
            alpha_crossing_ready=True,
        )

        self.assertEqual(verdict["curve_transfer_pass"], 1.0)
        self.assertEqual(verdict["derived_scalar_transfer_pass"], 1.0)
        self.assertEqual(verdict["joint_macro_transfer_pass"], 1.0)
        self.assertEqual(verdict["heldout_events_used_in_prediction"], 0.0)
        self.assertEqual(verdict["macro_fit_parameter_count"], 0.0)
        self.assertEqual(verdict["thermodynamic_claim_allowed"], 0.0)

    def test_hybrid_macro_crossover_selects_state_conditioned_kernel_not_new_clock(self):
        script_path = ROOT / "scripts" / "summarize_ka_hybrid_macro_transfer.py"
        spec = importlib.util.spec_from_file_location(
            "summarize_ka_hybrid_macro_transfer", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        result = module.classify_crossover(
            low={
                "diffusion_relative_error": "0.074",
                "alpha_relaxation_relative_error": "0.337",
                "maximum_ensemble_ngp_absolute_error": "0.528",
                "maximum_ensemble_fs_absolute_error": "0.0755",
                "maximum_count_tail_probability": "0.008",
                "jump_direction_correlation_included": "1",
                "joint_macro_transfer_pass": "0",
            },
            high={
                "diffusion_relative_error": "0.226",
                "maximum_ensemble_ngp_absolute_error": "0.527",
                "maximum_ensemble_fs_absolute_error": "0.087",
                "maximum_count_tail_probability": "0.015",
                "joint_macro_transfer_pass": "0",
            },
        )

        self.assertEqual(result["low_temperature_diffusion_transfer_pass"], 1.0)
        self.assertEqual(result["low_temperature_alpha_transfer_pass"], 0.0)
        self.assertEqual(result["low_temperature_count_kernel_identifiable"], 1.0)
        self.assertEqual(result["independent_count_jump_kernel_rejected"], 1.0)
        self.assertEqual(result["additional_exchange_clock_supported"], 0.0)
        self.assertEqual(
            result["next_minimal_extension"],
            "mobility_state_conditioned_jump_cage_kernel",
        )

    def test_ornstein_zernike_fit_recovers_length_and_rejects_negative_intercept(self):
        valid_rows = [
            {"wave_number": q, "s4": 10.0 / (1.0 + (2.0 * q) ** 2)}
            for q in (0.2, 0.3, 0.4, 0.5)
        ]
        valid = fit_ornstein_zernike_structure_factor(valid_rows)
        self.assertTrue(valid["fit_valid"])
        self.assertAlmostEqual(valid["amplitude"], 10.0)
        self.assertAlmostEqual(valid["correlation_length"], 2.0)

        invalid_rows = [
            {"wave_number": q, "s4": s4}
            for q, s4 in ((0.4, 25.0), (0.6, 4.5), (0.8, 2.0), (1.0, 1.1))
        ]
        invalid = fit_ornstein_zernike_structure_factor(invalid_rows)
        self.assertFalse(invalid["fit_valid"])
        self.assertLess(invalid["inverse_intercept"], 0.0)
        self.assertTrue(np.isnan(invalid["correlation_length"]))

    def test_overlap_s4_is_periodic_translation_invariant_and_has_q0_susceptibility(self):
        frame0 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        frame1 = frame0.copy()
        frame1[0, 0] += 1.0
        frame2 = frame1.copy()
        frame2[[0, 1, 2], 1] += 1.0
        trajectory = np.stack([frame0, frame1, frame2])

        rows = overlap_four_point_structure_factor(
            trajectory,
            box_lengths=np.array([4.0, 4.0, 4.0]),
            lag=1,
            overlap_radius=0.3,
            origin_stride=1,
            maximum_integer_squared=1,
        )
        shifted = overlap_four_point_structure_factor(
            trajectory + np.array([4.0, -4.0, 8.0]),
            box_lengths=np.array([4.0, 4.0, 4.0]),
            lag=1,
            overlap_radius=0.3,
            origin_stride=1,
            maximum_integer_squared=1,
        )

        self.assertEqual(rows[0]["integer_squared"], 0.0)
        self.assertAlmostEqual(rows[0]["s4"], 0.25)
        self.assertEqual(rows[1]["wavevector_count"], 6.0)
        self.assertGreaterEqual(rows[1]["s4"], 0.0)
        self.assertLessEqual(rows[1]["s4_wavevector_min"], rows[1]["s4"])
        self.assertGreaterEqual(rows[1]["s4_wavevector_max"], rows[1]["s4"])
        self.assertGreaterEqual(rows[1]["s4_wavevector_standard_deviation"], 0.0)
        np.testing.assert_allclose(
            [row["s4"] for row in rows],
            [row["s4"] for row in shifted],
        )

    def test_spatial_covariance_length_recovers_exponential_decay(self):
        rows = [
            {"distance_midpoint": r, "mean_covariance_excess": 2.0 * np.exp(-r / 1.5)}
            for r in (1.0, 2.0, 3.0, 4.0)
        ]

        fit = fit_spatial_covariance_length(rows, minimum_distance=1.0)

        self.assertAlmostEqual(fit["correlation_length"], 1.5)
        self.assertAlmostEqual(fit["amplitude"], 2.0)
        self.assertAlmostEqual(fit["log_space_r_squared"], 1.0)
        self.assertEqual(fit["fit_point_count"], 4.0)

    def test_distance_resolved_covariance_finds_close_pair_coactivity(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0], [5.5, 0.0, 0.0]]
        )
        events = {
            "particle": np.array([0, 1, 2, 3, 0, 1, 2, 3]),
            "time": np.array([1, 1, 11, 11, 21, 21, 31, 31]),
        }

        rows = distance_resolved_event_count_covariance(
            events,
            positions,
            np.array([20.0, 20.0, 20.0]),
            duration=40.0,
            count_window=10.0,
            distance_edges=np.array([0.0, 1.0, 10.0]),
        )

        close, far = rows
        self.assertEqual(close["pair_count"], 2.0)
        self.assertEqual(far["pair_count"], 4.0)
        self.assertGreater(close["covariance_excess_over_all_pairs"], 0.0)
        self.assertLess(far["covariance_excess_over_all_pairs"], 0.0)
        self.assertAlmostEqual(
            sum(row["pair_count"] * row["covariance_excess_over_all_pairs"] for row in rows),
            0.0,
        )

    def test_event_conditioned_neighbor_halo_detects_cooperative_motion(self):
        measure = getattr(ka_replicates, "event_conditioned_neighbor_displacement", None)
        self.assertIsNotNone(measure)
        frame0 = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ]
        )
        frame1 = frame0.copy()
        frame2 = frame1.copy()
        frame2[0, 0] += 1.0
        frame2[1, 0] += 0.5
        frame2[3, 0] += 0.1
        frame3 = frame2.copy()
        trajectory = np.stack([frame0, frame1, frame2, frame3])
        events = {
            "particle": np.array([0]),
            "time": np.array([2]),
            "jump_vector": np.array([[1.0, 0.0, 0.0]]),
        }

        rows, summary = measure(
            trajectory,
            events,
            box_lengths=np.array([100.0, 100.0, 100.0]),
            distance_edges=np.array([0.0, 2.0, 20.0]),
            half_window=1,
            event_indices=np.array([0]),
            control_particles=np.array([2]),
            integration_max_distance=2.0,
        )
        shifted, _ = measure(
            trajectory + np.array([100.0, -100.0, 200.0]),
            events,
            box_lengths=np.array([100.0, 100.0, 100.0]),
            distance_edges=np.array([0.0, 2.0, 20.0]),
            half_window=1,
            event_indices=np.array([0]),
            control_particles=np.array([2]),
            integration_max_distance=2.0,
        )

        self.assertAlmostEqual(rows[0]["event_mean_squared_displacement"], 0.25)
        self.assertAlmostEqual(rows[0]["control_mean_squared_displacement"], 0.01)
        self.assertAlmostEqual(rows[0]["event_to_control_squared_ratio"], 25.0)
        self.assertGreater(summary["integrated_neighbor_excess_over_self_jump_squared"], 0.0)
        np.testing.assert_allclose(
            [row["event_to_control_squared_ratio"] for row in rows],
            [row["event_to_control_squared_ratio"] for row in shifted],
        )

    def test_neighbor_halo_replicate_summary_requires_ci_above_control(self):
        summarize = getattr(ka_replicates, "summarize_neighbor_halo_replicates", None)
        self.assertIsNotNone(summarize)
        shell_rows = []
        for replicate, first, second in ((1, 2.0, 1.1), (2, 2.1, 1.0), (3, 1.9, 0.9)):
            shell_rows.extend(
                [
                    {
                        "replicate": float(replicate),
                        "distance_low": 0.0,
                        "distance_high": 2.0,
                        "distance_midpoint": 1.0,
                        "event_to_control_squared_ratio": first,
                    },
                    {
                        "replicate": float(replicate),
                        "distance_low": 2.0,
                        "distance_high": 4.0,
                        "distance_midpoint": 3.0,
                        "event_to_control_squared_ratio": second,
                    },
                ]
            )
        replicate_rows = [
            {
                "replicate": float(replicate),
                "integrated_neighbor_excess_over_self_jump_squared": value,
            }
            for replicate, value in ((1, 5.0), (2, 6.0), (3, 5.5))
        ]

        curve, verdict = summarize(shell_rows, replicate_rows)

        self.assertEqual(curve[0]["halo_detected_in_shell"], 1.0)
        self.assertEqual(curve[1]["halo_detected_in_shell"], 0.0)
        self.assertEqual(verdict["halo_radius_lower_bound"], 2.0)
        self.assertAlmostEqual(verdict["mean_integrated_neighbor_excess_over_self_jump_squared"], 5.5)
        self.assertEqual(verdict["spatial_measurement_claim_allowed"], 1.0)
        self.assertEqual(verdict["spatial_model_claim_allowed"], 0.0)

    def test_paired_curve_stability_requires_each_difference_ci_to_include_zero(self):
        summarize = getattr(ka_replicates, "summarize_paired_curve_stability", None)
        self.assertIsNotNone(summarize)
        calibration = []
        heldout = []
        for replicate, delta in ((1, -0.1), (2, 0.0), (3, 0.1)):
            for distance in (1.0, 2.0):
                calibration.append(
                    {"replicate": replicate, "distance_midpoint": distance, "value": distance}
                )
                heldout.append(
                    {
                        "replicate": replicate,
                        "distance_midpoint": distance,
                        "value": distance + delta,
                    }
                )

        curve, verdict = summarize(
            calibration,
            heldout,
            bin_key="distance_midpoint",
            metric_key="value",
            relative_equivalence_margin=0.3,
        )

        self.assertTrue(all(row["paired_difference_ci_includes_zero"] == 1.0 for row in curve))
        self.assertTrue(all(row["paired_difference_ci_within_margin"] == 1.0 for row in curve))
        self.assertEqual(verdict["paired_shift_not_detected"], 1.0)
        self.assertEqual(verdict["paired_curve_equivalent"], 1.0)
        shifted = [dict(row, value=float(row["value"]) + 1.0) for row in heldout]
        _, shifted_verdict = summarize(
            calibration,
            shifted,
            bin_key="distance_midpoint",
            metric_key="value",
            relative_equivalence_margin=0.3,
        )
        self.assertEqual(shifted_verdict["paired_curve_equivalent"], 0.0)

    def test_event_duration_and_spatiotemporal_clusters_are_data_derived(self):
        durations = getattr(ka_replicates, "event_activity_duration_statistics", None)
        clusters = getattr(ka_replicates, "spatiotemporal_event_cluster_statistics", None)
        self.assertIsNotNone(durations)
        self.assertIsNotNone(clusters)
        activity = np.array(
            [
                [0.0, 0.0],
                [0.6, 0.0],
                [0.8, 0.7],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        events = {
            "particle": np.array([0, 1, 0]),
            "time": np.array([3, 3, 10]),
            "pre_center": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
        }

        duration = durations(
            np.arange(1, 6),
            activity,
            {"particle": events["particle"][:2], "time": events["time"][:2]},
            threshold=0.5,
        )
        cluster = clusters(
            events,
            box_lengths=np.array([20.0, 20.0, 20.0]),
            maximum_time_separation=2,
            maximum_distance=2.0,
        )
        same_particle = clusters(
            {
                "particle": np.array([0, 0]),
                "time": np.array([1, 2]),
                "pre_center": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            },
            box_lengths=np.array([20.0, 20.0, 20.0]),
            maximum_time_separation=2,
            maximum_distance=2.0,
        )

        self.assertAlmostEqual(duration["median_duration"], 1.5)
        self.assertEqual(duration["cluster_time_window"], 2.0)
        self.assertEqual(cluster["cluster_count"], 2.0)
        self.assertAlmostEqual(cluster["mean_cluster_size"], 1.5)
        self.assertAlmostEqual(cluster["event_weighted_cluster_size"], 5.0 / 3.0)
        self.assertAlmostEqual(cluster["nontrivial_event_fraction"], 2.0 / 3.0)
        self.assertEqual(same_particle["cluster_count"], 2.0)
        self.assertEqual(same_particle["mean_cluster_size"], 1.0)

    def test_isolated_response_and_cooperative_diffusion_use_micro_inputs(self):
        response = getattr(ka_replicates, "isolated_event_response_amplitude", None)
        diffusion = getattr(ka_replicates, "cooperative_cluster_diffusion_coefficient", None)
        self.assertIsNotNone(response)
        self.assertIsNotNone(diffusion)
        trajectory = np.zeros((7, 1, 3), dtype=float)
        trajectory[2:4, 0, 0] = 1.0
        trajectory[4:, 0, 0] = 0.8
        events = {
            "particle": np.array([0]),
            "time": np.array([2]),
            "jump_vector": np.array([[1.0, 0.0, 0.0]]),
            "pre_center": np.array([[0.0, 0.0, 0.0]]),
        }

        measured = response(trajectory, events, response_lag=2, half_window=1)
        predicted = diffusion(
            event_rate=0.01,
            self_jump_squared=1.0,
            integrated_neighbor_excess=2.0,
            response_amplitude=0.5,
            mean_cluster_size=2.0,
            dimension=3,
        )

        self.assertAlmostEqual(measured["mean_response_amplitude"], 0.8)
        self.assertEqual(measured["isolated_event_count"], 1.0)
        self.assertAlmostEqual(predicted, 0.000625)

    def test_replicate_curve_summary_uses_between_trajectory_error(self):
        rows = [
            {"replicate": 1.0, "lag": 1.0, "msd": 1.0, "fs_k7p25": 0.8},
            {"replicate": 2.0, "lag": 1.0, "msd": 3.0, "fs_k7p25": 0.6},
            {"replicate": 1.0, "lag": 2.0, "msd": 2.0, "fs_k7p25": 0.5},
            {"replicate": 2.0, "lag": 2.0, "msd": 4.0, "fs_k7p25": 0.3},
        ]

        summary = summarize_replicate_curves(rows, metric_keys=["msd", "fs_k7p25"])

        msd_lag_one = next(
            row for row in summary if row["lag"] == 1.0 and row["metric"] == "msd"
        )
        self.assertEqual(msd_lag_one["independent_replicate_count"], 2.0)
        self.assertAlmostEqual(msd_lag_one["mean"], 2.0)
        self.assertAlmostEqual(msd_lag_one["standard_error"], 1.0)
        self.assertAlmostEqual(msd_lag_one["ci95_low"], 2.0 - 12.706204736432095)
        self.assertAlmostEqual(msd_lag_one["ci95_high"], 2.0 + 12.706204736432095)
        self.assertEqual(msd_lag_one["ci95_method"], "student_t_independent_replicates")

    def test_temperature_scan_requires_nonoverlapping_directional_intervals(self):
        high = [
            {"metric": "diffusion", "mean": 3.0, "ci95_low": 2.8, "ci95_high": 3.2},
            {"metric": "ngp_peak", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
            {"metric": "alpha_relaxation_time", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
            {"metric": "diffusion_alpha_product", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
            {"metric": "overlap_chi4_peak", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
        ]
        low = [
            {"metric": "diffusion", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
            {"metric": "ngp_peak", "mean": 1.5, "ci95_low": 1.3, "ci95_high": 1.7},
            {"metric": "alpha_relaxation_time", "mean": 1.5, "ci95_low": 1.3, "ci95_high": 1.7},
            {"metric": "diffusion_alpha_product", "mean": 1.5, "ci95_low": 1.3, "ci95_high": 1.7},
            {"metric": "overlap_chi4_peak", "mean": 1.5, "ci95_low": 1.3, "ci95_high": 1.7},
        ]

        rows = temperature_scan_verdict(high, low)

        diffusion = next(row for row in rows if row["metric"] == "diffusion")
        ngp = next(row for row in rows if row["metric"] == "ngp_peak")
        self.assertEqual(diffusion["effect"], "cooling_slowdown")
        self.assertAlmostEqual(diffusion["effect_ratio"], 3.0)
        self.assertTrue(diffusion["directional_ci95_separated"])
        self.assertEqual(ngp["effect"], "cooling_growth")
        self.assertAlmostEqual(ngp["effect_ratio"], 1.5)
        self.assertTrue(ngp["directional_ci95_separated"])

    def test_diffusion_estimate_uses_only_requested_heldout_origins(self):
        estimate = getattr(ka_replicates, "trajectory_diffusion_estimate", None)
        self.assertIsNotNone(estimate)
        positions = np.zeros((5, 2, 3), dtype=float)
        positions[:, :, 0] = np.arange(5)[:, None]

        diffusion = estimate(
            positions,
            lag=2,
            origin_stride=2,
            particle_mask=np.array([True, False]),
        )

        self.assertAlmostEqual(diffusion, 4.0 / 12.0)

    def test_heldout_transport_summary_rejects_event_clock_undercoverage(self):
        summarize = getattr(ka_replicates, "summarize_heldout_event_transport", None)
        self.assertIsNotNone(summarize)
        rows = [
            {
                "replicate": float(replicate),
                "observed_diffusion": observed,
                "uncorrelated_event_diffusion": 0.3 * observed,
                "correlated_event_diffusion": 0.4 * observed,
            }
            for replicate, observed in ((1, 1.0), (2, 1.1), (3, 0.9))
        ]

        summary, verdict = summarize(rows, minimum_coverage=0.8, maximum_coverage=1.2)

        correlated = next(row for row in summary if row["model"] == "correlated_event_clock")
        self.assertAlmostEqual(correlated["mean_coverage"], 0.4)
        self.assertEqual(verdict["heldout_transport_pass"], 0.0)
        self.assertEqual(verdict["primary_failure"], "correlated_event_clock_undercoverage")
        self.assertEqual(verdict["independent_replicate_count"], 3.0)

    def test_heldout_transport_summary_scores_cooperative_primary_model(self):
        rows = [
            {
                "replicate": float(replicate),
                "observed_diffusion": 1.0,
                "correlated_event_diffusion": 0.4,
                "cooperative_cluster_diffusion": coverage,
            }
            for replicate, coverage in ((1, 0.9), (2, 1.0), (3, 1.1))
        ]

        summary, verdict = ka_replicates.summarize_heldout_event_transport(
            rows,
            minimum_coverage=0.8,
            maximum_coverage=1.2,
            model_columns={
                "correlated_event_clock": "correlated_event_diffusion",
                "cooperative_cluster_response": "cooperative_cluster_diffusion",
            },
            primary_model="cooperative_cluster_response",
        )

        self.assertEqual([row["model"] for row in summary], [
            "correlated_event_clock",
            "cooperative_cluster_response",
        ])
        self.assertEqual(verdict["primary_model"], "cooperative_cluster_response")
        self.assertEqual(verdict["heldout_transport_pass"], 1.0)
        self.assertEqual(verdict["primary_failure"], "none")

    def test_load_lammps_dump_reconstructs_unwrapped_positions(self):
        dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-2 2
-2 2
-2 2
ITEM: ATOMS id type x y z ix iy iz
1 1 1.5 0 0 0 0 0
2 2 -1.5 1 0 0 0 0
ITEM: TIMESTEP
1000
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-2 2
-2 2
-2 2
ITEM: ATOMS id type x y z ix iy iz
1 1 -1.5 0 0 1 0 0
2 2 1.5 1 0 -1 0 0
"""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trajectory.lammpstrj"
            path.write_text(dump)

            trajectory = load_lammps_custom_trajectory(path)

        np.testing.assert_array_equal(trajectory["timesteps"], [0, 1000])
        np.testing.assert_array_equal(trajectory["particle_types"], [0, 1])
        np.testing.assert_allclose(trajectory["box_lengths"], [4.0, 4.0, 4.0])
        np.testing.assert_allclose(trajectory["wrapped_positions"][1, 0], [-1.5, 0.0, 0.0])
        np.testing.assert_allclose(trajectory["unwrapped_positions"][1, 0], [2.5, 0.0, 0.0])
        np.testing.assert_allclose(trajectory["unwrapped_positions"][1, 1], [-2.5, 1.0, 0.0])

    def test_initial_configuration_fs_uses_minimum_images_and_a_particles(self):
        box = np.array([10.0, 10.0, 10.0])
        reference = np.array([[4.9, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        displaced = np.array([[-4.9, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
        particle_mask = np.array([True, True, False])

        value = initial_configuration_fs(
            reference,
            displaced,
            box,
            particle_mask=particle_mask,
            wave_number=np.pi,
        )

        expected = (np.cos(0.2 * np.pi) + 4.0 + np.cos(2.0 * np.pi)) / 6.0
        self.assertAlmostEqual(value, expected)

    def test_independence_gate_rejects_correlated_frame_pair(self):
        positions = np.zeros((3, 2, 3), dtype=float)
        positions[1] = 0.01
        positions[2, :, 0] = np.array([0.5, -0.5])

        with self.assertRaisesRegex(ValueError, "not decorrelated"):
            validate_initial_frame_independence(
                positions,
                np.array([10.0, 10.0, 10.0]),
                np.array([True, True]),
                frame_indices=[0, 1, 2],
                wave_number=7.25,
                maximum_absolute_fs=0.2,
            )

    def test_prepare_replicate_writes_physical_protocol_and_provenance(self):
        box = np.array([[4.0, 4.0, 4.0]])
        types = np.array([[0, 0, 0, 1]])
        positions = [
            np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, 1.0, -1.0],
                ],
                dtype=np.float32,
            )
        ]
        payload = {"Box_size": box, "Particle_types": types, "Positions": positions}

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source.pkl"
            with source.open("wb") as handle:
                pickle.dump(payload, handle, protocol=4)
            output = root / "replicate"

            manifest = prepare_replicate(
                source,
                output,
                temperature=0.45,
                frame_index=0,
                velocity_seed=45117,
                equilibration_time=100.0,
                production_time=5000.0,
            )

            lammps_input = (output / "in.production").read_text()
            self.assertIn("pair_coeff 1 2 1.5 0.8 2.0", lammps_input)
            self.assertIn("pair_modify shift yes", lammps_input)
            self.assertIn("fix thermostat all nvt temp 0.45 0.45 10", lammps_input)
            self.assertIn("timestep 0.001", lammps_input)
            self.assertIn("dump trajectory all custom 1000", lammps_input)
            self.assertIn("restart 100000", lammps_input)
            self.assertIn("run 100000", lammps_input)
            self.assertIn("reset_timestep 0", lammps_input)
            self.assertIn("run 5000000", lammps_input)

            stored = json.loads((output / "manifest.json").read_text())
            self.assertEqual(stored, manifest)
            self.assertEqual(stored["source_doi"], "10.5281/zenodo.7469766")
            self.assertEqual(stored["independence_class"], "decorrelated_parent_frames_plus_velocity_seeds")
            self.assertFalse(stored["independently_prepared_parent_samples"])
            self.assertEqual(stored["saved_frame_interval_tau"], 1.0)
            self.assertEqual(stored["particle_counts"], {"A": 3, "B": 1, "total": 4})
            self.assertEqual(len(stored["source_sha256"]), 64)

    def test_prepare_ensemble_records_pairwise_decorrelation(self):
        rng = np.random.default_rng(17)
        payload = {
            "Box_size": np.array([[20.0, 20.0, 20.0]]),
            "Particle_types": np.array([[0] * 16 + [1] * 4]),
            "Positions": [rng.uniform(-10.0, 10.0, size=(20, 3)) for _ in range(2)],
        }
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source.pkl"
            with source.open("wb") as handle:
                pickle.dump(payload, handle, protocol=4)

            manifest = prepare_replicate_ensemble(
                source,
                root / "ensemble",
                temperature=0.7,
                frame_indices=[0, 1],
                velocity_seeds=[70117, 70139],
                maximum_absolute_fs=0.5,
                equilibration_time=0.1,
                production_time=1.0,
            )

            self.assertEqual(manifest["replicate_count"], 2)
            self.assertEqual(len(manifest["pairwise_initial_fs"]), 1)
            self.assertLess(abs(manifest["pairwise_initial_fs"][0]["fs"]), 0.5)
            self.assertTrue((root / "ensemble" / "replicate_01" / "in.production").is_file())
            self.assertTrue((root / "ensemble" / "replicate_02" / "manifest.json").is_file())
            self.assertEqual(
                json.loads((root / "ensemble" / "ensemble_manifest.json").read_text()),
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
