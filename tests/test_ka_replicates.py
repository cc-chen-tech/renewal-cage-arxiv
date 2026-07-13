import json
import importlib.util
import math
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
sys.path.insert(0, str(ROOT / "scripts"))

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
    prepare_isoconfigurational_langevin_clones,
    summarize_replicate_curves,
    temperature_scan_verdict,
    validate_initial_frame_independence,
)
from ka_local_cage import (  # noqa: E402
    frozen_minimum_environment_response,
    frozen_minimum_shell_environment_response,
    ensemble_displacement_observables,
    ka_frozen_neighbor_multistart_minima,
    driven_tagged_langevin_trajectory,
    driven_active_cluster_langevin_trajectory,
    active_cluster_source_velocity_ensemble,
    active_cluster_langevin_residual,
    nonlinear_core_harmonic_bath_langevin_trajectory,
    local_harmonic_displacement_memory_kernel,
    infer_causal_position_memory_kernel,
    finite_memory_position_response_holdout,
    fit_pair_force_response_auxiliary_embedding,
    fit_linear_response_auxiliary_embedding,
    force_generator_increment_diagnostic,
    ensemble_symmetric_response_summary,
    shared_response_prefix_length,
    propagate_causal_position_memory_response,
    symmetric_finite_difference_response,
    time_split_driven_response_conditioned_innovation_diagnostic,
    time_split_driven_response_empirical_innovation_diagnostic,
    time_split_graph_conditioned_increment_diagnostic,
    time_split_knn_state_conditioned_increment_diagnostic,
    time_split_force_auxiliary_markov_diagnostic,
    ka_local_cluster_hessian,
    ka_local_cluster_soft_mode_features,
    ka_local_cluster_anharmonic_barrier_features,
    frozen_minimum_response_jacobian,
    minimum_response_diagnostic,
    projected_overdamped_diffusion_tensor,
    projected_diffusion_increment_association,
    projected_overdamped_mixture_ngp,
    response_residual_memory_diagnostic,
    segmented_cage_center_event_statistics,
    ka_lj_local_energy_force_hessian,
    ka_lj_radial_derivatives,
    ka_lj_force_and_isotropic_curvature,
    ka_lj_force_generator_observables,
    ka_lj_second_force_generator,
    ka_lj_shell_forces,
    precursor_event_hazard_diagnostic,
    isoconfigurational_first_passage_diagnostic,
    isoconfigurational_state_rate_diagnostic,
    grouped_binomial_logistic_committor_diagnostic,
    heterogeneous_marked_poisson_prediction,
    underdamped_ou_cage_prediction,
    fit_underdamped_ou_cage_from_msd,
    hybrid_structural_clock_underdamped_cage_prediction,
    independent_gaussian_cage_marked_poisson_prediction,
    free_exponential_memory_gle_prediction,
    fit_free_exponential_memory_gle_from_msd,
    static_neighbor_cage_displacement,
    dynamic_neighbor_cage_displacement,
    hysteretic_neighbor_cage_displacement,
    local_bond_orientational_features,
    lagged_coordinate_increment_history,
    residual_ar1_memory_diagnostic,
    state_dependent_increment_diagnostic,
    time_split_empirical_increment_levy_diagnostic,
    time_split_ar1_empirical_innovation_diagnostic,
    time_split_arp_empirical_innovation_diagnostic,
    time_split_lagged_response_embedding_diagnostic,
    time_split_center_gle_diagnostic,
)
from ka_collective_memory import (  # noqa: E402
    discrete_volterra_memory_kernel,
    propagate_free_gle_velocity_correlation,
    long_wavelength_displacement_field,
    fourier_density_current_field,
    symmetric_quadratic_mode_products,
    local_affine_nonaffine_state,
    local_neighbor_velocity_field,
    local_neighbor_particle_velocity_field,
    discrete_mori_zwanzig_operators,
    simulate_discrete_mz_empirical_innovations,
    simulate_discrete_mz_block_innovations,
    time_split_shell_response_embedding_diagnostic,
    time_split_collective_velocity_diagnostic,
    time_split_affine_environment_embedding_diagnostic,
    time_split_collective_field_embedding_diagnostic,
    fixed_bath_particle_state,
    nearest_outer_bath_state,
)
from ka_generator_response import (  # noqa: E402
    extract_generator_response_path,
    fit_generator_constrained_response,
    generator_response_tangent_diagnostic,
    generator_response_lammps_input,
    matched_generator_response,
    right_censored_tangent_interval_mask,
    tangent_force_generator_noise_covariance_rate,
    tangent_noise_covariance_diagnostic,
)
from analyze_ka_active_cluster_residual import concatenate_residuals  # noqa: E402


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

    def test_posterior_weighted_displacement_kernel_links_fast_counts_to_large_motion(self):
        estimate = getattr(
            ka_replicates,
            "posterior_weighted_state_displacement_kernels",
            None,
        )
        self.assertIsNotNone(estimate)
        counts = np.array([0, 0, 1, 3, 4, 5])
        displacements = np.array(
            [
                [0.1, 0.0],
                [0.0, 0.1],
                [0.2, 0.0],
                [1.0, 0.0],
                [0.0, 1.2],
                [1.4, 0.0],
            ]
        )

        result = estimate(
            counts,
            displacements,
            slow_mean_count=0.2,
            fast_mean_count=3.5,
            stationary_slow_probability=0.6,
            stationary_fast_probability=0.4,
            wave_numbers=np.array([1.0]),
        )

        self.assertGreater(result["fast_effective_sample_size"], 2.0)
        self.assertGreater(result["slow_effective_sample_size"], 2.0)
        self.assertGreater(result["fast_msd"], 5.0 * result["slow_msd"])
        self.assertLess(result["fast_characteristic_k1"], result["slow_characteristic_k1"])

    def test_two_clock_state_displacement_propagates_markov_fourth_moment(self):
        propagate = getattr(
            ka_replicates,
            "two_clock_state_displacement_statistics",
            None,
        )
        self.assertIsNotNone(propagate)
        parameters = {
            "stationary_slow_probability": 0.7,
            "stationary_fast_probability": 0.3,
            "fast_clock_weight": 0.6,
            "slow_clock_weight": 0.4,
            "fast_clock_retention": 0.3,
            "slow_clock_retention": 0.9,
            "block_size": 20.0,
        }
        kernels = {
            "slow_msd": 0.2,
            "fast_msd": 1.0,
            "slow_fourth_moment": 0.08,
            "fast_fourth_moment": 2.0,
            "slow_characteristic_k1": 0.95,
            "fast_characteristic_k1": 0.65,
        }

        one = propagate(
            parameters,
            kernels,
            block_count=1,
            wave_number_key="k1",
            dimension=2,
        )
        two = propagate(
            parameters,
            kernels,
            block_count=2,
            wave_number_key="k1",
            dimension=2,
        )

        self.assertAlmostEqual(one["event_msd"], 0.7 * 0.2 + 0.3 * 1.0)
        self.assertAlmostEqual(one["event_fourth_moment"], 0.7 * 0.08 + 0.3 * 2.0)
        self.assertAlmostEqual(two["event_msd"], 2.0 * one["event_msd"])
        self.assertGreater(two["event_fourth_moment"], 2.0 * one["event_fourth_moment"])
        self.assertLess(two["event_characteristic"], one["event_characteristic"])

    def test_block_vector_correlation_curve_measures_cross_block_memory(self):
        correlate = getattr(ka_replicates, "block_vector_correlation_curve", None)
        self.assertIsNotNone(correlate)
        displacements = np.array(
            [
                [[1.0, 0.0], [-0.5, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, -0.5], [0.0, 1.0]],
            ]
        )

        rows = correlate(displacements, maximum_lag=1)

        self.assertAlmostEqual(rows[0]["block_vector_msd"], 0.75)
        self.assertAlmostEqual(rows[0]["block_dot_correlation"], -0.5)
        self.assertAlmostEqual(rows[0]["cumulative_green_kubo_factor"], -1.0 / 3.0)

        finite = getattr(ka_replicates, "finite_window_green_kubo_factor", None)
        self.assertIsNotNone(finite)
        self.assertAlmostEqual(finite(rows, block_count=1), 1.0)
        self.assertAlmostEqual(finite(rows, block_count=2), 1.0 / 3.0)

    def test_cumulative_block_observables_match_direct_windows(self):
        observe = getattr(ka_replicates, "cumulative_block_observables", None)
        self.assertIsNotNone(observe)
        blocks = np.array([[[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]]])

        result = observe(blocks, block_count=2, wave_numbers=np.array([1.0]))

        self.assertEqual(result["particle_window_count"], 2.0)
        self.assertAlmostEqual(result["msd"], 5.0)
        self.assertAlmostEqual(result["fourth_moment"], 25.0)
        self.assertAlmostEqual(result["ngp"], -0.5)
        self.assertAlmostEqual(
            result["characteristic_k1"],
            0.5 * (math.cos(1.0) + math.cos(2.0)),
        )

    def test_within_particle_shuffle_preserves_particle_vector_multisets(self):
        shuffle = getattr(ka_replicates, "within_particle_time_shuffle", None)
        self.assertIsNotNone(shuffle)
        blocks = np.arange(24, dtype=float).reshape(2, 4, 3)

        shuffled = shuffle(blocks, np.random.default_rng(7))

        for particle in range(blocks.shape[0]):
            original = sorted(map(tuple, blocks[particle]))
            randomized = sorted(map(tuple, shuffled[particle]))
            self.assertEqual(randomized, original)
        self.assertFalse(np.array_equal(shuffled, blocks))

    def test_direction_randomized_path_preserves_lengths_not_recoil(self):
        observe = getattr(
            ka_replicates,
            "direction_randomized_block_observables",
            None,
        )
        self.assertIsNotNone(observe)
        blocks = np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])

        result = observe(blocks, block_count=2, wave_numbers=np.array([1.0]))

        self.assertEqual(result["particle_window_count"], 1.0)
        self.assertAlmostEqual(result["msd"], 2.0)
        self.assertAlmostEqual(result["fourth_moment"], 16.0 / 3.0)
        self.assertAlmostEqual(result["ngp"], -0.2)
        self.assertAlmostEqual(
            result["characteristic_k1"],
            (math.sin(1.0) / 1.0) ** 2,
        )

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

        self.assertAlmostEqual(result["normalized_total_linear_change"], 0.4)
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

    def test_joint_state_kernel_does_not_double_count_external_cage_channel(self):
        script_path = ROOT / "scripts" / "predict_ka_hybrid_macro_observables.py"
        spec = importlib.util.spec_from_file_location(
            "predict_ka_hybrid_macro_observables_joint", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertTrue(module.kernel_uses_external_cage("correlated-global"))
        self.assertTrue(module.kernel_uses_external_cage("state-conditioned-finite-gk"))
        self.assertFalse(module.kernel_uses_external_cage("state-conditioned-joint"))
        self.assertFalse(module.kernel_uses_external_cage("state-conditioned-joint-finite-gk"))

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

    def test_state_kernel_crossover_selects_cooling_induced_multiblock_memory(self):
        script_path = ROOT / "scripts" / "summarize_ka_state_kernel_crossover.py"
        spec = importlib.util.spec_from_file_location(
            "summarize_ka_state_kernel_crossover", script_path
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        result = module.classify_state_kernel_crossover(
            low={
                "curve_transfer_pass": "0",
                "diffusion_relative_error": "0.059",
                "maximum_ensemble_ngp_absolute_error": "2.11",
                "maximum_ensemble_fs_absolute_error": "0.269",
            },
            high={
                "curve_transfer_pass": "1",
                "diffusion_relative_error": "0.08",
                "alpha_crossing_ready": "0",
                "maximum_ensemble_ngp_absolute_error": "0.15",
                "maximum_ensemble_fs_absolute_error": "0.021",
            },
        )

        self.assertEqual(result["high_temperature_curve_closure"], 1.0)
        self.assertEqual(result["low_temperature_curve_closure"], 0.0)
        self.assertEqual(result["diffusion_transfer_pass_both_temperatures"], 1.0)
        self.assertEqual(result["cooling_induced_higher_order_memory_required"], 1.0)
        self.assertEqual(result["additional_mobility_clock_supported"], 0.0)
        self.assertEqual(
            result["next_minimal_extension"],
            "non_markov_multiblock_orientation_cage_persistence_kernel",
        )

    def test_path_transfer_classifier_requires_curves_and_shuffle_precision(self):
        script_path = ROOT / "scripts" / "analyze_ka_empirical_path_transfer.py"
        spec = importlib.util.spec_from_file_location(
            "analyze_ka_empirical_path_transfer",
            script_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        summary_rows = [
            {
                "model": "contiguous_empirical_path",
                "lag": 100.0,
                "ensemble_msd_relative_error": 0.04,
                "ensemble_ngp_absolute_error": 0.05,
                "ensemble_absolute_error_fs_k2": 0.01,
                "ensemble_absolute_error_fs_k7p25": 0.02,
                "ensemble_msd_mc_relative_se": 0.0,
                "ensemble_ngp_mc_se": 0.0,
                "ensemble_fs_k2_mc_se": 0.0,
                "ensemble_fs_k7p25_mc_se": 0.0,
            },
            {
                "model": "within_particle_time_shuffle",
                "lag": 100.0,
                "ensemble_msd_relative_error": 0.06,
                "ensemble_ngp_absolute_error": 0.10,
                "ensemble_absolute_error_fs_k2": 0.01,
                "ensemble_absolute_error_fs_k7p25": 0.04,
                "ensemble_msd_mc_relative_se": 0.008,
                "ensemble_ngp_mc_se": 0.02,
                "ensemble_fs_k2_mc_se": 0.002,
                "ensemble_fs_k7p25_mc_se": 0.004,
            },
        ]
        replicate_rows = [
            {
                "model": model,
                "replicate": float(replicate),
                "ngp_absolute_error": error,
                "absolute_error_fs_k2": error / 10.0,
                "absolute_error_fs_k7p25": error / 10.0,
            }
            for replicate, contiguous, shuffled in (
                (1, 0.10, 0.40),
                (2, 0.12, 0.35),
                (3, 0.20, 0.15),
            )
            for model, error in (
                ("contiguous_empirical_path", contiguous),
                ("within_particle_time_shuffle", shuffled),
            )
        ]

        verdicts = {
            row["model"]: row
            for row in module.classify_path_model_transfer(
                summary_rows,
                replicate_rows=replicate_rows,
            )
        }

        self.assertEqual(verdicts["contiguous_empirical_path"]["curve_transfer_pass"], 1.0)
        self.assertEqual(verdicts["within_particle_time_shuffle"]["curve_transfer_pass"], 0.0)
        self.assertEqual(verdicts["within_particle_time_shuffle"]["shuffle_precision_pass"], 0.0)
        self.assertEqual(verdicts["contiguous_empirical_path"]["paired_contiguous_better_replicate_count"], 2.0)
        self.assertEqual(verdicts["contiguous_empirical_path"]["heldout_path_used_in_prediction"], 0.0)
        self.assertEqual(verdicts["contiguous_empirical_path"]["macro_fit_parameter_count"], 0.0)
        self.assertEqual(verdicts["contiguous_empirical_path"]["microdynamic_closure_claim_allowed"], 0.0)
        self.assertEqual(verdicts["contiguous_empirical_path"]["spatial_facilitation_claim_allowed"], 0.0)
        self.assertEqual(verdicts["contiguous_empirical_path"]["thermodynamic_claim_allowed"], 0.0)

    def test_path_transfer_seed_is_deterministic_and_replicate_specific(self):
        script_path = ROOT / "scripts" / "analyze_ka_empirical_path_transfer.py"
        spec = importlib.util.spec_from_file_location(
            "analyze_ka_empirical_path_transfer_seed",
            script_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        first = module.path_shuffle_seed(45101, replicate=2, realization=3)

        self.assertEqual(first, module.path_shuffle_seed(45101, replicate=2, realization=3))
        self.assertNotEqual(first, module.path_shuffle_seed(45101, replicate=3, realization=3))
        self.assertNotEqual(first, module.path_shuffle_seed(45101, replicate=2, realization=4))

    def test_empirical_path_crossover_requires_shared_higher_order_failure(self):
        script_path = ROOT / "scripts" / "summarize_ka_empirical_path_transfer.py"
        spec = importlib.util.spec_from_file_location(
            "summarize_ka_empirical_path_transfer",
            script_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        low = [
            {
                "model": "contiguous_empirical_path",
                "curve_transfer_pass": "1",
                "maximum_ensemble_ngp_absolute_error": "0.04",
                "maximum_ensemble_fs_absolute_error": "0.02",
                "paired_contiguous_better_replicate_count": "2",
                "paired_replicate_count": "3",
                "shuffle_precision_pass": "1",
            },
            {
                "model": "within_particle_time_shuffle",
                "curve_transfer_pass": "0",
                "maximum_ensemble_ngp_absolute_error": "1.2",
                "maximum_ensemble_fs_absolute_error": "0.18",
                "paired_contiguous_better_replicate_count": "2",
                "paired_replicate_count": "3",
                "shuffle_precision_pass": "1",
            },
            {
                "model": "direction_randomized_path",
                "curve_transfer_pass": "0",
                "maximum_ensemble_ngp_absolute_error": "1.0",
                "maximum_ensemble_fs_absolute_error": "0.15",
                "paired_contiguous_better_replicate_count": "2",
                "paired_replicate_count": "3",
                "shuffle_precision_pass": "1",
            },
        ]
        high = [
            {
                "model": "contiguous_empirical_path",
                "curve_transfer_pass": "1",
                "maximum_ensemble_ngp_absolute_error": "0.03",
                "maximum_ensemble_fs_absolute_error": "0.02",
            }
        ]
        markov_low = {
            "curve_transfer_pass": "0",
            "maximum_ensemble_ngp_absolute_error": "2.1",
            "maximum_ensemble_fs_absolute_error": "0.27",
        }

        result = module.classify_empirical_path_crossover(
            low,
            high,
            markov_low,
        )

        self.assertEqual(result["shared_low_temperature_higher_order_failure"], 1.0)
        self.assertEqual(result["replicate_consensus_pass"], 1.0)
        self.assertEqual(result["single_particle_multiblock_path_memory_required"], 1.0)
        self.assertEqual(result["amplitude_persistence_alone_sufficient"], 0.0)
        self.assertEqual(result["ordered_recoil_path_required"], 1.0)
        self.assertEqual(
            result["next_minimal_extension"],
            "conditional_reversible_cage_path_kernel",
        )
        self.assertEqual(result["microdynamic_closure_claim_allowed"], 0.0)
        self.assertEqual(result["spatial_facilitation_claim_allowed"], 0.0)
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)

    def test_ka_c3_switched_radial_derivatives_match_lj_and_vanish_through_third_order(self):
        epsilon = 1.0
        sigma = 1.0
        inner = 2.0 * sigma
        cutoff = 2.5 * sigma

        switched_inner = ka_lj_radial_derivatives(
            np.array([inner]),
            epsilon=epsilon,
            sigma=sigma,
            protocol="ka_lj_c3_switch",
        )
        lj_inner = ka_lj_radial_derivatives(
            np.array([inner]),
            epsilon=epsilon,
            sigma=sigma,
            protocol="ka_lj_cut",
        )
        for switched, lj in zip(switched_inner, lj_inner):
            np.testing.assert_allclose(switched, lj, rtol=0.0, atol=1e-13)

        switched_outer = ka_lj_radial_derivatives(
            np.array([cutoff, cutoff + 0.1]),
            epsilon=epsilon,
            sigma=sigma,
            protocol="ka_lj_c3_switch",
        )
        for derivative in switched_outer:
            np.testing.assert_allclose(derivative, 0.0, rtol=0.0, atol=1e-13)

        offset = 1e-7
        inner_sides = ka_lj_radial_derivatives(
            np.array([inner - offset, inner + offset]),
            epsilon=epsilon,
            sigma=sigma,
            protocol="ka_lj_c3_switch",
        )
        outer_inside = ka_lj_radial_derivatives(
            np.array([cutoff - offset]),
            epsilon=epsilon,
            sigma=sigma,
            protocol="ka_lj_c3_switch",
        )
        for derivative in inner_sides:
            self.assertLess(abs(float(derivative[1] - derivative[0])), 2e-3)
        for derivative in outer_inside:
            self.assertLess(abs(float(derivative[0])), 2e-3)

    def test_ka_c3_switched_radial_derivatives_form_one_consistent_derivative_chain(self):
        epsilon = 1.5
        sigma = 0.8
        step = 1e-5
        for radius in (1.1 * sigma, 2.25 * sigma):
            center = ka_lj_radial_derivatives(
                np.array([radius]),
                epsilon=epsilon,
                sigma=sigma,
                protocol="ka_lj_c3_switch",
            )
            plus = ka_lj_radial_derivatives(
                np.array([radius + step]),
                epsilon=epsilon,
                sigma=sigma,
                protocol="ka_lj_c3_switch",
            )
            minus = ka_lj_radial_derivatives(
                np.array([radius - step]),
                epsilon=epsilon,
                sigma=sigma,
                protocol="ka_lj_c3_switch",
            )
            for order in range(3):
                finite_difference = (plus[order] - minus[order]) / (2.0 * step)
                np.testing.assert_allclose(
                    finite_difference,
                    center[order + 1],
                    rtol=3e-5,
                    atol=3e-7,
                )

    def test_nearest_outer_bath_state_excludes_active_particles_and_uses_minimum_image(self):
        positions = np.array(
            [
                [
                    [9.8, 0.0, 0.0],
                    [9.9, 0.0, 0.0],
                    [0.2, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [8.9, 0.0, 0.0],
                ],
                [
                    [9.8, 0.0, 0.0],
                    [9.9, 0.0, 0.0],
                    [0.4, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [9.4, 0.0, 0.0],
                ],
            ]
        )
        velocities = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )

        state = nearest_outer_bath_state(
            positions,
            velocities=velocities,
            target_index=0,
            active_indices=np.array([0, 1]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
            neighbor_count=2,
        )

        np.testing.assert_array_equal(state["particle_indices"], [[2, 4], [4, 2]])
        np.testing.assert_allclose(state["relative_positions"][0, :, 0], [0.4, -0.9])
        np.testing.assert_allclose(state["relative_positions"][1, :, 0], [-0.4, 0.6])
        np.testing.assert_allclose(state["relative_velocities"][0, :, 0], [1.0, 3.0])

    def test_fixed_bath_particle_state_keeps_labels_and_uses_minimum_image(self):
        positions = np.array(
            [
                [[9.8, 0.0, 0.0], [0.2, 0.0, 0.0], [8.9, 0.0, 0.0]],
                [[9.7, 0.0, 0.0], [0.4, 0.0, 0.0], [8.6, 0.0, 0.0]],
            ]
        )
        velocities = np.array(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [4.5, 0.0, 0.0]],
            ]
        )

        state = fixed_bath_particle_state(
            positions,
            velocities=velocities,
            target_index=0,
            particle_indices=np.array([2, 1]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
        )

        np.testing.assert_array_equal(state["particle_indices"], [2, 1])
        np.testing.assert_allclose(state["relative_positions"][:, :, 0], [[-0.9, 0.4], [-1.1, 0.7]])
        np.testing.assert_allclose(state["relative_velocities"][:, :, 0], [[3.0, 1.0], [3.0, 1.0]])

    def test_ka_lj_shell_forces_sum_to_exact_pair_force(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.8, 0.0, 0.0]])
        types = np.array([0, 0, 1])
        target = np.array([0])
        pair_force, _ = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=types,
            box_lengths=np.array([12.0, 12.0, 12.0]),
            target_indices=target,
        )

        shell_force = ka_lj_shell_forces(
            positions,
            particle_types=types,
            box_lengths=np.array([12.0, 12.0, 12.0]),
            target_indices=target,
            shell_edges=np.array([0.0, 1.5, 2.5]),
        )

        np.testing.assert_allclose(np.sum(shell_force, axis=1), pair_force, atol=1e-12)

    def test_force_auxiliary_markov_diagnostic_recovers_synthetic_colored_force_process(self):
        rng = np.random.default_rng(959)
        frames, particles = 180, 500
        state = np.zeros((frames, particles, 2, 3))
        transition = np.array([[0.78, 0.06], [-0.22, 0.71]])
        for index in range(frames - 1):
            state[index + 1] = np.einsum("ab,pbc->pac", transition, state[index])
            state[index + 1] += rng.normal(scale=np.array([0.025, 0.06])[None, :, None], size=(particles, 2, 3))
        velocity, force = state[:, :, 0], state[:, :, 1]
        positions = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(velocity[:-1], axis=0)], axis=0)

        result = time_split_force_auxiliary_markov_diagnostic(
            positions,
            velocity,
            force,
            train_stop=90,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8]),
            wave_numbers=np.array([0.5, 1.0]),
            simulation_count=3000,
            seed=960,
        )

        np.testing.assert_allclose(result["transition_matrix"], transition, atol=0.04)
        self.assertGreater(float(result["heldout_state_r_squared"]), 0.35)
        self.assertLess(float(result["diffusion_relative_error"]), 0.15)

    def test_discrete_volterra_memory_kernel_recovers_causal_trapezoid_kernel(self):
        frame_time = 0.05
        expected_kernel = np.array([1.4, 0.6, -0.15, 0.04, 0.0])
        correlation = propagate_free_gle_velocity_correlation(
            expected_kernel,
            frame_time=frame_time,
            output_count=24,
        )

        result = discrete_volterra_memory_kernel(correlation, frame_time=frame_time, kernel_count=len(expected_kernel))

        np.testing.assert_allclose(result["kernel"], expected_kernel, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result["reconstructed_correlation"], correlation[: len(expected_kernel) + 1], rtol=1e-10, atol=1e-10)

    def test_lagged_coordinate_increment_history_is_causal_and_orders_recent_first(self):
        centers = np.array(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0]],
                [[1.0, 2.0, 0.0]],
                [[1.0, 2.0, 3.0]],
            ]
        )

        history = lagged_coordinate_increment_history(centers, order=2)

        self.assertTrue(np.isnan(history[:2]).all())
        np.testing.assert_allclose(history[2, 0], [0.0, 2.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(history[3, 0], [0.0, 0.0, 3.0, 0.0, 2.0, 0.0])

    def test_knn_state_conditioned_increment_diagnostic_recovers_continuous_state_law(self):
        rng = np.random.default_rng(947)
        frames, particles = 180, 80
        state = rng.normal(size=(frames, particles, 2))
        scale = 0.015 + 0.04 / (1.0 + np.exp(-state[:-1, :, :1]))
        drift = np.concatenate([0.02 * state[:-1, :, :1], np.zeros((frames - 1, particles, 2))], axis=2)
        increments = drift + rng.normal(size=(frames - 1, particles, 3)) * scale
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(increments, axis=0)])
        result = time_split_knn_state_conditioned_increment_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            state,
            train_stop=90,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8]),
            wave_numbers=np.array([0.5, 1.0]),
            neighbor_count=24,
            seed=948,
        )

        self.assertLess(float(result["diffusion_relative_error"]), 0.15)
        self.assertLess(float(result["knn_state_fs_max_relative_error"]), 0.10)
        self.assertLess(float(result["knn_state_ngp_max_absolute_error"]), 0.12)

    def test_graph_conditioned_increment_diagnostic_recovers_state_dependent_independent_increments(self):
        rng = np.random.default_rng(945)
        frames, particles = 260, 600
        state = rng.random((frames, particles)) < 0.25
        increments = rng.normal(size=(frames - 1, particles, 3)) * np.where(state[:-1, :, None], 0.07, 0.02)
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(increments, axis=0)])
        result = time_split_graph_conditioned_increment_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            state.astype(float),
            train_stop=130,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8, 16]),
            wave_numbers=np.array([0.5, 1.0]),
            seed=946,
        )

        self.assertLess(float(result["diffusion_relative_error"]), 0.12)
        self.assertLess(float(result["graph_conditioned_fs_max_relative_error"]), 0.08)
        self.assertLess(float(result["graph_conditioned_ngp_max_absolute_error"]), 0.10)

    def test_driven_response_conditioned_innovation_recovers_response_dependent_noise(self):
        rng = np.random.default_rng(943)
        frames, particles = 260, 700
        response = rng.normal(scale=0.04, size=(frames - 1, particles, 3))
        amplitude = np.linalg.norm(response, axis=2, keepdims=True)
        innovation = 0.25 * response + rng.normal(size=response.shape) * (0.01 + 0.20 * amplitude)
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(response + innovation, axis=0)])
        result = time_split_driven_response_conditioned_innovation_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            response,
            np.ones(response.shape[:2], dtype=bool),
            train_stop=130,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8, 16]),
            wave_numbers=np.array([0.5, 1.0]),
            bin_count=4,
            seed=944,
        )

        self.assertLess(float(result["diffusion_relative_error"]), 0.12)
        self.assertLess(float(result["driven_response_fs_max_relative_error"]), 0.08)
        self.assertLess(float(result["driven_response_ngp_max_absolute_error"]), 0.10)

    def test_driven_response_empirical_innovation_recovers_iid_response_plus_noise(self):
        rng = np.random.default_rng(941)
        frames, particles = 220, 500
        response = rng.normal(scale=0.04, size=(frames - 1, particles, 3))
        innovation = rng.normal(scale=0.025, size=(frames - 1, particles, 3))
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(response + innovation, axis=0)])
        result = time_split_driven_response_empirical_innovation_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            response,
            np.ones(response.shape[:2], dtype=bool),
            train_stop=110,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8, 16]),
            wave_numbers=np.array([0.5, 1.0]),
            seed=942,
        )

        self.assertLess(float(result["diffusion_relative_error"]), 0.10)
        self.assertLess(float(result["driven_response_fs_max_relative_error"]), 0.05)
        self.assertLess(float(result["driven_response_ngp_max_absolute_error"]), 0.05)

    def test_local_harmonic_memory_kernel_recovers_one_mode_green_function(self):
        hessian = np.zeros((6, 6))
        hessian[:3, :3] = 5.0 * np.eye(3)
        hessian[3:, 3:] = 2.0 * np.eye(3)
        hessian[0, 3] = hessian[3, 0] = 1.0
        times = np.array([0.0, 0.1])
        result = local_harmonic_displacement_memory_kernel(
            hessian,
            target_local_index=0,
            times=times,
            mass=1.0,
            damping=0.0,
        )

        expected = math.sin(math.sqrt(2.0) * 0.1) / math.sqrt(2.0)
        np.testing.assert_allclose(result["memory_kernel"][0], np.zeros((3, 3)), atol=1e-12)
        self.assertAlmostEqual(float(result["memory_kernel"][1, 0, 0]), expected, delta=1e-12)
        np.testing.assert_allclose(result["memory_kernel"][1, 1:, :], np.zeros((2, 3)), atol=1e-12)
        np.testing.assert_allclose(result["adiabatic_effective_hessian"], np.diag([4.5, 5.0, 5.0]), atol=1e-12)

    def test_symmetric_finite_difference_response_recovers_odd_linear_response(self):
        baseline = np.array(
            [
                [[1.0, -2.0, 0.5], [0.2, 0.1, -0.3]],
                [[0.4, 0.8, -0.6], [1.3, -0.7, 0.9]],
            ]
        )
        kernel = np.array(
            [
                [[2.0, 0.0, -1.0], [0.5, -1.5, 0.25]],
                [[-0.2, 0.7, 1.2], [0.8, 0.4, -0.9]],
            ]
        )
        displacement = 2.5e-3
        result = symmetric_finite_difference_response(
            baseline + displacement * kernel,
            baseline - displacement * kernel,
            displacement=displacement,
        )

        np.testing.assert_allclose(result["response"], kernel, atol=1e-12)
        np.testing.assert_allclose(result["even_component"], baseline, atol=1e-12)

    def test_ensemble_symmetric_response_summary_returns_mean_and_standard_error(self):
        responses = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[3.0, 6.0], [5.0, 8.0]],
                [[5.0, 10.0], [7.0, 12.0]],
            ]
        )

        result = ensemble_symmetric_response_summary(responses)

        np.testing.assert_allclose(result["mean"], [[3.0, 6.0], [5.0, 8.0]], atol=1e-12)
        expected_sem = np.std(responses, axis=0, ddof=1) / math.sqrt(3.0)
        np.testing.assert_allclose(result["standard_error"], expected_sem, atol=1e-12)
        self.assertEqual(int(result["member_count"]), 3)

    def test_shared_response_prefix_length_accepts_one_member_with_a_longer_tail(self):
        length = shared_response_prefix_length(
            [
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0, 2.0, 3.0]),
            ]
        )

        self.assertEqual(length, 3)

    def test_causal_position_memory_inference_recovers_discrete_kernel(self):
        frame_time = 0.1
        curvature = 2.0
        position_response = np.array([1.0, 0.92, 0.75, 0.51, 0.23])
        expected_memory = np.array([0.0, 0.7, -0.2, 0.15, 0.05])
        force_response = -curvature * position_response
        for frame in range(1, len(position_response)):
            force_response[frame] += frame_time * np.dot(
                expected_memory[1 : frame + 1], position_response[frame - 1 :: -1]
            )

        result = infer_causal_position_memory_kernel(
            position_response,
            force_response,
            frame_time=frame_time,
        )

        self.assertAlmostEqual(float(result["instantaneous_curvature"]), curvature, delta=1e-12)
        np.testing.assert_allclose(result["memory_kernel"], expected_memory, atol=1e-12)

    def test_position_memory_propagator_recovers_harmonic_limit(self):
        frame_time = 1e-3
        result = propagate_causal_position_memory_response(
            instantaneous_curvature=4.0,
            memory_kernel=np.zeros(2),
            frame_time=frame_time,
            mass=1.0,
            friction=0.0,
            initial_position_response=1.0,
            initial_velocity_response=0.0,
            frame_count=201,
        )

        times = np.arange(201) * frame_time
        np.testing.assert_allclose(result["position_response"], np.cos(2.0 * times), atol=2e-6)
        np.testing.assert_allclose(result["velocity_response"], -2.0 * np.sin(2.0 * times), atol=5e-6)

    def test_finite_memory_holdout_recovers_zero_memory_harmonic_response(self):
        generated = propagate_causal_position_memory_response(
            instantaneous_curvature=4.0,
            memory_kernel=np.zeros(2),
            frame_time=1e-3,
            mass=1.0,
            friction=1.0,
            frame_count=101,
        )
        result = finite_memory_position_response_holdout(
            generated["position_response"],
            -4.0 * generated["position_response"],
            frame_time=1e-3,
            mass=1.0,
            friction=1.0,
            fit_frames=20,
        )

        self.assertLess(float(result["heldout_relative_l2_error"]), 1e-12)
        self.assertLess(float(result["heldout_maximum_absolute_error"]), 1e-12)

    def test_pair_force_response_auxiliary_embedding_recovers_stable_linear_system(self):
        transition = np.array(
            [
                [0.95, 0.03, 0.00],
                [-0.10, 0.90, 0.02],
                [-0.30, 0.08, 0.75],
            ]
        )
        states = np.empty((80, 3))
        states[0] = [1.0, 0.0, -2.0]
        for frame in range(1, len(states)):
            states[frame] = transition @ states[frame - 1]

        result = fit_pair_force_response_auxiliary_embedding(states, fit_frames=30)

        np.testing.assert_allclose(result["transition_matrix"], transition, atol=1e-10)
        np.testing.assert_allclose(result["predicted_state_response"], states, atol=1e-9)
        self.assertLess(float(result["heldout_position_relative_l2_error"]), 1e-10)

    def test_linear_response_auxiliary_embedding_recovers_four_coordinate_system(self):
        transition = np.array(
            [
                [0.94, 0.04, 0.00, 0.00],
                [-0.08, 0.88, 0.03, 0.00],
                [-0.20, 0.05, 0.70, 0.06],
                [0.10, -0.03, -0.15, 0.66],
            ]
        )
        states = np.empty((90, 4))
        states[0] = [1.0, 0.1, -2.0, 0.5]
        for frame in range(1, len(states)):
            states[frame] = transition @ states[frame - 1]

        result = fit_linear_response_auxiliary_embedding(states, fit_frames=35)

        np.testing.assert_allclose(result["transition_matrix"], transition, atol=1e-10)
        np.testing.assert_allclose(result["predicted_state_response"], states, atol=1e-9)

    def test_driven_tagged_langevin_recovers_force_free_ballistic_limit(self):
        bath = np.array(
            [
                [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ]
        )
        result = driven_tagged_langevin_trajectory(
            bath,
            particle_types=np.array([0, 0]),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            target_index=0,
            initial_positions=np.array([[0.0, 0.0, 0.0]]),
            initial_velocities=np.array([[1.0, 0.0, 0.0]]),
            saved_frame_time=0.1,
            integration_time_step=0.01,
            mass=1.0,
            friction=0.0,
            temperature=0.0,
            rng=np.random.default_rng(101),
        )

        np.testing.assert_allclose(result["positions"][:, 0, 0], [0.0, 0.1, 0.2], atol=1e-12)
        np.testing.assert_allclose(result["velocities"][:, 0, 0], 1.0, atol=1e-12)

    def test_driven_active_cluster_langevin_recovers_force_free_ballistic_limit(self):
        external_bath = np.array(
            [
                [[8.0, 0.0, 0.0]],
                [[8.0, 0.0, 0.0]],
                [[8.0, 0.0, 0.0]],
            ]
        )
        result = driven_active_cluster_langevin_trajectory(
            external_bath,
            active_particle_types=np.array([0, 0]),
            external_particle_types=np.array([0]),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            initial_positions=np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]),
            initial_velocities=np.array([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
            saved_frame_time=0.1,
            integration_time_step=0.01,
            mass=1.0,
            friction=0.0,
            temperature=0.0,
            rng=np.random.default_rng(102),
        )

        np.testing.assert_allclose(result["positions"][:, 0, :, 0], [[0.0, 4.0], [0.1, 3.9], [0.2, 3.8]], atol=1e-12)

    def test_active_cluster_source_velocity_ensemble_repeats_selected_source_velocities(self):
        source_velocity = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=float
        )

        initial = active_cluster_source_velocity_ensemble(source_velocity, np.array([2, 0]), replicas=3)

        self.assertEqual(initial.shape, (3, 2, 3))
        np.testing.assert_allclose(initial[:, 0], np.repeat([[7.0, 8.0, 9.0]], 3, axis=0))
        np.testing.assert_allclose(initial[:, 1], np.repeat([[1.0, 2.0, 3.0]], 3, axis=0))

    def test_ensemble_displacement_observables_uses_all_samples_for_ngp_and_axis_averaged_fs(self):
        displacement = np.array(
            [[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]], [[0.0, -2.0, 0.0]]]
        )

        result = ensemble_displacement_observables(displacement, wave_numbers=np.array([1.0]))

        self.assertAlmostEqual(float(result["msd"]), 2.5)
        self.assertAlmostEqual(float(result["ngp"]), -0.184)
        self.assertAlmostEqual(float(result["fs_k_1"]), (4.0 + math.cos(1.0) + math.cos(2.0)) / 6.0)

    def test_active_cluster_langevin_residual_separates_included_and_omitted_forces(self):
        positions = np.repeat(
            np.array([[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [7.0, 0.0, 0.0]]]),
            3,
            axis=0,
        )
        velocities = np.zeros_like(positions)
        common = {
            "positions": positions,
            "velocities": velocities,
            "particle_types": np.array([0, 0, 0]),
            "box_lengths": np.array([20.0, 20.0, 20.0]),
            "active_indices": np.array([0]),
            "frame_time": 0.1,
            "friction": 0.0,
        }

        included = active_cluster_langevin_residual(
            external_indices=np.array([1, 2]), **common
        )
        omitted_neighbor = active_cluster_langevin_residual(
            external_indices=np.array([2]), **common
        )

        np.testing.assert_allclose(included["omitted_force"], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            omitted_neighbor["omitted_force"], omitted_neighbor["full_force"], atol=1e-12
        )

    def test_chunked_active_cluster_residual_keeps_force_and_residual_time_axes_aligned(self):
        positions = np.zeros((6, 2, 3))
        positions[:, 1, 0] = 8.0
        result = concatenate_residuals(
            positions,
            np.zeros_like(positions),
            particle_types=np.array([0, 0]),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            active_indices=np.array([0]),
            external_indices=np.empty(0, dtype=int),
            frame_time=0.1,
            friction=0.0,
            chunk_frames=2,
        )

        self.assertEqual(result["full_force"].shape[0], 6)
        self.assertEqual(result["omitted_force"].shape[0], 6)
        self.assertEqual(result["full_residual"].shape[0], 5)
        self.assertEqual(result["retained_residual"].shape[0], 5)

    def test_nonlinear_core_harmonic_bath_recovers_force_free_ballistic_limit(self):
        result = nonlinear_core_harmonic_bath_langevin_trajectory(
            np.empty((0, 3)),
            core_particle_types=np.array([0]),
            bath_particle_types=np.array([0]),
            fixed_outer_particle_types=np.empty(0, dtype=int),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            initial_core_positions=np.array([[[0.0, 0.0, 0.0]]]),
            initial_bath_positions=np.array([[[8.0, 0.0, 0.0]]]),
            initial_core_velocities=np.array([[[1.0, 0.0, 0.0]]]),
            initial_bath_velocities=np.zeros((1, 1, 3)),
            bath_initial_forces=np.zeros((1, 3)),
            bath_core_hessian=np.zeros((3, 3)),
            bath_hessian=np.zeros((3, 3)),
            saved_frame_time=0.1,
            integration_time_step=0.01,
            mass=1.0,
            friction=0.0,
            temperature=0.0,
            rng=np.random.default_rng(145),
        )

        np.testing.assert_allclose(result["core_positions"][:, 0, 0], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], atol=1e-12)

    def test_frozen_neighbor_multistart_recovers_one_symmetric_local_minimum(self):
        radius = 2.0 ** (1.0 / 6.0)
        environment = np.vstack(
            [
                np.zeros(3),
                radius * np.eye(3),
                -radius * np.eye(3),
            ]
        )
        result = ka_frozen_neighbor_multistart_minima(
            environment,
            particle_types=np.zeros(len(environment), dtype=int),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            target_index=0,
            seed_offsets=np.vstack([np.zeros(3), 0.08 * np.eye(3), -0.08 * np.eye(3)]),
        )

        self.assertEqual(int(result["minimum_count"]), 1)
        np.testing.assert_allclose(result["centers"][0], np.zeros(3), atol=1e-8)
        self.assertAlmostEqual(float(result["energies"][0]), -6.0, delta=1e-8)
        np.testing.assert_array_equal(result["seed_minimum_index"], np.zeros(7, dtype=int))

    def test_affine_environment_embedding_recovers_synthetic_causal_coupling(self):
        rng = np.random.default_rng(877)
        frames, particles = 420, 160
        affine = rng.normal(scale=0.08, size=(frames - 1, particles, 3, 3))
        d2min = np.exp(rng.normal(scale=0.25, size=(frames - 1, particles)))
        velocity = np.zeros((frames - 1, particles, 3))
        for index in range(1, len(velocity) - 1):
            velocity[index + 1] = (
                0.2 * velocity[index]
                + 0.7 * affine[index - 1, :, :, 0]
                + 0.12 * np.log(d2min[index - 1])[:, None]
                + rng.normal(scale=0.01, size=(particles, 3))
            )
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(velocity, axis=0)])
        result = time_split_affine_environment_embedding_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            affine,
            d2min,
            np.ones(d2min.shape, dtype=bool),
            train_stop=210,
        )

        self.assertGreater(float(result["heldout_affine_r_squared"]), 0.9)
        self.assertGreater(float(result["heldout_affine_r_squared_gain"]), 0.8)

    def test_local_bond_orientational_features_recovers_even_order_axial_pair(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        )
        result = local_bond_orientational_features(
            positions,
            particle_types=np.array([0, 0, 1, 1]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
            target_indices=np.array([0]),
            cutoff=1.5,
            orders=(4, 6, 8),
        )

        self.assertEqual(result["feature_names"], ("coordination_A", "coordination_B", "Q4", "Q6", "Q8"))
        np.testing.assert_allclose(result["features"][0], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_free_exponential_memory_gle_fit_recovers_synthetic_cage_msd(self):
        times = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 8.0, 20.0])
        expected = free_exponential_memory_gle_prediction(
            times,
            temperature=0.58,
            mass=14.0,
            memory_amplitude=95.0,
            memory_rate=3.5,
        )

        fit = fit_free_exponential_memory_gle_from_msd(
            times,
            expected["predicted_msd"],
            temperature=0.58,
        )

        np.testing.assert_allclose(fit["predicted_msd"], expected["predicted_msd"], rtol=0.03)
        self.assertLess(float(fit["relative_mse"]), 1e-3)

    def test_static_neighbor_cage_displacement_tracks_initial_neighbor_motion(self):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0], [4.4, 0.0, 0.0]],
            ]
        )
        result = static_neighbor_cage_displacement(
            positions,
            target_indices=np.array([0]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
            cutoff=1.5,
        )

        np.testing.assert_allclose(result["cage_displacement"][:, 0, 0], [0.0, 0.3])
        np.testing.assert_allclose(result["relative_displacement"][:, 0, 0], [0.0, -0.2])
        self.assertEqual(int(result["neighbor_count"][0]), 1)

    def test_dynamic_neighbor_cage_displacement_tracks_current_neighbor_graph(self):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [2.3, 0.0, 0.0], [1.2, 0.0, 0.0]],
            ]
        )
        result = dynamic_neighbor_cage_displacement(
            positions,
            target_indices=np.array([0]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
            cutoff=1.5,
        )

        np.testing.assert_allclose(result["cage_displacement"][:, 0, 0], [0.0, 0.2])
        np.testing.assert_allclose(result["relative_displacement"][:, 0, 0], [0.0, -0.1])
        np.testing.assert_allclose(
            result["tagged_displacement"], result["cage_displacement"] + result["relative_displacement"], atol=1e-12
        )
        np.testing.assert_array_equal(result["neighbor_count"][:, 0], [1, 1])

    def test_hysteretic_neighbor_cage_retains_neighbor_until_outer_cutoff(self):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.4, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.7, 0.0, 0.0], [1.2, 0.0, 0.0]],
            ]
        )
        result = hysteretic_neighbor_cage_displacement(
            positions,
            target_indices=np.array([0]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
            inner_cutoff=1.3,
            outer_cutoff=1.6,
        )

        np.testing.assert_allclose(result["cage_displacement"][:, 0, 0], [0.0, 0.5, 0.2])
        np.testing.assert_allclose(
            result["tagged_displacement"], result["cage_displacement"] + result["relative_displacement"], atol=1e-12
        )
        np.testing.assert_array_equal(result["neighbor_count"][:, 0], [1, 1, 1])
        np.testing.assert_array_equal(result["neighbor_membership_change_count"][:, 0], [0, 0, 2])

    def test_underdamped_ou_fit_recovers_synthetic_cage_msd(self):
        times = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.2])
        expected = underdamped_ou_cage_prediction(
            times,
            temperature=0.58,
            mass=1.0,
            stiffness=3.2,
            damping=1.1,
        )

        fit = fit_underdamped_ou_cage_from_msd(
            times,
            expected["predicted_msd"],
            temperature=0.58,
            mass=1.0,
        )

        self.assertAlmostEqual(float(fit["stiffness"]), 3.2, delta=0.35)
        self.assertAlmostEqual(float(fit["damping"]), 1.1, delta=0.20)
        np.testing.assert_allclose(fit["predicted_msd"], expected["predicted_msd"], rtol=0.02)

    def test_hybrid_clock_cage_prediction_composes_independent_displacements(self):
        rates = np.array([0.2])
        marks = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        times = np.array([1.0])
        wave_numbers = np.array([1.0])
        jump = heterogeneous_marked_poisson_prediction(rates, marks, times=times, wave_numbers=wave_numbers)
        cage = underdamped_ou_cage_prediction(
            times,
            temperature=0.58,
            mass=1.0,
            stiffness=3.2,
            damping=1.1,
        )

        hybrid = hybrid_structural_clock_underdamped_cage_prediction(
            rates,
            marks,
            times=times,
            wave_numbers=wave_numbers,
            temperature=0.58,
            mass=1.0,
            stiffness=3.2,
            damping=1.1,
        )

        coordinate_variance = float(cage["coordinate_displacement_variance"][0])
        self.assertAlmostEqual(float(hybrid["predicted_msd"][0]), float(jump["predicted_msd"][0]) + float(cage["predicted_msd"][0]))
        self.assertAlmostEqual(
            float(hybrid["predicted_fs"][1.0][0]),
            float(jump["predicted_fs"][1.0][0]) * np.exp(-0.5 * coordinate_variance),
        )
        self.assertGreater(float(hybrid["predicted_fourth"][0]), float(jump["predicted_fourth"][0]))

    def test_empirical_gaussian_cage_composes_with_structural_marked_clock(self):
        rates = np.array([0.2])
        marks = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        times = np.array([1.0])
        wave_numbers = np.array([1.0])
        jump = heterogeneous_marked_poisson_prediction(rates, marks, times=times, wave_numbers=wave_numbers)
        prediction = independent_gaussian_cage_marked_poisson_prediction(
            rates,
            marks,
            times=times,
            wave_numbers=wave_numbers,
            cage_coordinate_variance=np.array([0.2]),
        )

        self.assertAlmostEqual(float(prediction["predicted_msd"][0]), float(jump["predicted_msd"][0]) + 0.6)
        self.assertAlmostEqual(
            float(prediction["predicted_fs"][1.0][0]),
            float(jump["predicted_fs"][1.0][0]) * np.exp(-0.1),
        )

    def test_heterogeneous_marked_poisson_prediction_matches_analytic_moments(self):
        rates = np.array([0.1, 0.3])
        marks = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        result = heterogeneous_marked_poisson_prediction(
            rates,
            marks,
            times=np.array([2.0]),
            wave_numbers=np.array([1.0]),
        )

        self.assertAlmostEqual(float(result["predicted_msd"][0]), 0.4)
        self.assertAlmostEqual(
            float(result["predicted_fs"][1.0][0]),
            0.5
            * (
                np.exp(0.2 * ((np.cos(1.0) + 2.0) / 3.0 - 1.0))
                + np.exp(0.6 * ((np.cos(1.0) + 2.0) / 3.0 - 1.0))
            ),
        )
        expected_fourth = 0.4 + (5.0 / 3.0) * 4.0 * (0.1**2 + 0.3**2) / 2.0
        expected_ngp = 3.0 * expected_fourth / (5.0 * 0.4**2) - 1.0
        self.assertAlmostEqual(float(result["predicted_ngp"][0]), expected_ngp)
    def test_grouped_binomial_logistic_committor_generalizes_across_parent_groups(self):
        rng = np.random.default_rng(859)
        groups = np.repeat(np.arange(5), 400)
        feature = rng.normal(size=(len(groups), 2))
        probability = 1.0 / (1.0 + np.exp(-(-0.3 + 1.1 * feature[:, 0] - 0.7 * feature[:, 1])))
        trials = np.full(len(groups), 12)
        success = rng.binomial(trials, probability)
        result = grouped_binomial_logistic_committor_diagnostic(
            feature,
            success,
            trials,
            groups,
            l2_regularization=1.0,
        )

        self.assertGreater(float(result["mean_heldout_brier_skill"]), 0.10)
        self.assertGreater(float(result["mean_heldout_log_likelihood_gain_per_trial"]), 0.02)
    def test_isoconfigurational_state_rate_recovers_heldout_state_dependent_escape(self):
        rng = np.random.default_rng(853)
        clones, particles, horizon = 20, 1000, 12.0
        state = np.exp(rng.normal(scale=0.65, size=particles))
        rate = 0.025 * state**0.8
        first_passage = rng.exponential(1.0 / rate[None, :], size=(clones, particles))
        first_passage[first_passage > horizon] = np.inf
        train_mask = np.arange(particles) % 2 == 0
        result = isoconfigurational_state_rate_diagnostic(
            state,
            first_passage,
            train_mask=train_mask,
            horizon=horizon,
            bin_count=5,
        )

        self.assertGreater(float(result["heldout_brier_skill"]), 0.05)
        self.assertGreater(float(result["heldout_high_to_low_rate_ratio"]), 2.0)
        self.assertGreater(float(result["state_rate_log_slope"]), 0.4)
    def test_isoconfigurational_first_passage_recovers_censored_exponential_clock(self):
        rng = np.random.default_rng(839)
        rate, horizon = 0.08, 20.0
        first_passage = rng.exponential(1.0 / rate, size=(24, 1000))
        first_passage[first_passage > horizon] = np.inf
        result = isoconfigurational_first_passage_diagnostic(
            first_passage,
            horizon=horizon,
            survival_times=np.array([2.0, 5.0, 10.0, 20.0]),
        )

        self.assertAlmostEqual(float(result["escape_rate_mle"]), rate, delta=0.004)
        self.assertLess(float(result["survival_max_absolute_error"]), 0.025)
        self.assertGreater(float(result["committor_mean"]), 0.7)
    def test_prepare_isoconfigurational_langevin_clones_reuses_one_restart_and_varies_noise(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            restart = root / "parent.restart"
            restart.write_bytes(b"synthetic restart payload")
            output = root / "clones"

            manifest = prepare_isoconfigurational_langevin_clones(
                restart,
                output,
                temperature=0.58,
                velocity_seeds=[101, 103],
                langevin_seeds=[107, 109],
                damping=1.0,
                duration=2.0,
                dump_interval=0.05,
            )

            self.assertEqual(manifest["clone_count"], 2)
            self.assertEqual(manifest["parent_restart_path"], str(restart.resolve()))
            self.assertEqual(manifest["axis_semantics"], "isoconfigurational_langevin_clones")
            first = (output / "clone_001" / "in.clone").read_text()
            second = (output / "clone_002" / "in.clone").read_text()
            self.assertIn(f"read_restart {restart.resolve()}", first)
            self.assertIn("velocity all create 0.58 101", first)
            self.assertIn("fix bath all langevin 0.58 0.58 1 107", first)
            self.assertIn("velocity all create 0.58 103", second)
            self.assertIn("fix bath all langevin 0.58 0.58 1 109", second)

    def test_prepare_isoconfigurational_langevin_clones_can_dump_velocity_and_force(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            restart = root / "parent.restart"
            restart.write_bytes(b"synthetic restart payload")
            output = root / "clones"

            manifest = prepare_isoconfigurational_langevin_clones(
                restart,
                output,
                temperature=0.58,
                velocity_seeds=[101, 103],
                langevin_seeds=[107, 109],
                damping=1.0,
                duration=2.0,
                dump_interval=0.05,
                dump_velocity_force=True,
            )

            self.assertTrue(manifest["dump_velocity_force"])
            input_text = (output / "clone_001" / "in.clone").read_text()
            self.assertIn("id type x y z ix iy iz vx vy vz fx fy fz", input_text)
    def test_precursor_event_hazard_recovers_heldout_ready_escape_enrichment(self):
        rng = np.random.default_rng(827)
        frames, particles, precursor_lag = 240, 400, 2
        state = np.exp(rng.normal(scale=0.7, size=(frames, particles)))
        ready = state[:-precursor_lag] >= np.quantile(state[:120], 0.75)
        event_times: list[int] = []
        event_particles: list[int] = []
        for source_time in range(frames - precursor_lag):
            probability = np.where(ready[source_time], 0.30, 0.01)
            selected = np.flatnonzero(rng.random(particles) < probability)
            event_times.extend([source_time + precursor_lag] * len(selected))
            event_particles.extend(selected.tolist())

        result = precursor_event_hazard_diagnostic(
            state,
            np.asarray(event_particles),
            np.asarray(event_times),
            train_stop=120,
            precursor_lag=precursor_lag,
            frame_time=1.0,
            bin_count=4,
            held_source_start=130,
        )

        self.assertGreater(float(result["heldout_ready_to_unready_hazard_ratio"]), 10.0)
        self.assertGreater(float(result["heldout_brier_skill"]), 0.10)
        self.assertGreater(float(result["heldout_log_likelihood_gain_per_event"]), 0.10)
        self.assertGreater(float(result["training_ready_to_unready_hazard_ratio"]), 10.0)
    def test_arp_empirical_innovation_recovers_synthetic_two_step_memory(self):
        rng = np.random.default_rng(801)
        velocity = np.zeros((160, 500, 3))
        for index in range(2, len(velocity)):
            velocity[index] = (
                -0.35 * velocity[index - 1]
                + 0.22 * velocity[index - 2]
                + rng.normal(scale=0.06, size=(500, 3))
            )
        centers = np.concatenate([np.zeros((1, 500, 3)), np.cumsum(velocity, axis=0)])
        result = time_split_arp_empirical_innovation_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            train_stop=80,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8, 16]),
            wave_numbers=np.array([0.5, 1.0]),
            memory_order=2,
            simulation_count=12000,
            seed=803,
        )

        self.assertAlmostEqual(float(result["kernel_lag_1"]), -0.35, delta=0.04)
        self.assertAlmostEqual(float(result["kernel_lag_2"]), 0.22, delta=0.04)
        self.assertGreater(float(result["heldout_one_step_r_squared"]), 0.1)
        self.assertLess(float(result["diffusion_relative_error"]), 0.15)

    def test_arp_empirical_innovation_excludes_invalid_increment_endpoint(self):
        rng = np.random.default_rng(811)
        velocity = rng.normal(scale=0.04, size=(100, 200, 3))
        centers = np.concatenate([np.zeros((1, 200, 3)), np.cumsum(velocity, axis=0)])
        valid = np.ones(centers.shape[:2], dtype=bool)
        valid[30, 0] = False
        centers[30, 0] = np.nan

        result = time_split_arp_empirical_innovation_diagnostic(
            centers,
            valid,
            train_stop=50,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8]),
            wave_numbers=np.array([0.5, 1.0]),
            memory_order=2,
            simulation_count=1000,
            seed=813,
        )

        self.assertTrue(np.isfinite(float(result["kernel_lag_1"])))
        self.assertTrue(np.isfinite(float(result["heldout_one_step_r_squared"])))

    def test_long_wavelength_displacement_field_reconstructs_plane_wave_at_target(self):
        box = np.array([8.0, 8.0, 8.0])
        x = np.array([0.0, 2.0, 4.0, 6.0])
        positions = np.zeros((2, 4, 3))
        positions[0, :, 0] = x
        positions[1] = positions[0]
        positions[1, :, 1] = 0.2 * np.cos(2.0 * np.pi * x / box[0])
        field = long_wavelength_displacement_field(
            positions,
            target_indices=np.array([0]),
            box_lengths=box,
            integer_vectors=np.array([[1, 0, 0]]),
        )

        self.assertAlmostEqual(float(field[0, 0, 0, 1]), 0.2, delta=1e-12)

    def test_fourier_density_current_field_is_translation_covariant_at_tagged_particle(self):
        box = np.array([8.0, 8.0, 8.0])
        positions = np.zeros((1, 4, 3))
        positions[0, :, 0] = [0.0, 1.0, 3.0, 6.0]
        velocities = np.array([[[1.0, 0.2, 0.0], [-0.5, 0.0, 0.1], [0.25, 0.4, -0.2], [0.75, -0.3, 0.0]]])
        modes = np.array([[1, 0, 0]])
        original = fourier_density_current_field(
            positions,
            velocities=velocities,
            target_indices=np.array([0]),
            box_lengths=box,
            integer_vectors=modes,
        )
        shifted = fourier_density_current_field(
            np.mod(positions + np.array([0.37, 0.0, 0.0]), box),
            velocities=velocities,
            target_indices=np.array([0]),
            box_lengths=box,
            integer_vectors=modes,
        )
        phase = 2.0 * np.pi * positions[0, :, 0] / box[0]
        rho = np.mean(np.exp(-1j * phase))
        current = np.mean(velocities[0] * np.exp(-1j * phase)[:, None], axis=0)

        self.assertAlmostEqual(float(original["density"][0, 0, 0]), float(2.0 * np.real(rho)), delta=1e-12)
        np.testing.assert_allclose(original["current"][0, 0, 0], 2.0 * np.real(current), atol=1e-12)
        np.testing.assert_allclose(shifted["density"], original["density"], atol=1e-12)
        np.testing.assert_allclose(shifted["current"], original["current"], atol=1e-12)

    def test_symmetric_quadratic_mode_products_retains_each_unique_density_pair(self):
        modes = np.array([[[2.0, -3.0, 5.0]]])

        products = symmetric_quadratic_mode_products(modes)

        np.testing.assert_allclose(products[0, 0], [4.0, -6.0, 10.0, 9.0, -15.0, 25.0])

    def test_local_neighbor_particle_velocity_field_excludes_the_tagged_velocity(self):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [4.1, 0.0, 0.0]],
            ]
        )
        velocities = np.array(
            [
                [[9.0, 0.0, 0.0], [1.5, -0.5, 0.25], [-2.0, 0.0, 0.0]],
                [[8.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            ]
        )

        field, coordination = local_neighbor_particle_velocity_field(
            positions,
            velocities=velocities,
            target_indices=np.array([0]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
            cutoff=1.5,
        )

        self.assertEqual(coordination.tolist(), [[1], [1]])
        np.testing.assert_allclose(field[:, 0], velocities[:, 1])

    def test_discrete_mori_zwanzig_operators_reconstruct_training_correlations(self):
        rng = np.random.default_rng(901)
        samples, frames = 10000, 10
        state = np.empty((frames, samples, 1))
        state[0] = rng.normal(size=(samples, 1))
        state[1] = 0.6 * state[0] + rng.normal(scale=0.4, size=(samples, 1))
        for time in range(2, frames):
            state[time] = 0.5 * state[time - 1] - 0.2 * state[time - 2] + rng.normal(scale=0.4, size=(samples, 1))

        result = discrete_mori_zwanzig_operators(state, memory_order=4)

        np.testing.assert_allclose(
            result["reconstructed_correlation"],
            result["correlation"][:6],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_simulate_discrete_mz_empirical_innovations_respects_finite_memory_recursion(self):
        states = simulate_discrete_mz_empirical_innovations(
            np.array([[[2.0], [1.0]]]),
            np.array([[[0.5]], [[0.25]]]),
            np.zeros((1, 1)),
            output_count=3,
            rng=np.random.default_rng(902),
        )

        np.testing.assert_allclose(states[:, :, 0], [[1.0, 1.0, 0.75]])

    def test_simulate_discrete_mz_block_innovations_preserves_source_block_order(self):
        states = simulate_discrete_mz_block_innovations(
            np.array([[[0.0]]]),
            np.array([[[0.0]]]),
            np.array([[[1.0]], [[2.0]]]),
            output_count=5,
            block_length=2,
            rng=np.random.default_rng(903),
        )

        np.testing.assert_allclose(states[:, :, 0], [[0.0, 1.0, 2.0, 1.0, 2.0]])

    def test_local_cluster_hessian_is_symmetric_and_has_isolated_translation_mode(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        result = ka_local_cluster_hessian(
            positions,
            particle_types=np.array([0, 0]),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            target_index=0,
            cluster_cutoff=2.0,
        )

        hessian = result["hessian"]
        self.assertEqual(result["cluster_indices"].tolist(), [0, 1])
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
        np.testing.assert_allclose(hessian @ np.tile(np.eye(3), (2, 1)), 0.0, atol=1e-12)

    def test_local_cluster_soft_mode_features_are_target_symmetric_for_a_pair(self):
        features, names = ka_local_cluster_soft_mode_features(
            np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
            particle_types=np.array([0, 0]),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            target_indices=np.array([0, 1]),
            cluster_cutoff=2.0,
            ranks=(0,),
            eigenvalue_floor=1e-8,
        )

        self.assertEqual(names, ("inverse_eigenvalue_rank_0", "target_weighted_softness_rank_0"))
        self.assertEqual(features.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(features > 0.0))
        np.testing.assert_allclose(features[0], features[1], rtol=1e-12, atol=1e-12)

    def test_local_cluster_anharmonic_barrier_features_are_positive_and_pair_symmetric(self):
        features, names = ka_local_cluster_anharmonic_barrier_features(
            np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
            particle_types=np.array([0, 0]),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            target_indices=np.array([0, 1]),
            cluster_cutoff=2.0,
            ranks=(2,),
            eigenvalue_floor=1e-8,
            finite_difference_step=2e-3,
        )

        self.assertEqual(names, ("cubic_magnitude_rank_2", "anharmonic_barrier_proxy_rank_2"))
        self.assertEqual(features.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(features > 0.0))
        np.testing.assert_allclose(features[0], features[1], rtol=1e-10, atol=1e-10)

    def test_local_affine_nonaffine_state_recovers_pure_affine_neighbor_motion(self):
        reference = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        )
        gradient = np.array([[0.1, 0.03, 0.0], [0.0, -0.04, 0.02], [0.0, 0.0, 0.05]])
        displaced = reference.copy()
        displaced[1:] += reference[1:] @ gradient.T
        result = local_affine_nonaffine_state(
            np.stack([reference, displaced]),
            target_indices=np.array([0]),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            cutoff=1.5,
        )

        self.assertTrue(result["valid"][0, 0])
        np.testing.assert_allclose(result["affine_gradient"][0, 0], gradient, atol=1e-12)
        self.assertLess(float(result["d2min"][0, 0]), 1e-20)

    def test_shell_response_embedding_recovers_synthetic_spatial_coefficients(self):
        rng = np.random.default_rng(701)
        frames, particles, shells = 500, 150, 3
        shell_response = rng.normal(scale=0.1, size=(frames - 1, particles, shells, 3))
        velocity = np.zeros((frames - 1, particles, 3))
        for index in range(2, len(velocity)):
            velocity[index] = (
                0.1 * velocity[index - 1]
                + 0.7 * shell_response[index - 2, :, 0]
                - 0.45 * shell_response[index - 2, :, 1]
                + rng.normal(scale=0.015, size=(particles, 3))
            )
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(velocity, axis=0)])
        result = time_split_shell_response_embedding_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            shell_response,
            np.ones(shell_response.shape[:3], dtype=bool),
            train_stop=250,
        )

        self.assertAlmostEqual(float(result["shell_0_coefficient"]), 0.7, delta=0.04)
        self.assertAlmostEqual(float(result["shell_1_coefficient"]), -0.45, delta=0.04)
        self.assertGreater(float(result["heldout_shell_r_squared_gain"]), 0.75)

    def test_lagged_response_embedding_recovers_synthetic_collective_state(self):
        rng = np.random.default_rng(611)
        frames, particles = 500, 200
        response = rng.normal(scale=0.12, size=(frames - 1, particles, 3))
        velocity = np.zeros_like(response)
        for index in range(1, len(velocity)):
            velocity[index] = (
                0.15 * velocity[index - 1]
                + 0.8 * response[index - 2]
                + rng.normal(scale=0.02, size=(particles, 3))
            )
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(velocity, axis=0)])
        result = time_split_lagged_response_embedding_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            response,
            np.ones(response.shape[:2], dtype=bool),
            train_stop=250,
        )

        self.assertAlmostEqual(float(result["lagged_response_coefficient"]), 0.8, delta=0.04)
        self.assertGreater(float(result["heldout_response_r_squared_gain"]), 0.7)

    def test_ar1_empirical_innovation_diagnostic_recovers_synthetic_underdamped_walk(self):
        rng = np.random.default_rng(515)
        velocity = np.zeros((140, 600, 3))
        for index in range(1, len(velocity)):
            velocity[index] = -0.35 * velocity[index - 1] + rng.normal(
                scale=0.08, size=(600, 3)
            )
        centers = np.concatenate([np.zeros((1, 600, 3)), np.cumsum(velocity, axis=0)])
        result = time_split_ar1_empirical_innovation_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            train_stop=70,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8, 16]),
            wave_numbers=np.array([0.5, 1.0]),
            simulation_count=12000,
            seed=517,
        )

        self.assertAlmostEqual(float(result["ar1_coefficient"]), -0.35, delta=0.03)
        self.assertLess(float(result["diffusion_relative_error"]), 0.12)
        self.assertLess(float(result["ar1_fs_max_relative_error"]), 0.08)
        self.assertLess(float(result["ar1_ngp_max_absolute_error"]), 0.05)

    def test_empirical_increment_levy_diagnostic_recovers_iid_gaussian_walk(self):
        rng = np.random.default_rng(417)
        increments = rng.normal(scale=0.08, size=(120, 1500, 3))
        centers = np.concatenate([np.zeros((1, 1500, 3)), np.cumsum(increments, axis=0)])
        result = time_split_empirical_increment_levy_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            train_stop=60,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8, 16]),
            wave_numbers=np.array([0.5, 1.0]),
        )

        self.assertLess(float(result["diffusion_relative_error"]), 0.08)
        self.assertLess(float(result["levy_fs_max_relative_error"]), 0.05)
        self.assertLess(float(result["levy_ngp_max_absolute_error"]), 0.04)
        self.assertEqual(float(result["fit_parameters_from_macro_observables"]), 0.0)

    def test_response_residual_memory_diagnostic_recovers_white_gaussian_force(self):
        rng = np.random.default_rng(392)
        residual = rng.normal(scale=0.03, size=(800, 40, 3))
        centers = np.concatenate([np.zeros((1, 40, 3)), np.cumsum(residual, axis=0)])
        result = response_residual_memory_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            np.zeros_like(residual),
            np.ones(residual.shape[:2], dtype=bool),
            frame_time=0.05,
        )

        self.assertEqual(float(result["white_noise_candidate"]), 1.0)
        self.assertLess(abs(float(result["response_residual_ngp"])), 0.04)

    def test_precursor_hazard_uses_only_explicitly_valid_state_exposure(self):
        frames, particles = 80, 4
        state = np.exp(0.25 * np.sin(0.31 * np.arange(frames)[:, None]) + 0.03 * np.arange(particles)[None, :])
        valid = np.ones((frames, particles), dtype=bool)
        valid[12:18, 0] = False
        state[~valid] = np.nan
        events = [(particle, time) for particle in range(particles) for time in (16, 28, 44, 60)]
        result = precursor_event_hazard_diagnostic(
            state,
            np.array([particle for particle, _ in events]),
            np.array([time for _, time in events]),
            train_stop=36,
            precursor_lag=4,
            frame_time=0.1,
            state_valid=valid,
        )

        self.assertLess(float(result["training_state_valid_fraction"]), 1.0)
        self.assertEqual(float(result["heldout_state_valid_fraction"]), 1.0)

    def test_projected_diffusion_increment_association_detects_monotone_conditional_mobility(self):
        tensor = np.array([scale * np.eye(3) for scale in (1.0, 2.0, 3.0, 4.0, 5.0)])
        increment = np.array([[np.sqrt(scale), 0.0, 0.0] for scale in (1.0, 2.0, 3.0, 4.0, 5.0)])

        result = projected_diffusion_increment_association(tensor, increment, tail_fraction=0.2)

        self.assertAlmostEqual(float(result["rank_correlation"]), 1.0)
        self.assertGreater(float(result["upper_tail_increment_ratio"]), 1.0)

    def test_projected_overdamped_mixture_ngp_recovers_isotropic_diffusing_diffusivity_limit(self):
        tensor = np.array([np.eye(3), 2.0 * np.eye(3)])

        result = projected_overdamped_mixture_ngp(tensor)

        self.assertAlmostEqual(float(result["predicted_infinitesimal_ngp"]), 2.5 / 1.5**2 - 1.0)

    def test_projected_overdamped_diffusion_tensor_follows_ito_jacobian_rule(self):
        jacobian = np.zeros((1, 1, 2, 3, 3))
        jacobian[0, 0, 0] = np.eye(3)
        jacobian[0, 0, 1] = 2.0 * np.eye(3)

        result = projected_overdamped_diffusion_tensor(
            jacobian, temperature=0.6, bare_mobility=1.5
        )
        flattened = projected_overdamped_diffusion_tensor(
            jacobian[0, 0], temperature=0.6, bare_mobility=1.5
        )

        np.testing.assert_allclose(result["diffusion_tensor"][0, 0], 4.5 * np.eye(3))
        np.testing.assert_allclose(flattened["diffusion_tensor"], 4.5 * np.eye(3))
        self.assertAlmostEqual(float(result["isotropic_diffusion"][0, 0]), 4.5)
        self.assertAlmostEqual(float(result["anisotropy_ratio"][0, 0]), 1.0)

    def test_frozen_minimum_response_jacobian_obeys_translation_sum_rule(self):
        environment = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [-1.1, 0.0, 0.0],
                [0.0, 1.1, 0.0],
                [0.0, -1.1, 0.0],
                [0.0, 0.0, 1.1],
                [0.0, 0.0, -1.1],
            ]
        )
        result = frozen_minimum_response_jacobian(
            np.stack([environment, environment]),
            centers=np.zeros((2, 1, 3)),
            valid=np.ones((2, 1), dtype=bool),
            particle_types=np.zeros(len(environment), dtype=int),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            target_indices=np.array([0]),
        )

        self.assertTrue(result["response_valid"][0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(result["jacobian"][0, 0, 0])), 0.0)
        np.testing.assert_allclose(
            np.sum(result["jacobian"][0, 0], axis=0), np.eye(3), atol=1e-12
        )

    def test_minimum_response_diagnostic_recovers_parameter_free_linear_response(self):
        rng = np.random.default_rng(120)
        increments = rng.normal(scale=0.1, size=(300, 80, 3))
        centers = np.concatenate([np.zeros((1, 80, 3)), np.cumsum(increments, axis=0)])
        response = increments + rng.normal(scale=0.005, size=increments.shape)
        result = minimum_response_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            response,
            np.ones(response.shape[:2], dtype=bool),
            train_stop=150,
        )

        self.assertGreater(float(result["heldout_response_r_squared"]), 0.99)
        self.assertLess(abs(float(result["heldout_response_scale"])-1.0), 0.02)
        self.assertEqual(float(result["fit_parameters_from_macro_observables"]), 0.0)

    def test_frozen_minimum_response_tracks_uniform_environment_translation(self):
        environment = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [-1.1, 0.0, 0.0],
                [0.0, 1.1, 0.0],
                [0.0, -1.1, 0.0],
                [0.0, 0.0, 1.1],
                [0.0, 0.0, -1.1],
            ]
        )
        translation = np.array([2e-5, -3e-5, 1e-5])
        positions = np.stack([environment, environment + translation])
        result = frozen_minimum_environment_response(
            positions,
            centers=np.zeros((2, 1, 3)),
            valid=np.ones((2, 1), dtype=bool),
            particle_types=np.zeros(len(environment), dtype=int),
            box_lengths=np.array([20.0, 20.0, 20.0]),
            target_indices=np.array([0]),
        )

        self.assertTrue(result["response_valid"][0, 0])
        np.testing.assert_allclose(result["predicted_center_increment"][0, 0], translation, atol=1e-12)

    def test_frozen_minimum_shell_response_sums_to_exact_total_response(self):
        environment = np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [-1.1, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, -1.1, 0.0]]
        )
        translation = np.array([2e-5, -3e-5, 1e-5])
        positions = np.stack([environment, environment + translation])
        common = {
            "centers": np.zeros((2, 1, 3)),
            "valid": np.ones((2, 1), dtype=bool),
            "particle_types": np.zeros(len(environment), dtype=int),
            "box_lengths": np.array([20.0, 20.0, 20.0]),
            "target_indices": np.array([0]),
        }
        total = frozen_minimum_environment_response(positions, **common)
        shell = frozen_minimum_shell_environment_response(
            positions, shell_edges=np.array([0.0, 1.5, 2.5]), **common
        )

        self.assertTrue(shell["response_valid"][0, 0])
        np.testing.assert_allclose(
            np.sum(shell["shell_response"][0, 0], axis=0),
            total["predicted_center_increment"][0, 0],
            atol=1e-12,
        )

    def test_local_neighbor_velocity_field_excludes_tagged_and_distant_particles(self):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0], [4.5, 0.0, 0.0]],
                [[0.2, 0.0, 0.0], [1.5, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ]
        )
        mean_velocity, coordination = local_neighbor_velocity_field(
            positions,
            target_indices=np.array([0]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
            cutoff=2.0,
        )

        np.testing.assert_allclose(mean_velocity[:, 0, 0], [0.3, 0.2])
        np.testing.assert_array_equal(coordination[:, 0], [1, 1])

    def test_collective_velocity_embedding_recovers_synthetic_environment_coupling(self):
        rng = np.random.default_rng(811)
        frames, particles = 500, 300
        neighbor_velocity = rng.normal(scale=0.12, size=(frames - 1, particles, 3))
        center_velocity = np.zeros((frames - 1, particles, 3))
        for index in range(1, len(center_velocity)):
            center_velocity[index] = (
                0.25 * center_velocity[index - 1]
                + 0.85 * neighbor_velocity[index - 1]
                + rng.normal(scale=0.02, size=(particles, 3))
            )
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(center_velocity, axis=0)])
        result = time_split_collective_velocity_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            neighbor_velocity,
            train_stop=250,
        )

        self.assertAlmostEqual(float(result["collective_velocity_coefficient"]), 0.85, delta=0.04)
        self.assertGreater(float(result["heldout_collective_r_squared"]), 0.9)
        self.assertLess(abs(float(result["collective_residual_ngp"])), 0.05)

    def test_collective_field_embedding_recovers_synthetic_causal_coupling(self):
        rng = np.random.default_rng(157)
        frames, particles, modes = 460, 180, 4
        field = rng.normal(scale=0.09, size=(frames - 1, particles, modes, 3))
        velocity = np.zeros((frames - 1, particles, 3))
        coupling = np.array([0.65, -0.35, 0.22, -0.11])
        for index in range(1, len(velocity) - 1):
            velocity[index + 1] = (
                0.18 * velocity[index]
                + np.einsum("pmc,m->pc", field[index - 1], coupling)
                + rng.normal(scale=0.015, size=(particles, 3))
            )
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(velocity, axis=0)])
        result = time_split_collective_field_embedding_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            field,
            train_stop=230,
        )

        self.assertGreater(float(result["heldout_collective_field_r_squared"]), 0.9)
        self.assertGreater(float(result["heldout_collective_field_r_squared_gain"]), 0.85)

    def test_state_dependent_increment_diagnostic_recovers_diffusing_diffusivity_mixture(self):
        rng = np.random.default_rng(119)
        frames, particles = 400, 600
        state = np.exp(rng.normal(scale=0.45, size=(frames, particles)))
        q = 0.03 * state[:-1] ** -0.8
        increments = rng.normal(scale=np.sqrt(q[:, :, None] / 3.0), size=(frames - 1, particles, 3))
        centers = np.concatenate([np.zeros((1, particles, 3)), np.cumsum(increments, axis=0)])
        result = state_dependent_increment_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            state,
            train_stop=200,
            frame_time=1.0,
        )

        self.assertLess(float(result["diffusion_relative_error"]), 0.06)
        self.assertLess(float(result["mixture_ngp_absolute_error"]), 0.03)
        self.assertLess(float(result["state_mobility_log_slope"]), -0.5)
        self.assertEqual(float(result["fit_parameters_from_macro_observables"]), 0.0)

    def test_time_split_center_gle_recovers_brownian_transport_and_gaussian_scattering(self):
        rng = np.random.default_rng(73)
        increments = rng.normal(scale=0.1, size=(480, 320, 3))
        centers = np.concatenate([np.zeros((1, 320, 3)), np.cumsum(increments, axis=0)])
        result = time_split_center_gle_diagnostic(
            centers,
            np.ones(centers.shape[:2], dtype=bool),
            train_stop=240,
            frame_time=1.0,
            lags=np.array([1, 2, 4, 8, 16]),
            wave_numbers=np.array([0.5, 1.0]),
        )

        self.assertLess(float(result["diffusion_relative_error"]), 0.12)
        self.assertLess(float(result["gaussian_fs_max_relative_error"]), 0.04)
        self.assertLess(abs(float(result["observed_heldout_ngp_peak"])), 0.05)
        self.assertEqual(float(result["fit_parameters_from_macro_observables"]), 0.0)

    def test_segmented_cage_center_clock_uses_only_valid_exposure(self):
        centers = np.zeros((13, 1, 3))
        centers[2:5, 0, 0] = 1.0
        centers[10:, 0, 0] = 1.0
        valid = np.ones((13, 1), dtype=bool)
        valid[5:8, 0] = False

        result = segmented_cage_center_event_statistics(
            centers,
            valid,
            threshold=0.1,
            half_window=1,
            frame_time=2.0,
        )

        self.assertEqual(float(result["segment_count"]), 2.0)
        self.assertEqual(float(result["valid_exposure_time"]), 16.0)
        self.assertEqual(float(result["event_count"]), 2.0)
        self.assertAlmostEqual(float(result["event_rate"]), 0.125)

    def test_frozen_neighbor_energy_explicitly_excludes_moved_tagged_particle_self_pair(self):
        environment = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        candidate = np.array([[0.1, 0.0, 0.0]])
        energy, force, hessian = ka_lj_local_energy_force_hessian(
            candidate,
            environment_positions=environment,
            target_indices=np.array([0]),
            particle_types=np.array([0, 0]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
        )

        distance = 1.1
        self.assertAlmostEqual(energy[0], 4.0 * (distance**-12 - distance**-6))
        self.assertTrue(np.all(np.isfinite(force)))
        self.assertTrue(np.all(np.isfinite(hessian)))
        self.assertLess(float(np.max(np.abs(hessian))), 1e6)
        self.assertLess(force[0, 0], 0.0)

    def test_ka_pair_force_and_isotropic_curvature_match_analytic_pair_formula(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        forces, curvature = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=np.array([0, 0]),
            box_lengths=np.array([10.0, 10.0, 10.0]),
        )

        distance = 1.2
        force_magnitude = 24.0 * (2.0 / distance**13 - 1.0 / distance**7)
        expected_curvature = 8.0 * (22.0 / distance**14 - 5.0 / distance**8)
        self.assertAlmostEqual(forces[0, 0], -force_magnitude)
        self.assertAlmostEqual(forces[1, 0], force_magnitude)
        self.assertAlmostEqual(curvature[0], expected_curvature)
        self.assertAlmostEqual(curvature[1], expected_curvature)

    def test_ka_c3_switched_pair_force_and_hessian_match_radial_derivatives(self):
        box_lengths = np.array([20.0, 20.0, 20.0])
        cases = (
            (np.array([0, 0]), 1.0, 1.0),
            (np.array([0, 1]), 1.5, 0.8),
            (np.array([1, 1]), 0.5, 0.88),
        )
        for particle_types, epsilon, sigma in cases:
            for scaled_radius in (1.2, 2.25, 2.499):
                radius = scaled_radius * sigma
                positions = np.array([[radius, 0.0, 0.0], [0.0, 0.0, 0.0]])
                result = ka_lj_force_generator_observables(
                    positions,
                    velocities=np.zeros_like(positions),
                    particle_types=particle_types,
                    box_lengths=box_lengths,
                    target_indices=np.array([0]),
                    friction=1.0,
                    temperature=0.58,
                    potential_protocol="ka_lj_c3_switch",
                )
                _, first, second, _ = ka_lj_radial_derivatives(
                    np.array([radius]),
                    epsilon=epsilon,
                    sigma=sigma,
                    protocol="ka_lj_c3_switch",
                )
                expected_force = np.array([-first[0], 0.0, 0.0])
                expected_hessian = np.diag([second[0], first[0] / radius, first[0] / radius])
                np.testing.assert_allclose(result["force"][0], expected_force, rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(
                    result["target_pair_hessian"][0, 1],
                    expected_hessian,
                    rtol=2e-12,
                    atol=2e-12,
                )

    def test_ka_c3_switched_force_generators_match_full_phase_space_drift(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.77, 0.17, -0.08]])
        velocities = np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, -0.4]])
        particle_types = np.array([0, 1])
        box_lengths = np.array([20.0, 20.0, 20.0])
        target = np.array([0])
        protocol = "ka_lj_c3_switch"
        result = ka_lj_force_generator_observables(
            positions,
            velocities=velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=1.0,
            temperature=0.58,
            potential_protocol=protocol,
        )
        step = 1e-6
        force_plus = ka_lj_force_and_isotropic_curvature(
            positions + step * velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            potential_protocol=protocol,
        )[0]
        force_minus = ka_lj_force_and_isotropic_curvature(
            positions - step * velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            potential_protocol=protocol,
        )[0]
        np.testing.assert_allclose(
            result["force_generator"],
            (force_plus - force_minus) / (2.0 * step),
            rtol=3e-8,
            atol=3e-8,
        )

        second = ka_lj_second_force_generator(
            positions,
            velocities=velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=1.0,
            directional_step=1e-5,
            potential_protocol=protocol,
        )
        force = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            potential_protocol=protocol,
        )[0]
        acceleration = force - velocities
        drift_step = 2e-6
        plus = ka_lj_force_generator_observables(
            positions + drift_step * velocities,
            velocities=velocities + drift_step * acceleration,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=1.0,
            temperature=0.0,
            potential_protocol=protocol,
        )["force_generator"]
        minus = ka_lj_force_generator_observables(
            positions - drift_step * velocities,
            velocities=velocities - drift_step * acceleration,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=1.0,
            temperature=0.0,
            potential_protocol=protocol,
        )["force_generator"]
        np.testing.assert_allclose(second, (plus - minus) / (2.0 * drift_step), rtol=3e-5, atol=3e-5)

    def test_ka_force_generator_matches_directional_force_derivative_and_hessian_noise(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.13, 0.17, -0.08]])
        velocities = np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, -0.4]])
        particle_types = np.array([0, 1])
        box_lengths = np.array([20.0, 20.0, 20.0])
        target = np.array([0])
        result = ka_lj_force_generator_observables(
            positions,
            velocities=velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=1.0,
            temperature=0.58,
        )
        step = 1e-6
        force_plus = ka_lj_force_and_isotropic_curvature(
            positions + step * velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
        )[0]
        force_minus = ka_lj_force_and_isotropic_curvature(
            positions - step * velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
        )[0]
        np.testing.assert_allclose(
            result["force_generator"],
            (force_plus - force_minus) / (2.0 * step),
            rtol=2e-8,
            atol=2e-8,
        )
        cluster = ka_local_cluster_hessian(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_index=0,
            cluster_cutoff=5.0,
        )["hessian"]
        target_row = cluster[:3]
        expected_covariance_rate = 2.0 * 1.0 * 0.58 * target_row @ target_row.T
        np.testing.assert_allclose(
            result["force_generator_noise_covariance_rate"][0],
            expected_covariance_rate,
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(result["target_pair_active"].shape, (1, 2))
        self.assertEqual(result["target_pair_hessian"].shape, (1, 2, 3, 3))
        self.assertFalse(bool(result["target_pair_active"][0, 0]))
        self.assertTrue(bool(result["target_pair_active"][0, 1]))
        self.assertEqual(int(result["nearest_cutoff_particle_index"][0]), 1)
        expected_gap = float(np.linalg.norm(positions[0] - positions[1]) - 2.5 * 0.8)
        self.assertAlmostEqual(float(result["nearest_cutoff_signed_gap"][0]), expected_gap)

    def test_ka_second_force_generator_matches_first_generator_drift(self):
        positions = np.array([[0.0, 0.0, 0.0], [1.13, 0.17, -0.08]])
        velocities = np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, -0.4]])
        particle_types = np.array([0, 1])
        box_lengths = np.array([20.0, 20.0, 20.0])
        target = np.array([0])
        result = ka_lj_second_force_generator(
            positions,
            velocities=velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=1.0,
            directional_step=1e-5,
        )
        force = ka_lj_force_and_isotropic_curvature(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
        )[0]
        acceleration = force - velocities
        step = 2e-6
        plus = ka_lj_force_generator_observables(
            positions + step * velocities,
            velocities=velocities + step * acceleration,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=1.0,
            temperature=0.0,
        )["force_generator"]
        minus = ka_lj_force_generator_observables(
            positions - step * velocities,
            velocities=velocities - step * acceleration,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=1.0,
            temperature=0.0,
        )["force_generator"]
        np.testing.assert_allclose(result, (plus - minus) / (2.0 * step), rtol=2e-5, atol=2e-5)

        step_results = [
            ka_lj_second_force_generator(
                positions,
                velocities=velocities,
                particle_types=particle_types,
                box_lengths=box_lengths,
                target_indices=target,
                friction=1.0,
                directional_step=directional_step,
            )
            for directional_step in (3e-6, 1e-5, 3e-5)
        ]
        reference_norm = float(np.linalg.norm(step_results[1]))
        for current in (step_results[0], step_results[2]):
            self.assertLess(float(np.linalg.norm(current - step_results[1]) / reference_norm), 2e-5)

    def test_force_generator_increment_diagnostic_recovers_exact_derivative_and_known_innovation(self):
        frame_time = 0.1
        force = np.zeros((5, 3))
        force[:, 0] = [0.0, 0.0, 0.2, 0.0, 0.0]
        force_generator = np.zeros((5, 3))
        force_generator[:, 0] = [0.0, 1.0, 0.0, -1.0, 0.0]
        second_force_generator = np.zeros((5, 3))
        covariance_rate = np.repeat((np.eye(3) / frame_time)[None, :, :], 5, axis=0)

        result = force_generator_increment_diagnostic(
            force,
            force_generator,
            second_force_generator,
            covariance_rate,
            frame_time=frame_time,
        )

        self.assertLess(float(result["force_derivative_relative_l2"]), 1e-12)
        self.assertAlmostEqual(float(result["force_derivative_correlation"]), 1.0, delta=1e-12)
        self.assertAlmostEqual(float(result["innovation_trace_variance_ratio"]), 1.0 / 3.0, delta=1e-12)
        self.assertAlmostEqual(float(result["innovation_mean_squared_mahalanobis"]), 1.0, delta=1e-12)
        self.assertLess(float(result["innovation_normalized_mean"]), 1e-12)
        self.assertEqual(float(result["thermodynamic_claim_allowed"]), 0.0)

    def test_force_generator_cli_exposes_microscopic_controls(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "analyze_ka_force_generator.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--target-id",
            "--temperature",
            "--friction",
            "--integration-time-step",
            "--directional-step",
            "--stochastic-frame-limit",
        ):
            self.assertIn(option, completed.stdout)

    def test_generator_response_lammps_input_fixes_common_noise_and_full_state_dump(self):
        text = generator_response_lammps_input(
            parent_restart=Path("/tmp/parent.restart"),
            target_id=821,
            displacement=0.001,
            temperature=0.58,
            friction=1.0,
            velocity_seed=82101,
            langevin_seed=83101,
            run_steps=1000,
            dump_interval_steps=5,
            trajectory_name="trajectory.lammpstrj",
        )

        self.assertIn("displace_atoms tagged move 0.001 0 0 units box", text)
        self.assertIn("velocity all create 0.58 82101", text)
        self.assertIn("fix bath all langevin 0.58 0.58 1 83101", text)
        self.assertIn(
            "dump trajectory all custom 5 trajectory.lammpstrj id type x y z ix iy iz vx vy vz",
            text,
        )
        self.assertIn("dump_modify trajectory sort id format float %.17g", text)
        self.assertIn("run 1000", text)

    def test_extract_generator_response_path_reads_full_microscopic_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.lammpstrj"
            blocks = []
            for frame, timestep in enumerate((0, 5, 10, 15, 20)):
                first = np.array([1.0 + 0.01 * frame, 2.0 + 0.005 * frame, 3.0])
                second = np.array([2.13 - 0.002 * frame, 2.17, 2.92])
                blocks.append(
                    "\n".join(
                        [
                            "ITEM: TIMESTEP",
                            str(timestep),
                            "ITEM: NUMBER OF ATOMS",
                            "2",
                            "ITEM: BOX BOUNDS pp pp pp",
                            "0 20",
                            "0 20",
                            "0 20",
                            "ITEM: ATOMS id type x y z ix iy iz vx vy vz fx fy fz",
                            f"1 1 {first[0]} {first[1]} {first[2]} 0 0 0 0.4 -0.2 0.1 0 0 0",
                            f"2 2 {second[0]} {second[1]} {second[2]} 0 0 0 -0.3 0.5 -0.4 0 0 0",
                        ]
                    )
                )
            path.write_text("\n".join(blocks) + "\n")

            result = extract_generator_response_path(
                path,
                target_id=1,
                temperature=0.58,
                friction=1.0,
                integration_time_step=0.001,
                directional_step=1e-5,
                potential_protocol="ka_lj_c3_switch",
            )

        np.testing.assert_allclose(result["time"], np.arange(5) * 0.005)
        for key in ("position", "velocity", "force", "force_generator", "second_force_generator"):
            self.assertEqual(np.asarray(result[key]).shape, (5, 3))
            self.assertTrue(np.all(np.isfinite(result[key])))
        self.assertEqual(np.asarray(result["force_generator_noise_covariance_rate"]).shape, (5, 3, 3))
        self.assertEqual(np.asarray(result["target_pair_active"]).shape, (5, 2))
        self.assertEqual(np.asarray(result["target_pair_hessian"]).shape, (5, 2, 3, 3))
        self.assertEqual(np.asarray(result["nearest_cutoff_signed_gap"]).shape, (5,))
        self.assertEqual(np.asarray(result["nearest_cutoff_particle_index"]).shape, (5,))
        self.assertEqual(str(result["potential_protocol"]), "ka_lj_c3_switch")
        self.assertEqual(float(result["thermodynamic_claim_allowed"]), 0.0)

    def test_generator_response_runner_exposes_low_disk_protocol_controls(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_ka_generator_response.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--lammps-binary",
            "--parent-restart",
            "--output-directory",
            "--velocity-seeds",
            "--langevin-seeds",
            "--epsilons",
            "--duration",
            "--dump-interval",
            "--retain-audit-raw",
        ):
            self.assertIn(option, completed.stdout)

    def test_generator_constrained_response_recovers_exact_stable_krylov_system(self):
        friction = 1.0
        coefficient_blocks = [[] for _ in range(4)]
        for roots in (
            (-0.5, -0.8, -1.1, -1.4),
            (-0.6, -0.9, -1.2, -1.5),
            (-0.7, -1.0, -1.3, -1.6),
        ):
            polynomial = np.poly(roots)
            a_g = friction - polynomial[1]
            a_f = -polynomial[2] - friction * a_g
            a_v = -polynomial[3] - friction * a_f
            a_x = -polynomial[4]
            for block, value in zip(coefficient_blocks, (a_x, a_v, a_f, a_g)):
                block.append(value)
        final_block = np.hstack([np.diag(values) for values in coefficient_blocks])
        generator = np.zeros((12, 12))
        generator[0:3, 3:6] = np.eye(3)
        generator[3:6, 3:6] = -friction * np.eye(3)
        generator[3:6, 6:9] = np.eye(3)
        generator[6:9, 9:12] = np.eye(3)
        generator[9:12] = final_block
        frame_time = 0.02
        transition = np.eye(12)
        power = np.eye(12)
        for order in range(1, 5):
            power = power @ (frame_time * generator) / order
            transition += power
        states = np.empty((3, 160, 12))
        states[:, 0] = np.random.default_rng(981).normal(size=(3, 12))
        for frame in range(1, states.shape[1]):
            states[:, frame] = np.einsum("ab,pb->pa", transition, states[:, frame - 1])
        second = np.einsum("ab,ptb->pta", final_block, states)

        result = fit_generator_constrained_response(
            states,
            second,
            frame_time=frame_time,
            friction=friction,
            fit_frames=100,
        )

        np.testing.assert_allclose(result["fitted_second_generator_block"], final_block, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(result["predicted_state_response"], states, rtol=1e-8, atol=1e-8)
        self.assertLess(float(result["spectral_radius"]), 1.0)
        self.assertLess(float(result["heldout_position_relative_l2_error"]), 1e-8)
        self.assertEqual(float(result["thermodynamic_claim_allowed"]), 0.0)

    def test_matched_generator_response_preserves_exact_state_order(self):
        epsilon = 0.002
        time = np.arange(4, dtype=float) * 0.005
        baseline = np.arange(12, dtype=float).reshape(4, 3)
        responses = {
            "position": np.full((4, 3), 1.0),
            "velocity": np.full((4, 3), 2.0),
            "force": np.full((4, 3), 3.0),
            "force_generator": np.full((4, 3), 4.0),
            "second_force_generator": np.full((4, 3), 5.0),
        }
        positive = {"time": time, **{key: baseline + epsilon * value for key, value in responses.items()}}
        negative = {"time": time, **{key: baseline - epsilon * value for key, value in responses.items()}}

        result = matched_generator_response(positive, negative, epsilon=epsilon)

        np.testing.assert_allclose(result["time"], time)
        np.testing.assert_allclose(result["state_response"], np.tile([1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3, (4, 1)))
        np.testing.assert_allclose(result["second_force_response"], 5.0)
        self.assertEqual(float(result["thermodynamic_claim_allowed"]), 0.0)

    def test_tangent_force_generator_noise_covariance_uses_pair_hessian_response(self):
        epsilon = 0.002
        delta_pair = np.zeros((2, 3, 3))
        delta_pair[1] = np.diag([1.0, 2.0, 3.0])
        positive = epsilon * delta_pair
        negative = -epsilon * delta_pair

        result = tangent_force_generator_noise_covariance_rate(
            positive,
            negative,
            epsilon=epsilon,
            friction=1.5,
            temperature=0.4,
        )

        diagonal = np.sum(delta_pair, axis=0)
        expected = 2.0 * 1.5 * 0.4 * (
            diagonal @ diagonal.T + np.einsum("nab,ncb->ac", delta_pair, delta_pair)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_right_censored_tangent_interval_mask_discards_all_intervals_after_first_cutoff_crossing(self):
        mismatch = np.zeros(201, dtype=bool)
        mismatch[7] = True

        result = right_censored_tangent_interval_mask(
            mismatch,
            stride=5,
            interval_count=40,
        )

        np.testing.assert_array_equal(result[:3], [True, False, False])
        self.assertEqual(int(np.sum(result)), 1)
        np.testing.assert_array_equal(
            right_censored_tangent_interval_mask(
                np.zeros(201, dtype=bool),
                stride=5,
                interval_count=40,
            ),
            np.ones(40, dtype=bool),
        )

    def test_tangent_noise_covariance_diagnostic_calibrates_heteroscedastic_gaussian_noise(self):
        rng = np.random.default_rng(991)
        member_count = 48
        interval_count = 1200
        phase = np.linspace(0.0, 8.0 * np.pi, interval_count, endpoint=False)
        scale = np.exp(0.8 * np.sin(phase))[None, :, None]
        diagonal = scale * np.array([0.7, 1.1, 1.6])[None, None, :]
        covariance = np.zeros((member_count, interval_count, 3, 3))
        covariance[..., 0, 0] = diagonal[..., 0]
        covariance[..., 1, 1] = diagonal[..., 1]
        covariance[..., 2, 2] = diagonal[..., 2]
        normal = rng.normal(size=(member_count, interval_count, 3))
        residual = normal * np.sqrt(diagonal)
        valid = np.ones((member_count, interval_count), dtype=bool)
        valid[:, ::113] = False

        result = tangent_noise_covariance_diagnostic(residual, covariance, valid_mask=valid)

        self.assertAlmostEqual(float(result["trace_variance_ratio"]), 1.0, delta=0.02)
        self.assertAlmostEqual(float(result["mean_squared_mahalanobis"]), 3.0, delta=0.04)
        self.assertLess(float(result["whitened_max_abs_component_excess_kurtosis"]), 0.08)
        self.assertLess(abs(float(result["whitened_lag1_correlation"])), 0.02)
        self.assertGreater(float(result["observed_predicted_energy_correlation"]), 0.25)
        self.assertEqual(float(result["thermodynamic_claim_allowed"]), 0.0)

        single = tangent_noise_covariance_diagnostic(
            residual[0],
            covariance[0],
            valid_mask=valid[0],
        )
        self.assertEqual(float(single["valid_sample_count"]), float(np.sum(valid[0])))

    def test_generator_response_tangent_diagnostic_recovers_exact_smooth_chain(self):
        frame_time = 0.001
        friction = 1.0
        time = np.arange(1001, dtype=float) * frame_time
        rates = -np.array(
            [
                [0.4, 0.7, 1.1],
                [0.5, 0.9, 1.3],
                [0.6, 1.0, 1.5],
            ]
        )
        amplitude = np.array(
            [
                [1.0, 0.8, 1.2],
                [0.7, 1.1, 0.9],
                [1.3, 0.6, 1.0],
            ]
        )
        position = amplitude[:, None, :] * np.exp(rates[:, None, :] * time[None, :, None])
        velocity = rates[:, None, :] * position
        force = (rates**2 + friction * rates)[:, None, :] * position
        generator = rates[:, None, :] * force
        second = rates[:, None, :] * generator
        state = np.concatenate([position, velocity, force, generator], axis=2)

        result = generator_response_tangent_diagnostic(
            state,
            second,
            frame_time=frame_time,
            friction=friction,
        )

        for identity in ("position_velocity", "velocity_force", "force_generator", "generator_second"):
            self.assertLess(float(result[f"{identity}_relative_l2_error"]), 1e-6)
            self.assertGreater(float(result[f"{identity}_correlation"]), 1.0 - 1e-10)
        self.assertLess(float(result["tangent_innovation_rms"]), 1e-5)
        self.assertLess(float(result["symmetric_tangent_residual_rms"]), 1e-9)
        self.assertEqual(float(result["thermodynamic_claim_allowed"]), 0.0)

    def test_generator_response_tangent_diagnostic_detects_white_member_noise_suppression(self):
        rng = np.random.default_rng(987)
        member_count = 64
        frame_count = 2001
        frame_time = 0.005
        relaxation_rate = -20.0
        state = np.zeros((member_count, frame_count, 12))
        increments = rng.normal(scale=0.08, size=(member_count, frame_count - 1, 3))
        for frame in range(frame_count - 1):
            state[:, frame + 1, 9:12] = (
                state[:, frame, 9:12]
                + frame_time * relaxation_rate * state[:, frame, 9:12]
                + increments[:, frame]
            )
        second = relaxation_rate * state[:, :, 9:12]

        result = generator_response_tangent_diagnostic(
            state,
            second,
            frame_time=frame_time,
            friction=1.0,
        )

        np.testing.assert_allclose(result["tangent_innovation"], increments, atol=1e-12)
        self.assertLess(abs(float(result["tangent_innovation_lag1_correlation"])), 0.02)
        self.assertLess(abs(float(result["tangent_innovation_squared_norm_lag1_correlation"])), 0.02)
        self.assertLess(float(result["tangent_innovation_max_abs_component_excess_kurtosis"]), 0.08)
        self.assertAlmostEqual(float(result["tangent_innovation_scaled_ensemble_suppression"]), 1.0, delta=0.08)
        self.assertLess(float(result["tangent_innovation_normalized_mean"]), 3.0)

    def test_generator_response_closure_cli_exposes_preregistered_gates(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_generator_response_closure.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in ("--output-prefix", "--fit-times", "--horizons", "--linearity-tolerance"):
            self.assertIn(option, completed.stdout)

    def test_generator_response_resolution_cli_exposes_stride_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_generator_response_resolution.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in ("--output", "--strides", "--maximum-time"):
            self.assertIn(option, completed.stdout)

    def test_generator_response_cutoff_cli_exposes_mechanism_filter(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_generator_response_cutoff.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in ("--output", "--stride", "--maximum-time"):
            self.assertIn(option, completed.stdout)

    def test_tangent_noise_covariance_cli_exposes_microscopic_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_tangent_noise_covariance.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in ("--output", "--stride", "--maximum-time"):
            self.assertIn(option, completed.stdout)

    def test_residual_ar1_memory_diagnostic_recovers_white_innovation_limit(self):
        rng = np.random.default_rng(91)
        residual = np.zeros((4000, 2, 3))
        for frame in range(1, len(residual)):
            residual[frame] = 0.7 * residual[frame - 1] + rng.normal(
                scale=0.4, size=(2, 3)
            )

        result = residual_ar1_memory_diagnostic(residual, frame_time=0.5)

        self.assertAlmostEqual(result["ar1_coefficient"], 0.7, delta=0.03)
        self.assertLess(abs(result["innovation_lag1_correlation"]), 0.04)
        self.assertEqual(result["white_noise_candidate"], 1.0)

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
            limited = load_lammps_custom_trajectory(path, maximum_frame_count=1)

        np.testing.assert_array_equal(trajectory["timesteps"], [0, 1000])
        np.testing.assert_array_equal(trajectory["particle_types"], [0, 1])
        np.testing.assert_allclose(trajectory["box_lengths"], [4.0, 4.0, 4.0])
        np.testing.assert_allclose(trajectory["wrapped_positions"][1, 0], [-1.5, 0.0, 0.0])
        np.testing.assert_allclose(trajectory["unwrapped_positions"][1, 0], [2.5, 0.0, 0.0])
        np.testing.assert_allclose(trajectory["unwrapped_positions"][1, 1], [-2.5, 1.0, 0.0])
        np.testing.assert_array_equal(limited["timesteps"], [0])
        self.assertEqual(limited["unwrapped_positions"].shape, (1, 2, 3))

    def test_load_lammps_dump_reads_optional_velocity_and_force_columns(self):
        dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-2 2
-2 2
-2 2
ITEM: ATOMS id type x y z ix iy iz vx vy vz fx fy fz
1 1 1.5 0 0 0 0 0 0.1 0.2 0.3 1.1 1.2 1.3
2 2 -1.5 1 0 0 0 0 -0.1 -0.2 -0.3 -1.1 -1.2 -1.3
"""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trajectory.lammpstrj"
            path.write_text(dump)

            trajectory = load_lammps_custom_trajectory(path)

        np.testing.assert_allclose(trajectory["velocities"][0, 0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(trajectory["forces"][0, 1], [-1.1, -1.2, -1.3])

    def test_load_lammps_dump_reads_velocity_without_force_columns(self):
        dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-2 2
-2 2
-2 2
ITEM: ATOMS id type x y z ix iy iz vx vy vz
1 1 1.5 0 0 0 0 0 0.1 0.2 0.3
2 2 -1.5 1 0 0 0 0 -0.1 -0.2 -0.3
"""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trajectory.lammpstrj"
            path.write_text(dump)

            trajectory = load_lammps_custom_trajectory(path)

        np.testing.assert_allclose(trajectory["velocities"][0, 0], [0.1, 0.2, 0.3])
        self.assertNotIn("forces", trajectory)

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

    def test_prepare_replicate_writes_high_resolution_langevin_protocol(self):
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
                temperature=0.58,
                frame_index=0,
                velocity_seed=45117,
                dynamics="langevin",
                langevin_damping=1.0,
                langevin_seed=99173,
                equilibration_time=1.0,
                production_time=2.0,
                dump_interval_time=1.0,
                high_resolution_duration=0.5,
                high_resolution_dump_interval=0.05,
            )

            lammps_input = (output / "in.production").read_text()
            self.assertIn("fix integrator all nve", lammps_input)
            self.assertIn("fix bath all langevin 0.58 0.58 1 99173", lammps_input)
            self.assertNotIn("fix thermostat all nvt", lammps_input)
            self.assertIn("dump high_resolution all custom 50 high_resolution.lammpstrj", lammps_input)
            self.assertIn("run 500", lammps_input)
            self.assertIn("undump high_resolution", lammps_input)
            self.assertEqual(manifest["ensemble"], "NVE_plus_Langevin")
            self.assertEqual(manifest["high_resolution_saved_frame_interval_tau"], 0.05)

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

    def test_prepare_ensemble_propagates_langevin_protocol(self):
        rng = np.random.default_rng(29)
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
                temperature=0.58,
                frame_indices=[0, 1],
                velocity_seeds=[90117, 90139],
                maximum_absolute_fs=0.5,
                equilibration_time=0.1,
                production_time=1.0,
                dynamics="langevin",
                langevin_damping=1.0,
                langevin_seeds=[99173, 99191],
                high_resolution_duration=0.05,
                high_resolution_dump_interval=0.01,
            )

            input_text = (root / "ensemble" / "replicate_01" / "in.production").read_text()
            self.assertEqual(manifest["dynamics"], "langevin")
            self.assertIn("fix bath all langevin 0.58 0.58 1 99173", input_text)


if __name__ == "__main__":
    unittest.main()
