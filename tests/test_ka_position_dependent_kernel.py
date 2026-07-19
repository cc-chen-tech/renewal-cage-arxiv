import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class PositionDependentKernelTests(unittest.TestCase):
    def test_training_only_radial_basis_and_jacobian(self):
        from ka_position_dependent_kernel import (
            fit_radial_basis_scale,
            radial_vector_basis,
            radial_vector_basis_jacobian,
        )

        training = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [1.0, 2.0, 2.0],
                [2.0, -1.0, 1.0],
            ]
        )
        scale = fit_radial_basis_scale(training)
        frozen = dict(scale)
        held = np.array([[100.0, -50.0, 25.0], [-80.0, 20.0, 40.0]])
        held_basis = radial_vector_basis(held, scale)
        self.assertEqual(scale, frozen)
        self.assertEqual(held_basis.shape, (2, 3, 3))

        radial_square = np.sum(training**2, axis=-1)
        self.assertAlmostEqual(scale["mu_r2"], float(np.mean(radial_square)))
        self.assertAlmostEqual(scale["sigma_r2"], float(np.std(radial_square)))
        self.assertAlmostEqual(
            scale["epsilon_r2"],
            float(np.percentile(radial_square[radial_square > 0.0], 1.0)),
        )

        positions = np.array(
            [
                [0.7, -0.4, 0.2],
                [-0.3, 0.6, 0.8],
            ]
        )
        analytic = radial_vector_basis_jacobian(positions, scale)
        self.assertEqual(analytic.shape, (2, 3, 3, 3))
        step = 1e-6
        numerical = np.empty_like(analytic)
        for coordinate in range(3):
            displacement = np.zeros(3)
            displacement[coordinate] = step
            forward = radial_vector_basis(positions + displacement, scale)
            backward = radial_vector_basis(positions - displacement, scale)
            numerical[..., coordinate] = (forward - backward) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-9, atol=2e-9)

    def test_regularized_mz_system_recovers_causal_kernel(self):
        from ka_position_dependent_kernel import (
            assemble_mz_volterra_system,
            fit_radial_basis_scale,
            predict_mz_drift,
            radial_vector_basis,
            radial_vector_basis_jacobian,
            solve_regularized_mz_kernel,
        )

        rng = np.random.default_rng(20260719)
        position = rng.normal(size=(3, 15, 5, 3))
        velocity = rng.normal(size=position.shape)
        scale = fit_radial_basis_scale(position)
        support = 4
        mean_force = np.array([0.20, -0.10, 0.05])
        memory = np.array(
            [
                [0.30, -0.08, 0.04],
                [0.20, 0.05, -0.03],
                [0.10, -0.02, 0.01],
                [0.04, 0.01, 0.02],
            ]
        )
        basis = radial_vector_basis(position, scale)
        jacobian = radial_vector_basis_jacobian(position, scale)
        jacobian_velocity = np.einsum(
            "ctpbik,ctpk->ctpbi",
            jacobian,
            velocity,
        )
        acceleration = np.zeros_like(position)
        for time_index in range(support - 1, position.shape[1]):
            value = np.einsum(
                "b,cpbi->cpi",
                mean_force,
                basis[:, time_index],
            )
            for lag in range(support):
                value -= np.einsum(
                    "b,cpbi->cpi",
                    memory[lag],
                    jacobian_velocity[:, time_index - lag],
                )
            acceleration[:, time_index] = value

        system = assemble_mz_volterra_system(
            position,
            velocity,
            acceleration,
            scale=scale,
            support=support,
        )
        fitted = solve_regularized_mz_kernel(system, ridge=0.0)
        np.testing.assert_allclose(
            fitted["mean_force_coefficients"],
            mean_force,
            rtol=2e-12,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            fitted["memory_coefficients"],
            memory,
            rtol=2e-12,
            atol=2e-12,
        )
        predicted = predict_mz_drift(
            position,
            velocity,
            scale=scale,
            mean_force_coefficients=fitted["mean_force_coefficients"],
            memory_coefficients=fitted["memory_coefficients"],
        )
        np.testing.assert_allclose(
            predicted,
            acceleration[:, support - 1 :],
            rtol=2e-12,
            atol=2e-12,
        )
        regularized = solve_regularized_mz_kernel(system, ridge=1e-2)
        self.assertTrue(np.isfinite(regularized["condition_number"]))
        self.assertLess(
            np.linalg.norm(regularized["memory_coefficients"]),
            np.linalg.norm(fitted["memory_coefficients"]),
        )

    def test_stationary_scalar_baseline_uses_only_linear_radial_basis(self):
        from ka_position_dependent_kernel import (
            assemble_mz_volterra_system,
            fit_radial_basis_scale,
            predict_mz_drift,
            solve_regularized_mz_kernel,
        )

        rng = np.random.default_rng(44)
        position = rng.normal(size=(2, 9, 4, 3))
        velocity = rng.normal(size=position.shape)
        acceleration = 0.4 * position.copy()
        acceleration[:, 1:] -= 0.2 * velocity[:, :-1]
        scale = fit_radial_basis_scale(position)
        system = assemble_mz_volterra_system(
            position,
            velocity,
            acceleration,
            scale=scale,
            support=2,
            basis_indices=(0,),
        )
        self.assertEqual(system["design"].shape[1], 3)
        self.assertEqual(system["basis_indices"], (0,))
        fitted = solve_regularized_mz_kernel(system, ridge=0.0)
        self.assertEqual(fitted["mean_force_coefficients"].shape, (1,))
        self.assertEqual(fitted["memory_coefficients"].shape, (2, 1))
        predicted = predict_mz_drift(
            position,
            velocity,
            scale=scale,
            mean_force_coefficients=fitted["mean_force_coefficients"],
            memory_coefficients=fitted["memory_coefficients"],
            basis_indices=(0,),
        )
        np.testing.assert_allclose(predicted, acceleration[:, 1:], atol=2e-12)

    def test_mz_system_rejects_invalid_paths_support_and_ridge(self):
        from ka_position_dependent_kernel import (
            assemble_mz_volterra_system,
            fit_radial_basis_scale,
            solve_regularized_mz_kernel,
        )

        rng = np.random.default_rng(8)
        position = rng.normal(size=(2, 8, 3, 3))
        velocity = rng.normal(size=position.shape)
        acceleration = rng.normal(size=position.shape)
        scale = fit_radial_basis_scale(position)

        nonfinite = position.copy()
        nonfinite[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            assemble_mz_volterra_system(
                nonfinite,
                velocity,
                acceleration,
                scale=scale,
                support=2,
            )
        with self.assertRaisesRegex(ValueError, "align"):
            assemble_mz_volterra_system(
                position,
                velocity[:, :-1],
                acceleration,
                scale=scale,
                support=2,
            )
        with self.assertRaisesRegex(ValueError, "support"):
            assemble_mz_volterra_system(
                position,
                velocity,
                acceleration,
                scale=scale,
                support=9,
            )

        system = assemble_mz_volterra_system(
            position,
            velocity,
            acceleration,
            scale=scale,
            support=2,
        )
        with self.assertRaisesRegex(ValueError, "ridge"):
            solve_regularized_mz_kernel(system, ridge=-1e-4)
        unresolved = dict(system)
        unresolved["design"] = np.zeros_like(system["design"])
        with self.assertRaisesRegex(ValueError, "unresolved columns"):
            solve_regularized_mz_kernel(unresolved, ridge=1e-4)

    def test_two_position_auxiliary_recursion_matches_direct_convolution(self):
        from ka_position_dependent_kernel import (
            fit_radial_basis_scale,
            real_pole_history_features,
            reconstruct_auxiliary_innovations,
            two_position_auxiliary_features,
        )

        rng = np.random.default_rng(73)
        position = rng.normal(size=(2, 9, 4, 3))
        velocity = rng.normal(size=position.shape)
        scale = fit_radial_basis_scale(position)
        rates = np.array([0.3, 1.2])
        coefficients = np.array([[1.0, 0.2, 0.1], [0.5, -0.1, 0.2]])
        frame_time = 0.01
        result = two_position_auxiliary_features(
            position,
            velocity,
            scale=scale,
            decay_rates=rates,
            coupling_coefficients=coefficients,
            frame_time=frame_time,
        )
        coupling = result["coupling"]
        rho = np.exp(-rates * frame_time)
        forcing = -np.expm1(-rates * frame_time) / rates
        direct = np.zeros_like(position)
        for time_index in range(1, position.shape[1]):
            for history_index in range(time_index):
                weight = forcing * rho ** (time_index - 1 - history_index)
                historical_projection = np.einsum(
                    "cpaij,cpj->cpai",
                    np.swapaxes(coupling[:, history_index], -1, -2),
                    velocity[:, history_index],
                )
                direct[:, time_index] -= np.einsum(
                    "a,cpaij,cpaj->cpi",
                    weight,
                    coupling[:, time_index],
                    historical_projection,
                )
        np.testing.assert_allclose(
            result["force"],
            direct,
            rtol=3e-14,
            atol=3e-14,
        )
        innovation = reconstruct_auxiliary_innovations(
            result["auxiliary"],
            position,
            velocity,
            scale=scale,
            decay_rates=rates,
            coupling_coefficients=coefficients,
            frame_time=frame_time,
        )
        np.testing.assert_allclose(innovation, 0.0, rtol=0.0, atol=3e-16)
        self.assertEqual(result["positive_prony_factorization"], 1.0)

        real_pole = real_pole_history_features(
            position,
            velocity,
            scale=scale,
            decay_rates=rates,
            frame_time=frame_time,
        )
        self.assertEqual(real_pole["positive_prony_factorization"], 0.0)
        self.assertEqual(real_pole["history_features"].shape, (2, 9, 4, 2, 3, 3))

    def test_classifier_separates_mz_real_pole_and_positive_prony_claims(self):
        from ka_position_dependent_kernel import (
            classify_position_dependent_kernel_gate,
        )

        def passing_rows():
            rows = []
            model_values = {
                "stationary_scalar_nonparametric_volterra": (1.0, 1.0),
                "finite_basis_mz_position_kernel": (0.8, 0.8),
                "past_position_real_pole": (0.85, 0.82),
                "two_position_positive_prony": (0.86, 0.82),
                "time_permuted_position_null": (0.95, 0.95),
            }
            for held_clone in range(1, 5):
                for model, (rmse, nll) in model_values.items():
                    rows.append(
                        {
                            "held_clone_index": float(held_clone),
                            "model": model,
                            "drift_rmse": rmse,
                            "drift_nll": nll,
                            "held_scalar_component_count": 100.0,
                            "maximum_normalized_resolved_basis_residual_correlation": 0.1,
                            "maximum_normalized_auxiliary_innovation_autocorrelation": 0.1,
                            "second_fdt_covariance_normalized_rmse": 0.2,
                            "auxiliary_diagnostics_applicable": 1.0,
                            "selected_auxiliary_rank": 2.0,
                            "all_selected_decay_rates_positive": 1.0,
                            "all_fitted_arrays_finite": 1.0,
                        }
                    )
            return rows

        passed = classify_position_dependent_kernel_gate(passing_rows())
        self.assertEqual(passed["real_ka_position_dependent_mz_kernel_identified"], 1.0)
        self.assertEqual(passed["past_position_real_pole_identified_in_ka"], 1.0)
        self.assertEqual(passed["two_position_positive_prony_identified_in_ka"], 1.0)
        self.assertEqual(passed["positive_prony_kernel_identified_in_ka"], 1.0)
        self.assertEqual(passed["finite_auxiliary_rank_identified_in_ka"], 1.0)
        self.assertEqual(passed["selected_auxiliary_rank"], 2.0)
        self.assertEqual(passed["oscillatory_matrix_bath_authorized"], 0.0)
        self.assertLess(passed["mz_rmse_ratio_t95_high"], 1.0)

        inconsistent = passing_rows()
        for row in inconsistent:
            if row["model"] == "two_position_positive_prony" and row[
                "held_clone_index"
            ] == 4.0:
                row["selected_auxiliary_rank"] = 4.0
        inconsistent_verdict = classify_position_dependent_kernel_gate(inconsistent)
        self.assertEqual(
            inconsistent_verdict["two_position_positive_prony_identified_in_ka"],
            1.0,
        )
        self.assertEqual(
            inconsistent_verdict["finite_auxiliary_rank_identified_in_ka"],
            0.0,
        )

        mz_failed = passing_rows()
        for row in mz_failed:
            if row["model"] == "finite_basis_mz_position_kernel" and row[
                "held_clone_index"
            ] == 1.0:
                row["drift_rmse"] = 0.95
        failed_verdict = classify_position_dependent_kernel_gate(mz_failed)
        self.assertEqual(
            failed_verdict["real_ka_position_dependent_mz_kernel_identified"],
            0.0,
        )
        self.assertEqual(failed_verdict["positive_prony_kernel_identified_in_ka"], 0.0)

        realization_failed = passing_rows()
        for row in realization_failed:
            if row["model"] in {
                "past_position_real_pole",
                "two_position_positive_prony",
            }:
                row["second_fdt_covariance_normalized_rmse"] = 0.4
        unresolved = classify_position_dependent_kernel_gate(realization_failed)
        self.assertEqual(unresolved["real_ka_position_dependent_mz_kernel_identified"], 1.0)
        self.assertEqual(unresolved["past_position_real_pole_identified_in_ka"], 0.0)
        self.assertEqual(unresolved["positive_prony_kernel_identified_in_ka"], 0.0)
        self.assertEqual(unresolved["oscillatory_matrix_bath_authorized"], 1.0)

        for verdict in (passed, inconsistent_verdict, failed_verdict, unresolved):
            self.assertEqual(verdict["real_ka_kernel_identifiability_test_required"], 1.0)
            for claim in (
                "autonomous_single_particle_gle_allowed",
                "complete_event_clock_closure_allowed",
                "kramers_escape_claim_allowed",
                "spatial_facilitation_claim_allowed",
                "thermodynamic_claim_allowed",
            ):
                self.assertEqual(verdict[claim], 0.0)

    def test_held_diagnostics_use_frozen_training_variance_and_aligned_features(self):
        from ka_position_dependent_kernel import (
            fit_radial_basis_scale,
            held_kernel_diagnostics,
        )

        rng = np.random.default_rng(101)
        position = rng.normal(size=(1, 12, 6, 3))
        velocity = rng.normal(size=position.shape)
        observed = rng.normal(size=position.shape)
        residual = 0.15 * rng.normal(size=position.shape)
        predicted = observed - residual
        scale = fit_radial_basis_scale(position)
        innovation = rng.normal(size=(1, 11, 6, 2, 3))
        observed_fdt = np.array([1.0, 0.4, 0.2])
        target_fdt = np.array([1.0, 0.5, 0.1])
        diagnostics = held_kernel_diagnostics(
            observed,
            predicted,
            position,
            velocity,
            scale=scale,
            training_residual_variance=2.0,
            maximum_lag=2,
            auxiliary_innovation=innovation,
            observed_fdt_covariance=observed_fdt,
            target_fdt_covariance=target_fdt,
        )
        expected_rmse = np.sqrt(np.mean(residual**2)) / np.sqrt(
            np.mean(observed**2)
        )
        expected_nll = 0.5 * residual.size * np.log(4.0 * np.pi) + 0.25 * np.sum(
            residual**2
        )
        self.assertAlmostEqual(diagnostics["drift_rmse"], expected_rmse)
        self.assertAlmostEqual(diagnostics["drift_nll"], expected_nll)
        self.assertLessEqual(
            diagnostics[
                "maximum_normalized_resolved_basis_residual_correlation"
            ],
            1.0,
        )
        self.assertLessEqual(
            diagnostics[
                "maximum_normalized_auxiliary_innovation_autocorrelation"
            ],
            1.0,
        )
        expected_fdt = np.sqrt(np.mean((observed_fdt - target_fdt) ** 2)) / np.sqrt(
            np.mean(target_fdt**2)
        )
        self.assertAlmostEqual(
            diagnostics["second_fdt_covariance_normalized_rmse"],
            expected_fdt,
        )
        self.assertEqual(diagnostics["held_scalar_component_count"], residual.size)
        self.assertEqual(diagnostics["auxiliary_diagnostics_applicable"], 1.0)
        self.assertEqual(diagnostics["thermodynamic_claim_allowed"], 0.0)


if __name__ == "__main__":
    unittest.main()
