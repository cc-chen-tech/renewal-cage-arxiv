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


if __name__ == "__main__":
    unittest.main()
