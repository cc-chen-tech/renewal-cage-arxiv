import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class NonlinearBathGleTests(unittest.TestCase):
    @staticmethod
    def controls(**updates):
        from nonlinear_bath_gle import NonlinearBathControls

        values = {
            "temperature": 0.58,
            "friction": 1.0,
            "period": 1.0,
            "barrier": 1.74,
            "rates": np.array([0.20, 1.00]),
            "amplitudes": np.array([1.00, 0.55]),
            "modulation": np.array([0.45, 0.25]),
            "phases": np.array([0.0, 0.5 * np.pi]),
            "time_step": 0.001,
        }
        values.update(updates)
        return NonlinearBathControls(**values)

    def test_periodic_potential_gradient_matches_centered_difference(self):
        from nonlinear_bath_gle import (
            periodic_potential,
            periodic_potential_gradient,
        )

        position = np.array([-0.37, -0.11, 0.0, 0.19, 0.44])
        step = 1e-7
        numerical = (
            periodic_potential(position + step, barrier=1.74, period=1.0)
            - periodic_potential(position - step, barrier=1.74, period=1.0)
        ) / (2.0 * step)
        actual = periodic_potential_gradient(position, barrier=1.74, period=1.0)
        np.testing.assert_allclose(actual, numerical, rtol=2e-9, atol=2e-9)

    def test_periodic_coupling_uses_every_frozen_mode(self):
        from nonlinear_bath_gle import periodic_coupling

        controls = self.controls()
        position = np.array([0.0, 0.25])
        actual = periodic_coupling(position, controls=controls)
        expected = controls.amplitudes[None] * (
            1.0
            + controls.modulation[None]
            * np.cos(
                2.0 * np.pi * position[:, None] / controls.period
                + controls.phases[None]
            )
        )
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual.shape, (2, 2))

    def test_antisymmetric_coupling_energy_error_is_second_order(self):
        from nonlinear_bath_gle import nonlinear_bath_step

        position = np.array([0.13])
        momentum = np.array([0.8])
        auxiliary = np.array([[0.4, -0.7]])
        normal_p = np.zeros_like(momentum)
        normal_z = np.zeros_like(auxiliary)
        initial_energy = 0.5 * (
            momentum[0] ** 2 + float(np.sum(auxiliary[0] ** 2))
        )
        errors = []
        for step in (1e-3, 5e-4):
            controls = self.controls(
                temperature=0.0,
                friction=0.0,
                barrier=0.0,
                rates=np.zeros(2),
                modulation=np.zeros(2),
                time_step=step,
            )
            result = nonlinear_bath_step(
                position,
                momentum,
                auxiliary,
                normal_p=normal_p,
                normal_z=normal_z,
                controls=controls,
            )
            final_energy = 0.5 * (
                float(result["momentum"][0] ** 2)
                + float(np.sum(result["auxiliary"][0] ** 2))
            )
            errors.append(abs(final_energy - initial_energy))
        self.assertGreater(errors[0], 0.0)
        self.assertLess(errors[1] / errors[0], 0.26)

    def test_exact_ou_reconstruction_matches_supplied_noise_path(self):
        from nonlinear_bath_gle import (
            nonlinear_bath_step,
            reconstruct_auxiliary_path,
        )

        controls = self.controls()
        rng = np.random.default_rng(20260811)
        steps, trajectories = 25, 4
        normal_p = rng.normal(size=(steps, trajectories))
        normal_z = rng.normal(size=(steps, trajectories, 2))
        position = np.linspace(-0.2, 0.2, trajectories)
        momentum = rng.normal(size=trajectories)
        auxiliary = rng.normal(size=(trajectories, 2))
        positions = [position.copy()]
        momenta = [momentum.copy()]
        auxiliaries = [auxiliary.copy()]
        for index in range(steps):
            result = nonlinear_bath_step(
                position,
                momentum,
                auxiliary,
                normal_p=normal_p[index],
                normal_z=normal_z[index],
                controls=controls,
            )
            position = np.asarray(result["position"])
            momentum = np.asarray(result["momentum"])
            auxiliary = np.asarray(result["auxiliary"])
            positions.append(position.copy())
            momenta.append(momentum.copy())
            auxiliaries.append(auxiliary.copy())
        reconstructed = reconstruct_auxiliary_path(
            np.asarray(auxiliaries[0]),
            positions=np.asarray(positions[:-1]),
            momenta=np.asarray(momenta[:-1]),
            normal_increments=normal_z,
            controls=controls,
        )
        np.testing.assert_allclose(
            reconstructed,
            np.asarray(auxiliaries),
            rtol=0.0,
            atol=5e-14,
        )

    def test_eliminated_kernel_matches_mode_sum_and_keeps_claims_closed(self):
        from nonlinear_bath_gle import eliminated_memory_kernel

        controls = self.controls()
        left = np.array([0.10, 0.20])
        right = np.array([-0.15, 0.25])
        lag = 0.37
        result = eliminated_memory_kernel(
            left,
            right,
            lag=lag,
            controls=controls,
        )
        coupling_left = controls.amplitudes[None] * (
            1.0
            + controls.modulation[None]
            * np.cos(
                2.0 * np.pi * left[:, None] / controls.period
                + controls.phases[None]
            )
        )
        coupling_right = controls.amplitudes[None] * (
            1.0
            + controls.modulation[None]
            * np.cos(
                2.0 * np.pi * right[:, None] / controls.period
                + controls.phases[None]
            )
        )
        expected = np.sum(
            coupling_left
            * np.exp(-controls.rates[None] * lag)
            * coupling_right,
            axis=1,
        )
        np.testing.assert_allclose(result["kernel"], expected)
        self.assertEqual(result["exact_nonlinear_bath_elimination_supported"], 0.0)
        self.assertEqual(result["autonomous_single_particle_gle_allowed"], 0.0)
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)


if __name__ == "__main__":
    unittest.main()
