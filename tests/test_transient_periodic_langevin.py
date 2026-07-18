import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transient_periodic_langevin import (  # noqa: E402
    TransientPeriodicParams,
    conservative_forces,
    potential_energy,
    simulate_transient_periodic_langevin,
)


class TransientPotentialTests(unittest.TestCase):
    def setUp(self):
        self.params = TransientPeriodicParams(
            temperature=0.7,
            period=1.3,
            base_barrier=2.5,
            elastic_stiffness=0.8,
            barrier_stiffness=1.1,
            barrier_coupling=0.6,
            gamma_x=1.2,
            gamma_q=4.0,
            gamma_z=3.0,
        )

    def test_forces_are_negative_potential_gradients(self):
        x = np.array([[0.17, -0.28]])
        q = np.array([[-0.04, 0.09]])
        z = np.array([0.31])
        force_x, force_q, force_z = conservative_forces(x, q, z, self.params)
        epsilon = 1e-6

        for coordinate in range(x.shape[1]):
            plus = x.copy()
            minus = x.copy()
            plus[0, coordinate] += epsilon
            minus[0, coordinate] -= epsilon
            derivative = (
                potential_energy(plus, q, z, self.params)[0]
                - potential_energy(minus, q, z, self.params)[0]
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(force_x[0, coordinate], -derivative, delta=2e-6)

        for coordinate in range(q.shape[1]):
            plus = q.copy()
            minus = q.copy()
            plus[0, coordinate] += epsilon
            minus[0, coordinate] -= epsilon
            derivative = (
                potential_energy(x, plus, z, self.params)[0]
                - potential_energy(x, minus, z, self.params)[0]
            ) / (2.0 * epsilon)
            self.assertAlmostEqual(force_q[0, coordinate], -derivative, delta=2e-6)

        plus = z.copy()
        minus = z.copy()
        plus[0] += epsilon
        minus[0] -= epsilon
        derivative = (
            potential_energy(x, q, plus, self.params)[0]
            - potential_energy(x, q, minus, self.params)[0]
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(force_z[0], -derivative, delta=2e-6)

    def test_joint_integer_period_translation_is_invariant(self):
        x = np.array([[0.17, -0.28], [0.4, 0.2]])
        q = np.array([[-0.04, 0.09], [0.1, -0.3]])
        z = np.array([0.31, -0.2])
        shift = 2.0 * self.params.period

        before_energy = potential_energy(x, q, z, self.params)
        before_forces = conservative_forces(x, q, z, self.params)
        after_energy = potential_energy(x + shift, q + shift, z, self.params)
        after_forces = conservative_forces(x + shift, q + shift, z, self.params)

        np.testing.assert_allclose(after_energy, before_energy, atol=1e-12)
        for after, before in zip(after_forces, before_forces):
            np.testing.assert_allclose(after, before, atol=1e-12)

    def test_parameters_reject_nonphysical_domains(self):
        base = dict(
            temperature=1.0,
            period=1.0,
            base_barrier=2.0,
            elastic_stiffness=0.5,
            barrier_stiffness=1.0,
            barrier_coupling=0.5,
            gamma_x=1.0,
            gamma_q=1.0,
            gamma_z=1.0,
        )
        for key in (
            "temperature",
            "period",
            "base_barrier",
            "barrier_stiffness",
            "gamma_x",
            "gamma_q",
            "gamma_z",
        ):
            values = dict(base)
            values[key] = 0.0
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    TransientPeriodicParams(**values)
        for key in ("elastic_stiffness", "barrier_coupling"):
            values = dict(base)
            values[key] = -0.1
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    TransientPeriodicParams(**values)


class TransientIntegratorTests(unittest.TestCase):
    @staticmethod
    def params(**updates):
        values = dict(
            temperature=0.7,
            period=1.0,
            base_barrier=2.5,
            elastic_stiffness=0.6,
            barrier_stiffness=1.0,
            barrier_coupling=0.4,
            gamma_x=1.0,
            gamma_q=4.0,
            gamma_z=3.0,
        )
        values.update(updates)
        return TransientPeriodicParams(**values)

    def test_simulator_is_seed_deterministic(self):
        kwargs = dict(
            trajectory_count=12,
            dimension=2,
            dt=0.001,
            burnin_steps=30,
            production_steps=50,
            record_stride=5,
        )
        first = simulate_transient_periodic_langevin(
            self.params(), seed=17, **kwargs
        )
        second = simulate_transient_periodic_langevin(
            self.params(), seed=17, **kwargs
        )
        third = simulate_transient_periodic_langevin(
            self.params(), seed=18, **kwargs
        )

        for key in ("positions", "environment_positions", "barrier_coordinates"):
            self.assertTrue(np.array_equal(first[key], second[key]))
        self.assertFalse(np.array_equal(first["positions"], third["positions"]))
        self.assertEqual(first["all_finite"], 1.0)
        self.assertAlmostEqual(first["record_dt"], 0.005)

    def test_high_barrier_wrapped_variance_matches_local_equipartition(self):
        params = self.params(
            temperature=0.5,
            base_barrier=6.0,
            elastic_stiffness=0.0,
            barrier_coupling=0.0,
        )
        result = simulate_transient_periodic_langevin(
            params,
            trajectory_count=800,
            dimension=1,
            dt=0.0005,
            burnin_steps=3000,
            production_steps=2000,
            record_stride=20,
            seed=29,
        )
        positions = result["positions"]
        wrapped = positions - np.floor(positions / params.period + 0.5) * params.period
        observed = float(np.var(wrapped))
        expected = params.temperature / (
            2.0 * np.pi**2 * params.base_barrier / params.period**2
        )
        self.assertAlmostEqual(observed / expected, 1.0, delta=0.15)

    def test_elastic_environment_does_not_pin_common_translation(self):
        params = self.params(
            temperature=1.0,
            base_barrier=0.8,
            elastic_stiffness=0.5,
            barrier_coupling=0.0,
            gamma_q=2.0,
        )
        result = simulate_transient_periodic_langevin(
            params,
            trajectory_count=256,
            dimension=1,
            dt=0.002,
            burnin_steps=1000,
            production_steps=4000,
            record_stride=100,
            seed=31,
        )
        endpoint = result["positions"][-1, :, 0] - result["positions"][0, :, 0]
        self.assertGreater(float(np.mean(endpoint**2)), 0.1)

    def test_integrator_rejects_reference_curvature_instability(self):
        with self.assertRaisesRegex(ValueError, "stability"):
            simulate_transient_periodic_langevin(
                self.params(base_barrier=3.0),
                trajectory_count=4,
                dimension=1,
                dt=0.01,
                burnin_steps=1,
                production_steps=2,
                record_stride=1,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
