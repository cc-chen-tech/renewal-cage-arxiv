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


if __name__ == "__main__":
    unittest.main()
