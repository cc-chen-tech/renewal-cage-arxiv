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


if __name__ == "__main__":
    unittest.main()
