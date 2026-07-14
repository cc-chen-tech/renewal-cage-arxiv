import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_markov_bath import (  # noqa: E402
    fit_stationary_vector_autoregression,
    simulate_mori_with_var_bath,
    vector_autoregressive_residual,
)


class MarkovBathTests(unittest.TestCase):
    def test_stationary_var_fit_recovers_known_two_component_var2(self):
        coefficient = np.array(
            [
                [[0.55, 0.08], [-0.04, 0.42]],
                [[-0.18, 0.03], [0.02, -0.12]],
            ]
        )
        rng = np.random.default_rng(20260714)
        series = np.zeros((1600, 256, 2))
        for frame in range(2, len(series)):
            series[frame] = (
                series[frame - 1] @ coefficient[0].T
                + series[frame - 2] @ coefficient[1].T
                + rng.normal(scale=[0.35, 0.25], size=(series.shape[1], 2))
            )
        series = series[200:]

        fit = fit_stationary_vector_autoregression(
            series,
            order=2,
            ridge_regularization=1e-8,
        )

        np.testing.assert_allclose(fit["coefficients"], coefficient, atol=0.015)
        self.assertLess(float(fit["spectral_radius"]), 1.0)
        self.assertGreater(float(fit["minimum_noise_covariance_eigenvalue"]), 0.0)
        residual = vector_autoregressive_residual(
            series,
            np.asarray(fit["coefficients"]),
            mean=np.asarray(fit["mean"]),
        )
        lag_one = np.einsum("tni,tnj->ij", residual[1:], residual[:-1])
        lag_one /= (len(residual) - 1) * residual.shape[1]
        self.assertLess(float(np.max(np.abs(lag_one))), 0.002)

    def test_mori_var_simulator_uses_white_driven_colored_innovation(self):
        initial_state = np.zeros((8000, 1, 1))
        initial_innovation = np.zeros((8000, 1, 1))
        state, innovation = simulate_mori_with_var_bath(
            initial_state,
            np.array([[[0.6]]]),
            initial_innovation,
            np.array([[[0.7]]]),
            innovation_mean=np.zeros(1),
            white_noise_covariance=np.array([[0.51]]),
            output_count=400,
            rng=np.random.default_rng(11),
        )

        innovation = innovation[:, 100:, 0]
        state = state[:, 100:, 0]
        self.assertAlmostEqual(float(np.var(innovation)), 1.0, delta=0.025)
        self.assertAlmostEqual(
            float(np.mean(innovation[:, 1:] * innovation[:, :-1])),
            0.7,
            delta=0.025,
        )
        self.assertGreater(float(np.var(state)), float(np.var(innovation)))

    def test_mori_var_simulator_can_draw_empirical_white_residuals(self):
        state, innovation = simulate_mori_with_var_bath(
            np.zeros((4, 1, 1)),
            np.zeros((1, 1, 1)),
            np.zeros((4, 1, 1)),
            np.zeros((1, 1, 1)),
            innovation_mean=np.zeros(1),
            white_noise_covariance=np.ones((1, 1)),
            white_noise_pool=np.array([[-2.0], [3.0]]),
            output_count=20,
            rng=np.random.default_rng(8),
        )

        self.assertTrue(set(np.unique(innovation[:, 1:])).issubset({-2.0, 3.0}))
        np.testing.assert_allclose(state[:, 1:], innovation[:, 1:])


if __name__ == "__main__":
    unittest.main()
