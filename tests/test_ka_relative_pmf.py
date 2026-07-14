import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_relative_pmf import (  # noqa: E402
    fit_gaussian_relative_pmf,
    underdamped_ou_propagator,
)


class RelativePmfTests(unittest.TestCase):
    def test_gaussian_state_counting_recovers_fdt_and_harmonic_acceleration(self):
        rng = np.random.default_rng(20260714)
        sigma_u_squared = 0.004
        sigma_p_squared = 0.22
        friction = 1.1
        relative_position = rng.normal(
            scale=np.sqrt(sigma_u_squared), size=(20000, 3)
        )
        relative_velocity = rng.normal(
            scale=np.sqrt(sigma_p_squared), size=(20000, 3)
        )
        result = fit_gaussian_relative_pmf(
            relative_position,
            relative_velocity,
            relative_noise_variance_rate=2.0 * friction * sigma_p_squared,
            friction=friction,
        )

        self.assertAlmostEqual(
            float(result["relative_position_variance"]), sigma_u_squared, delta=8e-5
        )
        self.assertAlmostEqual(
            float(result["relative_velocity_variance"]), sigma_p_squared, delta=0.004
        )
        self.assertAlmostEqual(
            float(result["fdt_velocity_variance"]), sigma_p_squared, places=12
        )
        self.assertAlmostEqual(
            float(result["harmonic_acceleration_stiffness"]),
            sigma_p_squared / sigma_u_squared,
            delta=1.5,
        )

    def test_underdamped_ou_propagator_obeys_semigroup_and_initial_generator(self):
        kappa = 240.0
        friction = 1.0
        step = 1e-5
        identity = underdamped_ou_propagator(kappa, friction, 0.0)
        first = underdamped_ou_propagator(kappa, friction, step)
        double = underdamped_ou_propagator(kappa, friction, 2.0 * step)
        generator = np.array([[0.0, 1.0], [-kappa, -friction]])

        np.testing.assert_allclose(identity, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(first @ first, double, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            (first - identity) / step,
            generator,
            rtol=2e-3,
            atol=2e-3,
        )

    def test_invalid_pmf_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            underdamped_ou_propagator(-1.0, 1.0, 0.1)
        with self.assertRaisesRegex(ValueError, "align"):
            fit_gaussian_relative_pmf(
                np.zeros((4, 3)),
                np.zeros((5, 3)),
                relative_noise_variance_rate=1.0,
                friction=1.0,
            )

    def test_real_data_cli_exposes_fixed_pmf_and_ou_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_relative_pmf_ou_boundary.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for required in (
            "--drift-cache-directory",
            "--covariance-cache-directory",
            "--radial-bin-count",
            "--correlation-lags",
        ):
            self.assertIn(required, completed.stdout)


if __name__ == "__main__":
    unittest.main()
