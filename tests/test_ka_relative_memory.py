import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_relative_memory import (  # noqa: E402
    estimate_isoconfigurational_bias,
    invert_harmonic_velocity_memory_kernel,
    propagate_harmonic_gle_correlations,
)


class RelativeMemoryTests(unittest.TestCase):
    def test_harmonic_volterra_inversion_recovers_causal_kernel(self):
        kernel = np.array([8.0, 3.0, -0.6, 0.2, 0.0])
        frame_time = 0.02
        kappa = 4.0
        friction = 0.5
        velocity_variance = 0.7
        position_variance = velocity_variance / kappa
        correlations = propagate_harmonic_gle_correlations(
            kernel,
            harmonic_acceleration_stiffness=kappa,
            friction=friction,
            frame_time=frame_time,
            output_count=30,
            position_variance=position_variance,
            velocity_variance=velocity_variance,
        )

        recovered = invert_harmonic_velocity_memory_kernel(
            correlations["velocity_velocity_correlation"],
            correlations["position_velocity_correlation"],
            harmonic_acceleration_stiffness=kappa,
            friction=friction,
            frame_time=frame_time,
            kernel_count=len(kernel),
        )

        np.testing.assert_allclose(recovered, kernel, rtol=1e-10, atol=1e-10)

    def test_isoconfigurational_bias_averages_clone_and_time_axes_only(self):
        paths = np.zeros((3, 4, 2, 3))
        paths[:, :, 0, 0] = np.arange(3)[:, None] + np.arange(4)[None, :]
        paths[:, :, 1, 1] = 7.0

        bias = estimate_isoconfigurational_bias(paths)

        self.assertEqual(bias.shape, (2, 3))
        self.assertAlmostEqual(float(bias[0, 0]), 2.5)
        self.assertAlmostEqual(float(bias[1, 1]), 7.0)

    def test_invalid_kernel_controls_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            propagate_harmonic_gle_correlations(
                np.ones(2),
                harmonic_acceleration_stiffness=-1.0,
                friction=1.0,
                frame_time=0.1,
                output_count=4,
                position_variance=1.0,
                velocity_variance=1.0,
            )

    def test_real_data_cli_exposes_fixed_volterra_and_fdt_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "analyze_ka_relative_harmonic_volterra_fdt.py"
                ),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for required in (
            "--drift-cache-directory",
            "--covariance-cache-directory",
            "--kernel-count",
            "--maximum-lag",
        ):
            self.assertIn(required, completed.stdout)


if __name__ == "__main__":
    unittest.main()
