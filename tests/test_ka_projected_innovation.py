import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_projected_innovation import (  # noqa: E402
    block_projected_ito_increments,
    fit_constant_joint_covariance,
    multivariate_noise_covariance_diagnostic,
)


class ProjectedInnovationTests(unittest.TestCase):
    def test_left_ito_blocks_sum_drift_and_covariance_without_overlap(self):
        frame_time = 0.25
        state = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)
        drift = np.full_like(state, 0.5)
        covariance_rate = np.broadcast_to(np.eye(2), (5, 2, 2, 2)).copy()

        result = block_projected_ito_increments(
            state,
            drift,
            covariance_rate,
            frame_time=frame_time,
            stride=2,
            scheme="left",
        )

        expected = state[2::2] - state[:-2:2] - 2.0 * frame_time * 0.5
        np.testing.assert_allclose(result["residual"], expected)
        np.testing.assert_allclose(
            result["integrated_covariance"],
            2.0 * frame_time * np.broadcast_to(np.eye(2), (2, 2, 2, 2)),
        )
        np.testing.assert_allclose(result["starting_state"], state[:-2:2])

    def test_multivariate_diagnostic_calibrates_correlated_gaussian_noise(self):
        rng = np.random.default_rng(20260714)
        members = 80
        intervals = 800
        dimension = 6
        transform = rng.normal(size=(members, intervals, dimension, dimension))
        covariance = np.einsum("...ik,...jk->...ij", transform, transform)
        covariance += 0.5 * np.eye(dimension)
        normal = rng.normal(size=(members, intervals, dimension))
        square_root = np.linalg.cholesky(covariance)
        residual = np.einsum("...ij,...j->...i", square_root, normal)
        state = rng.normal(size=(members, intervals, 9))

        result = multivariate_noise_covariance_diagnostic(
            residual,
            covariance,
            starting_state=state,
        )

        self.assertAlmostEqual(float(result["trace_variance_ratio"]), 1.0, delta=0.02)
        self.assertAlmostEqual(
            float(result["mean_squared_mahalanobis_per_dimension"]), 1.0, delta=0.02
        )
        self.assertLess(float(result["maximum_absolute_whitened_mean"]), 0.02)
        self.assertLess(float(result["maximum_absolute_whitened_covariance_error"]), 0.02)
        self.assertLess(float(result["maximum_absolute_whitened_lag1_correlation"]), 0.02)
        self.assertLess(float(result["maximum_absolute_whitened_state_correlation"]), 0.02)
        self.assertEqual(float(result["thermodynamic_claim_allowed"]), 0.0)

    def test_adams_bashforth_two_uses_only_current_and_past_drift(self):
        state = np.arange(6, dtype=float)[:, None, None]
        drift = np.arange(6, dtype=float)[:, None, None] / 10.0
        covariance = np.ones((6, 1, 1, 1))

        result = block_projected_ito_increments(
            state,
            drift,
            covariance,
            frame_time=0.5,
            stride=2,
            scheme="adams_bashforth2",
        )

        base = state[2:] - state[1:-1] - 0.5 * (
            1.5 * drift[1:-1] - 0.5 * drift[:-2]
        )
        np.testing.assert_allclose(
            result["residual"], base[:4].reshape(2, 2, 1, 1).sum(axis=1)
        )
        np.testing.assert_array_equal(result["starting_frame_indices"], [1, 3])

    def test_invalid_covariance_or_stride_is_rejected(self):
        state = np.zeros((3, 1, 2))
        drift = np.zeros_like(state)
        covariance = np.broadcast_to(np.eye(2), (3, 1, 2, 2)).copy()
        covariance[0, 0, 0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive"):
            block_projected_ito_increments(
                state,
                drift,
                covariance,
                frame_time=0.1,
                stride=1,
                scheme="left",
            )
        with self.assertRaisesRegex(ValueError, "stride"):
            block_projected_ito_increments(
                state,
                drift,
                np.broadcast_to(np.eye(2), (3, 1, 2, 2)),
                frame_time=0.1,
                stride=0,
                scheme="left",
            )

    def test_constant_joint_covariance_models_preserve_requested_symmetry(self):
        rng = np.random.default_rng(91)
        raw = rng.normal(size=(30, 6, 6))
        covariance = np.einsum("nik,njk->nij", raw, raw) + np.eye(6)

        full = fit_constant_joint_covariance(covariance, model="full")
        isotropic = fit_constant_joint_covariance(covariance, model="block_isotropic")
        uncorrelated = fit_constant_joint_covariance(
            covariance, model="block_isotropic_uncorrelated"
        )
        scalar = fit_constant_joint_covariance(covariance, model="single_scalar")

        np.testing.assert_allclose(full, np.mean(covariance, axis=0))
        for block in (isotropic[:3, :3], isotropic[3:, 3:]):
            np.testing.assert_allclose(block, np.eye(3) * np.trace(block) / 3.0)
        np.testing.assert_allclose(
            isotropic[:3, 3:], np.eye(3) * np.trace(isotropic[:3, 3:]) / 3.0
        )
        np.testing.assert_allclose(uncorrelated[:3, 3:], np.zeros((3, 3)))
        np.testing.assert_allclose(scalar, np.eye(6) * np.trace(scalar) / 6.0)
        for fitted in (full, isotropic, uncorrelated, scalar):
            self.assertGreater(float(np.min(np.linalg.eigvalsh(fitted))), 0.0)

    def test_real_data_cli_exposes_fixed_ito_sensitivity_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_projected_ito_innovations.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for required in (
            "--drift-cache-directory",
            "--covariance-cache-directory",
            "--strides",
            "--schemes",
        ):
            self.assertIn(required, completed.stdout)

    def test_additive_noise_cli_exposes_fixed_held_clone_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "analyze_ka_additive_correlated_noise_closure.py"
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
            "--maximum-exact-metric-difference",
        ):
            self.assertIn(required, completed.stdout)


if __name__ == "__main__":
    unittest.main()
