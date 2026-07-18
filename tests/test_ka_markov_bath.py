import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_ka_relative_second_generator_closure import (  # noqa: E402
    aggregate_summary,
    white_residual_shape_diagnostic,
)

from ka_markov_bath import (  # noqa: E402
    fit_stationary_vector_autoregression,
    simulate_mori_with_var_bath,
    vector_autoregressive_residual,
)


class MarkovBathTests(unittest.TestCase):
    def test_second_generator_verdict_separates_lp_improvement_from_full_closure(self):
        common = {
            "record": "model_fold",
            "protocol": "synthetic",
            "fold_index": 1.0,
            "memory_order": 2.0,
            "bath_order": 1.0,
            "potential_protocol": "ka_lj_cut",
            "trace_probe_count": 4.0,
            "var_spectral_radius": 0.8,
            "minimum_white_noise_covariance_eigenvalue": 0.1,
            "maximum_held_white_residual_correlation": 0.02,
            "stationary_covariance_maximum_error": 0.1,
            "target_correlation_maximum_error": 0.1,
            "maximum_simulated_absolute_state": 4.0,
            "white_residual_gate_pass": 0.0,
            "gaussian_closure_gate_pass": 0.0,
        }
        baseline = {
            **common,
            "model": "relative_phase_generator",
            "maximum_held_squared_white_residual_correlation": 0.30,
            "maximum_absolute_held_white_residual_excess_kurtosis": 3.0,
            "lp_held_white_residual_correlation": 0.02,
            "lp_held_squared_white_residual_correlation": 0.30,
            "lp_held_white_residual_excess_kurtosis": 3.0,
        }
        extension = {
            **common,
            "model": "relative_phase_generator_l2p",
            "maximum_held_squared_white_residual_correlation": 0.40,
            "maximum_absolute_held_white_residual_excess_kurtosis": 5.0,
            "lp_held_white_residual_correlation": 0.01,
            "lp_held_squared_white_residual_correlation": 0.18,
            "lp_held_white_residual_excess_kurtosis": 1.5,
        }

        verdict = aggregate_summary([baseline, extension])[-1]

        self.assertEqual(verdict["l2p_improves_lp_shape_on_aggregate"], 1.0)
        self.assertEqual(verdict["finite_discrete_gaussian_l2p_closure_supported"], 0.0)

    def test_white_residual_shape_diagnostic_recognizes_gaussian_iid_driving(self):
        rng = np.random.default_rng(20260718)
        residual = rng.normal(size=(800, 400, 2))

        result = white_residual_shape_diagnostic(residual, maximum_lag=20)

        self.assertLess(
            float(np.max(result["maximum_absolute_correlation"])), 0.01
        )
        self.assertLess(
            float(np.max(result["maximum_absolute_squared_correlation"])), 0.01
        )
        self.assertLess(float(np.max(np.abs(result["excess_kurtosis"]))), 0.03)

    def test_white_residual_shape_diagnostic_detects_hidden_scale_memory(self):
        rng = np.random.default_rng(20260719)
        log_scale = np.zeros((700, 600, 1))
        for frame in range(1, len(log_scale)):
            log_scale[frame] = (
                0.94 * log_scale[frame - 1]
                + rng.normal(scale=0.18, size=log_scale.shape[1:])
            )
        residual = np.exp(0.65 * log_scale) * rng.normal(size=log_scale.shape)

        result = white_residual_shape_diagnostic(residual, maximum_lag=20)

        self.assertLess(float(result["maximum_absolute_correlation"][0]), 0.02)
        self.assertGreater(
            float(result["maximum_absolute_squared_correlation"][0]), 0.10
        )
        self.assertGreater(float(result["excess_kurtosis"][0]), 0.20)

    def test_relative_second_generator_closure_cli_fixes_mori_and_var_orders(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "analyze_ka_relative_second_generator_closure.py"
                ),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for required in (
            "--training-drift-cache-directory",
            "--training-second-generator-cache-directories",
            "--validation-drift-cache-directory",
            "--validation-second-generator-cache-directories",
            "--memory-order",
            "--bath-order",
            "--maximum-frame-count",
            "--output-prefix",
        ):
            self.assertIn(required, completed.stdout)

    def test_relative_second_generator_closure_cli_runs_matched_synthetic_folds(self):
        rng = np.random.default_rng(20260720)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            drift_directory = root / "drift"
            second_directory = root / "second"
            drift_directory.mkdir()
            second_directory.mkdir()
            targets = np.array([3, 7])
            for clone_index in range(1, 4):
                state = np.zeros((90, len(targets), 3, 4))
                for frame in range(1, len(state)):
                    state[frame] = 0.55 * state[frame - 1] + rng.normal(
                        scale=[0.4, 0.5, 0.6, 0.7], size=state.shape[1:]
                    )
                source_hash = f"synthetic-{clone_index}"
                shape = state.shape[:3]
                np.savez(
                    drift_directory / f"clone_{clone_index:03d}_decomposed_drift.npz",
                    relative_position=state[..., 0],
                    relative_velocity=state[..., 1],
                    center_velocity=np.zeros(shape),
                    relative_drift=state[..., 2],
                    center_drift=np.zeros(shape),
                    target_indices=targets,
                    trajectory_sha256=np.asarray(source_hash),
                    thermodynamic_claim_allowed=0.0,
                )
                np.savez(
                    second_directory
                    / f"clone_{clone_index:03d}_relative_second_generator.npz",
                    second_relative_generator=state[..., 3],
                    target_indices=targets,
                    trajectory_sha256=np.asarray(source_hash),
                    completed_frame_count=float(len(state)),
                    requested_frame_count=float(len(state)),
                    potential_protocol=np.asarray("ka_lj_cut"),
                    trace_probe_count=4.0,
                    thermodynamic_claim_allowed=0.0,
                )
            output_prefix = root / "synthetic_closure"

            subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts"
                        / "analyze_ka_relative_second_generator_closure.py"
                    ),
                    "--training-drift-cache-directory",
                    str(drift_directory),
                    "--training-second-generator-cache-directories",
                    str(second_directory),
                    "--memory-order",
                    "2",
                    "--bath-order",
                    "1",
                    "--maximum-lag",
                    "5",
                    "--simulation-count",
                    "200",
                    "--output-prefix",
                    str(output_prefix),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            for suffix in ("_details.csv", "_summary.csv", "_correlation.csv"):
                self.assertTrue(output_prefix.with_name(output_prefix.name + suffix).is_file())

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
