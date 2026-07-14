import sys
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ka_relative_mori import (  # noqa: E402
    bias_centered_phase_state,
    discrete_mori_gfd_diagnostic,
    discrete_mori_noise,
    finite_memory_innovation_series,
    parity_detailed_balance_diagnostic,
    propagate_discrete_mori_correlation,
)
from ka_collective_memory import discrete_mori_zwanzig_operators  # noqa: E402


class RelativeMoriTests(unittest.TestCase):
    def test_bias_centered_phase_state_flattens_particle_components(self):
        position = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
        velocity = 100.0 + position
        bias = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        state = bias_centered_phase_state(position, velocity, bias=bias)

        self.assertEqual(state.shape, (2, 6, 2))
        np.testing.assert_allclose(state[:, :, 0], (position - bias).reshape(2, 6))
        np.testing.assert_allclose(state[:, :, 1], velocity.reshape(2, 6))

    def test_discrete_mori_noise_recovers_known_finite_memory_residual(self):
        operators = np.array([[[0.7]], [[-0.2]]])
        innovation = np.array([0.3, -0.1, 0.4, 0.2, -0.5, 0.1])
        state = np.empty((len(innovation) + 2, 1, 1))
        state[:2, 0, 0] = [1.0, 0.5]
        for frame in range(1, len(state) - 1):
            memory = operators[0, 0, 0] * state[frame, 0, 0]
            if frame >= 1:
                memory += operators[1, 0, 0] * state[frame - 1, 0, 0]
            state[frame + 1, 0, 0] = memory + innovation[frame - 1]

        noise = discrete_mori_noise(state, operators)

        expected_w0 = state[1:, 0, 0] - 0.7 * state[:-1, 0, 0]
        expected_w1 = (
            state[2:, 0, 0]
            - 0.7 * state[1:-1, 0, 0]
            + 0.2 * state[:-2, 0, 0]
        )
        np.testing.assert_allclose(noise[0, :, 0, 0], expected_w0[: len(noise[0])])
        np.testing.assert_allclose(noise[1, :, 0, 0], expected_w1[: len(noise[1])])

    def test_finite_memory_innovation_series_recovers_ar2_driving(self):
        operators = np.array([[[0.7]], [[-0.2]]])
        driving = np.array([0.3, -0.1, 0.4, 0.2, -0.5, 0.1])
        state = np.empty((len(driving) + 2, 1, 1))
        state[:2, 0, 0] = [1.0, 0.5]
        for frame, value in enumerate(driving, start=1):
            state[frame + 1, 0, 0] = (
                0.7 * state[frame, 0, 0]
                - 0.2 * state[frame - 1, 0, 0]
                + value
            )

        innovation = finite_memory_innovation_series(state, operators)

        np.testing.assert_allclose(innovation[:, 0, 0], driving)

    def test_correlation_propagation_uses_finite_memory_tail(self):
        operators = np.array([[[0.8]], [[-0.15]]])
        initial = np.array([[[1.0]], [[0.8]], [[0.49]]])

        propagated = propagate_discrete_mori_correlation(
            operators,
            initial_correlation=initial,
            output_count=6,
        )

        self.assertEqual(propagated.shape, (6, 1, 1))
        self.assertAlmostEqual(float(propagated[3, 0, 0]), 0.8 * 0.49 - 0.15 * 0.8)
        self.assertAlmostEqual(
            float(propagated[4, 0, 0]),
            0.8 * propagated[3, 0, 0] - 0.15 * 0.49,
        )

    def test_discrete_gfd_diagnostic_passes_independent_stationary_ar2(self):
        rng = np.random.default_rng(20260714)

        def paths() -> np.ndarray:
            state = np.zeros((1800, 500, 1))
            state[:2, :, 0] = rng.normal(size=(2, state.shape[1]))
            for frame in range(1, len(state) - 1):
                state[frame + 1, :, 0] = (
                    0.65 * state[frame, :, 0]
                    - 0.22 * state[frame - 1, :, 0]
                    + rng.normal(scale=0.7, size=state.shape[1])
                )
            return state[200:]

        training = paths()
        held = paths()
        operators = discrete_mori_zwanzig_operators(
            training, memory_order=2
        )["operators"]

        result = discrete_mori_gfd_diagnostic(held, operators)

        self.assertLess(float(result["maximum_noise_initial_state_correlation"]), 0.02)
        self.assertLess(float(result["gfd_operator_normalized_rmse"]), 0.04)
        self.assertGreater(float(result["gfd_operator_shape_correlation"]), 0.99)

    def test_parity_detailed_balance_recovers_reversible_oscillator(self):
        rng = np.random.default_rng(20260715)
        samples = 400
        time = np.arange(120)[:, None]
        phase = 0.08 * time
        cosine = np.cos(phase)
        sine = np.sin(phase)
        amplitude = rng.normal(size=(1, samples))
        quadrature = rng.normal(size=(1, samples))
        position = amplitude * cosine + quadrature * sine
        velocity = -amplitude * sine + quadrature * cosine
        static_even = np.broadcast_to(rng.normal(size=(1, samples)), position.shape)
        state = np.stack([position, velocity, static_even], axis=2)

        correct = parity_detailed_balance_diagnostic(
            state, parity=np.array([1, -1, 1]), maximum_lag=40
        )
        wrong = parity_detailed_balance_diagnostic(
            state, parity=np.ones(3), maximum_lag=40
        )

        self.assertLess(float(correct["parity_defect_normalized_rmse"]), 0.03)
        self.assertGreater(float(wrong["parity_defect_normalized_rmse"]), 0.5)

    def test_invalid_bias_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bias"):
            bias_centered_phase_state(
                np.zeros((3, 2, 3)),
                np.zeros((3, 2, 3)),
                bias=np.zeros((3, 3)),
            )

    def test_real_data_cli_separates_discovery_and_validation_inputs(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_relative_generator_mori.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for required in (
            "--training-drift-cache-directory",
            "--validation-drift-cache-directory",
            "--memory-orders",
            "--fixed-memory-order",
        ):
            self.assertIn(required, completed.stdout)

    def test_cache_only_cli_exposes_microscopic_drift_inputs(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "cache_ka_decomposed_drift.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for required in (
            "--reduced-cache-directory",
            "--cage-cache-directory",
            "--drift-cache-directory",
            "--lammps-binary",
            "--parent-restart",
        ):
            self.assertIn(required, completed.stdout)

    def test_parity_audit_cli_exposes_separate_training_and_validation(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_relative_generator_parity.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--training-drift-cache-directory", completed.stdout)
        self.assertIn("--validation-drift-cache-directory", completed.stdout)

    def test_noise_closure_cli_exposes_block_discovery_and_fixed_validation(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "analyze_ka_relative_generator_noise_closure.py"
                ),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for required in (
            "--innovation-block-lengths",
            "--fixed-innovation-block-length",
            "--validation-drift-cache-directory",
        ):
            self.assertIn(required, completed.stdout)


if __name__ == "__main__":
    unittest.main()
