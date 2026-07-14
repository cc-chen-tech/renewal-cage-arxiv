import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SecondGeneratorKrylovTests(unittest.TestCase):
    @staticmethod
    def synthetic_second_generator_states():
        frame_time = 0.005
        friction = 1.0
        third_block = np.zeros((3, 15))
        for block, coefficient in enumerate((-0.2, -0.5, -1.0, -2.0, -3.0)):
            third_block[:, 3 * block : 3 * (block + 1)] = coefficient * np.eye(3)
        generator = np.zeros((15, 15))
        generator[0:3, 3:6] = np.eye(3)
        generator[3:6, 3:6] = -friction * np.eye(3)
        generator[3:6, 6:9] = np.eye(3)
        generator[6:9, 9:12] = np.eye(3)
        generator[9:12, 12:15] = np.eye(3)
        generator[12:15] = third_block
        identity = np.eye(15)
        transition = np.linalg.solve(
            identity - 0.5 * frame_time * generator,
            identity + 0.5 * frame_time * generator,
        )
        rng = np.random.default_rng(90210)
        states = np.empty((8, 201, 15))
        states[:, 0] = rng.normal(size=(8, 15))
        for frame in range(1, states.shape[1]):
            states[:, frame] = np.einsum(
                "ab,pb->pa", transition, states[:, frame - 1]
            )
        return states, third_block, frame_time, friction

    def test_assemble_second_generator_state_preserves_microscopic_order(self):
        from ka_generator_krylov import assemble_second_generator_state

        state = np.arange(2 * 7 * 12, dtype=float).reshape(2, 7, 12)
        second = np.arange(2 * 7 * 3, dtype=float).reshape(2, 7, 3)

        result = assemble_second_generator_state(state, second)

        self.assertEqual(result.shape, (2, 7, 15))
        np.testing.assert_array_equal(result[..., :12], state)
        np.testing.assert_array_equal(result[..., 12:15], second)

    def test_assemble_second_generator_state_rejects_misaligned_second_mode(self):
        from ka_generator_krylov import assemble_second_generator_state

        with self.assertRaisesRegex(ValueError, "second_force_response"):
            assemble_second_generator_state(
                np.zeros((7, 12)),
                np.zeros((6, 3)),
            )

    def test_weak_form_fit_recovers_exact_second_generator_chain(self):
        from ka_generator_krylov import fit_second_generator_constrained_response

        states, exact_block, frame_time, friction = (
            self.synthetic_second_generator_states()
        )

        fit = fit_second_generator_constrained_response(
            states,
            frame_time=frame_time,
            friction=friction,
            fit_frames=41,
        )

        np.testing.assert_allclose(
            fit["fitted_third_generator_block"],
            exact_block,
            rtol=2e-3,
            atol=2e-3,
        )
        self.assertEqual(fit["design_rank"], 15.0)
        self.assertLess(fit["heldout_position_relative_l2_error"], 2e-3)
        self.assertLessEqual(fit["spectral_radius"], 1.0 + 1e-6)

    def test_free_transition_and_propagation_recover_synthetic_chain(self):
        from ka_generator_krylov import (
            fit_free_second_generator_transition,
            propagate_linear_response,
        )

        states, _, _, _ = self.synthetic_second_generator_states()

        fit = fit_free_second_generator_transition(states, fit_frames=41)
        predicted = propagate_linear_response(
            fit["transition_matrix"],
            states[0, 0],
            states.shape[1],
        )

        self.assertEqual(fit["design_rank"], 15.0)
        self.assertEqual(predicted.shape, states[0].shape)
        np.testing.assert_array_equal(predicted[0], states[0, 0])
        np.testing.assert_allclose(predicted, states[0], rtol=1e-9, atol=1e-9)

    def test_second_generator_residual_vanishes_for_exact_weak_chain(self):
        from ka_generator_krylov import second_generator_residual_diagnostic

        states, exact_block, frame_time, _ = self.synthetic_second_generator_states()

        diagnostic = second_generator_residual_diagnostic(
            states,
            exact_block,
            frame_time=frame_time,
        )

        self.assertLess(diagnostic["residual_relative_l2"], 1e-10)
        self.assertEqual(diagnostic["maximum_abs_residual_state_correlation"], 0.0)


if __name__ == "__main__":
    unittest.main()
