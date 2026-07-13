import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class SmoothCageTests(unittest.TestCase):
    @staticmethod
    def microscopic_configuration():
        return {
            "positions": np.array(
                [
                    [0.10, -0.20, 0.30],
                    [1.05, -0.10, 0.25],
                    [-0.65, 0.55, 0.35],
                    [0.05, -1.15, 0.60],
                ]
            ),
            "particle_types": np.array([0, 0, 1, 0]),
            "box_lengths": np.array([20.0, 20.0, 20.0]),
            "target_index": 0,
        }

    def test_wendland_weight_and_derivative_are_compact_and_smooth(self):
        from ka_smooth_cage import wendland_c4_weight

        scaled_distance = np.array([0.0, 0.5, 1.0, 1.1])
        weight, derivative = wendland_c4_weight(scaled_distance)

        self.assertAlmostEqual(weight[0], 3.0)
        self.assertGreater(weight[1], 0.0)
        np.testing.assert_array_equal(weight[2:], 0.0)
        np.testing.assert_array_equal(derivative[2:], 0.0)

    def test_smooth_cage_is_translation_invariant_and_decomposes_position(self):
        from ka_smooth_cage import smooth_force_support_cage

        inputs = self.microscopic_configuration()
        result = smooth_force_support_cage(**inputs)
        translation = np.array([2.3, -1.7, 0.8])
        translated = smooth_force_support_cage(
            **{**inputs, "positions": inputs["positions"] + translation}
        )

        np.testing.assert_allclose(
            result["relative_position"],
            inputs["positions"][inputs["target_index"]] - result["cage_position"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            translated["relative_position"], result["relative_position"], atol=1e-12
        )
        np.testing.assert_allclose(
            translated["cage_position"], result["cage_position"] + translation, atol=1e-12
        )
        np.testing.assert_allclose(
            np.sum(result["jacobian"], axis=0), np.zeros((3, 3)), atol=1e-12
        )

    def test_smooth_cage_analytic_jacobian_matches_centered_difference(self):
        from ka_smooth_cage import smooth_force_support_cage

        inputs = self.microscopic_configuration()
        result = smooth_force_support_cage(**inputs)
        step = 1e-6
        numerical = np.empty_like(result["jacobian"])
        for particle in range(len(inputs["positions"])):
            for component in range(3):
                plus_positions = inputs["positions"].copy()
                minus_positions = inputs["positions"].copy()
                plus_positions[particle, component] += step
                minus_positions[particle, component] -= step
                plus = smooth_force_support_cage(
                    **{**inputs, "positions": plus_positions}
                )["relative_position"]
                minus = smooth_force_support_cage(
                    **{**inputs, "positions": minus_positions}
                )["relative_position"]
                numerical[particle, :, component] = (plus - minus) / (2.0 * step)

        relative_l2 = np.linalg.norm(result["jacobian"] - numerical) / np.linalg.norm(
            numerical
        )
        self.assertLess(relative_l2, 1e-6)


if __name__ == "__main__":
    unittest.main()
