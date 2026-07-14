import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class StructuralPrecursorTests(unittest.TestCase):
    def test_species_resolved_radial_features_match_manual_shell_weights(self):
        from ka_structural_precursor import species_resolved_radial_features

        positions = np.array(
            [
                [0.1, 0.1, 0.1],
                [1.1, 0.1, 0.1],
                [0.1, 1.6, 0.1],
            ]
        )
        particle_types = np.array([0, 0, 1])
        box_lengths = np.array([10.0, 10.0, 10.0])
        radii = np.array([1.0, 1.5])
        width = 0.2
        cutoff = 2.5

        feature = species_resolved_radial_features(
            positions,
            particle_types,
            box_lengths,
            np.array([0]),
            radii=radii,
            width=width,
            cutoff=cutoff,
        )
        translated = species_resolved_radial_features(
            positions + np.array([9.7, -3.2, 4.4]),
            particle_types,
            box_lengths,
            np.array([0]),
            radii=radii,
            width=width,
            cutoff=cutoff,
        )

        def cutoff_weight(distance):
            return 0.5 * (np.cos(np.pi * distance / cutoff) + 1.0)

        expected = np.array(
            [
                cutoff_weight(1.0)
                * np.exp(-0.5 * ((1.0 - radii[0]) / width) ** 2),
                cutoff_weight(1.0)
                * np.exp(-0.5 * ((1.0 - radii[1]) / width) ** 2),
                cutoff_weight(1.5)
                * np.exp(-0.5 * ((1.5 - radii[0]) / width) ** 2),
                cutoff_weight(1.5)
                * np.exp(-0.5 * ((1.5 - radii[1]) / width) ** 2),
            ]
        )
        self.assertEqual(feature.shape, (1, 4))
        np.testing.assert_allclose(feature[0], expected, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(translated, feature, rtol=0.0, atol=1e-14)

    def test_species_resolved_radial_features_reject_invalid_grid(self):
        from ka_structural_precursor import species_resolved_radial_features

        with self.assertRaisesRegex(ValueError, "radii"):
            species_resolved_radial_features(
                np.zeros((2, 3)),
                np.array([0, 1]),
                np.ones(3),
                np.array([0]),
                radii=np.array([1.0, 0.9]),
                width=0.2,
                cutoff=2.5,
            )


if __name__ == "__main__":
    unittest.main()
