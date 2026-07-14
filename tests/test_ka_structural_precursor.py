import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


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

    def test_isoconfigurational_expansion_preserves_parent_clone_target_order(self):
        from ka_structural_precursor import expand_isoconfigurational_structural_rows

        features = np.array([[[10.0], [11.0]], [[20.0], [21.0]]])
        first_passage = np.array(
            [
                [[1.0, 20.0], [2.0, 3.0], [20.0, 4.0]],
                [[5.0, 20.0], [6.0, 7.0], [8.0, 20.0]],
            ]
        )
        escaped = first_passage < 20.0

        result = expand_isoconfigurational_structural_rows(
            features, first_passage, escaped, horizon=20.0
        )

        np.testing.assert_array_equal(
            result["features"][:, 0],
            [10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0, 20.0, 21.0],
        )
        np.testing.assert_array_equal(
            result["groups"], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        )
        np.testing.assert_array_equal(
            result["successes"], escaped.sum(axis=1).reshape(-1)
        )
        np.testing.assert_array_equal(result["trials"], np.full(4, 3))
        np.testing.assert_array_equal(
            result["configuration_groups"], [0, 0, 1, 1]
        )

    def test_isoconfigurational_expansion_requires_horizon_censoring(self):
        from ka_structural_precursor import expand_isoconfigurational_structural_rows

        with self.assertRaisesRegex(ValueError, "horizon censoring"):
            expand_isoconfigurational_structural_rows(
                np.ones((2, 1, 1)),
                np.array([[[1.0], [19.0]], [[2.0], [20.0]]]),
                np.array([[[True], [False]], [[True], [False]]]),
                horizon=20.0,
            )

    def test_radial_residual_analysis_freezes_protocol_and_claim_boundaries(self):
        source = (
            ROOT / "scripts" / "analyze_ka_radial_precursor_residual.py"
        ).read_text()

        for required in (
            "RADII = np.array([0.8, 1.05, 1.3, 1.55, 1.8, 2.05, 2.3]",
            "RADIAL_WIDTH = 0.12",
            "RADIAL_CUTOFF = 2.5",
            "EXPECTED_EVENT_COUNT = 1731",
            "EXPECTED_CENSORED_COUNT = 829",
            '"geometry_radial"',
            "grouped_exponential_escape_diagnostic",
            "grouped_binomial_logistic_committor_diagnostic",
            "geometry_reproduction_gate_pass",
            "clone_invariance_gate_pass",
            '"event_clock_claim_allowed": False',
            '"autonomous_single_particle_gle_claim_allowed": False',
            '"kramers_escape_claim_allowed": False',
            '"thermodynamic_claim_allowed": False',
        ):
            self.assertIn(required, source)

    def test_radial_precursor_gate_requires_uniform_held_parent_transfer(self):
        from analyze_ka_radial_precursor_residual import (
            evaluate_radial_precursor_gates,
        )

        metrics = {
            "integrity_gate_pass": True,
            "geometry_reproduction_gate_pass": True,
            "clone_invariance_gate_pass": True,
            "geometry_mean_heldout_brier_skill": 0.008,
            "radial_mean_heldout_brier_skill": 0.015,
            "geometry_radial_mean_heldout_brier_skill": 0.04,
            "geometry_radial_mean_heldout_log_likelihood_gain_per_observation": 0.01,
            "geometry_radial_minimum_group_log_likelihood_gain": 0.1,
            "geometry_radial_maximum_heldout_survival_calibration_error": 0.09,
            "geometry_binomial_mean_heldout_brier_skill": 0.01,
            "geometry_radial_binomial_mean_heldout_brier_skill": 0.03,
        }

        passing = evaluate_radial_precursor_gates(metrics)
        self.assertTrue(passing["static_radial_precursor_allowed"])
        failing = evaluate_radial_precursor_gates(
            {
                **metrics,
                "geometry_radial_minimum_group_log_likelihood_gain": -1e-6,
            }
        )
        self.assertFalse(failing["likelihood_gate_pass"])
        self.assertFalse(failing["static_radial_precursor_allowed"])

    def test_local_soft_mode_features_match_exact_cluster_hessian(self):
        from ka_local_cage import ka_local_cluster_hessian
        from ka_structural_precursor import instantaneous_local_soft_mode_features

        positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        particle_types = np.array([0, 0])
        box_lengths = np.array([20.0, 20.0, 20.0])
        floor = 1e-8
        cluster = ka_local_cluster_hessian(
            positions,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_index=0,
            cluster_cutoff=2.0,
        )
        eigenvalue, eigenvector = np.linalg.eigh(cluster["hessian"])
        positive = np.flatnonzero(eigenvalue > floor)
        mode = int(positive[0])
        local_target = int(np.flatnonzero(cluster["cluster_indices"] == 0)[0])
        target_weight = float(
            np.sum(
                eigenvector[
                    3 * local_target : 3 * local_target + 3, mode
                ]
                ** 2
            )
        )
        expected = np.array(
            [
                np.log(1.0 / eigenvalue[mode]),
                np.log(target_weight / eigenvalue[mode]),
                np.log(len(cluster["cluster_indices"])),
                np.sum(eigenvalue <= floor),
            ]
        )

        feature, names = instantaneous_local_soft_mode_features(
            positions,
            particle_types,
            box_lengths,
            np.array([0]),
            cluster_cutoff=2.0,
            ranks=(0,),
            eigenvalue_floor=floor,
        )

        self.assertEqual(
            names,
            (
                "log_inverse_eigenvalue_rank_0",
                "log_target_weighted_softness_rank_0",
                "log_cluster_particle_count",
                "negative_or_zero_mode_count",
            ),
        )
        np.testing.assert_allclose(feature[0], expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
