import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class L2pConditionalDiffusionTests(unittest.TestCase):
    def test_rademacher_probes_are_seeded_nested_and_sign_valued(self):
        from ka_l2p_conditional_diffusion import rademacher_velocity_probes

        first = rademacher_velocity_probes(
            probe_count=32,
            particle_count=5,
            seed=20260719,
        )
        repeated = rademacher_velocity_probes(
            probe_count=32,
            particle_count=5,
            seed=20260719,
        )
        shorter = rademacher_velocity_probes(
            probe_count=16,
            particle_count=5,
            seed=20260719,
        )

        np.testing.assert_array_equal(first, repeated)
        np.testing.assert_array_equal(first[:16], shorter)
        self.assertEqual(first.shape, (32, 5, 3))
        self.assertEqual(set(np.unique(first)), {-1.0, 1.0})

    def test_nested_diffusion_estimates_match_direct_probe_outer_products(self):
        from ka_l2p_conditional_diffusion import nested_diffusion_estimates

        rng = np.random.default_rng(91)
        responses = rng.normal(size=(8, 3, 3))
        result = nested_diffusion_estimates(
            responses,
            prefix_counts=(2, 4, 8),
            friction=0.7,
            temperature=0.58,
        )

        expected = []
        for prefix in (2, 4, 8):
            expected.append(
                2.0
                * 0.7
                * 0.58
                * np.einsum(
                    "pti,ptj->tij", responses[:prefix], responses[:prefix]
                )
                / prefix
            )
        np.testing.assert_allclose(result["diffusion_prefixes"], expected)
        self.assertEqual(result["primary_probe_count"], 8.0)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(result["diffusion_prefixes"]))),
            -1e-12,
        )
        self.assertEqual(result["thermodynamic_claim_allowed"], 0.0)

    def test_probe_estimator_converges_to_known_linear_velocity_diffusion(self):
        from ka_l2p_conditional_diffusion import (
            nested_diffusion_estimates,
            rademacher_velocity_probes,
        )

        probes = rademacher_velocity_probes(
            probe_count=32768,
            particle_count=4,
            seed=77,
        )
        matrix = np.random.default_rng(12).normal(size=(3, probes.shape[1] * 3))
        responses = np.einsum("ij,pj->pi", matrix, probes.reshape(len(probes), -1))
        result = nested_diffusion_estimates(
            responses[:, None, :],
            prefix_counts=(128, 2048, 32768),
            friction=0.9,
            temperature=0.4,
        )
        expected = 2.0 * 0.9 * 0.4 * matrix @ matrix.T
        errors = [
            np.linalg.norm(value[0] - expected) / np.linalg.norm(expected)
            for value in result["diffusion_prefixes"]
        ]

        self.assertLess(errors[-1], 0.02)
        self.assertLess(errors[-1], errors[0])

    def test_diffusion_convergence_reports_distribution_and_rejects_bad_inputs(self):
        from ka_l2p_conditional_diffusion import (
            diffusion_convergence_summary,
            nested_diffusion_estimates,
            rademacher_velocity_probes,
        )

        reference = np.broadcast_to(np.eye(3), (4, 5, 3, 3)).copy()
        candidate = 0.9 * reference
        summary = diffusion_convergence_summary(candidate, reference)
        self.assertAlmostEqual(summary["median_relative_frobenius_error"], 0.1)
        self.assertAlmostEqual(summary["p95_relative_frobenius_error"], 0.1)
        with self.assertRaisesRegex(ValueError, "prefix"):
            nested_diffusion_estimates(
                np.ones((4, 2, 3)),
                prefix_counts=(2, 5),
                friction=1.0,
                temperature=0.58,
            )
        with self.assertRaisesRegex(ValueError, "probe_count"):
            rademacher_velocity_probes(
                probe_count=0,
                particle_count=2,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
