import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


class L3pQuotientTests(unittest.TestCase):
    @staticmethod
    def synthetic_clone(seed):
        rng = np.random.default_rng(seed)
        shape = (90, 4, 3)
        return {
            "relative_position": rng.normal(size=shape),
            "relative_velocity": rng.normal(size=shape),
            "relative_drift": rng.normal(size=shape),
            "second_relative_generator": rng.normal(size=shape),
            "l3p_generator": rng.normal(size=shape),
            "target_indices": np.arange(4, dtype=int),
            "trajectory_sha256": f"hash-{seed}",
            "potential_protocol": "ka_lj_cut",
            "thermodynamic_claim_allowed": 0.0,
        }

    @staticmethod
    def predictive_generator_clone(seed):
        rng = np.random.default_rng(seed)
        times, targets = 320, 8
        shape = (times, targets, 3)

        def ar_path(coefficient):
            values = np.zeros(shape)
            noise = rng.normal(size=shape)
            for time in range(1, times):
                values[time] = coefficient * values[time - 1] + noise[time]
            return values

        l3p = rng.normal(size=shape)
        l2p = np.zeros(shape)
        noise = 0.20 * rng.normal(size=shape)
        for time in range(times - 1):
            l2p[time + 1] = 0.20 * l2p[time] + 1.50 * l3p[time] + noise[time + 1]
        return {
            "relative_position": ar_path(0.75),
            "relative_velocity": ar_path(0.55),
            "relative_drift": ar_path(0.35),
            "second_relative_generator": l2p,
            "l3p_generator": l3p,
            "target_indices": np.arange(targets, dtype=int),
            "trajectory_sha256": f"predictive-{seed}",
            "potential_protocol": "ka_lj_cut",
            "thermodynamic_claim_allowed": 0.0,
        }

    def test_fifth_coordinate_models_are_causal_and_target_paired(self):
        from ka_l3p_quotient import quotient_fifth_coordinate

        times = 8
        targets = 2
        l2p = np.arange(times * targets * 3, dtype=float).reshape(times, targets, 3)
        l3p = 100.0 * np.arange(times)[:, None, None] + np.array(
            [[[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]]]
        )

        real = quotient_fifth_coordinate(
            l2p,
            l3p,
            model="l3p_generator",
            frame_time=0.01,
            seed=20260802,
        )
        np.testing.assert_array_equal(real, l3p[1:])

        backward = quotient_fifth_coordinate(
            l2p,
            l3p,
            model="l2p_backward_difference",
            frame_time=0.5,
            seed=20260802,
        )
        np.testing.assert_array_equal(backward, np.diff(l2p, axis=0) / 0.5)

        permuted = quotient_fifth_coordinate(
            l2p,
            l3p,
            model="l3p_time_permuted",
            frame_time=0.01,
            seed=20260802,
        )
        self.assertFalse(np.array_equal(permuted, real))
        for target in range(targets):
            np.testing.assert_array_equal(permuted[:, target, 1] - permuted[:, target, 0], 1.0)
            np.testing.assert_array_equal(permuted[:, target, 2] - permuted[:, target, 0], 2.0)
            self.assertEqual(
                sorted(permuted[:, target, 0].tolist()),
                sorted(real[:, target, 0].tolist()),
            )

        baseline = quotient_fifth_coordinate(
            l2p,
            l3p,
            model="l2p_exact_q_baseline",
            frame_time=0.01,
            seed=20260802,
        )
        self.assertIsNone(baseline)

    def test_l3p_verdict_requires_paired_improvements_and_closes_broad_claims(self):
        from ka_l3p_quotient import L3P_QUOTIENT_MODELS, classify_l3p_quotient

        rows = []
        for fold in range(1, 5):
            metrics = {
                "l2p_exact_q_baseline": (10.0, 0.20, 0.80),
                "l3p_generator": (8.0, 0.04, 0.30),
                "l3p_time_permuted": (9.0, 0.18, 0.70),
                "l2p_backward_difference": (9.2, 0.19, 0.72),
            }
            for model in L3P_QUOTIENT_MODELS:
                nll, squared_memory, kurtosis = metrics[model]
                rows.append(
                    {
                        "fold_index": float(fold),
                        "model": model,
                        "negative_log_likelihood": nll,
                        "maximum_absolute_squared_whitened_correlation": squared_memory,
                        "maximum_absolute_component_excess_kurtosis": kurtosis,
                        "maximum_absolute_whitened_correlation": 0.04,
                        "maximum_absolute_whitened_covariance_error": 0.08,
                        "mean_squared_mahalanobis_per_dimension": 1.0,
                        "isotropic_floor_variance_fraction": 0.10,
                    }
                )
        numerical = [
            {"fold_index": float(fold), "l3p_numerical_gate_pass": 1.0}
            for fold in range(1, 5)
        ]
        verdict = classify_l3p_quotient(rows, numerical)
        self.assertEqual(verdict["l3p_generator_coordinate_informative"], 1.0)
        self.assertEqual(verdict["l2p_residual_closed_by_l3p"], 1.0)
        self.assertEqual(verdict["l3p_diffusion_derivation_authorized"], 1.0)
        self.assertEqual(verdict["finite_l3p_gaussian_closure_supported"], 0.0)
        for claim in (
            "microscopic_environment_coordinate_z_allowed",
            "continuous_gaussian_langevin_bath_allowed",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "spatial_facilitation_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(verdict[claim], 0.0)

        failed_null = [dict(row) for row in rows]
        for row in failed_null:
            if row["model"] == "l3p_time_permuted":
                row["negative_log_likelihood"] = 7.9
        unresolved = classify_l3p_quotient(failed_null, numerical)
        self.assertEqual(unresolved["l3p_generator_coordinate_informative"], 0.0)
        self.assertEqual(unresolved["l3p_diffusion_derivation_authorized"], 0.0)

        failed_numerical = [dict(row) for row in numerical]
        failed_numerical[2]["l3p_numerical_gate_pass"] = 0.0
        unresolved = classify_l3p_quotient(rows, failed_numerical)
        self.assertEqual(unresolved["l3p_generator_coordinate_informative"], 0.0)

    def test_fold_extraction_shares_base_normalization_and_exact_frame_alignment(self):
        from ka_l3p_quotient import (
            augment_l3p_quotient_clone,
            extract_l3p_quotient_fold,
        )

        clones = [self.synthetic_clone(seed) for seed in range(4)]
        baseline = [
            augment_l3p_quotient_clone(
                clone,
                model="l2p_exact_q_baseline",
                frame_time=0.01,
                permutation_seed=20260802 + index,
            )
            for index, clone in enumerate(clones)
        ]
        generator = [
            augment_l3p_quotient_clone(
                clone,
                model="l3p_generator",
                frame_time=0.01,
                permutation_seed=20260802 + index,
            )
            for index, clone in enumerate(clones)
        ]
        baseline_fold = extract_l3p_quotient_fold(
            baseline[:3],
            baseline[3],
            memory_order=3,
            bath_order=2,
            ridge_regularization=1e-8,
            var_ridge_regularization=1e-6,
        )
        generator_fold = extract_l3p_quotient_fold(
            generator[:3],
            generator[3],
            memory_order=3,
            bath_order=2,
            ridge_regularization=1e-8,
            var_ridge_regularization=1e-6,
        )
        np.testing.assert_allclose(
            baseline_fold["normalization_mean"],
            generator_fold["normalization_mean"][..., :4],
        )
        np.testing.assert_allclose(
            baseline_fold["normalization_scale"],
            generator_fold["normalization_scale"][..., :4],
        )
        self.assertEqual(baseline_fold["held_state"].shape[-1], 4)
        self.assertEqual(generator_fold["held_state"].shape[-1], 5)
        frames = np.asarray(generator_fold["held_source_frame_indices"])
        self.assertTrue(np.all(np.diff(frames) == 1))
        self.assertGreaterEqual(frames[0], 1)
        self.assertLess(frames[-1], 90)
        residual = np.asarray(generator_fold["held_l2p_vector_innovation"])
        self.assertEqual(residual.shape, (len(frames), 4, 3))

    def test_analysis_cli_and_loader_require_frozen_resolved_l3p_caches(self):
        from analyze_ka_l3p_generator_quotient import load_l3p_caches

        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_l3p_generator_quotient.py"),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for phrase in (
            "--memory-order",
            "40",
            "--bath-order",
            "16",
            "--time-permutation-seed",
            "20260802",
            "--maximum-lag",
            "--l3p-cache-directory",
            "--conditional-diffusion-cache-directory",
        ):
            self.assertIn(phrase, completed.stdout)

        clone = self.synthetic_clone(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clone_001_l3p_generator.npz"
            payload = {
                "trajectory_sha256": np.asarray(clone["trajectory_sha256"]),
                "target_indices": clone["target_indices"],
                "potential_protocol": np.asarray("ka_lj_cut"),
                "estimator": np.asarray("microscopic_l3p_generator_quotient"),
                "l3p": clone["l3p_generator"],
                "completed_frame_count": 90.0,
                "requested_frame_count": 90.0,
                "l3p_numerical_gate_pass": 1.0,
                "numerical_state": np.asarray("l3p_generator_numerically_resolved"),
                "numerical_classifier_revision": np.asarray(
                    "sqrt_epsilon_monotonic_equivalence_v2"
                ),
                "prefix_16_32_error": np.zeros((1, 4)),
                "position_primary_reference_error": np.zeros((1, 4)),
                "position_coarse_reference_error": np.zeros((1, 4)),
                "cage_primary_reference_error": np.zeros((1, 4)),
                "cage_coarse_reference_error": np.zeros((1, 4)),
                "acceleration_directional_error": np.zeros((1, 4)),
                "finite_l3p_gaussian_closure_supported": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
            np.savez_compressed(path, **payload)
            loaded = load_l3p_caches(Path(directory), [clone])
            np.testing.assert_array_equal(
                loaded[0]["l3p_generator"],
                clone["l3p_generator"],
            )

            payload["l3p_numerical_gate_pass"] = 0.0
            payload["numerical_state"] = np.asarray(
                "l3p_generator_numerically_unresolved"
            )
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, "numerical gate"):
                load_l3p_caches(Path(directory), [clone])

    def test_real_generator_coordinate_beats_time_and_history_nulls_end_to_end(self):
        from ka_l3p_quotient import (
            L3P_QUOTIENT_MODELS,
            augment_l3p_quotient_clone,
            extract_l3p_quotient_fold,
        )

        clones = [self.predictive_generator_clone(seed) for seed in range(4)]
        held_variance = {}
        for model in L3P_QUOTIENT_MODELS:
            augmented = [
                augment_l3p_quotient_clone(
                    clone,
                    model=model,
                    frame_time=0.01,
                    permutation_seed=20260802 + index,
                )
                for index, clone in enumerate(clones)
            ]
            fold = extract_l3p_quotient_fold(
                augmented[:3],
                augmented[3],
                memory_order=1,
                bath_order=1,
                ridge_regularization=1e-8,
                var_ridge_regularization=1e-6,
            )
            residual = np.asarray(fold["held_l2p_vector_innovation"])
            held_variance[model] = float(np.mean(residual**2))

        real = held_variance["l3p_generator"]
        self.assertLess(real, 0.25 * held_variance["l2p_exact_q_baseline"])
        self.assertLess(real, 0.25 * held_variance["l3p_time_permuted"])
        self.assertLess(real, 0.25 * held_variance["l2p_backward_difference"])


if __name__ == "__main__":
    unittest.main()
