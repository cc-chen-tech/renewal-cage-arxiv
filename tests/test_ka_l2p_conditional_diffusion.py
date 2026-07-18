import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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

    def test_deterministic_conditional_diffusion_is_exact_psd_and_probe_free(self):
        from ka_l2p_conditional_diffusion import deterministic_conditional_diffusion

        jacobian = np.random.default_rng(20260721).normal(size=(5, 3, 4, 3))
        result = deterministic_conditional_diffusion(
            jacobian,
            friction=0.7,
            temperature=0.58,
        )
        expected = 2.0 * 0.7 * 0.58 * np.einsum(
            "tanb,tcnb->tac", jacobian, jacobian
        )

        np.testing.assert_allclose(
            result["conditional_diffusion"], expected, rtol=2e-15, atol=2e-15
        )
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(result["conditional_diffusion"]))),
            -1e-12,
        )
        self.assertNotIn("probe_count", result)
        self.assertNotIn("primary_probe_count", result)
        self.assertEqual(result["estimator"], "deterministic_velocity_jacobian")
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

    def test_cache_cli_exposes_frozen_probe_and_checkpoint_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "cache_ka_l2p_conditional_diffusion.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for option in (
            "--clone-directory",
            "--drift-cache-directory",
            "--second-generator-cache-directory",
            "--output-cache-directory",
            "--expected-clone-count",
            "--probe-prefix-counts",
            "--velocity-step",
            "--sensitivity-velocity-steps",
            "--probe-seed",
            "--maximum-frame-count",
            "--checkpoint-interval",
        ):
            self.assertIn(option, completed.stdout)

    def test_deterministic_cache_cli_exposes_steps_provenance_and_checkpoint_controls(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "cache_ka_l2p_deterministic_diffusion.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for option in (
            "--clone-directory",
            "--drift-cache-directory",
            "--second-generator-cache-directory",
            "--output-cache-directory",
            "--expected-clone-count",
            "--primary-jacobian-step",
            "--reference-jacobian-step",
            "--coarse-jacobian-step",
            "--sensitivity-frame-count",
            "--maximum-frame-count",
            "--checkpoint-interval",
        ):
            self.assertIn(option, completed.stdout)
        self.assertNotIn("--probe", completed.stdout)

    def test_deterministic_numerical_canary_is_mechanical_and_fail_closed(self):
        from ka_l2p_conditional_diffusion import classify_deterministic_numerical_canary

        pass_result = classify_deterministic_numerical_canary(
            a_primary_reference_error=np.array([0.001, 0.002, 0.003, 0.004]),
            q_primary_reference_error=np.array([0.002, 0.003, 0.004, 0.005]),
            directional_response_error=np.array([0.003, 0.004, 0.005, 0.006]),
            a_coarse_reference_error=np.array([0.01, 0.02, 0.03, 0.04]),
            q_coarse_reference_error=np.array([0.02, 0.03, 0.04, 0.05]),
            q_minimum_eigenvalue=np.array([0.1, 0.2, 0.3, 0.4]),
            q_trace=np.array([1.0, 2.0, 3.0, 4.0]),
        )
        self.assertEqual(pass_result["deterministic_numerical_gate_pass"], 1.0)
        self.assertEqual(pass_result["a_step_gate_pass"], 1.0)
        self.assertEqual(pass_result["q_step_gate_pass"], 1.0)
        self.assertEqual(pass_result["directional_identity_gate_pass"], 1.0)
        self.assertEqual(pass_result["step_monotonicity_gate_pass"], 1.0)
        self.assertEqual(pass_result["positive_semidefinite_gate_pass"], 1.0)

        failed = classify_deterministic_numerical_canary(
            a_primary_reference_error=np.array([0.001, 0.002, 0.003, 0.004]),
            q_primary_reference_error=np.array([0.002, 0.003, 0.004, 0.005]),
            directional_response_error=np.array([0.01, 0.02, 0.20, 0.30]),
            a_coarse_reference_error=np.array([0.01, 0.02, 0.03, 0.04]),
            q_coarse_reference_error=np.array([0.02, 0.03, 0.04, 0.05]),
            q_minimum_eigenvalue=np.array([0.1, 0.2, 0.3, 0.4]),
            q_trace=np.array([1.0, 2.0, 3.0, 4.0]),
        )
        self.assertEqual(failed["deterministic_numerical_gate_pass"], 0.0)
        self.assertEqual(failed["directional_identity_gate_pass"], 0.0)
        for flag in (
            "microscopic_environment_coordinate_z_allowed",
            "continuous_gaussian_langevin_bath_allowed",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(pass_result[flag], 0.0)
            self.assertEqual(failed[flag], 0.0)

    def test_cache_alignment_rejects_hash_target_and_protocol_mismatch(self):
        module = load_script(
            "cache_ka_l2p_conditional_diffusion.py",
            "cache_ka_l2p_conditional_diffusion_test",
        )
        drift = {
            "trajectory_sha256": "trajectory-a",
            "target_indices": np.array([1, 3]),
        }
        second = {
            "trajectory_sha256": "trajectory-a",
            "target_indices": np.array([1, 3]),
            "potential_protocol": "ka_lj_cut",
            "thermodynamic_claim_allowed": 0.0,
        }
        module.validate_cache_alignment(
            trajectory_sha256="trajectory-a",
            drift=drift,
            second=second,
            target_count=2,
        )
        for update, message in (
            ({"trajectory_sha256": "trajectory-b"}, "SHA256"),
            ({"target_indices": np.array([1, 4])}, "target"),
            ({"potential_protocol": "unknown"}, "protocol"),
            ({"thermodynamic_claim_allowed": 1.0}, "claim"),
        ):
            with self.subTest(update=update):
                with self.assertRaisesRegex(ValueError, message):
                    module.validate_cache_alignment(
                        trajectory_sha256="trajectory-a",
                        drift=drift,
                        second={**second, **update},
                        target_count=2,
                    )

    def test_scaled_conditional_covariance_fit_recovers_synthetic_parameters(self):
        from ka_l2p_conditional_diffusion import (
            fit_scaled_conditional_covariance,
        )

        rng = np.random.default_rng(20260720)
        raw = rng.normal(size=(300, 8, 3, 3))
        q = np.einsum("...ik,...jk->...ij", raw, raw) / 3.0 + 0.2 * np.eye(3)
        expected_scale = 0.65
        expected_floor = 0.18
        covariance = expected_scale * q + expected_floor * np.eye(3)
        residual = np.einsum(
            "...ij,...j->...i",
            np.linalg.cholesky(covariance),
            rng.normal(size=(300, 8, 3)),
        )

        fitted = fit_scaled_conditional_covariance(residual, q)

        self.assertAlmostEqual(fitted["scale"], expected_scale, delta=0.08)
        self.assertAlmostEqual(fitted["isotropic_floor"], expected_floor, delta=0.08)
        self.assertEqual(fitted["fit_uses_held_samples"], 0.0)

    def test_tensor_whitening_beats_constant_and_permuted_volatility_nulls(self):
        from ka_l2p_conditional_diffusion import (
            conditional_covariance_diagnostic,
            fit_constant_covariance,
            fit_scaled_conditional_covariance,
        )

        rng = np.random.default_rng(404)
        times = 1400
        targets = 12
        log_scale = np.empty((times, targets))
        log_scale[0] = rng.normal(scale=0.5, size=targets)
        for time in range(1, times):
            log_scale[time] = 0.92 * log_scale[time - 1] + rng.normal(
                scale=0.18, size=targets
            )
        q = np.exp(log_scale)[..., None, None] * np.diag([0.4, 1.0, 2.0])
        residual = np.einsum(
            "...ij,...j->...i",
            np.linalg.cholesky(q),
            rng.normal(size=(times, targets, 3)),
        )
        split = 800
        tensor_fit = fit_scaled_conditional_covariance(residual[:split], q[:split])
        tensor_covariance = (
            tensor_fit["scale"] * q[split:]
            + tensor_fit["isotropic_floor"] * np.eye(3)
        )
        constant_covariance = np.broadcast_to(
            fit_constant_covariance(residual[:split]), q[split:].shape
        )
        permutation = np.random.default_rng(11).permutation(times - split)
        permuted_covariance = tensor_fit["scale"] * q[split:][permutation] + tensor_fit[
            "isotropic_floor"
        ] * np.eye(3)
        tensor = conditional_covariance_diagnostic(
            residual[split:], tensor_covariance, maximum_lag=40, gaussian_seed=9
        )
        constant = conditional_covariance_diagnostic(
            residual[split:], constant_covariance, maximum_lag=40, gaussian_seed=9
        )
        permuted = conditional_covariance_diagnostic(
            residual[split:], permuted_covariance, maximum_lag=40, gaussian_seed=9
        )

        self.assertLess(tensor["negative_log_likelihood"], constant["negative_log_likelihood"])
        self.assertLess(tensor["negative_log_likelihood"], permuted["negative_log_likelihood"])
        self.assertLess(tensor["maximum_absolute_squared_whitened_correlation"], 0.06)
        self.assertGreater(
            constant["maximum_absolute_squared_whitened_correlation"],
            tensor["maximum_absolute_squared_whitened_correlation"],
        )

    def test_replicate_first_interval_uses_four_clones_not_pooled_samples(self):
        from ka_l2p_conditional_diffusion import replicate_first_t_interval

        result = replicate_first_t_interval(np.array([0.4, 0.7, 0.5, 0.8]))
        expected_se = np.std([0.4, 0.7, 0.5, 0.8], ddof=1) / 2.0
        self.assertAlmostEqual(result["mean"], 0.6)
        self.assertAlmostEqual(result["standard_error"], expected_se)
        self.assertAlmostEqual(
            result["ci95_low"], 0.6 - 3.182446305284263 * expected_se
        )
        self.assertEqual(result["replicate_count"], 4.0)
        with self.assertRaisesRegex(ValueError, "four"):
            replicate_first_t_interval(np.ones(8))

    def test_analysis_cli_exposes_frozen_fold_and_covariance_models(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_ka_l2p_conditional_diffusion.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for option in (
            "--drift-cache-directory",
            "--second-generator-cache-directories",
            "--conditional-diffusion-cache-directory",
            "--memory-order",
            "--bath-order",
            "--maximum-lag",
            "--frame-time",
            "--output-prefix",
        ):
            self.assertIn(option, completed.stdout)

    def test_verdict_requires_all_four_folds_and_keeps_broad_claims_closed(self):
        module = load_script(
            "analyze_ka_l2p_conditional_diffusion.py",
            "analyze_ka_l2p_conditional_diffusion_test",
        )
        rows = []
        for fold in range(1, 5):
            for model, nll, squared, floor in (
                ("constant_full", 2.0, 0.12, 0.0),
                ("constant_isotropic", 2.2, 0.13, 0.0),
                ("trace_only", 1.8, 0.07, 0.10),
                ("exact_tensor", 1.5, 0.04, 0.10),
                ("permuted_tensor", 2.1, 0.11, 0.10),
            ):
                rows.append(
                    {
                        "fold_index": float(fold),
                        "model": model,
                        "negative_log_likelihood": nll + 0.01 * fold,
                        "maximum_absolute_squared_whitened_correlation": squared,
                        "maximum_absolute_whitened_correlation": 0.03,
                        "maximum_absolute_component_excess_kurtosis": 0.20,
                        "mean_squared_mahalanobis_per_dimension": 1.0,
                        "maximum_absolute_whitened_covariance_error": 0.06,
                        "isotropic_floor_variance_fraction": floor,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
        convergence = [
            {
                "fold_index": float(fold),
                "prefix_median_relative_frobenius_error": 0.05,
                "prefix_p95_relative_frobenius_error": 0.15,
                "step_median_relative_frobenius_error": 0.04,
                "step_p95_relative_frobenius_error": 0.14,
            }
            for fold in range(1, 5)
        ]

        verdict = module.classify_l2p_conditional_diffusion(rows, convergence)

        self.assertEqual(verdict["l2p_diffusion_probe_converged"], 1.0)
        self.assertEqual(verdict["l2p_conditional_diffusion_supported"], 1.0)
        self.assertEqual(verdict["l2p_tensor_orientation_required"], 1.0)
        for claim in (
            "microscopic_environment_coordinate_z_allowed",
            "continuous_gaussian_langevin_bath_allowed",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(verdict[claim], 0.0)

        with self.assertRaisesRegex(ValueError, "complete"):
            module.classify_l2p_conditional_diffusion(rows[:-1], convergence)
        failed = [dict(row) for row in rows]
        next(row for row in failed if row["model"] == "exact_tensor")[
            "maximum_absolute_component_excess_kurtosis"
        ] = 0.8
        failed_verdict = module.classify_l2p_conditional_diffusion(failed, convergence)
        self.assertEqual(failed_verdict["l2p_conditional_diffusion_supported"], 0.0)
        self.assertEqual(
            failed_verdict["l2p_conditional_diffusion_informative_but_insufficient"],
            1.0,
        )
        self.assertEqual(failed_verdict["l3p_derivation_authorized"], 1.0)


if __name__ == "__main__":
    unittest.main()
