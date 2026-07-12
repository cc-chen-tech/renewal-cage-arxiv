import json
import importlib.util
import pickle
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ka_replicates

from ka_replicates import (  # noqa: E402
    distance_resolved_event_count_covariance,
    fit_spatial_covariance_length,
    fit_ornstein_zernike_structure_factor,
    initial_configuration_fs,
    load_lammps_custom_trajectory,
    overlap_four_point_structure_factor,
    prepare_replicate,
    prepare_replicate_ensemble,
    summarize_replicate_curves,
    temperature_scan_verdict,
    validate_initial_frame_independence,
)


class KAReplicatePreparationTests(unittest.TestCase):
    def test_small_sample_confidence_interval_uses_student_t(self):
        interval = getattr(ka_replicates, "independent_sample_ci95", None)
        self.assertIsNotNone(interval)

        low, high, critical = interval(mean=5.0, standard_error=2.0, sample_count=3)

        self.assertAlmostEqual(critical, 4.302652729911275)
        self.assertAlmostEqual(low, 5.0 - 2.0 * critical)
        self.assertAlmostEqual(high, 5.0 + 2.0 * critical)

    def test_spatial_script_aggregates_independent_replicates_without_enabling_model_claim(self):
        script_path = ROOT / "scripts" / "analyze_ka_spatial_covariance.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_spatial_covariance", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        aggregate = getattr(module, "aggregate_block_rows", None)
        self.assertIsNotNone(aggregate)
        rows = []
        for replicate, length in ((1.0, 1.0), (2.0, 2.0)):
            for block in (0.0, 1.0):
                for distance in (1.0, 2.0, 3.0, 4.0):
                    rows.append(
                        {
                            "replicate": replicate,
                            "block_index": block,
                            "distance_midpoint": distance,
                            "covariance_excess_over_all_pairs": 2.0 * np.exp(-distance / length),
                        }
                    )

        _, summary, _, fit_summary = aggregate(rows, minimum_distance=1.0)

        self.assertEqual(summary[0]["uncertainty_scope"], "independent_replicates")
        self.assertEqual(fit_summary["spatial_measurement_claim_allowed"], 1.0)
        self.assertEqual(fit_summary["spatial_model_claim_allowed"], 0.0)
        self.assertEqual(fit_summary["independent_replicate_count"], 2.0)

    def test_spatial_cli_aggregate_mode_does_not_require_temperature(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inputs = []
            for replicate, length in ((1, 1.0), (2, 2.0)):
                path = root / f"replicate_{replicate}.csv"
                rows = [
                    {
                        "block_index": block,
                        "distance_midpoint": distance,
                        "covariance_excess_over_all_pairs": 2.0 * np.exp(-distance / length),
                    }
                    for block in (0, 1)
                    for distance in (1.0, 2.0, 3.0, 4.0)
                ]
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                inputs.append(path)
            output = root / "ensemble"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_ka_spatial_covariance.py"),
                    "--aggregate-block-files",
                    *(str(path) for path in inputs),
                    "--fit-minimum-distance",
                    "1.0",
                    "--output-prefix",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "ensemble_fit_summary.csv").is_file())

    def test_binned_metric_summary_uses_replicate_means_for_uncertainty(self):
        summarize = getattr(ka_replicates, "summarize_replicate_binned_metric", None)
        self.assertIsNotNone(summarize)
        rows = [
            {"replicate": 1.0, "distance": 1.0, "block": 0.0, "covariance": 2.0},
            {"replicate": 1.0, "distance": 1.0, "block": 1.0, "covariance": 4.0},
            {"replicate": 2.0, "distance": 1.0, "block": 0.0, "covariance": 6.0},
            {"replicate": 2.0, "distance": 1.0, "block": 1.0, "covariance": 8.0},
            {"replicate": 1.0, "distance": 2.0, "block": 0.0, "covariance": 1.0},
            {"replicate": 2.0, "distance": 2.0, "block": 0.0, "covariance": 3.0},
        ]

        replicate_rows, summary_rows = summarize(
            rows,
            bin_key="distance",
            metric_key="covariance",
        )

        first_bin_replicates = [row for row in replicate_rows if row["distance"] == 1.0]
        first_bin_summary = next(row for row in summary_rows if row["distance"] == 1.0)
        self.assertEqual([row["covariance"] for row in first_bin_replicates], [3.0, 7.0])
        self.assertAlmostEqual(first_bin_summary["mean"], 5.0)
        self.assertAlmostEqual(first_bin_summary["standard_error"], 2.0)
        self.assertEqual(first_bin_summary["independent_replicate_count"], 2.0)
        self.assertEqual(first_bin_summary["within_replicate_block_count_min"], 2.0)
        self.assertEqual(first_bin_summary["within_replicate_block_count_max"], 2.0)

    def test_spatial_covariance_ensemble_fits_lengths_per_replicate(self):
        summarize = getattr(ka_replicates, "summarize_spatial_covariance_replicates", None)
        self.assertIsNotNone(summarize)
        rows = []
        for replicate, length in ((1.0, 1.0), (2.0, 2.0)):
            for block in (0.0, 1.0):
                for distance in (1.0, 2.0, 3.0, 4.0):
                    rows.append(
                        {
                            "replicate": replicate,
                            "block_index": block,
                            "distance_midpoint": distance,
                            "covariance_excess_over_all_pairs": 2.0 * np.exp(-distance / length),
                        }
                    )

        _, curve, fits, fit_summary = summarize(rows, minimum_distance=1.0)

        self.assertEqual(len(curve), 4)
        self.assertEqual([row["replicate"] for row in fits], [1.0, 2.0])
        self.assertAlmostEqual(fits[0]["correlation_length"], 1.0)
        self.assertAlmostEqual(fits[1]["correlation_length"], 2.0)
        self.assertAlmostEqual(fit_summary["mean_correlation_length"], 1.5)
        self.assertAlmostEqual(fit_summary["standard_error_correlation_length"], 0.5)
        self.assertEqual(fit_summary["independent_replicate_count"], 2.0)

    def test_overlap_s4_replicate_gate_rejects_one_invalid_oz_fit(self):
        summarize = getattr(ka_replicates, "summarize_overlap_s4_replicates", None)
        self.assertIsNotNone(summarize)
        rows = []
        for replicate in (1.0, 2.0):
            rows.append({"replicate": replicate, "integer_squared": 0.0, "wave_number": 0.0, "s4": 2.0})
        for q, valid_s4, invalid_s4 in (
            (0.2, 10.0 / 1.16, 25.0),
            (0.3, 10.0 / 1.36, 4.5),
            (0.4, 10.0 / 1.64, 2.0),
            (0.5, 10.0 / 2.0, 1.1),
        ):
            rows.append({"replicate": 1.0, "integer_squared": q, "wave_number": q, "s4": valid_s4})
            rows.append({"replicate": 2.0, "integer_squared": q, "wave_number": q, "s4": invalid_s4})

        _, fits, verdict = summarize(rows, ensemble_correction_available=False)

        self.assertEqual(sum(not row["fit_valid"] for row in fits), 1)
        self.assertEqual(verdict["invalid_replicate_fit_count"], 1.0)
        self.assertEqual(verdict["raw_oz_fit_reproducible"], 0.0)
        self.assertEqual(verdict["xi4_identifiable"], 0.0)
        self.assertEqual(verdict["xi4_claim_allowed"], 0.0)

    def test_overlap_s4_script_preserves_missing_ensemble_correction_gate(self):
        script_path = ROOT / "scripts" / "analyze_ka_overlap_s4.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_overlap_s4", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        aggregate = getattr(module, "aggregate_curve_rows", None)
        self.assertIsNotNone(aggregate)
        rows = []
        for replicate in (1.0, 2.0):
            rows.append({"replicate": replicate, "integer_squared": 0.0, "wave_number": 0.0, "s4": 2.0})
            for q in (0.2, 0.3, 0.4, 0.5):
                rows.append(
                    {
                        "replicate": replicate,
                        "integer_squared": q,
                        "wave_number": q,
                        "s4": 10.0 / (1.0 + (2.0 * q) ** 2),
                    }
                )

        _, _, verdict = aggregate(rows)

        self.assertEqual(verdict["raw_oz_fit_reproducible"], 1.0)
        self.assertEqual(verdict["ensemble_correction_available"], 0.0)
        self.assertEqual(verdict["xi4_identifiable"], 0.0)
        self.assertEqual(verdict["xi4_claim_allowed"], 0.0)

    def test_overlap_s4_cli_aggregate_mode_does_not_require_temperature_or_lag(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inputs = []
            for replicate in (1, 2):
                path = root / f"replicate_{replicate}.csv"
                rows = [{"integer_squared": 0.0, "wave_number": 0.0, "s4": 2.0}]
                rows.extend(
                    {
                        "integer_squared": q,
                        "wave_number": q,
                        "s4": 10.0 / (1.0 + (2.0 * q) ** 2),
                    }
                    for q in (0.2, 0.3, 0.4, 0.5)
                )
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                inputs.append(path)
            output = root / "ensemble"

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "analyze_ka_overlap_s4.py"),
                    "--aggregate-curve-files",
                    *(str(path) for path in inputs),
                    "--output-prefix",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "ensemble_verdict.csv").is_file())

    def test_temperature_event_trend_reports_unequal_replicate_counts(self):
        script_path = ROOT / "scripts" / "summarize_ka_replicate_scan.py"
        spec = importlib.util.spec_from_file_location("summarize_ka_replicate_scan", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        metrics = (
            "event_rate",
            "exchange_mean",
            "stationary_persistence_mean",
            "persistence_exchange_ratio",
            "count_fano",
            "correlated_diffusion",
        )
        high = [
            {"metric": metric, "mean": 2.0, "ci95_low": 1.8, "ci95_high": 2.2, "independent_replicate_count": 5.0}
            for metric in metrics
        ]
        low = [
            {"metric": metric, "mean": 1.0, "ci95_low": 0.8, "ci95_high": 1.2, "independent_replicate_count": 3.0}
            for metric in metrics
        ]

        rows = module.event_temperature_trends(high, low)

        self.assertEqual(rows[0]["high_temperature_replicate_count"], 5.0)
        self.assertEqual(rows[0]["low_temperature_replicate_count"], 3.0)
        self.assertNotIn("independent_replicates_per_temperature", rows[0])

    def test_waiting_diagnostic_summary_requires_replica_consensus(self):
        script_path = ROOT / "scripts" / "analyze_ka_replicates.py"
        spec = importlib.util.spec_from_file_location("analyze_ka_replicates", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        summarize = getattr(module, "summarize_waiting_diagnostic_rows", None)
        self.assertIsNotNone(summarize)
        rows = []
        for replicate in (1.0, 2.0, 3.0):
            for window in (512.0, 1024.0):
                rows.append(
                    {
                        "replicate": replicate,
                        "count_window": window,
                        "empirical_iid_relative_error": 0.08,
                        "gamma_iid_relative_error": 0.30,
                        "temporal_memory_excess_fraction": 0.03,
                        "persistent_environment_excess_fraction": 0.25,
                        "waiting_lag1_correlation": 0.01,
                        "shuffle_lag1_correlation_mean": 0.0,
                        "shuffle_lag1_correlation_standard_deviation": 0.02,
                        "complete_wait_particle_fraction": 0.5,
                        "complete_waiting_time_count": 200.0,
                        "particles_with_complete_wait": 100.0,
                        "window_count": 8.0,
                        "collective_covariance_ratio": 4.0,
                    }
                )

        replicate_rows, verdict = summarize(rows)

        self.assertEqual(len(replicate_rows), 3)
        self.assertTrue(all(row["waiting_failure_verdict"] == "empirical_iid_waiting_law_sufficient" for row in replicate_rows))
        self.assertEqual(verdict["consensus_verdict"], "empirical_iid_waiting_law_sufficient")
        self.assertEqual(verdict["consensus_replicate_count"], 3.0)
        self.assertEqual(verdict["independent_replicate_count"], 3.0)
        self.assertTrue(all(row["persistent_environment_identifiable"] == 0.0 for row in replicate_rows))
        self.assertEqual(verdict["persistent_environment_identifiable"], 0.0)
        self.assertEqual(verdict["collective_covariance_replicate_count"], 3.0)
        self.assertEqual(verdict["spatial_cooperation_test_required"], 1.0)
        self.assertEqual(verdict["spatial_cooperation_proven"], 0.0)

    def test_ornstein_zernike_fit_recovers_length_and_rejects_negative_intercept(self):
        valid_rows = [
            {"wave_number": q, "s4": 10.0 / (1.0 + (2.0 * q) ** 2)}
            for q in (0.2, 0.3, 0.4, 0.5)
        ]
        valid = fit_ornstein_zernike_structure_factor(valid_rows)
        self.assertTrue(valid["fit_valid"])
        self.assertAlmostEqual(valid["amplitude"], 10.0)
        self.assertAlmostEqual(valid["correlation_length"], 2.0)

        invalid_rows = [
            {"wave_number": q, "s4": s4}
            for q, s4 in ((0.4, 25.0), (0.6, 4.5), (0.8, 2.0), (1.0, 1.1))
        ]
        invalid = fit_ornstein_zernike_structure_factor(invalid_rows)
        self.assertFalse(invalid["fit_valid"])
        self.assertLess(invalid["inverse_intercept"], 0.0)
        self.assertTrue(np.isnan(invalid["correlation_length"]))

    def test_overlap_s4_is_periodic_translation_invariant_and_has_q0_susceptibility(self):
        frame0 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        frame1 = frame0.copy()
        frame1[0, 0] += 1.0
        frame2 = frame1.copy()
        frame2[[0, 1, 2], 1] += 1.0
        trajectory = np.stack([frame0, frame1, frame2])

        rows = overlap_four_point_structure_factor(
            trajectory,
            box_lengths=np.array([4.0, 4.0, 4.0]),
            lag=1,
            overlap_radius=0.3,
            origin_stride=1,
            maximum_integer_squared=1,
        )
        shifted = overlap_four_point_structure_factor(
            trajectory + np.array([4.0, -4.0, 8.0]),
            box_lengths=np.array([4.0, 4.0, 4.0]),
            lag=1,
            overlap_radius=0.3,
            origin_stride=1,
            maximum_integer_squared=1,
        )

        self.assertEqual(rows[0]["integer_squared"], 0.0)
        self.assertAlmostEqual(rows[0]["s4"], 0.25)
        self.assertEqual(rows[1]["wavevector_count"], 6.0)
        self.assertGreaterEqual(rows[1]["s4"], 0.0)
        self.assertLessEqual(rows[1]["s4_wavevector_min"], rows[1]["s4"])
        self.assertGreaterEqual(rows[1]["s4_wavevector_max"], rows[1]["s4"])
        self.assertGreaterEqual(rows[1]["s4_wavevector_standard_deviation"], 0.0)
        np.testing.assert_allclose(
            [row["s4"] for row in rows],
            [row["s4"] for row in shifted],
        )

    def test_spatial_covariance_length_recovers_exponential_decay(self):
        rows = [
            {"distance_midpoint": r, "mean_covariance_excess": 2.0 * np.exp(-r / 1.5)}
            for r in (1.0, 2.0, 3.0, 4.0)
        ]

        fit = fit_spatial_covariance_length(rows, minimum_distance=1.0)

        self.assertAlmostEqual(fit["correlation_length"], 1.5)
        self.assertAlmostEqual(fit["amplitude"], 2.0)
        self.assertAlmostEqual(fit["log_space_r_squared"], 1.0)
        self.assertEqual(fit["fit_point_count"], 4.0)

    def test_distance_resolved_covariance_finds_close_pair_coactivity(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0], [5.5, 0.0, 0.0]]
        )
        events = {
            "particle": np.array([0, 1, 2, 3, 0, 1, 2, 3]),
            "time": np.array([1, 1, 11, 11, 21, 21, 31, 31]),
        }

        rows = distance_resolved_event_count_covariance(
            events,
            positions,
            np.array([20.0, 20.0, 20.0]),
            duration=40.0,
            count_window=10.0,
            distance_edges=np.array([0.0, 1.0, 10.0]),
        )

        close, far = rows
        self.assertEqual(close["pair_count"], 2.0)
        self.assertEqual(far["pair_count"], 4.0)
        self.assertGreater(close["covariance_excess_over_all_pairs"], 0.0)
        self.assertLess(far["covariance_excess_over_all_pairs"], 0.0)
        self.assertAlmostEqual(
            sum(row["pair_count"] * row["covariance_excess_over_all_pairs"] for row in rows),
            0.0,
        )

    def test_replicate_curve_summary_uses_between_trajectory_error(self):
        rows = [
            {"replicate": 1.0, "lag": 1.0, "msd": 1.0, "fs_k7p25": 0.8},
            {"replicate": 2.0, "lag": 1.0, "msd": 3.0, "fs_k7p25": 0.6},
            {"replicate": 1.0, "lag": 2.0, "msd": 2.0, "fs_k7p25": 0.5},
            {"replicate": 2.0, "lag": 2.0, "msd": 4.0, "fs_k7p25": 0.3},
        ]

        summary = summarize_replicate_curves(rows, metric_keys=["msd", "fs_k7p25"])

        msd_lag_one = next(
            row for row in summary if row["lag"] == 1.0 and row["metric"] == "msd"
        )
        self.assertEqual(msd_lag_one["independent_replicate_count"], 2.0)
        self.assertAlmostEqual(msd_lag_one["mean"], 2.0)
        self.assertAlmostEqual(msd_lag_one["standard_error"], 1.0)
        self.assertAlmostEqual(msd_lag_one["ci95_low"], 2.0 - 12.706204736432095)
        self.assertAlmostEqual(msd_lag_one["ci95_high"], 2.0 + 12.706204736432095)
        self.assertEqual(msd_lag_one["ci95_method"], "student_t_independent_replicates")

    def test_temperature_scan_requires_nonoverlapping_directional_intervals(self):
        high = [
            {"metric": "diffusion", "mean": 3.0, "ci95_low": 2.8, "ci95_high": 3.2},
            {"metric": "ngp_peak", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
            {"metric": "alpha_relaxation_time", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
            {"metric": "diffusion_alpha_product", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
            {"metric": "overlap_chi4_peak", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
        ]
        low = [
            {"metric": "diffusion", "mean": 1.0, "ci95_low": 0.9, "ci95_high": 1.1},
            {"metric": "ngp_peak", "mean": 1.5, "ci95_low": 1.3, "ci95_high": 1.7},
            {"metric": "alpha_relaxation_time", "mean": 1.5, "ci95_low": 1.3, "ci95_high": 1.7},
            {"metric": "diffusion_alpha_product", "mean": 1.5, "ci95_low": 1.3, "ci95_high": 1.7},
            {"metric": "overlap_chi4_peak", "mean": 1.5, "ci95_low": 1.3, "ci95_high": 1.7},
        ]

        rows = temperature_scan_verdict(high, low)

        diffusion = next(row for row in rows if row["metric"] == "diffusion")
        ngp = next(row for row in rows if row["metric"] == "ngp_peak")
        self.assertEqual(diffusion["effect"], "cooling_slowdown")
        self.assertAlmostEqual(diffusion["effect_ratio"], 3.0)
        self.assertTrue(diffusion["directional_ci95_separated"])
        self.assertEqual(ngp["effect"], "cooling_growth")
        self.assertAlmostEqual(ngp["effect_ratio"], 1.5)
        self.assertTrue(ngp["directional_ci95_separated"])

    def test_diffusion_estimate_uses_only_requested_heldout_origins(self):
        estimate = getattr(ka_replicates, "trajectory_diffusion_estimate", None)
        self.assertIsNotNone(estimate)
        positions = np.zeros((5, 2, 3), dtype=float)
        positions[:, :, 0] = np.arange(5)[:, None]

        diffusion = estimate(
            positions,
            lag=2,
            origin_stride=2,
            particle_mask=np.array([True, False]),
        )

        self.assertAlmostEqual(diffusion, 4.0 / 12.0)

    def test_heldout_transport_summary_rejects_event_clock_undercoverage(self):
        summarize = getattr(ka_replicates, "summarize_heldout_event_transport", None)
        self.assertIsNotNone(summarize)
        rows = [
            {
                "replicate": float(replicate),
                "observed_diffusion": observed,
                "uncorrelated_event_diffusion": 0.3 * observed,
                "correlated_event_diffusion": 0.4 * observed,
            }
            for replicate, observed in ((1, 1.0), (2, 1.1), (3, 0.9))
        ]

        summary, verdict = summarize(rows, minimum_coverage=0.8, maximum_coverage=1.2)

        correlated = next(row for row in summary if row["model"] == "correlated_event_clock")
        self.assertAlmostEqual(correlated["mean_coverage"], 0.4)
        self.assertEqual(verdict["heldout_transport_pass"], 0.0)
        self.assertEqual(verdict["primary_failure"], "correlated_event_clock_undercoverage")
        self.assertEqual(verdict["independent_replicate_count"], 3.0)

    def test_load_lammps_dump_reconstructs_unwrapped_positions(self):
        dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-2 2
-2 2
-2 2
ITEM: ATOMS id type x y z ix iy iz
1 1 1.5 0 0 0 0 0
2 2 -1.5 1 0 0 0 0
ITEM: TIMESTEP
1000
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-2 2
-2 2
-2 2
ITEM: ATOMS id type x y z ix iy iz
1 1 -1.5 0 0 1 0 0
2 2 1.5 1 0 -1 0 0
"""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "trajectory.lammpstrj"
            path.write_text(dump)

            trajectory = load_lammps_custom_trajectory(path)

        np.testing.assert_array_equal(trajectory["timesteps"], [0, 1000])
        np.testing.assert_array_equal(trajectory["particle_types"], [0, 1])
        np.testing.assert_allclose(trajectory["box_lengths"], [4.0, 4.0, 4.0])
        np.testing.assert_allclose(trajectory["wrapped_positions"][1, 0], [-1.5, 0.0, 0.0])
        np.testing.assert_allclose(trajectory["unwrapped_positions"][1, 0], [2.5, 0.0, 0.0])
        np.testing.assert_allclose(trajectory["unwrapped_positions"][1, 1], [-2.5, 1.0, 0.0])

    def test_initial_configuration_fs_uses_minimum_images_and_a_particles(self):
        box = np.array([10.0, 10.0, 10.0])
        reference = np.array([[4.9, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        displaced = np.array([[-4.9, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
        particle_mask = np.array([True, True, False])

        value = initial_configuration_fs(
            reference,
            displaced,
            box,
            particle_mask=particle_mask,
            wave_number=np.pi,
        )

        expected = (np.cos(0.2 * np.pi) + 4.0 + np.cos(2.0 * np.pi)) / 6.0
        self.assertAlmostEqual(value, expected)

    def test_independence_gate_rejects_correlated_frame_pair(self):
        positions = np.zeros((3, 2, 3), dtype=float)
        positions[1] = 0.01
        positions[2, :, 0] = np.array([0.5, -0.5])

        with self.assertRaisesRegex(ValueError, "not decorrelated"):
            validate_initial_frame_independence(
                positions,
                np.array([10.0, 10.0, 10.0]),
                np.array([True, True]),
                frame_indices=[0, 1, 2],
                wave_number=7.25,
                maximum_absolute_fs=0.2,
            )

    def test_prepare_replicate_writes_physical_protocol_and_provenance(self):
        box = np.array([[4.0, 4.0, 4.0]])
        types = np.array([[0, 0, 0, 1]])
        positions = [
            np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, 1.0, -1.0],
                ],
                dtype=np.float32,
            )
        ]
        payload = {"Box_size": box, "Particle_types": types, "Positions": positions}

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source.pkl"
            with source.open("wb") as handle:
                pickle.dump(payload, handle, protocol=4)
            output = root / "replicate"

            manifest = prepare_replicate(
                source,
                output,
                temperature=0.45,
                frame_index=0,
                velocity_seed=45117,
                equilibration_time=100.0,
                production_time=5000.0,
            )

            lammps_input = (output / "in.production").read_text()
            self.assertIn("pair_coeff 1 2 1.5 0.8 2.0", lammps_input)
            self.assertIn("pair_modify shift yes", lammps_input)
            self.assertIn("fix thermostat all nvt temp 0.45 0.45 10", lammps_input)
            self.assertIn("timestep 0.001", lammps_input)
            self.assertIn("dump trajectory all custom 1000", lammps_input)
            self.assertIn("restart 100000", lammps_input)
            self.assertIn("run 100000", lammps_input)
            self.assertIn("reset_timestep 0", lammps_input)
            self.assertIn("run 5000000", lammps_input)

            stored = json.loads((output / "manifest.json").read_text())
            self.assertEqual(stored, manifest)
            self.assertEqual(stored["source_doi"], "10.5281/zenodo.7469766")
            self.assertEqual(stored["independence_class"], "decorrelated_parent_frames_plus_velocity_seeds")
            self.assertFalse(stored["independently_prepared_parent_samples"])
            self.assertEqual(stored["saved_frame_interval_tau"], 1.0)
            self.assertEqual(stored["particle_counts"], {"A": 3, "B": 1, "total": 4})
            self.assertEqual(len(stored["source_sha256"]), 64)

    def test_prepare_ensemble_records_pairwise_decorrelation(self):
        rng = np.random.default_rng(17)
        payload = {
            "Box_size": np.array([[20.0, 20.0, 20.0]]),
            "Particle_types": np.array([[0] * 16 + [1] * 4]),
            "Positions": [rng.uniform(-10.0, 10.0, size=(20, 3)) for _ in range(2)],
        }
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source.pkl"
            with source.open("wb") as handle:
                pickle.dump(payload, handle, protocol=4)

            manifest = prepare_replicate_ensemble(
                source,
                root / "ensemble",
                temperature=0.7,
                frame_indices=[0, 1],
                velocity_seeds=[70117, 70139],
                maximum_absolute_fs=0.5,
                equilibration_time=0.1,
                production_time=1.0,
            )

            self.assertEqual(manifest["replicate_count"], 2)
            self.assertEqual(len(manifest["pairwise_initial_fs"]), 1)
            self.assertLess(abs(manifest["pairwise_initial_fs"][0]["fs"]), 0.5)
            self.assertTrue((root / "ensemble" / "replicate_01" / "in.production").is_file())
            self.assertTrue((root / "ensemble" / "replicate_02" / "manifest.json").is_file())
            self.assertEqual(
                json.loads((root / "ensemble" / "ensemble_manifest.json").read_text()),
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
