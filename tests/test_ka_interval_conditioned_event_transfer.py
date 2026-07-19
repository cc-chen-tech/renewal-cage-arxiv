import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(filename: str, module_name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IntervalConditionedEventTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script(
            "summarize_ka_interval_conditioned_event_transfer.py",
            "summarize_ka_interval_conditioned_event_transfer",
        )
        cls.generator = load_script(
            "analyze_ka_interval_conditioned_event_transfer.py",
            "analyze_ka_interval_conditioned_event_transfer",
        )

    def test_legacy_interval_cache_is_canonicalized_without_changing_values(self):
        legacy = {
            "pmf": np.array([0.7, 0.3]),
            "samples": np.array([7.0, 3.0]),
            "msd": np.array([0.0, 2.0]),
            "fourth": np.array([0.0, 5.0]),
            "fs_k2": np.array([1.0, 0.4]),
            "fs_k4": np.array([1.0, -0.1]),
            "fs_k7.25": np.array([1.0, 0.2]),
        }

        canonical = self.generator.canonicalize_interval_statistics(legacy)

        np.testing.assert_array_equal(canonical["count_pmf"], legacy["pmf"])
        np.testing.assert_array_equal(canonical["sample_count"], legacy["samples"])
        np.testing.assert_array_equal(canonical["conditional_msd"], legacy["msd"])
        np.testing.assert_array_equal(
            canonical["conditional_fourth_moment"], legacy["fourth"]
        )
        np.testing.assert_array_equal(
            canonical["conditional_characteristic_k7p25"], legacy["fs_k7.25"]
        )

    def test_interval_cache_round_trip_preserves_lag_grid(self):
        statistics = {
            2: {
                "count_pmf": np.array([0.5, 0.5]),
                "sample_count": np.array([2.0, 2.0]),
                "conditional_msd": np.array([0.0, 1.0]),
                "conditional_fourth_moment": np.array([0.0, 1.0]),
                "conditional_characteristic_k2": np.array([1.0, 0.2]),
                "conditional_characteristic_k4": np.array([1.0, -0.1]),
                "conditional_characteristic_k7p25": np.array([1.0, 0.05]),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            self.generator.save_interval_statistics(path, statistics)
            loaded = self.generator.load_interval_statistics(path, (2,))

        self.assertEqual(set(loaded), {2})
        for key, expected in statistics[2].items():
            np.testing.assert_array_equal(loaded[2][key], expected)

    def test_three_channel_shapley_attribution_is_exact(self):
        values = {
            "ccc": 1.0,
            "hcc": 2.0,
            "chc": 4.0,
            "cch": 8.0,
            "hhc": 7.0,
            "hch": 12.0,
            "chh": 15.0,
            "hhh": 24.0,
        }

        attribution = self.analysis.three_channel_shapley(values)

        self.assertEqual(set(attribution), {"count", "kernel", "residual"})
        self.assertAlmostEqual(sum(attribution.values()), values["hhh"] - values["ccc"])

    def test_committed_low_temperature_canary_passes_only_at_ensemble_level(self):
        rows, curves, summaries, verdicts, _, ablation_rows, _, ablation = (
            self.analysis.analyze_committed_tables(ROOT)
        )
        low_summary = next(
            row
            for row in summaries
            if math.isclose(float(row["temperature"]), 0.45)
            and row["model_code"] == "ccc"
        )
        low_verdict = next(
            row for row in verdicts if math.isclose(float(row["temperature"]), 0.45)
        )

        self.assertEqual(len(rows), 496)
        self.assertEqual(len(curves), 128)
        self.assertEqual(len(ablation_rows), 124)
        self.assertAlmostEqual(
            float(low_summary["maximum_ensemble_msd_relative_error"]),
            0.07234265580045962,
        )
        self.assertAlmostEqual(
            float(low_summary["maximum_ensemble_ngp_absolute_error"]),
            0.23622943262158413,
        )
        self.assertAlmostEqual(
            float(low_summary["maximum_ensemble_fs_absolute_error"]),
            0.023856888817672783,
        )
        self.assertAlmostEqual(float(low_summary["diffusion_relative_error"]), 0.03433959173510892)
        self.assertAlmostEqual(
            float(low_summary["alpha_relaxation_relative_error"]),
            0.030553750757048803,
        )
        self.assertAlmostEqual(
            float(low_summary["diffusion_alpha_product_relative_error"]),
            0.0048350443050334535,
        )
        self.assertEqual(float(low_verdict["retrospective_ensemble_canary_pass"]), 1.0)
        self.assertEqual(float(low_verdict["replicate_level_transfer_pass"]), 0.0)
        self.assertEqual(float(low_verdict["independent_parent_sample_count"]), 1.0)
        self.assertEqual(float(low_verdict["independent_parent_sufficiency_pass"]), 0.0)
        self.assertEqual(float(low_verdict["previous_global_hybrid_joint_pass"]), 0.0)
        low_pooled = next(
            row
            for row in ablation
            if math.isclose(float(row["temperature"]), 0.45)
            and row["kernel_model"] == "lag_pooled"
        )
        self.assertAlmostEqual(
            float(low_pooled["maximum_ensemble_msd_relative_error"]),
            0.08874278377079559,
        )
        self.assertAlmostEqual(
            float(low_pooled["maximum_ensemble_ngp_absolute_error"]),
            0.1892839855418178,
        )
        self.assertAlmostEqual(
            float(low_pooled["maximum_ensemble_fs_absolute_error"]),
            0.03307694369951619,
        )
        self.assertEqual(float(low_pooled["ensemble_curve_pass"]), 0.0)
        self.assertEqual(
            float(low_verdict["joint_empirical_interval_closure_repairs_old_hybrid_canary"]),
            1.0,
        )
        self.assertEqual(
            float(low_verdict["lag_conditioning_required_for_frozen_multik_gate"]),
            1.0,
        )
        self.assertGreater(
            float(low_verdict["lag_conditioning_maximum_fs_error_reduction_fraction"]),
            0.27,
        )
        self.assertEqual(
            float(low_verdict["restart_labels_with_lower_mean_fs_error"]), 2.0
        )
        self.assertEqual(
            float(low_verdict["uniform_restart_mean_fs_improvement"]), 0.0
        )
        self.assertEqual(
            float(low_verdict["lag_conditioning_generalization_claim_allowed"]), 0.0
        )
        self.assertEqual(float(low_verdict["preregistered_prediction_claim_allowed"]), 0.0)
        self.assertEqual(float(low_verdict["finite_memory_parametric_claim_allowed"]), 0.0)
        self.assertEqual(float(low_verdict["thermodynamic_claim_allowed"]), 0.0)

    def test_high_temperature_oracle_rejects_universal_cage_event_factorization(self):
        _, _, _, verdicts, _, _, _, _ = self.analysis.analyze_committed_tables(ROOT)
        high = next(
            row for row in verdicts if math.isclose(float(row["temperature"]), 0.58)
        )

        self.assertEqual(float(high["heldout_oracle_factorization_pass"]), 0.0)
        self.assertGreater(float(high["oracle_maximum_ensemble_msd_relative_error"]), 0.3)
        self.assertGreater(float(high["oracle_maximum_ensemble_ngp_absolute_error"]), 0.5)
        self.assertEqual(float(high["universal_cage_event_representation_claim_allowed"]), 0.0)
        self.assertEqual(float(high["thermodynamic_claim_allowed"]), 0.0)

    def test_provenance_names_obadiya_sussman_archive_not_glassbench(self):
        path = ROOT / "data" / "renewal_cage_ka_interval_conditioned_event_transfer_provenance.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual({row["source_doi"] for row in rows}, {"10.5281/zenodo.7469766"})
        self.assertTrue(all("Obadiya" in row["source_dataset"] for row in rows))
        self.assertTrue(all("GlassBench" not in row["source_dataset"] for row in rows))
        self.assertTrue(all(float(row["independently_prepared_parent_samples"]) == 0.0 for row in rows))

    def test_analysis_fails_closed_when_counterfactual_grid_is_incomplete(self):
        rows, _, _, _, _, ablation_rows, _, _ = self.analysis.analyze_committed_tables(ROOT)
        truncated = rows[:-1]

        with self.assertRaisesRegex(ValueError, "three-channel grid"):
            self.analysis.validate_transfer_rows(truncated)

        with self.assertRaisesRegex(ValueError, "kernel-ablation grid"):
            self.analysis.validate_kernel_ablation_rows(ablation_rows[:-1])

    def test_svg_exposes_positive_canary_and_negative_controls(self):
        _, curves, summaries, verdicts, attributions, _, ablation_curves, ablation = (
            self.analysis.analyze_committed_tables(ROOT)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interval-transfer.svg"
            self.analysis.write_svg(
                path,
                curves,
                summaries,
                verdicts,
                attributions,
                ablation_curves,
                ablation,
            )
            svg = path.read_text()

        self.assertIn("Lag-conditioned event transfer", svg)
        self.assertIn("ensemble canary", svg)
        self.assertIn("replicate transfer: fail", svg)
        self.assertIn("high-T oracle factorization: fail", svg)
        self.assertIn("independent parents: 1", svg)
        self.assertIn("lag-pooled K(n): fail", svg)
        self.assertIn("lag-conditioned Kt(n): pass", svg)


if __name__ == "__main__":
    unittest.main()
