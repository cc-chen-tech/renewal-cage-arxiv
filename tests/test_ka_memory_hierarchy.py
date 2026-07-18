import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "summarize_ka_memory_hierarchy.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_ka_memory_hierarchy_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_inputs():
    return {
        "path": {
            "low_temperature_contiguous_closure": "1",
            "high_temperature_contiguous_closure": "1",
            "low_temperature_markov_failure": "1",
            "shared_low_temperature_higher_order_failure": "1",
            "shuffle_precision_pass": "1",
            "within_particle_time_shuffle_curve_transfer_pass": "0",
            "direction_randomized_path_curve_transfer_pass": "0",
            "replicate_consensus_pass": "1",
            "single_particle_multiblock_path_memory_required": "1",
            "amplitude_persistence_alone_sufficient": "0",
            "ordered_recoil_path_required": "1",
            "microdynamic_closure_claim_allowed": "0",
            "spatial_facilitation_claim_allowed": "0",
            "thermodynamic_claim_allowed": "0",
        },
        "anchor": {
            "mechanism_state": "anchor_aware_model_rejected",
            "provenance_and_competitor_completeness_pass": "1",
            "all_low_anchor_replicates_improve_over_recoil": "1",
            "low_anchor_aware_semi_markov_curve_transfer_pass": "0",
            "high_anchor_aware_semi_markov_curve_transfer_pass": "0",
            "microdynamic_closure_claim_allowed": "0",
            "spatial_facilitation_claim_allowed": "0",
            "thermodynamic_claim_allowed": "0",
        },
        "waiting": {
            "threshold_robust_dominant_mechanism": "1",
            "dominant_mechanism": "mixed_particle_environment_and_event_memory",
            "median_window_particle_conditioned_shuffle_sufficient": "1",
            "all_window_particle_conditioned_shuffle_sufficient": "0",
            "temporal_waiting_memory_supported": "1",
            "temporal_waiting_memory_parameter_claim_allowed": "0",
            "persistent_particle_environment_supported": "1",
            "finite_exchange_supported_by_prior_identity_decay": "1",
            "minimum_temporal_ordering_contribution_fraction": "0.072",
            "minimum_particle_identity_contribution_fraction": "0.087",
            "spatial_cooperation_test_required": "1",
            "spatial_cooperation_proven": "0",
            "thermodynamic_claim_allowed": "0",
        },
        "environment": {
            "waiting_mechanism_crossover_detected": "1",
            "exchange_time_growth_detected_all_block_sizes": "1",
            "minimum_exchange_time_growth_ratio": "2.96",
            "minimum_exchange_time_growth_ci95_low": "1.32",
            "cross_half_identity_correlation_growth_ratio": "5.13",
            "cross_half_identity_correlation_growth_ci95_low": "1.67",
            "pure_static_particle_rate_disorder_rejected": "1",
            "finite_exchange_environment_claim_allowed": "1",
            "finite_waiting_sequence_memory_required": "0",
            "spatial_facilitation_claim_allowed": "0",
            "thermodynamic_claim_allowed": "0",
        },
        "spatial_rows": [
            {
                "distance_midpoint": "0.55",
                "mean": "0.0014",
                "ci95_low": "0.0012",
                "ci95_high": "0.0016",
                "independent_replicate_count": "3",
                "spatial_measurement_claim_allowed": "1",
                "spatial_model_claim_allowed": "0",
                "thermodynamic_claim_allowed": "0",
            },
            {
                "distance_midpoint": "4.25",
                "mean": "0.0001",
                "ci95_low": "-0.0001",
                "ci95_high": "0.0003",
                "independent_replicate_count": "3",
                "spatial_measurement_claim_allowed": "1",
                "spatial_model_claim_allowed": "0",
                "thermodynamic_claim_allowed": "0",
            },
        ],
        "spatial_fit": {
            "mean_correlation_length": "0.695",
            "ci95_low_correlation_length": "0.451",
            "ci95_high_correlation_length": "0.940",
            "independent_replicate_count": "3",
            "fit_status": "between_replicate_uncertainty",
            "spatial_measurement_claim_allowed": "1",
            "spatial_model_claim_allowed": "0",
            "thermodynamic_claim_allowed": "0",
        },
    }


class MemoryHierarchyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_selects_ordered_path_and_supports_finite_exchange(self):
        verdict = self.module.classify_memory_hierarchy(**fixture_inputs())

        self.assertEqual(
            verdict["mechanism_state"],
            "ordered_particle_path_required_finite_exchange_supported",
        )
        self.assertEqual(verdict["evidence_completeness_pass"], 1.0)
        self.assertEqual(verdict["pooled_one_step_rejected"], 1.0)
        self.assertEqual(verdict["pooled_anchor_semi_markov_rejected"], 1.0)
        self.assertEqual(verdict["particle_identity_without_order_sufficient"], 0.0)
        self.assertEqual(verdict["ordered_particle_path_upper_bound_closes"], 1.0)
        self.assertEqual(verdict["finite_exchange_environment_supported"], 1.0)
        self.assertEqual(
            verdict["next_minimal_model_candidate"],
            "particle_conditioned_finite_exchange_ordered_path_kernel",
        )
        self.assertEqual(
            verdict["selection_scope"], "within_frozen_single_particle_alternatives"
        )
        self.assertEqual(verdict["static_environment_family_excluded"], 0.0)
        self.assertEqual(verdict["cross_particle_mechanism_excluded"], 0.0)

    def test_rejects_hierarchy_when_ordered_path_does_not_close(self):
        inputs = fixture_inputs()
        inputs["path"]["low_temperature_contiguous_closure"] = "0"
        inputs["path"]["single_particle_multiblock_path_memory_required"] = "0"
        inputs["path"]["ordered_recoil_path_required"] = "0"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(verdict["mechanism_state"], "ordered_path_hierarchy_rejected")

    def test_leaves_environment_unresolved_without_finite_exchange_evidence(self):
        inputs = fixture_inputs()
        inputs["environment"]["finite_exchange_environment_claim_allowed"] = "0"
        inputs["environment"]["minimum_exchange_time_growth_ci95_low"] = "0.90"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(
            verdict["mechanism_state"], "ordered_path_required_environment_unresolved"
        )

    def test_spatial_measurement_never_becomes_spatial_mechanism_claim(self):
        verdict = self.module.classify_memory_hierarchy(**fixture_inputs())
        self.assertEqual(verdict["positive_short_range_spatial_covariance_measured"], 1.0)
        self.assertAlmostEqual(verdict["spatial_correlation_length"], 0.695)
        self.assertEqual(verdict["spatial_facilitation_required"], 0.0)
        self.assertEqual(verdict["spatial_facilitation_claim_allowed"], 0.0)

    def test_upstream_claim_overreach_invalidates_hierarchy(self):
        inputs = fixture_inputs()
        inputs["anchor"]["microdynamic_closure_claim_allowed"] = "1"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(verdict["mechanism_state"], "mechanism_hierarchy_unresolved")
        self.assertEqual(verdict["evidence_completeness_pass"], 0.0)

    def test_missing_claim_guard_invalidates_hierarchy(self):
        inputs = fixture_inputs()
        inputs["path"].pop("thermodynamic_claim_allowed")
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(verdict["mechanism_state"], "mechanism_hierarchy_unresolved")
        self.assertEqual(verdict["evidence_completeness_pass"], 0.0)

    def test_waiting_memory_parameter_overclaim_invalidates_hierarchy(self):
        inputs = fixture_inputs()
        inputs["waiting"]["temporal_waiting_memory_parameter_claim_allowed"] = "1"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(verdict["mechanism_state"], "mechanism_hierarchy_unresolved")
        self.assertEqual(verdict["evidence_completeness_pass"], 0.0)

    def test_missing_particle_conditioned_shuffle_support_leaves_environment_open(self):
        inputs = fixture_inputs()
        inputs["waiting"]["median_window_particle_conditioned_shuffle_sufficient"] = "0"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(
            verdict["mechanism_state"], "ordered_path_required_environment_unresolved"
        )

    def test_shuffle_precision_failure_is_unresolved_not_identity_sufficient(self):
        inputs = fixture_inputs()
        inputs["path"]["shuffle_precision_pass"] = "0"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(verdict["particle_identity_without_order_state"], "unresolved")
        self.assertEqual(verdict["mechanism_state"], "mechanism_hierarchy_unresolved")

    def test_far_range_positive_bin_is_not_called_short_range_covariance(self):
        inputs = fixture_inputs()
        inputs["spatial_rows"][0]["ci95_low"] = "-0.001"
        inputs["spatial_rows"][1]["distance_midpoint"] = "8.0"
        inputs["spatial_rows"][1]["ci95_low"] = "0.0001"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(verdict["positive_short_range_spatial_covariance_measured"], 0.0)
        self.assertEqual(verdict["spatial_facilitation_claim_allowed"], 0.0)

    def test_spatial_model_overclaim_invalidates_hierarchy(self):
        inputs = fixture_inputs()
        inputs["spatial_fit"]["spatial_model_claim_allowed"] = "1"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(verdict["mechanism_state"], "mechanism_hierarchy_unresolved")
        self.assertEqual(verdict["evidence_completeness_pass"], 0.0)

    def test_invalid_spatial_fit_status_invalidates_hierarchy(self):
        inputs = fixture_inputs()
        inputs["spatial_fit"]["fit_status"] = "single_curve_only"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        self.assertEqual(verdict["mechanism_state"], "mechanism_hierarchy_unresolved")
        self.assertEqual(verdict["evidence_completeness_pass"], 0.0)

    def test_counterfactual_artifacts_do_not_keep_selected_finite_exchange_text(self):
        inputs = fixture_inputs()
        inputs["environment"]["finite_exchange_environment_claim_allowed"] = "0"
        inputs["environment"]["minimum_exchange_time_growth_ci95_low"] = "0.9"
        verdict = self.module.classify_memory_hierarchy(**inputs)
        with tempfile.TemporaryDirectory() as directory:
            figure = Path(directory) / "counterfactual.svg"
            rows = self.module.build_evidence_rows(verdict)
            self.module.write_svg(figure, verdict)
            svg = figure.read_text()
        finite_exchange_row = next(
            row for row in rows if row["evidence_stage"] == "finite_exchange_environment"
        )
        self.assertEqual(finite_exchange_row["heldout_result"], "not supported")
        self.assertIn("environment mechanism unresolved", svg.lower())
        self.assertNotIn("selected next candidate", svg.lower())

    def test_all_ablation_interpretations_follow_counterfactual_results(self):
        verdict = self.module.classify_memory_hierarchy(**fixture_inputs())
        verdict["pooled_one_step_rejected"] = 0.0
        verdict["pooled_anchor_semi_markov_rejected"] = 0.0
        verdict["particle_identity_without_order_state"] = "unresolved"
        verdict["ordered_particle_path_upper_bound_closes"] = 0.0
        rows = {
            row["evidence_stage"]: row
            for row in self.module.build_evidence_rows(verdict)
        }
        self.assertIn("not established", rows["pooled_one_step_recoil"]["interpretation"])
        self.assertIn("not established", rows["pooled_anchor_semi_markov"]["interpretation"])
        self.assertIn("unresolved", rows["particle_identity_without_order"]["interpretation"])
        self.assertIn("not established", rows["ordered_particle_path"]["interpretation"])
        self.assertNotIn("insufficient", rows["pooled_one_step_recoil"]["interpretation"])
        self.assertNotIn("selects", rows["ordered_particle_path"]["interpretation"])

    def test_reports_independent_identity_and_ordering_effect_sizes(self):
        verdict = self.module.classify_memory_hierarchy(**fixture_inputs())
        self.assertAlmostEqual(verdict["minimum_temporal_ordering_contribution_fraction"], 0.072)
        self.assertAlmostEqual(verdict["minimum_particle_identity_contribution_fraction"], 0.087)
        self.assertAlmostEqual(verdict["minimum_exchange_time_growth_ci95_low"], 1.32)

    def test_real_inputs_recompute_hierarchy_and_write_evidence_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "hierarchy.csv"
            evidence = root / "evidence.csv"
            figure = root / "hierarchy.svg"
            self.module.main(
                [
                    "--data-dir",
                    str(ROOT / "data"),
                    "--output-csv",
                    str(output),
                    "--output-evidence-csv",
                    str(evidence),
                    "--output-svg",
                    str(figure),
                ]
            )
            with output.open() as handle:
                verdict = next(csv.DictReader(handle))
            with evidence.open() as handle:
                rows = list(csv.DictReader(handle))
            svg = figure.read_text()
        self.assertEqual(
            verdict["mechanism_state"],
            "ordered_particle_path_required_finite_exchange_supported",
        )
        self.assertEqual(float(verdict["evidence_completeness_pass"]), 1.0)
        self.assertEqual(float(verdict["environment_recomputed_from_raw"]), 1.0)
        self.assertEqual(
            float(verdict["environment_waiting_consensus_recomputed_from_windows"]),
            1.0,
        )
        self.assertEqual(float(verdict["waiting_recomputed_from_thresholds"]), 1.0)
        self.assertEqual(
            float(verdict["stored_environment_waiting_consistency_pass"]), 1.0
        )
        self.assertEqual(float(verdict["stored_subgate_consistency_pass"]), 1.0)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1]["heldout_result"], "measurement only")
        self.assertEqual(rows[-1]["mechanism_claim_allowed"], "0.0")
        self.assertIn("Information-ablation hierarchy", svg)
        self.assertIn("measurement only", svg.lower())
        self.assertNotIn(">nan<", svg.lower())
        self.assertNotIn(">inf<", svg.lower())


if __name__ == "__main__":
    unittest.main()
