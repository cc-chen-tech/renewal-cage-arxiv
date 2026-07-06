import sys
import tempfile
import time
import unittest
import zipfile
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_arxiv_package import build_arxiv_package  # noqa: E402


class ArxivPackageTests(unittest.TestCase):
    def test_sota_comparison_table_keeps_scope_boundaries(self):
        path = ROOT / "data" / "renewal_cage_sota_comparison.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        phenomena = {row["phenomenon"]: row for row in rows}
        self.assertIn("thermodynamic_glass_transition", phenomena)
        self.assertIn("spatial_chi4_length", phenomena)
        self.assertIn("persistence_exchange_decoupling", phenomena)
        self.assertIn("mct_beta_relaxation", phenomena)
        self.assertEqual(phenomena["thermodynamic_glass_transition"]["model_status"], "partial")
        self.assertEqual(phenomena["spatial_chi4_length"]["model_status"], "partial")
        self.assertEqual(phenomena["persistence_exchange_decoupling"]["model_status"], "supported")
        self.assertEqual(phenomena["mct_beta_relaxation"]["model_status"], "partial")
        self.assertGreaterEqual(len(rows), 10)

    def test_sota_benchmark_consistency_contains_multiple_mechanism_checks(self):
        path = ROOT / "data" / "renewal_cage_sota_benchmark_consistency.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["benchmark_id"]: row for row in rows}
        self.assertIn("debye_waller_cage_localization", by_id)
        self.assertIn("kob_andersen_1995_beta_window", by_id)
        self.assertIn("kob_andersen_1995_mct_exponent_parameter", by_id)
        self.assertIn("gaussian_recovery_finite_exchange_vs_static_disorder", by_id)
        self.assertIn("ngp_peak_shift_on_cooling", by_id)
        self.assertIn("stokes_einstein_fractional_decoupling", by_id)
        self.assertIn("dynamic_heterogeneity_chi4_growth", by_id)
        self.assertIn("spatial_facilitation_constant_front_law", by_id)
        self.assertIn("alpha_tts_breakdown_shape_residual", by_id)
        self.assertIn("kww_alpha_stretching_on_cooling", by_id)
        self.assertIn("persistence_exchange_transport_inversion", by_id)
        self.assertIn("joint_persistence_exchange_multik_chi4_protocol", by_id)
        self.assertIn("kob_andersen_van_hove_tail_recovery", by_id)
        self.assertIn("angell_adam_gibbs_fragility_growth", by_id)
        self.assertIn("thermodynamic_transition_scope_boundary", by_id)
        self.assertEqual(float(by_id["debye_waller_cage_localization"]["overall_consistent"]), 1.0)
        self.assertLess(float(by_id["debye_waller_cage_localization"]["renewal_msd_fraction"]), 0.05)
        self.assertGreater(float(by_id["debye_waller_cage_localization"]["alpha_to_cage_time_ratio"]), 20.0)
        self.assertEqual(float(by_id["kob_andersen_1995_beta_window"]["overall_consistent"]), 1.0)
        self.assertEqual(float(by_id["kob_andersen_1995_mct_exponent_parameter"]["overall_consistent"]), 1.0)
        self.assertLess(
            float(by_id["kob_andersen_1995_mct_exponent_parameter"]["lambda_relative_mismatch"]),
            0.05,
        )
        self.assertEqual(
            float(by_id["gaussian_recovery_finite_exchange_vs_static_disorder"]["mechanism_selection_consistent"]),
            1.0,
        )
        self.assertEqual(float(by_id["ngp_peak_shift_on_cooling"]["overall_consistent"]), 1.0)
        self.assertGreater(float(by_id["ngp_peak_shift_on_cooling"]["peak_time_growth"]), 2.0)
        self.assertGreater(float(by_id["ngp_peak_shift_on_cooling"]["peak_height_growth"]), 1.1)
        self.assertLess(float(by_id["ngp_peak_shift_on_cooling"]["late_ngp"]), 0.05)
        self.assertEqual(float(by_id["stokes_einstein_fractional_decoupling"]["overall_consistent"]), 1.0)
        self.assertEqual(float(by_id["dynamic_heterogeneity_chi4_growth"]["overall_consistent"]), 1.0)
        self.assertEqual(float(by_id["spatial_facilitation_constant_front_law"]["overall_consistent"]), 1.0)
        self.assertLess(
            float(by_id["spatial_facilitation_constant_front_law"]["facilitation_diffusivity_relative_std"]),
            0.05,
        )
        self.assertEqual(float(by_id["alpha_tts_breakdown_shape_residual"]["overall_consistent"]), 1.0)
        self.assertGreater(float(by_id["alpha_tts_breakdown_shape_residual"]["cold_shape_residual"]), 0.25)
        self.assertGreater(float(by_id["alpha_tts_breakdown_shape_residual"]["alpha_shape_control_growth"]), 2.0)
        self.assertEqual(float(by_id["kww_alpha_stretching_on_cooling"]["overall_consistent"]), 1.0)
        self.assertLess(float(by_id["kww_alpha_stretching_on_cooling"]["cold_kww_beta"]), 0.9)
        self.assertGreater(float(by_id["kww_alpha_stretching_on_cooling"]["kww_beta_drop"]), 0.05)
        self.assertEqual(float(by_id["persistence_exchange_transport_inversion"]["overall_consistent"]), 1.0)
        self.assertGreater(
            float(by_id["persistence_exchange_transport_inversion"]["inferred_persistence_exchange_ratio"]),
            2.0,
        )
        self.assertLess(
            abs(float(by_id["persistence_exchange_transport_inversion"]["late_ngp_log_residual_benchmark"])),
            0.1,
        )
        self.assertEqual(float(by_id["joint_persistence_exchange_multik_chi4_protocol"]["overall_consistent"]), 1.0)
        self.assertGreater(
            float(by_id["joint_persistence_exchange_multik_chi4_protocol"]["joint_stokes_einstein_growth_over_poisson"]),
            2.0,
        )
        self.assertLess(
            float(by_id["joint_persistence_exchange_multik_chi4_protocol"]["joint_multik_tau_alpha_abs_log_residual"]),
            0.02,
        )
        self.assertGreater(
            float(by_id["joint_persistence_exchange_multik_chi4_protocol"]["rejected_mismatch_abs_log_residual"]),
            0.1,
        )
        self.assertEqual(float(by_id["kob_andersen_van_hove_tail_recovery"]["overall_consistent"]), 1.0)
        self.assertGreater(float(by_id["kob_andersen_van_hove_tail_recovery"]["peak_tail_ratio"]), 1.5)
        self.assertLess(float(by_id["kob_andersen_van_hove_tail_recovery"]["late_tail_abs_deviation"]), 0.15)
        self.assertEqual(float(by_id["angell_adam_gibbs_fragility_growth"]["overall_consistent"]), 1.0)
        self.assertGreater(float(by_id["angell_adam_gibbs_fragility_growth"]["fragility_index_growth"]), 1.5)
        self.assertEqual(
            float(by_id["angell_adam_gibbs_fragility_growth"]["fragility_scope_boundary_consistent"]),
            1.0,
        )
        self.assertEqual(float(by_id["thermodynamic_transition_scope_boundary"]["overall_consistent"]), 1.0)
        self.assertEqual(
            float(
                by_id["thermodynamic_transition_scope_boundary"][
                    "model_predicts_heat_capacity_anomaly_from_dynamics"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            float(
                by_id["thermodynamic_transition_scope_boundary"][
                    "model_predicts_kauzmann_transition_from_dynamics"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            float(by_id["thermodynamic_transition_scope_boundary"]["entropy_closure_required"]),
            1.0,
        )
        self.assertGreater(
            float(by_id["thermodynamic_transition_scope_boundary"]["thermodynamic_adam_gibbs_slowdown"]),
            10.0,
        )

    def test_literature_inversion_readiness_marks_public_benchmarks_as_not_yet_quantitative(self):
        path = ROOT / "data" / "renewal_cage_literature_inversion_readiness.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["benchmark_id"]: row for row in rows}
        self.assertIn("kob_andersen_van_hove_1995", by_id)
        self.assertIn("kob_andersen_intermediate_scattering_1995", by_id)
        self.assertIn("hedges_persistence_exchange_2007", by_id)
        self.assertGreaterEqual(float(by_id["kob_andersen_van_hove_1995"]["observable_coverage_fraction"]), 0.5)
        self.assertEqual(float(by_id["kob_andersen_van_hove_1995"]["qualitative_comparison_ready"]), 1.0)
        self.assertEqual(float(by_id["kob_andersen_van_hove_1995"]["quantitative_inversion_ready"]), 0.0)
        self.assertEqual(float(by_id["hedges_persistence_exchange_2007"]["uncertainty_weighted_ready"]), 0.0)

    def test_sota_claim_alignment_preserves_scope_boundaries(self):
        path = ROOT / "data" / "renewal_cage_sota_claim_alignment.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["claim_id"]: row for row in rows}
        self.assertEqual(by_id["hedges_persistence_exchange_decoupling"]["claim_alignment"], "supported")
        self.assertEqual(by_id["lacevic_four_point_dynamic_length"]["claim_alignment"], "partial")
        thermodynamic = by_id["kauzmann_adam_gibbs_entropy_boundary"]
        self.assertEqual(thermodynamic["claim_alignment"], "scope_boundary")
        self.assertEqual(float(thermodynamic["requires_external_closure"]), 1.0)
        self.assertEqual(float(thermodynamic["model_overclaims_source"]), 0.0)

    def test_sota_signed_constraints_audit_preserves_required_trends_and_boundaries(self):
        path = ROOT / "data" / "renewal_cage_sota_signed_constraints.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["constraint_id"]: row for row in rows}
        self.assertEqual(
            by_id["kob_andersen_van_hove_signed_constraints"]["signed_constraint_class"],
            "sota_consistent",
        )
        self.assertEqual(
            by_id["kob_andersen_van_hove_signed_constraints"]["missing_expected_signatures"],
            "none",
        )
        self.assertEqual(
            by_id["lacevic_four_point_signed_constraints"]["signed_constraint_class"],
            "closure_assisted_consistent",
        )
        self.assertEqual(
            float(by_id["lacevic_four_point_signed_constraints"]["requires_external_closure"]),
            1.0,
        )
        thermodynamic = by_id["kauzmann_thermodynamic_signed_boundary"]
        self.assertEqual(thermodynamic["signed_constraint_class"], "scope_boundary_consistent")
        self.assertEqual(thermodynamic["forbidden_claims_made"], "none")
        self.assertEqual(float(thermodynamic["publishable_alignment"]), 1.0)

    def test_sota_evidence_verdict_aggregates_claim_strength_without_overclaim(self):
        path = ROOT / "data" / "renewal_cage_sota_evidence_verdict.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["verdict_id"]: row for row in rows}
        dynamic = by_id["kob_andersen_van_hove_caging_ngp_verdict"]
        self.assertEqual(dynamic["evidence_grade"], "direct_dynamical_support")
        self.assertEqual(dynamic["allowed_manuscript_claim"], "dynamical_signature_supported")
        self.assertEqual(float(dynamic["publishable_without_overclaim"]), 1.0)

        spatial = by_id["lacevic_four_point_dynamic_length_verdict"]
        self.assertEqual(spatial["evidence_grade"], "closure_assisted_support")
        self.assertEqual(float(spatial["requires_external_closure"]), 1.0)

        thermodynamic = by_id["kauzmann_adam_gibbs_entropy_boundary_verdict"]
        self.assertEqual(thermodynamic["evidence_grade"], "thermodynamic_scope_boundary")
        self.assertEqual(thermodynamic["allowed_manuscript_claim"], "scope_boundary_only")

        pending = by_id["glassbench_reanalysis_state_verdict"]
        self.assertEqual(pending["evidence_grade"], "pending_trajectory_reanalysis")
        self.assertEqual(float(pending["trajectory_reanalysis_required"]), 1.0)
        self.assertEqual(float(pending["publishable_without_overclaim"]), 0.0)

    def test_sota_evidence_class_gate_separates_experiments_from_quantitative_inversion(self):
        path = ROOT / "data" / "renewal_cage_sota_evidence_class.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["class_id"]: row for row in rows}
        experiment = by_id["near_tg_experimental_heterogeneity_class"]
        self.assertEqual(experiment["source_modality"], "experiment")
        self.assertEqual(experiment["evidence_class"], "closure_assisted_experimental_constraint")
        self.assertEqual(float(experiment["quantitative_inversion_allowed"]), 0.0)
        self.assertEqual(float(experiment["trend_comparison_allowed"]), 1.0)
        self.assertEqual(experiment["primary_blocker"], "particle_trajectories")

        glassbench = by_id["glassbench_repository_reanalysis_class"]
        self.assertEqual(glassbench["evidence_class"], "metadata_reanalysis_candidate")
        self.assertEqual(glassbench["primary_blocker"], "physical_time")
        self.assertEqual(float(glassbench["quantitative_inversion_allowed"]), 0.0)

        canary = by_id["synthetic_member_ensemble_canary_class"]
        self.assertEqual(canary["evidence_class"], "uncertainty_weighted_quantitative_test")
        self.assertEqual(float(canary["quantitative_inversion_allowed"]), 1.0)

        thermodynamic = by_id["kauzmann_adam_gibbs_thermodynamic_class"]
        self.assertEqual(thermodynamic["evidence_class"], "thermodynamic_scope_boundary")
        self.assertEqual(thermodynamic["primary_blocker"], "renewal_dynamics_not_thermodynamic_theory")

    def test_simultaneous_closure_gate_requires_heldout_dynamical_predictions(self):
        path = ROOT / "data" / "renewal_cage_simultaneous_closure.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["protocol_id"]: row for row in rows}
        passed = by_id["synthetic_minimal_dynamical_closure"]
        self.assertEqual(
            passed["closure_stage"],
            "simultaneous_dynamical_signature_closure_passed",
        )
        self.assertEqual(float(passed["simultaneous_closure_ready"]), 1.0)
        self.assertEqual(float(passed["all_required_dynamical_predictions_pass"]), 1.0)
        self.assertEqual(float(passed["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(float(passed["heldout_count"]), 4.0)

        chi4_mismatch = by_id["synthetic_chi4_mismatch_closure"]
        self.assertEqual(chi4_mismatch["closure_stage"], "dynamical_heldout_prediction_failed")
        self.assertEqual(chi4_mismatch["primary_blocker"], "chi4_peak_z_consistent")

    def test_real_benchmark_assimilation_gate_marks_fit_readiness_and_blockers(self):
        path = ROOT / "data" / "renewal_cage_real_benchmark_assimilation_gate.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["benchmark_id"]: row for row in rows}
        structural = by_id["kob_andersen_structural_digitization_candidate"]
        self.assertEqual(structural["assimilation_stage"], "structural_digitization_ready")
        self.assertEqual(float(structural["structural_inversion_ready"]), 1.0)
        self.assertEqual(float(structural["uncertainty_weighted_ready"]), 0.0)
        self.assertEqual(structural["primary_blocker"], "uncertainty_columns")

        hedges = by_id["hedges_persistence_exchange_published_curves"]
        self.assertEqual(hedges["assimilation_stage"], "qualitative_alignment_only")
        self.assertIn("late_ngp", hedges["missing_observables"])

        thermo = by_id["kauzmann_adam_gibbs_entropy_boundary"]
        self.assertEqual(thermo["assimilation_stage"], "scope_boundary_only")
        self.assertEqual(thermo["primary_blocker"], "renewal_dynamics_not_thermodynamic_theory")

    def test_cross_observable_prediction_ledger_separates_fit_inputs_from_predictions(self):
        path = ROOT / "data" / "renewal_cage_cross_observable_prediction_ledger.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["protocol_id"]: row for row in rows}
        joint = by_id["joint_persistence_exchange_multik_chi4"]
        self.assertEqual(joint["prediction_class"], "predictive_diagnostic")
        self.assertIn("diffusion", joint["calibration_observables"])
        self.assertIn("late_ngp", joint["heldout_predictions"])
        self.assertEqual(float(joint["fit_only_overclaim_risk"]), 0.0)

        alpha_only = by_id["single_alpha_fit_only_null"]
        self.assertEqual(alpha_only["prediction_class"], "underconstrained_fit")
        self.assertEqual(float(alpha_only["fit_only_overclaim_risk"]), 1.0)

        spatial = by_id["spatial_chi4_front_closure"]
        self.assertEqual(spatial["prediction_class"], "closure_assisted_prediction")
        self.assertEqual(float(spatial["requires_external_closure"]), 1.0)

        thermo = by_id["thermodynamic_entropy_boundary"]
        self.assertEqual(thermo["prediction_class"], "scope_boundary")

    def test_inversion_identifiability_audit_marks_protocols_before_real_fit(self):
        path = ROOT / "data" / "renewal_cage_inversion_identifiability_audit.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["protocol_id"]: row for row in rows}
        joint = by_id["joint_persistence_exchange_multik_chi4"]
        self.assertEqual(joint["identifiability_class"], "identifiable_prediction")
        self.assertEqual(float(joint["rank_margin"]), 0.0)
        self.assertIn("late_ngp", joint["heldout_predictions"])
        self.assertEqual(float(joint["overclaim_risk"]), 0.0)

        alpha_only = by_id["single_alpha_fit_only_null"]
        self.assertEqual(alpha_only["identifiability_class"], "underidentified_fit")
        self.assertEqual(float(alpha_only["overclaim_risk"]), 1.0)

        spatial = by_id["spatial_chi4_front_closure"]
        self.assertEqual(spatial["identifiability_class"], "conditionally_identifiable")
        self.assertEqual(float(spatial["requires_external_closure"]), 1.0)

        thermo = by_id["thermodynamic_entropy_boundary"]
        self.assertEqual(thermo["identifiability_class"], "scope_boundary")

    def test_frontier_benchmark_horizon_marks_next_sota_reanalysis_targets(self):
        path = ROOT / "data" / "renewal_cage_frontier_benchmark_horizon.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["benchmark_id"]: row for row in rows}
        glassbench = by_id["glassbench_trajectory_horizon"]
        self.assertEqual(glassbench["horizon_class"], "trajectory_reanalysis_candidate")
        self.assertEqual(float(glassbench["can_compute_missing_from_trajectories"]), 1.0)
        self.assertIn("self_intermediate_scattering", glassbench["computable_missing_observables"])

        gst = by_id["gst_nn_potential_transport_horizon"]
        self.assertEqual(gst["horizon_class"], "transport_heterogeneity_candidate")
        self.assertEqual(gst["primary_blocker"], "late_ngp")

        molecular_motion = by_id["near_tg_molecular_motion_rotational_gap"]
        self.assertEqual(molecular_motion["horizon_class"], "structural_inversion_candidate")
        self.assertEqual(molecular_motion["primary_blocker"], "uncertainty_estimates")
        self.assertEqual(float(molecular_motion["model_extension_required"]), 0.0)

        thermo = by_id["heat_capacity_entropy_frontier"]
        self.assertEqual(thermo["horizon_class"], "scope_boundary")

    def test_sota_source_provenance_gate_marks_reanalysis_sources_without_overclaim(self):
        path = ROOT / "data" / "renewal_cage_sota_source_provenance.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["source_id"]: row for row in rows}
        glassbench = by_id["glassbench_zenodo_trajectory_release"]
        self.assertEqual(glassbench["provenance_stage"], "trajectory_reanalysis_source")
        self.assertEqual(float(glassbench["can_enter_trajectory_protocol"]), 1.0)
        self.assertEqual(float(glassbench["requires_digitization"]), 0.0)
        self.assertEqual(glassbench["primary_blocker"], "none")

        hedges = by_id["hedges_persistence_exchange_jcp_article"]
        self.assertEqual(hedges["provenance_stage"], "citation_only_source")
        self.assertEqual(float(hedges["requires_digitization"]), 1.0)
        self.assertEqual(hedges["primary_blocker"], "machine_readable_files")

        thermo = by_id["kauzmann_entropy_thermodynamic_boundary"]
        self.assertEqual(thermo["provenance_stage"], "scope_boundary_source")
        self.assertEqual(float(thermo["scope_boundary"]), 1.0)

    def test_sota_data_accession_manifest_records_remote_archive_without_claiming_cache(self):
        path = ROOT / "data" / "renewal_cage_sota_data_accession.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["accession_id"]: row for row in rows}
        glassbench = by_id["glassbench_zenodo_10118191"]
        self.assertEqual(glassbench["doi"], "10.5281/zenodo.10118191")
        self.assertEqual(glassbench["archive_name"], "GlassBench.zip")
        self.assertEqual(glassbench["archive_md5"], "82c83a7146eb749e13417e4350022417")
        self.assertEqual(glassbench["license_id"], "cc-by-4.0")
        self.assertEqual(glassbench["accession_stage"], "remote_trajectory_accession_ready")
        self.assertEqual(float(glassbench["accession_ready"]), 1.0)
        self.assertEqual(float(glassbench["ready_for_local_reanalysis"]), 0.0)
        self.assertEqual(glassbench["primary_blocker"], "local_cache")
        self.assertGreater(float(glassbench["archive_size_gb"]), 5.0)

        hedges = by_id["hedges_jcp_article_no_archive"]
        self.assertEqual(hedges["accession_stage"], "citation_only_no_accession")
        self.assertEqual(hedges["primary_blocker"], "downloadable_archive")

        thermo = by_id["kauzmann_entropy_scope_boundary"]
        self.assertEqual(thermo["accession_stage"], "scope_boundary_accession")
        self.assertEqual(float(thermo["scope_boundary"]), 1.0)

    def test_sota_zenodo_record_fingerprint_verifies_cached_remote_record(self):
        record_path = ROOT / "data" / "third_party" / "glassbench" / "zenodo_record_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_zenodo_record_fingerprint.csv"
        self.assertTrue(record_path.exists())
        self.assertTrue(path.exists())
        record = json.loads(record_path.read_text())
        self.assertEqual(record["doi"], "10.5281/zenodo.10118191")
        file_by_key = {entry["key"]: entry for entry in record["files"]}
        self.assertEqual(file_by_key["GlassBench.zip"]["size"], 6042260027)
        self.assertEqual(file_by_key["README"]["checksum"], "md5:f1a192f54a2fa7a2b3533af0011b80dc")

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["fingerprint_id"]: row for row in rows}
        glassbench = by_id["glassbench_zenodo_record_fingerprint"]
        self.assertEqual(glassbench["fingerprint_stage"], "zenodo_record_verified")
        self.assertEqual(float(glassbench["zenodo_record_fingerprint_ready"]), 1.0)
        self.assertEqual(float(glassbench["archive_md5_matches"]), 1.0)
        self.assertEqual(float(glassbench["readme_md5_matches"]), 1.0)
        self.assertEqual(float(glassbench["full_archive_download_required"]), 1.0)
        self.assertEqual(float(glassbench["real_reanalysis_ready"]), 0.0)
        self.assertEqual(glassbench["primary_blocker"], "archive_cache")

    def test_sota_archive_preflight_requires_policy_before_full_glassbench_download(self):
        path = ROOT / "data" / "renewal_cage_sota_archive_preflight.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["preflight_id"]: row for row in rows}
        glassbench = by_id["glassbench_archive_preflight"]
        self.assertEqual(glassbench["preflight_stage"], "large_archive_approval_required")
        self.assertEqual(glassbench["archive_name"], "GlassBench.zip")
        self.assertEqual(glassbench["readme_name"], "README")
        self.assertEqual(glassbench["archive_md5"], "82c83a7146eb749e13417e4350022417")
        self.assertEqual(glassbench["readme_md5"], "f1a192f54a2fa7a2b3533af0011b80dc")
        self.assertEqual(float(glassbench["ready_for_readme_schema_cache"]), 1.0)
        self.assertEqual(float(glassbench["ready_for_local_reanalysis"]), 0.0)
        self.assertEqual(float(glassbench["large_archive"]), 1.0)
        self.assertEqual(glassbench["primary_blocker"], "large_archive_download_approval")
        self.assertIn("_trajectories", glassbench["required_schema_tokens"])

        synthetic = by_id["synthetic_archive_preflight"]
        self.assertEqual(synthetic["preflight_stage"], "local_archive_reanalysis_ready")
        self.assertEqual(float(synthetic["ready_for_local_reanalysis"]), 1.0)
        self.assertEqual(synthetic["primary_blocker"], "none")

    def test_sota_readme_digest_verifies_local_glassbench_cache(self):
        cache_path = ROOT / "data" / "third_party" / "glassbench" / "README"
        path = ROOT / "data" / "renewal_cage_sota_readme_digest.csv"
        self.assertTrue(cache_path.exists())
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["digest_id"]: row for row in rows}
        glassbench = by_id["glassbench_readme_digest"]
        self.assertEqual(glassbench["digest_stage"], "readme_digest_verified")
        self.assertEqual(glassbench["observed_md5"], "f1a192f54a2fa7a2b3533af0011b80dc")
        self.assertEqual(glassbench["expected_md5"], "f1a192f54a2fa7a2b3533af0011b80dc")
        self.assertEqual(float(glassbench["observed_size_bytes"]), 2147.0)
        self.assertEqual(float(glassbench["readme_digest_ready"]), 1.0)
        self.assertEqual(float(glassbench["schema_token_coverage"]), 1.0)
        self.assertEqual(float(glassbench["citation_coverage"]), 1.0)
        self.assertEqual(float(glassbench["license_phrase_present"]), 1.0)
        self.assertIn("_trajectories", glassbench["required_tokens"])
        self.assertIn("10.1103/PhysRevLett.130.238202", glassbench["required_citation_dois"])

    def test_sota_local_cache_verification_marks_readme_verified_archive_missing(self):
        path = ROOT / "data" / "renewal_cage_sota_local_cache_verification.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["cache_id"]: row for row in rows}
        glassbench = by_id["glassbench_local_cache"]
        self.assertEqual(glassbench["cache_stage"], "archive_cache_missing")
        self.assertEqual(float(glassbench["readme_cache_verified"]), 1.0)
        self.assertEqual(float(glassbench["archive_cache_verified"]), 0.0)
        self.assertEqual(float(glassbench["ready_for_local_reanalysis"]), 0.0)
        self.assertEqual(glassbench["observed_readme_md5"], "f1a192f54a2fa7a2b3533af0011b80dc")
        self.assertEqual(glassbench["expected_archive_md5"], "82c83a7146eb749e13417e4350022417")
        self.assertEqual(glassbench["primary_blocker"], "archive_path")

    def test_sota_zip_structure_gate_waits_for_cached_glassbench_archive(self):
        path = ROOT / "data" / "renewal_cage_sota_zip_structure.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["structure_id"]: row for row in rows}
        glassbench = by_id["glassbench_zip_structure"]
        self.assertEqual(glassbench["zip_structure_stage"], "zip_archive_missing")
        self.assertEqual(float(glassbench["zip_structure_ready"]), 0.0)
        self.assertEqual(float(glassbench["zip_present"]), 0.0)
        self.assertEqual(glassbench["primary_blocker"], "archive_path")
        self.assertIn("KA/_trajectories", glassbench["required_roots"])
        self.assertIn("KA2D/_results", glassbench["missing_roots"])

    def test_sota_remote_zip_central_directory_verifies_glassbench_structure_without_cache(self):
        manifest_path = ROOT / "data" / "third_party" / "glassbench" / "remote_zip_central_directory_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_remote_zip_central_directory.csv"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["entry_count"], 70)
        self.assertEqual(manifest["central_directory_size_bytes"], 7915)
        self.assertTrue(manifest["range_supported"])
        self.assertTrue(manifest["zip64"])
        self.assertIn("GlassBench/KA_trajectories/", manifest["entries"])
        self.assertIn("GlassBench/KA2D_results/", manifest["entries"])

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["remote_structure_id"]: row for row in rows}
        glassbench = by_id["glassbench_remote_zip_central_directory"]
        self.assertEqual(glassbench["remote_zip_structure_stage"], "remote_zip_structure_verified")
        self.assertEqual(float(glassbench["remote_zip_structure_ready"]), 1.0)
        self.assertEqual(float(glassbench["root_coverage"]), 1.0)
        self.assertEqual(float(glassbench["entry_count"]), 70.0)
        self.assertEqual(float(glassbench["full_archive_cached"]), 0.0)
        self.assertEqual(float(glassbench["real_reanalysis_ready"]), 0.0)
        self.assertEqual(glassbench["primary_blocker"], "archive_cache")

    def test_sota_glassbench_payload_index_maps_remote_system_payloads(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_payload_index.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_system = {row["system_id"]: row for row in rows}
        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["payload_stage"], "remote_payload_index_verified")
        self.assertEqual(float(ka2d["payload_index_ready"]), 1.0)
        self.assertEqual(float(ka2d["trajectory_payload_count"]), 2.0)
        self.assertEqual(float(ka2d["model_payload_count"]), 2.0)
        self.assertIn("0.23", ka2d["common_temperatures"])
        self.assertIn("0.30", ka2d["common_temperatures"])
        self.assertEqual(float(ka2d["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d["primary_blocker"], "archive_cache")

        ka = by_system["KA"]
        self.assertEqual(ka["payload_stage"], "remote_payload_missing_trajectory")
        self.assertEqual(float(ka["payload_index_ready"]), 0.0)
        self.assertEqual(float(ka["model_result_index_ready"]), 1.0)
        self.assertEqual(float(ka["trajectory_payload_count"]), 0.0)
        self.assertIn("0.44", ka["common_model_result_temperatures"])
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_payload_locator_records_file_level_targets(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_payload_locator.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_023 = by_key[("KA2D", "0.23")]
        self.assertEqual(ka2d_023["source_path"], "GlassBench/KA2D_trajectories/T0.23.tar.xz")
        self.assertEqual(ka2d_023["payload_format"], "tar.xz")
        self.assertEqual(ka2d_023["locator_stage"], "remote_trajectory_payload_located")
        self.assertEqual(float(ka2d_023["remote_payload_located"]), 1.0)
        self.assertEqual(float(ka2d_023["range_supported"]), 1.0)
        self.assertEqual(float(ka2d_023["entry_metadata_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["range_fetch_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "zip_entry_metadata")

        ka = [row for row in rows if row["system_id"] == "KA"][0]
        self.assertEqual(ka["locator_stage"], "remote_trajectory_payload_missing")
        self.assertEqual(ka["source_path"], "none")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_entry_metadata_records_remote_zip_ranges(self):
        manifest_path = ROOT / "data" / "third_party" / "glassbench" / "trajectory_entry_metadata_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_entry_metadata.csv"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source"], "remote_zip_central_directory_and_local_header_range_reads")
        self.assertEqual(manifest["central_directory_range"], "6042252014-6042259928")

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["metadata_stage"], "trajectory_entry_metadata_ready_payload_size_blocked")
        self.assertEqual(ka2d_030["source_path"], "GlassBench/KA2D_trajectories/T0.30.tar.xz")
        self.assertEqual(ka2d_030["compression_method"], "deflate")
        self.assertEqual(float(ka2d_030["entry_metadata_ready"]), 1.0)
        self.assertEqual(float(ka2d_030["local_header_verified"]), 1.0)
        self.assertEqual(float(ka2d_030["compressed_size_bytes"]), 2_980_137_961.0)
        self.assertEqual(float(ka2d_030["compressed_data_range_start"]), 464_175.0)
        self.assertEqual(float(ka2d_030["compressed_data_range_end"]), 2_980_602_135.0)
        self.assertEqual(float(ka2d_030["full_member_fetch_within_policy"]), 0.0)
        self.assertEqual(float(ka2d_030["trajectory_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "member_payload_size_policy")

        ka2d_023 = by_key[("KA2D", "0.23")]
        self.assertEqual(float(ka2d_023["compressed_size_bytes"]), 397_505_592.0)
        self.assertEqual(float(ka2d_023["full_member_fetch_within_policy"]), 0.0)

    def test_sota_glassbench_trajectory_member_stream_probe_verifies_xz_prefix(self):
        manifest_path = ROOT / "data" / "third_party" / "glassbench" / "trajectory_member_stream_probe_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_member_stream_probe.csv"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source"], "remote_zip_member_prefix_range_reads")
        self.assertEqual(manifest["probe_compressed_bytes"], 65_536)

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["probe_stage"], "trajectory_member_prefix_verified_streaming_extraction_blocked")
        self.assertEqual(ka2d_030["source_path"], "GlassBench/KA2D_trajectories/T0.30.tar.xz")
        self.assertEqual(float(ka2d_030["compressed_probe_bytes"]), 65_536.0)
        self.assertEqual(float(ka2d_030["stream_inflate_ready"]), 1.0)
        self.assertEqual(float(ka2d_030["xz_magic_verified"]), 1.0)
        self.assertEqual(float(ka2d_030["member_prefix_verified"]), 1.0)
        self.assertEqual(float(ka2d_030["trajectory_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "streaming_member_extraction_policy")
        self.assertEqual(ka2d_030["inflated_prefix_hex"], "fd377a585a000004e6d6b44602002101")

    def test_sota_glassbench_trajectory_inner_tar_header_probe_records_npz_layout(self):
        manifest_path = (
            ROOT / "data" / "third_party" / "glassbench" / "trajectory_inner_tar_header_probe_10118191.json"
        )
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.csv"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source"], "remote_zip_member_to_xz_to_tar_header_range_reads")
        self.assertEqual(manifest["compressed_probe_bytes"], 4_194_304)
        self.assertEqual(manifest["tar_probe_limit_bytes"], 1_048_576)

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["tar_probe_stage"], "trajectory_inner_tar_layout_verified_extraction_blocked")
        self.assertEqual(ka2d_030["source_path"], "GlassBench/KA2D_trajectories/T0.30.tar.xz")
        self.assertEqual(ka2d_030["root_directory"], "T0.30/")
        self.assertEqual(ka2d_030["first_npz_member"], "T0.30/train/N1290T0.30_3_tc01.npz")
        self.assertEqual(ka2d_030["split_labels_in_probe"], "train")
        self.assertEqual(float(ka2d_030["tar_magic_verified"]), 1.0)
        self.assertEqual(float(ka2d_030["npz_member_header_verified"]), 1.0)
        self.assertEqual(float(ka2d_030["trajectory_layout_ready"]), 1.0)
        self.assertEqual(float(ka2d_030["trajectory_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "streaming_npz_extraction_policy")

    def test_sota_glassbench_trajectory_npz_schema_probe_records_coordinate_arrays(self):
        manifest_path = ROOT / "data" / "third_party" / "glassbench" / "trajectory_npz_schema_probe_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_npz_schema_probe.csv"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source"], "remote_zip_member_to_first_npz_schema_probe")
        self.assertEqual(manifest["required_arrays"], ["box.npy", "types.npy", "positions.npy"])

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["schema_probe_stage"], "trajectory_npz_coordinate_schema_verified")
        self.assertEqual(ka2d_030["first_npz_member"], "T0.30/train/N1290T0.30_3_tc01.npz")
        self.assertEqual(float(ka2d_030["npz_magic_verified"]), 1.0)
        self.assertEqual(float(ka2d_030["npz_schema_ready"]), 1.0)
        self.assertEqual(float(ka2d_030["coordinate_array_ready"]), 1.0)
        self.assertEqual(float(ka2d_030["particle_count"]), 1290.0)
        self.assertEqual(float(ka2d_030["frame_count"]), 20.0)
        self.assertEqual(float(ka2d_030["spatial_dimension"]), 2.0)
        self.assertIn("positions.npy:20x1290x2", ka2d_030["array_shapes"])
        self.assertEqual(float(ka2d_030["trajectory_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "full_npz_ensemble_extraction_policy")

    def test_sota_glassbench_trajectory_first_npz_observable_smoke_records_msd_ngp(self):
        manifest_path = (
            ROOT / "data" / "third_party" / "glassbench" / "trajectory_first_npz_observable_smoke_10118191.json"
        )
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.csv"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source"], "remote_zip_member_first_npz_minimal_image_observable_smoke")
        self.assertEqual(manifest["observable_method"], "minimal_image_displacement_from_first_frame")

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["smoke_stage"], "first_npz_msd_ngp_smoke_ready_reanalysis_blocked")
        self.assertEqual(ka2d_030["observable_method"], "minimal_image_displacement_from_first_frame")
        self.assertEqual(float(ka2d_030["observable_smoke_ready"]), 1.0)
        self.assertAlmostEqual(float(ka2d_030["final_msd"]), 0.005414723094117662)
        self.assertAlmostEqual(float(ka2d_030["final_ngp_2d"]), 0.05197783140666812)
        self.assertEqual(float(ka2d_030["peak_ngp_frame_index"]), 11.0)
        self.assertAlmostEqual(float(ka2d_030["peak_ngp_2d"]), 0.17874626903381952)
        self.assertEqual(float(ka2d_030["trajectory_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "single_npz_no_time_or_uncertainty")

    def test_sota_glassbench_trajectory_first_npz_observable_curve_records_frame_rows(self):
        manifest_path = (
            ROOT / "data" / "third_party" / "glassbench" / "trajectory_first_npz_observable_curve_10118191.json"
        )
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.csv"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source"], "remote_zip_member_first_npz_minimal_image_observable_curve")
        self.assertEqual(manifest["observable_method"], "minimal_image_displacement_from_first_frame")

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"], row["frame_index"]): row for row in rows}
        peak = by_key[("KA2D", "0.30", "11.0")]
        self.assertEqual(peak["curve_stage"], "first_npz_observable_curve_ready_reanalysis_blocked")
        self.assertAlmostEqual(float(peak["msd"]), 0.005117245596)
        self.assertAlmostEqual(float(peak["ngp_2d"]), 0.178746269034)
        self.assertEqual(float(peak["real_reanalysis_ready"]), 0.0)
        self.assertEqual(peak["primary_blocker"], "single_npz_frame_index_curve")

        final = by_key[("KA2D", "0.30", "19.0")]
        self.assertAlmostEqual(float(final["msd"]), 0.005414723094)
        self.assertAlmostEqual(float(final["ngp_2d"]), 0.051977831407)

    def test_sota_glassbench_short_window_trend_canary_records_real_data_sanity_check(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_short_window_trend_canary.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["system_id"], "KA2D")
        self.assertEqual(row["cold_temperature"], "0.23")
        self.assertEqual(row["hot_temperature"], "0.30")
        self.assertEqual(row["canary_stage"], "short_window_real_data_canary_ready_inversion_blocked")
        self.assertGreater(float(row["hot_to_cold_final_msd_ratio"]), 1.1)
        self.assertEqual(float(row["short_window_msd_slowdown_pass"]), 1.0)
        self.assertEqual(float(row["positive_ngp_canary_pass"]), 1.0)
        self.assertEqual(float(row["short_window_real_data_canary_ready"]), 1.0)
        self.assertEqual(float(row["sota_inversion_ready"]), 0.0)
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(row["primary_blocker"], "physical_time_ensemble_uncertainty")

    def test_sota_glassbench_trajectory_timebase_bridge_blocks_unmapped_result_grids(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_timebase_bridge.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        cold = by_key[("KA2D", "0.23")]
        hot = by_key[("KA2D", "0.30")]
        self.assertEqual(float(cold["frame_count"]), 20.0)
        self.assertEqual(float(cold["time_point_count"]), 8.0)
        self.assertEqual(float(hot["frame_count"]), 20.0)
        self.assertEqual(float(hot["time_point_count"]), 6.0)
        for row in [cold, hot]:
            self.assertEqual(row["timebase_stage"], "trajectory_result_timebase_length_mismatch")
            self.assertEqual(row["primary_blocker"], "frame_time_point_count")
            self.assertEqual(float(row["time_grid_available"]), 1.0)
            self.assertEqual(float(row["frame_time_point_count_match"]), 0.0)
            self.assertEqual(float(row["trajectory_timebase_ready"]), 0.0)
            self.assertEqual(float(row["sota_inversion_ready"]), 0.0)
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_frame_time_mapping_audit_blocks_interpolation_only_bridge(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_frame_time_mapping_audit.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        for key, ratio in [(("KA2D", "0.23"), 19.0 / 7.0), (("KA2D", "0.30"), 19.0 / 5.0)]:
            row = by_key[key]
            self.assertEqual(row["mapping_audit_stage"], "ambiguous_frame_time_mapping")
            self.assertEqual(float(row["exact_count_match"]), 0.0)
            self.assertEqual(float(row["integer_stride_subsample_candidate"]), 0.0)
            self.assertEqual(float(row["endpoint_interpolation_candidate"]), 1.0)
            self.assertAlmostEqual(float(row["frame_to_result_stride_ratio"]), ratio)
            self.assertEqual(row["accepted_mapping_class"], "none")
            self.assertEqual(row["provisional_mapping_class"], "endpoint_interpolation_requires_metadata")
            self.assertEqual(float(row["publishable_frame_time_mapping_ready"]), 0.0)
            self.assertIn("result_time_generation_script", row["minimum_required_metadata"])
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_real_inversion_gap_ledger_keeps_claim_level_safe(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_real_inversion_gap_ledger.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        for key in [("KA2D", "0.23"), ("KA2D", "0.30")]:
            row = by_key[key]
            self.assertEqual(row["ledger_stage"], "real_data_canary_timebase_blocked")
            self.assertEqual(row["allowed_claim_level"], "short_window_coordinate_trend_only")
            self.assertEqual(row["primary_blocker"], "frame_time_point_count")
            self.assertEqual(float(row["short_window_claim_ready"]), 1.0)
            self.assertEqual(float(row["trajectory_timebase_ready"]), 0.0)
            self.assertEqual(float(row["quantitative_real_inversion_ready"]), 0.0)
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
            self.assertIn("derive_or_fetch_trajectory_frame_time_mapping", row["next_required_actions"])

    def test_sota_glassbench_real_inversion_unlock_protocol_lists_minimum_payload(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_real_inversion_unlock_protocol.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        for key, time_point_count in [(("KA2D", "0.23"), 8.0), (("KA2D", "0.30"), 6.0)]:
            row = by_key[key]
            self.assertEqual(row["unlock_stage"], "minimum_real_inversion_payload_missing")
            self.assertEqual(row["current_claim_level"], "short_window_coordinate_trend_only")
            self.assertEqual(row["post_unlock_claim_level"], "uncertainty_weighted_real_trajectory_inversion")
            self.assertEqual(float(row["minimum_unlock_ready"]), 0.0)
            self.assertEqual(float(row["frame_time_mapping_required"]), 1.0)
            self.assertEqual(float(row["frame_time_mapping_present"]), 0.0)
            self.assertEqual(float(row["observed_prefix_member_count"]), 3.0)
            self.assertEqual(float(row["required_member_count"]), 4.0)
            self.assertEqual(float(row["additional_member_count_needed"]), 1.0)
            self.assertEqual(float(row["frame_count"]), 20.0)
            self.assertEqual(float(row["time_point_count"]), time_point_count)
            self.assertIn("frame_time_mapping", row["minimum_required_payload"])
            self.assertIn("one_more_independent_npz_member", row["minimum_required_payload"])
            self.assertIn("chi4_overlap", row["minimum_required_payload"])
            self.assertIn("sigma_chi4_overlap", row["minimum_required_payload"])
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_trajectory_first_npz_inversion_readiness_records_blockers(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["readiness_stage"], "frame_index_curve_only")
        self.assertEqual(ka2d_030["primary_blocker"], "physical_time_semantics")
        self.assertEqual(float(ka2d_030["frame_count"]), 20.0)
        self.assertEqual(float(ka2d_030["member_count"]), 1.0)
        self.assertEqual(float(ka2d_030["physical_time_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["ensemble_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["sota_inversion_ready"]), 0.0)
        self.assertIn("lag_time", ka2d_030["missing_observables"])
        self.assertIn("self_intermediate_scattering_by_k", ka2d_030["missing_observables"])
        self.assertIn("sigma_msd", ka2d_030["missing_uncertainty_columns"])

        ka = by_key[("KA", "none")]
        self.assertEqual(ka["readiness_stage"], "upstream_curve_incomplete")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_observable_coverage_audit_blocks_proxy_substitution(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_observable_coverage_audit.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        for key in [("KA2D", "0.23"), ("KA2D", "0.30")]:
            row = by_key[key]
            self.assertEqual(row["observable_audit_stage"], "frame_index_msd_ngp_only")
            self.assertEqual(row["available_trajectory_observables"], "frame_index;msd;ngp_2d")
            self.assertIn("lag_time", row["missing_observables"])
            self.assertIn("self_intermediate_scattering_by_k", row["missing_observables"])
            self.assertIn("chi4_overlap", row["missing_observables"])
            self.assertEqual(float(row["proxy_observable_substitution_allowed"]), 0.0)
            self.assertEqual(float(row["observable_coverage_ready"]), 0.0)
            self.assertEqual(float(row["publishable_real_inversion_observable_set_ready"]), 0.0)
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
            self.assertEqual(row["primary_blocker"], "observable_set")
            self.assertIn("compute_multi_k_self_intermediate_scattering", row["next_required_actions"])
            self.assertIn("do_not_substitute_rhomax_or_ml_feature_curves_for_fs_chi4", row["next_required_actions"])

    def test_sota_glassbench_first_npz_structural_observable_plan_marks_extractable_observables(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_first_npz_structural_observable_plan.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        for key, npz_bytes in [(("KA2D", "0.23"), 465710.0), (("KA2D", "0.30"), 444786.0)]:
            row = by_key[key]
            self.assertEqual(row["compute_plan_stage"], "coordinate_schema_ready_positions_bytes_missing")
            self.assertEqual(float(row["coordinate_schema_ready"]), 1.0)
            self.assertEqual(float(row["raw_coordinate_bytes_cached"]), 0.0)
            self.assertEqual(float(row["computable_after_npz_extraction"]), 1.0)
            self.assertEqual(float(row["immediately_computable_from_current_cache"]), 0.0)
            self.assertEqual(float(row["npz_member_bytes"]), npz_bytes)
            self.assertIn("self_intermediate_scattering_by_k", row["implemented_observable_protocol"])
            self.assertIn("chi4_overlap", row["implemented_observable_protocol"])
            self.assertEqual(row["remaining_missing_after_structural_compute"], "lag_time")
            self.assertEqual(row["primary_blocker"], "raw_coordinate_bytes")
            self.assertIn("extract_first_npz_positions_box_types", row["next_required_actions"])
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_trajectory_npz_ensemble_horizon_records_prefix_member_gap(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(float(ka2d_030["prefix_npz_member_count"]), 3.0)
        self.assertEqual(float(ka2d_030["extracted_curve_member_count"]), 1.0)
        self.assertEqual(float(ka2d_030["member_count_gap_to_threshold"]), 1.0)
        self.assertEqual(float(ka2d_030["prefix_member_horizon_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["multi_npz_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "additional_npz_member_headers")
        self.assertEqual(ka2d_030["horizon_stage"], "prefix_member_horizon_short")

    def test_sota_glassbench_visible_member_ensemble_audit_blocks_prefix_only_members(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_visible_member_ensemble_audit.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        for key, first_member_id, split_label in [
            (("KA2D", "0.23"), "N1290T0.23_202_tc05", "test"),
            (("KA2D", "0.30"), "N1290T0.30_3_tc01", "train"),
        ]:
            row = by_key[key]
            self.assertEqual(row["first_member_id"], first_member_id)
            self.assertEqual(row["split_labels_in_probe"], split_label)
            self.assertEqual(float(row["prefix_npz_member_count"]), 3.0)
            self.assertEqual(float(row["required_member_count"]), 4.0)
            self.assertEqual(float(row["additional_member_count_needed"]), 1.0)
            self.assertEqual(float(row["first_member_id_visible"]), 1.0)
            self.assertEqual(float(row["full_member_id_list_visible"]), 0.0)
            self.assertEqual(float(row["member_count_threshold_pass"]), 0.0)
            self.assertEqual(float(row["publishable_ensemble_uncertainty_ready"]), 0.0)
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
            self.assertEqual(row["primary_blocker"], "member_count_and_full_member_list")
            self.assertEqual(row["ensemble_audit_stage"], "visible_prefix_not_publishable_ensemble")

    def test_sota_remote_result_curve_cache_records_range_cached_numeric_curves(self):
        manifest_path = ROOT / "data" / "third_party" / "glassbench" / "range_result_curve_cache_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_remote_result_curve_cache.csv"
        self.assertTrue(manifest_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["source"], "remote_zip_range_reads")
        self.assertGreaterEqual(len(manifest["entries"]), 10)
        self.assertTrue(all(entry["crc32_matches"] for entry in manifest["entries"]))

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_system = {row["system_id"]: row for row in rows}
        ka = by_system["KA"]
        self.assertEqual(ka["curve_cache_stage"], "range_result_curves_verified")
        self.assertEqual(float(ka["curve_cache_ready"]), 1.0)
        self.assertIn("time_grid", ka["available_roles"])
        self.assertIn("rhomax_md", ka["available_roles"])
        self.assertIn("0.44", ka["temperature_grid"])
        self.assertEqual(float(ka["real_inversion_ready"]), 0.0)
        self.assertEqual(ka["primary_blocker"], "raw_curve_adapter")

        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["curve_cache_stage"], "range_result_curves_verified")
        self.assertEqual(float(ka2d["curve_cache_ready"]), 1.0)
        self.assertIn("rhomax_bb", ka2d["available_roles"])
        self.assertIn("0.30", ka2d["temperature_grid"])
        self.assertEqual(ka2d["primary_blocker"], "raw_curve_adapter")

    def test_sota_remote_result_curve_fetch_gap_marks_chi4_target_before_comparison(self):
        central_directory_path = ROOT / "data" / "third_party" / "glassbench" / "remote_zip_central_directory_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_remote_result_curve_fetch_gap.csv"
        self.assertTrue(central_directory_path.exists())
        self.assertTrue(path.exists())

        central_directory = json.loads(central_directory_path.read_text())
        self.assertIn(
            "GlassBench/KA_results/chi4_KA_T0.44_update.dat",
            central_directory["entries"],
        )

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_path = {row["target_path"]: row for row in rows}
        chi4 = by_path["GlassBench/KA_results/chi4_KA_T0.44_update.dat"]
        self.assertEqual(chi4["fetch_gap_stage"], "remote_target_present_range_cache_missing")
        self.assertEqual(chi4["candidate_observable"], "dynamic_heterogeneity_chi4_proxy")
        self.assertEqual(float(chi4["central_directory_present"]), 1.0)
        self.assertEqual(float(chi4["range_cache_present"]), 0.0)
        self.assertEqual(float(chi4["targeted_fetch_ready"]), 1.0)
        self.assertEqual(float(chi4["observable_comparison_ready"]), 0.0)
        self.assertEqual(float(chi4["real_inversion_ready"]), 0.0)
        self.assertEqual(chi4["primary_blocker"], "range_result_curve_cache")

    def test_sota_remote_result_curve_target_fetch_marks_chi4_header_only_payload(self):
        target_fetch_path = ROOT / "data" / "third_party" / "glassbench" / "range_result_curve_target_fetch_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_remote_result_curve_target_fetch.csv"
        self.assertTrue(target_fetch_path.exists())
        self.assertTrue(path.exists())

        manifest = json.loads(target_fetch_path.read_text())
        by_path = {entry["path"]: entry for entry in manifest["entries"]}
        target = by_path["GlassBench/KA_results/chi4_KA_T0.44_update.dat"]
        self.assertEqual(target["md5"], "2def7c42b63e7c347b8c4747974d8323")
        self.assertEqual(target["content_preview"], "t True Shiba Alkemade Jung Francois")
        self.assertEqual(target["numeric_row_count"], 0)
        self.assertEqual(target["header"], ["t", "True", "Shiba", "Alkemade", "Jung", "Francois"])

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_target = {row["target_path"]: row for row in rows}
        chi4 = by_target["GlassBench/KA_results/chi4_KA_T0.44_update.dat"]
        self.assertEqual(chi4["target_fetch_stage"], "target_fetch_header_only_parse_blocked")
        self.assertEqual(float(chi4["target_fetch_present"]), 1.0)
        self.assertEqual(float(chi4["header_only_payload"]), 1.0)
        self.assertEqual(float(chi4["numeric_payload_ready"]), 0.0)
        self.assertEqual(float(chi4["observable_comparison_ready"]), 0.0)
        self.assertEqual(float(chi4["real_inversion_ready"]), 0.0)
        self.assertEqual(chi4["primary_blocker"], "numeric_rows")

    def test_sota_remote_result_curve_published_semantics_blocks_ml_feature_curves(self):
        path = ROOT / "data" / "renewal_cage_sota_remote_result_curve_published_semantics.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_path = {row["source_path"]: row for row in rows}
        fig3 = by_path["GlassBench/KA2D_results/FIG3.dat"]
        self.assertEqual(fig3["semantic_stage"], "published_curve_ml_benchmark_not_physical_observable")
        self.assertEqual(fig3["header_tokens"], "t;BOTAN;CAGE;GlassMLP;SE3;DEN;EPOT;PSI4;TT")
        self.assertEqual(float(fig3["time_axis_present"]), 1.0)
        self.assertEqual(float(fig3["header_semantics_ready"]), 1.0)
        self.assertEqual(float(fig3["physical_observable_label_match"]), 0.0)
        self.assertEqual(float(fig3["ml_feature_column_count"]), 8.0)
        self.assertEqual(float(fig3["observable_comparison_ready"]), 0.0)
        self.assertEqual(float(fig3["real_inversion_ready"]), 0.0)
        self.assertEqual(fig3["primary_blocker"], "physical_observable_label")

    def test_sota_remote_result_curve_payload_adapter_pairs_cached_values(self):
        payload_path = ROOT / "data" / "third_party" / "glassbench" / "range_result_curve_values_10118191.json"
        path = ROOT / "data" / "renewal_cage_sota_remote_result_curve_payload_adapter.csv"
        self.assertTrue(payload_path.exists())
        self.assertTrue(path.exists())

        payload = json.loads(payload_path.read_text())
        self.assertEqual(payload["source"], "remote_zip_range_numeric_payload_cache")
        self.assertGreaterEqual(len(payload["entries"]), 10)

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {
            (row["system_id"], row["temperature"], row["curve_role"]): row
            for row in rows
        }
        ka2d_md = by_key[("KA2D", "0.30", "rhomax_md")]
        self.assertEqual(ka2d_md["adapter_stage"], "range_curve_payload_adapter_ready")
        self.assertEqual(float(ka2d_md["structural_adapter_ready"]), 1.0)
        self.assertEqual(float(ka2d_md["time_grid_matches_value_time"]), 1.0)
        self.assertEqual(ka2d_md["available_columns"], "temperature;time;rhomax")
        self.assertEqual(float(ka2d_md["uncertainty_adapter_ready"]), 0.0)
        self.assertEqual(float(ka2d_md["real_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_md["primary_blocker"], "sigma_rhomax")

        ka2d_bb = by_key[("KA2D", "0.30", "rhomax_bb")]
        self.assertEqual(ka2d_bb["adapter_stage"], "range_curve_payload_adapter_ready")
        self.assertEqual(float(ka2d_bb["value_point_count"]), 6.0)

        ka = by_key[("KA", "0.44", "rhomax_md")]
        self.assertEqual(ka["adapter_stage"], "range_curve_payload_parse_blocked")
        self.assertEqual(float(ka["structural_adapter_ready"]), 0.0)
        self.assertEqual(ka["primary_blocker"], "numeric_rows")

    def test_sota_remote_result_curve_observable_semantics_keeps_proxy_boundary(self):
        path = ROOT / "data" / "renewal_cage_sota_remote_result_curve_observable_semantics.csv"
        self.assertTrue(path.exists())

        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {
            (row["system_id"], row["temperature"], row["curve_role"]): row
            for row in rows
        }
        ka2d_md = by_key[("KA2D", "0.30", "rhomax_md")]
        self.assertEqual(
            ka2d_md["semantics_stage"],
            "proxy_observable_ready_model_semantics_incomplete",
        )
        self.assertEqual(ka2d_md["candidate_observable"], "overlap_density_proxy")
        self.assertEqual(float(ka2d_md["proxy_observable_ready"]), 1.0)
        self.assertEqual(float(ka2d_md["diagnostic_semantics_ready"]), 0.0)
        self.assertIn("alpha_decay", ka2d_md["missing_model_semantics"])
        self.assertIn("diffusion", ka2d_md["missing_model_semantics"])
        self.assertIn("late_ngp", ka2d_md["missing_model_semantics"])
        self.assertIn("chi4_proxy", ka2d_md["missing_model_semantics"])
        self.assertEqual(float(ka2d_md["real_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_md["primary_blocker"], "model_observable_semantics")

        ka = by_key[("KA", "0.44", "rhomax_md")]
        self.assertEqual(ka["semantics_stage"], "structural_adapter_blocked")
        self.assertEqual(float(ka["proxy_observable_ready"]), 0.0)
        self.assertEqual(ka["primary_blocker"], "numeric_rows")

    def test_sota_reanalysis_state_summarizes_current_glassbench_blocker(self):
        path = ROOT / "data" / "renewal_cage_sota_reanalysis_state.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["state_id"]: row for row in rows}
        glassbench = by_id["glassbench_reanalysis_state"]
        self.assertEqual(glassbench["reanalysis_stage"], "awaiting_full_archive_cache")
        self.assertEqual(glassbench["claim_level"], "metadata_verified_not_reanalysis")
        self.assertEqual(float(glassbench["accession_ready"]), 1.0)
        self.assertEqual(float(glassbench["readme_digest_ready"]), 1.0)
        self.assertEqual(float(glassbench["local_cache_verified"]), 0.0)
        self.assertEqual(float(glassbench["ready_for_model_comparison"]), 0.0)
        self.assertEqual(glassbench["primary_blocker"], "archive_path")
        self.assertEqual(glassbench["next_action"], "cache_full_archive_and_verify_checksum")

    def test_sota_readme_schema_gate_records_remote_schema_without_archive_inspection(self):
        path = ROOT / "data" / "renewal_cage_sota_readme_schema.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["schema_id"]: row for row in rows}
        glassbench = by_id["glassbench_readme_schema"]
        self.assertEqual(glassbench["schema_stage"], "remote_readme_schema_ready")
        self.assertEqual(float(glassbench["has_ka_system"]), 1.0)
        self.assertEqual(float(glassbench["has_ka2d_system"]), 1.0)
        self.assertEqual(float(glassbench["has_trajectory_folder"]), 1.0)
        self.assertEqual(float(glassbench["has_model_folder"]), 1.0)
        self.assertEqual(float(glassbench["has_results_folder"]), 1.0)
        self.assertEqual(float(glassbench["schema_ready"]), 1.0)
        self.assertEqual(float(glassbench["ready_for_local_adapter"]), 0.0)
        self.assertEqual(glassbench["primary_blocker"], "local_archive_inspection")

        missing = by_id["hedges_schema_missing_trajectories"]
        self.assertEqual(missing["schema_stage"], "metadata_incomplete_schema")
        self.assertEqual(missing["primary_blocker"], "trajectory_folder")

    def test_trajectory_adapter_contract_keeps_glassbench_remote_until_local_archive_inspected(self):
        path = ROOT / "data" / "renewal_cage_trajectory_adapter_contract.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["contract_id"]: row for row in rows}
        ka = by_id["glassbench_ka_remote_contract"]
        self.assertEqual(ka["adapter_stage"], "remote_adapter_contract_only")
        self.assertEqual(ka["system_id"], "KA")
        self.assertEqual(float(ka["adapter_ready"]), 0.0)
        self.assertEqual(float(ka["local_archive_inspected"]), 0.0)
        self.assertEqual(ka["primary_blocker"], "coordinate_file")
        self.assertIn("coordinate_file", ka["missing_local_fields"])

        ka2d = by_id["glassbench_ka2d_remote_contract"]
        self.assertEqual(ka2d["adapter_stage"], "remote_adapter_contract_only")
        self.assertEqual(ka2d["system_id"], "KA2D")
        self.assertEqual(float(ka2d["adapter_ready"]), 0.0)
        self.assertEqual(ka2d["primary_blocker"], "coordinate_file")

        synthetic = by_id["synthetic_local_trajectory_adapter"]
        self.assertEqual(synthetic["adapter_stage"], "local_trajectory_adapter_ready")
        self.assertEqual(float(synthetic["adapter_ready"]), 1.0)
        self.assertEqual(float(synthetic["local_archive_inspected"]), 1.0)
        self.assertEqual(synthetic["primary_blocker"], "none")

    def test_observable_falsification_matrix_maps_literature_to_diagnostic_blockers(self):
        path = ROOT / "data" / "renewal_cage_observable_falsification_matrix.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["benchmark_id"], row["diagnostic_id"]): row for row in rows}
        self.assertIn(("kob_andersen_intermediate_scattering_1995", "multi_k_alpha_shape"), by_key)
        self.assertIn(("hedges_persistence_exchange_2007", "joint_persistence_exchange_chi4"), by_key)
        self.assertEqual(
            float(
                by_key[("kob_andersen_intermediate_scattering_1995", "multi_k_alpha_shape")][
                    "structural_falsification_ready"
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(
                by_key[("kob_andersen_intermediate_scattering_1995", "multi_k_alpha_shape")][
                    "quantitative_falsification_ready"
                ]
            ),
            0.0,
        )
        self.assertEqual(
            by_key[("hedges_persistence_exchange_2007", "joint_persistence_exchange_chi4")][
                "missing_observables"
            ],
            "diffusion;late_ngp;chi4_peak",
        )
        self.assertEqual(
            by_key[("hedges_persistence_exchange_2007", "joint_persistence_exchange_chi4")]["primary_blocker"],
            "diffusion",
        )

    def test_benchmark_fusion_readiness_keeps_cross_paper_splicing_honest(self):
        path = ROOT / "data" / "renewal_cage_benchmark_fusion_readiness.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["fusion_id"]: row for row in rows}
        self.assertIn("kob_andersen_i_ii_dynamic_closure", by_id)
        self.assertIn("ka_lacevic_four_point_splice", by_id)
        self.assertEqual(float(by_id["kob_andersen_i_ii_dynamic_closure"]["structural_fusion_ready"]), 1.0)
        self.assertEqual(float(by_id["kob_andersen_i_ii_dynamic_closure"]["quantitative_fusion_ready"]), 0.0)
        self.assertEqual(by_id["kob_andersen_i_ii_dynamic_closure"]["primary_blocker"], "machine_readable_data")
        self.assertEqual(float(by_id["ka_lacevic_four_point_splice"]["shared_system_consistent"]), 1.0)
        self.assertEqual(float(by_id["ka_lacevic_four_point_splice"]["shared_temperature_grid_consistent"]), 0.0)
        self.assertEqual(float(by_id["ka_lacevic_four_point_splice"]["shared_ensemble_consistent"]), 0.0)
        self.assertEqual(float(by_id["ka_lacevic_four_point_splice"]["structural_fusion_ready"]), 0.0)
        self.assertEqual(by_id["ka_lacevic_four_point_splice"]["primary_blocker"], "temperature_grid_mismatch")

    def test_raw_curve_ingestion_contract_requires_uncertainty_columns_for_real_fit(self):
        path = ROOT / "data" / "renewal_cage_raw_curve_ingestion_contract.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_observable = {row["observable_id"]: row for row in rows}
        self.assertIn("ka_self_intermediate_scattering", by_observable)
        self.assertIn("ka_van_hove_ngp", by_observable)
        self.assertEqual(
            float(by_observable["ka_self_intermediate_scattering"]["structural_ingestion_ready"]),
            1.0,
        )
        self.assertEqual(
            float(by_observable["ka_self_intermediate_scattering"]["uncertainty_ingestion_ready"]),
            0.0,
        )
        self.assertEqual(
            by_observable["ka_self_intermediate_scattering"]["missing_uncertainty_columns"],
            "sigma_F_s",
        )
        self.assertEqual(
            float(by_observable["ka_van_hove_ngp"]["uncertainty_ingestion_ready"]),
            0.0,
        )
        self.assertEqual(by_observable["ka_van_hove_ngp"]["primary_blocker"], "sigma_G_s")

    def test_raw_curve_diagnostic_readiness_requires_uncertainty_before_real_inversion(self):
        path = ROOT / "data" / "renewal_cage_raw_curve_diagnostic_readiness.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["diagnostic_id"]: row for row in rows}
        self.assertIn("multi_k_alpha_shape", by_id)
        self.assertIn("van_hove_gaussian_recovery", by_id)
        self.assertIn("combined_alpha_vanhove_transport_closure", by_id)
        self.assertEqual(float(by_id["multi_k_alpha_shape"]["structural_diagnostic_ready"]), 1.0)
        self.assertEqual(float(by_id["multi_k_alpha_shape"]["uncertainty_diagnostic_ready"]), 0.0)
        self.assertEqual(by_id["multi_k_alpha_shape"]["primary_blocker"], "sigma_F_s")
        self.assertEqual(float(by_id["van_hove_gaussian_recovery"]["structural_diagnostic_ready"]), 1.0)
        self.assertEqual(float(by_id["van_hove_gaussian_recovery"]["uncertainty_diagnostic_ready"]), 0.0)
        self.assertEqual(by_id["van_hove_gaussian_recovery"]["primary_blocker"], "sigma_G_s")
        self.assertEqual(
            float(by_id["combined_alpha_vanhove_transport_closure"]["uncertainty_diagnostic_ready"]),
            0.0,
        )

    def test_raw_curve_persistence_exchange_protocol_has_pass_and_rejection_cases(self):
        path = ROOT / "data" / "renewal_cage_raw_curve_persistence_exchange_protocol.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_scenario = {row["scenario"]: row for row in rows}
        self.assertEqual(float(by_scenario["consistent_raw_curves"]["raw_curve_protocol_passes"]), 1.0)
        self.assertEqual(float(by_scenario["late_ngp_mismatch"]["raw_curve_protocol_passes"]), 0.0)
        self.assertGreater(
            float(by_scenario["consistent_raw_curves"]["persistence_exchange_ratio"]),
            6.0,
        )
        self.assertGreater(float(by_scenario["late_ngp_mismatch"]["late_ngp_z"]), 3.0)

    def test_trajectory_observable_protocol_exports_raw_trajectory_bridge(self):
        path = ROOT / "data" / "renewal_cage_trajectory_observable_protocol.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        self.assertGreaterEqual(len(rows), 4)
        peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
        self.assertEqual(peak["structural_observable_set"], "msd;ngp;self_intermediate_scattering;overlap_chi4")
        self.assertGreater(float(peak["chi4_overlap"]), 0.1)
        self.assertGreater(float(peak["ngp"]), 0.0)
        self.assertIn("1.1", peak["wave_numbers"])

    def test_trajectory_adapter_demo_exports_observables_from_local_table(self):
        path = ROOT / "data" / "renewal_cage_trajectory_adapter_demo.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        self.assertGreaterEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["adapter_source"], "synthetic_local_particle_table")
            self.assertEqual(float(row["adapter_ready"]), 1.0)
            self.assertEqual(float(row["frame_count"]), 9.0)
            self.assertEqual(float(row["particle_count"]), 12.0)
            self.assertEqual(float(row["dimension"]), 1.0)
            self.assertIn("msd", row["structural_observable_set"])
        peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
        self.assertGreater(float(peak["chi4_overlap"]), 0.1)
        self.assertGreater(float(peak["ngp"]), 0.0)

    def test_trajectory_csv_adapter_demo_gates_file_metadata_and_exports_observables(self):
        path = ROOT / "data" / "renewal_cage_trajectory_csv_adapter_demo.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        self.assertGreaterEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row["adapter_source"], "synthetic_local_csv_file")
            self.assertEqual(row["adapter_stage"], "local_csv_trajectory_ready")
            self.assertEqual(float(row["adapter_ready"]), 1.0)
            self.assertEqual(row["missing_metadata_fields"], "none")
            self.assertEqual(row["units_metadata"], "reduced_LJ_units")
            self.assertEqual(float(row["row_count"]), 108.0)
            self.assertIn("frame", row["csv_columns"])
            self.assertIn("particle_id", row["csv_columns"])
        peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
        self.assertGreater(float(peak["chi4_overlap"]), 0.1)

    def test_trajectory_curve_bridge_summarizes_csv_observables_for_inversion_gate(self):
        path = ROOT / "data" / "renewal_cage_trajectory_curve_bridge.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["benchmark_id"]: row for row in rows}
        ready = by_id["synthetic_alpha_crossing_csv_bridge"]
        self.assertEqual(ready["bridge_stage"], "trajectory_curve_bridge_ready")
        self.assertEqual(float(ready["curve_bridge_ready"]), 1.0)
        self.assertEqual(ready["primary_blocker"], "none")
        self.assertGreater(float(ready["diffusion_coefficient"]), 0.0)
        self.assertGreater(float(ready["d_tau_alpha_product"]), 0.0)
        self.assertIn("1.1:", ready["tau_alpha_by_k"])

        short = by_id["synthetic_short_csv_bridge"]
        self.assertEqual(short["bridge_stage"], "trajectory_curve_bridge_incomplete")
        self.assertEqual(float(short["curve_bridge_ready"]), 0.0)
        self.assertEqual(short["primary_blocker"], "alpha_threshold_crossing")

    def test_trajectory_curve_pe_gate_runs_joint_protocol_after_bridge(self):
        path = ROOT / "data" / "renewal_cage_trajectory_curve_pe_gate.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["benchmark_id"]: row for row in rows}
        ready = by_id["synthetic_bridge_pe_protocol_ready"]
        self.assertEqual(ready["gate_stage"], "trajectory_persistence_exchange_protocol_ready")
        self.assertEqual(float(ready["trajectory_pe_protocol_ready"]), 1.0)
        self.assertEqual(float(ready["passes_uncertainty_protocol"]), 1.0)
        self.assertGreater(float(ready["persistence_exchange_ratio"]), 6.0)
        self.assertLess(float(ready["max_multik_tau_alpha_z"]), 1.0)

        blocked = by_id["synthetic_short_csv_bridge"]
        self.assertEqual(blocked["gate_stage"], "trajectory_curve_bridge_incomplete")
        self.assertEqual(float(blocked["trajectory_pe_protocol_ready"]), 0.0)
        self.assertEqual(blocked["primary_blocker"], "alpha_threshold_crossing")

    def test_trajectory_pe_heldout_predictions_score_unfitted_observables(self):
        path = ROOT / "data" / "renewal_cage_trajectory_pe_heldout_predictions.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["benchmark_id"]: row for row in rows}
        ready = by_id["synthetic_bridge_pe_protocol_ready"]
        self.assertEqual(ready["prediction_stage"], "trajectory_pe_heldout_prediction_ready")
        self.assertEqual(float(ready["heldout_prediction_ready"]), 1.0)
        self.assertEqual(float(ready["heldout_predictions_pass"]), 1.0)
        self.assertLess(float(ready["heldout_tau_alpha_z"]), 1.0)
        self.assertLess(float(ready["heldout_late_ngp_z"]), 1.0)

        blocked = by_id["synthetic_short_csv_bridge"]
        self.assertEqual(blocked["prediction_stage"], "trajectory_pe_gate_incomplete")
        self.assertEqual(float(blocked["heldout_prediction_ready"]), 0.0)
        self.assertEqual(blocked["primary_blocker"], "alpha_threshold_crossing")

    def test_trajectory_prediction_falsification_gate_separates_fit_from_heldout(self):
        path = ROOT / "data" / "renewal_cage_trajectory_prediction_falsification.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["protocol_id"]: row for row in rows}
        passed = by_id["synthetic_trajectory_pe_heldout_protocol"]
        self.assertEqual(
            passed["falsification_stage"],
            "trajectory_prediction_falsification_passed",
        )
        self.assertEqual(float(passed["trajectory_falsification_ready"]), 1.0)
        self.assertEqual(float(passed["trajectory_predictions_falsified"]), 0.0)
        self.assertEqual(float(passed["fit_only_overclaim_risk"]), 0.0)
        self.assertGreaterEqual(float(passed["heldout_count"]), 2.0)
        self.assertIn("tau_alpha(k=1.35)", passed["heldout_observables"])

        upstream = by_id["short_trajectory_upstream_blocker"]
        self.assertEqual(upstream["falsification_stage"], "upstream_prediction_incomplete")
        self.assertEqual(float(upstream["trajectory_falsification_ready"]), 0.0)
        self.assertEqual(upstream["primary_blocker"], "alpha_threshold_crossing")

        fit_only = by_id["fit_only_negative_control"]
        self.assertEqual(fit_only["falsification_stage"], "fit_only_overclaim_risk")
        self.assertEqual(float(fit_only["fit_only_overclaim_risk"]), 1.0)
        self.assertEqual(float(fit_only["heldout_count"]), 0.0)
        self.assertEqual(fit_only["primary_blocker"], "heldout_observables")

    def test_benchmark_publication_ladder_separates_real_reanalysis_from_protocol_canary(self):
        path = ROOT / "data" / "renewal_cage_benchmark_publication_ladder.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["ladder_id"]: row for row in rows}
        glassbench = by_id["glassbench_current_publication_state"]
        self.assertEqual(glassbench["publication_stage"], "metadata_verified_not_reanalysis")
        self.assertEqual(glassbench["allowed_manuscript_claim"], "metadata_readiness_only")
        self.assertEqual(float(glassbench["real_data_quantitative_comparison"]), 0.0)
        self.assertEqual(float(glassbench["claim_overreach_if_called_fit"]), 1.0)

        canary = by_id["synthetic_trajectory_canary"]
        self.assertEqual(canary["publication_stage"], "synthetic_prediction_canary_passed")
        self.assertEqual(canary["allowed_manuscript_claim"], "protocol_canary_passed")
        self.assertEqual(float(canary["publishable_protocol_evidence"]), 1.0)
        self.assertEqual(float(canary["real_data_quantitative_comparison"]), 0.0)

        fit_only = by_id["fit_only_negative_control_publication_state"]
        self.assertEqual(fit_only["publication_stage"], "fit_only_overclaim_blocked")
        self.assertEqual(fit_only["allowed_manuscript_claim"], "do_not_claim_prediction")
        self.assertEqual(float(fit_only["claim_overreach_if_called_fit"]), 1.0)

    def test_trajectory_uncertainty_protocol_exports_jackknife_sigmas(self):
        path = ROOT / "data" / "renewal_cage_trajectory_uncertainty_protocol.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        self.assertGreaterEqual(len(rows), 4)
        peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
        self.assertEqual(peak["uncertainty_method"], "time_origin_block_jackknife")
        self.assertEqual(float(peak["uncertainty_estimates"]), 1.0)
        self.assertEqual(peak["primary_blocker"], "none")
        self.assertGreater(float(peak["sigma_msd"]), 0.0)
        self.assertGreater(float(peak["sigma_self_intermediate_scattering"]), 0.0)
        self.assertGreaterEqual(float(peak["sigma_chi4_overlap"]), 0.0)

    def test_trajectory_member_ensemble_uncertainty_exports_member_sigmas(self):
        path = ROOT / "data" / "renewal_cage_trajectory_member_ensemble_uncertainty.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 3)
        peak = max(rows, key=lambda row: float(row["chi4_overlap"]))
        self.assertEqual(peak["uncertainty_method"], "member_ensemble_standard_error")
        self.assertEqual(float(peak["member_count"]), 4.0)
        self.assertEqual(float(peak["ensemble_uncertainty_ready"]), 1.0)
        self.assertEqual(peak["primary_blocker"], "none")
        self.assertGreater(float(peak["sigma_msd"]), 0.0)
        self.assertGreater(float(peak["sigma_self_intermediate_scattering"]), 0.0)
        self.assertGreater(float(peak["sigma_chi4_overlap"]), 0.0)

    def test_trajectory_inversion_readiness_gate_promotes_uncertainty_weighted_trajectory_rows(self):
        path = ROOT / "data" / "renewal_cage_trajectory_inversion_readiness.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["benchmark_id"]: row for row in rows}
        ready = by_id["synthetic_intermittent_trajectory_uncertainty"]
        self.assertEqual(ready["readiness_stage"], "uncertainty_weighted_trajectory_inversion")
        self.assertEqual(ready["primary_blocker"], "none")
        self.assertEqual(float(ready["uncertainty_weighted_ready"]), 1.0)
        structural = by_id["synthetic_intermittent_trajectory_structural_only"]
        self.assertEqual(structural["readiness_stage"], "structural_trajectory_only")
        self.assertEqual(structural["primary_blocker"], "sigma_msd")
        member_ensemble = by_id["synthetic_member_ensemble_trajectory_uncertainty"]
        self.assertEqual(member_ensemble["readiness_stage"], "uncertainty_weighted_trajectory_inversion")
        self.assertEqual(float(member_ensemble["uncertainty_weighted_ready"]), 1.0)

    def test_translation_rotation_protocol_detects_rotational_decoupling_gap(self):
        path = ROOT / "data" / "renewal_cage_translation_rotation_protocol.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_scenario = {row["scenario"]: row for row in rows}
        coupled = by_scenario["coupled_clock"]
        rotational = by_scenario["rotationally_slow_clock"]
        self.assertEqual(float(coupled["translation_rotation_decoupling_detected"]), 0.0)
        self.assertEqual(float(rotational["translation_rotation_decoupling_detected"]), 1.0)
        self.assertGreater(float(rotational["rotational_to_translational_persistence_ratio"]), 2.0)
        self.assertGreater(
            float(rotational["rotational_dse_product"]),
            float(coupled["rotational_dse_product"]),
        )

    def test_build_arxiv_package_creates_source_zip_with_pdf_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = build_arxiv_package(output_dir=Path(tmpdir))
            self.assertTrue(zip_path.exists())
            self.assertGreater(zip_path.stat().st_size, 10_000)

            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())

            self.assertIn("main.tex", names)
            self.assertIn("references.bib", names)
            self.assertIn("figures/renewal_cage_results.pdf", names)
            self.assertIn("figures/renewal_cage_dimensionless.pdf", names)
            self.assertIn("figures/renewal_cage_scattering.pdf", names)
            self.assertIn("figures/renewal_cage_temperature.pdf", names)
            self.assertIn("figures/renewal_cage_barrier.pdf", names)
            self.assertIn("figures/renewal_cage_heterogeneity.pdf", names)
            self.assertIn("figures/renewal_cage_heterogeneity_map.pdf", names)
            self.assertIn("figures/renewal_cage_static_null.pdf", names)
            self.assertIn("figures/renewal_cage_alpha_shape.pdf", names)
            self.assertIn("figures/renewal_cage_facilitated_exchange.pdf", names)
            self.assertIn("figures/renewal_cage_glass_audit.pdf", names)
            self.assertIn("figures/renewal_cage_glass_phase_diagram.pdf", names)
            self.assertIn("figures/renewal_cage_spatial_chi4.pdf", names)
            self.assertIn("figures/renewal_cage_thermodynamic_closure.pdf", names)
            self.assertIn("figures/renewal_cage_mct_beta_closure.pdf", names)
            self.assertIn("figures/renewal_cage_sota_benchmark_consistency.pdf", names)
            self.assertIn("figures/renewal_cage_sota_claim_alignment.pdf", names)
            self.assertIn("figures/renewal_cage_sota_signed_constraints.pdf", names)
            self.assertIn("figures/renewal_cage_sota_evidence_verdict.pdf", names)
            self.assertIn("figures/renewal_cage_sota_evidence_class.pdf", names)
            self.assertIn("figures/renewal_cage_real_benchmark_assimilation_gate.pdf", names)
            self.assertIn("figures/renewal_cage_cross_observable_prediction_ledger.pdf", names)
            self.assertIn("figures/renewal_cage_inversion_identifiability_audit.pdf", names)
            self.assertIn("figures/renewal_cage_frontier_benchmark_horizon.pdf", names)
            self.assertIn("figures/renewal_cage_sota_source_provenance.pdf", names)
            self.assertIn("figures/renewal_cage_sota_data_accession.pdf", names)
            self.assertIn("figures/renewal_cage_sota_zenodo_record_fingerprint.pdf", names)
            self.assertIn("figures/renewal_cage_sota_remote_zip_central_directory.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_payload_index.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_payload_locator.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_entry_metadata.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_member_stream_probe.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_npz_schema_probe.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_short_window_trend_canary.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_timebase_bridge.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_frame_time_mapping_audit.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_real_inversion_gap_ledger.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_real_inversion_unlock_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_visible_member_ensemble_audit.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_observable_coverage_audit.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_first_npz_structural_observable_plan.pdf", names)
            self.assertIn("figures/renewal_cage_sota_remote_result_curve_cache.pdf", names)
            self.assertIn("figures/renewal_cage_sota_remote_result_curve_fetch_gap.pdf", names)
            self.assertIn("figures/renewal_cage_sota_remote_result_curve_target_fetch.pdf", names)
            self.assertIn("figures/renewal_cage_sota_remote_result_curve_published_semantics.pdf", names)
            self.assertIn("figures/renewal_cage_sota_remote_result_curve_payload_adapter.pdf", names)
            self.assertIn("figures/renewal_cage_sota_remote_result_curve_observable_semantics.pdf", names)
            self.assertIn("figures/renewal_cage_sota_readme_schema.pdf", names)
            self.assertIn("figures/renewal_cage_trajectory_adapter_contract.pdf", names)
            self.assertIn("figures/renewal_cage_literature_inversion_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_observable_falsification_matrix.pdf", names)
            self.assertIn("figures/renewal_cage_benchmark_fusion_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_raw_curve_ingestion_contract.pdf", names)
            self.assertIn("figures/renewal_cage_raw_curve_diagnostic_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_raw_curve_persistence_exchange_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_trajectory_observable_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_trajectory_uncertainty_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_trajectory_member_ensemble_uncertainty.pdf", names)
            self.assertIn("figures/renewal_cage_trajectory_inversion_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_benchmark_publication_ladder.pdf", names)
            self.assertIn("figures/renewal_cage_barrier_requirements.pdf", names)
            self.assertIn("figures/renewal_cage_mechanism_selection.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_joint_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_uncertainty_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_translation_rotation_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_simultaneous_closure.pdf", names)
            self.assertIn("figures/renewal_cage_inversion.pdf", names)

    def test_main_tex_uses_arxiv_safe_pdf_figures(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()

        self.assertIn("figures/renewal_cage_results.pdf", main_tex)
        self.assertIn("figures/renewal_cage_dimensionless.pdf", main_tex)
        self.assertIn("figures/renewal_cage_scattering.pdf", main_tex)
        self.assertIn("figures/renewal_cage_temperature.pdf", main_tex)
        self.assertIn("figures/renewal_cage_alpha_shape.pdf", main_tex)
        self.assertIn("figures/renewal_cage_facilitated_exchange.pdf", main_tex)
        self.assertIn("figures/renewal_cage_glass_audit.pdf", main_tex)
        self.assertIn("figures/renewal_cage_glass_phase_diagram.pdf", main_tex)
        self.assertIn("figures/renewal_cage_spatial_chi4.pdf", main_tex)
        self.assertIn("figures/renewal_cage_thermodynamic_closure.pdf", main_tex)
        self.assertIn("figures/renewal_cage_mct_beta_closure.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_benchmark_consistency.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_claim_alignment.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_evidence_verdict.pdf", main_tex)
        self.assertIn("figures/renewal_cage_real_benchmark_assimilation_gate.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_source_provenance.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_data_accession.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_zenodo_record_fingerprint.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_remote_zip_central_directory.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_payload_index.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_payload_locator.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_entry_metadata.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_member_stream_probe.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_npz_schema_probe.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_remote_result_curve_cache.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_remote_result_curve_fetch_gap.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_remote_result_curve_target_fetch.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_remote_result_curve_published_semantics.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_remote_result_curve_payload_adapter.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_remote_result_curve_observable_semantics.pdf", main_tex)
        self.assertIn("figures/renewal_cage_benchmark_publication_ladder.pdf", main_tex)
        self.assertIn("figures/renewal_cage_persistence_exchange.pdf", main_tex)
        self.assertIn("figures/renewal_cage_persistence_exchange_protocol.pdf", main_tex)
        self.assertIn("figures/renewal_cage_persistence_exchange_joint_protocol.pdf", main_tex)
        self.assertIn("figures/renewal_cage_persistence_exchange_uncertainty_protocol.pdf", main_tex)
        self.assertNotIn(".svg", main_tex)

    def test_readiness_figures_use_page_float_specifiers(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()
        readiness_figures = [
            "figures/renewal_cage_sota_benchmark_consistency.pdf",
            "figures/renewal_cage_sota_claim_alignment.pdf",
            "figures/renewal_cage_sota_evidence_verdict.pdf",
            "figures/renewal_cage_real_benchmark_assimilation_gate.pdf",
            "figures/renewal_cage_sota_source_provenance.pdf",
            "figures/renewal_cage_sota_data_accession.pdf",
            "figures/renewal_cage_sota_zenodo_record_fingerprint.pdf",
            "figures/renewal_cage_sota_remote_zip_central_directory.pdf",
            "figures/renewal_cage_sota_glassbench_payload_index.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_payload_locator.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_entry_metadata.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_member_stream_probe.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_npz_schema_probe.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.pdf",
            "figures/renewal_cage_sota_remote_result_curve_cache.pdf",
            "figures/renewal_cage_sota_remote_result_curve_fetch_gap.pdf",
            "figures/renewal_cage_sota_remote_result_curve_target_fetch.pdf",
            "figures/renewal_cage_sota_remote_result_curve_published_semantics.pdf",
            "figures/renewal_cage_sota_remote_result_curve_payload_adapter.pdf",
            "figures/renewal_cage_sota_remote_result_curve_observable_semantics.pdf",
        ]
        for figure in readiness_figures:
            figure_index = main_tex.index(figure)
            preceding_begin = main_tex.rfind("\\begin{figure}", 0, figure_index)
            self.assertNotEqual(preceding_begin, -1)
            self.assertIn("\\begin{figure}[p]", main_tex[preceding_begin:figure_index])

    def test_readiness_page_float_batch_flushes_before_discussion(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()
        ladder_index = main_tex.index(
            "figures/renewal_cage_benchmark_publication_ladder.pdf"
        )
        discussion_index = main_tex.index("\\section{Discussion}")
        self.assertIn("\\clearpage", main_tex[ladder_index:discussion_index])

    def test_sota_trajectory_payload_locator_flushes_float_batch(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()
        locator_index = main_tex.index("figures/renewal_cage_sota_glassbench_trajectory_payload_locator.pdf")
        reanalysis_index = main_tex.index("The reanalysis-state ledger")

        self.assertIn("\\clearpage", main_tex[locator_index:reanalysis_index])

    def test_main_text_does_not_overclaim_complete_glass_transition_theory(self):
        text = (ROOT / "paper" / "main.tex").read_text().lower()
        normalized = " ".join(text.split())
        forbidden_claims = [
            "explains all glass-transition phenomena",
            "explains all glass transition phenomena",
            "complete glass transition theory",
            "derives the thermodynamic glass transition",
            "derives an ideal glass transition",
        ]
        for claim in forbidden_claims:
            self.assertNotIn(claim, normalized)
        self.assertIn("dynamical glass signatures", normalized)
        self.assertIn("thermodynamic glass-transition phenomena left as explicit closures", normalized)

    def test_supplemental_large_figures_are_packaged_not_embedded(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()
        supplemental_figures = [
            "figures/renewal_cage_cross_observable_prediction_ledger.pdf",
            "figures/renewal_cage_inversion_identifiability_audit.pdf",
            "figures/renewal_cage_frontier_benchmark_horizon.pdf",
            "figures/renewal_cage_sota_signed_constraints.pdf",
            "figures/renewal_cage_sota_evidence_class.pdf",
            "figures/renewal_cage_simultaneous_closure.pdf",
            "figures/renewal_cage_sota_glassbench_short_window_trend_canary.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_timebase_bridge.pdf",
            "figures/renewal_cage_sota_glassbench_frame_time_mapping_audit.pdf",
            "figures/renewal_cage_sota_glassbench_real_inversion_gap_ledger.pdf",
            "figures/renewal_cage_sota_glassbench_real_inversion_unlock_protocol.pdf",
            "figures/renewal_cage_sota_glassbench_observable_coverage_audit.pdf",
            "figures/renewal_cage_sota_glassbench_first_npz_structural_observable_plan.pdf",
            "figures/renewal_cage_literature_inversion_readiness.pdf",
            "figures/renewal_cage_sota_readme_schema.pdf",
            "figures/renewal_cage_trajectory_adapter_contract.pdf",
            "figures/renewal_cage_observable_falsification_matrix.pdf",
            "figures/renewal_cage_benchmark_fusion_readiness.pdf",
            "figures/renewal_cage_raw_curve_ingestion_contract.pdf",
            "figures/renewal_cage_raw_curve_diagnostic_readiness.pdf",
            "figures/renewal_cage_raw_curve_persistence_exchange_protocol.pdf",
            "figures/renewal_cage_translation_rotation_protocol.pdf",
            "figures/renewal_cage_trajectory_observable_protocol.pdf",
            "figures/renewal_cage_trajectory_uncertainty_protocol.pdf",
            "figures/renewal_cage_trajectory_member_ensemble_uncertainty.pdf",
            "figures/renewal_cage_trajectory_inversion_readiness.pdf",
            "figures/renewal_cage_barrier_requirements.pdf",
            "figures/renewal_cage_barrier.pdf",
            "figures/renewal_cage_heterogeneity.pdf",
            "figures/renewal_cage_heterogeneity_map.pdf",
            "figures/renewal_cage_static_null.pdf",
            "figures/renewal_cage_mechanism_selection.pdf",
            "figures/renewal_cage_inversion.pdf",
        ]
        for figure in supplemental_figures:
            self.assertNotIn(figure, main_tex)

    def test_final_comparison_table_is_not_a_late_float(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()
        discussion_index = main_tex.index("Table~\\ref{tab:comparison}")
        table_index = main_tex.index("\\label{tab:comparison}")
        self.assertIn("\\refstepcounter{table}", main_tex[table_index - 40:table_index])
        bibliography_index = main_tex.index("\\bibliographystyle")
        self.assertNotIn("\\begin{table}", main_tex[discussion_index:bibliography_index])

    def test_build_arxiv_package_generates_deterministic_pdf_figures(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            build_arxiv_package(output_dir=Path(first))
            first_results = (ROOT / "paper" / "figures" / "renewal_cage_results.pdf").read_bytes()
            first_dimensionless = (ROOT / "paper" / "figures" / "renewal_cage_dimensionless.pdf").read_bytes()
            first_scattering = (ROOT / "paper" / "figures" / "renewal_cage_scattering.pdf").read_bytes()
            first_temperature = (ROOT / "paper" / "figures" / "renewal_cage_temperature.pdf").read_bytes()
            first_barrier = (ROOT / "paper" / "figures" / "renewal_cage_barrier.pdf").read_bytes()
            first_heterogeneity = (ROOT / "paper" / "figures" / "renewal_cage_heterogeneity.pdf").read_bytes()
            first_heterogeneity_map = (ROOT / "paper" / "figures" / "renewal_cage_heterogeneity_map.pdf").read_bytes()
            first_static_null = (ROOT / "paper" / "figures" / "renewal_cage_static_null.pdf").read_bytes()
            first_alpha_shape = (ROOT / "paper" / "figures" / "renewal_cage_alpha_shape.pdf").read_bytes()
            first_facilitated_exchange = (
                ROOT / "paper" / "figures" / "renewal_cage_facilitated_exchange.pdf"
            ).read_bytes()
            first_glass_audit = (ROOT / "paper" / "figures" / "renewal_cage_glass_audit.pdf").read_bytes()
            first_glass_phase_diagram = (
                ROOT / "paper" / "figures" / "renewal_cage_glass_phase_diagram.pdf"
            ).read_bytes()
            first_spatial_chi4 = (ROOT / "paper" / "figures" / "renewal_cage_spatial_chi4.pdf").read_bytes()
            first_thermodynamic_closure = (
                ROOT / "paper" / "figures" / "renewal_cage_thermodynamic_closure.pdf"
            ).read_bytes()
            first_mct_beta_closure = (
                ROOT / "paper" / "figures" / "renewal_cage_mct_beta_closure.pdf"
            ).read_bytes()
            first_sota_benchmark_consistency = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_benchmark_consistency.pdf"
            ).read_bytes()
            first_sota_claim_alignment = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_claim_alignment.pdf"
            ).read_bytes()
            first_sota_signed_constraints = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_signed_constraints.pdf"
            ).read_bytes()
            first_sota_evidence_verdict = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_evidence_verdict.pdf"
            ).read_bytes()
            first_sota_evidence_class = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_evidence_class.pdf"
            ).read_bytes()
            first_real_benchmark_assimilation_gate = (
                ROOT / "paper" / "figures" / "renewal_cage_real_benchmark_assimilation_gate.pdf"
            ).read_bytes()
            first_cross_observable_prediction_ledger = (
                ROOT / "paper" / "figures" / "renewal_cage_cross_observable_prediction_ledger.pdf"
            ).read_bytes()
            first_inversion_identifiability_audit = (
                ROOT / "paper" / "figures" / "renewal_cage_inversion_identifiability_audit.pdf"
            ).read_bytes()
            first_frontier_benchmark_horizon = (
                ROOT / "paper" / "figures" / "renewal_cage_frontier_benchmark_horizon.pdf"
            ).read_bytes()
            first_sota_source_provenance = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_source_provenance.pdf"
            ).read_bytes()
            first_sota_data_accession = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_data_accession.pdf"
            ).read_bytes()
            first_sota_zenodo_record_fingerprint = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_zenodo_record_fingerprint.pdf"
            ).read_bytes()
            first_sota_remote_zip_central_directory = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_zip_central_directory.pdf"
            ).read_bytes()
            first_sota_glassbench_payload_index = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_payload_index.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_payload_locator = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_payload_locator.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_entry_metadata = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_entry_metadata.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_member_stream_probe = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_member_stream_probe.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_inner_tar_header_probe = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_npz_schema_probe = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_npz_schema_probe.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_first_npz_observable_smoke = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_first_npz_observable_curve = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_first_npz_inversion_readiness = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_npz_ensemble_horizon = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.pdf"
            ).read_bytes()
            first_sota_remote_result_curve_cache = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_cache.pdf"
            ).read_bytes()
            first_sota_remote_result_curve_fetch_gap = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_fetch_gap.pdf"
            ).read_bytes()
            first_sota_remote_result_curve_target_fetch = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_target_fetch.pdf"
            ).read_bytes()
            first_sota_remote_result_curve_published_semantics = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_published_semantics.pdf"
            ).read_bytes()
            first_sota_remote_result_curve_payload_adapter = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_payload_adapter.pdf"
            ).read_bytes()
            first_sota_remote_result_curve_observable_semantics = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_observable_semantics.pdf"
            ).read_bytes()
            first_sota_readme_schema = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_readme_schema.pdf"
            ).read_bytes()
            first_trajectory_adapter_contract = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_adapter_contract.pdf"
            ).read_bytes()
            first_literature_inversion_readiness = (
                ROOT / "paper" / "figures" / "renewal_cage_literature_inversion_readiness.pdf"
            ).read_bytes()
            first_observable_falsification_matrix = (
                ROOT / "paper" / "figures" / "renewal_cage_observable_falsification_matrix.pdf"
            ).read_bytes()
            first_benchmark_fusion_readiness = (
                ROOT / "paper" / "figures" / "renewal_cage_benchmark_fusion_readiness.pdf"
            ).read_bytes()
            first_raw_curve_ingestion_contract = (
                ROOT / "paper" / "figures" / "renewal_cage_raw_curve_ingestion_contract.pdf"
            ).read_bytes()
            first_raw_curve_diagnostic_readiness = (
                ROOT / "paper" / "figures" / "renewal_cage_raw_curve_diagnostic_readiness.pdf"
            ).read_bytes()
            first_raw_curve_persistence_exchange_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_raw_curve_persistence_exchange_protocol.pdf"
            ).read_bytes()
            first_trajectory_observable_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_observable_protocol.pdf"
            ).read_bytes()
            first_trajectory_uncertainty_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_uncertainty_protocol.pdf"
            ).read_bytes()
            first_trajectory_member_ensemble_uncertainty = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_member_ensemble_uncertainty.pdf"
            ).read_bytes()
            first_trajectory_inversion_readiness = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_inversion_readiness.pdf"
            ).read_bytes()
            first_barrier_requirements = (
                ROOT / "paper" / "figures" / "renewal_cage_barrier_requirements.pdf"
            ).read_bytes()
            first_mechanism_selection = (
                ROOT / "paper" / "figures" / "renewal_cage_mechanism_selection.pdf"
            ).read_bytes()
            first_persistence_exchange = (
                ROOT / "paper" / "figures" / "renewal_cage_persistence_exchange.pdf"
            ).read_bytes()
            first_persistence_exchange_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_persistence_exchange_protocol.pdf"
            ).read_bytes()
            first_persistence_exchange_joint_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_persistence_exchange_joint_protocol.pdf"
            ).read_bytes()
            first_persistence_exchange_uncertainty_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_persistence_exchange_uncertainty_protocol.pdf"
            ).read_bytes()
            first_translation_rotation_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_translation_rotation_protocol.pdf"
            ).read_bytes()
            first_inversion = (ROOT / "paper" / "figures" / "renewal_cage_inversion.pdf").read_bytes()

            time.sleep(1.1)
            build_arxiv_package(output_dir=Path(second))
            second_results = (ROOT / "paper" / "figures" / "renewal_cage_results.pdf").read_bytes()
            second_dimensionless = (ROOT / "paper" / "figures" / "renewal_cage_dimensionless.pdf").read_bytes()
            second_scattering = (ROOT / "paper" / "figures" / "renewal_cage_scattering.pdf").read_bytes()
            second_temperature = (ROOT / "paper" / "figures" / "renewal_cage_temperature.pdf").read_bytes()
            second_barrier = (ROOT / "paper" / "figures" / "renewal_cage_barrier.pdf").read_bytes()
            second_heterogeneity = (ROOT / "paper" / "figures" / "renewal_cage_heterogeneity.pdf").read_bytes()
            second_heterogeneity_map = (ROOT / "paper" / "figures" / "renewal_cage_heterogeneity_map.pdf").read_bytes()
            second_static_null = (ROOT / "paper" / "figures" / "renewal_cage_static_null.pdf").read_bytes()
            second_alpha_shape = (ROOT / "paper" / "figures" / "renewal_cage_alpha_shape.pdf").read_bytes()
            second_facilitated_exchange = (
                ROOT / "paper" / "figures" / "renewal_cage_facilitated_exchange.pdf"
            ).read_bytes()
            second_glass_audit = (ROOT / "paper" / "figures" / "renewal_cage_glass_audit.pdf").read_bytes()
            second_glass_phase_diagram = (
                ROOT / "paper" / "figures" / "renewal_cage_glass_phase_diagram.pdf"
            ).read_bytes()
            second_spatial_chi4 = (ROOT / "paper" / "figures" / "renewal_cage_spatial_chi4.pdf").read_bytes()
            second_thermodynamic_closure = (
                ROOT / "paper" / "figures" / "renewal_cage_thermodynamic_closure.pdf"
            ).read_bytes()
            second_mct_beta_closure = (
                ROOT / "paper" / "figures" / "renewal_cage_mct_beta_closure.pdf"
            ).read_bytes()
            second_sota_benchmark_consistency = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_benchmark_consistency.pdf"
            ).read_bytes()
            second_sota_claim_alignment = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_claim_alignment.pdf"
            ).read_bytes()
            second_sota_signed_constraints = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_signed_constraints.pdf"
            ).read_bytes()
            second_sota_evidence_verdict = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_evidence_verdict.pdf"
            ).read_bytes()
            second_sota_evidence_class = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_evidence_class.pdf"
            ).read_bytes()
            second_real_benchmark_assimilation_gate = (
                ROOT / "paper" / "figures" / "renewal_cage_real_benchmark_assimilation_gate.pdf"
            ).read_bytes()
            second_cross_observable_prediction_ledger = (
                ROOT / "paper" / "figures" / "renewal_cage_cross_observable_prediction_ledger.pdf"
            ).read_bytes()
            second_inversion_identifiability_audit = (
                ROOT / "paper" / "figures" / "renewal_cage_inversion_identifiability_audit.pdf"
            ).read_bytes()
            second_frontier_benchmark_horizon = (
                ROOT / "paper" / "figures" / "renewal_cage_frontier_benchmark_horizon.pdf"
            ).read_bytes()
            second_sota_source_provenance = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_source_provenance.pdf"
            ).read_bytes()
            second_sota_data_accession = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_data_accession.pdf"
            ).read_bytes()
            second_sota_zenodo_record_fingerprint = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_zenodo_record_fingerprint.pdf"
            ).read_bytes()
            second_sota_remote_zip_central_directory = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_zip_central_directory.pdf"
            ).read_bytes()
            second_sota_glassbench_payload_index = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_payload_index.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_payload_locator = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_payload_locator.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_entry_metadata = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_entry_metadata.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_member_stream_probe = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_member_stream_probe.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_inner_tar_header_probe = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_inner_tar_header_probe.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_npz_schema_probe = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_glassbench_trajectory_npz_schema_probe.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_first_npz_observable_smoke = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_smoke.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_first_npz_observable_curve = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_first_npz_observable_curve.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_first_npz_inversion_readiness = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_first_npz_inversion_readiness.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_npz_ensemble_horizon = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.pdf"
            ).read_bytes()
            second_sota_remote_result_curve_cache = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_cache.pdf"
            ).read_bytes()
            second_sota_remote_result_curve_fetch_gap = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_fetch_gap.pdf"
            ).read_bytes()
            second_sota_remote_result_curve_target_fetch = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_target_fetch.pdf"
            ).read_bytes()
            second_sota_remote_result_curve_published_semantics = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_published_semantics.pdf"
            ).read_bytes()
            second_sota_remote_result_curve_payload_adapter = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_payload_adapter.pdf"
            ).read_bytes()
            second_sota_remote_result_curve_observable_semantics = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_remote_result_curve_observable_semantics.pdf"
            ).read_bytes()
            second_sota_readme_schema = (
                ROOT / "paper" / "figures" / "renewal_cage_sota_readme_schema.pdf"
            ).read_bytes()
            second_trajectory_adapter_contract = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_adapter_contract.pdf"
            ).read_bytes()
            second_literature_inversion_readiness = (
                ROOT / "paper" / "figures" / "renewal_cage_literature_inversion_readiness.pdf"
            ).read_bytes()
            second_observable_falsification_matrix = (
                ROOT / "paper" / "figures" / "renewal_cage_observable_falsification_matrix.pdf"
            ).read_bytes()
            second_benchmark_fusion_readiness = (
                ROOT / "paper" / "figures" / "renewal_cage_benchmark_fusion_readiness.pdf"
            ).read_bytes()
            second_raw_curve_ingestion_contract = (
                ROOT / "paper" / "figures" / "renewal_cage_raw_curve_ingestion_contract.pdf"
            ).read_bytes()
            second_raw_curve_diagnostic_readiness = (
                ROOT / "paper" / "figures" / "renewal_cage_raw_curve_diagnostic_readiness.pdf"
            ).read_bytes()
            second_raw_curve_persistence_exchange_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_raw_curve_persistence_exchange_protocol.pdf"
            ).read_bytes()
            second_trajectory_observable_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_observable_protocol.pdf"
            ).read_bytes()
            second_trajectory_uncertainty_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_uncertainty_protocol.pdf"
            ).read_bytes()
            second_trajectory_member_ensemble_uncertainty = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_member_ensemble_uncertainty.pdf"
            ).read_bytes()
            second_trajectory_inversion_readiness = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_inversion_readiness.pdf"
            ).read_bytes()
            second_barrier_requirements = (
                ROOT / "paper" / "figures" / "renewal_cage_barrier_requirements.pdf"
            ).read_bytes()
            second_mechanism_selection = (
                ROOT / "paper" / "figures" / "renewal_cage_mechanism_selection.pdf"
            ).read_bytes()
            second_persistence_exchange = (
                ROOT / "paper" / "figures" / "renewal_cage_persistence_exchange.pdf"
            ).read_bytes()
            second_persistence_exchange_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_persistence_exchange_protocol.pdf"
            ).read_bytes()
            second_persistence_exchange_joint_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_persistence_exchange_joint_protocol.pdf"
            ).read_bytes()
            second_persistence_exchange_uncertainty_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_persistence_exchange_uncertainty_protocol.pdf"
            ).read_bytes()
            second_translation_rotation_protocol = (
                ROOT / "paper" / "figures" / "renewal_cage_translation_rotation_protocol.pdf"
            ).read_bytes()
            second_inversion = (ROOT / "paper" / "figures" / "renewal_cage_inversion.pdf").read_bytes()

        self.assertEqual(first_results, second_results)
        self.assertEqual(first_dimensionless, second_dimensionless)
        self.assertEqual(first_scattering, second_scattering)
        self.assertEqual(first_temperature, second_temperature)
        self.assertEqual(first_barrier, second_barrier)
        self.assertEqual(first_heterogeneity, second_heterogeneity)
        self.assertEqual(first_heterogeneity_map, second_heterogeneity_map)
        self.assertEqual(first_static_null, second_static_null)
        self.assertEqual(first_alpha_shape, second_alpha_shape)
        self.assertEqual(first_facilitated_exchange, second_facilitated_exchange)
        self.assertEqual(first_glass_audit, second_glass_audit)
        self.assertEqual(first_glass_phase_diagram, second_glass_phase_diagram)
        self.assertEqual(first_spatial_chi4, second_spatial_chi4)
        self.assertEqual(first_thermodynamic_closure, second_thermodynamic_closure)
        self.assertEqual(first_mct_beta_closure, second_mct_beta_closure)
        self.assertEqual(first_sota_benchmark_consistency, second_sota_benchmark_consistency)
        self.assertEqual(first_sota_claim_alignment, second_sota_claim_alignment)
        self.assertEqual(first_sota_signed_constraints, second_sota_signed_constraints)
        self.assertEqual(first_sota_evidence_verdict, second_sota_evidence_verdict)
        self.assertEqual(first_sota_evidence_class, second_sota_evidence_class)
        self.assertEqual(first_real_benchmark_assimilation_gate, second_real_benchmark_assimilation_gate)
        self.assertEqual(first_cross_observable_prediction_ledger, second_cross_observable_prediction_ledger)
        self.assertEqual(first_inversion_identifiability_audit, second_inversion_identifiability_audit)
        self.assertEqual(first_frontier_benchmark_horizon, second_frontier_benchmark_horizon)
        self.assertEqual(first_sota_source_provenance, second_sota_source_provenance)
        self.assertEqual(first_sota_data_accession, second_sota_data_accession)
        self.assertEqual(first_sota_zenodo_record_fingerprint, second_sota_zenodo_record_fingerprint)
        self.assertEqual(first_sota_remote_zip_central_directory, second_sota_remote_zip_central_directory)
        self.assertEqual(first_sota_glassbench_payload_index, second_sota_glassbench_payload_index)
        self.assertEqual(
            first_sota_glassbench_trajectory_payload_locator,
            second_sota_glassbench_trajectory_payload_locator,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_entry_metadata,
            second_sota_glassbench_trajectory_entry_metadata,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_member_stream_probe,
            second_sota_glassbench_trajectory_member_stream_probe,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_inner_tar_header_probe,
            second_sota_glassbench_trajectory_inner_tar_header_probe,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_npz_schema_probe,
            second_sota_glassbench_trajectory_npz_schema_probe,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_first_npz_observable_smoke,
            second_sota_glassbench_trajectory_first_npz_observable_smoke,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_first_npz_observable_curve,
            second_sota_glassbench_trajectory_first_npz_observable_curve,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_first_npz_inversion_readiness,
            second_sota_glassbench_trajectory_first_npz_inversion_readiness,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_npz_ensemble_horizon,
            second_sota_glassbench_trajectory_npz_ensemble_horizon,
        )
        self.assertEqual(first_sota_remote_result_curve_cache, second_sota_remote_result_curve_cache)
        self.assertEqual(first_sota_remote_result_curve_fetch_gap, second_sota_remote_result_curve_fetch_gap)
        self.assertEqual(first_sota_remote_result_curve_target_fetch, second_sota_remote_result_curve_target_fetch)
        self.assertEqual(
            first_sota_remote_result_curve_published_semantics,
            second_sota_remote_result_curve_published_semantics,
        )
        self.assertEqual(
            first_sota_remote_result_curve_payload_adapter,
            second_sota_remote_result_curve_payload_adapter,
        )
        self.assertEqual(
            first_sota_remote_result_curve_observable_semantics,
            second_sota_remote_result_curve_observable_semantics,
        )
        self.assertEqual(first_sota_readme_schema, second_sota_readme_schema)
        self.assertEqual(first_trajectory_adapter_contract, second_trajectory_adapter_contract)
        self.assertEqual(first_literature_inversion_readiness, second_literature_inversion_readiness)
        self.assertEqual(first_observable_falsification_matrix, second_observable_falsification_matrix)
        self.assertEqual(first_benchmark_fusion_readiness, second_benchmark_fusion_readiness)
        self.assertEqual(first_raw_curve_ingestion_contract, second_raw_curve_ingestion_contract)
        self.assertEqual(first_raw_curve_diagnostic_readiness, second_raw_curve_diagnostic_readiness)
        self.assertEqual(
            first_raw_curve_persistence_exchange_protocol,
            second_raw_curve_persistence_exchange_protocol,
        )
        self.assertEqual(first_trajectory_observable_protocol, second_trajectory_observable_protocol)
        self.assertEqual(first_trajectory_uncertainty_protocol, second_trajectory_uncertainty_protocol)
        self.assertEqual(
            first_trajectory_member_ensemble_uncertainty,
            second_trajectory_member_ensemble_uncertainty,
        )
        self.assertEqual(first_trajectory_inversion_readiness, second_trajectory_inversion_readiness)
        self.assertEqual(first_barrier_requirements, second_barrier_requirements)
        self.assertEqual(first_mechanism_selection, second_mechanism_selection)
        self.assertEqual(first_persistence_exchange, second_persistence_exchange)
        self.assertEqual(first_persistence_exchange_protocol, second_persistence_exchange_protocol)
        self.assertEqual(first_persistence_exchange_joint_protocol, second_persistence_exchange_joint_protocol)
        self.assertEqual(first_persistence_exchange_uncertainty_protocol, second_persistence_exchange_uncertainty_protocol)
        self.assertEqual(first_translation_rotation_protocol, second_translation_rotation_protocol)
        self.assertEqual(first_inversion, second_inversion)


if __name__ == "__main__":
    unittest.main()
