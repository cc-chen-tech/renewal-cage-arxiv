import sys
import tempfile
import time
import unittest
import zipfile
import csv
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
            self.assertIn("figures/renewal_cage_real_benchmark_assimilation_gate.pdf", names)
            self.assertIn("figures/renewal_cage_cross_observable_prediction_ledger.pdf", names)
            self.assertIn("figures/renewal_cage_inversion_identifiability_audit.pdf", names)
            self.assertIn("figures/renewal_cage_frontier_benchmark_horizon.pdf", names)
            self.assertIn("figures/renewal_cage_literature_inversion_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_observable_falsification_matrix.pdf", names)
            self.assertIn("figures/renewal_cage_benchmark_fusion_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_raw_curve_ingestion_contract.pdf", names)
            self.assertIn("figures/renewal_cage_raw_curve_diagnostic_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_raw_curve_persistence_exchange_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_trajectory_observable_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_barrier_requirements.pdf", names)
            self.assertIn("figures/renewal_cage_mechanism_selection.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_joint_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_uncertainty_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_translation_rotation_protocol.pdf", names)
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
        self.assertIn("figures/renewal_cage_real_benchmark_assimilation_gate.pdf", main_tex)
        self.assertIn("figures/renewal_cage_literature_inversion_readiness.pdf", main_tex)
        self.assertIn("figures/renewal_cage_observable_falsification_matrix.pdf", main_tex)
        self.assertIn("figures/renewal_cage_benchmark_fusion_readiness.pdf", main_tex)
        self.assertIn("figures/renewal_cage_raw_curve_ingestion_contract.pdf", main_tex)
        self.assertIn("figures/renewal_cage_raw_curve_diagnostic_readiness.pdf", main_tex)
        self.assertIn("figures/renewal_cage_raw_curve_persistence_exchange_protocol.pdf", main_tex)
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
            "figures/renewal_cage_real_benchmark_assimilation_gate.pdf",
            "figures/renewal_cage_literature_inversion_readiness.pdf",
            "figures/renewal_cage_observable_falsification_matrix.pdf",
            "figures/renewal_cage_benchmark_fusion_readiness.pdf",
            "figures/renewal_cage_raw_curve_ingestion_contract.pdf",
            "figures/renewal_cage_raw_curve_diagnostic_readiness.pdf",
            "figures/renewal_cage_raw_curve_persistence_exchange_protocol.pdf",
        ]
        for figure in readiness_figures:
            figure_index = main_tex.index(figure)
            preceding_begin = main_tex.rfind("\\begin{figure}", 0, figure_index)
            self.assertNotEqual(preceding_begin, -1)
            self.assertIn("\\begin{figure}[p]", main_tex[preceding_begin:figure_index])

    def test_readiness_page_float_batch_flushes_before_discussion(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()
        raw_protocol_index = main_tex.index(
            "figures/renewal_cage_raw_curve_persistence_exchange_protocol.pdf"
        )
        discussion_index = main_tex.index("\\section{Discussion}")
        self.assertIn("\\clearpage", main_tex[raw_protocol_index:discussion_index])

    def test_supplemental_large_figures_are_packaged_not_embedded(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()
        supplemental_figures = [
            "figures/renewal_cage_cross_observable_prediction_ledger.pdf",
            "figures/renewal_cage_inversion_identifiability_audit.pdf",
            "figures/renewal_cage_frontier_benchmark_horizon.pdf",
            "figures/renewal_cage_sota_signed_constraints.pdf",
            "figures/renewal_cage_translation_rotation_protocol.pdf",
            "figures/renewal_cage_trajectory_observable_protocol.pdf",
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
        self.assertEqual(first_real_benchmark_assimilation_gate, second_real_benchmark_assimilation_gate)
        self.assertEqual(first_cross_observable_prediction_ledger, second_cross_observable_prediction_ledger)
        self.assertEqual(first_inversion_identifiability_audit, second_inversion_identifiability_audit)
        self.assertEqual(first_frontier_benchmark_horizon, second_frontier_benchmark_horizon)
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
