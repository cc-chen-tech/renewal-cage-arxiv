import csv
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from renewal_cage import (  # noqa: E402
    ActivatedBarrierParams,
    DelayedRenewalCageParams,
    FacilitatedExchangeLawParams,
    GammaExchangeParams,
    LangevinCageLandscapeParams,
    TemperatureLawParams,
    alpha_relaxation_time,
    apparent_alpha_activation_energies,
    activated_barrier_temperature_law,
    alpha_tts_benchmark_consistency,
    alpha_relaxation_shape_curve,
    alpha_shape_superposition_residual,
    barrier_amplification_laws,
    benchmark_fusion_readiness,
    cage_localization_benchmark_consistency,
    cage_localization_diagnostics,
    correlated_domain_susceptibility,
    classify_delay_exponent,
    delayed_poisson_mean,
    delayed_renewal_shape,
    dimensionless_peak_prediction,
    fractional_stokes_einstein_exponents,
    gamma_exchange_temperature_scan,
    glass_phenomenon_audit,
    glass_signature_phase_diagram,
    dynamic_heterogeneity_benchmark_consistency,
    dynamic_signature_alignment_ledger,
    infer_parameters_from_full_observables,
    infer_renewal_correlation_size,
    infer_parameters_from_scattering_transport,
    inversion_identifiability_audit,
    joint_inversion_benchmark_consistency,
    literature_inversion_readiness,
    generalized_delay_ngp_short_time,
    gaussian_radial_3d,
    gamma_exchange_alpha_relaxation_time,
    gamma_exchange_count_moments,
    gamma_exchange_asymptotic_diagnostics,
    gamma_exchange_diagnostic_map,
    infer_gamma_exchange_from_late_observables,
    infer_gamma_exchange_multik_collapse,
    infer_gamma_exchange_uncertainty_from_late_observables,
    infer_gamma_exchange_ratio_from_alpha_rate,
    gamma_exchange_ngp_1d,
    gamma_exchange_normalized_alpha_decay,
    gamma_exchange_scattering_susceptibility,
    gamma_exchange_self_intermediate_scattering,
    fragility_benchmark_consistency,
    frontier_benchmark_horizon,
    gaussian_recovery_benchmark_consistency,
    infer_spatial_facilitation_diffusivity,
    kww_alpha_fit,
    long_time_diffusion_coefficient,
    local_alpha_stretching_exponent,
    late_mechanism_selection,
    kramers_escape_rate,
    langevin_bare_diffusion,
    langevin_cage_ou_parameters,
    langevin_first_principles_bridge_audit,
    langevin_to_persistence_exchange,
    minimal_barrier_requirements,
    MCTBetaParams,
    mct_beta_correlator,
    mct_beta_benchmark_consistency,
    mct_exponent_benchmark_consistency,
    mct_beta_temperature_scan,
    ngp_peak_benchmark_consistency,
    observable_consistency_diagnostics,
    observable_falsification_matrix,
    raw_curve_ingestion_contract,
    raw_curve_diagnostic_readiness,
    raw_curve_persistence_exchange_protocol,
    real_benchmark_assimilation_gate,
    cross_observable_prediction_ledger,
    persistence_exchange_benchmark_consistency,
    radial_van_hove_3d,
    van_hove_tail_benchmark_consistency,
    local_cage_variance,
    moments_1d,
    moments_3d,
    ngp_1d,
    ngp_3d,
    normalized_alpha_decay,
    plateau_ngp_branches,
    plateau_peak_diagnostics,
    peak_relaxation_coupling,
    PersistenceExchangeParams,
    persistence_exchange_alpha_relaxation_time,
    persistence_exchange_count_distribution,
    persistence_exchange_count_pgf,
    persistence_exchange_count_moments,
    persistence_exchange_diffusion_coefficient,
    persistence_exchange_data_protocol,
    persistence_exchange_joint_diagnostic,
    simultaneous_dynamical_signature_closure_gate,
    persistence_exchange_ngp_1d,
    persistence_exchange_normalized_alpha_decay,
    persistence_exchange_scan,
    persistence_exchange_scattering_susceptibility,
    renewal_scattering_susceptibility,
    self_intermediate_scattering,
    static_gamma_asymptotic_diagnostics,
    static_gamma_count_moments,
    static_gamma_ngp_1d,
    static_gamma_normalized_alpha_decay,
    spatial_facilitation_growth_law_consistency,
    stokes_einstein_benchmark_consistency,
    stretched_alpha_benchmark_consistency,
    stokes_einstein_product,
    sota_archive_preflight_gate,
    sota_claim_alignment,
    sota_data_accession_gate,
    sota_evidence_verdict,
    sota_evidence_class_gate,
    sota_glassbench_trajectory_entry_metadata_gate,
    sota_glassbench_visible_member_ensemble_audit_gate,
    sota_glassbench_trajectory_npz_ensemble_horizon_gate,
    sota_glassbench_real_inversion_gap_ledger_gate,
    sota_glassbench_real_inversion_unlock_protocol_gate,
    sota_glassbench_frame_time_mapping_audit_gate,
    sota_glassbench_first_npz_structural_observable_plan_gate,
    glassbench_alpha_threshold_horizon_audit,
    glassbench_alpha_anchor_rescue_protocol,
    glassbench_alpha_anchor_cached_fs_audit,
    glassbench_direct_alpha_curve_audit,
    glassbench_direct_alpha_displacement_tail_bound,
    glassbench_direct_alpha_event_clock_extraction_contract,
    glassbench_direct_alpha_multilag_crossing_canary,
    glassbench_direct_alpha_transport_coupling_audit,
    glassbench_direct_alpha_pe_feasibility_bound,
    glassbench_sparse_lag_event_clock_audit,
    glassbench_interval_censored_first_crossing_clock,
    glassbench_interval_censored_persistence_fit,
    glassbench_finite_exchange_falsification_envelope,
    glassbench_late_recovery_falsification_protocol,
    glassbench_late_recovery_ingestion_contract,
    glassbench_late_recovery_timecode_target,
    glassbench_late_recovery_cache_request_contract,
    glassbench_late_recovery_membership_probe_contract,
    glassbench_late_recovery_public_timecode_ceiling,
    glassbench_censored_window_claim_audit,
    glassbench_sota_public_window_verdict,
    glassbench_late_recovery_experiment_design,
    glassbench_late_recovery_uncertainty_verdict,
    glassbench_late_recovery_outcome_matrix,
    glassbench_cage_jump_proxy_canary,
    glassbench_event_clock_threshold_readiness_gate,
    glassbench_cached_particle_timecode_bridge,
    glassbench_multilag_particle_cache_targets,
    glassbench_cached_particle_observable_semantics_audit,
    glassbench_first_npz_particle_cache_contract_gate,
    glassbench_microdynamic_closed_loop_audit,
    glassbench_timecode_signature_support_gate,
    glassbench_timecode_curve_bridge,
    sota_glassbench_ka2d_timecode_semantics_gate,
    sota_glassbench_observable_coverage_audit_gate,
    sota_glassbench_trajectory_first_npz_observable_curve_gate,
    sota_glassbench_trajectory_first_npz_inversion_readiness_gate,
    sota_glassbench_trajectory_member_ensemble_observable_gate,
    sota_glassbench_trajectory_npz_member_index_gate,
    sota_glassbench_short_window_trend_canary_gate,
    sota_glassbench_trajectory_timebase_bridge_gate,
    sota_glassbench_trajectory_inner_tar_header_probe_gate,
    sota_glassbench_trajectory_member_stream_probe_gate,
    sota_glassbench_trajectory_first_npz_observable_smoke_gate,
    sota_glassbench_trajectory_npz_schema_probe_gate,
    sota_glassbench_trajectory_payload_locator_gate,
    sota_local_cache_verification_gate,
    sota_readme_digest_gate,
    sota_glassbench_payload_index_gate,
    sota_remote_result_curve_cache_gate,
    sota_remote_result_curve_fetch_gap_gate,
    sota_remote_result_curve_payload_adapter_gate,
    sota_remote_result_curve_published_semantic_audit_gate,
    sota_remote_result_curve_target_fetch_gate,
    sota_remote_result_curve_observable_semantics_gate,
    sota_remote_zip_central_directory_gate,
    sota_zenodo_record_fingerprint_gate,
    sota_readme_schema_gate,
    sota_reanalysis_state_gate,
    sota_source_provenance_gate,
    sota_zip_structure_gate,
    sota_signed_constraint_audit,
    thermodynamic_scope_benchmark_consistency,
    temperature_dependent_params,
    temperature_dependent_gamma_exchange,
    temperature_scan,
    trajectory_adapter_contract,
    benchmark_publication_ladder,
    trajectory_inversion_readiness_gate,
    trajectory_cage_jump_event_protocol,
    trajectory_event_clock_macro_prediction_protocol,
    trajectory_event_clock_threshold_robustness_protocol,
    trajectory_observable_protocol,
    trajectory_observable_uncertainty_protocol,
    trajectory_table_csv_adapter,
    trajectory_table_adapter,
    trajectory_observable_curve_bridge,
    trajectory_curve_persistence_exchange_gate,
    trajectory_member_ensemble_uncertainty_protocol,
    trajectory_pe_heldout_prediction_gate,
    trajectory_prediction_falsification_gate,
    TranslationRotationExchangeParams,
    translation_rotation_decoupling_diagnostic,
    translation_rotation_inversion_protocol,
    translation_rotation_rotational_relaxation_time,
)


class DelayedRenewalCageTests(unittest.TestCase):
    def test_delayed_poisson_mean_has_cubic_short_time_onset(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=1.0,
            jump_variance=0.5,
            renewal_rate=2.0,
            renewal_delay=4.0,
        )
        t = np.array([1e-4, 2e-4, 4e-4])
        mean = delayed_poisson_mean(t, params)
        expected = params.renewal_rate * t**3 / (3.0 * params.renewal_delay**2)

        np.testing.assert_allclose(mean, expected, rtol=2e-4, atol=1e-16)

    def test_glassbench_timecode_curve_bridge_keeps_real_curve_before_inversion(self):
        rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "time_code": "tc05",
                "lag_time": 0.1,
                "tau_alpha": 918306.0,
                "timecode_curve_ready": 1.0,
                "member_count": 9.0,
                "msd": 0.0035,
                "sigma_msd_member_sem": 1e-5,
                "ngp_2d": 0.008,
                "sigma_ngp_2d_member_sem": 0.002,
                "wave_numbers": "0.7;1.1;1.6",
                "self_intermediate_scattering_by_k": "0.999;0.998;0.997",
                "sigma_self_intermediate_scattering_by_k_member_sem": "1e-6;2e-6;3e-6",
                "chi4_overlap_replica": 0.04,
                "sigma_chi4_overlap_member_sem": 0.005,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "time_code": "tc40",
                "lag_time": 1500000.0,
                "tau_alpha": 918306.0,
                "timecode_curve_ready": 1.0,
                "member_count": 6.0,
                "msd": 1.2,
                "sigma_msd_member_sem": 0.1,
                "ngp_2d": 1.9,
                "sigma_ngp_2d_member_sem": 0.16,
                "wave_numbers": "0.7;1.1;1.6",
                "self_intermediate_scattering_by_k": "0.879;0.763;0.640",
                "sigma_self_intermediate_scattering_by_k_member_sem": "0.009;0.016;0.023",
                "chi4_overlap_replica": 3.17,
                "sigma_chi4_overlap_member_sem": 0.79,
            },
        ]

        row = glassbench_timecode_curve_bridge(
            benchmark_id="glassbench_ka2d_t023_timecode_curve",
            rows=rows,
            required_wave_numbers=[0.7, 1.1, 1.6],
            anchor_wave_number=1.1,
        )[0]

        self.assertEqual(row["bridge_stage"], "glassbench_timecode_curve_bridge_incomplete")
        self.assertEqual(float(row["timecode_curve_ready"]), 1.0)
        self.assertEqual(float(row["real_time_observable_curve_ready"]), 1.0)
        self.assertEqual(float(row["curve_bridge_ready"]), 0.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "alpha_threshold_crossing")
        self.assertEqual(float(row["lag_count"]), 2.0)
        self.assertGreater(float(row["latest_lag_time_over_tau_alpha"]), 1.0)
        self.assertGreater(float(row["latest_self_intermediate_scattering_anchor"]), math.exp(-1.0))
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_timecode_signature_support_scores_real_dynamic_signatures(self):
        timecode_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "time_code": "tc05",
                "lag_time": 0.1,
                "timecode_curve_ready": 1.0,
                "msd": 0.003,
                "ngp_2d": 0.01,
                "wave_numbers": "0.7;1.1;1.6",
                "self_intermediate_scattering_by_k": "0.999;0.998;0.996",
                "chi4_overlap_replica": 0.05,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "time_code": "tc35",
                "lag_time": 142587.0,
                "timecode_curve_ready": 1.0,
                "msd": 0.15,
                "ngp_2d": 5.2,
                "wave_numbers": "0.7;1.1;1.6",
                "self_intermediate_scattering_by_k": "0.98;0.96;0.93",
                "chi4_overlap_replica": 5.7,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "time_code": "tc40",
                "lag_time": 1500000.0,
                "timecode_curve_ready": 1.0,
                "msd": 1.2,
                "ngp_2d": 1.9,
                "wave_numbers": "0.7;1.1;1.6",
                "self_intermediate_scattering_by_k": "0.88;0.76;0.64",
                "chi4_overlap_replica": 3.2,
            },
        ]
        bridge_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "real_time_observable_curve_ready": 1.0,
                "real_pe_inversion_ready": 0.0,
                "primary_blocker": "alpha_threshold_crossing",
            }
        ]

        row = glassbench_timecode_signature_support_gate(
            support_id="glassbench_signature_support",
            timecode_rows=timecode_rows,
            bridge_rows=bridge_rows,
            anchor_wave_number=1.1,
        )[0]

        self.assertEqual(row["signature_stage"], "real_curve_dynamic_signature_support_preinversion")
        self.assertEqual(float(row["real_time_observable_curve_ready"]), 1.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(float(row["msd_growth_signature"]), 1.0)
        self.assertEqual(float(row["self_intermediate_decay_signature"]), 1.0)
        self.assertEqual(float(row["transient_ngp_peak_signature"]), 1.0)
        self.assertEqual(float(row["transient_chi4_peak_signature"]), 1.0)
        self.assertEqual(float(row["alpha_threshold_crossed"]), 0.0)
        self.assertGreaterEqual(float(row["supported_dynamical_signature_count"]), 4.0)
        self.assertEqual(row["primary_blocker"], "alpha_threshold_crossing")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_alpha_threshold_horizon_audit_flags_metadata_anchor_mismatch(self):
        timecode_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "time_code": "tc20",
                "lag_time": 50.0,
                "tau_alpha": 100.0,
                "timecode_curve_ready": 1.0,
                "wave_numbers": "0.7;1.1;1.6",
                "self_intermediate_scattering_by_k": "0.93;0.88;0.75",
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "time_code": "tc40",
                "lag_time": 200.0,
                "tau_alpha": 100.0,
                "timecode_curve_ready": 1.0,
                "wave_numbers": "0.7;1.1;1.6",
                "self_intermediate_scattering_by_k": "0.80;0.70;0.55",
            },
        ]
        bridge_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "real_time_observable_curve_ready": 1.0,
                "real_pe_inversion_ready": 0.0,
                "primary_blocker": "alpha_threshold_crossing",
            }
        ]

        row = glassbench_alpha_threshold_horizon_audit(
            audit_id="glassbench_alpha_threshold_horizon",
            timecode_rows=timecode_rows,
            bridge_rows=bridge_rows,
            anchor_wave_number=1.1,
        )[0]

        self.assertEqual(row["audit_stage"], "metadata_tau_alpha_anchor_fs_mismatch")
        self.assertEqual(float(row["real_time_observable_curve_ready"]), 1.0)
        self.assertEqual(float(row["metadata_tau_alpha_reached"]), 1.0)
        self.assertEqual(float(row["alpha_threshold_crossed"]), 0.0)
        self.assertEqual(float(row["metadata_tau_alpha_consistent_with_anchor_fs"]), 0.0)
        self.assertGreater(float(row["latest_lag_time_over_tau_alpha_metadata"]), 1.0)
        self.assertGreater(float(row["latest_self_intermediate_scattering_anchor"]), math.exp(-1.0))
        self.assertGreater(float(row["estimated_threshold_wave_number_at_latest_lag"]), 1.6)
        self.assertGreater(float(row["threshold_wave_number_over_max_observed"]), 1.0)
        self.assertEqual(float(row["alpha_threshold_wave_number_covered"]), 0.0)
        self.assertGreater(float(row["estimated_lag_extension_factor"]), 1.0)
        self.assertEqual(row["primary_blocker"], "alpha_anchor_wave_number_outside_observed_grid")
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_alpha_anchor_rescue_protocol_separates_anchor_from_event_clock(self):
        alpha_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "estimated_threshold_wave_number_at_latest_lag": 2.7,
                "threshold_wave_number_over_max_observed": 1.69,
                "alpha_threshold_wave_number_covered": 0.0,
                "metadata_tau_alpha_reached": 1.0,
                "alpha_threshold_crossed": 0.0,
                "metadata_tau_alpha_consistent_with_anchor_fs": 0.0,
                "primary_blocker": "alpha_anchor_wave_number_outside_observed_grid",
                "audit_stage": "metadata_tau_alpha_anchor_fs_mismatch",
            }
        ]
        event_clock_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "threshold_sweep_event_clock_ready": 0.0,
                "macro_heldout_observables_ready": 0.0,
                "real_event_clock_threshold_robustness_ready": 0.0,
                "missing_real_threshold_inputs": "physical_time_semantics;macro_heldout_observables;threshold_sweep_event_clock",
                "primary_blocker": "physical_time_semantics",
            }
        ]
        closed_loop_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "missing_closed_loop_inputs": "physical_time_semantics;cage_jump_event_segmentation;persistence_exchange_event_clock;alpha_definition_consistency;real_persistence_exchange_inversion",
                "closed_loop_ready": 0.0,
                "primary_blocker": "physical_time_semantics",
            }
        ]

        row = glassbench_alpha_anchor_rescue_protocol(
            protocol_id="glassbench_alpha_anchor_rescue",
            alpha_horizon_rows=alpha_rows,
            event_clock_rows=event_clock_rows,
            closed_loop_rows=closed_loop_rows,
        )[0]

        self.assertEqual(row["rescue_stage"], "alpha_anchor_rescue_design_ready_real_event_clock_blocked")
        self.assertEqual(float(row["required_anchor_wave_number"]), 2.7)
        self.assertGreater(float(row["required_anchor_wave_number_over_observed_max"]), 1.0)
        self.assertEqual(float(row["alpha_anchor_measurement_required"]), 1.0)
        self.assertEqual(float(row["alpha_anchor_rescue_design_ready"]), 1.0)
        self.assertEqual(float(row["post_rescue_alpha_definition_consistent"]), 1.0)
        self.assertEqual(float(row["post_rescue_real_closed_loop_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "physical_time_semantics")
        self.assertIn("threshold_sweep_event_clock", row["remaining_post_rescue_blockers"])
        self.assertIn("persistence_exchange_event_clock", row["remaining_post_rescue_blockers"])
        self.assertNotIn("alpha_definition_consistency", row["remaining_post_rescue_blockers"])
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_alpha_anchor_cached_fs_audit_refines_required_k(self):
        rescue_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "required_anchor_wave_number": 2.7,
                "alpha_anchor_rescue_design_ready": 1.0,
                "post_rescue_real_closed_loop_ready": 0.0,
            }
        ]
        cached_anchor_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "time_code": "tc40",
                "lag_time": 1500000.0,
                "candidate_anchor_wave_number": 2.7,
                "cached_fs_at_candidate_anchor": 0.52,
                "latest_wave_numbers": "0.7;1.1;1.6",
                "latest_cached_fs_by_k": "0.90;0.80;0.69",
                "direct_threshold_wave_number": 4.8,
                "direct_fs_at_threshold_wave_number": math.exp(-1.0),
                "direct_root_bracketed": 1.0,
            }
        ]

        row = glassbench_alpha_anchor_cached_fs_audit(
            audit_id="glassbench_cached_alpha_anchor_fs",
            rescue_rows=rescue_rows,
            cached_anchor_rows=cached_anchor_rows,
        )[0]

        self.assertEqual(row["cached_anchor_stage"], "cached_direct_anchor_root_refines_required_k")
        self.assertEqual(float(row["candidate_anchor_wave_number"]), 2.7)
        self.assertGreater(float(row["cached_fs_at_candidate_anchor"]), math.exp(-1.0))
        self.assertEqual(float(row["candidate_anchor_threshold_crossed"]), 0.0)
        self.assertGreater(float(row["cached_structure_threshold_wave_number"]), 2.7)
        self.assertGreater(float(row["cached_structure_threshold_over_candidate"]), 1.0)
        self.assertAlmostEqual(float(row["cached_direct_threshold_wave_number"]), 4.8)
        self.assertGreater(float(row["cached_direct_threshold_over_candidate"]), 1.5)
        self.assertEqual(float(row["cached_direct_root_bracketed"]), 1.0)
        self.assertEqual(float(row["cached_alpha_anchor_rescue_ready"]), 0.0)
        self.assertEqual(float(row["post_rescue_real_closed_loop_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "cached_direct_anchor_wave_number_higher_than_protocol")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_direct_alpha_curve_audit_marks_cached_curve_not_event_clock(self):
        root_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "cached_direct_threshold_wave_number": 4.8,
                "cached_direct_root_bracketed": 1.0,
                "cached_anchor_stage": "cached_direct_anchor_root_refines_required_k",
            }
        ]
        curve_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "time_code": "tc05",
                "lag_time": 0.1,
                "direct_alpha_wave_number": 4.8,
                "direct_alpha_fs": 0.98,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "time_code": "tc10",
                "lag_time": 1.1,
                "direct_alpha_wave_number": 4.8,
                "direct_alpha_fs": 0.88,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "time_code": "tc40",
                "lag_time": 1500000.0,
                "direct_alpha_wave_number": 4.8,
                "direct_alpha_fs": math.exp(-1.0),
            },
        ]

        row = glassbench_direct_alpha_curve_audit(
            audit_id="glassbench_direct_alpha_curve",
            root_rows=root_rows,
            curve_rows=curve_rows,
        )[0]

        self.assertEqual(row["direct_alpha_curve_stage"], "cached_direct_alpha_curve_ready_event_clock_blocked")
        self.assertEqual(float(row["direct_alpha_wave_number"]), 4.8)
        self.assertEqual(float(row["lag_count"]), 3.0)
        self.assertAlmostEqual(float(row["threshold_crossing_lag_time"]), 1500000.0)
        self.assertEqual(row["threshold_crossing_time_code"], "tc40")
        self.assertEqual(float(row["alpha_threshold_crossed"]), 1.0)
        self.assertEqual(float(row["strictly_monotone_decay"]), 1.0)
        self.assertEqual(float(row["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "event_clock_trajectory")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_direct_alpha_transport_coupling_matches_crossing_observable(self):
        direct_rows = [
            {
                "audit_id": "glassbench_ka2d_direct_alpha_curve",
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "direct_alpha_wave_number": 4.8,
                "alpha_threshold_crossed": 1.0,
                "threshold_crossing_lag_time": 1500000.0,
                "threshold_crossing_time_code": "tc40",
                "event_clock_trajectory_ready": 0.0,
                "real_pe_inversion_ready": 0.0,
                "direct_alpha_curve_stage": "cached_direct_alpha_curve_ready_event_clock_blocked",
            }
        ]
        observable_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "time_code": "tc40",
                "lag_time": 1500000.0,
                "official_msd": 0.9747508405755333,
                "initial_reference_msd": 0.9747508405755335,
                "official_ngp_2d": 2.1239947887392923,
                "official_displacement_observable_reproducible": 1.0,
                "event_clock_trajectory_ready": 0.0,
                "observable_semantics_stage": "official_displacement_observable_reproduced",
            }
        ]

        row = glassbench_direct_alpha_transport_coupling_audit(
            audit_id="glassbench_direct_alpha_transport",
            direct_alpha_rows=direct_rows,
            observable_semantics_rows=observable_rows,
            dimension=2,
        )[0]

        self.assertEqual(row["transport_coupling_stage"], "cached_direct_alpha_transport_proxy_ready_event_clock_blocked")
        self.assertEqual(float(row["direct_alpha_transport_proxy_ready"]), 1.0)
        self.assertAlmostEqual(float(row["tau_alpha_direct"]), 1500000.0)
        self.assertAlmostEqual(float(row["matched_msd"]), 0.9747508405755333)
        self.assertAlmostEqual(float(row["apparent_diffusion_coefficient"]), 1.6245847342925555e-7)
        self.assertAlmostEqual(float(row["apparent_stokes_einstein_product"]), 0.24368771014388332)
        self.assertAlmostEqual(float(row["matched_ngp_2d"]), 2.1239947887392923)
        self.assertEqual(float(row["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "event_clock_trajectory")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_direct_alpha_pe_feasibility_bound_constrains_jump_variance(self):
        transport_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "direct_alpha_wave_number": 4.7984485103142,
                "tau_alpha_direct": 1500000.0,
                "matched_msd": 0.9747508405755333,
                "matched_ngp_2d": 2.1239947887392923,
                "apparent_diffusion_coefficient": 1.6245847342925555e-7,
                "apparent_stokes_einstein_product": 0.24368771014388332,
                "direct_alpha_transport_proxy_ready": 1.0,
                "real_pe_inversion_ready": 0.0,
            }
        ]

        row = glassbench_direct_alpha_pe_feasibility_bound(
            audit_id="glassbench_direct_alpha_pe_bound",
            transport_rows=transport_rows,
            reference_jump_variance_fraction=0.2,
        )[0]

        self.assertEqual(row["pe_feasibility_stage"], "direct_alpha_transport_bounds_pe_but_event_clock_missing")
        self.assertEqual(float(row["pe_feasibility_bound_ready"]), 1.0)
        self.assertEqual(float(row["full_msd_jump_variance_feasible"]), 0.0)
        self.assertAlmostEqual(float(row["jump_variance_upper_bound"]), 0.4855550202214052)
        self.assertAlmostEqual(float(row["jump_variance_upper_over_msd"]), 0.4981324457588704)
        self.assertAlmostEqual(float(row["reference_jump_variance"]), 0.19495016811510667)
        self.assertAlmostEqual(float(row["reference_exchange_mean"]), 600000.0)
        self.assertAlmostEqual(float(row["reference_persistence_mean"]), 1409293.5403982885)
        self.assertAlmostEqual(float(row["reference_persistence_exchange_ratio"]), 2.3488225673304806)
        self.assertEqual(float(row["conditional_pe_inference_ready"]), 1.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "event_clock_jump_variance")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_direct_alpha_displacement_tail_bound_marks_segmentation_target(self):
        pe_bound_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "direct_alpha_wave_number": 4.7984485103142,
                "tau_alpha_direct": 1500000.0,
                "matched_msd": 0.9747508405755333,
                "jump_variance_upper_bound": 0.4855550202214053,
                "full_msd_jump_variance_feasible": 0.0,
                "pe_feasibility_bound_ready": 1.0,
                "real_pe_inversion_ready": 0.0,
            }
        ]
        displacement_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "time_code": "tc40",
                "sample_count": 25800.0,
                "q_all": 0.48737542028776676,
                "q_bound": 0.4855550202214053,
                "fraction_q_le_bound": 0.7650387596899225,
                "fraction_q_gt_bound": 0.2349612403100775,
                "mean_q_above_bound": 1.7929841231781933,
                "q_median": 0.06625069389999988,
                "q_p90": 1.3898784621800007,
                "q_p95": 2.373878811687498,
            }
        ]

        row = glassbench_direct_alpha_displacement_tail_bound(
            audit_id="glassbench_direct_alpha_displacement_tail_bound",
            pe_bound_rows=pe_bound_rows,
            displacement_rows=displacement_rows,
            min_tail_fraction=0.05,
        )[0]

        self.assertEqual(row["tail_bound_stage"], "direct_displacement_tail_exceeds_pe_single_event_bound")
        self.assertEqual(float(row["tail_bound_ready"]), 1.0)
        self.assertAlmostEqual(float(row["fraction_q_gt_bound"]), 0.2349612403100775)
        self.assertAlmostEqual(float(row["mean_q_above_bound"]), 1.7929841231781933)
        self.assertAlmostEqual(float(row["mean_q_above_over_bound"]), 3.692648718492544)
        self.assertAlmostEqual(float(row["q_all_over_bound"]), 1.003749111821625)
        self.assertEqual(float(row["event_segmentation_required"]), 1.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "event_segmentation")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_direct_alpha_multilag_crossing_canary_blocks_replica_axis_clock(self):
        pe_bound_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "direct_alpha_wave_number": 4.7984485103142,
                "tau_alpha_direct": 1500000.0,
                "jump_variance_upper_bound": 0.4855550202214053,
                "pe_feasibility_bound_ready": 1.0,
            }
        ]
        crossing_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "axis0_semantics": "isoconfigurational_trajectory_replicates",
                "time_codes": "tc05;tc10;tc15;tc20;tc25;tc30;tc35;tc40",
                "lag_times": "0.1;1.1;11.64;122.47;1288.41;13554.0;142587.0;1500000.0",
                "sample_count": 25800.0,
                "above_bound_fractions_by_lag": "0;0;0;0;3.875968992248062e-05;0.001434108527131783;0.022325581395348838;0.2349612403100775",
                "ever_crossed_fraction": 0.24007751937984495,
                "never_crossed_fraction": 0.759922480620155,
                "post_crossing_recross_fraction": 0.23599320882852293,
                "first_crossing_fractions_by_time_code": "tc25:3.875968992248062e-05;tc30:0.001434108527131783;tc35:0.02135658914728682;tc40:0.21724806201550387",
                "first_crossing_q_mean": 1.671772382881059,
                "first_crossing_q_median": 1.1118363749999993,
                "first_crossing_q_p90": 3.4652622499604004,
            }
        ]

        row = glassbench_direct_alpha_multilag_crossing_canary(
            audit_id="glassbench_direct_alpha_multilag_crossing_canary",
            pe_bound_rows=pe_bound_rows,
            crossing_rows=crossing_rows,
            min_crossing_fraction=0.05,
        )[0]

        self.assertEqual(row["crossing_canary_stage"], "multilag_displacement_crossing_canary_ready_replica_axis_blocked")
        self.assertEqual(float(row["crossing_canary_ready"]), 1.0)
        self.assertAlmostEqual(float(row["ever_crossed_fraction"]), 0.24007751937984495)
        self.assertAlmostEqual(float(row["post_crossing_recross_fraction"]), 0.23599320882852293)
        self.assertAlmostEqual(float(row["first_crossing_q_mean_over_bound"]), 3.4430132801814253)
        self.assertEqual(float(row["event_segmentation_target_ready"]), 1.0)
        self.assertEqual(float(row["persistence_exchange_event_clock_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "frame_axis_is_isoconfigurational_replicates")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_direct_alpha_event_clock_contract_requires_true_time_axis(self):
        pe_bound_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "direct_alpha_wave_number": 4.7984485103142,
                "tau_alpha_direct": 1500000.0,
                "jump_variance_upper_bound": 0.4855550202214053,
                "conditional_pe_inference_ready": 1.0,
                "pe_feasibility_bound_ready": 1.0,
                "real_pe_inversion_ready": 0.0,
            }
        ]
        tail_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "q_all_over_bound": 1.003749111821625,
                "fraction_q_gt_bound": 0.2349612403100775,
                "mean_q_above_over_bound": 3.692648718492544,
                "event_segmentation_required": 1.0,
                "tail_bound_ready": 1.0,
                "primary_blocker": "event_segmentation",
            }
        ]
        crossing_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "axis0_semantics": "isoconfigurational_trajectory_replicates",
                "axis0_is_isoconfigurational_replica": 1.0,
                "time_codes": "tc05;tc10;tc15;tc20;tc25;tc30;tc35;tc40",
                "lag_times": "0.1;1.1;11.64;122.47;1288.41;13554.0;142587.0;1500000.0",
                "sample_count": 25800.0,
                "ever_crossed_fraction": 0.24007751937984495,
                "post_crossing_recross_fraction": 0.23599320882852293,
                "first_crossing_q_mean_over_bound": 3.4430132801814253,
                "event_segmentation_target_ready": 1.0,
                "crossing_canary_ready": 1.0,
                "persistence_exchange_event_clock_ready": 0.0,
                "primary_blocker": "frame_axis_is_isoconfigurational_replicates",
            }
        ]

        row = glassbench_direct_alpha_event_clock_extraction_contract(
            audit_id="glassbench_direct_alpha_event_clock_contract",
            pe_bound_rows=pe_bound_rows,
            tail_rows=tail_rows,
            crossing_rows=crossing_rows,
        )[0]

        self.assertEqual(row["event_clock_contract_stage"], "segmentation_target_ready_true_event_clock_missing")
        self.assertEqual(float(row["conditional_pe_inference_ready"]), 1.0)
        self.assertEqual(float(row["direct_displacement_tail_ready"]), 1.0)
        self.assertEqual(float(row["event_segmentation_target_ready"]), 1.0)
        self.assertEqual(float(row["cached_replica_ladder_ready"]), 1.0)
        self.assertEqual(float(row["axis0_is_physical_time"]), 0.0)
        self.assertEqual(float(row["requires_true_time_trajectory"]), 1.0)
        self.assertEqual(float(row["event_clock_extraction_ready"]), 0.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "physical_time_trajectory_axis")
        self.assertIn("positions[time,particle,dimension]", row["required_arrays"])
        self.assertIn("isoconfigurational_replica_axis", row["forbidden_substitutes"])
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_sparse_lag_event_clock_audit_promotes_candidate_not_real_inversion(self):
        cache_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "structure_id": "151",
                "time_code": code,
                "lag_time": lag,
                "positions_shape": "20x1290x2",
                "initial_reference_positions_cached": 1.0,
                "particle_resolved_positions_cached": 1.0,
                "max_initial_position_mismatch": 0.0,
            }
            for code, lag in [
                ("tc05", 0.1),
                ("tc10", 1.1),
                ("tc15", 11.64),
                ("tc20", 122.47),
                ("tc25", 1288.41),
                ("tc30", 13554.0),
                ("tc35", 142587.0),
                ("tc40", 1500000.0),
            ]
        ]
        contract_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "q_bound": 0.4855550202214053,
                "event_segmentation_target_ready": 1.0,
                "cached_replica_ladder_ready": 1.0,
                "real_pe_inversion_ready": 0.0,
            }
        ]

        row = glassbench_sparse_lag_event_clock_audit(
            audit_id="glassbench_sparse_lag_event_clock",
            cache_rows=cache_rows,
            contract_rows=contract_rows,
            required_time_codes=("tc05", "tc10", "tc15", "tc20", "tc25", "tc30", "tc35", "tc40"),
        )[0]

        self.assertEqual(row["sparse_lag_event_clock_stage"], "sparse_lag_tensor_ready_replica_identity_unverified")
        self.assertEqual(float(row["physical_lag_tensor_ready"]), 1.0)
        self.assertEqual(float(row["same_initial_structure_verified"]), 1.0)
        self.assertEqual(float(row["same_shape_across_lags"]), 1.0)
        self.assertEqual(float(row["time_code_coverage_fraction"]), 1.0)
        self.assertEqual(float(row["coarse_event_clock_candidate_ready"]), 1.0)
        self.assertEqual(float(row["replica_identity_alignment_ready"]), 0.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "replica_identity_alignment")
        self.assertEqual(row["event_clock_resolution"], "sparse_lag_interval")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_interval_censored_first_crossing_clock_quantifies_sparse_candidate(self):
        sparse_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "coarse_event_clock_candidate_ready": 1.0,
                "event_clock_resolution": "sparse_lag_interval",
                "real_pe_inversion_ready": 0.0,
            }
        ]
        crossing_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "lag_times": "0.1;1.1;11.64;122.47;1288.41;13554.0;142587.0;1500000.0",
                "time_codes": "tc05;tc10;tc15;tc20;tc25;tc30;tc35;tc40",
                "ever_crossed_fraction": 0.24007751937984495,
                "never_crossed_fraction": 0.759922480620155,
                "first_crossing_fractions_by_time_code": "tc25:3.875968992248062e-05;tc30:0.001434108527131783;tc35:0.02135658914728682;tc40:0.21724806201550387",
            }
        ]

        row = glassbench_interval_censored_first_crossing_clock(
            audit_id="glassbench_interval_censored_first_crossing_clock",
            sparse_lag_rows=sparse_rows,
            crossing_rows=crossing_rows,
        )[0]

        self.assertEqual(row["interval_clock_stage"], "interval_censored_persistence_clock_candidate")
        self.assertEqual(float(row["interval_clock_candidate_ready"]), 1.0)
        self.assertAlmostEqual(float(row["crossed_fraction"]), 0.24007751937984495)
        self.assertAlmostEqual(float(row["right_censored_fraction"]), 0.759922480620155)
        self.assertAlmostEqual(float(row["mean_first_crossing_lower_bound"]), 130241.55354213755)
        self.assertAlmostEqual(float(row["mean_first_crossing_upper_bound"]), 1370127.2559589928)
        self.assertAlmostEqual(float(row["mean_first_crossing_midpoint"]), 750184.4047505651)
        self.assertGreater(float(row["mean_interval_width"]), 1.0e6)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "interval_censoring_and_replica_identity")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_interval_censored_first_crossing_clock_blocks_missing_lag_distribution(self):
        sparse_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "structure_id": "3",
                "coarse_event_clock_candidate_ready": 0.0,
                "event_clock_resolution": "none",
            }
        ]
        crossing_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "structure_id": "3",
                "lag_times": "none",
                "time_codes": "none",
                "ever_crossed_fraction": 0.0,
                "never_crossed_fraction": 0.0,
                "first_crossing_fractions_by_time_code": "none",
            }
        ]

        row = glassbench_interval_censored_first_crossing_clock(
            audit_id="glassbench_interval_censored_first_crossing_clock",
            sparse_lag_rows=sparse_rows,
            crossing_rows=crossing_rows,
        )[0]

        self.assertEqual(row["interval_clock_stage"], "interval_clock_sparse_lag_upstream_incomplete")
        self.assertEqual(float(row["interval_clock_candidate_ready"]), 0.0)
        self.assertEqual(float(row["latest_lag_time"]), 0.0)
        self.assertEqual(row["primary_blocker"], "sparse_lag_event_clock")
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)

    def test_glassbench_interval_censored_persistence_fit_estimates_exponential_scale(self):
        interval_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "interval_clock_candidate_ready": 1.0,
                "first_crossing_intervals": "tc25:122.47:1288.4100000000001:3.8759689922480622e-05;tc30:1288.4100000000001:13554:0.001434108527131783;tc35:13554:142587:0.02135658914728682;tc40:142587:1500000:0.21724806201550387",
                "right_censored_fraction": 0.759922480620155,
                "latest_lag_time": 1500000.0,
            }
        ]
        direct_alpha_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "threshold_crossing_lag_time": 1500000.0,
                "alpha_threshold_crossed": 1.0,
            }
        ]

        row = glassbench_interval_censored_persistence_fit(
            fit_id="glassbench_interval_censored_persistence_fit",
            interval_clock_rows=interval_rows,
            direct_alpha_rows=direct_alpha_rows,
        )[0]

        self.assertEqual(row["persistence_fit_stage"], "interval_censored_exponential_persistence_fit_ready")
        self.assertEqual(float(row["persistence_fit_ready"]), 1.0)
        self.assertAlmostEqual(float(row["exponential_rate_mle"]), 1.827224516438915e-07, delta=1e-15)
        self.assertAlmostEqual(float(row["exponential_mean_persistence_time"]), 5472781.209990023, delta=1.0)
        self.assertAlmostEqual(float(row["predicted_crossed_fraction_at_latest_lag"]), 0.2397315447385533, delta=1e-7)
        self.assertAlmostEqual(float(row["observed_crossed_fraction"]), 0.24007751937984495)
        self.assertAlmostEqual(float(row["mean_persistence_over_tau_alpha_direct"]), 3.6485208066600154, delta=1e-6)
        self.assertEqual(float(row["exchange_clock_ready"]), 0.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "exchange_clock_and_replica_identity")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_finite_exchange_envelope_turns_censored_fit_into_followup_horizon(self):
        fit_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "persistence_fit_ready": 1.0,
                "exponential_mean_persistence_time": 5472781.231591796,
                "tau_alpha_direct": 1500000.0,
                "latest_lag_time": 1500000.0,
                "primary_blocker": "exchange_clock_and_replica_identity",
            }
        ]

        row = glassbench_finite_exchange_falsification_envelope(
            envelope_id="glassbench_finite_exchange_falsification_envelope",
            persistence_fit_rows=fit_rows,
            max_exchange_mean_over_tau_alpha=1.0,
            min_exchange_events_for_gaussian_recovery=25.0,
        )[0]

        self.assertEqual(row["envelope_stage"], "finite_exchange_falsification_horizon_ready")
        self.assertEqual(float(row["envelope_ready"]), 1.0)
        self.assertAlmostEqual(row["conditional_exchange_mean_upper_bound"], 1500000.0)
        self.assertAlmostEqual(row["conditional_persistence_exchange_ratio_lower_bound"], 3.6485208210611972)
        self.assertAlmostEqual(row["gaussian_recovery_lag_upper_bound"], 42972781.2315918, delta=1e-3)
        self.assertAlmostEqual(row["required_followup_lag_multiplier_over_current"], 28.6485208210612, delta=1e-12)
        self.assertEqual(float(row["current_window_has_gaussian_recovery_power"]), 0.0)
        self.assertEqual(float(row["late_ngp_followup_ready"]), 0.0)
        self.assertEqual(float(row["exchange_clock_ready"]), 0.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "late_ngp_followup_and_exchange_clock")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_late_recovery_protocol_classifies_support_and_rejection(self):
        envelope_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "envelope_ready": 1.0,
                "gaussian_recovery_lag_upper_bound": 42972781.2315918,
                "conditional_persistence_exchange_ratio_lower_bound": 3.6485208210611972,
                "primary_blocker": "late_ngp_followup_and_exchange_clock",
            }
        ]
        support = glassbench_late_recovery_falsification_protocol(
            protocol_id="glassbench_late_recovery_protocol",
            envelope_rows=envelope_rows,
            late_observable_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "structure_id": "151",
                    "observed_lag_time": 50000000.0,
                    "observed_late_ngp": 0.012,
                    "observed_tail_gaussian_recovery": 1.0,
                    "static_gamma_late_ngp_plateau": 0.24,
                }
            ],
            max_finite_exchange_late_ngp=0.05,
            min_static_plateau_rejection_gap=0.05,
        )[0]
        reject = glassbench_late_recovery_falsification_protocol(
            protocol_id="glassbench_late_recovery_protocol",
            envelope_rows=envelope_rows,
            late_observable_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "structure_id": "151",
                    "observed_lag_time": 50000000.0,
                    "observed_late_ngp": 0.16,
                    "observed_tail_gaussian_recovery": 0.0,
                    "static_gamma_late_ngp_plateau": 0.24,
                }
            ],
            max_finite_exchange_late_ngp=0.05,
            min_static_plateau_rejection_gap=0.05,
        )[0]

        self.assertEqual(support["late_recovery_stage"], "finite_exchange_late_recovery_supported")
        self.assertEqual(float(support["mechanism_selection_ready"]), 1.0)
        self.assertEqual(float(support["finite_exchange_supported"]), 1.0)
        self.assertEqual(float(support["finite_exchange_rejected"]), 0.0)
        self.assertEqual(float(support["static_disorder_rejected"]), 1.0)
        self.assertEqual(float(support["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(support["primary_blocker"], "exchange_clock")
        self.assertEqual(float(support["thermodynamic_claim_allowed"]), 0.0)

        self.assertEqual(reject["late_recovery_stage"], "finite_exchange_late_recovery_failed")
        self.assertEqual(float(reject["mechanism_selection_ready"]), 1.0)
        self.assertEqual(float(reject["finite_exchange_supported"]), 0.0)
        self.assertEqual(float(reject["finite_exchange_rejected"]), 1.0)
        self.assertEqual(float(reject["static_disorder_rejected"]), 0.0)
        self.assertEqual(reject["primary_blocker"], "late_gaussian_recovery")

    def test_glassbench_late_recovery_ingestion_contract_requires_horizon_and_uncertainty(self):
        envelope_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "envelope_ready": 1.0,
                "gaussian_recovery_lag_upper_bound": 42972781.2315918,
            }
        ]
        rows = glassbench_late_recovery_ingestion_contract(
            contract_id="glassbench_late_recovery_ingestion_contract",
            envelope_rows=envelope_rows,
            candidate_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "structure_id": "151",
                    "observed_lag_time": 50000000.0,
                    "observed_late_ngp": 0.012,
                    "sigma_late_ngp": 0.003,
                    "observed_tail_gaussian_recovery": 1.0,
                    "sigma_tail_recovery": 0.05,
                    "machine_readable": 1.0,
                    "shared_time_units": 1.0,
                    "source_trajectory_identity": "KA2D_0.23_structure151_member_ensemble",
                },
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "structure_id": "151",
                    "observed_lag_time": 1500000.0,
                    "observed_late_ngp": 0.02,
                    "sigma_late_ngp": 0.004,
                    "observed_tail_gaussian_recovery": 1.0,
                    "sigma_tail_recovery": 0.05,
                    "machine_readable": 1.0,
                    "shared_time_units": 1.0,
                    "source_trajectory_identity": "KA2D_0.23_structure151_member_ensemble",
                },
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "structure_id": "151",
                    "observed_lag_time": 50000000.0,
                    "observed_late_ngp": 0.02,
                    "observed_tail_gaussian_recovery": 1.0,
                    "machine_readable": 1.0,
                    "shared_time_units": 1.0,
                    "source_trajectory_identity": "KA2D_0.23_structure151_member_ensemble",
                },
            ],
        )

        by_stage = {row["candidate_id"]: row for row in rows}
        ready = by_stage["KA2D:0.23:151:0"]
        short = by_stage["KA2D:0.23:151:1"]
        no_uncertainty = by_stage["KA2D:0.23:151:2"]

        self.assertEqual(ready["late_recovery_ingestion_stage"], "late_recovery_observation_ingestion_ready")
        self.assertEqual(float(ready["late_recovery_observation_ready"]), 1.0)
        self.assertEqual(float(ready["horizon_satisfied"]), 1.0)
        self.assertEqual(float(ready["uncertainty_ready"]), 1.0)
        self.assertEqual(ready["primary_blocker"], "none")

        self.assertEqual(short["late_recovery_ingestion_stage"], "late_recovery_horizon_incomplete")
        self.assertEqual(float(short["late_recovery_observation_ready"]), 0.0)
        self.assertEqual(short["primary_blocker"], "late_recovery_horizon")

        self.assertEqual(no_uncertainty["late_recovery_ingestion_stage"], "late_recovery_uncertainty_incomplete")
        self.assertEqual(no_uncertainty["missing_uncertainty_columns"], "sigma_late_ngp;sigma_tail_recovery")
        self.assertEqual(no_uncertainty["primary_blocker"], "sigma_late_ngp")

    def test_glassbench_late_recovery_timecode_target_maps_required_lag_to_next_cache(self):
        envelope_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "envelope_ready": 1.0,
                "gaussian_recovery_lag_upper_bound": 42972781.2315918,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "structure_id": "3",
                "envelope_ready": 0.0,
                "gaussian_recovery_lag_upper_bound": 0.0,
            },
        ]
        interval_clock_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "interval_clock_candidate_ready": 1.0,
                "time_codes": "tc05;tc10;tc15;tc20;tc25;tc30;tc35;tc40",
                "lag_times": "0.10000000000000001;1.1000000000000001;11.640000000000001;122.47;1288.4100000000001;13554;142587;1500000",
            },
        ]

        rows = glassbench_late_recovery_timecode_target(
            target_id="glassbench_late_recovery_timecode_target",
            envelope_rows=envelope_rows,
            interval_clock_rows=interval_clock_rows,
        )

        by_key = {(row["system_id"], row["temperature"], row["structure_id"]): row for row in rows}
        ready = by_key[("KA2D", "0.23", "151")]
        incomplete = by_key[("KA2D", "0.30", "3")]

        self.assertEqual(ready["timecode_target_stage"], "late_recovery_timecode_target_ready")
        self.assertEqual(float(ready["timecode_target_ready"]), 1.0)
        self.assertEqual(ready["current_max_time_code"], "tc40")
        self.assertAlmostEqual(float(ready["current_max_lag_time"]), 1500000.0)
        self.assertAlmostEqual(float(ready["required_followup_lag_time"]), 42972781.2315918, delta=1e-3)
        self.assertEqual(ready["target_time_code"], "tc50")
        self.assertAlmostEqual(float(ready["target_lag_time"]), 166002226.81761542, delta=1e-3)
        self.assertAlmostEqual(float(ready["current_lag_over_required"]), 0.034905816123841256, delta=1e-12)
        self.assertGreater(float(ready["target_lag_over_required"]), 3.8)
        self.assertAlmostEqual(float(ready["timecode_log_lag_step"]), 2.3532680473611762, delta=1e-12)
        self.assertEqual(float(ready["late_recovery_observation_ready"]), 0.0)
        self.assertEqual(float(ready["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ready["primary_blocker"], "late_recovery_time_code_cache")
        self.assertEqual(
            ready["next_required_action"],
            "extract_or_cache_glassbench_time_code_tc50_late_recovery_observables",
        )
        self.assertEqual(float(ready["thermodynamic_claim_allowed"]), 0.0)

        self.assertEqual(incomplete["timecode_target_stage"], "late_recovery_timecode_target_upstream_incomplete")
        self.assertEqual(incomplete["primary_blocker"], "finite_exchange_envelope")

    def test_glassbench_late_recovery_cache_request_contract_marks_tc50_metadata_gap(self):
        timecode_target_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "timecode_target_ready": 1.0,
                "required_followup_lag_time": 42972781.2315918,
                "current_max_time_code": "tc40",
                "current_max_lag_time": 1500000.0,
                "target_time_code": "tc50",
                "target_lag_time": 166002226.81761542,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "structure_id": "3",
                "timecode_target_ready": 0.0,
                "target_time_code": "none",
                "target_lag_time": 0.0,
            },
        ]
        multilag_target_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "selected_structure_id": "151",
                "selected_time_codes": "tc05;tc10;tc15;tc20;tc25;tc30;tc35;tc40",
                "target_members": (
                    "T0.23/test/N1290T0.23_151_tc05.npz;"
                    "T0.23/test/N1290T0.23_151_tc10.npz;"
                    "T0.23/test/N1290T0.23_151_tc15.npz;"
                    "T0.23/test/N1290T0.23_151_tc20.npz;"
                    "T0.23/test/N1290T0.23_151_tc25.npz;"
                    "T0.23/test/N1290T0.23_151_tc30.npz;"
                    "T0.23/test/N1290T0.23_151_tc35.npz;"
                    "T0.23/test/N1290T0.23_151_tc40.npz"
                ),
                "target_member_md5s": "md5-05;md5-10;md5-15;md5-20;md5-25;md5-30;md5-35;md5-40",
                "target_lag_times": "0.1;1.1;11.64;122.47;1288.41;13554.0;142587.0;1500000.0",
            },
        ]
        cache_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "time_code": "tc40",
                "target_member": "T0.23/test/N1290T0.23_151_tc40.npz",
                "target_member_md5": "md5-40",
                "particle_cache_path": (
                    "data/third_party/glassbench/particle_cache/"
                    "glassbench_ka2d_T0_23_N1290T0.23_151_tc40_positions.npz"
                ),
                "particle_resolved_positions_cached": 1.0,
            },
        ]

        rows = glassbench_late_recovery_cache_request_contract(
            contract_id="glassbench_late_recovery_cache_request_contract",
            timecode_target_rows=timecode_target_rows,
            multilag_target_rows=multilag_target_rows,
            cache_rows=cache_rows,
        )

        by_key = {(row["system_id"], row["temperature"], row["structure_id"]): row for row in rows}
        request = by_key[("KA2D", "0.23", "151")]
        incomplete = by_key[("KA2D", "0.30", "3")]

        self.assertEqual(request["cache_request_stage"], "late_recovery_member_metadata_required")
        self.assertEqual(float(request["cache_request_ready"]), 1.0)
        self.assertEqual(request["source_path"], "GlassBench/KA2D_trajectories/T0.23.tar.xz")
        self.assertEqual(request["current_max_time_code"], "tc40")
        self.assertAlmostEqual(float(request["current_max_lag_time"]), 1500000.0)
        self.assertEqual(request["target_time_code"], "tc50")
        self.assertAlmostEqual(float(request["target_lag_time"]), 166002226.81761542, delta=1e-3)
        self.assertEqual(request["inferred_target_member"], "T0.23/test/N1290T0.23_151_tc50.npz")
        self.assertEqual(float(request["inferred_member_path_ready"]), 1.0)
        self.assertEqual(float(request["official_target_member_metadata_ready"]), 0.0)
        self.assertEqual(request["target_member_md5"], "none")
        self.assertEqual(
            request["expected_particle_cache_path"],
            "data/third_party/glassbench/particle_cache/glassbench_ka2d_T0_23_N1290T0.23_151_tc50_positions.npz",
        )
        self.assertEqual(float(request["particle_cache_ready"]), 0.0)
        self.assertEqual(float(request["late_recovery_observable_ready"]), 0.0)
        self.assertEqual(float(request["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(request["primary_blocker"], "late_recovery_npz_member_metadata")
        self.assertEqual(
            request["next_required_action"],
            "verify_glassbench_archive_contains_time_code_tc50_for_structure_151",
        )
        self.assertEqual(float(request["thermodynamic_claim_allowed"]), 0.0)

        self.assertEqual(incomplete["cache_request_stage"], "late_recovery_timecode_target_incomplete")
        self.assertEqual(incomplete["inferred_target_member"], "none")
        self.assertEqual(float(incomplete["inferred_member_path_ready"]), 0.0)
        self.assertEqual(incomplete["primary_blocker"], "late_recovery_timecode_target")

    def test_glassbench_late_recovery_membership_probe_contract_records_prefix_absence(self):
        cache_request_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "cache_request_ready": 1.0,
                "target_time_code": "tc50",
                "target_lag_time": 166002226.81761542,
                "inferred_target_member": "T0.23/test/N1290T0.23_151_tc50.npz",
            },
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "structure_id": "3",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "cache_request_ready": 0.0,
                "target_time_code": "none",
                "inferred_target_member": "none",
            },
        ]
        member_index_manifest = {
            "compressed_probe_bytes": 12582912,
            "tar_probe_limit_bytes": 25165824,
            "entries": [
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                    "compressed_probe_range_start": 2980602255,
                    "compressed_probe_range_end": 2993185166,
                    "compressed_probe_bytes": 12582912,
                    "tar_probe_bytes": 25165824,
                    "npz_member_count_in_probe": 55,
                    "npz_members": [
                        {"name": "T0.23/test/N1290T0.23_151_tc05.npz", "size_bytes": 465710},
                        {"name": "T0.23/test/N1290T0.23_151_tc10.npz", "size_bytes": 465710},
                        {"name": "T0.23/test/N1290T0.23_151_tc15.npz", "size_bytes": 465710},
                        {"name": "T0.23/test/N1290T0.23_151_tc20.npz", "size_bytes": 465710},
                        {"name": "T0.23/test/N1290T0.23_151_tc25.npz", "size_bytes": 465710},
                        {"name": "T0.23/test/N1290T0.23_151_tc30.npz", "size_bytes": 465710},
                        {"name": "T0.23/test/N1290T0.23_151_tc35.npz", "size_bytes": 465710},
                        {"name": "T0.23/test/N1290T0.23_151_tc40.npz", "size_bytes": 465710},
                    ],
                }
            ],
        }

        rows = glassbench_late_recovery_membership_probe_contract(
            probe_id="glassbench_late_recovery_membership_probe_contract",
            cache_request_rows=cache_request_rows,
            member_index_manifest=member_index_manifest,
        )

        by_key = {(row["system_id"], row["temperature"], row["structure_id"]): row for row in rows}
        cold = by_key[("KA2D", "0.23", "151")]
        warm = by_key[("KA2D", "0.30", "3")]

        self.assertEqual(cold["membership_probe_stage"], "late_recovery_target_absent_from_extended_prefix")
        self.assertEqual(float(cold["membership_probe_ready"]), 1.0)
        self.assertEqual(cold["target_time_code"], "tc50")
        self.assertEqual(cold["inferred_target_member"], "T0.23/test/N1290T0.23_151_tc50.npz")
        self.assertEqual(float(cold["target_member_visible_in_probe"]), 0.0)
        self.assertEqual(float(cold["same_structure_member_count_in_probe"]), 8.0)
        self.assertEqual(cold["same_structure_visible_time_codes"], "tc05;tc10;tc15;tc20;tc25;tc30;tc35;tc40")
        self.assertEqual(cold["max_visible_time_code"], "tc40")
        self.assertAlmostEqual(float(cold["compressed_probe_bytes"]), 12582912.0)
        self.assertAlmostEqual(float(cold["tar_probe_bytes"]), 25165824.0)
        self.assertEqual(float(cold["late_recovery_observable_ready"]), 0.0)
        self.assertEqual(cold["primary_blocker"], "late_recovery_member_index_depth")
        self.assertEqual(
            cold["next_required_action"],
            "extend_glassbench_tar_prefix_probe_for_structure_151_tc50_or_full_index",
        )
        self.assertEqual(float(cold["thermodynamic_claim_allowed"]), 0.0)

        self.assertEqual(warm["membership_probe_stage"], "late_recovery_cache_request_incomplete")
        self.assertEqual(warm["primary_blocker"], "late_recovery_cache_request")

    def test_glassbench_late_recovery_public_timecode_ceiling_blocks_unpublished_tc50(self):
        timecode_target_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "timecode_target_ready": 1.0,
                "current_max_time_code": "tc40",
                "current_max_lag_time": 1500000.0,
                "target_time_code": "tc50",
                "target_lag_time": 166002226.81761542,
                "required_followup_lag_time": 42972781.2315918,
            }
        ]
        semantics_manifest = {
            "entries": [
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                    "members": [
                        {
                            "structure_id": 151,
                            "time_code": "tc05",
                            "lag_time": 0.1,
                            "member": "T0.23/test/N1290T0.23_151_tc05.npz",
                        },
                        {
                            "structure_id": 151,
                            "time_code": "tc40",
                            "lag_time": 1500000.0,
                            "member": "T0.23/test/N1290T0.23_151_tc40.npz",
                        },
                        {
                            "structure_id": 152,
                            "time_code": "tc40",
                            "lag_time": 1500000.0,
                            "member": "T0.23/test/N1290T0.23_152_tc40.npz",
                        },
                    ],
                }
            ]
        }
        membership_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "membership_probe_ready": 1.0,
                "target_member_visible_in_probe": 0.0,
                "max_visible_time_code": "tc40",
                "same_structure_visible_time_codes": "tc05;tc40",
                "membership_probe_stage": "late_recovery_target_absent_from_extended_prefix",
            }
        ]

        rows = glassbench_late_recovery_public_timecode_ceiling(
            ceiling_id="glassbench_late_recovery_public_timecode_ceiling",
            timecode_target_rows=timecode_target_rows,
            semantics_manifest=semantics_manifest,
            membership_probe_rows=membership_rows,
        )

        row = rows[0]
        self.assertEqual(row["public_ceiling_stage"], "late_recovery_beyond_public_timecode_ceiling")
        self.assertEqual(float(row["public_ceiling_ready"]), 1.0)
        self.assertEqual(row["target_time_code"], "tc50")
        self.assertEqual(row["public_max_time_code"], "tc40")
        self.assertEqual(row["structure_max_time_code"], "tc40")
        self.assertAlmostEqual(float(row["public_max_lag_time"]), 1500000.0)
        self.assertAlmostEqual(float(row["target_lag_time"]), 166002226.81761542, delta=1e-3)
        self.assertGreater(float(row["target_lag_over_public_max"]), 100.0)
        self.assertEqual(float(row["target_time_code_published"]), 0.0)
        self.assertEqual(float(row["late_recovery_observation_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "public_glassbench_timecode_ceiling")
        self.assertEqual(
            row["next_required_action"],
            "obtain_new_glassbench_export_or_trajectory_beyond_tc40_for_late_recovery",
        )
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_censored_window_claim_audit_keeps_late_recovery_claim_blocked(self):
        public_ceiling_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "timecode_target_ready": 1.0,
                "target_time_code": "tc50",
                "target_lag_time": 166002226.81761542,
                "public_max_time_code": "tc40",
                "public_max_lag_time": 1500000.0,
                "target_lag_over_public_max": 110.66815121174359,
                "public_ceiling_stage": "late_recovery_beyond_public_timecode_ceiling",
                "primary_blocker": "public_glassbench_timecode_ceiling",
                "next_required_action": "obtain_new_glassbench_export_or_trajectory_beyond_tc40_for_late_recovery",
            }
        ]
        envelope_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "envelope_ready": 1.0,
                "tau_alpha_direct": 1500000.0,
                "latest_lag_time": 1500000.0,
                "gaussian_recovery_lag_upper_bound": 166002226.81761542,
            }
        ]

        rows = glassbench_censored_window_claim_audit(
            audit_id="glassbench_censored_window_claim_audit",
            public_ceiling_rows=public_ceiling_rows,
            finite_exchange_envelope_rows=envelope_rows,
        )

        row = rows[0]
        self.assertEqual(row["censored_window_stage"], "alpha_anchor_ready_late_recovery_censored")
        self.assertEqual(float(row["alpha_anchor_window_ready"]), 1.0)
        self.assertAlmostEqual(float(row["public_window_fraction_of_target_lag"]), 1500000.0 / 166002226.81761542)
        self.assertGreater(float(row["target_lag_over_public_max"]), 100.0)
        self.assertEqual(float(row["short_window_dynamic_claim_allowed"]), 1.0)
        self.assertEqual(float(row["alpha_relaxation_claim_allowed"]), 1.0)
        self.assertEqual(float(row["late_gaussian_recovery_claim_allowed"]), 0.0)
        self.assertEqual(float(row["static_vs_finite_exchange_rejection_ready"]), 0.0)
        self.assertEqual(float(row["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(row["allowed_public_claim_level"], "alpha_anchor_and_pre_late_dynamic_signatures")
        self.assertEqual(row["primary_blocker"], "public_glassbench_timecode_ceiling")
        self.assertEqual(
            row["next_required_action"],
            "obtain_new_glassbench_export_or_trajectory_beyond_tc40_for_late_recovery",
        )

    def test_glassbench_sota_public_window_verdict_separates_supported_and_censored_claims(self):
        censored_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "alpha_anchor_window_ready": 1.0,
                "short_window_dynamic_claim_allowed": 1.0,
                "alpha_relaxation_claim_allowed": 1.0,
                "late_gaussian_recovery_claim_allowed": 0.0,
                "static_vs_finite_exchange_rejection_ready": 0.0,
                "public_window_fraction_of_target_lag": 0.009036023364,
                "target_lag_over_public_max": 110.66815121174359,
                "allowed_public_claim_level": "alpha_anchor_and_pre_late_dynamic_signatures",
                "primary_blocker": "public_glassbench_timecode_ceiling",
            }
        ]
        signature_rows = [
            {
                "signature": "self_intermediate_alpha",
                "phenomenon": "self_intermediate_scattering_alpha_relaxation",
                "model_support": 1.0,
                "literature_qualitative_support": 1.0,
            },
            {
                "signature": "late_gaussian_recovery",
                "phenomenon": "long_time_gaussian_recovery",
                "model_support": 1.0,
                "literature_qualitative_support": 1.0,
            },
            {
                "signature": "persistence_exchange_decoupling",
                "phenomenon": "persistence_exchange_decoupling",
                "model_support": 1.0,
                "literature_qualitative_support": 1.0,
            },
            {
                "signature": "thermodynamic_transition",
                "phenomenon": "configurational_entropy_and_ideal_glass_scope",
                "model_support": 0.0,
                "literature_qualitative_support": 1.0,
            },
        ]

        rows = glassbench_sota_public_window_verdict(
            verdict_id="glassbench_sota_public_window_verdict",
            censored_window_rows=censored_rows,
            dynamic_signature_rows=signature_rows,
        )

        by_signature = {row["signature"]: row for row in rows}
        alpha = by_signature["self_intermediate_alpha"]
        self.assertEqual(alpha["public_window_verdict_stage"], "public_window_sota_consistent")
        self.assertEqual(float(alpha["public_glassbench_claim_allowed"]), 1.0)
        self.assertEqual(float(alpha["late_recovery_required"]), 0.0)
        self.assertEqual(float(alpha["thermodynamic_claim_allowed"]), 0.0)

        recovery = by_signature["late_gaussian_recovery"]
        self.assertEqual(recovery["public_window_verdict_stage"], "public_window_censored_sota_unresolved")
        self.assertEqual(float(recovery["public_glassbench_claim_allowed"]), 0.0)
        self.assertEqual(float(recovery["late_recovery_required"]), 1.0)
        self.assertEqual(recovery["primary_blocker"], "public_glassbench_timecode_ceiling")

        pe = by_signature["persistence_exchange_decoupling"]
        self.assertEqual(pe["public_window_verdict_stage"], "mechanism_selection_censored_unresolved")
        self.assertEqual(float(pe["mechanism_rejection_ready"]), 0.0)
        self.assertEqual(float(pe["public_glassbench_claim_allowed"]), 0.0)

        thermo = by_signature["thermodynamic_transition"]
        self.assertEqual(thermo["public_window_verdict_stage"], "scope_boundary_not_tested")
        self.assertEqual(thermo["allowed_public_claim"], "not_a_thermodynamic_glass_transition_test")
        self.assertEqual(float(thermo["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_late_recovery_experiment_design_targets_minimal_tc50_followup(self):
        protocol_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "envelope_ready": 1.0,
                "required_followup_lag_time": 42972781.2315918,
                "max_finite_exchange_late_ngp": 0.08,
                "static_gamma_late_ngp_plateau": 0.5,
                "late_recovery_stage": "late_recovery_acquisition_required",
            }
        ]
        timecode_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "timecode_target_ready": 1.0,
                "current_max_time_code": "tc40",
                "current_max_lag_time": 1500000.0,
                "target_time_code": "tc50",
                "target_lag_time": 166002226.81761542,
                "target_lag_over_required": 3.862962136962582,
            }
        ]
        public_verdict_rows = [
            {
                "signature": "late_gaussian_recovery",
                "public_window_verdict_stage": "public_window_censored_sota_unresolved",
                "primary_blocker": "public_glassbench_timecode_ceiling",
            },
            {
                "signature": "persistence_exchange_decoupling",
                "public_window_verdict_stage": "mechanism_selection_censored_unresolved",
                "primary_blocker": "public_glassbench_timecode_ceiling",
            },
        ]

        rows = glassbench_late_recovery_experiment_design(
            design_id="glassbench_late_recovery_experiment_design",
            late_recovery_protocol_rows=protocol_rows,
            timecode_target_rows=timecode_rows,
            public_window_verdict_rows=public_verdict_rows,
        )

        row = rows[0]
        self.assertEqual(row["experiment_design_stage"], "minimal_tc50_followup_ready")
        self.assertEqual(row["required_time_code"], "tc50")
        self.assertAlmostEqual(float(row["minimum_required_lag_time"]), 42972781.2315918)
        self.assertAlmostEqual(float(row["planned_lag_time"]), 166002226.81761542, delta=1e-3)
        self.assertGreater(float(row["planned_lag_over_minimum_required"]), 3.0)
        self.assertEqual(row["required_observables"], "MSD;NGP;F_s(k,t);self_van_hove_tail;member_uncertainty")
        self.assertEqual(row["finite_exchange_support_rule"], "late_ngp <= max_finite_exchange_late_ngp")
        self.assertEqual(row["static_disorder_rejection_rule"], "late_ngp + 2sigma < static_gamma_late_ngp_plateau")
        self.assertEqual(float(row["max_finite_exchange_late_ngp"]), 0.08)
        self.assertEqual(float(row["static_gamma_late_ngp_plateau"]), 0.5)
        self.assertEqual(float(row["late_recovery_claim_ready_after_measurement"]), 1.0)
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(row["primary_blocker"], "public_glassbench_timecode_ceiling")

    def test_glassbench_late_recovery_uncertainty_verdict_requires_2sigma_decision_power(self):
        protocol_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "envelope_ready": 1.0,
                "required_followup_lag_time": 42972781.2315918,
                "max_finite_exchange_late_ngp": 0.05,
                "static_gamma_late_ngp_plateau": 0.1,
                "min_static_plateau_rejection_gap": 0.05,
                "late_recovery_stage": "late_recovery_acquisition_required",
            }
        ]
        ingestion_rows = [
            {
                "candidate_id": "KA2D:0.23:151:0",
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "required_followup_lag_time": 42972781.2315918,
                "observed_lag_time": 50000000.0,
                "observed_late_ngp": 0.012,
                "sigma_late_ngp": 0.003,
                "observed_tail_gaussian_recovery": 1.0,
                "sigma_tail_recovery": 0.05,
                "late_recovery_observation_ready": 1.0,
                "late_recovery_ingestion_stage": "late_recovery_observation_ingestion_ready",
            },
            {
                "candidate_id": "KA2D:0.23:151:1",
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "required_followup_lag_time": 42972781.2315918,
                "observed_lag_time": 50000000.0,
                "observed_late_ngp": 0.045,
                "sigma_late_ngp": 0.01,
                "observed_tail_gaussian_recovery": 1.0,
                "sigma_tail_recovery": 0.05,
                "late_recovery_observation_ready": 1.0,
                "late_recovery_ingestion_stage": "late_recovery_observation_ingestion_ready",
            },
        ]

        rows = glassbench_late_recovery_uncertainty_verdict(
            verdict_id="glassbench_late_recovery_uncertainty_verdict",
            late_recovery_protocol_rows=protocol_rows,
            ingestion_rows=ingestion_rows,
        )

        by_candidate = {row["candidate_id"]: row for row in rows}
        supported = by_candidate["KA2D:0.23:151:0"]
        wide = by_candidate["KA2D:0.23:151:1"]

        self.assertEqual(
            supported["uncertainty_verdict_stage"],
            "uncertainty_weighted_finite_exchange_supported_static_disorder_rejected",
        )
        self.assertAlmostEqual(float(supported["late_ngp_upper_2sigma"]), 0.018)
        self.assertAlmostEqual(float(supported["tail_recovery_lower_2sigma"]), 0.9)
        self.assertGreater(float(supported["finite_exchange_support_margin"]), 0.03)
        self.assertGreater(float(supported["static_disorder_rejection_margin"]), 0.08)
        self.assertEqual(float(supported["finite_exchange_uncertainty_supported"]), 1.0)
        self.assertEqual(float(supported["static_disorder_uncertainty_rejected"]), 1.0)
        self.assertEqual(float(supported["uncertainty_decision_ready"]), 1.0)
        self.assertEqual(float(supported["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(float(supported["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(supported["primary_blocker"], "exchange_clock")

        self.assertEqual(wide["uncertainty_verdict_stage"], "late_recovery_uncertainty_indeterminate")
        self.assertGreater(float(wide["late_ngp_upper_2sigma"]), 0.05)
        self.assertEqual(float(wide["finite_exchange_uncertainty_supported"]), 0.0)
        self.assertEqual(float(wide["static_disorder_uncertainty_rejected"]), 1.0)
        self.assertEqual(float(wide["uncertainty_decision_ready"]), 0.0)
        self.assertEqual(wide["primary_blocker"], "late_ngp_uncertainty")

    def test_glassbench_late_recovery_outcome_matrix_preregisters_support_reject_and_indeterminate_paths(self):
        design_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "structure_id": "151",
                "required_time_code": "tc50",
                "minimum_required_lag_time": 42972781.2315918,
                "planned_lag_time": 166002226.81761542,
                "planned_lag_over_minimum_required": 3.862962136962582,
                "max_finite_exchange_late_ngp": 0.05,
                "static_gamma_late_ngp_plateau": 0.1,
                "late_recovery_claim_ready_after_measurement": 1.0,
                "experiment_design_stage": "minimal_tc50_followup_ready",
            }
        ]

        rows = glassbench_late_recovery_outcome_matrix(
            matrix_id="glassbench_late_recovery_outcome_matrix",
            experiment_design_rows=design_rows,
        )

        by_scenario = {row["outcome_scenario"]: row for row in rows}
        support = by_scenario["low_late_ngp_gaussian_recovery"]
        reject = by_scenario["high_late_ngp_or_missing_recovery"]
        wide = by_scenario["wide_uncertainty_requires_more_data"]

        self.assertEqual(len(rows), 3)
        self.assertEqual(support["target_time_code"], "tc50")
        self.assertEqual(
            support["predicted_uncertainty_verdict_stage"],
            "uncertainty_weighted_finite_exchange_supported_static_disorder_rejected",
        )
        self.assertEqual(support["claim_if_observed"], "finite_exchange_supported_static_disorder_rejected")
        self.assertGreater(float(support["finite_exchange_support_margin"]), 0.0)
        self.assertGreater(float(support["static_disorder_rejection_margin"]), 0.0)
        self.assertEqual(float(support["uncertainty_decision_ready"]), 1.0)

        self.assertEqual(reject["predicted_uncertainty_verdict_stage"], "uncertainty_weighted_finite_exchange_rejected")
        self.assertEqual(reject["claim_if_observed"], "finite_exchange_rejected_or_model_reparameterization_required")
        self.assertLess(float(reject["finite_exchange_support_margin"]), 0.0)
        self.assertEqual(float(reject["uncertainty_decision_ready"]), 1.0)

        self.assertEqual(wide["predicted_uncertainty_verdict_stage"], "late_recovery_uncertainty_indeterminate")
        self.assertEqual(wide["claim_if_observed"], "no_mechanism_selection_claim")
        self.assertEqual(float(wide["uncertainty_decision_ready"]), 0.0)
        self.assertEqual(wide["primary_blocker"], "late_ngp_uncertainty")
        self.assertEqual(float(wide["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(wide["outcome_matrix_stage"], "tc50_outcome_matrix_preregistered")

    def test_glassbench_microdynamic_closed_loop_audit_keeps_real_data_blockers_explicit(self):
        trajectory_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "frame_index": 0.0,
                "member_count": 4.0,
                "msd": 0.0,
                "ngp_2d": 0.0,
                "self_intermediate_scattering_by_k": "1;1;1",
                "frame_index_uncertainty_ready": 1.0,
                "physical_time_ready": 0.0,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "frame_index": 1.0,
                "member_count": 4.0,
                "msd": 0.006,
                "ngp_2d": 0.08,
                "self_intermediate_scattering_by_k": "0.999;0.998;0.996",
                "frame_index_uncertainty_ready": 1.0,
                "physical_time_ready": 0.0,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "frame_index": 2.0,
                "member_count": 4.0,
                "msd": 0.008,
                "ngp_2d": 0.05,
                "self_intermediate_scattering_by_k": "0.998;0.997;0.995",
                "frame_index_uncertainty_ready": 1.0,
                "physical_time_ready": 0.0,
            },
        ]
        signature_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "real_time_observable_curve_ready": 1.0,
                "real_pe_inversion_ready": 0.0,
                "supported_dynamical_signature_count": 4.0,
                "alpha_threshold_crossed": 0.0,
                "primary_blocker": "alpha_threshold_crossing",
            }
        ]
        alpha_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "metadata_tau_alpha_consistent_with_anchor_fs": 0.0,
                "primary_blocker": "anchor_wave_number_or_alpha_definition_mismatch",
            }
        ]

        row = glassbench_microdynamic_closed_loop_audit(
            audit_id="glassbench_microdynamic_closed_loop",
            trajectory_rows=trajectory_rows,
            signature_rows=signature_rows,
            alpha_horizon_rows=alpha_rows,
        )[0]

        self.assertEqual(row["closed_loop_stage"], "real_microstats_macro_signatures_closed_loop_blocked")
        self.assertEqual(float(row["frame_index_microstats_ready"]), 1.0)
        self.assertEqual(float(row["physical_time_microstats_ready"]), 0.0)
        self.assertEqual(float(row["macro_signature_ready"]), 1.0)
        self.assertEqual(float(row["micro_to_macro_prediction_ready"]), 0.0)
        self.assertEqual(float(row["closed_loop_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "physical_time_semantics")
        self.assertIn("cage_jump_event_segmentation", row["missing_closed_loop_inputs"])
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_cage_jump_proxy_canary_extracts_aggregate_event_candidates(self):
        trajectory_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "frame_index": 0.0,
                "member_count": 4.0,
                "msd": 0.0,
                "ngp_2d": 0.0,
                "self_intermediate_scattering_by_k": "1;1;1",
                "frame_index_uncertainty_ready": 1.0,
                "physical_time_ready": 0.0,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "frame_index": 1.0,
                "member_count": 4.0,
                "msd": 0.01,
                "ngp_2d": 0.02,
                "self_intermediate_scattering_by_k": "0.998;0.995;0.990",
                "frame_index_uncertainty_ready": 1.0,
                "physical_time_ready": 0.0,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "frame_index": 2.0,
                "member_count": 4.0,
                "msd": 0.09,
                "ngp_2d": 0.25,
                "self_intermediate_scattering_by_k": "0.970;0.940;0.900",
                "frame_index_uncertainty_ready": 1.0,
                "physical_time_ready": 0.0,
            },
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "frame_index": 3.0,
                "member_count": 4.0,
                "msd": 0.08,
                "ngp_2d": 0.10,
                "self_intermediate_scattering_by_k": "0.975;0.950;0.920",
                "frame_index_uncertainty_ready": 1.0,
                "physical_time_ready": 0.0,
            },
        ]

        row = glassbench_cage_jump_proxy_canary(
            canary_id="glassbench_cage_jump_proxy",
            trajectory_rows=trajectory_rows,
        )[0]

        self.assertEqual(row["canary_stage"], "aggregate_cage_jump_proxy_ready_particle_events_blocked")
        self.assertEqual(float(row["aggregate_jump_proxy_ready"]), 1.0)
        self.assertEqual(float(row["particle_resolved_jump_events_ready"]), 0.0)
        self.assertEqual(float(row["physical_time_jump_clock_ready"]), 0.0)
        self.assertEqual(float(row["peak_proxy_event_frame"]), 2.0)
        self.assertGreater(float(row["proxy_jump_length"]), 0.0)
        self.assertEqual(row["primary_blocker"], "particle_resolved_displacements")
        self.assertIn("particle_resolved_displacements", row["missing_event_clock_inputs"])
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_event_clock_threshold_readiness_gate_blocks_without_particle_cache(self):
        rows = glassbench_event_clock_threshold_readiness_gate(
            benchmark_id="glassbench_ka2d_threshold_readiness",
            system_id="KA2D",
            temperature=0.23,
            positions_schema_ready=True,
            first_npz_observable_curve_ready=True,
            member_ensemble_observable_ready=True,
            particle_resolved_positions_cached=False,
            physical_time_semantics_ready=False,
            event_clock_threshold_protocol_available=True,
            macro_heldout_observables_ready=False,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["readiness_stage"], "real_event_clock_threshold_robustness_blocked")
        self.assertEqual(row["primary_blocker"], "particle_resolved_positions_cache")
        self.assertEqual(float(row["positions_schema_ready"]), 1.0)
        self.assertEqual(float(row["member_ensemble_observable_ready"]), 1.0)
        self.assertEqual(float(row["particle_resolved_positions_cached"]), 0.0)
        self.assertEqual(float(row["real_event_clock_threshold_robustness_ready"]), 0.0)
        self.assertIn("threshold_sweep_event_clock", row["missing_real_threshold_inputs"])
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_event_clock_threshold_readiness_gate_accepts_complete_real_inputs(self):
        rows = glassbench_event_clock_threshold_readiness_gate(
            benchmark_id="glassbench_ka2d_threshold_readiness",
            system_id="KA2D",
            temperature=0.23,
            positions_schema_ready=True,
            first_npz_observable_curve_ready=True,
            member_ensemble_observable_ready=True,
            particle_resolved_positions_cached=True,
            physical_time_semantics_ready=True,
            event_clock_threshold_protocol_available=True,
            macro_heldout_observables_ready=True,
        )

        row = rows[0]
        self.assertEqual(row["readiness_stage"], "real_event_clock_threshold_robustness_ready")
        self.assertEqual(row["primary_blocker"], "none")
        self.assertEqual(float(row["threshold_sweep_event_clock_ready"]), 1.0)
        self.assertEqual(float(row["real_event_clock_threshold_robustness_ready"]), 1.0)
        self.assertEqual(float(row["real_benchmark_closed_loop_ready"]), 1.0)
        self.assertEqual(row["missing_real_threshold_inputs"], "none")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_first_npz_particle_cache_contract_pins_coordinate_payload(self):
        schema_entries = [
            {
                "path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "system_id": "KA2D",
                "temperature": "0.23",
                "first_npz_member": "T0.23/test/N1290T0.23_202_tc05.npz",
                "npz_member_bytes": 465710,
                "npz_member_md5": "26b4b9af10138fbd04a840fe8275de8e",
                "arrays": [
                    {"name": "positions.npy", "shape": [20, 1290, 2], "dtype": "float64"},
                    {"name": "box.npy", "shape": [], "dtype": "float64"},
                ],
            }
        ]
        curve_entries = [
            {
                "path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "system_id": "KA2D",
                "temperature": "0.23",
                "first_npz_member": "T0.23/test/N1290T0.23_202_tc05.npz",
                "compressed_probe_range_start": 2980602255,
                "compressed_probe_range_end": 2984796558,
                "compressed_probe_bytes": 4194304,
                "npz_member_bytes": 465710,
                "npz_member_md5": "26b4b9af10138fbd04a840fe8275de8e",
            }
        ]

        rows = glassbench_first_npz_particle_cache_contract_gate(
            contract_id="glassbench_first_npz_particle_cache_contract",
            schema_entries=schema_entries,
            curve_entries=curve_entries,
            cache_root="data/third_party/glassbench/particle_cache",
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["cache_contract_stage"], "first_npz_particle_cache_contract_ready_cache_missing")
        self.assertEqual(row["positions_shape"], "20x1290x2")
        self.assertEqual(float(row["frame_count"]), 20.0)
        self.assertEqual(float(row["particle_count"]), 1290.0)
        self.assertEqual(float(row["spatial_dimension"]), 2.0)
        self.assertEqual(float(row["particle_cache_contract_ready"]), 1.0)
        self.assertEqual(float(row["particle_resolved_positions_cached"]), 0.0)
        self.assertEqual(row["primary_blocker"], "persist_particle_coordinate_cache")
        self.assertIn("glassbench_ka2d_T0_23_first_npz_positions.npz", row["particle_cache_target"])
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_first_npz_particle_cache_contract_accepts_existing_cache(self):
        schema_entries = [
            {
                "path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "system_id": "KA2D",
                "temperature": "0.30",
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                "npz_member_bytes": 444786,
                "npz_member_md5": "f51fd76f59b8288405a9e7abb61cdd0a",
                "arrays": [{"name": "positions.npy", "shape": [20, 1290, 2], "dtype": "float64"}],
            }
        ]
        curve_entries = [
            {
                "path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "system_id": "KA2D",
                "temperature": "0.30",
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                "compressed_probe_range_start": 464175,
                "compressed_probe_range_end": 4658478,
                "compressed_probe_bytes": 4194304,
                "npz_member_bytes": 444786,
                "npz_member_md5": "f51fd76f59b8288405a9e7abb61cdd0a",
            }
        ]
        cache_path = "data/third_party/glassbench/particle_cache/glassbench_ka2d_T0_30_first_npz_positions.npz"

        row = glassbench_first_npz_particle_cache_contract_gate(
            contract_id="glassbench_first_npz_particle_cache_contract",
            schema_entries=schema_entries,
            curve_entries=curve_entries,
            cache_root="data/third_party/glassbench/particle_cache",
            cached_particle_cache_targets=[cache_path],
            physical_time_semantics_ready=True,
        )[0]

        self.assertEqual(row["cache_contract_stage"], "first_npz_particle_cache_ready_for_threshold_sweep")
        self.assertEqual(float(row["particle_resolved_positions_cached"]), 1.0)
        self.assertEqual(float(row["threshold_sweep_event_clock_ready"]), 1.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_glassbench_cached_particle_timecode_bridge_attaches_lag_without_time_axis(self):
        cache_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "particle_cache_path": "data/third_party/glassbench/particle_cache/cache.npz",
                "first_npz_member": "T0.23/test/N1290T0.23_202_tc05.npz",
                "npz_member_md5": "26b4b9af10138fbd04a840fe8275de8e",
                "positions_shape": "20x1290x2",
                "particle_resolved_positions_cached": 1.0,
            }
        ]
        semantics_manifest = {
            "entries": [
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "tau_alpha": 918305.0,
                    "members": [
                        {
                            "member": "T0.23/test/N1290T0.23_202_tc05.npz",
                            "member_md5": "26b4b9af10138fbd04a840fe8275de8e",
                            "time_code": "tc05",
                            "lag_time": 0.1,
                            "lag_time_over_tau_alpha": 1.0889616315258749e-07,
                            "axis0_semantics": "isoconfigurational_trajectory_replicates",
                            "replica_count": 20,
                            "positions_shape": [20, 1290, 2],
                        }
                    ],
                }
            ]
        }

        row = glassbench_cached_particle_timecode_bridge(
            bridge_id="glassbench_cached_particle_timecode_bridge",
            cache_rows=cache_rows,
            semantics_manifest=semantics_manifest,
        )[0]

        self.assertEqual(row["timecode_bridge_stage"], "cached_particle_lag_time_ready_event_clock_blocked")
        self.assertEqual(float(row["particle_resolved_positions_cached"]), 1.0)
        self.assertEqual(float(row["physical_lag_time_ready"]), 1.0)
        self.assertEqual(float(row["frame_axis_is_physical_time"]), 0.0)
        self.assertEqual(float(row["axis0_is_isoconfigurational_replica"]), 1.0)
        self.assertEqual(float(row["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "frame_axis_is_isoconfigurational_replicates")
        self.assertEqual(row["next_required_action"], "extract_multi_lag_particle_cache_or_true_trajectory")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_multilag_particle_cache_targets_selects_structure_ladder(self):
        semantics_manifest = {
            "entries": [
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "tau_alpha": 918306.0,
                    "members": [
                        {
                            "member": "T0.23/test/N1290T0.23_151_tc05.npz",
                            "member_md5": "md5-151-05",
                            "time_code": "tc05",
                            "lag_time": 0.1,
                            "lag_time_over_tau_alpha": 1.1e-7,
                            "structure_id": 151,
                        },
                        {
                            "member": "T0.23/test/N1290T0.23_151_tc10.npz",
                            "member_md5": "md5-151-10",
                            "time_code": "tc10",
                            "lag_time": 1.1,
                            "lag_time_over_tau_alpha": 1.2e-6,
                            "structure_id": 151,
                        },
                        {
                            "member": "T0.23/test/N1290T0.23_151_tc15.npz",
                            "member_md5": "md5-151-15",
                            "time_code": "tc15",
                            "lag_time": 11.64,
                            "lag_time_over_tau_alpha": 1.3e-5,
                            "structure_id": 151,
                        },
                        {
                            "member": "T0.23/test/N1290T0.23_202_tc05.npz",
                            "member_md5": "md5-202-05",
                            "time_code": "tc05",
                            "lag_time": 0.1,
                            "lag_time_over_tau_alpha": 1.1e-7,
                            "structure_id": 202,
                        },
                    ],
                },
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "tau_alpha": 2200.0,
                    "members": [
                        {
                            "member": "T0.30/train/N1290T0.30_10_tc01.npz",
                            "member_md5": "md5-10-01",
                            "time_code": "tc01",
                            "lag_time": 0.11,
                            "lag_time_over_tau_alpha": 5.0e-5,
                            "structure_id": 10,
                        }
                    ],
                },
            ]
        }
        cache_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "first_npz_member": "T0.23/test/N1290T0.23_202_tc05.npz",
                "npz_member_md5": "md5-202-05",
                "particle_resolved_positions_cached": 1.0,
            }
        ]

        rows = glassbench_multilag_particle_cache_targets(
            target_id="glassbench_multilag_particle_cache_targets",
            semantics_manifest=semantics_manifest,
            cache_rows=cache_rows,
            minimum_time_codes=3,
        )
        by_temp = {row["temperature"]: row for row in rows}

        cold = by_temp["0.23"]
        self.assertEqual(cold["selected_structure_id"], "151")
        self.assertEqual(cold["selected_time_codes"], "tc05;tc10;tc15")
        self.assertEqual(float(cold["official_multi_lag_ladder_ready"]), 1.0)
        self.assertEqual(float(cold["target_member_count"]), 3.0)
        self.assertEqual(float(cold["cached_target_member_count"]), 0.0)
        self.assertEqual(float(cold["particle_lag_ladder_cache_ready"]), 0.0)
        self.assertEqual(float(cold["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(cold["primary_blocker"], "multi_lag_particle_cache_missing")
        self.assertEqual(cold["next_required_action"], "extract_structure_matched_multi_lag_npz_members")
        self.assertEqual(float(cold["thermodynamic_claim_allowed"]), 0.0)

        warm = by_temp["0.30"]
        self.assertEqual(float(warm["official_multi_lag_ladder_ready"]), 0.0)
        self.assertEqual(float(warm["target_member_count"]), 1.0)
        self.assertEqual(warm["primary_blocker"], "official_multi_lag_semantics")

    def test_glassbench_cached_particle_observable_semantics_audit_blocks_without_initial_reference(self):
        rows = glassbench_cached_particle_observable_semantics_audit(
            audit_id="glassbench_cached_particle_observable_semantics",
            cached_observable_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "structure_id": "151",
                    "time_code": "tc05",
                    "lag_time": 0.1,
                    "target_member": "T0.23/test/N1290T0.23_151_tc05.npz",
                    "raw_coordinate_msd": 333.9,
                    "replica_spread_msd": 0.00186,
                    "initial_reference_positions_ready": 0.0,
                    "particle_resolved_positions_cached": 1.0,
                }
            ],
            official_observable_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "time_code": "tc05",
                    "member": "T0.23/test/N1290T0.23_151_tc05.npz",
                    "msd": 0.003511,
                    "ngp_2d": 0.00047,
                }
            ],
            max_reproducible_relative_error=0.05,
        )

        row = rows[0]
        self.assertGreater(float(row["raw_coordinate_msd_relative_error"]), 1.0e4)
        self.assertEqual(float(row["cached_coordinate_proxy_ready"]), 1.0)
        self.assertEqual(float(row["initial_reference_positions_ready"]), 0.0)
        self.assertEqual(float(row["official_displacement_observable_reproducible"]), 0.0)
        self.assertEqual(float(row["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "initial_positions_reference_missing")
        self.assertEqual(row["observable_semantics_stage"], "cached_coordinate_proxy_ready_initial_reference_blocked")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_glassbench_cached_particle_observable_semantics_audit_reproduces_displacement_with_initial_reference(self):
        rows = glassbench_cached_particle_observable_semantics_audit(
            audit_id="glassbench_cached_particle_observable_semantics",
            cached_observable_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "structure_id": "151",
                    "time_code": "tc05",
                    "lag_time": 0.1,
                    "target_member": "T0.23/test/N1290T0.23_151_tc05.npz",
                    "raw_coordinate_msd": 333.9,
                    "replica_spread_msd": 0.00186,
                    "initial_reference_msd": 0.00350,
                    "initial_reference_ngp_2d": 0.00048,
                    "initial_reference_fs_by_k": [0.99, 0.97],
                    "initial_reference_fs_formula": "axis_average_cos_xy",
                    "single_axis_x_fs_by_k": [0.98, 0.95],
                    "initial_reference_positions_ready": 1.0,
                    "particle_resolved_positions_cached": 1.0,
                }
            ],
            official_observable_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "time_code": "tc05",
                    "member": "T0.23/test/N1290T0.23_151_tc05.npz",
                    "msd": 0.003511,
                    "ngp_2d": 0.00047,
                    "self_intermediate_scattering_by_k": [0.9901, 0.9698],
                }
            ],
            max_reproducible_relative_error=0.05,
        )

        row = rows[0]
        self.assertLess(float(row["initial_reference_msd_relative_error"]), 0.05)
        self.assertLess(float(row["initial_reference_ngp_2d_relative_error"]), 0.05)
        self.assertLess(float(row["initial_reference_fs_max_abs_error"]), 0.001)
        self.assertEqual(float(row["official_displacement_observable_reproducible"]), 1.0)
        self.assertEqual(float(row["official_ngp_2d_reproducible"]), 1.0)
        self.assertEqual(float(row["official_fs_reproducible"]), 1.0)
        self.assertEqual(row["initial_reference_fs_formula"], "axis_average_cos_xy")
        self.assertEqual(float(row["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "none")
        self.assertEqual(row["observable_semantics_stage"], "official_displacement_observable_reproduced")
        self.assertEqual(row["next_required_action"], "run_structure_matched_displacement_inversion")

    def test_dynamic_signature_alignment_ledger_combines_model_literature_and_real_curve(self):
        claim_rows = [
            {
                "phenomenon": "cage_plateau_transient_ngp_van_hove_tail",
                "claim_alignment": "supported",
                "model_support_level": "derived",
                "primary_blocker": "uncertainty_columns",
            },
            {
                "phenomenon": "self_intermediate_scattering_alpha_relaxation",
                "claim_alignment": "supported",
                "model_support_level": "derived",
                "primary_blocker": "uncertainty_columns",
            },
            {
                "phenomenon": "persistence_exchange_decoupling",
                "claim_alignment": "supported",
                "model_support_level": "derived",
                "primary_blocker": "machine_readable_joint_curves",
            },
            {
                "phenomenon": "chi4_peak_and_dynamic_length_growth",
                "claim_alignment": "partial",
                "model_support_level": "effective_closure",
                "primary_blocker": "shared_transport_and_four_point_grid",
            },
            {
                "phenomenon": "configurational_entropy_and_ideal_glass_scope",
                "claim_alignment": "scope_boundary",
                "model_support_level": "closure_only",
                "primary_blocker": "thermodynamic_input_law",
            },
        ]
        literature_rows = [
            {"benchmark_source": "kob1995vanhove", "qualitative_comparison_ready": 1.0},
            {"benchmark_source": "kob1995intermediate", "qualitative_comparison_ready": 1.0},
            {"benchmark_source": "hedges2007persistence", "qualitative_comparison_ready": 1.0},
            {"benchmark_source": "lacevic2003fourpoint", "qualitative_comparison_ready": 1.0},
        ]
        glassbench_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "real_time_observable_curve_ready": 1.0,
                "real_pe_inversion_ready": 0.0,
                "msd_growth_signature": 1.0,
                "self_intermediate_decay_signature": 1.0,
                "transient_ngp_peak_signature": 1.0,
                "transient_chi4_peak_signature": 1.0,
                "alpha_threshold_crossed": 0.0,
                "primary_blocker": "alpha_threshold_crossing",
            }
        ]

        rows = dynamic_signature_alignment_ledger(
            alignment_id="sota_dynamic_signature_alignment",
            claim_rows=claim_rows,
            literature_rows=literature_rows,
            glassbench_signature_rows=glassbench_rows,
        )
        by_signature = {row["signature"]: row for row in rows}

        self.assertEqual(by_signature["transient_ngp_peak"]["alignment_stage"], "real_curve_supported")
        self.assertEqual(float(by_signature["transient_ngp_peak"]["real_glassbench_support"]), 1.0)
        self.assertEqual(float(by_signature["transient_ngp_peak"]["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(
            by_signature["self_intermediate_alpha"]["alignment_stage"],
            "real_curve_supported_pre_alpha_threshold",
        )
        self.assertEqual(by_signature["persistence_exchange_decoupling"]["alignment_stage"], "model_literature_supported_real_inversion_blocked")
        self.assertEqual(by_signature["persistence_exchange_decoupling"]["primary_blocker"], "alpha_threshold_crossing")
        self.assertEqual(by_signature["thermodynamic_transition"]["alignment_stage"], "scope_boundary_not_explained")
        self.assertEqual(float(by_signature["thermodynamic_transition"]["real_glassbench_support"]), 0.0)

    def test_langevin_bare_diffusion_and_ou_cage_follow_einstein_and_equipartition(self):
        landscape = LangevinCageLandscapeParams(
            temperature=0.8,
            friction=4.0,
            cage_curvature=10.0,
            saddle_curvature=6.0,
            barrier_height=3.2,
            jump_length=1.5,
        )

        self.assertAlmostEqual(langevin_bare_diffusion(landscape), 0.2)
        ou = langevin_cage_ou_parameters(landscape)

        self.assertAlmostEqual(ou["cage_variance"], 0.08)
        self.assertAlmostEqual(ou["cage_tau"], 0.4)

    def test_kramers_escape_rate_has_overdamped_arrhenius_barrier_scaling(self):
        fast = kramers_escape_rate(
            temperature=1.0,
            friction=2.0,
            basin_curvature=8.0,
            saddle_curvature=4.0,
            barrier_height=2.0,
        )
        slow = kramers_escape_rate(
            temperature=1.0,
            friction=2.0,
            basin_curvature=8.0,
            saddle_curvature=4.0,
            barrier_height=3.0,
        )
        expected_ratio = math.exp(1.0)

        self.assertGreater(fast, slow)
        self.assertAlmostEqual(fast / slow, expected_ratio)

    def test_langevin_landscape_coarse_grains_to_persistence_exchange_parameters(self):
        landscape = LangevinCageLandscapeParams(
            temperature=0.7,
            friction=3.0,
            cage_curvature=7.0,
            saddle_curvature=5.0,
            barrier_height=2.4,
            jump_length=1.2,
            persistence_barrier_extra=1.4,
            exchange_barrier_extra=0.2,
            dimension=3,
        )
        params = langevin_to_persistence_exchange(landscape)

        self.assertIsInstance(params, PersistenceExchangeParams)
        self.assertAlmostEqual(params.cage_variance, landscape.temperature / landscape.cage_curvature)
        self.assertAlmostEqual(params.cage_tau, landscape.friction / landscape.cage_curvature)
        self.assertAlmostEqual(params.jump_variance, landscape.jump_length**2 / landscape.dimension)
        self.assertGreater(params.persistence_mean, params.exchange_mean)
        self.assertGreater(
            persistence_exchange_diffusion_coefficient(params)
            * persistence_exchange_alpha_relaxation_time(1.0, params, threshold=math.exp(-1.0)),
            params.jump_variance,
        )

    def test_langevin_bridge_audit_marks_derived_effective_theory_and_remaining_assumptions(self):
        landscape = LangevinCageLandscapeParams(
            temperature=0.75,
            friction=2.5,
            cage_curvature=6.0,
            saddle_curvature=4.0,
            barrier_height=2.0,
            jump_length=1.0,
            persistence_barrier_extra=0.8,
            exchange_barrier_extra=0.1,
        )
        row = langevin_first_principles_bridge_audit(landscape)

        self.assertEqual(row["bridge_stage"], "langevin_kramers_to_renewal_effective_theory")
        self.assertEqual(float(row["langevin_equation_specified"]), 1.0)
        self.assertEqual(float(row["kramers_rates_derived"]), 1.0)
        self.assertEqual(float(row["persistence_exchange_params_derived"]), 1.0)
        self.assertEqual(float(row["full_many_body_first_principles_claim_allowed"]), 0.0)
        self.assertEqual(row["remaining_assumption"], "metastable_basin_partition_and_barrier_inputs")

    def test_local_cage_variance_has_plateau(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.7,
            cage_tau=0.8,
            jump_variance=0.5,
            renewal_rate=0.2,
            renewal_delay=2.0,
        )
        t = np.array([0.0, 10.0])
        variance = local_cage_variance(t, params)

        self.assertAlmostEqual(variance[0], 0.0)
        self.assertAlmostEqual(variance[-1], params.cage_variance, delta=1e-5)

    def test_moments_generate_plateau_then_long_time_growth(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.8,
            jump_variance=0.4,
            renewal_rate=0.25,
            renewal_delay=2.5,
        )
        early = moments_1d(np.array([1.0]), params)["m2"][0]
        middle = moments_1d(np.array([8.0]), params)["m2"][0]
        late = moments_1d(np.array([80.0]), params)["m2"][0]

        self.assertGreater(middle, early)
        self.assertLess(abs(middle - params.cage_variance), 0.5)
        self.assertGreater(late, middle + 5.0)

    def test_ngp_starts_near_zero_has_peak_and_decays(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        t = np.linspace(1e-4, 220.0, 2600)
        alpha = ngp_1d(t, params)
        peak_idx = int(np.argmax(alpha))

        self.assertLess(alpha[0], 1e-3)
        self.assertGreater(alpha[peak_idx], 0.1)
        self.assertGreater(peak_idx, 10)
        self.assertLess(alpha[-1], alpha[peak_idx] / 5.0)

    def test_long_time_ngp_matches_inverse_renewal_count(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        t = np.array([500.0])
        renewal_count = delayed_poisson_mean(t, params)[0]
        alpha = ngp_1d(t, params)[0]

        self.assertAlmostEqual(alpha * renewal_count, 1.0, delta=0.03)

    def test_three_dimensional_ngp_matches_variance_mixture_ngp(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        t = np.linspace(0.1, 40.0, 200)

        np.testing.assert_allclose(ngp_3d(t, params), ngp_1d(t, params), rtol=1e-12, atol=1e-12)

        moments = moments_3d(t, params)
        self.assertTrue(np.all(moments["r2"] > 0.0))
        self.assertTrue(np.all(moments["r4"] > 0.0))

    def test_self_intermediate_scattering_has_cage_plateau_and_alpha_decay(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.array([0.0, 1.0, 600.0])
        scattering = self_intermediate_scattering(wave_number, times, params)
        plateau = math.exp(-0.5 * wave_number**2 * params.cage_variance)
        alpha_rate = params.renewal_rate * (1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance))

        self.assertAlmostEqual(scattering[0], 1.0)
        self.assertAlmostEqual(scattering[1], plateau, delta=0.02)
        self.assertAlmostEqual(-math.log(scattering[-1] / plateau) / times[-1], alpha_rate, delta=0.003)

    def test_normalized_alpha_decay_removes_cage_debye_waller_factor(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.array([0.0, 8.0, 600.0])
        decay = normalized_alpha_decay(wave_number, times, params)
        alpha_rate = params.renewal_rate * (1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance))

        self.assertAlmostEqual(decay[0], 1.0)
        self.assertLess(decay[1], decay[0])
        self.assertAlmostEqual(-math.log(decay[-1]) / times[-1], alpha_rate, delta=0.003)

    def test_alpha_relaxation_time_solves_cage_normalized_decay_threshold(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        tau_alpha = alpha_relaxation_time(wave_number, params)
        decay = normalized_alpha_decay(wave_number, np.array([tau_alpha]), params)[0]

        self.assertAlmostEqual(decay, math.exp(-1.0), delta=1e-12)
        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        self.assertGreater(tau_alpha, 1.0 / (params.renewal_rate * gamma))

    def test_peak_relaxation_coupling_links_ngp_peak_to_alpha_time(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        threshold = math.exp(-1.0)

        coupling = peak_relaxation_coupling(wave_number, params, threshold=threshold)
        peak = dimensionless_peak_prediction(params)
        tau_alpha = alpha_relaxation_time(wave_number, params, threshold=threshold)
        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        alpha_count = -math.log(threshold) / gamma
        peak_count = params.cage_variance / params.jump_variance

        self.assertAlmostEqual(coupling["peak_time"], peak["peak_time"], delta=1e-12)
        self.assertAlmostEqual(coupling["tau_alpha"], tau_alpha, delta=1e-12)
        self.assertAlmostEqual(coupling["peak_ngp"], peak["peak_ngp"], delta=1e-12)
        self.assertAlmostEqual(coupling["gamma_k"], gamma, delta=1e-12)
        self.assertAlmostEqual(coupling["peak_renewal_count"], peak_count, delta=1e-12)
        self.assertAlmostEqual(coupling["alpha_renewal_count"], alpha_count, delta=1e-12)
        self.assertAlmostEqual(
            coupling["alpha_to_peak_renewal_count_ratio"],
            alpha_count / peak_count,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            coupling["tau_alpha_over_peak_time"],
            tau_alpha / peak["peak_time"],
            delta=1e-12,
        )
        self.assertGreater(coupling["tau_alpha_over_peak_time"], 1.0)

    def test_alpha_shape_curve_depends_only_on_delay_control(self):
        params_a = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        params_b = DelayedRenewalCageParams(
            cage_variance=1.8,
            cage_tau=0.9,
            jump_variance=0.8,
            renewal_rate=0.09,
            renewal_delay=6.0,
        )
        scaled_times = np.geomspace(0.15, 4.0, 120)

        curve_a = alpha_relaxation_shape_curve(1.1, params_a, scaled_times)
        curve_b = alpha_relaxation_shape_curve(1.1, params_b, scaled_times)

        np.testing.assert_allclose(curve_a, curve_b, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(alpha_relaxation_shape_curve(1.1, params_a, np.array([1.0]))[0], 1.0)

    def test_alpha_shape_superposition_residual_detects_control_change(self):
        reference = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        same_shape = DelayedRenewalCageParams(
            cage_variance=0.7,
            cage_tau=0.4,
            jump_variance=0.8,
            renewal_rate=0.09,
            renewal_delay=6.0,
        )
        different_shape = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=12.0,
        )
        scaled_times = np.geomspace(0.15, 4.0, 120)

        collapsed = alpha_shape_superposition_residual(
            1.1,
            reference,
            same_shape,
            scaled_times,
        )
        broken = alpha_shape_superposition_residual(
            1.1,
            reference,
            different_shape,
            scaled_times,
        )

        self.assertLess(collapsed["rms_log_shape_residual"], 1e-12)
        self.assertGreater(broken["rms_log_shape_residual"], 0.2)
        self.assertAlmostEqual(collapsed["reference_control"], collapsed["candidate_control"], delta=1e-12)
        self.assertGreater(broken["candidate_control"], broken["reference_control"])

    def test_renewal_scattering_susceptibility_is_closed_form_variance(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.linspace(0.0, 220.0, 900)
        susceptibility = renewal_scattering_susceptibility(wave_number, times, params)
        scattering = self_intermediate_scattering(wave_number, times, params)
        renewal = delayed_poisson_mean(times, params)
        jump_characteristic = math.exp(-0.5 * wave_number**2 * params.jump_variance)
        relative = np.exp(renewal * (jump_characteristic - 1.0) ** 2) - 1.0

        self.assertAlmostEqual(susceptibility[0], 0.0)
        self.assertTrue(np.all(susceptibility >= -1e-14))
        self.assertGreater(float(np.max(susceptibility)), 0.02)
        self.assertLess(susceptibility[-1], float(np.max(susceptibility)) / 4.0)
        np.testing.assert_allclose(susceptibility, scattering**2 * relative, rtol=1e-12, atol=1e-14)

    def test_gamma_exchange_count_moments_add_recovering_overdispersion(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        times = np.array([20.0, 120.0, 30000.0])

        count = gamma_exchange_count_moments(times, params, heterogeneity)
        renewal = delayed_poisson_mean(times, params)
        kappa_eff = heterogeneity.shape * (1.0 + renewal / heterogeneity.exchange_renewal_count)

        np.testing.assert_allclose(count["mean"], renewal, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(count["variance"], renewal + renewal**2 / kappa_eff, rtol=1e-12, atol=1e-14)
        self.assertGreater(count["variance"][1] / count["mean"][1], 2.0)

        alpha = gamma_exchange_ngp_1d(times, params, heterogeneity)
        self.assertGreater(alpha[1], ngp_1d(np.array([times[1]]), params)[0])
        self.assertLess(alpha[-1], alpha[1] / 20.0)

    def test_gamma_exchange_alpha_decay_has_stretched_like_window_and_recovery(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1
        times = np.linspace(0.02, 400.0, 2400)

        decay = gamma_exchange_normalized_alpha_decay(wave_number, times, params, heterogeneity)
        renewal = delayed_poisson_mean(times, params)
        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        kappa_eff = heterogeneity.shape * (1.0 + renewal / heterogeneity.exchange_renewal_count)
        expected = (1.0 + gamma * renewal / kappa_eff) ** (-kappa_eff)
        exponent = local_alpha_stretching_exponent(times, decay)
        alpha_window = (-np.log(decay) > 0.5) & (-np.log(decay) < 2.0)

        np.testing.assert_allclose(decay, expected, rtol=1e-12, atol=1e-14)
        self.assertLess(float(np.nanmedian(exponent[alpha_window])), 0.9)
        self.assertLess(decay[-1], 0.01)

    def test_gamma_exchange_scattering_susceptibility_matches_negative_binomial_variance(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.6, exchange_renewal_count=8.0)
        wave_number = 1.1
        times = np.linspace(0.0, 220.0, 900)

        susceptibility = gamma_exchange_scattering_susceptibility(wave_number, times, params, heterogeneity)
        scattering = gamma_exchange_self_intermediate_scattering(wave_number, times, params, heterogeneity)
        renewal = delayed_poisson_mean(times, params)
        local = local_cage_variance(times, params)
        kappa_eff = heterogeneity.shape * (1.0 + renewal / heterogeneity.exchange_renewal_count)
        jump_characteristic = math.exp(-0.5 * wave_number**2 * params.jump_variance)
        second_moment = np.exp(-wave_number**2 * local) * (
            1.0 + renewal * (1.0 - jump_characteristic**2) / kappa_eff
        ) ** (-kappa_eff)

        self.assertAlmostEqual(susceptibility[0], 0.0)
        self.assertTrue(np.all(susceptibility >= -1e-14))
        np.testing.assert_allclose(susceptibility, second_moment - scattering**2, rtol=1e-12, atol=1e-14)
        self.assertGreater(float(np.max(susceptibility)), float(np.max(renewal_scattering_susceptibility(wave_number, times, params))))

    def test_gamma_exchange_asymptotics_link_late_ngp_and_alpha_rate(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1
        late_time = np.array([30000.0])

        diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)
        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        heterogeneity_ratio = heterogeneity.exchange_renewal_count / heterogeneity.shape
        expected_rate_per_renewal = math.log1p(gamma * heterogeneity_ratio) / heterogeneity_ratio
        renewal = delayed_poisson_mean(late_time, params)[0]
        ngp = gamma_exchange_ngp_1d(late_time, params, heterogeneity)[0]
        decay = gamma_exchange_normalized_alpha_decay(wave_number, late_time, params, heterogeneity)[0]

        self.assertAlmostEqual(diagnostics["heterogeneity_ratio"], heterogeneity_ratio, delta=1e-12)
        self.assertAlmostEqual(diagnostics["late_ngp_renewal_amplitude"], 1.0 + heterogeneity_ratio, delta=1e-12)
        self.assertAlmostEqual(diagnostics["late_alpha_decay_per_renewal"], expected_rate_per_renewal, delta=1e-12)
        self.assertAlmostEqual(
            diagnostics["late_alpha_rate"],
            params.renewal_rate * expected_rate_per_renewal,
            delta=1e-12,
        )
        self.assertAlmostEqual(renewal * ngp, diagnostics["late_ngp_renewal_amplitude"], delta=0.08)
        self.assertAlmostEqual(-math.log(decay) / renewal, expected_rate_per_renewal, delta=2e-4)
        self.assertLess(diagnostics["alpha_rate_renormalization"], 1.0)

    def test_gamma_exchange_alpha_rate_inverts_heterogeneity_ratio(self):
        gamma = 0.38368679808771045
        heterogeneity_ratio = 25.0
        observed_rate_per_renewal = math.log1p(gamma * heterogeneity_ratio) / heterogeneity_ratio

        inferred = infer_gamma_exchange_ratio_from_alpha_rate(
            gamma_k=gamma,
            observed_decay_per_renewal=observed_rate_per_renewal,
        )

        self.assertAlmostEqual(inferred, heterogeneity_ratio, delta=1e-10)
        self.assertAlmostEqual(
            infer_gamma_exchange_ratio_from_alpha_rate(
                gamma_k=gamma,
                observed_decay_per_renewal=gamma,
            ),
            0.0,
            delta=1e-12,
        )

    def test_gamma_exchange_diagnostic_map_classifies_observable_window(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )

        rows = gamma_exchange_diagnostic_map(
            wave_number=1.1,
            params=params,
            shape=0.4,
            heterogeneity_ratios=[0.0, 2.0, 25.0],
        )

        self.assertEqual([row["heterogeneity_ratio"] for row in rows], [0.0, 2.0, 25.0])
        self.assertAlmostEqual(rows[0]["late_ngp_renewal_amplitude"], 1.0, delta=1e-12)
        self.assertAlmostEqual(rows[0]["alpha_rate_renormalization"], 1.0, delta=1e-12)
        self.assertEqual(rows[0]["passes_joint_criterion"], 0.0)
        self.assertEqual(rows[1]["passes_joint_criterion"], 1.0)
        self.assertEqual(rows[2]["passes_joint_criterion"], 1.0)
        self.assertGreater(rows[2]["late_ngp_renewal_amplitude"], rows[1]["late_ngp_renewal_amplitude"])
        self.assertLess(rows[2]["alpha_rate_renormalization"], rows[1]["alpha_rate_renormalization"])
        self.assertAlmostEqual(rows[2]["inferred_ratio_from_alpha_rate"], 25.0, delta=1e-10)
        self.assertAlmostEqual(rows[2]["log_ratio_residual"], 0.0, delta=1e-12)

    def test_gamma_exchange_late_observable_protocol_accepts_consistent_data(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1
        late_time = np.array([30000.0])
        renewal = delayed_poisson_mean(late_time, params)[0]
        late_ngp = gamma_exchange_ngp_1d(late_time, params, heterogeneity)[0]
        diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)

        inferred = infer_gamma_exchange_from_late_observables(
            wave_number=wave_number,
            params=params,
            late_renewal_count=renewal,
            late_ngp=late_ngp,
            observed_alpha_decay_per_renewal=diagnostics["late_alpha_decay_per_renewal"],
        )

        self.assertAlmostEqual(inferred["ratio_from_late_ngp"], 25.0, delta=0.08)
        self.assertAlmostEqual(inferred["ratio_from_alpha_rate"], 25.0, delta=1e-10)
        self.assertLess(abs(inferred["log_ratio_residual"]), 0.004)
        self.assertEqual(inferred["passes_consistency"], 1.0)

    def test_gamma_exchange_late_observable_protocol_rejects_inconsistent_data(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        gamma = 1.0 - math.exp(-0.5 * 1.1**2 * params.jump_variance)
        alpha_rate_for_c2 = math.log1p(gamma * 2.0) / 2.0

        inferred = infer_gamma_exchange_from_late_observables(
            wave_number=1.1,
            params=params,
            late_renewal_count=5400.0,
            late_ngp=26.0 / 5400.0,
            observed_alpha_decay_per_renewal=alpha_rate_for_c2,
        )

        self.assertAlmostEqual(inferred["ratio_from_late_ngp"], 25.0, delta=1e-12)
        self.assertAlmostEqual(inferred["ratio_from_alpha_rate"], 2.0, delta=1e-10)
        self.assertGreater(abs(inferred["log_ratio_residual"]), 2.0)
        self.assertEqual(inferred["passes_consistency"], 0.0)

    def test_gamma_exchange_late_observable_uncertainty_scores_residuals(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1
        late_time = np.array([30000.0])
        renewal = delayed_poisson_mean(late_time, params)[0]
        late_ngp = gamma_exchange_ngp_1d(late_time, params, heterogeneity)[0]
        diagnostics = gamma_exchange_asymptotic_diagnostics(wave_number, params, heterogeneity)

        scored = infer_gamma_exchange_uncertainty_from_late_observables(
            wave_number=wave_number,
            params=params,
            late_renewal_count=renewal,
            late_ngp=late_ngp,
            observed_alpha_decay_per_renewal=diagnostics["late_alpha_decay_per_renewal"],
            late_renewal_count_std=0.01 * renewal,
            late_ngp_std=0.01 * late_ngp,
            alpha_decay_per_renewal_std=0.002,
        )

        self.assertGreater(scored["ratio_from_late_ngp_std"], 0.0)
        self.assertGreater(scored["ratio_from_alpha_rate_std"], 0.0)
        self.assertGreater(scored["log_ratio_residual_std"], 0.0)
        self.assertLess(scored["log_ratio_z_score"], 1.0)
        self.assertEqual(scored["passes_statistical_consistency"], 1.0)

        gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
        alpha_rate_for_c2 = math.log1p(gamma * 2.0) / 2.0
        mismatched = infer_gamma_exchange_uncertainty_from_late_observables(
            wave_number=wave_number,
            params=params,
            late_renewal_count=renewal,
            late_ngp=late_ngp,
            observed_alpha_decay_per_renewal=alpha_rate_for_c2,
            late_renewal_count_std=0.01 * renewal,
            late_ngp_std=0.01 * late_ngp,
            alpha_decay_per_renewal_std=0.002,
        )

        self.assertGreater(mismatched["log_ratio_z_score"], 20.0)
        self.assertEqual(mismatched["passes_statistical_consistency"], 0.0)

    def test_gamma_exchange_multik_collapse_accepts_shared_exchange_ratio(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_numbers = [0.6, 1.1, 1.8]
        shared_ratio = 25.0
        rates = []
        for wave_number in wave_numbers:
            gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
            rates.append(math.log1p(gamma * shared_ratio) / shared_ratio)

        collapse = infer_gamma_exchange_multik_collapse(
            wave_numbers=wave_numbers,
            params=params,
            late_renewal_count=5400.0,
            late_ngp=26.0 / 5400.0,
            observed_alpha_decay_per_renewal=rates,
            alpha_decay_per_renewal_std=[0.002, 0.002, 0.002],
            late_renewal_count_std=54.0,
            late_ngp_std=0.01 * 26.0 / 5400.0,
        )

        self.assertEqual(len(collapse["per_wave_number"]), 3)
        self.assertAlmostEqual(collapse["ratio_from_late_ngp"], 25.0, delta=1e-12)
        self.assertAlmostEqual(collapse["weighted_mean_ratio_from_alpha"], 25.0, delta=1e-8)
        self.assertLess(collapse["collapse_z_score"], 1.0)
        self.assertEqual(collapse["passes_multik_collapse"], 1.0)

    def test_gamma_exchange_multik_collapse_rejects_wave_number_mismatch(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_numbers = [0.6, 1.1, 1.8]
        rates = []
        for idx, wave_number in enumerate(wave_numbers):
            target_ratio = 2.0 if idx == 1 else 25.0
            gamma = 1.0 - math.exp(-0.5 * wave_number**2 * params.jump_variance)
            rates.append(math.log1p(gamma * target_ratio) / target_ratio)

        collapse = infer_gamma_exchange_multik_collapse(
            wave_numbers=wave_numbers,
            params=params,
            late_renewal_count=5400.0,
            late_ngp=26.0 / 5400.0,
            observed_alpha_decay_per_renewal=rates,
            alpha_decay_per_renewal_std=[0.002, 0.002, 0.002],
            late_renewal_count_std=54.0,
            late_ngp_std=0.01 * 26.0 / 5400.0,
        )

        self.assertGreater(collapse["collapse_z_score"], 20.0)
        self.assertEqual(collapse["passes_multik_collapse"], 0.0)

    def test_late_mechanism_selection_identifies_finite_exchange_recovery(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        times = np.array([10000.0, 30000.0])
        renewal = delayed_poisson_mean(times, params)
        alpha = gamma_exchange_ngp_1d(times, params, heterogeneity)
        observed_slope = gamma_exchange_asymptotic_diagnostics(1.1, params, heterogeneity)[
            "late_alpha_decay_per_renewal"
        ]

        selection = late_mechanism_selection(
            wave_number=1.1,
            params=params,
            earlier_renewal_count=float(renewal[0]),
            earlier_ngp=float(alpha[0]),
            later_renewal_count=float(renewal[1]),
            later_ngp=float(alpha[1]),
            observed_alpha_decay_per_renewal=observed_slope,
        )

        self.assertEqual(selection["best_model"], "finite_exchange")
        self.assertEqual(selection["finite_exchange"]["passes"], 1.0)
        self.assertEqual(selection["poisson"]["passes"], 0.0)
        self.assertEqual(selection["static_gamma"]["passes"], 0.0)
        self.assertAlmostEqual(selection["finite_exchange"]["inferred_exchange_ratio"], 25.0, delta=0.15)

    def test_late_mechanism_selection_identifies_static_gamma_plateau(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        shape = 0.4
        times = np.array([10000.0, 30000.0])
        renewal = delayed_poisson_mean(times, params)
        alpha = static_gamma_ngp_1d(times, params, shape)
        decay = static_gamma_normalized_alpha_decay(1.1, np.array([times[1]]), params, shape)[0]
        observed_slope = -math.log(decay) / renewal[1]

        selection = late_mechanism_selection(
            wave_number=1.1,
            params=params,
            earlier_renewal_count=float(renewal[0]),
            earlier_ngp=float(alpha[0]),
            later_renewal_count=float(renewal[1]),
            later_ngp=float(alpha[1]),
            observed_alpha_decay_per_renewal=float(observed_slope),
        )

        self.assertEqual(selection["best_model"], "static_gamma")
        self.assertEqual(selection["static_gamma"]["passes"], 1.0)
        self.assertEqual(selection["poisson"]["passes"], 0.0)
        self.assertEqual(selection["finite_exchange"]["passes"], 0.0)
        self.assertAlmostEqual(selection["static_gamma"]["inferred_static_shape"], shape, delta=1e-3)

    def test_late_mechanism_selection_identifies_minimal_poisson_renewal(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        times = np.array([10000.0, 30000.0])
        renewal = delayed_poisson_mean(times, params)
        alpha = ngp_1d(times, params)
        gamma = 1.0 - math.exp(-0.5 * 1.1**2 * params.jump_variance)

        selection = late_mechanism_selection(
            wave_number=1.1,
            params=params,
            earlier_renewal_count=float(renewal[0]),
            earlier_ngp=float(alpha[0]),
            later_renewal_count=float(renewal[1]),
            later_ngp=float(alpha[1]),
            observed_alpha_decay_per_renewal=gamma,
        )

        self.assertEqual(selection["best_model"], "poisson")
        self.assertEqual(selection["poisson"]["passes"], 1.0)
        self.assertEqual(selection["static_gamma"]["passes"], 0.0)
        self.assertEqual(selection["finite_exchange"]["passes"], 0.0)

    def test_static_gamma_null_has_nonzero_long_time_ngp_plateau(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        shape = 0.4
        times = np.array([30.0, 30000.0])

        count = static_gamma_count_moments(times, params, shape)
        renewal = delayed_poisson_mean(times, params)
        alpha = static_gamma_ngp_1d(times, params, shape)

        np.testing.assert_allclose(count["mean"], renewal, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(count["variance"], renewal + renewal**2 / shape, rtol=1e-12, atol=1e-14)
        self.assertGreater(alpha[-1], 2.45)
        self.assertAlmostEqual(alpha[-1], 1.0 / shape, delta=0.01)

    def test_static_gamma_alpha_decay_per_renewal_vanishes_at_long_times(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        shape = 0.4
        wave_number = 1.1
        times = np.array([300.0, 30000.0, 60000.0])

        decay = static_gamma_normalized_alpha_decay(wave_number, times, params, shape)
        renewal = delayed_poisson_mean(times, params)
        slopes = -np.log(decay) / renewal
        diagnostics = static_gamma_asymptotic_diagnostics(wave_number, params, shape)

        self.assertLess(slopes[-1], slopes[0] / 20.0)
        self.assertLess(slopes[-1], 0.002)
        self.assertEqual(diagnostics["late_alpha_decay_per_renewal"], 0.0)
        self.assertAlmostEqual(diagnostics["late_ngp_plateau"], 1.0 / shape, delta=1e-12)

    def test_static_gamma_null_contrasts_finite_exchange_gaussian_recovery(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        late_time = np.array([30000.0])

        static_alpha = static_gamma_ngp_1d(late_time, params, heterogeneity.shape)[0]
        exchange_alpha = gamma_exchange_ngp_1d(late_time, params, heterogeneity)[0]

        self.assertGreater(static_alpha, 2.45)
        self.assertLess(exchange_alpha, 0.01)
        self.assertGreater(static_alpha / exchange_alpha, 500.0)

    def test_correlated_domain_susceptibility_scales_renewal_component(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.linspace(0.0, 220.0, 900)
        single_particle = renewal_scattering_susceptibility(wave_number, times, params)
        correlated = correlated_domain_susceptibility(
            wave_number,
            times,
            params,
            correlation_size=7.5,
        )

        np.testing.assert_allclose(correlated, 7.5 * single_particle, rtol=1e-12, atol=1e-14)
        self.assertGreater(float(np.max(correlated)), float(np.max(single_particle)))

    def test_renewal_correlation_size_inverts_observed_chi4_peak(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        times = np.linspace(0.0, 220.0, 900)
        single_particle = renewal_scattering_susceptibility(wave_number, times, params)
        observed_peak = 12.0 * float(np.max(single_particle))

        inferred = infer_renewal_correlation_size(
            observed_chi4_peak=observed_peak,
            wave_number=wave_number,
            t=times,
            params=params,
        )

        self.assertAlmostEqual(inferred["correlation_size"], 12.0)
        self.assertAlmostEqual(inferred["model_single_particle_peak"], float(np.max(single_particle)))
        self.assertGreater(inferred["peak_time"], 0.0)

    def test_renewal_correlation_size_validates_observed_peak(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        with self.assertRaises(ValueError):
            infer_renewal_correlation_size(
                observed_chi4_peak=0.0,
                wave_number=1.1,
                t=np.linspace(0.0, 10.0, 20),
                params=params,
            )

    def test_spatial_facilitation_domain_maps_persistence_time_to_correlation_volume(self):
        domain_fn = getattr(sys.modules["renewal_cage"], "spatial_facilitation_domain", None)
        if domain_fn is None:
            self.fail("spatial_facilitation_domain is missing")

        domain = domain_fn(
            persistence_time=9.0,
            dimension=3,
            particle_density=0.85,
            facilitation_diffusivity=0.05,
            microscopic_length=1.0,
        )
        expected_length = math.sqrt(1.0**2 + 2.0 * 3.0 * 0.05 * 9.0)
        expected_volume = 4.0 * math.pi * expected_length**3 / 3.0

        self.assertAlmostEqual(domain["dynamic_correlation_length"], expected_length)
        self.assertAlmostEqual(domain["correlation_size"], 0.85 * expected_volume)
        self.assertEqual(domain["front_dynamic_exponent"], 2.0)

    def test_spatial_facilitation_chi4_scan_grows_on_cooling(self):
        scan_fn = getattr(sys.modules["renewal_cage"], "spatial_facilitation_chi4_scan", None)
        if scan_fn is None:
            self.fail("spatial_facilitation_chi4_scan is missing")
        law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.25,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=1.0,
            delay_activation=2.5,
        )

        rows = scan_fn(
            temperatures=np.array([1.0, 0.8, 0.65]),
            law=law,
            wave_number=1.1,
            facilitation_diffusivity=0.04,
            particle_density=0.85,
            time_points=250,
        )

        lengths = [row["dynamic_correlation_length"] for row in rows]
        sizes = [row["correlation_size"] for row in rows]
        peaks = [row["chi4_peak"] for row in rows]
        self.assertTrue(all(later > earlier for earlier, later in zip(lengths, lengths[1:])))
        self.assertTrue(all(later > earlier for earlier, later in zip(sizes, sizes[1:])))
        self.assertTrue(all(later > earlier for earlier, later in zip(peaks, peaks[1:])))
        self.assertGreater(rows[-1]["chi4_peak_growth"], 1.0)

    def test_spatial_facilitation_diffusivity_inversion_recovers_constant_front_law(self):
        persistence_times = np.array([3.0, 7.0, 15.0])
        true_diffusivity = 0.04
        observed_lengths = np.sqrt(1.0 + 2.0 * 3.0 * true_diffusivity * persistence_times)

        inferred = infer_spatial_facilitation_diffusivity(
            persistence_times=persistence_times,
            observed_dynamic_lengths=observed_lengths,
            dimension=3,
            microscopic_length=1.0,
        )
        summary = spatial_facilitation_growth_law_consistency(
            persistence_times=persistence_times,
            observed_dynamic_lengths=observed_lengths,
            observed_diffusive_front_growth=True,
            dimension=3,
            microscopic_length=1.0,
            max_diffusivity_relative_std=1.0e-12,
            min_length_growth=1.5,
        )

        self.assertTrue(all(abs(row["inferred_facilitation_diffusivity"] - true_diffusivity) < 1.0e-12 for row in inferred))
        self.assertLess(summary["facilitation_diffusivity_relative_std"], 1.0e-12)
        self.assertGreater(summary["length_growth"], 1.5)
        self.assertEqual(summary["model_predicts_diffusive_front_growth"], 1.0)
        self.assertEqual(summary["facilitation_growth_law_consistent"], 1.0)
        self.assertEqual(summary["overall_consistent"], 1.0)

    def test_spatial_facilitation_growth_law_rejects_nonconstant_front_diffusivity(self):
        persistence_times = np.array([3.0, 7.0, 15.0])
        true_diffusivity = 0.04
        observed_lengths = np.sqrt(1.0 + 2.0 * 3.0 * true_diffusivity * persistence_times)
        observed_lengths[-1] *= 1.25

        summary = spatial_facilitation_growth_law_consistency(
            persistence_times=persistence_times,
            observed_dynamic_lengths=observed_lengths,
            observed_diffusive_front_growth=True,
            dimension=3,
            microscopic_length=1.0,
            max_diffusivity_relative_std=0.05,
            min_length_growth=1.5,
        )

        self.assertGreater(summary["facilitation_diffusivity_relative_std"], 0.05)
        self.assertEqual(summary["model_predicts_diffusive_front_growth"], 0.0)
        self.assertEqual(summary["facilitation_growth_law_consistent"], 0.0)
        self.assertEqual(summary["overall_consistent"], 0.0)

    def test_activated_barrier_gap_controls_delayed_renewal_product(self):
        barrier = ActivatedBarrierParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            renewal_rate_barrier=2.0,
            delay_onset_barrier=5.0,
            cage_stiffening_barrier=0.2,
            jump_to_cage_barrier=0.25,
        )
        law = activated_barrier_temperature_law(barrier)
        hot = temperature_dependent_params(1.0, law)
        cold_temperature = 0.62
        cold = temperature_dependent_params(cold_temperature, law)
        delta = 1.0 / cold_temperature - 1.0 / barrier.reference_temperature
        expected_ratio = math.exp((barrier.delay_onset_barrier - barrier.renewal_rate_barrier) * delta)

        self.assertAlmostEqual(law.delay_activation - law.rate_activation, 3.0)
        self.assertAlmostEqual(
            cold.renewal_rate * cold.renewal_delay / (hot.renewal_rate * hot.renewal_delay),
            expected_ratio,
        )
        self.assertGreater(cold.renewal_rate * cold.renewal_delay, hot.renewal_rate * hot.renewal_delay)

    def test_temperature_law_encodes_cooling_trends(self):
        law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=3.0,
            cage_stiffening=0.25,
            jump_to_cage_growth=0.35,
        )
        hot = temperature_dependent_params(1.0, law)
        cold = temperature_dependent_params(0.62, law)

        self.assertLess(cold.renewal_rate, hot.renewal_rate)
        self.assertGreater(cold.renewal_delay, hot.renewal_delay)
        self.assertLess(cold.cage_variance, hot.cage_variance)
        self.assertGreater(cold.jump_variance / cold.cage_variance, hot.jump_variance / hot.cage_variance)

    def test_temperature_scan_produces_stokes_einstein_decoupling(self):
        law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=5.0,
            cage_stiffening=0.2,
            jump_to_cage_growth=0.25,
        )
        temperatures = np.array([1.0, 0.85, 0.72, 0.62])
        rows = temperature_scan(temperatures, law, wave_number=1.1)

        diffusion = np.array([row["diffusion_coefficient"] for row in rows])
        tau_alpha = np.array([row["tau_alpha"] for row in rows])
        se_ratio = np.array([row["normalized_stokes_einstein_product"] for row in rows])

        self.assertTrue(np.all(np.diff(diffusion) < 0.0))
        self.assertTrue(np.all(np.diff(tau_alpha) > 0.0))
        self.assertGreater(se_ratio[-1], 2.0)
        self.assertAlmostEqual(se_ratio[0], 1.0)
        first = temperature_dependent_params(float(temperatures[0]), law)
        self.assertAlmostEqual(rows[0]["stokes_einstein_product"], stokes_einstein_product(1.1, first))
        self.assertAlmostEqual(rows[0]["diffusion_coefficient"], long_time_diffusion_coefficient(first))
        self.assertIn("fractional_stokes_einstein_exponent", rows[-1])
        self.assertGreater(rows[-1]["fractional_stokes_einstein_exponent"], 0.0)
        self.assertLess(rows[-1]["fractional_stokes_einstein_exponent"], 1.0)
        self.assertIn("apparent_alpha_activation_energy", rows[-1])
        self.assertIn("local_fragility_index", rows[-1])
        self.assertGreater(rows[-1]["apparent_alpha_activation_energy"], rows[0]["apparent_alpha_activation_energy"])
        self.assertGreater(rows[-1]["local_fragility_index"], rows[0]["local_fragility_index"])

    def test_configurational_entropy_law_extrapolates_to_kauzmann_temperature(self):
        entropy_cls = getattr(sys.modules["renewal_cage"], "ConfigurationalEntropyParams", None)
        entropy_fn = getattr(sys.modules["renewal_cage"], "configurational_entropy", None)
        heat_fn = getattr(sys.modules["renewal_cage"], "excess_heat_capacity", None)
        if entropy_cls is None or entropy_fn is None or heat_fn is None:
            self.fail("thermodynamic entropy closure is missing")

        law = entropy_cls(reference_temperature=1.0, entropy_ref=1.2, kauzmann_temperature=0.45)
        temperatures = np.array([1.0, 0.7, 0.5])
        entropy = entropy_fn(temperatures, law)
        heat_capacity = heat_fn(temperatures, law)

        self.assertTrue(np.all(np.diff(entropy) < 0.0))
        self.assertAlmostEqual(float(entropy_fn(np.array([0.45]), law)[0]), 0.0)
        self.assertTrue(np.all(heat_capacity > 0.0))
        self.assertGreater(1.0 / (temperatures[-1] * entropy[-1]), 1.0 / (temperatures[0] * entropy[0]))

    def test_adam_gibbs_thermodynamic_scan_links_entropy_to_renewal_slowdown(self):
        entropy_cls = getattr(sys.modules["renewal_cage"], "ConfigurationalEntropyParams", None)
        scan_fn = getattr(sys.modules["renewal_cage"], "adam_gibbs_thermodynamic_scan", None)
        if entropy_cls is None or scan_fn is None:
            self.fail("Adam-Gibbs thermodynamic scan is missing")
        law = entropy_cls(reference_temperature=1.0, entropy_ref=1.2, kauzmann_temperature=0.45)
        temperatures = np.array([1.0, 0.8, 0.62, 0.5])

        rows = scan_fn(
            temperatures=temperatures,
            entropy_law=law,
            activation_free_energy=1.6,
            tau_ref=3.0,
            renewal_rate_ref=0.18,
            wave_number=1.1,
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
        )

        entropy = np.array([row["configurational_entropy"] for row in rows])
        tau_ag = np.array([row["adam_gibbs_tau"] for row in rows])
        tau_alpha = np.array([row["tau_alpha"] for row in rows])
        self.assertTrue(np.all(np.diff(entropy) < 0.0))
        self.assertTrue(np.all(np.diff(tau_ag) > 0.0))
        self.assertTrue(np.all(np.diff(tau_alpha) > 0.0))
        self.assertGreater(rows[-1]["thermodynamic_slowdown"], 10.0)
        self.assertGreater(rows[-1]["inverse_entropy_control"], rows[0]["inverse_entropy_control"])
        self.assertGreater(rows[-1]["excess_heat_capacity"], 0.0)

    def test_mct_beta_correlator_has_critical_and_von_schweidler_slopes(self):
        beta = MCTBetaParams(
            plateau=0.72,
            critical_amplitude=0.09,
            von_schweidler_amplitude=0.08,
            critical_exponent=0.31,
            von_schweidler_exponent=0.58,
            beta_time=12.0,
        )
        time = np.geomspace(1.2, 120.0, 160)
        correlator = mct_beta_correlator(time, beta)

        early = time < beta.beta_time
        late = time > beta.beta_time
        early_slope = np.polyfit(
            np.log(time[early] / beta.beta_time),
            np.log(correlator[early] - beta.plateau),
            1,
        )[0]
        late_slope = np.polyfit(
            np.log(time[late] / beta.beta_time),
            np.log(beta.plateau - correlator[late]),
            1,
        )[0]

        self.assertAlmostEqual(early_slope, -beta.critical_exponent, delta=0.01)
        self.assertAlmostEqual(late_slope, beta.von_schweidler_exponent, delta=0.01)
        self.assertLess(np.max(correlator), 1.0)
        self.assertGreater(np.min(correlator), 0.0)

    def test_mct_beta_temperature_scan_links_plateau_window_to_alpha_crossover(self):
        base = MCTBetaParams(
            plateau=0.68,
            critical_amplitude=0.08,
            von_schweidler_amplitude=0.05,
            critical_exponent=0.32,
            von_schweidler_exponent=0.6,
            beta_time=4.0,
        )
        temperatures = np.array([1.0, 0.82, 0.68, 0.58])

        rows = mct_beta_temperature_scan(
            temperatures=temperatures,
            base=base,
            beta_time_activation=2.4,
            plateau_growth=0.14,
            alpha_time_ref=30.0,
            alpha_activation=5.0,
        )

        beta_time = np.array([row["beta_time"] for row in rows])
        plateau = np.array([row["plateau"] for row in rows])
        separation = np.array([row["alpha_beta_separation"] for row in rows])
        self.assertTrue(np.all(np.diff(beta_time) > 0.0))
        self.assertTrue(np.all(np.diff(plateau) > 0.0))
        self.assertTrue(np.all(np.diff(separation) > 0.0))
        self.assertGreater(rows[-1]["von_schweidler_exit_time"], rows[-1]["beta_time"])
        self.assertLess(rows[-1]["von_schweidler_exit_time"], rows[-1]["alpha_time"])

    def test_mct_beta_benchmark_consistency_matches_kob_andersen_window(self):
        beta = MCTBetaParams(
            plateau=0.68,
            critical_amplitude=0.08,
            von_schweidler_amplitude=0.05,
            critical_exponent=0.32,
            von_schweidler_exponent=0.6,
            beta_time=4.0,
        )

        row = mct_beta_benchmark_consistency(
            beta,
            benchmark_id="kob_andersen_1995_beta_window",
            observed_critical_decay=False,
            observed_von_schweidler=True,
            observation_min_time=0.85 * beta.beta_time,
            observation_max_time=500.0 * beta.beta_time,
            alpha_time=80.0 * beta.beta_time,
            required_decades=0.5,
        )

        self.assertLess(row["critical_window_decades"], 0.5)
        self.assertGreater(row["von_schweidler_window_decades"], 0.5)
        self.assertEqual(row["model_predicts_visible_critical_decay"], 0.0)
        self.assertEqual(row["model_predicts_visible_von_schweidler"], 1.0)
        self.assertEqual(row["critical_decay_consistent"], 1.0)
        self.assertEqual(row["von_schweidler_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_mct_exponent_benchmark_consistency_checks_common_lambda_relation(self):
        row = mct_exponent_benchmark_consistency(
            benchmark_id="kob_andersen_1995_mct_exponent_parameter",
            observed_common_exponent_parameter=True,
            critical_exponent=0.32,
            von_schweidler_exponent=0.60,
            max_lambda_relative_mismatch=0.05,
        )

        self.assertEqual(row["model_predicts_common_exponent_parameter"], 1.0)
        self.assertLess(row["lambda_relative_mismatch"], 0.05)
        self.assertAlmostEqual(row["lambda_from_a"], 0.716312910468668, places=12)
        self.assertAlmostEqual(row["lambda_from_b"], 0.7246032624007417, places=12)
        self.assertEqual(row["mct_exponent_parameter_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_cage_localization_diagnostics_quantifies_debye_waller_plateau(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )

        row = cage_localization_diagnostics(
            wave_number=1.1,
            plateau_time=1.0,
            params=params,
        )

        self.assertAlmostEqual(row["debye_waller_plateau"], math.exp(-0.5 * 1.1**2), places=12)
        self.assertLess(row["renewal_msd_fraction"], 0.02)
        self.assertGreater(row["alpha_to_cage_time_ratio"], 50.0)
        self.assertGreater(row["cage_plateau_msd"], 0.95)
        self.assertLess(row["cage_plateau_msd"], 1.05)

    def test_cage_localization_benchmark_consistency_detects_cage_plateau(self):
        row = cage_localization_benchmark_consistency(
            benchmark_id="debye_waller_cage_localization",
            observed_cage_localization=True,
            debye_waller_plateau=0.5460744266397094,
            renewal_msd_fraction=0.0048,
            alpha_to_cage_time_ratio=147.0,
            min_debye_waller_plateau=0.2,
            max_debye_waller_plateau=0.95,
            max_renewal_msd_fraction=0.05,
            min_alpha_to_cage_time_ratio=20.0,
        )

        self.assertEqual(row["model_predicts_cage_localization"], 1.0)
        self.assertEqual(row["debye_waller_consistent"], 1.0)
        self.assertEqual(row["renewal_fraction_consistent"], 1.0)
        self.assertEqual(row["alpha_separation_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_gaussian_recovery_benchmark_consistency_rejects_static_disorder(self):
        row = gaussian_recovery_benchmark_consistency(
            benchmark_id="gaussian_recovery_finite_exchange_vs_static_disorder",
            observed_gaussian_recovery=True,
            finite_exchange_late_ngp=0.0048,
            static_gamma_late_ngp=2.5,
            recovery_threshold=0.05,
        )

        self.assertEqual(row["model_predicts_gaussian_recovery"], 1.0)
        self.assertEqual(row["static_null_predicts_gaussian_recovery"], 0.0)
        self.assertEqual(row["finite_exchange_recovery_consistent"], 1.0)
        self.assertEqual(row["static_null_recovery_consistent"], 0.0)
        self.assertEqual(row["mechanism_selection_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_ngp_peak_benchmark_consistency_detects_cooling_peak_shift(self):
        row = ngp_peak_benchmark_consistency(
            benchmark_id="ngp_peak_shift_on_cooling",
            observed_transient_ngp_peak=True,
            hot_peak_time=11.0,
            cold_peak_time=70.0,
            hot_peak_ngp=0.12,
            cold_peak_ngp=0.28,
            late_ngp=0.0048,
            min_peak_time_growth=2.0,
            min_peak_height=0.05,
            min_peak_height_growth=1.2,
            max_late_ngp=0.05,
        )

        self.assertEqual(row["model_predicts_transient_ngp_peak"], 1.0)
        self.assertEqual(row["peak_time_growth_consistent"], 1.0)
        self.assertEqual(row["peak_height_consistent"], 1.0)
        self.assertEqual(row["peak_height_growth_consistent"], 1.0)
        self.assertEqual(row["late_recovery_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["peak_time_growth"], row["min_peak_time_growth"])
        self.assertGreater(row["peak_height_growth"], row["min_peak_height_growth"])

    def test_stokes_einstein_benchmark_consistency_detects_fractional_decoupling(self):
        row = stokes_einstein_benchmark_consistency(
            benchmark_id="stokes_einstein_fractional_decoupling",
            observed_stokes_einstein_violation=True,
            hot_se_product=1.0,
            cold_se_product=2.03,
            cold_fractional_exponent=0.568,
            min_product_growth=1.5,
            max_fractional_exponent=0.9,
        )

        self.assertEqual(row["model_predicts_stokes_einstein_violation"], 1.0)
        self.assertEqual(row["se_product_growth_consistent"], 1.0)
        self.assertEqual(row["fractional_exponent_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["se_product_growth"], 2.0)
        self.assertLess(row["cold_fractional_exponent"], 1.0)

    def test_dynamic_heterogeneity_benchmark_consistency_detects_chi4_growth(self):
        row = dynamic_heterogeneity_benchmark_consistency(
            benchmark_id="dynamic_heterogeneity_chi4_growth",
            observed_dynamic_heterogeneity_growth=True,
            length_growth=3.09,
            correlation_size_growth=29.5,
            chi4_peak_growth=34.7,
            min_length_growth=1.5,
            min_correlation_size_growth=2.0,
            min_chi4_peak_growth=2.0,
        )

        self.assertEqual(row["model_predicts_dynamic_heterogeneity_growth"], 1.0)
        self.assertEqual(row["length_growth_consistent"], 1.0)
        self.assertEqual(row["correlation_size_growth_consistent"], 1.0)
        self.assertEqual(row["chi4_peak_growth_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_alpha_tts_benchmark_consistency_detects_shape_breakdown(self):
        row = alpha_tts_benchmark_consistency(
            benchmark_id="alpha_tts_breakdown_shape_residual",
            observed_tts_breakdown=True,
            cold_shape_residual=0.611,
            alpha_shape_control_growth=6.44,
            residual_threshold=0.25,
            min_control_growth=2.0,
        )

        self.assertEqual(row["model_predicts_tts_breakdown"], 1.0)
        self.assertEqual(row["tts_residual_consistent"], 1.0)
        self.assertEqual(row["tts_control_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["cold_shape_residual"], row["residual_threshold"])
        self.assertGreater(row["alpha_shape_control_growth"], row["min_control_growth"])

    def test_kww_alpha_fit_recovers_stretched_exponent_from_window(self):
        time = np.logspace(-1.0, 2.0, 300)
        tau = 7.5
        beta = 0.62
        decay = np.exp(-((time / tau) ** beta))

        fit = kww_alpha_fit(
            time,
            decay,
            min_decay=0.12,
            max_decay=0.88,
        )

        self.assertAlmostEqual(fit["kww_beta"], beta, places=12)
        self.assertAlmostEqual(fit["kww_tau"], tau, places=12)
        self.assertLess(fit["rms_log_residual"], 1.0e-12)
        self.assertGreaterEqual(fit["points_used"], 50)

    def test_stretched_alpha_benchmark_consistency_detects_cooling_stretching(self):
        row = stretched_alpha_benchmark_consistency(
            benchmark_id="kww_alpha_stretching_on_cooling",
            observed_stretched_alpha=True,
            hot_kww_beta=0.92,
            cold_kww_beta=0.62,
            min_beta_drop=0.15,
            max_cold_beta=0.8,
            max_fit_residual=0.04,
            cold_fit_residual=0.012,
        )

        self.assertEqual(row["model_predicts_stretched_alpha"], 1.0)
        self.assertEqual(row["beta_drop_consistent"], 1.0)
        self.assertEqual(row["cold_beta_consistent"], 1.0)
        self.assertEqual(row["fit_quality_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertAlmostEqual(row["kww_beta_drop"], 0.30)

    def test_persistence_exchange_benchmark_consistency_checks_inversion_protocol(self):
        row = persistence_exchange_benchmark_consistency(
            benchmark_id="persistence_exchange_transport_inversion",
            observed_persistence_exchange_decoupling=True,
            inferred_persistence_exchange_ratio=9.0,
            late_ngp_log_residual=0.0,
            invalid_poisson_alpha_rejected=True,
            min_persistence_exchange_ratio=2.0,
            max_late_ngp_abs_log_residual=0.1,
        )

        self.assertEqual(row["model_predicts_persistence_exchange_decoupling"], 1.0)
        self.assertEqual(row["persistence_exchange_ratio_consistent"], 1.0)
        self.assertEqual(row["persistence_exchange_late_ngp_consistent"], 1.0)
        self.assertEqual(row["persistence_exchange_rejection_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["inferred_persistence_exchange_ratio"], row["min_persistence_exchange_ratio"])
        self.assertLess(abs(row["late_ngp_log_residual_benchmark"]), row["max_late_ngp_abs_log_residual"])

    def test_joint_inversion_benchmark_consistency_checks_multik_chi4_and_rejection(self):
        row = joint_inversion_benchmark_consistency(
            benchmark_id="joint_persistence_exchange_multik_chi4_protocol",
            observed_joint_inversion_closure=True,
            inferred_persistence_exchange_ratio=8.0,
            stokes_einstein_growth_over_poisson=3.56,
            max_multik_tau_alpha_abs_log_residual=0.0,
            late_ngp_log_residual=0.0,
            chi4_peak_growth_over_poisson=2.58,
            rejected_mismatch_abs_log_residual=0.223,
            min_persistence_exchange_ratio=2.0,
            min_stokes_einstein_growth=2.0,
            max_multik_abs_log_residual=0.02,
            max_late_ngp_abs_log_residual=0.02,
            min_chi4_peak_growth=1.5,
            min_rejected_mismatch_abs_log_residual=0.1,
        )

        self.assertEqual(row["model_predicts_joint_inversion_closure"], 1.0)
        self.assertEqual(row["joint_ratio_consistent"], 1.0)
        self.assertEqual(row["joint_se_consistent"], 1.0)
        self.assertEqual(row["joint_multik_consistent"], 1.0)
        self.assertEqual(row["joint_late_ngp_consistent"], 1.0)
        self.assertEqual(row["joint_chi4_consistent"], 1.0)
        self.assertEqual(row["joint_mismatch_rejected"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_literature_inversion_readiness_separates_qualitative_from_quantitative_data(self):
        row = literature_inversion_readiness(
            benchmark_id="kob_andersen_van_hove_1995",
            benchmark_source="kob1995vanhove",
            required_observables=["time_grid", "van_hove_tail", "ngp", "diffusion"],
            available_observables=["time_grid", "van_hove_tail", "ngp"],
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
            next_action="digitize curves or rerun public simulation",
        )

        self.assertAlmostEqual(row["observable_coverage_fraction"], 0.75)
        self.assertEqual(row["missing_observables"], "diffusion")
        self.assertEqual(row["qualitative_comparison_ready"], 1.0)
        self.assertEqual(row["quantitative_inversion_ready"], 0.0)
        self.assertEqual(row["uncertainty_weighted_ready"], 0.0)

    def test_real_benchmark_assimilation_gate_marks_uncertainty_weighted_inversion_ready(self):
        row = real_benchmark_assimilation_gate(
            benchmark_id="public_ka_alpha_vanhove_release",
            source_key="kob1995vanhove;kob1995intermediate",
            target_protocol="alpha_vanhove_transport",
            available_observables=[
                "time_grid",
                "temperature_grid",
                "wave_numbers",
                "self_intermediate_scattering",
                "van_hove_tail",
                "ngp",
                "diffusion",
            ],
            has_shared_system=True,
            has_machine_readable_curves=True,
            has_uncertainty_estimates=True,
            model_scope="dynamical_signature",
        )

        self.assertEqual(row["assimilation_stage"], "uncertainty_weighted_inversion")
        self.assertEqual(row["structural_inversion_ready"], 1.0)
        self.assertEqual(row["uncertainty_weighted_ready"], 1.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_real_benchmark_assimilation_gate_blocks_structural_data_without_uncertainties(self):
        row = real_benchmark_assimilation_gate(
            benchmark_id="digitized_ka_alpha_vanhove_candidate",
            source_key="kob1995vanhove;kob1995intermediate",
            target_protocol="alpha_vanhove_transport",
            available_observables=[
                "time_grid",
                "temperature_grid",
                "wave_numbers",
                "self_intermediate_scattering",
                "van_hove_tail",
                "ngp",
                "diffusion",
            ],
            has_shared_system=True,
            has_machine_readable_curves=True,
            has_uncertainty_estimates=False,
            model_scope="dynamical_signature",
        )

        self.assertEqual(row["assimilation_stage"], "structural_digitization_ready")
        self.assertEqual(row["structural_inversion_ready"], 1.0)
        self.assertEqual(row["uncertainty_weighted_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "uncertainty_columns")

    def test_real_benchmark_assimilation_gate_keeps_thermodynamics_as_scope_boundary(self):
        row = real_benchmark_assimilation_gate(
            benchmark_id="kauzmann_entropy_boundary",
            source_key="kauzmann1948nature;adam1965temperature",
            target_protocol="thermodynamic_entropy_closure",
            available_observables=["temperature_grid", "configurational_entropy", "tau_alpha"],
            has_shared_system=True,
            has_machine_readable_curves=True,
            has_uncertainty_estimates=True,
            model_scope="thermodynamic_transition",
        )

        self.assertEqual(row["assimilation_stage"], "scope_boundary_only")
        self.assertEqual(row["structural_inversion_ready"], 0.0)
        self.assertEqual(row["uncertainty_weighted_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "renewal_dynamics_not_thermodynamic_theory")

    def test_cross_observable_prediction_ledger_marks_heldout_predictive_diagnostic(self):
        row = cross_observable_prediction_ledger(
            protocol_id="joint_persistence_exchange_multik_chi4",
            source_key="synthetic_joint_protocol",
            model_scope="dynamical_signature",
            support_level="derived",
            calibration_observables=["diffusion", "anchor_tau_alpha"],
            heldout_predictions=["multi_k_tau_alpha", "late_ngp", "stokes_einstein_product"],
            closure_observables=[],
            failed_predictions=[],
        )

        self.assertEqual(row["prediction_class"], "predictive_diagnostic")
        self.assertEqual(row["calibration_count"], 2.0)
        self.assertEqual(row["heldout_prediction_count"], 3.0)
        self.assertEqual(row["fit_only_overclaim_risk"], 0.0)
        self.assertEqual(row["all_heldout_predictions_pass"], 1.0)

    def test_cross_observable_prediction_ledger_flags_fit_only_overclaim_risk(self):
        row = cross_observable_prediction_ledger(
            protocol_id="single_alpha_fit_only",
            source_key="hypothetical_alpha_fit",
            model_scope="dynamical_signature",
            support_level="derived",
            calibration_observables=["tau_alpha"],
            heldout_predictions=[],
            closure_observables=[],
            failed_predictions=[],
        )

        self.assertEqual(row["prediction_class"], "underconstrained_fit")
        self.assertEqual(row["heldout_prediction_count"], 0.0)
        self.assertEqual(row["fit_only_overclaim_risk"], 1.0)

    def test_cross_observable_prediction_ledger_keeps_closure_boundary_separate(self):
        row = cross_observable_prediction_ledger(
            protocol_id="spatial_chi4_front_closure",
            source_key="lacevic2003fourpoint",
            model_scope="spatial_heterogeneity",
            support_level="effective_closure",
            calibration_observables=["tau_alpha", "diffusion"],
            heldout_predictions=["chi4_peak", "dynamic_length"],
            closure_observables=["front_diffusivity"],
            failed_predictions=[],
        )

        self.assertEqual(row["prediction_class"], "closure_assisted_prediction")
        self.assertEqual(row["closure_observable_count"], 1.0)
        self.assertEqual(row["requires_external_closure"], 1.0)
        self.assertEqual(row["fit_only_overclaim_risk"], 0.0)

    def test_inversion_identifiability_audit_marks_heldout_protocol_identifiable(self):
        row = inversion_identifiability_audit(
            protocol_id="joint_persistence_exchange_multik_chi4",
            source_key="raw_curve_persistence_exchange_protocol",
            model_scope="transport_decoupling",
            fit_observables=["diffusion", "anchor_tau_alpha"],
            inferred_parameters=["exchange_time", "persistence_time"],
            heldout_predictions=["multi_k_tau_alpha", "late_ngp", "stokes_einstein_product"],
            external_closures=[],
            degenerate_parameters=[],
        )

        self.assertEqual(row["identifiability_class"], "identifiable_prediction")
        self.assertEqual(row["rank_margin"], 0.0)
        self.assertEqual(row["heldout_prediction_count"], 3.0)
        self.assertEqual(row["overclaim_risk"], 0.0)

    def test_inversion_identifiability_audit_flags_alpha_only_underidentified_fit(self):
        row = inversion_identifiability_audit(
            protocol_id="single_alpha_fit_only_null",
            source_key="hypothetical_alpha_only_fit",
            model_scope="dynamical_signature",
            fit_observables=["tau_alpha"],
            inferred_parameters=["exchange_time", "persistence_time"],
            heldout_predictions=[],
            external_closures=[],
            degenerate_parameters=[],
        )

        self.assertEqual(row["identifiability_class"], "underidentified_fit")
        self.assertEqual(row["rank_margin"], -1.0)
        self.assertEqual(row["overclaim_risk"], 1.0)

    def test_inversion_identifiability_audit_separates_closure_and_thermodynamic_boundaries(self):
        spatial = inversion_identifiability_audit(
            protocol_id="spatial_chi4_front_closure",
            source_key="lacevic2003fourpoint",
            model_scope="spatial_heterogeneity",
            fit_observables=["tau_alpha", "diffusion", "chi4_peak"],
            inferred_parameters=["correlation_length", "front_diffusivity"],
            heldout_predictions=["dynamic_length"],
            external_closures=["front_diffusivity_law"],
            degenerate_parameters=[],
        )
        thermo = inversion_identifiability_audit(
            protocol_id="thermodynamic_entropy_boundary",
            source_key="kauzmann1948nature",
            model_scope="thermodynamic_transition",
            fit_observables=["configurational_entropy", "temperature_grid"],
            inferred_parameters=["kauzmann_temperature", "entropy_slope"],
            heldout_predictions=["heat_capacity_anomaly"],
            external_closures=["entropy_law"],
            degenerate_parameters=[],
        )

        self.assertEqual(spatial["identifiability_class"], "conditionally_identifiable")
        self.assertEqual(spatial["requires_external_closure"], 1.0)
        self.assertEqual(spatial["overclaim_risk"], 0.0)
        self.assertEqual(thermo["identifiability_class"], "scope_boundary")
        self.assertEqual(thermo["overclaim_risk"], 0.0)

    def test_frontier_benchmark_horizon_marks_trajectory_reanalysis_candidate(self):
        row = frontier_benchmark_horizon(
            benchmark_id="glassbench_trajectory_horizon",
            source_key="jung2025roadmap_glassbench",
            source_year=2025,
            model_scope="dynamical_signature",
            target_protocol="alpha_vanhove_transport",
            available_observables=[
                "particle_trajectories",
                "time_grid",
                "temperature_grid",
                "structure",
                "local_mobility_labels",
            ],
            required_observables=[
                "particle_trajectories",
                "time_grid",
                "temperature_grid",
                "self_intermediate_scattering",
                "ngp",
                "diffusion",
            ],
            has_machine_readable_repository=True,
            has_uncertainty_estimates=False,
            has_shared_transport_grid=True,
            requires_external_closure=False,
            model_extension_required=False,
        )

        self.assertEqual(row["horizon_class"], "trajectory_reanalysis_candidate")
        self.assertEqual(row["can_compute_missing_from_trajectories"], 1.0)
        self.assertIn("self_intermediate_scattering", row["computable_missing_observables"])
        self.assertEqual(row["overclaim_risk"], 0.0)

    def test_frontier_benchmark_horizon_marks_transport_heterogeneity_candidate(self):
        row = frontier_benchmark_horizon(
            benchmark_id="gst_nn_potential_transport_horizon",
            source_key="marcorini2025gst_dynamic_heterogeneity",
            source_year=2025,
            model_scope="transport_decoupling",
            target_protocol="joint_persistence_exchange_multik_chi4",
            available_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "stokes_einstein_product",
                "chi4_peak",
                "fragility_proxy",
            ],
            required_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "late_ngp",
                "multi_k_tau_alpha",
                "chi4_peak",
            ],
            has_machine_readable_repository=True,
            has_uncertainty_estimates=False,
            has_shared_transport_grid=True,
            requires_external_closure=False,
            model_extension_required=False,
        )

        self.assertEqual(row["horizon_class"], "transport_heterogeneity_candidate")
        self.assertEqual(row["primary_blocker"], "late_ngp")
        self.assertGreater(row["frontier_priority_score"], 0.5)

    def test_frontier_benchmark_horizon_marks_translation_rotation_candidate_and_scope_boundary(self):
        rotational = frontier_benchmark_horizon(
            benchmark_id="near_tg_molecular_motion_rotational_gap",
            source_key="simon2026molecular_motion",
            source_year=2026,
            model_scope="transport_decoupling",
            target_protocol="translation_rotation_persistence_exchange",
            available_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "rotational_relaxation",
                "stokes_einstein_product",
            ],
            required_observables=[
                "temperature_grid",
                "diffusion",
                "tau_alpha",
                "rotational_relaxation",
            ],
            has_machine_readable_repository=True,
            has_uncertainty_estimates=False,
            has_shared_transport_grid=True,
            requires_external_closure=False,
            model_extension_required=False,
        )
        thermo = frontier_benchmark_horizon(
            benchmark_id="heat_capacity_entropy_frontier",
            source_key="thermodynamic_calorimetry_candidate",
            source_year=2025,
            model_scope="thermodynamic_transition",
            target_protocol="thermodynamic_entropy_closure",
            available_observables=["temperature_grid", "configurational_entropy", "heat_capacity"],
            required_observables=["temperature_grid", "configurational_entropy", "heat_capacity"],
            has_machine_readable_repository=True,
            has_uncertainty_estimates=True,
            has_shared_transport_grid=True,
            requires_external_closure=True,
            model_extension_required=False,
        )

        self.assertEqual(rotational["horizon_class"], "structural_inversion_candidate")
        self.assertEqual(rotational["primary_blocker"], "uncertainty_estimates")
        self.assertEqual(rotational["model_extension_required"], 0.0)
        self.assertEqual(thermo["horizon_class"], "scope_boundary")
        self.assertEqual(thermo["overclaim_risk"], 0.0)

    def test_sota_source_provenance_gate_promotes_public_trajectory_repository(self):
        row = sota_source_provenance_gate(
            source_id="glassbench_zenodo_trajectory_release",
            citation_key="jung2025roadmap",
            source_type="dataset_repository",
            model_scope="dynamical_signature",
            provenance_items=[
                "doi",
                "repository_url",
                "machine_readable_files",
                "raw_particle_trajectories",
                "simulation_protocol_metadata",
                "license_or_terms",
            ],
            supported_observables=[
                "particle_trajectories",
                "time_grid",
                "temperature_grid",
                "structure",
            ],
            required_downstream_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_inversion_readiness_gate",
            ],
            has_reanalysis_permission=True,
        )

        self.assertEqual(row["provenance_stage"], "trajectory_reanalysis_source")
        self.assertEqual(row["can_enter_trajectory_protocol"], 1.0)
        self.assertEqual(row["can_enter_raw_curve_protocol"], 0.0)
        self.assertEqual(row["requires_digitization"], 0.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_sota_source_provenance_gate_blocks_citation_only_published_curves(self):
        row = sota_source_provenance_gate(
            source_id="hedges_persistence_exchange_jcp_article",
            citation_key="hedges2007persistence",
            source_type="article",
            model_scope="transport_decoupling",
            provenance_items=["doi", "published_figures"],
            supported_observables=["persistence_time", "exchange_time", "tau_alpha"],
            required_downstream_protocols=[
                "persistence_exchange_protocol",
                "persistence_exchange_uncertainty_protocol",
            ],
            has_reanalysis_permission=False,
        )

        self.assertEqual(row["provenance_stage"], "citation_only_source")
        self.assertEqual(row["can_enter_trajectory_protocol"], 0.0)
        self.assertEqual(row["can_enter_raw_curve_protocol"], 0.0)
        self.assertEqual(row["requires_digitization"], 1.0)
        self.assertEqual(row["primary_blocker"], "machine_readable_files")

    def test_sota_source_provenance_gate_keeps_thermodynamics_as_scope_boundary(self):
        row = sota_source_provenance_gate(
            source_id="kauzmann_entropy_thermodynamic_boundary",
            citation_key="kauzmann1948nature",
            source_type="article",
            model_scope="thermodynamic_transition",
            provenance_items=["doi", "published_figures", "thermodynamic_observables"],
            supported_observables=["configurational_entropy", "heat_capacity"],
            required_downstream_protocols=["thermodynamic_entropy_closure"],
            has_reanalysis_permission=True,
        )

        self.assertEqual(row["provenance_stage"], "scope_boundary_source")
        self.assertEqual(row["primary_blocker"], "renewal_dynamics_not_thermodynamic_theory")
        self.assertEqual(row["scope_boundary"], 1.0)

    def test_sota_data_accession_gate_marks_glassbench_remote_archive_ready(self):
        row = sota_data_accession_gate(
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            citation_key="jung2025roadmap",
            model_scope="dynamical_signature",
            landing_url="https://zenodo.org/records/10118191",
            doi="10.5281/zenodo.10118191",
            archive_name="GlassBench.zip",
            archive_md5="82c83a7146eb749e13417e4350022417",
            archive_size_bytes=6042260027,
            license_id="cc-by-4.0",
            has_public_landing_page=True,
            has_downloadable_archive=True,
            has_schema_or_readme=True,
            has_trajectory_files=True,
            has_precomputed_descriptors=True,
            local_cache_present=False,
            intended_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_inversion_readiness_gate",
            ],
        )

        self.assertEqual(row["accession_stage"], "remote_trajectory_accession_ready")
        self.assertEqual(row["accession_ready"], 1.0)
        self.assertEqual(row["ready_for_local_reanalysis"], 0.0)
        self.assertEqual(row["primary_blocker"], "local_cache")
        self.assertEqual(row["download_required"], 1.0)
        self.assertGreater(row["archive_size_gb"], 5.0)

    def test_sota_zenodo_record_fingerprint_verifies_glassbench_files_without_reanalysis(self):
        record = {
            "id": 10118191,
            "doi": "10.5281/zenodo.10118191",
            "metadata": {"license": {"id": "cc-by-4.0"}},
            "files": [
                {
                    "key": "README",
                    "size": 2147,
                    "checksum": "md5:f1a192f54a2fa7a2b3533af0011b80dc",
                },
                {
                    "key": "GlassBench.zip",
                    "size": 6042260027,
                    "checksum": "md5:82c83a7146eb749e13417e4350022417",
                },
            ],
        }

        row = sota_zenodo_record_fingerprint_gate(
            fingerprint_id="glassbench_zenodo_record_fingerprint",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            record=record,
            expected_doi="10.5281/zenodo.10118191",
            expected_license_id="cc-by-4.0",
            expected_archive_name="GlassBench.zip",
            expected_archive_md5="82c83a7146eb749e13417e4350022417",
            expected_archive_size_bytes=6042260027,
            expected_readme_name="README",
            expected_readme_md5="f1a192f54a2fa7a2b3533af0011b80dc",
            expected_readme_size_bytes=2147,
            large_archive_threshold_bytes=100_000_000,
        )

        self.assertEqual(row["fingerprint_stage"], "zenodo_record_verified")
        self.assertEqual(row["zenodo_record_fingerprint_ready"], 1.0)
        self.assertEqual(row["archive_size_matches"], 1.0)
        self.assertEqual(row["archive_md5_matches"], 1.0)
        self.assertEqual(row["readme_size_matches"], 1.0)
        self.assertEqual(row["license_matches"], 1.0)
        self.assertEqual(row["full_archive_download_required"], 1.0)
        self.assertEqual(row["real_reanalysis_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "archive_cache")

    def test_sota_data_accession_gate_blocks_article_without_archive(self):
        row = sota_data_accession_gate(
            accession_id="hedges_jcp_article_no_archive",
            source_id="hedges_persistence_exchange_jcp_article",
            citation_key="hedges2007persistence",
            model_scope="transport_decoupling",
            landing_url="https://doi.org/10.1063/1.2817607",
            doi="10.1063/1.2817607",
            archive_name="none",
            archive_md5="none",
            archive_size_bytes=0,
            license_id="article",
            has_public_landing_page=True,
            has_downloadable_archive=False,
            has_schema_or_readme=False,
            has_trajectory_files=False,
            has_precomputed_descriptors=False,
            local_cache_present=False,
            intended_protocols=["persistence_exchange_uncertainty_protocol"],
        )

        self.assertEqual(row["accession_stage"], "citation_only_no_accession")
        self.assertEqual(row["accession_ready"], 0.0)
        self.assertEqual(row["ready_for_local_reanalysis"], 0.0)
        self.assertEqual(row["primary_blocker"], "downloadable_archive")

    def test_sota_data_accession_gate_keeps_thermodynamic_accession_as_scope_boundary(self):
        row = sota_data_accession_gate(
            accession_id="kauzmann_entropy_scope_boundary",
            source_id="kauzmann_entropy_thermodynamic_boundary",
            citation_key="kauzmann1948nature",
            model_scope="thermodynamic_transition",
            landing_url="https://doi.org/10.1021/cr60135a002",
            doi="10.1021/cr60135a002",
            archive_name="none",
            archive_md5="none",
            archive_size_bytes=0,
            license_id="article",
            has_public_landing_page=True,
            has_downloadable_archive=False,
            has_schema_or_readme=False,
            has_trajectory_files=False,
            has_precomputed_descriptors=False,
            local_cache_present=False,
            intended_protocols=["thermodynamic_entropy_closure"],
        )

        self.assertEqual(row["accession_stage"], "scope_boundary_accession")
        self.assertEqual(row["primary_blocker"], "renewal_dynamics_not_thermodynamic_theory")
        self.assertEqual(row["scope_boundary"], 1.0)

    def test_sota_archive_preflight_gate_requires_large_download_approval_for_glassbench(self):
        row = sota_archive_preflight_gate(
            preflight_id="glassbench_preflight",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            archive_name="GlassBench.zip",
            archive_size_bytes=6042260027,
            archive_md5="82c83a7146eb749e13417e4350022417",
            readme_name="README",
            readme_size_bytes=2147,
            readme_md5="f1a192f54a2fa7a2b3533af0011b80dc",
            max_automatic_download_bytes=100_000_000,
            full_archive_download_approved=False,
            local_readme_present=False,
            local_archive_present=False,
            required_schema_tokens=["KA", "KA2D", "_trajectories", "_models", "_results"],
            observed_schema_tokens=["KA", "KA2D", "_trajectories", "_models", "_results"],
            required_local_fields=[
                "coordinate_file",
                "time_grid",
                "particle_identity",
                "box_geometry",
                "temperature_or_state_point",
                "species_labels",
                "units_metadata",
            ],
            available_local_fields=[],
        )

        self.assertEqual(row["preflight_stage"], "large_archive_approval_required")
        self.assertEqual(row["ready_for_readme_schema_cache"], 1.0)
        self.assertEqual(row["ready_for_local_reanalysis"], 0.0)
        self.assertEqual(row["large_archive"], 1.0)
        self.assertEqual(row["primary_blocker"], "large_archive_download_approval")
        self.assertGreater(row["archive_size_gb"], 6.0)
        self.assertEqual(row["missing_schema_tokens"], "none")

    def test_sota_archive_preflight_gate_marks_local_archive_ready(self):
        row = sota_archive_preflight_gate(
            preflight_id="synthetic_archive_preflight",
            accession_id="synthetic_local_cache",
            source_id="synthetic_intermediate_scattering_fixture",
            archive_name="synthetic.zip",
            archive_size_bytes=2500000,
            archive_md5="0123456789abcdef0123456789abcdef",
            readme_name="README.md",
            readme_size_bytes=2048,
            readme_md5="abcdef0123456789abcdef0123456789",
            max_automatic_download_bytes=100_000_000,
            full_archive_download_approved=False,
            local_readme_present=True,
            local_archive_present=True,
            required_schema_tokens=["synthetic", "_trajectories"],
            observed_schema_tokens=["synthetic", "_trajectories"],
            required_local_fields=["coordinate_file", "time_grid", "particle_identity"],
            available_local_fields=["coordinate_file", "time_grid", "particle_identity"],
        )

        self.assertEqual(row["preflight_stage"], "local_archive_reanalysis_ready")
        self.assertEqual(row["ready_for_local_reanalysis"], 1.0)
        self.assertEqual(row["full_archive_download_allowed"], 1.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_sota_archive_preflight_gate_blocks_incomplete_readme_schema(self):
        row = sota_archive_preflight_gate(
            preflight_id="missing_schema_preflight",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            archive_name="GlassBench.zip",
            archive_size_bytes=6042260027,
            archive_md5="82c83a7146eb749e13417e4350022417",
            readme_name="README",
            readme_size_bytes=2147,
            readme_md5="f1a192f54a2fa7a2b3533af0011b80dc",
            max_automatic_download_bytes=100_000_000,
            full_archive_download_approved=True,
            local_readme_present=False,
            local_archive_present=False,
            required_schema_tokens=["KA", "KA2D", "_trajectories", "_models", "_results"],
            observed_schema_tokens=["KA", "_models", "_results"],
            required_local_fields=["coordinate_file"],
            available_local_fields=[],
        )

        self.assertEqual(row["preflight_stage"], "readme_schema_incomplete")
        self.assertEqual(row["ready_for_readme_schema_cache"], 0.0)
        self.assertEqual(row["ready_for_local_reanalysis"], 0.0)
        self.assertEqual(row["primary_blocker"], "schema_tokens")
        self.assertIn("_trajectories", row["missing_schema_tokens"])

    def test_sota_readme_digest_gate_verifies_cached_glassbench_readme(self):
        readme_text = (
            "GlassBench\n\n"
            "This dataset contains two different systems, the three-dimensional "
            "Kob-Andersen mixture (KA) and the two-dimensional ternary mixture "
            "(KA2D). For each system, we provide the pure simulation data "
            "(_trajectories), model predictions and derived structural properties "
            "(_models), as well as scripts to evaluate the data and reproduce the "
            "figures in the roadmap (_results).\n\n"
            "The dataset is uploaded with the license \"Creative Commons Attribution "
            "4.0 International\".\n\n"
            "DOI: 10.1063/5.0129791\n"
            "DOI: 10.1103/PhysRevLett.130.238202\n"
        )
        row = sota_readme_digest_gate(
            digest_id="glassbench_readme_digest",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            readme_text=readme_text,
            expected_size_bytes=len(readme_text.encode("utf-8")),
            expected_md5="use-computed",
            required_tokens=["KA", "KA2D", "_trajectories", "_models", "_results"],
            required_citation_dois=["10.1063/5.0129791", "10.1103/PhysRevLett.130.238202"],
            required_license_phrase="Creative Commons Attribution 4.0 International",
            local_cache_path="data/third_party/glassbench/README",
        )

        self.assertEqual(row["digest_stage"], "readme_digest_verified")
        self.assertEqual(row["readme_digest_ready"], 1.0)
        self.assertEqual(row["size_matches_expected"], 1.0)
        self.assertEqual(row["md5_matches_expected"], 1.0)
        self.assertEqual(row["schema_token_coverage"], 1.0)
        self.assertEqual(row["citation_coverage"], 1.0)
        self.assertEqual(row["license_phrase_present"], 1.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_sota_readme_digest_gate_blocks_missing_citation_guidance(self):
        readme_text = (
            "GlassBench KA KA2D _trajectories _models _results "
            "Creative Commons Attribution 4.0 International "
            "DOI: 10.1063/5.0129791"
        )
        row = sota_readme_digest_gate(
            digest_id="missing_citation_digest",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            readme_text=readme_text,
            expected_size_bytes=len(readme_text.encode("utf-8")),
            expected_md5="use-computed",
            required_tokens=["KA", "KA2D", "_trajectories", "_models", "_results"],
            required_citation_dois=["10.1063/5.0129791", "10.1103/PhysRevLett.130.238202"],
            required_license_phrase="Creative Commons Attribution 4.0 International",
            local_cache_path="data/third_party/glassbench/README",
        )

        self.assertEqual(row["digest_stage"], "citation_guidance_incomplete")
        self.assertEqual(row["readme_digest_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "citation_dois")
        self.assertIn("10.1103/PhysRevLett.130.238202", row["missing_citation_dois"])

    def test_sota_local_cache_verification_gate_marks_missing_archive_after_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readme = root / "README"
            archive = root / "GlassBench.zip"
            readme.write_text("GlassBench KA KA2D _trajectories\n", encoding="utf-8")

            row = sota_local_cache_verification_gate(
                cache_id="glassbench_local_cache",
                accession_id="glassbench_zenodo_10118191",
                source_id="glassbench_zenodo_trajectory_release",
                readme_path=readme,
                expected_readme_size_bytes=readme.stat().st_size,
                expected_readme_md5="use-computed",
                archive_path=archive,
                expected_archive_size_bytes=6042260027,
                expected_archive_md5="82c83a7146eb749e13417e4350022417",
            )

        self.assertEqual(row["cache_stage"], "archive_cache_missing")
        self.assertEqual(row["readme_cache_verified"], 1.0)
        self.assertEqual(row["archive_cache_verified"], 0.0)
        self.assertEqual(row["ready_for_local_reanalysis"], 0.0)
        self.assertEqual(row["primary_blocker"], "archive_path")

    def test_sota_local_cache_verification_gate_marks_complete_local_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            readme = root / "README"
            archive = root / "synthetic.zip"
            readme.write_text("README\n", encoding="utf-8")
            archive.write_bytes(b"synthetic archive bytes")

            row = sota_local_cache_verification_gate(
                cache_id="synthetic_local_cache",
                accession_id="synthetic_local_cache",
                source_id="synthetic_fixture",
                readme_path=readme,
                expected_readme_size_bytes=readme.stat().st_size,
                expected_readme_md5="use-computed",
                archive_path=archive,
                expected_archive_size_bytes=archive.stat().st_size,
                expected_archive_md5="use-computed",
            )

        self.assertEqual(row["cache_stage"], "local_archive_cache_verified")
        self.assertEqual(row["local_cache_verified"], 1.0)
        self.assertEqual(row["ready_for_local_reanalysis"], 1.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_sota_zip_structure_gate_blocks_missing_glassbench_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "GlassBench.zip"
            row = sota_zip_structure_gate(
                structure_id="glassbench_zip_structure",
                accession_id="glassbench_zenodo_10118191",
                source_id="glassbench_zenodo_trajectory_release",
                archive_path=archive,
                required_roots=[
                    "KA/_trajectories",
                    "KA/_models",
                    "KA/_results",
                    "KA2D/_trajectories",
                    "KA2D/_models",
                    "KA2D/_results",
                ],
            )

        self.assertEqual(row["zip_structure_stage"], "zip_archive_missing")
        self.assertEqual(row["zip_structure_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "archive_path")
        self.assertIn("KA/_trajectories", row["missing_roots"])

    def test_sota_zip_structure_gate_reads_synthetic_central_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "synthetic_glassbench.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                for name in [
                    "KA/_trajectories/traj_a.csv",
                    "KA/_models/model.json",
                    "KA/_results/figure.csv",
                    "KA2D/_trajectories/traj_a.csv",
                    "KA2D/_models/model.json",
                    "KA2D/_results/figure.csv",
                ]:
                    zf.writestr(name, "fixture\n")

            row = sota_zip_structure_gate(
                structure_id="synthetic_zip_structure",
                accession_id="synthetic_local_cache",
                source_id="synthetic_fixture",
                archive_path=archive,
                required_roots=[
                    "KA/_trajectories",
                    "KA/_models",
                    "KA/_results",
                    "KA2D/_trajectories",
                    "KA2D/_models",
                    "KA2D/_results",
                ],
            )

        self.assertEqual(row["zip_structure_stage"], "zip_structure_ready")
        self.assertEqual(row["zip_structure_ready"], 1.0)
        self.assertEqual(row["root_coverage"], 1.0)
        self.assertEqual(row["entry_count"], 6.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_sota_remote_zip_central_directory_verifies_glassbench_roots_without_download(self):
        manifest = {
            "archive_url": "https://zenodo.org/api/records/10118191/files/GlassBench.zip/content",
            "archive_size_bytes": 6042260027,
            "range_supported": True,
            "zip64": True,
            "tail_probe_bytes": 1048576,
            "central_directory_size_bytes": 7915,
            "central_directory_offset": 6042252014,
            "entry_count": 70,
            "entries": [
                "GlassBench/",
                "GlassBench/README",
                "GlassBench/KA_trajectories/",
                "GlassBench/KA_trajectories/T0.44.xyz",
                "GlassBench/KA_models/model.json",
                "GlassBench/KA_results/FIG2.dat",
                "GlassBench/KA2D_trajectories/T0.30.xyz",
                "GlassBench/KA2D_models/model.json",
                "GlassBench/KA2D_results/FIG3.dat",
            ],
        }
        row = sota_remote_zip_central_directory_gate(
            remote_structure_id="glassbench_remote_zip_central_directory",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            manifest=manifest,
            expected_archive_size_bytes=6042260027,
            required_roots=[
                "GlassBench/KA_trajectories",
                "GlassBench/KA_models",
                "GlassBench/KA_results",
                "GlassBench/KA2D_trajectories",
                "GlassBench/KA2D_models",
                "GlassBench/KA2D_results",
            ],
            full_archive_cached=False,
        )

        self.assertEqual(row["remote_zip_structure_stage"], "remote_zip_structure_verified")
        self.assertEqual(row["remote_zip_structure_ready"], 1.0)
        self.assertEqual(row["zip64"], 1.0)
        self.assertEqual(row["range_supported"], 1.0)
        self.assertEqual(row["root_coverage"], 1.0)
        self.assertEqual(row["entry_count"], 70.0)
        self.assertEqual(row["full_archive_cached"], 0.0)
        self.assertEqual(row["real_reanalysis_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "archive_cache")

    def test_sota_glassbench_payload_index_maps_system_temperatures_without_reanalysis(self):
        manifest = {
            "entries": [
                "GlassBench/KA_results/times_0.44.dat",
                "GlassBench/KA_results/times_0.50.dat",
                "GlassBench/KA_results/times_0.56.dat",
                "GlassBench/KA_results/times_0.64.dat",
                "GlassBench/KA_results/chi4_KA_T0.44_update.dat",
                "GlassBench/KA_trajectories/README",
                "GlassBench/KA_models/T0.44.tar.gz",
                "GlassBench/KA_models/T0.50.tar.gz",
                "GlassBench/KA_models/T0.56.tar.gz",
                "GlassBench/KA_models/T0.64.tar.gz",
                "GlassBench/KA2D_results/times_0.23.dat",
                "GlassBench/KA2D_results/times_0.30.dat",
                "GlassBench/KA2D_results/rhomax_T0.30_MD.dat",
                "GlassBench/KA2D_results/rhomax_T0.30_BB.dat",
                "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "GlassBench/KA2D_models/T0.23.tar.gz",
                "GlassBench/KA2D_models/T0.30.tar.gz",
            ],
        }

        rows = sota_glassbench_payload_index_gate(
            payload_index_id="glassbench_payload_index",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            manifest=manifest,
            systems=["KA", "KA2D"],
            full_archive_cached=False,
        )

        by_system = {row["system_id"]: row for row in rows}
        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["payload_stage"], "remote_payload_index_verified")
        self.assertEqual(ka2d["payload_index_ready"], 1.0)
        self.assertEqual(ka2d["trajectory_payload_count"], 2.0)
        self.assertEqual(ka2d["model_payload_count"], 2.0)
        self.assertEqual(ka2d["result_curve_count"], 4.0)
        self.assertEqual(ka2d["common_temperatures"], "0.23;0.30")
        self.assertEqual(ka2d["real_reanalysis_ready"], 0.0)
        self.assertEqual(ka2d["primary_blocker"], "archive_cache")

        ka = by_system["KA"]
        self.assertEqual(ka["payload_stage"], "remote_payload_missing_trajectory")
        self.assertEqual(ka["payload_index_ready"], 0.0)
        self.assertEqual(ka["trajectory_payload_count"], 0.0)
        self.assertEqual(ka["model_result_index_ready"], 1.0)
        self.assertEqual(ka["common_model_result_temperatures"], "0.44;0.50;0.56;0.64")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_payload_locator_marks_ka2d_remote_archives(self):
        manifest = {
            "archive_url": "https://zenodo.org/api/records/10118191/files/GlassBench.zip/content",
            "range_supported": True,
            "entries": [
                "GlassBench/KA_trajectories/README",
                "GlassBench/KA2D_trajectories/README",
                "GlassBench/KA2D_trajectories/example_T0.23.py",
                "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "GlassBench/KA2D_models/T0.30.tar.gz",
                "GlassBench/KA2D_results/times_0.30.dat",
            ],
        }

        rows = sota_glassbench_trajectory_payload_locator_gate(
            locator_id="glassbench_trajectory_payload_locator",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            manifest=manifest,
            systems=["KA", "KA2D"],
            full_archive_cached=False,
            entry_metadata_ready=False,
        )

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["source_path"], "GlassBench/KA2D_trajectories/T0.30.tar.xz")
        self.assertEqual(ka2d_030["payload_format"], "tar.xz")
        self.assertEqual(ka2d_030["locator_stage"], "remote_trajectory_payload_located")
        self.assertEqual(ka2d_030["remote_payload_located"], 1.0)
        self.assertEqual(ka2d_030["range_supported"], 1.0)
        self.assertEqual(ka2d_030["entry_metadata_ready"], 0.0)
        self.assertEqual(ka2d_030["range_fetch_ready"], 0.0)
        self.assertEqual(ka2d_030["real_reanalysis_ready"], 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "zip_entry_metadata")

        ka = [row for row in rows if row["system_id"] == "KA"][0]
        self.assertEqual(ka["locator_stage"], "remote_trajectory_payload_missing")
        self.assertEqual(ka["source_path"], "none")
        self.assertEqual(ka["temperature"], "none")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_entry_metadata_blocks_large_deflated_members(self):
        locator_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "remote_payload_located": 1.0,
                "range_supported": 1.0,
            },
            {
                "system_id": "KA",
                "temperature": "none",
                "source_path": "none",
                "remote_payload_located": 0.0,
                "range_supported": 1.0,
            },
        ]
        metadata_manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "compression_method": "deflate",
                    "crc32": "5e6a4b14",
                    "compressed_size_bytes": 2_980_137_961,
                    "uncompressed_size_bytes": 2_979_229_176,
                    "local_header_offset": 464_072,
                    "compressed_data_range_start": 464_175,
                    "compressed_data_range_end": 2_980_602_135,
                    "local_header_verified": True,
                }
            ]
        }

        rows = sota_glassbench_trajectory_entry_metadata_gate(
            metadata_id="glassbench_trajectory_entry_metadata",
            accession_id="glassbench_zenodo_10118191",
            locator_rows=locator_rows,
            metadata_manifest=metadata_manifest,
            max_policy_member_bytes=250_000_000,
        )

        by_system = {row["system_id"]: row for row in rows}
        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["metadata_stage"], "trajectory_entry_metadata_ready_payload_size_blocked")
        self.assertEqual(ka2d["source_path"], "GlassBench/KA2D_trajectories/T0.30.tar.xz")
        self.assertEqual(ka2d["compression_method"], "deflate")
        self.assertEqual(ka2d["entry_metadata_ready"], 1.0)
        self.assertEqual(ka2d["local_header_verified"], 1.0)
        self.assertEqual(ka2d["compressed_size_bytes"], 2_980_137_961.0)
        self.assertEqual(ka2d["compressed_data_range_start"], 464_175.0)
        self.assertEqual(ka2d["compressed_data_range_end"], 2_980_602_135.0)
        self.assertEqual(ka2d["full_member_fetch_within_policy"], 0.0)
        self.assertEqual(ka2d["trajectory_extraction_ready"], 0.0)
        self.assertEqual(ka2d["real_reanalysis_ready"], 0.0)
        self.assertEqual(ka2d["primary_blocker"], "member_payload_size_policy")

        ka = by_system["KA"]
        self.assertEqual(ka["metadata_stage"], "trajectory_payload_missing")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_member_stream_probe_verifies_xz_prefix(self):
        metadata_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "entry_metadata_ready": 1.0,
                "compressed_size_bytes": 397_505_592.0,
                "primary_blocker": "member_payload_size_policy",
            },
            {
                "system_id": "KA",
                "temperature": "none",
                "source_path": "none",
                "entry_metadata_ready": 0.0,
                "compressed_size_bytes": 0.0,
                "primary_blocker": "trajectory_payload",
            },
        ]
        probe_manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                    "compressed_probe_range_start": 2_980_602_255,
                    "compressed_probe_range_end": 2_980_667_790,
                    "compressed_probe_bytes": 65_536,
                    "compressed_probe_md5": "ba323cefe12381456e5e8ac6e27a5cd9",
                    "inflated_prefix_bytes": 1024,
                    "inflated_prefix_hex": "fd377a585a000004e6d6b44602002101",
                    "xz_magic_verified": True,
                    "stream_inflate_ready": True,
                }
            ]
        }

        rows = sota_glassbench_trajectory_member_stream_probe_gate(
            probe_id="glassbench_trajectory_member_stream_probe",
            accession_id="glassbench_zenodo_10118191",
            metadata_rows=metadata_rows,
            probe_manifest=probe_manifest,
        )

        by_system = {row["system_id"]: row for row in rows}
        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["probe_stage"], "trajectory_member_prefix_verified_streaming_extraction_blocked")
        self.assertEqual(ka2d["source_path"], "GlassBench/KA2D_trajectories/T0.23.tar.xz")
        self.assertEqual(ka2d["compressed_probe_bytes"], 65_536.0)
        self.assertEqual(ka2d["stream_inflate_ready"], 1.0)
        self.assertEqual(ka2d["xz_magic_verified"], 1.0)
        self.assertEqual(ka2d["member_prefix_verified"], 1.0)
        self.assertEqual(ka2d["trajectory_extraction_ready"], 0.0)
        self.assertEqual(ka2d["real_reanalysis_ready"], 0.0)
        self.assertEqual(ka2d["primary_blocker"], "streaming_member_extraction_policy")

        ka = by_system["KA"]
        self.assertEqual(ka["probe_stage"], "trajectory_entry_metadata_incomplete")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_inner_tar_header_probe_verifies_npz_members(self):
        member_probe_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "member_prefix_verified": 1.0,
                "primary_blocker": "streaming_member_extraction_policy",
            },
            {
                "system_id": "KA",
                "temperature": "none",
                "source_path": "none",
                "member_prefix_verified": 0.0,
                "primary_blocker": "trajectory_payload",
            },
        ]
        tar_probe_manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "compressed_probe_bytes": 4_194_304,
                    "xz_prefix_bytes": 4_193_024,
                    "tar_probe_bytes": 1_048_576,
                    "tar_magic_verified": True,
                    "root_directory": "T0.30/",
                    "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                    "first_npz_size_bytes": 444_786,
                    "npz_member_count_in_probe": 3,
                    "split_labels_in_probe": ["train"],
                }
            ]
        }

        rows = sota_glassbench_trajectory_inner_tar_header_probe_gate(
            tar_probe_id="glassbench_trajectory_inner_tar_header_probe",
            accession_id="glassbench_zenodo_10118191",
            member_probe_rows=member_probe_rows,
            tar_probe_manifest=tar_probe_manifest,
        )

        by_system = {row["system_id"]: row for row in rows}
        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["tar_probe_stage"], "trajectory_inner_tar_layout_verified_extraction_blocked")
        self.assertEqual(ka2d["source_path"], "GlassBench/KA2D_trajectories/T0.30.tar.xz")
        self.assertEqual(ka2d["root_directory"], "T0.30/")
        self.assertEqual(ka2d["first_npz_member"], "T0.30/train/N1290T0.30_3_tc01.npz")
        self.assertEqual(ka2d["split_labels_in_probe"], "train")
        self.assertEqual(ka2d["tar_magic_verified"], 1.0)
        self.assertEqual(ka2d["npz_member_header_verified"], 1.0)
        self.assertEqual(ka2d["trajectory_layout_ready"], 1.0)
        self.assertEqual(ka2d["trajectory_extraction_ready"], 0.0)
        self.assertEqual(ka2d["real_reanalysis_ready"], 0.0)
        self.assertEqual(ka2d["primary_blocker"], "streaming_npz_extraction_policy")

        ka = by_system["KA"]
        self.assertEqual(ka["tar_probe_stage"], "trajectory_member_prefix_incomplete")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_npz_schema_probe_verifies_coordinate_arrays(self):
        tar_probe_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                "trajectory_layout_ready": 1.0,
                "primary_blocker": "streaming_npz_extraction_policy",
            },
            {
                "system_id": "KA",
                "temperature": "none",
                "source_path": "none",
                "first_npz_member": "none",
                "trajectory_layout_ready": 0.0,
                "primary_blocker": "trajectory_payload",
            },
        ]
        schema_manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                    "npz_member_bytes": 444_786,
                    "npz_member_md5": "f51fd76f59b8288405a9e7abb61cdd0a",
                    "npz_magic_verified": True,
                    "arrays": [
                        {"name": "box.npy", "shape": [], "dtype": "float64"},
                        {"name": "types.npy", "shape": [1290], "dtype": "int64"},
                        {"name": "initial_positions.npy", "shape": [1290, 2], "dtype": "float64"},
                        {"name": "positions.npy", "shape": [20, 1290, 2], "dtype": "float64"},
                    ],
                }
            ]
        }

        rows = sota_glassbench_trajectory_npz_schema_probe_gate(
            schema_probe_id="glassbench_trajectory_npz_schema_probe",
            accession_id="glassbench_zenodo_10118191",
            tar_probe_rows=tar_probe_rows,
            schema_probe_manifest=schema_manifest,
            required_arrays=["box.npy", "types.npy", "positions.npy"],
        )

        by_system = {row["system_id"]: row for row in rows}
        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["schema_probe_stage"], "trajectory_npz_coordinate_schema_verified")
        self.assertEqual(ka2d["first_npz_member"], "T0.30/train/N1290T0.30_3_tc01.npz")
        self.assertEqual(ka2d["npz_magic_verified"], 1.0)
        self.assertEqual(ka2d["npz_schema_ready"], 1.0)
        self.assertEqual(ka2d["coordinate_array_ready"], 1.0)
        self.assertEqual(ka2d["particle_count"], 1290.0)
        self.assertEqual(ka2d["frame_count"], 20.0)
        self.assertEqual(ka2d["spatial_dimension"], 2.0)
        self.assertEqual(ka2d["array_names"], "box.npy;types.npy;initial_positions.npy;positions.npy")
        self.assertEqual(ka2d["trajectory_extraction_ready"], 0.0)
        self.assertEqual(ka2d["real_reanalysis_ready"], 0.0)
        self.assertEqual(ka2d["primary_blocker"], "full_npz_ensemble_extraction_policy")

        ka = by_system["KA"]
        self.assertEqual(ka["schema_probe_stage"], "trajectory_inner_tar_layout_incomplete")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_first_npz_observable_smoke_computes_msd_ngp(self):
        schema_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                "npz_schema_ready": 1.0,
                "coordinate_array_ready": 1.0,
                "particle_count": 1290.0,
                "frame_count": 20.0,
                "spatial_dimension": 2.0,
                "primary_blocker": "full_npz_ensemble_extraction_policy",
            },
            {
                "system_id": "KA",
                "temperature": "none",
                "source_path": "none",
                "first_npz_member": "none",
                "npz_schema_ready": 0.0,
                "coordinate_array_ready": 0.0,
                "primary_blocker": "trajectory_payload",
            },
        ]
        observable_manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                    "observable_method": "minimal_image_displacement_from_first_frame",
                    "box_length": 32.8962,
                    "positions_md5": "2fd3bc6e386a636f893ca513234edc72",
                    "frame_count": 20,
                    "particle_count": 1290,
                    "spatial_dimension": 2,
                    "final_frame_index": 19,
                    "final_msd": 0.005414723094117662,
                    "final_ngp_2d": 0.05197783140666812,
                    "peak_ngp_frame_index": 11,
                    "peak_ngp_2d": 0.17874626903381952,
                    "msd_at_peak_ngp": 0.005117245595512797,
                    "max_abs_min_image_displacement": 0.26752539804661485,
                }
            ]
        }

        rows = sota_glassbench_trajectory_first_npz_observable_smoke_gate(
            smoke_id="glassbench_trajectory_first_npz_observable_smoke",
            accession_id="glassbench_zenodo_10118191",
            schema_probe_rows=schema_rows,
            observable_manifest=observable_manifest,
            required_method="minimal_image_displacement_from_first_frame",
        )

        by_system = {row["system_id"]: row for row in rows}
        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["smoke_stage"], "first_npz_msd_ngp_smoke_ready_reanalysis_blocked")
        self.assertEqual(ka2d["observable_method"], "minimal_image_displacement_from_first_frame")
        self.assertEqual(ka2d["observable_smoke_ready"], 1.0)
        self.assertAlmostEqual(ka2d["final_msd"], 0.005414723094117662)
        self.assertAlmostEqual(ka2d["final_ngp_2d"], 0.05197783140666812)
        self.assertEqual(ka2d["peak_ngp_frame_index"], 11.0)
        self.assertAlmostEqual(ka2d["peak_ngp_2d"], 0.17874626903381952)
        self.assertEqual(ka2d["trajectory_extraction_ready"], 0.0)
        self.assertEqual(ka2d["real_reanalysis_ready"], 0.0)
        self.assertEqual(ka2d["primary_blocker"], "single_npz_no_time_or_uncertainty")

        ka = by_system["KA"]
        self.assertEqual(ka["smoke_stage"], "trajectory_npz_schema_incomplete")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_first_npz_observable_curve_exports_frame_rows(self):
        smoke_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                "observable_method": "minimal_image_displacement_from_first_frame",
                "observable_smoke_ready": 1.0,
                "primary_blocker": "single_npz_no_time_or_uncertainty",
            },
            {
                "system_id": "KA",
                "temperature": "none",
                "source_path": "none",
                "first_npz_member": "none",
                "observable_method": "none",
                "observable_smoke_ready": 0.0,
                "primary_blocker": "trajectory_payload",
            },
        ]
        curve_manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                    "observable_method": "minimal_image_displacement_from_first_frame",
                    "frame_indices": [0, 1, 11, 19],
                    "msd": [0.0, 0.005484922658, 0.005117245596, 0.005414723094],
                    "ngp_2d": [0.0, 0.048418253757, 0.178746269034, 0.051977831407],
                    "wave_numbers": [0.7, 1.1, 1.6],
                    "self_intermediate_scattering_by_k": [
                        "1;1;1",
                        "0.999;0.998;0.997",
                        "0.995;0.991;0.984",
                        "0.994;0.989;0.982",
                    ],
                    "overlap_radius": 0.3,
                    "chi4_overlap": [0.0, 0.01, 0.18, 0.05],
                }
            ]
        }

        rows = sota_glassbench_trajectory_first_npz_observable_curve_gate(
            curve_id="glassbench_trajectory_first_npz_observable_curve",
            accession_id="glassbench_zenodo_10118191",
            smoke_rows=smoke_rows,
            curve_manifest=curve_manifest,
            required_method="minimal_image_displacement_from_first_frame",
        )

        by_key = {(row["system_id"], row["temperature"], row["frame_index"]): row for row in rows}
        peak = by_key[("KA2D", "0.30", 11.0)]
        self.assertEqual(peak["curve_stage"], "first_npz_observable_curve_ready_reanalysis_blocked")
        self.assertEqual(peak["observable_method"], "minimal_image_displacement_from_first_frame")
        self.assertEqual(peak["observable_curve_ready"], 1.0)
        self.assertAlmostEqual(peak["msd"], 0.005117245596)
        self.assertAlmostEqual(peak["ngp_2d"], 0.178746269034)
        self.assertEqual(peak["wave_numbers"], "0.7;1.1;1.6")
        self.assertEqual(peak["self_intermediate_scattering_by_k"], "0.995;0.991;0.984")
        self.assertAlmostEqual(peak["self_intermediate_scattering"], 0.995)
        self.assertAlmostEqual(peak["overlap_radius"], 0.3)
        self.assertAlmostEqual(peak["chi4_overlap"], 0.18)
        self.assertEqual(peak["trajectory_extraction_ready"], 0.0)
        self.assertEqual(peak["real_reanalysis_ready"], 0.0)
        self.assertEqual(peak["primary_blocker"], "single_npz_frame_index_curve")

        final = by_key[("KA2D", "0.30", 19.0)]
        self.assertAlmostEqual(final["msd"], 0.005414723094)
        self.assertAlmostEqual(final["ngp_2d"], 0.051977831407)

        blocked = by_key[("KA", "none", -1.0)]
        self.assertEqual(blocked["curve_stage"], "first_npz_observable_smoke_incomplete")
        self.assertEqual(blocked["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_first_npz_inversion_readiness_blocks_single_member_frame_curves(self):
        curve_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                "observable_curve_ready": 1.0,
                "frame_index": float(frame),
                "msd": 0.005 + 0.0001 * frame,
                "ngp_2d": 0.05 + 0.001 * frame,
            }
            for frame in range(20)
        ]
        curve_rows.append(
            {
                "system_id": "KA",
                "temperature": "none",
                "source_path": "none",
                "first_npz_member": "none",
                "observable_curve_ready": 0.0,
                "frame_index": -1.0,
                "msd": 0.0,
                "ngp_2d": 0.0,
                "primary_blocker": "trajectory_payload",
            }
        )

        rows = sota_glassbench_trajectory_first_npz_inversion_readiness_gate(
            benchmark_id="glassbench_first_npz_sota_inversion_readiness",
            accession_id="glassbench_zenodo_10118191",
            curve_rows=curve_rows,
            required_observables=[
                "lag_time",
                "msd",
                "ngp_2d",
                "self_intermediate_scattering_by_k",
                "chi4_overlap",
            ],
            required_uncertainty_columns=[
                "sigma_msd",
                "sigma_ngp_2d",
                "sigma_self_intermediate_scattering_by_k",
                "sigma_chi4_overlap",
            ],
            min_member_count=4,
            min_frame_count=20,
            has_physical_time=False,
        )

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d["readiness_stage"], "frame_index_curve_only")
        self.assertEqual(ka2d["primary_blocker"], "physical_time_semantics")
        self.assertEqual(float(ka2d["frame_count"]), 20.0)
        self.assertEqual(float(ka2d["member_count"]), 1.0)
        self.assertEqual(float(ka2d["structural_curve_ready"]), 0.0)
        self.assertEqual(float(ka2d["sota_inversion_ready"]), 0.0)
        self.assertIn("lag_time", ka2d["missing_observables"])
        self.assertIn("self_intermediate_scattering_by_k", ka2d["missing_observables"])
        self.assertIn("sigma_msd", ka2d["missing_uncertainty_columns"])
        self.assertEqual(ka2d["next_required_action"], "attach_physical_lag_time_and_units")

        ka = by_key[("KA", "none")]
        self.assertEqual(ka["readiness_stage"], "upstream_curve_incomplete")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_first_npz_inversion_readiness_promotes_uncertainty_weighted_ensemble(self):
        curve_rows = []
        for member in range(4):
            for frame in range(20):
                curve_rows.append(
                    {
                        "system_id": "KA2D",
                        "temperature": "0.30",
                        "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                        "first_npz_member": f"T0.30/train/member_{member}.npz",
                        "observable_curve_ready": 1.0,
                        "frame_index": float(frame),
                        "lag_time": 0.1 * frame,
                        "msd": 0.005 + 0.0001 * frame,
                        "ngp_2d": 0.05 + 0.001 * frame,
                        "self_intermediate_scattering_by_k": "1.1:0.4",
                        "chi4_overlap": 0.2 + 0.01 * frame,
                        "sigma_msd": 1e-4,
                        "sigma_ngp_2d": 1e-3,
                        "sigma_self_intermediate_scattering_by_k": 2e-3,
                        "sigma_chi4_overlap": 3e-3,
                    }
                )

        rows = sota_glassbench_trajectory_first_npz_inversion_readiness_gate(
            benchmark_id="glassbench_first_npz_sota_inversion_readiness",
            accession_id="glassbench_zenodo_10118191",
            curve_rows=curve_rows,
            required_observables=[
                "lag_time",
                "msd",
                "ngp_2d",
                "self_intermediate_scattering_by_k",
                "chi4_overlap",
            ],
            required_uncertainty_columns=[
                "sigma_msd",
                "sigma_ngp_2d",
                "sigma_self_intermediate_scattering_by_k",
                "sigma_chi4_overlap",
            ],
            min_member_count=4,
            min_frame_count=20,
            has_physical_time=True,
        )

        ready = rows[0]
        self.assertEqual(ready["readiness_stage"], "uncertainty_weighted_sota_inversion_ready")
        self.assertEqual(ready["primary_blocker"], "none")
        self.assertEqual(float(ready["structural_curve_ready"]), 1.0)
        self.assertEqual(float(ready["ensemble_ready"]), 1.0)
        self.assertEqual(float(ready["uncertainty_ready"]), 1.0)
        self.assertEqual(float(ready["sota_inversion_ready"]), 1.0)

    def test_sota_glassbench_observable_coverage_audit_blocks_frame_index_msd_ngp_only(self):
        curve_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                "first_npz_member": "T0.23/test/N1290T0.23_202_tc05.npz",
                "observable_curve_ready": 1.0,
                "frame_index": float(frame),
                "msd": 0.004 + 0.0001 * frame,
                "ngp_2d": 0.08,
            }
            for frame in range(20)
        ]
        readiness_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "available_observables": "msd;ngp_2d",
                "missing_observables": "lag_time;self_intermediate_scattering_by_k;chi4_overlap",
                "structural_curve_ready": 0.0,
                "sota_inversion_ready": 0.0,
            }
        ]
        semantics_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.23",
                "curve_role": "rhomax_md",
                "candidate_observable": "overlap_density_proxy",
                "proxy_observable_ready": 0.0,
                "real_inversion_ready": 0.0,
                "primary_blocker": "structural_adapter",
            }
        ]

        rows = sota_glassbench_observable_coverage_audit_gate(
            audit_id="glassbench_observable_coverage_audit",
            accession_id="glassbench_zenodo_10118191",
            curve_rows=curve_rows,
            inversion_readiness_rows=readiness_rows,
            observable_semantics_rows=semantics_rows,
            required_observables=[
                "lag_time",
                "msd",
                "ngp_2d",
                "self_intermediate_scattering_by_k",
                "chi4_overlap",
            ],
        )

        row = rows[0]
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

    def test_sota_glassbench_observable_coverage_audit_accepts_complete_observable_set(self):
        curve_rows = []
        for frame in range(20):
            curve_rows.append(
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "first_npz_member": "T0.30/train/member_1.npz",
                    "observable_curve_ready": 1.0,
                    "frame_index": float(frame),
                    "lag_time": 0.1 * frame,
                    "msd": 0.005 + 0.0001 * frame,
                    "ngp_2d": 0.05,
                    "self_intermediate_scattering_by_k": "0.9;0.8",
                    "chi4_overlap": 0.2,
                }
            )
        readiness_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "available_observables": "lag_time;msd;ngp_2d;self_intermediate_scattering_by_k;chi4_overlap",
                "missing_observables": "none",
                "structural_curve_ready": 1.0,
                "sota_inversion_ready": 1.0,
            }
        ]
        semantics_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "curve_role": "direct_fs_chi4",
                "candidate_observable": "direct_trajectory_observables",
                "proxy_observable_ready": 1.0,
                "real_inversion_ready": 1.0,
                "primary_blocker": "none",
            }
        ]

        rows = sota_glassbench_observable_coverage_audit_gate(
            audit_id="glassbench_observable_coverage_audit",
            accession_id="glassbench_zenodo_10118191",
            curve_rows=curve_rows,
            inversion_readiness_rows=readiness_rows,
            observable_semantics_rows=semantics_rows,
            required_observables=[
                "lag_time",
                "msd",
                "ngp_2d",
                "self_intermediate_scattering_by_k",
                "chi4_overlap",
            ],
        )

        row = rows[0]
        self.assertEqual(row["observable_audit_stage"], "real_inversion_observable_set_ready")
        self.assertEqual(row["missing_observables"], "none")
        self.assertEqual(row["primary_blocker"], "none")
        self.assertEqual(row["next_required_actions"], "attach_uncertainties_and_run_real_inversion")
        self.assertEqual(float(row["remote_result_semantics_ready"]), 1.0)
        self.assertEqual(float(row["observable_coverage_ready"]), 1.0)
        self.assertEqual(float(row["publishable_real_inversion_observable_set_ready"]), 1.0)
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_first_npz_structural_observable_plan_marks_positions_bytes_blocker(self):
        schema_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                "npz_schema_ready": 1.0,
                "coordinate_array_ready": 1.0,
                "trajectory_extraction_ready": 0.0,
                "npz_member_bytes": 444786.0,
                "array_names": "box.npy;types.npy;initial_positions.npy;positions.npy",
                "array_shapes": "box.npy:scalar;types.npy:1290;positions.npy:20x1290x2",
                "frame_count": 20.0,
                "particle_count": 1290.0,
                "spatial_dimension": 2.0,
            }
        ]
        coverage_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "available_trajectory_observables": "frame_index;msd;ngp_2d",
                "missing_observables": "lag_time;self_intermediate_scattering_by_k;chi4_overlap",
                "observable_coverage_ready": 0.0,
            }
        ]

        rows = sota_glassbench_first_npz_structural_observable_plan_gate(
            plan_id="glassbench_first_npz_structural_observable_plan",
            accession_id="glassbench_zenodo_10118191",
            schema_probe_rows=schema_rows,
            observable_coverage_rows=coverage_rows,
            implemented_observables=[
                "msd",
                "ngp_2d",
                "self_intermediate_scattering_by_k",
                "chi4_overlap",
            ],
        )

        row = rows[0]
        self.assertEqual(row["compute_plan_stage"], "coordinate_schema_ready_positions_bytes_missing")
        self.assertEqual(float(row["coordinate_schema_ready"]), 1.0)
        self.assertEqual(float(row["raw_coordinate_bytes_cached"]), 0.0)
        self.assertEqual(float(row["computable_after_npz_extraction"]), 1.0)
        self.assertEqual(float(row["immediately_computable_from_current_cache"]), 0.0)
        self.assertIn("self_intermediate_scattering_by_k", row["implemented_observable_protocol"])
        self.assertIn("chi4_overlap", row["implemented_observable_protocol"])
        self.assertEqual(row["remaining_missing_after_structural_compute"], "lag_time")
        self.assertEqual(row["primary_blocker"], "raw_coordinate_bytes")
        self.assertIn("extract_first_npz_positions_box_types", row["next_required_actions"])
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_first_npz_structural_observable_plan_accepts_cached_positions(self):
        rows = sota_glassbench_first_npz_structural_observable_plan_gate(
            plan_id="glassbench_first_npz_structural_observable_plan",
            accession_id="glassbench_zenodo_10118191",
            schema_probe_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "first_npz_member": "T0.30/train/member_1.npz",
                    "npz_schema_ready": 1.0,
                    "coordinate_array_ready": 1.0,
                    "trajectory_extraction_ready": 1.0,
                    "npz_member_bytes": 444786.0,
                    "array_names": "box.npy;types.npy;positions.npy",
                    "array_shapes": "box.npy:scalar;types.npy:1290;positions.npy:20x1290x2",
                    "frame_count": 20.0,
                    "particle_count": 1290.0,
                    "spatial_dimension": 2.0,
                }
            ],
            observable_coverage_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "available_trajectory_observables": "frame_index;msd;ngp_2d",
                    "missing_observables": "lag_time;self_intermediate_scattering_by_k;chi4_overlap",
                    "observable_coverage_ready": 0.0,
                }
            ],
            implemented_observables=[
                "msd",
                "ngp_2d",
                "self_intermediate_scattering_by_k",
                "chi4_overlap",
            ],
        )

        row = rows[0]
        self.assertEqual(row["compute_plan_stage"], "structural_observable_compute_ready")
        self.assertEqual(float(row["raw_coordinate_bytes_cached"]), 1.0)
        self.assertEqual(float(row["immediately_computable_from_current_cache"]), 1.0)
        self.assertEqual(row["remaining_missing_after_structural_compute"], "lag_time")
        self.assertEqual(row["primary_blocker"], "physical_time_semantics")
        self.assertEqual(row["next_required_actions"], "run_trajectory_observable_protocol_on_cached_npz")

    def test_sota_glassbench_first_npz_structural_observable_plan_marks_cached_structural_observables(self):
        rows = sota_glassbench_first_npz_structural_observable_plan_gate(
            plan_id="glassbench_first_npz_structural_observable_plan",
            accession_id="glassbench_zenodo_10118191",
            schema_probe_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "first_npz_member": "T0.30/train/member_1.npz",
                    "npz_schema_ready": 1.0,
                    "coordinate_array_ready": 1.0,
                    "trajectory_extraction_ready": 0.0,
                    "npz_member_bytes": 444786.0,
                    "array_names": "box.npy;types.npy;positions.npy",
                    "array_shapes": "box.npy:scalar;types.npy:1290;positions.npy:20x1290x2",
                    "frame_count": 20.0,
                    "particle_count": 1290.0,
                    "spatial_dimension": 2.0,
                }
            ],
            observable_coverage_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "available_trajectory_observables": "frame_index;msd;ngp_2d;self_intermediate_scattering_by_k;chi4_overlap",
                    "missing_observables": "lag_time",
                    "observable_coverage_ready": 0.0,
                }
            ],
            implemented_observables=[
                "msd",
                "ngp_2d",
                "self_intermediate_scattering_by_k",
                "chi4_overlap",
            ],
        )

        row = rows[0]
        self.assertEqual(row["compute_plan_stage"], "structural_observables_cached_raw_coordinates_not_retained")
        self.assertEqual(float(row["structural_observables_cached"]), 1.0)
        self.assertEqual(float(row["raw_coordinate_bytes_cached"]), 0.0)
        self.assertEqual(row["remaining_missing_after_structural_compute"], "lag_time")
        self.assertEqual(row["primary_blocker"], "physical_time_semantics")
        self.assertEqual(row["next_required_actions"], "attach_physical_lag_time_and_units")

    def test_sota_glassbench_short_window_trend_canary_detects_real_cooling_slowdown(self):
        curve_rows = []
        for temp, msd_values, ngp_values in [
            ("0.23", [0.0, 0.004, 0.0041, 0.0040], [0.0, 0.05, 0.11, 0.04]),
            ("0.30", [0.0, 0.005, 0.0054, 0.0053], [0.0, 0.04, 0.12, 0.03]),
        ]:
            for frame, (msd, ngp) in enumerate(zip(msd_values, ngp_values)):
                curve_rows.append(
                    {
                        "curve_id": f"curve_ka2d_t{temp}_frame_{frame}",
                        "accession_id": "glassbench_zenodo_10118191",
                        "system_id": "KA2D",
                        "temperature": temp,
                        "source_path": f"GlassBench/KA2D_trajectories/T{temp}.tar.xz",
                        "first_npz_member": f"T{temp}/member.npz",
                        "observable_method": "minimal_image_displacement_from_first_frame",
                        "frame_index": float(frame),
                        "msd": msd,
                        "ngp_2d": ngp,
                        "observable_curve_ready": 1.0,
                        "trajectory_extraction_ready": 0.0,
                        "real_reanalysis_ready": 0.0,
                        "primary_blocker": "single_npz_frame_index_curve",
                        "curve_stage": "first_npz_observable_curve_ready_reanalysis_blocked",
                    }
                )

        rows = sota_glassbench_short_window_trend_canary_gate(
            canary_id="glassbench_short_window_trend_canary",
            accession_id="glassbench_zenodo_10118191",
            curve_rows=curve_rows,
            cold_temperature="0.23",
            hot_temperature="0.30",
            min_common_frame_count=4,
            min_msd_slowdown_ratio=1.1,
            min_peak_ngp=0.05,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["system_id"], "KA2D")
        self.assertGreater(float(row["hot_to_cold_final_msd_ratio"]), 1.1)
        self.assertEqual(float(row["short_window_msd_slowdown_pass"]), 1.0)
        self.assertEqual(float(row["positive_ngp_canary_pass"]), 1.0)
        self.assertEqual(float(row["short_window_real_data_canary_ready"]), 1.0)
        self.assertEqual(float(row["sota_inversion_ready"]), 0.0)
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(row["canary_stage"], "short_window_real_data_canary_ready_inversion_blocked")

    def test_sota_glassbench_short_window_trend_canary_rejects_inverted_msd_trend(self):
        curve_rows = []
        for temp, final_msd in [("0.23", 0.006), ("0.30", 0.004)]:
            for frame in range(4):
                curve_rows.append(
                    {
                        "system_id": "KA2D",
                        "temperature": temp,
                        "frame_index": float(frame),
                        "msd": final_msd * frame / 3.0,
                        "ngp_2d": 0.08 if frame == 2 else 0.02,
                        "observable_curve_ready": 1.0,
                    }
                )

        rows = sota_glassbench_short_window_trend_canary_gate(
            canary_id="glassbench_short_window_trend_canary",
            accession_id="glassbench_zenodo_10118191",
            curve_rows=curve_rows,
            cold_temperature="0.23",
            hot_temperature="0.30",
            min_common_frame_count=4,
            min_msd_slowdown_ratio=1.1,
            min_peak_ngp=0.05,
        )

        self.assertEqual(float(rows[0]["short_window_real_data_canary_ready"]), 0.0)
        self.assertEqual(rows[0]["primary_blocker"], "short_window_msd_slowdown")
        self.assertEqual(rows[0]["canary_stage"], "short_window_trend_canary_failed")

    def test_sota_glassbench_trajectory_timebase_bridge_blocks_mismatched_result_grid(self):
        curve_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "frame_index": float(frame),
                "msd": 0.005 + frame * 1e-4,
                "ngp_2d": 0.05,
                "observable_curve_ready": 1.0,
            }
            for frame in range(20)
        ]
        payload_adapter_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "curve_role": "rhomax_md",
                "time_grid_path": "GlassBench/KA2D_results/times_0.30.dat",
                "time_point_count": 6.0,
                "structural_adapter_ready": 1.0,
            }
        ]

        rows = sota_glassbench_trajectory_timebase_bridge_gate(
            bridge_id="glassbench_trajectory_timebase_bridge",
            accession_id="glassbench_zenodo_10118191",
            curve_rows=curve_rows,
            payload_adapter_rows=payload_adapter_rows,
            require_explicit_frame_time_mapping=True,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["system_id"], "KA2D")
        self.assertEqual(row["temperature"], "0.30")
        self.assertEqual(float(row["frame_count"]), 20.0)
        self.assertEqual(float(row["time_point_count"]), 6.0)
        self.assertEqual(float(row["frame_time_point_count_match"]), 0.0)
        self.assertEqual(float(row["trajectory_timebase_ready"]), 0.0)
        self.assertEqual(float(row["sota_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "frame_time_point_count")
        self.assertEqual(row["timebase_stage"], "trajectory_result_timebase_length_mismatch")

    def test_sota_glassbench_trajectory_timebase_bridge_accepts_explicit_matching_grid(self):
        curve_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "frame_index": float(frame),
                "msd": 0.005 + frame * 1e-4,
                "ngp_2d": 0.05,
                "observable_curve_ready": 1.0,
            }
            for frame in range(4)
        ]
        payload_adapter_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "curve_role": "rhomax_md",
                "time_grid_path": "GlassBench/KA2D_results/times_0.30.dat",
                "time_point_count": 4.0,
                "structural_adapter_ready": 1.0,
            }
        ]

        rows = sota_glassbench_trajectory_timebase_bridge_gate(
            bridge_id="glassbench_trajectory_timebase_bridge",
            accession_id="glassbench_zenodo_10118191",
            curve_rows=curve_rows,
            payload_adapter_rows=payload_adapter_rows,
            require_explicit_frame_time_mapping=True,
            explicit_frame_time_mappings={("KA2D", "0.30"): True},
        )

        row = rows[0]
        self.assertEqual(float(row["time_grid_available"]), 1.0)
        self.assertEqual(float(row["frame_time_point_count_match"]), 1.0)
        self.assertEqual(float(row["explicit_frame_time_mapping"]), 1.0)
        self.assertEqual(float(row["trajectory_timebase_ready"]), 1.0)
        self.assertEqual(float(row["sota_inversion_ready"]), 0.0)
        self.assertEqual(row["timebase_stage"], "trajectory_timebase_ready_observable_inversion_blocked")

    def test_sota_glassbench_real_inversion_gap_ledger_keeps_current_claim_short_window_only(self):
        rows = sota_glassbench_real_inversion_gap_ledger_gate(
            ledger_id="glassbench_real_inversion_gap_ledger",
            accession_id="glassbench_zenodo_10118191",
            short_window_rows=[
                {
                    "system_id": "KA2D",
                    "cold_temperature": "0.23",
                    "hot_temperature": "0.30",
                    "short_window_real_data_canary_ready": 1.0,
                    "hot_to_cold_final_msd_ratio": 1.36,
                }
            ],
            timebase_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "trajectory_timebase_ready": 0.0,
                    "primary_blocker": "frame_time_point_count",
                    "next_required_action": "derive_or_fetch_trajectory_frame_time_mapping",
                },
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "trajectory_timebase_ready": 0.0,
                    "primary_blocker": "frame_time_point_count",
                    "next_required_action": "derive_or_fetch_trajectory_frame_time_mapping",
                },
            ],
            ensemble_horizon_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "prefix_member_horizon_ready": 0.0,
                    "member_count_gap_to_threshold": 1.0,
                    "next_required_action": "extend_tar_probe_or_index_full_member_list",
                },
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "prefix_member_horizon_ready": 0.0,
                    "member_count_gap_to_threshold": 1.0,
                    "next_required_action": "extend_tar_probe_or_index_full_member_list",
                },
            ],
            inversion_readiness_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "structural_curve_ready": 0.0,
                    "uncertainty_ready": 0.0,
                    "sota_inversion_ready": 0.0,
                    "missing_observables": "lag_time;self_intermediate_scattering_by_k;chi4_overlap",
                    "missing_uncertainty_columns": "sigma_msd;sigma_ngp_2d",
                },
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "structural_curve_ready": 0.0,
                    "uncertainty_ready": 0.0,
                    "sota_inversion_ready": 0.0,
                    "missing_observables": "lag_time;self_intermediate_scattering_by_k;chi4_overlap",
                    "missing_uncertainty_columns": "sigma_msd;sigma_ngp_2d",
                },
            ],
            observable_semantics_rows=[],
        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(float(row["short_window_claim_ready"]), 1.0)
            self.assertEqual(float(row["trajectory_timebase_ready"]), 0.0)
            self.assertEqual(float(row["quantitative_real_inversion_ready"]), 0.0)
            self.assertEqual(row["allowed_claim_level"], "short_window_coordinate_trend_only")
            self.assertEqual(row["primary_blocker"], "frame_time_point_count")
            self.assertEqual(row["ledger_stage"], "real_data_canary_timebase_blocked")

    def test_sota_glassbench_real_inversion_gap_ledger_promotes_only_all_ready_rows(self):
        rows = sota_glassbench_real_inversion_gap_ledger_gate(
            ledger_id="glassbench_real_inversion_gap_ledger",
            accession_id="glassbench_zenodo_10118191",
            short_window_rows=[
                {
                    "system_id": "KA2D",
                    "cold_temperature": "0.23",
                    "hot_temperature": "0.30",
                    "short_window_real_data_canary_ready": 1.0,
                    "hot_to_cold_final_msd_ratio": 1.36,
                }
            ],
            timebase_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "trajectory_timebase_ready": 1.0,
                    "primary_blocker": "observable_set_and_uncertainty",
                    "next_required_action": "compute_fs_chi4_and_uncertainties_on_timebase",
                }
            ],
            ensemble_horizon_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "prefix_member_horizon_ready": 1.0,
                    "member_count_gap_to_threshold": 0.0,
                    "next_required_action": "extract_visible_npz_members_and_compute_uncertainties",
                }
            ],
            inversion_readiness_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "structural_curve_ready": 1.0,
                    "uncertainty_ready": 1.0,
                    "sota_inversion_ready": 1.0,
                    "missing_observables": "none",
                    "missing_uncertainty_columns": "none",
                }
            ],
            observable_semantics_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "real_inversion_ready": 1.0,
                }
            ],
        )

        row = rows[0]
        self.assertEqual(float(row["quantitative_real_inversion_ready"]), 1.0)
        self.assertEqual(row["allowed_claim_level"], "uncertainty_weighted_real_trajectory_inversion")
        self.assertEqual(row["primary_blocker"], "none")
        self.assertEqual(row["ledger_stage"], "real_data_quantitative_inversion_ready")

    def test_sota_glassbench_real_inversion_unlock_protocol_lists_minimum_missing_payload(self):
        rows = sota_glassbench_real_inversion_unlock_protocol_gate(
            protocol_id="glassbench_real_inversion_unlock_protocol",
            accession_id="glassbench_zenodo_10118191",
            ledger_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "short_window_claim_ready": 1.0,
                    "trajectory_timebase_ready": 0.0,
                    "ensemble_horizon_ready": 0.0,
                    "structural_observable_ready": 0.0,
                    "uncertainty_ready": 0.0,
                    "observable_semantics_ready": 0.0,
                    "quantitative_real_inversion_ready": 0.0,
                    "allowed_claim_level": "short_window_coordinate_trend_only",
                    "primary_blocker": "frame_time_point_count",
                }
            ],
            timebase_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "frame_count": 20.0,
                    "time_point_count": 8.0,
                    "explicit_frame_time_mapping_required": 1.0,
                    "explicit_frame_time_mapping": 0.0,
                    "trajectory_timebase_ready": 0.0,
                    "primary_blocker": "frame_time_point_count",
                }
            ],
            ensemble_horizon_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "npz_member_count_in_probe": 3.0,
                    "min_member_count": 4.0,
                    "member_count_gap_to_threshold": 1.0,
                    "prefix_member_horizon_ready": 0.0,
                }
            ],
            inversion_readiness_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "observed_member_count": 1.0,
                    "min_member_count": 4.0,
                    "frame_count": 20.0,
                    "min_frame_count": 20.0,
                    "missing_observables": "lag_time;self_intermediate_scattering_by_k;chi4_overlap",
                    "missing_uncertainty_columns": "sigma_msd;sigma_ngp_2d;sigma_self_intermediate_scattering_by_k;sigma_chi4_overlap",
                    "sota_inversion_ready": 0.0,
                }
            ],
        )

        row = rows[0]
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
        self.assertEqual(float(row["time_point_count"]), 8.0)
        self.assertEqual(row["missing_observables"], "lag_time;self_intermediate_scattering_by_k;chi4_overlap")
        self.assertIn("sigma_chi4_overlap", row["missing_uncertainty_columns"])
        self.assertEqual(
            row["minimum_required_payload"],
            "frame_time_mapping;one_more_independent_npz_member;lag_time;self_intermediate_scattering_by_k;chi4_overlap;sigma_msd;sigma_ngp_2d;sigma_self_intermediate_scattering_by_k;sigma_chi4_overlap",
        )
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_frame_time_mapping_audit_rejects_ambiguous_subsampling(self):
        rows = sota_glassbench_frame_time_mapping_audit_gate(
            audit_id="glassbench_frame_time_mapping_audit",
            accession_id="glassbench_zenodo_10118191",
            timebase_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "frame_count": 20.0,
                    "time_point_count": 8.0,
                    "time_grid_available": 1.0,
                    "time_grid_path": "GlassBench/KA2D_results/times_0.23.dat",
                    "explicit_frame_time_mapping_required": 1.0,
                    "explicit_frame_time_mapping": 0.0,
                    "trajectory_timebase_ready": 0.0,
                },
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "frame_count": 20.0,
                    "time_point_count": 6.0,
                    "time_grid_available": 1.0,
                    "time_grid_path": "GlassBench/KA2D_results/times_0.30.dat",
                    "explicit_frame_time_mapping_required": 1.0,
                    "explicit_frame_time_mapping": 0.0,
                    "trajectory_timebase_ready": 0.0,
                },
            ],
        )

        by_temperature = {row["temperature"]: row for row in rows}
        cold = by_temperature["0.23"]
        hot = by_temperature["0.30"]
        self.assertEqual(float(cold["exact_count_match"]), 0.0)
        self.assertEqual(float(hot["exact_count_match"]), 0.0)
        self.assertEqual(float(cold["integer_stride_subsample_candidate"]), 0.0)
        self.assertEqual(float(hot["integer_stride_subsample_candidate"]), 0.0)
        self.assertEqual(float(cold["endpoint_interpolation_candidate"]), 1.0)
        self.assertEqual(float(hot["endpoint_interpolation_candidate"]), 1.0)
        for row in rows:
            self.assertEqual(float(row["publishable_frame_time_mapping_ready"]), 0.0)
            self.assertEqual(row["accepted_mapping_class"], "none")
            self.assertEqual(row["provisional_mapping_class"], "endpoint_interpolation_requires_metadata")
            self.assertIn("dump_interval", row["minimum_required_metadata"])
            self.assertIn("trajectory_frame_origin", row["minimum_required_metadata"])
            self.assertEqual(row["mapping_audit_stage"], "ambiguous_frame_time_mapping")
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_frame_time_mapping_audit_accepts_explicit_count_match(self):
        rows = sota_glassbench_frame_time_mapping_audit_gate(
            audit_id="glassbench_frame_time_mapping_audit",
            accession_id="glassbench_zenodo_10118191",
            timebase_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "frame_count": 6.0,
                    "time_point_count": 6.0,
                    "time_grid_available": 1.0,
                    "time_grid_path": "GlassBench/KA2D_results/times_0.30.dat",
                    "explicit_frame_time_mapping_required": 1.0,
                    "explicit_frame_time_mapping": 1.0,
                    "trajectory_timebase_ready": 1.0,
                }
            ],
        )

        row = rows[0]
        self.assertEqual(float(row["exact_count_match"]), 1.0)
        self.assertEqual(float(row["publishable_frame_time_mapping_ready"]), 1.0)
        self.assertEqual(row["accepted_mapping_class"], "explicit_count_matched_frame_time_mapping")
        self.assertEqual(row["minimum_required_metadata"], "none")
        self.assertEqual(row["mapping_audit_stage"], "frame_time_mapping_ready")

    def test_sota_glassbench_trajectory_npz_ensemble_horizon_counts_prefix_members_without_claiming_extraction(self):
        tar_probe_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "trajectory_layout_ready": 1.0,
                "npz_member_count_in_probe": 3.0,
                "split_labels_in_probe": "train",
                "tar_probe_bytes": 1048576.0,
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
            },
            {
                "system_id": "KA",
                "temperature": "none",
                "source_path": "none",
                "trajectory_layout_ready": 0.0,
                "npz_member_count_in_probe": 0.0,
                "split_labels_in_probe": "none",
                "tar_probe_bytes": 0.0,
                "first_npz_member": "none",
                "primary_blocker": "trajectory_payload",
            },
        ]
        readiness_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "member_count": 1.0,
                "sota_inversion_ready": 0.0,
                "readiness_stage": "frame_index_curve_only",
            },
            {
                "system_id": "KA",
                "temperature": "none",
                "member_count": 0.0,
                "sota_inversion_ready": 0.0,
                "readiness_stage": "upstream_curve_incomplete",
            },
        ]

        rows = sota_glassbench_trajectory_npz_ensemble_horizon_gate(
            horizon_id="glassbench_npz_ensemble_horizon",
            accession_id="glassbench_zenodo_10118191",
            tar_probe_rows=tar_probe_rows,
            inversion_readiness_rows=readiness_rows,
            min_member_count=4,
        )

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d = by_key[("KA2D", "0.30")]
        self.assertEqual(float(ka2d["prefix_npz_member_count"]), 3.0)
        self.assertEqual(float(ka2d["extracted_curve_member_count"]), 1.0)
        self.assertEqual(float(ka2d["member_count_gap_to_threshold"]), 1.0)
        self.assertEqual(float(ka2d["prefix_member_horizon_ready"]), 0.0)
        self.assertEqual(float(ka2d["multi_npz_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d["primary_blocker"], "additional_npz_member_headers")
        self.assertEqual(ka2d["horizon_stage"], "prefix_member_horizon_short")
        self.assertEqual(ka2d["next_required_action"], "extend_tar_probe_or_index_full_member_list")

        ka = by_key[("KA", "none")]
        self.assertEqual(ka["horizon_stage"], "trajectory_layout_incomplete")
        self.assertEqual(ka["primary_blocker"], "trajectory_payload")

    def test_sota_glassbench_trajectory_npz_member_index_promotes_member_list_without_inversion(self):
        tar_probe_rows = [
            {
                "system_id": "KA2D",
                "temperature": "0.30",
                "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                "trajectory_layout_ready": 1.0,
                "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                "npz_member_count_in_probe": 3.0,
                "split_labels_in_probe": "train",
            }
        ]
        member_index_manifest = {
            "source": "remote_zip_member_to_extended_tar_member_index",
            "entries": [
                {
                    "path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "compressed_probe_bytes": 8_388_608,
                    "tar_probe_bytes": 4_194_304,
                    "member_index_complete_for_probe": True,
                    "npz_members": [
                        {"name": "T0.30/train/N1290T0.30_3_tc01.npz", "size_bytes": 444_786},
                        {"name": "T0.30/train/N1290T0.30_10_tc01.npz", "size_bytes": 444_786},
                        {"name": "T0.30/train/N1290T0.30_19_tc01.npz", "size_bytes": 444_786},
                        {"name": "T0.30/train/N1290T0.30_28_tc01.npz", "size_bytes": 444_786},
                        {"name": "T0.30/train/N1290T0.30_30_tc01.npz", "size_bytes": 444_786},
                    ],
                }
            ],
        }

        rows = sota_glassbench_trajectory_npz_member_index_gate(
            index_id="glassbench_npz_member_index",
            accession_id="glassbench_zenodo_10118191",
            tar_probe_rows=tar_probe_rows,
            member_index_manifest=member_index_manifest,
            min_member_count=4,
        )

        row = rows[0]
        self.assertEqual(row["member_index_stage"], "member_index_threshold_ready_extraction_pending")
        self.assertEqual(float(row["indexed_npz_member_count"]), 5.0)
        self.assertEqual(float(row["required_member_count"]), 4.0)
        self.assertEqual(float(row["member_count_threshold_pass"]), 1.0)
        self.assertEqual(float(row["full_member_id_list_visible"]), 1.0)
        self.assertEqual(
            row["first_four_member_ids"],
            "N1290T0.30_3_tc01;N1290T0.30_10_tc01;N1290T0.30_19_tc01;N1290T0.30_28_tc01",
        )
        self.assertEqual(row["split_labels_in_index"], "train")
        self.assertEqual(float(row["multi_npz_extraction_ready"]), 0.0)
        self.assertEqual(float(row["real_reanalysis_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "multi_npz_observable_extraction")

    def test_sota_glassbench_trajectory_member_ensemble_observables_add_frame_uncertainties(self):
        manifest = {
            "source": "remote_zip_member_first_four_npz_observable_ensemble",
            "entries": [
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "wave_numbers": [0.7, 1.1],
                    "overlap_radius": 0.1,
                    "members": [
                        {
                            "member": f"T0.30/train/N1290T0.30_{member_id}_tc01.npz",
                            "frame_indices": [0, 1],
                            "msd": [0.0, 0.004 + 0.001 * offset],
                            "ngp_2d": [0.0, 0.05 + 0.01 * offset],
                            "self_intermediate_scattering_by_k": ["1;1", f"{0.99 - 0.01 * offset};{0.98 - 0.01 * offset}"],
                            "chi4_overlap": [0.0, 1.0 + offset],
                        }
                        for offset, member_id in enumerate([3, 10, 19, 28])
                    ],
                }
            ],
        }

        rows = sota_glassbench_trajectory_member_ensemble_observable_gate(
            ensemble_id="glassbench_member_ensemble_observable",
            accession_id="glassbench_zenodo_10118191",
            member_observable_manifest=manifest,
            min_member_count=4,
        )

        by_frame = {float(row["frame_index"]): row for row in rows}
        frame1 = by_frame[1.0]
        self.assertEqual(float(frame1["member_count"]), 4.0)
        self.assertEqual(float(frame1["ensemble_member_threshold_pass"]), 1.0)
        self.assertAlmostEqual(float(frame1["msd"]), 0.0055)
        self.assertGreater(float(frame1["sigma_msd"]), 0.0)
        self.assertGreater(float(frame1["sigma_ngp_2d"]), 0.0)
        self.assertEqual(frame1["wave_numbers"], "0.7;1.1")
        self.assertEqual(frame1["self_intermediate_scattering_by_k"], "0.975;0.965")
        self.assertGreater(float(frame1["sigma_chi4_overlap"]), 0.0)
        self.assertEqual(float(frame1["frame_index_uncertainty_ready"]), 1.0)
        self.assertEqual(float(frame1["physical_time_ready"]), 0.0)
        self.assertEqual(float(frame1["sota_inversion_ready"]), 0.0)
        self.assertEqual(frame1["primary_blocker"], "physical_time_semantics")

    def test_sota_glassbench_ka2d_timecode_semantics_corrects_replica_axis(self):
        manifest = {
            "source": "remote_zip_ka2d_trajectory_readme_timecode_semantics_and_corrected_member_observables",
            "axis_semantics_evidence": "positions has shape (20,1290,2) for the 20 isoconfigurational trajectories",
            "entries": [
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "tau_alpha": 2200.0,
                    "time_code_map": {"tc01": 0.11, "tc02": 1.25},
                    "members": [
                        {
                            "member": "T0.30/train/N1290T0.30_3_tc01.npz",
                            "time_code": "tc01",
                            "lag_time": 0.11,
                            "lag_time_over_tau_alpha": 0.00005,
                            "axis0_semantics": "isoconfigurational_trajectory_replicates",
                            "replica_count": 20,
                            "msd": 0.005,
                            "ngp_2d": 0.04,
                            "self_intermediate_scattering_by_k": [0.99, 0.98],
                            "chi4_overlap_replica": 0.08,
                        },
                        {
                            "member": "T0.30/train/N1290T0.30_10_tc01.npz",
                            "time_code": "tc01",
                            "lag_time": 0.11,
                            "lag_time_over_tau_alpha": 0.00005,
                            "axis0_semantics": "isoconfigurational_trajectory_replicates",
                            "replica_count": 20,
                            "msd": 0.007,
                            "ngp_2d": 0.06,
                            "self_intermediate_scattering_by_k": [0.97, 0.96],
                            "chi4_overlap_replica": 0.12,
                        },
                    ],
                }
            ],
        }

        rows = sota_glassbench_ka2d_timecode_semantics_gate(
            semantics_id="glassbench_ka2d_timecode_semantics",
            accession_id="glassbench_zenodo_10118191",
            semantics_manifest=manifest,
            min_members_per_time_code=2,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["time_code"], "tc01")
        self.assertEqual(float(row["lag_time"]), 0.11)
        self.assertEqual(float(row["physical_lag_time_ready"]), 1.0)
        self.assertEqual(float(row["axis0_is_isoconfigurational_replica"]), 1.0)
        self.assertEqual(float(row["frame_axis_is_physical_time"]), 0.0)
        self.assertEqual(float(row["member_count"]), 2.0)
        self.assertAlmostEqual(float(row["msd"]), 0.006)
        self.assertGreater(float(row["sigma_msd_member_sem"]), 0.0)
        self.assertEqual(row["self_intermediate_scattering_by_k"], "0.98;0.97")
        self.assertEqual(float(row["all_time_codes_observed"]), 0.0)
        self.assertEqual(float(row["timecode_curve_ready"]), 0.0)
        self.assertEqual(float(row["sota_inversion_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "sparse_time_code_coverage")

    def test_sota_glassbench_visible_member_ensemble_audit_blocks_first_member_only_prefix(self):
        rows = sota_glassbench_visible_member_ensemble_audit_gate(
            audit_id="glassbench_visible_member_ensemble_audit",
            accession_id="glassbench_zenodo_10118191",
            tar_probe_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "source_path": "GlassBench/KA2D_trajectories/T0.23.tar.xz",
                    "first_npz_member": "T0.23/test/N1290T0.23_202_tc05.npz",
                    "split_labels_in_probe": "test",
                    "npz_member_count_in_probe": 3.0,
                    "trajectory_layout_ready": 1.0,
                }
            ],
            ensemble_horizon_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.23",
                    "prefix_npz_member_count": 3.0,
                    "min_member_count": 4.0,
                    "member_count_gap_to_threshold": 1.0,
                    "prefix_member_horizon_ready": 0.0,
                }
            ],
        )

        row = rows[0]
        self.assertEqual(row["first_member_id"], "N1290T0.23_202_tc05")
        self.assertEqual(row["split_labels_in_probe"], "test")
        self.assertEqual(float(row["prefix_npz_member_count"]), 3.0)
        self.assertEqual(float(row["required_member_count"]), 4.0)
        self.assertEqual(float(row["additional_member_count_needed"]), 1.0)
        self.assertEqual(float(row["first_member_id_visible"]), 1.0)
        self.assertEqual(float(row["full_member_id_list_visible"]), 0.0)
        self.assertEqual(float(row["split_policy_documented"]), 1.0)
        self.assertEqual(float(row["publishable_ensemble_uncertainty_ready"]), 0.0)
        self.assertEqual(row["primary_blocker"], "member_count_and_full_member_list")
        self.assertIn("index_full_npz_member_list", row["next_required_actions"])
        self.assertIn("extract_at_least_4_independent_members_per_temperature", row["next_required_actions"])
        self.assertEqual(row["ensemble_audit_stage"], "visible_prefix_not_publishable_ensemble")

    def test_sota_glassbench_visible_member_ensemble_audit_accepts_complete_member_list(self):
        rows = sota_glassbench_visible_member_ensemble_audit_gate(
            audit_id="glassbench_visible_member_ensemble_audit",
            accession_id="glassbench_zenodo_10118191",
            tar_probe_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "source_path": "GlassBench/KA2D_trajectories/T0.30.tar.xz",
                    "first_npz_member": "T0.30/train/N1290T0.30_3_tc01.npz",
                    "split_labels_in_probe": "train",
                    "npz_member_count_in_probe": 4.0,
                    "visible_npz_members": "T0.30/train/member_1.npz;T0.30/train/member_2.npz;T0.30/train/member_3.npz;T0.30/train/member_4.npz",
                    "trajectory_layout_ready": 1.0,
                }
            ],
            ensemble_horizon_rows=[
                {
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "prefix_npz_member_count": 4.0,
                    "min_member_count": 4.0,
                    "member_count_gap_to_threshold": 0.0,
                    "prefix_member_horizon_ready": 1.0,
                }
            ],
        )

        row = rows[0]
        self.assertEqual(float(row["full_member_id_list_visible"]), 1.0)
        self.assertEqual(float(row["member_count_threshold_pass"]), 1.0)
        self.assertEqual(float(row["publishable_ensemble_uncertainty_ready"]), 1.0)
        self.assertEqual(row["primary_blocker"], "none")
        self.assertEqual(row["next_required_actions"], "compute_member_resolved_observables_and_uncertainties")
        self.assertEqual(row["ensemble_audit_stage"], "visible_member_ensemble_ready_for_uncertainty")

    def test_sota_remote_result_curve_cache_verifies_range_cached_dat_files(self):
        manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA_results/times_0.44.dat",
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "time_grid",
                    "crc32_matches": True,
                    "md5": "aaaabbbbccccdddd",
                    "uncompressed_size_bytes": 59,
                    "numeric_row_count": 4,
                    "numeric_column_count": 1,
                    "range_start": 11340,
                    "range_end": 11428,
                },
                {
                    "path": "GlassBench/KA_results/chi4_KA_T0.44_update.dat",
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "chi4_proxy",
                    "crc32_matches": True,
                    "md5": "eeeeffff00001111",
                    "uncompressed_size_bytes": 36,
                    "numeric_row_count": 2,
                    "numeric_column_count": 2,
                    "range_start": 18780,
                    "range_end": 18850,
                },
                {
                    "path": "GlassBench/KA2D_results/times_0.30.dat",
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "curve_role": "time_grid",
                    "crc32_matches": True,
                    "md5": "2222333344445555",
                    "uncompressed_size_bytes": 37,
                    "numeric_row_count": 3,
                    "numeric_column_count": 1,
                    "range_start": 3600,
                    "range_end": 3700,
                },
                {
                    "path": "GlassBench/KA2D_results/rhomax_T0.30_MD.dat",
                    "system_id": "KA2D",
                    "temperature": "0.30",
                    "curve_role": "rhomax_md",
                    "crc32_matches": True,
                    "md5": "6666777788889999",
                    "uncompressed_size_bytes": 152,
                    "numeric_row_count": 5,
                    "numeric_column_count": 2,
                    "range_start": 5300,
                    "range_end": 5420,
                },
            ]
        }

        rows = sota_remote_result_curve_cache_gate(
            curve_cache_id="glassbench_range_result_curves",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            manifest=manifest,
            required_roles_by_system={
                "KA": ["time_grid", "chi4_proxy"],
                "KA2D": ["time_grid", "rhomax_md"],
            },
            max_uncompressed_size_bytes=10_000,
        )

        by_system = {row["system_id"]: row for row in rows}
        ka = by_system["KA"]
        self.assertEqual(ka["curve_cache_stage"], "range_result_curves_verified")
        self.assertEqual(ka["curve_cache_ready"], 1.0)
        self.assertEqual(ka["curve_file_count"], 2.0)
        self.assertEqual(ka["temperature_count"], 1.0)
        self.assertEqual(ka["available_roles"], "chi4_proxy;time_grid")
        self.assertEqual(ka["real_inversion_ready"], 0.0)
        self.assertEqual(ka["primary_blocker"], "raw_curve_adapter")

        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["curve_cache_stage"], "range_result_curves_verified")
        self.assertEqual(ka2d["curve_cache_ready"], 1.0)
        self.assertEqual(ka2d["temperature_grid"], "0.30")
        self.assertEqual(ka2d["primary_blocker"], "raw_curve_adapter")

    def test_sota_remote_result_curve_fetch_gap_marks_chi4_target_missing_from_cache(self):
        central_directory_manifest = {
            "entries": [
                "GlassBench/KA_results/times_0.44.dat",
                "GlassBench/KA_results/chi4_KA_T0.44_update.dat",
                "GlassBench/KA_results/rhomax_T0.44_MD.dat",
            ]
        }
        range_cache_manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA_results/times_0.44.dat",
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "time_grid",
                    "crc32_matches": True,
                    "md5": "time-md5",
                    "numeric_row_count": 10,
                    "numeric_column_count": 1,
                    "range_start": 100,
                    "range_end": 150,
                }
            ]
        }

        rows = sota_remote_result_curve_fetch_gap_gate(
            gap_id="glassbench_range_curve_fetch_gap",
            accession_id="glassbench_zenodo_10118191",
            central_directory_manifest=central_directory_manifest,
            range_cache_manifest=range_cache_manifest,
            target_curve_specs=[
                {
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "chi4_proxy",
                    "path": "GlassBench/KA_results/chi4_KA_T0.44_update.dat",
                    "candidate_observable": "dynamic_heterogeneity_chi4_proxy",
                }
            ],
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["fetch_gap_stage"], "remote_target_present_range_cache_missing")
        self.assertEqual(row["candidate_observable"], "dynamic_heterogeneity_chi4_proxy")
        self.assertEqual(row["central_directory_present"], 1.0)
        self.assertEqual(row["range_cache_present"], 0.0)
        self.assertEqual(row["targeted_fetch_ready"], 1.0)
        self.assertEqual(row["observable_comparison_ready"], 0.0)
        self.assertEqual(row["real_inversion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "range_result_curve_cache")

    def test_sota_remote_result_curve_target_fetch_marks_header_only_chi4_payload(self):
        central_directory_manifest = {
            "entries": [
                "GlassBench/KA_results/chi4_KA_T0.44_update.dat",
            ]
        }
        target_fetch_manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA_results/chi4_KA_T0.44_update.dat",
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "chi4_proxy",
                    "candidate_observable": "dynamic_heterogeneity_chi4_proxy",
                    "crc32_matches": True,
                    "md5": "2def7c42b63e7c347b8c4747974d8323",
                    "uncompressed_size_bytes": 36,
                    "numeric_row_count": 0,
                    "numeric_column_count": 0,
                    "header": ["t", "True", "Shiba", "Alkemade", "Jung", "Francois"],
                    "rows": [],
                    "range_start": 18760,
                    "range_end": 18797,
                }
            ]
        }

        rows = sota_remote_result_curve_target_fetch_gate(
            target_fetch_id="glassbench_range_curve_target_fetch",
            accession_id="glassbench_zenodo_10118191",
            central_directory_manifest=central_directory_manifest,
            target_fetch_manifest=target_fetch_manifest,
            target_paths=["GlassBench/KA_results/chi4_KA_T0.44_update.dat"],
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["target_fetch_stage"], "target_fetch_header_only_parse_blocked")
        self.assertEqual(row["candidate_observable"], "dynamic_heterogeneity_chi4_proxy")
        self.assertEqual(row["central_directory_present"], 1.0)
        self.assertEqual(row["target_fetch_present"], 1.0)
        self.assertEqual(row["target_fetch_checksum_ready"], 1.0)
        self.assertEqual(row["header_only_payload"], 1.0)
        self.assertEqual(row["numeric_payload_ready"], 0.0)
        self.assertEqual(row["observable_comparison_ready"], 0.0)
        self.assertEqual(row["real_inversion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "numeric_rows")

    def test_sota_remote_result_curve_published_semantic_audit_blocks_ml_feature_curves(self):
        payload_cache = {
            "entries": [
                {
                    "path": "GlassBench/KA2D_results/FIG3.dat",
                    "system_id": "KA2D",
                    "temperature": "none",
                    "curve_role": "published_figure_curve",
                    "header": ["t", "BOTAN", "CAGE", "GlassMLP", "SE3", "DEN", "EPOT", "PSI4", "TT"],
                    "numeric_row_count": 6,
                    "numeric_column_count": 9,
                    "rows": [[0.11, 0.92, 0.39, 0.26, 0.89, -0.03, 0.02, 0.01, -0.01]],
                }
            ]
        }

        rows = sota_remote_result_curve_published_semantic_audit_gate(
            audit_id="glassbench_published_curve_semantic_audit",
            accession_id="glassbench_zenodo_10118191",
            payload_cache=payload_cache,
            physical_observable_labels=["msd", "f_s", "ngp", "alpha2", "chi4", "diffusion", "tau_alpha"],
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["semantic_stage"], "published_curve_ml_benchmark_not_physical_observable")
        self.assertEqual(row["source_path"], "GlassBench/KA2D_results/FIG3.dat")
        self.assertEqual(row["time_axis_present"], 1.0)
        self.assertEqual(row["header_semantics_ready"], 1.0)
        self.assertEqual(row["physical_observable_label_match"], 0.0)
        self.assertEqual(row["ml_feature_column_count"], 8.0)
        self.assertEqual(row["published_curve_ready"], 1.0)
        self.assertEqual(row["observable_comparison_ready"], 0.0)
        self.assertEqual(row["real_inversion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "physical_observable_label")

    def test_sota_remote_result_curve_payload_adapter_pairs_time_and_rhomax(self):
        manifest = {
            "entries": [
                {
                    "path": "GlassBench/KA_results/times_0.44.dat",
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "time_grid",
                    "crc32": "aaaa1111",
                    "md5": "time-md5",
                    "numeric_row_count": 3,
                    "numeric_column_count": 1,
                },
                {
                    "path": "GlassBench/KA_results/rhomax_T0.44_MD.dat",
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "rhomax_md",
                    "crc32": "bbbb2222",
                    "md5": "rho-md5",
                    "numeric_row_count": 3,
                    "numeric_column_count": 2,
                },
            ]
        }
        payload_cache = {
            "entries": [
                {
                    "path": "GlassBench/KA_results/times_0.44.dat",
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "time_grid",
                    "crc32": "aaaa1111",
                    "md5": "time-md5",
                    "numeric_row_count": 3,
                    "numeric_column_count": 1,
                    "rows": [[0.13], [1.3], [13.0]],
                },
                {
                    "path": "GlassBench/KA_results/rhomax_T0.44_MD.dat",
                    "system_id": "KA",
                    "temperature": "0.44",
                    "curve_role": "rhomax_md",
                    "crc32": "bbbb2222",
                    "md5": "rho-md5",
                    "numeric_row_count": 3,
                    "numeric_column_count": 2,
                    "rows": [[0.13, 0.94], [1.3, 0.91], [13.0, 0.82]],
                },
            ]
        }

        rows = sota_remote_result_curve_payload_adapter_gate(
            payload_adapter_id="glassbench_range_curve_payload_adapter",
            accession_id="glassbench_zenodo_10118191",
            manifest=manifest,
            payload_cache=payload_cache,
            paired_value_roles_by_system={"KA": ["rhomax_md"]},
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["adapter_stage"], "range_curve_payload_adapter_ready")
        self.assertEqual(row["system_id"], "KA")
        self.assertEqual(row["temperature"], "0.44")
        self.assertEqual(row["curve_role"], "rhomax_md")
        self.assertEqual(row["available_columns"], "temperature;time;rhomax")
        self.assertEqual(row["time_grid_matches_value_time"], 1.0)
        self.assertEqual(row["structural_adapter_ready"], 1.0)
        self.assertEqual(row["uncertainty_adapter_ready"], 0.0)
        self.assertEqual(row["real_inversion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "sigma_rhomax")

    def test_sota_remote_result_curve_observable_semantics_keeps_rhomax_as_proxy(self):
        payload_rows = [
            {
                "payload_adapter_id": "glassbench_range_curve_payload_adapter_ka2d_0.30_rhomax_md",
                "accession_id": "glassbench_zenodo_10118191",
                "system_id": "KA2D",
                "temperature": "0.30",
                "curve_role": "rhomax_md",
                "available_columns": "temperature;time;rhomax",
                "structural_adapter_ready": 1.0,
                "uncertainty_adapter_ready": 0.0,
                "primary_blocker": "sigma_rhomax",
            },
            {
                "payload_adapter_id": "glassbench_range_curve_payload_adapter_ka_0.44_rhomax_md",
                "accession_id": "glassbench_zenodo_10118191",
                "system_id": "KA",
                "temperature": "0.44",
                "curve_role": "rhomax_md",
                "available_columns": "none",
                "structural_adapter_ready": 0.0,
                "uncertainty_adapter_ready": 0.0,
                "primary_blocker": "numeric_rows",
            },
        ]

        rows = sota_remote_result_curve_observable_semantics_gate(
            semantics_id="glassbench_range_curve_observable_semantics",
            accession_id="glassbench_zenodo_10118191",
            payload_adapter_rows=payload_rows,
            role_semantics={
                "rhomax_md": {
                    "candidate_observable": "overlap_density_proxy",
                    "available_semantics": ["temperature", "time", "rhomax", "curve_role_label"],
                }
            },
            required_model_semantics=[
                "alpha_decay",
                "diffusion",
                "late_ngp",
                "chi4_proxy",
                "uncertainty",
            ],
        )

        by_system = {row["system_id"]: row for row in rows}
        ka2d = by_system["KA2D"]
        self.assertEqual(ka2d["semantics_stage"], "proxy_observable_ready_model_semantics_incomplete")
        self.assertEqual(ka2d["candidate_observable"], "overlap_density_proxy")
        self.assertEqual(ka2d["proxy_observable_ready"], 1.0)
        self.assertEqual(ka2d["diagnostic_semantics_ready"], 0.0)
        self.assertIn("alpha_decay", ka2d["missing_model_semantics"])
        self.assertEqual(ka2d["primary_blocker"], "model_observable_semantics")
        self.assertEqual(ka2d["real_inversion_ready"], 0.0)

        ka = by_system["KA"]
        self.assertEqual(ka["semantics_stage"], "structural_adapter_blocked")
        self.assertEqual(ka["proxy_observable_ready"], 0.0)
        self.assertEqual(ka["primary_blocker"], "numeric_rows")

    def test_sota_reanalysis_state_gate_marks_metadata_verified_not_reanalysis(self):
        row = sota_reanalysis_state_gate(
            state_id="glassbench_reanalysis_state",
            source_id="glassbench_zenodo_trajectory_release",
            accession_ready=True,
            readme_digest_ready=True,
            local_cache_verified=False,
            zip_structure_ready=False,
            adapter_ready=False,
            local_cache_blocker="archive_path",
            zip_structure_blocker="archive_path",
            adapter_blocker="coordinate_file",
            required_final_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_curve_persistence_exchange_gate",
                "trajectory_prediction_falsification_gate",
            ],
        )

        self.assertEqual(row["reanalysis_stage"], "awaiting_full_archive_cache")
        self.assertEqual(row["claim_level"], "metadata_verified_not_reanalysis")
        self.assertEqual(row["ready_for_trajectory_reanalysis"], 0.0)
        self.assertEqual(row["ready_for_model_comparison"], 0.0)
        self.assertEqual(row["primary_blocker"], "archive_path")
        self.assertEqual(row["next_action"], "cache_full_archive_and_verify_checksum")

    def test_sota_reanalysis_state_gate_promotes_adapter_ready_state(self):
        row = sota_reanalysis_state_gate(
            state_id="synthetic_reanalysis_state",
            source_id="synthetic_fixture",
            accession_ready=True,
            readme_digest_ready=True,
            local_cache_verified=True,
            zip_structure_ready=True,
            adapter_ready=True,
            local_cache_blocker="none",
            zip_structure_blocker="none",
            adapter_blocker="none",
            required_final_protocols=[
                "trajectory_observable_protocol",
                "trajectory_curve_persistence_exchange_gate",
            ],
        )

        self.assertEqual(row["reanalysis_stage"], "ready_for_trajectory_observable_protocol")
        self.assertEqual(row["claim_level"], "local_archive_adapter_ready")
        self.assertEqual(row["ready_for_trajectory_reanalysis"], 1.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_sota_readme_schema_gate_marks_glassbench_remote_schema_ready(self):
        row = sota_readme_schema_gate(
            schema_id="glassbench_readme_schema",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            systems=["KA", "KA2D"],
            folder_tokens=["_trajectories", "_models", "_results"],
            license_statement="Creative Commons Attribution 4.0 International",
            required_citations=["10.1063/5.0129791", "10.1103/PhysRevLett.130.238202"],
            intended_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_inversion_readiness_gate",
            ],
            local_archive_inspected=False,
        )

        self.assertEqual(row["schema_stage"], "remote_readme_schema_ready")
        self.assertEqual(row["has_ka_system"], 1.0)
        self.assertEqual(row["has_ka2d_system"], 1.0)
        self.assertEqual(row["has_trajectory_folder"], 1.0)
        self.assertEqual(row["has_model_folder"], 1.0)
        self.assertEqual(row["has_results_folder"], 1.0)
        self.assertEqual(row["citation_count"], 2.0)
        self.assertEqual(row["schema_ready"], 1.0)
        self.assertEqual(row["ready_for_local_adapter"], 0.0)
        self.assertEqual(row["primary_blocker"], "local_archive_inspection")

    def test_sota_readme_schema_gate_blocks_missing_trajectory_folder(self):
        row = sota_readme_schema_gate(
            schema_id="article_supplement_schema_missing_trajectories",
            accession_id="hedges_jcp_article_no_archive",
            source_id="hedges_persistence_exchange_jcp_article",
            systems=["KA"],
            folder_tokens=["_models", "_results"],
            license_statement="article",
            required_citations=["10.1063/1.2817607"],
            intended_protocols=["trajectory_observable_protocol"],
            local_archive_inspected=False,
        )

        self.assertEqual(row["schema_stage"], "metadata_incomplete_schema")
        self.assertEqual(row["schema_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "trajectory_folder")

    def test_trajectory_adapter_contract_blocks_remote_glassbench_without_local_files(self):
        row = trajectory_adapter_contract(
            contract_id="glassbench_ka_remote_contract",
            accession_id="glassbench_zenodo_10118191",
            source_id="glassbench_zenodo_trajectory_release",
            system_id="KA",
            expected_archive_roots=["KA/_trajectories", "KA/_models", "KA/_results"],
            required_local_fields=[
                "archive_root",
                "trajectory_folder",
                "coordinate_file",
                "time_grid",
                "particle_identity",
                "box_geometry",
                "temperature_or_state_point",
                "species_labels",
                "units_metadata",
            ],
            available_local_fields=["archive_root", "trajectory_folder"],
            intended_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_inversion_readiness_gate",
            ],
            local_archive_inspected=False,
        )

        self.assertEqual(row["adapter_stage"], "remote_adapter_contract_only")
        self.assertEqual(row["adapter_ready"], 0.0)
        self.assertEqual(row["local_archive_inspected"], 0.0)
        self.assertEqual(row["available_required_field_count"], 2.0)
        self.assertEqual(row["primary_blocker"], "coordinate_file")
        self.assertIn("coordinate_file", row["missing_local_fields"])

    def test_trajectory_adapter_contract_promotes_local_synthetic_adapter(self):
        required_fields = [
            "archive_root",
            "trajectory_folder",
            "coordinate_file",
            "time_grid",
            "particle_identity",
            "box_geometry",
            "temperature_or_state_point",
            "species_labels",
            "units_metadata",
        ]
        row = trajectory_adapter_contract(
            contract_id="synthetic_local_trajectory_adapter",
            accession_id="synthetic_local_cache",
            source_id="synthetic_intermediate_scattering_fixture",
            system_id="synthetic",
            expected_archive_roots=["synthetic/_trajectories"],
            required_local_fields=required_fields,
            available_local_fields=required_fields,
            intended_protocols=[
                "trajectory_observable_protocol",
                "trajectory_uncertainty_protocol",
                "trajectory_inversion_readiness_gate",
            ],
            local_archive_inspected=True,
        )

        self.assertEqual(row["adapter_stage"], "local_trajectory_adapter_ready")
        self.assertEqual(row["adapter_ready"], 1.0)
        self.assertEqual(row["local_archive_inspected"], 1.0)
        self.assertEqual(row["missing_local_fields"], "none")
        self.assertEqual(row["primary_blocker"], "none")

    def test_sota_claim_alignment_scores_supported_dynamic_claim(self):
        row = sota_claim_alignment(
            claim_id="hedges_persistence_exchange_decoupling",
            source_key="hedges2007persistence",
            phenomenon="persistence_exchange_decoupling",
            claim_type="dynamical_signature",
            observed_claim="persistence and exchange times decouple on cooling",
            model_diagnostic="raw_curve_persistence_exchange_protocol",
            model_support_level="derived",
            data_readiness="qualitative",
            primary_blocker="machine_readable_curves",
        )

        self.assertEqual(row["claim_alignment"], "supported")
        self.assertEqual(row["model_overclaims_source"], 0.0)
        self.assertEqual(row["requires_external_closure"], 0.0)
        self.assertEqual(row["quantitative_fit_ready"], 0.0)

    def test_sota_claim_alignment_marks_thermodynamic_boundary_as_not_derived(self):
        row = sota_claim_alignment(
            claim_id="kauzmann_entropy_transition",
            source_key="kauzmann1948nature",
            phenomenon="thermodynamic_glass_transition",
            claim_type="thermodynamic_transition",
            observed_claim="configurational entropy extrapolates toward an ideal-glass limit",
            model_diagnostic="adam_gibbs_entropy_closure",
            model_support_level="closure_only",
            data_readiness="qualitative",
            primary_blocker="thermodynamic_input_law",
        )

        self.assertEqual(row["claim_alignment"], "scope_boundary")
        self.assertEqual(row["model_overclaims_source"], 0.0)
        self.assertEqual(row["requires_external_closure"], 1.0)
        self.assertEqual(row["quantitative_fit_ready"], 0.0)

    def test_sota_claim_alignment_rejects_overclaimed_thermodynamic_derivation(self):
        with self.assertRaises(ValueError):
            sota_claim_alignment(
                claim_id="bad_thermodynamic_claim",
                source_key="kauzmann1948nature",
                phenomenon="thermodynamic_glass_transition",
                claim_type="thermodynamic_transition",
                observed_claim="ideal glass thermodynamics",
                model_diagnostic="renewal_dynamics",
                model_support_level="derived",
                data_readiness="qualitative",
                primary_blocker="none",
            )

    def test_sota_evidence_verdict_grades_direct_dynamic_support(self):
        row = sota_evidence_verdict(
            verdict_id="kob_andersen_van_hove_verdict",
            source_key="kob1995vanhove",
            phenomenon="cage_plateau_transient_ngp_van_hove_tail",
            claim_alignment="supported",
            signed_constraint_class="sota_consistent",
            data_readiness="structural_raw",
            requires_external_closure=False,
            quantitative_fit_ready=False,
            model_overclaims_source=False,
            reanalysis_stage="not_required_for_literature_claim",
        )

        self.assertEqual(row["evidence_grade"], "direct_dynamical_support")
        self.assertEqual(row["allowed_manuscript_claim"], "dynamical_signature_supported")
        self.assertEqual(row["publishable_without_overclaim"], 1.0)
        self.assertEqual(row["trajectory_reanalysis_required"], 0.0)

    def test_sota_evidence_verdict_keeps_boundaries_and_pending_reanalysis_separate(self):
        thermodynamic = sota_evidence_verdict(
            verdict_id="kauzmann_boundary_verdict",
            source_key="kauzmann1948nature;adam1965temperature",
            phenomenon="configurational_entropy_and_ideal_glass_scope",
            claim_alignment="scope_boundary",
            signed_constraint_class="scope_boundary_consistent",
            data_readiness="qualitative",
            requires_external_closure=True,
            quantitative_fit_ready=False,
            model_overclaims_source=False,
            reanalysis_stage="not_required_for_literature_claim",
        )
        pending = sota_evidence_verdict(
            verdict_id="glassbench_pending_verdict",
            source_key="glassbench_zenodo_trajectory_release",
            phenomenon="trajectory_level_persistence_exchange_test",
            claim_alignment="partial",
            signed_constraint_class="closure_assisted_consistent",
            data_readiness="metadata_verified_not_reanalysis",
            requires_external_closure=True,
            quantitative_fit_ready=False,
            model_overclaims_source=False,
            reanalysis_stage="awaiting_full_archive_cache",
        )

        self.assertEqual(thermodynamic["evidence_grade"], "thermodynamic_scope_boundary")
        self.assertEqual(thermodynamic["allowed_manuscript_claim"], "scope_boundary_only")
        self.assertEqual(thermodynamic["publishable_without_overclaim"], 1.0)
        self.assertEqual(pending["evidence_grade"], "pending_trajectory_reanalysis")
        self.assertEqual(pending["trajectory_reanalysis_required"], 1.0)
        self.assertEqual(pending["publishable_without_overclaim"], 0.0)

    def test_sota_evidence_class_gate_separates_experimental_trends_from_quantitative_fit(self):
        row = sota_evidence_class_gate(
            class_id="near_tg_experimental_heterogeneity_class",
            source_key="berthier2024experimental",
            source_modality="experiment",
            evidence_grade="closure_assisted_support",
            observed_signatures=[
                "dynamic_heterogeneity_growth",
                "spatial_correlation_growth",
            ],
            model_supported_signatures=[
                "dynamic_heterogeneity_growth",
            ],
            available_quantitative_inputs=[
                "trend_direction",
                "temperature_series",
            ],
            required_quantitative_inputs=[
                "particle_trajectories",
                "self_intermediate_scattering",
                "ngp",
                "uncertainty",
            ],
            requires_external_closure=True,
            has_machine_readable_curves=False,
            has_uncertainties=False,
            has_shared_ensemble=False,
        )

        self.assertEqual(row["evidence_class"], "closure_assisted_experimental_constraint")
        self.assertEqual(row["quantitative_inversion_allowed"], 0.0)
        self.assertEqual(row["trend_comparison_allowed"], 1.0)
        self.assertEqual(row["primary_blocker"], "particle_trajectories")
        self.assertIn("spatial_correlation_growth", row["missing_model_supported_signatures"])

    def test_sota_evidence_class_gate_promotes_only_uncertainty_weighted_machine_readable_rows(self):
        row = sota_evidence_class_gate(
            class_id="synthetic_member_ensemble_class",
            source_key="synthetic_member_ensemble_trajectory",
            source_modality="simulation",
            evidence_grade="direct_dynamical_support",
            observed_signatures=[
                "msd",
                "ngp",
                "self_intermediate_scattering",
                "chi4_overlap",
            ],
            model_supported_signatures=[
                "msd",
                "ngp",
                "self_intermediate_scattering",
                "chi4_overlap",
            ],
            available_quantitative_inputs=[
                "particle_trajectories",
                "self_intermediate_scattering",
                "ngp",
                "uncertainty",
                "shared_ensemble",
            ],
            required_quantitative_inputs=[
                "particle_trajectories",
                "self_intermediate_scattering",
                "ngp",
                "uncertainty",
                "shared_ensemble",
            ],
            requires_external_closure=False,
            has_machine_readable_curves=True,
            has_uncertainties=True,
            has_shared_ensemble=True,
        )

        self.assertEqual(row["evidence_class"], "uncertainty_weighted_quantitative_test")
        self.assertEqual(row["quantitative_inversion_allowed"], 1.0)
        self.assertEqual(row["trend_comparison_allowed"], 1.0)
        self.assertEqual(row["primary_blocker"], "none")

    def test_sota_signed_constraint_audit_accepts_dynamic_signatures_without_forbidden_claims(self):
        row = sota_signed_constraint_audit(
            constraint_id="kob_andersen_van_hove_signed_constraints",
            source_key="kob1995vanhove",
            model_scope="dynamical_signature",
            source_observation="KA cooling shows cage plateau, transient NGP, broad van-Hove tails, and recovery",
            expected_signatures=[
                "msd_plateau",
                "transient_ngp_peak",
                "van_hove_tail",
                "late_gaussian_recovery",
            ],
            passed_signatures=[
                "msd_plateau",
                "transient_ngp_peak",
                "van_hove_tail",
                "late_gaussian_recovery",
            ],
            forbidden_claims=["thermodynamic_transition_derived"],
            made_claims=["finite_exchange_dynamic_diagnostic"],
            support_level="derived",
            quantitative_fit_ready=False,
        )

        self.assertEqual(row["signed_constraint_class"], "sota_consistent")
        self.assertEqual(row["missing_expected_signatures"], "none")
        self.assertEqual(row["forbidden_claims_made"], "none")
        self.assertEqual(row["all_required_signatures_pass"], 1.0)
        self.assertEqual(row["publishable_alignment"], 1.0)

    def test_sota_signed_constraint_audit_keeps_spatial_and_thermodynamic_boundaries(self):
        spatial = sota_signed_constraint_audit(
            constraint_id="lacevic_four_point_signed_constraints",
            source_key="lacevic2003fourpoint",
            model_scope="spatial_heterogeneity",
            source_observation="four-point susceptibility and dynamic length grow on cooling",
            expected_signatures=["chi4_peak_growth", "dynamic_length_growth"],
            passed_signatures=["chi4_peak_growth", "dynamic_length_growth"],
            forbidden_claims=["microscopic_dynamic_length_derived"],
            made_claims=["chi4_proxy_closure"],
            support_level="effective_closure",
            quantitative_fit_ready=False,
        )
        thermodynamic = sota_signed_constraint_audit(
            constraint_id="kauzmann_thermodynamic_signed_boundary",
            source_key="kauzmann1948nature;adam1965temperature",
            model_scope="thermodynamic_transition",
            source_observation="entropy extrapolation and heat-capacity anomalies require thermodynamic input",
            expected_signatures=["entropy_closure_required", "heat_capacity_not_derived"],
            passed_signatures=["entropy_closure_required", "heat_capacity_not_derived"],
            forbidden_claims=["ideal_glass_transition_derived", "heat_capacity_anomaly_derived"],
            made_claims=["thermodynamic_scope_boundary"],
            support_level="closure_only",
            quantitative_fit_ready=False,
        )
        overclaim = sota_signed_constraint_audit(
            constraint_id="bad_thermodynamic_overclaim",
            source_key="kauzmann1948nature",
            model_scope="thermodynamic_transition",
            source_observation="entropy anomaly",
            expected_signatures=["entropy_closure_required"],
            passed_signatures=["entropy_closure_required"],
            forbidden_claims=["ideal_glass_transition_derived"],
            made_claims=["ideal_glass_transition_derived"],
            support_level="closure_only",
            quantitative_fit_ready=False,
        )

        self.assertEqual(spatial["signed_constraint_class"], "closure_assisted_consistent")
        self.assertEqual(spatial["requires_external_closure"], 1.0)
        self.assertEqual(spatial["publishable_alignment"], 1.0)
        self.assertEqual(thermodynamic["signed_constraint_class"], "scope_boundary_consistent")
        self.assertEqual(thermodynamic["requires_external_closure"], 1.0)
        self.assertEqual(thermodynamic["publishable_alignment"], 1.0)
        self.assertEqual(overclaim["signed_constraint_class"], "overclaimed_boundary")
        self.assertEqual(overclaim["publishable_alignment"], 0.0)

    def test_observable_falsification_matrix_marks_diagnostic_blockers(self):
        rows = observable_falsification_matrix(
            benchmark_id="kob_andersen_combined_1995",
            benchmark_source="kob1995vanhove;kob1995intermediate",
            available_observables=[
                "time_grid",
                "self_intermediate_scattering",
                "tau_alpha",
                "wave_numbers",
                "van_hove_tail",
                "ngp",
            ],
            diagnostic_requirements={
                "multi_k_alpha_shape": [
                    "time_grid",
                    "self_intermediate_scattering",
                    "tau_alpha",
                    "wave_numbers",
                ],
                "van_hove_gaussian_recovery": ["time_grid", "van_hove_tail", "ngp", "diffusion"],
                "joint_persistence_exchange_chi4": [
                    "diffusion",
                    "tau_alpha",
                    "persistence_time",
                    "exchange_time",
                    "late_ngp",
                    "chi4_peak",
                ],
            },
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
        )

        by_diagnostic = {row["diagnostic_id"]: row for row in rows}
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["structural_falsification_ready"], 1.0)
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["quantitative_falsification_ready"], 0.0)
        self.assertEqual(by_diagnostic["van_hove_gaussian_recovery"]["missing_observables"], "diffusion")
        self.assertEqual(by_diagnostic["van_hove_gaussian_recovery"]["primary_blocker"], "diffusion")
        self.assertEqual(
            by_diagnostic["joint_persistence_exchange_chi4"]["missing_observables"],
            "diffusion;persistence_time;exchange_time;late_ngp;chi4_peak",
        )
        self.assertEqual(by_diagnostic["joint_persistence_exchange_chi4"]["structural_falsification_ready"], 0.0)

    def test_benchmark_fusion_readiness_requires_shared_system_and_grid(self):
        row = benchmark_fusion_readiness(
            fusion_id="kob_andersen_i_ii_dynamic_closure",
            benchmark_sources=["kob1995vanhove", "kob1995intermediate"],
            required_observables=[
                "time_grid",
                "van_hove_tail",
                "ngp",
                "diffusion",
                "self_intermediate_scattering",
                "tau_alpha",
                "wave_numbers",
            ],
            available_observables_by_benchmark={
                "kob1995vanhove": ["time_grid", "van_hove_tail", "ngp", "diffusion"],
                "kob1995intermediate": [
                    "time_grid",
                    "self_intermediate_scattering",
                    "tau_alpha",
                    "wave_numbers",
                ],
            },
            system_tags={
                "kob1995vanhove": "kob_andersen_binary_lj",
                "kob1995intermediate": "kob_andersen_binary_lj",
            },
            temperature_grid_tags={
                "kob1995vanhove": "ka_1995_grid",
                "kob1995intermediate": "ka_1995_grid",
            },
            ensemble_tags={
                "kob1995vanhove": "ka_1995_simulation",
                "kob1995intermediate": "ka_1995_simulation",
            },
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
        )

        self.assertEqual(row["missing_observables"], "none")
        self.assertEqual(row["shared_system_consistent"], 1.0)
        self.assertEqual(row["shared_temperature_grid_consistent"], 1.0)
        self.assertEqual(row["shared_ensemble_consistent"], 1.0)
        self.assertEqual(row["structural_fusion_ready"], 1.0)
        self.assertEqual(row["quantitative_fusion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "machine_readable_data")

    def test_benchmark_fusion_readiness_rejects_cross_ensemble_splicing(self):
        row = benchmark_fusion_readiness(
            fusion_id="ka_lacevic_four_point_splice",
            benchmark_sources=["kob1995intermediate", "lacevic2003fourpoint"],
            required_observables=[
                "self_intermediate_scattering",
                "tau_alpha",
                "chi4_peak",
                "dynamic_length",
            ],
            available_observables_by_benchmark={
                "kob1995intermediate": ["self_intermediate_scattering", "tau_alpha"],
                "lacevic2003fourpoint": ["tau_alpha", "chi4_peak", "dynamic_length"],
            },
            system_tags={
                "kob1995intermediate": "kob_andersen_binary_lj",
                "lacevic2003fourpoint": "kob_andersen_binary_lj",
            },
            temperature_grid_tags={
                "kob1995intermediate": "ka_1995_grid",
                "lacevic2003fourpoint": "lacevic_2003_grid",
            },
            ensemble_tags={
                "kob1995intermediate": "ka_1995_simulation",
                "lacevic2003fourpoint": "lacevic_2003_simulation",
            },
            has_machine_readable_data=False,
            has_uncertainty_estimates=False,
        )

        self.assertEqual(row["missing_observables"], "none")
        self.assertEqual(row["shared_system_consistent"], 1.0)
        self.assertEqual(row["shared_temperature_grid_consistent"], 0.0)
        self.assertEqual(row["shared_ensemble_consistent"], 0.0)
        self.assertEqual(row["structural_fusion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "temperature_grid_mismatch")

    def test_raw_curve_ingestion_contract_marks_missing_uncertainty_columns(self):
        rows = raw_curve_ingestion_contract(
            benchmark_id="kob_andersen_i_ii_dynamic_closure",
            observable_requirements={
                "self_intermediate_scattering": {
                    "required_columns": ["temperature", "wave_number", "time", "F_s"],
                    "uncertainty_columns": ["sigma_F_s"],
                    "target_diagnostic": "multi_k_alpha_shape",
                },
                "van_hove_ngp": {
                    "required_columns": ["temperature", "time", "radius", "G_s", "alpha2", "diffusion"],
                    "uncertainty_columns": ["sigma_G_s", "sigma_alpha2", "sigma_diffusion"],
                    "target_diagnostic": "van_hove_gaussian_recovery",
                },
            },
            available_columns_by_observable={
                "self_intermediate_scattering": ["temperature", "wave_number", "time", "F_s"],
                "van_hove_ngp": [
                    "temperature",
                    "time",
                    "radius",
                    "G_s",
                    "alpha2",
                    "diffusion",
                    "sigma_G_s",
                    "sigma_alpha2",
                    "sigma_diffusion",
                ],
            },
            machine_readable=True,
            shared_temperature_grid=True,
            shared_time_units=True,
        )

        by_observable = {row["observable_id"]: row for row in rows}
        self.assertEqual(by_observable["self_intermediate_scattering"]["structural_ingestion_ready"], 1.0)
        self.assertEqual(by_observable["self_intermediate_scattering"]["uncertainty_ingestion_ready"], 0.0)
        self.assertEqual(by_observable["self_intermediate_scattering"]["missing_columns"], "none")
        self.assertEqual(by_observable["self_intermediate_scattering"]["missing_uncertainty_columns"], "sigma_F_s")
        self.assertEqual(by_observable["self_intermediate_scattering"]["primary_blocker"], "sigma_F_s")
        self.assertEqual(by_observable["van_hove_ngp"]["uncertainty_ingestion_ready"], 1.0)
        self.assertEqual(by_observable["van_hove_ngp"]["primary_blocker"], "none")

    def test_raw_curve_ingestion_contract_rejects_unshared_units_for_fused_input(self):
        rows = raw_curve_ingestion_contract(
            benchmark_id="kob_andersen_i_ii_dynamic_closure",
            observable_requirements={
                "self_intermediate_scattering": {
                    "required_columns": ["temperature", "wave_number", "time", "F_s"],
                    "uncertainty_columns": [],
                    "target_diagnostic": "multi_k_alpha_shape",
                }
            },
            available_columns_by_observable={
                "self_intermediate_scattering": ["temperature", "wave_number", "time", "F_s"],
            },
            machine_readable=True,
            shared_temperature_grid=False,
            shared_time_units=True,
        )

        row = rows[0]
        self.assertEqual(row["structural_ingestion_ready"], 0.0)
        self.assertEqual(row["primary_blocker"], "temperature_grid_mismatch")

    def test_raw_curve_diagnostic_readiness_aggregates_contract_blockers(self):
        contract_rows = raw_curve_ingestion_contract(
            benchmark_id="kob_andersen_i_ii_dynamic_closure",
            observable_requirements={
                "self_intermediate_scattering": {
                    "required_columns": ["temperature", "wave_number", "time", "F_s"],
                    "uncertainty_columns": ["sigma_F_s"],
                    "target_diagnostic": "multi_k_alpha_shape",
                },
                "van_hove_ngp": {
                    "required_columns": ["temperature", "time", "radius", "G_s", "alpha2", "diffusion"],
                    "uncertainty_columns": ["sigma_G_s"],
                    "target_diagnostic": "van_hove_gaussian_recovery",
                },
            },
            available_columns_by_observable={
                "self_intermediate_scattering": ["temperature", "wave_number", "time", "F_s"],
                "van_hove_ngp": ["temperature", "time", "radius", "G_s", "alpha2", "diffusion", "sigma_G_s"],
            },
            machine_readable=True,
            shared_temperature_grid=True,
            shared_time_units=True,
        )
        rows = raw_curve_diagnostic_readiness(
            benchmark_id="kob_andersen_i_ii_dynamic_closure",
            contract_rows=contract_rows,
            diagnostic_observables={
                "multi_k_alpha_shape": ["self_intermediate_scattering"],
                "van_hove_gaussian_recovery": ["van_hove_ngp"],
                "combined_alpha_vanhove_closure": ["self_intermediate_scattering", "van_hove_ngp"],
            },
        )

        by_diagnostic = {row["diagnostic_id"]: row for row in rows}
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["structural_diagnostic_ready"], 1.0)
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["uncertainty_diagnostic_ready"], 0.0)
        self.assertEqual(by_diagnostic["multi_k_alpha_shape"]["primary_blocker"], "sigma_F_s")
        self.assertEqual(by_diagnostic["van_hove_gaussian_recovery"]["uncertainty_diagnostic_ready"], 1.0)
        self.assertEqual(by_diagnostic["van_hove_gaussian_recovery"]["primary_blocker"], "none")
        self.assertEqual(by_diagnostic["combined_alpha_vanhove_closure"]["structural_diagnostic_ready"], 1.0)
        self.assertEqual(by_diagnostic["combined_alpha_vanhove_closure"]["uncertainty_diagnostic_ready"], 0.0)
        self.assertEqual(by_diagnostic["combined_alpha_vanhove_closure"]["primary_blocker"], "sigma_F_s")

    def test_raw_curve_persistence_exchange_protocol_extracts_observables_and_passes(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=7.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        time_grid = np.geomspace(0.02, 800.0, 1600)
        alpha_curves = {
            wave_number: (
                time_grid,
                persistence_exchange_normalized_alpha_decay(wave_number, time_grid, params),
            )
            for wave_number in wave_numbers
        }
        late_time = 80.0 * params.persistence_mean
        ngp_time = np.geomspace(0.1, 1200.0, 1400)
        ngp_curve = (ngp_time, persistence_exchange_ngp_1d(ngp_time, params))
        chi4_time = np.geomspace(0.02, 400.0, 900)
        chi4_curve = (
            chi4_time,
            persistence_exchange_scattering_susceptibility(1.1, chi4_time, params),
        )

        row = raw_curve_persistence_exchange_protocol(
            benchmark_id="synthetic_raw_curve_closure",
            anchor_wave_number=1.1,
            alpha_curves_by_k=alpha_curves,
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            ngp_curve=ngp_curve,
            chi4_curve=chi4_curve,
            tau_alpha_relative_error_by_k={wave_number: 0.03 for wave_number in wave_numbers},
            late_ngp_relative_error=0.05,
            chi4_peak_relative_error=0.05,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=3.0,
        )

        self.assertEqual(row["benchmark_id"], "synthetic_raw_curve_closure")
        self.assertEqual(row["raw_curve_protocol_passes"], 1.0)
        self.assertAlmostEqual(row["persistence_exchange_ratio"], 7.0, delta=0.1)
        self.assertGreater(row["stokes_einstein_growth_over_poisson"], 2.0)
        self.assertLess(row["max_multik_tau_alpha_z"], 1.0)
        self.assertLess(row["late_ngp_z"], 1.0)
        self.assertLess(row["chi4_peak_z"], 1.0)

    def test_raw_curve_persistence_exchange_protocol_rejects_late_ngp_mismatch(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=7.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        time_grid = np.geomspace(0.02, 800.0, 1600)
        alpha_curves = {
            wave_number: (
                time_grid,
                persistence_exchange_normalized_alpha_decay(wave_number, time_grid, params),
            )
            for wave_number in wave_numbers
        }
        late_time = 80.0 * params.persistence_mean
        ngp_time = np.geomspace(0.1, 1200.0, 1400)
        corrupted_ngp = 1.8 * persistence_exchange_ngp_1d(ngp_time, params)
        chi4_time = np.geomspace(0.02, 400.0, 900)
        chi4_curve = (
            chi4_time,
            persistence_exchange_scattering_susceptibility(1.1, chi4_time, params),
        )

        row = raw_curve_persistence_exchange_protocol(
            benchmark_id="synthetic_raw_curve_late_ngp_mismatch",
            anchor_wave_number=1.1,
            alpha_curves_by_k=alpha_curves,
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            ngp_curve=(ngp_time, corrupted_ngp),
            chi4_curve=chi4_curve,
            tau_alpha_relative_error_by_k={wave_number: 0.03 for wave_number in wave_numbers},
            late_ngp_relative_error=0.05,
            chi4_peak_relative_error=0.05,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=3.0,
        )

        self.assertEqual(row["late_ngp_z_consistent"], 0.0)
        self.assertEqual(row["raw_curve_protocol_passes"], 0.0)
        self.assertGreater(row["late_ngp_z"], 3.0)

    def test_trajectory_observable_protocol_extracts_msd_ngp_fs_and_chi4(self):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.array(
            [
                [[0.0], [0.0], [0.0], [0.0]],
                [[0.0], [0.0], [2.0], [2.0]],
                [[0.0], [0.0], [2.0], [2.0]],
            ]
        )

        rows = trajectory_observable_protocol(
            positions=positions,
            times=times,
            lag_indices=[1, 2],
            wave_numbers=[math.pi],
            overlap_radius=0.5,
        )

        by_lag = {int(row["lag_index"]): row for row in rows}
        lag1 = by_lag[1]
        self.assertAlmostEqual(lag1["lag_time"], 1.0)
        self.assertAlmostEqual(lag1["msd"], 1.0)
        self.assertAlmostEqual(lag1["ngp"], 1.0 / 3.0)
        self.assertAlmostEqual(lag1["self_intermediate_scattering"], 1.0)
        self.assertAlmostEqual(lag1["overlap_mean"], 0.75)
        self.assertAlmostEqual(lag1["chi4_overlap"], 0.25)
        self.assertEqual(lag1["structural_observable_set"], "msd;ngp;self_intermediate_scattering;overlap_chi4")

        lag2 = by_lag[2]
        self.assertAlmostEqual(lag2["lag_time"], 2.0)
        self.assertAlmostEqual(lag2["msd"], 2.0)
        self.assertAlmostEqual(lag2["ngp"], -1.0 / 3.0)
        self.assertAlmostEqual(lag2["overlap_mean"], 0.5)

    def test_trajectory_cage_jump_event_protocol_extracts_persistence_and_exchange_clocks(self):
        positions = np.array(
            [
                [[0.0], [0.0], [0.0]],
                [[0.1], [0.0], [0.0]],
                [[1.4], [0.0], [0.0]],
                [[1.5], [1.2], [0.0]],
                [[2.8], [1.3], [1.1]],
            ],
            dtype=float,
        )
        times = np.arange(5.0)

        row = trajectory_cage_jump_event_protocol(
            protocol_id="synthetic_particle_cage_jump_events",
            positions=positions,
            times=times,
            jump_displacement_threshold=1.0,
            min_particles_with_jumps=2,
            min_exchange_interval_count=1,
        )

        self.assertEqual(row["event_protocol_stage"], "particle_resolved_cage_jump_event_clock_ready")
        self.assertEqual(float(row["particle_resolved_jump_events_ready"]), 1.0)
        self.assertEqual(float(row["physical_time_jump_clock_ready"]), 1.0)
        self.assertEqual(float(row["persistence_exchange_event_clock_ready"]), 1.0)
        self.assertEqual(float(row["total_jump_event_count"]), 4.0)
        self.assertEqual(float(row["particles_with_jump_count"]), 3.0)
        self.assertEqual(float(row["exchange_interval_count"]), 1.0)
        self.assertAlmostEqual(float(row["persistence_mean"]), 3.0)
        self.assertAlmostEqual(float(row["exchange_mean"]), 2.0)
        self.assertAlmostEqual(float(row["mean_squared_jump_length"]), 1.5075)
        self.assertGreater(float(row["jump_length_variance"]), 0.0)
        self.assertEqual(row["primary_blocker"], "none")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_trajectory_event_clock_macro_prediction_scores_heldout_signatures(self):
        event_row = {
            "protocol_id": "synthetic_particle_cage_jump_events",
            "persistence_exchange_event_clock_ready": 1.0,
            "persistence_mean": 3.0,
            "exchange_mean": 2.0,
            "mean_squared_jump_length": 1.5075,
            "dimension": 1.0,
        }
        params = PersistenceExchangeParams(
            cage_variance=0.5,
            cage_tau=0.2,
            jump_variance=1.5075,
            persistence_mean=3.0,
            exchange_mean=2.0,
        )
        wave_numbers = [0.8, 1.1]
        late_time = 12.0
        time_grid = np.geomspace(0.05, 30.0, 800)
        observed_tau_alpha_by_k = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])
        observed_chi4_peak = float(np.max(persistence_exchange_scattering_susceptibility(0.8, time_grid, params)))

        row = trajectory_event_clock_macro_prediction_protocol(
            protocol_id="synthetic_event_clock_macro_prediction",
            event_row=event_row,
            anchor_wave_number=0.8,
            wave_numbers=wave_numbers,
            observed_diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            diffusion_relative_error=0.05,
            observed_tau_alpha_by_k=observed_tau_alpha_by_k,
            tau_alpha_relative_error_by_k={0.8: 0.05, 1.1: 0.05},
            late_time=late_time,
            observed_late_ngp=observed_late_ngp,
            late_ngp_relative_error=0.10,
            observed_chi4_peak=observed_chi4_peak,
            chi4_peak_relative_error=0.10,
            time_grid=time_grid,
            cage_variance=0.5,
            cage_tau=0.2,
        )

        self.assertEqual(row["prediction_stage"], "event_clock_micro_to_macro_prediction_ready")
        self.assertEqual(float(row["micro_to_macro_prediction_ready"]), 1.0)
        self.assertEqual(float(row["micro_to_macro_predictions_pass"]), 1.0)
        self.assertEqual(float(row["calibrated_from_event_clock_only"]), 1.0)
        self.assertEqual(float(row["fit_parameters_from_macro_observables"]), 0.0)
        self.assertLess(float(row["diffusion_z"]), 1.0)
        self.assertLess(float(row["max_tau_alpha_z"]), 1.0)
        self.assertLess(float(row["late_ngp_z"]), 1.0)
        self.assertLess(float(row["chi4_peak_z"]), 1.0)
        self.assertEqual(row["primary_blocker"], "none")
        self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_trajectory_event_clock_threshold_robustness_detects_stable_window(self):
        positions = np.array(
            [
                [[0.0], [0.0], [0.0]],
                [[0.1], [0.0], [0.0]],
                [[1.4], [0.0], [0.0]],
                [[1.5], [1.2], [0.0]],
                [[2.8], [1.3], [1.1]],
            ],
            dtype=float,
        )
        times = np.arange(5.0)
        rows = trajectory_event_clock_threshold_robustness_protocol(
            protocol_id="synthetic_event_clock_threshold_robustness",
            positions=positions,
            times=times,
            thresholds=[0.05, 0.9, 1.0, 1.35],
            reference_threshold=1.0,
            anchor_wave_number=0.8,
            wave_numbers=[0.8, 1.1],
            late_time=12.0,
            time_grid=np.geomspace(0.05, 30.0, 800),
            min_particles_with_jumps=2,
            min_exchange_interval_count=1,
            cage_variance=0.5,
            cage_tau=0.2,
        )

        by_threshold = {float(row["jump_displacement_threshold"]): row for row in rows}
        self.assertEqual(by_threshold[1.0]["robustness_stage"], "event_clock_threshold_prediction_passed")
        self.assertEqual(by_threshold[0.9]["robustness_stage"], "event_clock_threshold_prediction_passed")
        self.assertEqual(float(by_threshold[1.0]["threshold_prediction_pass"]), 1.0)
        self.assertEqual(float(by_threshold[0.9]["threshold_prediction_pass"]), 1.0)
        self.assertGreaterEqual(float(by_threshold[1.0]["stable_threshold_window_count"]), 2.0)

        self.assertEqual(by_threshold[0.05]["robustness_stage"], "event_clock_threshold_prediction_failed")
        self.assertEqual(float(by_threshold[0.05]["threshold_prediction_pass"]), 0.0)
        self.assertEqual(by_threshold[0.05]["primary_blocker"], "threshold_macro_signature_mismatch")
        self.assertEqual(by_threshold[1.35]["robustness_stage"], "event_clock_threshold_event_clock_incomplete")
        self.assertEqual(by_threshold[1.35]["primary_blocker"], "jump_displacement_threshold")
        for row in rows:
            self.assertEqual(float(row["fit_parameters_from_macro_observables"]), 0.0)
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_trajectory_observable_protocol_validates_inputs(self):
        with self.assertRaises(ValueError):
            trajectory_observable_protocol(
                positions=np.zeros((2, 3)),
                times=np.array([0.0, 1.0]),
                lag_indices=[1],
                wave_numbers=[1.0],
                overlap_radius=0.5,
            )

    def test_trajectory_observable_curve_bridge_extracts_inversion_inputs(self):
        rows = [
            {
                "lag_time": 1.0,
                "dimension": 1.0,
                "msd": 1.0,
                "ngp": 0.7,
                "wave_numbers": "0.7;1.1",
                "self_intermediate_scattering_by_k": "0.9;0.8",
                "chi4_overlap": 0.2,
            },
            {
                "lag_time": 2.0,
                "dimension": 1.0,
                "msd": 3.6,
                "ngp": 0.35,
                "wave_numbers": "0.7;1.1",
                "self_intermediate_scattering_by_k": "0.5;0.3",
                "chi4_overlap": 1.4,
            },
            {
                "lag_time": 4.0,
                "dimension": 1.0,
                "msd": 8.0,
                "ngp": 0.12,
                "wave_numbers": "0.7;1.1",
                "self_intermediate_scattering_by_k": "0.2;0.1",
                "chi4_overlap": 0.6,
            },
        ]

        bridge = trajectory_observable_curve_bridge(
            benchmark_id="synthetic_csv_curve_bridge",
            rows=rows,
            required_wave_numbers=[0.7, 1.1],
            anchor_wave_number=1.1,
        )

        self.assertEqual(bridge["bridge_stage"], "trajectory_curve_bridge_ready")
        self.assertEqual(bridge["curve_bridge_ready"], 1.0)
        self.assertEqual(bridge["primary_blocker"], "none")
        self.assertAlmostEqual(bridge["diffusion_coefficient"], 1.0)
        self.assertAlmostEqual(bridge["late_time"], 4.0)
        self.assertAlmostEqual(bridge["late_ngp"], 0.12)
        self.assertAlmostEqual(bridge["chi4_peak"], 1.4)
        self.assertGreater(bridge["anchor_tau_alpha"], 1.0)
        self.assertGreater(bridge["d_tau_alpha_product"], 1.0)
        self.assertIn("1.1:", bridge["tau_alpha_by_k"])

    def test_trajectory_observable_curve_bridge_blocks_without_alpha_crossing(self):
        rows = [
            {
                "lag_time": 1.0,
                "dimension": 1.0,
                "msd": 1.0,
                "ngp": 0.4,
                "wave_numbers": "1.1",
                "self_intermediate_scattering_by_k": "0.95",
                "chi4_overlap": 0.2,
            },
            {
                "lag_time": 2.0,
                "dimension": 1.0,
                "msd": 2.0,
                "ngp": 0.3,
                "wave_numbers": "1.1",
                "self_intermediate_scattering_by_k": "0.8",
                "chi4_overlap": 0.4,
            },
        ]

        bridge = trajectory_observable_curve_bridge(
            benchmark_id="short_csv_curve_bridge",
            rows=rows,
            required_wave_numbers=[1.1],
            anchor_wave_number=1.1,
        )

        self.assertEqual(bridge["bridge_stage"], "trajectory_curve_bridge_incomplete")
        self.assertEqual(bridge["curve_bridge_ready"], 0.0)
        self.assertEqual(bridge["primary_blocker"], "alpha_threshold_crossing")
        self.assertEqual(bridge["tau_alpha_by_k"], "none")

    def test_trajectory_curve_persistence_exchange_gate_runs_joint_protocol_from_bridge(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=7.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        tau_by_k = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        late_time = 80.0 * params.persistence_mean
        time_grid = np.geomspace(0.02, 400.0, 900)
        bridge_row = {
            "benchmark_id": "synthetic_bridge_joint_protocol",
            "bridge_stage": "trajectory_curve_bridge_ready",
            "curve_bridge_ready": 1.0,
            "wave_numbers": "0.7;1.1;1.6",
            "anchor_wave_number": 1.1,
            "tau_alpha_by_k": ";".join(f"{wave_number}:{tau_by_k[wave_number]}" for wave_number in wave_numbers),
            "diffusion_coefficient": persistence_exchange_diffusion_coefficient(params),
            "late_time": late_time,
            "late_ngp": float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0]),
            "chi4_peak": float(
                np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, params))
            ),
        }

        gate = trajectory_curve_persistence_exchange_gate(
            bridge_row=bridge_row,
            jump_variance=params.jump_variance,
            tau_alpha_relative_error_by_k={wave_number: 0.03 for wave_number in wave_numbers},
            late_ngp_relative_error=0.05,
            chi4_peak_relative_error=0.05,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=3.0,
        )

        self.assertEqual(gate["gate_stage"], "trajectory_persistence_exchange_protocol_ready")
        self.assertEqual(gate["trajectory_pe_protocol_ready"], 1.0)
        self.assertEqual(gate["primary_blocker"], "none")
        self.assertEqual(gate["passes_uncertainty_protocol"], 1.0)
        self.assertAlmostEqual(gate["persistence_exchange_ratio"], 7.0, delta=0.1)
        self.assertLess(gate["max_multik_tau_alpha_z"], 1.0)
        self.assertLess(gate["late_ngp_z"], 1.0)
        self.assertLess(gate["chi4_peak_z"], 1.0)

    def test_trajectory_curve_persistence_exchange_gate_blocks_incomplete_bridge(self):
        bridge_row = {
            "benchmark_id": "short_bridge",
            "bridge_stage": "trajectory_curve_bridge_incomplete",
            "curve_bridge_ready": 0.0,
            "primary_blocker": "alpha_threshold_crossing",
        }

        gate = trajectory_curve_persistence_exchange_gate(
            bridge_row=bridge_row,
            jump_variance=0.7,
            tau_alpha_relative_error_by_k={1.1: 0.03},
            late_ngp_relative_error=0.05,
            chi4_peak_relative_error=0.05,
        )

        self.assertEqual(gate["gate_stage"], "trajectory_curve_bridge_incomplete")
        self.assertEqual(gate["trajectory_pe_protocol_ready"], 0.0)
        self.assertEqual(gate["primary_blocker"], "alpha_threshold_crossing")

    def test_trajectory_pe_heldout_prediction_gate_scores_unfitted_observables(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=7.0,
            exchange_mean=1.0,
        )
        heldout_wave_number = 1.35
        heldout_time = 120.0 * params.persistence_mean
        pe_gate_row = {
            "benchmark_id": "synthetic_bridge_pe_protocol_ready",
            "gate_stage": "trajectory_persistence_exchange_protocol_ready",
            "trajectory_pe_protocol_ready": 1.0,
            "exchange_mean": params.exchange_mean,
            "persistence_mean": params.persistence_mean,
        }

        gate = trajectory_pe_heldout_prediction_gate(
            pe_gate_row=pe_gate_row,
            jump_variance=params.jump_variance,
            heldout_wave_number=heldout_wave_number,
            observed_heldout_tau_alpha=persistence_exchange_alpha_relaxation_time(
                heldout_wave_number,
                params,
            ),
            heldout_tau_alpha_relative_error=0.03,
            heldout_late_time=heldout_time,
            observed_heldout_late_ngp=float(
                persistence_exchange_ngp_1d(np.array([heldout_time]), params)[0]
            ),
            heldout_late_ngp_relative_error=0.05,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=3.0,
        )

        self.assertEqual(gate["prediction_stage"], "trajectory_pe_heldout_prediction_ready")
        self.assertEqual(gate["heldout_prediction_ready"], 1.0)
        self.assertEqual(gate["primary_blocker"], "none")
        self.assertEqual(gate["heldout_predictions_pass"], 1.0)
        self.assertLess(gate["heldout_tau_alpha_z"], 1.0)
        self.assertLess(gate["heldout_late_ngp_z"], 1.0)
        self.assertGreater(gate["predicted_heldout_tau_alpha"], 0.0)
        self.assertGreater(gate["predicted_heldout_late_ngp"], 0.0)

    def test_trajectory_pe_heldout_prediction_gate_blocks_incomplete_pe_gate(self):
        gate = trajectory_pe_heldout_prediction_gate(
            pe_gate_row={
                "benchmark_id": "short_bridge",
                "gate_stage": "trajectory_curve_bridge_incomplete",
                "trajectory_pe_protocol_ready": 0.0,
                "primary_blocker": "alpha_threshold_crossing",
            },
            jump_variance=0.7,
            heldout_wave_number=1.35,
            observed_heldout_tau_alpha=4.0,
            heldout_tau_alpha_relative_error=0.03,
            heldout_late_time=20.0,
            observed_heldout_late_ngp=0.02,
            heldout_late_ngp_relative_error=0.05,
        )

        self.assertEqual(gate["prediction_stage"], "trajectory_pe_gate_incomplete")
        self.assertEqual(gate["heldout_prediction_ready"], 0.0)
        self.assertEqual(gate["primary_blocker"], "alpha_threshold_crossing")

    def test_trajectory_prediction_falsification_gate_accepts_passed_heldouts(self):
        row = trajectory_prediction_falsification_gate(
            protocol_id="synthetic_trajectory_pe_heldout_protocol",
            prediction_row={
                "benchmark_id": "synthetic_bridge_pe_protocol_ready",
                "heldout_prediction_ready": 1.0,
                "heldout_tau_alpha_pass": 1.0,
                "heldout_late_ngp_pass": 1.0,
                "heldout_predictions_pass": 1.0,
                "primary_blocker": "none",
            },
            calibration_observables=[
                "tau_alpha(k=0.7)",
                "tau_alpha(k=1.1)",
                "tau_alpha(k=1.6)",
                "late_ngp",
                "chi4_peak",
            ],
            heldout_observables=["tau_alpha(k=1.35)", "late_ngp(t=120tau_p)"],
            required_prediction_passes=["heldout_tau_alpha_pass", "heldout_late_ngp_pass"],
        )

        self.assertEqual(row["falsification_stage"], "trajectory_prediction_falsification_passed")
        self.assertEqual(row["trajectory_falsification_ready"], 1.0)
        self.assertEqual(row["trajectory_predictions_falsified"], 0.0)
        self.assertEqual(row["all_required_predictions_pass"], 1.0)
        self.assertEqual(row["fit_only_overclaim_risk"], 0.0)
        self.assertEqual(row["primary_blocker"], "none")
        self.assertEqual(row["calibration_count"], 5.0)
        self.assertEqual(row["heldout_count"], 2.0)
        self.assertIn("late_ngp(t=120tau_p)", str(row["heldout_observables"]))

    def test_trajectory_prediction_falsification_gate_blocks_upstream_incomplete(self):
        row = trajectory_prediction_falsification_gate(
            protocol_id="short_trajectory_upstream_blocker",
            prediction_row={
                "benchmark_id": "synthetic_short_csv_bridge",
                "heldout_prediction_ready": 0.0,
                "heldout_predictions_pass": 0.0,
                "primary_blocker": "alpha_threshold_crossing",
            },
            calibration_observables=["tau_alpha(k=0.7)", "late_ngp"],
            heldout_observables=["tau_alpha(k=1.35)", "late_ngp(t=120tau_p)"],
            required_prediction_passes=["heldout_tau_alpha_pass", "heldout_late_ngp_pass"],
        )

        self.assertEqual(row["falsification_stage"], "upstream_prediction_incomplete")
        self.assertEqual(row["trajectory_falsification_ready"], 0.0)
        self.assertEqual(row["trajectory_predictions_falsified"], 0.0)
        self.assertEqual(row["primary_blocker"], "alpha_threshold_crossing")

    def test_trajectory_prediction_falsification_gate_flags_fit_only_overclaim(self):
        row = trajectory_prediction_falsification_gate(
            protocol_id="fit_only_negative_control",
            prediction_row={
                "benchmark_id": "synthetic_bridge_pe_protocol_ready",
                "heldout_prediction_ready": 1.0,
                "heldout_tau_alpha_pass": 1.0,
                "heldout_late_ngp_pass": 1.0,
                "heldout_predictions_pass": 1.0,
                "primary_blocker": "none",
            },
            calibration_observables=["tau_alpha(k=0.7)", "late_ngp"],
            heldout_observables=[],
            required_prediction_passes=["heldout_tau_alpha_pass"],
        )

        self.assertEqual(row["falsification_stage"], "fit_only_overclaim_risk")
        self.assertEqual(row["trajectory_falsification_ready"], 0.0)
        self.assertEqual(row["fit_only_overclaim_risk"], 1.0)
        self.assertEqual(row["primary_blocker"], "heldout_observables")

    def test_benchmark_publication_ladder_separates_metadata_canary_and_overclaim(self):
        glassbench = benchmark_publication_ladder(
            ladder_id="glassbench_current_publication_state",
            source_key="glassbench_zenodo_10118191",
            source_class="real_public_data",
            evidence_grade="pending_trajectory_reanalysis",
            reanalysis_stage="awaiting_full_archive_cache",
            readiness_stage="none",
            falsification_stage="none",
            primary_blocker="archive_path",
        )
        self.assertEqual(glassbench["publication_stage"], "metadata_verified_not_reanalysis")
        self.assertEqual(glassbench["allowed_manuscript_claim"], "metadata_readiness_only")
        self.assertEqual(float(glassbench["real_data_quantitative_comparison"]), 0.0)
        self.assertEqual(float(glassbench["claim_overreach_if_called_fit"]), 1.0)
        self.assertEqual(glassbench["next_required_action"], "cache_full_archive_and_verify_checksum")

        canary = benchmark_publication_ladder(
            ladder_id="synthetic_trajectory_canary",
            source_key="synthetic_intermittent_trajectory",
            source_class="synthetic_canary",
            evidence_grade="direct_dynamical_support",
            reanalysis_stage="ready_for_trajectory_observable_protocol",
            readiness_stage="uncertainty_weighted_trajectory_inversion",
            falsification_stage="trajectory_prediction_falsification_passed",
            primary_blocker="none",
        )
        self.assertEqual(canary["publication_stage"], "synthetic_prediction_canary_passed")
        self.assertEqual(canary["allowed_manuscript_claim"], "protocol_canary_passed")
        self.assertEqual(float(canary["publishable_protocol_evidence"]), 1.0)
        self.assertEqual(float(canary["real_data_quantitative_comparison"]), 0.0)

        fit_only = benchmark_publication_ladder(
            ladder_id="fit_only_negative_control",
            source_key="synthetic_intermittent_trajectory",
            source_class="synthetic_canary",
            evidence_grade="direct_dynamical_support",
            reanalysis_stage="ready_for_trajectory_observable_protocol",
            readiness_stage="uncertainty_weighted_trajectory_inversion",
            falsification_stage="fit_only_overclaim_risk",
            primary_blocker="heldout_observables",
        )
        self.assertEqual(fit_only["publication_stage"], "fit_only_overclaim_blocked")
        self.assertEqual(fit_only["allowed_manuscript_claim"], "do_not_claim_prediction")
        self.assertEqual(float(fit_only["publishable_protocol_evidence"]), 0.0)
        self.assertEqual(float(fit_only["claim_overreach_if_called_fit"]), 1.0)

    def test_trajectory_table_adapter_orders_frames_particles_and_extracts_arrays(self):
        records = [
            {"frame": 1, "time": 1.0, "particle_id": "b", "x": 2.0, "y": 0.0},
            {"frame": 0, "time": 0.0, "particle_id": "a", "x": 0.0, "y": 0.0},
            {"frame": 1, "time": 1.0, "particle_id": "a", "x": 1.0, "y": 0.0},
            {"frame": 0, "time": 0.0, "particle_id": "b", "x": 0.0, "y": 1.0},
        ]

        adapted = trajectory_table_adapter(
            records=records,
            frame_column="frame",
            time_column="time",
            particle_column="particle_id",
            coordinate_columns=["x", "y"],
        )

        self.assertEqual(adapted["frame_ids"], "0;1")
        self.assertEqual(adapted["particle_ids"], "a;b")
        self.assertEqual(adapted["frame_count"], 2.0)
        self.assertEqual(adapted["particle_count"], 2.0)
        self.assertEqual(adapted["dimension"], 2.0)
        np.testing.assert_allclose(adapted["times"], np.array([0.0, 1.0]))
        np.testing.assert_allclose(
            adapted["positions"],
            np.array(
                [
                    [[0.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [2.0, 0.0]],
                ]
            ),
        )

    def test_trajectory_table_adapter_rejects_missing_frame_particle_rows(self):
        records = [
            {"frame": 0, "time": 0.0, "particle_id": "a", "x": 0.0},
            {"frame": 0, "time": 0.0, "particle_id": "b", "x": 1.0},
            {"frame": 1, "time": 1.0, "particle_id": "a", "x": 2.0},
        ]

        with self.assertRaisesRegex(ValueError, "complete rectangular"):
            trajectory_table_adapter(
                records=records,
                frame_column="frame",
                time_column="time",
                particle_column="particle_id",
                coordinate_columns=["x"],
            )

    def test_trajectory_table_csv_adapter_loads_file_and_metadata(self):
        rows = [
            {"frame": 1, "time": 1.0, "particle_id": "b", "x": 2.0, "y": 0.0},
            {"frame": 0, "time": 0.0, "particle_id": "a", "x": 0.0, "y": 0.0},
            {"frame": 1, "time": 1.0, "particle_id": "a", "x": 1.0, "y": 0.0},
            {"frame": 0, "time": 0.0, "particle_id": "b", "x": 0.0, "y": 1.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["frame", "time", "particle_id", "x", "y"])
                writer.writeheader()
                writer.writerows(rows)

            adapted = trajectory_table_csv_adapter(
                csv_path=path,
                frame_column="frame",
                time_column="time",
                particle_column="particle_id",
                coordinate_columns=["x", "y"],
                metadata={
                    "box_geometry": "orthorhombic",
                    "temperature_or_state_point": "synthetic_T_0.45",
                    "species_labels": "A;B",
                    "units_metadata": "reduced_LJ_units",
                },
            )

        self.assertEqual(adapted["adapter_stage"], "local_csv_trajectory_ready")
        self.assertEqual(adapted["adapter_ready"], 1.0)
        self.assertEqual(adapted["row_count"], 4.0)
        self.assertEqual(adapted["missing_metadata_fields"], "none")
        self.assertEqual(adapted["primary_blocker"], "none")
        np.testing.assert_allclose(adapted["times"], np.array([0.0, 1.0]))
        np.testing.assert_allclose(adapted["positions"][1, 1], np.array([2.0, 0.0]))

    def test_trajectory_table_csv_adapter_blocks_missing_units_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["frame", "time", "particle_id", "x"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"frame": 0, "time": 0.0, "particle_id": "a", "x": 0.0},
                        {"frame": 1, "time": 1.0, "particle_id": "a", "x": 1.0},
                    ]
                )

            adapted = trajectory_table_csv_adapter(
                csv_path=path,
                frame_column="frame",
                time_column="time",
                particle_column="particle_id",
                coordinate_columns=["x"],
                metadata={
                    "box_geometry": "orthorhombic",
                    "temperature_or_state_point": "synthetic_T_0.45",
                    "species_labels": "A",
                },
            )

        self.assertEqual(adapted["adapter_stage"], "metadata_incomplete_csv_adapter")
        self.assertEqual(adapted["adapter_ready"], 0.0)
        self.assertEqual(adapted["primary_blocker"], "units_metadata")
        self.assertIn("units_metadata", adapted["missing_metadata_fields"])

    def test_trajectory_observable_uncertainty_protocol_adds_jackknife_sigmas(self):
        times = np.arange(6.0)
        increments = np.array(
            [
                [[0.0], [0.0], [2.0], [2.0]],
                [[0.0], [2.0], [0.0], [2.0]],
                [[0.0], [0.0], [0.0], [2.0]],
                [[2.0], [0.0], [2.0], [0.0]],
                [[0.0], [2.0], [0.0], [0.0]],
            ]
        )
        positions = np.concatenate([np.zeros((1, 4, 1)), np.cumsum(increments, axis=0)])

        full = trajectory_observable_protocol(
            positions=positions,
            times=times,
            lag_indices=[1],
            wave_numbers=[1.1],
            overlap_radius=0.5,
        )[0]
        uncertain = trajectory_observable_uncertainty_protocol(
            positions=positions,
            times=times,
            lag_indices=[1],
            wave_numbers=[1.1],
            overlap_radius=0.5,
            block_count=3,
        )[0]

        self.assertAlmostEqual(uncertain["msd"], full["msd"])
        self.assertAlmostEqual(uncertain["ngp"], full["ngp"])
        self.assertAlmostEqual(
            uncertain["self_intermediate_scattering"],
            full["self_intermediate_scattering"],
        )
        self.assertEqual(uncertain["uncertainty_method"], "time_origin_block_jackknife")
        self.assertEqual(uncertain["uncertainty_estimates"], 1.0)
        self.assertEqual(uncertain["primary_blocker"], "none")
        self.assertGreater(uncertain["sigma_msd"], 0.0)
        self.assertGreater(uncertain["sigma_ngp"], 0.0)
        self.assertGreater(uncertain["sigma_self_intermediate_scattering"], 0.0)
        self.assertGreaterEqual(uncertain["sigma_chi4_overlap"], 0.0)

    def test_trajectory_observable_uncertainty_protocol_requires_multiple_blocks(self):
        with self.assertRaises(ValueError):
            trajectory_observable_uncertainty_protocol(
                positions=np.zeros((3, 2, 1)),
                times=np.array([0.0, 1.0, 2.0]),
                lag_indices=[1],
                wave_numbers=[1.0],
                overlap_radius=0.5,
                block_count=1,
            )

    def test_trajectory_inversion_readiness_gate_promotes_uncertainty_weighted_rows(self):
        rows = [
            {
                "lag_time": 1.0,
                "structural_observable_set": "msd;ngp;self_intermediate_scattering;overlap_chi4",
                "msd": 1.0,
                "ngp": 0.2,
                "self_intermediate_scattering": 0.5,
                "chi4_overlap": 0.3,
                "sigma_msd": 0.1,
                "sigma_ngp": 0.02,
                "sigma_self_intermediate_scattering": 0.03,
                "sigma_chi4_overlap": 0.04,
            },
            {
                "lag_time": 2.0,
                "structural_observable_set": "msd;ngp;self_intermediate_scattering;overlap_chi4",
                "msd": 2.0,
                "ngp": 0.1,
                "self_intermediate_scattering": 0.2,
                "chi4_overlap": 0.4,
                "sigma_msd": 0.2,
                "sigma_ngp": 0.03,
                "sigma_self_intermediate_scattering": 0.04,
                "sigma_chi4_overlap": 0.05,
            },
        ]

        gate = trajectory_inversion_readiness_gate(
            benchmark_id="glassbench_like_trajectory",
            source_key="trajectory_reanalysis_candidate",
            target_protocol="alpha_vanhove_chi4_transport",
            trajectory_rows=rows,
            required_observables=["msd", "ngp", "self_intermediate_scattering", "overlap_chi4"],
            required_uncertainty_columns=[
                "sigma_msd",
                "sigma_ngp",
                "sigma_self_intermediate_scattering",
                "sigma_chi4_overlap",
            ],
            has_shared_time_grid=True,
            has_shared_particle_identity=True,
        )

        self.assertEqual(gate["readiness_stage"], "uncertainty_weighted_trajectory_inversion")
        self.assertEqual(gate["primary_blocker"], "none")
        self.assertEqual(gate["structural_trajectory_ready"], 1.0)
        self.assertEqual(gate["uncertainty_weighted_ready"], 1.0)
        self.assertEqual(gate["lag_count"], 2.0)

    def test_trajectory_inversion_readiness_gate_blocks_missing_uncertainty_columns(self):
        rows = [
            {
                "lag_time": 1.0,
                "structural_observable_set": "msd;ngp;self_intermediate_scattering;overlap_chi4",
                "msd": 1.0,
                "ngp": 0.2,
                "self_intermediate_scattering": 0.5,
                "chi4_overlap": 0.3,
                "sigma_msd": 0.1,
            }
        ]

        gate = trajectory_inversion_readiness_gate(
            benchmark_id="structural_only_trajectory",
            source_key="trajectory_reanalysis_candidate",
            target_protocol="alpha_vanhove_chi4_transport",
            trajectory_rows=rows,
            required_observables=["msd", "ngp", "self_intermediate_scattering", "overlap_chi4"],
            required_uncertainty_columns=["sigma_msd", "sigma_ngp"],
            has_shared_time_grid=True,
            has_shared_particle_identity=True,
        )

        self.assertEqual(gate["readiness_stage"], "structural_trajectory_only")
        self.assertEqual(gate["primary_blocker"], "sigma_ngp")
        self.assertEqual(gate["structural_trajectory_ready"], 1.0)
        self.assertEqual(gate["uncertainty_weighted_ready"], 0.0)

    def test_trajectory_member_ensemble_uncertainty_protocol_adds_member_sigmas(self):
        member_rows = []
        for member_index in range(4):
            for lag_time, msd, ngp, fs_values, chi4 in [
                (1.0, 1.0, 0.62, [0.82, 0.70], 0.4),
                (2.0, 3.8, 0.31, [0.44, 0.28], 1.5),
                (4.0, 8.4, 0.11, [0.20, 0.08], 0.7),
            ]:
                scale = 1.0 + 0.04 * member_index
                member_rows.append(
                    {
                        "member_id": f"traj_{member_index}",
                        "lag_index": lag_time,
                        "lag_time": lag_time,
                        "time_origin_count": 8.0,
                        "particle_count": 64.0,
                        "dimension": 2.0,
                        "msd": msd * scale,
                        "ngp": ngp / scale,
                        "wave_numbers": "0.7;1.1",
                        "self_intermediate_scattering_by_k": ";".join(
                            f"{value / scale:.12g}" for value in fs_values
                        ),
                        "self_intermediate_scattering": fs_values[0] / scale,
                        "overlap_radius": 0.5,
                        "overlap_mean": 0.3 / scale,
                        "chi4_overlap": chi4 * scale,
                    }
                )

        rows = trajectory_member_ensemble_uncertainty_protocol(
            member_rows=member_rows,
            min_member_count=4,
        )

        self.assertEqual(len(rows), 3)
        middle = rows[1]
        self.assertEqual(middle["member_count"], 4.0)
        self.assertEqual(middle["ensemble_uncertainty_ready"], 1.0)
        self.assertEqual(middle["primary_blocker"], "none")
        self.assertGreater(middle["sigma_msd"], 0.0)
        self.assertGreater(middle["sigma_ngp"], 0.0)
        self.assertGreater(middle["sigma_self_intermediate_scattering"], 0.0)
        self.assertGreater(middle["sigma_chi4_overlap"], 0.0)
        self.assertEqual(middle["uncertainty_method"], "member_ensemble_standard_error")
        self.assertEqual(middle["ensemble_stage"], "member_ensemble_uncertainty_ready")
        self.assertIn(";", middle["sigma_self_intermediate_scattering_by_k"])

    def test_trajectory_member_ensemble_uncertainty_protocol_blocks_small_ensemble(self):
        member_rows = [
            {
                "member_id": f"traj_{member_index}",
                "lag_index": 1.0,
                "lag_time": 1.0,
                "time_origin_count": 8.0,
                "particle_count": 64.0,
                "dimension": 2.0,
                "msd": 1.0 + 0.1 * member_index,
                "ngp": 0.5,
                "wave_numbers": "0.7",
                "self_intermediate_scattering_by_k": f"{0.8 - 0.05 * member_index:.12g}",
                "self_intermediate_scattering": 0.8 - 0.05 * member_index,
                "overlap_radius": 0.5,
                "overlap_mean": 0.3,
                "chi4_overlap": 0.4,
            }
            for member_index in range(3)
        ]

        rows = trajectory_member_ensemble_uncertainty_protocol(
            member_rows=member_rows,
            min_member_count=4,
        )

        self.assertEqual(rows[0]["member_count"], 3.0)
        self.assertEqual(rows[0]["ensemble_uncertainty_ready"], 0.0)
        self.assertEqual(rows[0]["primary_blocker"], "member_count")
        self.assertEqual(rows[0]["ensemble_stage"], "member_ensemble_below_threshold")

    def test_van_hove_tail_benchmark_consistency_detects_transient_tail_and_recovery(self):
        row = van_hove_tail_benchmark_consistency(
            benchmark_id="kob_andersen_van_hove_tail_recovery",
            observed_transient_van_hove_tail=True,
            observed_late_gaussian_recovery=True,
            peak_tail_ratio=2.895,
            late_tail_ratio=0.966,
            peak_ngp=0.126,
            min_peak_tail_ratio=1.5,
            max_late_tail_deviation=0.15,
            min_peak_ngp=0.05,
        )

        self.assertEqual(row["model_predicts_transient_van_hove_tail"], 1.0)
        self.assertEqual(row["model_predicts_tail_gaussian_recovery"], 1.0)
        self.assertEqual(row["van_hove_tail_consistent"], 1.0)
        self.assertEqual(row["tail_recovery_consistent"], 1.0)
        self.assertEqual(row["peak_ngp_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["peak_tail_ratio"], row["min_peak_tail_ratio"])
        self.assertLess(row["late_tail_abs_deviation"], row["max_late_tail_deviation"])

    def test_fragility_benchmark_consistency_checks_super_arrhenius_growth_without_origin_claim(self):
        row = fragility_benchmark_consistency(
            benchmark_id="angell_adam_gibbs_fragility_growth",
            observed_fragility_growth=True,
            observed_adam_gibbs_slowdown=True,
            hot_activation_energy=2.69,
            cold_activation_energy=3.43,
            hot_fragility_index=1.17,
            cold_fragility_index=2.41,
            adam_gibbs_slowdown=1.48e8,
            material_specific_origin_claimed=False,
            min_activation_growth=1.2,
            min_fragility_growth=1.5,
            min_adam_gibbs_slowdown=10.0,
        )

        self.assertEqual(row["model_predicts_fragility_growth"], 1.0)
        self.assertEqual(row["activation_growth_consistent"], 1.0)
        self.assertEqual(row["fragility_index_consistent"], 1.0)
        self.assertEqual(row["adam_gibbs_slowdown_consistent"], 1.0)
        self.assertEqual(row["fragility_scope_boundary_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)
        self.assertGreater(row["activation_energy_growth"], row["min_activation_growth"])
        self.assertGreater(row["fragility_index_growth"], row["min_fragility_growth"])

    def test_thermodynamic_scope_benchmark_consistency_keeps_entropy_closure_boundary(self):
        row = thermodynamic_scope_benchmark_consistency(
            benchmark_id="thermodynamic_transition_scope_boundary",
            observed_heat_capacity_anomaly=True,
            observed_kauzmann_extrapolation=True,
            dynamic_model_derives_entropy=False,
            entropy_closure_supplied=True,
            adam_gibbs_slowdown=1.48e8,
            min_adam_gibbs_slowdown=10.0,
            material_specific_entropy_origin_claimed=False,
        )

        self.assertEqual(row["model_predicts_heat_capacity_anomaly_from_dynamics"], 0.0)
        self.assertEqual(row["model_predicts_kauzmann_transition_from_dynamics"], 0.0)
        self.assertEqual(row["entropy_closure_required"], 1.0)
        self.assertEqual(row["adam_gibbs_slowdown_consistent"], 1.0)
        self.assertEqual(row["thermodynamic_scope_boundary_consistent"], 1.0)
        self.assertEqual(row["overall_consistent"], 1.0)

    def test_facilitated_exchange_law_grows_exchange_ratio_on_cooling(self):
        law = FacilitatedExchangeLawParams(
            reference_temperature=1.0,
            shape_ref=0.4,
            exchange_renewal_count_ref=10.0,
            shape_broadening_barrier=1.5,
            exchange_slowing_barrier=2.5,
        )

        hot = temperature_dependent_gamma_exchange(1.0, law)
        cold_temperature = 0.62
        cold = temperature_dependent_gamma_exchange(cold_temperature, law)
        delta = 1.0 / cold_temperature - 1.0 / law.reference_temperature
        expected_ratio_growth = math.exp(
            (law.shape_broadening_barrier + law.exchange_slowing_barrier) * delta
        )

        self.assertAlmostEqual(hot.shape, 0.4)
        self.assertAlmostEqual(hot.exchange_renewal_count, 10.0)
        self.assertLess(cold.shape, hot.shape)
        self.assertGreater(cold.exchange_renewal_count, hot.exchange_renewal_count)
        self.assertAlmostEqual(
            (cold.exchange_renewal_count / cold.shape) / (hot.exchange_renewal_count / hot.shape),
            expected_ratio_growth,
            delta=1e-12,
        )

    def test_gamma_exchange_temperature_scan_links_cooling_to_alpha_slowing(self):
        cage_law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=5.0,
            cage_stiffening=0.2,
            jump_to_cage_growth=0.25,
        )
        exchange_law = FacilitatedExchangeLawParams(
            reference_temperature=1.0,
            shape_ref=0.4,
            exchange_renewal_count_ref=10.0,
            shape_broadening_barrier=1.5,
            exchange_slowing_barrier=2.5,
        )
        temperatures = np.array([1.0, 0.78, 0.62])

        rows = gamma_exchange_temperature_scan(
            temperatures,
            cage_law,
            exchange_law,
            wave_number=1.1,
        )

        ratios = np.array([row["heterogeneity_ratio"] for row in rows])
        amplitudes = np.array([row["late_ngp_renewal_amplitude"] for row in rows])
        alpha_renormalization = np.array([row["alpha_rate_renormalization"] for row in rows])

        self.assertTrue(np.all(np.diff(ratios) > 0.0))
        self.assertTrue(np.all(np.diff(amplitudes) > 0.0))
        self.assertTrue(np.all(np.diff(alpha_renormalization) < 0.0))
        self.assertGreater(rows[-1]["late_ngp_renewal_amplitude"], 50.0)
        self.assertLess(rows[-1]["alpha_rate_renormalization"], rows[0]["alpha_rate_renormalization"] / 2.0)

    def test_gamma_exchange_alpha_relaxation_time_solves_finite_exchange_decay(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        heterogeneity = GammaExchangeParams(shape=0.4, exchange_renewal_count=10.0)
        wave_number = 1.1

        tau_exchange = gamma_exchange_alpha_relaxation_time(wave_number, params, heterogeneity)
        tau_poisson = alpha_relaxation_time(wave_number, params)
        decay = gamma_exchange_normalized_alpha_decay(
            wave_number,
            np.array([tau_exchange]),
            params,
            heterogeneity,
        )[0]

        self.assertGreater(tau_exchange, tau_poisson)
        self.assertAlmostEqual(decay, math.exp(-1.0), delta=1e-10)

    def test_glass_phenomenon_audit_separates_supported_dynamics_from_thermodynamic_transition(self):
        cage_law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=5.0,
            cage_stiffening=0.2,
            jump_to_cage_growth=0.25,
        )
        exchange_law = FacilitatedExchangeLawParams(
            reference_temperature=1.0,
            shape_ref=0.4,
            exchange_renewal_count_ref=10.0,
            shape_broadening_barrier=1.5,
            exchange_slowing_barrier=2.5,
        )
        audit = glass_phenomenon_audit(
            np.array([1.0, 0.85, 0.72, 0.62]),
            cage_law,
            exchange_law,
            wave_number=1.1,
        )
        rows = audit["rows"]

        self.assertEqual(audit["diffusion_slowdown"], 1.0)
        self.assertEqual(audit["alpha_slowdown"], 1.0)
        self.assertEqual(audit["ngp_peak_shift"], 1.0)
        self.assertEqual(audit["stokes_einstein_violation"], 1.0)
        self.assertEqual(audit["fragility_growth"], 1.0)
        self.assertEqual(audit["heterogeneity_growth"], 1.0)
        self.assertEqual(audit["stretched_alpha_window"], 1.0)
        self.assertEqual(audit["chi4_peak_growth"], 1.0)
        self.assertEqual(audit["gaussian_recovery"], 1.0)
        self.assertEqual(audit["thermodynamic_transition"], 0.0)
        self.assertGreater(audit["supported_dynamic_signatures"], 8.0)
        self.assertGreater(rows[-1]["tau_alpha_exchange"] / rows[0]["tau_alpha_exchange"], 10.0)
        self.assertGreater(rows[-1]["chi4_peak"] / rows[0]["chi4_peak"], 1.0)

    def test_glass_signature_phase_diagram_identifies_barrier_facilitation_window(self):
        base_cage_law = TemperatureLawParams(
            reference_temperature=1.0,
            cage_variance_ref=1.0,
            cage_tau_ref=0.7,
            jump_to_cage_ref=0.8,
            renewal_rate_ref=0.18,
            renewal_delay_ref=3.0,
            rate_activation=2.0,
            delay_activation=2.0,
            cage_stiffening=0.2,
            jump_to_cage_growth=0.25,
        )
        base_exchange_law = FacilitatedExchangeLawParams(
            reference_temperature=1.0,
            shape_ref=0.4,
            exchange_renewal_count_ref=10.0,
            shape_broadening_barrier=1.5,
            exchange_slowing_barrier=2.5,
        )

        rows = glass_signature_phase_diagram(
            np.array([1.0, 0.85, 0.72, 0.62]),
            base_cage_law,
            base_exchange_law,
            wave_number=1.1,
            delay_barrier_gaps=[0.0, 1.5, 3.0],
            exchange_barrier_sums=[0.0, 2.0, 4.0],
        )

        self.assertEqual(len(rows), 9)
        weak = next(row for row in rows if row["delay_barrier_gap"] == 0.0 and row["exchange_barrier_sum"] == 0.0)
        strong = next(row for row in rows if row["delay_barrier_gap"] == 3.0 and row["exchange_barrier_sum"] == 4.0)
        same_exchange_stronger_delay = next(
            row for row in rows if row["delay_barrier_gap"] == 3.0 and row["exchange_barrier_sum"] == 0.0
        )
        same_delay_stronger_exchange = next(
            row for row in rows if row["delay_barrier_gap"] == 0.0 and row["exchange_barrier_sum"] == 4.0
        )

        self.assertEqual(weak["complete_dynamic_closure"], 0.0)
        self.assertEqual(strong["complete_dynamic_closure"], 1.0)
        self.assertEqual(strong["thermodynamic_transition"], 0.0)
        self.assertGreater(strong["supported_dynamic_signatures"], weak["supported_dynamic_signatures"])
        self.assertGreater(same_exchange_stronger_delay["cold_se_product_ratio"], weak["cold_se_product_ratio"])
        self.assertGreater(
            same_delay_stronger_exchange["cold_heterogeneity_growth_ratio"],
            weak["cold_heterogeneity_growth_ratio"],
        )

    def test_barrier_amplification_laws_give_closed_cooling_factors(self):
        laws = barrier_amplification_laws(
            hot_temperature=1.0,
            cold_temperature=0.62,
            delay_barrier_gap=3.0,
            exchange_barrier_sum=4.0,
        )
        delta = 1.0 / 0.62 - 1.0

        self.assertAlmostEqual(laws["inverse_temperature_interval"], delta)
        self.assertAlmostEqual(laws["lambda_tau_delay_growth"], math.exp(3.0 * delta))
        self.assertAlmostEqual(laws["heterogeneity_ratio_growth"], math.exp(4.0 * delta))
        self.assertAlmostEqual(laws["combined_slowing_growth"], math.exp(7.0 * delta))

    def test_minimal_barrier_requirements_invert_target_growth_factors(self):
        requirements = minimal_barrier_requirements(
            hot_temperature=1.0,
            cold_temperature=0.62,
            target_lambda_tau_delay_growth=math.exp(3.0 * (1.0 / 0.62 - 1.0)),
            target_heterogeneity_ratio_growth=math.exp(4.0 * (1.0 / 0.62 - 1.0)),
        )

        self.assertAlmostEqual(requirements["required_delay_barrier_gap"], 3.0)
        self.assertAlmostEqual(requirements["required_exchange_barrier_sum"], 4.0)
        self.assertAlmostEqual(requirements["target_combined_growth"], requirements["target_lambda_tau_delay_growth"] * requirements["target_heterogeneity_ratio_growth"])

    def test_minimal_barrier_requirements_validate_temperature_order(self):
        with self.assertRaises(ValueError):
            minimal_barrier_requirements(
                hot_temperature=0.62,
                cold_temperature=1.0,
                target_lambda_tau_delay_growth=2.0,
                target_heterogeneity_ratio_growth=3.0,
            )

    def test_fractional_stokes_einstein_exponents_recover_power_law_slope(self):
        tau_alpha = np.array([2.0, 4.0, 8.0, 16.0, 32.0])
        diffusion = 3.0 * tau_alpha ** (-0.72)
        exponents = fractional_stokes_einstein_exponents(diffusion, tau_alpha)

        np.testing.assert_allclose(exponents, 0.72, rtol=1e-12, atol=1e-12)

    def test_fractional_stokes_einstein_exponents_validate_inputs(self):
        with self.assertRaises(ValueError):
            fractional_stokes_einstein_exponents(np.array([1.0]), np.array([2.0]))
        with self.assertRaises(ValueError):
            fractional_stokes_einstein_exponents(np.array([1.0, -1.0]), np.array([2.0, 3.0]))

    def test_apparent_alpha_activation_energies_recover_arrhenius_barrier(self):
        temperatures = np.array([1.0, 0.9, 0.8, 0.7, 0.62])
        barrier = 4.2
        tau_alpha = 0.3 * np.exp(barrier / temperatures)
        energies = apparent_alpha_activation_energies(temperatures, tau_alpha)

        np.testing.assert_allclose(energies, barrier, rtol=1e-12, atol=1e-12)

    def test_apparent_alpha_activation_energies_validate_inputs(self):
        with self.assertRaises(ValueError):
            apparent_alpha_activation_energies(np.array([1.0]), np.array([2.0]))
        with self.assertRaises(ValueError):
            apparent_alpha_activation_energies(np.array([1.0, 0.0]), np.array([2.0, 3.0]))

    def test_radial_van_hove_distribution_normalizes(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        radius = np.linspace(0.0, 12.0, 4000)
        density = radial_van_hove_3d(radius, time=12.0, params=params, max_count=80)
        integral = np.trapezoid(density, radius)

        self.assertAlmostEqual(integral, 1.0, delta=2e-4)
        self.assertGreater(density[1], 0.0)

    def test_gaussian_radial_baseline_normalizes(self):
        radius = np.linspace(0.0, 12.0, 4000)
        density = gaussian_radial_3d(radius, coordinate_variance=2.5)
        integral = np.trapezoid(density, radius)

        self.assertAlmostEqual(integral, 1.0, delta=2e-4)

    def test_renewal_van_hove_has_heavier_peak_time_tail_than_gaussian_baseline(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        radius = np.linspace(0.0, 18.0, 5000)
        peak_time = 11.30637498610339
        renewal_density = radial_van_hove_3d(radius, time=peak_time, params=params, max_count=120)
        coordinate_variance = moments_1d(np.array([peak_time]), params)["m2"][0]
        gaussian_density = gaussian_radial_3d(radius, coordinate_variance=coordinate_variance)
        tail_mask = radius > 5.0
        renewal_tail = np.trapezoid(renewal_density[tail_mask], radius[tail_mask])
        gaussian_tail = np.trapezoid(gaussian_density[tail_mask], radius[tail_mask])

        self.assertGreater(renewal_tail / gaussian_tail, 1.5)

    def test_dimensionless_peak_prediction_matches_numeric_peak_in_plateau_regime(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        t = np.linspace(0.01, 180.0, 5000)
        alpha = ngp_1d(t, params)
        peak_time = float(t[int(np.argmax(alpha))])
        predicted = dimensionless_peak_prediction(params)

        self.assertAlmostEqual(peak_time, predicted["peak_time"], delta=0.2)
        self.assertAlmostEqual(float(np.max(alpha)), predicted["peak_ngp"], delta=0.005)

    def test_plateau_peak_diagnostics_recover_peak_scale_parameters(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        predicted = dimensionless_peak_prediction(params)
        diagnostics = plateau_peak_diagnostics(
            peak_ngp=predicted["peak_ngp"],
            peak_time=predicted["peak_time"],
            renewal_delay=params.renewal_delay,
        )

        self.assertAlmostEqual(diagnostics["jump_to_cage_variance"], params.jump_variance / params.cage_variance)
        self.assertAlmostEqual(diagnostics["target_renewal_count"], params.cage_variance / params.jump_variance)
        self.assertAlmostEqual(diagnostics["renewal_rate"], params.renewal_rate, delta=1e-12)

    def test_plateau_ngp_branches_invert_same_observed_value(self):
        beta = 0.8
        observed_ngp = 0.05
        branches = plateau_ngp_branches(jump_to_cage_variance=beta, observed_ngp=observed_ngp)

        self.assertLess(branches["early_y"], 1.0)
        self.assertGreater(branches["late_y"], 1.0)
        self.assertAlmostEqual(branches["early_y"] * branches["late_y"], 1.0)
        for branch in ("early_y", "late_y"):
            y = branches[branch]
            reconstructed = beta * y / (1.0 + y) ** 2
            self.assertAlmostEqual(reconstructed, observed_ngp, delta=1e-14)

    def test_plateau_ngp_branches_reject_values_above_peak_bound(self):
        with self.assertRaises(ValueError):
            plateau_ngp_branches(jump_to_cage_variance=0.8, observed_ngp=0.21)

    def test_observable_consistency_diagnostics_compare_peak_and_late_ngp(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        predicted = dimensionless_peak_prediction(params)
        late_time = 1200.0
        late_ngp = float(ngp_1d(np.array([late_time]), params)[0])
        diagnostics = observable_consistency_diagnostics(
            peak_ngp=predicted["peak_ngp"],
            peak_time=predicted["peak_time"],
            renewal_delay=params.renewal_delay,
            late_time=late_time,
            late_ngp=late_ngp,
        )

        self.assertAlmostEqual(diagnostics["jump_to_cage_variance"], params.jump_variance / params.cage_variance)
        self.assertAlmostEqual(diagnostics["peak_renewal_rate"], params.renewal_rate, delta=1e-12)
        self.assertAlmostEqual(diagnostics["late_renewal_rate_exact"], params.renewal_rate, delta=1e-6)
        self.assertAlmostEqual(diagnostics["late_renewal_rate_asymptotic"], params.renewal_rate, delta=2e-3)
        self.assertLess(abs(diagnostics["log_exact_rate_residual"]), 1e-5)
        self.assertLess(abs(diagnostics["log_asymptotic_rate_residual"]), 0.01)

    def test_observable_consistency_diagnostics_validate_inputs(self):
        with self.assertRaises(ValueError):
            observable_consistency_diagnostics(
                peak_ngp=0.2,
                peak_time=10.0,
                renewal_delay=3.0,
                late_time=0.0,
                late_ngp=0.01,
            )

    def test_scattering_transport_inversion_recovers_minimal_parameters(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        plateau = math.exp(-0.5 * wave_number**2 * params.cage_variance)
        diffusion = long_time_diffusion_coefficient(params)
        tau_alpha = alpha_relaxation_time(wave_number, params)
        peak = dimensionless_peak_prediction(params)

        inferred = infer_parameters_from_scattering_transport(
            wave_number=wave_number,
            debye_waller_plateau=plateau,
            diffusion_coefficient=diffusion,
            tau_alpha=tau_alpha,
            renewal_delay=params.renewal_delay,
        )

        self.assertAlmostEqual(inferred["cage_variance"], params.cage_variance)
        self.assertAlmostEqual(inferred["jump_variance"], params.jump_variance, delta=1e-11)
        self.assertAlmostEqual(inferred["renewal_rate"], params.renewal_rate, delta=1e-12)
        self.assertGreater(inferred["existence_margin"], 1.0)
        self.assertAlmostEqual(inferred["reconstructed_debye_waller_plateau"], plateau)
        self.assertAlmostEqual(inferred["reconstructed_diffusion_coefficient"], diffusion)
        self.assertAlmostEqual(inferred["reconstructed_tau_alpha"], tau_alpha, delta=1e-10)
        self.assertAlmostEqual(inferred["predicted_ngp_peak"], peak["peak_ngp"], delta=1e-12)
        self.assertAlmostEqual(inferred["predicted_ngp_peak_time"], peak["peak_time"], delta=1e-10)

    def test_scattering_transport_inversion_rejects_impossible_observables(self):
        with self.assertRaisesRegex(ValueError, "existence"):
            infer_parameters_from_scattering_transport(
                wave_number=1.1,
                debye_waller_plateau=0.55,
                diffusion_coefficient=1e-4,
                tau_alpha=10.0,
                renewal_delay=3.0,
            )

    def test_full_observable_inversion_recovers_renewal_delay_without_external_tau_d(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.25,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        wave_number = 1.1
        plateau = math.exp(-0.5 * wave_number**2 * params.cage_variance)
        diffusion = long_time_diffusion_coefficient(params)
        tau_alpha = alpha_relaxation_time(wave_number, params)
        peak = dimensionless_peak_prediction(params)

        inferred = infer_parameters_from_full_observables(
            wave_number=wave_number,
            debye_waller_plateau=plateau,
            diffusion_coefficient=diffusion,
            tau_alpha=tau_alpha,
            peak_time=peak["peak_time"],
            peak_ngp=peak["peak_ngp"],
        )

        self.assertAlmostEqual(inferred["cage_variance"], params.cage_variance)
        self.assertAlmostEqual(inferred["jump_variance"], params.jump_variance)
        self.assertAlmostEqual(inferred["renewal_rate"], params.renewal_rate)
        self.assertAlmostEqual(inferred["renewal_delay"], params.renewal_delay, delta=1e-12)
        self.assertAlmostEqual(inferred["reconstructed_tau_alpha"], tau_alpha, delta=1e-10)
        self.assertLess(abs(inferred["log_tau_alpha_residual"]), 1e-12)

    def test_full_observable_inversion_rejects_impossible_peak_timing(self):
        with self.assertRaisesRegex(ValueError, "peak timing"):
            infer_parameters_from_full_observables(
                wave_number=1.1,
                debye_waller_plateau=0.55,
                diffusion_coefficient=0.02,
                tau_alpha=10.0,
                peak_time=1.0,
                peak_ngp=0.2,
            )

    def test_delayed_renewal_shape_is_positive_and_matches_integral(self):
        scaled_time = 2.5
        shape = delayed_renewal_shape(scaled_time)
        numeric = np.trapezoid((1.0 - np.exp(-np.linspace(0.0, scaled_time, 2000))) ** 2, np.linspace(0.0, scaled_time, 2000))

        self.assertGreater(shape, 0.0)
        self.assertAlmostEqual(shape, numeric, delta=1e-6)

    def test_generalized_delay_short_time_law_matches_square_delay_model(self):
        params = DelayedRenewalCageParams(
            cage_variance=1.0,
            cage_tau=0.7,
            jump_variance=0.8,
            renewal_rate=0.18,
            renewal_delay=3.0,
        )
        asymptotic = generalized_delay_ngp_short_time(params, delay_exponent=2.0)
        t = np.array([1e-5, 2e-5, 4e-5])
        alpha = ngp_1d(t, params)

        self.assertAlmostEqual(asymptotic["power"], 1.0)
        np.testing.assert_allclose(alpha / t, asymptotic["prefactor"], rtol=1e-3)

    def test_delay_exponent_classification_explains_square_delay_choice(self):
        self.assertEqual(classify_delay_exponent(0.5), "singular_origin")
        self.assertEqual(classify_delay_exponent(1.0), "finite_origin")
        self.assertEqual(classify_delay_exponent(2.0), "regular_zero_origin")

    def test_persistence_exchange_distribution_normalizes_and_starts_unrenewed(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )

        probability = persistence_exchange_count_distribution(np.array([0.0, 4.0, 20.0]), params, max_count=80)

        self.assertAlmostEqual(probability[0, 0], 1.0)
        self.assertTrue(np.allclose(np.sum(probability, axis=1), 1.0, atol=1e-8))
        self.assertGreater(probability[1, 0], 0.55)
        self.assertLess(probability[2, 0], 0.1)

    def test_persistence_exchange_poisson_limit_recovers_count_moments(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=2.0,
            exchange_mean=2.0,
        )
        times = np.array([1.0, 3.0, 7.0])

        moments = persistence_exchange_count_moments(times, params, max_count=80)

        np.testing.assert_allclose(moments["mean"], times / 2.0, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(moments["variance"], times / 2.0, rtol=2e-5, atol=2e-5)

    def test_persistence_exchange_pgf_matches_alpha_decay(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_number = 1.1
        times = np.array([0.5, 5.0, 20.0])
        jump_factor = math.exp(-0.5 * wave_number**2 * params.jump_variance)

        pgf = persistence_exchange_count_pgf(jump_factor, times, params)
        decay = persistence_exchange_normalized_alpha_decay(wave_number, times, params, max_count=400)

        np.testing.assert_allclose(pgf, decay, rtol=2e-5, atol=2e-8)
        np.testing.assert_allclose(persistence_exchange_count_pgf(1.0, times, params), np.ones_like(times))

    def test_persistence_exchange_closed_moments_match_distribution_moments(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        times = np.array([2.0, 8.0, 30.0])

        probability = persistence_exchange_count_distribution(times, params, max_count=200)
        counts = np.arange(probability.shape[1], dtype=float)
        distribution_mean = probability @ counts
        distribution_variance = probability @ (counts**2) - distribution_mean**2
        closed = persistence_exchange_count_moments(times, params)

        np.testing.assert_allclose(closed["mean"], distribution_mean, rtol=5e-5, atol=5e-6)
        np.testing.assert_allclose(closed["variance"], distribution_variance, rtol=5e-5, atol=5e-6)

    def test_persistence_exchange_decoupling_increases_stokes_einstein_product(self):
        base = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=1.0,
            exchange_mean=1.0,
        )
        decoupled = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=12.0,
            exchange_mean=1.0,
        )
        wave_number = 1.1

        base_tau = persistence_exchange_alpha_relaxation_time(wave_number, base)
        decoupled_tau = persistence_exchange_alpha_relaxation_time(wave_number, decoupled)

        self.assertAlmostEqual(persistence_exchange_diffusion_coefficient(base), persistence_exchange_diffusion_coefficient(decoupled))
        self.assertGreater(decoupled_tau / base_tau, 3.0)
        self.assertGreater(
            persistence_exchange_diffusion_coefficient(decoupled) * decoupled_tau,
            3.0 * persistence_exchange_diffusion_coefficient(base) * base_tau,
        )

    def test_translation_rotation_exchange_detects_rotational_decoupling(self):
        coupled = TranslationRotationExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            translational_persistence_mean=4.0,
            translational_exchange_mean=1.0,
            rotational_persistence_mean=4.0,
            rotational_exchange_mean=1.0,
            rotational_step_correlation=0.62,
        )
        decoupled = TranslationRotationExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            translational_persistence_mean=4.0,
            translational_exchange_mean=1.0,
            rotational_persistence_mean=14.0,
            rotational_exchange_mean=1.0,
            rotational_step_correlation=0.62,
        )

        coupled_row = translation_rotation_decoupling_diagnostic(
            "coupled",
            coupled,
            wave_number=1.1,
        )
        decoupled_row = translation_rotation_decoupling_diagnostic(
            "rotationally_slow",
            decoupled,
            wave_number=1.1,
        )

        self.assertAlmostEqual(coupled_row["translation_rotation_ratio"], 1.0, delta=0.05)
        self.assertGreater(decoupled_row["translation_rotation_ratio"], 2.5)
        self.assertGreater(decoupled_row["rotational_dse_product"], coupled_row["rotational_dse_product"])

    def test_translation_rotation_inversion_recovers_hidden_rotational_clock(self):
        params = TranslationRotationExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            translational_persistence_mean=6.0,
            translational_exchange_mean=1.0,
            rotational_persistence_mean=15.0,
            rotational_exchange_mean=1.0,
            rotational_step_correlation=0.62,
        )
        wave_number = 1.1
        tau_alpha = persistence_exchange_alpha_relaxation_time(
            wave_number,
            PersistenceExchangeParams(
                cage_variance=params.cage_variance,
                cage_tau=params.cage_tau,
                jump_variance=params.jump_variance,
                persistence_mean=params.translational_persistence_mean,
                exchange_mean=params.translational_exchange_mean,
            ),
        )
        tau_rot = translation_rotation_rotational_relaxation_time(params)

        row = translation_rotation_inversion_protocol(
            benchmark_id="near_tg_molecular_motion_synthetic",
            wave_number=wave_number,
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(
                PersistenceExchangeParams(
                    cage_variance=params.cage_variance,
                    cage_tau=params.cage_tau,
                    jump_variance=params.jump_variance,
                    persistence_mean=params.translational_persistence_mean,
                    exchange_mean=params.translational_exchange_mean,
                )
            ),
            observed_tau_alpha=tau_alpha,
            observed_rotational_relaxation_time=tau_rot,
            rotational_step_correlation=params.rotational_step_correlation,
            rotational_exchange_mean=params.rotational_exchange_mean,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
        )

        self.assertAlmostEqual(row["translational_persistence_mean"], 6.0, places=7)
        self.assertAlmostEqual(row["rotational_persistence_mean"], 15.0, places=7)
        self.assertGreater(row["rotational_to_translational_persistence_ratio"], 2.0)
        self.assertEqual(row["translation_rotation_decoupling_detected"], 1.0)

    def test_persistence_exchange_transport_alpha_inversion_predicts_late_ngp(self):
        inference_fn = getattr(sys.modules["renewal_cage"], "infer_persistence_exchange_from_alpha_transport", None)
        if inference_fn is None:
            self.fail("infer_persistence_exchange_from_alpha_transport is missing")
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=9.0,
            exchange_mean=1.0,
        )
        wave_number = 1.1
        tau_alpha = persistence_exchange_alpha_relaxation_time(wave_number, params)
        diffusion = persistence_exchange_diffusion_coefficient(params)
        late_time = 80.0 * params.persistence_mean
        observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])

        inferred = inference_fn(
            wave_number=wave_number,
            jump_variance=params.jump_variance,
            diffusion_coefficient=diffusion,
            observed_tau_alpha=tau_alpha,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            late_time=late_time,
            observed_late_ngp=observed_late_ngp,
        )

        self.assertAlmostEqual(inferred["exchange_mean"], params.exchange_mean, places=10)
        self.assertAlmostEqual(inferred["persistence_mean"], params.persistence_mean, places=8)
        self.assertAlmostEqual(inferred["persistence_exchange_ratio"], 9.0, places=8)
        self.assertAlmostEqual(inferred["predicted_late_ngp"], observed_late_ngp, places=10)
        self.assertLess(abs(inferred["late_ngp_log_residual"]), 1e-8)

    def test_persistence_exchange_transport_alpha_inversion_rejects_too_fast_alpha(self):
        inference_fn = getattr(sys.modules["renewal_cage"], "infer_persistence_exchange_from_alpha_transport", None)
        if inference_fn is None:
            self.fail("infer_persistence_exchange_from_alpha_transport is missing")
        poisson_params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=1.0,
            exchange_mean=1.0,
        )
        wave_number = 1.1
        poisson_tau = persistence_exchange_alpha_relaxation_time(wave_number, poisson_params)

        with self.assertRaises(ValueError):
            inference_fn(
                wave_number=wave_number,
                jump_variance=poisson_params.jump_variance,
                diffusion_coefficient=persistence_exchange_diffusion_coefficient(poisson_params),
                observed_tau_alpha=0.8 * poisson_tau,
                cage_variance=poisson_params.cage_variance,
                cage_tau=poisson_params.cage_tau,
            )

    def test_persistence_exchange_joint_diagnostic_predicts_multik_late_ngp_and_chi4_proxy(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        diffusion = persistence_exchange_diffusion_coefficient(params)
        late_time = 80.0 * params.persistence_mean
        observed_late_ngp = float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0])
        time_grid = np.geomspace(0.05, 300.0, 260)

        diagnostic = persistence_exchange_joint_diagnostic(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            jump_variance=params.jump_variance,
            diffusion_coefficient=diffusion,
            late_time=late_time,
            observed_late_ngp=observed_late_ngp,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            max_multik_abs_log_residual=0.02,
            max_late_ngp_abs_log_residual=0.02,
            min_chi4_peak_growth=1.5,
        )
        poisson = PersistenceExchangeParams(
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            jump_variance=params.jump_variance,
            persistence_mean=params.exchange_mean,
            exchange_mean=params.exchange_mean,
        )
        inferred = PersistenceExchangeParams(
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            jump_variance=params.jump_variance,
            persistence_mean=diagnostic["persistence_mean"],
            exchange_mean=diagnostic["exchange_mean"],
        )
        predicted_chi4 = np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, inferred))
        poisson_chi4 = np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, poisson))

        self.assertAlmostEqual(diagnostic["persistence_exchange_ratio"], 8.0, places=7)
        self.assertLess(diagnostic["max_multik_tau_alpha_abs_log_residual"], 1e-8)
        self.assertLess(abs(diagnostic["late_ngp_log_residual"]), 1e-8)
        self.assertGreater(diagnostic["stokes_einstein_growth_over_poisson"], 2.0)
        self.assertGreater(diagnostic["chi4_peak_growth_over_poisson"], 1.5)
        self.assertAlmostEqual(diagnostic["chi4_peak"], predicted_chi4, places=10)
        self.assertAlmostEqual(diagnostic["poisson_chi4_peak"], poisson_chi4, places=10)
        self.assertEqual(diagnostic["passes_joint_protocol"], 1.0)

    def test_persistence_exchange_joint_diagnostic_rejects_multik_alpha_mismatch(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        observed_tau_alpha[1.6] *= 1.25
        diagnostic = persistence_exchange_joint_diagnostic(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=80.0 * params.persistence_mean,
            observed_late_ngp=float(persistence_exchange_ngp_1d(np.array([80.0 * params.persistence_mean]), params)[0]),
            time_grid=np.geomspace(0.05, 300.0, 260),
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            max_multik_abs_log_residual=0.02,
            max_late_ngp_abs_log_residual=0.02,
            min_chi4_peak_growth=1.5,
        )

        self.assertGreater(diagnostic["max_multik_tau_alpha_abs_log_residual"], 0.02)
        self.assertEqual(diagnostic["multik_tau_alpha_consistent"], 0.0)
        self.assertEqual(diagnostic["passes_joint_protocol"], 0.0)

    def test_persistence_exchange_data_protocol_scores_uncertainty_weighted_observables(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        time_grid = np.geomspace(0.05, 300.0, 260)
        observed_chi4_peak = float(np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, params)))
        late_time = 80.0 * params.persistence_mean

        scored = persistence_exchange_data_protocol(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            tau_alpha_relative_error_by_k={wave_number: 0.05 for wave_number in wave_numbers},
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            observed_late_ngp=float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0]),
            late_ngp_relative_error=0.08,
            observed_chi4_peak=observed_chi4_peak,
            chi4_peak_relative_error=0.1,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=2.0,
        )

        self.assertLess(scored["max_multik_tau_alpha_z"], 1e-8)
        self.assertLess(scored["late_ngp_z"], 1e-8)
        self.assertLess(scored["chi4_peak_z"], 1e-8)
        self.assertEqual(scored["multik_tau_alpha_z_consistent"], 1.0)
        self.assertEqual(scored["late_ngp_z_consistent"], 1.0)
        self.assertEqual(scored["chi4_peak_z_consistent"], 1.0)
        self.assertEqual(scored["passes_uncertainty_protocol"], 1.0)

    def test_persistence_exchange_data_protocol_rejects_chi4_mismatch_beyond_uncertainty(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        time_grid = np.geomspace(0.05, 300.0, 260)
        predicted_chi4_peak = float(np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, params)))
        late_time = 80.0 * params.persistence_mean

        scored = persistence_exchange_data_protocol(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            tau_alpha_relative_error_by_k={wave_number: 0.05 for wave_number in wave_numbers},
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            observed_late_ngp=float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0]),
            late_ngp_relative_error=0.08,
            observed_chi4_peak=2.0 * predicted_chi4_peak,
            chi4_peak_relative_error=0.1,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=2.0,
        )

        self.assertGreater(scored["chi4_peak_z"], 2.0)
        self.assertEqual(scored["chi4_peak_z_consistent"], 0.0)
        self.assertEqual(scored["passes_uncertainty_protocol"], 0.0)

    def test_simultaneous_dynamical_signature_closure_gate_accepts_minimal_anchor_inversion(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        time_grid = np.geomspace(0.05, 300.0, 260)
        late_time = 80.0 * params.persistence_mean
        scored = persistence_exchange_data_protocol(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            tau_alpha_relative_error_by_k={wave_number: 0.05 for wave_number in wave_numbers},
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            observed_late_ngp=float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0]),
            late_ngp_relative_error=0.08,
            observed_chi4_peak=float(
                np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, params))
            ),
            chi4_peak_relative_error=0.1,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=2.0,
        )

        gate = simultaneous_dynamical_signature_closure_gate(
            protocol_id="synthetic_minimal_dynamical_closure",
            scored_row=scored,
            calibration_observables=["diffusion_coefficient", "tau_alpha(k_anchor)"],
            heldout_observables=[
                "tau_alpha(k!=anchor)",
                "late_ngp_recovery",
                "stokes_einstein_growth",
                "chi4_proxy_peak",
            ],
            required_consistency_flags=[
                "multik_tau_alpha_z_consistent",
                "late_ngp_z_consistent",
                "chi4_peak_z_consistent",
            ],
            min_stokes_einstein_growth_over_poisson=2.0,
        )

        self.assertEqual(gate["closure_stage"], "simultaneous_dynamical_signature_closure_passed")
        self.assertEqual(gate["simultaneous_closure_ready"], 1.0)
        self.assertEqual(gate["all_required_dynamical_predictions_pass"], 1.0)
        self.assertEqual(gate["thermodynamic_claim_allowed"], 0.0)
        self.assertEqual(gate["calibration_count"], 2.0)
        self.assertEqual(gate["heldout_count"], 4.0)
        self.assertGreater(gate["stokes_einstein_growth_over_poisson"], 2.0)
        self.assertEqual(gate["primary_blocker"], "none")

    def test_simultaneous_dynamical_signature_closure_gate_rejects_chi4_proxy_mismatch(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=8.0,
            exchange_mean=1.0,
        )
        wave_numbers = [0.7, 1.1, 1.6]
        observed_tau_alpha = {
            wave_number: persistence_exchange_alpha_relaxation_time(wave_number, params)
            for wave_number in wave_numbers
        }
        time_grid = np.geomspace(0.05, 300.0, 260)
        predicted_chi4_peak = float(
            np.max(persistence_exchange_scattering_susceptibility(1.1, time_grid, params))
        )
        late_time = 80.0 * params.persistence_mean
        scored = persistence_exchange_data_protocol(
            anchor_wave_number=1.1,
            wave_numbers=wave_numbers,
            observed_tau_alpha_by_k=observed_tau_alpha,
            tau_alpha_relative_error_by_k={wave_number: 0.05 for wave_number in wave_numbers},
            jump_variance=params.jump_variance,
            diffusion_coefficient=persistence_exchange_diffusion_coefficient(params),
            late_time=late_time,
            observed_late_ngp=float(persistence_exchange_ngp_1d(np.array([late_time]), params)[0]),
            late_ngp_relative_error=0.08,
            observed_chi4_peak=2.0 * predicted_chi4_peak,
            chi4_peak_relative_error=0.1,
            time_grid=time_grid,
            cage_variance=params.cage_variance,
            cage_tau=params.cage_tau,
            z_threshold=2.0,
        )

        gate = simultaneous_dynamical_signature_closure_gate(
            protocol_id="synthetic_chi4_mismatch_closure",
            scored_row=scored,
            calibration_observables=["diffusion_coefficient", "tau_alpha(k_anchor)"],
            heldout_observables=[
                "tau_alpha(k!=anchor)",
                "late_ngp_recovery",
                "stokes_einstein_growth",
                "chi4_proxy_peak",
            ],
            required_consistency_flags=[
                "multik_tau_alpha_z_consistent",
                "late_ngp_z_consistent",
                "chi4_peak_z_consistent",
            ],
            min_stokes_einstein_growth_over_poisson=2.0,
        )

        self.assertEqual(gate["closure_stage"], "dynamical_heldout_prediction_failed")
        self.assertEqual(gate["simultaneous_closure_ready"], 1.0)
        self.assertEqual(gate["all_required_dynamical_predictions_pass"], 0.0)
        self.assertEqual(gate["primary_blocker"], "chi4_peak_z_consistent")
        self.assertEqual(gate["thermodynamic_claim_allowed"], 0.0)

    def test_persistence_exchange_long_time_ngp_recovers_to_zero(self):
        params = PersistenceExchangeParams(
            cage_variance=1.0,
            cage_tau=0.2,
            jump_variance=0.7,
            persistence_mean=12.0,
            exchange_mean=1.0,
        )
        times = np.array([8.0, 30.0, 650.0])

        alpha = persistence_exchange_ngp_1d(times, params, max_count=500)
        decay = persistence_exchange_normalized_alpha_decay(1.1, times, params, max_count=500)

        self.assertGreater(alpha[0], 0.05)
        self.assertLess(alpha[-1], alpha[0] / 20.0)
        self.assertLess(decay[-1], 1e-20)

    def test_persistence_exchange_scan_identifies_ratio_as_se_control(self):
        rows = persistence_exchange_scan(
            ratios=[1.0, 2.0, 4.0, 8.0, 12.0],
            exchange_mean=1.0,
            wave_number=1.1,
        )

        ratios = np.array([row["persistence_exchange_ratio"] for row in rows])
        se_products = np.array([row["stokes_einstein_product"] for row in rows])
        late_ngp = np.array([row["late_ngp"] for row in rows])

        np.testing.assert_allclose(ratios, [1.0, 2.0, 4.0, 8.0, 12.0])
        self.assertTrue(np.all(np.diff(se_products) > 0.0))
        self.assertLess(late_ngp[-1], 0.02)


if __name__ == "__main__":
    unittest.main()
