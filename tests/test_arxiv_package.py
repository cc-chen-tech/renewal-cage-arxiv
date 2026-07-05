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
            self.assertIn("figures/renewal_cage_barrier_requirements.pdf", names)
            self.assertIn("figures/renewal_cage_mechanism_selection.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_joint_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_persistence_exchange_uncertainty_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_inversion.pdf", names)

    def test_main_tex_uses_arxiv_safe_pdf_figures(self):
        main_tex = (ROOT / "paper" / "main.tex").read_text()

        self.assertIn("figures/renewal_cage_results.pdf", main_tex)
        self.assertIn("figures/renewal_cage_dimensionless.pdf", main_tex)
        self.assertIn("figures/renewal_cage_scattering.pdf", main_tex)
        self.assertIn("figures/renewal_cage_temperature.pdf", main_tex)
        self.assertIn("figures/renewal_cage_barrier.pdf", main_tex)
        self.assertIn("figures/renewal_cage_heterogeneity.pdf", main_tex)
        self.assertIn("figures/renewal_cage_heterogeneity_map.pdf", main_tex)
        self.assertIn("figures/renewal_cage_static_null.pdf", main_tex)
        self.assertIn("figures/renewal_cage_alpha_shape.pdf", main_tex)
        self.assertIn("figures/renewal_cage_facilitated_exchange.pdf", main_tex)
        self.assertIn("figures/renewal_cage_glass_audit.pdf", main_tex)
        self.assertIn("figures/renewal_cage_glass_phase_diagram.pdf", main_tex)
        self.assertIn("figures/renewal_cage_spatial_chi4.pdf", main_tex)
        self.assertIn("figures/renewal_cage_thermodynamic_closure.pdf", main_tex)
        self.assertIn("figures/renewal_cage_mct_beta_closure.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_benchmark_consistency.pdf", main_tex)
        self.assertIn("figures/renewal_cage_barrier_requirements.pdf", main_tex)
        self.assertIn("figures/renewal_cage_mechanism_selection.pdf", main_tex)
        self.assertIn("figures/renewal_cage_persistence_exchange.pdf", main_tex)
        self.assertIn("figures/renewal_cage_persistence_exchange_protocol.pdf", main_tex)
        self.assertIn("figures/renewal_cage_persistence_exchange_joint_protocol.pdf", main_tex)
        self.assertIn("figures/renewal_cage_persistence_exchange_uncertainty_protocol.pdf", main_tex)
        self.assertIn("figures/renewal_cage_inversion.pdf", main_tex)
        self.assertNotIn(".svg", main_tex)

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
        self.assertEqual(first_barrier_requirements, second_barrier_requirements)
        self.assertEqual(first_mechanism_selection, second_mechanism_selection)
        self.assertEqual(first_persistence_exchange, second_persistence_exchange)
        self.assertEqual(first_persistence_exchange_protocol, second_persistence_exchange_protocol)
        self.assertEqual(first_persistence_exchange_joint_protocol, second_persistence_exchange_joint_protocol)
        self.assertEqual(first_persistence_exchange_uncertainty_protocol, second_persistence_exchange_uncertainty_protocol)
        self.assertEqual(first_inversion, second_inversion)


if __name__ == "__main__":
    unittest.main()
