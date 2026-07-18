Warning: truncated output (original token count: 113516)
Total output lines: 8250

import sys
import math
import tempfile
import time
import unittest
import zipfile
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_arxiv_package import build_arxiv_package  # noqa: E402
from summarize_ka_cage_anchor_gate import classify_cage_anchor_gate  # noqa: E402
from summarize_ka_anchor_semi_markov_gate import (  # noqa: E402
    classify_anchor_semi_markov_gate,
)
from summarize_ka_memory_hierarchy import (  # noqa: E402
    main as summarize_memory_hierarchy,
    recompute_hierarchy,
)
from summarize_ka_segment_splice_gate import (  # noqa: E402
    classify_segment_splice_gate,
)
from summarize_ka_segment_splice_paired_excess import (  # noqa: E402
    classify_paired_excess_gate,
    main as summarize_segment_splice_paired_excess,
)
from summarize_ka_transport_clock_shape_quotient import (  # noqa: E402
    CLOSED_GATE_FIELDS as TRANSPORT_CLOCK_CLOSED_GATE_FIELDS,
    main as summarize_transport_clock_shape_quotient,
)
from summarize_ka_gamma_variance_mixture import (  # noqa: E402
    CLOSED_GATE_FIELDS as GAMMA_VARIANCE_CLOSED_GATE_FIELDS,
    main as summarize_gamma_variance_mixture,
)
from summarize_ka_activated_cage_geometry import (  # noqa: E402
    STRONG_ZERO_FLAGS as ACTIVATED_GEOMETRY_ZERO_FLAGS,
    main as summarize_activated_cage_geometry,
)
from summarize_ka_variance_mixture_shape_quotient import (  # noqa: E402
    OUTPUT_CLOSED_CLAIMS as VARIANCE_MIXTURE_CLOSED_CLAIMS,
    main as summarize_variance_mixture_shape_quotient,
)


class ArxivPackageTests(unittest.TestCase):
    def assertSerializedValueMatches(self, stored, key, expected):
        self.assertIn(key, stored)
        try:
            stored_value = float(stored[key])
            expected_value = float(expected)
        except (TypeError, ValueError):
            self.assertEqual(stored[key], str(expected))
        else:
            tolerance = max(1e-10, abs(expected_value) * 1e-11)
            self.assertAlmostEqual(stored_value, expected_value, delta=tolerance)

    def assertCsvRowsMatchComputedRows(self, stored_rows, computed_rows):
        self.assertEqual(len(stored_rows), len(computed_rows))
        for stored_row, computed_row in zip(stored_rows, computed_rows):
            self.assertEqual(set(stored_row), set(computed_row))
            for key, value in computed_row.items():
                self.assertSerializedValueMatches(stored_row, key, value)

    def assertCsvGateMatchesComputedGate(self, stored_gate, computed_gate):
        self.assertEqual(set(stored_gate), set(computed_gate))
        for key, value in computed_gate.items():
            self.assertSerializedValueMatches(stored_gate, key, value)

    def test_memory_hierarchy_is_recomputed_and_claim_limited(self):
        data = ROOT / "data"
        verdict_path = data / "renewal_cage_ka_memory_hierarchy.csv"
        evidence_path = data / "renewal_cage_ka_memory_hierarchy_evidence.csv"
        figure_path = ROOT / "figures" / "renewal_cage_ka_memory_hierarchy.svg"
        for path in (verdict_path, evidence_path, figure_path):
            self.assertTrue(path.is_file(), path)

        with verdict_path.open() as handle:
            stored = next(csv.DictReader(handle))
        with evidence_path.open() as handle:
            evidence = list(csv.DictReader(handle))
        recomputed = recompute_hierarchy(data)

        self.assertEqual(
            stored["mechanism_state"],
            "ordered_particle_path_required_finite_exchange_supported",
        )
        for key, value in recomputed.items():
            if isinstance(value, str):
                self.assertEqual(stored[key], value)
            else:
                self.assertAlmostEqual(float(stored[key]), float(value))
        self.assertEqual(len(evidence), 6)
        self.assertEqual(evidence[-1]["heldout_result"], "measurement only")
        self.assertEqual(float(evidence[-1]["mechanism_claim_allowed"]), 0.0)
        self.assertEqual(float(stored["microdynamic_closure_claim_allowed"]), 0.0)
        self.assertEqual(float(stored["spatial_facilitation_claim_allowed"]), 0.0)
        self.assertEqual(float(stored["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(float(stored["static_environment_family_excluded"]), 0.0)
        self.assertEqual(float(stored["cross_particle_mechanism_excluded"]), 0.0)
        self.assertEqual(float(stored["environment_recomputed_from_raw"]), 1.0)
        self.assertEqual(
            float(stored["environment_waiting_consensus_recomputed_from_windows"]),
            1.0,
        )
        self.assertEqual(float(stored["waiting_recomputed_from_thresholds"]), 1.0)
        self.assertEqual(
            float(stored["stored_environment_waiting_consistency_pass"]), 1.0
        )
        self.assertEqual(float(stored["stored_subgate_consistency_pass"]), 1.0)
        figure = figure_path.read_text()
        self.assertIn("Information-ablation hierarchy", figure)
        self.assertIn("measurement only", figure.lower())
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            generated_verdict = temporary / "verdict.csv"
            generated_evidence = temporary / "evidence.csv"
            generated_figure = temporary / "figure.svg"
            summarize_memory_hierarchy(
                [
                    "--data-dir",
                    str(data),
                    "--output-csv",
                    str(generated_verdict),
                    "--output-evidence-csv",
                    str(generated_evidence),
                    "--output-svg",
                    str(generated_figure),
                ]
            )
            self.assertEqual(verdict_path.read_bytes(), generated_verdict.read_bytes())
            self.assertEqual(evidence_path.read_bytes(), generated_evidence.read_bytes())
            self.assertEqual(figure_path.read_bytes(), generated_figure.read_bytes())

    def test_segment_splice_memory_gate_is_recomputed_and_claim_limited(self):
        data = ROOT / "data"
        tables = {}
        for code, temperature, lengths, replicates in (
            ("045", 0.45, (1, 2, 5, 10, 25, 50, 125, 250), (1, 2, 3)),
            ("058", 0.58, (1, 2, 4, 8, 16, 32, 37), (1, 2, 3, 4, 5)),
        ):
            tables[temperature] = {}
            for suffix in ("quality", "rows", "summary", "cells", "replicate_scores"):
                path = data / f"renewal_cage_ka_replicates_T{code}_segment_splice_{suffix}.csv"
                self.assertTrue(path.is_file(), path)
                with path.open() as handle:
                    tables[temperature][suffix] = list(csv.DictReader(handle))
                self.assertTrue(tables[temperature][suffix])
            cells = tables[temperature]["cells"]
            self.assertEqual(
                {
                    (row["model"], int(float(row["segment_length"])))
                    for row in cells
                },
                {
                    (model, length)
                    for model in (
                        "within_particle_segment_shuffle",
                        "cross_particle_segment_splice",
                    )
                    for length in lengths
                },
            )
            self.assertEqual(
                {float(row["required_realization_count"]) for row in cells},
                {64.0},
            )
            quality = tables[temperature]["quality"]
            self.assertEqual(
                {
                    (int(float(row["replicate"])), int(float(row["realization"])))
                    for row in quality
                    if row["model"] == "within_particle_segment_shuffle"
                    and int(float(row["segment_length"])) == lengths[0]
                },
                {(replicate, realization) for replicate in replicates for realization in range(64)},
            )
            for suffix, rows in tables[temperature].items():
                for row in rows:
                    self.assertEqual(float(row["microdynamic_closure_claim_allowed"]), 0.0)
                    self.assertEqual(float(row["spatial_facilitation_claim_allowed"]), 0.0)
                    self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
            self.assertTrue(
                all(
                    float(row["global_source_segment_schedule_preserved"]) == 1.0
                    for row in quality + cells
                )
            )

        low_cells = tables[0.45]["cells"]
        high_cells = tables[0.58]["cells"]
        low_precision_failures = {
            (row["model"], int(float(row["segment_length"])))
            for row in low_cells
            if float(row["precision_pass"]) == 0.0
        }
        self.assertEqual(
            low_precision_failures,
            {
                ("within_particle_segment_shuffle", 1),
                ("within_particle_segment_shuffle", 2),
            },
        )
        self.assertEqual(
            {
                (row["model"], int(float(row["segment_length"])))
                for row in low_cells
                if float(row["curve_transfer_pass"]) == 1.0
            },
            {
                ("within_particle_segment_shuffle", 250),
                ("cross_particle_segment_splice", 250),
            },
        )
        self.assertTrue(all(float(row["stationarity_control_pass"]) == 0.0 for row in high_cells))

        recomputed = classify_segment_splice_gate(
            low_cells,
            high_cells,
            tables[0.45]["replicate_scores"],
            tables[0.58]["replicate_scores"],
        )
        gate_path = data / "renewal_cage_ka_segment_splice_gate.csv"
        with gate_path.open() as handle:
            stored = next(csv.DictReader(handle))
        for key, value in recomputed.items():
            self.assertEqual(stored[key], str(value))
        self.assertEqual(stored["mechanism_state"], "mechanism_unresolved")
        self.assertEqual(stored["low_temperature_mechanism_state"], "mechanism_unresolved")
        self.assertEqual(float(stored["low_temperature_gate_resolved"]), 0.0)
        self.assertEqual(float(stored["high_temperature_control_resolved"]), 0.0)
        self.assertEqual(float(stored["low_within_memory_length_resolved"]), 0.0)
        self.assertEqual(float(stored["low_cross_memory_length_resolved"]), 0.0)
        self.assertEqual(float(stored["cell_grid_and_row_completeness_pass"]), 1.0)
        self.assertEqual(float(stored["global_source_segment_schedule_preserved"]), 1.0)
        self.assertEqual(float(stored["low_largest_nonfull_length"]), 125.0)
        self.assertEqual(float(stored["low_largest_nonfull_tau"]), 2500.0)
        self.assertEqual(float(stored["low_largest_nonfull_within_precision_pass"]), 1.0)
        self.assertEqual(float(stored["low_largest_nonfull_cross_precision_pass"]), 1.0)
        self.assertEqual(float(stored["low_largest_nonfull_within_curve_pass"]), 0.0)
        self.assertEqual(float(stored["low_largest_nonfull_cross_curve_pass"]), 0.0)
        self.assertEqual(
            float(stored["low_largest_nonfull_both_models_ensemble_curve_rejected"]),
            1.0,
        )
        self.assertEqual(float(stored["low_full_path_control_ensemble_pass"]), 1.0)
        self.assertEqual(float(stored["low_full_path_control_all_replicates_pass"]), 0.0)
        self.assertEqual(float(stored["low_full_path_control_failed_replicate_count"]), 2.0)
        self.assertEqual(
            float(stored["low_mechanism_identifiable_against_full_path_control"]),
            0.0,
        )
        self.assertEqual(float(stored["ensemble_cancellation_detected"]), 1.0)
        self.assertEqual(
            float(stored["independent_replicate_memory_lower_bound_claim_allowed"]),
            0.0,
        )
        self.assertEqual(float(stored["low_owner_identity_paired_ordering_count"]), 21.0)
        self.assertEqual(float(stored["low_owner_identity_paired_ordering_total"]), 21.0)
        self.assertAlmostEqual(
            float(stored["low_owner_identity_replicate_first_mean_score_difference"]),
            2.0718621990041886,
            places=12,
        )
        self.assertAlmostEqual(
            float(stored["low_owner_identity_replicate_first_standard_error"]),
            0.27773289433991805,
            places=12,
        )
        self.assertAlmostEqual(
            float(stored["low_owner_identity_replicate_first_t95_ci_low"]),
            0.8768740029863806,
            places=12,
        )
        self.assertAlmostEqual(
            float(stored["low_owner_identity_replicate_first_t95_ci_high"]),
            3.2668503950219963,
            places=12,
        )
        self.assertEqual(float(stored["owner_identity_information_supported_exploratory"]), 1.0)
        self.assertEqual(float(stored["owner_identity_sufficiency_claim_allowed"]), 0.0)
        self.assertEqual(float(stored["static_vs_finite_exchange_resolved"]), 0.0)
        self.assertEqual(
            stored["substantive_interpretation_condition"],
            "conditional_on_preserved_global_source_segment_schedule",
        )

        svg = (ROOT / "figures" / "renewal_cage_ka_segment_splice_gate.svg").read_text()
        self.assertIn("full-path control", svg)
        self.assertIn("normalized maximum error (clipped at 2.5)", svg)
        self.assertIn("&gt;=2.5", svg)
        self.assertIn('data-clipped="true"', svg)
        self.assertIn("no microscopic, spatial-facilitation, or thermodynamic claim", svg)
        coordinates = [float(value) for value in re.findall(r'(?:(?:x|y)[12]?|cx|cy)="([0-9.]+)"', svg)]
        self.assertTrue(coordinates)
        self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in coordinates))
        note = (ROOT / "docs" / "segment-splice-memory-gate.md").read_text()
        self.assertIn("conditional on the preserved global source-segment schedule", note)
        self.assertIn("does not identify finite single-particle memory", note)

    def test_segment_splice_paired_excess_gate_is_recomputed_and_claim_limited(self):
        data = ROOT / "data"
        rows_path = data / "renewal_cage_ka_segment_splice_paired_excess_rows.csv"
        gate_path = data / "renewal_cage_ka_segment_splice_paired_excess_gate.csv"
        figure_path = ROOT / "figures" / "renewal_cage_ka_segment_splice_paired_excess.svg"
        note_path = ROOT / "docs" / "segment-splice-paired-excess.md"
        for path in (rows_path, gate_path, figure_path, note_path):
            self.assertTrue(path.is_file(), path)

        with rows_path.open() as handle:
            rows = list(csv.DictReader(handle))
        with gate_path.open() as handle:
            stored = next(csv.DictReader(handle))
        self.assertEqual(len(rows), 14)
        self.assertEqual(
            {
                (row["model"], int(float(row["segment_length"])))
                for row in rows
            },
            {
                (model, length)
                for model in (
                    "within_particle_segment_shuffle",
                    "cross_particle_segment_splice",
                )
                for length in (1, 2, 5, 10, 25, 50, 125)
            },
        )
        for row in rows:
            self.assertEqual(float(row["microdynamic_closure_claim_allowed"]), 0.0)
            self.assertEqual(float(row["spatial_facilitation_claim_allowed"]), 0.0)
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
            self.assertEqual(float(row["post_run_exploratory"]), 1.0)
            self.assertEqual(float(row["global_source_segment_schedule_preserved"]), 1.0)
            self.assertEqual(float(row["replicate_provenance_validation_pass"]), 1.0)
            self.assertEqual(float(row["independent_replicate_count"]), 0.0)
            self.assertEqual(float(row["replicate_count"]), 3.0)
            self.assertEqual(float(row["parent_sample_count"]), 1.0)
            self.assertEqual(
                float(row["replicate_first_interval_independence_claim_allowed"]),
                0.0,
            )
            self.assertEqual(
                row["ci95_method"],
                "student_t_replicate_first_correlated_parent_exploratory_df2",
            )

        with (data / "renewal_cage_ka_replicates_T045_segment_splice_replicate_scores.csv").open() as handle:
            replicate_scores = list(csv.DictReader(handle))
        with (data / "renewal_cage_ka_replicates_T045_segment_splice_cells.csv").open() as handle:
            cells = list(csv.DictReader(handle))
        with (data / "renewal_cage_ka_segment_splice_gate.csv").open() as handle:
            source_verdict = list(csv.DictReader(handle))
        with (data / "renewal_cage_ka_replicates_T058_T045_provenance.csv").open() as handle:
            provenance = list(csv.DictReader(handle))
        recomputed_rows, recomputed_gate = classify_paired_excess_gate(
            replicate_scores,
            cells,
            source_verdict,
            provenance,
        )
        self.assertEqual(len(recomputed_rows), len(rows))
        self.assertCsvRowsMatchComputedRows(rows, recomputed_rows)
        self.assertCsvGateMatchesComputedGate(stored, recomputed_gate)

        self.assertEqual(stored["mechanism_state"], "mechanism_unresolved")
        self.assertEqual(float(stored["input_completeness_pass"]), 1.0)
        self.assertEqual(float(stored["paired_input_exactness_pass"]), 1.0)
        self.assertEqual(float(stored["source_verdict_fail_closed"]), 0.0)
        self.assertEqual(float(stored["full_path_model_agreement_pass"]), 1.0)
        self.assertEqual(float(stored["low_full_path_control_all_replicates_pass"]), 0.0)
        self.assertEqual(float(stored["low_full_path_control_failed_replicate_count"]), 2.0)
        self.assertEqual(float(stored["replicate_provenance_validation_pass"]), 1.0)
        self.assertEqual(float(stored["replicate_count"]), 3.0)
        self.assertEqual(float(stored["parent_sample_count"]), 1.0)
        self.assertEqual(float(stored["independent_replicate_count"]), 0.0)
        self.assertEqual(float(stored["independently_prepared_parent_samples"]), 0.0)
        self.assertEqual(float(stored["identified_prefix_max_segment_length"]), 10.0)
        self.assertEqual(float(stored["identified_prefix_max_tau"]), 200.0)
        self.assertEqual(
            float(stored["short_horizon_information_loss_supported_exploratory"]),
            1.0,
        )
        self.assertEqual(float(stored["owner_identity_paired_ordering_count"]), 21.0)
        self.assertEqual(float(stored["owner_identity_paired_ordering_total"]), 21.0)
        self.assertEqual(float(stored["owner_identity_information_supported_exploratory"]), 1.0)
        self.assertEqual(float(stored["paired_excess_equivalence_claim_allowed"]), 0.0)
        self.assertEqual(
            float(stored["independent_replicate_memory_lower_bound_claim_allowed"]),
            0.0,
        )
        self.assertEqual(float(stored["finite_memory_state_addition_allowed"]), 0.0)
        self.assertEqual(float(stored["owner_identity_sufficiency_claim_allowed"]), 0.0)
        self.assertEqual(float(stored["microdynamic_closure_claim_allowed"]), 0.0)
        self.assertEqual(float(stored["spatial_facilitation_claim_allowed"]), 0.0)
        self.assertEqual(float(stored["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(
            stored["next_required_action"],
            "replicate_resolved_full_path_baseline_or_new_trajectory_validation",
        )

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            generated_rows = temporary / "rows.csv"
            generated_gate = temporary / "gate.csv"
            generated_figure = temporary / "figure.svg"
            summarize_segment_splice_paired_excess(
                [
                    "--replicate-scores",
                    str(data / "renewal_cage_ka_replicates_T045_segment_splice_replicate_scores.csv"),
                    "--cells",
                    str(data / "renewal_cage_ka_replicates_T045_segment_splice_cells.csv"),
                    "--source-verdict",
                    str(data / "renewal_cage_ka_segment_splice_gate.csv"),
                    "--provenance",
                    str(data / "renewal_cage_ka_replicates_T058_T045_provenance.csv"),
                    "--output-rows",
                    str(generated_rows),
                    "--output-gate",
                    str(generated_gate),
                    "--output-svg",
                    str(generated_figure),
                ]
            )
            self.assertEqual(rows_path.read_bytes(), generated_rows.read_bytes())
            self.assertEqual(gate_path.read_bytes(), generated_gate.read_bytes())
            self.assertEqual(figure_path.read_bytes(), generated_figure.read_bytes())

        svg = figure_path.read_text()
        self.assertIn("Paired excess over replicate full-path baseline", svg)
        self.assertIn("post-run exploratory", svg)
        self.assertIn("mechanism unresolved", svg)
        self.assertIn("no independent-sample CI, sufficiency, microscopic, spatial, or thermodynamic claim", svg)
        coordinates = [
            float(value)
            for value in re.findall(r'(?:(?:x|y)[12]?|cx|cy)="([0-9.]+)"', svg)
        ]
        self.assertTrue(coordinates)
        self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in coordinates))
        note = note_path.read_text()
        self.assertIn("full-path replicate baseline itself does not close", note)
        self.assertIn("finite_memory_state_addition_allowed = 0", note)

    def test_gamma_variance_mixture_is_recomputed_and_claim_limited(self):
        data = ROOT / "data"
        rows_path = data / "renewal_cage_ka_gamma_variance_mixture_rows.csv"
        gate_path = data / "renewal_cage_ka_gamma_variance_mixture_gate.csv"
        simulation_path = (
            data / "renewal_cage_gamma_variance_mixture_langevin_validation.csv"
        )
        figure_path = ROOT / "figures" / "renewal_cage_ka_gamma_variance_mixture.svg"
        note_path = ROOT / "docs" / "microscopic-gamma-variance-mixture.md"
        for path in (rows_path, gate_path, simulation_path, figure_path, note_path):
            self.assertTrue(path.is_file(), path)

        with rows_path.open() as handle:
            rows = list(csv.DictReader(handle))
        with gate_path.open() as handle:
            gates = list(csv.DictReader(handle))
        with simulation_path.open() as handle:
            simulation = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 46)
        self.assertEqual(len(gates), 2)
        self.assertEqual(len(simulation), 3)
        low = next(row for row in gates if float(row["temperature"]) == 0.45)
        high = next(row for row in gates if float(row["temperature"]) == 0.58)
        self.assertEqual(low["analysis_status"], "scalar_mobility_cage_scale_residual")
        self.assertAlmostEqual(
            float(low["fs_k7p25_gamma_max_normalized_error"]),
            1.38542105249768,
            delta=5e-9,
        )
        self.assertAlmostEqual(
            float(low["minimum_cage_root_support_fraction"]),
            3.0 / 7.0,
            delta=5e-9,
        )
        self.assertEqual(
            float(low["scalar_mobility_low_intermediate_k_supported_exploratory"]),
            1.0,
        )
        self.assertEqual(
            float(low["scalar_mobility_shape_closure_supported_exploratory"]),
            0.0,
        )
        self.assertEqual(float(low["cage_plus_mobility_support_coverage_pass"]), 0.0)
        self.assertEqual(high["analysis_status"], "high_temperature_canary_only")
        self.assertEqual(float(high["high_temperature_control_resolved"]), 0.0)
        for gate in gates:
            self.assertEqual(float(gate["replicate_provenance_validation_pass"]), 1.0)
            self.assertEqual(float(gate["parent_sample_count"]), 1.0)
            self.assertEqual(float(gate["independent_replicate_count"]), 0.0)
            for field in GAMMA_VARIANCE_CLOSED_GATE_FIELDS:
                self.assertEqual(float(gate[field]), 0.0, (gate["temperature"], field))

        slow = next(row for row in simulation if float(row["tau_D_over_t"]) == 100.0)
        self.assertEqual(float(slow["slow_environment_limit_validation_pass"]), 1.0)
        self.assertLess(float(slow["msd_relative_error"]), 0.015)
        self.assertLess(float(slow["ngp_absolute_error"]), 0.035)
        self.assertLess(float(slow["maximum_fs_absolute_error"]), 0.012)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            generated = [
                temporary / "rows.csv",
                temporary / "gate.csv",
                temporary / "simulation.csv",
                temporary / "figure.svg",
            ]
            summarize_gamma_variance_mixture(
                [
                    "--low-rows",
                    str(data / "renewal_cage_ka_replicates_T045_segment_splice_rows.csv"),
                    "--high-rows",
                    str(data / "renewal_cage_ka_replicates_T058_segment_splice_rows.csv"),
                    "--low-stationarity",
                    str(data / "renewal_cage_ka_replicates_T045_nonlinear_path_stationarity.csv"),
                    "--high-stationarity",
                    str(data / "renewal_cage_ka_replicates_T058_block20_nonlinear_path_stationarity.csv"),
                    "--provenance",
                    str(data / "renewal_cage_ka_replicates_T058_T045_provenance.csv"),
                    "--output-rows",
                    str(generated[0]),
                    "--output-gate",
                    str(generated[1]),
                    "--output-simulation",
                    str(generated[2]),
                    "--output-svg",
                    str(generated[3]),
                ]
            )
            for stored, recomputed in zip(
                (rows_path, gate_path, simulation_path, figure_path), generated
            ):
                self.assertEqual(stored.read_bytes(), recomputed.read_bytes())

        svg = figure_path.read_text()
        self.assertIn("maximum normalized Fs error (tolerance units)", svg)
        self.assertIn("scalar mobility fails at cage scale", svg)
        self.assertIn("T=0.58 canary only", svg)
        note = re.sub(r"\s+", " ", note_path.read_text())
        self.assertIn("linear Gaussian GLE is an exact null", note)
        self.assertIn("heldout MSD and NGP are diagnostic inputs", note)
        self.assertIn("microdynamic_closure_claim_allowed = 0", note)
        readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text())
        self.assertIn("gamma variance-mixture Langevin", readme)
        self.assertIn("fail at the T=0.45 cage-scale wave number", readme)

    def test_activated_cage_geometry_is_recomputed_and_claim_limited(self):
        data = ROOT / "data"
        low_geometry_path = (
            data / "renewal_cage_ka_replicates_T045_calibration_jump_geometry.csv"
        )
        high_geometry_path = (
            data / "renewal_cage_ka_replicates_T058_calibration_jump_geometry.csv"
        )
        rows_path = data / "renewal_cage_ka_activated_cage_geometry_rows.csv"
        gate_path = data / "renewal_cage_ka_activated_cage_geometry_gate.csv"
        figure_path = ROOT / "figures" / "renewal_cage_ka_activated_cage_geometry.svg"
        note_path = ROOT / "docs" / "microscopic-activated-cage-geometry.md"
        for path in (
            low_geometry_path,
            high_geometry_path,
            rows_path,
            gate_path,
            figure_path,
            note_path,
        ):
            self.assertTrue(path.is_file(), path)

        with low_geometry_path.open() as handle:
            low_geometry = list(csv.DictReader(handle))
        with high_geometry_path.open() as handle:
            high_geometry = list(csv.DictReader(handle))
        with rows_path.open() as handle:
            rows = list(csv.DictReader(handle))
        with gate_path.open() as handle:
            gates = list(csv.DictReader(handle))
        self.assertEqual(len(low_geometry), 3)
        self.assertEqual(len(high_geometry), 5)
        self.assertEqual(len(rows), 46)
        self.assertEqual(len(gates), 2)
        for row in low_geometry + high_geometry:
            self.assertEqual(float(row["calibration_events_only"]), 1.0)
            self.assertEqual(float(row["heldout_events_used"]), 0.0)
            for field in ACTIVATED_GEOMETRY_ZERO_FLAGS:
                self.assertEqual(float(row[field]), 0.0)

        low = next(row for row in gates if float(row["temperature"]) == 0.45)
        high = next(row for row in gates if float(row["temperature"]) == 0.58)
        self.assertEqual(
            low["analysis_status"],
            "compound_poisson_cage_decomposition_unsupported",
        )
        self.assertEqual(float(low["empirical_supported_row_count"]), 8.0)
        self.assertAlmostEqual(
            float(low["empirical_support_fraction"]),
            8.0 / 21.0,
            delta=5e-10,
        )
        self.assertEqual(float(low["fixed_length_supported_row_count"]), 3.0)
        self.assertEqual(float(low["source_stationarity_pass"]), 1.0)
        self.assertAlmostEqual(
            float(low["fs_k7p25_empirical_max_absolute_error"]),
            0.04164052592,
            delta=5e-10,
        )
        self.assertEqual(float(low["curve_transfer_pass"]), 0.0)
        self.assertEqual(
            float(low["empirical_activated_jump_geometry_supported_exploratory"]),
            0.0,
        )
        self.assertEqual(
            high["analysis_status"],
            "canary_only_nonstationary_source",
        )
        self.assertEqual(float(high["empirical_supported_row_count"]), 25.0)
        self.assertEqual(float(high["source_stationarity_pass"]), 0.0)
        self.assertEqual(float(high["high_temperature_canary_only"]), 1.0)
        self.assertAlmostEqual(
            float(high["fs_k7p25_empirical_max_absolute_error"]),
            0.04341535083,
            delta=5e-10,
        )
        self.assertEqual(
            float(high["empirical_activated_jump_geometry_supported_exploratory"]),
            0.0,
        )
        for row in rows:
            for field in ACTIVATED_GEOMETRY_ZERO_FLAGS:
                self.assertEqual(float(row[field]), 0.0)
        for gate in gates:
            for field in ACTIVATED_GEOMETRY_ZERO_FLAGS:
                self.assertEqual(float(gate[field]), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            generated_rows = temporary / "rows.csv"
            generated_gate = temporary / "gate.csv"
            generated_figure = temporary / "figure.svg"
            summarize_activated_cage_geometry(
                [
                    "--low-geometry",
                    str(low_geometry_path),
                    "--high-geometry",
                    str(high_geometry_path),
                    "--gamma-rows",
                    str(data / "renewal_cage_ka_gamma_variance_mixture_rows.csv"),
                    "--low-stationarity",
                    str(data / "renewal_cage_ka_replicates_T045_nonlinear_path_stationarity.csv"),
                    "--high-stationarity",
                    str(data / "renewal_cage_ka_replicates_T058_block20_nonlinear_path_stationarity.csv"),
                    "--provenance",
                    str(data / "renewal_cage_ka_replicates_T058_T045_provenance.csv"),
                    "--output-rows",
                    str(generated_rows),
                    "--output-gate",
                    str(generated_gate),
                    "--output-svg",
                    str(generated_figure),
                ]
            )
            self.assertEqual(rows_path.read_bytes(), generated_rows.read_bytes())
            self.assertEqual(gate_path.read_bytes(), generated_gate.read_bytes())
            self.assertEqual(figure_path.read_bytes(), generated_figure.read_bytes())

        svg = figure_path.read_text()
        self.assertIn("Activated cage-jump geometry quotient", svg)
        self.assertIn("absolute Fs error (tolerance = 0.03)", svg)
        self.assertIn("fixed-length null: 3/21 supported at T=0.45", svg)
        self.assertIn("support gate failed", svg)
        self.assertIn("nonstationary canary", svg)
        self.assertNotIn("compound_poisson_cage_decomposition_unsupported", svg)
        self.assertNotIn("canary_only_nonstationary_source", svg)
        note = re.sub(r"\s+", " ", note_path.read_text())
        self.assertIn("held-out MSD and NGP are diagnostic inputs", note)
        self.assertIn("activated_cage_geometry_resolved = 0", note)
        readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text())
        self.assertIn("activated cage-jump geometry quotient", readme)

    def test_transport_clock_shape_quotient_is_recomputed_and_claim_limited(self):
        data = ROOT / "data"
        rows_path = data / "renewal_cage_ka_transport_clock_shape_quotient_rows.csv"
        gate_path = data / "renewal_cage_ka_transport_clock_shape_quotient_gate.csv"
        figure_path = ROOT / "figures" / "renewal_cage_ka_transport_clock_shape_quotient.svg"
        for path in (rows_path, gate_path, figure_path):
            self.assertTrue(path.is_file(), path)

        with rows_path.open() as handle:
            rows = list(csv.DictReader(handle))
        with gate_path.open() as handle:
            gates = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 184)
        self.assertEqual(len(gates), 2)
        self.assertEqual({float(row["temperature"]) for row in gates}, {0.45, 0.58})
        for row in rows:
            self.assertEqual(float(row["heldout_msd_used_as_diagnostic_input"]), 1.0)
            self.assertEqual(float(row["blind_prediction_claim_allowed"]), 0.0)
            self.assertEqual(float(row["microdynamic_closure_claim_allowed"]), 0.0)
            self.assertEqual(float(row["spatial_facilitation_claim_allowed"]), 0.0)
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

        low = next(row for row in gates if float(row["temperature"]) == 0.45)
        high = next(row for row in gates if float(row["temperature"]) == 0.58)
        self.assertEqual(low["analysis_status"], "clock_shape_separation_exploratory")
        self.assertEqual(float(low["clock_shape_separation_supported_exploratory"]), 1.0)
        self.assertAlmostEqual(
            float(low["fs_k4_max_normalized_msd_matched_error"]),
            0.508749096734,
            places=12,
        )
        self.assertAlmostEqual(
            float(low["fs_k7p25_max_normalized_msd_matched_error"]),
            1.13359180984,
            places=11,
        )
        self.assertEqual(high["analysis_status"], "high_temperature_canary_only")
        self.assertEqual(float(high["high_temperature_canary_only"]), 1.0)
        self.assertEqual(float(high["high_temperature_control_resolved"]), 0.0)
        for gate in gates:
            self.assertEqual(float(gate["replicate_provenance_validation_pass"]), 1.0)
            self.assertEqual(float(gate["parent_sample_count"]), 1.0)
            self.assertEqual(float(gate["independent_replicate_count"]), 0.0)
            self.assertEqual(float(gate["independently_prepared_parent_samples"]), 0.0)
            self.assertEqual(
                gate["independence_class"],
                "decorrelated_parent_frames_plus_velocity_seeds",
            )
            self.assertEqual(
                float(gate["confirmatory_independent_parent_replication_required"]),
                1.0,
            )
            for field in TRANSPORT_CLOCK_CLOSED_GATE_FIELDS:
                self.assertEqual(float(gate[field]), 0.0, (gate["temperature"], field))

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            generated_rows = temporary / "rows.csv"
            generated_gate = temporary / "gate.csv"
            generated_figure = temporary / "figure.svg"
            summarize_transport_clock_shape_quotient(
                [
                    "--low-rows",
                    str(data / "renewal_cage_ka_replicates_T045_segment_splice_rows.csv"),
                    "--high-rows",
                    str(data / "renewal_cage_ka_replicates_T058_segment_splice_rows.csv"),
                    "--low-stationarity",
                    str(data / "renewal_cage_ka_replicates_T045_nonlinear_path_stationarity.csv"),
                    "--high-stationarity",
                    str(data / "renewal_cage_ka_replicates_T058_block20_nonlinear_path_stationarity.csv"),
                    "--provenance",
                    str(data / "renewal_cage_ka_replicates_T058_T045_provenance.csv"),
                    "--output-rows",
                    str(generated_rows),
                    "--output-gate",
                    str(generated_gate),
                    "--output-svg",
                    str(generated_figure),
                ]
            )
            self.assertEqual(rows_path.read_bytes(), generated_rows.read_bytes())
            self.assertEqual(gate_path.read_bytes(), generated_gate.read_bytes())
            self.assertEqual(figure_path.read_bytes(), generated_figure.read_bytes())

        svg = figure_path.read_text()
        self.assertIn("maximum normalized error (tolerance units)", svg)
        self.assertIn("heldout MSD is a diagnostic input, not a blind prediction", svg)
        self.assertIn("T=0.58 canary only", svg)
        self.assertIn("clock-only closure rejected", svg)
        coordinates = [
            float(value)
            for value in re.findall(r'(?:(?:x|y)[12]?|cx|cy)="([0-9.]+)"', svg)
        ]
        self.assertTrue(coordinates)
        self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in coordinates))
        readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text())
        self.assertIn("transport-clock / shape quotient", readme)
        self.assertIn("diagnostic input, not a blind prediction", readme)
        self.assertIn("T=0.58 remains a canary", readme)

    def test_variance_mixture_shape_quotient_is_recomputed_and_claim_limited(self):
        data = ROOT / "data"
        rows_path = data / "renewal_cage_ka_variance_mixture_shape_quotient_rows.csv"
        gate_path = data / "renewal_cage_ka_variance_mixture_shape_quotient_gate.csv"
        figure_path = ROOT / "figures" / "renewal_cage_ka_variance_mixture_shape_quotient.svg"
        for path in (rows_path, gate_path, figure_path):
            self.assertTrue(path.is_file(), path)

        with rows_path.open() as handle:
            rows = list(csv.DictReader(handle))
        with gate_path.open() as handle:
            gates = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 114)
        self.assertEqual(len(gates), 2)
        self.assertEqual({float(row["temperature"]) for row in gates}, {0.45, 0.58})
        low = next(row for row in gates if float(row["temperature"]) == 0.45)
        high = next(row for row in gates if float(row["temperature"]) == 0.58)
        self.assertEqual(
            low["analysis_status"],
            "variance_mixture_shape_closure_exploratory",
        )
        self.assertAlmostEqual(
            float(low["maximum_baseline_normalized_error"]),
            1.13359180983653,
            delta=5e-8,
        )
        self.assertAlmostEqual(
            float(low["maximum_fourth_normalized_error"]),
            5.66701641386661,
            delta=5e-8,
        )
        self.assertAlmostEqual(
            float(low["maximum_gamma_normalized_error"]),
            0.465837665728119,
            delta=5e-8,
        )
        self.assertAlmostEqual(
            float(low["maximum_inverse_gaussian_normalized_error"]),
            0.422604946010197,
            delta=5e-8,
        )
        self.assertEqual(float(low["family_robust_resummation_pass"]), 1.0)
        self.assertEqual(
            float(
                low[
                    "marginal_variance_mixture_shape_closure_supported_exploratory"
                ]
            ),
            1.0,
        )
        self.assertEqual(float(low["variance_mixture_family_selected"]), 0.0)
        self.assertEqual(high["analysis_status"], "high_temperature_canary_only")
        self.assertEqual(float(high["source_stationarity_pass"]), 0.0)
        self.assertEqual(float(high["high_temperature_canary_only"]), 1.0)
        for gate in gates:
            self.assertEqual(float(gate["replicate_provenance_validation_pass"]), 1.0)
            self.assertEqual(float(gate["parent_sample_count"]), 1.0)
            self.assertEqual(float(gate["independent_replicate_count"]), 0.0)
            self.assertEqual(float(gate["heldout_msd_used_as_diagnostic_input"]), 1.0)
            self.assertEqual(float(gate["heldout_ngp_used_as_diagnostic_input"]), 1.0)
            self.assertEqual(float(gate["macro_fit_parameter_count"]), 0.0)
            for field in VARIANCE_MIXTURE_CLOSED_CLAIMS:
                self.assertEqual(float(gate[field]), 0.0, (gate["temperature"], field))
        for row in rows:
            self.assertEqual(float(row["heldout_msd_used_as_diagnostic_input"]), 1.0)
            self.assertEqual(float(row["heldout_ngp_used_as_diagnostic_input"]), 1.0)
            self.assertEqual(float(row["macro_fit_parameter_count"]), 0.0)
            for field in VARIANCE_MIXTURE_CLOSED_CLAIMS:
                self.assertEqual(float(row[field]), 0.0, (row["temperature"], field))

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            generated_rows = temporary / "rows.csv"
            generated_gate = temporary / "gate.csv"
            generated_figure = temporary / "figure.svg"
            summarize_variance_mixture_shape_quotient(
                [
                    "--quotient-rows",
                    str(data / "renewal_cage_ka_transport_clock_shape_quotient_rows.csv"),
                    "--source-gate",
                    str(data / "renewal_cage_ka_transport_clock_shape_quotient_gate.csv"),
                    "--output-rows",
                    str(generated_rows),
                    "--output-gate",
                    str(generated_gate),
                    "--output-svg",
                    str(generated_figure),
                ]
            )
            self.assertEqual(rows_path.read_bytes(), generated_rows.read_bytes())
            self.assertEqual(gate_path.read_bytes(), generated_gate.read_bytes())
            self.assertEqual(figure_path.read_bytes(), generated_figure.read_bytes())

        svg = figure_path.read_text()
        self.assertIn("Variance-mixture shape quotient", svg)
        self.assertIn("held-out MSD and NGP are diagnostic inputs", svg)
        self.assertIn("family unresolved", svg)
        self.assertIn("T=0.58 canary only", svg)
        coordinates = [
            float(value)
            for value in re.findall(r'(?:(?:x|y)[12]?|cx|cy)="([0-9.]+)"', svg)
        ]
        self.assertTrue(coordinates)
        self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in coordinates))
        readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text())
        self.assertIn("variance-mixture shape quotient", readme)
        self.assertIn("held-out MSD and NGP are diagnostic inputs", readme)
        self.assertIn("does not select a unique variance-mixture family", readme)

    def test_anchor_semi_markov_gate_is_recomputed_and_claim_limited(self):
        data = ROOT / "data"
        prefixes = {
            "low": data / "renewal_cage_ka_replicates_T045_anchor_semi_markov",
            "high": data / "renewal_cage_ka_replicates_T058_anchor_semi_markov",
        }
        tables = {}
        for label, prefix in prefixes.items():
            tables[label] = {}
            for suffix in (
                "transitions",
                "quality",
                "realizations",
                "rows",
                "summary",
                "verdict",
            ):
                path = prefix.with_name(f"{prefix.name}_{suffix}.csv")
                self.assertTrue(path.is_file(), path)
                with path.open() as handle:
                    tables[label][suffix] = list(csv.DictReader(handle))

        gate_path = data / "renewal_cage_ka_anchor_semi_markov_gate.csv"
        sota_path = data / "renewal_cage_ka_anchor_semi_markov_sota_audit.csv"
        svg_path = ROOT / "figures" / "renewal_cage_ka_anchor_semi_markov_gate.svg"
        self.assertTrue(gate_path.is_file())
        self.assertTrue(sota_path.is_file())
        self.assertTrue(svg_path.is_file())
        with gate_path.open() as handle:
            stored = next(csv.DictReader(handle))
        with sota_path.open() as handle:
            sota_rows = list(csv.DictReader(handle))
        with (
            data / "renewal_cage_ka_replicates_T045_recoil_markov_rows.csv"
        ).open() as handle:
            low_recoil_rows = list(csv.DictReader(handle))
        recoil_verdicts = []
        empirical_verdicts = []
        for label in ("T045", "T058"):
            with (
                data / f"renewal_cage_ka_replicates_{label}_recoil_markov_verdict.csv"
            ).open() as handle:
                recoil_verdicts.extend(csv.DictReader(handle))
            with (
                data / f"renewal_cage_ka_replicates_{label}_empirical_path_verdict.csv"
            ).open() as handle:
                empirical_verdicts.extend(
                    row
                    for row in csv.DictReader(handle)
                    if row["model"] == "contiguous_empirical_path"
                )
        recomputed = classify_anchor_semi_markov_gate(
            low_quality_rows=tables["low"]["quality"],
            low_summary_rows=tables["low"]["summary"],
            low_replicate_rows=tables["low"]["rows"],
            high_quality_rows=tables["high"]["quality"],
            high_summary_rows=tables["high"]["summary"],
            high_replicate_rows=tables["high"]["rows"],
            low_recoil_rows=low_recoil_rows,
            recoil_verdict_rows=recoil_verdicts,
            empirical_verdict_rows=empirical_verdicts,
        )

        self.assertEqual(len(tables["low"]["quality"]), 3 * 16 * 2)
        self.assertEqual(len(tables["high"]["quality"]), 5 * 16 * 2)
        for label, replicate_count in (("low", 3), ("high", 5)):
            quality = tables[label]["quality"]
            for model in (
                "anchor_aware_semi_markov",
                "state_schedule_without_anchor_geometry",
            ):
                grid = {
                    (int(float(row["replicate"])), int(float(row["realization"])))
                    for row in quality
                    if row["model"] == model
                }
                self.assertEqual(
                    grid,
                    {
                        (replicate, realization)
                        for replicate in range(1, replicate_count + 1)
                        for realization in range(16)
                    },
                )
            for suffix in ("quality", "realizations", "rows", "summary", "verdict"):
                for row in tables[label][suffix]:
                    self.assertEqual(float(row["microdynamic_closure_claim_allowed"]), 0.0)
                    self.assertEqual(float(row["spatial_facilitation_claim_allowed"]), 0.0)
                    self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
            for row in tables[label]["quality"]:
                self.assertEqual(float(row["calibration_events_only"]), 1.0)
                self.assertEqual(float(row["external_cage_channel_used"]), 1.0)
                self.assertEqual(float(row["calibration_cage_residual_transfer"]), 1.0)
                self.assertEqual(float(row["heldout_events_used_in_calibration"]), 0.0)
                self.assertEqual(float(row["heldout_cage_residual_used_in_prediction"]), 0.0)

        self.assertEqual(stored["mechanism_state"], recomputed["mechanism_state"])
        self.assertEqual(stored["mechanism_state"], "anchor_aware_model_rejected")
        self.assertEqual(len(sota_rows), 4)
        self.assertEqual(
            {row["alignment"] for row in sota_rows},
            {
                "consistent_event_signal",
                "consistent_missing_memory",
                "consistent_nonseparability_warning",
                "consistent_need_for_richer_escape_path",
            },
        )
        self.assertTrue(all(row["primary_url"].startswith("https://doi.org/") for row in sota_rows))
        for key in (
            "provenance_and_competitor_completeness_pass",
            "all_low_anchor_replicates_improve_over_recoil",
            "microdynamic_closure_claim_allowed",
            "spatial_facilitation_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(stored[key]), float(recomputed[key]))
        svg = svg_path.read_text()
        self.assertIn("Anchor-aware semi-Markov held-out gate", svg)
        self.assertIn("anchor_aware_model_rejected", svg)
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf", svg.lower())

    def test_softmode_precursor_residual_gate_is_complete(self):
        document_path = ROOT / "docs" / "microscopic-softmode-precursor-residual.md"
        summary_path = (
            ROOT / "data" / "renewal_cage_ka_softmode_precursor_T058_summary.csv"
        )
        model_path = (
            ROOT / "data" / "renewal_cage_ka_softmode_precursor_T058_models.csv"
        )
        committor_path = (
            ROOT / "data" / "renewal_cage_ka_softmode_precursor_T058_committor.csv"
        )
        for path in (document_path, summary_path, model_path, committor_path):
            self.assertTrue(path.is_file())
        document = document_path.read_text()
        for required in (
            "-0.00799",
            "-0.00087",
            "-0.00096",
            "-3.7375",
            "instantaneous_local_softmode_precursor_allowed = 0",
            "event_clock_claim_allowed = 0",
            "autonomous_single_particle_gle_claim_allowed = 0",
            "kramers_escape_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
            "stop adding scalar static local descriptors",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            summary = next(csv.DictReader(handle))
        for key, expected in (
            ("parent_count", 5),
            ("distinct_parent_restart_hash_count", 5),
            ("clone_count_per_parent", 8),
            ("target_count", 64),
            ("observation_count", 2560),
            ("configuration_count", 320),
            ("event_count", 1731),
            ("censored_count", 829),
        ):
            self.assertEqual(int(summary[key]), expected)
        self.assertEqual(float(summary["maximum_clone_position_difference"]), 0.0)
        self.assertEqual(float(summary["maximum_geometry_reference_error"]), 0.0)
        self.assertEqual(summary["integrity_gate_pass"], "True")
        self.assertEqual(summary["geometry_reproduction_gate_pass"], "True")
        self.assertEqual(summary["clone_invariance_gate_pass"], "True")
        self.assertEqual(summary["brier_increment_gate_pass"], "False")
        self.assertEqual(summary["brier_reference_gate_pass"], "False")
        self.assertEqual(summary["likelihood_gate_pass"], "False")
        self.assertEqual(summary["survival_gate_pass"], "True")
        self.assertEqual(summary["binomial_gate_pass"], "False")
        self.assertEqual(
            summary["instantaneous_local_softmode_precursor_allowed"], "False"
        )
        self.assertEqual(summary["event_clock_claim_allowed"], "False")
        self.assertEqual(
            summary["autonomous_single_particle_gle_claim_allowed"], "False"
        )
        self.assertEqual(summary["kramers_escape_claim_allowed"], "False")
        self.assertEqual(summary["thermodynamic_claim_allowed"], "False")
        self.assertLess(float(summary["softmode_mean_heldout_brier_skill"]), 0.0)
        self.assertLess(
            float(summary["geometry_softmode_mean_heldout_brier_skill"]), 0.0
        )
        self.assertLess(
            float(summary["geometry_softmode_minimum_group_log_likelihood_gain"]),
            0.0,
        )
        self.assertLess(
            float(summary["geometry_softmode_binomial_mean_heldout_brier_skill"]),
            0.0,
        )

        with model_path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            {row["model"] for row in rows if row["record"] == "censored_model"},
            {"geometry", "softmode", "geometry_softmode"},
        )
        self.assertEqual(sum(row["record"] == "held_parent" for row in rows), 15)

    def test_radial_precursor_residual_gate_is_complete_and_claim_limited(self):
        document_path = ROOT / "docs" / "microscopic-radial-precursor-residual.md"
        summary_path = (
            ROOT / "data" / "renewal_cage_ka_radial_precursor_T058_summary.csv"
        )
        model_path = (
            ROOT / "data" / "renewal_cage_ka_radial_precursor_T058_models.csv"
        )
        for path in (
            document_path,
            summary_path,
            model_path,
            ROOT / "data" / "renewal_cage_ka_radial_precursor_T058_details.csv",
            ROOT / "data" / "renewal_cage_ka_radial_precursor_T058_survival.csv",
            ROOT / "data" / "renewal_cage_ka_radial_precursor_T058_committor.csv",
        ):
            self.assertTrue(path.is_file())
        document = document_path.read_text()
        for required in (
            "0.00825",
            "0.00798",
            "0.00848",
            "0.00744",
            "static_radial_precursor_allowed = 0",
            "event_clock_claim_allowed = 0",
            "autonomous_single_particle_gle_claim_allowed = 0",
            "kramers_escape_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            summary = next(csv.DictReader(handle))
        self.assertEqual(int(summary["parent_count"]), 5)
        self.assertEqual(int(summary["distinct_parent_restart_hash_count"]), 5)
        self.assertEqual(int(summary["clone_count_per_parent"]), 8)
        self.assertEqual(int(summary["target_count"]), 64)
        self.assertEqual(int(summary["observation_count"]), 2560)
        self.assertEqual(int(summary["configuration_count"]), 320)
        self.assertEqual(int(summary["event_count"]), 1731)
        self.assertEqual(int(summary["censored_count"]), 829)
        self.assertEqual(float(summary["maximum_clone_position_difference"]), 0.0)
        self.assertEqual(float(summary["maximum_geometry_reference_error"]), 0.0)
        self.assertEqual(summary["integrity_gate_pass"], "True")
        self.assertEqual(summary["geometry_reproduction_gate_pass"], "True")
        self.assertEqual(summary["clone_invariance_gate_pass"], "True")
        self.assertEqual(summary["brier_increment_gate_pass"], "False")
        self.assertEqual(summary["brier_reference_gate_pass"], "False")
        self.assertEqual(summary["likelihood_gate_pass"], "True")
        self.assertEqual(summary["survival_gate_pass"], "True")
        self.assertEqual(summary["binomial_gate_pass"], "False")
        self.assertEqual(summary["static_radial_precursor_allowed"], "False")
        self.assertEqual(summary["event_clock_claim_allowed"], "False")
        self.assertEqual(
            summary["autonomous_single_particle_gle_claim_allowed"], "False"
        )
        self.assertEqual(summary["kramers_escape_claim_allowed"], "False")
        self.assertEqual(summary["thermodynamic_claim_allowed"], "False")
        self.assertLess(
            float(summary["geometry_radial_mean_heldout_brier_skill"]),
            float(summary["geometry_mean_heldout_brier_skill"]),
        )
        self.assertLess(
            float(summary["geometry_radial_mean_heldout_brier_skill"]),
            float(summary["radial_mean_heldout_brier_skill"]),
        )
        self.assertLess(
            float(summary["geometry_radial_binomial_mean_heldout_brier_skill"]),
            float(summary["geometry_binomial_mean_heldout_brier_skill"]),
        )

        with model_path.open() as handle:
            rows = list(csv.DictReader(handle))
        models = {row["model"] for row in rows if row["record"] == "censored_model"}
        self.assertEqual(models, {"geometry", "radial", "geometry_radial"})
        self.assertEqual(sum(row["record"] == "held_parent" for row in rows), 15)

    def test_smooth_cage_event_clock_is_complete_and_claim_limited(self):
        document_path = ROOT / "docs" / "microscopic-smooth-cage-event-clock.md"
        summary_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_smooth_cage_event_clock_T058_summary.csv"
        )
        model_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_smooth_cage_event_clock_T058_models.csv"
        )
        for path in (
            document_path,
            summary_path,
            model_path,
            ROOT
            / "data"
            / "renewal_cage_ka_smooth_cage_event_clock_T058_details.csv",
            ROOT
            / "data"
            / "renewal_cage_ka_smooth_cage_event_clock_T058_survival.csv",
        ):
            self.assertTrue(path.is_file())
        document = document_path.read_text()
        for required in (
            "1731",
            "829",
            "0.00529",
            "0.00836",
            "event_clock_claim_allowed = 0",
            "autonomous_single_particle_gle_claim_allowed = 0",
            "kramers_escape_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            summary = next(csv.DictReader(handle))
        self.assertEqual(int(summary["parent_count"]), 5)
        self.assertEqual(int(summary["distinct_parent_restart_hash_count"]), 5)
        self.assertEqual(int(summary["clone_count_per_parent"]), 8)
        self.assertEqual(int(summary["target_count"]), 64)
        self.assertEqual(int(summary["observation_count"]), 2560)
        self.assertEqual(int(summary["event_count"]), 1731)
        self.assertEqual(int(summary["censored_count"]), 829)
        self.assertEqual(summary["integrity_gate_pass"], "True")
        self.assertEqual(summary["survival_gate_pass"], "True")
        self.assertEqual(summary["microscopic_initial_escape_state_allowed"], "False")
        self.assertEqual(summary["event_clock_claim_allowed"], "False")
        self.assertEqual(summary["autonomous_single_particle_gle_claim_allowed"], "False")
        self.assertEqual(summary["kramers_escape_claim_allowed"], "False")
        self.assertEqual(summary["thermodynamic_claim_allowed"], "False")
        self.assertLess(
            float(summary["full_mean_heldout_brier_skill"]),
            float(summary["geometry_mean_heldout_brier_skill"]),
        )
        self.assertLess(
            float(summary["full_mean_heldout_brier_skill"]),
            float(summary["structural_brier_reference"]),
        )
        self.assertLess(float(summary["full_minimum_group_log_likelihood_gain"]), 0.0)

        with model_path.open() as handle:
            models = {
                row["model"]: row
                for row in csv.DictReader(handle)
                if row["record"] == "model"
            }
        self.assertEqual(set(models), {"geometry", "kinematic", "full"})

    def test_smooth_cage_microscopic_projection_is_complete_and_claim_limited(self):
        document_path = ROOT / "docs" / "microscopic-smooth-cage-projection.md"
        summary_path = ROOT / "data" / "renewal_cage_ka_smooth_cage_tangent_T058_summary.csv"
        required_paths = (
            ROOT / "scripts" / "run_ka_smooth_cage_response.py",
            ROOT / "scripts" / "analyze_ka_smooth_cage_tangent.py",
            ROOT / "data" / "renewal_cage_ka_smooth_cage_tangent_stride1_T058.csv",
            ROOT / "data" / "renewal_cage_ka_smooth_cage_tangent_stride2_T058.csv",
            ROOT / "data" / "renewal_cage_ka_smooth_cage_tangent_stride5_T058.csv",
        )
        self.assertTrue(document_path.is_file())
        self.assertTrue(all(path.is_file() for path in required_paths))
        document = document_path.read_text()
        for required in (
            "wendland_c4",
            "dp_i",
            "delta J",
            "0.8457",
            "320/320",
            "event_clock_claim_allowed = 0",
            "autonomous_single_particle_gle_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            summary = next(csv.DictReader(handle))
        self.assertEqual(summary["potential_protocol"], "ka_lj_c3_switch")
        self.assertEqual(int(summary["minimum_valid_intervals_per_epsilon"]), 320)
        self.assertEqual(summary["microscopic_smooth_cage_tangent_gate_pass"], "True")
        self.assertEqual(summary["event_clock_claim_allowed"], "False")
        self.assertEqual(summary["autonomous_single_particle_gle_claim_allowed"], "False")
        self.assertEqual(summary["thermodynamic_claim_allowed"], "False")
        self.assertLess(float(summary["maximum_jacobian_gram_condition_number"]), 2.1)
        self.assertLess(
            float(summary["maximum_cross_epsilon_covariance_relative_l2"]), 1.0e-3
        )
        self.assertGreater(
            float(summary["minimum_cross_epsilon_covariance_correlation"]), 0.9999
        )
    def test_cage_anchor_gate_artifacts_enforce_frozen_crossover(self):
        data = ROOT / "data"
        figure = ROOT / "figures" / "renewal_cage_ka_cage_anchor_gate.svg"
        return_rows_path = data / "renewal_cage_ka_cage_anchor_returns_rows.csv"
        return_verdict_path = data / "renewal_cage_ka_cage_anchor_returns_verdict.csv"
        recoil_prefixes = {
            "low": data / "renewal_cage_ka_replicates_T045_recoil_markov",
            "high": data / "renewal_cage_ka_replicates_T058_recoil_markov",
        }
        gate_path = data / "renewal_cage_ka_cage_anchor_gate.csv"
        ordered_path = data / "renewal_cage_ka_replicates_T045_empirical_path_verdict.csv"

        with return_rows_path.open() as handle:
            return_rows = list(csv.DictReader(handle))
        with return_verdict_path.open() as handle:
            return_verdict = next(csv.DictReader(handle))
        recoil = {}
        for temperature, prefix in recoil_prefixes.items():
            tables = {}
            for suffix in ("rows", "quality", "summary", "verdict"):
                with prefix.with_name(f"{prefix.name}_{suffix}.csv").open() as handle:
                    tables[suffix] = list(csv.DictReader(handle))
            recoil[temperature] = tables
        with gate_path.open() as handle:
            gate = next(csv.DictReader(handle))
        with ordered_path.open() as handle:
            ordered = next(
                row
                for row in csv.DictReader(handle)
                if row["model"] == "contiguous_empirical_path"
            )

        self.assertEqual(len(return_rows), 24)
        self.assertEqual(len(recoil["low"]["quality"]), 48)
        self.assertEqual(len(recoil["high"]["quality"]), 80)
        self.assertEqual(float(return_verdict["low_temperature_replicate_count"]), 3.0)
        self.assertEqual(float(return_verdict["high_temperature_replicate_count"]), 5.0)
        self.assertEqual(float(return_verdict["all_radius_scales_separated"]), 1.0)
        self.assertEqual(float(return_verdict["primary_radius_null_excess_pass"]), 1.0)
        self.assertGreaterEqual(
            float(return_verdict["minimum_primary_low_return_excess_ratio"]),
            1.35,
        )
        self.assertEqual({float(row["radius_scale"]) for row in return_rows}, {0.5, 1.0, 1.5})

        for temperature, expected_count, calibration_time in (
            ("low", 3.0, 5000.0),
            ("high", 5.0, 750.0),
        ):
            verdict = recoil[temperature]["verdict"][0]
            quality = recoil[temperature]["quality"]
            self.assertEqual(float(verdict["independent_replicate_count"]), expected_count)
            self.assertEqual(float(verdict["calibration_time"]), calibration_time)
            self.assertEqual(float(verdict["block_size"]), 20.0)
            self.assertEqual(float(verdict["radial_bin_count"]), 8.0)
            self.assertEqual(float(verdict["required_realizations_per_replicate"]), 16.0)
            self.assertEqual(float(verdict["quality_realization_completeness_pass"]), 1.0)
            self.assertEqual(float(verdict["quality_pass"]), 1.0)
            self.assertEqual(float(verdict["precision_pass"]), 1.0)
            self.assertLessEqual(float(verdict["maximum_ensemble_msd_mc_relative_se"]), 0.01)
            self.assertLessEqual(float(verdict["maximum_ensemble_ngp_mc_se"]), 0.03)
            self.assertLessEqual(float(verdict["maximum_ensemble_fs_mc_se"]), 0.003)
            self.assertLessEqual(max(float(row["radial_mean_relative_error"]) for row in quality), 0.02)
            self.assertLessEqual(max(float(row["radial_standard_deviation_relative_error"]) for row in quality), 0.02)
            self.assertLessEqual(max(float(row["lag_one_cosine_mean_absolute_error"]) for row in quality), 0.02)
            self.assertLessEqual(max(float(row["lag_one_cosine_quantile_maximum_absolute_error"]) for row in quality), 0.03)
            self.assertLessEqual(max(float(row["normalized_lag_one_dot_correlation_absolute_error"]) for row in quality), 0.02)
            self.assertTrue(all(float(row["radial_bin_count"]) == 8.0 for row in quality))
            self.assertTrue(all(float(row["replicate_first_aggregation"]) == 1.0 for row in recoil[temperature]["summary"]))

        low = recoil["low"]["verdict"][0]
        high = recoil["high"]["verdict"][0]
        self.assertGreater(float(low["maximum_ensemble_ngp_absolute_error"]), 0.30)
        self.assertGreater(float(low["maximum_ensemble_fs_absolute_error"]), 0.03)
        self.assertEqual(float(high["curve_transfer_pass"]), 1.0)
        self.assertLessEqual(float(high["maximum_ensemble_msd_relative_error"]), 0.10)
        self.assertLessEqual(float(high["maximum_ensemble_ngp_absolute_error"]), 0.30)
        self.assertLessEqual(float(high["maximum_ensemble_fs_absolute_error"]), 0.03)

        for key, expected in (
            ("block_size", 20.0),
            ("radial_bin_count", 8.0),
            ("recoil_realizations_per_replicate", 16.0),
            ("primary_radius_scale", 1.0),
            ("msd_relative_error_tolerance", 0.10),
            ("ngp_absolute_error_tolerance", 0.30),
            ("fs_absolute_error_tolerance", 0.03),
        ):
            self.assertEqual(float(gate[key]), expected)
        self.assertEqual(float(gate["ordered_calibration_path_upper_bound_pass"]), 1.0)
        self.assertEqual(float(gate["cage_anchor_memory_required"]), 1.0)
        self.assertEqual(gate["mechanism_state"], "cage_anchor_memory_required")
        recomputed = classify_cage_anchor_gate(
            [row for row in return_rows if float(row["temperature"]) == 0.45],
            [row for row in return_rows if float(row["temperature"]) == 0.58],
            recoil["low"]["quality"],
            recoil["high"]["quality"],
            recoil["low"]["verdict"][0],
            recoil["high"]["verdict"][0],
            ordered,
        )
        self.assertEqual(set(recomputed), set(gate))
        for key, value in recomputed.items():
            if isinstance(value, str):
                self.assertEqual(gate[key], value)
            else:
                self.assertEqual(float(gate[key]), value)
        for key in (
            "microdynamic_closure_claim_allowed",
            "spatial_facilitation_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(gate[key]), 0.0)
            self.assertTrue(all(float(row[key]) == 0.0 for row in return_rows))
            for tables in recoil.values():
                for table in tables.values():
                    self.assertTrue(all(float(row[key]) == 0.0 for row in table))

        svg = figure.read_text()
        self.assertIn("Cooling-induced cage-anchor memory gate", svg)
        self.assertIn("T=0.45", svg)
        self.assertIn("T=0.58", svg)
        self.assertIn(
            'y="528" font-family="Arial, sans-serif" font-size="11" fill="#596268">Dynamical mechanism only; closure, facilitation,</text>',
            svg,
        )
        self.assertIn(
            'y="543" font-family="Arial, sans-serif" font-size="11" fill="#596268">and thermodynamic claims remain 0.</text>',
            svg,
        )
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf", svg.lower())

    def test_nonlinear_path_gate_artifacts_preserve_claim_boundaries(self):
        gate_path = ROOT / "data" / "renewal_cage_ka_nonlinear_path_gate.csv"
        quality_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_nonlinear_path_quality.csv"
        )
        validity_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_path_cumulant_validity.csv"
        )
        low_verdict_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_nonlinear_path_verdict.csv"
        )
        high_verdict_paths = [
            ROOT
            / "data"
            / f"renewal_cage_ka_replicates_T058_{block}_nonlinear_path_verdict.csv"
            for block in ("block20", "block10")
        ]
        svg_path = ROOT / "figures" / "renewal_cage_ka_nonlinear_path_gate.svg"
        with gate_path.open() as handle:
            gate = next(csv.DictReader(handle))
        with quality_path.open() as handle:
            quality = list(csv.DictReader(handle))
        with validity_path.open() as handle:
            validity = list(csv.DictReader(handle))
        with low_verdict_path.open() as handle:
            low_verdict = next(csv.DictReader(handle))
        high_verdicts = []
        for path in high_verdict_paths:
            with path.open() as handle:
                high_verdicts.append(next(csv.DictReader(handle)))

        self.assertEqual(float(gate["low_temperature_gate_ready"]), 1.0)
        self.assertEqual(
            float(gate["low_temperature_nonlinear_path_memory_required"]),
            1.0,
        )
        self.assertEqual(
            float(gate["low_temperature_surrogate_failure_replicate_count"]),
            3.0,
        )
        self.assertEqual(
            float(gate["low_temperature_paired_contiguous_better_replicate_count"]),
            3.0,
        )
        self.assertEqual(float(gate["high_temperature_resolution_sensitivity"]), 1.0)
        self.assertEqual(float(gate["high_temperature_mechanism_resolved"]), 0.0)
        self.assertEqual(
            float(gate["binary_temperature_crossover_claim_allowed"]),
            0.0,
        )
        self.assertEqual(float(gate["unique_microscopic_model_selected"]), 0.0)
        self.assertEqual(float(gate["low_cumulant_horizon_k2"]), 4096.0)
        self.assertEqual(float(gate["low_cumulant_horizon_k4"]), 200.0)
        self.assertEqual(float(gate["low_cumulant_horizon_k7p25"]), 20.0)
        self.assertEqual(len(quality), 24)
        self.assertEqual(
            float(low_verdict["surrogate_realization_completeness_pass"]),
            1.0,
        )
        self.assertEqual(
            float(low_verdict["required_surrogate_realizations_per_replicate"]),
            8.0,
        )
        self.assertEqual(
            float(
                low_verdict[
                    "one_block_radial_plus_two_point_spectrum_sufficiency_resolved"
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(low_verdict["one_block_radial_plus_two_point_spectrum_sufficient"]),
            0.0,
        )
        for verdict in high_verdicts:
            self.assertEqual(
                float(
                    verdict[
                        "one_block_radial_plus_two_point_spectrum_sufficiency_resolved"
                    ]
                ),
                0.0,
            )
            self.assertEqual(
                float(verdict["calibration_nonstationarity_assessment_resolved"]),
                0.0,
            )
        self.assertLess(
            max(float(row["cross_spectral_matrix_nrmse"]) for row in quality),
            0.012,
        )
        for row in quality:
            self.assertEqual(float(row["calibration_time"]), 5000.0)
            self.assertEqual(float(row["independent_replicate_count"]), 3.0)
            self.assertEqual(float(row["surrogate_realizations_per_replicate"]), 8.0)
            self.assertEqual(float(row["surrogate_iteration_count"]), 110.0)
            self.assertEqual(float(row["surrogate_base_seed"]), 211003.0)
            self.assertEqual(float(row["heldout_path_used_in_prediction"]), 0.0)
        for row in validity:
            self.assertEqual(float(row["independent_replicate_count"]), 3.0)
            self.assertEqual(row["replicate_ids"], "1;2;3")
            self.assertEqual(float(row["replicate_moments_pooled"]), 1.0)
            self.assertEqual(float(row["heldout_prediction_claim_allowed"]), 0.0)
        for key in (
            "microdynamic_closure_claim_allowed",
            "spatial_facilitation_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(gate[key]), 0.0)
            self.assertTrue(all(float(row[key]) == 0.0 for row in quality))
            self.assertTrue(all(float(row[key]) == 0.0 for row in validity))
        for key, value in gate.items():
            if key in (
                "high_block20_state",
                "high_block10_state",
                "next_minimal_model_candidate",
            ):
                continue
            self.assertTrue(math.isfinite(float(value)), key)
        svg = svg_path.read_text()
        self.assertIn("Nonlinear cage-path cumulant gate", svg)
        self.assertIn("B  Observed-cumulant Fs error", svg)
        self.assertNotIn("nan", svg.lower())
        self.assertNotIn("inf", svg.lower())
    def test_ordered_empirical_paths_close_heldout_curves_but_nulls_fail(self):
        low_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_empirical_path_verdict.csv"
        )
        high_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T058_empirical_path_verdict.csv"
        )
        crossover_path = ROOT / "data" / "renewal_cage_ka_empirical_path_crossover.csv"
        with low_path.open() as handle:
            low = {row["model"]: row for row in csv.DictReader(handle)}
        with high_path.open() as handle:
            high = {row["model"]: row for row in csv.DictReader(handle)}
        with crossover_path.open() as handle:
            crossover = next(csv.DictReader(handle))

        low_contiguous = low["contiguous_empirical_path"]
        self.assertLess(float(low_contiguous["maximum_ensemble_msd_relative_error"]), 0.056)
        self.assertLess(float(low_contiguous["maximum_ensemble_ngp_absolute_error"]), 0.030)
        self.assertLess(float(low_contiguous["maximum_ensemble_fs_absolute_error"]), 0.022)
        self.assertEqual(float(low_contiguous["curve_transfer_pass"]), 1.0)
        self.assertEqual(
            float(low_contiguous["paired_contiguous_better_replicate_count"]),
            3.0,
        )
        low_shuffle = low["within_particle_time_shuffle"]
        self.assertGreater(float(low_shuffle["maximum_ensemble_msd_relative_error"]), 6.6)
        self.assertGreater(float(low_shuffle["maximum_ensemble_ngp_absolute_error"]), 2.0)
        self.assertGreater(float(low_shuffle["maximum_ensemble_fs_absolute_error"]), 0.65)
        self.assertEqual(float(low_shuffle["shuffle_precision_pass"]), 1.0)
        self.assertEqual(float(low_shuffle["curve_transfer_pass"]), 0.0)
        self.assertEqual(float(low["direction_randomized_path"]["curve_transfer_pass"]), 0.0)

        high_contiguous = high["contiguous_empirical_path"]
        self.assertLess(float(high_contiguous["maximum_ensemble_msd_relative_error"]), 0.088)
        self.assertLess(float(high_contiguous["maximum_ensemble_ngp_absolute_error"]), 0.026)
        self.assertLess(float(high_contiguous["maximum_ensemble_fs_absolute_error"]), 0.030)
        self.assertEqual(float(high_contiguous["curve_transfer_pass"]), 1.0)
        self.assertEqual(
            float(crossover["single_particle_multiblock_path_memory_required"]),
            1.0,
        )
        self.assertEqual(float(crossover["amplitude_persistence_alone_sufficient"]), 0.0)
        self.assertEqual(float(crossover["ordered_recoil_path_required"]), 1.0)
        self.assertEqual(
            crossover["next_minimal_extension"],
            "conditional_reversible_cage_path_kernel",
        )
        for key in (
            "microdynamic_closure_claim_allowed",
            "spatial_facilitation_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(crossover[key]), 0.0)
            self.assertEqual(float(low_contiguous[key]), 0.0)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_empirical_path_crossover.svg").is_file()
        )

    def test_state_joint_kernel_closes_high_temperature_curves_but_breaks_on_cooling(self):
        low_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_state_joint_finite_gk_macro_verdict.csv"
        )
        high_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T058_state_joint_finite_gk_macro_verdict.csv"
        )
        crossover_path = ROOT / "data" / "renewal_cage_ka_state_kernel_crossover.csv"
        with low_path.open() as handle:
            low = next(csv.DictReader(handle))
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        with crossover_path.open() as handle:
            crossover = next(csv.DictReader(handle))

        self.assertLess(float(low["maximum_ensemble_msd_relative_error"]), 0.059)
        self.assertLess(float(low["diffusion_relative_error"]), 0.059)
        self.assertGreater(float(low["maximum_ensemble_ngp_absolute_error"]), 2.11)
        self.assertGreater(float(low["maximum_ensemble_fs_absolute_error"]), 0.268)
        self.assertEqual(float(low["curve_transfer_pass"]), 0.0)
        self.assertEqual(float(low["calibration_only_joint_displacement_distribution"]), 1.0)
        self.assertEqual(float(low["calibration_cage_residual_transfer"]), 0.0)
        self.assertEqual(float(low["block_direction_correlation_lag_count"]), 8.0)
        self.assertEqual(float(low["unmeasured_block_correlation_assumed_zero"]), 1.0)
        self.assertLess(float(high["maximum_ensemble_msd_relative_error"]), 0.081)
        self.assertLess(float(high["diffusion_relative_error"]), 0.081)
        self.assertLess(float(high["maximum_ensemble_ngp_absolute_error"]), 0.151)
        self.assertLess(float(high["maximum_ensemble_fs_absolute_error"]), 0.022)
        self.assertEqual(float(high["curve_transfer_pass"]), 1.0)
        self.assertEqual(float(high["alpha_crossing_ready"]), 0.0)
        self.assertEqual(
            float(crossover["cooling_induced_higher_order_memory_required"]),
            1.0,
        )
        self.assertEqual(float(crossover["additional_mobility_clock_supported"]), 0.0)
        self.assertEqual(
            crossover["next_minimal_extension"],
            "non_markov_multiblock_orientation_cage_persistence_kernel",
        )
        self.assertEqual(float(crossover["heldout_macro_closure_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_state_kernel_crossover.svg").is_file()
        )

    def test_correlated_jump_kernel_closes_low_diffusion_but_rejects_macro_independence(self):
        low_path = ROOT / "data" / "renewal_cage_ka_replicates_T045_hybrid_macro_verdict.csv"
        high_path = ROOT / "data" / "renewal_cage_ka_replicates_T058_hybrid_macro_verdict.csv"
        crossover_path = ROOT / "data" / "renewal_cage_ka_hybrid_macro_crossover.csv"
        with low_path.open() as handle:
            low = next(csv.DictReader(handle))
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        with crossover_path.open() as handle:
            crossover = next(csv.DictReader(handle))

        self.assertLess(float(low["diffusion_relative_error"]), 0.075)
        self.assertLess(float(low["maximum_count_tail_probability"]), 0.0081)
        self.assertGreater(float(low["alpha_relaxation_relative_error"]), 0.33)
        self.assertGreater(float(low["maximum_ensemble_ngp_absolute_error"]), 0.52)
        self.assertGreater(float(low["maximum_ensemble_fs_absolute_error"]), 0.075)
        self.assertEqual(float(low["calibration_correlated_jump_kernel"]), 1.0)
        self.assertEqual(float(low["jump_direction_correlation_included"]), 1.0)
        self.assertEqual(float(low["joint_macro_transfer_pass"]), 0.0)
        self.assertEqual(float(low["preregistered_heldout_prediction_claim_allowed"]), 0.0)
        self.assertGreater(float(high["diffusion_relative_error"]), 0.22)
        self.assertEqual(float(crossover["independent_count_jump_kernel_rejected"]), 1.0)
        self.assertEqual(float(crossover["additional_exchange_clock_supported"]), 0.0)
        self.assertEqual(
            crossover["next_minimal_extension"],
            "mobility_state_conditioned_jump_cage_kernel",
        )
        self.assertEqual(float(crossover["heldout_macro_closure_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_hybrid_macro_crossover.svg").is_file()
        )

    def test_three_temperature_restart_uncertainty_supports_scalar_trends_not_parent_scope(self):
        verdict_path = ROOT / "data" / "renewal_cage_ka_three_temperature_uncertainty_verdict.csv"
        trends_path = ROOT / "data" / "renewal_cage_ka_three_temperature_uncertainty_trends.csv"
        with verdict_path.open() as handle:
            verdict = next(csv.DictReader(handle))
        with trends_path.open() as handle:
            trends = list(csv.DictReader(handle))
        by_transition_metric = {
            (
                float(row["high_temperature"]),
                float(row["low_temperature"]),
                row["metric"],
            ): row
            for row in trends
        }

        self.assertEqual(verdict["restart_replicate_counts_by_temperature"], "0.45:3;0.58:5;0.7:5")
        self.assertEqual(float(verdict["physical_time_definition_consistent"]), 1.0)
        self.assertEqual(float(verdict["saved_frame_interval_tau"]), 1.0)
        self.assertEqual(float(verdict["restart_ensemble_uncertainty_ready"]), 1.0)
        self.assertEqual(float(verdict["core_scalar_uncertainty_ready"]), 1.0)
        self.assertEqual(float(verdict["core_scalar_precision_ready"]), 1.0)
        self.assertEqual(float(verdict["curve_uncertainty_ready"]), 1.0)
        self.assertEqual(float(verdict["curve_precision_ready"]), 0.0)
        self.assertEqual(
            verdict["precision_blockers"],
            "T0.45:fs_k5;T0.45:fs_k7p25;T0.45:fs_k9;T0.45:ngp_3d",
        )
        self.assertEqual(float(verdict["cooling_trend_pass_count"]), 10.0)
        self.assertEqual(float(verdict["cooling_trend_test_count"]), 10.0)
        self.assertEqual(float(verdict["three_temperature_trend_chain_pass"]), 1.0)
        self.assertEqual(float(verdict["independently_prepared_parent_ensemble_ready"]), 0.0)
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)
        first_se = by_transition_metric[(0.7, 0.58, "diffusion_alpha_product")]
        second_se = by_transition_metric[(0.58, 0.45, "diffusion_alpha_product")]
        self.assertGreater(float(first_se["ci95_low_ratio"]), 1.14)
        self.assertGreater(float(second_se["ci95_low_ratio"]), 1.94)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_three_temperature_uncertainty.svg").is_file()
        )

    def test_waiting_failure_mechanism_is_threshold_robust_but_spatially_unresolved(self):
        verdict_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_waiting_threshold_sensitivity_verdict.csv"
        )
        threshold_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_waiting_threshold_sensitivity_thresholds.csv"
        )
        with verdict_path.open() as handle:
            verdict = next(csv.DictReader(handle))
        with threshold_path.open() as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 9)
        self.assertEqual(verdict["threshold_scales"], "0.9;1;1.1")
        self.assertEqual(float(verdict["independent_replicate_count"]), 3.0)
        self.assertEqual(float(verdict["threshold_robust_dominant_mechanism"]), 1.0)
        self.assertEqual(
            verdict["dominant_mechanism"],
            "mixed_particle_environment_and_event_memory",
        )
        self.assertEqual(float(verdict["gamma_shape_misspecification_supported"]), 1.0)
        self.assertEqual(float(verdict["empirical_waiting_law_sufficient"]), 0.0)
        self.assertEqual(float(verdict["gamma_shape_misspecification_sufficient"]), 0.0)
        self.assertEqual(float(verdict["temporal_waiting_memory_supported"]), 1.0)
        self.assertEqual(float(verdict["temporal_waiting_memory_dominant"]), 0.0)
        self.assertEqual(
            float(verdict["temporal_waiting_memory_parameter_claim_allowed"]),
            0.0,
        )
        self.assertEqual(
            float(verdict["median_window_particle_conditioned_shuffle_sufficient"]),
            1.0,
        )
        self.assertEqual(
            float(verdict["all_window_particle_conditioned_shuffle_sufficient"]),
            0.0,
        )
        self.assertEqual(float(verdict["long_window_shuffle_failure_any_threshold"]), 1.0)
        self.assertEqual(float(verdict["long_window_shuffle_failure_all_thresholds"]), 0.0)
        self.assertEqual(float(verdict["persistent_particle_environment_supported"]), 1.0)
        self.assertLess(
            float(verdict["maximum_median_sequence_shuffle_relative_error"]),
            0.12,
        )
        self.assertGreater(float(verdict["maximum_sequence_shuffle_relative_error"]), 0.19)
        self.assertGreater(float(verdict["minimum_temporal_ordering_contribution_fraction"]), 0.07)
        self.assertGreater(float(verdict["minimum_particle_identity_contribution_fraction"]), 0.08)
        self.assertEqual(float(verdict["spatial_cooperation_test_required"]), 1.0)
        self.assertEqual(float(verdict["spatial_cooperation_proven"]), 0.0)
        self.assertEqual(
            verdict["minimal_model_implication"],
            "finite_exchange_particle_conditioned_renewal",
        )
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue(
            (
                ROOT
                / "figures"
                / "renewal_cage_ka_replicates_T045_waiting_threshold_sensitivity.svg"
            ).is_file()
        )

    def test_rate_anomaly_is_threshold_robust_but_remains_single_replica_evidence(self):
        verdict_path = (
            ROOT / "data" / "renewal_cage_ka_rate_threshold_sensitivity_verdict.csv"
        )
        stability_path = (
            ROOT / "data" / "renewal_cage_ka_rate_threshold_sensitivity_stability.csv"
        )
        with verdict_path.open() as handle:
            verdict = next(csv.DictReader(handle))
        with stability_path.open() as handle:
            stability = {
                int(float(row["replicate"])): row for row in csv.DictReader(handle)
            }

        self.assertEqual(verdict["threshold_scales"], "0.9;1;1.1")
        self.assertEqual(float(verdict["failed_rate_replicate"]), 3.0)
        self.assertEqual(float(verdict["failed_replicate_threshold_robust_trend"]), 1.0)
        self.assertEqual(float(verdict["jump_threshold_artifact_supported"]), 0.0)
        self.assertEqual(float(verdict["threshold_robust_rate_anomaly_detected"]), 1.0)
        self.assertLess(
            float(stability[3]["trend_amplitude_span_across_thresholds"]),
            0.03,
        )
        self.assertGreater(
            float(stability[3]["minimum_absolute_total_linear_change"]),
            0.20,
        )
        self.assertEqual(float(verdict["systematic_rate_nonstationarity_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["new_rate_state_parameter_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["macro_observable_prediction_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_rate_threshold_sensitivity.svg").is_file()
        )

    def test_six_window_rate_audit_blocks_new_rate_state_and_macro_claims(self):
        verdict_path = ROOT / "data" / "renewal_cage_ka_rate_stability_verdict.csv"
        rows_path = ROOT / "data" / "renewal_cage_ka_rate_stability_rows.csv"
        with verdict_path.open() as handle:
            verdict = next(csv.DictReader(handle))
        with rows_path.open() as handle:
            rows = list(csv.DictReader(handle))
        low = {
            int(float(row["replicate"])): row
            for row in rows
            if row["temperature_group"] == "low" and float(row["block_size"]) == 20.0
        }

        self.assertEqual(verdict["failed_absolute_rate_gate_replicates"], "3")
        self.assertEqual(float(verdict["strict_low_temperature_trend_replicate_count"]), 0.0)
        self.assertEqual(float(verdict["borderline_low_temperature_trend_replicate_count"]), 1.0)
        self.assertGreater(float(low[3]["exact_two_sided_permutation_p_value"]), 0.05)
        self.assertLess(float(low[3]["exact_two_sided_permutation_p_value"]), 0.06)
        self.assertEqual(float(verdict["systematic_rate_nonstationarity_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["new_rate_state_parameter_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["rate_stationarity_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["macro_observable_prediction_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue((ROOT / "figures" / "renewal_cage_ka_rate_stability.svg").is_file())

    def test_two_clock_hmm_hybrid_closes_low_temperature_event_shape_but_not_rate_drift(self):
        high_path = ROOT / "data" / "renewal_cage_ka_replicates_T058_two_clock_hmm_verdict.csv"
        low_path = ROOT / "data" / "renewal_cage_ka_replicates_T045_two_clock_hmm_verdict.csv"
        crossover_path = ROOT / "data" / "renewal_cage_ka_two_clock_hmm_crossover.csv"
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        with low_path.open() as handle:
            low = next(csv.DictReader(handle))
        with crossover_path.open() as handle:
            crossover = next(csv.DictReader(handle))

        self.assertEqual(float(high["second_clock_calibration_selection_fraction"]), 0.0)
        self.assertEqual(float(high["full_hybrid_transfer_pass_fraction"]), 1.0)
        self.assertEqual(float(low["second_clock_calibration_selection_fraction"]), 1.0)
        self.assertEqual(float(low["conditional_shape_distribution_pass_fraction"]), 1.0)
        self.assertAlmostEqual(float(low["full_hybrid_transfer_pass_fraction"]), 4.0 / 6.0)
        self.assertLess(float(low["maximum_fano_relative_error"]), 0.092)
        self.assertLess(float(low["maximum_count_tv_distance"]), 0.020)
        self.assertLess(float(low["maximum_identity_rmse"]), 0.037)
        self.assertLess(float(low["maximum_pair_tv_distance"]), 0.030)
        self.assertEqual(float(crossover["cooling_induced_hybrid_clock_selected"]), 1.0)
        self.assertEqual(float(crossover["conditional_event_shape_crossover_closure"]), 1.0)
        self.assertEqual(float(crossover["absolute_rate_drift_unresolved"]), 1.0)
        self.assertEqual(crossover["rate_drift_replicates"], "3")
        self.assertEqual(float(crossover["macro_observable_prediction_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_two_clock_hmm_crossover.svg").is_file()
        )

    def test_gamma_refresh_cox_clock_closes_heldout_count_moments_across_cooling(self):
        high_path = ROOT / "data" / "renewal_cage_ka_replicates_T058_gamma_refresh_cox_verdict.csv"
        low_path = ROOT / "data" / "renewal_cage_ka_replicates_T045_gamma_refresh_cox_verdict.csv"
        crossover_path = ROOT / "data" / "renewal_cage_ka_gamma_refresh_crossover.csv"
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        with low_path.open() as handle:
            low = next(csv.DictReader(handle))
        with crossover_path.open() as handle:
            crossover = next(csv.DictReader(handle))

        self.assertEqual(
            high["event_level_outcome"],
            "single_clock_gamma_refresh_count_moment_closure",
        )
        self.assertEqual(float(high["single_clock_transfer_pass_fraction"]), 1.0)
        self.assertEqual(
            low["event_level_outcome"],
            "two_clock_gamma_refresh_conditional_shape_closure",
        )
        self.assertEqual(float(low["two_clock_selection_fraction"]), 1.0)
        self.assertAlmostEqual(float(low["two_clock_transfer_pass_fraction"]), 4.0 / 6.0)
        self.assertEqual(
            float(low["conditional_moment_marginal_memory_pass_fraction"]),
            1.0,
        )
        self.assertEqual(float(low["count_moment_closure_claim_allowed"]), 0.0)
        self.assertLess(float(low["maximum_heldout_fano_relative_error"]), 0.024)
        self.assertLess(float(low["maximum_two_clock_identity_rmse"]), 0.024)
        self.assertLess(float(low["maximum_heldout_count_tv_distance"]), 0.03)
        self.assertEqual(float(low["marginal_count_distribution_claim_allowed"]), 1.0)
        self.assertAlmostEqual(float(low["gamma_pair_distribution_pass_fraction"]), 4.0 / 6.0)
        self.assertEqual(float(low["hmm_pair_distribution_pass_fraction"]), 1.0)
        self.assertEqual(float(low["joint_count_pair_distribution_claim_allowed"]), 0.0)
        self.assertEqual(float(low["hybrid_semimarkov_emission_model_required"]), 1.0)
        self.assertEqual(float(low["full_count_sequence_likelihood_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["cooling_induced_second_refresh_clock_required"]), 1.0)
        self.assertEqual(float(crossover["conditional_shape_crossover_closure"]), 1.0)
        self.assertEqual(float(crossover["count_moment_crossover_closure"]), 0.0)
        self.assertEqual(float(crossover["positive_intensity_generator"]), 1.0)
        self.assertEqual(float(crossover["finite_recovery_enforced"]), 1.0)
        self.assertEqual(
            float(crossover["marginal_count_distribution_crossover_closure"]),
            1.0,
        )
        self.assertLess(float(crossover["low_maximum_count_tv_distance"]), 0.03)
        self.assertEqual(
            float(crossover["full_count_sequence_likelihood_claim_allowed"]),
            0.0,
        )
        self.assertAlmostEqual(float(crossover["low_gamma_pair_pass_fraction"]), 4.0 / 6.0)
        self.assertEqual(float(crossover["low_hmm_pair_pass_fraction"]), 1.0)
        self.assertEqual(float(crossover["joint_count_pair_crossover_closure"]), 0.0)
        self.assertEqual(float(crossover["hybrid_semimarkov_emission_model_required"]), 1.0)
        self.assertEqual(float(crossover["full_count_distribution_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["heldout_macro_prediction_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_gamma_refresh_crossover.svg").is_file()
        )

    def test_cooling_selects_two_mode_finite_exchange_spectrum_without_full_closure(self):
        high_path = ROOT / "data" / "renewal_cage_ka_replicates_T058_exchange_spectrum_verdict.csv"
        low_path = ROOT / "data" / "renewal_cage_ka_replicates_T045_exchange_spectrum_verdict.csv"
        crossover_path = ROOT / "data" / "renewal_cage_ka_exchange_spectrum_crossover.csv"
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        with low_path.open() as handle:
            low = next(csv.DictReader(handle))
        with crossover_path.open() as handle:
            crossover = next(csv.DictReader(handle))

        self.assertEqual(high["event_level_outcome"], "single_mode_exchange_sufficient")
        self.assertEqual(float(high["two_mode_calibration_selection_fraction"]), 0.0)
        self.assertEqual(
            low["event_level_outcome"],
            "two_mode_finite_exchange_spectrum_closure",
        )
        self.assertEqual(float(low["two_mode_calibration_selection_fraction"]), 1.0)
        self.assertEqual(float(low["two_mode_heldout_transfer_pass_fraction"]), 1.0)
        self.assertEqual(float(low["identity_spectrum_closure_claim_allowed"]), 1.0)
        self.assertEqual(float(low["full_event_clock_closure_claim_allowed"]), 0.0)
        self.assertEqual(low["excluded_underidentified_block_sizes"], "100")
        self.assertEqual(float(crossover["low_two_mode_pass_count"]), 6.0)
        self.assertEqual(float(crossover["low_markov_hmm_pass_count"]), 1.0)
        self.assertEqual(float(crossover["cooling_induced_exchange_spectrum_broadening"]), 1.0)
        self.assertEqual(float(crossover["semi_markov_generator_required"]), 1.0)
        self.assertEqual(float(crossover["identified_resolution_closure"]), 1.0)
        self.assertEqual(float(crossover["full_resolution_scope"]), 0.0)
        self.assertEqual(float(crossover["full_event_clock_closure_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["heldout_macro_prediction_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_exchange_spectrum_crossover.svg").is_file()
        )

    def test_finite_exchange_hmm_is_sufficient_high_but_requires_broad_low_spectrum(self):
        high_path = (
            ROOT / "data" / "renewal_cage_ka_replicates_T058_finite_exchange_hmm_verdict.csv"
        )
        low_path = (
            ROOT / "data" / "renewal_cage_ka_replicates_T045_finite_exchange_hmm_verdict.csv"
        )
        crossover_path = ROOT / "data" / "renewal_cage_ka_finite_exchange_hmm_crossover.csv"
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        with low_path.open() as handle:
            low = next(csv.DictReader(handle))
        with crossover_path.open() as handle:
            crossover = next(csv.DictReader(handle))

        self.assertEqual(float(high["replica_block_pass_fraction"]), 1.0)
        self.assertEqual(float(high["two_state_poisson_hmm_sufficient"]), 1.0)
        self.assertEqual(float(high["scoring_horizon"]), 600.0)
        self.assertEqual(float(low["scoring_horizon"]), 600.0)
        self.assertLess(float(low["replica_block_pass_fraction"]), 0.25)
        self.assertEqual(float(low["non_single_exponential_exchange_required"]), 1.0)
        self.assertEqual(float(crossover["high_positive_late_excess_block_count"]), 0.0)
        self.assertEqual(float(crossover["low_positive_late_excess_block_count"]), 3.0)
        self.assertEqual(float(crossover["finite_exchange_spectrum_broadening_detected"]), 1.0)
        self.assertEqual(float(crossover["spatial_facilitation_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["heldout_macro_prediction_claim_allowed"]), 0.0)
        self.assertEqual(float(crossover["thermodynamic_claim_allowed"]), 0.0)
        self.assertTrue(
            (ROOT / "figures" / "renewal_cage_ka_finite_exchange_hmm_crossover.svg").is_file()
        )

    def test_calibration_jump_cage_channels_transfer_ensemble_observables_without_macro_fit(self):
        path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_calibration_channel_transfer_verdict.csv"
        )
        with path.open() as handle:
            verdict = next(csv.DictReader(handle))
        high_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T058_calibration_channel_transfer_verdict.csv"
        )
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        self.assertEqual(float(verdict["retrospective_ensemble_transfer_candidate_pass"]), 1.0)
        self.assertLess(float(verdict["maximum_ensemble_msd_relative_error"]), 0.1)
        self.assertLess(float(verdict["maximum_ensemble_ngp_absolute_error"]), 0.3)
        self.assertLess(float(verdict["maximum_ensemble_fs_absolute_error"]), 0.03)
        self.assertEqual(float(verdict["lag_count"]), 9.0)
        self.assertEqual(float(verdict["wave_number_count"]), 3.0)
        self.assertEqual(float(verdict["calibration_only_microchannel_input"]), 1.0)
        self.assertEqual(float(verdict["heldout_events_used_in_prediction"]), 0.0)
        self.assertEqual(float(verdict["macro_fit_parameter_count"]), 0.0)
        self.assertEqual(float(verdict["individual_trajectory_forecast_pass"]), 0.0)
        self.assertEqual(float(verdict["early_ngp_significant_mismatch_lag_count"]), 3.0)
        self.assertEqual(float(verdict["preregistered_heldout_prediction_claim_allowed"]), 0.0)
        self.assertEqual(
            verdict["next_required_test"],
            "new_independent_trajectory_preregistered_channel_transfer",
        )
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(float(high["retrospective_ensemble_transfer_candidate_pass"]), 0.0)
        self.assertGreater(float(high["maximum_ensemble_msd_relative_error"]), 0.3)
        crossover_path = ROOT / "data" / "renewal_cage_ka_channel_transfer_crossover.csv"
        with crossover_path.open() as handle:
            crossover = next(csv.DictReader(handle))
        self.assertEqual(
            float(crossover["jump_cage_scale_separation_emerges_on_cooling"]),
            1.0,
        )
        self.assertEqual(float(crossover["heldout_events_used_in_prediction"]), 0.0)
        self.assertEqual(
            float(crossover["preregistered_heldout_prediction_claim_allowed"]),
            0.0,
        )

    def test_oracle_jump_cage_factorization_closes_low_temperature_multobservables(self):
        low_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_event_oracle_factorization_verdict.csv"
        )
        high_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T058_event_oracle_factorization_verdict.csv"
        )
        with low_path.open() as handle:
            low = next(csv.DictReader(handle))
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        self.assertEqual(float(low["oracle_jump_cage_factorization_supported"]), 1.0)
        self.assertLess(float(low["maximum_msd_relative_error"]), 0.1)
        self.assertLess(float(low["maximum_ngp_absolute_error"]), 0.3)
        self.assertLess(float(low["maximum_fs_absolute_error"]), 0.03)
        self.assertEqual(float(low["independent_replicate_count"]), 3.0)
        self.assertEqual(float(low["lag_count"]), 9.0)
        self.assertEqual(float(low["wave_number_count"]), 3.0)
        self.assertEqual(float(high["oracle_jump_cage_factorization_supported"]), 0.0)
        self.assertGreater(float(high["maximum_msd_relative_error"]), 0.3)
        self.assertEqual(float(low["calibration_prediction_claim_allowed"]), 0.0)
        self.assertEqual(low["gate_scope"], "posthoc_oracle_representation_diagnostic")
        self.assertEqual(float(low["thermodynamic_claim_allowed"]), 0.0)

    def test_debye_waller_waiting_law_crosses_from_iid_to_persistent_environment(self):
        high_path = (
            ROOT / "data" / "renewal_cage_ka_replicates_T058_debye_waller_waiting_verdict.csv"
        )
        low_path = (
            ROOT / "data" / "renewal_cage_ka_replicates_T045_debye_waller_waiting_verdict.csv"
        )
        with high_path.open() as handle:
            high = next(csv.DictReader(handle))
        with low_path.open() as handle:
            low = next(csv.DictReader(handle))
        self.assertEqual(high["consensus_verdict"], "empirical_iid_waiting_law_sufficient")
        self.assertEqual(low["consensus_verdict"], "persistent_particle_environment_required")
        self.assertEqual(float(high["independent_replicate_count"]), 5.0)
        self.assertEqual(float(low["independent_replicate_count"]), 3.0)
        self.assertEqual(float(low["persistent_environment_identifiable"]), 1.0)
        self.assertEqual(float(low["finite_memory_model_required"]), 0.0)
        self.assertGreater(
            float(low["mean_median_persistent_environment_excess_fraction"]),
            2.8,
        )

    def test_mobility_identity_decay_selects_finite_exchange_over_static_disorder(self):
        verdict_path = (
            ROOT / "data" / "renewal_cage_ka_debye_waller_environment_crossover_verdict.csv"
        )
        growth_path = (
            ROOT / "data" / "renewal_cage_ka_debye_waller_environment_crossover_growth.csv"
        )
        with verdict_path.open() as handle:
            verdict = next(csv.DictReader(handle))
        with growth_path.open() as handle:
            growth = list(csv.DictReader(handle))
        self.assertEqual({float(row["block_size"]) for row in growth}, {20.0, 50.0, 100.0})
        self.assertTrue(all(float(row["growth_detected"]) == 1.0 for row in growth))
        self.assertTrue(all(float(row["ci95_low_ratio"]) > 1.0 for row in growth))
        self.assertEqual(float(verdict["waiting_mechanism_crossover_detected"]), 1.0)
        self.assertEqual(float(verdict["pure_static_particle_rate_disorder_rejected"]), 1.0)
        self.assertEqual(float(verdict["finite_exchange_environment_claim_allowed"]), 1.0)
        self.assertEqual(float(verdict["finite_waiting_sequence_memory_required"]), 0.0)
        self.assertEqual(float(verdict["spatial_facilitation_claim_allowed"]), 0.0)
        self.assertGreater(float(verdict["minimum_exchange_time_growth_ci95_low"]), 1.3)
        self.assertGreater(
            float(verdict["cross_half_identity_correlation_growth_ci95_low"]),
            1.6,
        )
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)

    def test_debye_waller_event_clock_closes_high_temperature_but_not_every_low_replica(self):
        high_summary = (
            ROOT / "data" / "renewal_cage_ka_replicates_T058_debye_waller_heldout_summary.csv"
        )
        low_summary = (
            ROOT / "data" / "renewal_cage_ka_replicates_T045_debye_waller_heldout_summary.csv"
        )
        with high_summary.open() as handle:
            high = {row["model"]: row for row in csv.DictReader(handle)}
        with low_summary.open() as handle:
            low = {row["model"]: row for row in csv.DictReader(handle)}
        high_dw = high["debye_waller_direction_correlated"]
        low_dw = low["debye_waller_direction_correlated"]
        self.assertEqual(float(high_dw["replicates_within_tolerance"]), 5.0)
        self.assertGreater(float(high_dw["mean_coverage"]), 0.8)
        self.assertLess(float(high_dw["mean_coverage"]), 1.2)
        self.assertEqual(float(low_dw["replicates_within_tolerance"]), 1.0)
        self.assertAlmostEqual(float(low_dw["mean_coverage"]), 0.9779045410378439)
        self.assertGreater(
            float(low["debye_waller_uncorrelated"]["mean_coverage"]),
            1.9,
        )
        self.assertLess(float(low["fixed_phop_direction_correlated"]["mean_coverage"]), 0.5)

    def test_debye_waller_directional_crossover_is_sota_aligned_but_full_closure_stays_blocked(self):
        verdict_path = ROOT / "data" / "renewal_cage_ka_debye_waller_crossover_verdict.csv"
        convergence_path = ROOT / "data" / "renewal_cage_ka_debye_waller_crossover_convergence.csv"
        with verdict_path.open() as handle:
            verdict = next(csv.DictReader(handle))
        with convergence_path.open() as handle:
            convergence = {row["temperature_group"]: row for row in csv.DictReader(handle)}
        self.assertEqual(float(verdict["lag1_positive_to_negative_reversal"]), 1.0)
        self.assertGreater(float(verdict["lag1_high_ci95_low_over_q"]), 0.0)
        self.assertLess(float(verdict["lag1_low_ci95_high_over_q"]), 0.0)
        self.assertEqual(float(verdict["low_temperature_backtracking_required"]), 1.0)
        self.assertEqual(float(verdict["directional_crossover_claim_allowed"]), 1.0)
        self.assertEqual(float(verdict["cross_temperature_transport_closure_claim_allowed"]), 0.0)
        self.assertEqual(
            verdict["primary_remaining_failure"],
            "low_temperature_replicate_transport_inconsistency",
        )
        self.assertTrue(
            all(
                float(row["equivalent_within_twenty_percent"]) == 1.0
                for row in convergence.values()
            )
        )
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)

    def test_phop_cage_jump_sota_audit_exposes_reversed_decoupling_order(self):
        path = ROOT / "data" / "renewal_cage_ka_cage_jump_sota_audit_verdict.csv"
        with path.open() as handle:
            verdict = next(csv.DictReader(handle))
        self.assertEqual(float(verdict["low_temperature_ctwr_failure_consistent"]), 1.0)
        self.assertEqual(float(verdict["microscopic_invariant_consistent"]), 1.0)
        self.assertEqual(float(verdict["relative_decoupling_order_consistent"]), 0.0)
        self.assertEqual(float(verdict["current_phop_elementary_cage_jump_claim_allowed"]), 0.0)
        self.assertEqual(
            verdict["primary_mismatch"],
            "microscopic_decoupling_weaker_than_macroscopic",
        )

    def test_t045_halo_profile_is_stable_but_transport_closure_is_rejected(self):
        curve_path = (
            ROOT / "data" / "renewal_cage_ka_replicates_T045_halo_split_stability_curve.csv"
        )
        verdict_path = (
            ROOT / "data" / "renewal_cage_ka_replicates_T045_halo_split_stability_verdict.csv"
        )
        with curve_path.open() as handle:
            curve = list(csv.DictReader(handle))
        self.assertEqual(len(curve), 8)
        self.assertTrue(
            all(float(row["paired_difference_ci_includes_zero"]) == 1.0 for row in curve)
        )
        with verdict_path.open() as handle:
            verdict = next(csv.DictReader(handle))
        self.assertEqual(float(verdict["paired_shift_not_detected"]), 1.0)
        self.assertEqual(float(verdict["paired_curve_equivalent"]), 1.0)
        self.assertEqual(float(verdict["relative_equivalence_margin"]), 0.2)
        self.assertEqual(float(verdict["binary_radius_gate_stable"]), 0.0)
        self.assertEqual(
            verdict["radius_difference_interpretation"],
            "ci_significance_boundary_not_profile_shift",
        )
        self.assertEqual(float(verdict["closure_rejection_robust_to_radius_boundary"]), 1.0)
        self.assertEqual(float(verdict["spatial_measurement_claim_allowed"]), 1.0)
        self.assertEqual(float(verdict["spatial_model_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)

    def test_t045_radius4_closure_is_posthoc_and_cannot_enable_model_claim(self):
        path = (
            ROOT
            / "data"
            / "renewal_cage_ka_replicates_T045_cooperative_closure_radius4_sensitivity_verdict.csv"
        )
        with path.open() as handle:
            verdict = next(csv.DictReader(handle))
        self.assertEqual(verdict["halo_radius_source"], "posthoc_sensitivity")
        self.assertEqual(float(verdict["posthoc_sensitivity_only"]), 1.0)
        self.assertEqual(float(verdict["heldout_transport_pass"]), 0.0)
        self.assertEqual(float(verdict["spatial_model_claim_allowed"]), 0.0)

    def test_t045_neighbor_halo_is_replicated_but_model_claim_stays_blocked(self):
        curve_path = (
            ROOT / "data" / "renewal_cage_ka_replicates_T045_neighbor_halo_curve_summary.csv"
        )
        verdict_path = ROOT / "data" / "renewal_cage_ka_replicates_T045_neighbor_halo_verdict.csv"
        with curve_path.open() as handle:
            curve = list(csv.DictReader(handle))
        self.assertEqual([float(row["halo_detected_in_shell"]) for row in curve[:4]], [1.0] * 4)
        self.assertEqual(float(curve[4]["halo_detected_in_shell"]), 0.0)
        self.assertGreater(float(curve[0]["ci95_low"]), 4.0)
        self.assertGreater(float(curve[3]["ci95_low"]), 1.0)
        self.assertLess(float(curve[4]["ci95_low"]), 1.0)
        self.assertTrue(all(row["ci95_method"] == "student_t_independent_replicates" for row in curve))
        with verdict_path.open() as handle:
            verdict = next(csv.DictReader(handle))
        self.assertEqual(float(verdict["halo_radius_lower_bound"]), 4.0)
        self.assertEqual(float(verdict["independent_replicate_count"]), 3.0)
        self.assertGreater(
            float(verdict["ci95_low_integrated_neighbor_excess_over_self_jump_squared"]),
            4.0,
        )
        self.assertEqual(float(verdict["spatial_measurement_claim_allowed"]), 1.0)
        self.assertEqual(float(verdict["spatial_model_claim_allowed"]), 0.0)
        self.assertEqual(float(verdict["thermodynamic_claim_allowed"]), 0.0)

    def test_neighbor_halo_sota_ledger_blocks_numeric_and_xi4_overclaim(self):
        path = ROOT / "data" / "renewal_cage_ka_neighbor_halo_sota_alignment.csv"
        with path.open() as handle:
            rows = {row["source_id"]: row for row in csv.DictReader(handle)}
        self.assertEqual(
            set(rows),
            {
                "gokhale_nagamanasa_ganapathy_sood_2014",
                "pastore_coniglio_pica_ciamarra_2015",
                "ortlieb_royall_et_al_2023",
                "keys_hedges_garrahan_glotzer_chandler_2011",
            },
        )
        self.assertTrue(
            all(float(row["numerical_comparison_allowed"]) == 0.0 for row in rows.values())
        )
        self.assertTrue(all(float(row["thermodynamic_claim_allowed"]) == 0.0 for row in rows.values()))
        self.assertIn("direct_model_and_temperature", rows["pastore_coniglio_pica_ciamarra_2015"]["definition_alignment"])
        self.assertIn("different_dimension", rows["gokhale_nagamanasa_ganapathy_sood_2014"]["definition_alignment"])

    def test_t045_independent_waiting_and_heldout_transport_gates(self):
        waiting_path = ROOT / "data" / "renewal_cage_ka_replicates_T045_waiting_verdict.csv"
        with waiting_path.open() as handle:
            waiting = next(csv.DictReader(handle))
        self.assertEqual(waiting["consensus_verdict"], "empirical_iid_waiting_law_sufficient")
        self.assertEqual(float(waiting["independent_replicate_count"]), 3.0)
        self.assertEqual(float(waiting["finite_memory_model_required"]), 0.0)
        self.assertEqual(float(waiting["persistent_environment_identifiable"]), 0.0)
        self.assertEqual(float(waiting["collective_covariance_replicate_count"]), 3.0)
        self.assertEqual(float(waiting["spatial_cooperation_test_required"]), 1.0)
        self.assertEqual(float(waiting["spatial_cooperation_proven"]), 0.0)

        paths = {
            0.15: ROOT / "data" / "renewal_cage_ka_replicates_T045_heldout_transport_threshold_0p15_summary.csv",
            0.20: ROOT / "data" / "renewal_cage_ka_replicates_T045_heldout_transport_summary.csv",
            0.25: ROOT / "data" / "renewal_cage_ka_replicates_T045_heldout_transport_threshold_0p25_summary.csv",
        }
        coverage = {}
        for threshold, path in paths.items():
            with path.open() as handle:
                rows = {row["model"]: row for row in csv.DictReader(handle)}
            correlated = rows["correlated_event_clock"]
            coverage[threshold] = float(correlated["mean_coverage"])
            self.assertEqual(float(correlated["replicates_within_tolerance"]), 0.0)
            self.assertLess(float(correlated["ci95_high_coverage"]), 0.8)
            self.assertEqual(float(correlated["independent_replicate_count"]), 3.0)
        self.assertGreater(coverage[0.15], coverage[0.20])
        self.assertGreater(coverage[0.20], coverage[0.25])

    def test_t045_spatial_length_replicates_but_overlap_xi4_does_not(self):
        spatial_path = (
            ROOT / "data" / "renewal_cage_ka_replicates_T045_spatial_covariance_ensemble_fit_summary.csv"
        )
        with spatial_path.open() as handle:
            spatial = next(csv.DictReader(handle))
        self.assertEqual(float(spatial["independent_replicate_count"]), 3.0)
        self.assertGreater(float(spatial["ci95_low_correlation_length"]), 0.4)
        self.assertLess(float(spatial["ci95_high_correlation_length"]), 1.0)
        self.assertEqual(spatial["ci95_method"], "student_t_independent_replicates")
        self.assertEqual(float(spatial["spatial_measurement_claim_allowed"]), 1.0)
        self.assertEqual(float(spatial["spatial_model_claim_allowed"]), 0.0)

        s4_path = ROOT / "data" / "renewal_cage_ka_replicates_T045_overlap_s4_ensemble_verdict.csv"
        with s4_path.open() as handle:
            s4 = next(csv.DictReader(handle))
        self.assertEqual(float(s4["independent_replicate_count"]), 3.0)
        self.assertEqual(float(s4["invalid_replicate_fit_count"]), 3.0)
        self.assertEqual(float(s4["xi4_identifiable"]), 0.0)
        self.assertEqual(float(s4["xi4_claim_allowed"]), 0.0)

    def test_three_temperature_replicate_trends_keep_chi4_uncertainty_failure(self):
        path = ROOT / "data" / "renewal_cage_ka_replicates_T058_T045_trend.csv"
        with path.open() as handle:
            rows = {row["metric"]: row for row in csv.DictReader(handle)}
        for metric in ("diffusion", "alpha_relaxation_time", "diffusion_alpha_product", "ngp_peak"):
            self.assertEqual(rows[metric]["trend_pass"], "True")
        self.assertEqual(rows["overlap_chi4_peak"]["trend_pass"], "False")
        self.assertEqual(float(rows["diffusion"]["high_temperature_replicate_count"]), 5.0)
        self.assertEqual(float(rows["diffusion"]["low_temperature_replicate_count"]), 3.0)
        self.assertTrue(all(row["thermodynamic_claim_allowed"] == "False" for row in rows.values()))

    def test_c3_microscopic_tangent_control_is_complete_and_claim_limited(self):
        document_path = ROOT / "docs" / "microscopic-c3-switched-tangent-control.md"
        control_path = ROOT / "data" / "renewal_cage_ka_c3_tangent_control_T058_summary.csv"
        fidelity_path = ROOT / "data" / "renewal_cage_ka_c3_physical_fidelity_T058_summary.csv"
        self.assertTrue(document_path.is_file())
        document = document_path.read_text()
        for required in (
            "ka_lj_c3_switch",
            "5.914e-15",
            "320/320",
            "generator_second",
            "renewal hazard",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with control_path.open() as handle:
            control = next(csv.DictReader(handle))
        self.assertEqual(control["potential_protocol"], "ka_lj_c3_switch")
        self.assertEqual(int(control["minimum_valid_intervals_per_epsilon"]), 320)
        self.assertEqual(int(control["maximum_right_censored_intervals_per_epsilon"]), 0)
        self.assertEqual(control["microscopic_tangent_covariance_gate_pass"], "True")
        self.assertEqual(control["all_original_design_gates_pass"], "False")
        self.assertEqual(control["thermodynamic_claim_allowed"], "False")

        with fidelity_path.open() as handle:
            fidelity = next(csv.DictReader(handle))
        self.assertLess(float(fidelity["scaled_pair_histogram_total_variation"]), 0.02)
        self.assertLess(abs(float(fidelity["force_norm_mean_relative_difference"])), 0.02)
        self.assertEqual(fidelity["thermodynamic_claim_allowed"], "False")

    def test_second_generator_response_is_complete_and_claim_limited(self):
        document_path = (
            ROOT / "docs" / "microscopic-second-generator-krylov-response.md"
        )
        summary_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_second_generator_response_T058_summary.csv"
        )
        curve_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_second_generator_response_T058_curve.csv"
        )
        for path in (document_path, summary_path, curve_path):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "32 matched paths",
            "0.17569",
            "0.23290",
            "second_generator_response_allowed = 0",
            "one_tau_generator_response_allowed = 0",
            "autonomous_stochastic_single_particle_gle_allowed = 0",
            "event_clock_claim_allowed = 0",
            "kramers_escape_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
            "explicit slow bath or state-dependent memory",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["integrity_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["primary_fit_time"]), 0.2)
        for key in (
            "second_generator_response_allowed",
            "one_tau_generator_response_allowed",
            "autonomous_stochastic_single_particle_gle_allowed",
            "event_clock_claim_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

        primary = {
            (row["model"], row["evaluation_epsilon"]): row
            for row in rows
            if row["record"] == "aggregate_gate"
            and row["fit_time"] == "0.2"
            and row["horizon_time"] == "0.2"
        }
        for epsilon in ("0.001", "0.002"):
            first = primary[("first_generator_constrained", epsilon)]
            second = primary[("second_generator_constrained", epsilon)]
            self.assertEqual(int(second["identified_fold_count"]), 8)
            self.assertEqual(int(second["evaluable_fold_count"]), 8)
            self.assertLess(float(first["position_relative_l2_error"]), 0.18)
            self.assertGreater(float(second["position_relative_l2_error"]), 0.23)
            self.assertLess(float(second["paired_improvement_fraction"]), -0.61)
            self.assertEqual(float(second["all_identified_folds_pass"]), 0.0)

    def test_hankel_slow_force_bath_is_complete_and_claim_limited(self):
        document_path = ROOT / "docs" / "microscopic-hankel-slow-force-bath.md"
        summary_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_hankel_slow_force_bath_long_T058_summary.csv"
        )
        detail_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_hankel_slow_force_bath_long_T058_details.csv"
        )
        curve_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_hankel_slow_force_bath_long_T058_curve.csv"
        )
        extension_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_hankel_slow_force_bath_rank_extension_T058_summary.csv"
        )
        for path in (document_path, summary_path, detail_path, curve_path, extension_path):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "0.87341",
            "0.93739",
            "21.2371",
            "17.1130",
            "rank 16",
            "hankel_slow_force_bath_allowed = 0",
            "state_dependent_memory_allowed = 0",
            "complete_event_clock_closure_allowed = 0",
            "kramers_escape_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
            "nonlinear state-dependent memory",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        models = {row["model"]: row for row in rows if row["record"] == "aggregate_model"}
        self.assertEqual(
            set(models),
            {
                "raw_force_delay_2",
                "hankel_slow_2",
                "hankel_slow_4",
                "hankel_slow_8",
                "hankel_slow_16",
            },
        )
        primary = models["hankel_slow_8"]
        self.assertEqual(int(float(primary["held_clone_count"])), 4)
        self.assertGreater(float(primary["captured_force_history_variance_fraction"]), 0.87)
        self.assertLess(float(primary["maximum_maximum_held_residual_state_correlation"]), 0.13)
        self.assertGreater(float(primary["maximum_maximum_held_residual_lag_correlation"]), 0.93)
        self.assertGreater(float(primary["terminal_diffusion_relative_error"]), 21.0)
        self.assertGreater(float(primary["event_rate_relative_error"]), 17.0)
        self.assertGreater(float(models["hankel_slow_16"]["terminal_diffusion_relative_error"]), 3.0)

        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["integrity_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["numerical_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["orthogonality_gate_pass"]), 0.0)
        self.assertEqual(float(verdict["macro_event_gate_pass"]), 0.0)
        self.assertEqual(float(verdict["hankel_slow_force_bath_allowed"]), 0.0)
        for key in (
            "state_dependent_memory_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

        with extension_path.open() as handle:
            extension = {
                int(float(row["slow_mode_count"])): row
                for row in csv.DictReader(handle)
                if row["record"] == "aggregate_model" and float(row["slow_mode_count"]) > 0
            }
        self.assertEqual(set(extension), {24, 32, 48, 64})
        self.assertGreater(float(extension[64]["terminal_diffusion_relative_error"]), 4.0)

    def test_bilinear_state_dependent_memory_is_complete_and_not_promoted(self):
        document_path = ROOT / "docs" / "microscopic-bilinear-state-dependent-memory.md"
        summary_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_bilinear_state_dependent_memory_T058_summary.csv"
        )
        detail_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_bilinear_state_dependent_memory_T058_details.csv"
        )
        sensitivity_path = (
            ROOT
            / "data"
            / "renewal_cage_ka_bilinear_state_dependent_memory_T058_ridge_sensitivity.csv"
        )
        for path in (document_path, summary_path, detail_path, sensitivity_path):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "0.02182",
            "0.86078",
            "1.00407",
            "None of the four held",
            "teacher_forced_state_dependent_memory_gate_pass = 0",
            "autonomous_state_dependent_gle_allowed = 0",
            "complete_event_clock_closure_allowed = 0",
            "kramers_escape_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        models = {row["model"]: row for row in rows if row["record"] == "aggregate_model"}
        self.assertEqual(
            set(models),
            {"stationary_rank16", "bilinear_energy", "bilinear_energy_power"},
        )
        baseline = models["stationary_rank16"]
        full = models["bilinear_energy_power"]
        self.assertEqual(int(float(full["held_clone_count"])), 4)
        self.assertLess(
            float(full["maximum_maximum_held_residual_state_correlation"]),
            0.022,
        )
        self.assertGreater(
            float(full["maximum_maximum_held_residual_lag_correlation"]),
            float(baseline["maximum_maximum_held_residual_lag_correlation"]),
        )

        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["integrity_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["velocity_prediction_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["residual_state_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["residual_lag_gate_pass"]), 0.0)
        self.assertEqual(float(verdict["every_fold_lag_improves"]), 0.0)
        self.assertGreater(float(verdict["bilinear_to_baseline_lag_ratio"]), 1.0)
        for key in (
            "teacher_forced_state_dependent_memory_gate_pass",
            "autonomous_state_dependent_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

        with sensitivity_path.open() as handle:
            sensitivity = list(csv.DictReader(handle))
        self.assertEqual(len(sensitivity), 5)
        self.assertTrue(
            all(float(row["teacher_forced_state_dependent_memory_gate_pass"]) == 0.0 for row in sensitivity)
        )
        self.assertTrue(
            all(float(row["every_fold_lag_improves"]) == 0.0 for row in sensitivity)
        )

    def test_smooth_cage_hankel_bath_is_complete_and_not_promoted(self):
        document_path = ROOT / "docs" / "microscopic-smooth-cage-hankel-bath.md"
        summary_path = (
            ROOT / "data" / "renewal_cage_ka_smooth_cage_hankel_bath_T058_summary.csv"
        )
        detail_path = (
            ROOT / "data" / "renewal_cage_ka_smooth_cage_hankel_bath_T058_details.csv"
        )
        curve_path = (
            ROOT / "data" / "renewal_cage_ka_smooth_cage_hankel_bath_T058_curve.csv"
        )
        for path in (document_path, summary_path, detail_path, curve_path):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "0.03973",
            "0.32131",
            "0.85589",
            "0.99836",
            "3.6344",
            "1.8431",
            "smooth_cage_hankel_bath_allowed = 0",
            "autonomous_single_particle_gle_allowed = 0",
            "complete_event_clock_closure_allowed = 0",
            "kramers_escape_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        models = {row["model"]: row for row in rows if row["record"] == "aggregate_model"}
        self.assertEqual(
            set(models),
            {
                "hankel_slow_16",
                "hankel_slow_16_position",
                "hankel_slow_16_position_velocity",
            },
        )
        baseline = models["hankel_slow_16"]
        primary = models["hankel_slow_16_position_velocity"]
        self.assertEqual(int(float(primary["held_clone_count"])), 4)
        self.assertLess(
            float(primary["maximum_maximum_held_velocity_residual_lag_correlation"]),
            float(baseline["maximum_maximum_held_velocity_residual_lag_correlation"]),
        )
        self.assertGreater(
            float(primary["maximum_maximum_held_force_residual_lag_correlation"]),
            0.85,
        )
        self.assertGreater(float(primary["terminal_diffusion_relative_error"]), 3.6)
        self.assertGreater(float(primary["event_rate_relative_error"]), 1.8)
        self.assertLess(float(primary["normalized_trapezoidal_kinematic_error"]), 0.04)

        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["integrity_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["numerical_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["every_fold_lag_improves"]), 1.0)
        self.assertEqual(float(verdict["residual_memory_gate_pass"]), 0.0)
        self.assertEqual(float(verdict["macro_event_gate_pass"]), 0.0)
        self.assertGreater(float(verdict["primary_to_baseline_lag_ratio"]), 0.99)
        for key in (
            "smooth_cage_hankel_bath_allowed",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

    def test_decomposed_cage_drift_bath_is_complete_and_not_promoted(self):
        stem = "renewal_cage_ka_decomposed_cage_drift_bath_T058"
        document_path = ROOT / "docs" / "microscopic-decomposed-cage-drift-bath.md"
        summary_path = ROOT / "data" / f"{stem}_summary.csv"
        detail_path = ROOT / "data" / f"{stem}_details.csv"
        curve_path = ROOT / "data" / f"{stem}_curve.csv"
        script_path = ROOT / "scripts" / "analyze_ka_decomposed_cage_drift_bath.py"
        for path in (document_path, summary_path, detail_path, curve_path, script_path):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "0.16663",
            "0.91516",
            "20.3305",
            "16.0207",
            "0.85097",
            "0.66952",
            "decomposed_cage_drift_bath_allowed = 0",
            "autonomous_single_particle_gle_allowed = 0",
            "complete_event_clock_closure_allowed = 0",
            "kramers_escape_claim_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        models = {row["model"]: row for row in rows if row["record"] == "aggregate_model"}
        self.assertEqual(
            set(models),
            {"raw_hankel_16", "split_cage_drift_8", "split_cage_drift_16"},
        )
        baseline = models["raw_hankel_16"]
        primary = models["split_cage_drift_8"]
        diagnostic = models["split_cage_drift_16"]
        self.assertEqual(int(float(primary["held_clone_count"])), 4)
        self.assertGreater(
            float(primary["maximum_maximum_held_residual_lag_correlation"]),
            float(baseline["maximum_maximum_held_residual_lag_correlation"]),
        )
        self.assertGreater(float(primary["terminal_diffusion_relative_error"]), 20.0)
        self.assertLess(
            float(diagnostic["maximum_maximum_held_residual_lag_correlation"]),
            float(baseline["maximum_maximum_held_residual_lag_correlation"]),
        )
        self.assertGreater(
            float(diagnostic["maximum_maximum_held_residual_state_correlation"]),
            0.66,
        )
        self.assertLess(float(primary["maximum_force_cache_relative_rms_error"]), 2e-5)
        self.assertGreater(float(primary["minimum_force_cache_correlation"]), 0.99999999)
        self.assertLess(
            float(primary["maximum_maximum_noise_reconstruction_relative_error"]),
            1e-10,
        )

        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["integrity_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["numerical_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["every_fold_lag_improves"]), 0.0)
        self.assertEqual(float(verdict["residual_memory_gate_pass"]), 0.0)
        self.assertEqual(float(verdict["macro_event_gate_pass"]), 0.0)
        self.assertGreater(float(verdict["primary_to_baseline_lag_ratio"]), 1.06)
        for key in (
            "decomposed_cage_drift_bath_allowed",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

    def test_projected_ito_innovation_audit_preserves_primary_failure(self):
        stem = "renewal_cage_ka_projected_ito_innovations_T058"
        document_path = ROOT / "docs" / "microscopic-projected-ito-innovation-audit.md"
        summary_path = ROOT / "data" / f"{stem}_summary.csv"
        detail_path = ROOT / "data" / f"{stem}_details.csv"
        script_path = ROOT / "scripts" / "analyze_ka_projected_ito_innovations.py"
        for path in (document_path, summary_path, detail_path, script_path):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "0.22274",
            "0.08674",
            "0.93279",
            "0.03043",
            "projected_ito_local_gate_pass = 0",
            "adapted_second_order_consistency_gate_pass = 1",
            "projected_sde_numerically_supported = 1",
            "microscopic_projected_sde_allowed = 0",
            "autonomous_single_particle_gle_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        pooled = {
            (row["scheme"], int(float(row["stride"]))): row
            for row in rows
            if row["record"] == "pooled"
        }
        self.assertEqual(
            set(pooled),
            {
                (scheme, stride)
                for scheme in ("left", "adams_bashforth2", "trapezoid")
                for stride in (1, 2, 4, 8)
            },
        )
        left = pooled[("left", 1)]
        adapted = pooled[("adams_bashforth2", 1)]
        trapezoid = pooled[("trapezoid", 1)]
        self.assertGreater(
            float(left["maximum_absolute_whitened_state_correlation"]), 0.22
        )
        self.assertGreater(
            float(left["maximum_absolute_whitened_lag1_correlation"]), 0.08
        )
        for row in (adapted, trapezoid):
            self.assertLess(
                float(row["maximum_absolute_whitened_state_correlation"]), 0.05
            )
            self.assertLess(
                float(row["maximum_absolute_whitened_lag1_correlation"]), 0.05
            )
            self.assertLess(
                float(row["maximum_absolute_whitened_covariance_error"]), 0.07
            )

        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["projected_ito_local_gate_pass"]), 0.0)
        self.assertEqual(
            float(verdict["adapted_second_order_consistency_gate_pass"]), 1.0
        )
        self.assertEqual(float(verdict["trapezoid_sensitivity_gate_pass"]), 1.0)
        self.assertEqual(float(verdict["projected_sde_numerically_supported"]), 1.0)
        for key in (
            "microscopic_projected_sde_allowed",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

    def test_additive_correlated_noise_closure_keeps_cross_noise(self):
        stem = "renewal_cage_ka_additive_correlated_noise_closure_T058"
        document_path = ROOT / "docs" / "microscopic-additive-correlated-noise-closure.md"
        summary_path = ROOT / "data" / f"{stem}_summary.csv"
        detail_path = ROOT / "data" / f"{stem}_details.csv"
        script_path = (
            ROOT / "scripts" / "analyze_ka_additive_correlated_noise_closure.py"
        )
        for path in (document_path, summary_path, detail_path, script_path):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "0.17633",
            "-0.86023",
            "0.00170",
            "0.82362",
            "constant_correlated_projected_noise_allowed = 1",
            "configuration_dependent_noise_required = 0",
            "center_relative_cross_noise_required = 1",
            "autonomous_drift_closure_allowed = 0",
            "autonomous_single_particle_gle_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        models = {row["model"]: row for row in rows if row["record"] == "pooled_model"}
        self.assertEqual(
            set(models),
            {
                "exact_configuration_dependent",
                "constant_full",
                "constant_block_isotropic",
                "constant_block_uncorrelated",
                "constant_single_scalar",
            },
        )
        primary = models["constant_block_isotropic"]
        uncorrelated = models["constant_block_uncorrelated"]
        self.assertEqual(
            float(primary["every_held_clone_local_noise_gate_pass"]), 1.0
        )
        self.assertLess(
            float(primary["maximum_absolute_whitened_covariance_error"]), 0.07
        )
        self.assertGreater(
            float(uncorrelated["maximum_absolute_whitened_covariance_error"]), 0.82
        )
        self.assertEqual(
            float(uncorrelated["every_held_clone_local_noise_gate_pass"]), 0.0
        )

        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertLess(
            float(verdict["maximum_primary_to_exact_metric_difference"]), 0.002
        )
        self.assertEqual(
            float(verdict["constant_correlated_projected_noise_allowed"]), 1.0
        )
        self.assertEqual(float(verdict["configuration_dependent_noise_required"]), 0.0)
        self.assertEqual(float(verdict["center_relative_cross_noise_required"]), 1.0)
        for key in (
            "autonomous_drift_closure_allowed",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

    def test_relative_pmf_closes_static_force_but_not_markov_dynamics(self):
        stem = "renewal_cage_ka_relative_pmf_ou_boundary_T058"
        document_path = ROOT / "docs" / "microscopic-relative-pmf-ou-boundary.md"
        summary_path = ROOT / "data" / f"{stem}_summary.csv"
        detail_path = ROOT / "data" / f"{stem}_details.csv"
        curve_path = ROOT / "data" / f"{stem}_curve.csv"
        script_path = ROOT / "scripts" / "analyze_ka_relative_pmf_ou_boundary.py"
        for path in (document_path, summary_path, detail_path, curve_path, script_path):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "-0.49809",
            "0.00585",
            "0.99056",
            "0.06961",
            "1.09340",
            "0.94227",
            "relative_pmf_static_closure_allowed = 1",
            "markovian_relative_ou_allowed = 0",
            "relative_force_memory_required = 1",
            "autonomous_relative_dynamics_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        aggregate = next(row for row in rows if row["record"] == "aggregate")
        self.assertEqual(int(float(aggregate["held_clone_count"])), 4)
        self.assertLess(
            float(aggregate["maximum_fdt_velocity_variance_relative_error"]), 0.006
        )
        self.assertGreater(
            float(aggregate["minimum_conditional_radial_force_correlation"]), 0.99
        )
        self.assertLess(
            float(aggregate["maximum_conditional_radial_force_normalized_rmse"]),
            0.07,
        )
        self.assertGreater(
            float(aggregate["minimum_temperature_naive_force_normalized_rmse"]),
            1.58,
        )
        self.assertGreater(
            float(aggregate["maximum_mean_force_residual_lag_correlation"]), 0.94
        )
        self.assertGreater(
            float(aggregate["maximum_maximum_markov_ou_correlation_error"]), 1.09
        )

        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["relative_pmf_static_closure_allowed"]), 1.0)
        self.assertEqual(float(verdict["markovian_relative_ou_allowed"]), 0.0)
        self.assertEqual(float(verdict["relative_force_memory_required"]), 1.0)
        for key in (
            "autonomous_relative_dynamics_allowed",
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

    def test_relative_volterra_predicts_correlations_but_fails_fdt(self):
        stem = "renewal_cage_ka_relative_harmonic_volterra_fdt_T058"
        document_path = ROOT / "docs" / "microscopic-relative-harmonic-volterra-fdt.md"
        summary_path = ROOT / "data" / f"{stem}_summary.csv"
        detail_path = ROOT / "data" / f"{stem}_details.csv"
        curve_path = ROOT / "data" / f"{stem}_curve.csv"
        kernel_path = ROOT / "data" / f"{stem}_kernel.csv"
        script_path = (
            ROOT / "scripts" / "analyze_ka_relative_harmonic_volterra_fdt.py"
        )
        for path in (
            document_path,
            summary_path,
            detail_path,
            curve_path,
            kernel_path,
            script_path,
        ):
            self.assertTrue(path.is_file())

        document = document_path.read_text()
        for required in (
            "0.79011",
            "0.06382",
            "0.18326",
            "0.27149",
            "0.94666",
            "isoconfigurational_cage_bias_supported = 1",
            "relative_correlation_volterra_allowed = 1",
            "relative_mori_fdt_closure_allowed = 0",
            "physical_scalar_relative_gle_allowed = 0",
            "relative_orthogonal_force_closure_required = 1",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        aggregate = next(row for row in rows if row["record"] == "aggregate")
        self.assertEqual(int(float(aggregate["held_clone_count"])), 4)
        self.assertGreater(float(aggregate["minimum_held_bias_correlation"]), 0.79)
        self.assertLess(
            float(aggregate["maximum_held_extrapolation_correlation_rmse"]), 0.064
        )
        self.assertLess(
            float(
                aggregate["maximum_held_extrapolation_maximum_correlation_error"]
            ),
            0.184,
        )
        self.assertLess(
            float(aggregate["maximum_fdt_random_force_shape_correlation"]), 0.28
        )
        self.assertGreater(
            float(aggregate["minimum_fdt_random_force_normalized_rmse"]), 0.94
        )
        self.assertLess(
            float(aggregate["maximum_kernel_toeplitz_minimum_eigenvalue"]), -3.8e4
        )

        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["isoconfigurational_cage_bias_supported"]), 1.0)
        self.assertEqual(float(verdict["relative_correlation_volterra_allowed"]), 1.0)
        self.assertEqual(float(verdict["relative_mori_fdt_closure_allowed"]), 0.0)
        self.assertEqual(float(verdict["physical_scalar_relative_gle_allowed"]), 0.0)
        for key in (
            "relative_orthogonal_force_closure_required",
        ):
            self.assertEqual(float(verdict[key]), 1.0)
        for key in (
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

    def test_generator_augmented_matrix_mori_passes_independent_clones(self):
        discovery_stem = "renewal_cage_ka_relative_generator_mori_discovery_T058"
        validation_stem = "renewal_cage_ka_relative_generator_mori_validation_T058"
        document_path = ROOT / "docs" / "microscopic-relative-generator-mori.md"
        script_path = ROOT / "scripts" / "analyze_ka_relative_generator_mori.py"
        cache_script_path = ROOT / "scripts" / "cache_ka_decomposed_drift.py"
        for path in (document_path, script_path, cache_script_path):
            self.assertTrue(path.is_file())
        for stem in (discovery_stem, validation_stem):
            for suffix in ("details", "summary", "curve"):
                self.assertTrue((ROOT / "data" / f"{stem}_{suffix}.csv").is_file())

        document = document_path.read_text()
        for required in (
            "0.09059",
            "0.19297",
            "0.04129",
            "0.07652",
            "0.99853",
            "0.15001",
            "projected_relative_generator_mori_representation_allowed = 1",
            "thermal_fdt_adjoint_audit_pass = 0",
            "physical_relative_generator_gle_allowed = 0",
            "orthogonal_noise_generation_closed = 0",
            "autonomous_single_particle_gle_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with (ROOT / "data" / f"{discovery_stem}_summary.csv").open() as handle:
            discovery = list(csv.DictReader(handle))
        selected = next(
            row
            for row in discovery
            if row["record"] == "model_aggregate"
            and row["basis"] == "relative_phase_generator"
            and int(float(row["memory_order"])) == 40
        )
        self.assertEqual(float(selected["physical_representation_gate_pass"]), 1.0)
        self.assertLess(float(selected["maximum_gfd_operator_normalized_rmse"]), 0.091)
        self.assertLess(
            float(selected["maximum_held_target_correlation_extrapolation_maximum_error"]),
            0.193,
        )
        discovery_verdict = next(
            row for row in discovery if row["record"] == "verdict"
        )
        self.assertEqual(
            float(discovery_verdict["hyperparameter_selection_uses_held_folds"]), 1.0
        )
        self.assertEqual(float(discovery_verdict["independent_validation_available"]), 0.0)
        self.assertEqual(float(discovery_verdict["physical_relative_generator_gle_allowed"]), 0.0)

        with (ROOT / "data" / f"{validation_stem}_summary.csv").open() as handle:
            validation = list(csv.DictReader(handle))
        aggregate = next(
            row for row in validation if row["record"] == "model_aggregate"
        )
        self.assertEqual(int(float(aggregate["held_clone_count"])), 2)
        self.assertEqual(float(aggregate["physical_representation_gate_pass"]), 1.0)
        self.assertLess(
            float(aggregate["maximum_maximum_noise_initial_state_correlation"]), 0.042
        )
        self.assertLess(float(aggregate["maximum_gfd_operator_normalized_rmse"]), 0.077)
        self.assertGreater(float(aggregate["minimum_gfd_operator_shape_correlation"]), 0.9985)
        self.assertLess(
            float(aggregate["maximum_held_target_correlation_extrapolation_rmse"]), 0.050
        )
        self.assertLess(
            float(aggregate["maximum_held_target_correlation_extrapolation_maximum_error"]),
            0.151,
        )
        verdict = next(row for row in validation if row["record"] == "verdict")
        self.assertEqual(float(verdict["hyperparameter_selection_uses_held_folds"]), 0.0)
        self.assertEqual(float(verdict["confirmatory_matrix_mori_gfd_closure_supported"]), 1.0)
        self.assertEqual(
            float(verdict["projected_relative_generator_mori_representation_allowed"]),
            1.0,
        )
        self.assertEqual(float(verdict["thermal_fdt_adjoint_audit_pass"]), 0.0)
        self.assertEqual(float(verdict["physical_relative_generator_gle_allowed"]), 0.0)
        self.assertEqual(float(verdict["orthogonal_noise_generation_closed"]), 0.0)
        for key in (
            "autonomous_single_particle_gle_allowed",
            "complete_event_clock_closure_allowed",
            "kramers_escape_claim_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

    def test_generator_parity_basis_passes_detailed_balance_necessary_condition(self):
        stem = "renewal_cage_ka_relative_generator_parity_validation_T058"
        document_path = ROOT / "docs" / "microscopic-relative-generator-parity.md"
        script_path = ROOT / "scripts" / "analyze_ka_relative_generator_parity.py"
        for path in (document_path, script_path):
            self.assertTrue(path.is_file())
        for suffix in ("details", "summary", "curve"):
            self.assertTrue((ROOT / "data" / f"{stem}_{suffix}.csv").is_file())

        document = document_path.read_text()
        for required in (
            "0.02192",
            "0.00871",
            "0.00078",
            "1.07727",
            "parity_definite_generator_basis_allowed = 1",
            "resolved_generalized_detailed_balance_supported = 1",
            "thermal_fdt_adjoint_audit_pass = 0",
            "physical_relative_generator_gle_allowed = 0",
        ):
            self.assertIn(required, document)

        with (ROOT / "data" / f"{stem}_summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        aggregate = next(row for row in rows if row["record"] == "aggregate")
        self.assertEqual(int(float(aggregate["held_clone_count"])), 2)
        self.assertLess(float(aggregate["maximum_parity_defect_normalized_rmse"]), 0.022)
        self.assertLess(
            float(aggregate["maximum_parity_defect_maximum_absolute_error"]), 0.0088
        )
        self.assertLess(
            float(aggregate["maximum_equal_time_maximum_forbidden_parity_correlation"]),
            0.0008,
        )
        self.assertGreater(
            float(aggregate["minimum_wrong_all_even_parity_defect_normalized_rmse"]),
            1.077,
        )
        verdict = next(row for row in rows if row["record"] == "verdict")
        self.assertEqual(float(verdict["parity_definite_generator_basis_allowed"]), 1.0)
        self.assertEqual(float(verdict["resolved_generalized_detailed_balance_supported"]), 1.0)
        self.assertEqual(float(verdict["wrong_all_even_parity_rejected"]), 1.0)
        for key in (
            "thermal_fdt_adjoint_audit_pass",
            "physical_relative_generator_gle_allowed",
            "orthogonal_noise_generation_closed",
            "autonomous_single_particle_gle_allowed",
            "thermodynamic_claim_allowed",
        ):
            self.assertEqual(float(verdict[key]), 0.0)

    def test_colored_block_noise_closes_relative_matrix_mori_simulation(self):
        discovery_stem = "renewal_cage_ka_relative_generator_noise_discovery_T058"
        validation_stems = (
            "renewal_cage_ka_relative_generator_noise_validation_highstat_T058",
            "renewal_cage_ka_relative_generator_noise_validation_highstat_seedB_T058",
        )
        required_paths = (
            ROOT / "scripts" / "analyze_ka_relative_generator_noise_closure.py",
            ROOT / "docs" / "microscopic-relative-generator-noise-closure.md",
            ROOT
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-07-14-relative-generator-noise-closure-design.md",
            ROOT
            / "docs"
            / "superpowers"
            / "plans"
            / "2026-07-14-relative-generator-noise-closure.md",
        )
        for path in required_paths:
            self.assertTrue(path.is_file(), path)
        for stem in (discovery_stem, *validation_stems):
            for suffix in ("details", "summary", "curve"):
                path = ROOT / "data" / f"{stem}_{suffix}.csv"
                self.assertTrue(path.is_file(), path)

        document = required_paths[1].read_text()
        for required in (
            "0.79545",
            "0.27484",
            "0.06023",
            "0.23928",
            "0.20661",
            "0.20338",
            "selected_innovation_block_length = 40",
            "iid_innovation_noise_allowed = 0",
            "colored_orthogonal_noise_required = 1",
            "empirical_block_noise_generation_closed = 1",
            "autonomous_relative_matrix_mori_simulation_allowed = 1",
            "thermal_fdt_adjoint_audit_pass = 0",
            "microscopic_thermal_noise_model_closed = 0",
            "autonomous_single_particle_gle_allowed = 0",
            "thermodynamic_claim_allowed = 0",
        ):
            self.assertIn(required, document)

        with (ROOT / "data" / f"{discovery_stem}_summary.csv").open() as handle:
            discovery = list(csv.DictReader(handle))
        iid = next(
            row
            for row in discovery
            if row["record"] == "model_aggregate"
            and int(float(row["innovation_block_length"])) == 1
        )
        selected = next(
            row
            for row in discovery
            if row["record"] == "model_aggregate"
            and int(float(row["innovation_block_length"])) == 40
        )
        self.assertEqual(float(iid["empirical_block_noise_gate_pass"]), 0.0)
        self.assertGreater(
            float(iid["maximum_stationary_covariance_maximum_error"]), 0.79
        )
        self.assertLess(float(iid["minimum_terminal_variance_ratio"]), 0.…33516 tokens truncated…t = {
            (row["target_time_code"], row["direct_alpha_wave_number"]): row
            for row in ka2d_023
        }
        root_tc45 = by_target[("tc45", "4.79844851031")]
        high_tc50 = by_target[("tc50", "6.0")]
        self.assertAlmostEqual(float(root_tc45["predicted_fs"]), 0.24140638485539556, delta=1e-12)
        self.assertAlmostEqual(float(high_tc50["predicted_fs"]), 0.12459213356529883, delta=1e-12)
        self.assertLess(float(high_tc50["acceptance_fs_high"]), 0.18)
        self.assertEqual(float(root_tc45["prediction_target_ready"]), 1.0)
        self.assertEqual(float(root_tc45["observed_post_window_fs_ready"]), 0.0)
        self.assertEqual(float(root_tc45["real_alpha_shape_claim_ready"]), 0.0)
        self.assertEqual(root_tc45["primary_blocker"], "post_alpha_window_observation")
        self.assertEqual(float(root_tc45["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_direct_alpha_post_window_verdict_waits_for_observation(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_direct_alpha_post_window_verdict.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = [
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        ]
        self.assertEqual(len(ka2d_023), 6)
        root_tc45 = next(
            row for row in ka2d_023
            if row["target_time_code"] == "tc45"
            and row["direct_alpha_wave_number"] == "4.79844851031"
        )
        self.assertEqual(root_tc45["post_window_verdict_stage"], "post_alpha_observation_not_ready")
        self.assertAlmostEqual(float(root_tc45["predicted_fs"]), 0.24140638485539556, delta=1e-12)
        self.assertEqual(float(root_tc45["observed_post_window_fs_ready"]), 0.0)
        self.assertEqual(float(root_tc45["post_window_prediction_supported"]), 0.0)
        self.assertEqual(float(root_tc45["post_window_prediction_rejected"]), 0.0)
        self.assertEqual(float(root_tc45["real_alpha_shape_claim_ready"]), 0.0)
        self.assertEqual(root_tc45["primary_blocker"], "post_alpha_window_observation")
        self.assertEqual(float(root_tc45["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_direct_alpha_transport_records_proxy_not_inversion(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_direct_alpha_transport.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(
            ka2d_023["transport_coupling_stage"],
            "cached_direct_alpha_transport_proxy_ready_event_clock_blocked",
        )
        self.assertAlmostEqual(float(ka2d_023["tau_alpha_direct"]), 1500000.0)
        self.assertAlmostEqual(float(ka2d_023["matched_msd"]), 0.9747508405755333)
        self.assertAlmostEqual(float(ka2d_023["apparent_diffusion_coefficient"]), 1.6245847342925555e-7)
        self.assertAlmostEqual(float(ka2d_023["apparent_stokes_einstein_product"]), 0.24368771014388332)
        self.assertAlmostEqual(float(ka2d_023["matched_ngp_2d"]), 2.1239947887392923)
        self.assertEqual(float(ka2d_023["direct_alpha_transport_proxy_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_direct_alpha_pe_bound_records_conditional_identifiability(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_direct_alpha_pe_bound.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(
            ka2d_023["pe_feasibility_stage"],
            "direct_alpha_transport_bounds_pe_but_event_clock_missing",
        )
        self.assertEqual(float(ka2d_023["full_msd_jump_variance_feasible"]), 0.0)
        self.assertAlmostEqual(float(ka2d_023["jump_variance_upper_bound"]), 0.4855550202214052)
        self.assertAlmostEqual(float(ka2d_023["jump_variance_upper_over_msd"]), 0.4981324457588704)
        self.assertAlmostEqual(float(ka2d_023["reference_jump_variance_fraction"]), 0.2)
        self.assertAlmostEqual(float(ka2d_023["reference_exchange_mean"]), 600000.0)
        self.assertAlmostEqual(float(ka2d_023["reference_persistence_mean"]), 1409293.5403982885)
        self.assertAlmostEqual(float(ka2d_023["reference_persistence_exchange_ratio"]), 2.3488225673304806)
        self.assertEqual(float(ka2d_023["conditional_pe_inference_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "event_clock_jump_variance")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_direct_alpha_displacement_tail_bound_requires_segmentation(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_direct_alpha_displacement_tail_bound.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(
            ka2d_023["tail_bound_stage"],
            "direct_displacement_tail_exceeds_pe_single_event_bound",
        )
        self.assertAlmostEqual(float(ka2d_023["q_bound"]), 0.4855550202214053)
        self.assertAlmostEqual(float(ka2d_023["q_all"]), 0.48737542028776676)
        self.assertAlmostEqual(float(ka2d_023["q_all_over_bound"]), 1.003749111821625)
        self.assertAlmostEqual(float(ka2d_023["fraction_q_gt_bound"]), 0.2349612403100775)
        self.assertAlmostEqual(float(ka2d_023["mean_q_above_bound"]), 1.7929841231781933)
        self.assertAlmostEqual(float(ka2d_023["mean_q_above_over_bound"]), 3.692648718492544)
        self.assertEqual(float(ka2d_023["event_segmentation_required"]), 1.0)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "event_segmentation")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_direct_alpha_multilag_crossing_canary_blocks_replica_axis_clock(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_direct_alpha_multilag_crossing_canary.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(
            ka2d_023["crossing_canary_stage"],
            "multilag_displacement_crossing_canary_ready_replica_axis_blocked",
        )
        self.assertEqual(float(ka2d_023["axis0_is_isoconfigurational_replica"]), 1.0)
        self.assertAlmostEqual(float(ka2d_023["ever_crossed_fraction"]), 0.24007751937984495)
        self.assertAlmostEqual(float(ka2d_023["never_crossed_fraction"]), 0.759922480620155)
        self.assertAlmostEqual(float(ka2d_023["post_crossing_recross_fraction"]), 0.23599320882852293)
        self.assertAlmostEqual(float(ka2d_023["first_crossing_q_mean_over_bound"]), 3.4430132801814253)
        self.assertIn("tc40:0.21724806201550387", ka2d_023["first_crossing_fractions_by_time_code"])
        self.assertEqual(float(ka2d_023["event_segmentation_target_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["persistence_exchange_event_clock_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "frame_axis_is_isoconfigurational_replicates")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_direct_alpha_event_clock_contract_records_missing_true_time_axis(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_direct_alpha_event_clock_contract.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(
            ka2d_023["event_clock_contract_stage"],
            "segmentation_target_ready_true_event_clock_missing",
        )
        self.assertEqual(float(ka2d_023["conditional_pe_inference_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["direct_displacement_tail_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["event_segmentation_target_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["cached_replica_ladder_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["axis0_is_physical_time"]), 0.0)
        self.assertEqual(float(ka2d_023["requires_true_time_trajectory"]), 1.0)
        self.assertEqual(float(ka2d_023["event_clock_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "physical_time_trajectory_axis")
        self.assertIn("positions[time,particle,dimension]", ka2d_023["required_arrays"])
        self.assertIn("isoconfigurational_replica_axis", ka2d_023["forbidden_substitutes"])
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_sparse_lag_event_clock_audit_marks_coarse_candidate(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_sparse_lag_event_clock.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(
            ka2d_023["sparse_lag_event_clock_stage"],
            "sparse_lag_tensor_ready_replica_identity_unverified",
        )
        self.assertEqual(float(ka2d_023["physical_lag_tensor_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["same_initial_structure_verified"]), 1.0)
        self.assertEqual(float(ka2d_023["same_shape_across_lags"]), 1.0)
        self.assertEqual(float(ka2d_023["time_code_coverage_fraction"]), 1.0)
        self.assertEqual(float(ka2d_023["coarse_event_clock_candidate_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["replica_identity_alignment_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "replica_identity_alignment")
        self.assertEqual(ka2d_023["event_clock_resolution"], "sparse_lag_interval")
        self.assertIn("tc40", ka2d_023["observed_time_codes"])
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_interval_censored_first_crossing_clock_quantifies_candidate(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_interval_censored_first_crossing_clock.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(
            ka2d_023["interval_clock_stage"],
            "interval_censored_persistence_clock_candidate",
        )
        self.assertEqual(float(ka2d_023["interval_clock_candidate_ready"]), 1.0)
        self.assertAlmostEqual(float(ka2d_023["crossed_fraction"]), 0.24007751937984495)
        self.assertAlmostEqual(float(ka2d_023["right_censored_fraction"]), 0.759922480620155)
        self.assertAlmostEqual(float(ka2d_023["mean_first_crossing_lower_bound"]), 130241.55354213755)
        self.assertAlmostEqual(float(ka2d_023["mean_first_crossing_upper_bound"]), 1370127.2559589928)
        self.assertAlmostEqual(float(ka2d_023["mean_first_crossing_midpoint"]), 750184.4047505651)
        self.assertGreater(float(ka2d_023["mean_interval_width"]), 1.0e6)
        self.assertIn("tc40:142587", ka2d_023["first_crossing_intervals"])
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "interval_censoring_and_replica_identity")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_interval_censored_persistence_fit_estimates_scale(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_interval_censored_persistence_fit.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(
            ka2d_023["persistence_fit_stage"],
            "interval_censored_exponential_persistence_fit_ready",
        )
        self.assertEqual(float(ka2d_023["persistence_fit_ready"]), 1.0)
        self.assertAlmostEqual(float(ka2d_023["exponential_rate_mle"]), 1.827224516438915e-07, delta=1e-15)
        self.assertAlmostEqual(float(ka2d_023["exponential_mean_persistence_time"]), 5472781.209990023, delta=1.0)
        self.assertAlmostEqual(float(ka2d_023["predicted_crossed_fraction_at_latest_lag"]), 0.2397315447385533, delta=1e-7)
        self.assertAlmostEqual(float(ka2d_023["observed_crossed_fraction"]), 0.24007751937984495)
        self.assertAlmostEqual(float(ka2d_023["mean_persistence_over_tau_alpha_direct"]), 3.6485208066600154, delta=1e-6)
        self.assertEqual(float(ka2d_023["exchange_clock_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "exchange_clock_and_replica_identity")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_finite_exchange_envelope_defines_late_ngp_followup_horizon(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_finite_exchange_envelope.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["envelope_stage"], "finite_exchange_falsification_horizon_ready")
        self.assertEqual(float(ka2d_023["envelope_ready"]), 1.0)
        self.assertAlmostEqual(float(ka2d_023["conditional_persistence_exchange_ratio_lower_bound"]), 3.6485208210611972)
        self.assertAlmostEqual(float(ka2d_023["gaussian_recovery_lag_upper_bound"]), 42972781.2315918, delta=1e-3)
        self.assertAlmostEqual(float(ka2d_023["required_followup_lag_multiplier_over_current"]), 28.6485208210612, delta=1e-12)
        self.assertEqual(float(ka2d_023["current_window_has_gaussian_recovery_power"]), 0.0)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "late_ngp_followup_and_exchange_clock")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_late_recovery_protocol_keeps_followup_missing_explicit(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_protocol.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["late_recovery_stage"], "late_recovery_acquisition_required")
        self.assertEqual(float(ka2d_023["mechanism_selection_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["finite_exchange_supported"]), 0.0)
        self.assertEqual(float(ka2d_023["finite_exchange_rejected"]), 0.0)
        self.assertEqual(float(ka2d_023["static_disorder_rejected"]), 0.0)
        self.assertAlmostEqual(float(ka2d_023["required_followup_lag_time"]), 42972781.2315918, delta=1e-3)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "late_recovery_observation")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_late_recovery_ingestion_contract_requires_machine_readable_uncertainty(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_ingestion_contract.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["late_recovery_ingestion_stage"], "late_recovery_observation_missing")
        self.assertEqual(float(ka2d_023["late_recovery_observation_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["machine_readable_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["uncertainty_ready"]), 0.0)
        self.assertAlmostEqual(float(ka2d_023["required_followup_lag_time"]), 42972781.2315918, delta=1e-3)
        self.assertEqual(ka2d_023["missing_columns"], "observed_lag_time;observed_late_ngp;observed_tail_gaussian_recovery;source_trajectory_identity")
        self.assertEqual(ka2d_023["missing_uncertainty_columns"], "sigma_late_ngp;sigma_tail_recovery")
        self.assertEqual(ka2d_023["primary_blocker"], "late_recovery_observation")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)
        ka2d_030 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.30"
            and row["structure_id"] == "3"
        )
        self.assertEqual(ka2d_030["late_recovery_ingestion_stage"], "late_recovery_envelope_upstream_incomplete")
        self.assertEqual(ka2d_030["primary_blocker"], "finite_exchange_envelope")

    def test_sota_glassbench_late_recovery_timecode_target_names_next_cache(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_timecode_target.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["timecode_target_stage"], "late_recovery_timecode_target_ready")
        self.assertEqual(float(ka2d_023["timecode_target_ready"]), 1.0)
        self.assertEqual(ka2d_023["current_max_time_code"], "tc40")
        self.assertEqual(ka2d_023["target_time_code"], "tc50")
        self.assertAlmostEqual(float(ka2d_023["current_max_lag_time"]), 1500000.0)
        self.assertAlmostEqual(float(ka2d_023["required_followup_lag_time"]), 42972781.2315918, delta=1e-3)
        self.assertGreater(float(ka2d_023["target_lag_over_required"]), 3.8)
        self.assertEqual(ka2d_023["primary_blocker"], "late_recovery_time_code_cache")
        self.assertEqual(float(ka2d_023["late_recovery_observation_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_late_recovery_cache_request_contract_marks_tc50_metadata_gap(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_cache_request_contract.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["cache_request_stage"], "late_recovery_member_metadata_required")
        self.assertEqual(float(ka2d_023["cache_request_ready"]), 1.0)
        self.assertEqual(ka2d_023["target_time_code"], "tc50")
        self.assertEqual(ka2d_023["inferred_target_member"], "T0.23/test/N1290T0.23_151_tc50.npz")
        self.assertEqual(float(ka2d_023["inferred_member_path_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["official_target_member_metadata_ready"]), 0.0)
        self.assertEqual(ka2d_023["target_member_md5"], "none")
        self.assertEqual(float(ka2d_023["particle_cache_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "late_recovery_npz_member_metadata")
        self.assertEqual(float(ka2d_023["late_recovery_observable_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)
        ka2d_030 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.30"
            and row["structure_id"] == "3"
        )
        self.assertEqual(ka2d_030["cache_request_stage"], "late_recovery_timecode_target_incomplete")
        self.assertEqual(ka2d_030["inferred_target_member"], "none")
        self.assertEqual(float(ka2d_030["inferred_member_path_ready"]), 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "late_recovery_timecode_target")

    def test_sota_glassbench_late_recovery_membership_probe_records_tc50_prefix_absence(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_membership_probe_contract.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["membership_probe_stage"], "late_recovery_target_absent_from_extended_prefix")
        self.assertEqual(float(ka2d_023["membership_probe_ready"]), 1.0)
        self.assertEqual(ka2d_023["target_time_code"], "tc50")
        self.assertEqual(ka2d_023["inferred_target_member"], "T0.23/test/N1290T0.23_151_tc50.npz")
        self.assertEqual(float(ka2d_023["target_member_visible_in_probe"]), 0.0)
        self.assertEqual(float(ka2d_023["same_structure_member_count_in_probe"]), 8.0)
        self.assertEqual(ka2d_023["max_visible_time_code"], "tc40")
        self.assertAlmostEqual(float(ka2d_023["compressed_probe_bytes"]), 12582912.0)
        self.assertEqual(ka2d_023["primary_blocker"], "late_recovery_member_index_depth")
        self.assertEqual(float(ka2d_023["late_recovery_observable_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_late_recovery_public_timecode_ceiling_blocks_tc50(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_public_timecode_ceiling.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["public_ceiling_stage"], "late_recovery_beyond_public_timecode_ceiling")
        self.assertEqual(float(ka2d_023["public_ceiling_ready"]), 1.0)
        self.assertEqual(ka2d_023["target_time_code"], "tc50")
        self.assertEqual(ka2d_023["public_max_time_code"], "tc40")
        self.assertEqual(ka2d_023["structure_max_time_code"], "tc40")
        self.assertEqual(float(ka2d_023["target_time_code_published"]), 0.0)
        self.assertGreater(float(ka2d_023["target_lag_over_public_max"]), 100.0)
        self.assertEqual(ka2d_023["primary_blocker"], "public_glassbench_timecode_ceiling")
        self.assertEqual(float(ka2d_023["late_recovery_observation_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_censored_window_claim_audit_limits_public_claims(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_censored_window_claim_audit.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["censored_window_stage"], "alpha_anchor_ready_late_recovery_censored")
        self.assertEqual(ka2d_023["allowed_public_claim_level"], "alpha_anchor_and_pre_late_dynamic_signatures")
        self.assertEqual(float(ka2d_023["alpha_anchor_window_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["short_window_dynamic_claim_allowed"]), 1.0)
        self.assertEqual(float(ka2d_023["alpha_relaxation_claim_allowed"]), 1.0)
        self.assertEqual(float(ka2d_023["late_gaussian_recovery_claim_allowed"]), 0.0)
        self.assertEqual(float(ka2d_023["static_vs_finite_exchange_rejection_ready"]), 0.0)
        self.assertLess(float(ka2d_023["public_window_fraction_of_target_lag"]), 0.01)
        self.assertGreater(float(ka2d_023["target_lag_over_public_max"]), 100.0)
        self.assertEqual(ka2d_023["primary_blocker"], "public_glassbench_timecode_ceiling")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_public_window_verdict_maps_sota_claims_to_censoring(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_public_window_verdict.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

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

        persistence_exchange = by_signature["persistence_exchange_decoupling"]
        self.assertEqual(
            persistence_exchange["public_window_verdict_stage"],
            "mechanism_selection_censored_unresolved",
        )
        self.assertEqual(float(persistence_exchange["mechanism_rejection_ready"]), 0.0)

        thermodynamic = by_signature["thermodynamic_transition"]
        self.assertEqual(thermodynamic["public_window_verdict_stage"], "scope_boundary_not_tested")
        self.assertEqual(thermodynamic["allowed_public_claim"], "not_a_thermodynamic_glass_transition_test")
        self.assertEqual(float(thermodynamic["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_late_recovery_experiment_design_names_minimal_tc50_followup(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_experiment_design.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["experiment_design_stage"], "minimal_tc50_followup_ready")
        self.assertEqual(ka2d_023["current_max_time_code"], "tc40")
        self.assertEqual(ka2d_023["required_time_code"], "tc50")
        self.assertAlmostEqual(float(ka2d_023["minimum_required_lag_time"]), 42972781.2315918, delta=1e-3)
        self.assertAlmostEqual(float(ka2d_023["planned_lag_time"]), 166002226.81761542, delta=1e-3)
        self.assertGreater(float(ka2d_023["planned_lag_over_minimum_required"]), 3.0)
        self.assertEqual(ka2d_023["required_observables"], "MSD;NGP;F_s(k,t);self_van_hove_tail;member_uncertainty")
        self.assertEqual(ka2d_023["finite_exchange_support_rule"], "late_ngp <= max_finite_exchange_late_ngp")
        self.assertEqual(ka2d_023["static_disorder_rejection_rule"], "late_ngp + 2sigma < static_gamma_late_ngp_plateau")
        self.assertEqual(float(ka2d_023["max_finite_exchange_late_ngp"]), 0.05)
        self.assertEqual(float(ka2d_023["static_gamma_late_ngp_plateau"]), 0.1)
        self.assertEqual(float(ka2d_023["late_recovery_claim_ready_after_measurement"]), 1.0)
        self.assertEqual(ka2d_023["primary_blocker"], "public_glassbench_timecode_ceiling")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_late_recovery_uncertainty_verdict_waits_for_tc50_measurement(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_uncertainty_verdict.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(ka2d_023["uncertainty_verdict_stage"], "late_recovery_observation_not_ready")
        self.assertEqual(ka2d_023["candidate_id"], "KA2D:0.23:151:0")
        self.assertAlmostEqual(float(ka2d_023["minimum_required_lag_time"]), 42972781.2315918, delta=1e-3)
        self.assertEqual(float(ka2d_023["finite_exchange_uncertainty_supported"]), 0.0)
        self.assertEqual(float(ka2d_023["static_disorder_uncertainty_rejected"]), 0.0)
        self.assertEqual(float(ka2d_023["uncertainty_decision_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "late_recovery_observation")

    def test_sota_glassbench_late_recovery_outcome_matrix_preregisters_tc50_paths(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_outcome_matrix.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = [
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        ]
        by_scenario = {row["outcome_scenario"]: row for row in ka2d_023}
        self.assertEqual(len(ka2d_023), 3)
        self.assertEqual(by_scenario["low_late_ngp_gaussian_recovery"]["target_time_code"], "tc50")
        self.assertEqual(
            by_scenario["low_late_ngp_gaussian_recovery"]["claim_if_observed"],
            "finite_exchange_supported_static_disorder_rejected",
        )
        self.assertEqual(
            by_scenario["high_late_ngp_or_missing_recovery"]["claim_if_observed"],
            "finite_exchange_rejected_or_model_reparameterization_required",
        )
        self.assertEqual(
            by_scenario["wide_uncertainty_requires_more_data"]["claim_if_observed"],
            "no_mechanism_selection_claim",
        )
        self.assertEqual(
            by_scenario["wide_uncertainty_requires_more_data"]["outcome_matrix_stage"],
            "tc50_outcome_matrix_preregistered",
        )
        self.assertEqual(float(by_scenario["wide_uncertainty_requires_more_data"]["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_late_recovery_decision_power_plan_sizes_tc50_member_extension(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_late_recovery_decision_power_plan.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = [
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        ]
        by_scenario = {row["outcome_scenario"]: row for row in ka2d_023}
        sufficient = by_scenario["low_late_ngp_gaussian_recovery"]
        wide = by_scenario["wide_uncertainty_requires_more_data"]

        self.assertEqual(sufficient["decision_power_stage"], "decision_power_sufficient")
        self.assertEqual(float(sufficient["additional_member_count_needed"]), 0.0)
        self.assertEqual(wide["decision_power_stage"], "late_ngp_power_extension_required")
        self.assertGreater(float(wide["member_multiplier_needed"]), 1.0)
        self.assertEqual(float(wide["required_member_count"]), 128.0)
        self.assertEqual(float(wide["additional_member_count_needed"]), 120.0)
        self.assertGreater(float(wide["required_member_count"]), float(wide["current_member_count"]))
        self.assertEqual(wide["primary_blocker"], "late_ngp_uncertainty")
        self.assertEqual(float(wide["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_microdynamic_closed_loop_marks_real_blockers(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_microdynamic_closed_loop.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_023 = by_key[("KA2D", "0.23")]
        self.assertEqual(
            ka2d_023["closed_loop_stage"],
            "real_microstats_macro_signatures_closed_loop_blocked",
        )
        self.assertEqual(float(ka2d_023["frame_index_microstats_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["macro_signature_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["micro_to_macro_prediction_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["closed_loop_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "physical_time_semantics")
        self.assertIn("cage_jump_event_segmentation", ka2d_023["missing_closed_loop_inputs"])
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["closed_loop_stage"], "macro_timecode_upstream_incomplete")
        self.assertEqual(ka2d_030["primary_blocker"], "sparse_time_code_coverage")

    def test_sota_glassbench_cage_jump_proxy_canary_marks_proxy_not_event_clock(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_cage_jump_proxy_canary.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_023 = by_key[("KA2D", "0.23")]
        self.assertEqual(
            ka2d_023["canary_stage"],
            "aggregate_cage_jump_proxy_ready_particle_events_blocked",
        )
        self.assertEqual(float(ka2d_023["aggregate_jump_proxy_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["particle_resolved_jump_events_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["physical_time_jump_clock_ready"]), 0.0)
        self.assertGreater(float(ka2d_023["proxy_jump_length"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "particle_resolved_displacements")
        self.assertIn("persistence_exchange_event_clock", ka2d_023["missing_event_clock_inputs"])
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_event_clock_threshold_readiness_blocks_real_claim(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_event_clock_threshold_readiness.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        for key in [("KA2D", "0.23"), ("KA2D", "0.30")]:
            row = by_key[key]
            self.assertEqual(row["readiness_stage"], "real_event_clock_threshold_robustness_blocked")
            self.assertEqual(float(row["positions_schema_ready"]), 1.0)
            self.assertEqual(float(row["first_npz_observable_curve_ready"]), 1.0)
            self.assertEqual(float(row["member_ensemble_observable_ready"]), 1.0)
            self.assertEqual(float(row["particle_resolved_positions_cached"]), 1.0)
            self.assertEqual(float(row["threshold_sweep_event_clock_ready"]), 0.0)
            self.assertEqual(float(row["real_event_clock_threshold_robustness_ready"]), 0.0)
            self.assertEqual(row["primary_blocker"], "physical_time_semantics")
            self.assertNotIn("particle_resolved_positions_cache", row["missing_real_threshold_inputs"])
            self.assertIn("threshold_sweep_event_clock", row["missing_real_threshold_inputs"])
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_first_npz_particle_cache_contract_records_cached_positions(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_first_npz_particle_cache_contract.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_023 = by_key[("KA2D", "0.23")]
        self.assertEqual(
            ka2d_023["cache_contract_stage"],
            "first_npz_particle_cache_contract_ready_time_blocked",
        )
        self.assertEqual(ka2d_023["first_npz_member"], "T0.23/test/N1290T0.23_202_tc05.npz")
        self.assertEqual(ka2d_023["positions_shape"], "20x1290x2")
        self.assertEqual(float(ka2d_023["compressed_probe_range_start"]), 2980602255.0)
        self.assertEqual(float(ka2d_023["npz_member_bytes"]), 465710.0)
        self.assertEqual(ka2d_023["npz_member_md5"], "26b4b9af10138fbd04a840fe8275de8e")
        self.assertEqual(float(ka2d_023["particle_cache_contract_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["particle_resolved_positions_cached"]), 1.0)
        self.assertEqual(float(ka2d_023["threshold_sweep_event_clock_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "physical_time_semantics")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["positions_shape"], "20x1290x2")
        self.assertEqual(float(ka2d_030["npz_member_bytes"]), 444786.0)
        self.assertEqual(ka2d_030["npz_member_md5"], "f51fd76f59b8288405a9e7abb61cdd0a")
        self.assertEqual(float(ka2d_030["particle_resolved_positions_cached"]), 1.0)

    def test_sota_glassbench_first_npz_particle_cache_manifest_records_real_cache(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_first_npz_particle_cache_manifest.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        for key, expected_md5 in [
            (("KA2D", "0.23"), "26b4b9af10138fbd04a840fe8275de8e"),
            (("KA2D", "0.30"), "f51fd76f59b8288405a9e7abb61cdd0a"),
        ]:
            row = by_key[key]
            cache_path = Path(row["particle_cache_path"])
            self.assertTrue(cache_path.exists())
            self.assertEqual(row["cache_stage"], "particle_coordinate_cache_written")
            self.assertEqual(row["probe_encoding"], "zip_deflate_xz")
            self.assertEqual(row["positions_shape"], "20x1290x2")
            self.assertEqual(row["npz_member_md5"], expected_md5)
            self.assertEqual(float(row["particle_resolved_positions_cached"]), 1.0)
            self.assertEqual(float(row["threshold_sweep_event_clock_ready"]), 0.0)
            self.assertEqual(row["primary_blocker"], "physical_time_semantics")

    def test_sota_glassbench_cached_particle_timecode_bridge_keeps_replica_axis_blocker(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_cached_particle_timecode_bridge.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_023 = by_key[("KA2D", "0.23")]
        self.assertEqual(ka2d_023["time_code"], "tc05")
        self.assertAlmostEqual(float(ka2d_023["lag_time"]), 0.1)
        self.assertEqual(float(ka2d_023["physical_lag_time_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["particle_resolved_positions_cached"]), 1.0)
        self.assertEqual(float(ka2d_023["frame_axis_is_physical_time"]), 0.0)
        self.assertEqual(float(ka2d_023["axis0_is_isoconfigurational_replica"]), 1.0)
        self.assertEqual(float(ka2d_023["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "frame_axis_is_isoconfigurational_replicates")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(ka2d_030["time_code"], "tc01")
        self.assertAlmostEqual(float(ka2d_030["lag_time"]), 0.11)
        self.assertEqual(float(ka2d_030["physical_lag_time_ready"]), 1.0)

    def test_sota_glassbench_multilag_particle_cache_targets_identifies_next_members(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_multilag_particle_cache_targets.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_023 = by_key[("KA2D", "0.23")]
        self.assertEqual(ka2d_023["selected_structure_id"], "151")
        self.assertEqual(float(ka2d_023["official_multi_lag_ladder_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["target_member_count"]), 8.0)
        self.assertEqual(float(ka2d_023["cached_target_member_count"]), 8.0)
        self.assertEqual(float(ka2d_023["missing_target_member_count"]), 0.0)
        self.assertIn("T0.23/test/N1290T0.23_151_tc40.npz", ka2d_023["target_members"])
        self.assertIn("5160feded6ec1a1f366a6e55a7d33f70", ka2d_023["target_member_md5s"])
        self.assertEqual(float(ka2d_023["particle_lag_ladder_cache_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "frame_axis_is_isoconfigurational_replicates")

        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(float(ka2d_030["official_multi_lag_ladder_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["target_member_count"]), 1.0)
        self.assertEqual(ka2d_030["primary_blocker"], "official_multi_lag_semantics")

    def test_sota_glassbench_multilag_particle_cache_manifest_records_prefix_extracted_targets(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_multilag_particle_cache_manifest.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        cold_rows = [
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        ]
        self.assertEqual(len(cold_rows), 8)
        by_code = {row["time_code"]: row for row in cold_rows}
        for code in ["tc05", "tc10", "tc15", "tc20", "tc25", "tc30", "tc35", "tc40"]:
            row = by_code[code]
            self.assertEqual(float(row["member_in_bounded_prefix_index"]), 1.0)
            self.assertEqual(float(row["particle_resolved_positions_cached"]), 1.0)
            self.assertEqual(row["cache_stage"], "multi_lag_particle_coordinate_cache_written")
            self.assertTrue((ROOT / row["particle_cache_path"]).exists())
            self.assertEqual(row["positions_shape"], "20x1290x2")

    def test_sota_glassbench_cached_particle_observable_semantics_reproduces_official_displacements(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_cached_particle_observable_semantics.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        cold_rows = [
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        ]
        self.assertEqual(len(cold_rows), 8)
        by_code = {row["time_code"]: row for row in cold_rows}
        tc05 = by_code["tc05"]
        self.assertGreater(float(tc05["raw_coordinate_msd_relative_error"]), 1.0e4)
        self.assertEqual(float(tc05["cached_coordinate_proxy_ready"]), 1.0)
        self.assertEqual(float(tc05["initial_reference_positions_ready"]), 1.0)
        self.assertLess(float(tc05["initial_reference_msd_relative_error"]), 1.0e-12)
        self.assertEqual(float(tc05["official_displacement_observable_reproducible"]), 1.0)
        self.assertEqual(float(tc05["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(tc05["primary_blocker"], "none")
        self.assertEqual(tc05["observable_semantics_stage"], "official_displacement_observable_reproduced")
        self.assertEqual(float(tc05["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_cached_particle_observable_semantics_reproduces_official_ngp_formula(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_cached_particle_observable_semantics.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        tc30 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151" and row["time_code"] == "tc30"
        )
        self.assertEqual(tc30["initial_reference_ngp_2d_formula"], "mean_replica_alpha2_2d")
        self.assertGreater(float(tc30["pooled_initial_reference_ngp_2d_relative_error"]), 0.5)
        self.assertLess(float(tc30["initial_reference_ngp_2d_relative_error"]), 1.0e-12)
        self.assertEqual(float(tc30["official_ngp_2d_reproducible"]), 1.0)

    def test_sota_glassbench_cached_particle_observable_semantics_reproduces_official_fs_formula(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_cached_particle_observable_semantics.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        tc40 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151" and row["time_code"] == "tc40"
        )
        self.assertEqual(tc40["initial_reference_fs_formula"], "axis_average_cos_xy")
        self.assertLess(float(tc40["initial_reference_fs_max_abs_error"]), 1.0e-12)
        self.assertGreater(float(tc40["single_axis_x_fs_max_abs_error"]), 1.0e-3)
        self.assertEqual(float(tc40["official_fs_reproducible"]), 1.0)

    def test_sota_glassbench_observable_renewal_canary_rejects_naive_lag_clock(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_observable_renewal_canary.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        cold = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(float(cold["lag_count"]), 8.0)
        self.assertEqual(float(cold["real_displacement_ladder_ready"]), 1.0)
        self.assertEqual(float(cold["naive_lag_clock_renewal_fit_pass"]), 0.0)
        self.assertEqual(float(cold["naive_lag_clock_rejected"]), 1.0)
        self.assertGreater(float(cold["effective_jump_variance_cv"]), 1.0)
        self.assertGreater(float(cold["max_ngp_2d"]), 7.0)
        self.assertGreater(float(cold["max_fs_decay"]), 0.3)
        self.assertEqual(float(cold["event_clock_trajectory_ready"]), 0.0)
        self.assertEqual(float(cold["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(cold["primary_blocker"], "event_clock_segmentation_required")
        self.assertEqual(cold["canary_stage"], "real_observable_ladder_rejects_naive_lag_clock")

    def test_sota_glassbench_real_threshold_sweep_canary_blocks_unstable_event_clock(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_real_threshold_sweep_canary.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        cold = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
            and row["structure_id"] == "151"
        )
        self.assertEqual(float(cold["threshold_sweep_candidate_ready"]), 1.0)
        self.assertEqual(float(cold["threshold_robust_event_clock_ready"]), 0.0)
        self.assertEqual(float(cold["axis0_is_isoconfigurational_replica"]), 1.0)
        self.assertGreater(float(cold["mean_persistence_sensitivity_ratio"]), 2.0)
        self.assertGreater(float(cold["max_post_crossing_recross_fraction"]), 0.2)
        self.assertEqual(float(cold["recross_stable"]), 0.0)
        self.assertGreater(float(cold["anchor_mean_persistence_over_tau_alpha"]), 3.0)
        self.assertEqual(float(cold["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(cold["primary_blocker"], "threshold_sensitivity_recrossing_and_replica_identity")
        self.assertEqual(cold["threshold_sweep_stage"], "real_threshold_sweep_sensitive_replica_axis_blocked")

    def test_sota_glassbench_threshold_sweep_ensemble_verdict_marks_temperature_coverage_boundary(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_threshold_sweep_ensemble_verdict.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_temperature = {row["temperature"]: row for row in rows}
        cold = by_temperature["0.23"]
        hot = by_temperature["0.30"]
        self.assertEqual(float(cold["lag_count"]), 8.0)
        self.assertEqual(float(cold["threshold_sweep_candidate_ready"]), 1.0)
        self.assertEqual(float(cold["ensemble_threshold_robust_ready"]), 0.0)
        self.assertGreater(float(cold["mean_persistence_sensitivity_ratio"]), 2.0)
        self.assertGreater(float(cold["max_post_crossing_recross_fraction"]), 0.2)
        self.assertEqual(cold["ensemble_stage"], "ensemble_threshold_sensitive_blocked")
        self.assertEqual(float(hot["lag_count"]), 1.0)
        self.assertEqual(float(hot["threshold_sweep_candidate_ready"]), 0.0)
        self.assertEqual(hot["primary_blocker"], "insufficient_lag_coverage")
        self.assertEqual(hot["ensemble_stage"], "ensemble_threshold_sweep_coverage_blocked")
        self.assertTrue(all(float(row["real_pe_inversion_ready"]) == 0.0 for row in rows))

    def test_sota_glassbench_threshold_sweep_payload_contract_names_minimal_hot_payload(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_threshold_sweep_payload_contract.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        hot = next(row for row in rows if row["system_id"] == "KA2D" and row["temperature"] == "0.30")
        self.assertEqual(float(hot["lag_count"]), 1.0)
        self.assertEqual(float(hot["minimum_lag_count_for_threshold_sweep"]), 3.0)
        self.assertEqual(float(hot["minimum_additional_lag_count"]), 2.0)
        self.assertEqual(float(hot["official_member_ladder_known"]), 0.0)
        self.assertEqual(hot["known_time_codes"], "tc01")
        self.assertEqual(hot["requested_payload"], "official_multi_lag_member_index_and_particle_coordinates")
        self.assertEqual(hot["payload_contract_stage"], "multi_lag_payload_request_ready")
        self.assertEqual(float(hot["real_pe_inversion_ready"]), 0.0)

    def test_sota_glassbench_threshold_sweep_outcome_matrix_preregisters_hot_ladder_decision(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_threshold_sweep_outcome_matrix.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        hot = next(row for row in rows if row["system_id"] == "KA2D" and row["temperature"] == "0.30")
        self.assertEqual(hot["outcome_stage"], "awaiting_payload_preregistered_outcome")
        self.assertEqual(float(hot["minimum_additional_lag_count"]), 2.0)
        self.assertIn("mean_persistence_sensitivity_ratio <= 1.5", hot["preregistered_pass_condition"])
        self.assertIn("max_post_crossing_recross_fraction <= 0.05", hot["preregistered_pass_condition"])
        self.assertEqual(hot["claim_if_pass"], "threshold_robust_event_clock_candidate_not_pe_inversion")
        self.assertEqual(hot["claim_if_fail"], "fixed_lag_threshold_event_clock_rejected")
        self.assertEqual(float(hot["real_pe_inversion_ready"]), 0.0)
        self.assertEqual(float(hot["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_threshold_sweep_decision_power_plan_requires_member_uncertainty(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_threshold_sweep_decision_power_plan.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        hot = next(row for row in rows if row["system_id"] == "KA2D" and row["temperature"] == "0.30")
        self.assertEqual(float(hot["current_independent_member_count"]), 1.0)
        self.assertEqual(float(hot["minimum_independent_member_count"]), 3.0)
        self.assertEqual(float(hot["additional_independent_member_count_needed"]), 2.0)
        self.assertEqual(float(hot["pooled_particle_decision_allowed"]), 0.0)
        self.assertIn("member_mean_persistence_sensitivity_ratio", hot["required_uncertainty_columns"])
        self.assertEqual(hot["decision_power_stage"], "independent_member_extension_required")
        self.assertEqual(float(hot["real_pe_inversion_ready"]), 0.0)

    def test_sota_dynamic_signature_alignment_ledger_combines_literature_and_real_curve(self):
        path = ROOT / "data" / "renewal_cage_sota_dynamic_signature_alignment.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_signature = {row["signature"]: row for row in rows}
        ngp = by_signature["transient_ngp_peak"]
        self.assertEqual(ngp["alignment_stage"], "real_curve_supported")
        self.assertEqual(float(ngp["model_support"]), 1.0)
        self.assertEqual(float(ngp["literature_qualitative_support"]), 1.0)
        self.assertEqual(float(ngp["real_glassbench_support"]), 1.0)

        alpha = by_signature["self_intermediate_alpha"]
        self.assertEqual(alpha["alignment_stage"], "real_curve_supported_pre_alpha_threshold")
        self.assertEqual(float(alpha["real_glassbench_support"]), 1.0)
        self.assertEqual(float(alpha["real_quantitative_inversion_ready"]), 0.0)
        self.assertEqual(alpha["primary_blocker"], "alpha_threshold_crossing")

        pe = by_signature["persistence_exchange_decoupling"]
        self.assertEqual(pe["alignment_stage"], "model_literature_supported_real_inversion_blocked")
        self.assertEqual(float(pe["real_glassbench_support"]), 0.0)
        self.assertEqual(pe["primary_blocker"], "alpha_threshold_crossing")

        thermo = by_signature["thermodynamic_transition"]
        self.assertEqual(thermo["alignment_stage"], "scope_boundary_not_explained")
        self.assertEqual(float(thermo["thermodynamic_claim_allowed"]), 0.0)
        self.assertEqual(float(thermo["real_glassbench_support"]), 0.0)

    def test_sota_glassbench_direct_four_point_claim_gate_blocks_proxy_promotion(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_direct_four_point_claim_gate.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        ka2d_023 = next(
            row for row in rows
            if row["system_id"] == "KA2D" and row["temperature"] == "0.23"
        )
        self.assertEqual(
            ka2d_023["four_point_claim_stage"],
            "overlap_chi4_proxy_supported_direct_four_point_blocked",
        )
        self.assertEqual(float(ka2d_023["overlap_chi4_proxy_ready"]), 1.0)
        self.assertEqual(float(ka2d_023["direct_four_point_susceptibility_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["dynamic_length_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["direct_four_point_claim_ready"]), 0.0)
        self.assertEqual(float(ka2d_023["proxy_promotion_allowed"]), 0.0)
        self.assertEqual(ka2d_023["primary_blocker"], "direct_four_point_function_and_dynamic_length")
        self.assertEqual(float(ka2d_023["thermodynamic_claim_allowed"]), 0.0)

    def test_sota_glassbench_trajectory_npz_ensemble_horizon_records_prefix_member_gap(self):
        path = ROOT / "data" / "renewal_cage_sota_glassbench_trajectory_npz_ensemble_horizon.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_key = {(row["system_id"], row["temperature"]): row for row in rows}
        ka2d_030 = by_key[("KA2D", "0.30")]
        self.assertEqual(float(ka2d_030["prefix_npz_member_count"]), 10.0)
        self.assertEqual(float(ka2d_030["extracted_curve_member_count"]), 1.0)
        self.assertEqual(float(ka2d_030["member_count_gap_to_threshold"]), 0.0)
        self.assertEqual(float(ka2d_030["prefix_member_horizon_ready"]), 1.0)
        self.assertEqual(float(ka2d_030["multi_npz_extraction_ready"]), 0.0)
        self.assertEqual(float(ka2d_030["real_reanalysis_ready"]), 0.0)
        self.assertEqual(ka2d_030["primary_blocker"], "multi_npz_observable_extraction")
        self.assertEqual(ka2d_030["horizon_stage"], "member_index_horizon_ready_extraction_blocked")

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
            expected_members = 9.0 if key == ("KA2D", "0.23") else 10.0
            self.assertEqual(float(row["prefix_npz_member_count"]), expected_members)
            self.assertEqual(float(row["required_member_count"]), 4.0)
            self.assertEqual(float(row["additional_member_count_needed"]), 0.0)
            self.assertEqual(float(row["first_member_id_visible"]), 1.0)
            self.assertEqual(float(row["full_member_id_list_visible"]), 1.0)
            self.assertEqual(float(row["member_count_threshold_pass"]), 1.0)
            self.assertEqual(float(row["publishable_ensemble_uncertainty_ready"]), 1.0)
            self.assertEqual(float(row["thermodynamic_claim_allowed"]), 0.0)
            self.assertEqual(row["primary_blocker"], "none")
            self.assertEqual(row["ensemble_audit_stage"], "visible_member_ensemble_ready_for_uncertainty")

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

    def test_trajectory_cage_jump_events_extract_particle_clock_protocol(self):
        path = ROOT / "data" / "renewal_cage_trajectory_cage_jump_events.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["protocol_id"]: row for row in rows}
        ready = by_id["synthetic_particle_cage_jump_events"]
        self.assertEqual(
            ready["event_protocol_stage"],
            "particle_resolved_cage_jump_event_clock_ready",
        )
        self.assertEqual(float(ready["particle_resolved_jump_events_ready"]), 1.0)
        self.assertEqual(float(ready["physical_time_jump_clock_ready"]), 1.0)
        self.assertEqual(float(ready["persistence_exchange_event_clock_ready"]), 1.0)
        self.assertGreater(float(ready["total_jump_event_count"]), 0.0)
        self.assertGreater(float(ready["particles_with_jump_count"]), 0.0)
        self.assertGreater(float(ready["exchange_interval_count"]), 0.0)
        self.assertEqual(ready["primary_blocker"], "none")

    def test_trajectory_event_clock_macro_predictions_score_direct_micro_closure(self):
        path = ROOT / "data" / "renewal_cage_trajectory_event_clock_macro_predictions.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_id = {row["protocol_id"]: row for row in rows}
        ready = by_id["synthetic_event_clock_macro_prediction"]
        self.assertEqual(ready["prediction_stage"], "event_clock_micro_to_macro_prediction_ready")
        self.assertEqual(float(ready["micro_to_macro_prediction_ready"]), 1.0)
        self.assertEqual(float(ready["micro_to_macro_predictions_pass"]), 1.0)
        self.assertEqual(float(ready["calibrated_from_event_clock_only"]), 1.0)
        self.assertEqual(float(ready["fit_parameters_from_macro_observables"]), 0.0)
        self.assertLess(float(ready["diffusion_z"]), 1.0)
        self.assertLess(float(ready["max_tau_alpha_z"]), 1.0)
        self.assertLess(float(ready["late_ngp_z"]), 1.0)
        self.assertLess(float(ready["chi4_peak_z"]), 1.0)
        self.assertEqual(float(ready["thermodynamic_claim_allowed"]), 0.0)

        mismatch = by_id["synthetic_event_clock_macro_late_ngp_mismatch"]
        self.assertEqual(mismatch["prediction_stage"], "event_clock_micro_to_macro_prediction_failed")
        self.assertEqual(float(mismatch["micro_to_macro_predictions_pass"]), 0.0)
        self.assertEqual(mismatch["primary_blocker"], "heldout_macro_signature_mismatch")

    def test_trajectory_event_clock_threshold_robustness_records_stable_window(self):
        path = ROOT / "data" / "renewal_cage_trajectory_event_clock_threshold_robustness.csv"
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.DictReader(f))

        by_threshold = {float(row["jump_displacement_threshold"]): row for row in rows}
        self.assertEqual(by_threshold[1.0]["robustness_stage"], "event_clock_threshold_prediction_passed")
        self.assertEqual(by_threshold[0.9]["robustness_stage"], "event_clock_threshold_prediction_passed")
        self.assertGreaterEqual(float(by_threshold[1.0]["stable_threshold_window_count"]), 2.0)
        self.assertEqual(float(by_threshold[1.0]["fit_parameters_from_macro_observables"]), 0.0)
        self.assertEqual(float(by_threshold[1.0]["thermodynamic_claim_allowed"]), 0.0)

        self.assertEqual(by_threshold[0.05]["robustness_stage"], "event_clock_threshold_prediction_failed")
        self.assertEqual(by_threshold[0.05]["primary_blocker"], "threshold_macro_signature_mismatch")
        self.assertEqual(by_threshold[1.35]["robustness_stage"], "event_clock_threshold_event_clock_incomplete")
        self.assertEqual(by_threshold[1.35]["primary_blocker"], "jump_displacement_threshold")

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
            self.assertIn("figures/renewal_cage_spatial_covariance_closure.pdf", names)
            self.assertIn("figures/renewal_cage_thermodynamic_closure.pdf", names)
            self.assertIn("figures/renewal_cage_thermodynamic_nonidentifiability.pdf", names)
            self.assertIn("figures/renewal_cage_glass_signature_claim_ladder.pdf", names)
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
            self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_npz_member_index.pdf", names)
            self.assertIn(
                "figures/renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.pdf",
                names,
            )
            self.assertIn("figures/renewal_cage_sota_glassbench_ka2d_timecode_semantics.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_timecode_curve_bridge.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_timecode_signature_support.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_four_point_claim_gate.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_real_data_closure_priority.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_real_data_acquisition_design.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_real_data_acquisition_outcome_matrix.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_manuscript_claim_registry.pdf", names)
            self.assertIn("figures/renewal_cage_weeks_hard_colloid_true_time.pdf", names)
            self.assertIn("figures/renewal_cage_weeks_hard_colloid_event_clock_censoring.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_alpha_threshold_horizon.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_alpha_anchor_rescue_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_alpha_anchor_cached_fs.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_curve.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_shape_selection.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_multik_shape.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_multik_heldout_prediction.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_post_window_prediction_targets.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_post_window_verdict.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_transport.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_pe_bound.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_displacement_tail_bound.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_multilag_crossing_canary.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_real_threshold_sweep_canary.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_threshold_sweep_ensemble_verdict.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_threshold_sweep_payload_contract.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_threshold_sweep_outcome_matrix.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_threshold_sweep_decision_power_plan.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_direct_alpha_event_clock_contract.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_sparse_lag_event_clock.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_interval_censored_first_crossing_clock.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_interval_censored_persistence_fit.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_waiting_law_selection.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_finite_exchange_envelope.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_real_cached_microdynamic_verdict.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_protocol.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_ingestion_contract.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_timecode_target.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_cache_request_contract.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_membership_probe_contract.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_public_timecode_ceiling.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_censored_window_claim_audit.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_public_window_verdict.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_experiment_design.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_uncertainty_verdict.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_outcome_matrix.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_late_recovery_decision_power_plan.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_cage_jump_proxy_canary.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_cached_particle_timecode_bridge.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_multilag_particle_cache_targets.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_cached_particle_observable_semantics.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_observable_renewal_canary.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_event_clock_threshold_readiness.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_first_npz_particle_cache_contract.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_microdynamic_closed_loop.pdf", names)
            self.assertIn("figures/renewal_cage_sota_dynamic_signature_alignment.pdf", names)
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
            self.assertIn("figures/renewal_cage_trajectory_cage_jump_events.pdf", names)
            self.assertIn("figures/renewal_cage_trajectory_event_clock_macro_predictions.pdf", names)
            self.assertIn("figures/renewal_cage_trajectory_event_clock_threshold_robustness.pdf", names)
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
            self.assertIn("figures/renewal_cage_microdynamic_prediction_scorecard.pdf", names)
            self.assertIn("figures/renewal_cage_microdynamic_minimality_audit.pdf", names)
            self.assertIn("figures/renewal_cage_sota_experimental_verdict_matrix.pdf", names)
            self.assertIn("figures/renewal_cage_sota_glassbench_real_evidence_claim_synthesis.pdf", names)
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
        self.assertIn("figures/renewal_cage_thermodynamic_nonidentifiability.pdf", main_tex)
        self.assertIn("figures/renewal_cage_glass_signature_claim_ladder.pdf", main_tex)
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
        self.assertIn("figures/renewal_cage_sota_glassbench_trajectory_npz_member_index.pdf", main_tex)
        self.assertIn(
            "figures/renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.pdf",
            main_tex,
        )
        self.assertIn("figures/renewal_cage_sota_glassbench_ka2d_timecode_semantics.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_timecode_curve_bridge.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_timecode_signature_support.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_alpha_threshold_horizon.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_cage_jump_proxy_canary.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_observable_renewal_canary.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_glassbench_microdynamic_closed_loop.pdf", main_tex)
        self.assertIn("figures/renewal_cage_sota_dynamic_signature_alignment.pdf", main_tex)
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
            "figures/renewal_cage_sota_glassbench_trajectory_npz_member_index.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.pdf",
            "figures/renewal_cage_sota_glassbench_ka2d_timecode_semantics.pdf",
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
            "figures/renewal_cage_microdynamic_prediction_scorecard.pdf",
            "figures/renewal_cage_microdynamic_minimality_audit.pdf",
            "figures/renewal_cage_sota_experimental_verdict_matrix.pdf",
            "figures/renewal_cage_sota_glassbench_real_evidence_claim_synthesis.pdf",
            "figures/renewal_cage_sota_glassbench_direct_four_point_claim_gate.pdf",
            "figures/renewal_cage_sota_glassbench_real_data_closure_priority.pdf",
            "figures/renewal_cage_sota_glassbench_real_data_acquisition_design.pdf",
            "figures/renewal_cage_sota_glassbench_real_data_acquisition_outcome_matrix.pdf",
            "figures/renewal_cage_sota_glassbench_manuscript_claim_registry.pdf",
            "figures/renewal_cage_sota_glassbench_short_window_trend_canary.pdf",
            "figures/renewal_cage_sota_glassbench_trajectory_timebase_bridge.pdf",
            "figures/renewal_cage_sota_glassbench_frame_time_mapping_audit.pdf",
            "figures/renewal_cage_sota_glassbench_real_inversion_gap_ledger.pdf",
            "figures/renewal_cage_sota_glassbench_real_inversion_unlock_protocol.pdf",
            "figures/renewal_cage_sota_glassbench_alpha_anchor_rescue_protocol.pdf",
            "figures/renewal_cage_sota_glassbench_alpha_anchor_cached_fs.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_curve.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_shape_selection.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_multik_shape.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_multik_heldout_prediction.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_post_window_prediction_targets.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_post_window_verdict.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_transport.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_pe_bound.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_displacement_tail_bound.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_multilag_crossing_canary.pdf",
            "figures/renewal_cage_sota_glassbench_real_threshold_sweep_canary.pdf",
            "figures/renewal_cage_sota_glassbench_threshold_sweep_ensemble_verdict.pdf",
            "figures/renewal_cage_sota_glassbench_threshold_sweep_payload_contract.pdf",
            "figures/renewal_cage_sota_glassbench_threshold_sweep_outcome_matrix.pdf",
            "figures/renewal_cage_sota_glassbench_threshold_sweep_decision_power_plan.pdf",
            "figures/renewal_cage_sota_glassbench_direct_alpha_event_clock_contract.pdf",
            "figures/renewal_cage_sota_glassbench_sparse_lag_event_clock.pdf",
            "figures/renewal_cage_sota_glassbench_interval_censored_first_crossing_clock.pdf",
            "figures/renewal_cage_sota_glassbench_interval_censored_persistence_fit.pdf",
            "figures/renewal_cage_sota_glassbench_waiting_law_selection.pdf",
            "figures/renewal_cage_sota_glassbench_finite_exchange_envelope.pdf",
            "figures/renewal_cage_sota_glassbench_real_cached_microdynamic_verdict.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_protocol.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_ingestion_contract.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_timecode_target.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_cache_request_contract.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_membership_probe_contract.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_public_timecode_ceiling.pdf",
            "figures/renewal_cage_sota_glassbench_censored_window_claim_audit.pdf",
            "figures/renewal_cage_sota_glassbench_public_window_verdict.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_experiment_design.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_uncertainty_verdict.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_outcome_matrix.pdf",
            "figures/renewal_cage_sota_glassbench_late_recovery_decision_power_plan.pdf",
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
            "figures/renewal_cage_trajectory_cage_jump_events.pdf",
            "figures/renewal_cage_trajectory_event_clock_macro_predictions.pdf",
            "figures/renewal_cage_trajectory_event_clock_threshold_robustness.pdf",
            "figures/renewal_cage_sota_glassbench_cached_particle_timecode_bridge.pdf",
            "figures/renewal_cage_sota_glassbench_multilag_particle_cache_targets.pdf",
            "figures/renewal_cage_sota_glassbench_cached_particle_observable_semantics.pdf",
            "figures/renewal_cage_sota_glassbench_event_clock_threshold_readiness.pdf",
            "figures/renewal_cage_sota_glassbench_first_npz_particle_cache_contract.pdf",
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
            first_sota_glassbench_trajectory_npz_member_index = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_npz_member_index.pdf"
            ).read_bytes()
            first_sota_glassbench_trajectory_member_ensemble_observable = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.pdf"
            ).read_bytes()
            first_sota_glassbench_ka2d_timecode_semantics = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_ka2d_timecode_semantics.pdf"
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
            first_trajectory_cage_jump_events = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_cage_jump_events.pdf"
            ).read_bytes()
            first_trajectory_event_clock_macro_predictions = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_event_clock_macro_predictions.pdf"
            ).read_bytes()
            first_trajectory_event_clock_threshold_robustness = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_event_clock_threshold_robustness.pdf"
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
            second_sota_glassbench_trajectory_npz_member_index = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_npz_member_index.pdf"
            ).read_bytes()
            second_sota_glassbench_trajectory_member_ensemble_observable = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_trajectory_member_ensemble_observable.pdf"
            ).read_bytes()
            second_sota_glassbench_ka2d_timecode_semantics = (
                ROOT
                / "paper"
                / "figures"
                / "renewal_cage_sota_glassbench_ka2d_timecode_semantics.pdf"
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
            second_trajectory_cage_jump_events = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_cage_jump_events.pdf"
            ).read_bytes()
            second_trajectory_event_clock_macro_predictions = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_event_clock_macro_predictions.pdf"
            ).read_bytes()
            second_trajectory_event_clock_threshold_robustness = (
                ROOT / "paper" / "figures" / "renewal_cage_trajectory_event_clock_threshold_robustness.pdf"
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
            first_sota_glassbench_trajectory_npz_member_index,
            second_sota_glassbench_trajectory_npz_member_index,
        )
        self.assertEqual(
            first_sota_glassbench_trajectory_member_ensemble_observable,
            second_sota_glassbench_trajectory_member_ensemble_observable,
        )
        self.assertEqual(
            first_sota_glassbench_ka2d_timecode_semantics,
            second_sota_glassbench_ka2d_timecode_semantics,
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
        self.assertEqual(first_trajectory_cage_jump_events, second_trajectory_cage_jump_events)
        self.assertEqual(
            first_trajectory_event_clock_macro_predictions,
            second_trajectory_event_clock_macro_predictions,
        )
        self.assertEqual(
            first_trajectory_event_clock_threshold_robustness,
            second_trajectory_event_clock_threshold_robustness,
        )
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
