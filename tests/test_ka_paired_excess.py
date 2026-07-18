import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_summary():
    path = ROOT / "scripts" / "summarize_ka_segment_splice_paired_excess.py"
    spec = importlib.util.spec_from_file_location("paired_excess_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODELS = (
    "within_particle_segment_shuffle",
    "cross_particle_segment_splice",
)
LENGTHS = (1, 2, 5, 10, 25, 50, 125, 250)


def score_rows():
    baselines = {1: 0.9, 2: 2.1, 3: 3.3}
    within_excess = {
        1: (21.0, 18.0, 12.0, 9.0, 5.0, 3.0, 1.0),
        2: (21.0, 18.0, 13.0, 10.0, 6.0, 4.0, 3.0),
        3: (18.0, 14.0, 8.0, 5.0, 0.5, -2.0, -1.8),
    }
    rows = []
    for replicate in (1, 2, 3):
        for model in MODELS:
            finite = within_excess[replicate]
            if model == "cross_particle_segment_splice":
                finite = tuple(value + 2.0 for value in finite)
            for length, excess in zip(LENGTHS[:-1], finite):
                rows.append(
                    {
                        "temperature": 0.45,
                        "model": model,
                        "segment_length": float(length),
                        "replicate": float(replicate),
                        "higher_order_score": baselines[replicate] + excess,
                        "microdynamic_closure_claim_allowed": 0.0,
                        "spatial_facilitation_claim_allowed": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
            rows.append(
                {
                    "temperature": 0.45,
                    "model": model,
                    "segment_length": 250.0,
                    "replicate": float(replicate),
                    "higher_order_score": baselines[replicate],
                    "microdynamic_closure_claim_allowed": 0.0,
                    "spatial_facilitation_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
    return rows


def cell_rows():
    return [
        {
            "temperature": 0.45,
            "model": model,
            "segment_length": float(length),
            "global_source_segment_schedule_preserved": 1.0,
            "microdynamic_closure_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for model in MODELS
        for length in LENGTHS
    ]


def source_verdict():
    return [
        {
            "mechanism_state": "mechanism_unresolved",
            "low_mechanism_identifiable_against_full_path_control": 0.0,
            "low_full_path_control_all_replicates_pass": 0.0,
            "low_full_path_control_failed_replicate_count": 2.0,
            "global_source_segment_schedule_preserved": 1.0,
            "independent_replicate_memory_lower_bound_claim_allowed": 0.0,
            "owner_identity_sufficiency_claim_allowed": 0.0,
            "static_vs_finite_exchange_resolved": 0.0,
            "microdynamic_closure_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
    ]


def provenance_rows():
    return [
        {
            "temperature": 0.45,
            "replicate": float(replicate),
            "source_sha256": "shared-parent-sha256",
            "source_frame_index": float(frame),
            "velocity_seed": float(seed),
            "independence_class": "decorrelated_parent_frames_plus_velocity_seeds",
            "independently_prepared_parent_samples": False,
            "thermodynamic_claim_allowed": 0.0,
        }
        for replicate, frame, seed in (
            (1, 5000, 45117),
            (2, 35000, 45157),
            (3, 65000, 45201),
        )
    ]


class PairedExcessGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_summary()

    def test_centers_each_replicate_and_identifies_only_short_prefix(self):
        rows, gate = self.summary.classify_paired_excess_gate(
            score_rows(), cell_rows(), source_verdict(), provenance_rows()
        )
        self.assertEqual(len(rows), 14)
        self.assertEqual(
            {(row["model"], row["segment_length"]) for row in rows},
            {(model, float(length)) for model in MODELS for length in LENGTHS[:-1]},
        )
        within_one = next(
            row
            for row in rows
            if row["model"] == "within_particle_segment_shuffle"
            and row["segment_length"] == 1.0
        )
        self.assertAlmostEqual(within_one["replicate_1_paired_excess"], 21.0)
        self.assertEqual(within_one["paired_degradation_identified"], 1.0)
        for model in MODELS:
            identified = [
                int(row["segment_length"])
                for row in rows
                if row["model"] == model
                and row["paired_degradation_identified"] == 1.0
            ]
            self.assertEqual(identified, [1, 2, 5, 10])
        self.assertEqual(gate["identified_prefix_max_segment_length"], 10.0)
        self.assertEqual(gate["identified_prefix_max_tau"], 200.0)
        self.assertEqual(gate["short_horizon_information_loss_supported_exploratory"], 1.0)
        self.assertEqual(gate["owner_identity_information_supported_exploratory"], 1.0)
        self.assertEqual(gate["independently_prepared_parent_samples"], 0.0)
        self.assertEqual(gate["independent_replicate_count"], 0.0)
        self.assertEqual(gate["replicate_count"], 3.0)
        self.assertEqual(gate["replicate_provenance_validation_pass"], 1.0)
        self.assertEqual(
            gate["independence_class"],
            "decorrelated_parent_frames_plus_velocity_seeds",
        )
        self.assertEqual(within_one["independent_replicate_count"], 0.0)
        self.assertEqual(within_one["replicate_count"], 3.0)
        self.assertEqual(
            within_one["ci95_method"],
            "student_t_replicate_first_correlated_parent_exploratory_df2",
        )

        computed = self.summary.compute_paired_excess_rows(
            score_rows(), provenance_rows()
        )
        self.assertEqual(computed, rows)

    def test_rejects_incomplete_inputs_full_path_disagreement_and_open_claims(self):
        malformed_scores = []
        malformed_scores.append(score_rows()[:-1])
        mismatched = score_rows()
        mismatched[-1] = {**mismatched[-1], "higher_order_score": 3.4}
        malformed_scores.append(mismatched)
        fractional_length = score_rows()
        fractional_length[0] = {
            **fractional_length[0],
            "segment_length": 1.5,
        }
        malformed_scores.append(fractional_length)
        fractional_replicate = score_rows()
        fractional_replicate[0] = {
            **fractional_replicate[0],
            "replicate": 1.5,
        }
        malformed_scores.append(fractional_replicate)
        for scores in malformed_scores:
            with self.subTest():
                with self.assertRaises(ValueError):
                    self.summary.compute_paired_excess_rows(scores, provenance_rows())

        open_claim = cell_rows()
        open_claim[0] = {**open_claim[0], "microdynamic_closure_claim_allowed": 1.0}
        rows, gate = self.summary.classify_paired_excess_gate(
            score_rows(), open_claim, source_verdict(), provenance_rows()
        )
        self.assertEqual(rows, [])
        self.assertEqual(gate["paired_input_exactness_pass"], 0.0)

        resolved = source_verdict()
        resolved[0] = {**resolved[0], "mechanism_state": "resolved"}
        rows, gate = self.summary.classify_paired_excess_gate(
            score_rows(), cell_rows(), resolved, provenance_rows()
        )
        self.assertEqual(rows, [])
        self.assertEqual(gate["source_verdict_fail_closed"], 1.0)

        source_claim = source_verdict()
        source_claim[0] = {
            **source_claim[0],
            "owner_identity_sufficiency_claim_allowed": 1.0,
        }
        rows, gate = self.summary.classify_paired_excess_gate(
            score_rows(), cell_rows(), source_claim, provenance_rows()
        )
        self.assertEqual(rows, [])
        self.assertEqual(gate["source_verdict_fail_closed"], 1.0)

        inconsistent = source_verdict()
        inconsistent[0] = {
            **inconsistent[0],
            "low_full_path_control_all_replicates_pass": 1.0,
            "low_full_path_control_failed_replicate_count": 0.0,
        }
        rows, gate = self.summary.classify_paired_excess_gate(
            score_rows(), cell_rows(), inconsistent, provenance_rows()
        )
        self.assertEqual(rows, [])
        self.assertEqual(gate["source_verdict_fail_closed"], 1.0)

    def test_rejects_missing_or_false_independence_provenance(self):
        malformed = [provenance_rows()[:-1]]
        independent = provenance_rows()
        independent[0] = {
            **independent[0],
            "independently_prepared_parent_samples": True,
        }
        malformed.append(independent)
        mixed_parent = provenance_rows()
        mixed_parent[0] = {**mixed_parent[0], "source_sha256": "other-parent"}
        malformed.append(mixed_parent)
        open_claim = provenance_rows()
        open_claim[0] = {**open_claim[0], "thermodynamic_claim_allowed": True}
        malformed.append(open_claim)
        for provenance in malformed:
            with self.subTest():
                rows, gate = self.summary.classify_paired_excess_gate(
                    score_rows(), cell_rows(), source_verdict(), provenance
                )
                self.assertEqual(rows, [])
                self.assertEqual(gate["replicate_provenance_validation_pass"], 0.0)
                self.assertEqual(gate["independent_replicate_count"], 0.0)

    def test_gate_keeps_all_strong_claims_closed(self):
        _, gate = self.summary.classify_paired_excess_gate(
            score_rows(), cell_rows(), source_verdict(), provenance_rows()
        )
        self.assertEqual(gate["mechanism_state"], "mechanism_unresolved")
        self.assertEqual(gate["next_required_action"], "replicate_resolved_full_path_baseline_or_new_trajectory_validation")
        for field in self.summary.CLOSED_CLAIM_FIELDS:
            self.assertEqual(gate[field], 0.0, field)

    def test_csv_float_serialization_ignores_platform_last_bit_drift(self):
        self.assertEqual(self.summary.CSV_FLOAT_SIGNIFICANT_DIGITS, 15)
        self.assertEqual(
            self.summary.canonical_csv_value(2.0953229763876298),
            "2.09532297638763",
        )
        self.assertEqual(
            self.summary.canonical_csv_value(16.999163731272752),
            self.summary.canonical_csv_value(16.999163731272755),
        )
        self.assertEqual(
            self.summary.canonical_csv_value(0.2777328943399178),
            self.summary.canonical_csv_value(0.27773289433991805),
        )
        self.assertNotEqual(
            self.summary.canonical_csv_value(1.0),
            self.summary.canonical_csv_value(1.0001),
        )

    def test_svg_is_deterministic_and_states_exploratory_boundary(self):
        rows, gate = self.summary.classify_paired_excess_gate(
            score_rows(), cell_rows(), source_verdict(), provenance_rows()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.svg"
            self.summary.write_paired_excess_svg(path, rows, gate)
            first = path.read_bytes()
            self.summary.write_paired_excess_svg(path, rows, gate)
            self.assertEqual(first, path.read_bytes())
        text = first.decode()
        self.assertIn("Paired excess over replicate full-path baseline", text)
        self.assertIn("post-run exploratory", text)
        self.assertIn("mechanism unresolved", text)
        self.assertIn("data-identified=\"true\"", text)
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("inf", text.lower())


if __name__ == "__main__":
    unittest.main()
