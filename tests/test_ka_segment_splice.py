import sys
import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ka_segment_splice import (  # noqa: E402
    SegmentSurrogate,
    audit_segment_surrogate,
    cumulative_observables_many_lags,
    cross_particle_segment_splice,
    segment_slices,
    within_particle_segment_shuffle,
)
from ka_replicates import cumulative_block_observables  # noqa: E402


def load_script_module(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded_blocks(
    particle_count: int = 3,
    block_count: int = 7,
) -> np.ndarray:
    blocks = np.empty((particle_count, block_count, 3), dtype=float)
    for particle in range(particle_count):
        for block in range(block_count):
            blocks[particle, block] = [
                100.0 * particle + block,
                10.0 * particle + 0.1 * block,
                particle - block,
            ]
    return blocks


class SegmentRepresentationTests(unittest.TestCase):
    def test_segment_slices_keep_variable_terminal_segment(self):
        self.assertEqual(segment_slices(7, 3), ((0, 3), (3, 6), (6, 7)))
        self.assertEqual(segment_slices(6, 3), ((0, 3), (3, 6)))
        self.assertEqual(segment_slices(7, 7), ((0, 7),))

    def test_invalid_segment_inputs_fail(self):
        for block_count, segment_length in (
            (0, 1),
            (7, 0),
            (7, 8),
            (True, 1),
            (7, True),
        ):
            with self.assertRaises(ValueError):
                segment_slices(block_count, segment_length)

        blocks = encoded_blocks()
        invalid_arrays = (
            blocks[:, :, 0],
            blocks[:, :, :2],
            np.empty((0, 7, 3)),
            blocks.copy(),
        )
        invalid_arrays[-1][0, 0, 0] = np.nan
        for values in invalid_arrays:
            with self.assertRaises(ValueError):
                within_particle_segment_shuffle(
                    values,
                    segment_length=3,
                    rng=np.random.default_rng(1),
                )
        with self.assertRaises(ValueError):
            within_particle_segment_shuffle(
                blocks,
                segment_length=3,
                rng="not-a-generator",
            )


class WithinParticleShuffleTests(unittest.TestCase):
    def test_shuffle_preserves_particle_multisets_and_token_order(self):
        blocks = encoded_blocks()
        surrogate = within_particle_segment_shuffle(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(20260718),
        )

        self.assertEqual(surrogate.blocks.shape, blocks.shape)
        self.assertEqual(surrogate.source_particle.shape, (3, 3))
        self.assertEqual(surrogate.source_segment.shape, (3, 3))
        np.testing.assert_array_equal(
            surrogate.source_particle,
            np.broadcast_to(np.arange(3)[:, None], (3, 3)),
        )
        np.testing.assert_array_equal(
            surrogate.source_segment,
            np.broadcast_to(surrogate.source_segment[0], (3, 3)),
        )
        self.assertFalse(np.array_equal(surrogate.source_segment[0], np.arange(3)))
        np.testing.assert_array_equal(
            surrogate.target_segment_lengths,
            np.array([3 if segment < 2 else 1 for segment in surrogate.source_segment[0]]),
        )

        for particle in range(len(blocks)):
            source_rows = sorted(map(tuple, blocks[particle]))
            target_rows = sorted(map(tuple, surrogate.blocks[particle]))
            self.assertEqual(target_rows, source_rows)
            cursor = 0
            for source_segment, length in zip(
                surrogate.source_segment[particle],
                surrogate.target_segment_lengths,
                strict=True,
            ):
                source_start, source_stop = segment_slices(7, 3)[source_segment]
                self.assertEqual(source_stop - source_start, length)
                np.testing.assert_array_equal(
                    surrogate.blocks[particle, cursor : cursor + length],
                    blocks[particle, source_start:source_stop],
                )
                cursor += int(length)
            self.assertEqual(cursor, 7)

    def test_fixed_seed_is_deterministic_and_full_path_is_identity(self):
        blocks = encoded_blocks()
        first = within_particle_segment_shuffle(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(41),
        )
        second = within_particle_segment_shuffle(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(41),
        )
        np.testing.assert_array_equal(first.blocks, second.blocks)
        np.testing.assert_array_equal(first.source_segment, second.source_segment)

        full = within_particle_segment_shuffle(
            blocks,
            segment_length=7,
            rng=np.random.default_rng(41),
        )
        np.testing.assert_array_equal(full.blocks, blocks)
        np.testing.assert_array_equal(full.source_particle, np.arange(3)[:, None])
        np.testing.assert_array_equal(full.source_segment, np.zeros((3, 1), dtype=int))
        np.testing.assert_array_equal(full.target_segment_lengths, [7])


class CrossParticleSpliceTests(unittest.TestCase):
    def test_cross_splice_uses_every_token_once_without_owner_continuity(self):
        blocks = encoded_blocks(particle_count=7, block_count=7)
        within = within_particle_segment_shuffle(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(151),
        )
        cross = cross_particle_segment_splice(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(151),
        )

        np.testing.assert_array_equal(
            cross.target_segment_lengths,
            within.target_segment_lengths,
        )
        np.testing.assert_array_equal(cross.source_segment, within.source_segment)
        targets = np.arange(len(blocks))[:, None]
        self.assertTrue(np.all(cross.source_particle != targets))
        self.assertTrue(
            np.all(cross.source_particle[:, 1:] != cross.source_particle[:, :-1])
        )
        token_ids = (
            cross.source_particle * cross.source_particle.shape[1]
            + cross.source_segment
        )
        np.testing.assert_array_equal(
            np.sort(token_ids, axis=None),
            np.arange(token_ids.size),
        )

        slices = segment_slices(7, 3)
        for target in range(len(blocks)):
            cursor = 0
            for source_particle, source_segment, length in zip(
                cross.source_particle[target],
                cross.source_segment[target],
                cross.target_segment_lengths,
                strict=True,
            ):
                start, stop = slices[source_segment]
                self.assertEqual(stop - start, length)
                np.testing.assert_array_equal(
                    cross.blocks[target, cursor : cursor + length],
                    blocks[source_particle, start:stop],
                )
                cursor += int(length)

    def test_fixed_seed_is_deterministic_and_impossible_assignment_fails(self):
        blocks = encoded_blocks(particle_count=7, block_count=7)
        first = cross_particle_segment_splice(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(99),
        )
        second = cross_particle_segment_splice(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(99),
        )
        np.testing.assert_array_equal(first.blocks, second.blocks)
        np.testing.assert_array_equal(first.source_particle, second.source_particle)

        with self.assertRaises(ValueError):
            cross_particle_segment_splice(
                encoded_blocks(particle_count=2, block_count=7),
                segment_length=3,
                rng=np.random.default_rng(9),
                maximum_restarts=12,
            )
        for maximum_restarts in (0, True):
            with self.assertRaises(ValueError):
                cross_particle_segment_splice(
                    blocks,
                    segment_length=3,
                    rng=np.random.default_rng(9),
                    maximum_restarts=maximum_restarts,
                )


class SegmentAuditTests(unittest.TestCase):
    def test_exact_audit_separates_within_and_cross_information(self):
        blocks = encoded_blocks(particle_count=7, block_count=7)
        within = within_particle_segment_shuffle(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(17),
        )
        cross = cross_particle_segment_splice(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(17),
        )

        within_audit = audit_segment_surrogate(
            blocks,
            within,
            segment_length=3,
            model="within_particle_segment_shuffle",
        )
        cross_audit = audit_segment_surrogate(
            blocks,
            cross,
            segment_length=3,
            model="cross_particle_segment_splice",
        )
        for audit in (within_audit, cross_audit):
            self.assertEqual(audit["source_token_reuse_minimum"], 1.0)
            self.assertEqual(audit["source_token_reuse_maximum"], 1.0)
            self.assertEqual(audit["ordered_token_multiset_preserved"], 1.0)
            self.assertEqual(audit["global_block_provenance_multiset_preserved"], 1.0)
            self.assertEqual(audit["global_block_vector_multiset_preserved"], 1.0)
            self.assertEqual(audit["segment_length_histogram_preserved"], 1.0)
            self.assertEqual(audit["internal_adjacent_pair_multiset_preserved"], 1.0)
            self.assertEqual(audit["complete_particle_paths"], 1.0)
            self.assertEqual(audit["heldout_path_used_in_prediction"], 0.0)
            self.assertEqual(audit["macro_fit_parameter_count"], 0.0)
            self.assertEqual(audit["microdynamic_closure_claim_allowed"], 0.0)
            self.assertEqual(audit["spatial_facilitation_claim_allowed"], 0.0)
            self.assertEqual(audit["thermodynamic_claim_allowed"], 0.0)
            self.assertAlmostEqual(
                audit["guaranteed_internal_adjacency_fraction"],
                4.0 / 6.0,
            )
        self.assertEqual(within_audit["within_particle_vector_multiset_preserved"], 1.0)
        self.assertEqual(within_audit["same_source_assignment_fraction"], 1.0)
        self.assertEqual(cross_audit["within_particle_vector_multiset_preserved"], 0.0)
        self.assertEqual(cross_audit["same_source_assignment_fraction"], 0.0)
        self.assertEqual(cross_audit["adjacent_same_source_segment_fraction"], 0.0)

    def test_audit_uses_provenance_and_detects_tampering_with_duplicate_values(self):
        blocks = np.zeros((7, 7, 3), dtype=float)
        cross = cross_particle_segment_splice(
            blocks,
            segment_length=3,
            rng=np.random.default_rng(71),
        )
        audit = audit_segment_surrogate(
            blocks,
            cross,
            segment_length=3,
            model="cross_particle_segment_splice",
        )
        self.assertEqual(audit["global_block_provenance_multiset_preserved"], 1.0)

        duplicated_provenance = cross.source_particle.copy()
        duplicated_provenance[0, 0] = duplicated_provenance[1, 0]
        tampered_provenance = SegmentSurrogate(
            blocks=cross.blocks.copy(),
            source_particle=duplicated_provenance,
            source_segment=cross.source_segment.copy(),
            target_segment_lengths=cross.target_segment_lengths.copy(),
        )
        provenance_audit = audit_segment_surrogate(
            blocks,
            tampered_provenance,
            segment_length=3,
            model="cross_particle_segment_splice",
        )
        self.assertEqual(provenance_audit["global_block_provenance_multiset_preserved"], 0.0)
        self.assertEqual(provenance_audit["source_token_reuse_maximum"], 2.0)

        changed = cross.blocks.copy()
        changed[0, 0, 0] = 1.0
        tampered_values = SegmentSurrogate(
            blocks=changed,
            source_particle=cross.source_particle.copy(),
            source_segment=cross.source_segment.copy(),
            target_segment_lengths=cross.target_segment_lengths.copy(),
        )
        value_audit = audit_segment_surrogate(
            blocks,
            tampered_values,
            segment_length=3,
            model="cross_particle_segment_splice",
        )
        self.assertEqual(value_audit["global_block_vector_multiset_preserved"], 0.0)

    def test_full_path_cross_control_is_a_whole_path_permutation(self):
        blocks = encoded_blocks(particle_count=7, block_count=7)
        full = cross_particle_segment_splice(
            blocks,
            segment_length=7,
            rng=np.random.default_rng(33),
        )
        audit = audit_segment_surrogate(
            blocks,
            full,
            segment_length=7,
            model="cross_particle_segment_splice",
        )
        self.assertEqual(audit["full_path_control"], 1.0)
        self.assertEqual(audit["complete_path_ensemble_equal"], 1.0)
        np.testing.assert_array_equal(
            np.sort(full.source_particle[:, 0]),
            np.arange(len(blocks)),
        )
        for target, source in enumerate(full.source_particle[:, 0]):
            np.testing.assert_array_equal(full.blocks[target], blocks[source])


class MultiLagObservableTests(unittest.TestCase):
    def test_many_lag_observables_match_frozen_single_lag_semantics(self):
        rng = np.random.default_rng(8102)
        blocks = rng.normal(size=(4, 11, 3))
        block_counts = (1, 2, 5, 11)
        wave_numbers = np.array([2.0, 4.0, 7.25])

        observed = cumulative_observables_many_lags(
            blocks,
            block_counts=block_counts,
            wave_numbers=wave_numbers,
        )

        self.assertEqual(set(observed), set(block_counts))
        for block_count in block_counts:
            reference = cumulative_block_observables(
                blocks,
                block_count=block_count,
                wave_numbers=wave_numbers,
            )
            self.assertEqual(set(observed[block_count]), set(reference))
            for key, value in reference.items():
                self.assertAlmostEqual(observed[block_count][key], value, places=12)

    def test_many_lag_observables_reject_invalid_counts_and_wave_numbers(self):
        blocks = encoded_blocks()
        for block_counts, wave_numbers in (
            ((), np.array([2.0])),
            ((0,), np.array([2.0])),
            ((8,), np.array([2.0])),
            ((1, 1), np.array([2.0])),
            ((1,), np.array([])),
            ((1,), np.array([0.0])),
            ((1,), np.array([np.nan])),
        ):
            with self.assertRaises(ValueError):
                cumulative_observables_many_lags(
                    blocks,
                    block_counts=block_counts,
                    wave_numbers=wave_numbers,
                )


def valid_segment_classifier_inputs():
    models = (
        "within_particle_segment_shuffle",
        "cross_particle_segment_splice",
    )
    lengths = (1, 2)
    replicates = (1, 2)
    realizations = (0, 1)
    quality = []
    summary = []
    for model in models:
        for length in lengths:
            summary.append(
                {
                    "model": model,
                    "temperature": 0.45,
                    "segment_length": float(length),
                    "lag": 20.0,
                    "independent_replicate_count": 2.0,
                    "ensemble_msd_relative_error": 0.05,
                    "ensemble_ngp_absolute_error": 0.10,
                    "ensemble_absolute_error_fs_k2": 0.01,
                    "ensemble_msd_mc_relative_se": 0.001,
                    "ensemble_ngp_mc_se": 0.01,
                    "ensemble_fs_k2_mc_se": 0.001,
                    "heldout_path_used_in_prediction": 0.0,
                    "macro_fit_parameter_count": 0.0,
                    "microdynamic_closure_claim_allowed": 0.0,
                    "spatial_facilitation_claim_allowed": 0.0,
                    "thermodynamic_claim_allowed": 0.0,
                }
            )
            for replicate in replicates:
                for realization in realizations:
                    quality.append(
                        {
                            "model": model,
                            "temperature": 0.45,
                            "segment_length": float(length),
                            "replicate": float(replicate),
                            "realization": float(realization),
                            "source_token_reuse_minimum": 1.0,
                            "source_token_reuse_maximum": 1.0,
                            "ordered_token_multiset_preserved": 1.0,
                            "global_block_provenance_multiset_preserved": 1.0,
                            "global_block_vector_multiset_preserved": 1.0,
                            "segment_length_histogram_preserved": 1.0,
                            "internal_adjacent_pair_multiset_preserved": 1.0,
                            "complete_particle_paths": 1.0,
                            "within_particle_vector_multiset_preserved": (
                                1.0
                                if model == "within_particle_segment_shuffle"
                                else 0.0
                            ),
                            "same_source_assignment_fraction": (
                                1.0
                                if model == "within_particle_segment_shuffle"
                                else 0.0
                            ),
                            "adjacent_same_source_segment_fraction": 0.0,
                            "heldout_path_used_in_prediction": 0.0,
                            "macro_fit_parameter_count": 0.0,
                            "microdynamic_closure_claim_allowed": 0.0,
                            "spatial_facilitation_claim_allowed": 0.0,
                            "thermodynamic_claim_allowed": 0.0,
                        }
                    )
    stationarity = [
        {
            "temperature": 0.45,
            "comparison": comparison,
            "curve_transfer_pass": 1.0,
        }
        for comparison in ("early_late", "early_heldout", "late_heldout")
    ]
    return summary, quality, stationarity


class SegmentAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script_module(
            "analyze_ka_segment_splice_gate.py",
            "analyze_ka_segment_splice_gate_aggregation",
        )

    def test_replicates_are_averaged_equally_after_realization_statistics(self):
        rows = []
        for replicate, predicted_msd, predicted_ngp, predicted_fs in (
            (1, 1.0, 0.2, 0.7),
            (2, 3.0, 0.4, 0.9),
        ):
            rows.append(
                {
                    "temperature": 0.45,
                    "model": "within_particle_segment_shuffle",
                    "segment_length": 2.0,
                    "replicate": float(replicate),
                    "lag": 20.0,
                    "predicted_msd": predicted_msd,
                    "observed_msd": 2.0,
                    "predicted_msd_mc_se": 0.2,
                    "predicted_ngp": predicted_ngp,
                    "observed_ngp": 0.3,
                    "predicted_ngp_mc_se": 0.02,
                    "predicted_fs_k2": predicted_fs,
                    "observed_fs_k2": 0.8,
                    "predicted_fs_k2_mc_se": 0.004,
                }
            )
        summary = self.analysis.summarize_segment_rows(
            rows,
            fs_keys=("observed_fs_k2",),
        )

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["independent_replicate_count"], 2.0)
        self.assertAlmostEqual(row["predicted_msd"], 2.0)
        self.assertAlmostEqual(row["ensemble_msd_relative_error"], 0.0)
        self.assertAlmostEqual(row["predicted_ngp"], 0.3)
        self.assertAlmostEqual(row["ensemble_ngp_absolute_error"], 0.0)
        self.assertAlmostEqual(row["predicted_fs_k2"], 0.8)
        self.assertAlmostEqual(row["ensemble_absolute_error_fs_k2"], 0.0)
        self.assertAlmostEqual(
            row["ensemble_msd_mc_relative_se"],
            np.sqrt(0.2**2 + 0.2**2) / 2.0 / 2.0,
        )
        self.assertEqual(row["microdynamic_closure_claim_allowed"], 0.0)
        self.assertEqual(row["spatial_facilitation_claim_allowed"], 0.0)
        self.assertEqual(row["thermodynamic_claim_allowed"], 0.0)

    def test_empty_scattering_or_duplicate_replicate_rows_fail(self):
        with self.assertRaises(ValueError):
            self.analysis.summarize_segment_rows([], fs_keys=("observed_fs_k2",))
        with self.assertRaises(ValueError):
            self.analysis.summarize_segment_rows(
                [{"temperature": 0.45}],
                fs_keys=(),
            )
        duplicate = {
            "temperature": 0.45,
            "model": "within_particle_segment_shuffle",
            "segment_length": 2.0,
            "replicate": 1.0,
            "lag": 20.0,
            "predicted_msd": 1.0,
            "observed_msd": 1.0,
            "predicted_msd_mc_se": 0.0,
            "predicted_ngp": 0.0,
            "observed_ngp": 0.0,
            "predicted_ngp_mc_se": 0.0,
            "predicted_fs_k2": 1.0,
            "observed_fs_k2": 1.0,
            "predicted_fs_k2_mc_se": 0.0,
        }
        with self.assertRaises(ValueError):
            self.analysis.summarize_segment_rows(
                [duplicate, dict(duplicate)],
                fs_keys=("observed_fs_k2",),
            )


class SegmentCellClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script_module(
            "analyze_ka_segment_splice_gate.py",
            "analyze_ka_segment_splice_gate_classifier",
        )

    def classify(self, summary, quality, stationarity):
        return self.analysis.classify_segment_cells(
            summary,
            quality,
            stationarity,
            expected_grids={0.45: (1, 2)},
            expected_replicates={0.45: (1, 2)},
            expected_realizations=2,
        )

    def test_complete_cells_pass_only_from_recomputed_inputs(self):
        summary, quality, stationarity = valid_segment_classifier_inputs()
        for row in summary:
            row["cell_pass"] = 0.0
        cells = self.classify(summary, quality, stationarity)

        self.assertEqual(len(cells), 4)
        self.assertTrue(all(row["cell_pass"] == 1.0 for row in cells))
        self.assertTrue(all(row["realization_completeness_pass"] == 1.0 for row in cells))
        self.assertTrue(all(row["exact_information_pass"] == 1.0 for row in cells))
        self.assertTrue(all(row["stationarity_control_pass"] == 1.0 for row in cells))

    def test_each_quality_and_claim_boundary_can_reject_a_cell(self):
        quality_fields = (
            ("source_token_reuse_minimum", 0.0),
            ("source_token_reuse_maximum", 2.0),
            ("ordered_token_multiset_preserved", 0.0),
            ("global_block_provenance_multiset_preserved", 0.0),
            ("global_block_vector_multiset_preserved", 0.0),
            ("segment_length_histogram_preserved", 0.0),
            ("internal_adjacent_pair_multiset_preserved", 0.0),
            ("complete_particle_paths", 0.0),
            ("heldout_path_used_in_prediction", 1.0),
            ("macro_fit_parameter_count", 1.0),
            ("microdynamic_closure_claim_allowed", 1.0),
            ("spatial_facilitation_claim_allowed", 1.0),
            ("thermodynamic_claim_allowed", 1.0),
        )
        for field, bad_value in quality_fields:
            with self.subTest(field=field):
                summary, quality, stationarity = valid_segment_classifier_inputs()
                quality[0][field] = bad_value
                cells = self.classify(summary, quality, stationarity)
                selected = next(
                    row
                    for row in cells
                    if row["model"] == quality[0]["model"]
                    and row["segment_length"] == quality[0]["segment_length"]
                )
                self.assertEqual(selected["cell_pass"], 0.0)

        summary, quality, stationarity = valid_segment_classifier_inputs()
        cross = next(row for row in quality if row["model"].startswith("cross"))
        cross["same_source_assignment_fraction"] = 0.1
        cells = self.classify(summary, quality, stationarity)
        self.assertEqual(
            next(row for row in cells if row["model"].startswith("cross"))["cell_pass"],
            0.0,
        )

    def test_precision_curve_stationarity_and_completeness_fail_independently(self):
        summary_fields = (
            ("ensemble_msd_mc_relative_se", 0.011),
            ("ensemble_ngp_mc_se", 0.031),
            ("ensemble_fs_k2_mc_se", 0.0031),
            ("ensemble_msd_relative_error", 0.101),
            ("ensemble_ngp_absolute_error", 0.301),
            ("ensemble_absolute_error_fs_k2", 0.031),
            ("heldout_path_used_in_prediction", 1.0),
            ("macro_fit_parameter_count", 1.0),
            ("microdynamic_closure_claim_allowed", 1.0),
            ("spatial_facilitation_claim_allowed", 1.0),
            ("thermodynamic_claim_allowed", 1.0),
        )
        for field, bad_value in summary_fields:
            with self.subTest(field=field):
                summary, quality, stationarity = valid_segment_classifier_inputs()
                summary[0][field] = bad_value
                cells = self.classify(summary, quality, stationarity)
                selected = next(
                    row
                    for row in cells
                    if row["model"] == summary[0]["model"]
                    and row["segment_length"] == summary[0]["segment_length"]
                )
                self.assertEqual(selected["cell_pass"], 0.0)

        summary, quality, stationarity = valid_segment_classifier_inputs()
        stationarity[0]["curve_transfer_pass"] = 0.0
        self.assertTrue(
            all(row["cell_pass"] == 0.0 for row in self.classify(summary, quality, stationarity))
        )
        summary, quality, stationarity = valid_segment_classifier_inputs()
        quality.pop()
        cells = self.classify(summary, quality, stationarity)
        self.assertTrue(any(row["realization_completeness_pass"] == 0.0 for row in cells))
        self.assertTrue(any(row["cell_pass"] == 0.0 for row in cells))

    def test_exact_grids_and_replicate_sets_are_mandatory(self):
        summary, quality, stationarity = valid_segment_classifier_inputs()
        with self.assertRaises(ValueError):
            self.analysis.classify_segment_cells(
                summary,
                quality,
                stationarity,
                expected_grids={0.45: (1, 2, 3)},
                expected_replicates={0.45: (1, 2)},
                expected_realizations=2,
            )
        with self.assertRaises(ValueError):
            self.analysis.classify_segment_cells(
                summary,
                quality,
                stationarity,
                expected_grids={0.45: (1, 2)},
                expected_replicates={0.45: (1, 2, 3)},
                expected_realizations=2,
            )


if __name__ == "__main__":
    unittest.main()
