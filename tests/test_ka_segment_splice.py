import sys
import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
            self.assertEqual(audit["global_source_segment_schedule_preserved"], 1.0)
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

        changed_schedule = cross.source_segment.copy()
        changed_schedule[0] = np.roll(changed_schedule[0], 1)
        schedule_audit = audit_segment_surrogate(
            blocks,
            SegmentSurrogate(
                blocks=cross.blocks.copy(),
                source_particle=cross.source_particle.copy(),
                source_segment=changed_schedule,
                target_segment_lengths=cross.target_segment_lengths.copy(),
            ),
            segment_length=3,
            model="cross_particle_segment_splice",
        )
        self.assertEqual(schedule_audit["global_source_segment_schedule_preserved"], 0.0)

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
                            "global_source_segment_schedule_preserved": 1.0,
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
        self.assertEqual(
            {
                (row["segment_length"], row["full_path_control"], row["memory_length_selectable"])
                for row in cells
            },
            {(1.0, 0.0, 1.0), (2.0, 1.0, 0.0)},
        )

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
            ("global_source_segment_schedule_preserved", 0.0),
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


def synthetic_two_temperature_protocol():
    rng = np.random.default_rng(20260718)
    blocks_by_temperature = {
        0.45: {
            replicate: rng.normal(size=(5, 250, 3))
            for replicate in (1, 2, 3)
        },
        0.58: {
            replicate: rng.normal(size=(5, 37, 3))
            for replicate in (1, 2, 3, 4, 5)
        },
    }
    heldout_by_temperature = {}
    waves = np.array([2.0, 4.0, 7.25])
    for temperature, replicates in blocks_by_temperature.items():
        rows = []
        for replicate, blocks in replicates.items():
            observed = cumulative_observables_many_lags(
                blocks,
                block_counts=(1, 2),
                wave_numbers=waves,
            )
            for block_count in (1, 2):
                row = {
                    "replicate": float(replicate),
                    "temperature": temperature,
                    "lag": float(20 * block_count),
                    "observed_msd": observed[block_count]["msd"],
                    "observed_ngp": observed[block_count]["ngp"],
                }
                for wave in waves:
                    characteristic = f"characteristic_k{wave:g}".replace(".", "p")
                    fs_key = f"observed_fs_k{wave:g}".replace(".", "p")
                    row[fs_key] = observed[block_count][characteristic]
                rows.append(row)
        heldout_by_temperature[temperature] = rows
    stationarity = [
        {
            "temperature": temperature,
            "comparison": comparison,
            "curve_transfer_pass": 1.0,
        }
        for temperature in (0.45, 0.58)
        for comparison in ("early_late", "early_heldout", "late_heldout")
    ]
    return blocks_by_temperature, heldout_by_temperature, stationarity


class SegmentRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_script_module(
            "analyze_ka_segment_splice_gate.py",
            "analyze_ka_segment_splice_gate_runner",
        )

    def test_frozen_protocol_and_seed_grid_are_strict_and_deterministic(self):
        low = self.analysis.frozen_protocol(0.45)
        high = self.analysis.frozen_protocol(0.58)
        self.assertEqual(low["calibration_time"], 5000)
        self.assertEqual(low["block_count"], 250)
        self.assertEqual(low["segment_lengths"], (1, 2, 5, 10, 25, 50, 125, 250))
        self.assertEqual(low["replicates"], (1, 2, 3))
        self.assertEqual(high["calibration_time"], 750)
        self.assertEqual(high["block_count"], 37)
        self.assertEqual(high["segment_lengths"], (1, 2, 4, 8, 16, 32, 37))
        self.assertEqual(high["replicates"], (1, 2, 3, 4, 5))
        with self.assertRaises(ValueError):
            self.analysis.frozen_protocol(0.50)

        seeds = {
            self.analysis.segment_realization_seed(
                20260718,
                temperature=temperature,
                replicate=replicate,
                segment_length=length,
                realization=realization,
            )
            for temperature in (0.45, 0.58)
            for replicate in (1, 2)
            for length in (1, 2)
            for realization in range(4)
        }
        self.assertEqual(len(seeds), 2 * 2 * 2 * 4)
        self.assertEqual(
            self.analysis.segment_realization_seed(
                20260718,
                temperature=0.45,
                replicate=1,
                segment_length=2,
                realization=3,
            ),
            self.analysis.segment_realization_seed(
                20260718,
                temperature=0.45,
                replicate=1,
                segment_length=2,
                realization=3,
            ),
        )
        for bad in (
            {"base_seed": -1},
            {"replicate": 0},
            {"segment_length": 0},
            {"realization": -1},
        ):
            arguments = {
                "base_seed": 1,
                "temperature": 0.45,
                "replicate": 1,
                "segment_length": 1,
                "realization": 0,
            }
            arguments.update(bad)
            with self.assertRaises(ValueError):
                self.analysis.segment_realization_seed(**arguments)

    def test_replicate_analysis_is_nested_and_keeps_paired_segment_order(self):
        blocks_by_temperature, heldout, _ = synthetic_two_temperature_protocol()
        blocks = blocks_by_temperature[0.45][1]
        local_heldout = [row for row in heldout[0.45] if row["replicate"] == 1.0]
        first = self.analysis.analyze_segment_replicate(
            blocks,
            local_heldout,
            temperature=0.45,
            replicate=1,
            segment_lengths=(1, 2),
            realization_count=2,
            base_seed=11,
            block_size=20,
        )
        extended = self.analysis.analyze_segment_replicate(
            blocks,
            local_heldout,
            temperature=0.45,
            replicate=1,
            segment_lengths=(1, 2),
            realization_count=3,
            base_seed=11,
            block_size=20,
        )

        first_quality = first["quality_rows"]
        nested_quality = [
            row for row in extended["quality_rows"] if row["realization"] < 2.0
        ]
        self.assertEqual(first_quality, nested_quality)
        self.assertTrue(all(row["paired_segment_order_match"] == 1.0 for row in first_quality))
        self.assertTrue(all(row["heldout_path_used_in_prediction"] == 0.0 for row in first_quality))
        self.assertEqual(len(first["prediction_rows"]), 2 * 2 * 2)
        self.assertEqual(len(first["replicate_scores"]), 2 * 2)

    def test_two_temperature_protocol_extends_every_cell_together(self):
        blocks, heldout, stationarity = synthetic_two_temperature_protocol()
        result = self.analysis.run_two_temperature_from_blocks(
            blocks,
            heldout,
            stationarity,
            initial_realizations=2,
            extended_realizations=4,
            base_seed=20260718,
            block_size=20,
        )
        rerun = self.analysis.run_two_temperature_from_blocks(
            blocks,
            heldout,
            stationarity,
            initial_realizations=2,
            extended_realizations=4,
            base_seed=20260718,
            block_size=20,
        )

        self.assertEqual(result, rerun)
        self.assertEqual(result["realization_count"], 4)
        pairs_by_cell = {}
        for row in result["quality_rows"]:
            key = (row["temperature"], row["model"], row["segment_length"], row["replicate"])
            pairs_by_cell.setdefault(key, set()).add(int(row["realization"]))
        self.assertTrue(pairs_by_cell)
        self.assertTrue(all(realizations == set(range(4)) for realizations in pairs_by_cell.values()))
        self.assertEqual(
            {row["required_realization_count"] for row in result["cell_rows"]},
            {4.0},
        )

    def test_cli_controls_reject_nonfrozen_real_protocol(self):
        self.analysis.validate_real_protocol_controls(
            block_size=20,
            initial_realizations=16,
            extended_realizations=64,
        )
        for controls in (
            (10, 16, 64),
            (20, 8, 64),
            (20, 16, 32),
            (20, 64, 16),
        ):
            with self.assertRaises(ValueError):
                self.analysis.validate_real_protocol_controls(
                    block_size=controls[0],
                    initial_realizations=controls[1],
                    extended_realizations=controls[2],
                )

    def test_manifest_validation_and_type_a_block_extraction_are_strict(self):
        manifest = {
            "temperature": 0.45,
            "replicate_count": 3,
            "replicates": [
                {"replicate": replicate, "directory": f"replicate_{replicate:02d}"}
                for replicate in (1, 2, 3)
            ],
            "thermodynamic_claim_allowed": False,
        }
        validated = self.analysis.validate_ensemble_manifest(manifest, temperature=0.45)
        self.assertEqual(validated, ((1, "replicate_01"), (2, "replicate_02"), (3, "replicate_03")))
        for key, value in (
            ("temperature", 0.58),
            ("replicate_count", 2),
            ("thermodynamic_claim_allowed", True),
        ):
            changed = json.loads(json.dumps(manifest))
            changed[key] = value
            with self.assertRaises(ValueError):
                self.analysis.validate_ensemble_manifest(changed, temperature=0.45)

        positions = np.zeros((41, 4, 3), dtype=float)
        for frame in range(41):
            positions[frame, :, 0] = frame * np.array([1.0, 10.0, 2.0, 20.0])
        trajectory = {
            "unwrapped_positions": positions,
            "particle_types": np.array([0, 1, 0, 1]),
        }
        blocks = self.analysis.calibration_blocks_from_trajectory(
            trajectory,
            calibration_time=40,
            block_size=20,
        )
        self.assertEqual(blocks.shape, (2, 2, 3))
        np.testing.assert_array_equal(blocks[:, :, 0], [[20.0, 20.0], [40.0, 40.0]])
        with self.assertRaises(ValueError):
            self.analysis.calibration_blocks_from_trajectory(
                {**trajectory, "unwrapped_positions": positions[:40]},
                calibration_time=40,
                block_size=20,
            )

    def test_protocol_writer_emits_exact_deterministic_two_temperature_files(self):
        blocks, heldout, stationarity = synthetic_two_temperature_protocol()
        result = self.analysis.run_two_temperature_from_blocks(
            blocks,
            heldout,
            stationarity,
            initial_realizations=2,
            extended_realizations=2,
            base_seed=7,
            block_size=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.analysis.write_protocol_outputs(root, result)
            expected = {
                root / f"renewal_cage_ka_replicates_T{code}_segment_splice_{suffix}.csv"
                for code in ("045", "058")
                for suffix in (
                    "quality",
                    "rows",
                    "summary",
                    "cells",
                    "replicate_scores",
                )
            }
            self.assertEqual(set(paths), expected)
            self.assertTrue(all(path.is_file() for path in expected))
            snapshots = {path.name: path.read_bytes() for path in expected}
            self.analysis.write_protocol_outputs(root, result)
            self.assertEqual(
                snapshots,
                {path.name: path.read_bytes() for path in expected},
            )
            for path in expected:
                with path.open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows)
                self.assertEqual(
                    {float(row["temperature"]) for row in rows},
                    {0.45 if "T045" in path.name else 0.58},
                )

    def test_frozen_block_loader_reads_only_the_calibration_prefix(self):
        manifest = {
            "temperature": 0.58,
            "replicate_count": 5,
            "replicates": [
                {"replicate": replicate, "directory": f"replicate_{replicate:02d}"}
                for replicate in (1, 2, 3, 4, 5)
            ],
            "thermodynamic_claim_allowed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ensemble_manifest.json").write_text(json.dumps(manifest))
            for replicate in (1, 2, 3, 4, 5):
                replicate_root = root / f"replicate_{replicate:02d}"
                replicate_root.mkdir()
                (replicate_root / "trajectory.lammpstrj").write_text("fixture\n")

            positions = np.zeros((751, 3, 3), dtype=float)
            positions[:, :, 0] = np.arange(751)[:, None]
            trajectory = {
                "unwrapped_positions": positions,
                "particle_types": np.array([0, 1, 0]),
            }
            with mock.patch.object(
                self.analysis,
                "load_lammps_custom_trajectory",
                return_value=trajectory,
            ) as loader:
                blocks = self.analysis.load_frozen_blocks(
                    root,
                    temperature=0.58,
                    block_size=20,
                )

            self.assertEqual(set(blocks), {1, 2, 3, 4, 5})
            self.assertTrue(all(value.shape == (2, 37, 3) for value in blocks.values()))
            self.assertEqual(loader.call_count, 5)
            for replicate, call in enumerate(loader.call_args_list, start=1):
                self.assertEqual(
                    call.args[0],
                    root / f"replicate_{replicate:02d}" / "trajectory.lammpstrj",
                )
                self.assertEqual(call.kwargs, {"maximum_frame_count": 751})

    def test_cli_parser_requires_both_temperatures_and_all_controls(self):
        parser = self.analysis.build_parser()
        paths = [Path(f"input_{index}") for index in range(6)]
        args = parser.parse_args(
            [
                "--low-ensemble-directory",
                str(paths[0]),
                "--high-ensemble-directory",
                str(paths[1]),
                "--low-heldout-factorization",
                str(paths[2]),
                "--high-heldout-factorization",
                str(paths[3]),
                "--low-stationarity",
                str(paths[4]),
                "--high-stationarity",
                str(paths[5]),
                "--output-directory",
                "output",
            ]
        )
        self.assertEqual(args.block_size, 20)
        self.assertEqual(args.initial_realizations, 16)
        self.assertEqual(args.extended_realizations, 64)
        self.assertEqual(args.base_seed, 20260718)
        self.assertEqual(args.output_directory, Path("output"))


def segment_cell_fixture(
    temperature,
    *,
    within_start,
    cross_start,
    failed_support=None,
):
    grid = (
        (1, 2, 5, 10, 25, 50, 125, 250)
        if temperature == 0.45
        else (1, 2, 4, 8, 16, 32, 37)
    )
    starts = {
        "within_particle_segment_shuffle": within_start,
        "cross_particle_segment_splice": cross_start,
    }
    rows = []
    for model, start in starts.items():
        for length in grid:
            curve = length == grid[-1] or (
                start is not None and length >= start and length < grid[-1]
            )
            row = {
                "temperature": temperature,
                "model": model,
                "segment_length": float(length),
                "realization_completeness_pass": 1.0,
                "exact_information_pass": 1.0,
                "provenance_claim_boundary_pass": 1.0,
                "stationarity_control_pass": 1.0,
                "precision_pass": 1.0,
                "global_source_segment_schedule_preserved": 1.0,
                "curve_transfer_pass": float(curve),
                "cell_pass": float(curve),
                "full_path_control": float(length == grid[-1]),
                "memory_length_selectable": float(length < grid[-1]),
                "microdynamic_closure_claim_allowed": 0.0,
                "spatial_facilitation_claim_allowed": 0.0,
                "thermodynamic_claim_allowed": 0.0,
            }
            if failed_support is not None and model == failed_support[0] and length == failed_support[1]:
                row[failed_support[2]] = 0.0
                row["cell_pass"] = 0.0
            rows.append(row)
    return rows


def segment_score_fixture(temperature, *, within_score=0.4, cross_score=0.7):
    grid = (
        (1, 2, 5, 10, 25, 50, 125, 250)
        if temperature == 0.45
        else (1, 2, 4, 8, 16, 32, 37)
    )
    replicate_count = 3 if temperature == 0.45 else 5
    return [
        {
            "temperature": temperature,
            "model": model,
            "segment_length": float(length),
            "replicate": float(replicate),
            "higher_order_score": score,
            "microdynamic_closure_claim_allowed": 0.0,
            "spatial_facilitation_claim_allowed": 0.0,
            "thermodynamic_claim_allowed": 0.0,
        }
        for model, score in (
            ("within_particle_segment_shuffle", within_score),
            ("cross_particle_segment_splice", cross_score),
        )
        for length in grid
        for replicate in range(1, replicate_count + 1)
    ]


class SegmentMemorySelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_script_module(
            "summarize_ka_segment_splice_gate.py",
            "summarize_ka_segment_splice_gate_selection",
        )

    def test_selector_requires_a_complete_monotone_nonfull_tail(self):
        grid = (1, 2, 5, 10, 25, 50, 125, 250)
        rows = segment_cell_fixture(0.45, within_start=25, cross_start=None)
        self.assertEqual(
            self.summary.select_monotone_memory_length(
                rows,
                model="within_particle_segment_shuffle",
                temperature=0.45,
                block_count=250,
                required_grid=grid,
            ),
            25,
        )
        isolated = [dict(row) for row in rows]
        for row in isolated:
            if row["model"] == "within_particle_segment_shuffle":
                row["curve_transfer_pass"] = float(row["segment_length"] in {2.0, 250.0})
        self.assertIsNone(
            self.summary.select_monotone_memory_length(
                isolated,
                model="within_particle_segment_shuffle",
                temperature=0.45,
                block_count=250,
                required_grid=grid,
            )
        )
        only_full = [dict(row) for row in rows]
        for row in only_full:
            if row["model"] == "within_particle_segment_shuffle":
                row["curve_transfer_pass"] = float(row["segment_length"] == 250.0)
        self.assertIsNone(
            self.summary.select_monotone_memory_length(
                only_full,
                model="within_particle_segment_shuffle",
                temperature=0.45,
                block_count=250,
                required_grid=grid,
            )
        )

    def test_selector_rejects_incomplete_duplicate_or_unresolved_support(self):
        grid = (1, 2, 5, 10, 25, 50, 125, 250)
        rows = segment_cell_fixture(0.45, within_start=25, cross_start=None)
        local = [row for row in rows if row["model"] == "within_particle_segment_shuffle"]
        variants = (
            local[:-1],
            local + [dict(local[-1])],
            [{**row, "precision_pass": 0.0} if row["segment_length"] == 25.0 else row for row in local],
            [{**row, "memory_length_selectable": 1.0} if row["segment_length"] == 250.0 else row for row in local],
        )
        for variant in variants:
            with self.assertRaises(ValueError):
                self.summary.select_monotone_memory_length(
                    variant,
                    model="within_particle_segment_shuffle",
                    temperature=0.45,
                    block_count=250,
                    required_grid=grid,
                )


class SegmentGateDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_script_module(
            "summarize_ka_segment_splice_gate.py",
            "summarize_ka_segment_splice_gate_decision",
        )

    def classify(self, low_within, low_cross, *, failed_support=None, within_score=0.4, cross_score=0.7):
        return self.summary.classify_segment_splice_gate(
            segment_cell_fixture(
                0.45,
                within_start=low_within,
                cross_start=low_cross,
                failed_support=failed_support,
            ),
            segment_cell_fixture(0.58, within_start=8, cross_start=8),
            segment_score_fixture(
                0.45,
                within_score=within_score,
                cross_score=cross_score,
            ),
            segment_score_fixture(0.58),
        )

    def test_decision_table_covers_all_five_frozen_states(self):
        cases = (
            (
                25,
                25,
                "finite_single_particle_path_memory_sufficient_conditional_on_global_schedule",
            ),
            (25, 50, "persistent_environment_identity_required_beyond_local_path"),
            (None, None, "longer_or_richer_path_state_required"),
            (50, 25, "null_family_pathology_unresolved"),
        )
        for within, cross, expected in cases:
            with self.subTest(expected=expected):
                verdict = self.classify(within, cross)
                self.assertEqual(verdict["mechanism_state"], expected)
                self.assertEqual(verdict["global_source_segment_schedule_preserved"], 1.0)
                self.assertEqual(
                    verdict["substantive_interpretation_condition"],
                    "conditional_on_preserved_global_source_segment_schedule",
                )
                self.assertEqual(verdict["within_cooling_memory_growth"], 1.0 if within else 0.0)
                self.assertEqual(verdict["microdynamic_closure_claim_allowed"], 0.0)
                self.assertEqual(verdict["spatial_facilitation_claim_allowed"], 0.0)
                self.assertEqual(verdict["thermodynamic_claim_allowed"], 0.0)

        unresolved = self.classify(
            25,
            50,
            failed_support=("within_particle_segment_shuffle", 25, "precision_pass"),
        )
        self.assertEqual(unresolved["mechanism_state"], "mechanism_unresolved")

    def test_substantive_states_require_all_replicate_scores_and_strict_ordering(self):
        score_failure = self.classify(25, 25, within_score=1.01)
        self.assertEqual(score_failure["mechanism_state"], "mechanism_unresolved")
        tied_persistent = self.classify(25, 50, within_score=0.7, cross_score=0.7)
        self.assertEqual(tied_persistent["mechanism_state"], "mechanism_unresolved")

    def test_high_support_failure_preserves_the_low_temperature_specific_verdict(self):
        low = segment_cell_fixture(0.45, within_start=25, cross_start=50)
        high = segment_cell_fixture(
            0.58,
            within_start=8,
            cross_start=8,
            failed_support=(
                "within_particle_segment_shuffle",
                8,
                "stationarity_control_pass",
            ),
        )
        verdict = self.summary.classify_segment_splice_gate(
            low,
            high,
            segment_score_fixture(0.45),
            segment_score_fixture(0.58),
        )
        self.assertEqual(verdict["mechanism_state"], "mechanism_unresolved")
        self.assertEqual(
            verdict["low_temperature_mechanism_state"],
            "persistent_environment_identity_required_beyond_local_path",
        )
        self.assertEqual(verdict["low_temperature_gate_resolved"], 1.0)
        self.assertEqual(verdict["high_temperature_control_resolved"], 0.0)
        self.assertEqual(verdict["within_cooling_memory_growth"], 0.0)
        self.assertEqual(verdict["cross_cooling_memory_growth"], 0.0)
        self.assertEqual(verdict["cell_grid_and_row_completeness_pass"], 1.0)
        self.assertEqual(verdict["global_source_segment_schedule_preserved"], 1.0)

    def test_full_path_ensemble_cancellation_blocks_mechanism_identifiability(self):
        low = segment_cell_fixture(0.45, within_start=None, cross_start=None)
        for row in low:
            if row["segment_length"] == 250.0:
                row["curve_transfer_pass"] = 1.0
                row["cell_pass"] = 1.0
        low_scores = segment_score_fixture(0.45)
        for row in low_scores:
            if row["segment_length"] == 250.0:
                row["higher_order_score"] = 0.8 if row["replicate"] == 1.0 else 2.0
        verdict = self.summary.classify_segment_splice_gate(
            low,
            segment_cell_fixture(0.58, within_start=8, cross_start=8),
            low_scores,
            segment_score_fixture(0.58),
        )
        self.assertEqual(verdict["low_full_path_control_ensemble_pass"], 1.0)
        self.assertEqual(verdict["low_full_path_control_all_replicates_pass"], 0.0)
        self.assertEqual(verdict["low_full_path_control_failed_replicate_count"], 2.0)
        self.assertEqual(verdict["low_mechanism_identifiable_against_full_path_control"], 0.0)
        self.assertEqual(verdict["ensemble_cancellation_detected"], 1.0)
        self.assertEqual(verdict["independent_replicate_memory_lower_bound_claim_allowed"], 0.0)
        self.assertEqual(verdict["low_owner_identity_paired_ordering_count"], 21.0)
        self.assertEqual(verdict["low_owner_identity_paired_ordering_total"], 21.0)
        self.assertAlmostEqual(
            verdict["low_owner_identity_replicate_first_mean_score_difference"],
            0.3,
        )
        self.assertEqual(verdict["owner_identity_information_supported_exploratory"], 1.0)
        self.assertEqual(verdict["owner_identity_sufficiency_claim_allowed"], 0.0)
        self.assertEqual(verdict["static_vs_finite_exchange_resolved"], 0.0)

    def test_svg_is_deterministic_finite_and_marks_the_claim_boundary(self):
        low = segment_cell_fixture(0.45, within_start=25, cross_start=50)
        high = segment_cell_fixture(0.58, within_start=8, cross_start=8)
        for index, row in enumerate(low + high):
            row["maximum_ensemble_msd_relative_error"] = 0.01 + 0.001 * index
            row["maximum_ensemble_ngp_absolute_error"] = 0.02 + 0.002 * index
            row["maximum_ensemble_fs_absolute_error"] = 0.003 + 0.0001 * index
            row["tau_L"] = 20.0 * row["segment_length"]
        low[0]["maximum_ensemble_msd_relative_error"] = 1.0
        verdict = self.classify(25, 50)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gate.svg"
            self.summary.write_gate_svg(output, low + high, verdict)
            first = output.read_bytes()
            self.summary.write_gate_svg(output, low + high, verdict)
            self.assertEqual(first, output.read_bytes())
            text = first.decode()
        self.assertIn("tolerance = 1", text)
        self.assertIn("full-path control", text)
        self.assertIn("normalized maximum error (clipped at 2.5)", text)
        self.assertIn("&gt;=2.5", text)
        self.assertIn('data-clipped="true"', text)
        self.assertIn(verdict["mechanism_state"], text)
        self.assertIn("no microscopic, spatial-facilitation, or thermodynamic claim", text)


def paired_excess_score_fixture(*, bad_full_path_model_agreement=False):
    grid = (1, 2, 5, 10, 25, 50, 125, 250)
    full_scores = {1: 0.9, 2: 2.1, 3: 3.3}
    within_excess = {
        1: 18.0,
        2: 15.0,
        5: 11.0,
        10: 8.0,
        25: 4.0,
        50: 2.0,
        125: 0.2,
    }
    cross_excess = {
        1: 21.0,
        2: 18.0,
        5: 14.0,
        10: 11.0,
        25: 7.0,
        50: 5.0,
        125: 2.0,
    }
    rows = []
    for model in (
        "within_particle_segment_shuffle",
        "cross_particle_segment_splice",
    ):
        excess_by_length = (
            within_excess
            if model == "within_particle_segment_shuffle"
            else cross_excess
        )
        for length in grid:
            for replicate, baseline in full_scores.items():
                score = baseline if length == 250 else baseline + excess_by_length[length]
                if (
                    bad_full_path_model_agreement
                    and model == "cross_particle_segment_splice"
                    and length == 250
                    and replicate == 2
                ):
                    score += 0.01
                rows.append(
                    {
                        "temperature": 0.45,
                        "model": model,
                        "segment_length": float(length),
                        "replicate": float(replicate),
                        "higher_order_score": score,
                        "replicate_curve_pass": float(score <= 1.0),
                        "microdynamic_closure_claim_allowed": 0.0,
                        "spatial_facilitation_claim_allowed": 0.0,
                        "thermodynamic_claim_allowed": 0.0,
                    }
                )
    return rows


def paired_excess_source_verdict_fixture():
    return {
        "mechanism_state": "mechanism_unresolved",
        "low_mechanism_identifiable_against_full_path_control": 0.0,
        "low_full_path_control_all_replicates_pass": 0.0,
        "low_full_path_control_failed_replicate_count": 2.0,
        "global_source_segment_schedule_preserved": 1.0,
        "owner_identity_sufficiency_claim_allowed": 0.0,
        "independent_replicate_memory_lower_bound_claim_allowed": 0.0,
        "static_vs_finite_exchange_resolved": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }


def paired_excess_provenance_fixture():
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


class PairedExcessBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_script_module(
            "summarize_ka_segment_splice_paired_excess.py",
            "summarize_ka_segment_splice_paired_excess",
        )

    def classify(self, *, bad_full_path_model_agreement=False, source_verdict=None):
        return self.summary.classify_paired_excess_gate(
            paired_excess_score_fixture(
                bad_full_path_model_agreement=bad_full_path_model_agreement,
            ),
            segment_cell_fixture(0.45, within_start=None, cross_start=None),
            [source_verdict or paired_excess_source_verdict_fixture()],
            paired_excess_provenance_fixture(),
        )

    def test_excess_rows_are_replicate_paired_to_the_full_path_baseline(self):
        rows, gate = self.classify()

        self.assertEqual(len(rows), 14)
        selected = next(
            row
            for row in rows
            if row["model"] == "within_particle_segment_shuffle"
            and row["segment_length"] == 50.0
        )
        self.assertEqual(selected["independent_replicate_count"], 0.0)
        self.assertEqual(selected["replicate_count"], 3.0)
        self.assertEqual(selected["parent_sample_count"], 1.0)
        self.assertEqual(
            selected["ci95_method"],
            "student_t_replicate_first_correlated_parent_exploratory_df2",
        )
        self.assertAlmostEqual(selected["replicate_1_paired_excess"], 2.0)
        self.assertAlmostEqual(selected["replicate_2_paired_excess"], 2.0)
        self.assertAlmostEqual(selected["replicate_3_paired_excess"], 2.0)
        self.assertAlmostEqual(selected["replicate_first_mean_paired_excess"], 2.0)
        self.assertAlmostEqual(selected["replicate_first_t95_ci_low"], 2.0)
        self.assertAlmostEqual(selected["replicate_first_t95_ci_high"], 2.0)
        self.assertEqual(selected["paired_degradation_identified"], 1.0)
        self.assertEqual(selected["global_source_segment_schedule_preserved"], 1.0)
        self.assertEqual(selected["microdynamic_closure_claim_allowed"], 0.0)

        self.assertEqual(gate["mechanism_state"], "mechanism_unresolved")
        self.assertEqual(gate["input_completeness_pass"], 1.0)
        self.assertEqual(gate["full_path_model_agreement_pass"], 1.0)
        self.assertEqual(gate["low_full_path_control_all_replicates_pass"], 0.0)
        self.assertEqual(gate["replicate_provenance_validation_pass"], 1.0)
        self.assertEqual(gate["independent_replicate_count"], 0.0)
        self.assertEqual(gate["short_horizon_information_loss_supported_exploratory"], 1.0)
        self.assertEqual(gate["owner_identity_information_supported_exploratory"], 1.0)
        self.assertEqual(gate["finite_memory_state_addition_allowed"], 0.0)
        self.assertEqual(gate["next_required_action"], "replicate_resolved_full_path_baseline_or_new_trajectory_validation")

    def test_bad_source_verdict_or_full_path_disagreement_fails_closed(self):
        _, disagreement = self.classify(bad_full_path_model_agreement=True)
        self.assertEqual(disagreement["input_completeness_pass"], 1.0)
        self.assertEqual(disagreement["full_path_model_agreement_pass"], 0.0)
        self.assertEqual(disagreement["short_horizon_information_loss_supported_exploratory"], 0.0)
        self.assertEqual(disagreement["mechanism_state"], "mechanism_unresolved")

        bad_verdict = paired_excess_source_verdict_fixture()
        bad_verdict["mechanism_state"] = "finite_single_particle_path_memory_sufficient"
        _, rejected = self.classify(source_verdict=bad_verdict)
        self.assertEqual(rejected["source_verdict_fail_closed"], 1.0)
        self.assertEqual(rejected["input_completeness_pass"], 0.0)
        self.assertEqual(rejected["finite_memory_state_addition_allowed"], 0.0)

    def test_svg_is_deterministic_and_marks_exploratory_claim_boundary(self):
        rows, gate = self.classify()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "paired-excess.svg"
            self.summary.write_paired_excess_svg(output, rows, gate)
            first = output.read_bytes()
            self.summary.write_paired_excess_svg(output, rows, gate)
            self.assertEqual(first, output.read_bytes())
            text = first.decode()

        self.assertIn("Paired excess over replicate full-path baseline", text)
        self.assertIn("post-run exploratory", text)
        self.assertIn("mechanism unresolved", text)
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("inf", text.lower())
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("inf", text.lower())


if __name__ == "__main__":
    unittest.main()
