import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ka_segment_splice import (  # noqa: E402
    SegmentSurrogate,
    audit_segment_surrogate,
    cross_particle_segment_splice,
    segment_slices,
    within_particle_segment_shuffle,
)


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


if __name__ == "__main__":
    unittest.main()
