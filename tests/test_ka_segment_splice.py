import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ka_segment_splice import (  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
