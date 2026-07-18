"""Information-preserving segment surrogates for calibration cage paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentSurrogate:
    """A reconstructed block path with segment-level source provenance."""

    blocks: np.ndarray
    source_particle: np.ndarray
    source_segment: np.ndarray
    target_segment_lengths: np.ndarray


def segment_slices(
    block_count: int,
    segment_length: int,
) -> tuple[tuple[int, int], ...]:
    """Partition a complete block path without dropping a terminal remainder."""

    if (
        isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or block_count < 1
        or isinstance(segment_length, bool)
        or not isinstance(segment_length, int)
        or not 1 <= segment_length <= block_count
    ):
        raise ValueError("block_count and segment_length must be compatible positive integers")
    return tuple(
        (start, min(start + segment_length, block_count))
        for start in range(0, block_count, segment_length)
    )


def _validated_blocks(blocks: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(blocks, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("blocks must be a finite particle-block-vector array") from error
    if (
        values.ndim != 3
        or values.shape[0] < 1
        or values.shape[1] < 1
        or values.shape[2] != 3
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("blocks must be a finite particle-block-3 array")
    return values


def _segment_order(
    segment_count: int,
    rng: np.random.Generator,
    *,
    maximum_restarts: int = 100,
) -> np.ndarray:
    if segment_count == 1:
        return np.zeros(1, dtype=int)
    identity = np.arange(segment_count)
    for _ in range(maximum_restarts):
        order = rng.permutation(segment_count)
        if not np.array_equal(order, identity):
            return order
    raise ValueError("failed to draw a nonidentity segment permutation")


def within_particle_segment_shuffle(
    blocks: np.ndarray,
    *,
    segment_length: int,
    rng: np.random.Generator,
) -> SegmentSurrogate:
    """Reorder intact path segments while retaining their particle owner."""

    values = _validated_blocks(blocks)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    slices = segment_slices(values.shape[1], segment_length)
    order = _segment_order(len(slices), rng)
    target_lengths = np.asarray(
        [slices[index][1] - slices[index][0] for index in order],
        dtype=int,
    )
    reconstructed = np.concatenate(
        [values[:, slices[index][0] : slices[index][1]] for index in order],
        axis=1,
    )
    particle_count = values.shape[0]
    source_particle = np.broadcast_to(
        np.arange(particle_count, dtype=int)[:, None],
        (particle_count, len(slices)),
    ).copy()
    source_segment = np.broadcast_to(
        order[None, :],
        (particle_count, len(slices)),
    ).copy()
    return SegmentSurrogate(
        blocks=reconstructed,
        source_particle=source_particle,
        source_segment=source_segment,
        target_segment_lengths=target_lengths,
    )
