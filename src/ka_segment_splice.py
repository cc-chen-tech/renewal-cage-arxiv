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


def _particle_derangement(
    particle_count: int,
    rng: np.random.Generator,
    *,
    previous_source: np.ndarray | None,
    maximum_restarts: int,
) -> np.ndarray:
    targets = np.arange(particle_count)
    for _ in range(maximum_restarts):
        source = rng.permutation(particle_count)
        if np.any(source == targets):
            continue
        if previous_source is not None and np.any(source == previous_source):
            continue
        return source
    raise ValueError("failed to draw a constrained cross-particle assignment")


def cross_particle_segment_splice(
    blocks: np.ndarray,
    *,
    segment_length: int,
    rng: np.random.Generator,
    maximum_restarts: int = 100,
) -> SegmentSurrogate:
    """Splice intact segments across owners without adjacent owner continuity."""

    values = _validated_blocks(blocks)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    if (
        isinstance(maximum_restarts, bool)
        or not isinstance(maximum_restarts, int)
        or maximum_restarts < 1
    ):
        raise ValueError("maximum_restarts must be a positive integer")
    slices = segment_slices(values.shape[1], segment_length)
    order = _segment_order(len(slices), rng, maximum_restarts=maximum_restarts)
    target_lengths = np.asarray(
        [slices[index][1] - slices[index][0] for index in order],
        dtype=int,
    )
    particle_count = values.shape[0]
    source_particle_columns: list[np.ndarray] = []
    reconstructed_segments: list[np.ndarray] = []
    previous_source: np.ndarray | None = None
    for source_segment in order:
        source_particle = _particle_derangement(
            particle_count,
            rng,
            previous_source=previous_source,
            maximum_restarts=maximum_restarts,
        )
        start, stop = slices[int(source_segment)]
        reconstructed_segments.append(values[source_particle, start:stop])
        source_particle_columns.append(source_particle)
        previous_source = source_particle
    source_particle_array = np.column_stack(source_particle_columns)
    source_segment_array = np.broadcast_to(
        order[None, :],
        source_particle_array.shape,
    ).copy()
    return SegmentSurrogate(
        blocks=np.concatenate(reconstructed_segments, axis=1),
        source_particle=source_particle_array,
        source_segment=source_segment_array,
        target_segment_lengths=target_lengths,
    )


def audit_segment_surrogate(
    source_blocks: np.ndarray,
    surrogate: SegmentSurrogate,
    *,
    segment_length: int,
    model: str,
) -> dict[str, float]:
    """Audit exact segment and block provenance without relying on value uniqueness."""

    values = _validated_blocks(source_blocks)
    if model not in {
        "within_particle_segment_shuffle",
        "cross_particle_segment_splice",
    }:
        raise ValueError("model must name one frozen segment surrogate")
    if not isinstance(surrogate, SegmentSurrogate):
        raise ValueError("surrogate must be a SegmentSurrogate")
    slices = segment_slices(values.shape[1], segment_length)
    particle_count, block_count, _ = values.shape
    segment_count = len(slices)
    reconstructed = np.asarray(surrogate.blocks, dtype=float)
    source_particle = np.asarray(surrogate.source_particle)
    source_segment = np.asarray(surrogate.source_segment)
    target_lengths = np.asarray(surrogate.target_segment_lengths)
    provenance_shape_pass = (
        reconstructed.shape == values.shape
        and source_particle.shape == (particle_count, segment_count)
        and source_segment.shape == source_particle.shape
        and target_lengths.shape == (segment_count,)
        and np.issubdtype(source_particle.dtype, np.integer)
        and np.issubdtype(source_segment.dtype, np.integer)
        and np.issubdtype(target_lengths.dtype, np.integer)
        and np.all(np.isfinite(reconstructed))
        and np.all((0 <= source_particle) & (source_particle < particle_count))
        and np.all((0 <= source_segment) & (source_segment < segment_count))
        and np.all(target_lengths > 0)
        and int(np.sum(target_lengths)) == block_count
    )
    expected = np.empty_like(values)
    block_provenance: list[int] = []
    token_provenance: list[int] = []
    length_match = provenance_shape_pass
    cursor = 0
    if provenance_shape_pass:
        for target in range(particle_count):
            cursor = 0
            for slot in range(segment_count):
                local_source_particle = int(source_particle[target, slot])
                local_source_segment = int(source_segment[target, slot])
                start, stop = slices[local_source_segment]
                length = stop - start
                if length != int(target_lengths[slot]):
                    length_match = False
                    break
                expected[target, cursor : cursor + length] = values[
                    local_source_particle,
                    start:stop,
                ]
                block_provenance.extend(
                    local_source_particle * block_count + block
                    for block in range(start, stop)
                )
                token_provenance.append(
                    local_source_particle * segment_count + local_source_segment
                )
                cursor += length
            if cursor != block_count:
                length_match = False
                break
    token_counts = np.bincount(
        np.asarray(token_provenance, dtype=int),
        minlength=particle_count * segment_count,
    )
    token_minimum = float(np.min(token_counts)) if len(token_counts) else 0.0
    token_maximum = float(np.max(token_counts)) if len(token_counts) else 0.0
    token_multiset = bool(
        length_match
        and len(token_provenance) == particle_count * segment_count
        and np.all(token_counts == 1)
    )
    block_counts = np.bincount(
        np.asarray(block_provenance, dtype=int),
        minlength=particle_count * block_count,
    )
    block_multiset = bool(
        length_match
        and len(block_provenance) == particle_count * block_count
        and np.all(block_counts == 1)
    )
    value_match = bool(
        length_match
        and reconstructed.shape == expected.shape
        and np.array_equal(reconstructed, expected)
    )
    source_lengths = np.asarray([stop - start for start, stop in slices])
    length_histogram = bool(
        length_match
        and np.array_equal(
            np.sort(np.tile(target_lengths, particle_count)),
            np.sort(np.tile(source_lengths, particle_count)),
        )
    )
    targets = np.arange(particle_count)[:, None]
    same_source_fraction = (
        float(np.mean(source_particle == targets)) if provenance_shape_pass else 1.0
    )
    adjacent_same_source = (
        float(np.mean(source_particle[:, 1:] == source_particle[:, :-1]))
        if provenance_shape_pass and segment_count > 1
        else 0.0
    )
    internal_pairs = int(np.sum(np.maximum(target_lengths - 1, 0))) * particle_count
    total_pairs = particle_count * max(block_count - 1, 1)
    guaranteed_internal_fraction = float(internal_pairs / total_pairs)
    accidental_count = 0
    seam_count = particle_count * max(segment_count - 1, 0)
    if provenance_shape_pass:
        for target in range(particle_count):
            for slot in range(segment_count - 1):
                left_particle = int(source_particle[target, slot])
                right_particle = int(source_particle[target, slot + 1])
                left_segment = int(source_segment[target, slot])
                right_segment = int(source_segment[target, slot + 1])
                if (
                    left_particle == right_particle
                    and slices[left_segment][1] == slices[right_segment][0]
                ):
                    accidental_count += 1
    accidental_fraction = float(accidental_count / seam_count) if seam_count else 0.0
    within_particle_multiset = bool(
        token_multiset
        and np.all(source_particle == targets)
        and all(
            np.array_equal(
                np.sort(source_segment[target]),
                np.arange(segment_count),
            )
            for target in range(particle_count)
        )
    )
    full_path_control = segment_length == block_count
    complete_path_ensemble_equal = bool(
        full_path_control and token_multiset and block_multiset and value_match
    )
    return {
        "source_token_count": float(particle_count * segment_count),
        "target_token_count": float(len(token_provenance)),
        "source_token_reuse_minimum": token_minimum,
        "source_token_reuse_maximum": token_maximum,
        "ordered_token_multiset_preserved": float(token_multiset and value_match),
        "global_block_provenance_multiset_preserved": float(block_multiset),
        "global_block_vector_multiset_preserved": float(block_multiset and value_match),
        "segment_length_histogram_preserved": float(length_histogram),
        "internal_adjacent_pair_multiset_preserved": float(
            token_multiset and value_match
        ),
        "complete_particle_paths": float(provenance_shape_pass and length_match),
        "within_particle_vector_multiset_preserved": float(within_particle_multiset),
        "same_source_assignment_fraction": same_source_fraction,
        "adjacent_same_source_segment_fraction": adjacent_same_source,
        "guaranteed_internal_adjacency_fraction": guaranteed_internal_fraction,
        "accidental_seam_adjacency_fraction": accidental_fraction,
        "full_path_control": float(full_path_control),
        "complete_path_ensemble_equal": float(complete_path_ensemble_equal),
        "heldout_path_used_in_prediction": 0.0,
        "macro_fit_parameter_count": 0.0,
        "microdynamic_closure_claim_allowed": 0.0,
        "spatial_facilitation_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
