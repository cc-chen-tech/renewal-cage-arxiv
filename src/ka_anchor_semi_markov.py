"""Directly observed reversible-cage states and semi-Markov path surrogates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


RETURN_STATE = 1
ESCAPE_STATE = 0


def _validated_anchor_events(
    events: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("particle", "time", "jump_vector", "pre_center", "post_center")
    try:
        particle = np.asarray(events["particle"])
        time = np.asarray(events["time"], dtype=float)
        jump = np.asarray(events["jump_vector"], dtype=float)
        pre = np.asarray(events["pre_center"], dtype=float)
        post = np.asarray(events["post_center"], dtype=float)
    except KeyError as error:
        raise ValueError(f"events must contain {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise ValueError("event arrays must be numeric") from error
    if any(name not in events for name in required):
        raise ValueError("events are missing required cage-center fields")
    if (
        particle.ndim != 1
        or time.shape != particle.shape
        or jump.shape != (len(particle), 3)
        or pre.shape != jump.shape
        or post.shape != jump.shape
        or len(particle) < 1
    ):
        raise ValueError("events must contain aligned one-dimensional ids and 3D vectors")
    if not np.issubdtype(particle.dtype, np.integer) or np.any(particle < 0):
        raise ValueError("particle ids must be nonnegative integers")
    if np.any(~np.isfinite(time)) or any(
        np.any(~np.isfinite(values)) for values in (jump, pre, post)
    ):
        raise ValueError("event times and cage vectors must be finite")
    if np.any(np.diff(particle) < 0):
        raise ValueError("events must be grouped by particle")
    adjacent = particle[:-1] == particle[1:]
    if np.any(np.diff(time)[adjacent] <= 0.0):
        raise ValueError("event times must increase within each particle")
    if np.any(np.linalg.norm(jump, axis=1) <= 0.0):
        raise ValueError("jump vectors must be nonzero")
    if not np.allclose(jump, post - pre, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("jump vectors must equal post_center minus pre_center")
    return particle.astype(int, copy=False), time, jump, pre, post


def _rank_bin(sorted_values: np.ndarray, value: float, bin_count: int) -> int:
    rank = int(np.searchsorted(sorted_values, value, side="left"))
    return min(bin_count - 1, int(rank * bin_count // len(sorted_values)))


def _rank_quantile(sorted_values: np.ndarray, value: float) -> float:
    left = int(np.searchsorted(sorted_values, value, side="left"))
    right = int(np.searchsorted(sorted_values, value, side="right"))
    average_rank = 0.5 * (left + right - 1)
    return float((average_rank + 0.5) / len(sorted_values))


def extract_anchor_transition_kernel(
    events: dict[str, np.ndarray],
    *,
    debye_waller_factor: float,
    radial_bin_count: int,
) -> dict[str, Any]:
    """Extract observed return/escape Markov-renewal records without particle joins."""

    if not math.isfinite(debye_waller_factor) or debye_waller_factor <= 0.0:
        raise ValueError("debye_waller_factor must be positive and finite")
    if (
        isinstance(radial_bin_count, bool)
        or not isinstance(radial_bin_count, int)
        or radial_bin_count < 1
    ):
        raise ValueError("radial_bin_count must be a positive integer")
    particle, time, jump, pre, post = _validated_anchor_events(events)
    threshold = math.sqrt(debye_waller_factor)

    rows: dict[str, list[float | int]] = {
        "particle": [],
        "source_event_index": [],
        "target_event_index": [],
        "current_state": [],
        "next_state": [],
        "holding_time": [],
        "normalized_holding_time": [],
        "source_radius": [],
        "target_radius": [],
        "source_radius_bin": [],
        "target_radius_quantile": [],
        "relative_cosine": [],
        "closure_distance": [],
    }
    profiles: dict[int, dict[str, Any]] = {}
    active: list[int] = []
    non_propagating: list[int] = []

    starts = np.flatnonzero(np.r_[True, particle[1:] != particle[:-1]])
    ends = np.r_[starts[1:], len(particle)]
    for start, end in zip(starts, ends):
        indices = np.arange(start, end)
        particle_id = int(particle[start])
        local_times = time[indices]
        local_jumps = jump[indices]
        radii = np.linalg.norm(local_jumps, axis=1)
        sorted_radii = np.sort(radii)
        waits = np.diff(local_times)
        mean_wait = float(np.mean(waits)) if len(waits) else math.nan
        profile: dict[str, Any] = {
            "event_count": int(len(indices)),
            "mean_wait": mean_wait,
            "jump_radii": sorted_radii.copy(),
        }
        if len(indices) < 3:
            profiles[particle_id] = profile
            non_propagating.append(particle_id)
            continue

        states = np.full(len(indices), -1, dtype=int)
        closure = np.full(len(indices), math.nan, dtype=float)
        closure[1:] = np.linalg.norm(post[indices[1:]] - pre[indices[:-1]], axis=1)
        states[1:] = (closure[1:] <= threshold).astype(int)
        profile.update(
            {
                "initial_state": int(states[1]),
                "initial_vector": local_jumps[1].copy(),
            }
        )
        profiles[particle_id] = profile
        active.append(particle_id)

        for local_source in range(1, len(indices) - 1):
            source_index = int(indices[local_source])
            target_index = int(indices[local_source + 1])
            source_vector = jump[source_index]
            target_vector = jump[target_index]
            source_radius = float(np.linalg.norm(source_vector))
            target_radius = float(np.linalg.norm(target_vector))
            relative_cosine = float(
                np.dot(source_vector, target_vector) / (source_radius * target_radius)
            )
            holding_time = float(time[target_index] - time[source_index])
            values: dict[str, float | int] = {
                "particle": particle_id,
                "source_event_index": source_index,
                "target_event_index": target_index,
                "current_state": int(states[local_source]),
                "next_state": int(states[local_source + 1]),
                "holding_time": holding_time,
                "normalized_holding_time": holding_time / mean_wait,
                "source_radius": source_radius,
                "target_radius": target_radius,
                "source_radius_bin": _rank_bin(
                    sorted_radii,
                    source_radius,
                    radial_bin_count,
                ),
                "target_radius_quantile": _rank_quantile(
                    sorted_radii,
                    target_radius,
                ),
                "relative_cosine": relative_cosine,
                "closure_distance": float(closure[local_source + 1]),
            }
            for key, value in values.items():
                rows[key].append(value)

    if not rows["particle"]:
        raise ValueError("at least one within-particle event triplet is required")
    integer_fields = {
        "particle",
        "source_event_index",
        "target_event_index",
        "current_state",
        "next_state",
        "source_radius_bin",
    }
    result: dict[str, Any] = {
        key: np.asarray(values, dtype=int if key in integer_fields else float)
        for key, values in rows.items()
    }
    result.update(
        {
            "debye_waller_factor": float(debye_waller_factor),
            "radius_threshold": threshold,
            "radial_bin_count": int(radial_bin_count),
            "transition_count": len(rows["particle"]),
            "particle_profiles": profiles,
            "active_particle_ids": tuple(active),
            "non_propagating_particle_ids": tuple(non_propagating),
        }
    )
    return result
