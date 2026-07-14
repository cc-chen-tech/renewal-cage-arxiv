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


def _validate_simulation_kernel(kernel: dict[str, Any]) -> None:
    required_arrays = (
        "current_state",
        "next_state",
        "normalized_holding_time",
        "source_radius_bin",
        "target_radius_quantile",
        "relative_cosine",
        "closure_distance",
    )
    try:
        length = len(np.asarray(kernel["current_state"]))
        profiles = kernel["particle_profiles"]
        radial_bin_count = int(kernel["radial_bin_count"])
        threshold = float(kernel["radius_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("kernel is missing required simulation fields") from error
    if length < 1 or not isinstance(profiles, dict) or not profiles:
        raise ValueError("kernel must contain transitions and particle profiles")
    if radial_bin_count < 1 or not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("kernel radial controls must be positive")
    for name in required_arrays:
        values = np.asarray(kernel.get(name))
        if values.ndim != 1 or len(values) != length:
            raise ValueError(f"kernel field {name!r} must align with transitions")
        if np.any(~np.isfinite(values.astype(float))):
            raise ValueError(f"kernel field {name!r} must be finite")
    current = np.asarray(kernel["current_state"], dtype=int)
    following = np.asarray(kernel["next_state"], dtype=int)
    bins = np.asarray(kernel["source_radius_bin"], dtype=int)
    quantiles = np.asarray(kernel["target_radius_quantile"], dtype=float)
    if np.any(~np.isin(current, (ESCAPE_STATE, RETURN_STATE))) or np.any(
        ~np.isin(following, (ESCAPE_STATE, RETURN_STATE))
    ):
        raise ValueError("kernel states must be binary return/escape labels")
    if np.any(bins < 0) or np.any(bins >= radial_bin_count):
        raise ValueError("kernel source-radius bins are out of range")
    if np.any(quantiles < 0.0) or np.any(quantiles > 1.0):
        raise ValueError("kernel target-radius quantiles must lie in [0, 1]")


def _profile_radius(profile: dict[str, Any], quantile: float) -> float:
    radii = np.asarray(profile.get("jump_radii"), dtype=float)
    if radii.ndim != 1 or len(radii) < 1 or np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("particle profiles require positive finite jump radii")
    return float(np.quantile(radii, quantile, method="nearest"))


def _profile_radius_bin(profile: dict[str, Any], radius: float, bin_count: int) -> int:
    radii = np.asarray(profile["jump_radii"], dtype=float)
    return _rank_bin(np.sort(radii), radius, bin_count)


def _fixed_polar_direction(
    source_vector: np.ndarray,
    relative_cosine: float,
    rng: np.random.Generator,
) -> np.ndarray:
    radius = float(np.linalg.norm(source_vector))
    if radius <= 0.0 or not math.isfinite(relative_cosine) or abs(relative_cosine) > 1.0:
        raise ValueError("polar geometry requires a nonzero source and cosine in [-1, 1]")
    source_direction = source_vector / radius
    basis = np.zeros(3, dtype=float)
    basis[int(np.argmin(np.abs(source_direction)))] = 1.0
    first = np.cross(source_direction, basis)
    first /= np.linalg.norm(first)
    second = np.cross(source_direction, first)
    azimuth = float(rng.uniform(0.0, 2.0 * math.pi))
    transverse = math.sqrt(max(0.0, 1.0 - relative_cosine**2))
    return (
        relative_cosine * source_direction
        + transverse * (math.cos(azimuth) * first + math.sin(azimuth) * second)
    )


def simulate_anchor_semi_markov(
    kernel: dict[str, Any],
    rng: np.random.Generator,
    *,
    duration: int,
    maximum_lag: int,
    model: str,
) -> dict[str, np.ndarray]:
    """Generate paired anchor-aware or state-schedule event paths."""

    _validate_simulation_kernel(kernel)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator")
    if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
        raise ValueError("duration must be a positive integer")
    if (
        isinstance(maximum_lag, bool)
        or not isinstance(maximum_lag, int)
        or maximum_lag < 0
    ):
        raise ValueError("maximum_lag must be a nonnegative integer")
    allowed_models = {
        "anchor_aware_semi_markov",
        "state_schedule_without_anchor_geometry",
    }
    if model not in allowed_models:
        raise ValueError(f"model must be one of {sorted(allowed_models)}")

    state_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    geometry_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    current_states = np.asarray(kernel["current_state"], dtype=int)
    next_states = np.asarray(kernel["next_state"], dtype=int)
    normalized_waits = np.asarray(kernel["normalized_holding_time"], dtype=float)
    source_bins = np.asarray(kernel["source_radius_bin"], dtype=int)
    target_quantiles = np.asarray(kernel["target_radius_quantile"], dtype=float)
    relative_cosines = np.asarray(kernel["relative_cosine"], dtype=float)
    closure_distances = np.asarray(kernel["closure_distance"], dtype=float)
    radial_bin_count = int(kernel["radial_bin_count"])
    threshold = float(kernel["radius_threshold"])
    horizon = duration + maximum_lag

    output: dict[str, list[Any]] = {
        "particle": [],
        "time": [],
        "jump_vector": [],
        "source_vector": [],
        "current_state": [],
        "scheduled_state": [],
        "geometric_return": [],
        "holding_time": [],
        "target_radius": [],
        "sampled_closure_distance": [],
    }
    profiles: dict[int, dict[str, Any]] = kernel["particle_profiles"]
    for particle_id in kernel["active_particle_ids"]:
        profile = profiles[int(particle_id)]
        try:
            mean_wait = float(profile["mean_wait"])
            current_state = int(profile["initial_state"])
            current_vector = np.asarray(profile["initial_vector"], dtype=float).copy()
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("active particle profiles are incomplete") from error
        if (
            not math.isfinite(mean_wait)
            or mean_wait <= 0.0
            or current_state not in (ESCAPE_STATE, RETURN_STATE)
            or current_vector.shape != (3,)
            or np.any(~np.isfinite(current_vector))
            or np.linalg.norm(current_vector) <= 0.0
        ):
            raise ValueError("active particle profiles contain invalid values")
        current_radius = float(np.linalg.norm(current_vector))
        current_time = 0
        while current_time < horizon:
            source_bin = _profile_radius_bin(
                profile,
                current_radius,
                radial_bin_count,
            )
            candidates = np.flatnonzero(
                (current_states == current_state) & (source_bins == source_bin)
            )
            supported: list[int] = []
            for candidate in candidates:
                target_radius = _profile_radius(
                    profile,
                    float(target_quantiles[candidate]),
                )
                if next_states[candidate] != RETURN_STATE:
                    supported.append(int(candidate))
                    continue
                closure = float(closure_distances[candidate])
                if abs(current_radius - target_radius) <= closure <= current_radius + target_radius:
                    supported.append(int(candidate))
            if not supported:
                raise ValueError(
                    "empty conditional support for current state, radial bin, and geometry"
                )
            record = supported[int(state_rng.integers(0, len(supported)))]
            scheduled_state = int(next_states[record])
            holding_time = max(1, int(round(normalized_waits[record] * mean_wait)))
            event_time = current_time + holding_time
            if event_time > horizon:
                break
            target_radius = _profile_radius(profile, float(target_quantiles[record]))
            sampled_closure = float(closure_distances[record])
            if model == "anchor_aware_semi_markov" and scheduled_state == RETURN_STATE:
                relative_cosine = (
                    sampled_closure**2 - current_radius**2 - target_radius**2
                ) / (2.0 * current_radius * target_radius)
                if not -1.0 <= relative_cosine <= 1.0:
                    raise ValueError("unsupported return tuple cannot be clipped")
            elif model == "state_schedule_without_anchor_geometry":
                geometry_candidates = np.flatnonzero(source_bins == source_bin)
                if len(geometry_candidates) == 0:
                    raise ValueError("state-schedule control lacks recoil geometry support")
                geometry_record = geometry_candidates[
                    int(geometry_rng.integers(0, len(geometry_candidates)))
                ]
                relative_cosine = float(relative_cosines[geometry_record])
            else:
                relative_cosine = float(relative_cosines[record])
            target_vector = target_radius * _fixed_polar_direction(
                current_vector,
                relative_cosine,
                geometry_rng,
            )
            geometric_return = int(
                np.linalg.norm(current_vector + target_vector) <= threshold
            )
            output["particle"].append(int(particle_id))
            output["time"].append(event_time)
            output["jump_vector"].append(target_vector)
            output["source_vector"].append(current_vector.copy())
            output["current_state"].append(current_state)
            output["scheduled_state"].append(scheduled_state)
            output["geometric_return"].append(geometric_return)
            output["holding_time"].append(holding_time)
            output["target_radius"].append(target_radius)
            output["sampled_closure_distance"].append(sampled_closure)
            current_time = event_time
            current_state = scheduled_state
            current_vector = target_vector
            current_radius = target_radius

    if not output["particle"]:
        raise ValueError("simulation produced no events")
    vector_fields = {"jump_vector", "source_vector"}
    integer_fields = {
        "particle",
        "time",
        "current_state",
        "scheduled_state",
        "geometric_return",
        "holding_time",
    }
    result = {
        key: np.asarray(
            values,
            dtype=float if key in vector_fields or key not in integer_fields else int,
        )
        for key, values in output.items()
    }
    result["unsupported_tuple_count"] = np.array([0.0])
    result["active_particle_count"] = np.array([float(len(kernel["active_particle_ids"]))])
    result["non_propagating_particle_count"] = np.array(
        [float(len(kernel["non_propagating_particle_ids"]))]
    )
    return result


def _conditional_probability(current: np.ndarray, following: np.ndarray, state: int) -> float:
    selected = current == state
    return float(np.mean(following[selected])) if np.any(selected) else math.nan


def _maximum_quantile_error(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    relative: bool,
) -> float:
    if len(reference) == 0 or len(candidate) == 0:
        return math.inf
    errors = []
    for quantile in (0.25, 0.50, 0.75, 0.90):
        first = float(np.quantile(reference, quantile))
        second = float(np.quantile(candidate, quantile))
        if relative:
            errors.append(abs(second / first - 1.0) if first != 0.0 else math.inf)
        else:
            errors.append(abs(second - first))
    return max(errors)


def anchor_path_quality(
    kernel: dict[str, Any],
    synthetic: dict[str, np.ndarray],
    *,
    model: str,
) -> dict[str, float]:
    """Measure calibration-state, holding, radial, recoil, and closure fidelity."""

    _validate_simulation_kernel(kernel)
    if model not in {
        "anchor_aware_semi_markov",
        "state_schedule_without_anchor_geometry",
    }:
        raise ValueError("unknown anchor semi-Markov model")
    required = (
        "current_state",
        "scheduled_state",
        "geometric_return",
        "holding_time",
        "target_radius",
        "source_vector",
        "jump_vector",
    )
    try:
        length = len(np.asarray(synthetic["scheduled_state"]))
        if any(len(np.asarray(synthetic[name])) != length for name in required):
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("synthetic event fields must be present and aligned") from error
    if length < 1:
        raise ValueError("synthetic path must contain events")

    reference_current = np.asarray(kernel["current_state"], dtype=int)
    reference_next = np.asarray(kernel["next_state"], dtype=int)
    candidate_current = np.asarray(synthetic["current_state"], dtype=int)
    candidate_next = np.asarray(synthetic["scheduled_state"], dtype=int)
    reference_wait = np.asarray(kernel["holding_time"], dtype=float)
    candidate_wait = np.asarray(synthetic["holding_time"], dtype=float)
    reference_radius = np.asarray(kernel["target_radius"], dtype=float)
    candidate_radius = np.asarray(synthetic["target_radius"], dtype=float)
    reference_cosine = np.asarray(kernel["relative_cosine"], dtype=float)
    source = np.asarray(synthetic["source_vector"], dtype=float)
    target = np.asarray(synthetic["jump_vector"], dtype=float)
    candidate_cosine = np.sum(source * target, axis=1) / (
        np.linalg.norm(source, axis=1) * np.linalg.norm(target, axis=1)
    )
    threshold = float(kernel["radius_threshold"])
    reference_return = reference_next == RETURN_STATE
    candidate_return = candidate_next == RETURN_STATE
    geometric_return = np.asarray(synthetic["geometric_return"], dtype=int)
    reference_closure = np.asarray(kernel["closure_distance"], dtype=float)[
        reference_return
    ] / threshold
    candidate_closure = np.linalg.norm(source + target, axis=1)[candidate_return] / threshold

    result: dict[str, float] = {
        "scheduled_return_fraction_absolute_error": abs(
            float(np.mean(candidate_next)) - float(np.mean(reference_next))
        ),
        "scheduled_return_given_return_absolute_error": abs(
            _conditional_probability(candidate_current, candidate_next, RETURN_STATE)
            - _conditional_probability(reference_current, reference_next, RETURN_STATE)
        ),
        "scheduled_return_given_escape_absolute_error": abs(
            _conditional_probability(candidate_current, candidate_next, ESCAPE_STATE)
            - _conditional_probability(reference_current, reference_next, ESCAPE_STATE)
        ),
        "radial_mean_relative_error": abs(
            float(np.mean(candidate_radius)) / float(np.mean(reference_radius)) - 1.0
        ),
        "radial_standard_deviation_relative_error": abs(
            float(np.std(candidate_radius)) / float(np.std(reference_radius)) - 1.0
        ),
        "lag_one_cosine_mean_absolute_error": abs(
            float(np.mean(candidate_cosine)) - float(np.mean(reference_cosine))
        ),
        "lag_one_cosine_quantile_maximum_absolute_error": _maximum_quantile_error(
            reference_cosine,
            candidate_cosine,
            relative=False,
        ),
        "geometric_return_fraction_absolute_error": abs(
            float(np.mean(geometric_return)) - float(np.mean(reference_next))
        ),
        "return_closure_quantile_maximum_error_over_dw": _maximum_quantile_error(
            reference_closure,
            candidate_closure,
            relative=False,
        ),
        "geometric_return_quality_required": float(
            model == "anchor_aware_semi_markov"
        ),
        "unsupported_tuple_count": float(
            np.asarray(synthetic.get("unsupported_tuple_count", [math.inf]))[0]
        ),
        "active_particle_count": float(
            np.asarray(synthetic.get("active_particle_count", [math.nan]))[0]
        ),
        "non_propagating_particle_count": float(
            np.asarray(synthetic.get("non_propagating_particle_count", [math.nan]))[0]
        ),
    }
    for state, label in ((RETURN_STATE, "return"), (ESCAPE_STATE, "escape")):
        reference_selected = reference_wait[reference_next == state]
        candidate_selected = candidate_wait[candidate_next == state]
        result[f"{label}_holding_time_mean_relative_error"] = (
            abs(
                float(np.mean(candidate_selected))
                / float(np.mean(reference_selected))
                - 1.0
            )
            if len(reference_selected) and len(candidate_selected)
            else math.inf
        )
        result[f"{label}_holding_time_quantile_maximum_relative_error"] = (
            _maximum_quantile_error(
                reference_selected,
                candidate_selected,
                relative=True,
            )
        )
    return result
