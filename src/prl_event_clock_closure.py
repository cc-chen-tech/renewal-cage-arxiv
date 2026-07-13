"""Strict train-to-held-out event-clock closure for continuous trajectories.

This module deliberately keeps the microscopic inputs separate from the
held-out displacement observables.  It is an analysis protocol, not a claim
that a marked-renewal approximation is universally sufficient.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from renewal_cage import (
    block_trajectory_observables,
    event_clock_statistics,
    event_space_correlated_diffusion,
    finite_flight_self_intermediate_scattering,
    finite_flight_weight_integral,
    phop_values,
)


EVENT_DEFINITION = "candelier_phop_contiguous_peak_recursive_ABA_removal"
VARIANTS = (
    "full_event_clock",
    "constant_rate_ctrw",
    "static_rate_disorder",
    "uncorrelated_jumps",
    "instantaneous_flight",
)


def p_hop_increment_partition(
    positions: np.ndarray,
    *,
    threshold: float,
    half_window: int,
) -> dict[str, np.ndarray]:
    """Partition an exact trajectory into p_hop-active and complementary increments.

    An active p_hop center at time ``t`` uses the two averaging windows
    ``[t-half_window, t)`` and ``[t, t+half_window)``.  The union of those
    increment intervals is the active-flight mask.  The returned paths are
    therefore an identity, not a fitted event reconstruction:

    ``positions - positions[0] = active_positions + cage_positions``.
    """

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 3:
        raise ValueError("positions must have shape (at least 3 frames, particles, 3)")
    if np.any(~np.isfinite(positions)) or threshold <= 0.0 or half_window < 1:
        raise ValueError("positions must be finite and p_hop controls must be positive")
    times, activity = phop_values(positions, half_window=half_window)
    active_center = activity > threshold
    increment_mask = np.zeros((positions.shape[0] - 1, positions.shape[1]), dtype=bool)
    for time, selected in zip(times, active_center):
        if np.any(selected):
            start = max(0, int(time) - half_window)
            stop = min(len(increment_mask), int(time) + half_window)
            increment_mask[start:stop, selected] = True
    increments = np.diff(positions, axis=0)
    active_positions = np.concatenate(
        [np.zeros_like(positions[:1]), np.cumsum(increments * increment_mask[..., None], axis=0)], axis=0
    )
    cage_positions = np.concatenate(
        [np.zeros_like(positions[:1]), np.cumsum(increments * (~increment_mask)[..., None], axis=0)], axis=0
    )
    return {
        "activity_times": times,
        "phop_activity": activity,
        "active_increment_mask": increment_mask,
        "active_positions": active_positions,
        "cage_positions": cage_positions,
    }


def event_cage_cross_memory(
    active_positions: np.ndarray,
    cage_positions: np.ndarray,
    *,
    lags: np.ndarray,
    max_lag: int,
) -> dict[str, np.ndarray]:
    """Measure the increment cross-memory and its stationary MSD contribution.

    For ``Delta r_n = e_n + c_n``, define
    ``C_ec(s)=<e_n dot c_(n+s)>``.  Stationarity gives the cross term in the
    lag-``L`` MSD as ``2 sum_(|s|<L) (L-|s|) C_ec(s)``.  The function returns
    that kernel, the resulting stationary reconstruction where it is covered
    by ``max_lag``, and the direct finite-trajectory cross MSD.
    """

    active_positions = np.asarray(active_positions, dtype=float)
    cage_positions = np.asarray(cage_positions, dtype=float)
    lags = np.asarray(lags, dtype=int)
    if active_positions.ndim != 3 or active_positions.shape[2] != 3 or cage_positions.shape != active_positions.shape:
        raise ValueError("active_positions and cage_positions must be matching (frames, particles, 3) arrays")
    if not len(lags) or np.any(lags < 1) or np.any(lags >= len(active_positions)):
        raise ValueError("lags must be positive and fit inside the paths")
    if max_lag < 0 or max_lag >= len(active_positions) - 1:
        raise ValueError("max_lag must fit inside the increment paths")
    active_increment = np.diff(active_positions, axis=0)
    cage_increment = np.diff(cage_positions, axis=0)
    kernel_lags = np.arange(-max_lag, max_lag + 1)
    kernel = np.empty(len(kernel_lags), dtype=float)
    increment_count, particle_count = active_increment.shape[:2]
    fft_length = 1 << int(math.ceil(math.log2(2 * increment_count - 1)))
    cross_correlation = np.fft.irfft(
        np.fft.rfft(cage_increment, n=fft_length, axis=0) * np.conj(np.fft.rfft(active_increment, n=fft_length, axis=0)),
        n=fft_length,
        axis=0,
    )
    for index, offset in enumerate(kernel_lags):
        correlation_index = offset if offset >= 0 else fft_length + offset
        kernel[index] = float(np.sum(cross_correlation[correlation_index])) / (
            (increment_count - abs(offset)) * particle_count
        )
    direct = np.empty(len(lags), dtype=float)
    stationary = np.full(len(lags), np.nan, dtype=float)
    for index, lag in enumerate(lags):
        active_displacement = active_positions[lag:] - active_positions[:-lag]
        cage_displacement = cage_positions[lag:] - cage_positions[:-lag]
        direct[index] = float(2.0 * np.mean(np.sum(active_displacement * cage_displacement, axis=2)))
        if lag <= max_lag + 1:
            offsets = np.arange(-(lag - 1), lag)
            weights = lag - np.abs(offsets)
            stationary[index] = float(2.0 * np.sum(weights * kernel[offsets + max_lag]))
    return {
        "kernel_lags": kernel_lags,
        "increment_cross_correlation": kernel,
        "direct_cross_msd": direct,
        "stationary_cross_msd": stationary,
    }


def causal_event_cage_response_normal_equations(
    active_increment: np.ndarray,
    cage_increment: np.ndarray,
    *,
    start: int,
    stop: int,
    history_order: int,
) -> dict[str, np.ndarray]:
    """Build the isotropic causal event-to-cage response normal equations.

    The response model is ``c_n = sum_(s=1)^p g_s e_(n-s) + xi_n``.  Each
    Cartesian component and particle is a replicate of the same scalar memory
    kernel; the function returns sufficient statistics, so independent
    trajectories can be pooled without combining their raw arrays.
    """

    active_increment = np.asarray(active_increment, dtype=float)
    cage_increment = np.asarray(cage_increment, dtype=float)
    if active_increment.ndim != 3 or active_increment.shape[2] != 3 or cage_increment.shape != active_increment.shape:
        raise ValueError("active_increment and cage_increment must be matching (time, particles, 3) arrays")
    if not np.all(np.isfinite(active_increment)) or not np.all(np.isfinite(cage_increment)):
        raise ValueError("increments must be finite")
    if history_order < 1 or start < history_order or not (start < stop <= len(active_increment)):
        raise ValueError("history order and fitting range must be valid")
    gram = np.zeros((history_order, history_order), dtype=float)
    right = np.zeros(history_order, dtype=float)
    lags = np.arange(1, history_order + 1)
    for time in range(start, stop):
        design = np.moveaxis(active_increment[time - lags], 0, -1).reshape(-1, history_order)
        target = cage_increment[time].reshape(-1)
        gram += design.T @ design
        right += design.T @ target
    return {"gram_matrix": gram, "right_hand_side": right}


def causal_event_cage_response_prediction(active_increment: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return the causal event-induced cage increment from a scalar kernel."""

    active_increment = np.asarray(active_increment, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if active_increment.ndim != 3 or active_increment.shape[2] != 3 or not np.all(np.isfinite(active_increment)):
        raise ValueError("active_increment must be a finite (time, particles, 3) array")
    if kernel.ndim != 1 or not len(kernel) or not np.all(np.isfinite(kernel)):
        raise ValueError("kernel must be a nonempty finite vector")
    prediction = np.zeros_like(active_increment)
    lags = np.arange(1, len(kernel) + 1)
    for time in range(len(kernel), len(active_increment)):
        prediction[time] = np.tensordot(kernel, active_increment[time - lags], axes=(0, 0))
    return prediction


def event_age_since_active_increment(active_increment_mask: np.ndarray, *, maximum_age: int) -> np.ndarray:
    """Return the causal, capped interval age since each particle's active event.

    The supplied mask is indexed by increment time.  An active increment has
    age zero; inactive increments advance the preceding age by one up to the
    chosen cap.  The cap represents the stationary long-since-event class and
    also supplies a defined value at the finite record boundary.
    """

    mask = np.asarray(active_increment_mask, dtype=bool)
    if mask.ndim != 2 or not len(mask) or not mask.shape[1] or maximum_age < 0:
        raise ValueError("active_increment_mask must be nonempty (time, particles) and maximum_age nonnegative")
    age = np.empty(mask.shape, dtype=int)
    current = np.full(mask.shape[1], maximum_age, dtype=int)
    for time, active in enumerate(mask):
        current = np.where(active, 0, np.minimum(current + 1, maximum_age))
        age[time] = current
    return age


def event_age_bin_lower_edges(age_edges: np.ndarray) -> np.ndarray:
    """Return lower-inclusive labels for ``np.digitize(..., right=True)`` bins."""

    edges = np.asarray(age_edges, dtype=int)
    if edges.ndim != 1 or not len(edges) or edges[0] != 0 or np.any(edges < 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("age edges must be a sorted, distinct nonnegative vector starting at zero")
    return np.concatenate([np.array([0], dtype=int), edges + 1])


def _events_with_active_durations(
    positions: np.ndarray,
    *,
    threshold: float,
    half_window: int,
) -> dict[str, np.ndarray]:
    """Apply the fixed p_hop rule while retaining each contiguous active width."""

    times, activity = phop_values(positions, half_window=half_window)
    retained: list[tuple[int, int, float, np.ndarray, int, np.ndarray, np.ndarray]] = []
    radius = math.sqrt(threshold)
    for particle in range(positions.shape[1]):
        active = np.flatnonzero(activity[:, particle] > threshold)
        if not len(active):
            continue
        boundaries = np.flatnonzero(np.diff(active) > 1) + 1
        local_events: list[tuple[int, int, float, np.ndarray, int, np.ndarray, np.ndarray]] = []
        for group in np.split(active, boundaries):
            peak_index = int(group[np.argmax(activity[group, particle])])
            event_time = int(times[peak_index])
            before = np.mean(positions[event_time - half_window : event_time, particle], axis=0)
            after = np.mean(positions[event_time : event_time + half_window, particle], axis=0)
            event = (
                particle,
                event_time,
                float(activity[peak_index, particle]),
                after - before,
                int(len(group)),
                before,
                after,
            )
            if local_events and np.linalg.norm(after - local_events[-1][5]) <= radius:
                local_events.pop()
            else:
                local_events.append(event)
        retained.extend(local_events)
    if not retained:
        raise ValueError("fixed p_hop protocol found no nonrecrossing events")
    retained.sort(key=lambda item: (item[0], item[1]))
    return {
        "particle": np.array([item[0] for item in retained], dtype=int),
        "time": np.array([item[1] for item in retained], dtype=float),
        "phop": np.array([item[2] for item in retained], dtype=float),
        "jump_vector": np.stack([item[3] for item in retained]),
        "active_duration": np.array([item[4] for item in retained], dtype=float),
    }


def _cage_coordinate_variance(train: np.ndarray) -> float:
    """One-frame microscopic cage estimate, with no held-out macro fit."""

    increments = np.diff(train, axis=0)
    return max(float(np.mean(increments**2)) / 2.0, 1e-12)


def _compressed_mark_distribution(marks: np.ndarray, *, maximum_bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Compress measured jump components into a normalized training-only law."""

    marks = np.asarray(marks, dtype=float)
    if marks.ndim != 1 or not len(marks) or np.any(~np.isfinite(marks)):
        raise ValueError("marks must be a nonempty finite vector")
    bin_count = min(maximum_bins, max(16, int(math.sqrt(len(marks)))))
    counts, edges = np.histogram(marks, bins=bin_count)
    keep = counts > 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[keep], counts[keep].astype(float) / float(np.sum(counts[keep]))


def _marked_poisson_ngp_3d(
    times: np.ndarray,
    *,
    event_rate: float,
    jump_squared_mean: float,
    jump_fourth_mean: float,
    cage_coordinate_variance: float,
    flight_duration: float,
    count_fano: float,
) -> np.ndarray:
    """Isotropic marked-flight NGP approximation using only training statistics."""

    out = np.empty_like(times, dtype=float)
    for index, time in np.ndenumerate(times):
        weight2 = finite_flight_weight_integral(float(time), flight_duration, 2)
        weight4 = finite_flight_weight_integral(float(time), flight_duration, 4)
        coordinate_variance = cage_coordinate_variance + event_rate * weight2 * jump_squared_mean / 3.0
        second = 3.0 * coordinate_variance
        fourth = 15.0 * coordinate_variance**2
        # The final term is the marked-count fourth cumulant.  Fano=1 is the
        # constant-rate CTRW; Fano>1 is the measured static-rate-disorder null.
        fourth += event_rate * weight4 * jump_fourth_mean
        fourth += max(count_fano - 1.0, 0.0) * (event_rate * weight2 * jump_squared_mean) ** 2
        out[index] = 3.0 * fourth / (5.0 * second**2) - 1.0
    return out


def _score(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(predicted) & np.isfinite(observed) & (np.abs(observed) > 1e-14)
    if not np.any(finite):
        return {"relative_rmse": math.nan, "max_relative_error": math.nan, "available": 0.0}
    relative = (predicted[finite] - observed[finite]) / observed[finite]
    return {
        "relative_rmse": float(math.sqrt(np.mean(relative**2))),
        "max_relative_error": float(np.max(np.abs(relative))),
        "available": 1.0,
    }


def summarize_threshold_ensemble(
    rows: list[dict[str, float | str]],
    *,
    metric_keys: tuple[str, ...],
    pass_threshold: float = 0.20,
) -> list[dict[str, float]]:
    """Aggregate every preregistered threshold without selecting a best fit."""

    if not rows or not metric_keys or pass_threshold <= 0.0:
        raise ValueError("rows, metric_keys, and pass_threshold must be provided")
    output: list[dict[str, float]] = []
    thresholds = sorted({float(row["threshold"]) for row in rows})
    for threshold in thresholds:
        selected = [row for row in rows if float(row["threshold"]) == threshold]
        summary: dict[str, float] = {
            "threshold": threshold,
            "replicate_count": float(len(selected)),
            "acceptance_relative_error": float(pass_threshold),
        }
        metric_passes: list[bool] = []
        for key in metric_keys:
            values = np.array([float(row[key]) for row in selected], dtype=float)
            if np.any(~np.isfinite(values)):
                raise ValueError(f"{key} must be finite")
            summary[key] = float(np.mean(values))
            summary[f"{key}_se"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            metric_passes.append(bool(np.mean(values) <= pass_threshold))
        summary["all_observables_pass"] = float(all(metric_passes))
        output.append(summary)
    return output


def time_split_event_clock_closure(
    positions: np.ndarray,
    *,
    train_stop: int,
    threshold: float,
    half_window: int,
    lags: np.ndarray,
    wave_numbers: np.ndarray,
    overlap_radius: float,
    origin_stride: int,
) -> dict[str, Any]:
    """Predict held-out dynamics from a fixed p_hop clock in the training half.

    `positions` must be unwrapped and sampled on a physical, uniform time axis.
    The return value intentionally reports all variants, including failures.
    """

    positions = np.asarray(positions, dtype=float)
    lags = np.asarray(lags, dtype=int)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape (frames, particles, 3)")
    if not (2 * half_window + 3 < train_stop < positions.shape[0] - 2):
        raise ValueError("train_stop must leave usable train and held-out segments")
    if threshold <= 0.0 or half_window < 1:
        raise ValueError("threshold and half_window must be positive")
    if len(lags) == 0 or np.any(lags <= 0) or np.any(lags >= positions.shape[0] - train_stop):
        raise ValueError("lags must fit inside the held-out segment")
    if len(wave_numbers) == 0 or np.any(wave_numbers <= 0.0):
        raise ValueError("wave_numbers must be positive")

    train = positions[:train_stop]
    heldout = positions[train_stop:]
    events = _events_with_active_durations(train, threshold=threshold, half_window=half_window)
    stats = event_clock_statistics(
        events,
        duration=float(train_stop - 1),
        particle_count=positions.shape[1],
        dimension=3,
    )
    observed_rows = block_trajectory_observables(
        heldout,
        lags=lags,
        block_size=heldout.shape[0] - 1,
        wave_numbers=wave_numbers,
        overlap_radius=overlap_radius,
        origin_stride=origin_stride,
    )
    observed_rows = sorted(observed_rows, key=lambda row: float(row["lag"]))
    lag_values = np.array([float(row["lag"]) for row in observed_rows])
    observed_diffusion = float(observed_rows[-1]["msd"]) / (6.0 * lag_values[-1])
    observed_ngp = np.array([float(row["ngp_3d"]) for row in observed_rows])
    observed_chi4 = np.array([float(row["overlap_chi4"]) for row in observed_rows])
    jump_components, jump_probabilities = _compressed_mark_distribution(events["jump_vector"].reshape(-1))
    jump_squared = np.sum(events["jump_vector"] ** 2, axis=1)
    flight_duration = float(np.median(events["active_duration"]))
    cage_variance = _cage_coordinate_variance(train)
    correlations = [
        stats["jump_correlation_lag1_over_q"] * stats["jump_squared_mean"],
        stats["jump_correlation_lag2_over_q"] * stats["jump_squared_mean"],
    ]
    full_diffusion = event_space_correlated_diffusion(
        stats["event_rate"], stats["jump_squared_mean"], correlations, dimension=3
    )
    uncorrelated_diffusion = stats["uncorrelated_diffusion"]
    common = {
        "event_rate": float(stats["event_rate"]),
        "cage_variance": cage_variance,
        "cage_tau": 1.0,
        "mark_displacements": jump_components,
        "mark_probabilities": jump_probabilities,
    }

    variants: dict[str, dict[str, Any]] = {}
    for name in VARIANTS:
        instantaneous = name in {"constant_rate_ctrw", "instantaneous_flight"}
        local_flight_duration = 0.0 if instantaneous else flight_duration
        local_fano = 1.0 if name in {"constant_rate_ctrw", "uncorrelated_jumps", "instantaneous_flight"} else stats["count_fano"]
        diffusion = full_diffusion if name in {"full_event_clock", "instantaneous_flight"} else uncorrelated_diffusion
        predicted_fs = {
            float(k): finite_flight_self_intermediate_scattering(
                float(k), lag_values, flight_duration=local_flight_duration, **common
            )
            for k in wave_numbers
        }
        predicted_ngp = _marked_poisson_ngp_3d(
            lag_values,
            event_rate=float(stats["event_rate"]),
            jump_squared_mean=float(stats["jump_squared_mean"]),
            jump_fourth_mean=float(np.mean(jump_squared**2)),
            cage_coordinate_variance=cage_variance,
            flight_duration=local_flight_duration,
            count_fano=float(local_fano),
        )
        fs_scores = {
            f"fs_k{k:g}": _score(predicted_fs[float(k)], np.array([row[f"fs_k{k:g}".replace(".", "p")] for row in observed_rows]))
            for k in wave_numbers
        }
        variants[name] = {
            "predicted_diffusion": float(diffusion),
            "observed_diffusion": observed_diffusion,
            "diffusion_relative_error": abs(diffusion / observed_diffusion - 1.0),
            "predicted_ngp": predicted_ngp,
            "predicted_fs": predicted_fs,
            "ngp": _score(predicted_ngp, observed_ngp),
            "fs": fs_scores,
            "predicted_count_fano_proxy": float(local_fano),
            "observed_overlap_chi4_peak": float(np.max(observed_chi4)),
            "finite_flight_duration": float(local_flight_duration),
        }

    return {
        "event_definition": EVENT_DEFINITION,
        "train_stop": float(train_stop),
        "heldout_frame_count": float(heldout.shape[0]),
        "threshold": float(threshold),
        "half_window": float(half_window),
        "train_event_count": float(stats["event_count"]),
        "train_event_rate": float(stats["event_rate"]),
        "train_exchange_mean": float(stats["exchange_mean"]),
        "train_persistence_mean": float(stats["stationary_persistence_mean"]),
        "train_jump_squared_mean": float(stats["jump_squared_mean"]),
        "train_flight_duration": flight_duration,
        "train_cage_coordinate_variance": cage_variance,
        "fit_parameters_from_macro_observables": 0.0,
        "thermodynamic_claim_allowed": 0.0,
        "variants": VARIANTS,
        "variant_scores": variants,
        "observed_lags": lag_values,
        "observed_ngp": observed_ngp,
    }
