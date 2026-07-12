"""Reproducible preparation of decorrelated Kob-Andersen trajectory restarts."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np


SOURCE_DOI = "10.5281/zenodo.7469766"


def independent_sample_ci95(
    *,
    mean: float,
    standard_error: float,
    sample_count: int,
) -> tuple[float, float, float]:
    """Two-sided 95% Student-t interval for a mean of independent samples."""

    if sample_count < 2 or not math.isfinite(mean) or not math.isfinite(standard_error):
        raise ValueError("a finite mean, standard error, and at least two samples are required")
    if standard_error < 0.0:
        raise ValueError("standard_error must be nonnegative")
    critical_by_df = {
        1: 12.706204736432095,
        2: 4.302652729911275,
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.570581835636314,
        6: 2.446911848791681,
        7: 2.3646242510102993,
        8: 2.3060041350333704,
        9: 2.2621571628540993,
        10: 2.2281388519649385,
        11: 2.200985160082949,
        12: 2.1788128296634177,
        13: 2.1603686564610127,
        14: 2.1447866879169273,
        15: 2.131449545559323,
        16: 2.1199052992210112,
        17: 2.1098155778331806,
        18: 2.10092204024096,
        19: 2.093024054408263,
        20: 2.0859634472658364,
        21: 2.079613844727662,
        22: 2.0738730679040147,
        23: 2.0686576104190406,
        24: 2.0638985616280205,
        25: 2.059538552753294,
        26: 2.055529438642871,
        27: 2.0518305164802833,
        28: 2.048407141795244,
        29: 2.045229642132703,
        30: 2.0422724563012373,
    }
    critical = critical_by_df.get(sample_count - 1, 1.959963984540054)
    margin = critical * standard_error
    return mean - margin, mean + margin, critical


def summarize_replicate_binned_metric(
    rows: Sequence[dict[str, object]],
    *,
    bin_key: str,
    metric_key: str,
    replicate_key: str = "replicate",
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Average time blocks within replicas, then estimate error between replicas."""

    if not rows:
        raise ValueError("at least one binned replicate row is required")
    required = {bin_key, metric_key, replicate_key}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("every row must contain bin, metric, and replicate keys")

    grouped: dict[tuple[float, float], list[float]] = {}
    for row in rows:
        key = (float(row[bin_key]), float(row[replicate_key]))
        value = float(row[metric_key])
        if not all(math.isfinite(item) for item in (*key, value)):
            raise ValueError("bin, metric, and replicate values must be finite")
        grouped.setdefault(key, []).append(value)

    replicate_rows = [
        {
            bin_key: bin_value,
            replicate_key: replicate,
            metric_key: float(np.mean(values)),
            "within_replicate_block_count": float(len(values)),
        }
        for (bin_value, replicate), values in sorted(grouped.items())
    ]
    summary_rows: list[dict[str, float]] = []
    for bin_value in sorted({float(row[bin_key]) for row in replicate_rows}):
        selected = [row for row in replicate_rows if float(row[bin_key]) == bin_value]
        values = np.array([float(row[metric_key]) for row in selected])
        if len(values) < 2:
            raise ValueError("at least two independent replicas are required per bin")
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(values))
        block_counts = [float(row["within_replicate_block_count"]) for row in selected]
        mean = float(np.mean(values))
        ci_low, ci_high, critical = independent_sample_ci95(
            mean=mean,
            standard_error=standard_error,
            sample_count=len(values),
        )
        summary_rows.append(
            {
                bin_key: bin_value,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "ci95_critical_value": critical,
                "ci95_method": "student_t_independent_replicates",
                "independent_replicate_count": float(len(values)),
                "within_replicate_block_count_min": min(block_counts),
                "within_replicate_block_count_max": max(block_counts),
            }
        )
    return replicate_rows, summary_rows


def trajectory_diffusion_estimate(
    unwrapped_positions: np.ndarray,
    *,
    lag: int,
    origin_stride: int,
    particle_mask: np.ndarray | None = None,
) -> float:
    """Estimate diffusion from a fixed lag using only origins in one trajectory window."""

    positions = np.asarray(unwrapped_positions)
    if positions.ndim != 3 or positions.shape[2] < 1:
        raise ValueError("unwrapped_positions must have shape (frames, particles, dimensions)")
    if lag < 1 or lag >= len(positions) or origin_stride < 1:
        raise ValueError("lag and origin_stride must define at least one valid displacement")
    if particle_mask is None:
        mask = np.ones(positions.shape[1], dtype=bool)
    else:
        mask = np.asarray(particle_mask, dtype=bool)
        if mask.shape != (positions.shape[1],) or not np.any(mask):
            raise ValueError("particle_mask must select at least one particle")
    origins = np.arange(0, len(positions) - lag, origin_stride, dtype=int)
    displacement = positions[origins + lag][:, mask] - positions[origins][:, mask]
    msd = float(np.mean(np.sum(displacement**2, axis=2)))
    return msd / (2.0 * positions.shape[2] * lag)


def summarize_heldout_event_transport(
    rows: Sequence[dict[str, object]],
    *,
    minimum_coverage: float,
    maximum_coverage: float,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    """Score event-clock diffusion predictions against independent held-out windows."""

    if not rows or not 0.0 < minimum_coverage < maximum_coverage:
        raise ValueError("rows and an ordered positive coverage interval are required")
    models = {
        "uncorrelated_event_clock": "uncorrelated_event_diffusion",
        "correlated_event_clock": "correlated_event_diffusion",
    }
    summary: list[dict[str, float | str]] = []
    coverage_by_model: dict[str, np.ndarray] = {}
    for model, key in models.items():
        observed = np.array([float(row["observed_diffusion"]) for row in rows])
        predicted = np.array([float(row[key]) for row in rows])
        if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(predicted)):
            raise ValueError("held-out diffusion values must be finite")
        if np.any(observed <= 0.0) or np.any(predicted <= 0.0):
            raise ValueError("held-out diffusion values must be positive")
        coverage = predicted / observed
        coverage_by_model[model] = coverage
        standard_deviation = float(np.std(coverage, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(coverage))
        mean = float(np.mean(coverage))
        ci_low, ci_high, critical = independent_sample_ci95(
            mean=mean,
            standard_error=standard_error,
            sample_count=len(coverage),
        )
        summary.append(
            {
                "model": model,
                "mean_observed_diffusion": float(np.mean(observed)),
                "mean_predicted_diffusion": float(np.mean(predicted)),
                "mean_coverage": mean,
                "standard_deviation_coverage": standard_deviation,
                "standard_error_coverage": standard_error,
                "ci95_low_coverage": ci_low,
                "ci95_high_coverage": ci_high,
                "ci95_critical_value": critical,
                "ci95_method": "student_t_independent_replicates",
                "replicates_within_tolerance": float(
                    np.sum((coverage >= minimum_coverage) & (coverage <= maximum_coverage))
                ),
                "independent_replicate_count": float(len(coverage)),
            }
        )
    correlated = coverage_by_model["correlated_event_clock"]
    passed = bool(np.all((correlated >= minimum_coverage) & (correlated <= maximum_coverage)))
    if passed:
        failure = "none"
    elif float(np.mean(correlated)) < minimum_coverage:
        failure = "correlated_event_clock_undercoverage"
    elif float(np.mean(correlated)) > maximum_coverage:
        failure = "correlated_event_clock_overcoverage"
    else:
        failure = "replicate_transport_inconsistency"
    verdict: dict[str, float | str] = {
        "minimum_coverage": minimum_coverage,
        "maximum_coverage": maximum_coverage,
        "heldout_transport_pass": float(passed),
        "primary_failure": failure,
        "independent_replicate_count": float(len(rows)),
        "macro_fit_parameter_count": 0.0,
        "finite_memory_model_required": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    return summary, verdict


def fit_ornstein_zernike_structure_factor(
    rows: Sequence[dict[str, object]],
) -> dict[str, float | bool]:
    """Fit S(q)=A/[1+(q xi)^2] through a linear inverse transform."""

    q = np.array([float(row["wave_number"]) for row in rows], dtype=float)
    s4 = np.array([float(row["s4"]) for row in rows], dtype=float)
    selected = (q > 0.0) & (s4 > 0.0) & np.isfinite(q) & np.isfinite(s4)
    if np.sum(selected) < 3:
        raise ValueError("at least three positive finite-q S4 shells are required")
    x = q[selected] ** 2
    y = 1.0 / s4[selected]
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    residual_sum = float(np.sum((y - prediction) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    fit_valid = bool(intercept > 0.0 and slope > 0.0)
    return {
        "inverse_intercept": float(intercept),
        "inverse_slope": float(slope),
        "amplitude": float(1.0 / intercept) if fit_valid else math.nan,
        "correlation_length": float(math.sqrt(slope / intercept)) if fit_valid else math.nan,
        "inverse_space_r_squared": float(1.0 - residual_sum / total_sum)
        if total_sum > 0.0
        else 1.0,
        "fit_shell_count": float(np.sum(selected)),
        "fit_wave_number_min": float(np.min(q[selected])),
        "fit_wave_number_max": float(np.max(q[selected])),
        "fit_valid": fit_valid,
    }


def summarize_overlap_s4_replicates(
    rows: Sequence[dict[str, object]],
    *,
    ensemble_correction_available: bool,
) -> tuple[list[dict[str, float]], list[dict[str, float | bool]], dict[str, float | str]]:
    """Gate an overlap-S4 length using fits repeated across independent replicas."""

    _, summary_rows = summarize_replicate_binned_metric(
        rows,
        bin_key="integer_squared",
        metric_key="s4",
    )
    for summary in summary_rows:
        shell = float(summary["integer_squared"])
        selected = [row for row in rows if float(row["integer_squared"]) == shell]
        summary["wave_number"] = float(np.mean([float(row["wave_number"]) for row in selected]))

    fits: list[dict[str, float | bool]] = []
    for replicate in sorted({float(row["replicate"]) for row in rows}):
        selected = [row for row in rows if float(row["replicate"]) == replicate]
        shell_rows: list[dict[str, float]] = []
        for shell in sorted({float(row["integer_squared"]) for row in selected}):
            shell_selected = [row for row in selected if float(row["integer_squared"]) == shell]
            shell_rows.append(
                {
                    "wave_number": float(np.mean([float(row["wave_number"]) for row in shell_selected])),
                    "s4": float(np.mean([float(row["s4"]) for row in shell_selected])),
                }
            )
        fit = fit_ornstein_zernike_structure_factor(
            [row for row in shell_rows if row["wave_number"] > 0.0]
        )
        fit["replicate"] = replicate
        fits.append(fit)

    aggregate_fit = fit_ornstein_zernike_structure_factor(
        [
            {"wave_number": row["wave_number"], "s4": row["mean"]}
            for row in summary_rows
            if float(row["wave_number"]) > 0.0
        ]
    )
    invalid_count = sum(not bool(row["fit_valid"]) for row in fits)
    reproducible = invalid_count == 0 and bool(aggregate_fit["fit_valid"])
    identifiable = reproducible and ensemble_correction_available
    verdict: dict[str, float | str] = {
        "independent_replicate_count": float(len(fits)),
        "invalid_replicate_fit_count": float(invalid_count),
        "aggregate_fit_valid": float(bool(aggregate_fit["fit_valid"])),
        "aggregate_correlation_length": float(aggregate_fit["correlation_length"]),
        "raw_oz_fit_reproducible": float(reproducible),
        "ensemble_correction_available": float(ensemble_correction_available),
        "xi4_identifiable": float(identifiable),
        "xi4_claim_allowed": float(identifiable),
        "thermodynamic_claim_allowed": 0.0,
        "verdict": "xi4_identifiable_with_replicate_and_ensemble_gate"
        if identifiable
        else "xi4_not_identifiable",
    }
    return summary_rows, fits, verdict


def overlap_four_point_structure_factor(
    unwrapped_positions: np.ndarray,
    *,
    box_lengths: np.ndarray,
    lag: int,
    overlap_radius: float,
    origin_stride: int,
    maximum_integer_squared: int,
) -> list[dict[str, float]]:
    """Compute overlap S4 on reciprocal-box wavevector shells."""

    positions = np.asarray(unwrapped_positions, dtype=float)
    box_lengths = np.asarray(box_lengths, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != len(box_lengths):
        raise ValueError("unwrapped_positions must have shape (frames, particles, dimensions)")
    if positions.shape[0] <= lag or lag < 1:
        raise ValueError("lag must be positive and shorter than the trajectory")
    if origin_stride < 1 or maximum_integer_squared < 1:
        raise ValueError("origin_stride and maximum_integer_squared must be positive")
    if not math.isfinite(overlap_radius) or overlap_radius <= 0.0:
        raise ValueError("overlap_radius must be positive and finite")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("positions and positive box lengths must be finite")

    origins = np.arange(0, positions.shape[0] - lag, origin_stride, dtype=int)
    if len(origins) < 2:
        raise ValueError("at least two time origins are required")
    overlap = np.empty((len(origins), positions.shape[1]), dtype=float)
    for index, origin in enumerate(origins):
        displacement = positions[origin + lag] - positions[origin]
        overlap[index] = np.sum(displacement**2, axis=1) < overlap_radius**2
    overlap_mean = float(np.mean(overlap))
    fluctuation = overlap - overlap_mean
    overlap_fraction = np.mean(overlap, axis=1)
    particle_count = positions.shape[1]
    chi4 = particle_count * float(np.var(overlap_fraction))
    rows: list[dict[str, float]] = [
        {
            "integer_squared": 0.0,
            "wave_number": 0.0,
            "wavevector_count": 1.0,
            "s4": chi4,
            "s4_wavevector_standard_deviation": 0.0,
            "s4_wavevector_min": chi4,
            "s4_wavevector_max": chi4,
            "overlap_mean": overlap_mean,
            "origin_count": float(len(origins)),
            "particle_count": float(particle_count),
        }
    ]

    maximum_component = int(math.ceil(math.sqrt(maximum_integer_squared)))
    integer_vectors = np.array(
        [
            vector
            for vector in np.ndindex(*([2 * maximum_component + 1] * positions.shape[2]))
            if 0
            < sum((component - maximum_component) ** 2 for component in vector)
            <= maximum_integer_squared
        ],
        dtype=int,
    ) - maximum_component
    integer_squared = np.sum(integer_vectors**2, axis=1)
    for shell in sorted(set(int(value) for value in integer_squared)):
        vectors = integer_vectors[integer_squared == shell]
        wavevectors = 2.0 * math.pi * vectors / box_lengths
        vector_accumulated = np.zeros(len(vectors), dtype=float)
        for origin_index, origin in enumerate(origins):
            phase = positions[origin] @ wavevectors.T
            amplitude = fluctuation[origin_index] @ np.exp(1j * phase)
            vector_accumulated += np.abs(amplitude) ** 2
        vector_s4 = vector_accumulated / (particle_count * len(origins))
        rows.append(
            {
                "integer_squared": float(shell),
                "wave_number": float(np.mean(np.linalg.norm(wavevectors, axis=1))),
                "wavevector_count": float(len(vectors)),
                "s4": float(np.mean(vector_s4)),
                "s4_wavevector_standard_deviation": float(np.std(vector_s4)),
                "s4_wavevector_min": float(np.min(vector_s4)),
                "s4_wavevector_max": float(np.max(vector_s4)),
                "overlap_mean": overlap_mean,
                "origin_count": float(len(origins)),
                "particle_count": float(particle_count),
            }
        )
    return rows


def fit_spatial_covariance_length(
    rows: Sequence[dict[str, object]],
    *,
    minimum_distance: float,
) -> dict[str, float]:
    """Fit the positive covariance branch to A exp(-r/xi) in log space."""

    if not math.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("minimum_distance must be nonnegative and finite")
    distance = np.array([float(row["distance_midpoint"]) for row in rows])
    covariance = np.array([float(row["mean_covariance_excess"]) for row in rows])
    selected = (distance >= minimum_distance) & (covariance > 0.0)
    if np.sum(selected) < 3:
        raise ValueError("at least three positive covariance bins are required")
    x = distance[selected]
    log_y = np.log(covariance[selected])
    slope, intercept = np.polyfit(x, log_y, 1)
    if slope >= 0.0:
        raise ValueError("positive covariance branch does not decay with distance")
    prediction = intercept + slope * x
    residual_sum = float(np.sum((log_y - prediction) ** 2))
    total_sum = float(np.sum((log_y - np.mean(log_y)) ** 2))
    return {
        "amplitude": float(math.exp(intercept)),
        "correlation_length": float(-1.0 / slope),
        "log_space_r_squared": float(1.0 - residual_sum / total_sum) if total_sum > 0.0 else 1.0,
        "fit_point_count": float(np.sum(selected)),
        "fit_distance_min": float(np.min(x)),
        "fit_distance_max": float(np.max(x)),
    }


def summarize_spatial_covariance_replicates(
    rows: Sequence[dict[str, object]],
    *,
    minimum_distance: float,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, float]],
    list[dict[str, float]],
    dict[str, float],
]:
    """Estimate event-covariance curves and lengths using independent replicas."""

    replicate_rows, summary_rows = summarize_replicate_binned_metric(
        rows,
        bin_key="distance_midpoint",
        metric_key="covariance_excess_over_all_pairs",
    )
    fits: list[dict[str, float]] = []
    for replicate in sorted({float(row["replicate"]) for row in replicate_rows}):
        selected = [row for row in replicate_rows if float(row["replicate"]) == replicate]
        fit = fit_spatial_covariance_length(
            [
                {
                    "distance_midpoint": row["distance_midpoint"],
                    "mean_covariance_excess": row["covariance_excess_over_all_pairs"],
                }
                for row in selected
            ],
            minimum_distance=minimum_distance,
        )
        fit["replicate"] = replicate
        fits.append(fit)

    lengths = np.array([float(row["correlation_length"]) for row in fits])
    if len(lengths) < 2:
        raise ValueError("at least two independent spatial-covariance fits are required")
    standard_deviation = float(np.std(lengths, ddof=1))
    standard_error = standard_deviation / math.sqrt(len(lengths))
    mean = float(np.mean(lengths))
    ci_low, ci_high, critical = independent_sample_ci95(
        mean=mean,
        standard_error=standard_error,
        sample_count=len(lengths),
    )
    fit_summary = {
        "mean_correlation_length": mean,
        "standard_deviation_correlation_length": standard_deviation,
        "standard_error_correlation_length": standard_error,
        "ci95_low_correlation_length": ci_low,
        "ci95_high_correlation_length": ci_high,
        "ci95_critical_value": critical,
        "ci95_method": "student_t_independent_replicates",
        "independent_replicate_count": float(len(lengths)),
    }
    return replicate_rows, summary_rows, fits, fit_summary


def distance_resolved_event_count_covariance(
    events: dict[str, np.ndarray],
    reference_positions: np.ndarray,
    box_lengths: np.ndarray,
    *,
    duration: float,
    count_window: float,
    distance_edges: np.ndarray,
) -> list[dict[str, float]]:
    """Measure spatial count covariance after removing global activity fluctuations."""

    positions = np.asarray(reference_positions, dtype=float)
    box_lengths = np.asarray(box_lengths, dtype=float)
    edges = np.asarray(distance_edges, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != len(box_lengths):
        raise ValueError("reference_positions must have shape (particles, dimensions)")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise ValueError("positions and positive box lengths must be finite")
    if not math.isfinite(duration) or not math.isfinite(count_window) or duration <= 0.0 or count_window <= 0.0:
        raise ValueError("duration and count_window must be positive and finite")
    if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("distance_edges must be a strictly increasing finite sequence")
    particle = np.asarray(events["particle"], dtype=int)
    event_time = np.asarray(events["time"], dtype=float)
    if particle.shape != event_time.shape or np.any(particle < 0) or np.any(particle >= len(positions)):
        raise ValueError("event particles and times must be aligned and in range")
    window_count = int(math.floor(duration / count_window))
    if window_count < 2:
        raise ValueError("at least two complete count windows are required")

    counts = np.zeros((len(positions), window_count), dtype=float)
    window = np.floor(event_time / count_window).astype(int)
    retained = (event_time >= 0.0) & (window >= 0) & (window < window_count)
    np.add.at(counts, (particle[retained], window[retained]), 1.0)
    residual = counts - np.mean(counts, axis=0, keepdims=True)
    residual -= np.mean(residual, axis=1, keepdims=True)

    bin_count = np.zeros(len(edges) - 1, dtype=np.int64)
    bin_sum = np.zeros(len(edges) - 1, dtype=float)
    all_pair_sum = 0.0
    all_pair_count = 0
    for left in range(len(positions) - 1):
        displacement = positions[left + 1 :] - positions[left]
        displacement -= box_lengths * np.rint(displacement / box_lengths)
        distance = np.linalg.norm(displacement, axis=1)
        covariance = residual[left + 1 :] @ residual[left] / window_count
        all_pair_sum += float(np.sum(covariance))
        all_pair_count += len(covariance)
        bin_index = np.searchsorted(edges, distance, side="right") - 1
        valid = (bin_index >= 0) & (bin_index < len(bin_count))
        bin_count += np.bincount(bin_index[valid], minlength=len(bin_count))
        bin_sum += np.bincount(
            bin_index[valid],
            weights=covariance[valid],
            minlength=len(bin_count),
        )

    all_pair_mean = all_pair_sum / all_pair_count
    individual_variance = float(np.mean(residual**2))
    rows: list[dict[str, float]] = []
    for index, pair_count in enumerate(bin_count):
        if pair_count == 0:
            raise ValueError("every distance bin must contain at least one particle pair")
        covariance = bin_sum[index] / pair_count
        excess = covariance - all_pair_mean
        rows.append(
            {
                "distance_low": float(edges[index]),
                "distance_high": float(edges[index + 1]),
                "distance_midpoint": float((edges[index] + edges[index + 1]) / 2.0),
                "pair_count": float(pair_count),
                "count_window": count_window,
                "window_count": float(window_count),
                "retained_event_count": float(np.sum(retained)),
                "count_covariance": float(covariance),
                "all_pair_covariance_null": float(all_pair_mean),
                "covariance_excess_over_all_pairs": float(excess),
                "normalized_covariance_excess": float(excess / individual_variance)
                if individual_variance > 0.0
                else 0.0,
            }
        )
    return rows


def summarize_replicate_curves(
    rows: Sequence[dict[str, object]],
    *,
    metric_keys: Sequence[str],
) -> list[dict[str, float | str]]:
    """Attach between-trajectory uncertainty to each lag-resolved observable."""

    if not rows or not metric_keys:
        raise ValueError("rows and metric_keys must be nonempty")
    grouped: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row["lag"]), []).append(row)
    expected_replicates: set[int] | None = None
    summary: list[dict[str, float | str]] = []
    for lag, lag_rows in sorted(grouped.items()):
        replicates = {int(float(row["replicate"])) for row in lag_rows}
        if len(replicates) < 2 or len(replicates) != len(lag_rows):
            raise ValueError("each lag must contain one row per replicate and at least two replicates")
        if expected_replicates is None:
            expected_replicates = replicates
        elif replicates != expected_replicates:
            raise ValueError("replicate membership must be identical at every lag")
        for metric in metric_keys:
            if any(metric not in row for row in lag_rows):
                raise ValueError(f"missing curve metric {metric}")
            values = np.array([float(row[metric]) for row in lag_rows], dtype=float)
            if np.any(~np.isfinite(values)):
                raise ValueError("curve metrics must be finite")
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1))
            standard_error = standard_deviation / math.sqrt(len(values))
            ci_low, ci_high, critical = independent_sample_ci95(
                mean=mean,
                standard_error=standard_error,
                sample_count=len(values),
            )
            summary.append(
                {
                    "lag": lag,
                    "metric": metric,
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                    "standard_error": standard_error,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "ci95_critical_value": critical,
                    "ci95_method": "student_t_independent_replicates",
                    "independent_replicate_count": float(len(values)),
                }
            )
    return summary


def temperature_scan_verdict(
    high_temperature_rows: Sequence[dict[str, object]],
    low_temperature_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Quantify preregistered cooling trends using independent-replicate intervals."""

    expected_directions = {
        "diffusion": "decrease",
        "alpha_relaxation_time": "increase",
        "diffusion_alpha_product": "increase",
        "ngp_peak": "increase",
        "overlap_chi4_peak": "increase",
    }
    high = {str(row["metric"]): row for row in high_temperature_rows}
    low = {str(row["metric"]): row for row in low_temperature_rows}
    if any(metric not in high or metric not in low for metric in expected_directions):
        raise ValueError("both temperatures must contain every preregistered metric")

    rows: list[dict[str, object]] = []
    for metric, direction in expected_directions.items():
        high_mean = float(high[metric]["mean"])
        low_mean = float(low[metric]["mean"])
        if high_mean <= 0.0 or low_mean <= 0.0:
            raise ValueError("temperature-scan means must be positive")
        if direction == "decrease":
            ratio = high_mean / low_mean
            separated = float(low[metric]["ci95_high"]) < float(high[metric]["ci95_low"])
            effect = "cooling_slowdown"
        else:
            ratio = low_mean / high_mean
            separated = float(low[metric]["ci95_low"]) > float(high[metric]["ci95_high"])
            effect = "cooling_growth"
        rows.append(
            {
                "metric": metric,
                "effect": effect,
                "high_temperature_mean": high_mean,
                "high_temperature_ci95_low": float(high[metric]["ci95_low"]),
                "high_temperature_ci95_high": float(high[metric]["ci95_high"]),
                "low_temperature_mean": low_mean,
                "low_temperature_ci95_low": float(low[metric]["ci95_low"]),
                "low_temperature_ci95_high": float(low[metric]["ci95_high"]),
                "effect_ratio": ratio,
                "directional_ci95_separated": separated,
                "trend_pass": (ratio > 1.0 and separated),
                "thermodynamic_claim_allowed": False,
            }
        )
    return rows


def load_lammps_custom_trajectory(path: Path) -> dict[str, np.ndarray]:
    """Load the fixed custom-dump schema emitted by the replicate protocol."""

    path = Path(path)
    timesteps: list[int] = []
    wrapped_frames: list[np.ndarray] = []
    unwrapped_frames: list[np.ndarray] = []
    expected_types: np.ndarray | None = None
    expected_box: np.ndarray | None = None
    with path.open() as handle:
        while True:
            marker = handle.readline()
            if marker == "":
                break
            if marker.strip() != "ITEM: TIMESTEP":
                raise ValueError("unexpected LAMMPS dump timestep header")
            timesteps.append(int(handle.readline()))
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("unexpected LAMMPS dump atom-count header")
            particle_count = int(handle.readline())
            if particle_count < 1:
                raise ValueError("LAMMPS dump must contain particles")
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("unexpected LAMMPS dump box header")
            bounds = np.array(
                [[float(value) for value in handle.readline().split()[:2]] for _ in range(3)]
            )
            box_lengths = bounds[:, 1] - bounds[:, 0]
            if np.any(box_lengths <= 0.0):
                raise ValueError("LAMMPS dump box lengths must be positive")
            atom_header = handle.readline().strip()
            if atom_header != "ITEM: ATOMS id type x y z ix iy iz":
                raise ValueError("unexpected LAMMPS dump atom schema")
            rows = [handle.readline() for _ in range(particle_count)]
            if any(row == "" for row in rows):
                raise ValueError("truncated LAMMPS dump frame")
            values = np.fromstring("".join(rows), sep=" ")
            if values.size != particle_count * 8:
                raise ValueError("malformed LAMMPS dump atom row")
            values = values.reshape(particle_count, 8)
            atom_ids = values[:, 0].astype(int)
            if not np.array_equal(atom_ids, np.arange(1, particle_count + 1)):
                raise ValueError("LAMMPS dump atoms must be sorted by id")
            particle_types = values[:, 1].astype(int) - 1
            wrapped = values[:, 2:5].astype(np.float32)
            images = values[:, 5:8].astype(np.int64)
            unwrapped = (wrapped.astype(float) + images * box_lengths).astype(np.float32)
            if expected_types is None:
                expected_types = particle_types
                expected_box = box_lengths
            elif not np.array_equal(particle_types, expected_types):
                raise ValueError("particle types changed between dump frames")
            elif not np.allclose(box_lengths, expected_box):
                raise ValueError("box lengths changed between dump frames")
            wrapped_frames.append(wrapped)
            unwrapped_frames.append(unwrapped)

    if not timesteps:
        raise ValueError("LAMMPS dump contains no frames")
    timestep_array = np.asarray(timesteps, dtype=np.int64)
    if len(timestep_array) > 1 and np.any(np.diff(timestep_array) <= 0):
        raise ValueError("LAMMPS dump timesteps must increase strictly")
    return {
        "timesteps": timestep_array,
        "particle_types": np.asarray(expected_types, dtype=int),
        "box_lengths": np.asarray(expected_box, dtype=float),
        "wrapped_positions": np.stack(wrapped_frames),
        "unwrapped_positions": np.stack(unwrapped_frames),
    }


def initial_configuration_fs(
    reference: np.ndarray,
    comparison: np.ndarray,
    box_lengths: np.ndarray,
    *,
    particle_mask: np.ndarray,
    wave_number: float,
) -> float:
    """Axis-averaged self scattering between two periodic configurations."""

    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    box_lengths = np.asarray(box_lengths, dtype=float)
    particle_mask = np.asarray(particle_mask, dtype=bool)
    if reference.shape != comparison.shape or reference.ndim != 2:
        raise ValueError("configurations must have matching particle-by-dimension shapes")
    if box_lengths.shape != (reference.shape[1],) or np.any(box_lengths <= 0.0):
        raise ValueError("box_lengths must be positive and match the spatial dimension")
    if particle_mask.shape != (reference.shape[0],) or not np.any(particle_mask):
        raise ValueError("particle_mask must select at least one particle")
    if not math.isfinite(wave_number) or wave_number <= 0.0:
        raise ValueError("wave_number must be positive and finite")
    if np.any(~np.isfinite(reference)) or np.any(~np.isfinite(comparison)):
        raise ValueError("configurations must be finite")

    displacement = comparison[particle_mask] - reference[particle_mask]
    displacement -= box_lengths * np.rint(displacement / box_lengths)
    return float(np.mean(np.cos(wave_number * displacement)))


def validate_initial_frame_independence(
    positions: Sequence[np.ndarray],
    box_lengths: np.ndarray,
    particle_mask: np.ndarray,
    *,
    frame_indices: Sequence[int],
    wave_number: float = 7.25,
    maximum_absolute_fs: float = 0.1,
) -> list[dict[str, float | int]]:
    """Require every selected parent-frame pair to pass an Fs decorrelation gate."""

    if not 0.0 <= maximum_absolute_fs < 1.0:
        raise ValueError("maximum_absolute_fs must lie in [0, 1)")
    indices = [int(index) for index in frame_indices]
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError("frame_indices must contain at least two unique frames")
    if min(indices) < 0 or max(indices) >= len(positions):
        raise ValueError("frame index is outside the trajectory")

    rows: list[dict[str, float | int]] = []
    for left_offset, left in enumerate(indices[:-1]):
        for right in indices[left_offset + 1 :]:
            fs = initial_configuration_fs(
                positions[left],
                positions[right],
                box_lengths,
                particle_mask=particle_mask,
                wave_number=wave_number,
            )
            rows.append({"frame_i": left, "frame_j": right, "fs": fs})
            if abs(fs) > maximum_absolute_fs:
                raise ValueError(
                    f"selected parent frames are not decorrelated: "
                    f"frames {left} and {right} have Fs={fs:.6g}"
                )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_lammps_data(
    path: Path,
    positions: np.ndarray,
    particle_types: np.ndarray,
    box_lengths: np.ndarray,
) -> None:
    with path.open("w") as handle:
        handle.write("Kob-Andersen 80:20 restart from an equilibrated public frame\n\n")
        handle.write(f"{len(positions)} atoms\n2 atom types\n\n")
        for axis, length in zip("xyz", box_lengths):
            handle.write(f"{-length / 2.0:.10f} {length / 2.0:.10f} {axis}lo {axis}hi\n")
        handle.write("\nMasses\n\n1 1.0\n2 1.0\n\nAtoms # atomic\n\n")
        for atom_id, (particle_type, coordinate) in enumerate(
            zip(particle_types, positions), start=1
        ):
            handle.write(
                f"{atom_id} {int(particle_type) + 1} "
                f"{coordinate[0]:.9f} {coordinate[1]:.9f} {coordinate[2]:.9f}\n"
            )


def _lammps_input(
    *,
    temperature: float,
    velocity_seed: int,
    equilibration_steps: int,
    production_steps: int,
) -> str:
    return f"""units lj
atom_style atomic
boundary p p p
read_data initial.data

pair_style lj/cut 2.5
pair_modify shift yes
pair_coeff 1 1 1.0 1.0 2.5
pair_coeff 1 2 1.5 0.8 2.0
pair_coeff 2 2 0.5 0.88 2.2
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes

velocity all create {temperature:g} {velocity_seed} mom yes rot no dist gaussian
fix thermostat all nvt temp {temperature:g} {temperature:g} 10
timestep 0.001
thermo 10000
thermo_style custom step time temp pe ke etotal press

run {equilibration_steps}
write_restart equilibrated.restart
reset_timestep 0

dump trajectory all custom 1000 trajectory.lammpstrj id type x y z ix iy iz
dump_modify trajectory sort id
restart 100000 restart.*
run {production_steps}
write_restart final.restart
"""


def prepare_replicate(
    source_path: Path,
    output_directory: Path,
    *,
    temperature: float,
    frame_index: int,
    velocity_seed: int,
    equilibration_time: float = 100.0,
    production_time: float = 5000.0,
    _payload: dict[str, object] | None = None,
    _source_sha256: str | None = None,
) -> dict[str, object]:
    """Create one restart directory and a machine-readable protocol manifest."""

    source_path = Path(source_path).resolve()
    output_directory = Path(output_directory).resolve()
    if not source_path.is_file():
        raise ValueError("source_path must be an existing pickle trajectory")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    if isinstance(velocity_seed, bool) or not isinstance(velocity_seed, int) or velocity_seed <= 0:
        raise ValueError("velocity_seed must be a positive integer")
    for name, value in (
        ("equilibration_time", equilibration_time),
        ("production_time", production_time),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    if _payload is None:
        with source_path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        payload = _payload
    box_lengths = np.asarray(payload["Box_size"][0], dtype=float)
    particle_types = np.asarray(payload["Particle_types"][0], dtype=int)
    if frame_index >= len(payload["Positions"]):
        raise ValueError("frame_index is outside the source trajectory")
    positions = np.asarray(payload["Positions"][frame_index], dtype=float)
    if positions.shape != (len(particle_types), 3) or box_lengths.shape != (3,):
        raise ValueError("source trajectory must contain three-dimensional particle frames")
    if set(np.unique(particle_types)) - {0, 1}:
        raise ValueError("particle types must use zero for A and one for B")

    timestep = 0.001
    equilibration_steps = int(round(equilibration_time / timestep))
    production_steps = int(round(production_time / timestep))
    if not math.isclose(equilibration_steps * timestep, equilibration_time):
        raise ValueError("equilibration_time must be an integer multiple of 0.001 tau")
    if not math.isclose(production_steps * timestep, production_time):
        raise ValueError("production_time must be an integer multiple of 0.001 tau")

    output_directory.mkdir(parents=True, exist_ok=False)
    _write_lammps_data(output_directory / "initial.data", positions, particle_types, box_lengths)
    (output_directory / "in.production").write_text(
        _lammps_input(
            temperature=temperature,
            velocity_seed=velocity_seed,
            equilibration_steps=equilibration_steps,
            production_steps=production_steps,
        )
    )

    a_count = int(np.sum(particle_types == 0))
    b_count = int(np.sum(particle_types == 1))
    manifest: dict[str, object] = {
        "source_doi": SOURCE_DOI,
        "source_path": str(source_path),
        "source_sha256": _source_sha256 or _sha256(source_path),
        "source_frame_index": frame_index,
        "temperature": temperature,
        "velocity_seed": velocity_seed,
        "particle_counts": {"A": a_count, "B": b_count, "total": a_count + b_count},
        "box_lengths": box_lengths.tolist(),
        "density": float(len(particle_types) / np.prod(box_lengths)),
        "ensemble": "NVT",
        "thermostat": "LAMMPS_fix_nvt_Nose_Hoover",
        "thermostat_coupling_tau": 10.0,
        "timestep_tau": timestep,
        "equilibration_time_tau": equilibration_time,
        "production_time_tau": production_time,
        "saved_frame_interval_tau": 1.0,
        "restart_interval_tau": 100.0,
        "potential": "standard_80_20_Kob_Andersen_LJ_shifted_at_2p5_sigma",
        "independence_class": "decorrelated_parent_frames_plus_velocity_seeds",
        "independently_prepared_parent_samples": False,
        "thermodynamic_claim_allowed": False,
    }
    temporary_manifest = output_directory / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(output_directory / "manifest.json")
    return manifest


def prepare_replicate_ensemble(
    source_path: Path,
    output_directory: Path,
    *,
    temperature: float,
    frame_indices: Sequence[int],
    velocity_seeds: Sequence[int],
    wave_number: float = 7.25,
    maximum_absolute_fs: float = 0.1,
    equilibration_time: float = 100.0,
    production_time: float = 5000.0,
) -> dict[str, object]:
    """Prepare a gated ensemble without repeatedly loading or hashing its parent."""

    source_path = Path(source_path).resolve()
    output_directory = Path(output_directory).resolve()
    frames = [int(value) for value in frame_indices]
    seeds = [int(value) for value in velocity_seeds]
    if len(frames) != len(seeds) or len(frames) < 2:
        raise ValueError("frame_indices and velocity_seeds must have the same length of at least two")
    if len(set(seeds)) != len(seeds):
        raise ValueError("velocity_seeds must be unique")
    if not source_path.is_file():
        raise ValueError("source_path must be an existing pickle trajectory")

    with source_path.open("rb") as handle:
        payload = pickle.load(handle)
    box_lengths = np.asarray(payload["Box_size"][0], dtype=float)
    particle_types = np.asarray(payload["Particle_types"][0], dtype=int)
    pairwise_fs = validate_initial_frame_independence(
        payload["Positions"],
        box_lengths,
        particle_types == 0,
        frame_indices=frames,
        wave_number=wave_number,
        maximum_absolute_fs=maximum_absolute_fs,
    )
    source_sha256 = _sha256(source_path)
    output_directory.mkdir(parents=True, exist_ok=False)

    replicate_manifests: list[dict[str, object]] = []
    for replicate_index, (frame_index, velocity_seed) in enumerate(zip(frames, seeds), start=1):
        replicate_directory = output_directory / f"replicate_{replicate_index:02d}"
        replicate_manifest = prepare_replicate(
            source_path,
            replicate_directory,
            temperature=temperature,
            frame_index=frame_index,
            velocity_seed=velocity_seed,
            equilibration_time=equilibration_time,
            production_time=production_time,
            _payload=payload,
            _source_sha256=source_sha256,
        )
        replicate_manifests.append(
            {
                "replicate": replicate_index,
                "directory": replicate_directory.name,
                "source_frame_index": frame_index,
                "velocity_seed": velocity_seed,
            }
        )

    manifest: dict[str, object] = {
        "source_doi": SOURCE_DOI,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "temperature": temperature,
        "replicate_count": len(frames),
        "independence_class": "decorrelated_parent_frames_plus_velocity_seeds",
        "independently_prepared_parent_samples": False,
        "independence_wave_number": wave_number,
        "maximum_absolute_fs_allowed": maximum_absolute_fs,
        "maximum_absolute_fs_observed": max(abs(float(row["fs"])) for row in pairwise_fs),
        "pairwise_initial_fs": pairwise_fs,
        "replicates": replicate_manifests,
        "equilibration_time_tau": equilibration_time,
        "production_time_tau": production_time,
        "thermodynamic_claim_allowed": False,
    }
    temporary_manifest = output_directory / "ensemble_manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(output_directory / "ensemble_manifest.json")
    return manifest
