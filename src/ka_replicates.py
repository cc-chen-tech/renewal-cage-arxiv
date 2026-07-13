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


def independent_group_ratio(
    numerator_values: np.ndarray,
    denominator_values: np.ndarray,
    *,
    relative_equivalence_margin: float,
) -> dict[str, float | str]:
    """Compare positive independent-group means on a conservative log-ratio scale."""

    numerator = np.asarray(numerator_values, dtype=float)
    denominator = np.asarray(denominator_values, dtype=float)
    if numerator.ndim != 1 or denominator.ndim != 1 or min(len(numerator), len(denominator)) < 2:
        raise ValueError("each independent group must be a vector with at least two values")
    if np.any(~np.isfinite(numerator)) or np.any(~np.isfinite(denominator)):
        raise ValueError("independent group values must be finite")
    if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
        raise ValueError("independent group values must be positive")
    if not math.isfinite(relative_equivalence_margin) or not 0.0 < relative_equivalence_margin < 1.0:
        raise ValueError("relative_equivalence_margin must lie between zero and one")
    numerator_mean = float(np.mean(numerator))
    denominator_mean = float(np.mean(denominator))
    numerator_se = float(np.std(numerator, ddof=1) / math.sqrt(len(numerator)))
    denominator_se = float(np.std(denominator, ddof=1) / math.sqrt(len(denominator)))
    log_ratio = math.log(numerator_mean / denominator_mean)
    log_standard_error = math.sqrt(
        (numerator_se / numerator_mean) ** 2
        + (denominator_se / denominator_mean) ** 2
    )
    numerator_critical = independent_sample_ci95(
        mean=0.0,
        standard_error=1.0,
        sample_count=len(numerator),
    )[2]
    denominator_critical = independent_sample_ci95(
        mean=0.0,
        standard_error=1.0,
        sample_count=len(denominator),
    )[2]
    critical = max(numerator_critical, denominator_critical)
    lower = math.exp(log_ratio - critical * log_standard_error)
    upper = math.exp(log_ratio + critical * log_standard_error)
    return {
        "numerator_mean": numerator_mean,
        "denominator_mean": denominator_mean,
        "mean_ratio": numerator_mean / denominator_mean,
        "log_ratio_standard_error": log_standard_error,
        "ci95_low_ratio": lower,
        "ci95_high_ratio": upper,
        "ci95_critical_value": critical,
        "ci95_method": "conservative_student_t_delta_log_ratio",
        "numerator_replicate_count": float(len(numerator)),
        "denominator_replicate_count": float(len(denominator)),
        "relative_equivalence_margin": relative_equivalence_margin,
        "growth_detected": float(lower > 1.0),
        "decrease_detected": float(upper < 1.0),
        "equivalent_to_unity": float(
            lower >= 1.0 - relative_equivalence_margin
            and upper <= 1.0 + relative_equivalence_margin
        ),
    }


def signed_temperature_separation(
    high_temperature_values: np.ndarray,
    low_temperature_values: np.ndarray,
) -> dict[str, float | str]:
    """Compare signed observables between independent high- and low-temperature replicas."""

    high = np.asarray(high_temperature_values, dtype=float)
    low = np.asarray(low_temperature_values, dtype=float)
    if high.ndim != 1 or low.ndim != 1 or min(len(high), len(low)) < 2:
        raise ValueError("each temperature group must be a vector with at least two values")
    if np.any(~np.isfinite(high)) or np.any(~np.isfinite(low)):
        raise ValueError("temperature-group values must be finite")

    def summarize(values: np.ndarray) -> tuple[float, float, float, float, float]:
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(values))
        ci_low, ci_high, critical = independent_sample_ci95(
            mean=mean,
            standard_error=standard_error,
            sample_count=len(values),
        )
        return mean, standard_error, ci_low, ci_high, critical

    high_mean, high_se, high_low, high_high, high_critical = summarize(high)
    low_mean, low_se, low_low, low_high, low_critical = summarize(low)
    return {
        "high_mean": high_mean,
        "high_standard_error": high_se,
        "high_ci95_low": high_low,
        "high_ci95_high": high_high,
        "low_mean": low_mean,
        "low_standard_error": low_se,
        "low_ci95_low": low_low,
        "low_ci95_high": low_high,
        "high_ci95_critical_value": high_critical,
        "low_ci95_critical_value": low_critical,
        "ci95_method": "student_t_independent_replicates",
        "high_replicate_count": float(len(high)),
        "low_replicate_count": float(len(low)),
        "confidence_intervals_separated": float(
            high_low > low_high or low_low > high_high
        ),
        "positive_high_negative_low_reversal": float(high_low > 0.0 and low_high < 0.0),
    }


def jump_vector_correlation_curve(
    events: dict[str, np.ndarray],
    *,
    maximum_lag: int,
) -> list[dict[str, float]]:
    """Resolve event-indexed jump-vector memory and cumulative Green-Kubo factors."""

    particles = np.asarray(events["particle"], dtype=int)
    times = np.asarray(events["time"], dtype=float)
    jumps = np.asarray(events["jump_vector"], dtype=float)
    if particles.ndim != 1 or times.shape != particles.shape:
        raise ValueError("event particles and times must be aligned vectors")
    if jumps.ndim != 2 or jumps.shape[0] != len(particles) or jumps.shape[1] < 1:
        raise ValueError("jump vectors must align with events")
    if len(particles) == 0 or np.any(~np.isfinite(times)) or np.any(~np.isfinite(jumps)):
        raise ValueError("event arrays must be nonempty and finite")
    if isinstance(maximum_lag, bool) or not isinstance(maximum_lag, int) or maximum_lag < 1:
        raise ValueError("maximum_lag must be a positive integer")
    order = np.lexsort((times, particles))
    particles = particles[order]
    jumps = jumps[order]
    jump_squared_mean = float(np.mean(np.sum(jumps**2, axis=1)))
    if jump_squared_mean <= 0.0:
        raise ValueError("mean squared jump length must be positive")
    cumulative = 0.0
    rows: list[dict[str, float]] = []
    for lag in range(1, maximum_lag + 1):
        values: list[float] = []
        for particle in np.unique(particles):
            particle_jumps = jumps[particles == particle]
            if len(particle_jumps) > lag:
                values.extend(
                    np.sum(particle_jumps[:-lag] * particle_jumps[lag:], axis=1).tolist()
                )
        if not values:
            raise ValueError("maximum_lag exceeds all supported particle event sequences")
        correlation = float(np.mean(values))
        cumulative += correlation
        rows.append(
            {
                "event_lag": float(lag),
                "pair_count": float(len(values)),
                "jump_squared_mean": jump_squared_mean,
                "jump_dot_correlation": correlation,
                "correlation_over_q": correlation / jump_squared_mean,
                "cumulative_green_kubo_factor": 1.0
                + 2.0 * cumulative / jump_squared_mean,
            }
        )
    return rows


def block_vector_correlation_curve(
    block_displacements: np.ndarray,
    *,
    maximum_lag: int,
) -> list[dict[str, float]]:
    """Measure vector memory between consecutive fixed-time displacement blocks."""

    displacements = np.asarray(block_displacements, dtype=float)
    if (
        displacements.ndim != 3
        or min(displacements.shape) < 1
        or np.any(~np.isfinite(displacements))
    ):
        raise ValueError("block_displacements must be a finite particle-block-vector array")
    if (
        isinstance(maximum_lag, bool)
        or not isinstance(maximum_lag, int)
        or maximum_lag < 1
        or maximum_lag >= displacements.shape[1]
    ):
        raise ValueError("maximum_lag must fit inside the block axis")
    block_msd = float(np.mean(np.sum(displacements**2, axis=2)))
    if block_msd <= 0.0:
        raise ValueError("block displacement variance must be positive")
    cumulative = 0.0
    rows: list[dict[str, float]] = []
    for lag in range(1, maximum_lag + 1):
        correlation = float(
            np.mean(
                np.sum(
                    displacements[:, :-lag] * displacements[:, lag:], axis=2
                )
            )
        )
        cumulative += correlation
        rows.append(
            {
                "block_lag": float(lag),
                "pair_count": float(
                    displacements.shape[0] * (displacements.shape[1] - lag)
                ),
                "block_vector_msd": block_msd,
                "block_dot_correlation": correlation,
                "correlation_over_block_msd": correlation / block_msd,
                "cumulative_green_kubo_factor": 1.0
                + 2.0 * cumulative / block_msd,
            }
        )
    return rows


def finite_window_green_kubo_factor(
    correlation_rows: Sequence[dict[str, object]],
    *,
    block_count: int,
) -> float:
    """Convert block-vector correlations to the exact finite-window MSD factor."""

    if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count < 1:
        raise ValueError("block_count must be a positive integer")
    if block_count == 1:
        return 1.0
    selected = [
        row
        for row in correlation_rows
        if 1 <= int(float(row["block_lag"])) < block_count
    ]
    if not selected:
        return 1.0
    block_msd_values = {float(row["block_vector_msd"]) for row in selected}
    if len(block_msd_values) != 1 or min(block_msd_values) <= 0.0:
        raise ValueError("correlation rows must share one positive block MSD")
    block_msd = block_msd_values.pop()
    correction = sum(
        (1.0 - float(row["block_lag"]) / block_count)
        * float(row["block_dot_correlation"])
        for row in selected
    )
    return 1.0 + 2.0 * correction / block_msd


def _validate_block_path_inputs(
    block_displacements: np.ndarray,
    block_count: int,
    wave_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    displacements = np.asarray(block_displacements, dtype=float)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if (
        displacements.ndim != 3
        or min(displacements.shape) < 1
        or np.any(~np.isfinite(displacements))
    ):
        raise ValueError("block displacements must be a finite particle-block-vector array")
    if (
        isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or not 1 <= block_count <= displacements.shape[1]
    ):
        raise ValueError("block_count must fit inside the block axis")
    if (
        wave_numbers.ndim != 1
        or len(wave_numbers) < 1
        or np.any(~np.isfinite(wave_numbers))
        or np.any(wave_numbers <= 0.0)
    ):
        raise ValueError("wave numbers must be a positive finite vector")
    return displacements, wave_numbers


def cumulative_block_observables(
    block_displacements: np.ndarray,
    *,
    block_count: int,
    wave_numbers: np.ndarray,
) -> dict[str, float]:
    """Measure cumulative observables over all contiguous particle-block windows."""

    displacements, wave_numbers = _validate_block_path_inputs(
        block_displacements,
        block_count,
        wave_numbers,
    )
    prefix = np.concatenate(
        [
            np.zeros((displacements.shape[0], 1, displacements.shape[2])),
            np.cumsum(displacements, axis=1),
        ],
        axis=1,
    )
    cumulative = prefix[:, block_count:] - prefix[:, :-block_count]
    vectors = cumulative.reshape(-1, displacements.shape[2])
    squared = np.sum(vectors**2, axis=1)
    msd = float(np.mean(squared))
    fourth = float(np.mean(squared**2))
    ngp = (
        displacements.shape[2]
        / (displacements.shape[2] + 2.0)
        * fourth
        / msd**2
        - 1.0
        if msd > 0.0
        else math.nan
    )
    result = {
        "block_count": float(block_count),
        "particle_window_count": float(len(vectors)),
        "msd": msd,
        "fourth_moment": fourth,
        "ngp": ngp,
    }
    for wave_number in wave_numbers:
        key = f"characteristic_k{wave_number:g}".replace(".", "p")
        result[key] = float(np.mean(np.cos(wave_number * vectors)))
    return result


def within_particle_time_shuffle(
    block_displacements: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Destroy block order while preserving each particle's vector multiset."""

    displacements = np.asarray(block_displacements, dtype=float)
    if (
        displacements.ndim != 3
        or min(displacements.shape) < 1
        or np.any(~np.isfinite(displacements))
        or not isinstance(rng, np.random.Generator)
    ):
        raise ValueError("a finite block array and NumPy generator are required")
    order = np.argsort(rng.random(displacements.shape[:2]), axis=1)
    return np.take_along_axis(displacements, order[:, :, np.newaxis], axis=1)


def direction_randomized_block_observables(
    block_displacements: np.ndarray,
    *,
    block_count: int,
    wave_numbers: np.ndarray,
) -> dict[str, float]:
    """Average ordered block lengths over independent isotropic 3D directions."""

    displacements, wave_numbers = _validate_block_path_inputs(
        block_displacements,
        block_count,
        wave_numbers,
    )
    if displacements.shape[2] != 3:
        raise ValueError("the analytic direction-randomized kernel requires 3D vectors")
    squared_lengths = np.sum(displacements**2, axis=2)
    fourth_lengths = squared_lengths**2

    def window_sum(values: np.ndarray) -> np.ndarray:
        prefix = np.concatenate(
            [np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)],
            axis=1,
        )
        return prefix[:, block_count:] - prefix[:, :-block_count]

    sum_second = window_sum(squared_lengths)
    sum_fourth = window_sum(fourth_lengths)
    fourth_windows = sum_fourth + (5.0 / 3.0) * (
        sum_second**2 - sum_fourth
    )
    msd = float(np.mean(sum_second))
    fourth = float(np.mean(fourth_windows))
    result = {
        "block_count": float(block_count),
        "particle_window_count": float(sum_second.size),
        "msd": msd,
        "fourth_moment": fourth,
        "ngp": 3.0 / 5.0 * fourth / msd**2 - 1.0 if msd > 0.0 else math.nan,
    }
    lengths = np.sqrt(squared_lengths)
    window_count = lengths.shape[1] - block_count + 1
    for wave_number in wave_numbers:
        characteristic = np.ones((lengths.shape[0], window_count))
        for offset in range(block_count):
            characteristic *= np.sinc(
                wave_number
                * lengths[:, offset : offset + window_count]
                / math.pi
            )
        key = f"characteristic_k{wave_number:g}".replace(".", "p")
        result[key] = float(np.mean(characteristic))
    return result


def particle_event_count_matrix(
    events: dict[str, np.ndarray],
    *,
    duration: float,
    particle_count: int,
    block_size: float,
) -> np.ndarray:
    """Bin particle events into complete, consecutive observation blocks."""

    particles = np.asarray(events["particle"], dtype=int)
    times = np.asarray(events["time"], dtype=float)
    if particles.ndim != 1 or times.shape != particles.shape:
        raise ValueError("event particles and times must be aligned vectors")
    if particle_count < 2 or not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("particle_count and duration must define a nontrivial ensemble")
    if not math.isfinite(block_size) or block_size <= 0.0 or block_size > duration / 2.0:
        raise ValueError("block_size must leave at least two complete blocks")
    if np.any(particles < 0) or np.any(particles >= particle_count):
        raise ValueError("event particle indices are out of range")
    if np.any(~np.isfinite(times)) or np.any(times < 0.0) or np.any(times > duration):
        raise ValueError("event times must lie in the observation interval")
    block_count = int(duration // block_size)
    block_index = np.floor(times / block_size).astype(int)
    retained = block_index < block_count
    counts = np.zeros((particle_count, block_count), dtype=int)
    np.add.at(counts, (particles[retained], block_index[retained]), 1.0)
    return counts


def particle_event_count_correlation_curve(
    events: dict[str, np.ndarray],
    *,
    duration: float,
    particle_count: int,
    block_size: float,
    maximum_lag: int,
) -> list[dict[str, float]]:
    """Measure persistence of particle mobility identity across event-count blocks."""

    if isinstance(maximum_lag, bool) or not isinstance(maximum_lag, int) or maximum_lag < 1:
        raise ValueError("maximum_lag must be a positive integer")
    counts = particle_event_count_matrix(
        events,
        duration=duration,
        particle_count=particle_count,
        block_size=block_size,
    )
    block_count = counts.shape[1]
    if maximum_lag >= block_count:
        raise ValueError("maximum_lag must be smaller than the complete block count")
    residual = counts - np.mean(counts, axis=0, keepdims=True)
    rows: list[dict[str, float]] = []
    for lag in range(1, maximum_lag + 1):
        left = residual[:, :-lag].ravel()
        right = residual[:, lag:].ravel()
        denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        correlation = float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0
        rows.append(
            {
                "block_lag": float(lag),
                "lag_time": float(lag * block_size),
                "block_size": float(block_size),
                "block_count": float(block_count),
                "paired_particle_block_count": float(len(left)),
                "mean_events_per_particle_block": float(np.mean(counts)),
                "particle_identity_correlation": correlation,
            }
        )
    return rows


def particle_event_count_cross_window_correlation(
    first_events: dict[str, np.ndarray],
    second_events: dict[str, np.ndarray],
    *,
    particle_count: int,
) -> dict[str, float]:
    """Correlate particle mobility identities between two disjoint windows."""

    if particle_count < 2:
        raise ValueError("particle_count must be at least two")
    first_particles = np.asarray(first_events["particle"], dtype=int)
    second_particles = np.asarray(second_events["particle"], dtype=int)
    if first_particles.ndim != 1 or second_particles.ndim != 1:
        raise ValueError("event particle arrays must be vectors")
    if (
        np.any(first_particles < 0)
        or np.any(first_particles >= particle_count)
        or np.any(second_particles < 0)
        or np.any(second_particles >= particle_count)
    ):
        raise ValueError("event particle indices are out of range")
    first_counts = np.bincount(first_particles, minlength=particle_count).astype(float)
    second_counts = np.bincount(second_particles, minlength=particle_count).astype(float)
    first_residual = first_counts - np.mean(first_counts)
    second_residual = second_counts - np.mean(second_counts)
    denominator = math.sqrt(
        float(np.dot(first_residual, first_residual) * np.dot(second_residual, second_residual))
    )
    correlation = (
        float(np.dot(first_residual, second_residual) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return {
        "particle_identity_correlation": correlation,
        "first_mean_event_count": float(np.mean(first_counts)),
        "second_mean_event_count": float(np.mean(second_counts)),
        "first_event_count_variance": float(np.var(first_counts)),
        "second_event_count_variance": float(np.var(second_counts)),
        "particle_count": float(particle_count),
    }


def correlation_efold_crossing(
    rows: Sequence[dict[str, object]],
    *,
    lag_key: str = "lag_time",
    correlation_key: str = "particle_identity_correlation",
) -> dict[str, float]:
    """Locate the first 1/e decay of a positive correlation curve."""

    if len(rows) < 2:
        raise ValueError("an e-fold crossing requires at least two curve rows")
    ordered = sorted(rows, key=lambda row: float(row[lag_key]))
    lags = np.array([float(row[lag_key]) for row in ordered])
    correlations = np.array([float(row[correlation_key]) for row in ordered])
    if (
        np.any(~np.isfinite(lags))
        or np.any(~np.isfinite(correlations))
        or np.any(lags <= 0.0)
        or np.any(np.diff(lags) <= 0.0)
        or correlations[0] <= 0.0
    ):
        raise ValueError("correlation curve must have increasing positive lags and a positive start")
    target = float(correlations[0] / math.e)
    candidates = np.flatnonzero(correlations <= target)
    if len(candidates) == 0:
        raise ValueError("correlation curve does not reach its 1/e target")
    upper_index = int(candidates[0])
    if upper_index == 0:
        crossing = float(lags[0])
        lower_lag = float(lags[0])
    else:
        lower_index = upper_index - 1
        lower_lag = float(lags[lower_index])
        if correlations[upper_index] <= 0.0:
            crossing = float(lags[upper_index])
        else:
            fraction = (
                math.log(target) - math.log(float(correlations[lower_index]))
            ) / (
                math.log(float(correlations[upper_index]))
                - math.log(float(correlations[lower_index]))
            )
            crossing = math.exp(
                math.log(float(lags[lower_index]))
                + fraction
                * (math.log(float(lags[upper_index])) - math.log(float(lags[lower_index])))
            )
    return {
        "initial_lag_time": float(lags[0]),
        "initial_correlation": float(correlations[0]),
        "target_correlation": target,
        "efold_crossing_time": crossing,
        "crossing_lower_lag": lower_lag,
        "crossing_upper_lag": float(lags[upper_index]),
    }


def fit_two_state_poisson_hmm(
    counts: np.ndarray,
    *,
    block_size: float,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> dict[str, float | bool]:
    """Fit a slow/fast Markov-modulated Poisson clock to count sequences."""

    observations = np.asarray(counts, dtype=float)
    if (
        observations.ndim != 2
        or min(observations.shape) < 2
        or np.any(~np.isfinite(observations))
        or np.any(observations < 0.0)
        or np.any(observations != np.floor(observations))
    ):
        raise ValueError("counts must be a matrix of nonnegative integer observations")
    if float(np.sum(observations)) <= 0.0:
        raise ValueError("counts must contain at least one event")
    if not math.isfinite(block_size) or block_size <= 0.0:
        raise ValueError("block_size must be positive and finite")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    integer_counts = observations.astype(int)
    log_factorial = np.array(
        [math.lgamma(value + 1.0) for value in range(int(np.max(integer_counts)) + 1)]
    )
    mean_count = float(np.mean(observations))
    means = np.array(
        [max(0.25 * mean_count, 1e-4), max(2.0 * mean_count, 0.25)],
        dtype=float,
    )
    transition = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    initial = np.array([0.5, 0.5], dtype=float)
    floor = 1e-12

    def expectation() -> tuple[np.ndarray, np.ndarray, float]:
        log_emission = np.empty(observations.shape + (2,), dtype=float)
        for state in range(2):
            log_emission[:, :, state] = (
                observations * math.log(means[state])
                - means[state]
                - log_factorial[integer_counts]
            )
        offsets = np.max(log_emission, axis=2)
        emission = np.exp(log_emission - offsets[:, :, None])
        alpha = np.empty_like(emission)
        scales = np.empty(observations.shape, dtype=float)
        alpha[:, 0] = initial * emission[:, 0]
        scales[:, 0] = np.sum(alpha[:, 0], axis=1)
        alpha[:, 0] /= scales[:, 0, None]
        for block in range(1, observations.shape[1]):
            alpha[:, block] = (alpha[:, block - 1] @ transition) * emission[:, block]
            scales[:, block] = np.sum(alpha[:, block], axis=1)
            alpha[:, block] /= scales[:, block, None]

        beta = np.ones_like(emission)
        for block in range(observations.shape[1] - 2, -1, -1):
            beta[:, block] = (
                (emission[:, block + 1] * beta[:, block + 1]) @ transition.T
            ) / scales[:, block + 1, None]
        posterior = alpha * beta
        posterior /= np.sum(posterior, axis=2, keepdims=True)

        transition_counts = np.zeros((2, 2), dtype=float)
        for block in range(observations.shape[1] - 1):
            pair = (
                alpha[:, block, :, None]
                * transition[None, :, :]
                * emission[:, block + 1, None, :]
                * beta[:, block + 1, None, :]
            )
            pair /= np.sum(pair, axis=(1, 2), keepdims=True)
            transition_counts += np.sum(pair, axis=0)
        log_likelihood = float(np.sum(np.log(scales) + offsets))
        return posterior, transition_counts, log_likelihood

    previous_log_likelihood = -math.inf
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        posterior, transition_counts, log_likelihood = expectation()
        if (
            math.isfinite(previous_log_likelihood)
            and abs(log_likelihood - previous_log_likelihood)
            <= tolerance * (1.0 + abs(previous_log_likelihood))
        ):
            converged = True
            break
        initial = np.maximum(np.mean(posterior[:, 0], axis=0), floor)
        initial /= np.sum(initial)
        transition = np.maximum(transition_counts, floor)
        transition /= np.sum(transition, axis=1, keepdims=True)
        weights = np.sum(posterior, axis=(0, 1))
        means = np.maximum(
            np.sum(posterior * observations[:, :, None], axis=(0, 1)) / weights,
            floor,
        )
        previous_log_likelihood = log_likelihood
    if not converged:
        _, _, log_likelihood = expectation()

    order = np.argsort(means)
    means = means[order]
    transition = transition[np.ix_(order, order)]
    initial = initial[order]
    slow_to_fast = float(transition[0, 1])
    fast_to_slow = float(transition[1, 0])
    exchange_eigenvalue = float(1.0 - slow_to_fast - fast_to_slow)
    exchange_time = (
        -block_size / math.log(abs(exchange_eigenvalue))
        if 0.0 < abs(exchange_eigenvalue) < 1.0
        else 0.0
    )
    stationary_slow = fast_to_slow / (slow_to_fast + fast_to_slow)
    single_poisson_log_likelihood = float(
        np.sum(
            observations * math.log(mean_count)
            - mean_count
            - log_factorial[integer_counts]
        )
    )
    return {
        "slow_mean_count": float(means[0]),
        "fast_mean_count": float(means[1]),
        "slow_to_fast_probability": slow_to_fast,
        "fast_to_slow_probability": fast_to_slow,
        "stationary_slow_probability": stationary_slow,
        "stationary_fast_probability": 1.0 - stationary_slow,
        "initial_slow_probability": float(initial[0]),
        "exchange_eigenvalue": exchange_eigenvalue,
        "exchange_time": float(exchange_time),
        "block_size": float(block_size),
        "log_likelihood": log_likelihood,
        "single_poisson_log_likelihood": single_poisson_log_likelihood,
        "log_likelihood_gain_per_observation": (
            log_likelihood - single_poisson_log_likelihood
        ) / observations.size,
        "iterations": float(iterations),
        "converged": converged,
    }


def two_state_poisson_hmm_count_predictions(
    fitted: dict[str, object],
    *,
    maximum_lag: int,
) -> list[dict[str, float]]:
    """Predict stationary count fluctuations and identity decay from a fitted clock."""

    if isinstance(maximum_lag, bool) or not isinstance(maximum_lag, int) or maximum_lag < 1:
        raise ValueError("maximum_lag must be a positive integer")
    slow_mean = float(fitted["slow_mean_count"])
    fast_mean = float(fitted["fast_mean_count"])
    slow_probability = float(fitted["stationary_slow_probability"])
    fast_probability = float(fitted["stationary_fast_probability"])
    eigenvalue = float(fitted["exchange_eigenvalue"])
    values = np.array(
        [slow_mean, fast_mean, slow_probability, fast_probability, eigenvalue]
    )
    if (
        np.any(~np.isfinite(values))
        or min(slow_mean, fast_mean) < 0.0
        or slow_mean > fast_mean
        or min(slow_probability, fast_probability) < 0.0
        or not math.isclose(slow_probability + fast_probability, 1.0, abs_tol=1e-8)
        or abs(eigenvalue) >= 1.0
    ):
        raise ValueError("fitted parameters do not define a stationary two-state clock")
    mean_count = slow_probability * slow_mean + fast_probability * fast_mean
    if mean_count <= 0.0:
        raise ValueError("fitted mean count must be positive")
    environment_variance = (
        slow_probability * fast_probability * (fast_mean - slow_mean) ** 2
    )
    count_variance = mean_count + environment_variance
    return [
        {
            "block_lag": float(lag),
            "predicted_mean_count": mean_count,
            "predicted_count_variance": count_variance,
            "predicted_fano_factor": count_variance / mean_count,
            "predicted_environment_variance": environment_variance,
            "predicted_particle_identity_correlation": (
                environment_variance * eigenvalue**lag / count_variance
            ),
        }
        for lag in range(1, maximum_lag + 1)
    ]


def score_two_state_poisson_hmm(
    counts: np.ndarray,
    fitted: dict[str, object],
) -> dict[str, float]:
    """Evaluate count sequences under fixed fitted two-state Poisson-HMM parameters."""

    observations = np.asarray(counts, dtype=float)
    if (
        observations.ndim != 2
        or min(observations.shape) < 2
        or np.any(~np.isfinite(observations))
        or np.any(observations < 0.0)
        or np.any(observations != np.floor(observations))
    ):
        raise ValueError("counts must be a matrix of nonnegative integer observations")
    means = np.array(
        [float(fitted["slow_mean_count"]), float(fitted["fast_mean_count"])],
        dtype=float,
    )
    transition = np.array(
        [
            [
                1.0 - float(fitted["slow_to_fast_probability"]),
                float(fitted["slow_to_fast_probability"]),
            ],
            [
                float(fitted["fast_to_slow_probability"]),
                1.0 - float(fitted["fast_to_slow_probability"]),
            ],
        ],
        dtype=float,
    )
    initial = np.array(
        [
            float(fitted["stationary_slow_probability"]),
            float(fitted["stationary_fast_probability"]),
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(means))
        or np.any(means <= 0.0)
        or np.any(~np.isfinite(transition))
        or np.any(transition <= 0.0)
        or np.any(transition >= 1.0)
        or np.any(~np.isfinite(initial))
        or np.any(initial <= 0.0)
        or not np.allclose(np.sum(transition, axis=1), 1.0)
        or not math.isclose(float(np.sum(initial)), 1.0, abs_tol=1e-8)
    ):
        raise ValueError("fitted parameters do not define a positive two-state Poisson HMM")

    integer_counts = observations.astype(int)
    log_factorial = np.array(
        [math.lgamma(value + 1.0) for value in range(int(np.max(integer_counts)) + 1)]
    )
    log_emission = np.empty(observations.shape + (2,), dtype=float)
    for state in range(2):
        log_emission[:, :, state] = (
            observations * math.log(means[state])
            - means[state]
            - log_factorial[integer_counts]
        )
    offsets = np.max(log_emission, axis=2)
    emission = np.exp(log_emission - offsets[:, :, None])
    alpha = initial * emission[:, 0]
    scales = np.sum(alpha, axis=1)
    log_likelihood = float(np.sum(np.log(scales) + offsets[:, 0]))
    alpha /= scales[:, None]
    for block in range(1, observations.shape[1]):
        alpha = (alpha @ transition) * emission[:, block]
        scales = np.sum(alpha, axis=1)
        log_likelihood += float(np.sum(np.log(scales) + offsets[:, block]))
        alpha /= scales[:, None]
    return {
        "log_likelihood": log_likelihood,
        "mean_log_likelihood": log_likelihood / observations.size,
        "observation_count": float(observations.size),
        "sequence_count": float(observations.shape[0]),
        "block_count": float(observations.shape[1]),
    }


def exponential_correlation_spectrum(
    times: np.ndarray,
    fitted: dict[str, object],
) -> np.ndarray:
    """Evaluate a one- or two-mode finite exponential relaxation spectrum."""

    values = np.asarray(times, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("times must be finite and nonnegative")
    fast_amplitude = float(fitted["fast_amplitude"])
    fast_time = float(fitted["fast_time"])
    slow_amplitude = float(fitted["slow_amplitude"])
    slow_time = float(fitted["slow_time"])
    parameters = np.array([fast_amplitude, fast_time, slow_amplitude, slow_time])
    if (
        np.any(~np.isfinite(parameters))
        or min(fast_amplitude, slow_amplitude) < 0.0
        or min(fast_time, slow_time) <= 0.0
        or fast_time > slow_time
    ):
        raise ValueError("fitted spectrum must have nonnegative amplitudes and ordered times")
    return (
        fast_amplitude * np.exp(-values / fast_time)
        + slow_amplitude * np.exp(-values / slow_time)
    )


def fit_exponential_correlation_spectrum(
    times: np.ndarray,
    correlations: np.ndarray,
    *,
    component_count: int,
    grid_size: int = 80,
) -> dict[str, float]:
    """Fit a nonnegative finite relaxation spectrum by deterministic grid search."""

    lag_times = np.asarray(times, dtype=float)
    observed = np.asarray(correlations, dtype=float)
    if (
        lag_times.ndim != 1
        or observed.shape != lag_times.shape
        or len(lag_times) < 3
        or np.any(~np.isfinite(lag_times))
        or np.any(~np.isfinite(observed))
        or np.any(lag_times <= 0.0)
        or np.any(np.diff(lag_times) <= 0.0)
    ):
        raise ValueError("times must be a strictly increasing positive vector aligned with correlations")
    if component_count not in (1, 2):
        raise ValueError("component_count must be one or two")
    if component_count == 2 and len(lag_times) < 5:
        raise ValueError("two-component spectra require at least five correlation points")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 20:
        raise ValueError("grid_size must be an integer of at least twenty")

    minimum_spacing = float(np.min(np.diff(np.concatenate(([0.0], lag_times)))))
    time_grid = np.geomspace(minimum_spacing / 4.0, float(lag_times[-1]) * 20.0, grid_size)
    basis = np.exp(-lag_times[:, None] / time_grid[None, :])
    best_rss = math.inf
    best_indices = (0, 0)
    best_amplitudes = np.zeros(2, dtype=float)

    def score_columns(columns: np.ndarray) -> tuple[float, np.ndarray]:
        unconstrained = np.linalg.lstsq(columns, observed, rcond=None)[0]
        candidates = [np.maximum(unconstrained, 0.0)]
        for index in range(columns.shape[1]):
            amplitudes = np.zeros(columns.shape[1], dtype=float)
            denominator = float(np.dot(columns[:, index], columns[:, index]))
            amplitudes[index] = max(float(np.dot(columns[:, index], observed)) / denominator, 0.0)
            candidates.append(amplitudes)
        best_local = min(
            candidates,
            key=lambda amplitudes: float(np.sum((observed - columns @ amplitudes) ** 2)),
        )
        residual = observed - columns @ best_local
        return float(np.dot(residual, residual)), best_local

    if component_count == 1:
        for index in range(grid_size):
            rss, amplitudes = score_columns(basis[:, index : index + 1])
            if rss < best_rss:
                best_rss = rss
                best_indices = (index, index)
                best_amplitudes = np.array([amplitudes[0], 0.0])
    else:
        for fast_index in range(grid_size - 1):
            for slow_index in range(fast_index + 1, grid_size):
                rss, amplitudes = score_columns(basis[:, [fast_index, slow_index]])
                if rss < best_rss:
                    best_rss = rss
                    best_indices = (fast_index, slow_index)
                    best_amplitudes = amplitudes

    fast_time = float(time_grid[best_indices[0]])
    slow_time = float(time_grid[best_indices[1]])
    parameter_count = 2 * component_count
    mean_squared_error = best_rss / len(observed)
    return {
        "component_count": float(component_count),
        "fast_amplitude": float(best_amplitudes[0]),
        "fast_time": fast_time,
        "slow_amplitude": float(best_amplitudes[1]),
        "slow_time": slow_time,
        "total_amplitude": float(np.sum(best_amplitudes)),
        "time_scale_ratio": slow_time / fast_time,
        "rss": best_rss,
        "rmse": math.sqrt(mean_squared_error),
        "bic": len(observed) * math.log(max(mean_squared_error, 1e-300))
        + parameter_count * math.log(len(observed)),
        "fit_point_count": float(len(observed)),
        "grid_size": float(grid_size),
        "minimum_candidate_time": float(time_grid[0]),
        "maximum_candidate_time": float(time_grid[-1]),
    }


def fit_anchored_exponential_correlation_spectrum(
    times: np.ndarray,
    correlations: np.ndarray,
    *,
    total_amplitude: float,
    component_count: int,
    grid_size: int = 80,
) -> dict[str, float]:
    """Fit finite relaxation times with zero-lag amplitude fixed by count Fano."""

    lag_times = np.asarray(times, dtype=float)
    observed = np.asarray(correlations, dtype=float)
    if (
        lag_times.ndim != 1
        or observed.shape != lag_times.shape
        or len(lag_times) < 3
        or np.any(~np.isfinite(lag_times))
        or np.any(~np.isfinite(observed))
        or np.any(lag_times <= 0.0)
        or np.any(np.diff(lag_times) <= 0.0)
    ):
        raise ValueError("times must be a strictly increasing positive vector aligned with correlations")
    if not math.isfinite(total_amplitude) or not 0.0 < total_amplitude < 1.0:
        raise ValueError("total_amplitude must lie strictly between zero and one")
    if component_count not in (1, 2):
        raise ValueError("component_count must be one or two")
    if component_count == 2 and len(lag_times) < 5:
        raise ValueError("two-component spectra require at least five correlation points")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 20:
        raise ValueError("grid_size must be an integer of at least twenty")

    minimum_spacing = float(np.min(np.diff(np.concatenate(([0.0], lag_times)))))
    time_grid = np.geomspace(minimum_spacing / 4.0, float(lag_times[-1]) * 20.0, grid_size)
    basis = np.exp(-lag_times[:, None] / time_grid[None, :])
    best_rss = math.inf
    best_indices = (0, 0)
    best_fast_fraction = 1.0
    if component_count == 1:
        for index in range(grid_size):
            residual = observed - total_amplitude * basis[:, index]
            rss = float(np.dot(residual, residual))
            if rss < best_rss:
                best_rss = rss
                best_indices = (index, index)
    else:
        for fast_index in range(grid_size - 1):
            fast = basis[:, fast_index]
            for slow_index in range(fast_index + 1, grid_size):
                slow = basis[:, slow_index]
                direction = total_amplitude * (fast - slow)
                denominator = float(np.dot(direction, direction))
                fraction = float(
                    np.clip(
                        np.dot(direction, observed - total_amplitude * slow) / denominator,
                        0.0,
                        1.0,
                    )
                )
                prediction = total_amplitude * (
                    fraction * fast + (1.0 - fraction) * slow
                )
                residual = observed - prediction
                rss = float(np.dot(residual, residual))
                if rss < best_rss:
                    best_rss = rss
                    best_indices = (fast_index, slow_index)
                    best_fast_fraction = fraction

    fast_time = float(time_grid[best_indices[0]])
    slow_time = float(time_grid[best_indices[1]])
    fast_amplitude = total_amplitude * best_fast_fraction
    slow_amplitude = total_amplitude - fast_amplitude
    parameter_count = 1 if component_count == 1 else 3
    mean_squared_error = best_rss / len(observed)
    return {
        "component_count": float(component_count),
        "fast_amplitude": fast_amplitude,
        "fast_amplitude_fraction": best_fast_fraction,
        "fast_time": fast_time,
        "slow_amplitude": slow_amplitude,
        "slow_amplitude_fraction": 1.0 - best_fast_fraction,
        "slow_time": slow_time,
        "total_amplitude": total_amplitude,
        "time_scale_ratio": slow_time / fast_time,
        "rss": best_rss,
        "rmse": math.sqrt(mean_squared_error),
        "bic": len(observed) * math.log(max(mean_squared_error, 1e-300))
        + parameter_count * math.log(len(observed)),
        "fit_point_count": float(len(observed)),
        "grid_size": float(grid_size),
        "minimum_candidate_time": float(time_grid[0]),
        "maximum_candidate_time": float(time_grid[-1]),
    }


def gamma_refresh_cox_parameters(
    *,
    mean_count: float,
    fano_factor: float,
    block_size: float,
    fitted_spectrum: dict[str, object],
) -> dict[str, float]:
    """Map an anchored finite spectrum to a positive gamma-refresh Cox clock."""

    values = np.array([mean_count, fano_factor, block_size], dtype=float)
    if np.any(~np.isfinite(values)) or mean_count <= 0.0 or fano_factor <= 1.0 or block_size <= 0.0:
        raise ValueError("mean, super-Poisson Fano factor, and block size must be positive")
    fast_amplitude = float(fitted_spectrum["fast_amplitude"])
    slow_amplitude = float(fitted_spectrum["slow_amplitude"])
    fast_time = float(fitted_spectrum["fast_time"])
    slow_time = float(fitted_spectrum["slow_time"])
    total_amplitude = fast_amplitude + slow_amplitude
    fano_amplitude = 1.0 - 1.0 / fano_factor
    if (
        min(fast_amplitude, slow_amplitude) < 0.0
        or min(fast_time, slow_time) <= 0.0
        or not math.isclose(total_amplitude, fano_amplitude, rel_tol=1e-8, abs_tol=1e-10)
    ):
        raise ValueError("spectrum amplitude must be anchored by the supplied Fano factor")

    environment_variance = mean_count * (fano_factor - 1.0)
    fast_variance = environment_variance * fast_amplitude / total_amplitude
    slow_variance = environment_variance * slow_amplitude / total_amplitude
    standard_deviation_sum = math.sqrt(fast_variance) + math.sqrt(slow_variance)
    fast_mean = mean_count * math.sqrt(fast_variance) / standard_deviation_sum
    slow_mean = mean_count - fast_mean

    def gamma_parameters(mean: float, variance: float) -> tuple[float, float]:
        if variance == 0.0:
            return 0.0, 0.0
        return mean * mean / variance, variance / mean

    fast_shape, fast_scale = gamma_parameters(fast_mean, fast_variance)
    slow_shape, slow_scale = gamma_parameters(slow_mean, slow_variance)
    return {
        "mean_count": mean_count,
        "count_variance": mean_count * fano_factor,
        "fano_factor": fano_factor,
        "block_size": block_size,
        "fast_amplitude": fast_amplitude,
        "fast_time": fast_time,
        "fast_retention_probability": math.exp(-block_size / fast_time),
        "fast_intensity_mean": fast_mean,
        "fast_intensity_variance": fast_variance,
        "fast_gamma_shape": fast_shape,
        "fast_gamma_scale": fast_scale,
        "slow_amplitude": slow_amplitude,
        "slow_time": slow_time,
        "slow_retention_probability": math.exp(-block_size / slow_time),
        "slow_intensity_mean": slow_mean,
        "slow_intensity_variance": slow_variance,
        "slow_gamma_shape": slow_shape,
        "slow_gamma_scale": slow_scale,
        "total_amplitude": total_amplitude,
    }


def gamma_refresh_cox_count_predictions(
    parameters: dict[str, object],
    *,
    maximum_lag: int,
) -> list[dict[str, float]]:
    """Return exact stationary count moments for a gamma-refresh Cox clock."""

    if isinstance(maximum_lag, bool) or not isinstance(maximum_lag, int) or maximum_lag < 1:
        raise ValueError("maximum_lag must be a positive integer")
    fast_amplitude = float(parameters["fast_amplitude"])
    slow_amplitude = float(parameters["slow_amplitude"])
    fast_retention = float(parameters["fast_retention_probability"])
    slow_retention = float(parameters["slow_retention_probability"])
    return [
        {
            "block_lag": float(lag),
            "lag_time": lag * float(parameters["block_size"]),
            "predicted_mean_count": float(parameters["mean_count"]),
            "predicted_count_variance": float(parameters["count_variance"]),
            "predicted_fano_factor": float(parameters["fano_factor"]),
            "predicted_identity_correlation": (
                fast_amplitude * fast_retention**lag
                + slow_amplitude * slow_retention**lag
            ),
        }
        for lag in range(1, maximum_lag + 1)
    ]


def gamma_refresh_cox_count_pmf(
    parameters: dict[str, object],
    *,
    maximum_count: int,
) -> np.ndarray:
    """Exact stationary marginal count PMF from convolved Poisson-gamma channels."""

    if isinstance(maximum_count, bool) or not isinstance(maximum_count, int) or maximum_count < 0:
        raise ValueError("maximum_count must be a nonnegative integer")

    def channel_pmf(prefix: str) -> np.ndarray:
        variance = float(parameters[f"{prefix}_intensity_variance"])
        values = np.zeros(maximum_count + 1, dtype=float)
        if variance == 0.0:
            values[0] = 1.0
            return values
        shape = float(parameters[f"{prefix}_gamma_shape"])
        scale = float(parameters[f"{prefix}_gamma_scale"])
        ratio = scale / (1.0 + scale)
        values[0] = (1.0 + scale) ** (-shape)
        for count in range(1, maximum_count + 1):
            values[count] = (
                values[count - 1]
                * (count - 1.0 + shape)
                / count
                * ratio
            )
        return values

    fast = channel_pmf("fast")
    slow = channel_pmf("slow")
    return np.convolve(fast, slow)[: maximum_count + 1]


def gamma_refresh_cox_pair_pmf(
    parameters: dict[str, object],
    *,
    maximum_count: int,
    block_lag: int,
) -> np.ndarray:
    """Exact joint PMF of two counts separated by a finite block lag."""

    if isinstance(maximum_count, bool) or not isinstance(maximum_count, int) or maximum_count < 0:
        raise ValueError("maximum_count must be a nonnegative integer")
    if isinstance(block_lag, bool) or not isinstance(block_lag, int) or block_lag < 1:
        raise ValueError("block_lag must be a positive integer")

    def channel_pair(prefix: str) -> np.ndarray:
        variance = float(parameters[f"{prefix}_intensity_variance"])
        if variance == 0.0:
            result = np.zeros((maximum_count + 1, maximum_count + 1), dtype=float)
            result[0, 0] = 1.0
            return result
        shape = float(parameters[f"{prefix}_gamma_shape"])
        scale = float(parameters[f"{prefix}_gamma_scale"])
        marginal = np.zeros(maximum_count + 1, dtype=float)
        marginal[0] = (1.0 + scale) ** (-shape)
        ratio = scale / (1.0 + scale)
        for count in range(1, maximum_count + 1):
            marginal[count] = (
                marginal[count - 1]
                * (count - 1.0 + shape)
                / count
                * ratio
            )
        shared = np.empty((maximum_count + 1, maximum_count + 1), dtype=float)
        for first in range(maximum_count + 1):
            for second in range(maximum_count + 1):
                shared[first, second] = math.exp(
                    math.lgamma(shape + first + second)
                    - math.lgamma(shape)
                    - math.lgamma(first + 1.0)
                    - math.lgamma(second + 1.0)
                    + (first + second) * math.log(scale)
                    - (shape + first + second) * math.log1p(2.0 * scale)
                )
        retained = float(parameters[f"{prefix}_retention_probability"]) ** block_lag
        return retained * shared + (1.0 - retained) * np.outer(marginal, marginal)

    fast = channel_pair("fast")
    slow = channel_pair("slow")
    result = np.zeros(
        (2 * maximum_count + 1, 2 * maximum_count + 1),
        dtype=float,
    )
    for first in range(maximum_count + 1):
        for second in range(maximum_count + 1):
            result[
                first : first + maximum_count + 1,
                second : second + maximum_count + 1,
            ] += fast[first, second] * slow
    return result[: maximum_count + 1, : maximum_count + 1]


def simulate_gamma_refresh_cox_counts(
    parameters: dict[str, object],
    *,
    sequence_count: int,
    block_count: int,
    random_seed: int,
) -> np.ndarray:
    """Sample stationary event counts from two positive finite-refresh channels."""

    if min(sequence_count, block_count) < 2:
        raise ValueError("simulation requires at least two sequences and blocks")
    rng = np.random.default_rng(random_seed)

    def initialize(prefix: str) -> np.ndarray:
        variance = float(parameters[f"{prefix}_intensity_variance"])
        if variance == 0.0:
            return np.zeros(sequence_count, dtype=float)
        return rng.gamma(
            shape=float(parameters[f"{prefix}_gamma_shape"]),
            scale=float(parameters[f"{prefix}_gamma_scale"]),
            size=sequence_count,
        )

    fast = initialize("fast")
    slow = initialize("slow")
    counts = np.empty((sequence_count, block_count), dtype=int)
    for block in range(block_count):
        counts[:, block] = rng.poisson(fast + slow)
        if block == block_count - 1:
            continue
        for prefix, channel in (("fast", fast), ("slow", slow)):
            refresh = rng.random(sequence_count) > float(
                parameters[f"{prefix}_retention_probability"]
            )
            if np.any(refresh) and float(parameters[f"{prefix}_intensity_variance"]) > 0.0:
                channel[refresh] = rng.gamma(
                    shape=float(parameters[f"{prefix}_gamma_shape"]),
                    scale=float(parameters[f"{prefix}_gamma_scale"]),
                    size=int(np.sum(refresh)),
                )
    return counts


def two_clock_hmm_mixture_parameters(
    hmm_fitted: dict[str, object],
    fitted_spectrum: dict[str, object],
    *,
    block_size: float,
) -> dict[str, float]:
    """Combine HMM emissions with two finite exchange-rate classes."""

    slow_mean = float(hmm_fitted["slow_mean_count"])
    fast_mean = float(hmm_fitted["fast_mean_count"])
    slow_probability = float(hmm_fitted["stationary_slow_probability"])
    fast_probability = float(hmm_fitted["stationary_fast_probability"])
    fast_clock_weight = float(fitted_spectrum["fast_amplitude_fraction"])
    fast_time = float(fitted_spectrum["fast_time"])
    slow_time = float(fitted_spectrum["slow_time"])
    values = np.array(
        [
            slow_mean,
            fast_mean,
            slow_probability,
            fast_probability,
            fast_clock_weight,
            fast_time,
            slow_time,
            block_size,
        ]
    )
    if (
        np.any(~np.isfinite(values))
        or min(slow_mean, fast_mean) < 0.0
        or slow_mean > fast_mean
        or min(slow_probability, fast_probability) <= 0.0
        or not math.isclose(slow_probability + fast_probability, 1.0, abs_tol=1e-8)
        or not 0.0 <= fast_clock_weight <= 1.0
        or min(fast_time, slow_time, block_size) <= 0.0
        or fast_time > slow_time
    ):
        raise ValueError("HMM emissions and exchange spectrum must define two finite clocks")
    mean_count = slow_probability * slow_mean + fast_probability * fast_mean
    environment_variance = (
        slow_probability * fast_probability * (fast_mean - slow_mean) ** 2
    )
    count_variance = mean_count + environment_variance
    return {
        "slow_mean_count": slow_mean,
        "fast_mean_count": fast_mean,
        "stationary_slow_probability": slow_probability,
        "stationary_fast_probability": fast_probability,
        "fast_clock_weight": fast_clock_weight,
        "slow_clock_weight": 1.0 - fast_clock_weight,
        "fast_clock_time": fast_time,
        "slow_clock_time": slow_time,
        "fast_clock_retention": math.exp(-block_size / fast_time),
        "slow_clock_retention": math.exp(-block_size / slow_time),
        "block_size": block_size,
        "mean_count": mean_count,
        "environment_variance": environment_variance,
        "count_variance": count_variance,
        "fano_factor": count_variance / mean_count,
        "zero_lag_identity_amplitude": environment_variance / count_variance,
    }


def two_clock_hmm_mixture_count_predictions(
    parameters: dict[str, object],
    *,
    maximum_lag: int,
) -> list[dict[str, float]]:
    """Exact stationary moments of a two-exchange-rate HMM mixture."""

    if isinstance(maximum_lag, bool) or not isinstance(maximum_lag, int) or maximum_lag < 1:
        raise ValueError("maximum_lag must be a positive integer")
    return [
        {
            "block_lag": float(lag),
            "lag_time": lag * float(parameters["block_size"]),
            "predicted_mean_count": float(parameters["mean_count"]),
            "predicted_count_variance": float(parameters["count_variance"]),
            "predicted_fano_factor": float(parameters["fano_factor"]),
            "predicted_identity_correlation": float(
                parameters["zero_lag_identity_amplitude"]
            )
            * (
                float(parameters["fast_clock_weight"])
                * float(parameters["fast_clock_retention"]) ** lag
                + float(parameters["slow_clock_weight"])
                * float(parameters["slow_clock_retention"]) ** lag
            ),
        }
        for lag in range(1, maximum_lag + 1)
    ]


def two_clock_hmm_mixture_total_count_statistics(
    parameters: dict[str, object],
    *,
    block_count: int,
    pgf_argument: float,
) -> dict[str, float]:
    """Propagate the two-clock HMM through a finite cumulative count window."""

    if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count < 1:
        raise ValueError("block_count must be a positive integer")
    if not math.isfinite(pgf_argument):
        raise ValueError("pgf_argument must be finite")
    stationary = np.array(
        [
            float(parameters["stationary_slow_probability"]),
            float(parameters["stationary_fast_probability"]),
        ],
        dtype=float,
    )
    means = np.array(
        [
            float(parameters["slow_mean_count"]),
            float(parameters["fast_mean_count"]),
        ],
        dtype=float,
    )
    clock_weights = np.array(
        [
            float(parameters["fast_clock_weight"]),
            float(parameters["slow_clock_weight"]),
        ],
        dtype=float,
    )
    retentions = np.array(
        [
            float(parameters["fast_clock_retention"]),
            float(parameters["slow_clock_retention"]),
        ],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(stationary))
        or np.any(~np.isfinite(means))
        or np.any(~np.isfinite(clock_weights))
        or np.any(~np.isfinite(retentions))
        or np.any(stationary <= 0.0)
        or not math.isclose(float(np.sum(stationary)), 1.0, abs_tol=1e-8)
        or np.any(means < 0.0)
        or np.any(clock_weights < 0.0)
        or not math.isclose(float(np.sum(clock_weights)), 1.0, abs_tol=1e-8)
        or np.any(retentions < 0.0)
        or np.any(retentions > 1.0)
    ):
        raise ValueError("parameters must define stationary positive two-clock emissions")

    def propagate(retention: float) -> tuple[float, float, float]:
        transition = np.array(
            [
                [
                    1.0 - stationary[1] * (1.0 - retention),
                    stationary[1] * (1.0 - retention),
                ],
                [
                    stationary[0] * (1.0 - retention),
                    1.0 - stationary[0] * (1.0 - retention),
                ],
            ]
        )
        probability = stationary.copy()
        first = np.zeros(2, dtype=float)
        second = np.zeros(2, dtype=float)
        pgf = stationary.copy()
        emission_pgf = np.exp(means * (pgf_argument - 1.0))
        for _ in range(block_count):
            second = (second + 2.0 * first * means + probability * means**2) @ transition
            first = (first + probability * means) @ transition
            probability = probability @ transition
            pgf = (pgf * emission_pgf) @ transition
        return float(np.sum(first)), float(np.sum(second)), float(np.sum(pgf))

    values = [propagate(float(retention)) for retention in retentions]
    mean_count = float(sum(weight * value[0] for weight, value in zip(clock_weights, values)))
    factorial_second = float(
        sum(weight * value[1] for weight, value in zip(clock_weights, values))
    )
    count_pgf = float(sum(weight * value[2] for weight, value in zip(clock_weights, values)))
    return {
        "block_count": float(block_count),
        "lag_time": block_count * float(parameters["block_size"]),
        "mean_count": mean_count,
        "factorial_second_count": factorial_second,
        "count_variance": factorial_second + mean_count - mean_count**2,
        "pgf_argument": float(pgf_argument),
        "count_pgf": count_pgf,
    }


def two_clock_hmm_mixture_total_count_pmf(
    parameters: dict[str, object],
    *,
    block_count: int,
    maximum_count: int,
) -> dict[str, object]:
    """Return the exact truncated cumulative-count PMF of the two-clock HMM."""

    if isinstance(maximum_count, bool) or not isinstance(maximum_count, int) or maximum_count < 0:
        raise ValueError("maximum_count must be a nonnegative integer")
    two_clock_hmm_mixture_total_count_statistics(
        parameters,
        block_count=block_count,
        pgf_argument=1.0,
    )
    stationary = np.array(
        [
            float(parameters["stationary_slow_probability"]),
            float(parameters["stationary_fast_probability"]),
        ]
    )
    means = np.array(
        [
            float(parameters["slow_mean_count"]),
            float(parameters["fast_mean_count"]),
        ]
    )

    def poisson(mean: float) -> np.ndarray:
        values = np.zeros(maximum_count + 1, dtype=float)
        values[0] = math.exp(-mean)
        for count in range(1, maximum_count + 1):
            values[count] = values[count - 1] * mean / count
        return values

    emissions = [poisson(float(mean)) for mean in means]

    def propagate(retention: float) -> np.ndarray:
        transition = np.array(
            [
                [
                    1.0 - stationary[1] * (1.0 - retention),
                    stationary[1] * (1.0 - retention),
                ],
                [
                    stationary[0] * (1.0 - retention),
                    1.0 - stationary[0] * (1.0 - retention),
                ],
            ]
        )
        state_count = np.zeros((maximum_count + 1, 2), dtype=float)
        state_count[0] = stationary
        for _ in range(block_count):
            emitted = np.column_stack(
                [
                    np.convolve(state_count[:, state], emissions[state])[
                        : maximum_count + 1
                    ]
                    for state in range(2)
                ]
            )
            state_count = emitted @ transition
        return np.sum(state_count, axis=1)

    pmf = (
        float(parameters["fast_clock_weight"])
        * propagate(float(parameters["fast_clock_retention"]))
        + float(parameters["slow_clock_weight"])
        * propagate(float(parameters["slow_clock_retention"]))
    )
    return {
        "block_count": float(block_count),
        "lag_time": block_count * float(parameters["block_size"]),
        "maximum_count": float(maximum_count),
        "count_pmf": pmf,
        "tail_probability": max(0.0, 1.0 - float(np.sum(pmf))),
    }


def two_clock_hmm_mixture_pair_pmf(
    parameters: dict[str, object],
    *,
    maximum_count: int,
    block_lag: int,
) -> np.ndarray:
    """Exact joint count PMF for the two-exchange-rate HMM mixture."""

    if isinstance(maximum_count, bool) or not isinstance(maximum_count, int) or maximum_count < 0:
        raise ValueError("maximum_count must be a nonnegative integer")
    if isinstance(block_lag, bool) or not isinstance(block_lag, int) or block_lag < 1:
        raise ValueError("block_lag must be a positive integer")
    stationary = np.array(
        [
            float(parameters["stationary_slow_probability"]),
            float(parameters["stationary_fast_probability"]),
        ]
    )

    def poisson(mean: float) -> np.ndarray:
        values = np.zeros(maximum_count + 1, dtype=float)
        values[0] = math.exp(-mean)
        for count in range(1, maximum_count + 1):
            values[count] = values[count - 1] * mean / count
        return values

    emissions = [
        poisson(float(parameters["slow_mean_count"])),
        poisson(float(parameters["fast_mean_count"])),
    ]

    def clock_pair(retention: float) -> np.ndarray:
        lag_retention = retention**block_lag
        transition = np.array(
            [
                [
                    1.0 - stationary[1] * (1.0 - lag_retention),
                    stationary[1] * (1.0 - lag_retention),
                ],
                [
                    stationary[0] * (1.0 - lag_retention),
                    1.0 - stationary[0] * (1.0 - lag_retention),
                ],
            ]
        )
        return sum(
            stationary[first]
            * transition[first, second]
            * np.outer(emissions[first], emissions[second])
            for first in range(2)
            for second in range(2)
        )

    return (
        float(parameters["fast_clock_weight"])
        * clock_pair(float(parameters["fast_clock_retention"]))
        + float(parameters["slow_clock_weight"])
        * clock_pair(float(parameters["slow_clock_retention"]))
    )


def simulate_two_clock_hmm_mixture_counts(
    parameters: dict[str, object],
    *,
    sequence_count: int,
    block_count: int,
    random_seed: int,
) -> np.ndarray:
    """Sample count sequences from HMM emissions with two finite exchange rates."""

    if min(sequence_count, block_count) < 2:
        raise ValueError("simulation requires at least two sequences and blocks")
    rng = np.random.default_rng(random_seed)
    fast_clock = rng.random(sequence_count) < float(parameters["fast_clock_weight"])
    retention = np.where(
        fast_clock,
        float(parameters["fast_clock_retention"]),
        float(parameters["slow_clock_retention"]),
    )
    slow_probability = float(parameters["stationary_slow_probability"])
    fast_probability = float(parameters["stationary_fast_probability"])
    state = (rng.random(sequence_count) >= slow_probability).astype(int)
    means = np.array(
        [float(parameters["slow_mean_count"]), float(parameters["fast_mean_count"])]
    )
    counts = np.empty((sequence_count, block_count), dtype=int)
    for block in range(block_count):
        counts[:, block] = rng.poisson(means[state])
        if block == block_count - 1:
            continue
        uniforms = rng.random(sequence_count)
        slow_to_fast = fast_probability * (1.0 - retention)
        fast_to_slow = slow_probability * (1.0 - retention)
        state = np.where(
            state == 0,
            uniforms < slow_to_fast,
            uniforms >= fast_to_slow,
        ).astype(int)
    return counts


def event_cumulative_trajectory(
    events: dict[str, np.ndarray],
    *,
    frame_count: int,
    particle_count: int,
    dimension: int,
) -> np.ndarray:
    """Construct the event-space path obtained by accumulating jump vectors."""

    particles = np.asarray(events["particle"], dtype=int)
    times = np.asarray(events["time"], dtype=int)
    jumps = np.asarray(events["jump_vector"], dtype=float)
    if frame_count < 1 or particle_count < 1 or dimension < 1:
        raise ValueError("trajectory dimensions must be positive")
    if particles.ndim != 1 or times.shape != particles.shape:
        raise ValueError("event particles and times must be aligned vectors")
    if jumps.shape != (len(particles), dimension):
        raise ValueError("jump vectors must align with events and dimension")
    if (
        np.any(particles < 0)
        or np.any(particles >= particle_count)
        or np.any(times < 0)
        or np.any(times >= frame_count)
        or np.any(~np.isfinite(jumps))
    ):
        raise ValueError("event arrays contain out-of-range or nonfinite values")
    increments = np.zeros((frame_count, particle_count, dimension), dtype=float)
    np.add.at(increments, (times, particles), jumps)
    return np.cumsum(increments, axis=0)


def independent_isotropic_channel_moments(
    *,
    first_msd: float,
    first_ngp: float,
    second_msd: float,
    second_ngp: float,
    dimension: int,
) -> dict[str, float]:
    """Convolve second and fourth moments of independent isotropic displacement channels."""

    values = (first_msd, first_ngp, second_msd, second_ngp)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("channel moments must be finite")
    if first_msd < 0.0 or second_msd < 0.0:
        raise ValueError("channel MSD values must be nonnegative")
    if first_ngp < -1.0 or second_ngp < -1.0:
        raise ValueError("channel NGP values cannot be smaller than minus one")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    prefactor = (dimension + 2.0) / dimension
    first_fourth = prefactor * (1.0 + first_ngp) * first_msd**2
    second_fourth = prefactor * (1.0 + second_ngp) * second_msd**2
    cross = 2.0 * (1.0 + 2.0 / dimension) * first_msd * second_msd
    combined_msd = first_msd + second_msd
    combined_fourth = first_fourth + second_fourth + cross
    combined_ngp = (
        dimension
        * combined_fourth
        / ((dimension + 2.0) * combined_msd**2)
        - 1.0
        if combined_msd > 0.0
        else 0.0
    )
    return {
        "first_fourth_moment": first_fourth,
        "second_fourth_moment": second_fourth,
        "cross_fourth_moment": cross,
        "combined_msd": combined_msd,
        "combined_fourth_moment": combined_fourth,
        "combined_ngp": combined_ngp,
    }


def compound_jump_cage_observables(
    *,
    mean_count: float,
    factorial_second_count: float,
    jump_msd: float,
    jump_fourth_moment: float,
    count_pgf: float,
    cage_msd: float,
    cage_ngp: float,
    cage_fs: float,
    dimension: int,
) -> dict[str, float]:
    """Map a count law and iid isotropic jumps to cage-convolved observables."""

    values = (
        mean_count,
        factorial_second_count,
        jump_msd,
        jump_fourth_moment,
        count_pgf,
        cage_msd,
        cage_ngp,
        cage_fs,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("count, jump, and cage inputs must be finite")
    if (
        min(mean_count, factorial_second_count, jump_msd, jump_fourth_moment, cage_msd) < 0.0
        or cage_ngp < -1.0
        or not -1.0 <= count_pgf <= 1.0
        or not -1.0 <= cage_fs <= 1.0
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 1
    ):
        raise ValueError("count, jump, cage, and dimension inputs are outside their domains")
    jump_channel_msd = mean_count * jump_msd
    jump_channel_fourth = (
        mean_count * jump_fourth_moment
        + (1.0 + 2.0 / dimension)
        * factorial_second_count
        * jump_msd**2
    )
    jump_channel_ngp = (
        dimension
        * jump_channel_fourth
        / ((dimension + 2.0) * jump_channel_msd**2)
        - 1.0
        if jump_channel_msd > 0.0
        else 0.0
    )
    combined = independent_isotropic_channel_moments(
        first_msd=jump_channel_msd,
        first_ngp=jump_channel_ngp,
        second_msd=cage_msd,
        second_ngp=cage_ngp,
        dimension=dimension,
    )
    return {
        "jump_channel_msd": jump_channel_msd,
        "jump_channel_fourth_moment": jump_channel_fourth,
        "jump_channel_ngp": jump_channel_ngp,
        "factorized_msd": combined["combined_msd"],
        "factorized_fourth_moment": combined["combined_fourth_moment"],
        "factorized_ngp": combined["combined_ngp"],
        "factorized_fs": count_pgf * cage_fs,
    }


def green_kubo_renormalized_jump_statistics(
    jump_vectors: np.ndarray,
    *,
    green_kubo_factor: float,
    wave_numbers: np.ndarray,
) -> dict[str, float]:
    """Coarse-grain directional jump memory into calibration-measured moments."""

    jumps = np.asarray(jump_vectors, dtype=float)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if (
        jumps.ndim != 2
        or len(jumps) == 0
        or jumps.shape[1] < 1
        or np.any(~np.isfinite(jumps))
    ):
        raise ValueError("jump_vectors must be a nonempty finite matrix")
    if (
        not math.isfinite(green_kubo_factor)
        or green_kubo_factor <= 0.0
        or wave_numbers.ndim != 1
        or len(wave_numbers) == 0
        or np.any(~np.isfinite(wave_numbers))
        or np.any(wave_numbers <= 0.0)
    ):
        raise ValueError("green_kubo_factor and wave_numbers must be positive and finite")
    raw_squared = np.sum(jumps**2, axis=1)
    effective = math.sqrt(green_kubo_factor) * jumps
    effective_squared = np.sum(effective**2, axis=1)
    result = {
        "green_kubo_factor": float(green_kubo_factor),
        "raw_jump_msd": float(np.mean(raw_squared)),
        "raw_jump_fourth_moment": float(np.mean(raw_squared**2)),
        "effective_jump_msd": float(np.mean(effective_squared)),
        "effective_jump_fourth_moment": float(np.mean(effective_squared**2)),
    }
    for wave_number in wave_numbers:
        key = f"jump_characteristic_k{wave_number:g}".replace(".", "p")
        result[key] = float(np.mean(np.cos(wave_number * effective)))
    return result


def correlated_jump_propagator(
    events: dict[str, np.ndarray],
    *,
    maximum_count: int,
    wave_numbers: np.ndarray,
    minimum_sample_count: int,
) -> list[dict[str, float]]:
    """Estimate conditional net-displacement kernels for consecutive jumps."""

    particles = np.asarray(events["particle"], dtype=int)
    times = np.asarray(events["time"], dtype=float)
    jumps = np.asarray(events["jump_vector"], dtype=float)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if (
        particles.ndim != 1
        or times.shape != particles.shape
        or jumps.ndim != 2
        or jumps.shape[0] != len(particles)
        or jumps.shape[1] < 1
        or len(particles) == 0
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(jumps))
    ):
        raise ValueError("event particles, times, and jump vectors must be aligned and finite")
    if (
        isinstance(maximum_count, bool)
        or not isinstance(maximum_count, int)
        or maximum_count < 1
        or isinstance(minimum_sample_count, bool)
        or not isinstance(minimum_sample_count, int)
        or minimum_sample_count < 1
        or wave_numbers.ndim != 1
        or len(wave_numbers) == 0
        or np.any(~np.isfinite(wave_numbers))
        or np.any(wave_numbers <= 0.0)
    ):
        raise ValueError("count limits and wave numbers must be positive")
    order = np.lexsort((times, particles))
    particles = particles[order]
    jumps = jumps[order]
    particle_jumps = [jumps[particles == particle] for particle in np.unique(particles)]
    rows: list[dict[str, float]] = [
        {
            "jump_count": 0.0,
            "sample_count": float(len(particle_jumps)),
            "conditional_msd": 0.0,
            "conditional_fourth_moment": 0.0,
            **{
                f"conditional_characteristic_k{wave_number:g}".replace(".", "p"): 1.0
                for wave_number in wave_numbers
            },
        }
    ]
    for count in range(1, maximum_count + 1):
        displacements: list[np.ndarray] = []
        for values in particle_jumps:
            if len(values) < count:
                continue
            cumulative = np.vstack(
                [np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)]
            )
            displacements.append(cumulative[count:] - cumulative[:-count])
        sample_count = sum(len(values) for values in displacements)
        if sample_count < minimum_sample_count:
            break
        displacement = np.vstack(displacements)
        squared = np.sum(displacement**2, axis=1)
        row = {
            "jump_count": float(count),
            "sample_count": float(sample_count),
            "conditional_msd": float(np.mean(squared)),
            "conditional_fourth_moment": float(np.mean(squared**2)),
        }
        for wave_number in wave_numbers:
            key = f"conditional_characteristic_k{wave_number:g}".replace(".", "p")
            row[key] = float(np.mean(np.cos(wave_number * displacement)))
        rows.append(row)
    return rows


def posterior_weighted_state_displacement_kernels(
    counts: np.ndarray,
    displacements: np.ndarray,
    *,
    slow_mean_count: float,
    fast_mean_count: float,
    stationary_slow_probability: float,
    stationary_fast_probability: float,
    wave_numbers: np.ndarray,
) -> dict[str, float]:
    """Estimate slow/fast displacement kernels from count-emission posteriors."""

    counts = np.asarray(counts)
    displacements = np.asarray(displacements, dtype=float)
    wave_numbers = np.asarray(wave_numbers, dtype=float)
    if (
        counts.ndim != 1
        or displacements.ndim != 2
        or len(counts) != len(displacements)
        or len(counts) < 2
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
        or np.any(~np.isfinite(displacements))
    ):
        raise ValueError("counts and displacements must be aligned finite block samples")
    probabilities = np.array(
        [stationary_slow_probability, stationary_fast_probability], dtype=float
    )
    means = np.array([slow_mean_count, fast_mean_count], dtype=float)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or not math.isclose(float(np.sum(probabilities)), 1.0, abs_tol=1e-8)
        or np.any(~np.isfinite(means))
        or np.any(means <= 0.0)
        or means[0] > means[1]
        or wave_numbers.ndim != 1
        or len(wave_numbers) == 0
        or np.any(~np.isfinite(wave_numbers))
        or np.any(wave_numbers <= 0.0)
    ):
        raise ValueError("state probabilities, count means, and wave numbers are invalid")
    log_factorial = np.array([math.lgamma(float(value) + 1.0) for value in counts])
    log_weights = np.column_stack(
        [
            math.log(probabilities[state])
            + counts * math.log(means[state])
            - means[state]
            - log_factorial
            for state in range(2)
        ]
    )
    row_max = np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - row_max)
    weights /= np.sum(weights, axis=1, keepdims=True)
    squared = np.sum(displacements**2, axis=1)
    result: dict[str, float] = {}
    for state, prefix in enumerate(("slow", "fast")):
        state_weights = weights[:, state]
        weight_sum = float(np.sum(state_weights))
        result[f"{prefix}_posterior_weight_sum"] = weight_sum
        result[f"{prefix}_effective_sample_size"] = float(
            weight_sum**2 / np.sum(state_weights**2)
        )
        result[f"{prefix}_mean_count"] = float(
            np.sum(state_weights * counts) / weight_sum
        )
        result[f"{prefix}_msd"] = float(
            np.sum(state_weights * squared) / weight_sum
        )
        result[f"{prefix}_fourth_moment"] = float(
            np.sum(state_weights * squared**2) / weight_sum
        )
        for wave_number in wave_numbers:
            characteristic = np.mean(
                np.cos(wave_number * displacements), axis=1
            )
            key = f"{prefix}_characteristic_k{wave_number:g}".replace(".", "p")
            result[key] = float(
                np.sum(state_weights * characteristic) / weight_sum
            )
    return result


def two_clock_state_displacement_statistics(
    parameters: dict[str, object],
    kernels: dict[str, object],
    *,
    block_count: int,
    wave_number_key: str,
    dimension: int,
) -> dict[str, float]:
    """Propagate state-conditioned isotropic block displacements through two clocks."""

    if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count < 1:
        raise ValueError("block_count must be a positive integer")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    stationary = np.array(
        [
            float(parameters["stationary_slow_probability"]),
            float(parameters["stationary_fast_probability"]),
        ]
    )
    clock_weights = np.array(
        [
            float(parameters["fast_clock_weight"]),
            float(parameters["slow_clock_weight"]),
        ]
    )
    retentions = np.array(
        [
            float(parameters["fast_clock_retention"]),
            float(parameters["slow_clock_retention"]),
        ]
    )
    state_msd = np.array(
        [float(kernels["slow_msd"]), float(kernels["fast_msd"])]
    )
    state_fourth = np.array(
        [
            float(kernels["slow_fourth_moment"]),
            float(kernels["fast_fourth_moment"]),
        ]
    )
    state_characteristic = np.array(
        [
            float(kernels[f"slow_characteristic_{wave_number_key}"]),
            float(kernels[f"fast_characteristic_{wave_number_key}"]),
        ]
    )
    values = np.concatenate(
        [
            stationary,
            clock_weights,
            retentions,
            state_msd,
            state_fourth,
            state_characteristic,
        ]
    )
    if (
        np.any(~np.isfinite(values))
        or np.any(state_msd < 0.0)
        or np.any(state_fourth < 0.0)
        or np.any(np.abs(state_characteristic) > 1.0)
        or np.any(retentions < 0.0)
        or np.any(retentions > 1.0)
    ):
        raise ValueError("state displacement kernels and clocks are invalid")
    event_msd = block_count * float(stationary @ state_msd)
    single_fourth = block_count * float(stationary @ state_fourth)

    def propagate(retention: float) -> tuple[float, float]:
        transition = np.array(
            [
                [
                    1.0 - stationary[1] * (1.0 - retention),
                    stationary[1] * (1.0 - retention),
                ],
                [
                    stationary[0] * (1.0 - retention),
                    1.0 - stationary[0] * (1.0 - retention),
                ],
            ]
        )
        pair_sum = 0.0
        transition_power = transition.copy()
        for lag in range(1, block_count):
            pair_sum += (block_count - lag) * float(
                (stationary * state_msd) @ transition_power @ state_msd
            )
            transition_power = transition_power @ transition
        fourth = single_fourth + 2.0 * (1.0 + 2.0 / dimension) * pair_sum
        characteristic = stationary * state_characteristic
        for _ in range(1, block_count):
            characteristic = (characteristic @ transition) * state_characteristic
        return fourth, float(np.sum(characteristic))

    propagated = [propagate(float(retention)) for retention in retentions]
    return {
        "block_count": float(block_count),
        "lag_time": block_count * float(parameters["block_size"]),
        "event_msd": event_msd,
        "event_fourth_moment": float(
            sum(weight * value[0] for weight, value in zip(clock_weights, propagated))
        ),
        "event_characteristic": float(
            sum(weight * value[1] for weight, value in zip(clock_weights, propagated))
        ),
    }


def debye_waller_factor_from_msd(
    lags: np.ndarray,
    mean_squared_displacement: np.ndarray,
) -> dict[str, float]:
    """Select the Debye-Waller time at the minimum logarithmic MSD slope."""

    lag = np.asarray(lags, dtype=float)
    msd = np.asarray(mean_squared_displacement, dtype=float)
    if lag.ndim != 1 or msd.shape != lag.shape or len(lag) < 5:
        raise ValueError("lags and MSD must be aligned vectors with at least five values")
    if np.any(~np.isfinite(lag)) or np.any(~np.isfinite(msd)):
        raise ValueError("lags and MSD must be finite")
    if np.any(lag <= 0.0) or np.any(msd <= 0.0) or np.any(np.diff(lag) <= 0.0):
        raise ValueError("lags and MSD must be positive and lags strictly increasing")
    slope = np.gradient(np.log(msd), np.log(lag))
    index = int(np.argmin(slope[1:-1]) + 1)
    return {
        "debye_waller_lag": float(lag[index]),
        "debye_waller_factor": float(msd[index]),
        "minimum_log_msd_slope": float(slope[index]),
        "candidate_lag_count": float(len(lag)),
    }


def position_fluctuation_values(
    unwrapped_positions: np.ndarray,
    *,
    half_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling positional variance used by Debye-Waller cage segmentation."""

    positions = np.asarray(unwrapped_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] < 1:
        raise ValueError("unwrapped_positions must have shape (frames, particles, dimensions)")
    if isinstance(half_window, bool) or not isinstance(half_window, int) or half_window < 1:
        raise ValueError("half_window must be a positive integer")
    if len(positions) < 2 * half_window + 3 or np.any(~np.isfinite(positions)):
        raise ValueError("trajectory must be finite and longer than the fluctuation window")
    width = 2 * half_window + 1
    times = np.arange(half_window, len(positions) - half_window)
    prefix = np.concatenate(
        [np.zeros((1, positions.shape[1], positions.shape[2])), np.cumsum(positions, axis=0)],
        axis=0,
    )
    squared = np.sum(positions**2, axis=2)
    squared_prefix = np.concatenate(
        [np.zeros((1, positions.shape[1])), np.cumsum(squared, axis=0)],
        axis=0,
    )
    mean = (prefix[times + half_window + 1] - prefix[times - half_window]) / width
    mean_squared = (
        squared_prefix[times + half_window + 1] - squared_prefix[times - half_window]
    ) / width
    fluctuation = np.maximum(mean_squared - np.sum(mean**2, axis=2), 0.0)
    return times, fluctuation


def extract_debye_waller_cage_jumps(
    unwrapped_positions: np.ndarray,
    *,
    debye_waller_factor: float,
    half_window: int,
    activity_times: np.ndarray | None = None,
    activity_values: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Segment finite-duration jumps between adjacent inactive cage intervals."""

    positions = np.asarray(unwrapped_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] < 1 or np.any(~np.isfinite(positions)):
        raise ValueError("unwrapped_positions must be a finite trajectory array")
    if not math.isfinite(debye_waller_factor) or debye_waller_factor <= 0.0:
        raise ValueError("debye_waller_factor must be positive and finite")
    if (activity_times is None) != (activity_values is None):
        raise ValueError("activity_times and activity_values must be supplied together")
    if activity_times is None:
        times, activity = position_fluctuation_values(positions, half_window=half_window)
    else:
        times = np.asarray(activity_times, dtype=int)
        activity = np.asarray(activity_values, dtype=float)
        expected_length = len(positions) - 2 * half_window
        if times.shape != (expected_length,) or activity.shape != (
            expected_length,
            positions.shape[1],
        ):
            raise ValueError("precomputed fluctuation arrays do not match the trajectory")
    if np.any(~np.isfinite(activity)):
        raise ValueError("position fluctuation values must be finite")
    retained: list[tuple[object, ...]] = []
    for particle in range(positions.shape[1]):
        active = np.flatnonzero(activity[:, particle] > debye_waller_factor)
        if len(active) == 0:
            continue
        groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
        for group_index, group in enumerate(groups):
            previous_end = int(groups[group_index - 1][-1] + 1) if group_index else 0
            next_start = int(groups[group_index + 1][0]) if group_index + 1 < len(groups) else len(times)
            pre_indices = np.arange(previous_end, int(group[0]))
            post_indices = np.arange(int(group[-1] + 1), next_start)
            if len(pre_indices) == 0 or len(post_indices) == 0:
                continue
            pre_center = np.mean(positions[times[pre_indices], particle], axis=0)
            post_center = np.mean(positions[times[post_indices], particle], axis=0)
            peak_index = int(group[np.argmax(activity[group, particle])])
            retained.append(
                (
                    particle,
                    int(times[peak_index]),
                    float(activity[peak_index, particle]),
                    post_center - pre_center,
                    pre_center,
                    post_center,
                    int(times[group[0]]),
                    int(times[group[-1]]),
                    float(len(group)),
                    float(len(pre_indices)),
                    float(len(post_indices)),
                )
            )
    dimension = positions.shape[2]
    if not retained:
        return {
            "particle": np.empty(0, dtype=int),
            "time": np.empty(0, dtype=int),
            "activity": np.empty(0, dtype=float),
            "jump_vector": np.empty((0, dimension), dtype=float),
            "pre_center": np.empty((0, dimension), dtype=float),
            "post_center": np.empty((0, dimension), dtype=float),
            "jump_start": np.empty(0, dtype=int),
            "jump_end": np.empty(0, dtype=int),
            "jump_duration": np.empty(0, dtype=float),
            "pre_cage_duration": np.empty(0, dtype=float),
            "post_cage_duration": np.empty(0, dtype=float),
        }
    retained.sort(key=lambda event: (int(event[0]), int(event[1])))
    return {
        "particle": np.array([event[0] for event in retained], dtype=int),
        "time": np.array([event[1] for event in retained], dtype=int),
        "activity": np.array([event[2] for event in retained], dtype=float),
        "jump_vector": np.stack([event[3] for event in retained]),
        "pre_center": np.stack([event[4] for event in retained]),
        "post_center": np.stack([event[5] for event in retained]),
        "jump_start": np.array([event[6] for event in retained], dtype=int),
        "jump_end": np.array([event[7] for event in retained], dtype=int),
        "jump_duration": np.array([event[8] for event in retained], dtype=float),
        "pre_cage_duration": np.array([event[9] for event in retained], dtype=float),
        "post_cage_duration": np.array([event[10] for event in retained], dtype=float),
    }


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
    model_columns: dict[str, str] | None = None,
    primary_model: str = "correlated_event_clock",
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    """Score event-clock diffusion predictions against independent held-out windows."""

    if not rows or not 0.0 < minimum_coverage < maximum_coverage:
        raise ValueError("rows and an ordered positive coverage interval are required")
    models = model_columns or {
        "uncorrelated_event_clock": "uncorrelated_event_diffusion",
        "correlated_event_clock": "correlated_event_diffusion",
    }
    if not models or primary_model not in models:
        raise ValueError("primary_model must identify one requested transport model")
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
    primary = coverage_by_model[primary_model]
    passed = bool(np.all((primary >= minimum_coverage) & (primary <= maximum_coverage)))
    if passed:
        failure = "none"
    elif float(np.mean(primary)) < minimum_coverage:
        failure = f"{primary_model}_undercoverage"
    elif float(np.mean(primary)) > maximum_coverage:
        failure = f"{primary_model}_overcoverage"
    else:
        failure = "replicate_transport_inconsistency"
    verdict: dict[str, float | str] = {
        "minimum_coverage": minimum_coverage,
        "maximum_coverage": maximum_coverage,
        "heldout_transport_pass": float(passed),
        "primary_failure": failure,
        "primary_model": primary_model,
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


def event_conditioned_neighbor_displacement(
    unwrapped_positions: np.ndarray,
    events: dict[str, np.ndarray],
    *,
    box_lengths: np.ndarray,
    distance_edges: np.ndarray,
    half_window: int,
    event_indices: np.ndarray,
    control_particles: np.ndarray,
    integration_max_distance: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Compare neighbor motion around events with same-time random-center controls."""

    positions = np.asarray(unwrapped_positions, dtype=float)
    box = np.asarray(box_lengths, dtype=float)
    edges = np.asarray(distance_edges, dtype=float)
    selected = np.asarray(event_indices, dtype=int)
    controls = np.asarray(control_particles, dtype=int)
    particles = np.asarray(events["particle"], dtype=int)
    times = np.asarray(events["time"], dtype=int)
    jumps = np.asarray(events["jump_vector"], dtype=float)
    if positions.ndim != 3 or positions.shape[2] != len(box):
        raise ValueError("unwrapped_positions must have shape (frames, particles, dimensions)")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("positions and positive box lengths must be finite")
    if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("distance_edges must be a strictly increasing finite sequence")
    if half_window < 1 or not math.isfinite(integration_max_distance) or integration_max_distance <= 0.0:
        raise ValueError("half_window and integration_max_distance must be positive")
    if particles.shape != times.shape or jumps.shape != (len(particles), positions.shape[2]):
        raise ValueError("event particle, time, and jump arrays must align")
    if selected.ndim != 1 or len(selected) == 0 or controls.shape != selected.shape:
        raise ValueError("event_indices and control_particles must be aligned nonempty vectors")
    if np.any(selected < 0) or np.any(selected >= len(particles)):
        raise ValueError("event_indices are out of range")
    if np.any(controls < 0) or np.any(controls >= positions.shape[1]):
        raise ValueError("control_particles are out of range")
    selected_times = times[selected]
    if np.any(selected_times < half_window) or np.any(selected_times + half_window > len(positions)):
        raise ValueError("selected events lack complete pre/post windows")

    bin_count = len(edges) - 1
    event_count = np.zeros(bin_count, dtype=np.int64)
    control_count = np.zeros(bin_count, dtype=np.int64)
    event_squared_sum = np.zeros(bin_count, dtype=float)
    control_squared_sum = np.zeros(bin_count, dtype=float)
    event_projection_sum = np.zeros(bin_count, dtype=float)
    control_projection_sum = np.zeros(bin_count, dtype=float)
    for event_index, control in zip(selected, controls):
        time = int(times[event_index])
        particle = int(particles[event_index])
        jump = jumps[event_index]
        jump_squared = float(np.dot(jump, jump))
        if jump_squared <= 0.0 or not math.isfinite(jump_squared):
            raise ValueError("selected event jump vectors must have positive finite norm")
        pre = np.mean(positions[time - half_window : time], axis=0)
        post = np.mean(positions[time : time + half_window], axis=0)
        displacement = post - pre
        squared_displacement = np.sum(displacement**2, axis=1)
        projection = displacement @ jump / jump_squared

        event_separation = pre - pre[particle]
        event_separation -= box * np.rint(event_separation / box)
        event_bin = np.searchsorted(edges, np.linalg.norm(event_separation, axis=1), side="right") - 1
        retained = (
            (np.arange(len(pre)) != particle)
            & (event_bin >= 0)
            & (event_bin < bin_count)
        )
        event_count += np.bincount(event_bin[retained], minlength=bin_count)
        event_squared_sum += np.bincount(
            event_bin[retained], weights=squared_displacement[retained], minlength=bin_count
        )
        event_projection_sum += np.bincount(
            event_bin[retained], weights=projection[retained], minlength=bin_count
        )

        control_separation = pre - pre[control]
        control_separation -= box * np.rint(control_separation / box)
        control_bin = np.searchsorted(edges, np.linalg.norm(control_separation, axis=1), side="right") - 1
        retained = (
            (np.arange(len(pre)) != control)
            & (control_bin >= 0)
            & (control_bin < bin_count)
        )
        control_count += np.bincount(control_bin[retained], minlength=bin_count)
        control_squared_sum += np.bincount(
            control_bin[retained], weights=squared_displacement[retained], minlength=bin_count
        )
        control_projection_sum += np.bincount(
            control_bin[retained], weights=projection[retained], minlength=bin_count
        )

    if np.any(event_count == 0) or np.any(control_count == 0):
        raise ValueError("every distance bin must contain event and control pairs")
    event_mean = event_squared_sum / event_count
    control_mean = control_squared_sum / control_count
    rows: list[dict[str, float]] = []
    for index in range(bin_count):
        rows.append(
            {
                "distance_low": float(edges[index]),
                "distance_high": float(edges[index + 1]),
                "distance_midpoint": float(0.5 * (edges[index] + edges[index + 1])),
                "event_pair_count": float(event_count[index]),
                "control_pair_count": float(control_count[index]),
                "event_mean_squared_displacement": float(event_mean[index]),
                "control_mean_squared_displacement": float(control_mean[index]),
                "event_to_control_squared_ratio": float(event_mean[index] / control_mean[index])
                if control_mean[index] > 0.0
                else math.inf,
                "event_mean_longitudinal_projection": float(
                    event_projection_sum[index] / event_count[index]
                ),
                "control_mean_longitudinal_projection": float(
                    control_projection_sum[index] / control_count[index]
                ),
            }
        )
    integrated = np.array([edges[index + 1] <= integration_max_distance for index in range(bin_count)])
    if not np.any(integrated):
        raise ValueError("integration_max_distance must include at least one complete bin")
    excess_per_event = event_count / len(selected) * (event_mean - control_mean)
    mean_self_jump_squared = float(np.mean(np.sum(jumps[selected] ** 2, axis=1)))
    integrated_excess = float(np.sum(excess_per_event[integrated]))
    summary = {
        "sampled_event_count": float(len(selected)),
        "mean_self_jump_squared": mean_self_jump_squared,
        "integration_max_distance": float(integration_max_distance),
        "integrated_neighbor_excess": integrated_excess,
        "integrated_neighbor_excess_over_self_jump_squared": integrated_excess
        / mean_self_jump_squared,
    }
    return rows, summary


def summarize_paired_curve_stability(
    calibration_rows: Sequence[dict[str, object]],
    heldout_rows: Sequence[dict[str, object]],
    *,
    bin_key: str,
    metric_key: str,
    relative_equivalence_margin: float,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    """Test paired split-half shifts and equivalence across independent replicas."""

    if not math.isfinite(relative_equivalence_margin) or not 0.0 < relative_equivalence_margin < 1.0:
        raise ValueError("relative_equivalence_margin must lie between zero and one")

    def keyed(rows: Sequence[dict[str, object]]) -> dict[tuple[float, float], float]:
        result: dict[tuple[float, float], float] = {}
        for row in rows:
            key = (float(row["replicate"]), float(row[bin_key]))
            value = float(row[metric_key])
            if key in result or not math.isfinite(value):
                raise ValueError("paired curve rows must have unique finite replicate-bin values")
            result[key] = value
        return result

    calibration = keyed(calibration_rows)
    heldout = keyed(heldout_rows)
    if calibration.keys() != heldout.keys():
        raise ValueError("calibration and held-out curves must have identical replicate-bin keys")
    bins = sorted({key[1] for key in calibration})
    rows: list[dict[str, float | str]] = []
    for bin_value in bins:
        keys = sorted(key for key in calibration if key[1] == bin_value)
        if len(keys) < 2:
            raise ValueError("paired curve stability requires at least two independent replicas")
        differences = np.array([heldout[key] - calibration[key] for key in keys])
        mean = float(np.mean(differences))
        standard_deviation = float(np.std(differences, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(differences))
        ci_low, ci_high, critical = independent_sample_ci95(
            mean=mean,
            standard_error=standard_error,
            sample_count=len(differences),
        )
        calibration_mean = float(np.mean([calibration[key] for key in keys]))
        equivalence_margin = relative_equivalence_margin * abs(calibration_mean)
        rows.append(
            {
                bin_key: bin_value,
                "calibration_mean": calibration_mean,
                "heldout_mean": float(np.mean([heldout[key] for key in keys])),
                "mean_paired_difference": mean,
                "standard_deviation_paired_difference": standard_deviation,
                "standard_error_paired_difference": standard_error,
                "ci95_low_paired_difference": ci_low,
                "ci95_high_paired_difference": ci_high,
                "ci95_critical_value": critical,
                "ci95_method": "student_t_paired_independent_replicates",
                "independent_replicate_count": float(len(differences)),
                "paired_difference_ci_includes_zero": float(ci_low <= 0.0 <= ci_high),
                "relative_equivalence_margin": relative_equivalence_margin,
                "absolute_equivalence_margin": equivalence_margin,
                "paired_difference_ci_within_margin": float(
                    ci_low >= -equivalence_margin and ci_high <= equivalence_margin
                ),
            }
        )
    shift_not_detected = all(
        float(row["paired_difference_ci_includes_zero"]) == 1.0 for row in rows
    )
    equivalent = all(
        float(row["paired_difference_ci_within_margin"]) == 1.0 for row in rows
    )
    return rows, {
        "paired_shift_not_detected": float(shift_not_detected),
        "paired_curve_equivalent": float(equivalent),
        "relative_equivalence_margin": relative_equivalence_margin,
        "tested_bin_count": float(len(rows)),
        "nonequivalent_bin_count": float(
            sum(float(row["paired_difference_ci_within_margin"]) == 0.0 for row in rows)
        ),
        "independent_replicate_count": rows[0]["independent_replicate_count"],
        "ci95_method": "student_t_paired_independent_replicates",
    }


def summarize_neighbor_halo_replicates(
    shell_rows: Sequence[dict[str, object]],
    replicate_rows: Sequence[dict[str, object]],
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
    """Attach independent-replicate uncertainty to an event-conditioned halo."""

    _, curve = summarize_replicate_binned_metric(
        shell_rows,
        bin_key="distance_midpoint",
        metric_key="event_to_control_squared_ratio",
    )
    for row in curve:
        midpoint = float(row["distance_midpoint"])
        selected = [source for source in shell_rows if float(source["distance_midpoint"]) == midpoint]
        lows = {float(source["distance_low"]) for source in selected}
        highs = {float(source["distance_high"]) for source in selected}
        if len(lows) != 1 or len(highs) != 1:
            raise ValueError("distance-bin edges must agree across replicas")
        row["distance_low"] = lows.pop()
        row["distance_high"] = highs.pop()
        row["halo_detected_in_shell"] = float(float(row["ci95_low"]) > 1.0)
        row["uncertainty_scope"] = "independent_replicates"

    halo_radius = 0.0
    contiguous = True
    for row in sorted(curve, key=lambda item: float(item["distance_low"])):
        if contiguous and float(row["halo_detected_in_shell"]) == 1.0:
            halo_radius = float(row["distance_high"])
        else:
            contiguous = False

    values = np.array(
        [float(row["integrated_neighbor_excess_over_self_jump_squared"]) for row in replicate_rows]
    )
    replicates = [float(row["replicate"]) for row in replicate_rows]
    if len(values) < 2 or len(set(replicates)) != len(values):
        raise ValueError("at least two unique independent replicate halo summaries are required")
    if np.any(~np.isfinite(values)):
        raise ValueError("integrated halo ratios must be finite")
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    standard_error = standard_deviation / math.sqrt(len(values))
    ci_low, ci_high, critical = independent_sample_ci95(
        mean=mean,
        standard_error=standard_error,
        sample_count=len(values),
    )
    verdict: dict[str, float | str] = {
        "halo_radius_lower_bound": halo_radius,
        "mean_integrated_neighbor_excess_over_self_jump_squared": mean,
        "standard_deviation_integrated_neighbor_excess_over_self_jump_squared": standard_deviation,
        "standard_error_integrated_neighbor_excess_over_self_jump_squared": standard_error,
        "ci95_low_integrated_neighbor_excess_over_self_jump_squared": ci_low,
        "ci95_high_integrated_neighbor_excess_over_self_jump_squared": ci_high,
        "ci95_critical_value": critical,
        "ci95_method": "student_t_independent_replicates",
        "independent_replicate_count": float(len(values)),
        "spatial_measurement_claim_allowed": float(halo_radius > 0.0 and ci_low > 0.0),
        "spatial_model_claim_allowed": 0.0,
        "thermodynamic_claim_allowed": 0.0,
    }
    return curve, verdict


def event_activity_duration_statistics(
    activity_times: np.ndarray,
    activity_values: np.ndarray,
    events: dict[str, np.ndarray],
    *,
    threshold: float,
) -> dict[str, float]:
    """Measure contiguous above-threshold durations for retained event peaks."""

    times = np.asarray(activity_times, dtype=int)
    activity = np.asarray(activity_values, dtype=float)
    particles = np.asarray(events["particle"], dtype=int)
    event_times = np.asarray(events["time"], dtype=int)
    if times.ndim != 1 or activity.ndim != 2 or activity.shape[0] != len(times):
        raise ValueError("activity times and values must have aligned time axes")
    if particles.shape != event_times.shape or particles.ndim != 1 or len(particles) == 0:
        raise ValueError("event particle and time arrays must be aligned and nonempty")
    if np.any(particles < 0) or np.any(particles >= activity.shape[1]):
        raise ValueError("event particle index is out of range")
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")
    time_to_index = {int(time): index for index, time in enumerate(times)}
    durations: list[int] = []
    for particle, event_time in zip(particles, event_times):
        if int(event_time) not in time_to_index:
            raise ValueError("every event time must occur on the activity time axis")
        index = time_to_index[int(event_time)]
        if activity[index, particle] <= threshold:
            raise ValueError("event peak must exceed the activity threshold")
        lower = index
        upper = index
        while lower > 0 and activity[lower - 1, particle] > threshold:
            lower -= 1
        while upper + 1 < len(times) and activity[upper + 1, particle] > threshold:
            upper += 1
        durations.append(upper - lower + 1)
    values = np.asarray(durations, dtype=float)
    median = float(np.median(values))
    return {
        "event_count": float(len(values)),
        "mean_duration": float(np.mean(values)),
        "median_duration": median,
        "duration_p75": float(np.quantile(values, 0.75)),
        "duration_p90": float(np.quantile(values, 0.90)),
        "duration_p95": float(np.quantile(values, 0.95)),
        "maximum_duration": float(np.max(values)),
        "cluster_time_window": float(max(1, math.ceil(median))),
    }


def spatiotemporal_event_cluster_statistics(
    events: dict[str, np.ndarray],
    *,
    box_lengths: np.ndarray,
    maximum_time_separation: int,
    maximum_distance: float,
) -> dict[str, float]:
    """Cluster events connected within a measured time window and spatial halo."""

    particles = np.asarray(events["particle"], dtype=int)
    times = np.asarray(events["time"], dtype=float)
    centers = np.asarray(events["pre_center"], dtype=float)
    box = np.asarray(box_lengths, dtype=float)
    if (
        times.ndim != 1
        or len(times) == 0
        or particles.shape != times.shape
        or centers.shape != (len(times), len(box))
    ):
        raise ValueError("event times and centers must be aligned and nonempty")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(centers)):
        raise ValueError("event times and centers must be finite")
    if np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box lengths must be positive and finite")
    if maximum_time_separation < 0 or maximum_distance <= 0.0:
        raise ValueError("cluster time and distance scales must be nonnegative and positive")
    order = np.argsort(times)
    sorted_particles = particles[order]
    sorted_times = times[order]
    sorted_centers = centers[order]
    parent = np.arange(len(order))
    size = np.ones(len(order), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    squared_cutoff = maximum_distance**2
    for left in range(len(order)):
        right = left + 1
        while right < len(order) and sorted_times[right] - sorted_times[left] <= maximum_time_separation:
            if sorted_particles[right] == sorted_particles[left]:
                right += 1
                continue
            displacement = sorted_centers[right] - sorted_centers[left]
            displacement -= box * np.rint(displacement / box)
            if float(np.dot(displacement, displacement)) <= squared_cutoff:
                union(left, right)
            right += 1
    roots = np.array([find(index) for index in range(len(order))])
    counts = np.array([np.sum(roots == root) for root in np.unique(roots)], dtype=float)
    return {
        "event_count": float(len(order)),
        "cluster_count": float(len(counts)),
        "mean_cluster_size": float(np.mean(counts)),
        "event_weighted_cluster_size": float(np.sum(counts**2) / np.sum(counts)),
        "nontrivial_event_fraction": float(np.sum(counts[counts > 1.0]) / np.sum(counts)),
        "maximum_cluster_size": float(np.max(counts)),
        "maximum_time_separation": float(maximum_time_separation),
        "maximum_distance": float(maximum_distance),
    }


def isolated_event_response_amplitude(
    unwrapped_positions: np.ndarray,
    events: dict[str, np.ndarray],
    *,
    response_lag: int,
    half_window: int,
) -> dict[str, float]:
    """Measure the persistent displacement projected onto isolated event marks."""

    positions = np.asarray(unwrapped_positions, dtype=float)
    particles = np.asarray(events["particle"], dtype=int)
    times = np.asarray(events["time"], dtype=int)
    jumps = np.asarray(events["jump_vector"], dtype=float)
    pre_centers = np.asarray(events["pre_center"], dtype=float)
    if positions.ndim != 3 or positions.shape[2] < 1:
        raise ValueError("unwrapped_positions must have shape (frames, particles, dimensions)")
    if particles.shape != times.shape or jumps.shape != (len(times), positions.shape[2]):
        raise ValueError("event particle, time, and jump arrays must align")
    if pre_centers.shape != jumps.shape or len(times) == 0:
        raise ValueError("event pre-centers must align with a nonempty event table")
    if response_lag < 0 or half_window < 1:
        raise ValueError("response_lag and half_window must be nonnegative and positive")
    next_time = np.full(len(times), np.inf)
    for particle in np.unique(particles):
        indices = np.flatnonzero(particles == particle)
        indices = indices[np.argsort(times[indices])]
        next_time[indices[:-1]] = times[indices[1:]]
    jump_squared = np.sum(jumps**2, axis=1)
    stop = times + response_lag + half_window
    valid = (
        (times + response_lag >= 0)
        & (stop <= len(positions))
        & (next_time > stop)
        & (jump_squared > 0.0)
    )
    values: list[float] = []
    for index in np.flatnonzero(valid):
        time = int(times[index])
        future = np.mean(
            positions[
                time + response_lag : time + response_lag + half_window,
                particles[index],
            ],
            axis=0,
        )
        values.append(
            float(np.dot(future - pre_centers[index], jumps[index]) / jump_squared[index])
        )
    if not values:
        raise ValueError("no isolated events support the requested response lag")
    response = np.asarray(values, dtype=float)
    standard_deviation = float(np.std(response, ddof=1)) if len(response) > 1 else 0.0
    return {
        "isolated_event_count": float(len(response)),
        "response_lag": float(response_lag),
        "mean_response_amplitude": float(np.mean(response)),
        "median_response_amplitude": float(np.median(response)),
        "standard_deviation_response_amplitude": standard_deviation,
        "standard_error_response_amplitude": standard_deviation / math.sqrt(len(response)),
    }


def cooperative_cluster_diffusion_coefficient(
    *,
    event_rate: float,
    self_jump_squared: float,
    integrated_neighbor_excess: float,
    response_amplitude: float,
    mean_cluster_size: float,
    dimension: int = 3,
) -> float:
    """Diffusion from measured cooperative marks after cluster de-duplication."""

    values = (
        event_rate,
        self_jump_squared,
        integrated_neighbor_excess,
        response_amplitude,
        mean_cluster_size,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cooperative diffusion inputs must be finite")
    if event_rate <= 0.0 or self_jump_squared <= 0.0 or mean_cluster_size <= 0.0:
        raise ValueError("event rate, self mark, and cluster size must be positive")
    if integrated_neighbor_excess < 0.0 or response_amplitude < 0.0:
        raise ValueError("halo excess and response amplitude must be nonnegative")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    mark_squared = response_amplitude**2 * (
        self_jump_squared + integrated_neighbor_excess
    )
    return event_rate * mark_squared / (2.0 * dimension * mean_cluster_size)


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


def load_lammps_custom_trajectory(
    path: Path,
    *,
    maximum_frame_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Load the fixed custom-dump schema emitted by the replicate protocol."""

    path = Path(path)
    if maximum_frame_count is not None and (
        isinstance(maximum_frame_count, bool)
        or not isinstance(maximum_frame_count, int)
        or maximum_frame_count < 1
    ):
        raise ValueError("maximum_frame_count must be a positive integer")
    timesteps: list[int] = []
    wrapped_frames: list[np.ndarray] = []
    unwrapped_frames: list[np.ndarray] = []
    velocity_frames: list[np.ndarray] = []
    force_frames: list[np.ndarray] = []
    expected_types: np.ndarray | None = None
    expected_box: np.ndarray | None = None
    optional_schema: str | None = None
    with path.open() as handle:
        while True:
            if maximum_frame_count is not None and len(timesteps) >= maximum_frame_count:
                break
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
            base_header = "ITEM: ATOMS id type x y z ix iy iz"
            velocity_header = base_header + " vx vy vz"
            extended_header = base_header + " vx vy vz fx fy fz"
            schema_by_header = {
                base_header: "positions",
                velocity_header: "velocity",
                extended_header: "velocity_force",
            }
            schema = schema_by_header.get(atom_header)
            if schema is None:
                raise ValueError("unexpected LAMMPS dump atom schema")
            if optional_schema is None:
                optional_schema = schema
            elif schema != optional_schema:
                raise ValueError("LAMMPS dump schema changed between frames")
            rows = [handle.readline() for _ in range(particle_count)]
            if any(row == "" for row in rows):
                raise ValueError("truncated LAMMPS dump frame")
            values = np.fromstring("".join(rows), sep=" ")
            column_count = {"positions": 8, "velocity": 11, "velocity_force": 14}[schema]
            if values.size != particle_count * column_count:
                raise ValueError("malformed LAMMPS dump atom row")
            values = values.reshape(particle_count, column_count)
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
            if schema in {"velocity", "velocity_force"}:
                velocity_frames.append(values[:, 8:11].astype(np.float32))
            if schema == "velocity_force":
                force_frames.append(values[:, 11:14].astype(np.float32))

    if not timesteps:
        raise ValueError("LAMMPS dump contains no frames")
    timestep_array = np.asarray(timesteps, dtype=np.int64)
    if len(timestep_array) > 1 and np.any(np.diff(timestep_array) <= 0):
        raise ValueError("LAMMPS dump timesteps must increase strictly")
    result = {
        "timesteps": timestep_array,
        "particle_types": np.asarray(expected_types, dtype=int),
        "box_lengths": np.asarray(expected_box, dtype=float),
        "wrapped_positions": np.stack(wrapped_frames),
        "unwrapped_positions": np.stack(unwrapped_frames),
    }
    if optional_schema in {"velocity", "velocity_force"}:
        result["velocities"] = np.stack(velocity_frames)
    if optional_schema == "velocity_force":
        result["forces"] = np.stack(force_frames)
    return result


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
    dynamics: str,
    langevin_damping: float | None,
    langevin_seed: int | None,
    dump_interval_steps: int,
    high_resolution_steps: int,
    high_resolution_dump_interval_steps: int | None,
) -> str:
    if dynamics == "nvt":
        dynamics_lines = f"fix thermostat all nvt temp {temperature:g} {temperature:g} 10"
    elif dynamics == "langevin":
        if langevin_damping is None or langevin_seed is None:
            raise ValueError("langevin dynamics requires damping and a random seed")
        dynamics_lines = (
            "fix integrator all nve\n"
            f"fix bath all langevin {temperature:g} {temperature:g} {langevin_damping:g} {langevin_seed}"
        )
    else:
        raise ValueError("dynamics must be 'nvt' or 'langevin'")
    high_resolution_commands = ""
    if high_resolution_steps:
        if high_resolution_dump_interval_steps is None:
            raise ValueError("high-resolution dump interval is required when high-resolution steps are positive")
        high_resolution_commands = f"""dump high_resolution all custom {high_resolution_dump_interval_steps} high_resolution.lammpstrj id type x y z ix iy iz
dump_modify high_resolution sort id
run {high_resolution_steps}
undump high_resolution
"""
    remaining_steps = production_steps - high_resolution_steps
    remaining_commands = f"run {remaining_steps}\n" if remaining_steps else ""
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
{dynamics_lines}
timestep 0.001
thermo 10000
thermo_style custom step time temp pe ke etotal press

run {equilibration_steps}
write_restart equilibrated.restart
reset_timestep 0

dump trajectory all custom {dump_interval_steps} trajectory.lammpstrj id type x y z ix iy iz
dump_modify trajectory sort id
restart 100000 restart.*
{high_resolution_commands}{remaining_commands}write_restart final.restart
"""


def _isoconfigurational_lammps_input(
    *,
    parent_restart: Path,
    temperature: float,
    velocity_seed: int,
    langevin_seed: int,
    damping: float,
    duration_steps: int,
    dump_interval_steps: int,
    dump_velocity_force: bool,
) -> str:
    """Build one NVE-plus-Langevin clone from an identical parent restart."""

    dump_columns = "id type x y z ix iy iz vx vy vz fx fy fz" if dump_velocity_force else "id type x y z ix iy iz"
    return f"""units lj
atom_style atomic
read_restart {parent_restart}

reset_timestep 0
velocity all create {temperature:g} {velocity_seed} mom yes rot no dist gaussian
fix integrator all nve
fix bath all langevin {temperature:g} {temperature:g} {damping:g} {langevin_seed}
timestep 0.001
thermo 10000
thermo_style custom step time temp pe ke etotal press

dump clone all custom {dump_interval_steps} trajectory.lammpstrj {dump_columns}
dump_modify clone sort id
run {duration_steps}
write_restart final.restart
"""


def prepare_isoconfigurational_langevin_clones(
    parent_restart: Path,
    output_directory: Path,
    *,
    temperature: float,
    velocity_seeds: Sequence[int],
    langevin_seeds: Sequence[int],
    damping: float,
    duration: float,
    dump_interval: float,
    dump_velocity_force: bool = False,
) -> dict[str, object]:
    """Prepare independent-noise Langevin clones from one many-particle state.

    Each clone reads exactly the same restart coordinates, then independently
    samples Maxwell velocities and Langevin noise.  This is an
    isoconfigurational ensemble, not a continuation of a physical-time path.
    """

    parent_restart = Path(parent_restart).resolve()
    output_directory = Path(output_directory).resolve()
    if not parent_restart.is_file():
        raise ValueError("parent_restart must be an existing LAMMPS restart")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be positive and finite")
    if not math.isfinite(duration) or duration <= 0.0 or not math.isfinite(dump_interval) or dump_interval <= 0.0:
        raise ValueError("duration and dump_interval must be positive and finite")
    velocity = [int(value) for value in velocity_seeds]
    noise = [int(value) for value in langevin_seeds]
    if len(velocity) < 2 or len(velocity) != len(noise):
        raise ValueError("at least two aligned velocity and Langevin seeds are required")
    if any(value <= 0 for value in velocity + noise) or len(set(velocity)) != len(velocity) or len(set(noise)) != len(noise):
        raise ValueError("velocity and Langevin seeds must be positive and unique within each sequence")
    timestep = 0.001
    duration_steps = int(round(duration / timestep))
    dump_interval_steps = int(round(dump_interval / timestep))
    if not math.isclose(duration_steps * timestep, duration) or not math.isclose(dump_interval_steps * timestep, dump_interval):
        raise ValueError("duration and dump_interval must be integer multiples of 0.001 tau")
    if output_directory.exists():
        raise ValueError("output_directory must not already exist")

    output_directory.mkdir(parents=True)
    clone_rows: list[dict[str, object]] = []
    for clone_index, (velocity_seed, langevin_seed) in enumerate(zip(velocity, noise), start=1):
        clone_directory = output_directory / f"clone_{clone_index:03d}"
        clone_directory.mkdir()
        (clone_directory / "in.clone").write_text(
            _isoconfigurational_lammps_input(
                parent_restart=parent_restart,
                temperature=temperature,
                velocity_seed=velocity_seed,
                langevin_seed=langevin_seed,
                damping=damping,
                duration_steps=duration_steps,
                dump_interval_steps=dump_interval_steps,
                dump_velocity_force=dump_velocity_force,
            )
        )
        row = {
            "clone_index": clone_index,
            "velocity_seed": velocity_seed,
            "langevin_seed": langevin_seed,
            "parent_restart_path": str(parent_restart),
            "axis_semantics": "isoconfigurational_langevin_clones",
        }
        (clone_directory / "clone_manifest.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        clone_rows.append(row)
    manifest: dict[str, object] = {
        "clone_count": len(clone_rows),
        "parent_restart_path": str(parent_restart),
        "parent_restart_sha256": _sha256(parent_restart),
        "temperature": temperature,
        "dynamics": "nve_plus_langevin",
        "langevin_damping": damping,
        "duration_tau": duration,
        "saved_frame_interval_tau": dump_interval,
        "dump_velocity_force": bool(dump_velocity_force),
        "axis_semantics": "isoconfigurational_langevin_clones",
        "randomized_maxwell_velocities": True,
        "independent_bath_noise": True,
        "thermodynamic_claim_allowed": False,
        "clones": clone_rows,
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def prepare_replicate(
    source_path: Path,
    output_directory: Path,
    *,
    temperature: float,
    frame_index: int,
    velocity_seed: int,
    equilibration_time: float = 100.0,
    production_time: float = 5000.0,
    dynamics: str = "nvt",
    langevin_damping: float | None = None,
    langevin_seed: int | None = None,
    dump_interval_time: float = 1.0,
    high_resolution_duration: float = 0.0,
    high_resolution_dump_interval: float | None = None,
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
    if dynamics not in {"nvt", "langevin"}:
        raise ValueError("dynamics must be 'nvt' or 'langevin'")
    if dynamics == "langevin":
        if langevin_damping is None or not math.isfinite(langevin_damping) or langevin_damping <= 0.0:
            raise ValueError("langevin_damping must be positive and finite for Langevin dynamics")
        if isinstance(langevin_seed, bool) or not isinstance(langevin_seed, int) or langevin_seed <= 0:
            raise ValueError("langevin_seed must be a positive integer for Langevin dynamics")
    elif langevin_damping is not None or langevin_seed is not None:
        raise ValueError("Langevin controls are only valid when dynamics='langevin'")
    for name, value in (
        ("equilibration_time", equilibration_time),
        ("production_time", production_time),
        ("dump_interval_time", dump_interval_time),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(high_resolution_duration) or high_resolution_duration < 0.0:
        raise ValueError("high_resolution_duration must be finite and nonnegative")
    if high_resolution_duration > production_time:
        raise ValueError("high_resolution_duration cannot exceed production_time")
    if high_resolution_duration > 0.0:
        if high_resolution_dump_interval is None or not math.isfinite(high_resolution_dump_interval):
            raise ValueError("high_resolution_dump_interval is required for a positive high_resolution_duration")
        if high_resolution_dump_interval <= 0.0:
            raise ValueError("high_resolution_dump_interval must be positive")
    elif high_resolution_dump_interval is not None:
        raise ValueError("high_resolution_dump_interval requires a positive high_resolution_duration")

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
    dump_interval_steps = int(round(dump_interval_time / timestep))
    high_resolution_steps = int(round(high_resolution_duration / timestep))
    high_resolution_dump_interval_steps = (
        int(round(high_resolution_dump_interval / timestep))
        if high_resolution_dump_interval is not None
        else None
    )
    if not math.isclose(equilibration_steps * timestep, equilibration_time):
        raise ValueError("equilibration_time must be an integer multiple of 0.001 tau")
    if not math.isclose(production_steps * timestep, production_time):
        raise ValueError("production_time must be an integer multiple of 0.001 tau")
    if not math.isclose(dump_interval_steps * timestep, dump_interval_time):
        raise ValueError("dump_interval_time must be an integer multiple of 0.001 tau")
    if not math.isclose(high_resolution_steps * timestep, high_resolution_duration):
        raise ValueError("high_resolution_duration must be an integer multiple of 0.001 tau")
    if high_resolution_dump_interval is not None and not math.isclose(
        high_resolution_dump_interval_steps * timestep, high_resolution_dump_interval
    ):
        raise ValueError("high_resolution_dump_interval must be an integer multiple of 0.001 tau")

    output_directory.mkdir(parents=True, exist_ok=False)
    _write_lammps_data(output_directory / "initial.data", positions, particle_types, box_lengths)
    (output_directory / "in.production").write_text(
        _lammps_input(
            temperature=temperature,
            velocity_seed=velocity_seed,
            equilibration_steps=equilibration_steps,
            production_steps=production_steps,
            dynamics=dynamics,
            langevin_damping=langevin_damping,
            langevin_seed=langevin_seed,
            dump_interval_steps=dump_interval_steps,
            high_resolution_steps=high_resolution_steps,
            high_resolution_dump_interval_steps=high_resolution_dump_interval_steps,
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
        "ensemble": "NVT" if dynamics == "nvt" else "NVE_plus_Langevin",
        "dynamics": dynamics,
        "thermostat": "LAMMPS_fix_nvt_Nose_Hoover" if dynamics == "nvt" else "LAMMPS_fix_langevin",
        "thermostat_coupling_tau": 10.0 if dynamics == "nvt" else langevin_damping,
        "langevin_seed": langevin_seed,
        "timestep_tau": timestep,
        "equilibration_time_tau": equilibration_time,
        "production_time_tau": production_time,
        "saved_frame_interval_tau": dump_interval_time,
        "high_resolution_duration_tau": high_resolution_duration,
        "high_resolution_saved_frame_interval_tau": high_resolution_dump_interval,
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
    dynamics: str = "nvt",
    langevin_damping: float | None = None,
    langevin_seeds: Sequence[int] | None = None,
    dump_interval_time: float = 1.0,
    high_resolution_duration: float = 0.0,
    high_resolution_dump_interval: float | None = None,
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
    if dynamics == "langevin":
        if langevin_seeds is None:
            raise ValueError("Langevin ensembles require one langevin seed per replicate")
        bath_seeds = [int(value) for value in langevin_seeds]
        if len(bath_seeds) != len(frames) or len(set(bath_seeds)) != len(bath_seeds):
            raise ValueError("langevin_seeds must be unique and align with frame_indices")
    elif dynamics == "nvt":
        if langevin_seeds is not None:
            raise ValueError("langevin_seeds are only valid when dynamics='langevin'")
        bath_seeds = [None] * len(frames)
    else:
        raise ValueError("dynamics must be 'nvt' or 'langevin'")
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
    for replicate_index, (frame_index, velocity_seed, langevin_seed) in enumerate(
        zip(frames, seeds, bath_seeds), start=1
    ):
        replicate_directory = output_directory / f"replicate_{replicate_index:02d}"
        replicate_manifest = prepare_replicate(
            source_path,
            replicate_directory,
            temperature=temperature,
            frame_index=frame_index,
            velocity_seed=velocity_seed,
            equilibration_time=equilibration_time,
            production_time=production_time,
            dynamics=dynamics,
            langevin_damping=langevin_damping,
            langevin_seed=langevin_seed,
            dump_interval_time=dump_interval_time,
            high_resolution_duration=high_resolution_duration,
            high_resolution_dump_interval=high_resolution_dump_interval,
            _payload=payload,
            _source_sha256=source_sha256,
        )
        replicate_manifests.append(
            {
                "replicate": replicate_index,
                "directory": replicate_directory.name,
                "source_frame_index": frame_index,
                "velocity_seed": velocity_seed,
                "langevin_seed": langevin_seed,
            }
        )

    manifest: dict[str, object] = {
        "source_doi": SOURCE_DOI,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "temperature": temperature,
        "dynamics": dynamics,
        "langevin_damping": langevin_damping,
        "langevin_seeds": bath_seeds if dynamics == "langevin" else None,
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
        "saved_frame_interval_tau": dump_interval_time,
        "high_resolution_duration_tau": high_resolution_duration,
        "high_resolution_saved_frame_interval_tau": high_resolution_dump_interval,
        "thermodynamic_claim_allowed": False,
    }
    temporary_manifest = output_directory / "ensemble_manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(output_directory / "ensemble_manifest.json")
    return manifest
