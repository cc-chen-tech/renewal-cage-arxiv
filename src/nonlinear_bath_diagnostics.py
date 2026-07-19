"""Physical gates for nonlinear auxiliary-bath elimination experiments."""

from __future__ import annotations

import math

import numpy as np

from nonlinear_bath_gle import (
    NonlinearBathControls,
    eliminated_memory_kernel,
    periodic_potential,
)


_DELAY_TIME_GRID = np.geomspace(1e-3, 1e2, 801)
_CLOSED_CLAIMS = {
    "autonomous_single_particle_gle_allowed": 0.0,
    "complete_event_clock_closure_allowed": 0.0,
    "kramers_escape_claim_allowed": 0.0,
    "spatial_facilitation_claim_allowed": 0.0,
    "thermodynamic_claim_allowed": 0.0,
}


def _finite_aligned_state(
    position: np.ndarray,
    momentum: np.ndarray,
    auxiliary: np.ndarray,
    controls: NonlinearBathControls,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(position, dtype=float)
    velocity = np.asarray(momentum, dtype=float)
    bath = np.asarray(auxiliary, dtype=float)
    if (
        value.size < 1
        or velocity.shape != value.shape
        or bath.shape != value.shape + (len(controls.rates),)
        or np.any(~np.isfinite(value))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(bath))
    ):
        raise ValueError("equilibrium state arrays must be finite and aligned")
    if controls.temperature <= 0.0:
        raise ValueError("equilibrium diagnostics require positive temperature")
    return value, velocity, bath


def equilibrium_diagnostics(
    position: np.ndarray,
    momentum: np.ndarray,
    auxiliary: np.ndarray,
    *,
    controls: NonlinearBathControls,
    position_bin_count: int,
) -> dict[str, np.ndarray | float]:
    """Compare sampled one-point statistics with the invariant Gibbs density."""

    value, velocity, bath = _finite_aligned_state(
        position,
        momentum,
        auxiliary,
        controls,
    )
    if (
        isinstance(position_bin_count, bool)
        or not isinstance(position_bin_count, (int, np.integer))
        or position_bin_count < 2
    ):
        raise ValueError("position_bin_count must be an integer at least two")
    temperature = controls.temperature
    momentum_error = abs(float(np.mean(velocity**2)) / temperature - 1.0)
    bath_ratios = np.mean(bath**2, axis=tuple(range(bath.ndim - 1))) / temperature
    bath_error = float(np.max(np.abs(bath_ratios - 1.0)))
    centered_velocity = velocity.ravel() - float(np.mean(velocity))
    correlations = []
    for mode in range(bath.shape[-1]):
        centered_bath = bath[..., mode].ravel() - float(np.mean(bath[..., mode]))
        denominator = math.sqrt(
            float(np.mean(centered_velocity**2))
            * float(np.mean(centered_bath**2))
        )
        if denominator <= 0.0:
            raise ValueError("equilibrium correlations need nonzero variance")
        correlations.append(
            float(np.mean(centered_velocity * centered_bath)) / denominator
        )
    maximum_correlation = float(np.max(np.abs(correlations)))

    half_period = 0.5 * controls.period
    wrapped = (value.ravel() + half_period) % controls.period - half_period
    edges = np.linspace(-half_period, half_period, position_bin_count + 1)
    counts = np.histogram(wrapped, bins=edges)[0].astype(float)
    empirical = counts / np.sum(counts)
    centers = 0.5 * (edges[1:] + edges[:-1])
    log_weight = -periodic_potential(
        centers,
        barrier=controls.barrier,
        period=controls.period,
    ) / temperature
    weight = np.exp(log_weight - float(np.max(log_weight)))
    expected = weight / np.sum(weight)
    total_variation = 0.5 * float(np.sum(np.abs(empirical - expected)))
    passed = (
        momentum_error <= 0.05
        and bath_error <= 0.05
        and maximum_correlation <= 0.05
        and total_variation <= 0.08
    )
    return {
        "momentum_temperature_relative_error": momentum_error,
        "auxiliary_temperature_relative_error": np.abs(bath_ratios - 1.0),
        "maximum_auxiliary_temperature_relative_error": bath_error,
        "momentum_auxiliary_correlation": np.asarray(correlations),
        "maximum_momentum_auxiliary_correlation": maximum_correlation,
        "position_histogram": empirical,
        "position_gibbs_probability": expected,
        "position_gibbs_total_variation": total_variation,
        "position_bin_count": float(position_bin_count),
        "equilibrium_gate_pass": float(passed),
        **_CLOSED_CLAIMS,
    }


def _position_bins(
    positions: np.ndarray,
    *,
    period: float,
    count: int,
) -> np.ndarray:
    half_period = 0.5 * period
    wrapped = (positions + half_period) % period - half_period
    indices = np.floor((wrapped + half_period) * count / period).astype(int)
    return np.clip(indices, 0, count - 1)


def bath_replay_covariance(
    positions: np.ndarray,
    replay_forces: np.ndarray,
    *,
    lag_steps: np.ndarray,
    controls: NonlinearBathControls,
    position_bin_count: int,
) -> dict[str, np.ndarray | float]:
    """Compare fixed-path independent-bath replay covariance with the kernel."""

    paths = np.asarray(positions, dtype=float)
    forces = np.asarray(replay_forces, dtype=float)
    lags_raw = np.asarray(lag_steps)
    if (
        paths.ndim != 2
        or forces.ndim != 3
        or forces.shape[1:] != paths.shape
        or forces.shape[0] < 2
        or np.any(~np.isfinite(paths))
        or np.any(~np.isfinite(forces))
        or lags_raw.ndim != 1
        or len(lags_raw) < 1
        or np.any(lags_raw != lags_raw.astype(int))
        or np.any(lags_raw < 1)
        or np.any(lags_raw >= len(paths))
        or isinstance(position_bin_count, bool)
        or not isinstance(position_bin_count, (int, np.integer))
        or position_bin_count < 2
    ):
        raise ValueError("bath replay arrays, bins, and lags must be finite and aligned")
    lags = lags_raw.astype(int)
    empirical = np.empty((len(lags), position_bin_count), dtype=float)
    exact = np.empty_like(empirical)
    counts = np.empty_like(empirical)
    lag_errors = np.empty(len(lags), dtype=float)
    centered_forces = forces - np.mean(forces, axis=0, keepdims=True)
    for slot, lag in enumerate(lags):
        left = paths[lag:]
        right = paths[:-lag]
        products = centered_forces[:, lag:] * centered_forces[:, :-lag]
        bins = _position_bins(
            left,
            period=controls.period,
            count=position_bin_count,
        )
        exact_samples = controls.temperature * np.asarray(
            eliminated_memory_kernel(
                left,
                right,
                lag=lag * controls.time_step,
                controls=controls,
            )["kernel"]
        )
        for position_bin in range(position_bin_count):
            selected = bins == position_bin
            sample_count = int(np.sum(selected))
            if sample_count < 1:
                raise ValueError("every fixed replay position bin needs support")
            empirical[slot, position_bin] = float(np.mean(products[:, selected]))
            exact[slot, position_bin] = float(np.mean(exact_samples[selected]))
            counts[slot, position_bin] = float(sample_count)
        lag_errors[slot] = float(
            np.linalg.norm(empirical[slot] - exact[slot])
            / max(np.linalg.norm(exact[slot]), 1e-30)
        )
    pooled_error = float(
        np.linalg.norm(empirical - exact) / max(np.linalg.norm(exact), 1e-30)
    )
    passed = pooled_error <= 0.15 and bool(np.all(lag_errors <= 0.30))
    return {
        "empirical_covariance": empirical,
        "exact_covariance": exact,
        "position_bin_support": counts,
        "lag_steps": lags,
        "lag_times": lags * controls.time_step,
        "lagwise_normalized_rmse": lag_errors,
        "pooled_normalized_rmse": pooled_error,
        "bath_replay_gate_pass": float(passed),
        **_CLOSED_CLAIMS,
    }


def accepted_periodic_cage_events(
    positions: np.ndarray,
    *,
    sample_time: float,
    period: float,
    persistence: float,
) -> list[dict[str, float]]:
    """Return boundary crossings that remain in the new periodic cage."""

    values = np.asarray(positions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if (
        values.ndim != 2
        or len(values) < 2
        or np.any(~np.isfinite(values))
        or not math.isfinite(sample_time)
        or sample_time <= 0.0
        or not math.isfinite(period)
        or period <= 0.0
        or not math.isfinite(persistence)
        or persistence <= 0.0
    ):
        raise ValueError("event paths and controls must be finite and physical")
    persistence_steps = int(math.ceil(persistence / sample_time))
    cages = np.floor(values / period + 0.5).astype(int)
    records: list[dict[str, float]] = []
    for trajectory in range(values.shape[1]):
        current = int(cages[0, trajectory])
        previous_event_time = 0.0
        index = 1
        while index < len(values):
            candidate = int(cages[index, trajectory])
            if candidate == current:
                index += 1
                continue
            stop = index + persistence_steps + 1
            if stop <= len(values) and np.all(
                cages[index:stop, trajectory] == candidate
            ):
                event_time = index * sample_time
                records.append(
                    {
                        "trajectory_index": float(trajectory),
                        "event_frame": float(index),
                        "event_time": float(event_time),
                        "waiting_time": float(event_time - previous_event_time),
                        "from_cage": float(current),
                        "to_cage": float(candidate),
                    }
                )
                previous_event_time = event_time
                current = candidate
                index = stop
            else:
                index += 1
    return records


def _validated_waits(
    waiting_times: np.ndarray,
    censored: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    waits = np.asarray(waiting_times, dtype=float)
    censor = np.asarray(censored)
    if (
        waits.ndim != 1
        or len(waits) < 1
        or censor.shape != waits.shape
        or censor.dtype.kind != "b"
        or np.any(~np.isfinite(waits))
        or np.any(waits <= 0.0)
    ):
        raise ValueError("waiting times and censor flags must be finite and aligned")
    events = ~censor
    if not np.any(events):
        raise ValueError("hazard fit requires at least one observed event")
    return waits, censor.astype(bool), events


def fit_constant_hazard(
    waiting_times: np.ndarray,
    *,
    censored: np.ndarray,
) -> dict[str, float | str]:
    """Fit the exact censored exponential maximum-likelihood rate."""

    waits, _, events = _validated_waits(waiting_times, censored)
    event_count = int(np.sum(events))
    exposure = float(np.sum(waits))
    rate = event_count / exposure
    negative_log_likelihood = -event_count * math.log(rate) + rate * exposure
    return {
        "model": "constant_hazard",
        "rate": float(rate),
        "delay_time": 0.0,
        "event_count": float(event_count),
        "total_exposure": exposure,
        "negative_log_likelihood": float(negative_log_likelihood),
        "fit_uses_held_samples": 0.0,
        **_CLOSED_CLAIMS,
    }


def _delayed_exposure(waits: np.ndarray, delay: float) -> np.ndarray:
    scaled = waits / delay
    return (
        waits
        - 2.0 * delay * (-np.expm1(-scaled))
        + 0.5 * delay * (-np.expm1(-2.0 * scaled))
    )


def fit_delayed_square_hazard(
    waiting_times: np.ndarray,
    *,
    censored: np.ndarray,
) -> dict[str, float | str]:
    """Fit the frozen square-delayed hazard by profiled grid likelihood."""

    waits, _, events = _validated_waits(waiting_times, censored)
    event_count = int(np.sum(events))
    best: tuple[float, float, float, float] | None = None
    for delay in _DELAY_TIME_GRID:
        exposure = float(np.sum(_delayed_exposure(waits, float(delay))))
        if exposure <= 0.0:
            continue
        rate = event_count / exposure
        readiness = -np.expm1(-waits[events] / float(delay))
        if np.any(readiness <= 0.0):
            continue
        negative_log_likelihood = (
            -event_count * math.log(rate)
            - 2.0 * float(np.sum(np.log(readiness)))
            + rate * exposure
        )
        candidate = (negative_log_likelihood, float(delay), rate, exposure)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError("delayed hazard grid has no finite likelihood")
    return {
        "model": "delayed_square_hazard",
        "rate": float(best[2]),
        "delay_time": float(best[1]),
        "event_count": float(event_count),
        "total_integrated_readiness_exposure": float(best[3]),
        "negative_log_likelihood": float(best[0]),
        "delay_grid_minimum": float(_DELAY_TIME_GRID[0]),
        "delay_grid_maximum": float(_DELAY_TIME_GRID[-1]),
        "delay_grid_count": float(len(_DELAY_TIME_GRID)),
        "fit_uses_held_samples": 0.0,
        **_CLOSED_CLAIMS,
    }


def classify_nonlinear_bath_gate(
    *,
    maximum_reconstruction_relative_error: float,
    half_step_equilibrium_not_worse: bool,
    equilibrium_gate_pass: bool,
    bath_replay_gate_pass: bool,
    held_constant_negative_log_likelihood: float,
    held_delayed_negative_log_likelihood: float,
    held_constant_integrated_survival_error: float,
    held_delayed_integrated_survival_error: float,
    delay_time_bootstrap_ci95_low: float,
) -> dict[str, float | str]:
    """Apply the frozen synthetic authorization and delayed-clock gates."""

    numeric = (
        maximum_reconstruction_relative_error,
        held_constant_negative_log_likelihood,
        held_delayed_negative_log_likelihood,
        held_constant_integrated_survival_error,
        held_delayed_integrated_survival_error,
        delay_time_bootstrap_ci95_low,
    )
    if (
        any(not math.isfinite(float(value)) for value in numeric)
        or maximum_reconstruction_relative_error < 0.0
        or held_constant_integrated_survival_error < 0.0
        or held_delayed_integrated_survival_error < 0.0
        or not isinstance(half_step_equilibrium_not_worse, (bool, np.bool_))
        or not isinstance(equilibrium_gate_pass, (bool, np.bool_))
        or not isinstance(bath_replay_gate_pass, (bool, np.bool_))
    ):
        raise ValueError("nonlinear bath gate inputs must be finite and physical")
    exact = (
        maximum_reconstruction_relative_error <= 5e-11
        and bool(half_step_equilibrium_not_worse)
    )
    replay = exact and bool(bath_replay_gate_pass)
    authorized = exact and bool(equilibrium_gate_pass) and replay
    delayed = (
        held_delayed_negative_log_likelihood
        < held_constant_negative_log_likelihood
        and held_delayed_integrated_survival_error
        <= 0.90 * held_constant_integrated_survival_error
        and delay_time_bootstrap_ci95_low > 0.0
    )
    return {
        "record": "verdict",
        "maximum_reconstruction_relative_error": float(
            maximum_reconstruction_relative_error
        ),
        "half_step_equilibrium_not_worse": float(
            half_step_equilibrium_not_worse
        ),
        "equilibrium_gate_pass": float(equilibrium_gate_pass),
        "bath_replay_gate_pass": float(bath_replay_gate_pass),
        "held_constant_negative_log_likelihood": float(
            held_constant_negative_log_likelihood
        ),
        "held_delayed_negative_log_likelihood": float(
            held_delayed_negative_log_likelihood
        ),
        "held_constant_integrated_survival_error": float(
            held_constant_integrated_survival_error
        ),
        "held_delayed_integrated_survival_error": float(
            held_delayed_integrated_survival_error
        ),
        "delay_time_bootstrap_ci95_low": float(
            delay_time_bootstrap_ci95_low
        ),
        "exact_nonlinear_bath_elimination_supported": float(exact),
        "synthetic_bath_level_fdt_replay_supported": float(replay),
        "synthetic_delayed_hazard_emerges": float(delayed),
        "real_ka_position_dependent_kernel_authorized": float(authorized),
        **_CLOSED_CLAIMS,
    }
