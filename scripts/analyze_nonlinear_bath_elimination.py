#!/usr/bin/env python3
"""Fail-closed analysis of the frozen nonlinear auxiliary-bath experiment."""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nonlinear_bath_diagnostics import (  # noqa: E402
    accepted_periodic_cage_events,
    classify_nonlinear_bath_gate,
    equilibrium_diagnostics,
    fit_constant_hazard,
    fit_delayed_square_hazard,
)
from nonlinear_bath_gle import (  # noqa: E402
    NonlinearBathControls,
    gibbs_one_step_moment_bias,
    gibbs_stationarity_audit,
    periodic_coupling,
    reconstruct_auxiliary_path,
)
from simulate_nonlinear_bath_elimination import (  # noqa: E402
    checkpoint_metadata,
    file_sha256,
    frozen_simulation_protocol,
    validate_checkpoint_payload,
)


_MODES = (
    "canary",
    "canary-half-step",
    "production",
    "null-constant-coupling",
    "null-no-bath",
)
_TRAJECTORY_COUNTS = {
    "canary": 16,
    "canary-half-step": 16,
    "production": 256,
    "null-constant-coupling": 256,
    "null-no-bath": 256,
}
_SURVIVAL_TIME = np.linspace(0.0, 100.0, 201)
_BROAD_CLAIMS = (
    "autonomous_single_particle_gle_allowed",
    "complete_event_clock_closure_allowed",
    "kramers_escape_claim_allowed",
    "spatial_facilitation_claim_allowed",
    "thermodynamic_claim_allowed",
)


def _scalar_text(value: object) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("cache bundle fields must be scalar")
    return str(array.item())


def _scalar_float(value: object) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("cache bundle fields must be scalar")
    number = float(array)
    if not math.isfinite(number):
        raise ValueError("cache bundle fields must be finite")
    return number


def validate_bundle_headers(
    headers: dict[str, dict[str, object]],
    *,
    simulator_sha256: str,
    gle_sha256: str,
) -> dict[str, float]:
    """Require the exact five-cache grid and one common committed source."""

    if set(headers) != set(_MODES):
        raise ValueError("cache bundle must contain the exact frozen mode grid")
    for mode in _MODES:
        row = headers[mode]
        if _scalar_text(row.get("frozen_simulation_protocol", "")) != mode:
            raise ValueError(f"cache bundle mode mismatch for {mode}")
        if _scalar_text(row.get("source_sha256", "")) != simulator_sha256:
            raise ValueError(f"cache bundle simulator source mismatch for {mode}")
        if _scalar_text(row.get("gle_source_sha256", "")) != gle_sha256:
            raise ValueError(f"cache bundle GLE source mismatch for {mode}")
        if _scalar_float(row.get("cache_complete", float("nan"))) != 1.0:
            raise ValueError(f"cache bundle is incomplete for {mode}")
        if _scalar_float(row.get("trajectory_count", float("nan"))) != float(
            _TRAJECTORY_COUNTS[mode]
        ):
            raise ValueError(f"cache bundle trajectory count mismatch for {mode}")
    return {
        "cache_grid_complete": 1.0,
        "training_trajectory_count": 128.0,
        "held_trajectory_count": 128.0,
    }


def load_complete_cache(path: Path, *, expected_mode: str) -> dict[str, object]:
    """Load one cache only after recomputing source and frozen provenance."""

    if expected_mode not in _MODES:
        raise ValueError("unknown expected nonlinear-bath cache mode")
    protocol = frozen_simulation_protocol(expected_mode)
    simulator_path = ROOT / "scripts" / "simulate_nonlinear_bath_elimination.py"
    simulator_hash = file_sha256(simulator_path)
    expected = checkpoint_metadata(
        frozen_simulation_protocol=expected_mode,
        source_sha256=simulator_hash,
        requested_step_count=int(protocol["requested_step_count"]),
    )
    with np.load(path, allow_pickle=False) as cache:
        payload = validate_checkpoint_payload(
            cache,
            provenance=expected,
            protocol=protocol,
        )
        if int(payload["completed_step_count"]) != int(
            protocol["requested_step_count"]
        ):
            raise ValueError("analysis requires a complete checkpoint cache")
        header = {
            "frozen_simulation_protocol": _scalar_text(
                cache["frozen_simulation_protocol"]
            ),
            "source_sha256": _scalar_text(cache["source_sha256"]),
            "gle_source_sha256": _scalar_text(cache["gle_source_sha256"]),
            "cache_complete": _scalar_float(cache["cache_complete"]),
            "trajectory_count": _scalar_float(cache["trajectory_count"]),
        }
    controls = protocol["controls"]
    if not isinstance(controls, NonlinearBathControls):
        raise ValueError("loaded protocol has invalid controls")
    return {
        **payload,
        **header,
        "path": str(Path(path).resolve()),
        "controls": controls,
        "event_sample_time": float(protocol["event_sample_time"]),
        "equilibrium_sample_time": float(protocol["equilibrium_sample_time"]),
        "potential_amplitude": float(protocol["potential_amplitude"]),
        "physical_barrier_height": float(protocol["physical_barrier_height"]),
        "production_time": float(protocol["production_steps"])
        * controls.time_step,
    }


def event_wait_table(
    records: list[dict[str, float]],
    *,
    horizon: float,
    trajectory_count: int,
) -> dict[str, np.ndarray]:
    """Recompute event waits and one terminal censoring interval per path."""

    if (
        not math.isfinite(horizon)
        or horizon <= 0.0
        or isinstance(trajectory_count, bool)
        or not isinstance(trajectory_count, (int, np.integer))
        or trajectory_count < 1
    ):
        raise ValueError("wait-table controls must be finite and physical")
    event_times: list[list[float]] = [[] for _ in range(trajectory_count)]
    for record in records:
        owner_raw = float(record["trajectory_index"])
        event_time = float(record["event_time"])
        if (
            not math.isfinite(owner_raw)
            or owner_raw != round(owner_raw)
            or owner_raw < 0
            or owner_raw >= trajectory_count
            or not math.isfinite(event_time)
            or event_time <= 0.0
            or event_time >= horizon
        ):
            raise ValueError("event records are outside the frozen path horizon")
        event_times[int(owner_raw)].append(event_time)
    waits: list[float] = []
    censored: list[bool] = []
    owners: list[int] = []
    for owner, times in enumerate(event_times):
        previous = 0.0
        for event_time in sorted(times):
            wait = event_time - previous
            if wait <= 0.0:
                raise ValueError("event times must be strictly increasing per path")
            waits.append(wait)
            censored.append(False)
            owners.append(owner)
            previous = event_time
        terminal = horizon - previous
        if terminal > 0.0:
            waits.append(terminal)
            censored.append(True)
            owners.append(owner)
    return {
        "waiting_time": np.asarray(waits, dtype=float),
        "censored": np.asarray(censored, dtype=bool),
        "trajectory_index": np.asarray(owners, dtype=int),
    }


def _hazard_negative_log_likelihood(
    waits: np.ndarray,
    censored: np.ndarray,
    *,
    rate: float,
    delay: float,
) -> float:
    events = ~censored
    event_count = int(np.sum(events))
    if rate <= 0.0 or event_count < 1:
        raise ValueError("held hazard score requires events and a positive rate")
    if delay == 0.0:
        return float(-event_count * math.log(rate) + rate * np.sum(waits))
    readiness = -np.expm1(-waits[events] / delay)
    exposure = (
        waits
        - 2.0 * delay * (-np.expm1(-waits / delay))
        + 0.5 * delay * (-np.expm1(-2.0 * waits / delay))
    )
    return float(
        -event_count * math.log(rate)
        - 2.0 * np.sum(np.log(readiness))
        + rate * np.sum(exposure)
    )


def _model_survival(time: np.ndarray, *, rate: float, delay: float) -> np.ndarray:
    if delay == 0.0:
        exposure = time
    else:
        exposure = (
            time
            - 2.0 * delay * (-np.expm1(-time / delay))
            + 0.5 * delay * (-np.expm1(-2.0 * time / delay))
        )
    return np.exp(-rate * exposure)


def _kaplan_meier(
    waits: np.ndarray,
    censored: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:
    survival = np.ones_like(time, dtype=float)
    product = 1.0
    event_times = np.unique(waits[~censored])
    event_slots: list[tuple[float, float]] = []
    for event_time in event_times:
        at_risk = int(np.sum(waits >= event_time))
        events = int(np.sum((waits == event_time) & (~censored)))
        if at_risk < 1 or events < 1:
            continue
        product *= 1.0 - events / at_risk
        event_slots.append((float(event_time), product))
    for slot, query in enumerate(time):
        current = 1.0
        for event_time, value in event_slots:
            if event_time > query:
                break
            current = value
        survival[slot] = current
    return survival


def score_hazard_models(
    wait_table: dict[str, np.ndarray],
    *,
    bootstrap_count: int = 400,
    bootstrap_seed: int = 20260812,
) -> dict[str, object]:
    """Fit on owners 0:128 and score only owners 128:256."""

    waits = np.asarray(wait_table["waiting_time"], dtype=float)
    censored = np.asarray(wait_table["censored"])
    owners = np.asarray(wait_table["trajectory_index"])
    if (
        waits.ndim != 1
        or censored.shape != waits.shape
        or owners.shape != waits.shape
        or censored.dtype.kind != "b"
        or np.any(~np.isfinite(waits))
        or np.any(waits <= 0.0)
        or np.any(owners != owners.astype(int))
        or np.any(owners < 0)
        or np.any(owners >= 256)
        or isinstance(bootstrap_count, bool)
        or not isinstance(bootstrap_count, (int, np.integer))
        or bootstrap_count < 1
    ):
        raise ValueError("hazard score table must be finite and cover 256 owners")
    if set(np.unique(owners.astype(int))) != set(range(256)):
        raise ValueError("hazard score requires every frozen trajectory owner")
    training = owners < 128
    held = ~training
    constant = fit_constant_hazard(waits[training], censored=censored[training])
    delayed = fit_delayed_square_hazard(waits[training], censored=censored[training])
    constant_rate = float(constant["rate"])
    delayed_rate = float(delayed["rate"])
    delay_time = float(delayed["delay_time"])
    held_constant_nll = _hazard_negative_log_likelihood(
        waits[held],
        censored[held],
        rate=constant_rate,
        delay=0.0,
    )
    held_delayed_nll = _hazard_negative_log_likelihood(
        waits[held],
        censored[held],
        rate=delayed_rate,
        delay=delay_time,
    )
    empirical = _kaplan_meier(waits[held], censored[held], _SURVIVAL_TIME)
    constant_survival = _model_survival(
        _SURVIVAL_TIME,
        rate=constant_rate,
        delay=0.0,
    )
    delayed_survival = _model_survival(
        _SURVIVAL_TIME,
        rate=delayed_rate,
        delay=delay_time,
    )
    constant_error = float(
        np.trapz(np.abs(empirical - constant_survival), _SURVIVAL_TIME) / 100.0
    )
    delayed_error = float(
        np.trapz(np.abs(empirical - delayed_survival), _SURVIVAL_TIME) / 100.0
    )

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_delays = np.empty(bootstrap_count, dtype=float)
    training_owners = owners[training].astype(int)
    for sample in range(bootstrap_count):
        sampled_owners = rng.integers(0, 128, size=128)
        selected_waits = []
        selected_censors = []
        for owner in sampled_owners:
            mask = training_owners == owner
            selected_waits.append(waits[training][mask])
            selected_censors.append(censored[training][mask])
        fitted = fit_delayed_square_hazard(
            np.concatenate(selected_waits),
            censored=np.concatenate(selected_censors),
        )
        bootstrap_delays[sample] = float(fitted["delay_time"])
    ci_low, ci_high = np.quantile(bootstrap_delays, [0.025, 0.975])
    return {
        "training_trajectory_count": 128.0,
        "held_trajectory_count": 128.0,
        "fit_uses_held_samples": 0.0,
        "training_wait_count": float(np.sum(training)),
        "held_wait_count": float(np.sum(held)),
        "training_event_count": float(np.sum(~censored[training])),
        "held_event_count": float(np.sum(~censored[held])),
        "constant_rate": constant_rate,
        "delayed_rate": delayed_rate,
        "delayed_time": delay_time,
        "held_constant_negative_log_likelihood": held_constant_nll,
        "held_delayed_negative_log_likelihood": held_delayed_nll,
        "held_constant_integrated_survival_error": constant_error,
        "held_delayed_integrated_survival_error": delayed_error,
        "bootstrap_resamples": float(bootstrap_count),
        "bootstrap_seed": float(bootstrap_seed),
        "delay_time_bootstrap_ci95_low": float(ci_low),
        "delay_time_bootstrap_ci95_high": float(ci_high),
        "survival_time": _SURVIVAL_TIME.copy(),
        "empirical_survival": empirical,
        "constant_survival": constant_survival,
        "delayed_survival": delayed_survival,
    }


def fixed_path_bath_replay(
    positions: np.ndarray,
    *,
    controls: NonlinearBathControls,
    sample_time: float,
    replay_count: int = 512,
    replay_seed: int = 20260813,
    permutation_seed: int = 20260814,
) -> dict[str, np.ndarray | float]:
    """Stream fixed-path OU replay without materializing all bath histories."""

    path = np.asarray(positions, dtype=float)
    required_steps = int(round(20.0 / sample_time)) + 1
    if (
        path.ndim != 2
        or path.shape[0] < required_steps
        or path.shape[1] < 32
        or np.any(~np.isfinite(path))
        or sample_time <= 0.0
        or replay_count < 2
    ):
        raise ValueError("fixed-path replay lacks the frozen path support")
    path = path[:required_steps, :32]
    lag_times = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
    lag_steps = np.rint(lag_times / sample_time).astype(int)
    if np.any(np.abs(lag_steps * sample_time - lag_times) > 1e-12):
        raise ValueError("stored path spacing does not resolve frozen replay lags")
    position_bin_count = 12
    half_period = 0.5 * controls.period
    wrapped = (path + half_period) % controls.period - half_period
    bins = np.clip(
        np.floor((wrapped + half_period) * position_bin_count / controls.period),
        0,
        position_bin_count - 1,
    ).astype(int)
    true_coupling = periodic_coupling(path, controls=controls)
    permutation = np.random.default_rng(permutation_seed).permutation(len(path))
    permuted_coupling = periodic_coupling(path[permutation], controls=controls)
    decay = np.exp(-controls.rates * sample_time)
    noise_scale = np.sqrt(controls.temperature * (1.0 - decay**2))
    rng = np.random.default_rng(replay_seed)
    bath = rng.normal(
        scale=math.sqrt(controls.temperature),
        size=(replay_count, 32, len(controls.rates)),
    )
    maximum_lag = int(np.max(lag_steps))
    force_history = np.empty((maximum_lag + 1, replay_count, 32), dtype=float)
    empirical_sum = np.zeros((len(lag_steps), position_bin_count), dtype=float)
    exact_sum = np.zeros_like(empirical_sum)
    permuted_sum = np.zeros_like(empirical_sum)
    counts = np.zeros_like(empirical_sum)
    for time_index in range(len(path)):
        coupling = true_coupling[time_index]
        force = np.sum(bath * coupling[None], axis=-1)
        force_history[time_index % len(force_history)] = force
        centered = force - np.mean(force, axis=0, keepdims=True)
        for lag_slot, lag in enumerate(lag_steps):
            if time_index < lag:
                continue
            past = force_history[(time_index - lag) % len(force_history)]
            past_centered = past - np.mean(past, axis=0, keepdims=True)
            empirical_sample = np.sum(centered * past_centered, axis=0) / (
                replay_count - 1
            )
            exact_sample = controls.temperature * np.sum(
                true_coupling[time_index]
                * np.exp(-controls.rates * lag * sample_time)
                * true_coupling[time_index - lag],
                axis=-1,
            )
            permuted_sample = controls.temperature * np.sum(
                permuted_coupling[time_index]
                * np.exp(-controls.rates * lag * sample_time)
                * permuted_coupling[time_index - lag],
                axis=-1,
            )
            current_bins = bins[time_index]
            for position_bin in range(position_bin_count):
                selected = current_bins == position_bin
                count = int(np.sum(selected))
                if count < 1:
                    continue
                empirical_sum[lag_slot, position_bin] += float(
                    np.sum(empirical_sample[selected])
                )
                exact_sum[lag_slot, position_bin] += float(
                    np.sum(exact_sample[selected])
                )
                permuted_sum[lag_slot, position_bin] += float(
                    np.sum(permuted_sample[selected])
                )
                counts[lag_slot, position_bin] += count
        if time_index + 1 < len(path):
            bath = decay * bath + noise_scale * rng.normal(size=bath.shape)
    if np.any(counts < 1):
        raise ValueError("fixed-path replay has an unsupported position bin")
    empirical = empirical_sum / counts
    exact = exact_sum / counts
    permuted = permuted_sum / counts
    lag_errors = np.linalg.norm(empirical - exact, axis=1) / np.maximum(
        np.linalg.norm(exact, axis=1),
        1e-30,
    )
    permuted_lag_errors = np.linalg.norm(empirical - permuted, axis=1) / np.maximum(
        np.linalg.norm(permuted, axis=1),
        1e-30,
    )
    pooled = float(
        np.linalg.norm(empirical - exact) / max(np.linalg.norm(exact), 1e-30)
    )
    permuted_pooled = float(
        np.linalg.norm(empirical - permuted)
        / max(np.linalg.norm(permuted), 1e-30)
    )
    return {
        "lag_times": lag_times,
        "lagwise_normalized_rmse": lag_errors,
        "pooled_normalized_rmse": pooled,
        "time_permuted_lagwise_normalized_rmse": permuted_lag_errors,
        "time_permuted_pooled_normalized_rmse": permuted_pooled,
        "position_bin_support": counts,
        "bath_replay_gate_pass": float(
            pooled <= 0.15 and bool(np.all(lag_errors <= 0.30))
        ),
        "replay_path_count": 32.0,
        "replay_history_count": float(replay_count),
        "replay_duration": 20.0,
        "replay_seed": float(replay_seed),
        "time_permutation_seed": float(permutation_seed),
    }


def _polyline(
    values_x: np.ndarray,
    values_y: np.ndarray,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_max: float,
    color: str,
) -> str:
    x = left + width * (values_x - x_min) / max(x_max - x_min, 1e-30)
    y = top + height * (1.0 - values_y / max(y_max, 1e-30))
    points = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(x, y))
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'


def render_gate_svg(
    *,
    equilibrium_rows: list[dict[str, object]],
    replay_rows: list[dict[str, object]],
    survival_time: np.ndarray,
    empirical_survival: np.ndarray,
    constant_survival: np.ndarray,
    delayed_survival: np.ndarray,
) -> str:
    """Render three dynamically scaled panels without hidden axis truncation."""

    equilibrium_values = np.asarray(
        [float(row["normalized_error"]) for row in equilibrium_rows]
    )
    replay_values = np.asarray(
        [float(row["normalized_error"]) for row in replay_rows]
    )
    arrays = (
        equilibrium_values,
        replay_values,
        np.asarray(survival_time, dtype=float),
        np.asarray(empirical_survival, dtype=float),
        np.asarray(constant_survival, dtype=float),
        np.asarray(delayed_survival, dtype=float),
    )
    if any(array.size < 1 or np.any(~np.isfinite(array)) for array in arrays):
        raise ValueError("SVG inputs must be nonempty and finite")
    width, height = 1500.0, 520.0
    panel_width, panel_height = 390.0, 320.0
    top = 80.0
    lefts = (80.0, 555.0, 1030.0)
    eq_ymax = max(1.1, 1.10 * float(np.max(equilibrium_values)))
    replay_ymax = max(1.1, 1.10 * float(np.max(replay_values)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2933;letter-spacing:0} .axis{stroke:#52606d;stroke-width:1}</style>',
        '<text x="750" y="30" text-anchor="middle" font-size="20">Nonlinear auxiliary-bath microscopic gate</text>',
    ]
    for left in lefts:
        parts.append(
            f'<path class="axis" d="M {left} {top} V {top + panel_height} H {left + panel_width}" fill="none"/>'
        )
    parts.extend(
        [
            f'<text x="{lefts[0] + panel_width / 2}" y="58" text-anchor="middle" font-size="15">normalized equilibrium error</text>',
            f'<text x="{lefts[1] + panel_width / 2}" y="58" text-anchor="middle" font-size="15">bath-level FDT replay normalized error</text>',
            f'<text x="{lefts[2] + panel_width / 2}" y="58" text-anchor="middle" font-size="15">held survival probability</text>',
        ]
    )
    for panel_left, ymax in ((lefts[0], eq_ymax), (lefts[1], replay_ymax)):
        tolerance_y = top + panel_height * (1.0 - 1.0 / ymax)
        parts.append(
            f'<path d="M {panel_left} {tolerance_y:.2f} H {panel_left + panel_width}" stroke="#c2410c" stroke-dasharray="6 4"/>'
        )
        parts.append(
            f'<text x="{panel_left + panel_width - 4}" y="{tolerance_y - 6:.2f}" text-anchor="end" font-size="12">tolerance = 1</text>'
        )
        parts.append(
            f'<text x="{panel_left - 8}" y="{top + 4}" text-anchor="end" font-size="11">{ymax:.3g}</text>'
        )
        parts.append(
            f'<text x="{panel_left - 8}" y="{top + panel_height + 4}" text-anchor="end" font-size="11">0</text>'
        )
    eq_x = np.arange(len(equilibrium_values), dtype=float)
    replay_x = np.arange(len(replay_values), dtype=float)
    parts.append(
        _polyline(
            eq_x,
            equilibrium_values,
            left=lefts[0],
            top=top,
            width=panel_width,
            height=panel_height,
            x_min=0.0,
            x_max=max(float(len(eq_x) - 1), 1.0),
            y_max=eq_ymax,
            color="#2563eb",
        )
    )
    parts.append(
        _polyline(
            replay_x,
            replay_values,
            left=lefts[1],
            top=top,
            width=panel_width,
            height=panel_height,
            x_min=0.0,
            x_max=max(float(len(replay_x) - 1), 1.0),
            y_max=replay_ymax,
            color="#047857",
        )
    )
    for index, row in enumerate(equilibrium_rows):
        label = html.escape(str(row["metric"]))
        x = lefts[0] + panel_width * index / max(len(equilibrium_rows) - 1, 1)
        parts.append(
            f'<text x="{x:.2f}" y="{top + panel_height + 18}" text-anchor="middle" font-size="10">{label}</text>'
        )
    for index, row in enumerate(replay_rows):
        label = html.escape(f'{float(row["lag_time"]):g}')
        x = lefts[1] + panel_width * index / max(len(replay_rows) - 1, 1)
        parts.append(
            f'<text x="{x:.2f}" y="{top + panel_height + 18}" text-anchor="middle" font-size="10">{label}</text>'
        )
    for values, color in (
        (np.asarray(empirical_survival), "#111827"),
        (np.asarray(constant_survival), "#b91c1c"),
        (np.asarray(delayed_survival), "#7c3aed"),
    ):
        parts.append(
            _polyline(
                np.asarray(survival_time),
                values,
                left=lefts[2],
                top=top,
                width=panel_width,
                height=panel_height,
                x_min=float(np.min(survival_time)),
                x_max=float(np.max(survival_time)),
                y_max=1.0,
                color=color,
            )
        )
    parts.extend(
        [
            f'<text x="{lefts[1] + panel_width / 2}" y="{top + panel_height + 42}" text-anchor="middle" font-size="12">lag time (tau)</text>',
            f'<text x="{lefts[2] + panel_width / 2}" y="{top + panel_height + 42}" text-anchor="middle" font-size="12">time (tau)</text>',
            '<text x="1040" y="430" font-size="11" fill="#111827">empirical</text>',
            '<text x="1120" y="430" font-size="11" fill="#b91c1c">constant</text>',
            '<text x="1200" y="430" font-size="11" fill="#7c3aed">delayed</text>',
            '<text x="750" y="485" text-anchor="middle" font-size="13">broad microscopic and thermodynamic claims remain closed</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty analysis CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _equilibrium_rows(
    mode: str,
    diagnostic: dict[str, np.ndarray | float],
) -> list[dict[str, object]]:
    return [
        {
            "record": "equilibrium",
            "mode": mode,
            "metric": "momentum",
            "value": diagnostic["momentum_temperature_relative_error"],
            "tolerance": 0.05,
            "normalized_error": float(
                diagnostic["momentum_temperature_relative_error"]
            )
            / 0.05,
        },
        {
            "record": "equilibrium",
            "mode": mode,
            "metric": "auxiliary",
            "value": diagnostic["maximum_auxiliary_temperature_relative_error"],
            "tolerance": 0.05,
            "normalized_error": float(
                diagnostic["maximum_auxiliary_temperature_relative_error"]
            )
            / 0.05,
        },
        {
            "record": "equilibrium",
            "mode": mode,
            "metric": "p-z correlation",
            "value": diagnostic["maximum_momentum_auxiliary_correlation"],
            "tolerance": 0.05,
            "normalized_error": float(
                diagnostic["maximum_momentum_auxiliary_correlation"]
            )
            / 0.05,
        },
        {
            "record": "equilibrium",
            "mode": mode,
            "metric": "position TV",
            "value": diagnostic["position_gibbs_total_variation"],
            "tolerance": 0.08,
            "normalized_error": float(diagnostic["position_gibbs_total_variation"])
            / 0.08,
        },
    ]


def _cache_equilibrium(cache: dict[str, object]) -> dict[str, np.ndarray | float]:
    controls = cache["controls"]
    if not isinstance(controls, NonlinearBathControls):
        raise ValueError("cache controls are invalid")
    positions = np.asarray(cache["equilibrium_positions"])
    momenta = np.asarray(cache["equilibrium_momenta"])
    auxiliary = np.asarray(cache["equilibrium_auxiliary"])
    return equilibrium_diagnostics(
        positions.reshape(-1),
        momenta.reshape(-1),
        auxiliary.reshape(-1, len(controls.rates)),
        controls=controls,
        position_bin_count=40,
    )


def _reconstruction_error(cache: dict[str, object]) -> float:
    controls = cache["controls"]
    if not isinstance(controls, NonlinearBathControls):
        raise ValueError("cache controls are invalid")
    positions = np.asarray(cache["equilibrium_positions"])
    momenta = np.asarray(cache["equilibrium_momenta"])
    auxiliary = np.asarray(cache["equilibrium_auxiliary"])
    reconstructed = reconstruct_auxiliary_path(
        auxiliary[0],
        positions=positions[:-1],
        momenta=momenta[:-1],
        normal_increments=np.asarray(cache["canary_normal_z"]),
        controls=controls,
    )
    return float(
        np.max(np.abs(reconstructed - auxiliary))
        / max(float(np.max(np.abs(auxiliary))), math.sqrt(controls.temperature), 1e-30)
    )


def canary_preflight(
    canary: dict[str, object],
    half_step: dict[str, object],
) -> dict[str, float | str]:
    """Authorize production only after both exact-OU paths reconstruct."""

    canary_error = _reconstruction_error(canary)
    half_step_error = _reconstruction_error(half_step)
    maximum_error = max(canary_error, half_step_error)
    stationarity_results = []
    for cache in (canary, half_step):
        controls = cache["controls"]
        if not isinstance(controls, NonlinearBathControls):
            raise ValueError("canary controls are invalid")
        positions = np.asarray(cache["equilibrium_positions"])
        momenta = np.asarray(cache["equilibrium_momenta"])
        auxiliary = np.asarray(cache["equilibrium_auxiliary"])
        stationarity_results.append(
            gibbs_stationarity_audit(
                positions.reshape(-1),
                momenta.reshape(-1),
                auxiliary.reshape(-1, len(controls.rates)),
                controls=controls,
            )
        )
    maximum_stationarity_residual = max(
        float(result["maximum_normalized_stationarity_residual"])
        for result in stationarity_results
    )
    gibbs_derived = all(
        float(
            result["periodic_quotient_gibbs_invariant_density_derived"]
        )
        == 1.0
        for result in stationarity_results
    )
    full_bias = gibbs_one_step_moment_bias(controls=canary["controls"])
    half_bias = gibbs_one_step_moment_bias(controls=half_step["controls"])

    def contraction_ratio(half_value: float, full_value: float) -> float:
        if full_value == 0.0:
            return 0.0 if half_value == 0.0 else float("inf")
        return abs(half_value / full_value)

    momentum_bias_ratio = contraction_ratio(
        float(half_bias["momentum_variance_bias"]),
        float(full_bias["momentum_variance_bias"]),
    )
    auxiliary_bias_ratio = float(
        np.max(
            np.divide(
                np.abs(half_bias["auxiliary_variance_bias"]),
                np.abs(full_bias["auxiliary_variance_bias"]),
                out=np.zeros_like(half_bias["auxiliary_variance_bias"]),
                where=np.abs(full_bias["auxiliary_variance_bias"]) > 0.0,
            )
        )
    )
    passed = maximum_error <= 5e-11 and gibbs_derived
    return {
        "record": "canary_preflight",
        "canary_reconstruction_relative_error": canary_error,
        "half_step_reconstruction_relative_error": half_step_error,
        "maximum_reconstruction_relative_error": maximum_error,
        "reconstruction_tolerance": 5e-11,
        "maximum_normalized_stationarity_residual": (
            maximum_stationarity_residual
        ),
        "periodic_quotient_gibbs_invariant_density_derived": float(
            gibbs_derived
        ),
        "unwrapped_position_gibbs_probability_allowed": 0.0,
        "discrete_scheme_exact_gibbs_preserving": float(
            full_bias["discrete_scheme_exact_gibbs_preserving"]
        ),
        "full_step_momentum_variance_bias": float(
            full_bias["momentum_variance_bias"]
        ),
        "half_step_momentum_variance_bias": float(
            half_bias["momentum_variance_bias"]
        ),
        "half_step_momentum_bias_ratio": momentum_bias_ratio,
        "full_step_maximum_auxiliary_variance_bias": float(
            np.max(np.abs(full_bias["auxiliary_variance_bias"]))
        ),
        "half_step_maximum_auxiliary_variance_bias": float(
            np.max(np.abs(half_bias["auxiliary_variance_bias"]))
        ),
        "half_step_auxiliary_bias_maximum_ratio": auxiliary_bias_ratio,
        "analytic_half_step_local_bias_contraction_pass": float(
            momentum_bias_ratio <= 0.251 and auxiliary_bias_ratio <= 0.251
        ),
        "finite_state_and_provenance_validated": 1.0,
        "canary_preflight_pass": float(passed),
        "exact_nonlinear_bath_elimination_supported": 0.0,
        "synthetic_bath_level_fdt_replay_supported": 0.0,
        "synthetic_delayed_hazard_emerges": 0.0,
        "real_ka_position_dependent_kernel_authorized": 0.0,
        **{claim: 0.0 for claim in _BROAD_CLAIMS},
    }


def analyze_bundle(cache_paths: dict[str, Path], *, output_prefix: Path) -> dict[str, object]:
    """Recompute every gate and write details, summary, survival, and SVG."""

    if set(cache_paths) != set(_MODES):
        raise ValueError("analysis requires the exact cache path grid")
    caches = {
        mode: load_complete_cache(cache_paths[mode], expected_mode=mode)
        for mode in _MODES
    }
    simulator_hash = file_sha256(
        ROOT / "scripts" / "simulate_nonlinear_bath_elimination.py"
    )
    gle_hash = file_sha256(ROOT / "src" / "nonlinear_bath_gle.py")
    header_gate = validate_bundle_headers(
        {
            mode: {
                key: caches[mode][key]
                for key in (
                    "frozen_simulation_protocol",
                    "source_sha256",
                    "gle_source_sha256",
                    "cache_complete",
                    "trajectory_count",
                )
            }
            for mode in _MODES
        },
        simulator_sha256=simulator_hash,
        gle_sha256=gle_hash,
    )
    diagnostics = {mode: _cache_equilibrium(cache) for mode, cache in caches.items()}
    preflight = canary_preflight(
        caches["canary"],
        caches["canary-half-step"],
    )
    if float(preflight["canary_preflight_pass"]) != 1.0:
        raise ValueError("full analysis rejects a failed canary preflight")
    full_reconstruction = float(
        preflight["canary_reconstruction_relative_error"]
    )
    half_reconstruction = float(
        preflight["half_step_reconstruction_relative_error"]
    )
    maximum_reconstruction = float(
        preflight["maximum_reconstruction_relative_error"]
    )
    canary_rows = _equilibrium_rows("canary", diagnostics["canary"])
    half_rows = _equilibrium_rows(
        "canary-half-step",
        diagnostics["canary-half-step"],
    )
    half_step_not_worse = all(
        float(half["normalized_error"]) <= float(full["normalized_error"])
        for full, half in zip(canary_rows, half_rows)
    )
    production = caches["production"]
    controls = production["controls"]
    if not isinstance(controls, NonlinearBathControls):
        raise ValueError("production controls are invalid")
    replay = fixed_path_bath_replay(
        np.asarray(production["event_positions"]),
        controls=controls,
        sample_time=float(production["event_sample_time"]),
    )
    records = accepted_periodic_cage_events(
        np.asarray(production["event_positions"]),
        sample_time=float(production["event_sample_time"]),
        period=controls.period,
        persistence=0.10,
    )
    waits = event_wait_table(
        records,
        horizon=float(production["production_time"]),
        trajectory_count=256,
    )
    hazard = score_hazard_models(waits)
    verdict = classify_nonlinear_bath_gate(
        maximum_reconstruction_relative_error=maximum_reconstruction,
        half_step_equilibrium_not_worse=half_step_not_worse,
        equilibrium_gate_pass=bool(
            diagnostics["production"]["equilibrium_gate_pass"]
        ),
        bath_replay_gate_pass=bool(replay["bath_replay_gate_pass"]),
        held_constant_negative_log_likelihood=float(
            hazard["held_constant_negative_log_likelihood"]
        ),
        held_delayed_negative_log_likelihood=float(
            hazard["held_delayed_negative_log_likelihood"]
        ),
        held_constant_integrated_survival_error=float(
            hazard["held_constant_integrated_survival_error"]
        ),
        held_delayed_integrated_survival_error=float(
            hazard["held_delayed_integrated_survival_error"]
        ),
        delay_time_bootstrap_ci95_low=float(
            hazard["delay_time_bootstrap_ci95_low"]
        ),
    )
    details: list[dict[str, object]] = []
    for mode in _MODES:
        details.extend(_equilibrium_rows(mode, diagnostics[mode]))
    replay_rows = []
    for lag, error, permuted_error in zip(
        replay["lag_times"],
        replay["lagwise_normalized_rmse"],
        replay["time_permuted_lagwise_normalized_rmse"],
    ):
        row = {
            "record": "bath_replay",
            "mode": "production",
            "metric": "fixed_path_covariance",
            "lag_time": float(lag),
            "normalized_error": float(error),
            "time_permuted_normalized_error": float(permuted_error),
            "tolerance": 0.30,
        }
        replay_rows.append(row)
        details.append(row)
    mode_records = {"production": records}
    mode_hazards = {"production": hazard}
    for mode in ("production", "null-constant-coupling", "null-no-bath"):
        cache = caches[mode]
        local_controls = cache["controls"]
        if not isinstance(local_controls, NonlinearBathControls):
            raise ValueError("null controls are invalid")
        if mode in mode_records:
            local_records = mode_records[mode]
        else:
            local_records = accepted_periodic_cage_events(
                np.asarray(cache["event_positions"]),
                sample_time=float(cache["event_sample_time"]),
                period=local_controls.period,
                persistence=0.10,
            )
            mode_records[mode] = local_records
        if mode not in mode_hazards:
            local_waits = event_wait_table(
                local_records,
                horizon=float(cache["production_time"]),
                trajectory_count=256,
            )
            mode_hazards[mode] = score_hazard_models(local_waits)
        local_hazard = mode_hazards[mode]
        details.append(
            {
                "record": "event_count",
                "mode": mode,
                "metric": "accepted_nonrecrossing_events",
                "value": float(len(local_records)),
                "event_definition_persistence": 0.10,
            }
        )
        for model in ("constant", "delayed"):
            details.append(
                {
                    "record": "hazard_score",
                    "mode": mode,
                    "metric": f"held_{model}_hazard",
                    "held_negative_log_likelihood": local_hazard[
                        f"held_{model}_negative_log_likelihood"
                    ],
                    "held_integrated_survival_error": local_hazard[
                        f"held_{model}_integrated_survival_error"
                    ],
                    "fit_uses_held_samples": 0.0,
                }
            )
    null_summary: dict[str, object] = {}
    for mode, prefix in (
        ("null-constant-coupling", "constant_coupling_null"),
        ("null-no-bath", "no_bath_null"),
    ):
        local = mode_hazards[mode]
        null_summary.update(
            {
                f"{prefix}_accepted_event_count": float(len(mode_records[mode])),
                f"{prefix}_held_constant_negative_log_likelihood": local[
                    "held_constant_negative_log_likelihood"
                ],
                f"{prefix}_held_delayed_negative_log_likelihood": local[
                    "held_delayed_negative_log_likelihood"
                ],
                f"{prefix}_held_constant_integrated_survival_error": local[
                    "held_constant_integrated_survival_error"
                ],
                f"{prefix}_held_delayed_integrated_survival_error": local[
                    "held_delayed_integrated_survival_error"
                ],
                f"{prefix}_delay_time": local["delayed_time"],
                f"{prefix}_delay_time_bootstrap_ci95_low": local[
                    "delay_time_bootstrap_ci95_low"
                ],
                f"{prefix}_delayed_preference_descriptive": float(
                    float(local["held_delayed_negative_log_likelihood"])
                    < float(local["held_constant_negative_log_likelihood"])
                    and float(local["held_delayed_integrated_survival_error"])
                    <= 0.90
                    * float(local["held_constant_integrated_survival_error"])
                    and float(local["delay_time_bootstrap_ci95_low"]) > 1e-3
                ),
            }
        )
    summary = {
        "record": "verdict",
        **header_gate,
        "simulator_sha256": simulator_hash,
        "gle_source_sha256": gle_hash,
        "analyzer_sha256": file_sha256(Path(__file__)),
        "potential_amplitude": production["potential_amplitude"],
        "physical_barrier_height": production["physical_barrier_height"],
        "canary_reconstruction_relative_error": full_reconstruction,
        "half_step_reconstruction_relative_error": half_reconstruction,
        "maximum_reconstruction_relative_error": maximum_reconstruction,
        "maximum_normalized_stationarity_residual": preflight[
            "maximum_normalized_stationarity_residual"
        ],
        "periodic_quotient_gibbs_invariant_density_derived": preflight[
            "periodic_quotient_gibbs_invariant_density_derived"
        ],
        "unwrapped_position_gibbs_probability_allowed": 0.0,
        "discrete_scheme_exact_gibbs_preserving": preflight[
            "discrete_scheme_exact_gibbs_preserving"
        ],
        "full_step_momentum_variance_bias": preflight[
            "full_step_momentum_variance_bias"
        ],
        "half_step_momentum_variance_bias": preflight[
            "half_step_momentum_variance_bias"
        ],
        "half_step_momentum_bias_ratio": preflight[
            "half_step_momentum_bias_ratio"
        ],
        "full_step_maximum_auxiliary_variance_bias": preflight[
            "full_step_maximum_auxiliary_variance_bias"
        ],
        "half_step_maximum_auxiliary_variance_bias": preflight[
            "half_step_maximum_auxiliary_variance_bias"
        ],
        "half_step_auxiliary_bias_maximum_ratio": preflight[
            "half_step_auxiliary_bias_maximum_ratio"
        ],
        "analytic_half_step_local_bias_contraction_pass": preflight[
            "analytic_half_step_local_bias_contraction_pass"
        ],
        "half_step_equilibrium_not_worse": float(half_step_not_worse),
        "bath_replay_pooled_normalized_rmse": replay[
            "pooled_normalized_rmse"
        ],
        "time_permuted_replay_pooled_normalized_rmse": replay[
            "time_permuted_pooled_normalized_rmse"
        ],
        "replay_path_count": replay["replay_path_count"],
        "replay_history_count": replay["replay_history_count"],
        "replay_duration": replay["replay_duration"],
        "accepted_event_count": float(len(records)),
        **null_summary,
        **{
            key: value
            for key, value in hazard.items()
            if np.asarray(value).shape == ()
        },
        **verdict,
    }
    for claim in _BROAD_CLAIMS:
        if float(summary[claim]) != 0.0:
            raise ValueError(f"broad claim unexpectedly opened: {claim}")
    survival_rows = [
        {
            "time": float(time),
            "empirical_survival": float(empirical),
            "constant_survival": float(constant),
            "delayed_survival": float(delayed),
            "split": "held_trajectories_128_255",
        }
        for time, empirical, constant, delayed in zip(
            hazard["survival_time"],
            hazard["empirical_survival"],
            hazard["constant_survival"],
            hazard["delayed_survival"],
        )
    ]
    details_path = Path(f"{output_prefix}_details.csv")
    summary_path = Path(f"{output_prefix}_summary.csv")
    survival_path = Path(f"{output_prefix}_survival.csv")
    svg_path = Path(f"{output_prefix}.svg")
    _write_csv(details_path, details)
    _write_csv(summary_path, [summary])
    _write_csv(survival_path, survival_rows)
    svg = render_gate_svg(
        equilibrium_rows=_equilibrium_rows("production", diagnostics["production"]),
        replay_rows=replay_rows,
        survival_time=np.asarray(hazard["survival_time"]),
        empirical_survival=np.asarray(hazard["empirical_survival"]),
        constant_survival=np.asarray(hazard["constant_survival"]),
        delayed_survival=np.asarray(hazard["delayed_survival"]),
    )
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg, encoding="utf-8")
    return {
        **summary,
        "details_path": str(details_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "survival_path": str(survival_path.resolve()),
        "svg_path": str(svg_path.resolve()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--canary-cache", type=Path, required=True)
    parser.add_argument("--half-step-cache", type=Path, required=True)
    parser.add_argument("--production-cache", type=Path, required=True)
    parser.add_argument("--constant-coupling-cache", type=Path, required=True)
    parser.add_argument("--no-bath-cache", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = analyze_bundle(
        {
            "canary": args.canary_cache,
            "canary-half-step": args.half_step_cache,
            "production": args.production_cache,
            "null-constant-coupling": args.constant_coupling_cache,
            "null-no-bath": args.no_bath_cache,
        },
        output_prefix=args.output_prefix,
    )
    print(result)


if __name__ == "__main__":
    main()
