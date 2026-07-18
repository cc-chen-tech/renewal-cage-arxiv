"""Continuous transient-periodic Langevin dynamics and cage diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransientPeriodicParams:
    """Thermal parameters for a tagged coordinate and slow cage environment."""

    temperature: float
    period: float
    base_barrier: float
    elastic_stiffness: float
    barrier_stiffness: float
    barrier_coupling: float
    gamma_x: float
    gamma_q: float
    gamma_z: float

    def __post_init__(self) -> None:
        positive = (
            "temperature",
            "period",
            "base_barrier",
            "barrier_stiffness",
            "gamma_x",
            "gamma_q",
            "gamma_z",
        )
        nonnegative = ("elastic_stiffness", "barrier_coupling")
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")


def _validated_state(
    x: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tagged = np.asarray(x, dtype=float)
    environment = np.asarray(q, dtype=float)
    barrier = np.asarray(z, dtype=float)
    if (
        tagged.ndim != 2
        or environment.shape != tagged.shape
        or barrier.shape != (tagged.shape[0],)
        or tagged.shape[0] == 0
        or tagged.shape[1] == 0
        or np.any(~np.isfinite(tagged))
        or np.any(~np.isfinite(environment))
        or np.any(~np.isfinite(barrier))
    ):
        raise ValueError("x, q, and z must be aligned nonempty finite arrays")
    return tagged, environment, barrier


def potential_energy(
    x: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    params: TransientPeriodicParams,
) -> np.ndarray:
    """Evaluate the translationally invariant transient periodic potential."""

    tagged, environment, barrier_coordinate = _validated_state(x, q, z)
    phase = 2.0 * np.pi * tagged / params.period
    barrier = params.base_barrier + params.barrier_coupling * barrier_coordinate**2
    periodic = 0.5 * barrier[:, None] * (1.0 - np.cos(phase))
    elastic = 0.5 * params.elastic_stiffness * (tagged - environment) ** 2
    return (
        np.sum(periodic + elastic, axis=1)
        + 0.5 * params.barrier_stiffness * barrier_coordinate**2
    )


def conservative_forces(
    x: np.ndarray,
    q: np.ndarray,
    z: np.ndarray,
    params: TransientPeriodicParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return negative gradients of the transient periodic potential."""

    tagged, environment, barrier_coordinate = _validated_state(x, q, z)
    phase = 2.0 * np.pi * tagged / params.period
    barrier = params.base_barrier + params.barrier_coupling * barrier_coordinate**2
    difference = tagged - environment
    force_x = (
        -barrier[:, None] * np.pi / params.period * np.sin(phase)
        - params.elastic_stiffness * difference
    )
    force_q = params.elastic_stiffness * difference
    force_z = (
        -params.barrier_stiffness * barrier_coordinate
        - params.barrier_coupling
        * barrier_coordinate
        * np.sum(1.0 - np.cos(phase), axis=1)
    )
    return force_x, force_q, force_z


def simulate_transient_periodic_langevin(
    params: TransientPeriodicParams,
    *,
    trajectory_count: int,
    dimension: int,
    dt: float,
    burnin_steps: int,
    production_steps: int,
    record_stride: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Integrate the coupled overdamped Langevin equations with FDT noise."""

    integer_inputs = (
        ("trajectory_count", trajectory_count),
        ("dimension", dimension),
        ("burnin_steps", burnin_steps),
        ("production_steps", production_steps),
        ("record_stride", record_stride),
        ("seed", seed),
    )
    for name, value in integer_inputs:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if trajectory_count < 1 or dimension < 1:
        raise ValueError("trajectory_count and dimension must be positive")
    if burnin_steps < 0 or production_steps < 1 or record_stride < 1:
        raise ValueError("integration step counts are outside their domains")
    if production_steps % record_stride != 0:
        raise ValueError("production_steps must be divisible by record_stride")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    reference_curvature = (
        2.0 * math.pi**2 * params.base_barrier / params.period**2
        + params.elastic_stiffness
    )
    stability_number = dt * reference_curvature / params.gamma_x
    if stability_number >= 0.2:
        raise ValueError("Euler stability bound requires dt*kappa/gamma_x < 0.2")

    seed_streams = np.random.SeedSequence(seed).spawn(4)
    initial_barrier_rng, x_rng, q_rng, z_rng = (
        np.random.default_rng(stream) for stream in seed_streams
    )
    x = np.zeros((trajectory_count, dimension), dtype=float)
    q = np.zeros_like(x)
    z = initial_barrier_rng.normal(
        scale=math.sqrt(params.temperature / params.barrier_stiffness),
        size=trajectory_count,
    )
    x_noise = math.sqrt(2.0 * params.temperature * dt / params.gamma_x)
    q_noise = math.sqrt(2.0 * params.temperature * dt / params.gamma_q)
    z_noise = math.sqrt(2.0 * params.temperature * dt / params.gamma_z)
    maximum_euler_displacement = 0.0

    def advance() -> None:
        nonlocal x, q, z, maximum_euler_displacement
        force_x, force_q, force_z = conservative_forces(x, q, z, params)
        delta_x = (
            force_x * dt / params.gamma_x
            + x_noise * x_rng.normal(size=x.shape)
        )
        if params.elastic_stiffness > 0.0:
            delta_q = (
                force_q * dt / params.gamma_q
                + q_noise * q_rng.normal(size=q.shape)
            )
        else:
            delta_q = np.zeros_like(q)
        delta_z = (
            force_z * dt / params.gamma_z
            + z_noise * z_rng.normal(size=z.shape)
        )
        x += delta_x
        q += delta_q
        z += delta_z
        maximum_euler_displacement = max(
            maximum_euler_displacement,
            float(np.max(np.abs(delta_x))),
        )
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(q)) or np.any(~np.isfinite(z)):
            raise FloatingPointError("nonfinite state encountered during integration")

    for _ in range(burnin_steps):
        advance()

    record_count = production_steps // record_stride + 1
    positions = np.empty((record_count, trajectory_count, dimension), dtype=float)
    environment_positions = np.empty_like(positions)
    barrier_coordinates = np.empty((record_count, trajectory_count), dtype=float)
    positions[0] = x
    environment_positions[0] = q
    barrier_coordinates[0] = z
    record_index = 1
    for step in range(1, production_steps + 1):
        advance()
        if step % record_stride == 0:
            positions[record_index] = x
            environment_positions[record_index] = q
            barrier_coordinates[record_index] = z
            record_index += 1
    return {
        "positions": positions,
        "environment_positions": environment_positions,
        "barrier_coordinates": barrier_coordinates,
        "record_dt": dt * record_stride,
        "reference_stability_number": stability_number,
        "maximum_euler_displacement": maximum_euler_displacement,
        "all_finite": 1.0,
    }


def stable_cage_events(
    positions: np.ndarray,
    *,
    period: float,
    dwell_frames: int,
    frame_dt: float = 1.0,
) -> dict[str, np.ndarray]:
    """Extract nearest-well transitions that survive a non-recrossing dwell."""

    path = np.asarray(positions, dtype=float)
    if (
        path.ndim != 3
        or path.shape[0] < 2
        or path.shape[1] < 1
        or path.shape[2] < 1
        or np.any(~np.isfinite(path))
    ):
        raise ValueError("positions must be a finite frames-by-trajectories-by-dimension array")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("period must be positive and finite")
    if isinstance(dwell_frames, bool) or not isinstance(dwell_frames, int) or dwell_frames < 1:
        raise ValueError("dwell_frames must be a positive integer")
    if not math.isfinite(frame_dt) or frame_dt <= 0.0:
        raise ValueError("frame_dt must be positive and finite")
    cage_indices = np.floor(path / period + 0.5).astype(np.int64)
    frames: list[int] = []
    times: list[float] = []
    trajectories: list[int] = []
    dimensions: list[int] = []
    signed_steps: list[int] = []
    vectors: list[np.ndarray] = []
    for trajectory in range(path.shape[1]):
        for component in range(path.shape[2]):
            values = cage_indices[:, trajectory, component]
            current = int(values[0])
            for frame in range(1, len(values)):
                candidate = int(values[frame])
                if candidate == current or frame + dwell_frames > len(values):
                    continue
                if np.all(values[frame : frame + dwell_frames] == candidate):
                    step = candidate - current
                    vector = np.zeros(path.shape[2], dtype=float)
                    vector[component] = step * period
                    frames.append(frame)
                    times.append(frame * frame_dt)
                    trajectories.append(trajectory)
                    dimensions.append(component)
                    signed_steps.append(step)
                    vectors.append(vector)
                    current = candidate
    if frames:
        order = np.lexsort((np.asarray(dimensions), np.asarray(trajectories), np.asarray(frames)))
        vector_array = np.asarray(vectors, dtype=float)[order]
    else:
        order = np.empty(0, dtype=int)
        vector_array = np.empty((0, path.shape[2]), dtype=float)
    return {
        "frame": np.asarray(frames, dtype=int)[order],
        "time": np.asarray(times, dtype=float)[order],
        "trajectory": np.asarray(trajectories, dtype=int)[order],
        "dimension": np.asarray(dimensions, dtype=int)[order],
        "signed_cage_step": np.asarray(signed_steps, dtype=int)[order],
        "vector_step": vector_array,
    }


def event_clock_statistics(
    events: dict[str, np.ndarray],
    *,
    trajectory_count: int,
    dimension: int,
    duration: float,
    count_window: float,
) -> dict[str, float]:
    """Summarize event-count fluctuations, waiting times, and vector recoil."""

    if (
        isinstance(trajectory_count, bool)
        or not isinstance(trajectory_count, int)
        or trajectory_count < 1
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 1
        or not math.isfinite(duration)
        or duration <= 0.0
        or not math.isfinite(count_window)
        or count_window <= 0.0
    ):
        raise ValueError("event-clock dimensions and times must be positive")
    times = np.asarray(events["time"], dtype=float)
    trajectories = np.asarray(events["trajectory"], dtype=int)
    vectors = np.asarray(events["vector_step"], dtype=float)
    if (
        times.ndim != 1
        or trajectories.shape != times.shape
        or vectors.shape != (len(times), dimension)
        or np.any(~np.isfinite(times))
        or np.any(~np.isfinite(vectors))
        or np.any(times < 0.0)
        or np.any(trajectories < 0)
        or np.any(trajectories >= trajectory_count)
    ):
        raise ValueError("event arrays are not aligned with the requested ensemble")
    window_count = int(math.floor(duration / count_window + 1e-12))
    if window_count < 1:
        raise ValueError("duration must contain at least one count window")
    counts = np.zeros((trajectory_count, window_count), dtype=int)
    retained = times < window_count * count_window
    if np.any(retained):
        windows = np.floor(times[retained] / count_window).astype(int)
        np.add.at(counts, (trajectories[retained], windows), 1)
    mean_count = float(np.mean(counts))
    count_fano = float(np.var(counts) / mean_count) if mean_count > 0.0 else 0.0

    persistence: list[float] = []
    exchange: list[float] = []
    pair_dot: list[float] = []
    pair_norm: list[float] = []
    for trajectory in range(trajectory_count):
        indices = np.flatnonzero(trajectories == trajectory)
        if len(indices) == 0:
            continue
        local_order = indices[np.argsort(times[indices])]
        local_times = times[local_order]
        persistence.append(float(local_times[0]))
        if len(local_times) > 1:
            exchange.extend(np.diff(local_times).tolist())
            first = vectors[local_order[:-1]]
            second = vectors[local_order[1:]]
            pair_dot.extend(np.sum(first * second, axis=1).tolist())
            pair_norm.extend(np.sum(first**2, axis=1).tolist())
    persistence_supported = bool(persistence)
    exchange_supported = bool(exchange)
    mean_persistence = float(np.mean(persistence)) if persistence else math.nan
    mean_exchange = float(np.mean(exchange)) if exchange else math.nan
    correlation = (
        float(np.sum(pair_dot) / np.sum(pair_norm))
        if pair_norm and float(np.sum(pair_norm)) > 0.0
        else math.nan
    )
    return {
        "event_count": float(len(times)),
        "count_sample_count": float(counts.size),
        "mean_window_count": mean_count,
        "count_fano_factor": count_fano,
        "persistence_supported": float(persistence_supported),
        "exchange_supported": float(exchange_supported),
        "mean_persistence_time": mean_persistence,
        "mean_exchange_time": mean_exchange,
        "persistence_exchange_ratio": (
            mean_persistence / mean_exchange
            if persistence_supported and exchange_supported and mean_exchange > 0.0
            else math.nan
        ),
        "successive_vector_pair_count": float(len(pair_dot)),
        "successive_vector_correlation": correlation,
    }


def displacement_observables(
    positions: np.ndarray,
    *,
    lag_frames: list[int] | tuple[int, ...] | np.ndarray,
    wave_numbers: list[float] | tuple[float, ...] | np.ndarray,
) -> list[dict[str, float]]:
    """Compute time-origin averaged MSD, NGP, and component scattering."""

    path = np.asarray(positions, dtype=float)
    lags = np.asarray(lag_frames)
    waves = np.asarray(wave_numbers, dtype=float)
    if (
        path.ndim != 3
        or path.shape[0] < 2
        or path.shape[1] < 1
        or path.shape[2] < 1
        or np.any(~np.isfinite(path))
        or lags.ndim != 1
        or len(lags) == 0
        or np.any(lags != np.floor(lags))
        or np.any(lags < 1)
        or np.any(lags >= path.shape[0])
        or waves.ndim != 1
        or len(waves) == 0
        or np.any(~np.isfinite(waves))
        or np.any(waves <= 0.0)
    ):
        raise ValueError("trajectory, lags, and wave numbers are invalid")
    dimension = path.shape[2]
    rows: list[dict[str, float]] = []
    for lag_value in lags.astype(int):
        displacement = path[lag_value:] - path[:-lag_value]
        squared = np.sum(displacement**2, axis=2)
        msd = float(np.mean(squared))
        fourth = float(np.mean(squared**2))
        ngp = (
            dimension * fourth / ((dimension + 2.0) * msd**2) - 1.0
            if msd > 0.0
            else 0.0
        )
        row = {
            "lag_frames": float(lag_value),
            "sample_count": float(squared.size),
            "msd": msd,
            "fourth_moment": fourth,
            "ngp": ngp,
        }
        for wave_number in waves:
            key = f"fs_k{wave_number:g}".replace(".", "p")
            row[key] = float(np.mean(np.cos(wave_number * displacement)))
        rows.append(row)
    return rows
