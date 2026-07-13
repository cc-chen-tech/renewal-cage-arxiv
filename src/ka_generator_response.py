"""Low-disk matched-response tools for exact KA force-generator paths."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ka_local_cage import ka_lj_force_generator_observables, ka_lj_second_force_generator
from ka_replicates import load_lammps_custom_trajectory


def generator_response_lammps_input(
    *,
    parent_restart: Path,
    target_id: int,
    displacement: float,
    temperature: float,
    friction: float,
    velocity_seed: int,
    langevin_seed: int,
    run_steps: int,
    dump_interval_steps: int,
    trajectory_name: str,
) -> str:
    """Build one serial full-state matched-response KA Langevin input."""

    if isinstance(target_id, bool) or not isinstance(target_id, int) or target_id < 1:
        raise ValueError("target_id must be a positive integer")
    if not math.isfinite(displacement) or displacement == 0.0:
        raise ValueError("displacement must be finite and nonzero")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(friction) or friction <= 0.0:
        raise ValueError("friction must be finite and positive")
    for name, seed in (("velocity_seed", velocity_seed), ("langevin_seed", langevin_seed)):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(run_steps, bool) or not isinstance(run_steps, int) or run_steps < 1:
        raise ValueError("run_steps must be a positive integer")
    if isinstance(dump_interval_steps, bool) or not isinstance(dump_interval_steps, int) or dump_interval_steps < 1:
        raise ValueError("dump_interval_steps must be a positive integer")
    if run_steps % dump_interval_steps:
        raise ValueError("run_steps must be divisible by dump_interval_steps")
    if not trajectory_name or any(character.isspace() for character in trajectory_name):
        raise ValueError("trajectory_name must be nonempty and contain no whitespace")

    return f"""units lj
atom_style atomic
processors 1 1 1
read_restart {Path(parent_restart).resolve()}

reset_timestep 0
group tagged id {target_id}
displace_atoms tagged move {displacement:g} 0 0 units box
velocity all create {temperature:g} {velocity_seed} mom yes rot no dist gaussian
fix integrator all nve
fix bath all langevin {temperature:g} {temperature:g} {friction:g} {langevin_seed}
timestep 0.001
thermo 200
thermo_style custom step time temp pe ke etotal press

dump trajectory all custom {dump_interval_steps} {trajectory_name} id type x y z ix iy iz vx vy vz
dump_modify trajectory sort id format float %.17g
run {run_steps}
"""


def extract_generator_response_path(
    path: Path,
    *,
    target_id: int,
    temperature: float,
    friction: float,
    integration_time_step: float,
    directional_step: float,
    potential_protocol: str = "ka_lj_cut",
) -> dict[str, np.ndarray | float]:
    """Extract exact tagged `F`, `LF`, and `L2F` from one full-state path."""

    if isinstance(target_id, bool) or not isinstance(target_id, int) or target_id < 1:
        raise ValueError("target_id must be a positive integer")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(integration_time_step) or integration_time_step <= 0.0:
        raise ValueError("integration_time_step must be finite and positive")
    if not math.isfinite(directional_step) or directional_step <= 0.0:
        raise ValueError("directional_step must be finite and positive")

    trajectory = load_lammps_custom_trajectory(Path(path))
    if "velocities" not in trajectory:
        raise ValueError("trajectory must contain full particle velocities")
    positions = np.asarray(trajectory["unwrapped_positions"], dtype=float)
    velocities = np.asarray(trajectory["velocities"], dtype=float)
    particle_types = np.asarray(trajectory["particle_types"], dtype=int)
    box_lengths = np.asarray(trajectory["box_lengths"], dtype=float)
    timesteps = np.asarray(trajectory["timesteps"], dtype=np.int64)
    target_index = target_id - 1
    if target_index >= positions.shape[1]:
        raise ValueError("target_id lies outside the trajectory atom table")
    intervals = np.diff(timesteps)
    if len(intervals) == 0 or not np.all(intervals == intervals[0]):
        raise ValueError("saved trajectory timesteps must be uniform")

    target = np.array([target_index], dtype=int)
    force = np.empty((len(positions), 3), dtype=float)
    generator = np.empty_like(force)
    second = np.empty_like(force)
    covariance_rate = np.empty((len(positions), 3, 3), dtype=float)
    target_pair_active = np.empty((len(positions), positions.shape[1]), dtype=bool)
    target_pair_hessian = np.empty((len(positions), positions.shape[1], 3, 3), dtype=np.float32)
    nearest_cutoff_signed_gap = np.empty(len(positions), dtype=float)
    nearest_cutoff_particle_index = np.empty(len(positions), dtype=np.int64)
    for frame, (frame_positions, frame_velocities) in enumerate(zip(positions, velocities)):
        observable = ka_lj_force_generator_observables(
            frame_positions,
            velocities=frame_velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=friction,
            temperature=temperature,
            potential_protocol=potential_protocol,
        )
        force[frame] = observable["force"][0]
        generator[frame] = observable["force_generator"][0]
        covariance_rate[frame] = observable["force_generator_noise_covariance_rate"][0]
        target_pair_active[frame] = observable["target_pair_active"][0]
        target_pair_hessian[frame] = observable["target_pair_hessian"][0]
        nearest_cutoff_signed_gap[frame] = observable["nearest_cutoff_signed_gap"][0]
        nearest_cutoff_particle_index[frame] = observable["nearest_cutoff_particle_index"][0]
        second[frame] = ka_lj_second_force_generator(
            frame_positions,
            velocities=frame_velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=friction,
            directional_step=directional_step,
            potential_protocol=potential_protocol,
        )[0]

    frame_time = float(intervals[0]) * integration_time_step
    return {
        "time": (timesteps - timesteps[0]).astype(float) * integration_time_step,
        "position": positions[:, target_index],
        "velocity": velocities[:, target_index],
        "force": force,
        "force_generator": generator,
        "second_force_generator": second,
        "force_generator_noise_covariance_rate": covariance_rate,
        "target_pair_active": target_pair_active,
        "target_pair_hessian": target_pair_hessian,
        "nearest_cutoff_signed_gap": nearest_cutoff_signed_gap,
        "nearest_cutoff_particle_index": nearest_cutoff_particle_index,
        "potential_protocol": np.asarray(potential_protocol),
        "frame_time": frame_time,
        "target_id": float(target_id),
        "temperature": float(temperature),
        "friction": float(friction),
        "thermodynamic_claim_allowed": 0.0,
    }


def _rk4_linear_transition(generator: np.ndarray, frame_time: float) -> np.ndarray:
    transition = np.eye(len(generator))
    term = np.eye(len(generator))
    scaled = frame_time * generator
    for order in range(1, 5):
        term = term @ scaled / order
        transition += term
    return transition


def matched_generator_response(
    positive: dict[str, np.ndarray],
    negative: dict[str, np.ndarray],
    *,
    epsilon: float,
) -> dict[str, np.ndarray | float]:
    """Return the central tangent response in fixed `(x,v,F,LF)` order."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    positive_time = np.asarray(positive["time"], dtype=float)
    negative_time = np.asarray(negative["time"], dtype=float)
    if positive_time.ndim != 1 or len(positive_time) < 2 or not np.array_equal(positive_time, negative_time):
        raise ValueError("matched paths must have identical one-dimensional time grids")
    responses: dict[str, np.ndarray] = {}
    for key in ("position", "velocity", "force", "force_generator", "second_force_generator"):
        plus = np.asarray(positive[key], dtype=float)
        minus = np.asarray(negative[key], dtype=float)
        if plus.shape != (len(positive_time), 3) or minus.shape != plus.shape:
            raise ValueError(f"matched {key} arrays must have shape (frames, 3)")
        if np.any(~np.isfinite(plus)) or np.any(~np.isfinite(minus)):
            raise ValueError(f"matched {key} arrays must be finite")
        responses[key] = (plus - minus) / (2.0 * epsilon)
    return {
        "time": positive_time,
        "state_response": np.concatenate(
            [responses[key] for key in ("position", "velocity", "force", "force_generator")], axis=1
        ),
        "second_force_response": responses["second_force_generator"],
        "thermodynamic_claim_allowed": 0.0,
    }


def right_censored_tangent_interval_mask(
    frame_mismatch: np.ndarray,
    *,
    stride: int,
    interval_count: int,
) -> np.ndarray:
    """Retain only intervals ending before the first non-smooth frame.

    A hard-cutoff branch mismatch changes the subsequent matched paths, so
    deleting only the interval containing the crossing is not sufficient.
    """

    mismatch = np.asarray(frame_mismatch, dtype=bool)
    if mismatch.ndim != 1:
        raise ValueError("frame_mismatch must be one-dimensional")
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if (
        isinstance(interval_count, bool)
        or not isinstance(interval_count, (int, np.integer))
        or interval_count < 1
    ):
        raise ValueError("interval_count must be a positive integer")
    if interval_count * stride >= len(mismatch):
        raise ValueError("frame_mismatch does not cover all tangent interval endpoints")

    crossing = np.flatnonzero(mismatch)
    if len(crossing) == 0:
        return np.ones(interval_count, dtype=bool)
    interval_end_frames = stride * np.arange(1, interval_count + 1)
    return interval_end_frames < int(crossing[0])


def tangent_force_generator_noise_covariance_rate(
    positive_pair_hessian: np.ndarray,
    negative_pair_hessian: np.ndarray,
    *,
    epsilon: float,
    friction: float,
    temperature: float,
) -> np.ndarray:
    """Return the exact common-noise covariance rate of `delta LF`."""

    positive = np.asarray(positive_pair_hessian, dtype=float)
    negative = np.asarray(negative_pair_hessian, dtype=float)
    if positive.shape != negative.shape or positive.ndim < 3 or positive.shape[-2:] != (3, 3):
        raise ValueError("pair Hessians must have matching (..., particles, 3, 3) shape")
    if np.any(~np.isfinite(positive)) or np.any(~np.isfinite(negative)):
        raise ValueError("pair Hessians must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and nonnegative")
    delta_pair = (positive - negative) / (2.0 * epsilon)
    delta_diagonal = np.sum(delta_pair, axis=-3)
    covariance_geometry = np.einsum("...ab,...cb->...ac", delta_diagonal, delta_diagonal)
    covariance_geometry += np.einsum("...nab,...ncb->...ac", delta_pair, delta_pair)
    return 2.0 * friction * temperature * covariance_geometry


def tangent_noise_covariance_diagnostic(
    residual: np.ndarray,
    integrated_covariance: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Calibrate drift-subtracted tangent increments against exact FDT covariance."""

    values = np.asarray(residual, dtype=float)
    covariance = np.asarray(integrated_covariance, dtype=float)
    squeeze_input = values.ndim == 2
    if squeeze_input:
        values = values[None, ...]
        covariance = covariance[None, ...]
    if values.ndim != 3 or values.shape[1] < 2 or values.shape[2] != 3:
        raise ValueError("residual must have shape (members, at least 2 intervals, 3)")
    if covariance.shape != values.shape[:2] + (3, 3):
        raise ValueError("integrated covariance must align with residual and have shape (..., 3, 3)")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(covariance)):
        raise ValueError("residual and covariance must be finite")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError("integrated covariance must be symmetric")
    if valid_mask is None:
        valid = np.ones(values.shape[:2], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if squeeze_input and valid.ndim == 1:
            valid = valid[None, ...]
        if valid.shape != values.shape[:2]:
            raise ValueError("valid_mask must align with member and interval axes")
    if np.sum(valid) < 4:
        raise ValueError("at least four valid intervals are required")

    selected_values = values[valid]
    selected_covariance = covariance[valid]
    eigenvalues, eigenvectors = np.linalg.eigh(selected_covariance)
    scale = np.max(eigenvalues, axis=1, keepdims=True)
    if np.any(scale <= 0.0) or np.any(eigenvalues < -1e-10 * scale):
        raise ValueError("integrated covariance must be positive semidefinite")
    regularized = np.maximum(eigenvalues, 1e-12 * scale)
    projected = np.einsum("nrc,nr->nc", eigenvectors, selected_values)
    whitened_eigenbasis = projected / np.sqrt(regularized)
    whitened = np.einsum("nrc,nc->nr", eigenvectors, whitened_eigenbasis)
    squared_mahalanobis = np.sum(whitened_eigenbasis**2, axis=1)

    observed_energy = np.sum(selected_values**2, axis=1)
    predicted_energy = np.trace(selected_covariance, axis1=1, axis2=2)
    predicted_total = float(np.sum(predicted_energy))
    trace_ratio = float(np.sum(observed_energy) / predicted_total) if predicted_total > 0.0 else float("nan")
    if np.std(observed_energy) > 0.0 and np.std(predicted_energy) > 0.0:
        energy_correlation = float(np.corrcoef(observed_energy, predicted_energy)[0, 1])
    else:
        energy_correlation = float("nan")

    component_excess = np.full(3, np.nan, dtype=float)
    for component in range(3):
        centered = whitened[:, component] - np.mean(whitened[:, component])
        variance = float(np.mean(centered**2))
        if variance > 0.0:
            component_excess[component] = float(np.mean(centered**4) / variance**2 - 3.0)

    whitened_full = np.full_like(values, np.nan, dtype=float)
    whitened_full[valid] = whitened
    consecutive = valid[:, :-1] & valid[:, 1:]
    lag_source = whitened_full[:, :-1][consecutive].reshape(-1)
    lag_target = whitened_full[:, 1:][consecutive].reshape(-1)
    if len(lag_source) and np.std(lag_source) > 0.0 and np.std(lag_target) > 0.0:
        whitened_lag1 = float(np.corrcoef(lag_source, lag_target)[0, 1])
    else:
        whitened_lag1 = float("nan")

    summed_value = np.sum(selected_values, axis=0)
    summed_covariance = np.sum(selected_covariance, axis=0)
    summed_mean_mahalanobis = float(summed_value @ np.linalg.pinv(summed_covariance) @ summed_value)
    return {
        "trace_variance_ratio": trace_ratio,
        "mean_squared_mahalanobis": float(np.mean(squared_mahalanobis)),
        "summed_mean_squared_mahalanobis": summed_mean_mahalanobis,
        "observed_predicted_energy_correlation": energy_correlation,
        "whitened_residual": whitened,
        "whitened_component_excess_kurtosis": component_excess,
        "whitened_max_abs_component_excess_kurtosis": float(np.nanmax(np.abs(component_excess))),
        "whitened_lag1_correlation": whitened_lag1,
        "minimum_covariance_eigenvalue": float(np.min(eigenvalues)),
        "valid_sample_count": float(np.sum(valid)),
        "thermodynamic_claim_allowed": 0.0,
    }


def _identity_error(derivative: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    target_norm = float(np.linalg.norm(target))
    relative_l2 = float(np.linalg.norm(derivative - target) / target_norm) if target_norm > 0.0 else float("nan")
    derivative_flat = derivative.reshape(-1)
    target_flat = target.reshape(-1)
    if np.std(derivative_flat) > 0.0 and np.std(target_flat) > 0.0:
        correlation = float(np.corrcoef(derivative_flat, target_flat)[0, 1])
    else:
        correlation = float("nan")
    return relative_l2, correlation


def _tangent_identity_metrics(
    state: np.ndarray,
    second: np.ndarray,
    *,
    frame_time: float,
    friction: float,
) -> dict[str, float]:
    derivative = (state[:, 2:] - state[:, :-2]) / (2.0 * frame_time)
    centered = state[:, 1:-1]
    targets = {
        "position_velocity": centered[:, :, 3:6],
        "velocity_force": centered[:, :, 6:9] - friction * centered[:, :, 3:6],
        "force_generator": centered[:, :, 9:12],
        "generator_second": second[:, 1:-1],
    }
    slices = {
        "position_velocity": slice(0, 3),
        "velocity_force": slice(3, 6),
        "force_generator": slice(6, 9),
        "generator_second": slice(9, 12),
    }
    metrics: dict[str, float] = {}
    for name, target in targets.items():
        relative_l2, correlation = _identity_error(derivative[:, :, slices[name]], target)
        metrics[f"{name}_relative_l2_error"] = relative_l2
        metrics[f"{name}_correlation"] = correlation
    return metrics


def _increment_statistics(
    increment: np.ndarray,
    *,
    prefix: str,
) -> dict[str, np.ndarray | float]:
    member_count = increment.shape[0]
    rms = float(np.sqrt(np.mean(np.sum(increment**2, axis=2))))
    ensemble_increment = np.mean(increment, axis=0)
    ensemble_rms = float(np.sqrt(np.mean(np.sum(ensemble_increment**2, axis=1))))
    suppression = ensemble_rms / rms if rms > 0.0 else float("nan")

    lag_source = increment[:, :-1].reshape(-1)
    lag_target = increment[:, 1:].reshape(-1)
    if np.std(lag_source) > 0.0 and np.std(lag_target) > 0.0:
        lag1_correlation = float(np.corrcoef(lag_source, lag_target)[0, 1])
    else:
        lag1_correlation = float("nan")

    squared_norm = np.sum(increment**2, axis=2)
    energy_source = squared_norm[:, :-1].reshape(-1)
    energy_target = squared_norm[:, 1:].reshape(-1)
    if np.std(energy_source) > 0.0 and np.std(energy_target) > 0.0:
        squared_norm_lag1_correlation = float(np.corrcoef(energy_source, energy_target)[0, 1])
    else:
        squared_norm_lag1_correlation = float("nan")
    increment_norm = np.sqrt(squared_norm).reshape(-1)

    component_excess = np.full(3, np.nan, dtype=float)
    flattened = increment.reshape(-1, 3)
    for component in range(3):
        centered_component = flattened[:, component] - np.mean(flattened[:, component])
        variance = float(np.mean(centered_component**2))
        if variance > 0.0:
            component_excess[component] = float(np.mean(centered_component**4) / variance**2 - 3.0)

    sample_count = len(flattened)
    increment_mean = np.mean(flattened, axis=0)
    covariance = np.cov(flattened, rowvar=False, ddof=1) if sample_count > 1 else np.zeros((3, 3))
    standard_error_scale = math.sqrt(max(float(np.trace(covariance)), 0.0) / sample_count)
    if standard_error_scale > 0.0:
        normalized_mean = float(np.linalg.norm(increment_mean) / standard_error_scale)
    else:
        normalized_mean = 0.0 if np.linalg.norm(increment_mean) == 0.0 else float("inf")

    return {
        prefix: increment,
        f"ensemble_mean_{prefix}": ensemble_increment,
        f"{prefix}_covariance": covariance,
        f"{prefix}_component_excess_kurtosis": component_excess,
        f"{prefix}_rms": rms,
        f"ensemble_mean_{prefix}_rms": ensemble_rms,
        f"{prefix}_ensemble_suppression": suppression,
        f"{prefix}_scaled_ensemble_suppression": suppression * math.sqrt(member_count),
        f"{prefix}_lag1_correlation": lag1_correlation,
        f"{prefix}_squared_norm_lag1_correlation": squared_norm_lag1_correlation,
        f"{prefix}_norm_median": float(np.median(increment_norm)),
        f"{prefix}_norm_p95": float(np.quantile(increment_norm, 0.95)),
        f"{prefix}_norm_p99": float(np.quantile(increment_norm, 0.99)),
        f"{prefix}_norm_maximum": float(np.max(increment_norm)),
        f"{prefix}_max_abs_component_excess_kurtosis": float(np.nanmax(np.abs(component_excess))),
        f"{prefix}_normalized_mean": normalized_mean,
    }


def generator_response_tangent_diagnostic(
    state_response: np.ndarray,
    second_force_response: np.ndarray,
    *,
    frame_time: float,
    friction: float,
) -> dict[str, np.ndarray | float]:
    """Audit exact tangent identities and the omitted common-noise martingale.

    The final coordinate obeys
    `d(delta LF) = delta L2F dt - sqrt(2 gamma T) delta H dW`.
    Consequently the drift-subtracted increment, rather than `delta L2F`
    alone, is the microscopic residual relevant to a stochastic closure.
    """

    state = np.asarray(state_response, dtype=float)
    second = np.asarray(second_force_response, dtype=float)
    if state.ndim == 2:
        state = state[None, ...]
        second = second[None, ...]
    if state.ndim != 3 or state.shape[1] < 3 or state.shape[2] != 12:
        raise ValueError("state_response must have shape (members, at least 3 frames, 12)")
    if second.shape != state.shape[:2] + (3,):
        raise ValueError("second_force_response must align with state_response and have three components")
    if np.any(~np.isfinite(state)) or np.any(~np.isfinite(second)):
        raise ValueError("response arrays must be finite")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be finite and positive")
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")

    metrics: dict[str, np.ndarray | float] = _tangent_identity_metrics(
        state,
        second,
        frame_time=frame_time,
        friction=friction,
    )
    ensemble_state = np.mean(state, axis=0, keepdims=True)
    ensemble_second = np.mean(second, axis=0, keepdims=True)
    ensemble_metrics = _tangent_identity_metrics(
        ensemble_state,
        ensemble_second,
        frame_time=frame_time,
        friction=friction,
    )
    metrics.update({f"ensemble_mean_{key}": value for key, value in ensemble_metrics.items()})

    generator = state[:, :, 9:12]
    tangent_innovation = (
        generator[:, 1:]
        - generator[:, :-1]
        - frame_time * second[:, :-1]
    )
    symmetric_residual = (
        generator[:, 1:]
        - generator[:, :-1]
        - 0.5 * frame_time * (second[:, 1:] + second[:, :-1])
    )
    metrics.update(_increment_statistics(tangent_innovation, prefix="tangent_innovation"))
    metrics.update(_increment_statistics(symmetric_residual, prefix="symmetric_tangent_residual"))
    metrics["member_count"] = float(state.shape[0])
    metrics["thermodynamic_claim_allowed"] = 0.0
    return metrics


def fit_generator_constrained_response(
    state_response: np.ndarray,
    second_force_response: np.ndarray,
    *,
    frame_time: float,
    friction: float,
    fit_frames: int,
) -> dict[str, np.ndarray | float]:
    """Fit only the `L2F` row of a tagged 12-state tangent generator."""

    state = np.asarray(state_response, dtype=float)
    second = np.asarray(second_force_response, dtype=float)
    squeeze_output = state.ndim == 2
    if squeeze_output:
        state = state[None, ...]
        second = second[None, ...]
    if state.ndim != 3 or state.shape[2] != 12 or state.shape[1] < 5:
        raise ValueError("state_response must have shape (members, at least 5 frames, 12)")
    if second.shape != state.shape[:2] + (3,):
        raise ValueError("second_force_response must align with state_response and have three components")
    if np.any(~np.isfinite(state)) or np.any(~np.isfinite(second)):
        raise ValueError("response arrays must be finite")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be finite and positive")
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not isinstance(fit_frames, (int, np.integer)) or not (4 <= fit_frames < state.shape[1]):
        raise ValueError("fit_frames must leave a temporal holdout")

    training_state = state[:, :fit_frames].reshape(-1, 12)
    training_second = second[:, :fit_frames].reshape(-1, 3)
    scale = np.sqrt(np.mean(training_state**2, axis=0))
    if np.any(scale <= 1e-14):
        raise ValueError("each retained response coordinate must vary on the fitting interval")
    scaled_state = training_state / scale
    scaled_coefficient, _, rank, _ = np.linalg.lstsq(scaled_state, training_second, rcond=None)
    if rank < 12:
        raise ValueError("training responses do not identify the 12-coordinate generator projection")
    final_block = scaled_coefficient.T / scale[None, :]

    continuous_generator = np.zeros((12, 12), dtype=float)
    continuous_generator[0:3, 3:6] = np.eye(3)
    continuous_generator[3:6, 3:6] = -friction * np.eye(3)
    continuous_generator[3:6, 6:9] = np.eye(3)
    continuous_generator[6:9, 9:12] = np.eye(3)
    continuous_generator[9:12] = final_block
    transition = _rk4_linear_transition(continuous_generator, frame_time)

    predicted = np.empty_like(state)
    predicted[:, 0] = state[:, 0]
    for frame in range(1, state.shape[1]):
        predicted[:, frame] = np.einsum("ab,pb->pa", transition, predicted[:, frame - 1])

    fitted_training_second = training_state @ final_block.T
    second_norm = float(np.linalg.norm(training_second))
    training_position_norm = float(np.linalg.norm(state[:, :fit_frames, :3]))
    held_position_norm = float(np.linalg.norm(state[:, fit_frames:, :3]))
    if min(second_norm, training_position_norm, held_position_norm) <= 0.0:
        raise ValueError("training targets and position responses must have nonzero norm")
    residual = training_second - fitted_training_second
    residual_correlation = np.full((3, 12), np.nan, dtype=float)
    for output in range(3):
        for coordinate in range(12):
            if np.std(residual[:, output]) > 0.0 and np.std(training_state[:, coordinate]) > 0.0:
                residual_correlation[output, coordinate] = np.corrcoef(
                    residual[:, output], training_state[:, coordinate]
                )[0, 1]
    predicted_output = predicted[0] if squeeze_output else predicted
    return {
        "fitted_second_generator_block": final_block,
        "continuous_generator": continuous_generator,
        "transition_matrix": transition,
        "predicted_state_response": predicted_output,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
        "design_rank": float(rank),
        "training_second_generator_relative_l2_error": float(
            np.linalg.norm(residual) / second_norm
        ),
        "training_position_relative_l2_error": float(
            np.linalg.norm(predicted[:, :fit_frames, :3] - state[:, :fit_frames, :3]) / training_position_norm
        ),
        "heldout_position_relative_l2_error": float(
            np.linalg.norm(predicted[:, fit_frames:, :3] - state[:, fit_frames:, :3]) / held_position_norm
        ),
        "residual_state_correlation": residual_correlation,
        "thermodynamic_claim_allowed": 0.0,
    }
