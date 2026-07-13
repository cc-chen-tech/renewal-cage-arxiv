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
dump_modify trajectory sort id
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
    for frame, (frame_positions, frame_velocities) in enumerate(zip(positions, velocities)):
        observable = ka_lj_force_generator_observables(
            frame_positions,
            velocities=frame_velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=friction,
            temperature=temperature,
        )
        force[frame] = observable["force"][0]
        generator[frame] = observable["force_generator"][0]
        covariance_rate[frame] = observable["force_generator_noise_covariance_rate"][0]
        second[frame] = ka_lj_second_force_generator(
            frame_positions,
            velocities=frame_velocities,
            particle_types=particle_types,
            box_lengths=box_lengths,
            target_indices=target,
            friction=friction,
            directional_step=directional_step,
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
