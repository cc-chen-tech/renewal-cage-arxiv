"""Generator-constrained Krylov reductions of matched KA responses."""

from __future__ import annotations

import math

import numpy as np


def assemble_second_generator_state(
    state_response: np.ndarray,
    second_force_response: np.ndarray,
) -> np.ndarray:
    """Append ``delta L2F`` to the fixed ``(x,v,F,LF)`` response order."""

    state = np.asarray(state_response, dtype=float)
    second = np.asarray(second_force_response, dtype=float)
    if state.ndim not in (2, 3) or state.shape[-1] != 12:
        raise ValueError(
            "state_response must end in the 12 microscopic response coordinates"
        )
    if second.shape != state.shape[:-1] + (3,):
        raise ValueError("second_force_response must align with state_response")
    if np.any(~np.isfinite(state)) or np.any(~np.isfinite(second)):
        raise ValueError("response coordinates must be finite")
    return np.concatenate((state, second), axis=-1)


def _rk4_linear_transition(generator: np.ndarray, frame_time: float) -> np.ndarray:
    transition = np.eye(len(generator))
    term = np.eye(len(generator))
    scaled = frame_time * generator
    for order in range(1, 5):
        term = term @ scaled / order
        transition += term
    return transition


def fit_second_generator_constrained_response(
    second_generator_state: np.ndarray,
    *,
    frame_time: float,
    friction: float,
    fit_frames: int,
) -> dict[str, np.ndarray | float]:
    """Fit only the weak-form ``L3F`` row of a 15-state tangent chain."""

    state = np.asarray(second_generator_state, dtype=float)
    squeeze_output = state.ndim == 2
    if squeeze_output:
        state = state[None, ...]
    if state.ndim != 3 or state.shape[2] != 15 or state.shape[1] < 5:
        raise ValueError(
            "second_generator_state must have shape (members, at least 5 frames, 15)"
        )
    if np.any(~np.isfinite(state)):
        raise ValueError("second_generator_state must be finite")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be finite and positive")
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not isinstance(fit_frames, (int, np.integer)) or not (
        4 <= fit_frames < state.shape[1]
    ):
        raise ValueError("fit_frames must leave a temporal holdout")

    training = state[:, :fit_frames]
    source = 0.5 * (training[:, 1:] + training[:, :-1])
    target = (training[:, 1:, 12:15] - training[:, :-1, 12:15]) / frame_time
    source_rows = source.reshape(-1, 15)
    target_rows = target.reshape(-1, 3)
    scale = np.sqrt(np.mean(source_rows**2, axis=0))
    if np.any(scale <= 1e-14):
        raise ValueError("each retained response coordinate must vary on the fit interval")
    scaled_coefficient, _, rank, _ = np.linalg.lstsq(
        source_rows / scale,
        target_rows,
        rcond=None,
    )
    if rank < 15:
        raise ValueError("weak-form responses do not identify the 15-coordinate projection")
    third_block = scaled_coefficient.T / scale[None, :]

    continuous_generator = np.zeros((15, 15), dtype=float)
    continuous_generator[0:3, 3:6] = np.eye(3)
    continuous_generator[3:6, 3:6] = -friction * np.eye(3)
    continuous_generator[3:6, 6:9] = np.eye(3)
    continuous_generator[6:9, 9:12] = np.eye(3)
    continuous_generator[9:12, 12:15] = np.eye(3)
    continuous_generator[12:15] = third_block
    transition = _rk4_linear_transition(continuous_generator, frame_time)

    predicted = np.empty_like(state)
    predicted[:, 0] = state[:, 0]
    for frame in range(1, state.shape[1]):
        predicted[:, frame] = np.einsum(
            "ab,pb->pa", transition, predicted[:, frame - 1]
        )

    fitted_target = source_rows @ third_block.T
    residual = target_rows - fitted_target
    target_norm = float(np.linalg.norm(target_rows))
    held_position = state[:, fit_frames:, :3]
    held_norm = float(np.linalg.norm(held_position))
    if target_norm <= 0.0 or held_norm <= 0.0:
        raise ValueError("fit target and held position response must have nonzero norm")
    predicted_output = predicted[0] if squeeze_output else predicted
    return {
        "fitted_third_generator_block": third_block,
        "continuous_generator": continuous_generator,
        "transition_matrix": transition,
        "predicted_state_response": predicted_output,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
        "design_rank": float(rank),
        "training_third_generator_relative_l2_error": float(
            np.linalg.norm(residual) / target_norm
        ),
        "heldout_position_relative_l2_error": float(
            np.linalg.norm(predicted[:, fit_frames:, :3] - held_position) / held_norm
        ),
        "thermodynamic_claim_allowed": 0.0,
    }


def fit_free_second_generator_transition(
    second_generator_state: np.ndarray,
    *,
    fit_frames: int,
) -> dict[str, np.ndarray | float]:
    """Fit an unconstrained 15-state discrete transition as a control."""

    state = np.asarray(second_generator_state, dtype=float)
    if state.ndim == 2:
        state = state[None, ...]
    if state.ndim != 3 or state.shape[2] != 15 or state.shape[1] < 3:
        raise ValueError(
            "second_generator_state must have shape (members, at least 3 frames, 15)"
        )
    if np.any(~np.isfinite(state)):
        raise ValueError("second_generator_state must be finite")
    if not isinstance(fit_frames, (int, np.integer)) or not (
        3 <= fit_frames < state.shape[1]
    ):
        raise ValueError("fit_frames must leave a temporal holdout")
    source = state[:, : fit_frames - 1].reshape(-1, 15)
    target = state[:, 1:fit_frames].reshape(-1, 15)
    scale = np.sqrt(np.mean(source**2, axis=0))
    if np.any(scale <= 1e-14):
        raise ValueError("each retained response coordinate must vary on the fit interval")
    coefficient, _, rank, _ = np.linalg.lstsq(source / scale, target, rcond=None)
    if rank < 15:
        raise ValueError("free transition does not identify all 15 response coordinates")
    transition = coefficient.T / scale[None, :]
    return {
        "transition_matrix": transition,
        "design_rank": float(rank),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(transition)))),
        "thermodynamic_claim_allowed": 0.0,
    }


def propagate_linear_response(
    transition_matrix: np.ndarray,
    initial_state: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    """Propagate one autonomous linear response from a supplied initial state."""

    transition = np.asarray(transition_matrix, dtype=float)
    initial = np.asarray(initial_state, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition_matrix must be square")
    if initial.shape != (transition.shape[0],):
        raise ValueError("initial_state must align with transition_matrix")
    if np.any(~np.isfinite(transition)) or np.any(~np.isfinite(initial)):
        raise ValueError("transition_matrix and initial_state must be finite")
    if not isinstance(frame_count, (int, np.integer)) or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    predicted = np.empty((frame_count, len(initial)), dtype=float)
    predicted[0] = initial
    for frame in range(1, frame_count):
        predicted[frame] = transition @ predicted[frame - 1]
    return predicted


def second_generator_residual_diagnostic(
    second_generator_state: np.ndarray,
    third_generator_block: np.ndarray,
    *,
    frame_time: float,
) -> dict[str, np.ndarray | float]:
    """Measure weak-form ``L3F`` residuals and retained-state correlation."""

    state = np.asarray(second_generator_state, dtype=float)
    if state.ndim == 2:
        state = state[None, ...]
    block = np.asarray(third_generator_block, dtype=float)
    if state.ndim != 3 or state.shape[1] < 2 or state.shape[2] != 15:
        raise ValueError(
            "second_generator_state must have shape (members, at least 2 frames, 15)"
        )
    if block.shape != (3, 15):
        raise ValueError("third_generator_block must have shape (3, 15)")
    if np.any(~np.isfinite(state)) or np.any(~np.isfinite(block)):
        raise ValueError("state and block must be finite")
    if not math.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("frame_time must be finite and positive")

    midpoint = 0.5 * (state[:, 1:] + state[:, :-1])
    observed = state[:, 1:, 12:15] - state[:, :-1, 12:15]
    predicted = frame_time * np.einsum("...b,ab->...a", midpoint, block)
    residual = observed - predicted
    observed_norm = float(np.linalg.norm(observed))
    if observed_norm <= 0.0:
        raise ValueError("second-generator increments must have nonzero norm")
    relative_l2 = float(np.linalg.norm(residual) / observed_norm)
    residual_rows = residual.reshape(-1, 3)
    state_rows = midpoint.reshape(-1, 15)
    correlation = np.full((3, 15), np.nan, dtype=float)
    if relative_l2 > 1e-12:
        for output in range(3):
            for coordinate in range(15):
                if (
                    np.std(residual_rows[:, output]) > 0.0
                    and np.std(state_rows[:, coordinate]) > 0.0
                ):
                    correlation[output, coordinate] = np.corrcoef(
                        residual_rows[:, output], state_rows[:, coordinate]
                    )[0, 1]
    finite = np.abs(correlation[np.isfinite(correlation)])
    maximum = float(np.max(finite)) if len(finite) else 0.0
    return {
        "residual": residual[0] if second_generator_state.ndim == 2 else residual,
        "residual_state_correlation": correlation,
        "residual_relative_l2": relative_l2,
        "maximum_abs_residual_state_correlation": maximum,
        "thermodynamic_claim_allowed": 0.0,
    }
