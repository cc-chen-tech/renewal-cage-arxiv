"""Exact common-noise diffusion blocks of a microscopic generator ladder."""

from __future__ import annotations

import math

import numpy as np


_CLOSED_CLAIMS = {
    "finite_generator_ladder_closure_supported": 0.0,
    "microscopic_environment_coordinate_z_allowed": 0.0,
    "continuous_gaussian_langevin_bath_allowed": 0.0,
    "autonomous_single_particle_gle_allowed": 0.0,
    "complete_event_clock_closure_allowed": 0.0,
    "kramers_escape_claim_allowed": 0.0,
    "spatial_facilitation_claim_allowed": 0.0,
    "thermodynamic_claim_allowed": 0.0,
}


def lp_velocity_jacobian(
    cage_jacobian: np.ndarray,
    cage_jacobian_velocity_derivative: np.ndarray,
    *,
    friction: float,
) -> dict[str, np.ndarray | float]:
    """Return ``D_V(Lp)=2 D_R J[V]-gamma J`` in noise-matrix layout."""

    jacobian = np.asarray(cage_jacobian, dtype=float)
    derivative = np.asarray(cage_jacobian_velocity_derivative, dtype=float)
    if (
        jacobian.ndim != 4
        or jacobian.shape[2:] != (3, 3)
        or derivative.shape != jacobian.shape
        or len(jacobian) < 1
        or jacobian.shape[1] < 1
        or np.any(~np.isfinite(jacobian))
        or np.any(~np.isfinite(derivative))
    ):
        raise ValueError(
            "cage Jacobians must have finite shape (targets, particles, 3, 3)"
        )
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    matrix = np.transpose(
        2.0 * derivative - float(friction) * jacobian,
        (0, 2, 1, 3),
    )
    return {
        "lp_velocity_jacobian": matrix,
        "friction": float(friction),
        **_CLOSED_CLAIMS,
    }


def generator_ladder_conditional_diffusion(
    velocity_jacobians: np.ndarray,
    *,
    friction: float,
    temperature: float,
) -> dict[str, np.ndarray | float | str]:
    """Assemble every ``Q_rs=2 gamma T A_r A_s^T`` common-noise block."""

    jacobians = np.asarray(velocity_jacobians, dtype=float)
    if (
        jacobians.ndim != 5
        or jacobians.shape[2] != 3
        or jacobians.shape[4] != 3
        or jacobians.shape[0] < 1
        or jacobians.shape[1] < 1
        or jacobians.shape[3] < 1
        or np.any(~np.isfinite(jacobians))
    ):
        raise ValueError(
            "velocity_jacobians must have finite shape "
            "(levels, targets, 3, particles, 3)"
        )
    if not math.isfinite(friction) or friction < 0.0:
        raise ValueError("friction must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and nonnegative")
    scale = 2.0 * float(friction) * float(temperature)
    blocks = scale * np.einsum(
        "rtanb,stcnb->rstac",
        jacobians,
        jacobians,
    )
    level_count, _, target_count = blocks.shape[:3]
    joint = np.transpose(blocks, (2, 0, 3, 1, 4)).reshape(
        target_count,
        3 * level_count,
        3 * level_count,
    )
    joint = 0.5 * (joint + np.swapaxes(joint, -1, -2))
    minimum_eigenvalue = np.linalg.eigvalsh(joint)[:, 0]
    return {
        "diffusion_blocks": blocks,
        "joint_diffusion": joint,
        "minimum_joint_eigenvalue": minimum_eigenvalue,
        "noise_level_count": float(level_count),
        "noise_source_dimension": float(3 * jacobians.shape[3]),
        "friction": float(friction),
        "temperature": float(temperature),
        "estimator": "exact_generator_ladder_carre_du_champ",
        **_CLOSED_CLAIMS,
    }


def generator_ladder_local_characteristics(
    coordinates: np.ndarray,
    *,
    terminal_generator: np.ndarray,
    velocity_jacobians: np.ndarray,
    friction: float,
    temperature: float,
) -> dict[str, np.ndarray | float | str]:
    """Return the exact microscopic-frame drift and diffusion of a ladder.

    For ``Y=(g_0,...,g_{m-1})`` with ``g_{r+1}=L g_r``, the drift is
    ``(g_1,...,g_m)``.  This identity is conditioned on the full microscopic
    phase-space frame and therefore does not assert that ``Y`` is autonomous.
    """

    state = np.asarray(coordinates, dtype=float)
    terminal = np.asarray(terminal_generator, dtype=float)
    jacobians = np.asarray(velocity_jacobians, dtype=float)
    if (
        state.ndim != 3
        or state.shape[2] != 3
        or state.shape[0] < 1
        or state.shape[1] < 1
        or terminal.shape != state.shape[1:]
        or jacobians.ndim != 5
        or jacobians.shape[:3] != state.shape
        or jacobians.shape[4] != 3
        or jacobians.shape[3] < 1
        or np.any(~np.isfinite(state))
        or np.any(~np.isfinite(terminal))
        or np.any(~np.isfinite(jacobians))
    ):
        raise ValueError(
            "generator coordinates, terminal drift, and velocity Jacobians "
            "must be finite aligned microscopic-frame arrays"
        )
    diffusion = generator_ladder_conditional_diffusion(
        jacobians,
        friction=friction,
        temperature=temperature,
    )
    drift = np.concatenate((state[1:], terminal[None]), axis=0)
    return {
        "coordinates": state,
        "drift": drift,
        **diffusion,
        "microscopic_state_conditioned": 1.0,
        "projected_state_autonomous": 0.0,
        **_CLOSED_CLAIMS,
    }


def quadratic_test_generator(
    state: np.ndarray,
    *,
    drift: np.ndarray,
    diffusion: np.ndarray,
    linear: np.ndarray,
    hessian: np.ndarray,
) -> float:
    """Evaluate ``L f`` for ``f(y)=a.y + y.H.y/2`` from local characteristics."""

    value = np.asarray(state, dtype=float)
    local_drift = np.asarray(drift, dtype=float)
    local_diffusion = np.asarray(diffusion, dtype=float)
    linear_term = np.asarray(linear, dtype=float)
    curvature = np.asarray(hessian, dtype=float)
    dimension = value.size
    if (
        value.ndim != 1
        or local_drift.shape != value.shape
        or linear_term.shape != value.shape
        or local_diffusion.shape != (dimension, dimension)
        or curvature.shape != (dimension, dimension)
        or np.any(~np.isfinite(value))
        or np.any(~np.isfinite(local_drift))
        or np.any(~np.isfinite(local_diffusion))
        or np.any(~np.isfinite(linear_term))
        or np.any(~np.isfinite(curvature))
        or not np.allclose(curvature, curvature.T, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("quadratic generator inputs must be finite and aligned")
    gradient = linear_term + curvature @ value
    return float(
        gradient @ local_drift
        + 0.5 * np.einsum("ij,ij->", curvature, local_diffusion)
    )
