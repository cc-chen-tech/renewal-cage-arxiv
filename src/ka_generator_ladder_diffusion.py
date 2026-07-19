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
