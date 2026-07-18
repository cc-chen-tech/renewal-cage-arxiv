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
