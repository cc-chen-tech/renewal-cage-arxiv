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

    rng = np.random.default_rng(seed)
    x = np.zeros((trajectory_count, dimension), dtype=float)
    q = np.zeros_like(x)
    z = rng.normal(
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
            + x_noise * rng.normal(size=x.shape)
        )
        if params.elastic_stiffness > 0.0:
            delta_q = (
                force_q * dt / params.gamma_q
                + q_noise * rng.normal(size=q.shape)
            )
        else:
            delta_q = np.zeros_like(q)
        delta_z = (
            force_z * dt / params.gamma_z
            + z_noise * rng.normal(size=z.shape)
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
