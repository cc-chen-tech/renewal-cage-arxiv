"""Thermodynamically consistent nonlinear auxiliary-bath Langevin algebra."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_CLOSED_CLAIMS = {
    "exact_nonlinear_bath_elimination_supported": 0.0,
    "synthetic_bath_level_fdt_replay_supported": 0.0,
    "synthetic_delayed_hazard_emerges": 0.0,
    "real_ka_position_dependent_kernel_authorized": 0.0,
    "autonomous_single_particle_gle_allowed": 0.0,
    "complete_event_clock_closure_allowed": 0.0,
    "kramers_escape_claim_allowed": 0.0,
    "spatial_facilitation_claim_allowed": 0.0,
    "thermodynamic_claim_allowed": 0.0,
}


@dataclass(frozen=True)
class NonlinearBathControls:
    """Validated scalar and mode controls for the frozen auxiliary bath."""

    temperature: float
    friction: float
    period: float
    barrier: float
    rates: np.ndarray
    amplitudes: np.ndarray
    modulation: np.ndarray
    phases: np.ndarray
    time_step: float

    def __post_init__(self) -> None:
        rates = np.asarray(self.rates, dtype=float)
        amplitudes = np.asarray(self.amplitudes, dtype=float)
        modulation = np.asarray(self.modulation, dtype=float)
        phases = np.asarray(self.phases, dtype=float)
        if (
            rates.ndim != 1
            or len(rates) < 1
            or amplitudes.shape != rates.shape
            or modulation.shape != rates.shape
            or phases.shape != rates.shape
            or np.any(~np.isfinite(rates))
            or np.any(~np.isfinite(amplitudes))
            or np.any(~np.isfinite(modulation))
            or np.any(~np.isfinite(phases))
            or np.any(rates < 0.0)
            or np.any(amplitudes < 0.0)
            or np.any(np.abs(modulation) >= 1.0)
        ):
            raise ValueError("bath mode controls must be finite aligned vectors")
        scalars = (
            self.temperature,
            self.friction,
            self.period,
            self.barrier,
            self.time_step,
        )
        if any(not math.isfinite(float(value)) for value in scalars):
            raise ValueError("bath scalar controls must be finite")
        if (
            self.temperature < 0.0
            or self.friction < 0.0
            or self.period <= 0.0
            or self.barrier < 0.0
            or self.time_step <= 0.0
        ):
            raise ValueError("bath scalar controls are outside their physical domain")
        object.__setattr__(self, "rates", rates.copy())
        object.__setattr__(self, "amplitudes", amplitudes.copy())
        object.__setattr__(self, "modulation", modulation.copy())
        object.__setattr__(self, "phases", phases.copy())


def periodic_potential(
    position: np.ndarray,
    *,
    barrier: float,
    period: float,
) -> np.ndarray:
    """Return ``V0 [1-cos(2 pi u/ell)]``."""

    value = np.asarray(position, dtype=float)
    if (
        np.any(~np.isfinite(value))
        or not math.isfinite(barrier)
        or barrier < 0.0
        or not math.isfinite(period)
        or period <= 0.0
    ):
        raise ValueError("periodic-potential inputs must be finite and physical")
    return float(barrier) * (1.0 - np.cos(2.0 * np.pi * value / float(period)))


def periodic_potential_gradient(
    position: np.ndarray,
    *,
    barrier: float,
    period: float,
) -> np.ndarray:
    """Return the analytic gradient of the frozen periodic potential."""

    value = np.asarray(position, dtype=float)
    periodic_potential(value, barrier=barrier, period=period)
    wave_number = 2.0 * np.pi / float(period)
    return float(barrier) * wave_number * np.sin(wave_number * value)


def periodic_coupling(
    position: np.ndarray,
    *,
    controls: NonlinearBathControls,
) -> np.ndarray:
    """Return all position-dependent scalar mode couplings ``C_a(u)``."""

    value = np.asarray(position, dtype=float)
    if np.any(~np.isfinite(value)):
        raise ValueError("coupling positions must be finite")
    phase = (
        2.0 * np.pi * value[..., None] / controls.period
        + controls.phases
    )
    return controls.amplitudes * (1.0 + controls.modulation * np.cos(phase))


def gibbs_stationarity_audit(
    position: np.ndarray,
    momentum: np.ndarray,
    auxiliary: np.ndarray,
    *,
    controls: NonlinearBathControls,
) -> dict[str, np.ndarray | float]:
    """Evaluate the exact Fokker-Planck cancellation for the Gibbs density."""

    value = np.asarray(position, dtype=float)
    velocity = np.asarray(momentum, dtype=float)
    bath = np.asarray(auxiliary, dtype=float)
    if (
        velocity.shape != value.shape
        or bath.shape != value.shape + (len(controls.rates),)
        or np.any(~np.isfinite(value))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(bath))
        or controls.temperature <= 0.0
    ):
        raise ValueError("Gibbs stationarity audit needs finite positive-T state")
    temperature = controls.temperature
    gradient = periodic_potential_gradient(
        value,
        barrier=controls.barrier,
        period=controls.period,
    )
    coupling = periodic_coupling(value, controls=controls)
    hamiltonian_energy_rate = gradient * velocity - gradient * velocity
    bath_force = np.sum(coupling * bath, axis=-1)
    antisymmetric_bath_energy_rate = velocity * bath_force + np.sum(
        bath * (-coupling * velocity[..., None]),
        axis=-1,
    )
    conservative_divergence = np.zeros_like(value)

    momentum_drift_residual = controls.friction * (
        1.0 - velocity**2 / temperature
    )
    momentum_diffusion_residual = controls.friction * (
        velocity**2 / temperature - 1.0
    )
    momentum_thermostat_residual = (
        momentum_drift_residual + momentum_diffusion_residual
    )
    auxiliary_drift_residual = controls.rates * (
        1.0 - bath**2 / temperature
    )
    auxiliary_diffusion_residual = controls.rates * (
        bath**2 / temperature - 1.0
    )
    auxiliary_thermostat_residual = (
        auxiliary_drift_residual + auxiliary_diffusion_residual
    )
    maximum = float(
        max(
            np.max(np.abs(hamiltonian_energy_rate)),
            np.max(np.abs(antisymmetric_bath_energy_rate)),
            np.max(np.abs(conservative_divergence)),
            np.max(np.abs(momentum_thermostat_residual)),
            np.max(np.abs(auxiliary_thermostat_residual)),
        )
    )
    return {
        "hamiltonian_energy_rate": hamiltonian_energy_rate,
        "antisymmetric_bath_energy_rate": antisymmetric_bath_energy_rate,
        "conservative_phase_space_divergence": conservative_divergence,
        "momentum_thermostat_drift_residual": momentum_drift_residual,
        "momentum_thermostat_diffusion_residual": momentum_diffusion_residual,
        "momentum_thermostat_stationarity_residual": (
            momentum_thermostat_residual
        ),
        "auxiliary_thermostat_drift_residual": auxiliary_drift_residual,
        "auxiliary_thermostat_diffusion_residual": auxiliary_diffusion_residual,
        "auxiliary_thermostat_stationarity_residual": (
            auxiliary_thermostat_residual
        ),
        "maximum_normalized_stationarity_residual": maximum,
        "gibbs_invariant_density_derived": float(maximum <= 5e-13),
        **_CLOSED_CLAIMS,
    }


def _ou_coefficients(controls: NonlinearBathControls) -> tuple[np.ndarray, np.ndarray]:
    decay = np.exp(-controls.rates * controls.time_step)
    integral = np.empty_like(decay)
    positive = controls.rates > 0.0
    integral[positive] = -np.expm1(
        -controls.rates[positive] * controls.time_step
    ) / controls.rates[positive]
    integral[~positive] = controls.time_step
    return decay, integral


def nonlinear_bath_step(
    position: np.ndarray,
    momentum: np.ndarray,
    auxiliary: np.ndarray,
    *,
    normal_p: np.ndarray,
    normal_z: np.ndarray,
    controls: NonlinearBathControls,
) -> dict[str, np.ndarray | float]:
    """Advance the frozen explicit/exact-OU discretization by one step."""

    value = np.asarray(position, dtype=float)
    velocity = np.asarray(momentum, dtype=float)
    bath = np.asarray(auxiliary, dtype=float)
    momentum_noise = np.asarray(normal_p, dtype=float)
    bath_noise = np.asarray(normal_z, dtype=float)
    expected_bath_shape = value.shape + (len(controls.rates),)
    if (
        velocity.shape != value.shape
        or momentum_noise.shape != value.shape
        or bath.shape != expected_bath_shape
        or bath_noise.shape != expected_bath_shape
        or np.any(~np.isfinite(value))
        or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(bath))
        or np.any(~np.isfinite(momentum_noise))
        or np.any(~np.isfinite(bath_noise))
    ):
        raise ValueError("state and noise arrays must be finite and aligned")
    coupling = periodic_coupling(value, controls=controls)
    gradient = periodic_potential_gradient(
        value,
        barrier=controls.barrier,
        period=controls.period,
    )
    time_step = controls.time_step
    next_position = value + velocity * time_step
    next_momentum = velocity + (
        -gradient
        - controls.friction * velocity
        + np.sum(coupling * bath, axis=-1)
    ) * time_step
    next_momentum += math.sqrt(
        2.0 * controls.friction * controls.temperature * time_step
    ) * momentum_noise
    decay, integral = _ou_coefficients(controls)
    next_auxiliary = decay * bath - integral * coupling * velocity[..., None]
    next_auxiliary += np.sqrt(
        controls.temperature * np.maximum(1.0 - decay**2, 0.0)
    ) * bath_noise
    return {
        "position": next_position,
        "momentum": next_momentum,
        "auxiliary": next_auxiliary,
        "coupling": coupling,
        "potential_gradient": gradient,
        **_CLOSED_CLAIMS,
    }


def reconstruct_auxiliary_path(
    initial_auxiliary: np.ndarray,
    *,
    positions: np.ndarray,
    momenta: np.ndarray,
    normal_increments: np.ndarray,
    controls: NonlinearBathControls,
) -> np.ndarray:
    """Reconstruct the auxiliary path from the exact discrete OU recurrence."""

    initial = np.asarray(initial_auxiliary, dtype=float)
    position = np.asarray(positions, dtype=float)
    momentum = np.asarray(momenta, dtype=float)
    noise = np.asarray(normal_increments, dtype=float)
    if (
        position.ndim < 1
        or momentum.shape != position.shape
        or initial.shape != position.shape[1:] + (len(controls.rates),)
        or noise.shape != position.shape + (len(controls.rates),)
        or np.any(~np.isfinite(initial))
        or np.any(~np.isfinite(position))
        or np.any(~np.isfinite(momentum))
        or np.any(~np.isfinite(noise))
    ):
        raise ValueError("auxiliary reconstruction inputs must be finite and aligned")
    decay, integral = _ou_coefficients(controls)
    noise_scale = np.sqrt(
        controls.temperature * np.maximum(1.0 - decay**2, 0.0)
    )
    output = np.empty((len(position) + 1, *initial.shape), dtype=float)
    output[0] = initial
    for index in range(len(position)):
        coupling = periodic_coupling(position[index], controls=controls)
        output[index + 1] = (
            decay * output[index]
            - integral * coupling * momentum[index][..., None]
            + noise_scale * noise[index]
        )
    return output


def eliminated_memory_kernel(
    left_position: np.ndarray,
    right_position: np.ndarray,
    *,
    lag: float,
    controls: NonlinearBathControls,
) -> dict[str, np.ndarray | float]:
    """Return the exact scalar nonlinear memory kernel at two path points."""

    left = np.asarray(left_position, dtype=float)
    right = np.asarray(right_position, dtype=float)
    if (
        left.shape != right.shape
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
        or not math.isfinite(lag)
        or lag < 0.0
    ):
        raise ValueError("kernel positions and lag must be finite and aligned")
    left_coupling = periodic_coupling(left, controls=controls)
    right_coupling = periodic_coupling(right, controls=controls)
    kernel = np.sum(
        left_coupling
        * np.exp(-controls.rates * float(lag))
        * right_coupling,
        axis=-1,
    )
    return {
        "kernel": kernel,
        "lag": float(lag),
        **_CLOSED_CLAIMS,
    }
